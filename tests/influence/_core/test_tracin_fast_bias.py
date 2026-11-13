# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import os
import tempfile

import torch
import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from parameterized import parameterized
from torch.utils.data import TensorDataset


class _NetWithFinalBias(nn.Module):
    def __init__(self, bias: bool) -> None:
        super().__init__()
        self.body = nn.Linear(5, 4)
        self.fc = nn.Linear(4, 3, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.tanh(self.body(x)))


class TestTracInFastFinalLayerBias(BaseTest):
    @parameterized.expand(
        [
            (fast_cls.__name__, fast_cls, bias)
            for fast_cls in [TracInCPFast, TracInCPFastRandProj]
            for bias in [False, True]
        ]
    )
    def test_fast_matches_tracincp_on_final_layer(
        self, _name: str, fast_cls: type, bias: bool
    ) -> None:
        # The fast implementations only use last-layer gradients; they should match
        # TracInCP restricted to all parameters of that layer, including its bias.
        torch.manual_seed(0)
        net = _NetWithFinalBias(bias)
        inputs, labels = torch.randn(8, 5), torch.randint(0, 3, (8,))
        dataset = TensorDataset(inputs, labels)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(net.state_dict(), checkpoint)

            def load(model: nn.Module, path: str) -> float:
                model.load_state_dict(torch.load(path))
                return 1.0

            naive = TracInCP(
                net,
                dataset,
                [checkpoint],
                checkpoints_load_func=load,
                loss_fn=nn.CrossEntropyLoss(reduction="none"),
                batch_size=8,
                layers=["fc"],
                sample_wise_grads_per_batch=False,
            )
            fast = fast_cls(
                net,
                "fc",
                dataset,
                [checkpoint],
                checkpoints_load_func=load,
                loss_fn=nn.CrossEntropyLoss(reduction="sum"),
                batch_size=8,
            )

            test_examples = (inputs[:3], labels[:3])
            assertTensorAlmostEqual(
                self,
                fast.influence(test_examples, k=None),
                naive.influence(test_examples, k=None),
                delta=1e-4,
                mode="max",
            )
            if fast_cls is TracInCPFast:
                # `TracInCPFastRandProj` does not implement `self_influence`
                assertTensorAlmostEqual(
                    self,
                    fast.self_influence(),
                    naive.self_influence(),
                    delta=1e-4,
                    mode="max",
                )
