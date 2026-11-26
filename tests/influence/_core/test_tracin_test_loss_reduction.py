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
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from parameterized import parameterized
from torch.utils.data import TensorDataset


class TestTracInTestLossReduction(BaseTest):
    @parameterized.expand([("sum", "mean"), ("mean", "sum")])
    def test_test_loss_fn_with_different_reduction(
        self, train_reduction: str, test_reduction: str
    ) -> None:
        # `test_loss_fn` may use a different reduction than `loss_fn`; per-example
        # gradients, and hence influence scores, do not depend on the reduction.
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
        inputs, labels = torch.randn(8, 5), torch.randint(0, 3, (8,))
        dataset = TensorDataset(inputs, labels)
        test_examples = (inputs[:3], labels[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(net.state_dict(), checkpoint)

            def load(model: nn.Module, path: str) -> float:
                model.load_state_dict(torch.load(path))
                return 1.0

            def tracin(test_loss_reduction: str) -> TracInCP:
                return TracInCP(
                    net,
                    dataset,
                    [checkpoint],
                    checkpoints_load_func=load,
                    loss_fn=nn.CrossEntropyLoss(reduction=train_reduction),
                    test_loss_fn=nn.CrossEntropyLoss(reduction=test_loss_reduction),
                    batch_size=8,
                    sample_wise_grads_per_batch=True,
                )

            reference = tracin(train_reduction)
            different = tracin(test_reduction)

            assertTensorAlmostEqual(
                self,
                different.influence(test_examples, k=None),
                reference.influence(test_examples, k=None),
                delta=1e-5,
                mode="max",
            )
            assertTensorAlmostEqual(
                self,
                different.influence(test_examples, k=3).influence_scores,
                reference.influence(test_examples, k=3).influence_scores,
                delta=1e-5,
                mode="max",
            )
