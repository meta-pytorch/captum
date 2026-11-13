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
from torch.utils.data import TensorDataset


class TestTracInIntermediateQuantitiesLearningRate(BaseTest):
    def test_intermediate_quantities_dot_product_with_learning_rates(self) -> None:
        # The influence score sums learning_rate * <grad, grad'> over checkpoints, so
        # the intermediate quantities must give it as a dot product for any
        # learning rate, not only 1.
        torch.manual_seed(0)
        net = nn.Sequential(nn.Linear(5, 4), nn.Tanh(), nn.Linear(4, 3))
        inputs, labels = torch.randn(8, 5), torch.randint(0, 3, (8,))
        dataset = TensorDataset(inputs, labels)
        test_examples = (inputs[:3], labels[:3])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoints = []
            learning_rates = {}
            for i, learning_rate in enumerate([0.25, 0.5]):
                checkpoint = os.path.join(tmpdir, f"checkpoint_{i}.pt")
                with torch.no_grad():
                    for param in net.parameters():
                        param.add_(0.1 * torch.randn_like(param))
                torch.save(net.state_dict(), checkpoint)
                checkpoints.append(checkpoint)
                learning_rates[checkpoint] = learning_rate

            def load(model: nn.Module, path: str) -> float:
                model.load_state_dict(torch.load(path))
                return learning_rates[path]

            tracin = TracInCP(
                net,
                dataset,
                checkpoints,
                checkpoints_load_func=load,
                loss_fn=nn.CrossEntropyLoss(reduction="none"),
                batch_size=8,
                sample_wise_grads_per_batch=False,
            )

            influence = tracin.influence(test_examples, k=None)
            train_quantities = tracin.compute_intermediate_quantities((inputs, labels))
            test_quantities = tracin.compute_intermediate_quantities(test_examples)
            assertTensorAlmostEqual(
                self,
                test_quantities @ train_quantities.T,
                influence,
                delta=1e-5,
                mode="max",
            )

            aggregated = tracin.compute_intermediate_quantities(
                test_examples, aggregate=True
            )
            assertTensorAlmostEqual(
                self,
                aggregated @ train_quantities.T,
                influence.sum(dim=0, keepdim=True),
                delta=1e-5,
                mode="max",
            )
