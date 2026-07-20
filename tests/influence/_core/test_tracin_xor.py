# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from collections import OrderedDict
from typing import Callable, cast, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.influence._core.tracincp import TracInCP
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _wrap_model_in_dataparallel,
    BasicLinearNet,
    BinaryDataset,
    build_test_name_func,
    DataInfluenceConstructor,
)
from parameterized import parameterized


# pyrefly: ignore [invalid-inheritance]
class TestTracInXOR(BaseTest):

    # TODO: Move test setup to use setUp and tearDown method overrides.
    def _test_tracin_xor_setup(
        self, tmpdir: str, use_gpu: bool = False
    ) -> Tuple[BinaryDataset, ...]:
        net = BasicLinearNet(in_features=2, hidden_nodes=2, out_features=1)

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.2956, -1.4465], [-0.3890, -0.7420]]),
                ),
                ("linear1.bias", torch.Tensor([1.2924, 0.0021])),
                ("linear2.weight", torch.Tensor([[-1.2013, 0.7174]])),
                ("linear2.bias", torch.Tensor([0.5880])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "0" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.3238, -1.4899], [-0.4544, -0.7448]]),
                ),
                ("linear1.bias", torch.Tensor([1.3185, -0.0317])),
                ("linear2.weight", torch.Tensor([[-1.2342, 0.7741]])),
                ("linear2.bias", torch.Tensor([0.6234])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "1" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.3546, -1.5288], [-0.5250, -0.7591]]),
                ),
                ("linear1.bias", torch.Tensor([1.3432, -0.0684])),
                ("linear2.weight", torch.Tensor([[-1.2490, 0.8534]])),
                ("linear2.bias", torch.Tensor([0.6749])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "2" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.4022, -1.5485], [-0.5688, -0.7607]]),
                ),
                ("linear1.bias", torch.Tensor([1.3740, -0.1571])),
                ("linear2.weight", torch.Tensor([[-1.3412, 0.9013]])),
                ("linear2.bias", torch.Tensor([0.6468])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "3" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.4464, -1.5890], [-0.6348, -0.7665]]),
                ),
                ("linear1.bias", torch.Tensor([1.3791, -0.2008])),
                ("linear2.weight", torch.Tensor([[-1.3818, 0.9586]])),
                ("linear2.bias", torch.Tensor([0.6954])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "4" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5217, -1.6242], [-0.6644, -0.7842]]),
                ),
                ("linear1.bias", torch.Tensor([1.3500, -0.2418])),
                ("linear2.weight", torch.Tensor([[-1.4304, 0.9980]])),
                ("linear2.bias", torch.Tensor([0.7567])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "5" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5551, -1.6631], [-0.7420, -0.8025]]),
                ),
                ("linear1.bias", torch.Tensor([1.3508, -0.2618])),
                ("linear2.weight", torch.Tensor([[-1.4272, 1.0772]])),
                ("linear2.bias", torch.Tensor([0.8427])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "6" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        state = OrderedDict(
            [
                (
                    "linear1.weight",
                    torch.Tensor([[-1.5893, -1.6656], [-0.7863, -0.8369]]),
                ),
                ("linear1.bias", torch.Tensor([1.3949, -0.3215])),
                ("linear2.weight", torch.Tensor([[-1.4555, 1.1600]])),
                ("linear2.bias", torch.Tensor([0.8730])),
            ]
        )
        net.load_state_dict(state)
        net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net

        checkpoint_name = "-".join(["checkpoint", "class", "7" + ".pt"])
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        dataset = BinaryDataset(use_gpu)

        return net_adjusted, dataset  # type: ignore

    parametrized_list: List[
        Tuple[Optional[str], DataInfluenceConstructor, str, bool]
    ] = [
        (
            "none",
            DataInfluenceConstructor(
                TracInCP, name="TracInCP_linear1", layers=["linear1"]
            ),
            "check_idx",
            False,
        ),
        (
            "none",
            DataInfluenceConstructor(TracInCP, name="TracInCP_all_layers"),
            "check_idx",
            False,
        ),
        (
            None,
            DataInfluenceConstructor(TracInCP, name="TracInCP_all_layers"),
            "sample_wise_trick",
            False,
        ),
        (
            None,
            DataInfluenceConstructor(
                TracInCP, name="TracInCP_linear1_linear2", layers=["linear1", "linear2"]
            ),
            "sample_wise_trick",
            False,
        ),
    ]

    if torch.cuda.is_available() and torch.cuda.device_count() != 0:
        parametrized_list.extend(
            [
                (
                    "none",
                    DataInfluenceConstructor(TracInCP, name="TracInCP_all_layers"),
                    "check_idx",
                    True,
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_linear1_linear2",
                        layers=["module.linear1", "module.linear2"],
                    ),
                    "check_idx",
                    True,
                ),
            ],
        )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func($parameter$args_to_skip
    # = ["reduction"])` to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        parametrized_list,
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_xor(
        self,
        reduction: Optional[str],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
        mode: str,
        use_gpu: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            batch_size = 4

            net, dataset = self._test_tracin_xor_setup(tmpdir, use_gpu)

            testset = F.normalize(torch.empty(100, 2).normal_(mean=0, std=0.5), dim=1)
            mask = ~torch.logical_xor(testset[:, 0] > 0, testset[:, 1] > 0)
            testlabels = (
                torch.where(mask, torch.tensor(1), torch.tensor(-1))
                .unsqueeze(1)
                .float()
            )
            if use_gpu:
                testset = testset.cuda()
                testlabels = testlabels.cuda()

            self.assertTrue(callable(tracin_constructor))

            if mode == "check_idx":

                self.assertTrue(isinstance(reduction, str))
                # pyre-fixme[22]: The cast is redundant.
                criterion = nn.MSELoss(reduction=cast(str, reduction))

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )
                # pyre-fixme[16]: `object` has no attribute `influence`.
                test_scores = tracin.influence((testset, testlabels))
                idx = torch.argsort(test_scores, dim=1, descending=True)
                # check that top 5 influences have matching binary classification
                for i in range(len(idx)):
                    influence_labels = dataset.labels[idx[i][0:5], 0]
                    self.assertTrue(torch.all(testlabels[i, 0] == influence_labels))

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=True,
                )
                test_scores = tracin.influence((testset, testlabels))
                test_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    (testset, testlabels)
                )
                assertTensorAlmostEqual(
                    self, test_scores, test_scores_sample_wise_trick
                )
