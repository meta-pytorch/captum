# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
from typing import Callable

import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)
from parameterized import parameterized
from torch.utils.data import DataLoader


class TestTracInDataLoader(BaseTest):
    """
    This tests that the influence score computed when a Dataset is fed to the
    `self.tracin_constructor` and when a DataLoader constructed using the same
    Dataset is fed to `self.tracin_constructor` gives the same results.
    """

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `comprehension((reduction, constr, unpack_inputs) for
    # generators(generator(unpack_inputs in [False, True] if ),
    # generators(generator((reduction, constr) in
    # [("none", captum.testing.helpers.influence.common.DataInfluenceConstructor
    # (captum.influence._core.tracincp.TracInCP)),
    # ("sum", captum.testing.helpers.influence.common.DataInfluenceConstructor
    # (captum.influence._core.tracincp_fast_rand_proj.TracInCPFast)), ("sum",
    # captum.testing.helpers.influence.common.DataInfluenceConstructor(captum.influence._core.
    # tracincp_fast_rand_proj.TracInCPFastRandProj)), ("sum",
    # captum.testing.helpers.influence.common.DataInfluenceConstructor(
    # captum.influence._core.tracincp_fast_rand_proj.TracInCPFastRandProj,
    # $parameter$name = "TracInCPFastRandProj_1DProj",
    # $parameter$projection_dim = 1))] if ))))`
    # to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        [
            (
                reduction,
                constr,
                unpack_inputs,
            )
            for unpack_inputs in [False, True]
            for reduction, constr in [
                ("none", DataInfluenceConstructor(TracInCP)),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj,
                        name="TracInCPFastRandProj_1DProj",
                        projection_dim=1,
                    ),
                ),
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_dataloader(
        self,
        reduction: str,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
        unpack_inputs: bool,
    ) -> None:

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 5

            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

            self.assertTrue(isinstance(reduction, str))
            criterion = nn.MSELoss(reduction=reduction)

            self.assertTrue(callable(tracin_constructor))
            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # pyre-fixme[16]: `object` has no attribute `influence`.
            train_scores = tracin.influence(
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=None,
            )

            tracin_dataloader = tracin_constructor(
                net,
                DataLoader(train_dataset, batch_size=batch_size, shuffle=False),
                tmpdir,
                None,
                criterion,
            )

            train_scores_dataloader = tracin_dataloader.influence(
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=None,
            )

            assertTensorAlmostEqual(
                self,
                train_scores,
                train_scores_dataloader,
                delta=0.0,
                mode="max",
            )
