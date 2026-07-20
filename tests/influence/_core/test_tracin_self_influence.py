# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import tempfile
from typing import Callable, Optional

import torch
import torch.nn as nn
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction
from captum.influence._core.influence_function import NaiveInfluenceFunction
from captum.influence._core.tracincp import TracInCP, TracInCPBase
from captum.influence._core.tracincp_fast_rand_proj import TracInCPFast
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
    GPU_SETTING_LIST,
    is_gpu,
)
from parameterized import parameterized
from torch.utils.data import DataLoader


# pyrefly: ignore [invalid-inheritance]
class TestTracInSelfInfluence(BaseTest):

    param_list = []

    # add the tests for `TracInCPBase` implementations and `InfluenceFunctionBase`
    # implementations separately, because the latter does not support `DataParallel`

    # add tests for `TracInCPBase` implementations

    for unpack_inputs in [True, False]:
        for gpu_setting in GPU_SETTING_LIST:
            for reduction, constructor in [
                (
                    "none",
                    DataInfluenceConstructor(TracInCP, name="TracInCP_all_layers"),
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_linear1",
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_data_parallel"
                            else ["linear1"]
                        ),
                    ),
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_linear1_linear2",
                        layers=(
                            ["module.linear1", "module.linear2"]
                            if gpu_setting == "cuda_data_parallel"
                            else ["linear1", "linear2"]
                        ),
                    ),
                ),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_sample_wise_grads_per_batch_all_layers",
                        sample_wise_grads_per_batch=True,
                    ),
                ),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFast, "TracInCPFast_last_fc_layer"
                    ),
                ),
                (
                    "mean",
                    DataInfluenceConstructor(
                        TracInCPFast, "TracInCPFast_last_fc_layer"
                    ),
                ),
            ]:
                if not (
                    "sample_wise_grads_per_batch" in constructor.kwargs
                    and constructor.kwargs["sample_wise_grads_per_batch"]
                    and is_gpu(gpu_setting)
                ):
                    param_list.append(
                        (reduction, constructor, unpack_inputs, gpu_setting)
                    )

    # add tests for `InfluenceFunctionBase` implementations
    gpu_setting_list = (
        ["", "cuda"]
        if torch.cuda.is_available() and torch.cuda.device_count() != 0
        else [""]
    )

    for unpack_inputs in [True, False]:
        for gpu_setting in gpu_setting_list:
            for reduction, constructor in [
                (
                    "none",
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction, name="NaiveInfluenceFunction_all_layers"
                    ),
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        name="NaiveInfluenceFunction_linear1",
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_data_parallel"
                            else ["linear1"]
                        ),
                    ),
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        ArnoldiInfluenceFunction,
                        name="ArnoldiInfluenceFunction_all_layers",
                    ),
                ),
                (
                    "none",
                    DataInfluenceConstructor(
                        ArnoldiInfluenceFunction,
                        name="ArnoldiInfluenceFunction_linear1",
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_data_parallel"
                            else ["linear1"]
                        ),
                    ),
                ),
            ]:
                if not (
                    "sample_wise_grads_per_batch" in constructor.kwargs
                    and constructor.kwargs["sample_wise_grads_per_batch"]
                    and is_gpu(gpu_setting)
                ):
                    param_list.append(
                        (reduction, constructor, unpack_inputs, gpu_setting)
                    )

    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(),
    )
    def test_tracin_self_influence(
        self,
        reduction: str,
        tracin_constructor: Callable,
        unpack_inputs: bool,
        gpu_setting: Optional[str],
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            # pyrefly: ignore [bad-unpacking]
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                False,
                gpu_setting,
            )

            # compute tracin_scores of training data on training data
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )
            train_scores = tracin.influence(
                _format_batch_into_tuple(
                    # pyrefly: ignore [missing-attribute]
                    train_dataset.samples,
                    # pyrefly: ignore [missing-attribute]
                    train_dataset.labels,
                    unpack_inputs,
                ),
                k=None,
            )
            # calculate self_tracin_scores
            self_tracin_scores = tracin.self_influence()

            # check that self_tracin scores equals the diagonal of influence scores
            assertTensorAlmostEqual(
                self,
                torch.diagonal(train_scores),
                self_tracin_scores,
                delta=0.01,
                mode="max",
            )

            # check that setting `outer_loop_by_checkpoints=False` and
            # `outer_loop_by_checkpoints=True` gives the same self influence scores
            # this test is only relevant for implementations of `TracInCPBase`, as
            # implementations of `InfluenceFunctionBase` do not use checkpoints.
            if isinstance(tracin, TracInCPBase):
                self_tracin_scores_by_checkpoints = tracin.self_influence(  # type: ignore
                    DataLoader(train_dataset, batch_size=batch_size),
                    # pyrefly: ignore [unexpected-keyword]
                    outer_loop_by_checkpoints=True,
                )
                assertTensorAlmostEqual(
                    self,
                    self_tracin_scores_by_checkpoints,
                    self_tracin_scores,
                    delta=0.01,
                    mode="max",
                )

    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP,
                        sample_wise_grads_per_batch=True,
                    ),
                ),
                ("sum", DataInfluenceConstructor(TracInCPFast)),
                ("mean", DataInfluenceConstructor(TracInCPFast)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_self_influence_dataloader_vs_single_batch(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        # tests that the result of calling the public method `self_influence` for a
        # DataLoader of batches is the same as when the batches are collated into a
        # single batch
        with tempfile.TemporaryDirectory() as tmpdir:
            # pyrefly: ignore [bad-unpacking]
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=False)

            # create a single batch representing the entire dataset
            single_batch = next(
                # pyrefly: ignore [bad-argument-type]
                iter(DataLoader(train_dataset, batch_size=len(train_dataset)))
            )

            # create a dataloader that yields batches from the dataset
            dataloader = DataLoader(train_dataset, batch_size=5)

            # create tracin instance
            criterion = nn.MSELoss(reduction=reduction)
            batch_size = 5
            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # compute self influence using `self_influence` when passing in a single
            # batch
            single_batch_self_influence = tracin.self_influence(single_batch)

            # compute self influence using `self_influence` when passing in a
            # dataloader with the same examples
            dataloader_self_influence = tracin.self_influence(dataloader)

            # the two self influences should be equal
            assertTensorAlmostEqual(
                self,
                single_batch_self_influence,
                dataloader_self_influence,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )
