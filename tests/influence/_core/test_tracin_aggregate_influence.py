# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

import tempfile
from typing import Callable

import torch
import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from captum.testing.helpers.influence.common import (
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)
from parameterized import parameterized
from torch.utils.data import DataLoader


class TestTracInAggregateInfluence(BaseTest):
    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
                (
                    "sum",
                    DataInfluenceConstructor(
                        TracInCP, sample_wise_grads_per_batch=True
                    ),
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_aggregate_influence(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that calling `influence` with `aggregate=True`
        does give the same result as calling it with `aggregate=False`, and then
        summing
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                return_test_data=False,
            )

            # create a dataloader that yields batches from the dataset
            train_dataset = DataLoader(train_dataset, batch_size=5)

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

            train_scores = tracin.influence(train_dataset, aggregate=False)
            aggregated_train_scores = tracin.influence(train_dataset, aggregate=True)

            assertTensorAlmostEqual(
                self,
                torch.sum(train_scores, dim=0, keepdim=True),
                aggregated_train_scores,
                delta=1e-3,  # due to numerical issues, we can't set this to 0.0
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
                        TracInCP, sample_wise_grads_per_batch=True
                    ),
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_aggregate_influence_api(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that the result of calling the public method
        `influence` when `aggregate` is true for a DataLoader of batches is the same as
        when the batches are collated into a single batch
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                return_test_data=False,
            )

            # create a single batch representing the entire dataset
            single_batch = next(
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

            # compute influence scores using `influence`
            # when passing in a single batch
            single_batch_aggregated_train_scores = tracin.influence(
                single_batch, aggregate=True
            )

            # compute influence scores using `influence`
            # when passing in a dataloader with the same examples
            dataloader_aggregated_train_scores = tracin.influence(
                dataloader, aggregate=True
            )

            # the two influence scores should be equal
            assertTensorAlmostEqual(
                self,
                single_batch_aggregated_train_scores,
                dataloader_aggregated_train_scores,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )
