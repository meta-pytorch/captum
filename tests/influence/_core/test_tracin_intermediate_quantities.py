# pyre-unsafe
import tempfile
from typing import Callable

import torch
import torch.nn as nn
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction
from captum.influence._core.influence_function import NaiveInfluenceFunction
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


class TestTracInIntermediateQuantities(BaseTest):
    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("none", DataInfluenceConstructor(TracInCP)),
                ("none", DataInfluenceConstructor(NaiveInfluenceFunction)),
                ("none", DataInfluenceConstructor(ArnoldiInfluenceFunction)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_aggregate(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that calling `compute_intermediate_quantities` with `aggregate=True`
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

            intermediate_quantities = tracin.compute_intermediate_quantities(
                train_dataset, aggregate=False
            )
            aggregated_intermediate_quantities = tracin.compute_intermediate_quantities(
                train_dataset, aggregate=True
            )

            assertTensorAlmostEqual(
                self,
                torch.sum(intermediate_quantities, dim=0, keepdim=True),
                aggregated_intermediate_quantities,
                delta=1e-4,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )

    @parameterized.expand(
        [
            (reduction, constructor, unpack_inputs)
            for unpack_inputs in [True, False]
            for (reduction, constructor) in [
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
                ("none", DataInfluenceConstructor(TracInCP)),
                ("none", DataInfluenceConstructor(NaiveInfluenceFunction)),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_api(
        self, reduction: str, tracin_constructor: Callable, unpack_inputs: bool
    ) -> None:
        """
        tests that the result of calling the public method
        `compute_intermediate_quantities` for a DataLoader of batches is the same as
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

            # compute intermediate quantities using `compute_intermediate_quantities`
            # when passing in a single batch
            single_batch_intermediate_quantities = (
                tracin.compute_intermediate_quantities(single_batch)
            )

            # compute intermediate quantities using `compute_intermediate_quantities`
            # when passing in a dataloader with the same examples
            dataloader_intermediate_quantities = tracin.compute_intermediate_quantities(
                dataloader,
            )

            # the two self influences should be equal
            assertTensorAlmostEqual(
                self,
                single_batch_intermediate_quantities,
                dataloader_intermediate_quantities,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )

    @parameterized.expand(
        [
            (
                reduction,
                constructor,
                intermediate_quantities_tracin_constructor,
                unpack_inputs,
            )
            for unpack_inputs in [True, False]
            for (
                reduction,
                constructor,
                intermediate_quantities_tracin_constructor,
            ) in [
                (
                    "sum",
                    DataInfluenceConstructor(TracInCPFast),
                    DataInfluenceConstructor(TracInCPFastRandProj),
                ),
                (
                    "none",
                    DataInfluenceConstructor(TracInCP),
                    DataInfluenceConstructor(TracInCP),
                ),
                (
                    "none",
                    DataInfluenceConstructor(NaiveInfluenceFunction),
                    DataInfluenceConstructor(NaiveInfluenceFunction),
                ),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_consistent(
        self,
        reduction: str,
        tracin_constructor: Callable,
        intermediate_quantities_tracin_constructor: Callable,
        unpack_inputs: bool,
    ) -> None:
        """
        Since the influence score of a test batch on a training data should be the dot
        product of their intermediate quantities, checks that this is the case, by
        computing the influence score 2 different ways and checking they give the same
        results: 1) with the `influence` method, and by using the
        `compute_intermediate_quantities` method on the test and training data, and
        taking the dot product. No projection should be done.  Otherwise, the
        projection will cause error. For 1), we use an implementation that does not use
        intermediate quantities, i.e. `TracInCPFast`.  For 2), we use a method that
        does use intermediate quantities, i.e. `TracInCPFastRandProj`. Since the
        methods for the 2 cases are different, we need to parametrize the test with 2
        different tracin constructors. `tracin_constructor` is the constructor for the
        tracin implementation for case 1.  `intermediate_quantities_tracin_constructor`
        is the constructor for the tracin implementation for case 2. Note that we also
        use this test for implementations of `InfluenceFunctionBase`, where for the
        same method, both ways should give the same result by definition.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_features,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs, return_test_data=True)

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

            # create tracin instance which exposes `intermediate_quantities`
            intermediate_quantities_tracin = intermediate_quantities_tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # compute influence scores without using `compute_intermediate_quantities`
            test_batch = _format_batch_into_tuple(
                test_features, test_labels, unpack_inputs
            )
            scores = tracin.influence(
                test_batch,
            )

            # the influence score is the dot product of intermediate quantities
            intermediate_quantities_scores = torch.matmul(
                intermediate_quantities_tracin.compute_intermediate_quantities(
                    test_batch
                ),
                intermediate_quantities_tracin.compute_intermediate_quantities(
                    train_dataset
                ).T,
            )

            # the scores computed using the two methods should be the same
            assertTensorAlmostEqual(
                self,
                scores,
                intermediate_quantities_scores,
                delta=0.01,  # due to numerical issues, we can't set this to 0.0
                mode="max",
            )

    @parameterized.expand(
        [
            (reduction, constructor, projection_dim, unpack_inputs)
            for unpack_inputs in [False]
            for (reduction, constructor, projection_dim) in [
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), None),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), 2),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), 4),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), 9),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), 10),
                ("sum", DataInfluenceConstructor(TracInCPFastRandProj), 12),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_intermediate_quantities_projection_consistency(
        self,
        reduction: str,
        tracin_constructor: Callable,
        projection_dim: int,
        unpack_inputs: bool,
    ) -> None:
        """

        tests that the result of calling the public method
        "compute_intermediate_quantities" with TracInCPFastRandProj
        with/without projection_dim gives embedding of correct size.

        if projection_dim None, size should be dim of
        input to final layer * num classes * num checkpoints.
        otherwise it should be "at most" projection_dim * num checkpoints.
        See inline comments for "at most" caveat
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

            # create a single batch
            batch_size = 1
            single_batch = next(iter(DataLoader(train_dataset, batch_size=batch_size)))

            # NOW add projection_dim as a parameter passed in
            kwargs = {"projection_dim": projection_dim}

            # create tracin instance
            criterion = nn.MSELoss(reduction=reduction)
            tracin = tracin_constructor(
                net, train_dataset, tmpdir, batch_size, criterion, **kwargs
            )

            # compute intermediate quantities using `compute_intermediate_quantities`
            # when passing in a single batch
            single_batch_intermediate_quantities = (
                tracin.compute_intermediate_quantities(single_batch)
            )

            """
            net has
            in_features = 5,
            hidden_nodes (layer_input_dim) = 4,
            out_features (jacobian_dim) = 3
            and 5 checkpoints

            projection only happens
            (A) if project_dim < layer_input_dim * jacobian_dim  ( 4 * 3 = 12 here )

            also if jacobian_dim < int(sqrt(projection dim)),
            then jacobian_dim is not projected down
            similarly if layer_input_dim < int(sqrt(projection dim)),
            then it is not projected down

            in other words,
            jacobian_dim_post = min(jacobian_dim, int(sqrt(projection dim)))
            layer_input_dim_post = min(layer_input_dim, int(sqrt(projection dim)))

            and if not None and projection_dim < layer_input_dim * jacobian_dim
            (B) final_projection_dim =
                jacobian_dim_post * layer_input_dim_post * num_checkpoints


            if project dim = None we expect final dimension size of
            layer_input * jacobian_dim * num checkpoints = 4 * 3 * 5  = 60 dimension

            otherwise using (B) if
            project dim = 2  we expect 1 * 1 * 5 = 5
            project dim = 4  we expect 2 * 2 * 5 = 20
            project dim = 9  we expect 3 * 3 * 5 = 45
            project dim = 10 we expect 3 * 3 * 5 = 45
            project dim = 12 we expect 4 * 3 * 5 = 60 ( don't project since not (A))
            """

            # print(single_batch_intermediate_quantities.shape)
            expected_dim = {None: 60, 2: 5, 4: 20, 9: 45, 10: 45, 12: 60}
            self.assertEqual(
                expected_dim[projection_dim],
                single_batch_intermediate_quantities.shape[1],
            )
