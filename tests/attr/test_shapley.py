#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import unittest
import unittest.mock
import warnings
from collections.abc import Iterable, Sequence
from typing import Any, Callable, List, Tuple, Union

import torch
from captum._utils.typing import BaselineType, Tensor, TensorOrTupleOfTensorsGeneric
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModel_MultiLayer_MultiInput_with_Future,
    BasicModel_MultiLayer_with_Future,
    BasicModelBoolInput,
    BasicModelBoolInput_with_Future,
)
from parameterized import parameterized
from torch.futures import Future


class Test(BaseTest):
    @parameterized.expand(
        [
            ("dense_ids", torch.tensor([[0, 0, 1, 2, 2]]), 7),
            ("sparse_ids", torch.tensor([[0, 0, 2, 2, 2]]), 4),
        ]
    )
    def test_expected_forward_count_matches_execution(
        self, _name: str, feature_mask: Tensor, expected_forwards: int
    ) -> None:
        completed_forwards = 0

        def forward(inputs: Tensor) -> Tensor:
            nonlocal completed_forwards
            completed_forwards += 1
            return inputs.sum(dim=1)

        shapley = ShapleyValueSampling(forward)
        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        kwargs = {
            "feature_mask": feature_mask,
            "n_samples": 3,
            "perturbations_per_eval": 2,
        }
        planned_forwards = shapley.expected_forward_count(inputs, **kwargs)
        shapley.attribute(inputs, **kwargs)

        self.assertEqual(planned_forwards, expected_forwards)
        self.assertEqual(completed_forwards, expected_forwards)

    @parameterized.expand([True, False])
    def test_shapley_sampling_skips_absent_feature_ids(self, use_future: bool) -> None:
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        completed_forwards = 0

        def forward(inputs: Tensor) -> Tensor:
            nonlocal completed_forwards
            completed_forwards += 1
            return (inputs * weights).sum(dim=1)

        def forward_future(inputs: Tensor) -> Future[Tensor]:
            fut: Future[Tensor] = Future()
            fut.set_result(forward(inputs))
            return fut

        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]])
        feature_mask = torch.tensor([[-1, 3, 3, 7, 7]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if use_future:
                attributions = (
                    ShapleyValueSampling(forward_future)
                    .attribute_future(inputs, feature_mask=feature_mask, n_samples=2)
                    .wait()
                )
            else:
                attributions = ShapleyValueSampling(forward).attribute(
                    inputs, feature_mask=feature_mask, n_samples=2
                )

        self.assertEqual(caught, [])
        # 2 present feature IDs * 2 samples + 1 initial eval.
        self.assertEqual(completed_forwards, 5)
        assertTensorAlmostEqual(
            self,
            attributions,
            torch.tensor(
                [[0.0, 13.0, 13.0, 41.0, 41.0], [0.0, 17.0, 17.0, 13.0, 13.0]]
            ),
        )

    def test_shapley_values_skips_absent_feature_ids(self) -> None:
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        completed_forwards = 0

        def forward(inputs: Tensor) -> Tensor:
            nonlocal completed_forwards
            completed_forwards += 1
            return (inputs * weights).sum(dim=1)

        inputs = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            attributions = ShapleyValues(forward).attribute(
                inputs, feature_mask=torch.tensor([[0, 0, 3, 3, 3]])
            )

        self.assertEqual(caught, [])
        # 2! permutations * 2 present feature IDs + 1 initial eval.
        self.assertEqual(completed_forwards, 5)
        assertTensorAlmostEqual(
            self,
            attributions,
            torch.tensor([[5.0, 5.0, 50.0, 50.0, 50.0]]),
        )

    def test_shapley_values_feature_count_warning_ignores_absent_ids(self) -> None:
        class StopAttribution(Exception):
            pass

        def forward(inputs: Tensor) -> Tensor:
            # Stop before enumerating permutations, which would take 11! orders
            # if absent IDs were counted.
            raise StopAttribution

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with self.assertRaises(StopAttribution):
                ShapleyValues(forward).attribute(
                    torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]),
                    feature_mask=torch.tensor([[0, 0, 10, 10, 10]]),
                )

        self.assertEqual(caught, [])

    def test_expected_forward_count_rejects_unplanned_subclass(self) -> None:
        class UnplannedShapley(ShapleyValueSampling):
            pass

        with self.assertRaisesRegex(NotImplementedError, "exact forward plan"):
            UnplannedShapley(BasicModel_MultiLayer()).expected_forward_count(
                torch.tensor([[1.0, 2.0]])
            )

    def test_expected_forward_count_rejects_custom_permutation_generator(self) -> None:
        shapley = ShapleyValueSampling(BasicModel_MultiLayer())

        def custom_generator(
            num_features: int, num_samples: int
        ) -> Iterable[Sequence[int]]:
            del num_features, num_samples
            return ()

        shapley.permutation_generator = custom_generator

        with self.assertRaisesRegex(NotImplementedError, "permutation_generator"):
            shapley.expected_forward_count(torch.tensor([[1.0, 2.0]]))

    @parameterized.expand([True, False])
    def test_simple_shapley_sampling(self, use_future: bool) -> None:
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        if use_future:
            net_fut = BasicModel_MultiLayer_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(1, 2, 3),
                n_samples=250,
            )
        else:
            net = BasicModel_MultiLayer()
            self._shapley_test_assert(
                net,
                inp,
                [[76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(1, 2, 3),
                n_samples=250,
            )

    @parameterized.expand([True, False])
    def test_simple_shapley_sampling_with_mask(self, use_future: bool) -> None:
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        if use_future:
            net_fut = BasicModel_MultiLayer_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[275.0, 275.0, 115.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(1, 2, 3),
            )
        else:
            net = BasicModel_MultiLayer()
            self._shapley_test_assert(
                net,
                inp,
                [[275.0, 275.0, 115.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(1, 2, 3),
            )

    @parameterized.expand([True, False])
    def test_simple_shapley_sampling_boolean(self, use_future: bool) -> None:
        inp = torch.tensor([[True, False, True]])
        if use_future:
            net_fut = BasicModelBoolInput_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[35.0, 35.0, 35.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(1, 2, 3),
            )
        else:
            net = BasicModelBoolInput()
            self._shapley_test_assert(
                net,
                inp,
                [[35.0, 35.0, 35.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(1, 2, 3),
            )

    @parameterized.expand([True, False])
    def test_simple_shapley_sampling_boolean_with_baseline(
        self, use_future: bool
    ) -> None:
        inp = torch.tensor([[True, False, True]])
        if use_future:
            net_fut = BasicModelBoolInput_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[-40.0, -40.0, 0.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                baselines=True,
                perturbations_per_eval=(1, 2, 3),
            )
        else:
            net = BasicModelBoolInput()
            self._shapley_test_assert(
                net,
                inp,
                [[-40.0, -40.0, 0.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                baselines=True,
                perturbations_per_eval=(1, 2, 3),
            )

    @parameterized.expand([True, False])
    def test_simple_shapley_sampling_with_baselines(self, use_future: bool) -> None:
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        if use_future:
            net_fut = BasicModel_MultiLayer_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[248.0, 248.0, 104.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                baselines=4,
                perturbations_per_eval=(1, 2, 3),
            )
        else:
            net = BasicModel_MultiLayer()
            self._shapley_test_assert(
                net,
                inp,
                [[248.0, 248.0, 104.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                baselines=4,
                perturbations_per_eval=(1, 2, 3),
            )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_sampling(self, use_future: bool) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]])
        if use_future:
            net_fut = BasicModel_MultiLayer_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[7.0, 32.5, 10.5], [76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(1, 2, 3),
                n_samples=200,
            )
        else:
            net = BasicModel_MultiLayer()
            self._shapley_test_assert(
                net,
                inp,
                [[7.0, 32.5, 10.5], [76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(1, 2, 3),
                n_samples=200,
            )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_sampling_with_mask(self, use_future: bool) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        if use_future:
            net_fut = BasicModel_MultiLayer_with_Future()
            self._shapley_test_assert_future(
                net_fut,
                inp,
                [[39.5, 39.5, 10.5], [275.0, 275.0, 115.0]],
                feature_mask=mask,
                perturbations_per_eval=(1, 2, 3),
            )
        else:
            net = BasicModel_MultiLayer()
            self._shapley_test_assert(
                net,
                inp,
                [[39.5, 39.5, 10.5], [275.0, 275.0, 115.0]],
                feature_mask=mask,
                perturbations_per_eval=(1, 2, 3),
            )

    def test_multi_input_shapley_sampling_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected: Tuple[List[List[float]], ...] = (
            [[90.0, 0.0, 0.0], [78.0, 198.0, 118.0]],
            [[78.0, 0.0, 198.0], [0.0, 398.0, 0.0]],
            [[0.0, 398.0, 38.0], [0.0, 38.0, 0.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=200,
            test_true_shapley=False,
        )

    def test_multi_input_shapley_sampling_without_mask_future(self) -> None:
        net = BasicModel_MultiLayer_MultiInput_with_Future()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected_future: Tuple[List[List[float]], ...] = (
            [[90.0, 0.0, 0.0], [78.0, 198.0, 118.0]],
            [[78.0, 0.0, 198.0], [0.0, 398.0, 0.0]],
            [[0.0, 398.0, 38.0], [0.0, 38.0, 0.0]],
        )
        self._shapley_test_assert_future(
            net,
            (inp1, inp2, inp3),
            expected_future,
            additional_input=(1,),
            n_samples=200,
            test_true_shapley=False,
        )

    def test_multi_input_shapley_sampling_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected_mask: Tuple[List[List[float]], ...] = (
            [[1088.6666, 1088.6666, 1088.6666], [255.0, 595.0, 255.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 595.0, 0.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 255.0, 255.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_mask,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline: Tuple[List[List[float]], ...] = (
            [[1040.0, 1040.0, 1040.0], [184.0, 580.0, 184.0]],
            [[52.0, 1040.0, 132.0], [184.0, 580.0, -12.0]],
            [[52.0, 1040.0, 132.0], [184.0, 184.0, 184.0]],
        )
        self._shapley_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_shapley_sampling_with_mask_future(self) -> None:
        net = BasicModel_MultiLayer_MultiInput_with_Future()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected_mask_fut: Tuple[List[List[float]], ...] = (
            [[1088.6666, 1088.6666, 1088.6666], [255.0, 595.0, 255.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 595.0, 0.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 255.0, 255.0]],
        )
        self._shapley_test_assert_future(
            net,
            (inp1, inp2, inp3),
            expected_mask_fut,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline_fut: Tuple[List[List[float]], ...] = (
            [[1040.0, 1040.0, 1040.0], [184.0, 580.0, 184.0]],
            [[52.0, 1040.0, 132.0], [184.0, 580.0, -12.0]],
            [[52.0, 1040.0, 132.0], [184.0, 184.0, 184.0]],
        )
        self._shapley_test_assert_future(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline_fut,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    @parameterized.expand([True, False])
    def test_shapley_sampling_multi_task_output(self, use_future: bool) -> None:
        # return shape (batch size, 2)
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        if use_future:
            net1_fut: BasicModel_MultiLayer_with_Future = (
                BasicModel_MultiLayer_with_Future()
            )

            def forward_func(*args: Any, **kwargs: Any) -> Future[torch.Tensor]:
                net_output = net1_fut(*args, **kwargs)
                net_output.wait()
                batch_size = net_output.value().size(0)
                constant = torch.ones(batch_size, 2)
                output = torch.cat(
                    [
                        net_output.value(),
                        constant,
                    ],
                    dim=-1,
                )
                fut: Future[torch.Tensor] = Future()
                fut.set_result(output)
                return fut

            self._shapley_test_assert_future(
                forward_func,
                inp,
                [
                    [
                        [76.66666, 196.66666, 116.66666],
                        [76.66666, 196.66666, 116.66666],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ],
                target=None,  # no target, multi-task output for all classes
                perturbations_per_eval=(1, 2, 3),
                n_samples=150,
                test_true_shapley=True,
            )
        else:
            net1: BasicModel_MultiLayer = BasicModel_MultiLayer()

            def forward_func_sync(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                net_output = net1(*args, **kwargs)
                batch_size = net_output.size(0)
                constant = torch.ones(batch_size, 2)
                output = torch.cat(
                    [
                        net_output,
                        constant,
                    ],
                    dim=-1,
                )
                return output

            # return shape (batch size, 4)
            self._shapley_test_assert(
                forward_func_sync,
                inp,
                [
                    [
                        [76.66666, 196.66666, 116.66666],
                        [76.66666, 196.66666, 116.66666],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ],
                target=None,  # no target, multi-task output for all classes
                perturbations_per_eval=(1, 2, 3),
                n_samples=150,
                test_true_shapley=True,
            )

    @parameterized.expand([True, False])
    def test_shapley_sampling_multi_task_output_with_mask(
        self, use_future: bool
    ) -> None:
        # return shape (batch size, 2)
        inp = torch.tensor([[20.0, 50.0, 30.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[1, 1, 0], [0, 1, 1]])
        if use_future:
            net1_fut: BasicModel_MultiLayer_with_Future = (
                BasicModel_MultiLayer_with_Future()
            )

            # return shape (batch size, 4)
            def forward_func(*args: Any, **kwargs: Any) -> Future[torch.Tensor]:
                net_output = net1_fut(*args, **kwargs)
                net_output.wait()
                batch_size = net_output.value().size(0)
                constant = torch.ones(batch_size, 1)

                output = torch.cat(
                    [
                        net_output.value(),
                        constant,
                    ],
                    dim=-1,
                )
                fut: Future[torch.Tensor] = Future()
                fut.set_result(output)
                return fut

            self._shapley_test_assert_future(
                forward_func,
                inp,
                [
                    [
                        [275.0, 275.0, 115.0],
                        [275.0, 275.0, 115.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [75.0, 315.0, 315.0],
                        [75.0, 315.0, 315.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                target=None,  # no target, multi-task output for all classes
                perturbations_per_eval=(1, 2, 3),
                n_samples=150,
                test_true_shapley=True,
                feature_mask=mask,
            )
        else:

            net1: BasicModel_MultiLayer = BasicModel_MultiLayer()

            # return shape (batch size, 4)
            def forward_func_sync(*args: torch.Tensor, **kwargs: Any) -> torch.Tensor:
                net_output = net1(*args, **kwargs)
                batch_size = net_output.size(0)
                constant = torch.ones(batch_size, 1)

                output = torch.cat(
                    [
                        net_output,
                        constant,
                    ],
                    dim=-1,
                )
                return output

            self._shapley_test_assert(
                forward_func_sync,
                inp,
                [
                    [
                        [275.0, 275.0, 115.0],
                        [275.0, 275.0, 115.0],
                        [0.0, 0.0, 0.0],
                    ],
                    [
                        [75.0, 315.0, 315.0],
                        [75.0, 315.0, 315.0],
                        [0.0, 0.0, 0.0],
                    ],
                ],
                target=None,  # no target, multi-task output for all classes
                perturbations_per_eval=(1, 2, 3),
                n_samples=150,
                test_true_shapley=True,
                feature_mask=mask,
            )

    # Remaining tests are for cases where forward function returns a scalar
    # per batch, as either a float, integer, 0d tensor or 1d tensor.
    @parameterized.expand([True, False])
    def test_single_shapley_batch_scalar_float(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> float:
            return float(torch.sum(net(inp)).item())

        def func_future(inp: torch.Tensor) -> Future[float]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[float] = Future()
            fut.set_result(float(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[float, Future[float]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_batch_scalar_tensor_0d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp))

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_batch_scalar_tensor_1d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp)).reshape(1)

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()).reshape(1))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_batch_scalar_tensor_int(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> int:
            return int(torch.sum(net(inp)).item())

        def func_future(inp: torch.Tensor) -> Future[int]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[int] = Future()
            fut.set_result(int(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[int, Future[int]]] = func_future
        else:
            func_to_use = func
        self._single_input_one_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_int_batch_scalar_float(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> float:
            return float(torch.sum(net(inp.float())).item())

        def func_future(inp: torch.Tensor) -> Future[float]:
            temp = net_fut(inp.float())
            temp.wait()
            fut: Future[float] = Future()
            fut.set_result(float(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[float, Future[float]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_int_batch_scalar_tensor_0d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp.float()))

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp.float())
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_int_batch_scalar_tensor_1d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp.float())).reshape(1)

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp.float())
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()).reshape(1))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_single_shapley_int_batch_scalar_tensor_int(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> int:
            return int(torch.sum(net(inp.float())).item())

        def func_future(inp: torch.Tensor) -> Future[int]:
            temp = net_fut(inp.float())
            temp.wait()
            fut: Future[int] = Future()
            fut.set_result(int(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[int, Future[int]]] = func_future
        else:
            func_to_use = func
        self._single_int_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_batch_scalar_float(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> float:
            return float(torch.sum(net(inp)).item())

        def func_future(inp: torch.Tensor) -> Future[float]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[float] = Future()
            fut.set_result(float(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[float, Future[float]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_batch_scalar_tensor_0d(
        self, use_future: bool
    ) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp))

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_batch_scalar_tensor_1d(
        self, use_future: bool
    ) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(inp)).reshape(1)

        def func_future(inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()).reshape(1))
            return fut

        if use_future:
            func_to_use: Callable[
                [torch.Tensor], Union[torch.Tensor, Future[torch.Tensor]]
            ] = func_future
        else:
            func_to_use = func
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_sample_shapley_batch_scalar_tensor_int(
        self, use_future: bool
    ) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()
        net_fut: BasicModel_MultiLayer_with_Future = BasicModel_MultiLayer_with_Future()

        def func(inp: torch.Tensor) -> int:
            return int(torch.sum(net(inp)).item())

        def func_future(inp: torch.Tensor) -> Future[int]:
            temp = net_fut(inp)
            temp.wait()
            fut: Future[int] = Future()
            fut.set_result(int(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[[torch.Tensor], Union[int, Future[int]]] = func_future
        else:
            func_to_use = func
        self._single_input_multi_sample_batch_scalar_shapley_assert(
            lambda inp: func_to_use(inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_inp_shapley_batch_scalar_float(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer_MultiInput = BasicModel_MultiLayer_MultiInput()
        net_fut: BasicModel_MultiLayer_MultiInput_with_Future = (
            BasicModel_MultiLayer_MultiInput_with_Future()
        )

        def func(*inp: torch.Tensor) -> float:
            return float(torch.sum(net(*inp)).item())

        def func_future(*inp: torch.Tensor) -> Future[float]:
            temp = net_fut(*inp)
            temp.wait()
            fut: Future[float] = Future()
            fut.set_result(float(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[..., Union[float, Future[float]]] = func_future
        else:
            func_to_use = func
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: func_to_use(*inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_inp_shapley_batch_scalar_tensor_0d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer_MultiInput = BasicModel_MultiLayer_MultiInput()
        net_fut: BasicModel_MultiLayer_MultiInput_with_Future = (
            BasicModel_MultiLayer_MultiInput_with_Future()
        )

        def func(*inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(*inp))

        def func_future(*inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(*inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()))
            return fut

        if use_future:
            func_to_use: Callable[..., Union[torch.Tensor, Future[torch.Tensor]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: func_to_use(*inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_multi_inp_shapley_batch_scalar_tensor_1d(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer_MultiInput = BasicModel_MultiLayer_MultiInput()
        net_fut: BasicModel_MultiLayer_MultiInput_with_Future = (
            BasicModel_MultiLayer_MultiInput_with_Future()
        )

        def func(*inp: torch.Tensor) -> torch.Tensor:
            return torch.sum(net(*inp)).reshape(1)

        def func_future(*inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(*inp)
            temp.wait()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.sum(temp.value()).reshape(1))
            return fut

        if use_future:
            func_to_use: Callable[..., Union[torch.Tensor, Future[torch.Tensor]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: func_to_use(*inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_mutli_inp_shapley_batch_scalar_tensor_int(self, use_future: bool) -> None:
        net: BasicModel_MultiLayer_MultiInput = BasicModel_MultiLayer_MultiInput()
        net_fut: BasicModel_MultiLayer_MultiInput_with_Future = (
            BasicModel_MultiLayer_MultiInput_with_Future()
        )

        def func(*inp: torch.Tensor) -> int:
            return int(torch.sum(net(*inp)).item())

        def func_future(*inp: torch.Tensor) -> Future[int]:
            temp = net_fut(*inp)
            temp.wait()
            fut: Future[int] = Future()
            fut.set_result(int(torch.sum(temp.value()).item()))
            return fut

        if use_future:
            func_to_use: Callable[..., Union[int, Future[int]]] = func_future
        else:
            func_to_use = func
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: func_to_use(*inp), use_future=use_future
        )

    @parameterized.expand([True, False])
    def test_mutli_inp_shapley_batch_scalar_tensor_expanded(
        self, use_future: bool
    ) -> None:
        net: BasicModel_MultiLayer_MultiInput = BasicModel_MultiLayer_MultiInput()
        net_fut: BasicModel_MultiLayer_MultiInput_with_Future = (
            BasicModel_MultiLayer_MultiInput_with_Future()
        )

        def func(*inp: torch.Tensor) -> torch.Tensor:
            sum_val = torch.sum(net(*inp)).item()
            return torch.tensor([sum_val, sum_val + 2.0, sum_val + 3.0])

        def func_future(*inp: torch.Tensor) -> Future[torch.Tensor]:
            temp = net_fut(*inp)
            temp.wait()
            sum_val = torch.sum(temp.value()).item()
            fut: Future[torch.Tensor] = Future()
            fut.set_result(torch.tensor([sum_val, sum_val + 2.0, sum_val + 3.0]))
            return fut

        if use_future:
            func_to_use: Callable[..., Union[torch.Tensor, Future[torch.Tensor]]] = (
                func_future
            )
        else:
            func_to_use = func
        self._multi_input_batch_scalar_shapley_assert(
            lambda *inp: func_to_use(*inp), use_future=use_future, expanded_output=True
        )

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_shapley_sampling_with_show_progress(
        self, mock_stderr: io.StringIO
    ) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._shapley_test_assert(
                net,
                inp,
                [[76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(bsz,),
                n_samples=250,
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Shapley Value Sampling attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"
            assert (
                "Shapley Values attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_shapley_sampling_with_mask_and_show_progress(
        self, mock_stderr: io.StringIO
    ) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._shapley_test_assert(
                net,
                inp,
                [[275.0, 275.0, 115.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(bsz,),
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Shapley Value Sampling attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"
            assert (
                "Shapley Values attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    def _single_input_one_sample_batch_scalar_shapley_assert(
        self,
        func: Callable[..., Any],
        use_future: bool = False,
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])
        if use_future:
            self._shapley_test_assert_future(
                func,
                inp,
                [[79.0, 79.0, 21.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
            )
        else:
            self._shapley_test_assert(
                func,
                inp,
                [[79.0, 79.0, 21.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
            )

    def _single_input_multi_sample_batch_scalar_shapley_assert(
        self,
        func: Callable[..., Any],
        use_future: bool = False,
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])
        if use_future:
            self._shapley_test_assert_future(
                func,
                inp,
                [[629.0, 629.0, 251.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
                n_samples=2500,
            )
        else:
            self._shapley_test_assert(
                func,
                inp,
                [[629.0, 629.0, 251.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
                n_samples=2500,
            )

    def _single_int_input_multi_sample_batch_scalar_shapley_assert(
        self,
        func: Callable[..., Any],
        use_future: bool = False,
    ) -> None:
        inp = torch.tensor([[2, 10, 3], [20, 50, 30]])
        mask = torch.tensor([[0, 0, 1]])
        if use_future:
            self._shapley_test_assert_future(
                func,
                inp,
                [[629.0, 629.0, 251.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
                n_samples=2500,
            )
        else:
            self._shapley_test_assert(
                func,
                inp,
                [[629.0, 629.0, 251.0]],
                feature_mask=mask,
                perturbations_per_eval=(1,),
                target=None,
                n_samples=2500,
            )

    def _multi_input_batch_scalar_shapley_assert(
        self,
        func: Callable[..., Any],
        use_future: bool = False,
        expanded_output: bool = False,
    ) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [20.0, 10.0, 13.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        out_mult = 3 if expanded_output else 1
        expected = (
            [[3850.6666, 3850.6666, 3850.6666]] * out_mult,
            [[306.6666, 3850.6666, 410.6666]] * out_mult,
            [[306.6666, 3850.6666, 410.6666]] * out_mult,
        )
        if use_future:
            self._shapley_test_assert_future(
                func,
                (inp1, inp2, inp3),
                expected,
                additional_input=(1,),
                feature_mask=(mask1, mask2, mask3),
                perturbations_per_eval=(1,),
                target=None,
                n_samples=3500,
                delta=1.2,
            )
        else:
            self._shapley_test_assert(
                func,
                (inp1, inp2, inp3),
                expected,
                additional_input=(1,),
                feature_mask=(mask1, mask2, mask3),
                perturbations_per_eval=(1,),
                target=None,
                n_samples=3500,
                delta=1.2,
            )

    def _shapley_test_assert(
        self,
        model: Callable[..., Union[int, float, Tensor]],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr: Union[
            List[List[float]], Tuple[List[List[float]], ...], List[List[List[float]]]
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
        test_true_shapley: bool = True,
        show_progress: bool = False,
    ) -> None:
        for batch_size in perturbations_per_eval:
            shapley_samp = ShapleyValueSampling(model)
            attributions = shapley_samp.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                n_samples=n_samples,
                show_progress=show_progress,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_attr, delta=delta, mode="max"
            )
            if test_true_shapley:
                shapley_val = ShapleyValues(model)
                attributions = shapley_val.attribute(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    show_progress=show_progress,
                )
                assertTensorTuplesAlmostEqual(
                    self, attributions, expected_attr, mode="max", delta=0.001
                )

    def _shapley_test_assert_future(
        self,
        model: Callable[
            ..., Union[int, float, Tensor, Future[int], Future[float], Future[Tensor]]
        ],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr: Union[
            List[List[float]], Tuple[List[List[float]], ...], List[List[List[float]]]
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
        # leaving this false as it is not supported for future
        test_true_shapley: bool = False,
        show_progress: bool = False,
    ) -> None:
        for batch_size in perturbations_per_eval:
            shapley_samp = ShapleyValueSampling(model)  # pyre-ignore[6]
            attributions = shapley_samp.attribute_future(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                n_samples=n_samples,
                show_progress=show_progress,
            )
            attributions.wait()
            assertTensorTuplesAlmostEqual(
                self, attributions.value(), expected_attr, delta=delta, mode="max"
            )
            if test_true_shapley:
                shapley_val = ShapleyValues(model)  # pyre-ignore[6]
                attributions = shapley_val.attribute_future(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    show_progress=show_progress,
                )
                attributions.wait()
                assertTensorTuplesAlmostEqual(
                    self, attributions.value(), expected_attr, mode="max", delta=0.001
                )


if __name__ == "__main__":
    unittest.main()
