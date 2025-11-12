#!/usr/bin/env python3

# pyre-strict

import io
import threading
import time
import unittest
import unittest.mock
from typing import Any, cast, List, Tuple, Union

import torch
from captum._utils.common import _construct_future_forward
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.attribution import Attribution
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModelBoolInput,
    BasicModelWithSparseInputs,
)
from torch import Tensor


class Test(BaseTest):
    r"""
    The following conversion tests are underlying assumptions
    made by the rest of tests in this file.

    We are testing them explicitly just in case they break behaviour
    in the future. As in this case it will be easier to update the tests.
    """

    def test_python_float_conversion(self) -> None:
        x = torch.tensor(3, dtype=cast(torch.dtype, float))
        self.assertEqual(x.dtype, torch.float64)

    def test_python_int_conversion(self) -> None:
        x = torch.tensor(5, dtype=cast(torch.dtype, int))
        self.assertEqual(x.dtype, torch.int64)

    def test_float32_tensor_item_conversion(self) -> None:
        x = torch.tensor(5, dtype=torch.float32)
        y = torch.tensor(x.item())  # .item() returns a python float

        # for whatever reason it is only
        # dtype == torch.float64 if you provide dtype=float
        self.assertEqual(y.dtype, torch.float32)

    def test_int32_tensor_item_conversion(self) -> None:
        x = torch.tensor(5, dtype=torch.int32)
        y = torch.tensor(x.item())  # .item() returns a python int
        self.assertEqual(y.dtype, torch.int64)

    def test_simple_ablation(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            ablation_algo, inp, [[80.0, 200.0, 120.0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_int(self) -> None:
        ablation_algo = FeatureAblation(BasicModel())
        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            ablation_algo, inp, [[-3, 0, 0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_int_to_int_nt(self) -> None:
        ablation_algo = NoiseTunnel(FeatureAblation(BasicModel()))
        inp = torch.tensor([[-3, 1, 2]]).float()
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[-3.0, 0.0, 0.0]],
            perturbations_per_eval=(1, 2, 3),
            stdevs=1e-10,
        )

    def test_simple_ablation_int_to_float(self) -> None:
        net: BasicModel = BasicModel()

        def wrapper_func(inp: Tensor) -> Tensor:
            return net(inp).float()

        ablation_algo = FeatureAblation(wrapper_func)

        inp = torch.tensor([[-3, 1, 2]])
        self._ablation_test_assert(
            ablation_algo, inp, [[-3.0, 0.0, 0.0]], perturbations_per_eval=(1, 2, 3)
        )

    def test_simple_ablation_with_mask(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[280.0, 280.0, 120.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_with_baselines(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[248.0, 248.0, 104.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_boolean(self) -> None:
        ablation_algo = FeatureAblation(BasicModelBoolInput())
        inp = torch.tensor([[True, False, True]])
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[40.0, 40.0, 40.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_ablation_boolean_with_baselines(self) -> None:
        ablation_algo = FeatureAblation(BasicModelBoolInput())
        inp = torch.tensor([[True, False, True]])
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[-40.0, -40.0, 0.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=True,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[8.0, 35.0, 12.0], [80.0, 200.0, 120.0]],
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_mask(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[41.0, 41.0, 12.0], [280.0, 280.0, 120.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_mask_weighted(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        ablation_algo.use_weights = True
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]])
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            ablation_algo,
            inp,
            [[41.0, 41.0, 12.0], [280.0, 280.0, 120.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation_with_mask(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer_MultiInput())
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[3, 4, 2]])
        mask3 = torch.tensor([[5, 6, 7], [5, 5, 5]])
        expected = (
            [[492.0, 492.0, 492.0], [200.0, 200.0, 200.0]],
            [[80.0, 200.0, 120.0], [0.0, 400.0, 0.0]],
            [[0.0, 400.0, 40.0], [60.0, 60.0, 60.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2),
            expected[0:1],
            additional_input=(inp3, 1),
            feature_mask=(mask1, mask2),
            perturbations_per_eval=(1, 2, 3),
        )
        expected_with_baseline = (
            [[468.0, 468.0, 468.0], [184.0, 192.0, 184.0]],
            [[68.0, 188.0, 108.0], [-12.0, 388.0, -12.0]],
            [[-16.0, 384.0, 24.0], [12.0, 12.0, 12.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation_with_int_input_tensor_and_float_baseline(
        self,
    ) -> None:
        def sum_forward(*inps: torch.Tensor) -> torch.Tensor:
            flattened = [torch.flatten(inp, start_dim=1) for inp in inps]
            return torch.cat(flattened, dim=1).sum(1)

        ablation_algo = FeatureAblation(sum_forward)
        inp1 = torch.tensor([[0, 1], [3, 4]])
        inp2 = torch.tensor(
            [
                [[0.1, 0.2], [0.3, 0.2]],
                [[0.4, 0.5], [0.3, 0.2]],
            ]
        )
        inp3 = torch.tensor([[0], [1]])

        expected = (
            torch.tensor([[-0.2, 0.8], [2.8, 3.8]]),
            torch.tensor(
                [
                    [[-3.0, -2.9], [-2.8, -2.9]],
                    [[-2.7, -2.6], [-2.8, -2.9]],
                ]
            ),
            torch.tensor([[-0.4], [0.6]]),
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected,
            target=None,
            baselines=(0.2, 3.1, 0.4),
        )

    def test_multi_input_ablation_with_mask_weighted(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer_MultiInput())
        ablation_algo.use_weights = True
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[3, 4, 2]])
        mask3 = torch.tensor([[5, 6, 7], [5, 5, 5]])
        expected = (
            [[492.0, 492.0, 492.0], [200.0, 200.0, 200.0]],
            [[80.0, 200.0, 120.0], [0.0, 400.0, 0.0]],
            [[0.0, 400.0, 40.0], [60.0, 60.0, 60.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2),
            expected[0:1],
            additional_input=(inp3, 1),
            feature_mask=(mask1, mask2),
            perturbations_per_eval=(1, 2, 3),
        )
        expected_with_baseline = (
            [[468.0, 468.0, 468.0], [184.0, 192.0, 184.0]],
            [[68.0, 188.0, 108.0], [-12.0, 388.0, -12.0]],
            [[-16.0, 384.0, 24.0], [12.0, 12.0, 12.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation_with_mask_dupe_feature_idx(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer_MultiInput())
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected_out = (
            [[1092.0, 1092.0, 1092.0], [260.0, 600.0, 260.0]],
            [[80.0, 1092.0, 160.0], [260.0, 600.0, 0.0]],
            [[80.0, 1092.0, 160.0], [260.0, 260.0, 260.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected_out,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )

        expected_cross_tensor_with_baseline = (
            [[1040.0, 1040.0, 1040.0], [184.0, 580.0, 184.0]],
            [[52.0, 1040.0, 132.0], [184.0, 580.0, -12.0]],
            [[52.0, 1040.0, 132.0], [184.0, 184.0, 184.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected_cross_tensor_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation_with_mask_nt(self) -> None:
        ablation_algo = NoiseTunnel(FeatureAblation(BasicModel_MultiLayer_MultiInput()))
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[3, 4, 2]])
        mask3 = torch.tensor([[5, 6, 7], [5, 5, 5]])
        expected = (
            [[492.0, 492.0, 492.0], [200.0, 200.0, 200.0]],
            [[80.0, 200.0, 120.0], [0.0, 400.0, 0.0]],
            [[0.0, 400.0, 40.0], [60.0, 60.0, 60.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            stdevs=1e-10,
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2),
            expected[0:1],
            additional_input=(inp3, 1),
            feature_mask=(mask1, mask2),
            perturbations_per_eval=(1, 2, 3),
            stdevs=1e-10,
        )
        expected_with_baseline = (
            [[468.0, 468.0, 468.0], [184.0, 192.0, 184.0]],
            [[68.0, 188.0, 108.0], [-12.0, 388.0, -12.0]],
            [[-16.0, 384.0, 24.0], [12.0, 12.0, 12.0]],
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
            stdevs=1e-10,
        )

    def test_multi_input_ablation(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer_MultiInput())
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        baseline1 = torch.tensor([[3.0, 0.0, 0.0]])
        baseline2 = torch.tensor([[0.0, 1.0, 0.0]])
        baseline3 = torch.tensor([[1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            (
                [[80.0, 400.0, 0.0], [68.0, 200.0, 120.0]],
                [[80.0, 196.0, 120.0], [0.0, 396.0, 0.0]],
                [[-4.0, 392.0, 28.0], [4.0, 32.0, 0.0]],
            ),
            additional_input=(1,),
            baselines=(baseline1, baseline2, baseline3),
            perturbations_per_eval=(1, 2, 3),
        )
        baseline1_exp = torch.tensor([[3.0, 0.0, 0.0], [3.0, 0.0, 2.0]])
        baseline2_exp = torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 4.0]])
        baseline3_exp = torch.tensor([[3.0, 2.0, 4.0], [1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            (
                [[80.0, 400.0, 0.0], [68.0, 200.0, 112.0]],
                [[80.0, 196.0, 120.0], [0.0, 396.0, -16.0]],
                [[-12.0, 392.0, 24.0], [4.0, 32.0, 0.0]],
            ),
            additional_input=(1,),
            baselines=(baseline1_exp, baseline2_exp, baseline3_exp),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_multi_input_conv(self) -> None:
        ablation_algo = FeatureAblation(BasicModel_ConvNet_One_Conv())
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        self._ablation_test_assert(
            ablation_algo,
            (inp, inp2),
            (67 * torch.ones_like(inp), 13 * torch.ones_like(inp2)),
            feature_mask=(torch.tensor(0), torch.tensor(1)),
            perturbations_per_eval=(1, 2, 4, 8, 12, 16),
        )
        self._ablation_test_assert(
            ablation_algo,
            (inp, inp2),
            (
                [
                    [
                        [
                            [0.0, 2.0, 4.0, 3.0],
                            [4.0, 9.0, 10.0, 7.0],
                            [4.0, 13.0, 14.0, 11.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
                [
                    [
                        [
                            [1.0, 2.0, 2.0, 1.0],
                            [1.0, 2.0, 2.0, 1.0],
                            [1.0, 2.0, 2.0, 1.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
            ),
            perturbations_per_eval=(1, 3, 7, 14),
        )

    # Remaining tests are for cases where forward function returns a scalar
    # per batch, as either a float, integer, 0d tensor or 1d tensor.
    def test_error_perturbations_per_eval_limit_batch_scalar(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        ablation = FeatureAblation(lambda inp: torch.sum(net(inp)).item())
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, perturbations_per_eval=2)

    def test_error_agg_mode_arbitrary_output(self) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()

        # output 3 numbers for the entire batch
        # note that the batch size == 2
        def forward_func(inp: Tensor) -> Tensor:
            pred = net(inp)
            return torch.stack([pred.sum(), pred.max(), pred.min()])

        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        ablation = FeatureAblation(forward_func)
        with self.assertRaises(AssertionError):
            _ = ablation.attribute(inp, perturbations_per_eval=2)

    def test_empty_sparse_features(self) -> None:
        ablation_algo = FeatureAblation(BasicModelWithSparseInputs())
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        inp2 = torch.tensor([])
        exp: Tuple[List[List[float]], List[float]] = ([[9.0, -3.0, 12.0]], [0.0])
        self._ablation_test_assert(ablation_algo, (inp1, inp2), exp, target=None)

    def test_sparse_features(self) -> None:
        ablation_algo = FeatureAblation(BasicModelWithSparseInputs())
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        # Length of sparse index list may not match # of examples
        inp2 = torch.tensor([1, 7, 2, 4, 5, 3, 6])
        self._ablation_test_assert(
            ablation_algo, (inp1, inp2), ([[9.0, -3.0, 12.0]], [2.0]), target=None
        )

    def test_single_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)).item())
        self._single_input_one_sample_batch_scalar_ablation_assert(
            ablation_algo, dtype=torch.float64
        )

    def test_single_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)))
        self._single_input_one_sample_batch_scalar_ablation_assert(ablation_algo)

    def test_single_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)).reshape(1))
        self._single_input_one_sample_batch_scalar_ablation_assert(ablation_algo)

    def test_single_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: int(torch.sum(net(inp)).item()))
        self._single_input_one_sample_batch_scalar_ablation_assert(
            ablation_algo, dtype=torch.int64
        )

    def test_multi_sample_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)).item())
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            ablation_algo,
            dtype=torch.float64,
        )

    def test_multi_sample_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)))
        self._single_input_multi_sample_batch_scalar_ablation_assert(ablation_algo)

    def test_multi_sample_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: torch.sum(net(inp)).reshape(1))
        self._single_input_multi_sample_batch_scalar_ablation_assert(ablation_algo)

    def test_multi_sample_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer()
        ablation_algo = FeatureAblation(lambda inp: int(torch.sum(net(inp)).item()))
        self._single_input_multi_sample_batch_scalar_ablation_assert(
            ablation_algo, dtype=torch.int64
        )

    def test_multi_inp_ablation_batch_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        ablation_algo = FeatureAblation(lambda *inp: torch.sum(net(*inp)).item())
        self._multi_input_batch_scalar_ablation_assert(
            ablation_algo,
            dtype=torch.float64,
        )

    def test_multi_inp_ablation_batch_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        ablation_algo = FeatureAblation(lambda *inp: torch.sum(net(*inp)))
        self._multi_input_batch_scalar_ablation_assert(ablation_algo)

    def test_multi_inp_ablation_batch_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        ablation_algo = FeatureAblation(lambda *inp: torch.sum(net(*inp)).reshape(1))
        self._multi_input_batch_scalar_ablation_assert(ablation_algo)

    def test_mutli_inp_ablation_batch_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        ablation_algo = FeatureAblation(lambda *inp: int(torch.sum(net(*inp)).item()))
        self._multi_input_batch_scalar_ablation_assert(ablation_algo, dtype=torch.int64)

    def test_future_output(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            dummy_output = torch.ones(1, 5, 3, 2)
            return dummy_output

        abl = FeatureAblation(_construct_future_forward(forward_func))
        inp = torch.randn(10, 5)
        mask = torch.arange(5).unsqueeze(0)
        self._ablation_test_assert(
            ablation_algo=abl,
            test_input=inp,
            baselines=None,
            target=None,
            feature_mask=mask,
            perturbations_per_eval=(1,),
            expected_ablation=torch.zeros((5 * 3 * 2,) + inp[0].shape),
            test_future=True,
        )

    def test_future_output_2(self) -> None:
        net: BasicModel_MultiLayer = BasicModel_MultiLayer()

        def slow_set_future(fut: torch.futures.Future[Tensor], value: Tensor) -> None:
            time.sleep(10)
            out = net(value)
            fut.set_result(out)

        def forward_func(inp: Tensor) -> torch.futures.Future[Tensor]:
            # pyre-fixme[29]: `typing.Type[torch.futures.Future]` is not a function.
            fut: torch.futures.Future[Tensor] = torch.futures.Future()
            t = threading.Thread(target=slow_set_future, args=(fut, inp))
            t.start()
            return fut

        abl = FeatureAblation(forward_func)
        inp = torch.tensor([[20.0, 50.0, 30.0], [10.0, 40.0, 20.0]], requires_grad=True)
        self._ablation_test_assert(
            ablation_algo=abl,
            test_input=inp,
            baselines=None,
            target=0,
            perturbations_per_eval=(1,),
            expected_ablation=torch.tensor([[80.0, 200.0, 120.0], [40.0, 160.0, 80.0]]),
            test_future=True,
        )

    def test_future_wrong_usage(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            dummy_output = torch.ones(1, 5, 3, 2)
            return dummy_output

        abl = FeatureAblation(_construct_future_forward(forward_func))
        inp = torch.randn(10, 5)
        mask = torch.arange(5).unsqueeze(0)
        perturbations_per_eval = (1,)

        with self.assertRaises(AssertionError):
            for batch_size in perturbations_per_eval:
                attributions = abl.attribute(  # noqa
                    inp,
                    target=None,
                    feature_mask=mask,
                    additional_forward_args=None,
                    baselines=None,
                    perturbations_per_eval=batch_size,
                )

    def test_future_wrong_usage_2(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            dummy_output = torch.ones(1, 5, 3, 2)
            return dummy_output

        abl = FeatureAblation(forward_func)
        inp = torch.randn(10, 5)
        mask = torch.arange(5).unsqueeze(0)
        perturbations_per_eval = (1,)

        with self.assertRaises(AssertionError):
            for batch_size in perturbations_per_eval:
                attributions = abl.attribute_future(  # noqa
                    inp,
                    target=None,
                    feature_mask=mask,
                    additional_forward_args=None,
                    baselines=None,
                    perturbations_per_eval=batch_size,
                )

    def test_unassociated_output_3d_tensor(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            return torch.ones(1, 5, 3, 2)

        inp = torch.randn(10, 5)
        mask = torch.arange(5).unsqueeze(0)
        self._ablation_test_assert(
            ablation_algo=FeatureAblation(forward_func),
            test_input=inp,
            baselines=None,
            target=None,
            feature_mask=mask,
            perturbations_per_eval=(1,),
            expected_ablation=torch.zeros((5 * 3 * 2,) + inp[0].shape),
        )

    def test_single_inp_ablation_multi_output_aggr(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[0, 1, 2]])
        self._ablation_test_assert(
            ablation_algo=FeatureAblation(forward_func),
            test_input=inp,
            feature_mask=mask,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            # should just be the first input spread across each feature
            expected_ablation=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    def test_single_inp_ablation_multi_output_aggr_mask_none(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self._ablation_test_assert(
            ablation_algo=FeatureAblation(forward_func),
            test_input=inp,
            feature_mask=None,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            # should just be the first input spread across each feature
            expected_ablation=[[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    def test_single_inp_ablation_multi_output_aggr_non_standard(self) -> None:
        def forward_func(inp: Tensor) -> Tensor:
            return inp[0].unsqueeze(0)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        mask = torch.tensor([[0, 0, 1]])
        self._ablation_test_assert(
            ablation_algo=FeatureAblation(forward_func),
            test_input=inp,
            feature_mask=mask,
            baselines=None,
            target=None,
            perturbations_per_eval=(1,),
            expected_ablation=[[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        )

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_ablation_with_show_progress(
        self, mock_stderr: unittest.mock.Mock
    ) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._ablation_test_assert(
                ablation_algo,
                inp,
                [[80.0, 200.0, 120.0]],
                perturbations_per_eval=(bsz,),
                show_progress=True,
            )

            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Feature Ablation attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_ablation_with_mask_and_show_progress(
        self, mock_stderr: unittest.mock.Mock
    ) -> None:
        ablation_algo = FeatureAblation(BasicModel_MultiLayer())
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._ablation_test_assert(
                ablation_algo,
                inp,
                [[280.0, 280.0, 120.0]],
                feature_mask=torch.tensor([[0, 0, 1]]),
                perturbations_per_eval=(bsz,),
                show_progress=True,
            )

            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Feature Ablation attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    def _single_input_one_sample_batch_scalar_ablation_assert(
        self, ablation_algo: Attribution, dtype: torch.dtype = torch.float32
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            ablation_algo,
            inp,
            torch.tensor([[82.0, 82.0, 24.0]], dtype=torch.float32).to(dtype),
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def _single_input_multi_sample_batch_scalar_ablation_assert(
        self,
        ablation_algo: Attribution,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._ablation_test_assert(
            ablation_algo,
            inp,
            torch.tensor([[642.0, 642.0, 264.0]], dtype=torch.float32).to(dtype),
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def _multi_input_batch_scalar_ablation_assert(
        self,
        ablation_algo: Attribution,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 3, 2]])
        mask3 = torch.tensor([[4, 5, 6]])
        expected = (
            torch.tensor([[1784, 1784, 1784]], dtype=dtype),
            torch.tensor([[160, 1200, 240]], dtype=dtype),
            torch.tensor([[16, 880, 104]], dtype=dtype),
        )

        self._ablation_test_assert(
            ablation_algo,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
        )

    def _ablation_test_assert(
        self,
        ablation_algo: Attribution,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_ablation: Union[
            Tensor,
            Tuple[Tensor, ...],
            # NOTE: mypy doesn't support recursive types
            # would do a List[NestedList[Union[int, float]]
            # or Tuple[NestedList[Union[int, float]]
            # but... we can't.
            #
            # See https://github.com/python/mypy/issues/731
            List[Any],
            Tuple[List[Any], ...],
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: TargetType = 0,
        test_future: bool = False,
        **kwargs: Any,
    ) -> None:
        for batch_size in perturbations_per_eval:
            self.assertTrue(ablation_algo.multiplies_by_inputs)
            if isinstance(ablation_algo, FeatureAblation) and test_future:
                attributions = ablation_algo.attribute_future(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    **kwargs,
                ).wait()
            else:
                attributions = ablation_algo.attribute(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    **kwargs,
                )
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    expected = expected_ablation[i]
                    if not isinstance(expected, torch.Tensor):
                        expected = torch.tensor(expected)

                    self.assertEqual(attributions[i].shape, expected.shape)
                    self.assertEqual(attributions[i].dtype, expected.dtype)
                    assertTensorAlmostEqual(self, attributions[i], expected)
            else:
                if not isinstance(expected_ablation, torch.Tensor):
                    expected_ablation = torch.tensor(expected_ablation)

                self.assertEqual(attributions.shape, expected_ablation.shape)
                self.assertEqual(attributions.dtype, expected_ablation.dtype)
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
