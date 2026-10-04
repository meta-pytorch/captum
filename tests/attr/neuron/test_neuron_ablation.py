#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.typing import (
    BaselineType,
    TensorLikeList,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.neuron.neuron_feature_ablation import NeuronFeatureAblation
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from torch import Tensor
from torch.nn import Module


# pyrefly: ignore [invalid-inheritance]
class Test(BaseTest):
    def test_simple_ablation_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._ablation_test_assert(
            net,
            net.linear2,
            inp,
            [[280.0, 280.0, 120.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            net,
            net.linear2,
            inp,
            [[41.0, 41.0, 12.0], [280.0, 280.0, 120.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_sample_ablation_with_selector_fn(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            net,
            net.linear2,
            inp,
            [[82.0, 82.0, 24.0], [560.0, 560.0, 240.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
            neuron_selector=lambda x: torch.sum(x, dim=1),
        )

    def test_multi_sample_ablation_with_slice(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1], [1, 1, 0]])
        self._ablation_test_assert(
            net,
            net.linear2,
            inp,
            [[82.0, 82.0, 24.0], [560.0, 560.0, 240.0]],
            feature_mask=mask,
            perturbations_per_eval=(1, 2, 3),
            neuron_selector=(slice(0, 2, 1),),
        )

    def test_multi_input_ablation_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
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
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        self._ablation_test_assert(
            net,
            net.model.linear2,
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
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_ablation(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        baseline1 = torch.tensor([[3.0, 0.0, 0.0]])
        baseline2 = torch.tensor([[0.0, 1.0, 0.0]])
        baseline3 = torch.tensor([[1.0, 2.0, 3.0]])
        self._ablation_test_assert(
            net,
            net.model.linear2,
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
            net,
            net.model.linear2,
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
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        self._ablation_test_assert(
            net,
            net.relu2,
            (inp, inp2),
            (67 * torch.ones_like(inp), 13 * torch.ones_like(inp2)),
            feature_mask=(torch.tensor(0), torch.tensor(1)),
            perturbations_per_eval=(1, 2, 4, 8, 12, 16),
        )
        self._ablation_test_assert(
            net,
            net.relu2,
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

    def test_simple_multi_input_conv_intermediate(self) -> None:
        net = BasicModel_ConvNet_One_Conv(inplace=True)
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            (torch.zeros_like(inp), torch.zeros_like(inp2)),
            feature_mask=(torch.tensor(0), torch.tensor(1)),
            perturbations_per_eval=(1, 2, 4, 8, 12, 16),
            neuron_selector=(1, 0, 0),
        )
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            (45 * torch.ones_like(inp), 9 * torch.ones_like(inp2)),
            feature_mask=(torch.tensor(0), torch.tensor(1)),
            perturbations_per_eval=(1, 2, 4, 8, 12, 16),
            neuron_selector=(1, 0, 0),
            attribute_to_neuron_input=True,
        )
        self._ablation_test_assert(
            net,
            net.relu1,
            (inp, inp2),
            (
                [
                    [
                        [
                            [0.0, 1.0, 2.0, 0.0],
                            [4.0, 5.0, 6.0, 0.0],
                            [8.0, 9.0, 10.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
                [
                    [
                        [
                            [1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0],
                            [1.0, 1.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ],
            ),
            perturbations_per_eval=(1, 3, 7, 14),
            neuron_selector=(1, 0, 0),
            attribute_to_neuron_input=True,
        )

    def _ablation_test_assert(
        self,
        model: Module,
        layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_ablation: Union[
            TensorLikeList,
            Tuple[TensorLikeList, ...],
            Tuple[Tensor, ...],
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable] = 0,
        attribute_to_neuron_input: bool = False,
    ) -> None:
        for batch_size in perturbations_per_eval:
            ablation = NeuronFeatureAblation(model, layer)
            self.assertTrue(ablation.multiplies_by_inputs)
            attributions = ablation.attribute(
                test_input,
                neuron_selector=neuron_selector,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                attribute_to_neuron_input=attribute_to_neuron_input,
            )
            if isinstance(expected_ablation, tuple):
                for i in range(len(expected_ablation)):
                    assertTensorAlmostEqual(self, attributions[i], expected_ablation[i])
            else:
                assertTensorAlmostEqual(self, attributions, expected_ablation)


if __name__ == "__main__":
    unittest.main()
