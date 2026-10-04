#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any, List, Tuple, Union

import torch
import torch.nn as nn
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    Conv1dSeqModel,
)
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.linear0, inp, [[0.0, 100.0, 0.0]])

    def test_simple_linear_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.linear1, inp, [[90.0, 101.0, 101.0, 101.0]]
        )

    def test_simple_multi_linear_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._multiple_layer_activation_test_assert(
            net,
            [net.linear1, net.linear0],
            inp,
            ([[90.0, 101.0, 101.0, 101.0]], [[0.0, 100.0, 0.0]]),
        )

    def test_simple_relu_activation_input_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[2.0, -5.0, 4.0]])
        self._layer_activation_test_assert(
            net, net.relu, inp, ([-9.0, 2.0, 2.0, 2.0],), attribute_to_layer_input=True
        )

    def test_simple_linear_activation_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[2.0, -5.0, 4.0]])
        self._layer_activation_test_assert(
            net, net.linear1, inp, [[-9.0, 2.0, 2.0, 2.0]]
        )

    def test_simple_relu_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.relu, inp, [[0.0, 8.0, 8.0, 8.0]])

    def test_simple_output_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(net, net.linear2, inp, [[392.0, 394.0]])

    def test_simple_multi_output_activation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.multi_relu, inp, ([[0.0, 7.0, 7.0, 7.0]], [[0.0, 7.0, 7.0, 7.0]])
        )

    def test_simple_multi_layer_multi_output_activation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._multiple_layer_activation_test_assert(
            net,
            [net.multi_relu, net.linear0, net.linear1],
            inp,
            [
                ([[0.0, 7.0, 7.0, 7.0]], [[0.0, 7.0, 7.0, 7.0]]),
                [[0.0, 6.0, 0.0]],
                [[-4.0, 7.0, 7.0, 7.0]],
            ],
        )

    def test_simple_multi_input_activation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._layer_activation_test_assert(
            net,
            net.multi_relu,
            inp,
            ([[-4.0, 7.0, 7.0, 7.0]], [[-4.0, 7.0, 7.0, 7.0]]),
            attribute_to_layer_input=True,
        )

    def test_simple_multi_input_linear2_activation(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.linear2, (inp1, inp2, inp3), [[392.0, 394.0]], (4,)
        )

    def test_simple_multi_input_relu_activation(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.relu, (inp1, inp2), [[90.0, 101.0, 101.0, 101.0]], (inp3, 5)
        )

    def test_sequential_in_place(self) -> None:
        model = nn.Sequential(nn.Conv2d(3, 4, 3), nn.ReLU(inplace=True))
        layer_act = LayerActivation(model, model[0])
        input = torch.randn(1, 3, 5, 5)
        assertTensorAlmostEqual(self, layer_act.attribute(input), model[0](input))

    def test_sequential_module(self) -> None:
        model = Conv1dSeqModel()
        layer_act = LayerActivation(model, model.seq)
        input = torch.randn(2, 4, 1000)
        out = model(input)
        assertTensorAlmostEqual(self, layer_act.attribute(input), out)

    def _layer_activation_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_activation: Union[
            List[List[float]], Tuple[List[float], ...], Tuple[List[List[float]], ...]
        ],
        additional_input: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> None:
        layer_act = LayerActivation(model, target_layer)
        self.assertTrue(layer_act.multiplies_by_inputs)
        attributions = layer_act.attribute(
            test_input,
            additional_forward_args=additional_input,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assertTensorTuplesAlmostEqual(
            self, attributions, expected_activation, delta=0.01
        )

    def _multiple_layer_activation_test_assert(
        self,
        model: Module,
        target_layers: List[Module],
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_activation: Union[
            List, Tuple[List[float], ...], Tuple[List[List[float]], ...]
        ],
        additional_input: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> None:
        layer_act = LayerActivation(model, target_layers)
        self.assertTrue(layer_act.multiplies_by_inputs)
        attributions = layer_act.attribute(
            test_input,
            additional_forward_args=additional_input,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        for i in range(len(target_layers)):
            assertTensorTuplesAlmostEqual(
                self, attributions[i], expected_activation[i], delta=0.01
            )


if __name__ == "__main__":
    unittest.main()
