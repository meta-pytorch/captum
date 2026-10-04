#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any, Callable, cast, List, Tuple, Union

import torch
from captum._utils.gradient import _forward_layer_eval
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.neuron.neuron_gradient import NeuronGradient
from captum.attr._core.saliency import Saliency
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_gradient_input_linear2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._gradient_input_test_assert(net, net.linear2, inp, (0,), [[4.0, 4.0, 4.0]])

    def test_simple_gradient_input_linear1(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._gradient_input_test_assert(net, net.linear1, inp, (0,), [[1.0, 1.0, 1.0]])

    def test_simple_gradient_input_relu_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(
            net, net.relu, inp, (0,), [[1.0, 1.0, 1.0]], attribute_to_neuron_input=True
        )

    def test_simple_gradient_input_linear1_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(net, net.linear1, inp, (0,), [[1.0, 1.0, 1.0]])

    def test_simple_gradient_input_relu(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]], requires_grad=True)
        self._gradient_input_test_assert(net, net.relu, inp, 0, [[0.0, 0.0, 0.0]])

    def test_simple_gradient_input_relu2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(net, net.relu, inp, 1, [[1.0, 1.0, 1.0]])

    def test_simple_gradient_input_relu_selector_fn(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(
            net, net.relu, inp, lambda x: torch.sum(x), [[3.0, 3.0, 3.0]]
        )

    def test_simple_gradient_input_relu2_agg_neurons(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(
            net, net.relu, inp, (slice(0, 2, 1),), [[1.0, 1.0, 1.0]]
        )

    def test_simple_gradient_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 100.0, 0.0]])
        inp2 = torch.tensor([[0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 0.0]])
        self._gradient_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[12.0, 12.0, 12.0]], [[12.0, 12.0, 12.0]], [[12.0, 12.0, 12.0]]),
            (3,),
        )

    def test_simple_gradient_multi_input_linear1(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 100.0, 0.0]])
        inp2 = torch.tensor([[0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 0.0]])
        self._gradient_input_test_assert(
            net,
            net.model.linear1,
            (inp1, inp2),
            (0,),
            ([[5.0, 5.0, 5.0]], [[5.0, 5.0, 5.0]]),
            (inp3, 5),
        )

    def test_matching_output_gradient(self) -> None:
        net = BasicModel_ConvNet()
        inp = torch.randn(2, 1, 10, 10, requires_grad=True)
        self._gradient_matching_test_assert(net, net.softmax, inp)

    def test_matching_intermediate_gradient(self) -> None:
        net = BasicModel_ConvNet()
        inp = torch.randn(3, 1, 10, 10)
        self._gradient_matching_test_assert(net, net.relu2, inp)

    def _gradient_input_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        test_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
        expected_input_gradient: Union[
            List[List[float]], Tuple[List[List[float]], ...]
        ],
        additional_input: Any = None,
        attribute_to_neuron_input: bool = False,
    ) -> None:
        grad = NeuronGradient(model, target_layer)
        attributions = grad.attribute(
            test_input,
            test_neuron_selector,
            additional_forward_args=additional_input,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )
        assertTensorTuplesAlmostEqual(self, attributions, expected_input_gradient)

    def _gradient_matching_test_assert(
        self, model: Module, output_layer: Module, test_input: Tensor
    ) -> None:
        out = _forward_layer_eval(model, test_input, output_layer)
        # Select first element of tuple
        out = out[0]
        gradient_attrib = NeuronGradient(model, output_layer)
        self.assertFalse(gradient_attrib.multiplies_by_inputs)
        for i in range(cast(Tuple[int, ...], out.shape)[1]):
            neuron: Tuple[int, ...] = (i,)
            while len(neuron) < len(out.shape) - 1:
                neuron = neuron + (0,)
            input_attrib = Saliency(
                lambda x, neuron=neuron: _forward_layer_eval(
                    model, x, output_layer, grad_enabled=True
                )[0][(slice(None), *neuron)]
            )
            sal_vals = input_attrib.attribute(test_input, abs=False)
            grad_vals = gradient_attrib.attribute(test_input, neuron)
            # Verify matching sizes
            self.assertEqual(grad_vals.shape, sal_vals.shape)
            self.assertEqual(grad_vals.shape, test_input.shape)
            assertTensorAlmostEqual(self, sal_vals, grad_vals, delta=0.001, mode="max")


if __name__ == "__main__":
    unittest.main()
