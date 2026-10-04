#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any, Callable, Tuple, Union

import torch
from captum._utils.typing import TensorLikeList, TensorOrTupleOfTensorsGeneric
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
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
    def test_simple_ig_input_linear2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._ig_input_test_assert(net, net.linear2, inp, 0, [[0.0, 390.0, 0.0]])

    def test_simple_ig_input_linear2_wo_mult_by_inputs(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[100.0, 100.0, 100.0]])
        self._ig_input_test_assert(
            net, net.linear2, inp, 0, [[3.96, 3.96, 3.96]], multiply_by_inputs=False
        )

    def test_simple_ig_input_linear1(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.linear1, inp, (0,), [[0.0, 100.0, 0.0]])

    def test_simple_ig_input_relu(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 6.0, 14.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.relu, inp, (0,), [[0.0, 3.0, 7.0]])

    def test_simple_ig_input_relu2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._ig_input_test_assert(net, net.relu, inp, 1, [[0.0, 5.0, 4.0]])

    def test_simple_ig_input_relu_selector_fn(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._ig_input_test_assert(
            net, net.relu, inp, lambda x: torch.sum(x[:, 2:]), [[0.0, 10.0, 8.0]]
        )

    def test_simple_ig_input_relu2_agg_neurons(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._ig_input_test_assert(
            net, net.relu, inp, (slice(0, 2, 1),), [[0.0, 5.0, 4.0]]
        )

    def test_simple_ig_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[0.0, 156.0, 0.0]], [[0.0, 156.0, 0.0]], [[0.0, 78.0, 0.0]]),
            (4,),
        )

    def test_simple_ig_multi_input_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5]], [[0.0, 1.5, 3.5]]),
            (inp3, 0.5),
        )

    def test_simple_ig_multi_input_relu_batch(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (0,),
            ([[0.0, 1.5, 3.5], [0.0, 40.0, 0.0]], [[0.0, 1.5, 3.5], [0.0, 10.0, 0.0]]),
            (inp3, 0.5),
        )

    def test_simple_ig_multi_input_relu_batch_selector_fn(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._ig_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            lambda x: torch.sum(x),
            (
                [[0.0, 10.5, 24.5], [0.0, 160.0, 0.0]],
                [[0.0, 10.5, 24.5], [0.0, 40.0, 0.0]],
            ),
            (inp3, 0.5),
        )

    def test_matching_output_gradient(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10, requires_grad=True)
        self._ig_matching_test_assert(net, net.softmax, inp, baseline)

    def _ig_input_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        test_neuron: Union[int, Tuple[Union[int, slice], ...], Callable],
        expected_input_ig: Union[TensorLikeList, Tuple[TensorLikeList, ...]],
        additional_input: Any = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        for internal_batch_size in [None, 5, 20]:
            grad = NeuronIntegratedGradients(
                model, target_layer, multiply_by_inputs=multiply_by_inputs
            )
            self.assertEqual(grad.multiplies_by_inputs, multiply_by_inputs)
            attributions = grad.attribute(
                test_input,
                test_neuron,
                n_steps=200,
                method="gausslegendre",
                additional_forward_args=additional_input,
                internal_batch_size=internal_batch_size,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_input_ig, delta=0.1
            )

    def _ig_matching_test_assert(
        self,
        model: Module,
        output_layer: Module,
        test_input: Tensor,
        baseline: Union[None, Tensor] = None,
    ) -> None:
        out = model(test_input)
        input_attrib = IntegratedGradients(model)
        ig_attrib = NeuronIntegratedGradients(model, output_layer)
        for i in range(out.shape[1]):
            ig_vals = input_attrib.attribute(  # type: ignore[has-type]
                test_input, target=i, baselines=baseline
            )
            neuron_ig_vals = ig_attrib.attribute(test_input, (i,), baselines=baseline)
            assertTensorAlmostEqual(
                self, ig_vals, neuron_ig_vals, delta=0.001, mode="max"
            )
            self.assertEqual(neuron_ig_vals.shape, test_input.shape)


if __name__ == "__main__":
    unittest.main()
