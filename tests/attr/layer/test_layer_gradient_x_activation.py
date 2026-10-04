#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest
from typing import Any, List, Tuple, Union

import torch
from captum._utils.typing import ModuleOrModuleList
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.testing.helpers.basic import assertTensorTuplesAlmostEqual, BaseTest
from captum.testing.helpers.basic_models import (
    BasicEmbeddingModel,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.linear0, inp, [[0.0, 400.0, 0.0]])

    def test_simple_input_gradient_activation_no_grad(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        with torch.no_grad():
            self._layer_activation_test_assert(
                net, net.linear0, inp, [[0.0, 400.0, 0.0]]
            )

    def test_simple_linear_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.linear1, inp, [[90.0, 101.0, 101.0, 101.0]]
        )

    def test_multi_layer_linear_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        module_list: List[Module] = [net.linear0, net.linear1]
        self._layer_activation_test_assert(
            net,
            module_list,
            inp,
            ([[0.0, 400.0, 0.0]], [[90.0, 101.0, 101.0, 101.0]]),
        )

    def test_simple_linear_gradient_activation_no_grad(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])

        # this way we deactivate require_grad. Some models explicitly
        # do that before interpreting the model.
        for param in net.parameters():
            param.requires_grad = False

        self._layer_activation_test_assert(
            net, net.linear1, inp, [[90.0, 101.0, 101.0, 101.0]]
        )

    def test_simple_multi_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[3.0, 4.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.multi_relu, inp, ([[0.0, 8.0, 8.0, 8.0]], [[0.0, 8.0, 8.0, 8.0]])
        )

    def test_simple_relu_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._layer_activation_test_assert(net, net.relu, inp, [[0.0, 8.0, 8.0, 8.0]])

    def test_multi_layer_multi_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[3.0, 4.0, 0.0]])
        module_list: List[Module] = [net.multi_relu, net.linear0]
        self._layer_activation_test_assert(
            net,
            module_list,
            inp,
            [([[0.0, 8.0, 8.0, 8.0]], [[0.0, 8.0, 8.0, 8.0]]), [[9.0, 12.0, 0.0]]],
        )

    def test_simple_output_gradient_activation(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._layer_activation_test_assert(net, net.linear2, inp, [[392.0, 0.0]])

    def test_simple_gradient_activation_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.linear2, (inp1, inp2, inp3), [[392.0, 0.0]], (4,)
        )

    def test_simple_gradient_activation_multi_input_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._layer_activation_test_assert(
            net, net.model.relu, (inp1, inp2), [[90.0, 101.0, 101.0, 101.0]], (inp3, 5)
        )

    def test_gradient_activation_embedding(self) -> None:
        input1 = torch.tensor([2, 5, 0, 1])
        input2 = torch.tensor([3, 0, 0, 2])
        model = BasicEmbeddingModel()
        layer_act = LayerGradientXActivation(model, model.embedding1)
        self.assertEqual(
            list(layer_act.attribute(inputs=(input1, input2)).shape), [4, 100]
        )

    def test_gradient_activation_embedding_no_grad(self) -> None:
        input1 = torch.tensor([2, 5, 0, 1])
        input2 = torch.tensor([3, 0, 0, 2])
        model = BasicEmbeddingModel()
        for param in model.parameters():
            param.requires_grad = False

        with torch.no_grad():
            layer_act = LayerGradientXActivation(model, model.embedding1)
            self.assertEqual(
                list(layer_act.attribute(inputs=(input1, input2)).shape), [4, 100]
            )

    def test_simple_multi_gradient_activation_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )

        model = BasicModel_MultiLayer(multi_input_module=True)
        test_input = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        # pyre-fixme[6]: For 2nd argument expected `ModuleOrModuleList` but got
        #  `List[Union[ReLU, Linear]]`.
        layer_act = LayerGradientXActivation(model, [model.linear1, model.relu])
        attributions = layer_act.attribute(
            inputs=test_input, target=0, grad_kwargs={"materialize_grads": True}
        )
        self.assertEqual(len(attributions), 2)
        self.assertEqual(list(attributions[0].shape), [1, 4])
        self.assertEqual(list(attributions[1].shape), [1, 4])

    def _layer_activation_test_assert(
        self,
        model: Module,
        target_layer: ModuleOrModuleList,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        # `typing.List[<element type>]` to avoid runtime subscripting errors.
        expected_activation: Union[List, Tuple[List[List[float]], ...]],
        additional_input: Any = None,
    ) -> None:
        layer_act = LayerGradientXActivation(model, target_layer)
        self.assertTrue(layer_act.multiplies_by_inputs)
        attributions = layer_act.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        if isinstance(target_layer, Module):
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_activation, delta=0.01
            )
        else:
            # pyre-fixme[6]: For 1st argument expected
            #  `pyre_extensions.PyreReadOnly[Sized]` but got `ModuleOrModuleList`.
            for i in range(len(target_layer)):
                assertTensorTuplesAlmostEqual(
                    self, attributions[i], expected_activation[i], delta=0.01
                )
        # test Layer Gradient without multiplying with activations
        layer_grads = LayerGradientXActivation(
            model, target_layer, multiply_by_inputs=False
        )
        layer_act = LayerActivation(model, target_layer)
        self.assertFalse(layer_grads.multiplies_by_inputs)
        grads = layer_grads.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        acts = layer_act.attribute(test_input, additional_forward_args=additional_input)
        if isinstance(target_layer, Module):
            assertTensorTuplesAlmostEqual(
                self,
                attributions,
                tuple(act * grad for act, grad in zip(acts, grads)),
                delta=0.01,
            )
        else:
            # pyre-fixme[6]: For 1st argument expected
            #  `pyre_extensions.PyreReadOnly[Sized]` but got `ModuleOrModuleList`.
            for i in range(len(target_layer)):
                assertTensorTuplesAlmostEqual(
                    self,
                    attributions[i],
                    tuple(act * grad for act, grad in zip(acts[i], grads[i])),
                    delta=0.01,
                )


if __name__ == "__main__":
    unittest.main()
