#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Any, List, Tuple, Union

import torch
from captum._utils.typing import (
    SliceIntType,
    TensorLikeList,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.guided_backprop_deconvnet import GuidedBackprop
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronGuidedBackprop,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel_ConvNet_One_Conv
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_conv_gb(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 2.0, 2.0, 1.0],
                ]
            ]
        ]
        self._guided_backprop_test_assert(net, (inp,), (exp,))

    def test_simple_input_conv_neuron_gb(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 2.0, 2.0, 1.0],
                ]
            ]
        ]
        self._neuron_guided_backprop_test_assert(net, net.fc1, (0,), (inp,), (exp,))

    def test_simple_input_conv_neuron_gb_agg_neurons(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [
                [
                    [0.0, 1.0, 1.0, 1.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 3.0, 3.0, 2.0],
                    [1.0, 2.0, 2.0, 1.0],
                ]
            ]
        ]
        self._neuron_guided_backprop_test_assert(
            net, net.fc1, (slice(0, 1, 1),), (inp,), (exp,)
        )

    def test_simple_multi_input_conv_gb(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                    [2.0, 4.0, 4.0, 2.0],
                    [2.0, 4.0, 4.0, 2.0],
                    [1.0, 2.0, 2.0, 1.0],
                ]
            ]
        ]
        self._guided_backprop_test_assert(net, (inp, inp2), (ex_attr, ex_attr))

    def test_simple_multi_input_conv_neuron_gb(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [
                [
                    [1.0, 2.0, 2.0, 1.0],
                    [2.0, 4.0, 4.0, 2.0],
                    [2.0, 4.0, 4.0, 2.0],
                    [1.0, 2.0, 2.0, 1.0],
                ]
            ]
        ]
        self._neuron_guided_backprop_test_assert(
            net, net.fc1, (3,), (inp, inp2), (ex_attr, ex_attr)
        )

    def test_gb_matching(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 100.0 * torch.randn(1, 1, 4, 4)
        self._guided_backprop_matching_assert(net, net.relu2, inp)

    def _guided_backprop_test_assert(
        self,
        model: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected: Tuple[TensorLikeList, ...],
        additional_input: Any = None,
    ) -> None:
        guided_backprop = GuidedBackprop(model)
        attributions = guided_backprop.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(
                self,
                attributions[i],
                expected[i],
                delta=0.01,
            )

    def _neuron_guided_backprop_test_assert(
        self,
        model: Module,
        layer: Module,
        neuron_selector: Union[int, Tuple[Union[int, SliceIntType], ...]],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected: Tuple[List[List[List[List[float]]]], ...],
        additional_input: Any = None,
    ) -> None:
        guided_backprop = NeuronGuidedBackprop(model, layer)
        attributions = guided_backprop.attribute(
            test_input,
            neuron_selector=neuron_selector,
            additional_forward_args=additional_input,
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(
                self,
                attributions[i],
                expected[i],
                delta=0.01,
            )

    def _guided_backprop_matching_assert(
        self,
        model: Module,
        output_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
    ) -> None:
        out = model(test_input)
        attrib = GuidedBackprop(model)
        self.assertFalse(attrib.multiplies_by_inputs)
        neuron_attrib = NeuronGuidedBackprop(model, output_layer)
        for i in range(out.shape[1]):
            gbp_vals = attrib.attribute(test_input, target=i)
            neuron_gbp_vals = neuron_attrib.attribute(test_input, (i,))
            assertTensorAlmostEqual(self, gbp_vals, neuron_gbp_vals, delta=0.01)


if __name__ == "__main__":
    unittest.main()
