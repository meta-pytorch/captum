#!/usr/bin/env python3

# pyre-strict

from __future__ import print_function

import unittest
from typing import Any, Tuple, Union

import torch
from captum._utils.typing import SliceIntType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.guided_backprop_deconvnet import Deconvolution
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel_ConvNet_One_Conv
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_conv_deconv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        exp = torch.tensor(exp).view(1, 1, 4, 4)
        self._deconv_test_assert(net, (inp,), (exp,))

    def test_simple_input_conv_neuron_deconv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        exp = torch.tensor(exp).view(1, 1, 4, 4)
        self._neuron_deconv_test_assert(net, net.fc1, (0,), (inp,), (exp,))

    def test_simple_input_conv_neuron_deconv_agg_neurons(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        exp = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        exp = torch.tensor(exp).view(1, 1, 4, 4)
        self._neuron_deconv_test_assert(net, net.fc1, (slice(0, 1, 1),), (inp,), (exp,))

    def test_simple_multi_input_conv_deconv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        ex_attr = torch.tensor(ex_attr).view(1, 1, 4, 4)
        self._deconv_test_assert(net, (inp, inp2), (ex_attr, ex_attr))

    def test_simple_multi_input_conv_neuron_deconv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex_attr = [
            [2.0, 3.0, 3.0, 1.0],
            [3.0, 5.0, 5.0, 2.0],
            [3.0, 5.0, 5.0, 2.0],
            [1.0, 2.0, 2.0, 1.0],
        ]
        ex_attr = torch.tensor(ex_attr).view(1, 1, 4, 4)
        self._neuron_deconv_test_assert(
            net, net.fc1, (3,), (inp, inp2), (ex_attr, ex_attr)
        )

    def test_deconv_matching(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 100.0 * torch.randn(1, 1, 4, 4)
        self._deconv_matching_assert(net, net.relu2, inp)

    def _deconv_test_assert(
        self,
        model: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected: Tuple[torch.Tensor, ...],
        additional_input: Any = None,
    ) -> None:
        deconv = Deconvolution(model)
        attributions = deconv.attribute(
            test_input, target=0, additional_forward_args=additional_input
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _neuron_deconv_test_assert(
        self,
        model: Module,
        layer: Module,
        neuron_selector: Union[int, Tuple[Union[int, SliceIntType], ...]],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected: Tuple[torch.Tensor, ...],
        additional_input: Any = None,
    ) -> None:
        deconv = NeuronDeconvolution(model, layer)
        attributions = deconv.attribute(
            test_input,
            neuron_selector=neuron_selector,
            additional_forward_args=additional_input,
        )
        for i in range(len(test_input)):
            assertTensorAlmostEqual(self, attributions[i], expected[i], delta=0.01)

    def _deconv_matching_assert(
        self,
        model: Module,
        output_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
    ) -> None:
        out = model(test_input)
        attrib = Deconvolution(model)
        self.assertFalse(attrib.multiplies_by_inputs)
        neuron_attrib = NeuronDeconvolution(model, output_layer)
        for i in range(out.shape[1]):
            deconv_vals = attrib.attribute(test_input, target=i)
            neuron_deconv_vals = neuron_attrib.attribute(test_input, (i,))
            assertTensorAlmostEqual(self, deconv_vals, neuron_deconv_vals, delta=0.01)


if __name__ == "__main__":
    unittest.main()
