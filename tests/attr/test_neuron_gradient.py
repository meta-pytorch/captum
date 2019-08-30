from __future__ import print_function

import unittest

import torch
from captum.attr._core.saliency import Saliency
from captum.attr._core.neuron_gradient import NeuronGradient
from captum.attr._utils.common import _forward_layer_eval, _extend_index_list


from .helpers.basic_models import TestModel_ConvNet, TestModel_MultiLayer
from .helpers.utils import assertArraysAlmostEqual


class Test(unittest.TestCase):
    def test_simple_gradient_input_linear2(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._gradient_input_test_assert(net, net.linear2, inp, (0,), [4.0, 4.0, 4.0])

    def test_simple_gradient_input_linear1(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._gradient_input_test_assert(net, net.linear1, inp, (0,), [1.0, 1.0, 1.0])

    def test_simple_gradient_input_relu(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]], requires_grad=True)
        self._gradient_input_test_assert(net, net.relu, inp, 0, [0.0, 0.0, 0.0])

    def test_simple_gradient_input_relu2(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]])
        self._gradient_input_test_assert(net, net.relu, inp, 1, [1.0, 1.0, 1.0])

    def test_matching_output_gradient(self):
        net = TestModel_ConvNet()
        inp = torch.randn(2, 1, 10, 10, requires_grad=True)
        self._gradient_matching_test_assert(net, net.softmax, inp)

    def test_matching_intermediate_gradient(self):
        net = TestModel_ConvNet()
        inp = torch.randn(3, 1, 10, 10)
        self._gradient_matching_test_assert(net, net.relu2, inp)

    def _gradient_input_test_assert(
        self, model, target_layer, test_input, test_neuron, expected_input_gradient
    ):
        grad = NeuronGradient(model, target_layer)
        attributions = grad.attribute(test_input, test_neuron)
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_input_gradient, delta=0.1
        )

    def _gradient_matching_test_assert(self, model, output_layer, test_input):
        out = _forward_layer_eval(model, test_input, output_layer)
        gradient_attrib = NeuronGradient(model, output_layer)
        for i in range(out.shape[1]):
            neuron = (i,)
            while len(neuron) < len(out.shape) - 1:
                neuron = neuron + (0,)
            input_attrib = Saliency(
                lambda x: torch.stack(
                    [
                        _forward_layer_eval(model, x, output_layer)[index]
                        for index in _extend_index_list(test_input.shape[0], neuron)
                    ]
                )
            )
            sal_vals = input_attrib.attribute(test_input, abs=False)
            grad_vals = gradient_attrib.attribute(test_input, neuron)
            # Verify matching sizes
            self.assertEqual(grad_vals.shape, sal_vals.shape)
            self.assertEqual(grad_vals.shape, test_input.shape)
            assertArraysAlmostEqual(
                sal_vals.reshape(-1).tolist(),
                grad_vals.reshape(-1).tolist(),
                delta=0.001,
            )


if __name__ == "__main__":
    unittest.main()
