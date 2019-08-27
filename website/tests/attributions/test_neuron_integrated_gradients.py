from __future__ import print_function

import unittest

import torch
from captum.attributions.integrated_gradients import IntegratedGradients
from captum.attributions.neuron_integrated_gradients import NeuronIntegratedGradients

from .helpers.basic_models import TestModel_ConvNet, TestModel_MultiLayer
from .helpers.utils import assertArraysAlmostEqual


class Test(unittest.TestCase):
    def test_simple_ig_input_linear2(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.linear2, inp, 0, [0.0, 390.0, 0.0])

    def test_simple_ig_input_linear1(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.linear1, inp, (0,), [0.0, 100.0, 0.0])

    def test_simple_ig_input_relu(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 6.0, 14.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.relu, inp, (0,), [0.0, 3.0, 7.0])

    def test_simple_ig_input_relu2(self):
        net = TestModel_MultiLayer()
        inp = torch.tensor([[0.0, 5.0, 4.0]], requires_grad=True)
        self._ig_input_test_assert(net, net.relu, inp, 1, [0.0, 5.0, 4.0])

    def test_matching_output_gradient(self):
        net = TestModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10, requires_grad=True)
        self._ig_matching_test_assert(net, net.softmax, inp, baseline)

    def _ig_input_test_assert(
        self, model, target_layer, test_input, test_neuron, expected_input_ig
    ):
        grad = NeuronIntegratedGradients(model, target_layer)
        attributions = grad.attribute(
            test_input, test_neuron, n_steps=500, method="gausslegendre"
        )
        assertArraysAlmostEqual(
            attributions.squeeze(0).tolist(), expected_input_ig, delta=0.1
        )

    def _ig_matching_test_assert(self, model, output_layer, test_input, baseline=None):
        out = model(test_input)
        input_attrib = IntegratedGradients(model)
        ig_attrib = NeuronIntegratedGradients(model, output_layer)
        for i in range(out.shape[1]):
            ig_vals = input_attrib.attribute(test_input, target=i, baselines=baseline)[
                0
            ]
            neuron_ig_vals = ig_attrib.attribute(test_input, (i,), baselines=baseline)
            assertArraysAlmostEqual(
                ig_vals.reshape(-1).tolist(),
                neuron_ig_vals.reshape(-1).tolist(),
                delta=0.001,
            )
            self.assertEqual(neuron_ig_vals.shape, test_input.shape)


if __name__ == "__main__":
    unittest.main()
