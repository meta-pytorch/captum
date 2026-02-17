#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from captum._utils.typing import BaselineType, Tensor
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.classification_models import SigmoidModel, SoftmaxModel
from torch.nn import Module


class Test(BaseTest):
    def test_sigmoid_classification_vanilla(self) -> None:
        self._assert_sigmoid_classification("vanilla", "riemann_right")

    def test_sigmoid_classification_smoothgrad(self) -> None:
        self._assert_sigmoid_classification("smoothgrad", "riemann_left")

    def test_sigmoid_classification_smoothgrad_sq(self) -> None:
        self._assert_sigmoid_classification("smoothgrad_sq", "riemann_middle")

    def test_sigmoid_classification_vargrad(self) -> None:
        self._assert_sigmoid_classification("vargrad", "riemann_trapezoid")

    def test_softmax_classification_vanilla(self) -> None:
        self._assert_softmax_classification("vanilla", "gausslegendre")

    def test_softmax_classification_smoothgrad(self) -> None:
        self._assert_softmax_classification("smoothgrad", "riemann_right")

    def test_softmax_classification_smoothgrad_sq(self) -> None:
        self._assert_softmax_classification("smoothgrad_sq", "riemann_left")

    def test_softmax_classification_vargrad(self) -> None:
        self._assert_softmax_classification("vargrad", "riemann_middle")

    def test_softmax_classification_vanilla_batch(self) -> None:
        self._assert_softmax_classification_batch("vanilla", "riemann_trapezoid")

    def test_softmax_classification_smoothgrad_batch(self) -> None:
        self._assert_softmax_classification_batch("smoothgrad", "gausslegendre")

    def test_softmax_classification_smoothgrad_sq_batch(self) -> None:
        self._assert_softmax_classification_batch("smoothgrad_sq", "riemann_right")

    def test_softmax_classification_vargrad_batch(self) -> None:
        self._assert_softmax_classification_batch("vargrad", "riemann_left")

    def _assert_sigmoid_classification(
        self, type: str = "vanilla", approximation_method: str = "gausslegendre"
    ) -> None:
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(0)
        # TODO add test cases for multiple different layers
        model = SigmoidModel(num_in, 5, 1)
        self._validate_completness(model, input, target, type, approximation_method)

    def _assert_softmax_classification(
        self, type: str = "vanilla", approximation_method: str = "gausslegendre"
    ) -> None:
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        target = torch.tensor(5)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        self._validate_completness(model, input, target, type, approximation_method)

    def _assert_softmax_classification_batch(
        self, type: str = "vanilla", approximation_method: str = "gausslegendre"
    ) -> None:
        num_in = 40
        input = torch.arange(0.0, num_in * 3.0, requires_grad=True).reshape(3, num_in)
        target = torch.tensor([5, 5, 2])
        baseline = torch.zeros(1, num_in)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        self._validate_completness(
            model, input, target, type, approximation_method, baseline
        )

    def _validate_completness(
        self,
        model: Module,
        input: Tensor,
        target: Tensor,
        type: str = "vanilla",
        approximation_method: str = "gausslegendre",
        baseline: BaselineType = None,
    ) -> None:
        ig = IntegratedGradients(model.forward)
        model.zero_grad()
        if type == "vanilla":
            attributions, delta = ig.attribute(  # type: ignore[has-type]
                input,
                baselines=baseline,
                target=target,
                method=approximation_method,
                n_steps=200,
                return_convergence_delta=True,
            )
            delta_expected = ig.compute_convergence_delta(
                attributions, baseline, input, target
            )
            assertTensorAlmostEqual(self, delta_expected, delta)

            delta_condition = (delta.abs() < 0.005).all()
            self.assertTrue(
                delta_condition,
                "The sum of attribution values {} is not "
                "nearly equal to the difference between the endpoint for "
                "some samples".format(delta),
            )
            self.assertEqual([input.shape[0]], list(delta.shape))
        else:
            nt = NoiseTunnel(ig)
            n_samples = 10
            attributions, delta = nt.attribute(
                input,
                baselines=baseline,
                nt_type=type,
                nt_samples=n_samples,
                stdevs=0.0002,
                n_steps=100,
                target=target,
                method=approximation_method,
                return_convergence_delta=True,
            )
            self.assertEqual([input.shape[0] * n_samples], list(delta.shape))

        self.assertTrue((delta.abs() < 0.05).all())
        self.assertEqual(attributions.shape, input.shape)


if __name__ == "__main__":
    unittest.main()
