from __future__ import print_function

from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _run_forward

from .helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel3,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel6_MultiTensor,
    TestModel_MultiLayer,
)
from .helpers.utils import assertArraysAlmostEqual

import unittest
import torch


class Test(unittest.TestCase):
    def test_multivariable_vanilla(self):
        self._assert_multi_variable("vanilla")

    def test_multivariable_smoothgrad(self):
        self._assert_multi_variable("smoothgrad")

    def test_multivariable_vargrad(self):
        self._assert_multi_variable("vargrad")

    def test_multi_argument_vanilla(self):
        self._assert_multi_argument("vanilla")

    def test_multi_argument_smoothgrad(self):
        self._assert_multi_argument("smoothgrad")

    def test_multi_argument_vargrad(self):
        self._assert_multi_argument("vargrad")

    def test_univariable_vanilla(self):
        self._assert_univariable("vanilla")

    def test_univariable_smoothgrad(self):
        self._assert_univariable("smoothgrad")

    def test_univariable_vargrad(self):
        self._assert_univariable("vargrad")

    def test_multi_tensor_input_vanilla(self):
        self._assert_multi_tensor_input("vanilla")

    def test_multi_tensor_input_smoothgrad(self):
        self._assert_multi_tensor_input("smoothgrad")

    def test_multi_tensor_input_vargrad(self):
        self._assert_multi_tensor_input("vargrad")

    def test_batched_input_vanilla(self):
        self._assert_batched_tensor_input("vanilla")

    def test_batched_multi_input_vanilla(self):
        self._assert_batched_tensor_multi_input("vanilla")

    def _assert_multi_variable(self, type):
        model = BasicModel2()

        input1 = torch.tensor([3.0], requires_grad=True)
        input2 = torch.tensor([1.0], requires_grad=True)

        baseline1 = torch.tensor([0.0])
        baseline2 = torch.tensor([0.0])

        attributions1 = self._compute_attribution_and_evaluate(
            model, (input1, input2), (baseline1, baseline2), type=type
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions1[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions1[1].tolist(), [-0.5], delta=0.05)
        model = BasicModel3()
        attributions2 = self._compute_attribution_and_evaluate(
            model, (input1, input2), (baseline1, baseline2), type=type
        )
        if type == "vanilla":
            assertArraysAlmostEqual(attributions2[0].tolist(), [1.5], delta=0.05)
            assertArraysAlmostEqual(attributions2[1].tolist(), [-0.5], delta=0.05)
            # Verifies implementation invariance
            self.assertEqual(
                sum(attribution for attribution in attributions1),
                sum(attribution for attribution in attributions2),
            )

    def _assert_univariable(self, type):
        model = BasicModel()
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([0.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([-1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
        )

    def _assert_multi_argument(self, type):
        model = BasicModel4_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2]], requires_grad=True),
            ),
            baselines=(torch.zeros((1, 3)), torch.zeros((1, 3))),
            additional_forward_args=torch.arange(1.0, 4.0).reshape(1, 3),
            type=type,
        )
        # uses batching with an integer variable and nd-tensors as
        # additional forward arguments
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), torch.zeros((2, 3))),
            additional_forward_args=(torch.arange(1.0, 7.0).reshape(2, 3), 1),
            type=type,
        )
        # uses batching with an integer variable and python list
        # as additional forward arguments
        model = BasicModel5_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), torch.zeros((2, 3))),
            additional_forward_args=([2, 3], 1),
            type=type,
        )

    def _assert_multi_tensor_input(self, type):
        model = BasicModel6_MultiTensor()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2]], requires_grad=True),
            ),
            type=type,
        )

    def _assert_batched_tensor_input(self, type):
        model = TestModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(model, input, type=type, target=0)
        self._compute_attribution_batch_helper_evaluate(model, input, target=0)

    def _assert_batched_tensor_multi_input(self, type):
        model = TestModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.1, 1.9], [0.5, 0.0, 0.7], [1.5, 2.1, 1.1]], requires_grad=True
            ),
            torch.tensor(
                [[0.3, 1.9, 2.4], [0.5, 0.6, 2.1], [1.2, 2.1, 0.2]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(model, input, type=type, target=0)
        self._compute_attribution_batch_helper_evaluate(model, input, target=0)

    def _compute_attribution_and_evaluate(
        self,
        model,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        type="vanilla",
    ):
        r"""
            attrib_type: 'vanilla', 'smoothgrad', 'vargrad'
        """
        ig = IntegratedGradients(model.forward)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = ig.zero_baseline(inputs)

        forward_input = _run_forward(
            model,
            inputs,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        forward_baseline = _run_forward(
            model,
            baselines,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        for method in [
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]:
            if type == "vanilla":
                attributions, delta = ig.attribute(
                    inputs,
                    baselines,
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=1500,
                    target=target,
                )
                if isinstance(attributions, tuple):
                    attr_sum = sum(
                        torch.sum(attribution).item() for attribution in attributions
                    )
                else:
                    attr_sum = torch.sum(attributions).item()
                expected_delta = abs(
                    attr_sum
                    - (forward_input.sum().item() - forward_baseline.sum().item())
                )
                self.assertAlmostEqual(
                    attr_sum,
                    forward_input.sum().item() - forward_baseline.sum().item(),
                    delta=0.05,
                )
                self.assertAlmostEqual(delta, expected_delta, delta=0.005)
            else:
                nt = NoiseTunnel(ig)
                attributions, delta = nt.attribute(
                    inputs,
                    reg_type=type,
                    n_samples=10,
                    noise_frac=0.0002,
                    baselines=baselines,
                    additional_forward_args=additional_forward_args,
                    method=method,
                )

            if isinstance(inputs, tuple):
                for input, attribution in zip(inputs, attributions):
                    self.assertEqual(attribution.shape, input.shape)
            else:
                self.assertEqual(attributions.shape, inputs.shape)
        return attributions

    def _compute_attribution_batch_helper_evaluate(
        self, model, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        ig = IntegratedGradients(model.forward)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = ig.zero_baseline(inputs)

        for method in [
            "riemann_right",
            "riemann_left",
            "riemann_middle",
            "riemann_trapezoid",
            "gausslegendre",
        ]:
            attributions, delta = ig.attribute(
                inputs,
                baselines,
                additional_forward_args=additional_forward_args,
                method=method,
                n_steps=1500,
                target=target,
            )
            total_delta = 0
            for i in range(inputs[0].shape[0]):
                attributions_indiv, delta_indiv = ig.attribute(
                    tuple(input[i : i + 1] for input in inputs),
                    tuple(baseline[i : i + 1] for baseline in baselines),
                    additional_forward_args=additional_forward_args,
                    method=method,
                    n_steps=1500,
                    target=target,
                )
                total_delta += delta_indiv
                for j in range(len(attributions)):
                    assertArraysAlmostEqual(
                        attributions[j][i : i + 1].squeeze(0).tolist(),
                        attributions_indiv[j].squeeze(0).tolist(),
                    )
            self.assertAlmostEqual(delta, total_delta, delta=0.005)


if __name__ == "__main__":
    unittest.main()
