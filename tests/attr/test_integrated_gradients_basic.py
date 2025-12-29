#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import cast, Optional, Tuple, Union

import torch
from captum._utils.common import _zeros
from captum._utils.typing import BaselineType, Tensor, TensorOrTupleOfTensorsGeneric
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.common import _tensorize_baseline
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel3,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel6_MultiTensor,
    BasicModel_MultiLayer,
)
from torch.nn import Module


class Test(BaseTest):
    def test_multivariable_vanilla(self) -> None:
        self._assert_multi_variable("vanilla", "riemann_right")

    def test_multivariable_vanilla_wo_mutliplying_by_inputs(self) -> None:
        self._assert_multi_variable(
            "vanilla", "riemann_right", multiply_by_inputs=False
        )

    def test_multivariable_smoothgrad(self) -> None:
        self._assert_multi_variable("smoothgrad", "riemann_left")

    def test_multivariable_smoothgrad_sq(self) -> None:
        self._assert_multi_variable("smoothgrad_sq", "riemann_middle")

    def test_multivariable_vargrad(self) -> None:
        self._assert_multi_variable("vargrad", "riemann_trapezoid")

    def test_multi_argument_vanilla(self) -> None:
        self._assert_multi_argument("vanilla", "gausslegendre")

    def test_multi_argument_smoothgrad(self) -> None:
        self._assert_multi_argument("smoothgrad", "riemann_right")

    def test_multi_argument_smoothgrad_sq(self) -> None:
        self._assert_multi_argument("smoothgrad_sq", "riemann_left")

    def test_multi_argument_vargrad(self) -> None:
        self._assert_multi_argument("vargrad", "riemann_middle")

    def test_univariable_vanilla(self) -> None:
        self._assert_univariable("vanilla", "riemann_trapezoid")

    def test_univariable_smoothgrad(self) -> None:
        self._assert_univariable("smoothgrad", "gausslegendre")

    def test_univariable_smoothgrad_sq(self) -> None:
        self._assert_univariable("smoothgrad_sq", "riemann_right")

    def test_univariable_vargrad(self) -> None:
        self._assert_univariable("vargrad", "riemann_left")

    def test_multi_tensor_input_vanilla(self) -> None:
        self._assert_multi_tensor_input("vanilla", "riemann_middle")

    def test_multi_tensor_input_smoothgrad(self) -> None:
        self._assert_multi_tensor_input("smoothgrad", "riemann_trapezoid")

    def test_multi_tensor_input_smoothgrad_sq(self) -> None:
        self._assert_multi_tensor_input("smoothgrad_sq", "gausslegendre")

    def test_multi_tensor_input_vargrad(self) -> None:
        self._assert_multi_tensor_input("vargrad", "riemann_right")

    def test_batched_input_vanilla(self) -> None:
        self._assert_batched_tensor_input("vanilla", "riemann_left")

    def test_batched_input_smoothgrad(self) -> None:
        self._assert_batched_tensor_input("smoothgrad", "riemann_middle")

    def test_batched_input_smoothgrad_with_batch_size_1(self) -> None:
        self._assert_n_samples_batched_size("smoothgrad", "riemann_middle", 1)

    def test_batched_input_smoothgrad_with_batch_size_2(self) -> None:
        self._assert_n_samples_batched_size("vargrad", "riemann_middle", 2)

    def test_batched_input_smoothgrad_with_batch_size_3(self) -> None:
        self._assert_n_samples_batched_size("smoothgrad_sq", "riemann_middle", 3)

    def test_batched_input_smoothgrad_sq(self) -> None:
        self._assert_batched_tensor_input("smoothgrad_sq", "riemann_trapezoid")

    def test_batched_input_vargrad(self) -> None:
        self._assert_batched_tensor_input("vargrad", "gausslegendre")

    def test_batched_input_smoothgrad_wo_mutliplying_by_inputs(self) -> None:
        model = BasicModel_MultiLayer()
        inputs = torch.tensor(
            [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
        )
        ig_wo_mutliplying_by_inputs = IntegratedGradients(
            model, multiply_by_inputs=False
        )
        nt_wo_mutliplying_by_inputs = NoiseTunnel(ig_wo_mutliplying_by_inputs)

        ig = IntegratedGradients(model)
        nt = NoiseTunnel(ig)
        n_samples = 5
        target = 0
        type = "smoothgrad"
        attributions_wo_mutliplying_by_inputs = nt_wo_mutliplying_by_inputs.attribute(
            inputs,
            nt_type=type,
            nt_samples=n_samples,
            stdevs=0.0,
            target=target,
            n_steps=500,
        )
        attributions = nt.attribute(
            inputs,
            nt_type=type,
            nt_samples=n_samples,
            stdevs=0.0,
            target=target,
            n_steps=500,
        )
        assertTensorAlmostEqual(
            self, attributions_wo_mutliplying_by_inputs * inputs, attributions
        )

    def test_batched_multi_input_vanilla(self) -> None:
        self._assert_batched_tensor_multi_input("vanilla", "riemann_right")

    def test_batched_multi_input_smoothgrad(self) -> None:
        self._assert_batched_tensor_multi_input("smoothgrad", "riemann_left")

    def test_batched_multi_input_smoothgrad_sq(self) -> None:
        self._assert_batched_tensor_multi_input("smoothgrad_sq", "riemann_middle")

    def test_batched_multi_input_vargrad(self) -> None:
        self._assert_batched_tensor_multi_input("vargrad", "riemann_trapezoid")

    def test_batched_multi_input_vargrad_batch_size_1(self) -> None:
        self._assert_batched_tensor_multi_input("vargrad", "riemann_trapezoid", 1)

    def test_batched_multi_input_smooth_batch_size_2(self) -> None:
        self._assert_batched_tensor_multi_input("vargrad", "riemann_trapezoid", 2)

    def test_batched_multi_input_smoothgrad_sq_batch_size_3(self) -> None:
        self._assert_batched_tensor_multi_input("vargrad", "riemann_trapezoid", 3)

    def test_futures_not_implemented(self) -> None:
        model = BasicModel2()
        ig = IntegratedGradients(model, multiply_by_inputs=True)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = ig.attribute_future()  # type: ignore
        self.assertEqual(attributions, None)

    def _assert_multi_variable(
        self,
        type: str,
        approximation_method: str = "gausslegendre",
        multiply_by_inputs: bool = True,
    ) -> None:
        model = BasicModel2()

        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0], requires_grad=True)

        baseline1 = torch.tensor([0.0])
        baseline2 = torch.tensor([0.0])

        attributions1 = self._compute_attribution_and_evaluate(
            model,
            (input1, input2),
            (baseline1, baseline2),
            type=type,
            approximation_method=approximation_method,
            multiply_by_inputs=multiply_by_inputs,
        )
        if type == "vanilla":
            assertTensorAlmostEqual(
                self,
                attributions1[0],
                [1.5] if multiply_by_inputs else [0.5],
                delta=0.05,
                mode="max",
            )
            assertTensorAlmostEqual(
                self,
                attributions1[1],
                [-0.5] if multiply_by_inputs else [-0.5],
                delta=0.05,
                mode="max",
            )
        model = BasicModel3()
        attributions2 = self._compute_attribution_and_evaluate(
            model,
            (input1, input2),
            (baseline1, baseline2),
            type=type,
            approximation_method=approximation_method,
            multiply_by_inputs=multiply_by_inputs,
        )
        if type == "vanilla":
            assertTensorAlmostEqual(
                self,
                attributions2[0],
                [1.5] if multiply_by_inputs else [0.5],
                delta=0.05,
                mode="max",
            )
            assertTensorAlmostEqual(
                self,
                attributions2[1],
                [-0.5] if multiply_by_inputs else [-0.5],
                delta=0.05,
                mode="max",
            )
            # Verifies implementation invariance
            self.assertEqual(
                sum(attribution for attribution in attributions1),
                sum(attribution for attribution in attributions2),
            )

    def _assert_univariable(
        self, type: str, approximation_method: str = "gausslegendre"
    ) -> None:
        model = BasicModel()
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([1.0], requires_grad=True),
            torch.tensor([0.0]),
            type=type,
            approximation_method=approximation_method,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            type=type,
            approximation_method=approximation_method,
        )
        self._compute_attribution_and_evaluate(
            model,
            torch.tensor([-1.0], requires_grad=True),
            0.00001,
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_multi_argument(
        self, type: str, approximation_method: str = "gausslegendre"
    ) -> None:
        model = BasicModel4_MultiArgs()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2]], requires_grad=True),
            ),
            baselines=(0.0, torch.zeros((1, 3))),
            additional_forward_args=torch.arange(1.0, 4.0).reshape(1, 3),
            type=type,
            approximation_method=approximation_method,
        )
        # uses batching with an integer variable and nd-tensors as
        # additional forward arguments
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((2, 3)), 0.0),
            additional_forward_args=(torch.arange(1.0, 7.0).reshape(2, 3), 1),
            type=type,
            approximation_method=approximation_method,
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
            baselines=(0.0, 0.00001),
            additional_forward_args=([2, 3], 1),
            type=type,
            approximation_method=approximation_method,
        )
        # similar to previous case plus baseline consists of a tensor and
        # a single example
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
            ),
            baselines=(torch.zeros((1, 3)), 0.00001),
            additional_forward_args=([2, 3], 1),
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_multi_tensor_input(
        self, type: str, approximation_method: str = "gausslegendre"
    ) -> None:
        model = BasicModel6_MultiTensor()
        self._compute_attribution_and_evaluate(
            model,
            (
                torch.tensor([[1.5, 2.0, 3.3]], requires_grad=True),
                torch.tensor([[3.0, 3.5, 2.2]], requires_grad=True),
            ),
            type=type,
            approximation_method=approximation_method,
        )

    def _assert_batched_tensor_input(
        self, type: str, approximation_method: str = "gausslegendre"
    ) -> None:
        model = BasicModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(
            model, input, type=type, target=0, approximation_method=approximation_method
        )
        self._compute_attribution_batch_helper_evaluate(
            model, input, target=0, approximation_method=approximation_method
        )

    def _assert_batched_tensor_multi_input(
        self,
        type: str,
        approximation_method: str = "gausslegendre",
        nt_samples_batch_size: Optional[int] = None,
    ) -> None:
        model = BasicModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.1, 1.9], [0.5, 0.0, 0.7], [1.5, 2.1, 1.1]], requires_grad=True
            ),
            torch.tensor(
                [[0.3, 1.9, 2.4], [0.5, 0.6, 2.1], [1.2, 2.1, 0.2]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(
            model,
            input,
            type=type,
            target=0,
            approximation_method=approximation_method,
            nt_samples_batch_size=nt_samples_batch_size,
        )

    def _assert_n_samples_batched_size(
        self,
        type: str,
        approximation_method: str = "gausslegendre",
        nt_samples_batch_size: Optional[int] = None,
    ) -> None:
        model = BasicModel_MultiLayer()
        input = (
            torch.tensor(
                [[1.5, 2.0, 1.3], [0.5, 0.1, 2.3], [1.5, 2.0, 1.3]], requires_grad=True
            ),
        )
        self._compute_attribution_and_evaluate(
            model,
            input,
            type=type,
            target=0,
            nt_samples_batch_size=nt_samples_batch_size,
            approximation_method=approximation_method,
        )

    def _compute_attribution_and_evaluate(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: Union[None, int] = None,
        additional_forward_args: Optional[object] = None,
        type: str = "vanilla",
        approximation_method: str = "gausslegendre",
        multiply_by_inputs: bool = True,
        nt_samples_batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, ...]:
        r"""
        attrib_type: 'vanilla', 'smoothgrad', 'smoothgrad_sq', 'vargrad'
        """
        ig = IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs)
        self.assertEqual(ig.multiplies_by_inputs, multiply_by_inputs)
        inputs_tuple: Tuple[Tensor, ...]
        if not isinstance(inputs, tuple):
            inputs_tuple = (cast(Tensor, inputs),)
        else:
            inputs_tuple = inputs

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = _tensorize_baseline(inputs_tuple, _zeros(inputs_tuple))

        if type == "vanilla":
            attributions, delta = ig.attribute(  # type: ignore[has-type]
                inputs_tuple,
                baselines,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                target=target,
                return_convergence_delta=True,
            )
            model.zero_grad()
            attributions_without_delta, delta = ig.attribute(  # type: ignore[has-type]
                inputs_tuple,
                baselines,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                target=target,
                return_convergence_delta=True,
            )
            model.zero_grad()
            self.assertEqual([inputs_tuple[0].shape[0]], list(delta.shape))
            delta_external = ig.compute_convergence_delta(
                attributions,
                baselines,
                inputs_tuple,
                target=target,
                additional_forward_args=additional_forward_args,
            )
            assertTensorAlmostEqual(self, delta, delta_external, delta=0.0, mode="max")
        else:
            nt = NoiseTunnel(ig)
            n_samples = 5
            attributions, delta = nt.attribute(
                inputs_tuple,
                nt_type=type,
                nt_samples=n_samples,
                stdevs=0.00000002,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                return_convergence_delta=True,
                nt_samples_batch_size=nt_samples_batch_size,
            )
            attributions_without_delta = nt.attribute(
                inputs_tuple,
                nt_type=type,
                nt_samples=n_samples,
                stdevs=0.00000002,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=500,
                nt_samples_batch_size=3,
            )
            self.assertEqual(nt.multiplies_by_inputs, multiply_by_inputs)
            self.assertEqual([inputs_tuple[0].shape[0] * n_samples], list(delta.shape))

        for input, attribution in zip(inputs_tuple, attributions):
            self.assertEqual(attribution.shape, input.shape)
        if multiply_by_inputs:
            assertTensorAlmostEqual(self, delta, torch.zeros(delta.shape), 0.07, "max")

        # compare attributions retrieved with and without
        # `return_convergence_delta` flag

        for attribution, attribution_without_delta in zip(
            attributions, attributions_without_delta
        ):
            assertTensorAlmostEqual(
                self, attribution, attribution_without_delta, delta=0.05
            )

        return cast(Tuple[Tensor, ...], attributions)

    def _compute_attribution_batch_helper_evaluate(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        target: Optional[int] = None,
        additional_forward_args: Optional[object] = None,
        approximation_method: str = "gausslegendre",
    ) -> None:
        ig = IntegratedGradients(model)
        inputs_tuple: Tuple[Tensor, ...]
        if not isinstance(inputs, tuple):
            inputs_tuple = (cast(Tensor, inputs),)
        else:
            inputs_tuple = inputs

        if baselines is not None and not isinstance(baselines, tuple):
            baselines = (baselines,)

        if baselines is None:
            baselines = _tensorize_baseline(inputs_tuple, _zeros(inputs_tuple))

        for internal_batch_size in [None, 10, 20]:
            attributions, delta = ig.attribute(  # type: ignore[has-type]
                inputs_tuple,
                baselines,
                additional_forward_args=additional_forward_args,
                method=approximation_method,
                n_steps=100,
                target=target,
                internal_batch_size=internal_batch_size,
                return_convergence_delta=True,
            )
            total_delta = 0.0
            for i in range(inputs_tuple[0].shape[0]):
                attrib_indiv, delta_indiv = ig.attribute(  # type: ignore[has-type]
                    tuple(input[i : i + 1] for input in inputs_tuple),
                    tuple(baseline[i : i + 1] for baseline in baselines),
                    additional_forward_args=additional_forward_args,
                    method=approximation_method,
                    n_steps=100,
                    target=target,
                    internal_batch_size=internal_batch_size,
                    return_convergence_delta=True,
                )
                total_delta += abs(delta_indiv).sum().item()
                for j in range(len(attributions)):
                    assertTensorAlmostEqual(
                        self,
                        attributions[j][i : i + 1].squeeze(0),
                        attrib_indiv[j].squeeze(0),
                        delta=0.05,
                        mode="max",
                    )
            self.assertAlmostEqual(abs(delta).sum().item(), total_delta, delta=0.005)


if __name__ == "__main__":
    unittest.main()
