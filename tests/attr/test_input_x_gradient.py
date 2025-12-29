#!/usr/bin/env python3

# pyre-strict
from typing import cast, Optional

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.testing.attr.helpers.get_config_util import (
    get_basic_config,
    get_multiargs_basic_config,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.classification_models import SoftmaxModel
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_input_x_gradient_test_basic_vanilla(self) -> None:
        self._input_x_gradient_base_assert(*get_basic_config())

    def test_input_x_gradient_test_basic_smoothgrad(self) -> None:
        self._input_x_gradient_base_assert(*get_basic_config(), nt_type="smoothgrad")

    def test_input_x_gradient_test_basic_vargrad(self) -> None:
        self._input_x_gradient_base_assert(*get_basic_config(), nt_type="vargrad")

    def test_saliency_test_basic_multi_variable_vanilla(self) -> None:
        self._input_x_gradient_base_assert(*get_multiargs_basic_config())

    def test_saliency_test_basic_multi_variable_smoothgrad(self) -> None:
        self._input_x_gradient_base_assert(
            *get_multiargs_basic_config(), nt_type="smoothgrad"
        )

    def test_saliency_test_basic_multi_vargrad(self) -> None:
        self._input_x_gradient_base_assert(
            *get_multiargs_basic_config(), nt_type="vargrad"
        )

    def test_input_x_gradient_classification_vanilla(self) -> None:
        self._input_x_gradient_classification_assert()

    def test_input_x_gradient_classification_smoothgrad(self) -> None:
        self._input_x_gradient_classification_assert(nt_type="smoothgrad")

    def test_input_x_gradient_classification_vargrad(self) -> None:
        self._input_x_gradient_classification_assert(nt_type="vargrad")

    def test_futures_not_implemented(self) -> None:
        model = SoftmaxModel(5, 20, 10)
        input_x_grad = InputXGradient(model.forward)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = input_x_grad.attribute_future()  # type: ignore
        self.assertEqual(attributions, None)

    def _input_x_gradient_base_assert(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected_grads: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Optional[object] = None,
        nt_type: str = "vanilla",
    ) -> None:
        input_x_grad = InputXGradient(model)
        self.assertTrue(input_x_grad.multiplies_by_inputs)
        attributions: TensorOrTupleOfTensorsGeneric
        if nt_type == "vanilla":
            attributions = input_x_grad.attribute(
                inputs,
                additional_forward_args=additional_forward_args,
            )
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                inputs,
                nt_type=nt_type,
                nt_samples=10,
                stdevs=0.0002,
                additional_forward_args=additional_forward_args,
            )

        if isinstance(attributions, tuple):
            for input, attribution, expected_grad in zip(
                inputs, attributions, expected_grads
            ):
                if nt_type == "vanilla":
                    self._assert_attribution(expected_grad, input, attribution)
                self.assertEqual(input.shape, attribution.shape)
        elif isinstance(attributions, Tensor):
            if nt_type == "vanilla":
                self._assert_attribution(
                    cast(Tensor, expected_grads), cast(Tensor, inputs), attributions
                )
            self.assertEqual(cast(Tensor, inputs).shape, attributions.shape)

    def _assert_attribution(
        self, expected_grad: Tensor, input: Tensor, attribution: Tensor
    ) -> None:
        assertTensorAlmostEqual(
            self,
            attribution,
            (expected_grad * input),
            delta=0.05,
            mode="max",
        )

    def _input_x_gradient_classification_assert(self, nt_type: str = "vanilla") -> None:
        num_in = 5
        input = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0]], requires_grad=True)
        target = torch.tensor(5)

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        input_x_grad = InputXGradient(model.forward)
        if nt_type == "vanilla":
            attributions = input_x_grad.attribute(input, target)
            output = model(input)[:, target]
            output.backward()
            input_grad = input.grad
            assert input_grad is not None
            expected = input_grad * input
            assertTensorAlmostEqual(self, attributions, expected, 0.00001, "max")
        else:
            nt = NoiseTunnel(input_x_grad)
            attributions = nt.attribute(
                input, nt_type=nt_type, nt_samples=10, stdevs=1.0, target=target
            )

        self.assertEqual(attributions.shape, input.shape)
