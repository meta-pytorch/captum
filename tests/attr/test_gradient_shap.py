#!/usr/bin/env python3

# pyre-strict

from typing import cast, Tuple

import numpy as np
import numpy.typing as npt
import torch
from captum._utils.typing import Tensor
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.testing.attr.helpers.attribution_delta_util import (
    assert_attribution_delta,
    assert_delta,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicLinearModel, BasicModel2
from captum.testing.helpers.classification_models import SoftmaxModel


class Test(BaseTest):

    # This test reproduces some of the test cases from the original implementation
    # https://github.com/slundberg/shap/
    # explainers/test_gradient.py
    def test_basic_multi_input(self) -> None:
        batch_size = 10

        x1 = torch.ones(batch_size, 3)
        x2 = torch.ones(batch_size, 4)
        inputs = (x1, x2)

        batch_size_baselines = 20
        baselines = (
            torch.zeros(batch_size_baselines, 3),
            torch.zeros(batch_size_baselines, 4),
        )

        model = BasicLinearModel()
        model.eval()
        model.zero_grad()

        np.random.seed(0)
        torch.manual_seed(0)
        gradient_shap = GradientShap(model)
        n_samples = 50
        attributions, delta = cast(
            Tuple[Tuple[Tensor, ...], Tensor],
            gradient_shap.attribute(
                inputs, baselines, n_samples=n_samples, return_convergence_delta=True
            ),
        )
        attributions_without_delta = gradient_shap.attribute((x1, x2), baselines)

        assert_attribution_delta(self, inputs, attributions, n_samples, delta)
        # Compare with integrated gradients
        ig = IntegratedGradients(model)
        baselines = (torch.zeros(batch_size, 3), torch.zeros(batch_size, 4))
        attributions_ig = ig.attribute(  # type: ignore[has-type]
            inputs, baselines=baselines
        )
        self._assert_shap_ig_comparision(attributions, attributions_ig)

        # compare attributions retrieved with and without
        # `return_convergence_delta` flag
        for attribution, attribution_without_delta in zip(
            attributions, attributions_without_delta
        ):
            assertTensorAlmostEqual(self, attribution, attribution_without_delta)

    def test_basic_multi_input_wo_mutliplying_by_inputs(self) -> None:
        batch_size = 10

        x1 = torch.ones(batch_size, 3)
        x2 = torch.ones(batch_size, 4)
        inputs = (x1, x2)

        batch_size_baselines = 20
        baselines = (
            torch.ones(batch_size_baselines, 3) + 2e-5,
            torch.ones(batch_size_baselines, 4) + 2e-5,
        )

        model = BasicLinearModel()
        model.eval()
        model.zero_grad()

        np.random.seed(0)
        torch.manual_seed(0)
        gradient_shap = GradientShap(model)
        gradient_shap_wo_mutliplying_by_inputs = GradientShap(
            model, multiply_by_inputs=False
        )
        n_samples = 50
        attributions = cast(
            Tuple[Tuple[Tensor, ...], Tensor],
            gradient_shap.attribute(
                inputs,
                baselines,
                n_samples=n_samples,
                stdevs=0.0,
            ),
        )
        attributions_wo_mutliplying_by_inputs = cast(
            Tuple[Tuple[Tensor, ...], Tensor],
            gradient_shap_wo_mutliplying_by_inputs.attribute(
                inputs,
                baselines,
                n_samples=n_samples,
                stdevs=0.0,
            ),
        )
        assertTensorAlmostEqual(
            self,
            attributions_wo_mutliplying_by_inputs[0] * (x1 - baselines[0][0:1]),
            attributions[0],
        )
        assertTensorAlmostEqual(
            self,
            attributions_wo_mutliplying_by_inputs[1] * (x2 - baselines[1][0:1]),
            attributions[1],
        )

    def test_classification_baselines_as_function(self) -> None:
        num_in: int = 40
        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)

        def generate_baselines() -> Tensor:
            return torch.arange(0.0, num_in * 4.0).reshape(4, num_in)

        def generate_baselines_with_inputs(inputs: Tensor) -> Tensor:
            inp_shape = cast(Tuple[int, ...], inputs.shape)
            return torch.arange(0.0, inp_shape[1] * 2.0).reshape(2, inp_shape[1])

        def generate_baselines_returns_array() -> npt.NDArray:
            return np.arange(0.0, num_in * 4.0).reshape(4, num_in)

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()
        model.zero_grad()

        gradient_shap = GradientShap(model)
        n_samples = 10
        attributions, delta = gradient_shap.attribute(
            inputs,
            baselines=generate_baselines,
            target=torch.tensor(1),
            n_samples=n_samples,
            stdevs=0.009,
            return_convergence_delta=True,
        )
        assert_attribution_delta(self, (inputs,), (attributions,), n_samples, delta)

        attributions, delta = gradient_shap.attribute(
            inputs,
            baselines=generate_baselines_with_inputs,
            target=torch.tensor(1),
            n_samples=n_samples,
            stdevs=0.00001,
            return_convergence_delta=True,
        )
        assert_attribution_delta(self, (inputs,), (attributions,), n_samples, delta)

        with self.assertRaises(AssertionError):
            attributions, delta = gradient_shap.attribute(  # type: ignore
                inputs,
                # Intentionally passing wrong type.
                baselines=generate_baselines_returns_array,
                target=torch.tensor(1),
                n_samples=n_samples,
                stdevs=0.00001,
                return_convergence_delta=True,
            )

    def test_classification(self) -> None:
        num_in = 40
        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = torch.arange(0.0, num_in * 4.0).reshape(4, num_in)
        target = torch.tensor(1)
        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()
        model.zero_grad()

        gradient_shap = GradientShap(model)
        n_samples = 10
        attributions, delta = gradient_shap.attribute(
            inputs,
            baselines=baselines,
            target=target,
            n_samples=n_samples,
            stdevs=0.009,
            return_convergence_delta=True,
        )
        assert_attribution_delta(self, (inputs,), (attributions,), n_samples, delta)

        # try to call `compute_convergence_delta` externally
        with self.assertRaises(AssertionError):
            gradient_shap.compute_convergence_delta(
                attributions, inputs, baselines, target=target
            )
        # now, let's expand target and choose random baselines from `baselines` tensor
        rand_indices = np.random.choice(baselines.shape[0], inputs.shape[0]).tolist()
        chosen_baselines = baselines[rand_indices]

        target_extendes = torch.tensor([1, 1])
        external_delta = gradient_shap.compute_convergence_delta(
            attributions, chosen_baselines, inputs, target=target_extendes
        )
        assert_delta(self, external_delta)

        # Compare with integrated gradients
        ig = IntegratedGradients(model)
        baselines = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        attributions_ig = ig.attribute(  # type: ignore[has-type]
            inputs, baselines=baselines, target=target
        )
        self._assert_shap_ig_comparision((attributions,), (attributions_ig,))

    def test_basic_relu_multi_input(self) -> None:
        model = BasicModel2()

        input1 = torch.tensor([[3.0]])
        input2 = torch.tensor([[1.0]], requires_grad=True)

        baseline1 = torch.tensor([[0.0]])
        baseline2 = torch.tensor([[0.0]])
        inputs = (input1, input2)
        baselines = (baseline1, baseline2)

        gs = GradientShap(model)
        n_samples = 20000
        attributions, delta = cast(
            Tuple[Tuple[Tensor, ...], Tensor],
            gs.attribute(
                inputs,
                baselines=baselines,
                n_samples=n_samples,
                return_convergence_delta=True,
            ),
        )
        assert_attribution_delta(
            self, inputs, attributions, n_samples, delta, delta_thresh=0.008
        )

        ig = IntegratedGradients(model)
        attributions_ig = ig.attribute(  # type: ignore[has-type]
            inputs, baselines=baselines
        )
        self._assert_shap_ig_comparision(attributions, attributions_ig)

    def test_futures_not_implemented(self) -> None:
        model = BasicModel2()
        gs = GradientShap(model)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = gs.attribute_future()
        self.assertEqual(attributions, None)

    def _assert_shap_ig_comparision(
        self, attributions1: Tuple[Tensor, ...], attributions2: Tuple[Tensor, ...]
    ) -> None:
        for attribution1, attribution2 in zip(attributions1, attributions2):
            for attr_row1, attr_row2 in zip(attribution1, attribution2):
                assertTensorAlmostEqual(self, attr_row1, attr_row2, 0.05, "max")
