#!/usr/bin/env python3

# pyre-strict

from typing import TypeVar, Union

import torch
from captum._utils.typing import TargetType
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.testing.helpers.basic import assertAttributionComparision, BaseTest
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_ConvNet_MaxPool1d,
    BasicModel_ConvNet_MaxPool3d,
)
from captum.testing.helpers.classification_models import (
    SigmoidDeepLiftModel,
    SoftmaxDeepLiftModel,
)
from torch import Tensor
from torch.nn import Module

DeepLiftAttrMethod = TypeVar("DeepLiftAttrMethod", DeepLift, DeepLiftShap)


class Test(BaseTest):
    def test_sigmoid_classification(self) -> None:
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baseline = 0 * input
        target = torch.tensor(0)
        # TODO add test cases for multiple different layers
        model = SigmoidDeepLiftModel(num_in, 5, 1)
        dl = DeepLift(model)
        model.zero_grad()
        attributions, delta = dl.attribute(  # type: ignore[has-type]
            input, baseline, target=target, return_convergence_delta=True
        )
        self._assert_attributions(model, attributions, input, baseline, delta, target)

        # compare with integrated gradients
        ig = IntegratedGradients(model)
        attributions_ig = ig.attribute(  # type: ignore[has-type]
            input, baseline, target=target
        )
        assertAttributionComparision(self, (attributions,), (attributions_ig,))

    def test_softmax_classification_zero_baseline(self) -> None:
        num_in = 20
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baselines = 0.0

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baselines, torch.tensor(2))

    def test_softmax_classification_batch_zero_baseline(self) -> None:
        num_in = 40
        input = torch.arange(0.0, num_in * 3.0, requires_grad=True).reshape(3, num_in)
        baselines = 0
        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLift(model)

        self.softmax_classification(
            model, dl, input, baselines, torch.tensor([2, 2, 2])
        )

    def test_softmax_classification_batch_multi_target(self) -> None:
        num_in = 40
        inputs = (
            torch.arange(0.0, num_in * 3.0, requires_grad=True)
            .reshape(3, num_in)
            .double()
        )
        baselines = torch.arange(1.0, num_in + 1).reshape(1, num_in).double()
        model = SoftmaxDeepLiftModel(num_in, 20, 10).double()
        dl = DeepLift(model)

        self.softmax_classification(
            model, dl, inputs, baselines, torch.tensor([2, 2, 2])
        )

    def test_softmax_classification_multi_baseline(self) -> None:
        num_in = 40
        input = torch.arange(0.0, num_in * 1.0, requires_grad=True).unsqueeze(0)
        baselines = torch.randn(5, 40)

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLiftShap(model)

        self.softmax_classification(model, dl, input, baselines, torch.tensor(2))

    def test_softmax_classification_batch_multi_baseline(self) -> None:
        num_in = 40
        input = torch.arange(0.0, num_in * 2.0, requires_grad=True).reshape(2, num_in)
        baselines = torch.randn(5, 40)

        model = SoftmaxDeepLiftModel(num_in, 20, 10)
        dl = DeepLiftShap(model)

        self.softmax_classification(model, dl, input, baselines, torch.tensor(2))

    def test_convnet_with_maxpool3d(self) -> None:
        input = 100 * torch.randn(2, 1, 10, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10, 10)

        model = BasicModel_ConvNet_MaxPool3d()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def test_convnet_with_maxpool3d_large_baselines(self) -> None:
        input = 100 * torch.randn(2, 1, 10, 10, 10, requires_grad=True)
        baseline = 600 * torch.randn(2, 1, 10, 10, 10)

        model = BasicModel_ConvNet_MaxPool3d()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def test_convnet_with_maxpool2d(self) -> None:
        input = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10, 10)

        model = BasicModel_ConvNet()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def test_convnet_with_maxpool2d_large_baselines(self) -> None:
        input = 100 * torch.randn(2, 1, 10, 10, requires_grad=True)
        baseline = 500 * torch.randn(2, 1, 10, 10)

        model = BasicModel_ConvNet()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def test_convnet_with_maxpool1d(self) -> None:
        input = 100 * torch.randn(2, 1, 10, requires_grad=True)
        baseline = 20 * torch.randn(2, 1, 10)

        model = BasicModel_ConvNet_MaxPool1d()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def test_convnet_with_maxpool1d_large_baselines(self) -> None:
        input = 100 * torch.randn(2, 1, 10, requires_grad=True)
        baseline = 500 * torch.randn(2, 1, 10)

        model = BasicModel_ConvNet_MaxPool1d()
        dl = DeepLift(model)

        self.softmax_classification(model, dl, input, baseline, torch.tensor(2))

    def softmax_classification(
        self,
        model: Module,
        attr_method: DeepLiftAttrMethod,
        input: Tensor,
        baselines: Union[float, int, Tensor],
        target: TargetType,
    ) -> None:
        # TODO add test cases for multiple different layers
        if isinstance(attr_method, DeepLiftShap):
            assert isinstance(
                baselines, Tensor
            ), "Non-tensor baseline not supported for DeepLiftShap"

        model.zero_grad()
        attributions, delta = attr_method.attribute(  # type: ignore[has-type]
            input, baselines=baselines, target=target, return_convergence_delta=True
        )
        self._assert_attributions(model, attributions, input, baselines, delta, target)

        target2 = torch.tensor(1)
        attributions, delta = attr_method.attribute(  # type: ignore[has-type]
            input, baselines=baselines, target=target2, return_convergence_delta=True
        )

        self._assert_attributions(model, attributions, input, baselines, delta, target2)

    def _assert_attributions(
        self,
        model: Module,
        attributions: Tensor,
        inputs: Tensor,
        baselines: Union[Tensor, int, float],
        delta: Tensor,
        target: TargetType = None,
    ) -> None:
        self.assertEqual(inputs.shape, attributions.shape)

        delta_condition = (delta.abs() < 0.003).all()
        self.assertTrue(
            delta_condition,
            "The sum of attribution values {} is not "
            "nearly equal to the difference between the endpoint for "
            "some samples".format(delta),
        )
        # compare with integrated gradients
        if isinstance(baselines, (int, float)) or inputs.shape == baselines.shape:
            ig = IntegratedGradients(model)
            attributions_ig = ig.attribute(  # type: ignore[has-type]
                inputs, baselines=baselines, target=target
            )
            assertAttributionComparision(self, attributions, attributions_ig)
