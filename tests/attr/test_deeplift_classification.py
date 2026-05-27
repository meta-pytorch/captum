#!/usr/bin/env python3

# pyre-strict

from typing import TypeVar, Union

import torch
import torch.nn.functional as F
from captum._utils.typing import TargetType
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap, maxpool1d, softmax
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


class FunctionalExpModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.exp(input).sum(dim=1)


class FunctionalSquareModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        return (input * input).sum(dim=1)


class FunctionalSigmoidProductModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        return (self.sigmoid(1.702 * input) * input).sum(dim=1)


class FunctionalSoftmaxProfileModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        logits = input - input.mean(dim=-1, keepdim=True)
        probs = F.softmax(logits, dim=-1)
        return (probs * logits).sum(dim=-1)


class FunctionalUnaryNonlinearModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        positive_input = input + 4
        return (
            F.relu(input)
            + F.leaky_relu(input, negative_slope=0.2)
            + torch.sigmoid(input)
            + torch.tanh(input)
            + F.softplus(input)
            + F.elu(input)
            + F.selu(input)
            + F.gelu(input)
            + torch.rsqrt(positive_input)
            + torch.clamp(input, min=-0.25, max=1.25)
        ).sum(dim=1)


class FunctionalMinMaxModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        other = torch.flip(input, dims=[1]) + 0.25
        return (
            torch.minimum(input, other)
            + torch.maximum(0.5 * input, other)
            + torch.min(input + 0.1, other)
            + torch.max(input - 0.1, other)
        ).sum(dim=1)


class FunctionalConstantMinMaxModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        lower = torch.zeros_like(input)
        upper = torch.full_like(input, 0.5)
        return (torch.maximum(input, lower) + torch.minimum(input, upper)).sum(dim=1)


class FunctionalMatmulModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        left = input.unsqueeze(2)
        right = input.unsqueeze(1)
        return torch.matmul(left, right).sum(dim=(1, 2))


class FunctionalMaxReductionModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.amax(input, dim=1)


class FunctionalTorchMaxReductionModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        return torch.max(input, dim=1).values


class FunctionalMaxPool2dModel(Module):
    def forward(self, input: Tensor) -> Tensor:
        return F.max_pool2d(input, 2).flatten(1).sum(dim=1)


class SeluGeluModel(Module):
    def __init__(self) -> None:
        super().__init__()
        self.selu = torch.nn.SELU()
        self.gelu = torch.nn.GELU()

    def forward(self, input: Tensor) -> Tensor:
        return self.gelu(self.selu(input)).sum(dim=1)


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

    def test_functional_exp_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalExpModel(), input, baseline
        )

    def test_functional_multiply_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalSquareModel(), input, baseline
        )
        self._assert_functional_model_completeness(
            FunctionalSigmoidProductModel(), input, baseline
        )

    def test_functional_softmax_profile_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalSoftmaxProfileModel(), input, baseline
        )

    def test_functional_unary_nonlinear_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalUnaryNonlinearModel(), input, baseline
        )

    def test_functional_binary_min_max_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalMinMaxModel(), input, baseline
        )
        self._assert_functional_model_completeness(
            FunctionalConstantMinMaxModel(), input, baseline
        )

    def test_functional_matmul_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalMatmulModel(), input, baseline
        )

    def test_functional_max_reduction_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(
            FunctionalMaxReductionModel(), input, baseline
        )
        self._assert_functional_model_completeness(
            FunctionalTorchMaxReductionModel(), input, baseline
        )

    def test_functional_max_pool_rescale(self) -> None:
        input = torch.tensor(
            [
                [[[1.0, 0.0], [0.0, 0.0]]],
                [[[0.0, 3.0], [1.0, 2.0]]],
            ],
            requires_grad=True,
        )
        baseline = torch.tensor(
            [
                [[[0.0, 2.0], [0.0, 0.0]]],
                [[[1.0, 0.0], [4.0, 0.0]]],
            ]
        )

        self._assert_functional_model_completeness(
            FunctionalMaxPool2dModel(), input, baseline
        )

    def test_selu_gelu_module_rescale(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baseline = torch.zeros_like(input)

        self._assert_functional_model_completeness(SeluGeluModel(), input, baseline)

    def test_deeplift_shap_functional_tensor_ops(self) -> None:
        input = torch.tensor([[1.0, -2.0, 0.5], [-0.5, 2.0, 1.5]], requires_grad=True)
        baselines = torch.tensor(
            [[0.0, 0.0, 0.0], [0.25, -0.25, 0.5], [-0.5, 0.5, -0.25]]
        )

        for model in (
            FunctionalExpModel(),
            FunctionalSquareModel(),
            FunctionalSoftmaxProfileModel(),
            FunctionalUnaryNonlinearModel(),
            FunctionalMinMaxModel(),
            FunctionalConstantMinMaxModel(),
            FunctionalMatmulModel(),
            FunctionalMaxReductionModel(),
            FunctionalTorchMaxReductionModel(),
            SeluGeluModel(),
        ):
            _, delta = DeepLiftShap(model).attribute(  # type: ignore[has-type]
                input,
                baselines=baselines,
                return_convergence_delta=True,
            )
            self.assertTrue(
                (delta.abs() < 0.0001).all(),
                "Functional tensor op DeepLiftShap delta is too large: {}".format(
                    delta
                ),
            )

    def test_maxpool_uses_full_grad_input_for_equal_deltas(self) -> None:
        module = torch.nn.MaxPool1d(kernel_size=2, stride=2)
        inputs = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0]],
                [[5.0, 6.0, 7.0, 8.0]],
                [[1.0, 0.0, 9.0, 0.0]],
                [[0.0, 6.0, 0.0, 8.0]],
            ]
        )
        outputs = module(inputs)
        module.input = inputs  # type: ignore[attr-defined]
        grad_input = torch.arange(inputs.numel(), dtype=inputs.dtype).view_as(inputs)
        grad_output = torch.ones_like(outputs)

        multipliers = maxpool1d(module, inputs, outputs, grad_input, grad_output)

        zero_delta_mask = (inputs[:2] - inputs[2:]).abs() < 1e-10
        zero_delta_mask = torch.cat(2 * [zero_delta_mask])
        self.assertTrue(
            torch.equal(multipliers[zero_delta_mask], grad_input[zero_delta_mask])
        )

    def test_softmax_uses_rescale_without_normalization(self) -> None:
        module = torch.nn.Softmax(dim=1)
        inputs = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [3.0, 1.0, 0.0],
                [0.5, 1.0, 1.5],
                [1.0, 0.0, -1.0],
            ]
        )
        outputs = module(inputs)
        grad_input = torch.full_like(inputs, 0.25)
        grad_output = torch.ones_like(outputs)

        multipliers = softmax(module, inputs, outputs, grad_input, grad_output)

        delta_in = inputs[:2] - inputs[2:]
        delta_out = outputs[:2] - outputs[2:]
        delta_in = torch.cat(2 * [delta_in])
        delta_out = torch.cat(2 * [delta_out])
        expected = torch.where(
            delta_in.abs() < 1e-10, grad_input, grad_output * delta_out / delta_in
        )
        torch.testing.assert_close(multipliers, expected)

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

    def _assert_functional_model_completeness(
        self, model: Module, input: Tensor, baseline: Tensor
    ) -> None:
        attributions, delta = DeepLift(model).attribute(  # type: ignore[has-type]
            input,
            baselines=baseline,
            return_convergence_delta=True,
        )
        attr_sum = attributions.reshape(attributions.shape[0], -1).sum(dim=1)
        output_diff = model(input) - model(baseline)
        torch.testing.assert_close(attr_sum, output_diff, atol=0.0001, rtol=0.0001)
        self.assertTrue(
            (delta.abs() < 0.0001).all(),
            "Functional tensor op DeepLift delta is too large: {}".format(delta),
        )
