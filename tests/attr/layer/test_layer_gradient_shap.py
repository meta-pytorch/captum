#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.layer.layer_gradient_shap import (
    LayerGradientShap,
    LayerInputBaselineXGradient,
)
from captum.testing.attr.helpers.attribution_delta_util import assert_attribution_delta
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from captum.testing.helpers.classification_models import SoftmaxModel
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_basic_multilayer(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, -20.0, 10.0]])
        baselines = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        expected = [[-8.4, 0.0]]

        self._assert_attributions(model, model.linear2, inputs, baselines, 0, expected)

    def test_basic_multilayer_wo_multiplying_by_inputs(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, -20.0, 10.0]])
        baselines = torch.zeros(3, 3)
        lgs = LayerGradientShap(model, model.linear2, multiply_by_inputs=False)
        attrs = lgs.attribute(  # type: ignore[has-type]
            inputs,
            baselines,
            target=0,
            stdevs=0.0,
        )
        assertTensorAlmostEqual(self, attrs, torch.tensor([[1.0, 0.0]]))

    def test_basic_multi_tensor_output(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        model.eval()

        inputs = torch.tensor([[0.0, 100.0, 0.0]])
        expected = ([[90.0, 100.0, 100.0, 100.0]], [[90.0, 100.0, 100.0, 100.0]])
        self._assert_attributions(
            model,
            model.multi_relu,
            inputs,
            torch.zeros_like(inputs),
            0,
            expected,
            n_samples=5,
        )

    def test_basic_multilayer_with_add_args(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, -20.0, 10.0]])
        add_args = torch.ones(1, 3)
        baselines = torch.randn(30, 3)
        expected = [[-13.9510, 0.0]]

        self._assert_attributions(
            model, model.linear2, inputs, baselines, 0, expected, add_args=add_args
        )

    def test_basic_multilayer_compare_w_inp_features(self) -> None:
        model = BasicModel_MultiLayer()
        model.eval()

        inputs = torch.tensor([[10.0, 20.0, 10.0]])
        baselines = torch.randn(30, 3)

        gs = GradientShap(model)
        expected, delta = gs.attribute(
            inputs, baselines, target=0, return_convergence_delta=True
        )
        self.setUp()
        self._assert_attributions(
            model,
            model.linear0,
            inputs,
            baselines,
            0,
            expected,
            expected_delta=delta,
            attribute_to_layer_input=True,
        )

    def test_classification(self) -> None:
        def custom_baseline_fn(inputs: Tensor) -> Tensor:
            num_in = inputs.shape[1]
            return torch.arange(0.0, num_in * 4.0).reshape(4, num_in)

        num_in = 40
        n_samples = 10

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()

        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = custom_baseline_fn
        expected = torch.zeros(2, 20)

        self._assert_attributions(
            model, model.relu1, inputs, baselines, 1, expected, n_samples=n_samples
        )

    def test_basic_multi_input(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()

        inputs = (torch.tensor([[10.0, 20.0, 10.0]]), torch.tensor([[1.0, 2.0, 1.0]]))
        add_args = (torch.tensor([[1.0, 2.0, 3.0]]), 1.0)
        baselines = (torch.randn(30, 3), torch.randn(30, 3))
        expected = torch.tensor([[171.6841, 0.0]])
        self._assert_attributions(
            net, net.model.linear2, inputs, baselines, 0, expected, add_args=add_args
        )

    def test_relu_grad_shap_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )

        model = BasicModel_MultiLayer(inplace=True, multi_input_module=True)
        model.eval()

        inputs = torch.tensor([[1.0, -20.0, 10.0]], requires_grad=True)
        baselines = torch.zeros(3, 3)
        lgs = LayerInputBaselineXGradient(model, model.relu, multiply_by_inputs=False)
        attrs = lgs.attribute(
            inputs, baselines, target=0, grad_kwargs={"materialize_grads": True}
        )

        assertTensorAlmostEqual(
            self,
            attrs,
            torch.tensor(
                [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
            ),
        )

    def _assert_attributions(
        self,
        model: Module,
        layer: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        baselines: Union[TensorOrTupleOfTensorsGeneric, Callable],
        target: TargetType,
        expected: Union[
            Tensor,
            Tuple[Tensor, ...],
            List[float],
            List[List[float]],
            Tuple[List[float], ...],
            Tuple[List[List[float]], ...],
        ],
        expected_delta: Optional[Tensor] = None,
        n_samples: int = 5,
        attribute_to_layer_input: bool = False,
        add_args: Any = None,
    ) -> None:
        lgs = LayerGradientShap(model, layer)
        attrs, delta = lgs.attribute(  # type: ignore[has-type]
            inputs,
            baselines,
            target=target,
            additional_forward_args=add_args,
            n_samples=n_samples,
            stdevs=0.0009,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assertTensorTuplesAlmostEqual(self, attrs, expected, delta=0.005)
        if expected_delta is None:
            assert_attribution_delta(
                # pyre-fixme[6]: For 1st argument expected `FbBaseTest` but got `Test`.
                self,  # type: ignore
                inputs,
                attrs,
                n_samples,
                delta,
                is_layer=True,
            )
        else:
            for delta_i, expected_delta_i in zip(delta, expected_delta):
                assertTensorAlmostEqual(self, delta_i, expected_delta_i, delta=0.01)
