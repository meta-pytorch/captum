#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import cast, Tuple

import torch
import torch.nn as nn
from captum.attr import InputXGradient, LRP
from captum.attr._utils.lrp_rules import (
    Alpha1_Beta0_Rule,
    EpsilonRule,
    GammaRule,
    IdentityRule,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
    BasicModelWithReusedLinear,
    SimpleLRPModel,
)
from torch import Tensor
from torch.nn import Module


def _get_basic_config() -> Tuple[Module, Tensor]:
    input = torch.arange(16).view(1, 1, 4, 4).float()
    return BasicModel_ConvNet_One_Conv(), input


def _get_rule_config() -> Tuple[Tensor, Module, Tensor, Tensor]:
    relevance = torch.tensor([[[-0.0, 3.0]]])
    layer = nn.modules.Conv1d(1, 1, 2, bias=False)
    nn.init.constant_(layer.weight.data, 2)
    activations = torch.tensor([[[1.0, 5.0, 7.0]]])
    input = torch.tensor([[2, 0, -2]])
    return relevance, layer, activations, input


def _get_simple_model(inplace: bool = False) -> Tuple[Module, Tensor]:
    model = SimpleLRPModel(inplace)
    inputs = torch.tensor([[1.0, 2.0, 3.0]])

    return model, inputs


def _get_simple_model2(inplace: bool = False) -> Tuple[Module, Tensor]:
    class MyModel(nn.Module):
        def __init__(self, inplace) -> None:
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.lin.weight = nn.Parameter(torch.ones(2, 2))
            self.relu = torch.nn.ReLU(inplace=inplace)

        def forward(self, input):
            return self.relu(self.lin(input))[0].unsqueeze(0)

    input = torch.tensor([[1.0, 2.0], [1.0, 3.0]])
    model = MyModel(inplace)

    return model, input


# pyrefly: ignore [invalid-inheritance]
class Test(BaseTest):
    def _assert_lrp_supports_module(self, module: Module, inputs: Tensor) -> None:
        class SingleModuleModel(nn.Module):
            module: Module

            def __init__(self, module: Module) -> None:
                super().__init__()
                self.module = module

            def forward(self, input: Tensor) -> Tensor:
                return self.module(input).flatten(1).sum(dim=1)

        model = SingleModuleModel(module).eval()
        attributions = cast(Tensor, LRP(model).attribute(inputs))
        self.assertEqual(attributions.shape, inputs.shape)
        self.assertTrue(torch.isfinite(attributions).all())

    def test_lrp_creator(self) -> None:
        model, _ = _get_basic_config()
        model.conv1.rule = 1  # type: ignore
        self.assertRaises(TypeError, LRP, model)

    def test_lrp_creator_activation(self) -> None:
        model, inputs = _get_basic_config()
        model.add_module("sigmoid", nn.Sigmoid())
        lrp = LRP(model)
        self.assertRaises(TypeError, lrp.attribute, inputs)  # type: ignore[has-type]

    def test_lrp_basic_attributions(self) -> None:
        model, inputs = _get_basic_config()
        logits = model(inputs)
        _, classIndex = torch.max(logits, 1)
        lrp = LRP(model)
        relevance, delta = lrp.attribute(  # type: ignore[has-type]
            inputs, cast(int, classIndex.item()), return_convergence_delta=True
        )
        self.assertEqual(delta.item(), 0)  # type: ignore
        self.assertEqual(relevance.shape, inputs.shape)  # type: ignore
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor(
                [[[[0, 1, 2, 3], [0, 5, 6, 7], [0, 9, 10, 11], [0, 0, 0, 0]]]]
            ),
        )

    def test_lrp_simple_attributions(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance, torch.tensor([[18.0, 36.0, 54.0]]))

    def test_lrp_simple_attributions_batch(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = EpsilonRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        inputs = torch.cat((inputs, 3 * inputs))
        relevance, delta = lrp.attribute(  # type: ignore[has-type]
            inputs, target=0, return_convergence_delta=True
        )
        self.assertEqual(relevance.shape, inputs.shape)  # type: ignore
        self.assertEqual(delta.shape[0], inputs.shape[0])  # type: ignore
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[18.0, 36.0, 54.0], [54.0, 108.0, 162.0]])
        )

    def test_lrp_simple_repeat_attributions(self) -> None:
        model, inputs = _get_simple_model()
        model.eval()
        model.linear.rule = GammaRule()  # type: ignore
        model.linear2.rule = Alpha1_Beta0_Rule()  # type: ignore
        output = model(inputs)
        lrp = LRP(model)
        _ = lrp.attribute(inputs)  # type: ignore[has-type]
        output_after = model(inputs)
        assertTensorAlmostEqual(self, output, output_after)

    def test_lrp_preserves_custom_rules_for_repeated_attributions(self) -> None:
        class UnsupportedLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(1, 2))

            def forward(self, input: Tensor) -> Tensor:
                return input.matmul(self.weight.t())

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = UnsupportedLinear()
                self.layer.rule = EpsilonRule()  # type: ignore

            def forward(self, input: Tensor) -> Tensor:
                return self.layer(input)

        model = Model().eval()
        inputs = torch.tensor([[1.0, 2.0]])
        lrp = LRP(model)

        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        self.assertTrue(hasattr(model.layer, "rule"))
        repeated_relevance = lrp.attribute(inputs)  # type: ignore[has-type]

        assertTensorAlmostEqual(self, relevance, repeated_relevance)

    def test_lrp_simple_inplaceReLU(self) -> None:
        model_default, inputs = _get_simple_model()
        model_inplace, _ = _get_simple_model(inplace=True)
        for model in [model_default, model_inplace]:
            model.eval()
            model.linear.rule = EpsilonRule()  # type: ignore
            model.linear2.rule = EpsilonRule()  # type: ignore
        lrp_default = LRP(model_default)
        lrp_inplace = LRP(model_inplace)
        relevance_default = lrp_default.attribute(inputs)  # type: ignore[has-type]
        relevance_inplace = lrp_inplace.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance_default, relevance_inplace)

    def test_lrp_simple_tanh(self) -> None:
        class Model(nn.Module):
            def __init__(self) -> None:
                super(Model, self).__init__()
                self.linear = nn.Linear(3, 3, bias=False)
                self.linear.weight.data.fill_(0.1)
                self.tanh = torch.nn.Tanh()
                self.linear2 = nn.Linear(3, 1, bias=False)
                self.linear2.weight.data.fill_(0.1)

            def forward(self, x):
                return self.linear2(self.tanh(self.linear(x)))

        model = Model()
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        _ = model(inputs)
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[0.0269, 0.0537, 0.0806]])
        )  # Result if tanh is skipped for propagation

    def test_lrp_simple_attributions_GammaRule(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = GammaRule(gamma=1)  # type: ignore
        model.linear2.rule = GammaRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(
            self, relevance.data, torch.tensor([[28 / 3, 104 / 3, 52]])  # type: ignore
        )

    def test_lrp_simple_attributions_AlphaBeta(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = Alpha1_Beta0_Rule()  # type: ignore
        model.linear2.rule = Alpha1_Beta0_Rule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance, torch.tensor([[12, 33.6, 50.4]]))

    def test_lrp_Identity(self) -> None:
        model, inputs = _get_simple_model()
        with torch.no_grad():
            model.linear.weight.data[0][0] = -2  # type: ignore
        model.eval()
        model.linear.rule = IdentityRule()  # type: ignore
        model.linear2.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(inputs)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance, torch.tensor([[24.0, 36.0, 36.0]]))

    def test_lrp_simple2_attributions(self) -> None:
        model, input = _get_simple_model2()
        lrp = LRP(model)
        relevance = lrp.attribute(input, 0)  # type: ignore[has-type]
        self.assertEqual(relevance.shape, input.shape)  # type: ignore

    def test_lrp_skip_connection(self) -> None:
        # A custom addition module needs to be used so that relevance is
        # propagated correctly.
        class Addition_Module(nn.Module):
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
                return x1 + x2

        class SkipConnection(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(5)
                self.add = Addition_Module()

            def forward(self, input: Tensor) -> Module:
                x = self.add(self.linear(input), input)
                return x

        model = SkipConnection()
        input = torch.Tensor([[2, 3]])
        model.add.rule = EpsilonRule()  # type: ignore
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[10, 18]]))

    def test_lrp_maxpool1D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.linear = nn.Linear(2, 2, bias=False)
                self.linear.weight.data.fill_(2.0)
                self.maxpool = nn.MaxPool1d(2)

            def forward(self, input: Tensor) -> Module:
                return self.maxpool(self.linear(input))

        model = MaxPoolModel()
        input = torch.tensor([[[1.0, 2.0], [5.0, 6.0]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input, target=1)  # type: ignore[has-type]
        assertTensorAlmostEqual(self, relevance, torch.Tensor([[[0.0, 0.0], [10, 12]]]))

    def test_lrp_maxpool2D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool2d(2)

            def forward(self, input: Tensor) -> Module:
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[1.0, 2.0], [5.0, 6.0]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)  # type: ignore[has-type]
        assertTensorAlmostEqual(
            self, relevance, torch.Tensor([[[[0.0, 0.0], [0.0, 6.0]]]])
        )

    def test_lrp_maxpool3D(self) -> None:
        class MaxPoolModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.maxpool = nn.MaxPool3d(2)

            def forward(self, input: Tensor) -> Module:
                return self.maxpool(input)

        model = MaxPoolModel()
        input = torch.tensor([[[[[1.0, 2.0], [5.0, 6.0]], [[3.0, 4.0], [7.0, 8.0]]]]])
        lrp = LRP(model)
        relevance = lrp.attribute(input)  # type: ignore[has-type]
        assertTensorAlmostEqual(
            self,
            relevance,
            torch.Tensor([[[[[0.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 8.0]]]]]),
        )

    def test_lrp_conv1D_default_rule(self) -> None:
        conv = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        conv.weight.data.fill_(2.0)
        self._assert_lrp_supports_module(
            conv,
            torch.tensor([[[1.0, 2.0, 3.0]]]),
        )

    def test_lrp_conv3D_default_rule(self) -> None:
        conv = nn.Conv3d(1, 1, kernel_size=1, bias=False)
        conv.weight.data.fill_(2.0)
        self._assert_lrp_supports_module(
            conv,
            torch.arange(8, dtype=torch.float).view(1, 1, 2, 2, 2) + 1.0,
        )

    def test_lrp_avgpool1D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.AvgPool1d(kernel_size=2),
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        )

    def test_lrp_avgpool3D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.AvgPool3d(kernel_size=2),
            torch.arange(8, dtype=torch.float).view(1, 1, 2, 2, 2) + 1.0,
        )

    def test_lrp_adaptive_avgpool1D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.AdaptiveAvgPool1d(output_size=2),
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        )

    def test_lrp_adaptive_avgpool3D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            torch.arange(8, dtype=torch.float).view(1, 1, 2, 2, 2) + 1.0,
        )

    def test_lrp_batchnorm1D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.BatchNorm1d(num_features=1),
            torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]),
        )

    def test_lrp_batchnorm3D_default_rule(self) -> None:
        self._assert_lrp_supports_module(
            nn.BatchNorm3d(num_features=1),
            torch.arange(8, dtype=torch.float).view(1, 1, 2, 2, 2) + 1.0,
        )

    def test_lrp_multi(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.Tensor([[1, 2, 3]])
        add_input = 0
        output = model(input)
        output_add = model(input, add_input=add_input)
        self.assertTrue(torch.equal(output, output_add))
        lrp = LRP(model)
        attributions = lrp.attribute(input, target=0)  # type: ignore[has-type]
        attributions_add_input = lrp.attribute(  # type: ignore[has-type]
            input, target=0, additional_forward_args=(add_input,)
        )
        self.assertTrue(
            torch.equal(attributions, attributions_add_input)  # type: ignore
        )  # type: ignore

    def test_lrp_multi_inputs(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.Tensor([[1, 2, 3]])
        input = (input, 3 * input)
        lrp = LRP(model)
        attributions, delta = lrp.attribute(  # type: ignore[has-type]
            input, target=0, return_convergence_delta=True
        )
        self.assertEqual(len(input), 2)
        assertTensorAlmostEqual(self, attributions[0], torch.Tensor([[16, 32, 48]]))
        assertTensorAlmostEqual(self, delta, torch.Tensor([-104.0]))

    def test_lrp_ixg_equivalency(self) -> None:
        model, inputs = _get_simple_model()
        lrp = LRP(model)
        attributions_lrp = lrp.attribute(inputs)  # type: ignore[has-type]
        ixg = InputXGradient(model)
        attributions_ixg = ixg.attribute(inputs)
        assertTensorAlmostEqual(
            self, attributions_lrp, attributions_ixg
        )  # Divide by score because LRP relevance is normalized.

    def test_lrp_repeated_module(self) -> None:
        model = BasicModelWithReusedLinear()
        inp = torch.ones(2, 3)
        lrp = LRP(model)
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            lrp.attribute(inp, target=0)  # type: ignore[has-type]

    def test_futures_not_implemented(self) -> None:
        model = BasicModelWithReusedLinear()
        lrp = LRP(model)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = lrp.attribute_future()  # type: ignore
        self.assertEqual(attributions, None)
