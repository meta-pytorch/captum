#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, List, Tuple

import torch
from captum._utils.gradient import apply_gradient_requirements
from captum._utils.sample_gradient import (
    _reset_sample_grads,
    SampleGradientWrapper,
    SUPPORTED_MODULES,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_ConvNetWithPaddingDilation,
    BasicModel_MultiLayer,
)
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_sample_grads_linear_sum(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        inp = (torch.randn(6, 3), torch.randn(6, 3))
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def test_sample_grads_linear_mean(self) -> None:
        model = BasicModel_MultiLayer(multi_input_module=True)
        inp = (20 * torch.randn(6, 3),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.mean(x))

    def test_sample_grads_conv_sum(self) -> None:
        model = BasicModel_ConvNet_One_Conv()
        inp = (123 * torch.randn(6, 1, 4, 4),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def test_sample_grads_conv_mean_multi_inp(self) -> None:
        model = BasicModel_ConvNet_One_Conv()
        inp = (20 * torch.randn(6, 1, 4, 4), 9 * torch.randn(6, 1, 4, 4))
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.mean(x))

    def test_sample_grads_modified_conv_mean(self) -> None:
        model = BasicModel_ConvNetWithPaddingDilation()
        inp = (20 * torch.randn(6, 1, 5, 5),)
        self._compare_sample_grads_per_sample(
            model, inp, lambda x: torch.mean(x), "mean"
        )

    def test_sample_grads_modified_conv_sum(self) -> None:
        model = BasicModel_ConvNetWithPaddingDilation()
        inp = (20 * torch.randn(6, 1, 5, 5),)
        self._compare_sample_grads_per_sample(model, inp, lambda x: torch.sum(x), "sum")

    def _compare_sample_grads_per_sample(
        self,
        model: Module,
        inputs: Tuple[Tensor, ...],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Callable,
        loss_type: str = "mean",
    ) -> None:
        wrapper = SampleGradientWrapper(model)
        wrapper.add_hooks()
        apply_gradient_requirements(inputs)
        out = model(*inputs)
        wrapper.compute_param_sample_gradients(loss_fn(out), loss_type)

        batch_size = inputs[0].shape[0]
        for i in range(batch_size):
            model.zero_grad()
            single_inp = tuple(inp[i : i + 1] for inp in inputs)
            out = model(*single_inp)
            loss_fn(out).backward()
            for layer in model.modules():
                if isinstance(layer, tuple(SUPPORTED_MODULES.keys())):
                    assertTensorAlmostEqual(
                        self,
                        layer.weight.grad,
                        layer.weight.sample_grad[i],  # type: ignore
                        mode="max",
                    )
                    assertTensorAlmostEqual(
                        self,
                        layer.bias.grad,
                        layer.bias.sample_grad[i],  # type: ignore
                        mode="max",
                    )

    def test_sample_grads_layer_modules(self) -> None:
        """
        tests that if `layer_modules` argument is specified for `SampleGradientWrapper`
        that only per-sample gradients for the specified layers are calculated
        """
        model = BasicModel_ConvNet_One_Conv()
        inp = (20 * torch.randn(6, 1, 4, 4), 9 * torch.randn(6, 1, 4, 4))

        # possible candidates for `layer_modules`, which are the modules whose
        # parameters we want to compute sample grads for
        layer_moduless: List[List[Module]] = [
            [model.conv1],
            [model.fc1],
            [model.conv1, model.fc1],
        ]
        # hard coded all modules we want to check
        all_modules = [model.conv1, model.fc1]

        for layer_modules in layer_moduless:
            # we will call the wrapper multiple times, so should reset each time
            for module in all_modules:
                _reset_sample_grads(module)

            # compute sample grads
            wrapper = SampleGradientWrapper(model, layer_modules)
            wrapper.add_hooks()
            apply_gradient_requirements(inp)
            out = model(*inp)
            wrapper.compute_param_sample_gradients(torch.sum(out), "sum")

            for module in all_modules:
                if module in layer_modules:
                    # If we calculated the sample grads for the layer, none
                    # of its parameters' `sample_grad` attributes` would be an int,
                    # since even though they were all set to 0 in beginning of loop
                    # computing sample grads would override that 0.
                    # So, check that we did calculate sample grads for the desired
                    # layers via the above checking approach.
                    for parameter in module.parameters():
                        assert not isinstance(
                            parameter.sample_grad, int  # type: ignore
                        )
                else:
                    # For the layers we do not want sample grads for, their
                    # `sample_grad` should still be 0, since they should not have been
                    # over-written.
                    for parameter in module.parameters():
                        assert parameter.sample_grad == 0  # type: ignore
