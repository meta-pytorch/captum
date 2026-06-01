#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Callable, Tuple, Union

import torch
from captum.attr._core.neuron.neuron_gradient_shap import NeuronGradientShap
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel_MultiLayer
from captum.testing.helpers.classification_models import SoftmaxModel
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_basic_multilayer(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, 20.0, 10.0]])
        baselines = torch.zeros(2, 3)
        ngs = NeuronGradientShap(model, model.linear1, multiply_by_inputs=False)
        attr = ngs.attribute(inputs, 0, baselines=baselines, stdevs=0.0)
        self.assertFalse(ngs.multiplies_by_inputs)
        assertTensorAlmostEqual(self, attr, [[1.0, 1.0, 1.0]])

    def test_basic_multilayer_wo_mult_by_inputs(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, 20.0, 10.0]])
        baselines = torch.randn(2, 3)

        self._assert_attributions(model, model.linear1, inputs, baselines, 0, 60)

    def test_basic_multilayer_wo_mult_by_inputs_agg_neurons(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        model.eval()

        inputs = torch.tensor([[1.0, 20.0, 10.0]])
        baselines = torch.randn(2, 3)

        self._assert_attributions(
            model, model.linear1, inputs, baselines, (slice(0, 1, 1),), 60
        )
        self._assert_attributions(
            model, model.linear1, inputs, baselines, lambda x: x[:, 0:1], 60
        )

    def test_classification(self) -> None:
        def custom_baseline_fn(inputs: Tensor) -> Tensor:
            num_in = inputs.shape[1]  # type: ignore
            return torch.arange(0.0, num_in * 5.0).reshape(5, num_in)

        num_in = 40
        n_samples = 100

        # 10-class classification model
        model = SoftmaxModel(num_in, 20, 10)
        model.eval()

        inputs = torch.arange(0.0, num_in * 2.0).reshape(2, num_in)
        baselines = custom_baseline_fn

        self._assert_attributions(model, model.relu1, inputs, baselines, 1, n_samples)

    def _assert_attributions(
        self,
        model: Module,
        layer: Module,
        inputs: Tensor,
        baselines: Union[Tensor, Callable[..., Tensor]],
        neuron_ind: Union[int, Tuple[Union[int, slice], ...], Callable],
        n_samples: int = 5,
    ) -> None:
        ngs = NeuronGradientShap(model, layer)
        nig = NeuronIntegratedGradients(model, layer)
        attrs_gs = ngs.attribute(
            inputs, neuron_ind, baselines=baselines, n_samples=n_samples, stdevs=0.09
        )

        if callable(baselines):
            baselines = baselines(inputs)

        attrs_ig = []
        for baseline in torch.unbind(baselines):
            attrs_ig.append(
                nig.attribute(inputs, neuron_ind, baselines=baseline.unsqueeze(0))
            )
        combined_attrs_ig = torch.stack(attrs_ig, dim=0).mean(dim=0)
        self.assertTrue(ngs.multiplies_by_inputs)
        assertTensorAlmostEqual(self, attrs_gs, combined_attrs_ig, 0.5)
