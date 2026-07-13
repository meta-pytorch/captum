#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Tuple

import torch
from captum.attr import (
    AblationGuidedShapleyValueSampling,
    FeatureAblation,
    ShapleyValueSampling,
)
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from captum.testing.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from parameterized import parameterized


class Test(BaseTest):
    @parameterized.expand([(1,), (2,), (3,)])
    def test_additive_model_matches_feature_ablation_and_shapley(
        self, perturbations_per_eval: int
    ) -> None:
        def forward_func(inp: torch.Tensor) -> torch.Tensor:
            return 2.0 * inp[:, 0] - 3.0 * inp[:, 1] + 0.5 * inp[:, 2]

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, -1.0, 2.0]])
        ags = AblationGuidedShapleyValueSampling(forward_func)
        ags_attr = ags.attribute(
            inp,
            n_samples=7,
            perturbations_per_eval=perturbations_per_eval,
        )
        loo_attr = FeatureAblation(forward_func).attribute(
            inp,
            perturbations_per_eval=perturbations_per_eval,
        )
        shapley_attr = ShapleyValueSampling(forward_func).attribute(
            inp,
            n_samples=7,
            perturbations_per_eval=perturbations_per_eval,
        )

        assertTensorAlmostEqual(self, ags_attr, loo_attr, mode="max")
        assertTensorAlmostEqual(self, ags_attr, shapley_attr, mode="max")

    def test_multi_tensor_additive_model_matches_feature_ablation(self) -> None:
        def forward_func(
            inp1: torch.Tensor, inp2: torch.Tensor, scale: float
        ) -> torch.Tensor:
            return scale * (inp1.sum(dim=1) + 2.0 * inp2.sum(dim=1))

        inp1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        inp2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        inputs = (inp1, inp2)
        additional_args = (2.0,)
        ags_attr = AblationGuidedShapleyValueSampling(forward_func).attribute(
            inputs,
            additional_forward_args=additional_args,
            n_samples=11,
            perturbations_per_eval=2,
        )
        loo_attr = FeatureAblation(forward_func).attribute(
            inputs,
            additional_forward_args=additional_args,
            perturbations_per_eval=2,
        )

        assert isinstance(ags_attr, tuple)
        assert isinstance(loo_attr, tuple)
        for actual, expected in zip(ags_attr, loo_attr):
            assertTensorAlmostEqual(self, actual, expected, mode="max")

    def test_feature_mask_grouping_matches_feature_ablation(self) -> None:
        def forward_func(inp: torch.Tensor) -> torch.Tensor:
            return inp.sum(dim=1)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        feature_mask = torch.tensor([[0, 0, 1]])
        ags_attr = AblationGuidedShapleyValueSampling(forward_func).attribute(
            inp,
            feature_mask=feature_mask,
            n_samples=5,
        )
        loo_attr = FeatureAblation(forward_func).attribute(
            inp,
            feature_mask=feature_mask,
        )

        assertTensorAlmostEqual(self, ags_attr, loo_attr, mode="max")

    def test_multi_output_without_target_matches_shapley_shape(self) -> None:
        def forward_func(inp: torch.Tensor) -> torch.Tensor:
            return torch.stack((inp[:, 0] + inp[:, 1], 2.0 * inp[:, 2]), dim=1)

        inp = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ags_attr = AblationGuidedShapleyValueSampling(forward_func).attribute(
            inp,
            target=None,
            n_samples=5,
        )
        shapley_attr = ShapleyValueSampling(forward_func).attribute(
            inp,
            target=None,
            n_samples=5,
        )

        self.assertEqual(ags_attr.shape, shapley_attr.shape)
        assertTensorAlmostEqual(self, ags_attr, shapley_attr, mode="max")

    def test_n_samples_respects_torch_seed_for_reproducibility(self) -> None:
        def forward_func(inp: torch.Tensor) -> torch.Tensor:
            return inp[:, 0] * inp[:, 1] + inp[:, 1] * inp[:, 2] + inp[:, 0] * inp[:, 2]

        inp = torch.tensor([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]])
        ags = AblationGuidedShapleyValueSampling(forward_func)
        torch.manual_seed(123)
        first_attr = ags.attribute(inp, n_samples=13)
        torch.manual_seed(123)
        second_attr = ags.attribute(inp, n_samples=13)

        assertTensorAlmostEqual(self, first_attr, second_attr, delta=0.0, mode="max")

    def test_interacting_features_are_sampled_and_additive_features_keep_loo(
        self,
    ) -> None:
        def forward_func(inp: torch.Tensor) -> torch.Tensor:
            return inp[:, 0] * inp[:, 1] + inp[:, 2]

        inp = torch.tensor([[2.0, 3.0, 5.0]])
        torch.manual_seed(0)
        ags_attr = AblationGuidedShapleyValueSampling(forward_func).attribute(
            inp,
            n_samples=200,
        )
        loo_attr = FeatureAblation(forward_func).attribute(inp)

        assertTensorAlmostEqual(self, ags_attr, [[3.0, 3.0, 5.0]], delta=0.1)
        self.assertGreater(abs(float(loo_attr[0, 0] - ags_attr[0, 0])), 2.0)
        self.assertGreater(abs(float(loo_attr[0, 1] - ags_attr[0, 1])), 2.0)
        self.assertEqual(float(loo_attr[0, 2]), float(ags_attr[0, 2]))

    def test_output_shape_matches_single_and_multi_tensor_inputs(self) -> None:
        single_input = torch.tensor([[20.0, 50.0, 30.0], [2.0, 10.0, 3.0]])
        single_attr = AblationGuidedShapleyValueSampling(
            BasicModel_MultiLayer()
        ).attribute(single_input, target=0, n_samples=5)

        self.assertEqual(single_attr.shape, single_input.shape)

        multi_inputs: Tuple[torch.Tensor, ...] = (
            torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]]),
            torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]]),
            torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]]),
        )
        multi_attr = AblationGuidedShapleyValueSampling(
            BasicModel_MultiLayer_MultiInput()
        ).attribute(
            multi_inputs,
            target=0,
            additional_forward_args=(1,),
            n_samples=5,
        )

        assert isinstance(multi_attr, tuple)
        for actual, expected in zip(multi_attr, multi_inputs):
            self.assertEqual(actual.shape, expected.shape)
