#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, cast, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _get_max_feature_index,
    _is_tuple,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.shapley_value import _shape_feature_mask, ShapleyValueSampling
from captum.attr._utils.common import _format_input_baseline, _tensorize_baseline
from captum.log import log_usage
from torch import Tensor
from torch.futures import Future


class AblationGuidedShapleyValueSampling(ShapleyValueSampling):
    """
    Ablation-guided Shapley Value sampling.

    This method first runs two inexpensive ablation screens for each feature
    group: leave-one-out attribution, ``a_i = f(all) - f(all \\ {i})``, and
    inclusion attribution, ``m_i = f({i}) - f(empty)``. A feature is considered
    interacting when ``|a_i - m_i| / (|a_i| + eps)`` is greater than
    ``interaction_rel_threshold``. Non-interacting features receive their exact
    leave-one-out attribution, while interacting features are passed to
    :class:`~captum.attr.ShapleyValueSampling`.

    Shapley sampling is restricted to the active interacting feature groups.
    All non-active groups are pinned present at their input values, so the
    sampled game is ``v(S union pinned)`` while preserving the standard Captum
    attribution output shape.
    """

    def __init__(
        self,
        forward_func: Callable[
            ..., Union[int, float, Tensor, Future[int], Future[float], Future[Tensor]]
        ],
        interaction_rel_threshold: float = 0.25,
        eps: float = 1e-8,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it. The forward function can either
                        return a scalar per example, or a single scalar for the
                        full batch. If a single scalar is returned for the batch,
                        `perturbations_per_eval` must be 1, and the returned
                        attributions will have first dimension 1, corresponding to
                        feature importance across all examples in the batch.
            interaction_rel_threshold (float, optional): Relative threshold used
                        to identify interacting feature groups. A feature group
                        is sampled with Shapley Value Sampling when
                        ``mean(abs(LOO - inclusion)) / (mean(abs(LOO)) + eps)``
                        exceeds this value.
                        Default: 0.25
            eps (float, optional): Small positive value used to stabilize the
                        relative interaction score denominator.
                        Default: 1e-8
        """
        ShapleyValueSampling.__init__(self, forward_func)
        self.interaction_rel_threshold = interaction_rel_threshold
        self.eps = eps

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        interaction_rel_threshold: Optional[float] = None,
        eps: Optional[float] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

                inputs (Tensor or tuple[Tensor, ...]): Input for which
                            ablation-guided Shapley value sampling attributions
                            are computed. If forward_func takes a single tensor
                            as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a
                            tuple of the input tensors should be provided. It is
                            assumed that for all given input tensors, dimension 0
                            corresponds to the number of examples (aka batch size),
                            and if multiple input tensors are provided, the examples
                            must be aligned appropriately.
                baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                            Baselines define reference value which replaces each
                            feature when ablated.
                            Baselines can be provided as:

                            - a single tensor, if inputs is a single tensor, with
                              exactly the same dimensions as inputs or the first
                              dimension is one and the remaining dimensions match
                              with inputs.

                            - a single scalar, if inputs is a single tensor, which
                              will be broadcasted for each input value in input
                              tensor.

                            - a tuple of tensors or scalars, the baseline
                              corresponding to each tensor in the inputs' tuple can
                              be:

                              - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding input
                                tensor.

                              - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted for
                                corresponding input tensor.

                            In the cases when `baselines` is not provided, we
                            internally use zero scalar corresponding to each input
                            tensor.
                            Default: None
                target (int, tuple, Tensor, or list, optional): Output indices for
                            which difference is computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example, no
                            target index is necessary.
                            Default: None
                additional_forward_args (Any, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this
                            argument can be provided. It must be either a single
                            additional argument of a Tensor or arbitrary
                            (non-tuple) type or a tuple containing multiple
                            additional arguments including tensors or any arbitrary
                            python types. These arguments are provided to
                            forward_func in order following the arguments in
                            inputs.
                            Default: None
                feature_mask (Tensor or tuple[Tensor, ...], optional):
                            feature_mask defines a mask for the input, grouping
                            features which should be perturbed together.
                            feature_mask should contain the same number of tensors
                            as inputs. Each tensor should be the same size as the
                            corresponding input or broadcastable to match the input
                            tensor. Values across all tensors should be integers in
                            the range 0 to num_features - 1, and indices
                            corresponding to the same feature should have the same
                            value.
                            Default: None
                n_samples (int, optional): The number of feature permutations
                            tested for the interacting feature groups.
                            Default: 25
                perturbations_per_eval (int, optional): Allows multiple
                            perturbations to be processed simultaneously in one
                            call to forward_fn. Each forward pass will contain a
                            maximum of perturbations_per_eval * #examples samples.
                            If the forward function returns a single scalar per
                            batch, perturbations_per_eval must be set to 1.
                            Default: 1
                show_progress (bool, optional): Displays the progress of
                            computation. It will try to use tqdm if available for
                            advanced features (e.g. time estimation). Otherwise,
                            it will fallback to a simple output of progress.
                            Default: False
                interaction_rel_threshold (float, optional): Overrides the
                            constructor threshold for this attribution call.
                            Default: None
                eps (float, optional): Overrides the constructor denominator
                            stabilizer for this attribution call.
                            Default: None

        Returns:
                *Tensor* or *tuple[Tensor, ...]* of **attributions**:
                - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                            The attributions with respect to each input feature.
                            If the forward function returns a scalar value per
                            example, attributions will be the same size as the
                            provided inputs, with each value providing the
                            attribution of the corresponding input index. If the
                            forward function returns a scalar per batch, then
                            attribution tensor(s) will have first dimension 1 and
                            the remaining dimensions will match the input. If a
                            single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining AblationGuidedShapleyValueSampling interpreter
            >>> ags = AblationGuidedShapleyValueSampling(net)
            >>> # Computes exact leave-one-out attributions for additive
            >>> # feature groups and samples only groups with detected
            >>> # interactions.
            >>> attr = ags.attribute(input, target=1, n_samples=200)

            >>> # Feature groups are supported in the same way as
            >>> # ShapleyValueSampling.
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = ags.attribute(input, target=1, feature_mask=feature_mask)
        """
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple, formatted_baselines = _format_input_baseline(inputs, baselines)
        formatted_additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        formatted_feature_mask = _format_feature_mask(feature_mask, inputs_tuple)
        reshaped_feature_mask = _shape_feature_mask(
            formatted_feature_mask, inputs_tuple
        )
        threshold = (
            self.interaction_rel_threshold
            if interaction_rel_threshold is None
            else interaction_rel_threshold
        )
        denominator_eps = self.eps if eps is None else eps
        assert threshold >= 0, "interaction_rel_threshold must be non-negative."
        assert denominator_eps > 0, "eps must be positive."
        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be at least 1."

        with torch.no_grad():
            tensorized_baselines = _tensorize_baseline(
                inputs_tuple, formatted_baselines
            )
            initial_eval = self._strict_run_forward(
                self.forward_func,
                inputs_tuple,
                target,
                formatted_additional_forward_args,
            )
            output_shape = tuple(initial_eval.shape)
            total_features = _get_max_feature_index(reshaped_feature_mask) + 1
            loo_attributions = self._attribute_with_feature_ablation(
                inputs_tuple,
                tensorized_baselines,
                target,
                formatted_additional_forward_args,
                reshaped_feature_mask,
                perturbations_per_eval,
                show_progress,
                output_shape,
            )
            inclusion_attributions = tuple(
                -single_attr
                for single_attr in self._attribute_with_feature_ablation(
                    tensorized_baselines,
                    inputs_tuple,
                    target,
                    formatted_additional_forward_args,
                    reshaped_feature_mask,
                    perturbations_per_eval,
                    show_progress,
                    output_shape,
                )
            )
            active_feature_indices = self._find_active_feature_indices(
                loo_attributions,
                inclusion_attributions,
                reshaped_feature_mask,
                total_features,
                threshold,
                denominator_eps,
            )

            if len(active_feature_indices) == 0:
                return cast(
                    TensorOrTupleOfTensorsGeneric,
                    _format_output(is_inputs_tuple, loo_attributions),
                )

            active_feature_mask = self._construct_active_feature_mask(
                reshaped_feature_mask,
                active_feature_indices,
            )
            pinned_baselines = self._construct_pinned_baselines(
                inputs_tuple,
                tensorized_baselines,
                reshaped_feature_mask,
                active_feature_indices,
            )
            active_attributions = cast(
                Tuple[Tensor, ...],
                super().attribute.__wrapped__(
                    self,
                    inputs=inputs_tuple,
                    baselines=pinned_baselines,
                    target=target,
                    additional_forward_args=formatted_additional_forward_args,
                    feature_mask=active_feature_mask,
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    show_progress=show_progress,
                ),
            )
            attributions = self._merge_active_attributions(
                loo_attributions,
                active_attributions,
                reshaped_feature_mask,
                active_feature_indices,
            )
            return cast(
                TensorOrTupleOfTensorsGeneric,
                _format_output(is_inputs_tuple, attributions),
            )

    def _attribute_with_feature_ablation(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Optional[Tuple[object, ...]],
        feature_mask: Tuple[Tensor, ...],
        perturbations_per_eval: int,
        show_progress: bool,
        output_shape: Tuple[int, ...],
    ) -> Tuple[Tensor, ...]:
        ablator = FeatureAblation(self.forward_func)
        attributions = cast(
            Tuple[Tensor, ...],
            ablator.attribute(
                inputs,
                baselines=baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                feature_mask=feature_mask,
                perturbations_per_eval=perturbations_per_eval,
                show_progress=show_progress,
            ),
        )
        return self._reshape_feature_ablation_attributions(
            attributions,
            inputs,
            output_shape,
        )

    def _reshape_feature_ablation_attributions(
        self,
        attributions: Tuple[Tensor, ...],
        inputs: Tuple[Tensor, ...],
        output_shape: Tuple[int, ...],
    ) -> Tuple[Tensor, ...]:
        reshaped_attributions: List[Tensor] = []
        for single_attribution, single_input in zip(attributions, inputs):
            expected_shape = output_shape + tuple(single_input.shape[1:])
            if tuple(single_attribution.shape) == expected_shape:
                reshaped_attributions.append(single_attribution)
            else:
                reshaped_attributions.append(single_attribution.reshape(expected_shape))
        return tuple(reshaped_attributions)

    def _find_active_feature_indices(
        self,
        loo_attributions: Tuple[Tensor, ...],
        inclusion_attributions: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        total_features: int,
        interaction_rel_threshold: float,
        eps: float,
    ) -> List[int]:
        active_feature_indices: List[int] = []
        for feature_index in range(total_features):
            loo_values = self._gather_feature_attribution_values(
                loo_attributions,
                feature_mask,
                feature_index,
            )
            if loo_values.numel() == 0:
                continue
            inclusion_values = self._gather_feature_attribution_values(
                inclusion_attributions,
                feature_mask,
                feature_index,
            )
            loo_values = loo_values.float()
            inclusion_values = inclusion_values.float()
            loo_magnitude = torch.mean(torch.abs(loo_values))
            interaction_magnitude = torch.mean(torch.abs(loo_values - inclusion_values))
            interaction_ratio = interaction_magnitude / (loo_magnitude + eps)
            if float(interaction_ratio.item()) > interaction_rel_threshold:
                active_feature_indices.append(feature_index)
        return active_feature_indices

    def _gather_feature_attribution_values(
        self,
        attributions: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        feature_index: int,
    ) -> Tensor:
        values: List[Tensor] = []
        device = attributions[0].device
        for single_attribution, single_mask in zip(attributions, feature_mask):
            selector = self._reshape_mask_for_attribution(
                single_mask == feature_index,
                single_attribution,
            )
            selector = selector.to(single_attribution.device).expand_as(
                single_attribution
            )
            if bool(torch.any(selector).item()):
                values.append(
                    single_attribution[selector].reshape(-1).float().to(device)
                )
        if len(values) == 0:
            return torch.empty(0, device=device)
        return torch.cat(values)

    def _reshape_mask_for_attribution(
        self, mask: Tensor, attribution: Tensor
    ) -> Tensor:
        assert (
            attribution.dim() >= mask.dim()
        ), "Attribution must have at least as many dimensions as the feature mask."
        return mask.reshape(
            tuple(mask.shape[:1])
            + (attribution.dim() - mask.dim()) * (1,)
            + tuple(mask.shape[1:])
        )

    def _construct_active_feature_mask(
        self,
        feature_mask: Tuple[Tensor, ...],
        active_feature_indices: List[int],
    ) -> Tuple[Tensor, ...]:
        active_index_to_position: Dict[int, int] = {
            feature_index: active_position
            for active_position, feature_index in enumerate(active_feature_indices)
        }
        active_feature_masks: List[Tensor] = []
        for single_mask in feature_mask:
            active_mask = torch.full_like(single_mask, -1)
            for feature_index, active_position in active_index_to_position.items():
                active_mask = torch.where(
                    single_mask == feature_index,
                    torch.full_like(single_mask, active_position),
                    active_mask,
                )
            active_feature_masks.append(active_mask)
        return tuple(active_feature_masks)

    def _construct_pinned_baselines(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        active_feature_indices: List[int],
    ) -> Tuple[Tensor, ...]:
        pinned_baselines: List[Tensor] = []
        for single_input, single_baseline, single_mask in zip(
            inputs, baselines, feature_mask
        ):
            active_mask = self._get_active_mask(single_mask, active_feature_indices)
            pinned_baselines.append(
                torch.where(active_mask, single_baseline, single_input)
            )
        return tuple(pinned_baselines)

    def _merge_active_attributions(
        self,
        loo_attributions: Tuple[Tensor, ...],
        active_attributions: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        active_feature_indices: List[int],
    ) -> Tuple[Tensor, ...]:
        merged_attributions: List[Tensor] = []
        for loo_attr, active_attr, single_mask in zip(
            loo_attributions, active_attributions, feature_mask
        ):
            active_mask = self._get_active_mask(single_mask, active_feature_indices)
            selector = self._reshape_mask_for_attribution(active_mask, loo_attr)
            selector = selector.to(loo_attr.device).expand_as(loo_attr)
            merged_attributions.append(
                torch.where(
                    selector,
                    active_attr.to(device=loo_attr.device, dtype=loo_attr.dtype),
                    loo_attr,
                )
            )
        return tuple(merged_attributions)

    def _get_active_mask(
        self,
        feature_mask: Tensor,
        active_feature_indices: List[int],
    ) -> Tensor:
        active_mask = torch.zeros_like(feature_mask, dtype=torch.bool)
        for feature_index in active_feature_indices:
            active_mask = active_mask | (feature_mask == feature_index)
        return active_mask
