#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
from inspect import signature
from typing import Callable, List, Literal, Optional, Tuple, TYPE_CHECKING, Union

import torch
from captum._utils.common import (
    _format_baseline,
    _format_output,
    _format_tensor_into_tuples,
    _validate_input as _validate_input_basic,
)
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.approximation_methods import SUPPORTED_METHODS
from torch import Tensor

if TYPE_CHECKING:
    from captum.attr._utils.attribution import GradientAttribution


def _sum_rows(input: Tensor) -> Tensor:
    return input.reshape(input.shape[0], -1).sum(1)


def _validate_target(num_samples: int, target: TargetType) -> None:
    if isinstance(target, list) or (
        isinstance(target, torch.Tensor) and torch.numel(target) > 1
    ):
        assert num_samples == len(target), (
            "The number of samples provied in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )


def _validate_input(
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Union[Tensor, int, float], ...],
    n_steps: int = 50,
    method: str = "riemann_trapezoid",
    draw_baseline_from_distrib: bool = False,
) -> None:
    _validate_input_basic(inputs, baselines, draw_baseline_from_distrib)
    assert (
        n_steps >= 0
    ), "The number of steps must be a positive integer. " "Given: {}".format(n_steps)

    assert (
        method in SUPPORTED_METHODS
    ), "Approximation method must be one for the following {}. " "Given {}".format(
        SUPPORTED_METHODS, method
    )


def _validate_noise_tunnel_type(
    nt_type: str, supported_noise_tunnel_types: List[str]
) -> None:
    assert nt_type in supported_noise_tunnel_types, (
        "Noise types must be either `smoothgrad`, `smoothgrad_sq` or `vargrad`. "
        "Given {}".format(nt_type)
    )


@typing.overload
def _format_input_baseline(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    baselines: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


@typing.overload
def _format_input_baseline(  # type: ignore
    inputs: Union[Tensor, Tuple[Tensor, ...]], baselines: BaselineType
) -> Tuple[Tuple[Tensor, ...], Tuple[Union[Tensor, int, float], ...]]: ...


@typing.overload
def _format_input_baseline(  # type: ignore
    inputs: TensorOrTupleOfTensorsGeneric, baselines: BaselineType
) -> Tuple[Tuple[Tensor, ...], Tuple[Union[Tensor, int, float], ...]]: ...


def _format_input_baseline(
    inputs: Union[Tensor, Tuple[Tensor, ...]], baselines: BaselineType
) -> Tuple[Tuple[Tensor, ...], Tuple[Union[Tensor, int, float], ...]]:
    inputs = _format_tensor_into_tuples(inputs)
    baselines = _format_baseline(baselines, inputs)
    return inputs, baselines


# This function can potentially be merged with the `format_baseline` function
# however, since currently not all algorithms support baselines of type
# callable this will be kept in a separate function.
@typing.overload
def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        Tuple[Tensor, ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Tensor, ...]: ...


@typing.overload
def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        int,
        float,
        Tuple[Union[Tensor, int, float], ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Union[Tensor, int, float], ...]: ...


def _format_callable_baseline(
    baselines: Union[
        None,
        Callable[..., Union[Tensor, Tuple[Tensor, ...]]],
        Tensor,
        int,
        float,
        Tuple[Union[Tensor, int, float], ...],
    ],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Union[Tensor, int, float], ...]:
    if callable(baselines):
        # Note: this assumes that if baselines is a function and if it takes
        # arguments, then the first argument is the `inputs`.
        # This can be expanded in the future with better type checks
        baseline_parameters = signature(baselines).parameters
        if len(baseline_parameters) == 0:
            baselines = baselines()
        else:
            baselines = baselines(inputs)
    return _format_baseline(baselines, _format_tensor_into_tuples(inputs))


def _format_and_verify_strides(
    strides: Union[None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]],
    inputs: Tuple[Tensor, ...],
) -> Tuple[Union[int, Tuple[int, ...]], ...]:
    # Formats strides, which are necessary for occlusion
    # Assumes inputs are already formatted (in tuple)
    if strides is None:
        strides = tuple(1 for input in inputs)
    if len(inputs) == 1 and not (isinstance(strides, tuple) and len(strides) == 1):
        strides = (strides,)  # type: ignore
    assert isinstance(strides, tuple) and len(strides) == len(
        inputs
    ), "Strides must be provided for each input tensor."
    for i in range(len(inputs)):
        assert isinstance(strides[i], int) or (
            isinstance(strides[i], tuple)
            and len(strides[i]) == len(inputs[i].shape) - 1  # type: ignore
        ), (
            "Stride for input index {} is {}, which is invalid for input with "
            "shape {}. It must be either an int or a tuple with length equal to "
            "len(input_shape) - 1."
        ).format(
            i, strides[i], inputs[i].shape
        )

    # pyre-fixme[7]: Expected `Tuple[Union[int, typing.Tuple[int, ...]], ...]` but
    #  got `Union[Tuple[Union[int, typing.Tuple[Union[int, typing.Tuple[int, ...]],
    #  ...]]], typing.Tuple[Union[int, typing.Tuple[int, ...]], ...]]`.
    return strides


def _format_and_verify_sliding_window_shapes(
    sliding_window_shapes: Union[Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[int, ...], ...]:
    # Formats shapes of sliding windows, which is necessary for occlusion
    # Assumes inputs is already formatted (in tuple)
    if isinstance(sliding_window_shapes[0], int):
        sliding_window_shapes = (sliding_window_shapes,)  # type: ignore
    # pyre-fixme[35]: Target cannot be annotated.
    sliding_window_shapes: Tuple[Tuple[int, ...], ...]
    assert len(sliding_window_shapes) == len(
        inputs
    ), "Must provide sliding window dimensions for each input tensor."
    for i in range(len(inputs)):
        assert (
            isinstance(sliding_window_shapes[i], tuple)
            and len(sliding_window_shapes[i]) == len(inputs[i].shape) - 1
        ), (
            "Occlusion shape for input index {} is {} but should be a tuple with "
            "{} dimensions."
        ).format(
            i, sliding_window_shapes[i], len(inputs[i].shape) - 1
        )
    return sliding_window_shapes


@typing.overload
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Optional[object],
    target: TargetType,
    is_inputs_tuple: Literal[True],
) -> Union[Tuple[Tensor, ...], Tuple[Tuple[Tensor, ...], Tensor]]: ...


@typing.overload
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Optional[object],
    target: TargetType,
    is_inputs_tuple: Literal[False] = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]: ...


@typing.overload
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Optional[object],
    target: TargetType,
    is_inputs_tuple: bool = False,
) -> Union[
    Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
]: ...


# FIXME: GradientAttribution is provided as a string due to a circular import.
# This should be fixed when common is refactored into separate files.
def _compute_conv_delta_and_format_attrs(
    attr_algo: "GradientAttribution",
    return_convergence_delta: bool,
    attributions: Tuple[Tensor, ...],
    start_point: Union[int, float, Tensor, Tuple[Union[int, float, Tensor], ...]],
    end_point: Union[Tensor, Tuple[Tensor, ...]],
    additional_forward_args: Optional[object],
    target: TargetType,
    is_inputs_tuple: bool = False,
) -> Union[
    Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
]:
    if return_convergence_delta:
        # computes convergence error
        delta = attr_algo.compute_convergence_delta(
            attributions,
            start_point,
            end_point,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        return _format_output(is_inputs_tuple, attributions), delta
    else:
        return _format_output(is_inputs_tuple, attributions)


def _tensorize_baseline(
    inputs: Tuple[Tensor, ...], baselines: Tuple[Union[int, float, Tensor], ...]
) -> Tuple[Tensor, ...]:
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _tensorize_single_baseline(baseline, input):
        if isinstance(baseline, (int, float)):
            return torch.full_like(input, baseline)
        if input.shape[0] > baseline.shape[0] and baseline.shape[0] == 1:
            return torch.cat([baseline] * input.shape[0])
        return baseline

    assert isinstance(inputs, tuple) and isinstance(baselines, tuple), (
        "inputs and baselines must"
        "have tuple type but found baselines: {} and inputs: {}".format(
            type(baselines), type(inputs)
        )
    )
    return tuple(
        _tensorize_single_baseline(baseline, input)
        for baseline, input in zip(baselines, inputs)
    )


def _reshape_and_sum(
    tensor_input: Tensor, num_steps: int, num_examples: int, layer_size: Tuple[int, ...]
) -> Tensor:
    # Used for attribution methods which perform integration
    # Sums across integration steps by reshaping tensor to
    # (num_steps, num_examples, (layer_size)) and summing over
    # dimension 0. Returns a tensor of size (num_examples, (layer_size))
    return torch.sum(
        tensor_input.reshape((num_steps, num_examples) + layer_size), dim=0
    )


def _call_custom_attribution_func(
    custom_attribution_func: Callable[..., Tuple[Tensor, ...]],
    multipliers: Tuple[Tensor, ...],
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    assert callable(custom_attribution_func), (
        "`custom_attribution_func`"
        " must be a callable function but {} provided".format(
            type(custom_attribution_func)
        )
    )
    custom_attr_func_params = signature(custom_attribution_func).parameters

    if len(custom_attr_func_params) == 1:
        return custom_attribution_func(multipliers)
    elif len(custom_attr_func_params) == 2:
        return custom_attribution_func(multipliers, inputs)
    elif len(custom_attr_func_params) == 3:
        return custom_attribution_func(multipliers, inputs, baselines)
    else:
        raise AssertionError(
            "`custom_attribution_func` must take at least one and at most 3 arguments."
        )


def _find_output_mode_and_verify(
    initial_eval: Union[int, float, Tensor],
    num_examples: int,
    perturbations_per_eval: int,
    feature_mask: Union[None, TensorOrTupleOfTensorsGeneric],
    allow_multi_outputs: bool = False,
) -> bool:
    """
    This method identifies whether the model outputs a single output for a batch
    (agg_output_mode = True) or whether it outputs a single output per example
    (agg_output_mode = False) and returns agg_output_mode. The method also
    verifies that perturbations_per_eval is 1 in the case that agg_output_mode is True
    and also verifies that the first dimension of each feature mask if the model
    returns a single output for a batch.
    """
    if isinstance(initial_eval, (int, float)) or (
        isinstance(initial_eval, torch.Tensor)
        and (
            len(initial_eval.shape) == 0
            or (num_examples > 1 and initial_eval.numel() == 1)
        )
    ):
        agg_output_mode = True
        assert (
            perturbations_per_eval == 1
        ), "Cannot have perturbations_per_eval > 1 when function returns scalar."
        if feature_mask is not None:
            for single_mask in feature_mask:
                assert single_mask.shape[0] == 1, (
                    "Cannot provide different masks for each example when function "
                    "returns a scalar."
                )
    else:
        agg_output_mode = perturbations_per_eval == 1
        if not allow_multi_outputs:
            assert (
                isinstance(initial_eval, torch.Tensor) and initial_eval[0].numel() == 1
            ), "Target should identify a single element in the model output."
    return agg_output_mode


def _construct_default_feature_mask(
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], int]:
    feature_mask = []
    current_num_features = 0
    for i in range(len(inputs)):
        num_features = torch.numel(inputs[i][0])
        feature_mask.append(
            current_num_features
            + torch.reshape(
                torch.arange(num_features, device=inputs[i].device),
                inputs[i][0:1].shape,
            )
        )
        current_num_features += num_features
    total_features = current_num_features
    feature_mask = tuple(feature_mask)
    return feature_mask, total_features


def get_total_features_from_mask(
    feature_mask: Tuple[Tensor, ...],
) -> int:
    """
    Return the numbers of input features based on the total unique
    feature IDs/indices in the feature mask.
    """
    seen_idxs = set()
    for mask in feature_mask:
        seen_idxs |= set(torch.unique(mask).tolist())
    return len(seen_idxs)
