#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, cast, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    _run_forward,
    ExpansionTypes,
    safe_div,
)
from captum._utils.typing import (
    BaselineTupleType,
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.log import log_usage
from captum.metrics._utils.batching import _divide_and_aggregate_metrics
from torch import Tensor


def infidelity_perturb_func_decorator(
    multiply_by_inputs: bool = True,
    # pyre-ignore[34]: The type variable `Variable[TensorOrTupleOfTensorsGeneric
    # <: [torch._tensor.Tensor, typing.Tuple[torch._tensor.Tensor, ...]]]` isn't
    # present in the function's parameters.
) -> Callable[
    [Callable[..., TensorOrTupleOfTensorsGeneric]],
    Callable[
        [TensorOrTupleOfTensorsGeneric, BaselineType],
        Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    ],
]:
    r"""An auxiliary, decorator function that helps with computing
    perturbations given perturbed inputs. It can be useful for cases
    when `perturb_func` returns only perturbed inputs and we
    internally compute the perturbations as
    (input - perturbed_input) / (input - baseline) if
    multiply_by_inputs is set to True and
    (input - perturbed_input) otherwise.

    If users decorate their `perturb_func` with
    `@infidelity_perturb_func_decorator` function then their `perturb_func`
    needs to only return perturbed inputs.

    Args:

        multiply_by_inputs (bool): Indicates whether model inputs'
                multiplier is factored in the computation of
                attribution scores.

    """

    def sub_infidelity_perturb_func_decorator(
        perturb_func: Callable[..., TensorOrTupleOfTensorsGeneric],
    ) -> Callable[
        [TensorOrTupleOfTensorsGeneric, BaselineType],
        Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    ]:
        r"""
        Args:

            perturb_func(Callable): Input perturbation function that takes inputs
                and optionally baselines and returns perturbed inputs

        Returns:

            default_perturb_func(Callable): Internal default perturbation
            function that computes the perturbations internally and returns
            perturbations and perturbed inputs.

        Examples::
            >>> @infidelity_perturb_func_decorator(True)
            >>> def perturb_fn(inputs):
            >>>    noise = torch.tensor(np.random.normal(0, 0.003,
            >>>                         inputs.shape)).float()
            >>>    return inputs - noise
            >>> # Computes infidelity score using `perturb_fn`
            >>> infidelity = infidelity(model, perturb_fn, input, ...)

        """

        def default_perturb_func(
            inputs: TensorOrTupleOfTensorsGeneric, baselines: BaselineType = None
        ) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]:
            r""" """
            inputs_perturbed: TensorOrTupleOfTensorsGeneric = (
                perturb_func(inputs, baselines)
                if baselines is not None
                else perturb_func(inputs)
            )
            inputs_perturbed_formatted = _format_tensor_into_tuples(inputs_perturbed)
            inputs_formatted = _format_tensor_into_tuples(inputs)
            baselines = _format_baseline(baselines, inputs_formatted)
            if baselines is None:
                perturbations = tuple(
                    (
                        safe_div(
                            input - input_perturbed,
                            input,
                            default_denom=1.0,
                        )
                        if multiply_by_inputs
                        else input - input_perturbed
                    )
                    for input, input_perturbed in zip(
                        inputs_formatted, inputs_perturbed_formatted
                    )
                )
            else:
                perturbations = tuple(
                    (
                        safe_div(
                            input - input_perturbed,
                            input - baseline,
                            default_denom=1.0,
                        )
                        if multiply_by_inputs
                        else input - input_perturbed
                    )
                    for input, input_perturbed, baseline in zip(
                        inputs_formatted,
                        inputs_perturbed_formatted,
                        baselines,
                    )
                )
            return perturbations, inputs_perturbed_formatted

        return default_perturb_func

    return sub_infidelity_perturb_func_decorator


@log_usage(part_of_slo=True)
def infidelity(
    forward_func: Callable[..., Tensor],
    perturb_func: Callable[
        ..., Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
    ],
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Optional[object] = None,
    target: TargetType = None,
    n_perturb_samples: int = 10,
    max_examples_per_batch: Optional[int] = None,
    normalize: bool = False,
) -> Tensor:
    r"""
    Explanation infidelity represents the expected mean-squared error
    between the explanation multiplied by a meaningful input perturbation
    and the differences between the predictor function at its input
    and perturbed input.
    More details about the measure can be found in the following paper:
    https://arxiv.org/abs/1901.09392

    It is derived from the completeness property of well-known attribution
    algorithms and is a computationally more efficient and generalized
    notion of Sensitivy-n. The latter measures correlations between the sum
    of the attributions and the differences of the predictor function at
    its input and fixed baseline. More details about the Sensitivity-n can
    be found here:
    https://arxiv.org/abs/1711.06104

    The users can perturb the inputs any desired way by providing any
    perturbation function that takes the inputs (and optionally baselines)
    and returns perturbed inputs or perturbed inputs and corresponding
    perturbations.

    This specific implementation is primarily tested for attribution-based
    explanation methods but the idea can be expanded to use for non
    attribution-based interpretability methods as well.

    Args:

        forward_func (Callable):
                The forward function of the model or any modification of it.

        perturb_func (Callable):
                The perturbation function of model inputs. This function takes
                model inputs and optionally baselines as input arguments and returns
                either a tuple of perturbations and perturbed inputs or just
                perturbed inputs. For example:

                >>> def my_perturb_func(inputs):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbations, perturbed_inputs

                If we want to only return perturbed inputs and compute
                perturbations internally then we can wrap perturb_func with
                `infidelity_perturb_func_decorator` decorator such as:

                >>> from captum.metrics import infidelity_perturb_func_decorator

                >>> @infidelity_perturb_func_decorator(<multiply_by_inputs flag>)
                >>> def my_perturb_func(inputs):
                >>>   <MY-LOGIC-HERE>
                >>>   return perturbed_inputs

                In case `multiply_by_inputs` is False we compute perturbations by
                `input - perturbed_input` difference and in case `multiply_by_inputs`
                flag is True we compute it by dividing
                (input - perturbed_input) by (input - baselines).
                The user needs to only return perturbed inputs in `perturb_func`
                as described above.

                `infidelity_perturb_func_decorator` needs to be used with
                `multiply_by_inputs` flag set to False in case infidelity
                score is being computed for attribution maps that are local aka
                that do not factor in inputs in the final attribution score.
                Such attribution algorithms include Saliency, GradCam, Guided Backprop,
                or Integrated Gradients and DeepLift attribution scores that are already
                computed with `multiply_by_inputs=False` flag.

                If there are more than one inputs passed to infidelity function those
                will be passed to `perturb_func` as tuples in the same order as they
                are passed to infidelity function.

                If inputs
                 - is a single tensor, the function needs to return a tuple
                   of perturbations and perturbed input such as:
                   perturb, perturbed_input and only perturbed_input in case
                   `infidelity_perturb_func_decorator` is used.
                 - is a tuple of tensors, corresponding perturbations and perturbed
                   inputs must be computed and returned as tuples in the
                   following format:

                   (perturb1, perturb2, ... perturbN), (perturbed_input1,
                   perturbed_input2, ... perturbed_inputN)

                   Similar to previous case here as well we need to return only
                   perturbed inputs in case `infidelity_perturb_func_decorator`
                   decorates out `perturb_func`.

                It is important to note that for performance reasons `perturb_func`
                isn't called for each example individually but on a batch of
                input examples that are repeated `max_examples_per_batch / batch_size`
                times within the batch.

        inputs (Tensor or tuple[Tensor, ...]): Input for which
                attributions are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples (aka batch size), and if
                multiple input tensors are provided, the examples must
                be aligned appropriately.

        baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                Baselines define reference values which sometimes represent ablated
                values and are used to compare with the actual inputs to compute
                importance scores in attribution algorithms. They can be represented
                as:

                - a single tensor, if inputs is a single tensor, with
                  exactly the same dimensions as inputs or the first
                  dimension is one and the remaining dimensions match
                  with inputs.

                - a single scalar, if inputs is a single tensor, which will
                  be broadcasted for each input value in input tensor.

                - a tuple of tensors or scalars, the baseline corresponding
                  to each tensor in the inputs' tuple can be:

                - either a tensor with matching dimensions to
                  corresponding tensor in the inputs' tuple
                  or the first dimension is one and the remaining
                  dimensions match with the corresponding
                  input tensor.

                - or a scalar, corresponding to a tensor in the
                  inputs' tuple. This scalar value is broadcasted
                  for corresponding input tensor.

                Default: None

        attributions (Tensor or tuple[Tensor, ...]):
                Attribution scores computed based on an attribution algorithm.
                This attribution scores can be computed using the implementations
                provided in the `captum.attr` package. Some of those attribution
                approaches are so called global methods, which means that
                they factor in model inputs' multiplier, as described in:
                https://arxiv.org/abs/1711.06104
                Many global attribution algorithms can be used in local modes,
                meaning that the inputs multiplier isn't factored in the
                attribution scores.
                This can be done duing the definition of the attribution algorithm
                by passing `multiply_by_inputs=False` flag.
                For example in case of Integrated Gradients (IG) we can obtain
                local attribution scores if we define the constructor of IG as:
                ig = IntegratedGradients(multiply_by_inputs=False)

                Some attribution algorithms are inherently local.
                Examples of inherently local attribution methods include:
                Saliency, Guided GradCam, Guided Backprop and Deconvolution.

                For local attributions we can use real-valued perturbations
                whereas for global attributions that perturbation is binary.
                https://arxiv.org/abs/1901.09392

                If we want to compute the infidelity of global attributions we
                can use a binary perturbation matrix that will allow us to select
                a subset of features from `inputs` or `inputs - baselines` space.
                This will allow us to approximate sensitivity-n for a global
                attribution algorithm.

                `infidelity_perturb_func_decorator` function decorator is a helper
                function that computes perturbations under the hood if perturbed
                inputs are provided.

                For more details about how to use `infidelity_perturb_func_decorator`,
                please, read the documentation about `perturb_func`

                Attributions have the same shape and dimensionality as the inputs.
                If inputs is a single tensor then the attributions is a single
                tensor as well. If inputs is provided as a tuple of tensors
                then attributions will be tuples of tensors as well.

        additional_forward_args (Any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a tuple
                containing multiple additional arguments including tensors
                or any arbitrary python types. These arguments are provided to
                forward_func in order, following the arguments in inputs.
                Note that the perturbations are not computed with respect
                to these arguments. This means that these arguments aren't
                being passed to `perturb_func` as an input argument.

                Default: None
        target (int, tuple, Tensor, or list, optional): Indices for selecting
                predictions from output(for classification cases,
                this is usually the target class).
                If the network returns a scalar value per example, no target
                index is necessary.
                For general 2D outputs, targets can be either:

                - A single integer or a tensor containing a single
                  integer, which is applied to all input examples

                - A list of integers or a 1D tensor, with length matching
                  the number of examples in inputs (dim 0). Each integer
                  is applied as the target for the corresponding example.

                  For outputs with > 2 dimensions, targets can be either:

                - A single tuple, which contains #output_dims - 1
                  elements. This target index is applied to all examples.

                - A list of tuples with length equal to the number of
                  examples in inputs (dim 0), and each tuple containing
                  #output_dims - 1 elements. Each tuple is applied as the
                  target for the corresponding example.

                Default: None
        n_perturb_samples (int, optional): The number of times input tensors
                are perturbed. Each input example in the inputs tensor is expanded
                `n_perturb_samples`
                times before calling `perturb_func` function.

                Default: 10
        max_examples_per_batch (int, optional): The number of maximum input
                examples that are processed together. In case the number of
                examples (`input batch size * n_perturb_samples`) exceeds
                `max_examples_per_batch`, they will be sliced
                into batches of `max_examples_per_batch` examples and processed
                in a sequential order. If `max_examples_per_batch` is None, all
                examples are processed together. `max_examples_per_batch` should
                at least be equal `input batch size` and at most
                `input batch size * n_perturb_samples`.

                Default: None
        normalize (bool, optional): Normalize the dot product of the input
                perturbation and the attribution so the infidelity value is invariant
                to constant scaling of the attribution values. The normalization factor
                beta is defined as the ratio of two mean values:

                .. math::
                    \beta = \frac{
                        \mathbb{E}_{I \sim \mu_I} [ I^T \Phi(f, x) (f(x) - f(x - I)) ]
                    }{
                        \mathbb{E}_{I \sim \mu_I} [ (I^T \Phi(f, x))^2 ]
                    }

                Please refer the original paper for the meaning of the symbols. Same
                normalization can be found in the paper's official implementation
                https://github.com/chihkuanyeh/saliency_evaluation

                Default: False
    Returns:

        infidelities (Tensor): A tensor of scalar infidelity scores per
                input example. The first dimension is equal to the
                number of examples in the input batch and the second
                dimension is one.

    Examples::
        >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
        >>> # and returns an Nx10 tensor of class probabilities.
        >>> net = ImageClassifier()
        >>> saliency = Saliency(net)
        >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
        >>> # Computes saliency maps for class 3.
        >>> attribution = saliency.attribute(input, target=3)
        >>> # define a perturbation function for the input
        >>> def perturb_fn(inputs):
        >>>    noise = torch.tensor(np.random.normal(0, 0.003, inputs.shape)).float()
        >>>    return noise, inputs - noise
        >>> # Computes infidelity score for saliency maps
        >>> infid = infidelity(net, perturb_fn, input, attribution)
    """
    # perform argument formattings
    inputs_formatted = _format_tensor_into_tuples(inputs)
    baselines_formatted: BaselineTupleType = None
    if baselines is not None:
        baselines_formatted = _format_baseline(baselines, inputs_formatted)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    attributions_formatted = _format_tensor_into_tuples(attributions)

    # Make sure that inputs and corresponding attributions have matching sizes.
    assert len(inputs_formatted) == len(attributions_formatted), (
        "The number of tensors in the inputs and attributions must match. "
        f"Found number of tensors in the inputs is: {len(inputs_formatted)} and in "
        f"the attributions: {len(attributions_formatted)}"
    )
    for inp, attr in zip(inputs_formatted, attributions_formatted):
        assert inp.shape == attr.shape, (
            "Inputs and attributions must have matching shapes. "
            f"One of the input tensor's shape is {inp.shape} and the "
            f"attribution tensor's shape is: {attr.shape}"
        )

    bsz = inputs_formatted[0].size(0)

    _next_infidelity_tensors = _make_next_infidelity_tensors_func(
        forward_func,
        bsz,
        perturb_func,
        inputs_formatted,
        baselines_formatted,
        attributions_formatted,
        additional_forward_args,
        target,
        normalize,
    )

    with torch.no_grad():
        # if not normalize, directly return aggrgated MSE ((a-b)^2,)
        # else return aggregated MSE's polynomial expansion tensors (a^2, ab, b^2)
        agg_tensors = _divide_and_aggregate_metrics(
            inputs_formatted,
            n_perturb_samples,
            _next_infidelity_tensors,
            agg_func=_sum_infidelity_tensors,
            max_examples_per_batch=max_examples_per_batch,
        )

    if normalize:
        beta_num = agg_tensors[1]
        beta_denorm = agg_tensors[0]

        beta = safe_div(beta_num, beta_denorm)

        infidelity_values = (
            beta * beta * agg_tensors[0] - 2 * beta * agg_tensors[1] + agg_tensors[2]
        )
    else:
        infidelity_values = agg_tensors[0]

    infidelity_values /= n_perturb_samples

    return infidelity_values


def _generate_perturbations(
    current_n_perturb_samples: int,
    perturb_func: Callable[
        ..., Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
    ],
    inputs: Tuple[Tensor, ...],
    baselines: BaselineTupleType,
) -> Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]:
    r"""
    The perturbations are generated for each example
    `current_n_perturb_samples` times.

    For performance reasons we are not calling `perturb_func` on each example but
    on a batch that contains `current_n_perturb_samples`
    repeated instances per example.
    """

    def call_perturb_func() -> (
        Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
    ):
        r""" """
        baselines_pert: BaselineType = None
        inputs_pert: Union[Tensor, Tuple[Tensor, ...]]
        if len(inputs_expanded) == 1:
            inputs_pert = inputs_expanded[0]
            if baselines_expanded is not None:
                baselines_pert = baselines_expanded[0]
        else:
            inputs_pert = inputs_expanded
            baselines_pert = baselines_expanded
        return (
            perturb_func(inputs_pert, baselines_pert)
            if baselines_pert is not None
            else perturb_func(inputs_pert)
        )

    inputs_expanded: Tuple[Tensor, ...] = tuple(
        torch.repeat_interleave(input, current_n_perturb_samples, dim=0)
        for input in inputs
    )

    baselines_expanded: BaselineTupleType = baselines
    if baselines is not None:
        baselines_expanded = tuple(
            (
                baseline.repeat_interleave(current_n_perturb_samples, dim=0)
                if isinstance(baseline, torch.Tensor)
                and baseline.shape[0] == input.shape[0]
                and baseline.shape[0] > 1
                else baseline
            )
            for input, baseline in zip(inputs, baselines)
        )

    return call_perturb_func()


def _validate_inputs_and_perturbations(
    inputs: Tuple[Tensor, ...],
    inputs_perturbed: Tuple[Tensor, ...],
    perturbations: Tuple[Tensor, ...],
) -> None:
    # asserts the sizes of the perturbations and inputs
    assert len(perturbations) == len(inputs), (
        "The number of perturbed "
        "inputs and corresponding perturbations must have the same number of "
        f"elements. Found number of inputs is: {len(perturbations)} and "
        f"perturbations: {len(inputs)}"
    )

    # asserts the shapes of the perturbations and perturbed inputs
    for perturb, input_perturbed in zip(perturbations, inputs_perturbed):
        assert perturb[0].shape == input_perturbed[0].shape, (
            "Perturbed input "
            "and corresponding perturbation must have the same shape and "
            f"dimensionality. Found perturbation shape is: {perturb[0].shape} "
            f"and the input shape is: {input_perturbed[0].shape}"
        )


def _make_next_infidelity_tensors_func(
    forward_func: Callable[..., Tensor],
    bsz: int,
    perturb_func: Callable[
        ..., Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
    ],
    inputs: Tuple[Tensor, ...],
    baselines: BaselineTupleType,
    attributions: Tuple[Tensor, ...],
    additional_forward_args: Optional[Tuple[object, ...]] = None,
    target: TargetType = None,
    normalize: bool = False,
) -> Callable[[int], Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]]:

    def _next_infidelity_tensors(
        current_n_perturb_samples: int,
    ) -> Union[Tuple[Tensor], Tuple[Tensor, Tensor, Tensor]]:
        perturbations, inputs_perturbed = _generate_perturbations(
            current_n_perturb_samples, perturb_func, inputs, baselines
        )

        perturbations_formatted = _format_tensor_into_tuples(perturbations)
        inputs_perturbed_formatted = _format_tensor_into_tuples(inputs_perturbed)

        _validate_inputs_and_perturbations(
            inputs,
            inputs_perturbed_formatted,
            perturbations_formatted,
        )

        targets_expanded = _expand_target(
            target,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )
        additional_forward_args_expanded = _expand_additional_forward_args(
            additional_forward_args,
            current_n_perturb_samples,
            expansion_type=ExpansionTypes.repeat_interleave,
        )

        inputs_perturbed_fwd = _run_forward(
            forward_func,
            inputs_perturbed_formatted,
            targets_expanded,
            additional_forward_args_expanded,
        )
        if isinstance(inputs_perturbed_fwd, torch.futures.Future):
            raise NotImplementedError(
                f"Outputs from forward_func of type {type(inputs_perturbed_fwd)} are "
                "not yet supported."
            )
        inputs_fwd = _run_forward(forward_func, inputs, target, additional_forward_args)
        # _run_forward may return future of Tensor,
        # but we don't support it here now
        # And it will fail before here.
        inputs_fwd = cast(Tensor, inputs_fwd)
        inputs_fwd = torch.repeat_interleave(
            inputs_fwd, current_n_perturb_samples, dim=0
        )
        perturbed_fwd_diffs = inputs_fwd - inputs_perturbed_fwd
        attributions_expanded = tuple(
            torch.repeat_interleave(attribution, current_n_perturb_samples, dim=0)
            for attribution in attributions
        )

        attributions_times_perturb = tuple(
            (attribution_expanded * perturbation).view(attribution_expanded.size(0), -1)
            for attribution_expanded, perturbation in zip(
                attributions_expanded, perturbations_formatted
            )
        )

        attr_times_perturb_sums = sum(
            torch.sum(attribution_times_perturb, dim=1)
            for attribution_times_perturb in attributions_times_perturb
        )
        attr_times_perturb_sums = cast(Tensor, attr_times_perturb_sums)

        # reshape as Tensor(bsz, current_n_perturb_samples)
        attr_times_perturb_sums = attr_times_perturb_sums.view(bsz, -1)
        perturbed_fwd_diffs = perturbed_fwd_diffs.view(bsz, -1)

        if normalize:
            # in order to normalize, we have to aggregate the following tensors
            # to calculate MSE in its polynomial expansion:
            # (a-b)^2 = a^2 - 2ab + b^2
            return (
                attr_times_perturb_sums.pow(2).sum(-1),
                (attr_times_perturb_sums * perturbed_fwd_diffs).sum(-1),
                perturbed_fwd_diffs.pow(2).sum(-1),
            )
        else:
            # returns (a-b)^2 if no need to normalize
            return ((attr_times_perturb_sums - perturbed_fwd_diffs).pow(2).sum(-1),)

    return _next_infidelity_tensors


def _sum_infidelity_tensors(
    agg_tensors: Tuple[Tensor, ...], tensors: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    return tuple(agg_t + t for agg_t, t in zip(agg_tensors, tensors))
