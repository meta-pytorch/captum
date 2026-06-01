#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import torch
from captum._utils.common import _is_tuple
from captum._utils.typing import (
    BaselineType,
    TargetType,
    Tensor,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import (
    _compute_conv_delta_and_format_attrs,
    _format_callable_baseline,
    _format_input_baseline,
)
from captum.log import log_usage


class GradientShap(GradientAttribution):
    r"""
    Implements gradient SHAP based on the implementation from SHAP's primary
    author. For reference, please view the original
    `implementation
    <https://github.com/slundberg/shap#deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models>`_
    and the paper: `A Unified Approach to Interpreting Model Predictions
    <https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions>`_

    GradientShap approximates SHAP values by computing the expectations of
    gradients by randomly sampling from the distribution of baselines/references.
    It adds white noise to each input sample `n_samples` times, selects a
    random baseline from baselines' distribution and a random point along the
    path between the baseline and the input, and computes the gradient of outputs
    with respect to those selected random points. The final SHAP values represent
    the expected values of gradients * (inputs - baselines).

    GradientShap makes an assumption that the input features are independent
    and that the explanation model is linear, meaning that the explanations
    are modeled through the additive composition of feature effects.
    Under those assumptions, SHAP value can be approximated as the expectation
    of gradients that are computed for randomly generated `n_samples` input
    samples after adding gaussian noise `n_samples` times to each input for
    different baselines/references.

    In some sense it can be viewed as an approximation of integrated gradients
    by computing the expectations of gradients for different baselines.

    Current implementation uses Smoothgrad from :class:`.NoiseTunnel` in order to
    randomly draw samples from the distribution of baselines, add noise to input
    samples and compute the expectation (smoothgrad).
    """

    def __init__(
        self, forward_func: Callable[..., Tensor], multiply_by_inputs: bool = True
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                       any modification of it.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                    model inputs' multiplier in the final attribution scores.
                    In the literature this is also known as local vs global
                    attribution. If inputs' multiplier isn't factored in
                    then this type of attribution method is also called local
                    attribution. If it is, then that type of attribution
                    method is called global.
                    More detailed can be found here:
                    https://arxiv.org/abs/1711.06104

                    In case of gradient shap, if `multiply_by_inputs`
                    is set to True, the sensitivity scores of scaled inputs
                    are being multiplied by (inputs - baselines).
        """
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        n_samples: int = 5,
        stdevs: Union[float, Tuple[float, ...]] = 0.0,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        n_samples: int = 5,
        stdevs: Union[float, Tuple[float, ...]] = 0.0,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @log_usage(part_of_slo=True)
    # pyre-fixme[43]: This definition does not have the same decorators as the
    #  preceding overload(s).
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        n_samples: int = 5,
        stdevs: Union[float, Tuple[float, ...]] = 0.0,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which SHAP attribution
                        values are computed. If `forward_func` takes a single
                        tensor as input, a single input tensor should be provided.
                        If `forward_func` takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (Tensor, tuple[Tensor, ...], or Callable):
                        Baselines define the starting point from which expectation
                        is computed and can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          the first dimension equal to the number of examples
                          in the baselines' distribution. The remaining dimensions
                          must match with input tensor's dimension starting from
                          the second dimension.

                        - a tuple of tensors, if inputs is a tuple of tensors,
                          with the first dimension of any tensor inside the tuple
                          equal to the number of examples in the baseline's
                          distribution. The remaining dimensions must match
                          the dimensions of the corresponding input tensor
                          starting from the second dimension.

                        - callable function, optionally takes `inputs` as an
                          argument and either returns a single tensor
                          or a tuple of those.

                        It is recommended that the number of samples in the baselines'
                        tensors is larger than one.
            n_samples (int, optional): The number of randomly generated examples
                        per sample in the input batch. Random examples are
                        generated by adding gaussian random noise to each sample.
                        Default: `5` if `n_samples` is not provided.
            stdevs    (float or tuple of float, optional): The standard deviation
                        of gaussian noise with zero mean that is added to each
                        input in the batch. If `stdevs` is a single float value
                        then that same value is used for all inputs. If it is
                        a tuple, then it must have the same length as the inputs
                        tuple. In this case, each stdev value in the stdevs tuple
                        corresponds to the input with the same index in the inputs
                        tuple.
                        Default: 0.0
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
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
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It can contain a tuple of ND tensors or
                        any arbitrary python type of any shape.
                        In case of the ND tensor the first dimension of the
                        tensor must correspond to the batch size. It will be
                        repeated for each `n_steps` for each randomly generated
                        input sample.
                        Note that the gradients are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution score computed based on GradientSHAP with respect
                        to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                        This is computed using the property that the total
                        sum of forward_func(inputs) - forward_func(baselines)
                        must be very close to the total sum of the attributions
                        based on GradientSHAP.
                        Delta is calculated for each example in the input after adding
                        `n_samples` times gaussian noise to each of them. Therefore,
                        the dimensionality of the deltas tensor is equal to the
                        `number of examples in the input` * `n_samples`
                        The deltas are ordered by each input example and `n_samples`
                        noisy samples generated for it.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> gradient_shap = GradientShap(net)
            >>> input = torch.randn(3, 3, 32, 32, requires_grad=True)
            >>> # choosing baselines randomly
            >>> baselines = torch.randn(20, 3, 32, 32)
            >>> # Computes gradient shap for the input
            >>> # Attribution size matches input size: 3x3x32x32
            >>> attribution = gradient_shap.attribute(input, baselines,
                                                                target=5)

        """
        # since `baselines` is a distribution, we can generate it using a function
        # rather than passing it as an input argument
        formatted_baselines = _format_callable_baseline(baselines, inputs)
        assert isinstance(formatted_baselines[0], torch.Tensor), (
            "Baselines distribution has to be provided in a form "
            "of a torch.Tensor {}.".format(formatted_baselines[0])
        )

        input_min_baseline_x_grad = InputBaselineXGradient(
            self.forward_func, self.multiplies_by_inputs
        )
        input_min_baseline_x_grad.gradient_func = self.gradient_func

        nt = NoiseTunnel(input_min_baseline_x_grad)

        # NOTE: using attribute.__wrapped__ to not log
        attributions = nt.attribute.__wrapped__(
            nt,  # self
            inputs,
            nt_type="smoothgrad",
            nt_samples=n_samples,
            stdevs=stdevs,
            draw_baseline_from_distrib=True,
            baselines=formatted_baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            return_convergence_delta=return_convergence_delta,
        )

        return attributions

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for GradientShap.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for GradientShap"
        )

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs


class InputBaselineXGradient(GradientAttribution):
    _multiply_by_inputs: bool

    def __init__(
        self, forward_func: Callable[..., Tensor], multiply_by_inputs: bool = True
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then this type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of gradient shap, if `multiply_by_inputs`
                        is set to True, the sensitivity scores of scaled inputs
                        are being multiplied by (inputs - baselines).

        """
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        *,
        return_convergence_delta: Literal[True],
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: Literal[False] = False,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple, baselines = _format_input_baseline(inputs, baselines)

        rand_coefficient = torch.tensor(
            np.random.uniform(0.0, 1.0, inputs_tuple[0].shape[0]),
            device=inputs_tuple[0].device,
            dtype=inputs_tuple[0].dtype,
        )

        input_baseline_scaled = tuple(
            _scale_input(input, baseline, rand_coefficient)
            for input, baseline in zip(inputs_tuple, baselines)
        )
        grads = self.gradient_func(
            self.forward_func, input_baseline_scaled, target, additional_forward_args
        )

        if self.multiplies_by_inputs:
            input_baseline_diffs = tuple(
                input - baseline for input, baseline in zip(inputs_tuple, baselines)
            )
            attributions = tuple(
                input_baseline_diff * grad
                for input_baseline_diff, grad in zip(input_baseline_diffs, grads)
            )
        else:
            attributions = grads

        # pyre-fixme[7]: Expected `Union[Tuple[Variable[TensorOrTupleOfTensorsGeneric...
        return _compute_conv_delta_and_format_attrs(
            self,
            return_convergence_delta,
            attributions,
            baselines,
            inputs,
            additional_forward_args,
            target,
            is_inputs_tuple,
        )

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for InputBaseLineXGradient.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for InputBaseLineXGradient"
        )

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs


def _scale_input(
    input: Tensor, baseline: Union[Tensor, int, float], rand_coefficient: Tensor
) -> Tensor:
    # batch size
    bsz = input.shape[0]
    inp_shape_wo_bsz = input.shape[1:]
    inp_shape = (bsz,) + tuple([1] * len(inp_shape_wo_bsz))

    # expand and reshape the indices
    rand_coefficient = rand_coefficient.view(inp_shape)

    input_baseline_scaled = (
        rand_coefficient * input + (1.0 - rand_coefficient) * baseline
    ).requires_grad_()
    return input_baseline_scaled
