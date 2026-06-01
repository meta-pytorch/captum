#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import typing
from typing import Any, Callable, cast, Dict, Literal, Optional, Sequence, Tuple, Union

import torch
from captum._utils.common import (
    _expand_target,
    _format_additional_forward_args,
    _format_baseline,
    _format_tensor_into_tuples,
    ExpansionTypes,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._utils.attribution import LayerAttribution
from captum.attr._utils.common import (
    _call_custom_attribution_func,
    _compute_conv_delta_and_format_attrs,
    _format_callable_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class LayerDeepLift(LayerAttribution, DeepLift):
    r"""
    Implements DeepLIFT algorithm for the layer based on the following paper:
    Learning Important Features Through Propagating Activation Differences,
    Avanti Shrikumar, et. al.
    https://arxiv.org/abs/1704.02685

    and the gradient formulation proposed in:
    Towards better understanding of gradient-based attribution methods for
    deep neural networks,  Marco Ancona, et.al.
    https://openreview.net/pdf?id=Sy21R9JAW

    This implementation supports only Rescale rule. RevealCancel rule will
    be supported in later releases.
    Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
    Integrated Gradients, it runs significantly faster than Integrated
    Gradients and is preferred for large datasets.

    Currently we only support a limited number of non-linear activations
    but the plan is to expand the list in the future.

    Note: As we know, currently we cannot access the building blocks,
    of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
    Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
    with performance similar to built-in ones using TorchScript.
    More details on how to build custom RNNs can be found here:
    https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """

    def __init__(
        self,
        model: Module,
        layer: Module,
        multiply_by_inputs: bool = True,
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which attributions are computed.
                        The size and dimensionality of the attributions
                        corresponds to the size and dimensionality of the layer's
                        input or output depending on whether we attribute to the
                        inputs or outputs of the layer.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of Layer DeepLift, if `multiply_by_inputs`
                        is set to True, final sensitivity scores
                        are being multiplied by
                        layer activations for inputs - layer activations for baselines.
                        This flag applies only if `custom_attribution_func` is
                        set to None.
        """
        LayerAttribution.__init__(self, model, layer)
        DeepLift.__init__(self, model)
        self.model = model
        self._multiply_by_inputs = multiply_by_inputs

    # Ignoring mypy error for inconsistent signature with DeepLift
    @typing.overload  # type: ignore
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        *,
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]: ...

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attributions are computed. If model takes a
                        single tensor as input, a single input tensor should be
                        provided. If model takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

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

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
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
            additional_forward_args (Any, optional): If the model
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same length.
                        `custom_attribution_func` returns a tuple of attribution
                        tensors that have the same length as the `inputs`.
                        Default: None
            grad_kwargs (Dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Attribution score computed based on DeepLift's rescale rule with
                respect to layer's inputs or outputs. Attributions will always be the
                same size as the provided layer's inputs or outputs, depending on
                whether we attribute to the inputs or outputs of the layer.
                If the layer input / output is a single tensor, then
                just a tensor is returned; if the layer input / output
                has multiple tensors, then a corresponding tuple
                of tensors is returned.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                This is computed using the property that the total sum of
                model(inputs) - model(baselines) must equal the
                total sum of the attributions computed based on DeepLift's
                rescale rule.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in input.
                Note that the logic described for deltas is guaranteed
                when the default logic for attribution computations is used,
                meaning that the `custom_attribution_func=None`, otherwise
                it is not guaranteed and depends on the specifics of the
                `custom_attribution_func`.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = LayerDeepLift(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and class 3.
            >>> attribution = dl.attribute(input, target=1)
        """
        inputs = _format_tensor_into_tuples(inputs)
        baselines = _format_baseline(baselines, inputs)
        _validate_input(inputs, baselines)

        baselines = _tensorize_baseline(inputs, baselines)

        main_model_hooks = []
        try:
            main_model_hooks = self._hook_main_model()

            self.model.apply(
                lambda mod: self._register_hooks(
                    mod, attribute_to_layer_input=attribute_to_layer_input
                )
            )

            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )
            expanded_target = _expand_target(
                target, 2, expansion_type=ExpansionTypes.repeat
            )
            wrapped_forward_func = self._construct_forward_func(
                self.model,
                (inputs, baselines),
                expanded_target,
                additional_forward_args,
            )

            def chunk_output_fn(
                out: TensorOrTupleOfTensorsGeneric,
            ) -> Sequence[Union[Tensor, Sequence[Tensor]]]:
                if isinstance(out, Tensor):
                    return out.chunk(2)
                return tuple(out_sub.chunk(2) for out_sub in out)

            gradients, attrs = compute_layer_gradients_and_eval(
                wrapped_forward_func,
                self.layer,
                inputs,
                attribute_to_layer_input=attribute_to_layer_input,
                output_fn=lambda out: chunk_output_fn(out),
                grad_kwargs=grad_kwargs,
            )

            attr_inputs = tuple(attr[0] for attr in attrs)
            attr_baselines = tuple(attr[1] for attr in attrs)
            gradients = tuple(grad[0] for grad in gradients)

            if custom_attribution_func is None:
                if self.multiplies_by_inputs:
                    attributions = tuple(
                        (input - baseline) * gradient
                        for input, baseline, gradient in zip(
                            attr_inputs, attr_baselines, gradients
                        )
                    )
                else:
                    attributions = gradients
            else:
                attributions = _call_custom_attribution_func(
                    custom_attribution_func, gradients, attr_inputs, attr_baselines
                )
        finally:
            # remove hooks from all activations
            self._remove_hooks(main_model_hooks)

        return _compute_conv_delta_and_format_attrs(
            self,
            return_convergence_delta,
            attributions,
            baselines,
            inputs,
            additional_forward_args,
            target,
            cast(Union[Literal[True], Literal[False]], len(attributions) > 1),
        )

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs


class LayerDeepLiftShap(LayerDeepLift, DeepLiftShap):
    r"""
    Extends LayerDeepLift and DeepLiftShap algorithms and approximates SHAP
    values for given input `layer`.
    For each input sample - baseline pair it computes DeepLift attributions
    with respect to inputs or outputs of given `layer` averages
    resulting attributions across baselines. Whether to compute the attributions
    with respect to the inputs or outputs of the layer is defined by the
    input flag `attribute_to_layer_input`.
    More details about the algorithm can be found here:

    https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

    Note that the explanation model:

        1. Assumes that input features are independent of one another
        2. Is linear, meaning that the explanations are modeled through
            the additive composition of feature effects.

    Although, it assumes a linear model for each explanation, the overall
    model across multiple explanations can be complex and non-linear.
    """

    def __init__(
        self,
        model: Module,
        layer: Module,
        multiply_by_inputs: bool = True,
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which attributions are computed.
                        The size and dimensionality of the attributions
                        corresponds to the size and dimensionality of the layer's
                        input or output depending on whether we attribute to the
                        inputs or outputs of the layer.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of LayerDeepLiftShap, if `multiply_by_inputs`
                        is set to True, final sensitivity scores are being
                        multiplied by
                        layer activations for inputs - layer activations for baselines
                        This flag applies only if `custom_attribution_func` is
                        set to None.
        """
        LayerDeepLift.__init__(self, model, layer)
        DeepLiftShap.__init__(self, model, multiply_by_inputs)

    # Ignoring mypy error for inconsistent signature with DeepLiftShap
    @typing.overload  # type: ignore
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[
            Tensor, Tuple[Tensor, ...], Callable[..., Union[Tensor, Tuple[Tensor, ...]]]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        *,
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]: ...

    @typing.overload  # type: ignore
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[
            Tensor, Tuple[Tensor, ...], Callable[..., Union[Tensor, Tuple[Tensor, ...]]]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: Union[
            Tensor, Tuple[Tensor, ...], Callable[..., Union[Tensor, Tuple[Tensor, ...]]]
        ],
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> Union[
        Tensor, Tuple[Tensor, ...], Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attributions are computed. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (Tensor, tuple[Tensor, ...], or Callable):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references. Baselines can be provided as:

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
            additional_forward_args (Any, optional): If the model
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it assumes that both the inputs and
                        outputs of internal layers are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution score computed based on DeepLift's rescale rule
                        with respect to layer's inputs or outputs. Attributions
                        will always be the same size as the provided layer's inputs
                        or outputs, depending on whether we attribute to the inputs
                        or outputs of the layer.
                        Attributions are returned in a tuple based on whether
                        the layer inputs / outputs are contained in a tuple
                        from a forward hook. For standard modules, inputs of
                        a single tensor are usually wrapped in a tuple, while
                        outputs of a single tensor are not.
            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                        This is computed using the property that the
                        total sum of model(inputs) - model(baselines)
                        must be very close to the total sum of attributions
                        computed based on approximated SHAP values using
                        DeepLift's rescale rule.
                        Delta is calculated for each example input and baseline pair,
                        meaning that the number of elements in returned delta tensor
                        is equal to the
                        `number of examples in input` * `number of examples
                        in baseline`. The deltas are ordered in the first place by
                        input example, followed by the baseline.
                        Note that the logic described for deltas is guaranteed
                        when the default logic for attribution computations is used,
                        meaning that the `custom_attribution_func=None`, otherwise
                        it is not guaranteed and depends on the specifics of the
                        `custom_attribution_func`.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = LayerDeepLiftShap(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes shap values using deeplift for class 3.
            >>> attribution = dl.attribute(input, target=3)
        """
        inputs = _format_tensor_into_tuples(inputs)
        baselines = _format_callable_baseline(baselines, inputs)

        assert isinstance(baselines[0], torch.Tensor) and baselines[0].shape[0] > 1, (
            "Baselines distribution has to be provided in form of a torch.Tensor"
            " with more than one example but found: {}."
            " If baselines are provided in shape of scalars or with a single"
            " baseline example, `LayerDeepLift`"
            " approach can be used instead.".format(baselines[0])
        )

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]

        (
            exp_inp,
            exp_base,
            exp_target,
            exp_addit_args,
        ) = DeepLiftShap._expand_inputs_baselines_targets(
            self, baselines, inputs, target, additional_forward_args
        )
        attribs_layer_deeplift = LayerDeepLift.attribute.__wrapped__(  # type: ignore
            self,
            exp_inp,
            exp_base,
            target=exp_target,
            additional_forward_args=exp_addit_args,
            return_convergence_delta=cast(
                Literal[True, False],
                return_convergence_delta,
            ),
            attribute_to_layer_input=attribute_to_layer_input,
            custom_attribution_func=custom_attribution_func,
        )
        delta: Tensor
        attributions: Union[Tensor, Tuple[Tensor, ...]]
        if return_convergence_delta:
            attributions, delta = attribs_layer_deeplift
        else:
            attributions = attribs_layer_deeplift
        if isinstance(attributions, tuple):
            attributions = tuple(
                DeepLiftShap._compute_mean_across_baselines(
                    self, inp_bsz, base_bsz, cast(Tensor, attrib)
                )
                for attrib in attributions
            )
        else:
            attributions = DeepLiftShap._compute_mean_across_baselines(
                self, inp_bsz, base_bsz, attributions
            )
        if return_convergence_delta:
            return attributions, delta
        else:
            return cast(
                Union[
                    Tensor,
                    Tuple[Tensor, ...],
                    Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor],
                ],
                attributions,
            )

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs
