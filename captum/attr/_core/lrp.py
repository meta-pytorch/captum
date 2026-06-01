#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import typing
from collections import defaultdict
from typing import Any, cast, Dict, List, Literal, Optional, Tuple, Union

import torch.nn as nn
from captum._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
    _run_forward,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.attr._utils.common import _sum_rows
from captum.attr._utils.custom_modules import Addition_Module
from captum.attr._utils.lrp_rules import EpsilonRule, PropagationRule
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


class LRP(GradientAttribution):
    r"""
    Layer-wise relevance propagation is based on a backward propagation
    mechanism applied sequentially to all layers of the model. Here, the
    model output score represents the initial relevance which is decomposed
    into values for each neuron of the underlying layers. The decomposition
    is defined by rules that are chosen for each layer, involving its weights
    and activations. Details on the model can be found in the original paper
    [https://doi.org/10.1371/journal.pone.0130140]. The implementation is
    inspired by the tutorial of the same group
    [https://doi.org/10.1016/j.dsp.2017.10.011] and the publication by
    Ancona et al. [https://openreview.net/forum?id=Sy21R9JAW].
    """

    verbose: bool = False
    _original_state_dict: Dict[str, Any] = {}
    layers: List[Module] = []
    backward_handles: List[RemovableHandle] = []
    forward_handles: List[RemovableHandle] = []

    def __init__(self, model: Module) -> None:
        r"""
        Args:

            model (Module): The forward function of the model or any modification of
                it. Custom rules for a given layer need to be defined as attribute
                `module.rule` and need to be of type PropagationRule. If no rule is
                specified for a layer, a pre-defined default rule for the module type
                is used.
        """
        GradientAttribution.__init__(self, model)
        self.model = model
        self._layers_with_added_rules: List[Module] = []
        self._check_rules()

    @property
    def multiplies_by_inputs(self) -> bool:
        return True

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        *,
        return_convergence_delta: Literal[True],
        verbose: bool = False,
    ) -> Tuple[TensorOrTupleOfTensorsGeneric, Tensor]: ...

    @typing.overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: Literal[False] = False,
        verbose: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric: ...

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: bool = False,
        verbose: bool = False,
    ) -> Union[
        TensorOrTupleOfTensorsGeneric, Tuple[TensorOrTupleOfTensorsGeneric, Tensor]
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which relevance is
                        propagated. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.

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
            additional_forward_args (tuple, optional): If the forward function
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

            verbose (bool, optional): Indicates whether information on application
                    of rules is printed during propagation.

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**
            or 2-element tuple of **attributions**, **delta**:

              - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The propagated relevance values with respect to each
                        input feature. The values are normalized by the output score
                        value (sum(relevance)=1). To obtain values comparable to other
                        methods or implementations these values need to be multiplied
                        by the output score. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations.

              - **delta** (*Tensor*, returned if return_convergence_delta=True):
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        of examples in the inputs.

        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> lrp = LRP(net)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = lrp.attribute(input, target=5)

        """
        self.verbose = verbose
        self._original_state_dict = self.model.state_dict()
        self.layers = []
        self._get_layers(self.model)
        self._layers_with_added_rules = []
        self._check_and_attach_rules()
        self.backward_handles: List[RemovableHandle] = []
        self.forward_handles: List[RemovableHandle] = []

        is_inputs_tuple = _is_tuple(inputs)
        input_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(input_tuple)

        try:
            # 1. Forward pass: Change weights of layers according to selected rules.
            output = self._compute_output_and_change_weights(
                input_tuple,
                target,
                additional_forward_args,
            )
            # 2. Forward pass + backward pass: Register hooks to configure relevance
            # propagation and execute back-propagation.
            self._register_forward_hooks()
            normalized_relevances = self.gradient_func(
                self._forward_fn_wrapper, input_tuple, target, additional_forward_args
            )
            relevances = tuple(
                normalized_relevance
                * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
                for normalized_relevance in normalized_relevances
            )
        finally:
            self._restore_model()

        undo_gradient_requirements(input_tuple, gradient_mask)

        if return_convergence_delta:
            return (
                cast(
                    TensorOrTupleOfTensorsGeneric,
                    _format_output(is_inputs_tuple, relevances),
                ),
                self.compute_convergence_delta(relevances, output),
            )
        else:
            return _format_output(is_inputs_tuple, relevances)  # type: ignore

    def attribute_future(self) -> None:
        r"""
        This method is not implemented for LRP.
        """
        raise NotImplementedError("attribute_future is not implemented for LRP")

    def has_convergence_delta(self) -> bool:
        return True

    def compute_convergence_delta(
        self, attributions: Union[Tensor, Tuple[Tensor, ...]], output: Tensor
    ) -> Tensor:
        """
        Here, we use the completeness property of LRP: The relevance is conserved
        during the propagation through the models' layers. Therefore, the difference
        between the sum of attribution (relevance) values and model output is taken as
        the convergence delta. It should be zero for functional attribution. However,
        when rules with an epsilon value are used for stability reasons, relevance is
        absorbed during propagation and the convergence delta is non-zero.

        Args:

            attributions (Tensor or tuple[Tensor, ...]): Attribution scores that
                        are precomputed by an attribution algorithm.
                        Attributions can be provided in form of a single tensor
                        or a tuple of those. It is assumed that attribution
                        tensor's dimension 0 corresponds to the number of
                        examples, and if multiple input tensors are provided,
                        the examples must be aligned appropriately.

            output (Tensor): The output value with respect to which
                        the attribution values are computed. This value corresponds to
                        the target score of a classification model. The given tensor
                        should only have a single element.

        Returns:
            *Tensor*:
            - **delta** Difference of relevance in output layer and input layer.
        """
        if isinstance(attributions, tuple):
            for attr in attributions:
                summed_attr = cast(
                    Tensor, sum(_sum_rows(attr) for attr in attributions)
                )
        else:
            summed_attr = _sum_rows(attributions)
        return output.flatten() - summed_attr.flatten()

    def _get_layers(self, model: Module) -> None:
        for layer in model.children():
            if len(list(layer.children())) == 0:
                self.layers.append(layer)
            else:
                self._get_layers(layer)

    def _check_and_attach_rules(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "rule"):
                layer.activations = {}  # type: ignore
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
                pass
            elif type(layer) in SUPPORTED_LAYERS_WITH_RULES.keys():
                layer.activations = {}  # type: ignore
                layer.rule = SUPPORTED_LAYERS_WITH_RULES[type(layer)]()  # type: ignore
                self._layers_with_added_rules.append(layer)
                layer.rule.relevance_input = defaultdict(list)  # type: ignore
                layer.rule.relevance_output = {}  # type: ignore
            elif type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                layer.rule = None  # type: ignore
                self._layers_with_added_rules.append(layer)
            else:
                raise TypeError(
                    (
                        f"Module of type {type(layer)} has no rule defined and no"
                        "default rule exists for this module type. Please, set a rule"
                        "explicitly for this module and assure that it is appropriate"
                        "for this type of layer."
                    )
                )

    def _check_rules(self) -> None:
        for module in self.model.modules():
            if hasattr(module, "rule"):
                if (
                    not isinstance(module.rule, PropagationRule)
                    and module.rule is not None
                ):
                    raise TypeError(
                        (
                            f"Please select propagation rules inherited from class "
                            f"PropagationRule for module: {module}"
                        )
                    )

    def _register_forward_hooks(self) -> None:
        for layer in self.layers:
            if type(layer) in SUPPORTED_NON_LINEAR_LAYERS:
                backward_handles = _register_backward_hook(
                    layer, PropagationRule.backward_hook_activation, self
                )
                self.backward_handles.extend(backward_handles)
            else:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook  # type: ignore
                )
                self.forward_handles.append(forward_handle)
                if self.verbose:
                    print(f"Applied {layer.rule} on layer {layer}")

    def _register_weight_hooks(self) -> None:
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_hook(
                    layer.rule.forward_hook_weights  # type: ignore
                )
                self.forward_handles.append(forward_handle)

    def _register_pre_hooks(self) -> None:
        for layer in self.layers:
            if layer.rule is not None:
                forward_handle = layer.register_forward_pre_hook(
                    layer.rule.forward_pre_hook_activations  # type: ignore
                )
                self.forward_handles.append(forward_handle)

    def _compute_output_and_change_weights(
        self,
        inputs: Tuple[Tensor, ...],
        target: TargetType,
        additional_forward_args: Optional[object],
    ) -> Tensor:
        try:
            self._register_weight_hooks()
            output = _run_forward(self.model, inputs, target, additional_forward_args)
        finally:
            self._remove_forward_hooks()
        # Register pre_hooks that pass the initial activations from before weight
        # adjustments as inputs to the layers with adjusted weights. This procedure
        # is important for graph generation in the 2nd forward pass.
        self._register_pre_hooks()

        # _run_forward may return future of Tensor,
        # but we don't support it here now
        # And it will fail before here.
        return cast(Tensor, output)

    def _remove_forward_hooks(self) -> None:
        for forward_handle in self.forward_handles:
            forward_handle.remove()

    def _remove_backward_hooks(self) -> None:
        for backward_handle in self.backward_handles:
            backward_handle.remove()
        for layer in self.layers:
            if hasattr(layer.rule, "_handle_input_hooks"):
                for handle in layer.rule._handle_input_hooks:  # type: ignore
                    handle.remove()
            if hasattr(layer.rule, "_handle_output_hook"):
                layer.rule._handle_output_hook.remove()  # type: ignore

    def _remove_rules(self) -> None:
        for layer in self._layers_with_added_rules:
            if hasattr(layer, "rule"):
                del layer.rule

    def _clear_properties(self) -> None:
        for layer in self.layers:
            if hasattr(layer, "activation"):
                del layer.activation

    def _restore_state(self) -> None:
        self.model.load_state_dict(self._original_state_dict)  # type: ignore

    def _restore_model(self) -> None:
        self._restore_state()
        self._remove_backward_hooks()
        self._remove_forward_hooks()
        self._remove_rules()
        self._clear_properties()

    def _forward_fn_wrapper(self, *inputs: Tensor) -> Tensor:
        """
        Wraps a forward function with addition of zero as a workaround to
        https://github.com/pytorch/pytorch/issues/35802 discussed in
        https://github.com/pytorch/captum/issues/143#issuecomment-611750044

        #TODO: Remove when bugs are fixed
        """
        adjusted_inputs = tuple(
            input + 0 if input is not None else input for input in inputs
        )
        return self.model(*adjusted_inputs)


SUPPORTED_LAYERS_WITH_RULES = {
    nn.MaxPool1d: EpsilonRule,
    nn.MaxPool2d: EpsilonRule,
    nn.MaxPool3d: EpsilonRule,
    nn.Conv1d: EpsilonRule,
    nn.Conv2d: EpsilonRule,
    nn.Conv3d: EpsilonRule,
    nn.AvgPool1d: EpsilonRule,
    nn.AvgPool2d: EpsilonRule,
    nn.AvgPool3d: EpsilonRule,
    nn.AdaptiveAvgPool1d: EpsilonRule,
    nn.AdaptiveAvgPool2d: EpsilonRule,
    nn.AdaptiveAvgPool3d: EpsilonRule,
    nn.Linear: EpsilonRule,
    nn.BatchNorm1d: EpsilonRule,
    nn.BatchNorm2d: EpsilonRule,
    nn.BatchNorm3d: EpsilonRule,
    Addition_Module: EpsilonRule,
}

SUPPORTED_NON_LINEAR_LAYERS = [nn.ReLU, nn.Dropout, nn.Tanh]
