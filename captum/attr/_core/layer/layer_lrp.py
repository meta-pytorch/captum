#!/usr/bin/env python3

# pyre-strict
import typing
from typing import cast, Dict, List, Literal, Optional, Tuple, TypeVar, Union

import torch
from captum._utils.common import (
    _format_tensor_into_tuples,
    _reduce_list,
    _sort_key_list,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    undo_gradient_requirements,
)
from captum._utils.typing import (
    ModuleOrModuleList,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.lrp import LRP
from captum.attr._utils.attribution import LayerAttribution
from captum.attr._utils.lrp_rules import PropagationRule
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

T = TypeVar("T")


class LayerLRP(LRP, LayerAttribution):
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

    device_ids: List[int]
    verbose: bool
    layers: List[Module]
    attribute_to_layer_input: bool = False
    backward_handles: List[RemovableHandle]
    forward_handles: List[RemovableHandle]

    def __init__(self, model: Module, layer: ModuleOrModuleList) -> None:
        """
        Args:

            model (Module): The forward function of the model or
                        any modification of it. Custom rules for a given layer need to
                        be defined as attribute
                        `module.rule` and need to be of type PropagationRule.

            layer (torch.nn.Module or list(torch.nn.Module)): Layer or layers
                          for which attributions are computed.
                          The size and dimensionality of the attributions
                          corresponds to the size and dimensionality of the layer's
                          input or output depending on whether we attribute to the
                          inputs or outputs of the layer. If value is None, the
                          relevance for all layers is returned in attribution.
        """
        LayerAttribution.__init__(self, model, layer)
        LRP.__init__(self, model)
        if hasattr(self.model, "device_ids"):
            self.device_ids = cast(List[int], self.model.device_ids)

    @typing.overload  # type: ignore
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        *,
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Tuple[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Union[Tensor, List[Tensor]],
    ]: ...

    @typing.overload
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: Literal[False] = False,
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]: ...

    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        verbose: bool = False,
    ) -> Union[
        Tensor,
        Tuple[Tensor, ...],
        List[Union[Tensor, Tuple[Tensor, ...]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Union[Tensor, List[Tensor]],
        ],
    ]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which relevance is
                        propagated.
                        If model takes a single
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

            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.

            verbose (bool, optional): Indicates whether information on application
                    of rules is printed during propagation.
                    Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions** or 2-element tuple of
            **attributions**, **delta** or list of **attributions** and **delta**:

              - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The propagated relevance values with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned. The sum of attributions
                        is one and not corresponding to the prediction score as in other
                        implementations. If attributions for all layers are returned
                        (layer=None) a list of tensors or tuples of tensors is returned
                        with entries for each layer.
              - **delta** (*Tensor* or list of *Tensor*
                        returned if return_convergence_delta=True):
                        Delta is calculated per example, meaning that the number of
                        elements in returned delta tensor is equal to the number of
                        examples in input.
                        If attributions for all layers are returned (layer=None) a list
                        of tensors is returned with entries for
                        each layer.

        Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities. It has one
                >>> # Conv2D and a ReLU layer.
                >>> net = ImageClassifier()
                >>> layer_lrp = LayerLRP(net, net.conv1)
                >>> input = torch.randn(3, 3, 32, 32)
                >>> # Attribution size matches input size: 3x3x32x32
                >>> attribution = layer_lrp.attribute(input, target=5)

        """
        self.verbose = verbose
        self._original_state_dict = self.model.state_dict()
        self.layers = []
        self._get_layers(self.model)
        self._check_and_attach_rules()
        self.attribute_to_layer_input = attribute_to_layer_input
        self.backward_handles = []
        self.forward_handles = []

        inputs_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        try:
            # 1. Forward pass
            output = self._compute_output_and_change_weights(
                inputs_tuple,
                target,
                additional_forward_args,
            )
            self._register_forward_hooks()
            # 2. Forward pass + backward pass
            _ = compute_gradients(
                self._forward_fn_wrapper, inputs_tuple, target, additional_forward_args
            )
            relevances = self._get_output_relevance(output)
        finally:
            self._restore_model()
        undo_gradient_requirements(inputs_tuple, gradient_mask)

        if return_convergence_delta:
            delta: Union[Tensor, List[Tensor]]
            if isinstance(self.layer, list):
                delta = []
                for relevance_layer in relevances:
                    delta.append(
                        self.compute_convergence_delta(
                            cast(Union[Tensor, Tuple[Tensor, ...]], relevance_layer),
                            output,
                        )
                    )
            else:
                delta = self.compute_convergence_delta(
                    cast(Tuple[Tensor, ...], relevances), output
                )
            return relevances, delta  # type: ignore
        else:
            return relevances  # type: ignore

    def _get_single_output_relevance(
        self, layer: Module, output: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        if self.attribute_to_layer_input:
            normalized_relevances = cast(
                Dict[torch.device, Tensor],
                cast(PropagationRule, layer.rule).relevance_input,
            )
        else:
            normalized_relevances = cast(PropagationRule, layer.rule).relevance_output
        key_list = _sort_key_list(list(normalized_relevances.keys()), self.device_ids)
        normalized_relevances_reduced = _reduce_list(
            [normalized_relevances[device_id] for device_id in key_list]
        )

        if isinstance(normalized_relevances_reduced, tuple):
            return tuple(
                normalized_relevance
                * output.reshape((-1,) + (1,) * (normalized_relevance.dim() - 1))
                for normalized_relevance in normalized_relevances_reduced
            )
        else:
            return normalized_relevances_reduced * output.reshape(
                (-1,) + (1,) * (normalized_relevances_reduced.dim() - 1)
            )

    def _get_output_relevance(
        self, output: Tensor
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        if isinstance(self.layer, list):
            relevances = []
            for layer in self.layer:
                relevances.append(self._get_single_output_relevance(layer, output))
            return relevances
        else:
            return self._get_single_output_relevance(self.layer, output)

    @staticmethod
    def _convert_list_to_tuple(
        relevances: Union[List[T], Tuple[T, ...]],
    ) -> Tuple[T, ...]:
        if isinstance(relevances, list):
            return tuple(relevances)
        else:
            return relevances
