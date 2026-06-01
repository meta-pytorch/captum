#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable, cast, List, Optional, Tuple, Union

from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
)
from captum._utils.gradient import (
    _forward_layer_eval_with_neuron_grads,
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import SliceIntType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution, NeuronAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class NeuronGradient(NeuronAttribution, GradientAttribution):
    r"""
    Computes the gradient of the output of a particular neuron with
    respect to the inputs of the network.
    """

    def __init__(
        self,
        forward_func: Callable[..., Union[int, float, Tensor]],
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer.
                          Currently, it is assumed that the inputs or the outputs
                          of the layer, depending on which one is used for
                          attribution, can only be a single tensor.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        NeuronAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        neuron_selector: Union[
            int,
            Tuple[Union[int, SliceIntType], ...],
            Callable[[Union[Tensor, Tuple[Tensor, ...]]], Tensor],
        ],
        additional_forward_args: Optional[object] = None,
        attribute_to_neuron_input: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which neuron
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            neuron_selector (int, Callable, tuple[int], or slice):
                        Selector for neuron
                        in given layer for which attribution is desired.
                        Neuron selector can be provided as:

                        - a single integer, if the layer output is 2D. This integer
                          selects the appropriate neuron column in the layer input
                          or output

                        - a tuple of integers or slice objects. Length of this
                          tuple must be one less than the number of dimensions
                          in the input / output of the given layer (since
                          dimension 0 corresponds to number of examples).
                          The elements of the tuple can be either integers or
                          slice objects (slice object allows indexing a
                          range of neurons rather individual ones).

                          If any of the tuple elements is a slice object, the
                          indexed output tensor is used for attribution. Note
                          that specifying a slice of a tensor would amount to
                          computing the attribution of the sum of the specified
                          neurons, and not the individual neurons independently.

                        - a callable, which should
                          take the target layer as input (single tensor or tuple
                          if multiple tensors are in layer) and return a neuron or
                          aggregate of the layer's neurons for attribution.
                          For example, this function could return the
                          sum of the neurons in the layer or sum of neurons with
                          activations in a particular range. It is expected that
                          this function returns either a tensor with one element
                          or a 1D tensor with length equal to batch_size (one scalar
                          per input example)
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_neuron_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the neuron input
                        or output. If `attribute_to_neuron_input` is set to True
                        then the attributions will be computed with respect to
                        neuron's inputs, otherwise it will be computed with respect
                        to neuron's outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal neurons, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Gradients of particular neuron with respect to each input
                        feature. Attributions will always be the same size as the
                        provided inputs, with each value providing the attribution
                        of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> neuron_ig = NeuronGradient(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
            >>> # which indexes a particular neuron in the layer output.
            >>> # For this example, we choose the index (4,1,2).
            >>> # Computes neuron gradient for neuron with
            >>> # index (4,1,2).
            >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        _, input_grads = _forward_layer_eval_with_neuron_grads(
            self.forward_func,
            inputs,
            self.layer,
            additional_forward_args,
            gradient_neuron_selector=neuron_selector,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
        )

        undo_gradient_requirements(inputs_tuple, gradient_mask)

        return cast(
            TensorOrTupleOfTensorsGeneric, _format_output(is_inputs_tuple, input_grads)
        )
