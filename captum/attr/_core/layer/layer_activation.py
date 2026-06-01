#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable, cast, List, Optional, Tuple, Union

import torch
from captum._utils.common import _format_output
from captum._utils.gradient import _forward_layer_eval
from captum._utils.typing import ModuleOrModuleList
from captum.attr._utils.attribution import LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class LayerActivation(LayerAttribution):
    r"""
    Computes activation of selected layer for given input.
    """

    def __init__(
        self,
        forward_func: Callable[..., Union[int, float, Tensor]],
        layer: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                          modification of it
            layer (torch.nn.Module or list of torch.nn.Module): Layer or layers
                          for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer. If multiple layers are provided, attributions
                          are returned as a list, each element corresponding to the
                          activations of the corresponding layer.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        additional_forward_args: Optional[object] = None,
        attribute_to_layer_input: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        activation is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
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

        Returns:
            *Tensor* or *tuple[Tensor, ...]* or list of **attributions**:
            - **attributions** (*Tensor*, *tuple[Tensor, ...]*, or *list*):
                        Activation of each neuron in given layer output.
                        Attributions will always be the same size as the
                        output of the given layer.
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.
                        If multiple layers are provided, attributions
                        are returned as a list, each element corresponding to the
                        activations of the corresponding layer.



        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> layer_act = LayerActivation(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer activation.
            >>> # attribution is layer output, with size Nx12x32x32
            >>> attribution = layer_act.attribute(input)
        """
        with torch.no_grad():
            layer_eval = _forward_layer_eval(
                self.forward_func,
                inputs,
                self.layer,
                additional_forward_args,
                device_ids=self.device_ids,
                attribute_to_layer_input=attribute_to_layer_input,
            )
        if isinstance(self.layer, Module):
            return _format_output(
                len(layer_eval) > 1, cast(Tuple[Tensor, ...], layer_eval)
            )
        else:
            return [
                _format_output(len(single_layer_eval) > 1, single_layer_eval)
                for single_layer_eval in layer_eval
            ]

    @property
    def multiplies_by_inputs(self) -> bool:
        return True
