#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import TargetType
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class LayerGradCam(LayerAttribution, GradientAttribution):
    r"""
    Computes GradCAM attribution for chosen layer. GradCAM is designed for
    convolutional neural networks, and is usually applied to the last
    convolutional layer.

    GradCAM computes the gradients of the target output with respect to
    the given layer, averages for each output channel (dimension 2 of
    output), and multiplies the average gradient for each channel by the
    layer activations. The results are summed over all channels.

    Note that in the original GradCAM algorithm described in the paper,
    ReLU is applied to the output, returning only non-negative attributions.
    For providing more flexibility to the user, we choose to not perform the
    ReLU internally by default and return the sign information. To match the
    original GradCAM algorithm, it is necessary to pass the parameter
    relu_attributions=True to apply ReLU on the final
    attributions or alternatively only visualize the positive attributions.

    Note: this procedure sums over the second dimension (# of channels),
    so the output of GradCAM attributions will have a second
    dimension of 1, but all other dimensions will match that of the layer
    output.

    GradCAM attributions are generally upsampled and can be viewed as a
    mask to the input, since a convolutional layer output generally
    matches the input image spatially. This upsampling can be performed
    using LayerAttribution.interpolate, as shown in the example below.

    More details regarding the GradCAM method can be found in the
    original paper here:
    https://arxiv.org/abs/1610.02391
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: Module,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Output size of attribute matches this layer's output
                          dimensions, except for dimension 2, which will be 1,
                          since GradCAM sums over channels.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        attr_dim_summation: bool = True,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
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
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to the
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the outputs of internal layers, depending on whether we
                        attribute to the input or output, are single tensors.
                        Support for multiple tensors will be added later.
                        Default: False
            relu_attributions (bool, optional): Indicates whether to
                        apply a ReLU operation on the final attribution,
                        returning only non-negative attributions. Setting this
                        flag to True matches the original GradCAM algorithm,
                        otherwise, by default, both positive and negative
                        attributions are returned.
                        Default: False
            attr_dim_summation (bool, optional): Indicates whether to
                        sum attributions along dimension 1 (usually channel).
                        The default (True) means to sum along dimension 1.
                        Default: True
            grad_kwargs (Dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attributions based on GradCAM method.
                        Attributions will be the same size as the
                        output of the given layer, except for dimension 2,
                        which will be 1 due to summing over channels.
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.
        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains a layer conv4, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx50x8x8.
            >>> # It is the last convolution layer, which is the recommended
            >>> # use case for GradCAM.
            >>> net = ImageClassifier()
            >>> layer_gc = LayerGradCam(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer GradCAM for class 3.
            >>> # attribution size matches layer output except for dimension
            >>> # 1, so dimensions of attr would be Nx1x8x8.
            >>> attr = layer_gc.attribute(input, 3)
            >>> # GradCAM attributions are often upsampled and viewed as a
            >>> # mask to the input, since the convolutional layer output
            >>> # spatially matches the original input image.
            >>> # This can be done with LayerAttribution's interpolate method.
            >>> upsampled_attr = LayerAttribution.interpolate(attr, (32, 32))
        """
        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            cast(Module, self.layer),
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
            grad_kwargs=grad_kwargs,
        )

        summed_grads = tuple(
            (
                torch.mean(
                    layer_grad,
                    dim=tuple(x for x in range(2, len(layer_grad.shape))),
                    keepdim=True,
                )
                if len(layer_grad.shape) > 2
                else layer_grad
            )
            for layer_grad in layer_gradients
        )

        if attr_dim_summation:
            scaled_acts = tuple(
                torch.sum(summed_grad * layer_eval, dim=1, keepdim=True)
                for summed_grad, layer_eval in zip(summed_grads, layer_evals)
            )
        else:
            scaled_acts = tuple(
                summed_grad * layer_eval
                for summed_grad, layer_eval in zip(summed_grads, layer_evals)
            )

        if relu_attributions:
            scaled_acts = tuple(F.relu(scaled_act) for scaled_act in scaled_acts)
        return _format_output(len(scaled_acts) > 1, scaled_acts)
