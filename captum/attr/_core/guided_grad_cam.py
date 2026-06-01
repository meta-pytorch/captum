#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from typing import cast, List, Optional, Union

import torch
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.guided_backprop_deconvnet import GuidedBackprop
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class GuidedGradCam(GradientAttribution):
    r"""
    Computes element-wise product of guided backpropagation attributions
    with upsampled (non-negative) GradCAM attributions.
    GradCAM attributions are computed with respect to the layer
    provided in the constructor, and attributions
    are upsampled to match the input size. GradCAM is designed for
    convolutional neural networks, and is usually applied to the last
    convolutional layer.

    Note that if multiple input tensors are provided, attributions for
    each input tensor are computed by upsampling the GradCAM
    attributions to match that input's dimensions. If interpolation is
    not possible for the input tensor dimensions and interpolation mode,
    then an empty tensor is returned in the attributions for the
    corresponding position of that input tensor. This can occur if the
    input tensor does not have the same number of dimensions as the chosen
    layer's output or is not either 3D, 4D or 5D.

    Note that attributions are only meaningful for input tensors
    which are spatially alligned with the chosen layer, e.g. an input
    image tensor for a convolutional layer.

    More details regarding GuidedGradCAM can be found in the original
    GradCAM paper here:
    https://arxiv.org/abs/1610.02391

    Warning: Ensure that all ReLU operations in the forward function of the
    given model are performed using a module (nn.module.ReLU).
    If nn.functional.ReLU is used, gradients are not overridden appropriately.
    """

    def __init__(
        self, model: Module, layer: Module, device_ids: Union[None, List[int]] = None
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which GradCAM attributions are computed.
                          Currently, only layers with a single tensor output are
                          supported.
            device_ids (list[int]): Device ID list, necessary only if model
                          is a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If model is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        GradientAttribution.__init__(self, model)
        self.grad_cam = LayerGradCam(model, layer, device_ids)
        self.guided_backprop = GuidedBackprop(model)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which attributions
                        are computed. If model takes a single
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
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to model in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            interpolate_mode (str, optional): Method for interpolation, which
                        must be a valid input interpolation mode for
                        torch.nn.functional. These methods are
                        "nearest", "area", "linear" (3D-only), "bilinear"
                        (4D-only), "bicubic" (4D-only), "trilinear" (5D-only)
                        based on the number of dimensions of the chosen layer
                        output (which must also match the number of
                        dimensions for the input tensor). Note that
                        the original GradCAM paper uses "bilinear"
                        interpolation, but we default to "nearest" for
                        applicability to any of 3D, 4D or 5D tensors.
                        Default: "nearest"
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output in `LayerGradCam`.
                        If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer inputs, otherwise it will be computed with respect
                        to layer outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            *Tensor* of **attributions**:
            - **attributions** (*Tensor*):
                    Element-wise product of (upsampled) GradCAM
                    and Guided Backprop attributions.
                    If a single tensor is provided as inputs, a single tensor is
                    returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.
                    Attributions will be the same size as the provided inputs,
                    with each value providing the attribution of the
                    corresponding input index.
                    If the GradCAM attributions cannot be upsampled to the shape
                    of a given input tensor, None is returned in the corresponding
                    index position.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv4, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx50x8x8.
            >>> # It is the last convolution layer, which is the recommended
            >>> # use case for GuidedGradCAM.
            >>> net = ImageClassifier()
            >>> guided_gc = GuidedGradCam(net, net.conv4)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes guided GradCAM attributions for class 3.
            >>> # attribution size matches input size, Nx3x32x32
            >>> attribution = guided_gc.attribute(input, 3)
        """
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple = _format_tensor_into_tuples(inputs)
        grad_cam_attr = self.grad_cam.attribute.__wrapped__(
            self.grad_cam,  # self
            inputs=inputs_tuple,
            target=target,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=True,
        )
        if isinstance(grad_cam_attr, tuple):
            assert len(grad_cam_attr) == 1, (
                "GuidedGradCAM attributions for layer with multiple inputs / "
                "outputs is not supported."
            )
            grad_cam_attr = grad_cam_attr[0]

        guided_backprop_attr = self.guided_backprop.attribute.__wrapped__(
            self.guided_backprop,  # self
            inputs=inputs_tuple,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        output_attr: List[Tensor] = []
        for i in range(len(inputs_tuple)):
            try:
                output_attr.append(
                    guided_backprop_attr[i]
                    * LayerAttribution.interpolate(
                        grad_cam_attr,
                        tuple(inputs_tuple[i].shape[2:]),
                        interpolate_mode=interpolate_mode,
                    )
                )
            except Exception:
                warnings.warn(
                    "Couldn't appropriately interpolate GradCAM attributions for some "
                    "input tensors, returning empty tensor for corresponding "
                    "attributions.",
                    stacklevel=1,
                )
                output_attr.append(torch.empty(0))

        return cast(
            TensorOrTupleOfTensorsGeneric,
            _format_output(is_inputs_tuple, tuple(output_attr)),
        )
