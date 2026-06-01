#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

import torch
from captum._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _run_forward,
)
from captum._utils.gradient import _forward_layer_eval
from captum._utils.typing import BaselineType, TargetType
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._utils.attribution import LayerAttribution, PerturbationAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.scatter_gather import scatter


class LayerFeatureAblation(LayerAttribution, PerturbationAttribution):
    r"""
    A perturbation based approach to computing layer attribution, involving
    replacing values in the input / output of a layer with a given baseline /
    reference, and computing the difference in output. By default, each
    neuron (scalar input / output value) within the layer is replaced
    independently.
    Passing a layer mask allows grouping neurons to be
    ablated together.
    Each neuron in the group will be given the same attribution value
    equal to the change in target as a result of ablating the entire neuron
    group.
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
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself
                          (or otherwise has a device_ids attribute with the device
                          ID list), then it is not necessary to provide this
                          argument.
        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        PerturbationAttribution.__init__(self, forward_func)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        layer_baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        layer_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        attribute_to_layer_input: bool = False,
        perturbations_per_eval: int = 1,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            layer_baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Layer baselines define reference values which replace each
                        layer input / output value when ablated.
                        Layer baselines should be a single tensor with dimensions
                        matching the input / output of the target layer (or
                        broadcastable to match it), based
                        on whether we are attributing to the input or output
                        of the target layer.
                        In the cases when `baselines` is not provided, we internally
                        use zero as the baseline for each neuron.
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
            layer_mask (Tensor or tuple[Tensor, ...], optional):
                        layer_mask defines a mask for the layer, grouping
                        elements of the layer input / output which should be
                        ablated together.
                        layer_mask should be a single tensor with dimensions
                        matching the input / output of the target layer (or
                        broadcastable to match it), based
                        on whether we are attributing to the input or output
                        of the target layer. layer_mask
                        should contain integers in the range 0 to num_groups
                        - 1, and all elements with the same value are
                        considered to be in the same group.
                        If None, then a layer mask is constructed which assigns
                        each neuron within the layer as a separate group, which
                        is ablated independently.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer's inputs, otherwise it will be computed with respect
                        to layer's outputs.
                        Note that currently it is assumed that either the input
                        or the output of the layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            perturbations_per_eval (int, optional): Allows ablation of multiple
                        neuron (groups) to be processed simultaneously in one
                        call to forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        Default: 1

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution of each neuron in given layer input or
                        output. Attributions will always be the same size as
                        the input or output of the given layer, depending on
                        whether we attribute to the inputs or outputs
                        of the layer which is decided by the input flag
                        `attribute_to_layer_input`
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.


        Examples::

        >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
        >>> # and returns an Nx3 tensor of class probabilities.
        >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
        >>> # and the output of this layer has dimensions Nx12x3x3.
        >>> net = SimpleClassifier()
        >>> # Generating random input with size 2 x 4 x 4
        >>> input = torch.randn(2, 4, 4)
        >>> # Defining LayerFeatureAblation interpreter
        >>> ablator = LayerFeatureAblation(net, net.conv1)
        >>> # Computes ablation attribution, ablating each of the 108
        >>> # neurons independently.
        >>> attr = ablator.attribute(input, target=1)

        >>> # Alternatively, we may want to ablate neurons in groups, e.g.
        >>> # grouping all the layer outputs in the same row.
        >>> # This can be done by creating a layer mask as follows, which
        >>> # defines the groups of layer inputs / outouts, e.g.:
        >>> # +---+---+---+
        >>> # | 0 | 0 | 0 |
        >>> # +---+---+---+
        >>> # | 1 | 1 | 1 |
        >>> # +---+---+---+
        >>> # | 2 | 2 | 2 |
        >>> # +---+---+---+
        >>> # With this mask, all the 36 neurons in a row / channel are ablated
        >>> # simultaneously, and the attribution for each neuron in the same
        >>> # group (0 - 2) per example are the same.
        >>> # The attributions can be calculated as follows:
        >>> # layer mask has dimensions 1 x 3 x 3
        >>> layer_mask = torch.tensor([[[0,0,0],[1,1,1],
        >>>                             [2,2,2]]])
        >>> attr = ablator.attribute(input, target=1,
        >>>                          layer_mask=layer_mask)
        """

        def layer_forward_func(*args: Any) -> Union[Tensor]:
            r"""
            Args:
                args (Any): Tensors comprising the layer input and the original
                    inputs, and an int representing the length of the layer input
            """
            layer_length: int = args[-1]
            layer_input: Tuple[Tensor, ...] = args[:layer_length]
            original_inputs: Tuple[Tensor, ...] = args[layer_length:-1]

            device_ids = self.device_ids
            if device_ids is None:
                device_ids = getattr(self.forward_func, "device_ids", None)

            all_layer_inputs: Dict[torch.device, Tuple[Tensor, ...]] = {}
            if device_ids is not None:
                scattered_layer_input = scatter(layer_input, target_gpus=device_ids)
                for device_tensors in scattered_layer_input:
                    all_layer_inputs[device_tensors[0].device] = device_tensors
            else:
                all_layer_inputs[layer_input[0].device] = layer_input

            def forward_hook(
                module: Module,
                inp: Union[None, Tensor, Tuple[Tensor, ...]],
                out: Union[None, Tensor, Tuple[Tensor, ...]] = None,
            ) -> Union[Tensor, Tuple[Tensor, ...]]:
                device = _extract_device(module, inp, out)
                is_layer_tuple = (
                    isinstance(out, tuple)
                    if out is not None
                    else isinstance(inp, tuple)
                )
                if device not in all_layer_inputs:
                    raise AssertionError(
                        "Layer input not placed on appropriate "
                        "device. If using a DataParallel model, either provide the "
                        "DataParallel model as forward_func or provide device ids"
                        " to the constructor."
                    )
                if not is_layer_tuple:
                    return all_layer_inputs[device][0]
                return all_layer_inputs[device]

            hook = None
            try:
                if attribute_to_layer_input:
                    hook = self.layer.register_forward_pre_hook(forward_hook)
                else:
                    hook = self.layer.register_forward_hook(forward_hook)
                eval = _run_forward(self.forward_func, original_inputs, target=target)
            finally:
                if hook is not None:
                    hook.remove()

            # _run_forward may return future of Tensor,
            # but we don't support it here now
            # And it will fail before here.
            return cast(Tensor, eval)

        with torch.no_grad():
            inputs = _format_tensor_into_tuples(inputs)
            additional_forward_args = _format_additional_forward_args(
                additional_forward_args
            )
            layer_eval = _forward_layer_eval(
                self.forward_func,
                inputs,
                self.layer,
                additional_forward_args,
                device_ids=self.device_ids,
                attribute_to_layer_input=attribute_to_layer_input,
            )
            layer_eval_len = (len(layer_eval),)
            all_inputs = (
                (inputs + additional_forward_args + layer_eval_len)
                if additional_forward_args is not None
                else inputs + layer_eval_len
            )

            ablator = self.attributor(layer_forward_func)

            layer_attribs = ablator.attribute.__wrapped__(
                ablator,  # self
                layer_eval,
                baselines=layer_baselines,
                additional_forward_args=all_inputs,
                feature_mask=layer_mask,
                perturbations_per_eval=perturbations_per_eval,
            )
            _attr = _format_output(len(layer_attribs) > 1, layer_attribs)
        return _attr

    @property
    def attributor(self) -> Type[FeatureAblation]:
        return FeatureAblation
