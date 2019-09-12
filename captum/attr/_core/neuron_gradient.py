#!/usr/bin/env python3
from .._utils.attribution import NeuronAttribution
from .._utils.common import (
    _forward_layer_eval,
    _extend_index_list,
    format_input,
    _format_additional_forward_args,
    _format_attributions,
)
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements

import torch


class NeuronGradient(NeuronAttribution):
    def __init__(self, forward_func, layer):
        r"""
        Args

            forward_func (callable):  The forward function of the model or any
                          modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                          Neuron index in the attribute method refers to a particular
                          neuron in the output of this layer. Currently, only
                          layers which output a single tensor are supported.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not neccesary to provide this argument.
        """
        super().__init__(forward_func, layer)

    def attribute(self, inputs, neuron_index, additional_forward_args=None):
        r"""
            Computes gradient with respect to input of a particular neuron in
            the given hidden layer.


            Args

                inputs (tensor or tuple of tensors):  Input for which neuron
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                neuron_index (int or tuple): Index of neuron in output of given
                              layer for which attribution is desired. Length of
                              this tuple must be one less than the number of
                              dimensions in the output of the given layer (since
                              dimension 0 corresponds to number of examples).
                              An integer may be provided instead of a tuple of
                              length 1.
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be a tuple containing tensors or
                            any arbitrary python types. These arguments are provided to
                            forward_func in order following the arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None

            Return

                attributions (tensor or tuple of tensors): Gradients of
                            particular neuron with respect to each input feature.
                            Attributions will always be the same size as the provided
                            inputs, with each value providing the attribution of the
                            corresponding input index.
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
                >>> # Computes neuron gradient for neuron with
                >>> # index (4,1,2).
                >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """
        is_inputs_tuple = isinstance(inputs, tuple)
        inputs = format_input(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        gradient_mask = apply_gradient_requirements(inputs)

        layer_out = _forward_layer_eval(
            self.forward_func, inputs, self.layer, additional_forward_args
        )
        indices = _extend_index_list(inputs[0].shape[0], neuron_index)
        with torch.autograd.set_grad_enabled(True):
            input_grads = torch.autograd.grad(
                [layer_out[index] for index in indices], inputs
            )

        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, input_grads)
