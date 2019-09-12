#!/usr/bin/env python3

import torch

from .._utils.common import _format_attributions, format_input
from .._utils.attribution import GradientBasedAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


class Saliency(GradientBasedAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (callable): The forward function of the model or
                        any modification of it
        """
        super().__init__(forward_func)

    def attribute(self, inputs, target=None, abs=True, additional_forward_args=None):
        r""""
        A baseline approach for computing input attribution. It returns
        the gradients with respect to inputs. If `abs` is set to True, which is
        the default, the absolute value of the gradients is returned.

        More details about the approach can be found in the following paper:
            https://arxiv.org/pdf/1312.6034.pdf

        Args:

                inputs (tensor or tuple of tensors):  Input for which integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            mutliple input tensors are provided, the examples must
                            be aligned appropriately.
                target (int, optional):  Output index for which gradient is computed
                            (for classification cases, this is the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary. (Note: Tuples for multi
                            -dimensional output indices will be supported soon.)
                            Default: None
                abs (bool, optional): Returns absolute value of gradients if set
                            to True, otherwise returns the (signed) gradients if
                            False.
                            Defalut: True
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It must be a tuple containing tensors or
                            any arbitrary python types. These arguments are provided to
                            forward_func in order following the arguments in inputs.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None

        Return:

                attributions (tensor or tuple of tensors): The gradients with
                            respect to each input feature. Attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # Generating random input with size 2x3x3x32
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Defining Saliency interpreter
            >>> saliency = Saliency(net)
            >>> # Computes saliency maps for class 3.
            >>> attribution = saliency.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # No need to format additional_forward_args here.
        # They are being formated in the `_run_forward` function in `common.py`
        gradients = self.gradient_func(
            self.forward_func, inputs, target, additional_forward_args
        )
        if abs:
            attributions = tuple(torch.abs(gradient) for gradient in gradients)
        else:
            attributions = gradients
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, attributions)
