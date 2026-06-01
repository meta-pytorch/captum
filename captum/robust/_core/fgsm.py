#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Callable, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _select_targets,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    undo_gradient_requirements,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from captum.robust._core.perturbation import Perturbation
from torch import Tensor


class FGSM(Perturbation):
    r"""
    Fast Gradient Sign Method is a one-step method that can generate
    adversarial examples.

    For non-targeted attack, the formulation is::

        x' = x + epsilon * sign(gradient of L(theta, x, y))

    For targeted attack on t, the formulation is::

        x' = x - epsilon * sign(gradient of L(theta, x, t))

    ``L(theta, x, y)`` is the model's loss function with respect to model
    parameters, inputs and labels.

    More details on Fast Gradient Sign Method can be found in the original
    paper: https://arxiv.org/abs/1412.6572
    """

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        forward_func: Callable,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_func: Optional[Callable] = None,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
    ) -> None:
        r"""
        Args:
            forward_func (Callable): The pytorch model for which the attack is
                        computed.
            loss_func (Callable, optional): Loss function of which the gradient
                        computed. The loss function should take in outputs of the
                        model and labels, and return a loss tensor.
                        The default loss function is negative log.
            lower_bound (float, optional): Lower bound of input values.
                        Default: ``float("-inf")``
            upper_bound (float, optional): Upper bound of input values.
                        e.g. image pixels must be in the range 0-255
                        Default: ``float("inf")``

        Attributes:
            bound (Callable): A function that bounds the input values based on
                        given lower_bound and upper_bound. Can be overwritten for
                        custom use cases if necessary.
            zero_thresh (float): The threshold below which gradient will be treated
                        as zero. Can be modified for custom use cases if necessary.
        """
        super().__init__()
        self.forward_func = forward_func
        self.loss_func = loss_func
        # pyre-fixme[4]: Attribute must be annotated.
        self.bound = lambda x: torch.clamp(x, min=lower_bound, max=upper_bound)
        # pyre-fixme[4]: Attribute must be annotated.
        self.zero_thresh = 10**-6

    @log_usage(part_of_slo=False)
    def perturb(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        epsilon: float,
        target: Any,
        additional_forward_args: Optional[object] = None,
        targeted: bool = False,
        mask: Optional[TensorOrTupleOfTensorsGeneric] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method computes and returns the perturbed input for each input tensor.
        It supports both targeted and non-targeted attacks.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which adversarial
                        attack is computed. It can be provided as a single
                        tensor or a tuple of multiple tensors. If multiple
                        input tensors are provided, the batch sizes must be
                        aligned across all tensors.
            epsilon (float): Step size of perturbation.
            target (Any): True labels of inputs if non-targeted attack is
                        desired. Target class of inputs if targeted attack
                        is desired. Target will be passed to the loss function
                        to compute loss, so the type needs to match the
                        argument type of the loss function.

                        If using the default negative log as loss function,
                        labels should be of type int, tuple, tensor or list.
                        For general 2D outputs, labels can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the label for the corresponding example.

                        For outputs with > 2 dimensions, labels can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This label index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          label for the corresponding example.

            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. These arguments are provided to
                        forward_func in order following the arguments in inputs.
                        Default: None.
            targeted (bool, optional): If attack should be targeted.
                        Default: False.
            mask (Tensor or tuple[Tensor, ...], optional): mask of zeroes and ones
                        that defines which elements within the input tensor(s) are
                        perturbed. This mask must have the same shape and
                        dimensionality as the inputs. If this argument is not
                        provided, all elements will be perturbed.
                        Default: None.


        Returns:

            - **perturbed inputs** (*Tensor* or *tuple[Tensor, ...]*):
                        Perturbed input for each
                        input tensor. The perturbed inputs have the same shape and
                        dimensionality as the inputs.
                        If a single tensor is provided as inputs, a single tensor
                        is returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
        """
        is_inputs_tuple = _is_tuple(inputs)
        # pyre-fixme[35]: Target cannot be annotated.
        inputs: Tuple[Tensor, ...] = _format_tensor_into_tuples(inputs)
        # pyre-fixme[9]: masks has type `Union[typing.Tuple[int, ...],
        #  typing.Tuple[Tensor, ...]]`; used as `Tuple[Union[int, Tensor], ...]`.
        masks: Union[Tuple[int, ...], Tuple[Tensor, ...]] = (
            _format_tensor_into_tuples(mask)
            if (mask is not None)
            else (1,) * len(inputs)
        )
        gradient_mask = apply_gradient_requirements(inputs)

        def _forward_with_loss() -> Tensor:
            additional_inputs = _format_additional_forward_args(additional_forward_args)
            outputs = self.forward_func(  # type: ignore
                *(
                    (*inputs, *additional_inputs)  # type: ignore
                    if additional_inputs is not None
                    else inputs
                )
            )
            if self.loss_func is not None:
                return self.loss_func(outputs, target)
            else:
                loss = -torch.log(outputs)
                return _select_targets(loss, target)

        grads = compute_gradients(_forward_with_loss, inputs)
        undo_gradient_requirements(inputs, gradient_mask)
        perturbed_inputs = self._perturb(inputs, grads, epsilon, targeted, masks)
        perturbed_inputs = tuple(
            self.bound(perturbed_inputs[i]) for i in range(len(perturbed_inputs))
        )
        # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got
        #  `Tuple[Tensor, ...]`.
        return _format_output(is_inputs_tuple, perturbed_inputs)

    def _perturb(
        self,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        inputs: Tuple,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        grads: Tuple,
        epsilon: float,
        targeted: bool,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        masks: Tuple,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    ) -> Tuple:
        r"""
        A helper function to calculate the perturbed inputs given original
        inputs, gradient of loss function and epsilon. The calculation is
        different for targeted v.s. non-targeted as described above.
        """
        multiplier = -1 if targeted else 1
        inputs = tuple(
            torch.where(
                torch.abs(grad) > self.zero_thresh,
                inp + multiplier * epsilon * torch.sign(grad) * mask,
                inp,
            )
            for grad, inp, mask in zip(grads, inputs, masks)
        )
        return inputs
