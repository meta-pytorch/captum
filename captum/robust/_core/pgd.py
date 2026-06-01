#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Any, Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from captum._utils.common import _format_output, _format_tensor_into_tuples, _is_tuple
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.log import log_usage
from captum.robust._core.fgsm import FGSM
from captum.robust._core.perturbation import Perturbation
from torch import Tensor


class PGD(Perturbation):
    r"""
    Projected Gradient Descent is an iterative version of the one-step attack
    FGSM that can generate adversarial examples. It takes multiple gradient
    steps to search for an adversarial perturbation within the desired
    neighbor ball around the original inputs. In a non-targeted attack, the
    formulation is::

        x_0 = x
        x_(t+1) = Clip_r(x_t + alpha * sign(gradient of L(theta, x, t)))

    where Clip denotes the function that projects its argument to the r-neighbor
    ball around x so that the perturbation will be bounded. Alpha is the step
    size. L(theta, x, y) is the model's loss function with respect to model
    parameters, inputs and targets.
    In a targeted attack, the formulation is similar::

        x_0 = x
        x_(t+1) = Clip_r(x_t - alpha * sign(gradient of L(theta, x, t)))

    More details on Projected Gradient Descent can be found in the original
    paper: https://arxiv.org/abs/1706.06083
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
                        model and labels, and return the loss for each input tensor.
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
        """
        super().__init__()
        self.forward_func = forward_func
        self.fgsm = FGSM(forward_func, loss_func)
        # pyre-fixme[4]: Attribute must be annotated.
        self.bound = lambda x: torch.clamp(x, min=lower_bound, max=upper_bound)

    @log_usage(part_of_slo=False)
    def perturb(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        radius: float,
        step_size: float,
        step_num: int,
        target: Any,
        additional_forward_args: Optional[object] = None,
        targeted: bool = False,
        random_start: bool = False,
        norm: str = "Linf",
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
            radius (float): Radius of the neighbor ball centered around inputs.
                        The perturbation should be within this range.
            step_size (float): Step size of each gradient step.
            step_num (int): Step numbers. It usually guarantees that the perturbation
                        can reach the border.
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
                        Default: ``None``
            targeted (bool, optional): If attack should be targeted.
                        Default: ``False``
            random_start (bool, optional): If a random initialization is added to
                        inputs. Default: ``False``
            norm (str, optional): Specifies the norm to calculate distance from
                        original inputs: ``Linf`` | ``L2``.
                        Default: ``Linf``
            mask (Tensor or tuple[Tensor, ...], optional): mask of zeroes and ones
                        that defines which elements within the input tensor(s) are
                        perturbed. This mask must have the same shape and
                        dimensionality as the inputs. If this argument is not
                        provided, all elements are perturbed.
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

        def _clip(inputs: Tensor, outputs: Tensor) -> Tensor:
            diff = outputs - inputs
            if norm == "Linf":
                return inputs + torch.clamp(diff, -radius, radius)
            elif norm == "L2":
                return inputs + torch.renorm(diff, 2, 0, radius)
            else:
                raise AssertionError("Norm constraint must be L2 or Linf.")

        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs = _format_tensor_into_tuples(inputs)
        # pyre-fixme[9]: formatted_masks has type `Union[typing.Tuple[int, ...],
        #  typing.Tuple[Tensor, ...]]`; used as `Tuple[Union[int, Tensor], ...]`.
        formatted_masks: Union[Tuple[int, ...], Tuple[Tensor, ...]] = (
            _format_tensor_into_tuples(mask)
            if (mask is not None)
            else (1,) * len(formatted_inputs)
        )
        perturbed_inputs = formatted_inputs
        if random_start:
            perturbed_inputs = tuple(
                self.bound(
                    self._random_point(
                        formatted_inputs[i], radius, norm, formatted_masks[i]
                    )
                )
                for i in range(len(formatted_inputs))
            )
        for _i in range(step_num):
            perturbed_inputs = self.fgsm.perturb(
                perturbed_inputs,
                step_size,
                target,
                additional_forward_args,
                targeted,
                formatted_masks,
            )
            perturbed_inputs = tuple(
                _clip(formatted_inputs[j], perturbed_inputs[j])
                for j in range(len(perturbed_inputs))
            )
            # Detaching inputs to avoid dependency of gradient between steps
            perturbed_inputs = tuple(
                self.bound(perturbed_inputs[j]).detach()
                for j in range(len(perturbed_inputs))
            )
        # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got
        #  `Tuple[Tensor, ...]`.
        return _format_output(is_inputs_tuple, perturbed_inputs)

    def _random_point(
        self, center: Tensor, radius: float, norm: str, mask: Union[Tensor, int]
    ) -> Tensor:
        r"""
        A helper function that returns a uniform random point within the ball
        with the given center and radius. Norm should be either L2 or Linf.
        """
        if norm == "L2":
            u = torch.randn_like(center)
            unit_u = F.normalize(u.view(u.size(0), -1)).view(u.size())
            d = torch.numel(center[0])
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `float`.
            r = (torch.rand(u.size(0)) ** (1.0 / d)) * radius
            # pyre-fixme[16]: `float` has no attribute `__getitem__`.
            # pyre-fixme[16]: `float` has no attribute `dim`.
            r = r[(...,) + (None,) * (r.dim() - 1)]
            x = r * unit_u
            return center + (x * mask)
        elif norm == "Linf":
            x = torch.rand_like(center) * radius * 2 - radius
            return center + (x * mask)
        else:
            raise AssertionError("Norm constraint must be L2 or Linf.")
