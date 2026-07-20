#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from typing import cast, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from captum._utils.common import (
    _format_output,
    _format_tensor_into_tuples,
    _is_tuple,
    _register_backward_hook,
)
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import GradientAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


class ModifiedReluGradientAttribution(GradientAttribution):
    def __init__(self, model: Module, use_relu_grad_output: bool = False) -> None:
        r"""
        Args:

            model (nn.Module): The reference to PyTorch model instance.
        """
        GradientAttribution.__init__(self, model)
        self.model = model
        self.backward_hooks: List[RemovableHandle] = []
        self.use_relu_grad_output = use_relu_grad_output
        assert isinstance(self.model, torch.nn.Module), (
            "Given model must be an instance of torch.nn.Module to properly hook"
            " ReLU layers."
        )

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Computes attribution by overriding relu gradients. Based on constructor
        flag use_relu_grad_output, performs either GuidedBackpropagation if False
        and Deconvolution if True. This class is the parent class of both these
        methods, more information on usage can be found in the docstrings for each
        implementing class.
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        inputs_tuple = _format_tensor_into_tuples(inputs)
        gradient_mask = apply_gradient_requirements(inputs_tuple)

        # set hooks for overriding ReLU gradients
        warnings.warn(
            "Setting backward hooks on ReLU activations."
            "The hooks will be removed after the attribution is finished",
            stacklevel=1,
        )
        try:
            self.model.apply(self._register_hooks)

            gradients = self.gradient_func(
                self.forward_func, inputs_tuple, target, additional_forward_args
            )
        finally:
            self._remove_hooks()

        undo_gradient_requirements(inputs_tuple, gradient_mask)
        return cast(
            TensorOrTupleOfTensorsGeneric, _format_output(is_inputs_tuple, gradients)
        )

    def attribute_future(self) -> None:
        r"""
        This method is not implemented for ModifiedReluGradientAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for ModifiedReluGradientAttribution"
        )

    def _register_hooks(self, module: Module) -> None:
        if isinstance(module, torch.nn.ReLU):
            hooks = _register_backward_hook(module, self._backward_hook, self)
            self.backward_hooks.extend(hooks)

    def _backward_hook(
        self,
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ) -> Union[Tuple[Tensor], Tensor]:
        to_override_grads = grad_output if self.use_relu_grad_output else grad_input
        if isinstance(to_override_grads, tuple):
            # pyrefly: ignore [bad-return]
            return tuple(
                F.relu(to_override_grad) for to_override_grad in to_override_grads  # type: ignore # noqa: E501 line too long
            )
        else:
            return F.relu(to_override_grads)

    def _remove_hooks(self) -> None:
        for hook in self.backward_hooks:
            hook.remove()


class GuidedBackprop(ModifiedReluGradientAttribution):
    r"""
    Computes attribution using guided backpropagation. Guided backpropagation
    computes the gradient of the target output with respect to the input,
    but gradients of ReLU functions are overridden so that only
    non-negative gradients are backpropagated.

    More details regarding the guided backpropagation algorithm can be found
    in the original paper here:
    https://arxiv.org/abs/1412.6806

    Warning: Ensure that all ReLU operations in the forward function of the
    given model are performed using a module (nn.module.ReLU).
    If nn.functional.ReLU is used, gradients are not overridden appropriately.
    """

    def __init__(self, model: Module) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        ModifiedReluGradientAttribution.__init__(
            self, model, use_relu_grad_output=False
        )

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which
                        attributions are computed. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
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
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The guided backprop gradients with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> gbp = GuidedBackprop(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes Guided Backprop attribution scores for class 3.
            >>> attribution = gbp.attribute(input, target=3)
        """
        return super().attribute.__wrapped__(
            self, inputs, target, additional_forward_args
        )


class Deconvolution(ModifiedReluGradientAttribution):
    r"""
    Computes attribution using deconvolution. Deconvolution
    computes the gradient of the target output with respect to the input,
    but gradients of ReLU functions are overridden so that the gradient
    of the ReLU input is simply computed taking ReLU of the output gradient,
    essentially only propagating non-negative gradients (without
    dependence on the sign of the ReLU input).

    More details regarding the deconvolution algorithm can be found
    in these papers:
    https://arxiv.org/abs/1311.2901
    https://link.springer.com/chapter/10.1007/978-3-319-46466-4_8

    Warning: Ensure that all ReLU operations in the forward function of the
    given model are performed using a module (nn.module.ReLU).
    If nn.functional.ReLU is used, gradients are not overridden appropriately.
    """

    def __init__(self, model: Module) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        ModifiedReluGradientAttribution.__init__(self, model, use_relu_grad_output=True)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which
                        attributions are computed. If model takes a single
                        tensor as input, a single input tensor should be provided.
                        If model takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
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
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The deconvolution attributions with respect to each
                        input feature. Attributions will always
                        be the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> deconv = Deconvolution(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes Deconvolution attribution scores for class 3.
            >>> attribution = deconv.attribute(input, target=3)
        """
        return super().attribute.__wrapped__(
            self, inputs, target, additional_forward_args
        )
