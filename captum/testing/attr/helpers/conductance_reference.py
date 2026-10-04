#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable, cast, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from captum._utils.gradient import (
    apply_gradient_requirements,
    undo_gradient_requirements,
)
from captum._utils.typing import ModuleOrModuleList
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import LayerAttribution
from captum.attr._utils.common import _reshape_and_sum
from torch import Tensor
from torch.utils.hooks import RemovableHandle

"""
Note: This implementation of conductance follows the procedure described in the original
paper exactly (https://arxiv.org/abs/1805.12233), computing gradients of output with
respect to hidden neurons and each hidden neuron with respect to the input and summing
appropriately. Computing the gradient of each neuron with respect to the input is
not necessary to just compute the conductance of a given layer, so the main
implementationof conductance does not use this approach in order to compute layer
conductance more efficiently (https://arxiv.org/pdf/1807.09946.pdf).
This implementation is used only for testing to verify that the output matches
that of the main implementation.
"""


class ConductanceReference(LayerAttribution):
    def __init__(
        self, forward_func: Callable[..., Tensor], layer: ModuleOrModuleList
    ) -> None:
        r"""
        Args

            forward_func:  The forward function of the model or any modification of it
            layer: Layer for which output attributions are computed.
                   Output size of attribute matches that of layer output.
        """
        super().__init__(forward_func, layer)

    def _conductance_grads(
        self,
        forward_fn: Callable[..., Tensor],
        input: Tensor,
        target_ind: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, int]:
        with torch.autograd.set_grad_enabled(True):
            # Set a forward hook on specified module and run forward pass to
            # get output tensor size.
            saved_tensor = None

            def forward_hook(module: nn.Module, inp: Tensor, out: Tensor) -> None:
                nonlocal saved_tensor
                saved_tensor = out

            hook = self.layer.register_forward_hook(forward_hook)
            output = forward_fn(input)

            # Compute layer output tensor dimensions and total number of units.
            # The hidden layer tensor is assumed to have dimension (num_hidden, ...)
            # where the product of the dimensions >= 1 correspond to the total
            # number of hidden neurons in the layer.
            layer_size: Tuple[int, ...] = tuple(cast(Tensor, saved_tensor).size())[1:]
            layer_units: int = int(np.prod(layer_size))

            # Remove unnecessary forward hook.
            hook.remove()

            # Backward hook function to override gradients in order to obtain
            # just the gradient of each hidden unit with respect to input.
            saved_grads = None

            def backward_hook(grads: Tensor) -> Tensor:
                nonlocal saved_grads
                saved_grads = grads
                zero_mat = torch.zeros((1,) + layer_size)
                scatter_indices = torch.arange(0, layer_units).view_as(zero_mat)
                # Creates matrix with each layer containing a single unit with
                # value 1 and remaining zeros, which will provide gradients
                # with respect to each unit independently.
                to_return = torch.zeros((layer_units,) + layer_size).scatter(
                    0, scatter_indices, 1
                )
                to_repeat = [1] * len(to_return.shape)
                to_repeat[0] = grads.shape[0] // to_return.shape[0]
                expanded = to_return.repeat(to_repeat)
                return expanded

            # Create a forward hook in order to attach backward hook to appropriate
            # tensor. Save backward hook in order to remove hook appropriately.
            back_hook = None

            def forward_hook_register_back(
                module: nn.Module, inp: Tensor, out: Tensor
            ) -> None:
                nonlocal back_hook
                back_hook = out.register_hook(backward_hook)

            hook = self.layer.register_forward_hook(forward_hook_register_back)

            # Expand input to include layer_units copies of each input.
            # This allows obtaining gradient with respect to each hidden unit
            # in one pass.
            expanded_input = torch.repeat_interleave(input, layer_units, dim=0)
            output = forward_fn(expanded_input)
            hook.remove()
            output = output[:, target_ind] if target_ind is not None else output
            input_grads = torch.autograd.grad(torch.unbind(output), expanded_input)

            # Remove backwards hook
            cast(RemovableHandle, back_hook).remove()

            # Remove duplicates in gradient with respect to hidden layer,
            # choose one for each layer_units indices.
            output_mid_grads = torch.index_select(
                cast(Tensor, saved_grads),
                0,
                torch.tensor(range(0, input_grads[0].shape[0], layer_units)),
            )
        return input_grads[0], output_mid_grads, layer_units

    def attribute(
        self,
        inputs: Tensor,
        baselines: Union[None, int, Tensor] = None,
        target: Optional[Tensor] = None,
        n_steps: int = 500,
        method: str = "riemann_trapezoid",
    ) -> Tensor:
        r"""
        Computes conductance using gradients along the path, applying
        riemann's method or gauss-legendre.
        The details of the approach can be found here:
        https://arxiv.org/abs/1805.12233

        Args

            inputs:     A single high dimensional input tensor, in which
                        dimension 0 corresponds to number of examples.
            baselines:   A single high dimensional baseline tensor,
                        which has the same shape as the input
            target:     Predicted class index. This is necessary only for
                        classification use cases
            n_steps:    The number of steps used by the approximation method
            method:     Method for integral approximation, one of `riemann_right`,
                        `riemann_middle`, `riemann_trapezoid` or `gausslegendre`

        Return

            attributions: Total conductance with respect to each neuron in
                          output of given layer
        """
        if baselines is None:
            baselines = 0
        gradient_mask = apply_gradient_requirements((inputs,))
        # retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # compute scaled inputs from baseline to final input.
        scaled_features = torch.cat(
            [baselines + alpha * (inputs - baselines) for alpha in alphas],
            dim=0,
        )

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer, gradient of hidden layer with respect to input,
        # and number of hidden units.
        input_gradients, mid_layer_gradients, hidden_units = self._conductance_grads(
            self.forward_func, scaled_features, target
        )
        # Multiply gradient of hidden layer with respect to input by input - baseline
        scaled_input_gradients = torch.repeat_interleave(
            inputs - baselines, hidden_units, dim=0
        )
        scaled_input_gradients = input_gradients * scaled_input_gradients.repeat(
            *([len(alphas)] + [1] * (len(scaled_input_gradients.shape) - 1))
        )

        # Sum gradients for each input neuron in order to have total
        # for each hidden unit and reshape to match hidden layer shape
        summed_input_grads = torch.sum(
            scaled_input_gradients, tuple(range(1, len(scaled_input_gradients.shape)))
        ).view_as(mid_layer_gradients)

        # Rescale gradients of hidden layer by by step size.
        scaled_grads = mid_layer_gradients.contiguous().view(
            n_steps, -1
        ) * torch.tensor(step_sizes).view(n_steps, 1).to(mid_layer_gradients.device)

        undo_gradient_requirements((inputs,), gradient_mask)

        # Element-wise mutliply gradient of output with respect to hidden layer
        # and summed gradients with respect to input (chain rule) and sum across
        # stepped inputs.
        return _reshape_and_sum(
            scaled_grads.view(mid_layer_gradients.shape) * summed_input_grads,
            n_steps,
            inputs.shape[0],
            tuple(mid_layer_gradients.shape[1:]),
        )
