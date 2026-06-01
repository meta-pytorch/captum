# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from collections import defaultdict
from enum import Enum
from typing import cast, DefaultDict, Iterable, List, Optional, Tuple, Union

import torch
from captum._utils.common import _format_tensor_into_tuples, _register_backward_hook
from torch import Tensor
from torch.nn import Module


def _reset_sample_grads(module: Module) -> None:
    module.weight.sample_grad = 0  # type: ignore
    if module.bias is not None:
        module.bias.sample_grad = 0  # type: ignore


def linear_param_grads(
    module: Module, activation: Tensor, gradient_out: Tensor, reset: bool = False
) -> None:
    r"""
    Computes parameter gradients per sample for nn.Linear module, given module
    input activations and output gradients.

    Gradients are accumulated in the sample_grad attribute of each parameter
    (weight and bias). If reset = True, any current sample_grad values are reset,
    otherwise computed gradients are accumulated and added to the existing
    stored gradients.

    Inputs with more than 2 dimensions are only supported with torch 1.8 or later
    """
    if reset:
        _reset_sample_grads(module)

    module.weight.sample_grad += torch.einsum(  # type: ignore
        "n...i,n...j->nij", gradient_out, activation
    )
    if module.bias is not None:
        module.bias.sample_grad += torch.einsum(  # type: ignore
            "n...i->ni", gradient_out
        )


def conv2d_param_grads(
    module: Module, activation: Tensor, gradient_out: Tensor, reset: bool = False
) -> None:
    r"""
    Computes parameter gradients per sample for nn.Conv2d module, given module
    input activations and output gradients.

    nn.Conv2d modules with padding set to a string option ('same' or 'valid') are
    currently unsupported.

    Gradients are accumulated in the sample_grad attribute of each parameter
    (weight and bias). If reset = True, any current sample_grad values are reset,
    otherwise computed gradients are accumulated and added to the existing
    stored gradients.
    """
    if reset:
        _reset_sample_grads(module)

    # pyre-fixme[22]: The cast is redundant.
    batch_size = cast(int, activation.shape[0])
    unfolded_act = torch.nn.functional.unfold(
        activation,
        cast(Union[int, Tuple[int, ...]], module.kernel_size),
        dilation=cast(Union[int, Tuple[int, ...]], module.dilation),
        padding=cast(Union[int, Tuple[int, ...]], module.padding),
        stride=cast(Union[int, Tuple[int, ...]], module.stride),
    )
    reshaped_grad = gradient_out.reshape(batch_size, -1, unfolded_act.shape[-1])
    grad1 = torch.einsum("ijk,ilk->ijl", reshaped_grad, unfolded_act)
    shape = [batch_size] + list(cast(Iterable[int], module.weight.shape))
    module.weight.sample_grad += grad1.reshape(shape)  # type: ignore
    if module.bias is not None:
        module.bias.sample_grad += torch.sum(reshaped_grad, dim=2)  # type: ignore


SUPPORTED_MODULES = {
    torch.nn.Conv2d: conv2d_param_grads,
    torch.nn.Linear: linear_param_grads,
}


class LossMode(Enum):
    SUM = 0
    MEAN = 1


class SampleGradientWrapper:
    r"""
    Wrapper which allows computing sample-wise gradients in a single backward pass.

    This is accomplished by adding hooks to capture activations and output
    gradients for supported modules, and using these activations and gradients
    to compute the parameter gradients per-sample.

    Currently, only nn.Linear and nn.Conv2d modules are supported.

    Similar reference implementations of sample-based gradients include:
    - https://github.com/cybertronai/autograd-hacks
    - https://github.com/pytorch/opacus/tree/main/opacus/grad_sample
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, model, layer_modules: Optional[List[Module]] = None) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.model = model
        self.hooks_added = False
        self.activation_dict: DefaultDict[Module, List[Tensor]] = defaultdict(list)
        self.gradient_dict: DefaultDict[Module, List[Tensor]] = defaultdict(list)
        self.forward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.backward_hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.layer_modules: Optional[List[Module]] = layer_modules

    def add_hooks(self) -> None:
        self.hooks_added = True
        self.model.apply(self._register_module_hooks)

    def _register_module_hooks(self, module: torch.nn.Module) -> None:
        if (self.layer_modules is None or module in self.layer_modules) and isinstance(
            module, tuple(SUPPORTED_MODULES.keys())
        ):
            self.forward_hooks.append(
                module.register_forward_hook(self._forward_hook_fn)
            )
            self.backward_hooks.extend(
                _register_backward_hook(module, self._backward_hook_fn, None)
            )

    def _forward_hook_fn(
        self,
        module: Module,
        module_input: Union[Tensor, Tuple[Tensor, ...]],
        module_output: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        inp_tuple = _format_tensor_into_tuples(module_input)
        self.activation_dict[module].append(inp_tuple[0].clone().detach())

    def _backward_hook_fn(
        self,
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:
        grad_output_tuple = _format_tensor_into_tuples(grad_output)
        self.gradient_dict[module].append(grad_output_tuple[0].clone().detach())

    def remove_hooks(self) -> None:
        self.hooks_added = False

        for hook in self.forward_hooks:
            hook.remove()

        for hook in self.backward_hooks:
            hook.remove()

        self.forward_hooks = []
        self.backward_hooks = []

    def _reset(self) -> None:
        self.activation_dict = defaultdict(list)
        self.gradient_dict = defaultdict(list)

    def compute_param_sample_gradients(
        self, loss_blob: Tensor, loss_mode: str = "mean"
    ) -> None:
        assert (
            loss_mode.upper() in LossMode.__members__
        ), f"Provided loss mode {loss_mode} is not valid"
        mode = LossMode[loss_mode.upper()]

        self.model.zero_grad()
        loss_blob.backward(gradient=torch.ones_like(loss_blob))

        for module in self.gradient_dict:
            # pyre-fixme[6]: For 1st argument expected `Type[Union[Conv2d, Linear]]`
            #  but got `Type[Module]`.
            sample_grad_fn = SUPPORTED_MODULES[type(module)]
            activations = self.activation_dict[module]
            gradients = self.gradient_dict[module]
            assert len(activations) == len(gradients), (
                "Number of saved activations do not match number of saved gradients."
                " This may occur if multiple forward passes are run without calling"
                " reset or computing param gradients."
            )
            # Reversing grads since when a module is used multiple times,
            # the activations will be aligned with the reverse order of the gradients,
            # since the order is reversed in backprop.
            for i, (act, grad) in enumerate(
                zip(activations, list(reversed(gradients)))
            ):
                mult = 1 if mode is LossMode.SUM else act.shape[0]
                sample_grad_fn(module, act, grad * mult, reset=(i == 0))
        self._reset()
