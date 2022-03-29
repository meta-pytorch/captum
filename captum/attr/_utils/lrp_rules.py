#!/usr/bin/env python3

import copy
from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import torch

from ..._utils.common import _format_tensor_into_tuples
from ..._utils.typing import Module, Tensor, TensorOrTupleOfTensorsGeneric


class PropagationRule(ABC):
    """
    Base class for all propagation rule classes, also called Z-Rule.
    STABILITY_FACTOR is used to assure that no zero divison occurs.
    """

    STABILITY_FACTOR = 1e-9

    def forward_hook(
        self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor
    ) -> Tensor:
        """Register backward hooks on input and output
        tensors of linear layers in the model."""
        inputs = _format_tensor_into_tuples(inputs)
        self._has_single_input = len(inputs) == 1
        self._handle_input_hooks = []
        for input in inputs:
            if not hasattr(input, "hook_registered"):
                input_hook = self._create_backward_hook_input(input.data)
                self._handle_input_hooks.append(input.register_hook(input_hook))
                input.hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)
        return outputs.clone()

    @staticmethod
    def backward_hook_activation(
        module: Module,
        grad_input: Union[Tensor, Tuple[Tensor, ...]],
        grad_output: Union[Tensor, Tuple[Tensor, ...]],
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Backward hook to propagate relevance over non-linear activations."""
        if (
            isinstance(grad_input, tuple)
            and isinstance(grad_output, tuple)
            and len(grad_input) > len(grad_output)
        ):
            # Adds any additional elements of grad_input if applicable
            # This occurs when registering a backward hook on nn.Dropout
            # modules, which has an additional element of None in
            # grad_input
            return grad_output + grad_input[len(grad_output) :]
        return grad_output

    def _create_backward_hook_input(
        self, inputs: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            relevance = grad * inputs
            device = grad.device
            if self._has_single_input:
                self.relevance_input[device] = relevance.data
            else:
                self.relevance_input[device].append(relevance.data)
            return relevance

        return _backward_hook_input

    def _create_backward_hook_output(
        self, outputs: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> Tensor:
            sign = torch.sign(outputs)
            sign[sign == 0] = 1
            relevance = grad / (outputs + sign * self.STABILITY_FACTOR)
            self.relevance_output[grad.device] = grad.data
            return relevance

        return _backward_hook_output

    def forward_hook_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        """Save initial activations a_j before modules are changed"""
        device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        module.activations[device] = tuple(input.data for input in inputs)
        self._manipulate_weights(module, inputs, outputs)

    @abstractmethod
    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        raise NotImplementedError

    def forward_pre_hook_activations(self, module: Module, inputs: Tuple[Tensor, ...]):
        """Pass initial activations to graph generation pass"""
        device = inputs[0].device if isinstance(inputs, tuple) else inputs.device
        for input, activation in zip(inputs, module.activations[device]):
            input.data = activation
        return inputs


class EpsilonRule(PropagationRule):
    """
    Rule for relevance propagation using a small value of epsilon
    to avoid numerical instabilities and remove noise.

    Use for middle layers.

    Args:
        epsilon (integer, float): Value by which is added to the
        discriminator during propagation.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.STABILITY_FACTOR = epsilon

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        pass


class GammaRule(PropagationRule):
    """
    Gamma rule for relevance propagation, gives more importance to
    positive relevance.

    Use for lower layers.

    Args:
        gamma (float): The gamma parameter determines by how much
        the positive relevance is increased.
    """

    def __init__(self, gamma: float = 0.25, set_bias_to_zero: bool = False) -> None:
        self.gamma = gamma
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "weight"):
            module.weight.data = (
                module.weight.data + self.gamma * module.weight.data.clamp(min=0)
            )
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class ZPlusRule(PropagationRule):
    """
    Z^+ rule for relevance backpropagation closely related to
    Deep-Taylor Decomposition cf. https://doi.org/10.1016/j.patcog.2016.11.008.
    Only positive relevance is propagated, resulting in stable results,
    therefore recommended as the initial choice.

    Warning: Does not work for BatchNorm modules because weight and bias
    are defined differently.

    Use for lower layers.
    """

    def __init__(self, set_bias_to_zero: bool = False) -> None:
        self.set_bias_to_zero = set_bias_to_zero

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data.clamp(min=0)
        if self.set_bias_to_zero and hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)


class IdentityRule(EpsilonRule):
    """
    Identity rule for skipping layer manipulation and propagating the
    relevance over a layer. Only valid for modules with same dimensions for
    inputs and outputs.

    Can be used for BatchNorm2D.
    """

    def _create_backward_hook_output(
        self, outputs: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> None:
            self.relevance_output[grad.device] = grad.data

        return _backward_hook_output

    def _create_backward_hook_input(
        self, inputs: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            return self.relevance_output[grad.device]

        return _backward_hook_input


class WSquaredRule(PropagationRule):
    r"""
    W^2-rule for relevance propagation, cf. https://arxiv.org/pdf/1512.02479.
    Does not weight the relevance by the layers input.

    Use for input layer.
    """

    def __init__(self) -> None:
        self._input_shapes: Tuple[torch.Size, ...] = tuple()
        self._denominator = torch.Tensor()

    def forward_hook(
        self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor
    ) -> Tensor:
        self._compute_denominator(module, inputs)
        return super().forward_hook(module, inputs, outputs)

    def _compute_denominator(self, module: Module, inputs: Tuple[Tensor, ...]) -> None:
        input_shapes = tuple(x.shape[1:] for x in inputs)

        if input_shapes != self._input_shapes:
            self._input_shapes = input_shapes
            with torch.no_grad():
                self._denominator = module.forward(
                    *tuple(
                        torch.ones(input_shape).unsqueeze(dim=0)
                        for input_shape in self._input_shapes
                    )
                )
                self._denominator += self.STABILITY_FACTOR

    def _create_backward_hook_input(
        self, input_: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(
            grad: Tensor,
        ) -> None:
            pass

        return _backward_hook_input

    def _create_backward_hook_output(
        self, output: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> Tensor:
            relevance = grad / torch.cat(
                tuple(self._denominator for _ in range(output.shape[0]))
            )
            return relevance

        return _backward_hook_output

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
        if hasattr(module, "weight"):
            module.weight.data = module.weight.data ** 2


class FlatRule(WSquaredRule):
    r"""
    Flat rule for relevance backpropagation. This rule weights the relevance
    by the number of contributing input neurons.

    Use for lower layers.
    """

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                module.bias.data = torch.zeros_like(module.bias.data)
        if hasattr(module, "weight"):
            module.weight.data = torch.ones_like(module.weight.data)


class ZBoundRule(PropagationRule):
    r"""
    Z found rule for relevance backpropagation with bounded input values.

    Use for input layer for images.
    """

    def __init__(
        self,
        lower_bound: Union[int, float] = -1.0,
        upper_bound: Union[int, float] = 1.0,
        set_bias_to_zero: bool = True,
    ) -> None:
        """
        Args:
            lower_bound (Union[int, float], optional): Lower bound value for input
            features. Defaults to -1.0.
            upper_bound (Union[int, float], optional): Upper bound for input features.
            Defaults to 1.0.
            set_bias_to_zero (bool, optional): Parameter for setting bias to
            zero in relevance computations.
            Defaults to False.
        """
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.set_bias_to_zero = set_bias_to_zero

        self._lower_bound_tensor = torch.Tensor()
        self._upper_bound_tensor = torch.Tensor()

        self._module_pos: Module = None
        self._module_neg: Module = None
        self._bias_contrib: Tensor = None

        self._denominator_bound_contribution = torch.Tensor()
        self._input_shapes: Tuple[torch.Size, ...] = tuple()

    def forward_hook(
        self, module: Module, inputs: Tuple[Tensor, ...], outputs: Tensor
    ) -> Tensor:
        r"""
        Register backward hooks on input and output
        tensors of linear layers in the model.
        """
        if not hasattr(module, "weight"):
            raise RuntimeError(
                f"{self.__class__.__name__} assigned to module without weights:"
                + "{module}. This rule only supports modules with weight."
            )
        inputs = _format_tensor_into_tuples(inputs)
        self._handle_input_hooks = list()
        for input_index, input_ in enumerate(inputs):
            if not hasattr(input_, "hook_registered"):
                input_hook = self._create_backward_hook_input(input_.data, input_index)
                handle = input_.register_hook(input_hook)
                self._handle_input_hooks.append(handle)
                input_.hook_registered = True
        output_hook = self._create_backward_hook_output(outputs.data)
        self._handle_output_hook = outputs.register_hook(output_hook)

        self._create_auxiliary_quantities(inputs)

        return outputs.clone()

    def _create_backward_hook_input(
        self, input_: Tensor, input_index: int
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            relevance = (input_ - self._lower_bound_tensor[input_index]) * grad
            relevance += (
                input_ - self._upper_bound_tensor[input_index]
            ) * self.negative_weight_contraction[input_index]

            return relevance

        return _backward_hook_input

    def _create_backward_hook_output(
        self, output: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> Tensor:

            denominator = self.og_outputs[0] - self._denominator_bound_contribution
            if self.set_bias_to_zero and self._bias_contrib is not None:
                denominator -= torch.cat(
                    tuple(self._bias_contrib for _ in range(output.shape[0]))
                )
            denominator += self.STABILITY_FACTOR

            rescaled_relevance = grad / denominator
            self.negative_weight_contraction = torch.autograd.grad(
                outputs=self.upper_bound_contrib,
                inputs=self._upper_bound_tensor,
                grad_outputs=rescaled_relevance,
                retain_graph=True,
            )

            return rescaled_relevance

        return _backward_hook_output

    def _create_auxiliary_quantities(self, inputs: Tuple[Tensor, ...]) -> None:
        r"""
        Computes the l w^+ + h w^- term for the denominator.
        """
        input_shapes = tuple(x.shape for x in inputs)

        if input_shapes != self._input_shapes:
            self._input_shapes = input_shapes

            with torch.autograd.set_grad_enabled(True):
                self._upper_bound_tensor = tuple(
                    torch.full(
                        input_shape,
                        self.upper_bound,
                        dtype=torch.float,
                        requires_grad=True,
                    )
                    for input_shape in self._input_shapes
                )
                self.upper_bound_contrib = self._module_neg.forward(
                    *self._upper_bound_tensor
                )

            with torch.no_grad():
                self._lower_bound_tensor = tuple(
                    torch.full(input_shape, self.lower_bound, dtype=torch.float)
                    for input_shape in self._input_shapes
                )
                self._denominator_bound_contribution = (
                    self._module_pos.forward(*self._lower_bound_tensor)
                    + self.upper_bound_contrib
                )

    def forward_hook_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        self.og_outputs = outputs.detach()
        super().forward_hook_weights(module, inputs, outputs)

    def _separate_weights_by_sign(self, module: Module) -> None:
        if hasattr(module, "weight"):
            if self._module_neg is None:
                self._module_neg = copy.deepcopy(module)
                self._module_neg.weight.data = torch.clamp(
                    self._module_neg.weight.data, max=0.0
                )
            self._module_pos = module
            self._module_pos.weight.data = torch.clamp(
                self._module_pos.weight.data, min=0.0
            )

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                if self.set_bias_to_zero and self._bias_contrib is None:
                    with torch.no_grad():
                        self._bias_contrib = module.forward(
                            *(
                                torch.zeros(input_.shape[1:]).unsqueeze(dim=0)
                                for input_ in inputs
                            )
                        )
                module.bias.data = torch.zeros_like(module.bias.data)

        self._separate_weights_by_sign(module)


class AlphaBetaRule(ZBoundRule):
    r"""
    Alpha Beta rule for relevance backpropagation. This rule weights
    the positive and negative addends contributing to some pre-activation
    asymmetrically via coefficients alpha and beta (alpha + beta = 1).

    Use for lower layers.
    """

    def __init__(self, alpha: float = 1.0, set_bias_to_zero: bool = False) -> None:
        r"""
        Args:
            alpha (float, optional): Alpha parameter of alpha beta rule.
            Defaults to 1.

            set_bias_to_zero (bool, optional): Parameter for setting bias to
            zero in relevance computations.
            Defaults to False.
        """
        self.alpha = alpha
        self.beta = 1.0 - self.alpha
        self.set_bias_to_zero = set_bias_to_zero

        self._module_pos: Optional[Module] = None
        self._module_neg: Optional[Module] = None
        self._bias_contrib: Tensor = None

    def _create_backward_hook_input(
        self, input_: Tensor, input_index: int
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_input(grad: Tensor) -> Tensor:
            return self.out[input_index]

        return _backward_hook_input

    def _create_backward_hook_output(
        self, output: Tensor
    ) -> Callable[[Tensor], Optional[Tensor]]:
        def _backward_hook_output(grad: Tensor) -> None:
            out = self._compute_signed_contributions(
                grad, self.inputs_pos, self.inputs_neg, +1
            )

            if self.beta:
                out_beta = tuple(
                    self.beta * rel
                    for rel in self._compute_signed_contributions(
                        grad, self.inputs_neg, self.inputs_pos, -1
                    )
                )
                out = tuple(self.alpha * x + y for x, y in zip(out, out_beta))

            self.out = out

        return _backward_hook_output

    def _compute_signed_contributions(
        self,
        grad: Tensor,
        mod_pos_in: Tuple[Tensor, ...],
        mod_neg_in: Tuple[Tensor, ...],
        sign: int,
    ) -> Tuple[Tensor, ...]:
        r"""
        if mod_pos_in is the positive part of the inputs and mod_neg_in the negative
        this computes the alpha part
        if mod_pos_in is the negative part of the inputs and mod_neg_in the positive
        this computes the beta part
        """
        with torch.autograd.set_grad_enabled(True):
            mod_pos_out = self._module_pos.forward(*mod_pos_in)
            mod_neg_out = self._module_neg.forward(*mod_neg_in)

            denominator = mod_pos_out + mod_neg_out
            if not self.set_bias_to_zero and self._bias_contrib is not None:
                if sign == 1:
                    denominator += torch.cat(
                        tuple(self._bias_contrib for _ in range(denominator.shape[0]))
                    ).clamp(min=0)
                else:
                    denominator += torch.cat(
                        tuple(self._bias_contrib for _ in range(denominator.shape[0]))
                    ).clamp(max=0)

            # this might be unneccessary, simply adding epsilon may be enough
            denominator += sign * (torch.eq(denominator, 0.0)) * self.STABILITY_FACTOR

            rescaled_relevance = grad / denominator

            # getting contractions with transposed Jacobian
            positive_weight_contraction = torch.autograd.grad(
                outputs=mod_pos_out, inputs=mod_pos_in, grad_outputs=rescaled_relevance
            )
            negative_weight_contraction = torch.autograd.grad(
                outputs=mod_neg_out, inputs=mod_neg_in, grad_outputs=rescaled_relevance
            )

            out = tuple(
                (pos_in * jac_pos) + (neg_in * jac_neg)
                for jac_pos, pos_in, jac_neg, neg_in in zip(
                    positive_weight_contraction,
                    mod_pos_in,
                    negative_weight_contraction,
                    mod_neg_in,
                )
            )

        return out

    def _create_auxiliary_quantities(self, inputs: Tuple[Tensor, ...]) -> None:
        self.inputs_pos = tuple(input_.data.clamp(min=0) for input_ in inputs)
        self.inputs_neg = tuple(input_.data.clamp(max=0) for input_ in inputs)

        for input_pos, input_neg in zip(self.inputs_pos, self.inputs_neg):
            input_pos.requires_grad_(True)
            input_neg.requires_grad_(True)

    def _manipulate_weights(
        self,
        module: Module,
        inputs: Tuple[Tensor, ...],
        outputs: Tensor,
    ) -> None:
        if hasattr(module, "bias"):
            if module.bias is not None:
                if not self.set_bias_to_zero and self._bias_contrib is None:
                    with torch.no_grad():
                        self._bias_contrib = module.forward(
                            *(
                                torch.zeros(input_.shape[1:]).unsqueeze(dim=0)
                                for input_ in inputs
                            )
                        )
                module.bias.data = torch.zeros_like(module.bias.data)

        self._separate_weights_by_sign(module)
