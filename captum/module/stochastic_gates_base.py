#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Module


class StochasticGatesBase(Module, ABC):
    """
    Abstract module for Stochastic Gates.

    Stochastic Gates is a practical solution to add L0 norm regularization for neural
    networks. L0 regularization, which explicitly penalizes any present (non-zero)
    parameters, can help network pruning and feature selection, but directly optimizing
    L0 is a non-differentiable combinatorial problem. To surrogate L0, Stochastic Gate
    uses certain continuous probability distributions (e.g., Concrete, Gaussian) with
    hard-sigmoid rectification as a continuous smoothed Bernoulli distribution
    determining the weight of a parameter, i.e., gate. Then L0 is equal to the gates's
    non-zero probability represented by the parameters of the continuous probability
    distribution. The gate value can also be reparameterized to the distribution
    parameters with a noise. So the expected L0 can be optimized through learning
    the distribution parameters via stochastic gradients.

    This base class defines the shared variables and forward logic of how the input is
    gated regardless of the underneath distribution. The actual implementation should
    extend this class and implement the distribution specific functions.
    """

    mask: Optional[Tensor]

    def __init__(
        self,
        n_gates: int,
        mask: Optional[Tensor] = None,
        reg_weight: float = 1.0,
        reg_reduction: str = "sum",
    ) -> None:
        """
        Args:
            n_gates (int): number of gates.

            mask (Tensor, optional): If provided, this allows grouping multiple
                input tensor elements to share the same stochastic gate.
                This tensor should be broadcastable to match the input shape
                and contain integers in the range 0 to n_gates - 1.
                Indices grouped to the same stochastic gate should have the same value.
                If not provided, each element in the input tensor
                (on dimensions other than dim 0 - batch dim) is gated separately.
                Default: None

            reg_weight (float, optional): rescaling weight for L0 regularization term.
                Default: 1.0

            reg_reduction (str, optional): the reduction to apply to the regularization:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied and it will be the same as the return of ``get_active_probs``,
                ``'mean'``: the sum of the gates non-zero probabilities will be divided
                by the number of gates, ``'sum'``: the gates non-zero probabilities will
                be summed.
                Default: ``'sum'``
        """
        super().__init__()

        if mask is not None:
            max_mask_ind = mask.max().item()
            assert max_mask_ind == n_gates - 1, (
                f"the maximum mask index (received {max_mask_ind}) should be equal to"
                f" the number of gates - 1 (received {n_gates}) since each mask"
                " should correspond to a gate"
            )

        valid_reg_reduction = ["none", "mean", "sum"]
        assert (
            reg_reduction in valid_reg_reduction
        ), f"reg_reduction must be one of [none, mean, sum], received: {reg_reduction}"
        self.reg_reduction = reg_reduction

        self.n_gates = n_gates
        self.register_buffer(
            "mask", mask.detach().clone() if mask is not None else None
        )
        self.reg_weight = reg_weight

    def forward(self, input_tensor: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_tensor (Tensor): Tensor to be gated with stochastic gates


        Returns:
            tuple[Tensor, Tensor]:

            - gated_input (Tensor): Tensor of the same shape weighted by the sampled
                gate values

            - l0_reg (Tensor): L0 regularization term to be optimized together with
                model loss,
                e.g. loss(model_out, target) + l0_reg
        """
        if self.mask is None:
            n_ele = self._get_numel_of_input(input_tensor)
            assert n_ele == self.n_gates, (
                "if mask is not given, each example in the input batch should have the"
                " same number of elements"
                f" (received {n_ele}) as gates ({self.n_gates})"
            )

        input_size = input_tensor.size()
        batch_size = input_size[0]

        gate_values = self._sample_gate_values(batch_size)

        # hard-sigmoid rectification z=min(1,max(0,_z))
        gate_values = torch.clamp(gate_values, min=0, max=1)

        if self.mask is not None:
            # use expand_as not expand/broadcast_to which do not work with torch.fx
            input_mask = self.mask.expand_as(input_tensor)

            # flatten all dim except batch to gather from gate values
            flattened_mask = input_mask.reshape(batch_size, -1)
            gate_values = torch.gather(gate_values, 1, flattened_mask)

        # reshape gates(batch_size, n_elements) into input_size for point-wise mul
        gate_values = gate_values.reshape(input_size)
        gated_input = input_tensor * gate_values

        prob_density = self._get_gate_active_probs()
        if self.reg_reduction == "sum":
            l0_reg = prob_density.sum()
        elif self.reg_reduction == "mean":
            l0_reg = prob_density.mean()
        else:
            l0_reg = prob_density

        l0_reg *= self.reg_weight

        return gated_input, l0_reg

    def get_gate_values(self, clamp: bool = True) -> Tensor:
        """
        Get the gate values, which are the means of the underneath gate distributions,
        optionally clamped within 0 and 1.

        Args:
            clamp (bool, optional): whether to clamp the gate values or not. As smoothed
                Bernoulli variables, gate values are clamped within 0 and 1 by default.
                Turn this off to get the raw means of the underneath
                distribution (e.g., concrete, gaussian), which can be useful to
                differentiate the gates' importance when multiple gate
                values are beyond 0 or 1.
                Default: ``True``

        Returns:
            Tensor:
            - gate_values (Tensor): value of each gate in shape(n_gates)
        """
        gate_values = self._get_gate_values()
        if clamp:
            gate_values = torch.clamp(gate_values, min=0, max=1)

        return gate_values.detach()

    def get_gate_active_probs(self) -> Tensor:
        """
        Get the active probability of each gate, i.e, gate value > 0

        Returns:
            Tensor:
            - probs (Tensor): probabilities tensor of the gates are active
                in shape(n_gates)
        """
        return self._get_gate_active_probs().detach()

    @abstractmethod
    def _get_gate_values(self) -> Tensor:
        """
        Protected method to be override in the child depending on the chosen
        distribution. Get the raw gate values derived from the learned parameters of
        the according distribution without clamping.

        Returns:
            gate_values (Tensor): gate value tensor of shape(n_gates)
        """
        pass

    @abstractmethod
    def _sample_gate_values(self, batch_size: int) -> Tensor:
        """
        Protected method to be override in the child depending on the chosen
        distribution. Sample gate values for each example in the batch from a
        probability distribution

        Args:
            batch_size (int): input batch size

        Returns:
            gate_values (Tensor): gate value tensor of shape(batch_size, n_gates)
        """
        pass

    @abstractmethod
    def _get_gate_active_probs(self) -> Tensor:
        """
        Protected method to be override in the child depending on the chosen
        distribution. Get the active probability of each gate, i.e, gate value > 0

        Returns:
            probs (Tensor): probabilities tensor of the gates are active
                in shape(n_gates)
        """
        pass

    def _get_numel_of_input(self, input_tensor: Tensor) -> int:
        """
        Get the number of elements of a single example in the batched input tensor
        """
        assert input_tensor.dim() > 1, (
            "The input tensor must have more than 1 dimension with the 1st dimention"
            " being batch size;"
            f" received input tensor shape {input_tensor.size()}"
        )
        return input_tensor[0].numel()
