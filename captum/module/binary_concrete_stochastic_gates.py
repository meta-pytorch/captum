#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math
from typing import Any, Optional

import torch
from captum.module.stochastic_gates_base import StochasticGatesBase
from torch import nn, Tensor


def _torch_empty(batch_size: int, n_gates: int, device: torch.device) -> Tensor:
    return torch.empty(batch_size, n_gates, device=device)


# torch.fx is introduced in 1.8.0
if hasattr(torch, "fx"):
    torch.fx.wrap(_torch_empty)


class BinaryConcreteStochasticGates(StochasticGatesBase):
    """
    Stochastic Gates with binary concrete distribution.

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

    BinaryConcreteStochasticGates adopts a "stretched" binary concrete distribution as
    the smoothed Bernoulli distribution of gate. The binary concrete distribution does
    not include its lower and upper boundaries, 0 and 1, which are required by a
    Bernoulli distribution, so it needs to be linearly stretched beyond both boundaries.
    Then use hard-sigmoid rectification to "fold" the parts smaller than 0 or larger
    than 1 back to 0 and 1.

    More details can be found in the original paper:
    https://arxiv.org/abs/1712.01312

    Examples::

        >>> n_params = 5  # number of parameters
        >>> stg = BinaryConcreteStochasticGates(n_params, reg_weight=0.01)
        >>> inputs = torch.randn(3, n_params)  # mock inputs with batch size of 3
        >>> gated_inputs, reg = stg(mock_inputs)  # gate the inputs

    """

    def __init__(
        self,
        n_gates: int,
        mask: Optional[Tensor] = None,
        reg_weight: float = 1.0,
        temperature: float = 2 / 3,
        lower_bound: float = -0.1,
        upper_bound: float = 1.1,
        eps: float = 1e-8,
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
                (on dimensions other than dim 0, i.e., batch dim) is gated separately.
                Default: None

            reg_weight (float, optional): rescaling weight for L0 regularization term.
                Default: 1.0

            temperature (float, optional): temperature of the concrete distribution,
                controls the degree of approximation, as 0 means the original Bernoulli
                without relaxation. The value should be between 0 and 1.
                Default: 2/3

            lower_bound (float, optional): the lower bound to "stretch" the binary
                concrete distribution
                Default: -0.1

            upper_bound (float, optional): the upper bound to "stretch" the binary
                concrete distribution
                Default: 1.1

            eps (float, optional): term to improve numerical stability in binary
                concerete sampling
                Default: 1e-8

            reg_reduction (str, optional): the reduction to apply to the regularization:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied and it will be the same as the return of ``get_active_probs``,
                ``'mean'``: the sum of the gates non-zero probabilities will be divided
                by the number of gates, ``'sum'``: the gates non-zero probabilities will
                be summed.
                Default: ``'sum'``
        """
        super().__init__(
            n_gates, mask=mask, reg_weight=reg_weight, reg_reduction=reg_reduction
        )

        # avoid changing the tensor's variable name
        # when the module is used after compilation,
        # users may directly access this tensor by name
        log_alpha_param = torch.empty(n_gates)
        nn.init.normal_(log_alpha_param, mean=0.0, std=0.01)
        self.log_alpha_param = nn.Parameter(log_alpha_param)

        assert (
            0 < temperature < 1
        ), f"the temperature should be between 0 and 1, received {temperature}"
        self.temperature = temperature

        assert (
            lower_bound < 0
        ), f"the stretch lower bound should be smaller than 0, received {lower_bound}"
        self.lower_bound = lower_bound

        assert (
            upper_bound > 1
        ), f"the stretch upper bound should be larger than 1, received {upper_bound}"
        self.upper_bound = upper_bound

        self.eps = eps

        # pre-calculate the fixed term used in active prob
        self.active_prob_offset: float = temperature * math.log(
            -lower_bound / upper_bound
        )

    def _sample_gate_values(self, batch_size: int) -> Tensor:
        """
        Sample gate values for each example in the batch from the binary concrete
        distributions

        Args:
            batch_size (int): input batch size

        Returns:
            gate_values (Tensor): gate value tensor of shape(batch_size, n_gates)
        """
        if self.training:
            u = _torch_empty(
                batch_size, self.n_gates, device=self.log_alpha_param.device
            )
            u.uniform_(self.eps, 1 - self.eps)
            s = torch.sigmoid(
                (torch.logit(u) + self.log_alpha_param) / self.temperature
            )

        else:
            s = torch.sigmoid(self.log_alpha_param)
            s = s.expand(batch_size, self.n_gates)

        s_bar = s * (self.upper_bound - self.lower_bound) + self.lower_bound

        return s_bar

    def _get_gate_values(self) -> Tensor:
        """
        Get the raw gate values, which are the means of the underneath gate
        distributions, derived from learned log_alpha_param

        Returns:
            gate_values (Tensor): value of each gate after model is trained
        """
        gate_values = (
            torch.sigmoid(self.log_alpha_param) * (self.upper_bound - self.lower_bound)
            + self.lower_bound
        )
        return gate_values

    def _get_gate_active_probs(self) -> Tensor:
        """
        Get the active probability of each gate, i.e, gate value > 0, in the binary
        concrete distributions

        Returns:
            probs (Tensor): probabilities tensor of the gates are active
                in shape(n_gates)
        """
        return torch.sigmoid(self.log_alpha_param - self.active_prob_offset)

    @classmethod
    def _from_pretrained(
        cls, log_alpha_param: Tensor, *args: Any, **kwargs: Any
    ) -> "BinaryConcreteStochasticGates":
        """
        Private factory method to create an instance with pretrained parameters

        Args:
            log_alpha_param (Tensor): FloatTensor containing weights for
                the pretrained log_alpha

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

            temperature (float, optional): temperature of the concrete distribution,
                controls the degree of approximation, as 0 means the original Bernoulli
                without relaxation. The value should be between 0 and 1.
                Default: 2/3

            lower_bound (float, optional): the lower bound to "stretch" the binary
                concrete distribution
                Default: -0.1

            upper_bound (float, optional): the upper bound to "stretch" the binary
                concrete distribution
                Default: 1.1

            eps (float, optional): term to improve numerical stability in binary
                concerete sampling
                Default: 1e-8

            reg_reduction (str, optional): the reduction to apply to the regularization:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied and it will be the same as the return of ``get_active_probs``,
                ``'mean'``: the sum of the gates non-zero probabilities will be divided
                by the number of gates, ``'sum'``: the gates non-zero probabilities will
                be summed.
                Default: ``'sum'``

        Returns:
            stg (BinaryConcreteStochasticGates): StochasticGates instance
        """
        assert (
            log_alpha_param.dim() == 1
        ), "log_alpha_param is expected to be 1-dimensional"

        n_gates = log_alpha_param.numel()
        stg = cls(n_gates, *args, **kwargs)
        stg.load_state_dict({"log_alpha_param": log_alpha_param}, strict=False)

        return stg
