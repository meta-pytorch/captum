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


class GaussianStochasticGates(StochasticGatesBase):
    """
    Stochastic Gates with Gaussian distribution.

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

    GaussianStochasticGates adopts a gaussian distribution as the smoothed Bernoulli
    distribution of gate. While the smoothed Bernoulli distribution should be
    within 0 and 1, gaussian does not have boundaries. So hard-sigmoid rectification
    is used to "fold" the parts smaller than 0 or larger than 1 back to 0 and 1.

    More details can be found in the original paper:
    https://arxiv.org/abs/1810.04247

    Examples::

        >>> n_params = 5  # number of gates
        >>> stg = GaussianStochasticGates(n_params, reg_weight=0.01)
        >>> inputs = torch.randn(3, n_params)  # mock inputs with batch size of 3
        >>> gated_inputs, reg = stg(mock_inputs)  # gate the inputs
    """

    def __init__(
        self,
        n_gates: int,
        mask: Optional[Tensor] = None,
        reg_weight: float = 1.0,
        std: float = 0.5,
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

            std (float, optional): standard deviation that will be fixed throughout.
                Default: 0.5

            reg_reduction (str, optional): the reduction to apply to the regularization:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied and it will be the same as the return of ``get_active_probs``,
                ``'mean'``: the sum of the gates non-zero probabilities will be divided
                by the number of gates, ``'sum'``: the gates non-zero probabilities will
                be summed.
                Default: ``'sum'``
        """
        super().__init__(
            n_gates,
            mask=mask,
            reg_weight=reg_weight,
            reg_reduction=reg_reduction,
        )

        mu = torch.empty(n_gates)
        nn.init.normal_(mu, mean=0.5, std=0.01)
        self.mu = nn.Parameter(mu)

        assert 0 < std, f"the standard deviation should be positive, received {std}"  # type: ignore  # noqa: E501 line too long
        self.std = std

    def _sample_gate_values(self, batch_size: int) -> Tensor:
        """
        Sample gate values for each example in the batch from the Gaussian distribution

        Args:
            batch_size (int): input batch size

        Returns:
            gate_values (Tensor): gate value tensor of shape(batch_size, n_gates)
        """

        if self.training:
            n = torch.empty(batch_size, self.n_gates, device=self.mu.device)
            n.normal_(mean=0, std=self.std)
            return self.mu + n

        return self.mu.expand(batch_size, self.n_gates)

    def _get_gate_values(self) -> Tensor:
        """
        Get the raw gate values, which are the means of the underneath gate
        distributions, the learned mu

        Returns:
            gate_values (Tensor): value of each gate after model is trained
        """
        return self.mu

    def _get_gate_active_probs(self) -> Tensor:
        """
        Get the active probability of each gate, i.e, gate value > 0, in the
        Gaussian distribution

        Returns:
            probs (Tensor): probabilities tensor of the gates are active
                in shape(n_gates)
        """
        std = self.std
        assert std is not None, "std should not be None"
        x = self.mu / std
        return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

    @classmethod
    def _from_pretrained(
        cls, mu: Tensor, *args: Any, **kwargs: Any
    ) -> "GaussianStochasticGates":
        """
        Private factory method to create an instance with pretrained parameters

        Args:
            mu (Tensor): FloatTensor containing weights for the pretrained mu

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

            std (float, optional): standard deviation that will be fixed throughout.
                Default: 0.5

            reg_reduction (str, optional): the reduction to apply to the regularization:
                ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be
                applied and it will be the same as the return of ``get_active_probs``,
                ``'mean'``: the sum of the gates non-zero probabilities will be divided
                by the number of gates, ``'sum'``: the gates non-zero probabilities will
                be summed.
                Default: ``'sum'``

        Returns:
            stg (GaussianStochasticGates): StochasticGates instance
        """
        n_gates = mu.numel()
        stg = cls(n_gates, *args, **kwargs)
        stg.load_state_dict({"mu": mu}, strict=False)

        return stg
