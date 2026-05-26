#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from captum.module.binary_concrete_stochastic_gates import BinaryConcreteStochasticGates
from torch import Tensor

from .test_binary_concrete_stochastic_gates import TestBinaryConcreteStochasticGates


# CUDA RNG produces different sequences on different GPU architectures
# (e.g. V100 vs A100 vs H100) even with the same seed, causing flaky
# tests. By generating uniform samples on CPU and moving to the device,
# tests get consistent results regardless of which GPU type runs them.
def _cpu_rng_sample(self: BinaryConcreteStochasticGates, batch_size: int) -> Tensor:
    if self.training:
        u = torch.empty(batch_size, self.n_gates)
        u.uniform_(self.eps, 1 - self.eps)
        u = u.to(self.log_alpha_param.device)
        s = torch.sigmoid((torch.logit(u) + self.log_alpha_param) / self.temperature)
    else:
        s = torch.sigmoid(self.log_alpha_param)
        s = s.expand(batch_size, self.n_gates)

    s_bar = s * (self.upper_bound - self.lower_bound) + self.lower_bound
    return s_bar


class TestBinaryConcreteStochasticGatesCUDA(
    TestBinaryConcreteStochasticGates,
):
    testing_device: str = "cuda"

    def setUp(self) -> None:
        super().setUp()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[..., Tensor]`; used as `(...) -> Tensor`.
        patcher = patch.object(
            BinaryConcreteStochasticGates,
            "_sample_gate_values",
            _cpu_rng_sample,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
