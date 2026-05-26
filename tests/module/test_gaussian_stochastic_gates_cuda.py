#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest
from unittest.mock import patch

import torch
from captum.module.gaussian_stochastic_gates import GaussianStochasticGates
from torch import Tensor

from .test_gaussian_stochastic_gates import TestGaussianStochasticGates


# CUDA RNG produces different sequences on different GPU architectures
# (e.g. V100 vs A100 vs H100) even with the same seed, causing flaky tests.
# By generating noise on CPU (where torch.manual_seed is deterministic across
# all hardware) and moving to the device, tests get consistent results
# regardless of which GPU type runs them in CI.
def _cpu_rng_sample(self: GaussianStochasticGates, batch_size: int) -> Tensor:
    if self.training:
        n = torch.empty(batch_size, self.n_gates)
        n.normal_(mean=0, std=self.std)
        return self.mu + n.to(self.mu.device)
    return self.mu.expand(batch_size, self.n_gates)


class TestGaussianStochasticGatesCUDA(TestGaussianStochasticGates):
    testing_device: str = "cuda"

    def setUp(self) -> None:
        super().setUp()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")
        # pyre-fixme[8]: Attribute has type
        #  `BoundMethod[..., Tensor]`; used as `(...) -> Tensor`.
        patcher = patch.object(
            GaussianStochasticGates,
            "_sample_gate_values",
            _cpu_rng_sample,
        )
        patcher.start()
        self.addCleanup(patcher.stop)
