#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import unittest

import torch
from captum.attr._core.shapley_value import _shape_feature_mask, ShapleyValueSampling
from captum.testing.helpers.basic import BaseTest


class TestShapleyDeviceMismatch(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")

    def test_shape_feature_mask_multi_input_cpu_mask_cuda_input(self) -> None:
        """Test device mismatch with multiple inputs on CUDA."""
        inp1 = torch.tensor([[1.0, 2.0, 3.0]], device="cuda")
        inp2 = torch.tensor([[4.0, 5.0, 6.0]], device="cuda")
        mask1 = torch.tensor([[0, 0, 1]], device="cpu")
        mask2 = torch.tensor([[0, 1, 2]], device="cpu")
        result = _shape_feature_mask((mask1, mask2), (inp1, inp2))
        self.assertEqual(result[0].device, inp1.device)
        self.assertEqual(result[1].device, inp2.device)

    def test_multi_input_attributions_remain_on_each_input_device(self) -> None:
        cpu_input = torch.tensor([[1.0, 2.0]], device="cpu")
        cuda_input = torch.tensor([[3.0, 4.0]], device="cuda")

        def forward(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
            return first.to(second.device).sum(dim=1) + second.sum(dim=1)

        attributions = ShapleyValueSampling(forward).attribute(
            (cpu_input, cuda_input), n_samples=1
        )

        self.assertEqual(attributions[0].device, cpu_input.device)
        self.assertEqual(attributions[1].device, cuda_input.device)
        torch.testing.assert_close(attributions[0], cpu_input)
        torch.testing.assert_close(attributions[1], cuda_input)


if __name__ == "__main__":
    unittest.main()
