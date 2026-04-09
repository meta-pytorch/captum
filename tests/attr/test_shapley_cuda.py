#!/usr/bin/env python3

# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from captum.attr._core.shapley_value import _shape_feature_mask
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


if __name__ == "__main__":
    unittest.main()
