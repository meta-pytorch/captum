#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import unittest

import torch
from captum.module.gaussian_stochastic_gates import GaussianStochasticGates
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual


class TestGaussianStochasticGates(BaseTest):
    testing_device: str = "cpu"

    def setUp(self) -> None:
        super().setUp()
        if self.testing_device == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")

    def test_gstg_1d_input(self) -> None:

        dim = 3
        gstg = GaussianStochasticGates(dim).to(self.testing_device)

        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gated_input, reg = gstg(input_tensor)
        expected_reg = 2.5213
        expected_gated_input = [[0.0000, 0.0198, 0.1483], [0.1848, 0.3402, 0.1782]]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_gstg_1d_input_with_reg_reduction(self) -> None:
        dim = 3
        mean_gstg = GaussianStochasticGates(dim, reg_reduction="mean").to(
            self.testing_device
        )
        none_gstg = GaussianStochasticGates(dim, reg_reduction="none").to(
            self.testing_device
        )

        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        _, mean_reg = mean_gstg(input_tensor)
        _, none_reg = none_gstg(input_tensor)
        expected_mean_reg = 0.8404
        expected_none_reg = torch.tensor([0.8424, 0.8384, 0.8438])

        assertTensorAlmostEqual(self, mean_reg, expected_mean_reg)
        assertTensorAlmostEqual(self, none_reg, expected_none_reg)

    def test_gstg_1d_input_with_n_gates_error(self) -> None:

        dim = 3
        gstg = GaussianStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor([0.0, 0.1, 0.2]).to(self.testing_device)

        with self.assertRaises(AssertionError):
            gstg(input_tensor)

    def test_gstg_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 0, 1]).to(self.testing_device)
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gated_input, reg = gstg(input_tensor)
        expected_reg = 1.6849
        expected_gated_input = [[0.0000, 0.0000, 0.1225], [0.0583, 0.0777, 0.3779]]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_gates_values_matching_dim_when_eval(self) -> None:
        dim = 3
        gstg = GaussianStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gstg.train(False)
        gated_input, reg = gstg(input_tensor)
        assert gated_input.shape == input_tensor.shape

    def test_gstg_2d_input(self) -> None:

        dim = 3 * 2
        gstg = GaussianStochasticGates(dim).to(self.testing_device)

        # shape(2,3,2)
        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gated_input, reg = gstg(input_tensor)
        expected_reg = 5.0458
        expected_gated_input = [
            [[0.0000, 0.0851], [0.0713, 0.3000], [0.2180, 0.1878]],
            [[0.2538, 0.0000], [0.3391, 0.8501], [0.3633, 0.8913]],
        ]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_gstg_2d_input_with_n_gates_error(self) -> None:

        dim = 5
        gstg = GaussianStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
            ]
        ).to(self.testing_device)

        with self.assertRaises(AssertionError):
            gstg(input_tensor)

    def test_gstg_2d_input_with_mask(self) -> None:

        dim = 3
        mask = torch.tensor(
            [
                [0, 1],
                [1, 1],
                [0, 2],
            ]
        ).to(self.testing_device)
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)

        # shape(2,3,2)
        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gated_input, reg = gstg(input_tensor)
        expected_reg = 2.5213
        expected_gated_input = [
            [[0.0000, 0.0198], [0.0396, 0.0594], [0.2435, 0.3708]],
            [[0.3696, 0.5954], [0.6805, 0.7655], [0.6159, 0.3921]],
        ]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_get_gate_values_1d_input(self) -> None:

        dim = 3
        gstg = GaussianStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_values = gstg.get_gate_values()

        expected_gate_values = [0.5005, 0.5040, 0.4899]
        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 1, 1])
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_values = gstg.get_gate_values()

        expected_gate_values = [0.5005, 0.5040]
        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_2d_input(self) -> None:

        dim = 3 * 2
        gstg = GaussianStochasticGates(dim).to(self.testing_device)

        # shape(2,3,2)
        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_values = gstg.get_gate_values()

        expected_gate_values = [0.5005, 0.5040, 0.4899, 0.5022, 0.4939, 0.5050]
        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_2d_input_with_mask(self) -> None:

        dim = 3
        mask = torch.tensor(
            [
                [0, 1],
                [1, 1],
                [0, 2],
            ]
        )
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)

        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_values = gstg.get_gate_values()

        expected_gate_values = [0.5005, 0.5040, 0.4899]
        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_clamp(self) -> None:
        gstg = GaussianStochasticGates._from_pretrained(
            torch.tensor([2.0, -2.0, 2.0])
        ).to(self.testing_device)

        clamped_gate_values = gstg.get_gate_values().cpu().tolist()
        assert clamped_gate_values == [1.0, 0.0, 1.0]

        unclamped_gate_values = gstg.get_gate_values(clamp=False).cpu().tolist()
        assert (
            unclamped_gate_values[0] > 1
            and unclamped_gate_values[1] < 0
            and unclamped_gate_values[2] > 1
        )

    def test_get_gate_active_probs_1d_input(self) -> None:

        dim = 3
        gstg = GaussianStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_active_probs = gstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8416, 0.8433, 0.8364]
        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_get_gate_active_probs_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 1, 1])
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_active_probs = gstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8416, 0.8433]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_get_gate_active_probs_2d_input(self) -> None:

        dim = 3 * 2
        gstg = GaussianStochasticGates(dim).to(self.testing_device)

        # shape(2,3,2)
        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_active_probs = gstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8416, 0.8433, 0.8364, 0.8424, 0.8384, 0.8438]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_get_gate_active_probs_2d_input_with_mask(self) -> None:

        dim = 3
        mask = torch.tensor(
            [
                [0, 1],
                [1, 1],
                [0, 2],
            ]
        )
        gstg = GaussianStochasticGates(dim, mask=mask).to(self.testing_device)

        input_tensor = torch.tensor(
            [
                [
                    [0.0, 0.1],
                    [0.2, 0.3],
                    [0.4, 0.5],
                ],
                [
                    [0.6, 0.7],
                    [0.8, 0.9],
                    [1.0, 1.1],
                ],
            ]
        ).to(self.testing_device)

        gstg(input_tensor)
        gate_active_probs = gstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8416, 0.8433, 0.8364]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_from_pretrained(self) -> None:
        mu = torch.tensor([0.1, 0.2, 0.3, 0.4])
        kwargs = {
            "mask": torch.tensor([0, 1, 1, 0, 2, 3]),
            "reg_weight": 0.1,
            "std": 0.01,
        }
        stg = GaussianStochasticGates._from_pretrained(mu, **kwargs)

        for key, expected_val in kwargs.items():
            val = getattr(stg, key)
            if isinstance(expected_val, torch.Tensor):
                assertTensorAlmostEqual(self, val, expected_val, mode="max")
            else:
                assert val == expected_val
