#!/usr/bin/env python3

# pyre-strict

import unittest

import torch
from captum.module.binary_concrete_stochastic_gates import BinaryConcreteStochasticGates
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual


class TestBinaryConcreteStochasticGates(BaseTest):
    testing_device: str = "cpu"

    def setUp(self) -> None:
        super().setUp()
        if self.testing_device == "cuda" and not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")

    def test_bcstg_1d_input(self) -> None:

        dim = 3
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gated_input, reg = bcstg(input_tensor)
        expected_reg = 2.4947

        expected_gated_input = [[0.0000, 0.0212, 0.1892], [0.1839, 0.3753, 0.4937]]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_bcstg_1d_input_with_reg_reduction(self) -> None:

        dim = 3
        mean_bcstg = BinaryConcreteStochasticGates(dim, reg_reduction="mean").to(
            self.testing_device
        )
        none_bcstg = BinaryConcreteStochasticGates(dim, reg_reduction="none").to(
            self.testing_device
        )
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        mean_gated_input, mean_reg = mean_bcstg(input_tensor)
        none_gated_input, none_reg = none_bcstg(input_tensor)
        expected_mean_reg = 0.8316
        expected_none_reg = torch.tensor([0.8321, 0.8310, 0.8325])

        assertTensorAlmostEqual(self, mean_reg, expected_mean_reg)
        assertTensorAlmostEqual(self, none_reg, expected_none_reg)

    def test_bcstg_1d_input_with_n_gates_error(self) -> None:

        dim = 3
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor([0.0, 0.1, 0.2]).to(self.testing_device)

        with self.assertRaises(AssertionError):
            bcstg(input_tensor)

    def test_bcstg_num_mask_not_equal_dim_error(self) -> None:
        dim = 3
        mask = torch.tensor([0, 0, 1])  # only two distinct masks, but given dim is 3

        with self.assertRaises(AssertionError):
            BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)

    def test_gates_values_matching_dim_when_eval(self) -> None:
        dim = 3
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        bcstg.train(False)
        gated_input, reg = bcstg(input_tensor)
        assert gated_input.shape == input_tensor.shape

    def test_bcstg_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 0, 1]).to(self.testing_device)
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        gated_input, reg = bcstg(input_tensor)
        expected_reg = 1.6643

        expected_gated_input = [[0.0000, 0.0000, 0.1679], [0.0000, 0.0000, 0.2223]]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_bcstg_2d_input(self) -> None:

        dim = 3 * 2
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)

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

        gated_input, reg = bcstg(input_tensor)

        expected_reg = 4.9903
        expected_gated_input = [
            [[0.0000, 0.0990], [0.0261, 0.2431], [0.0551, 0.3863]],
            [[0.0476, 0.6177], [0.5400, 0.1530], [0.0984, 0.8013]],
        ]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_bcstg_2d_input_with_n_gates_error(self) -> None:

        dim = 5
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
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
            bcstg(input_tensor)

    def test_bcstg_2d_input_with_mask(self) -> None:

        dim = 3
        mask = torch.tensor(
            [
                [0, 1],
                [1, 1],
                [0, 2],
            ]
        ).to(self.testing_device)
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)

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

        gated_input, reg = bcstg(input_tensor)
        expected_reg = 2.4947

        expected_gated_input = [
            [[0.0000, 0.0212], [0.0424, 0.0636], [0.3191, 0.4730]],
            [[0.3678, 0.6568], [0.7507, 0.8445], [0.6130, 1.0861]],
        ]

        assertTensorAlmostEqual(self, gated_input, expected_gated_input, mode="max")
        assertTensorAlmostEqual(self, reg, expected_reg)

    def test_get_gate_values_1d_input(self) -> None:

        dim = 3
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        bcstg(input_tensor)
        gate_values = bcstg.get_gate_values()

        expected_gate_values = [0.5001, 0.5012, 0.4970]

        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 1, 1])
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        bcstg(input_tensor)
        gate_values = bcstg.get_gate_values()

        expected_gate_values = [0.5001, 0.5012]

        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_2d_input(self) -> None:

        dim = 3 * 2
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)

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

        bcstg(input_tensor)
        gate_values = bcstg.get_gate_values()

        expected_gate_values = [0.5001, 0.5012, 0.4970, 0.5007, 0.4982, 0.5015]

        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_values_clamp(self) -> None:
        # enlarge the bounds & extremify log_alpha to mock gate  values beyond 0 & 1
        bcstg = BinaryConcreteStochasticGates._from_pretrained(
            torch.tensor([10.0, -10.0, 10.0]),
            lower_bound=-2,
            upper_bound=2,
        ).to(self.testing_device)

        clamped_gate_values = bcstg.get_gate_values().cpu().tolist()
        assert clamped_gate_values == [1.0, 0.0, 1.0]

        unclamped_gate_values = bcstg.get_gate_values(clamp=False).cpu().tolist()
        assert (
            unclamped_gate_values[0] > 1
            and unclamped_gate_values[1] < 0
            and unclamped_gate_values[2] > 1
        )

    def test_get_gate_values_2d_input_with_mask(self) -> None:

        dim = 3
        mask = torch.tensor(
            [
                [0, 1],
                [1, 1],
                [0, 2],
            ]
        )
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)

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

        bcstg(input_tensor)
        gate_values = bcstg.get_gate_values()

        expected_gate_values = [0.5001, 0.5012, 0.4970]

        assertTensorAlmostEqual(self, gate_values, expected_gate_values, mode="max")

    def test_get_gate_active_probs_1d_input(self) -> None:

        dim = 3
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        bcstg(input_tensor)
        gate_active_probs = bcstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8319, 0.8324, 0.8304]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_get_gate_active_probs_1d_input_with_mask(self) -> None:

        dim = 2
        mask = torch.tensor([0, 1, 1])
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)
        input_tensor = torch.tensor(
            [
                [0.0, 0.1, 0.2],
                [0.3, 0.4, 0.5],
            ]
        ).to(self.testing_device)

        bcstg(input_tensor)
        gate_active_probs = bcstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8319, 0.8324]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_get_gate_active_probs_2d_input(self) -> None:

        dim = 3 * 2
        bcstg = BinaryConcreteStochasticGates(dim).to(self.testing_device)

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

        bcstg(input_tensor)
        gate_active_probs = bcstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8319, 0.8324, 0.8304, 0.8321, 0.8310, 0.8325]

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
        bcstg = BinaryConcreteStochasticGates(dim, mask=mask).to(self.testing_device)

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

        bcstg(input_tensor)
        gate_active_probs = bcstg.get_gate_active_probs()

        expected_gate_active_probs = [0.8319, 0.8324, 0.8304]

        assertTensorAlmostEqual(
            self, gate_active_probs, expected_gate_active_probs, mode="max"
        )

    def test_from_pretrained(self) -> None:
        log_alpha_param = torch.tensor([0.1, 0.2, 0.3, 0.4])
        kwargs = {
            "mask": torch.tensor([0, 1, 1, 0, 2, 3]),
            "reg_weight": 0.1,
            "lower_bound": -0.2,
            "upper_bound": 1.2,
        }
        stg = BinaryConcreteStochasticGates._from_pretrained(log_alpha_param, **kwargs)

        for key, expected_val in kwargs.items():
            val = getattr(stg, key)
            if isinstance(expected_val, torch.Tensor):
                assertTensorAlmostEqual(self, val, expected_val, mode="max")
            else:
                assert val == expected_val
