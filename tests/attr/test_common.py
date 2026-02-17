#!/usr/bin/env python3

# pyre-strict

import torch
from captum.attr._core.noise_tunnel import SUPPORTED_NOISE_TUNNEL_TYPES
from captum.attr._utils.common import _validate_input, _validate_noise_tunnel_type
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_validate_input(self) -> None:
        with self.assertRaises(AssertionError) as err:
            _validate_input(
                (torch.tensor([-1.0, 1.0]),), (torch.tensor([-2.0, 0.0, 1.0]),)
            )
        self.assertEqual(
            "Baseline can be provided as a tensor for just one input and "
            "broadcasted to the batch or input and baseline must have the "
            "same shape or the baseline corresponding to each input tensor "
            "must be a scalar. Found baseline: tensor([-2.,  0.,  1.]) and "
            "input: tensor([-1.,  1.])",
            str(err.exception),
        )

        with self.assertRaises(AssertionError) as err:
            _validate_input(
                (torch.tensor([-1.0, 1.0]),), (torch.tensor([-1.0, 1.0]),), n_steps=-1
            )
        self.assertEqual(
            "The number of steps must be a positive integer. Given: -1",
            str(err.exception),
        )

        with self.assertRaises(AssertionError) as err:
            _validate_input(
                (torch.tensor([-1.0, 1.0]),),
                (torch.tensor([-1.0, 1.0]),),
                method="abcde",
            )
        self.assertIn(
            "Approximation method must be one for the following",
            str(err.exception),
        )
        # any baseline which is broadcastable to match the input is supported, which
        # includes a scalar / single-element tensor.
        _validate_input((torch.tensor([-1.0, 1.0]),), (torch.tensor([-2.0]),))
        _validate_input((torch.tensor([-1.0]),), (torch.tensor([-2.0]),))
        _validate_input(
            (torch.tensor([-1.0]),), (torch.tensor([-2.0]),), method="gausslegendre"
        )

    def test_validate_nt_type(self) -> None:
        with self.assertRaises(
            AssertionError,
        ) as err:
            _validate_noise_tunnel_type("abc", SUPPORTED_NOISE_TUNNEL_TYPES)
        self.assertIn(
            "Noise types must be either `smoothgrad`, `smoothgrad_sq` or `vargrad`.",
            str(err.exception),
        )

        _validate_noise_tunnel_type("smoothgrad", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("smoothgrad_sq", SUPPORTED_NOISE_TUNNEL_TYPES)
        _validate_noise_tunnel_type("vargrad", SUPPORTED_NOISE_TUNNEL_TYPES)
