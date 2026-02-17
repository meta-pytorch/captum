# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-unsafe

# !/usr/bin/env python3

import torch
from captum.influence._utils.common import _jacobian_loss_wrt_inputs
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual


class TestCommon(BaseTest):
    def setUp(self) -> None:
        super().setUp()

    def test_jacobian_loss_wrt_inputs(self) -> None:
        with self.assertRaises(ValueError) as err:
            _jacobian_loss_wrt_inputs(
                torch.nn.BCELoss(reduction="sum"),
                torch.tensor([-1.0, 1.0]),
                torch.tensor([1.0]),
                True,
                "",
            )
        self.assertEqual(
            "`` is not a valid value for reduction_type. "
            "Must be either 'sum' or 'mean'.",
            str(err.exception),
        )

        with self.assertRaises(AssertionError) as err:
            _jacobian_loss_wrt_inputs(
                torch.nn.BCELoss(reduction="sum"),
                torch.tensor([-1.0, 1.0]),
                torch.tensor([1.0]),
                True,
                "mean",
            )
        self.assertEqual(
            "loss_fn.reduction `sum` does not matchreduction type `mean`."
            " Please ensure they are matching.",
            str(err.exception),
        )

        res = _jacobian_loss_wrt_inputs(
            torch.nn.BCELoss(reduction="sum"),
            torch.tensor([0.5, 1.0]),
            torch.tensor([0.0, 1.0]),
            True,
            "sum",
        )
        assertTensorAlmostEqual(
            self, res, torch.tensor([2.0, 0.0]), delta=0.0, mode="sum"
        )
