# pyre-unsafe
import tempfile
from typing import Callable

import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import TracInCPFast
from captum.testing.helpers import BaseTest
from captum.testing.helpers.influence.common import (
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
)
from parameterized import parameterized


class TestTracinValidator(BaseTest):

    param_list = [
        (
            "none",
            DataInfluenceConstructor(TracInCP, name="TracInCP"),
        ),
        (
            "mean",
            DataInfluenceConstructor(
                TracInCPFast,
                name="TracInCpFast",
            ),
        ),
    ]

    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(),
    )
    def test_tracin_require_inputs_dataset(
        self,
        reduction: str,
        tracin_constructor: Callable,
    ) -> None:
        """
        This test verifies that tracinCP and tracinCPFast
        influence methods required `inputs_dataset`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs=False)

            criterion = nn.MSELoss(reduction=reduction)

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                loss_fn=criterion,
                batch_size=1,
            )
            with self.assertRaisesRegex(AssertionError, "required."):
                tracin.influence(None, k=None)

    def test_tracincp_fast_rand_proj_inputs(self) -> None:
        """
        This test verifies that TracInCPFast should be initialized
        with a valid `final_fc_layer`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(tmpdir, unpack_inputs=False)

            with self.assertRaisesRegex(
                ValueError, 'Invalid final_fc_layer str: "invalid_layer" provided!'
            ):
                TracInCPFast(
                    net,
                    "invalid_layer",  # type: ignore
                    train_dataset,
                    tmpdir,
                    loss_fn=nn.MSELoss(),
                    batch_size=1,
                )

    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(),
    )
    def test_tracincp_input_checkpoints(
        self, reduction: str, tracin_constructor: Callable
    ) -> None:
        """
        This test verifies that tracinCP and tracinCPFast
        class should be initialized with valid `checkpoints`.
        """
        with tempfile.TemporaryDirectory() as invalid_tmpdir:
            with tempfile.TemporaryDirectory() as tmpdir:
                (
                    net,
                    train_dataset,
                    test_samples,
                    test_labels,
                ) = get_random_model_and_data(tmpdir, unpack_inputs=False)

            with self.assertRaisesRegex(
                ValueError, "Invalid checkpoints provided for TracIn class: "
            ):
                tracin_constructor(
                    net,
                    train_dataset,
                    invalid_tmpdir,
                    loss_fn=nn.MSELoss(),
                    batch_size=1,
                )
