# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import io
import tempfile
import unittest.mock
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
from torch.utils.data import DataLoader


class TestTracInShowProgress(BaseTest):
    """
    This tests that the progress bar correctly shows a "100%" message at some point in
    the relevant computations.  Progress bars are shown for calls to the `influence`
    method for all 3 modes.  This is why 3 different modes are tested, and the mode
    being tested is a parameter in the test.  `TracInCPFastRandProj.influence` is not
    tested, because none of its modes involve computations over the entire training
    dataset, so that no progress bar is shown (the computation is instead done in
    `TracInCPFastRandProj.__init__`.  TODO: add progress bar for computations done
    in `TracInCPFastRandProj.__init__`).
    """

    def _check_error_msg_multiplicity(
        self,
        mock_stderr: io.StringIO,
        msg: str,
        msg_multiplicity: int,
        greater_than: bool = True,
    ) -> None:
        """
        Checks that in `mock_stderr`, the error msg `msg` occurs `msg_multiplicity`
        times. If 'greater_than' is true, it checks that the `msg` occurs at least
        `msg_multiplicity` times. Otherwise, it checks that `msg` occurs exactly
        `msg_multiplicity` times. The reason to let `greater_than` as true by default
        is that tqdm sometimes displays the "100%" more than once for each progress bar
        because it may want to correct its estimation of it/s. In this case, the
        tqdm could remove the original "100%" and then re-display "100%" with the
        updated estimate of it/s.
        """
        output = mock_stderr.getvalue()
        actual_msg_multiplicity = output.count(msg)
        assert isinstance(actual_msg_multiplicity, int)
        error_msg = (
            f"Error in progress of batches with output looking for '{msg}'"
            f" at least {msg_multiplicity} times"
            f"(found {actual_msg_multiplicity}) in {repr(output)}"
        )
        if greater_than:
            self.assertGreaterEqual(
                actual_msg_multiplicity, msg_multiplicity, error_msg
            )
        else:
            self.assertEqual(
                actual_msg_multiplicity,
                msg_multiplicity,
                error_msg,
            )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `comprehension((reduction, constr, mode) for
    # generators(generator((reduction, constr) in
    # [("none", captum.testing.helpers.influence.common.DataInfluenceConstructor
    # (captum.influence._core.tracincp.TracInCP)),
    # ("sum", captum.testing.helpers.influence.common.DataInfluenceConstructor
    # (captum.influence._core.tracincp_fast_rand_proj.TracInCPFast))] if ),
    # generators(generator(mode in ["self influence by checkpoints",
    # "self influence by batches", "influence", "k-most"] if ))))`
    # to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        [
            (
                reduction,
                constr,
                mode,
            )
            for reduction, constr in [
                (
                    "none",
                    DataInfluenceConstructor(TracInCP),
                ),
                (
                    "sum",
                    DataInfluenceConstructor(TracInCPFast),
                ),
            ]
            for mode in [
                "self influence by checkpoints",
                "self influence by batches",
                "influence",
                "k-most",
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_show_progress(
        self,
        reduction: str,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
        mode: str,
    ) -> None:

        with unittest.mock.patch("sys.stderr", new_callable=io.StringIO) as mock_stderr:

            with tempfile.TemporaryDirectory() as tmpdir:

                batch_size = 5

                (
                    net,
                    train_dataset,
                    test_samples,
                    test_labels,
                ) = get_random_model_and_data(
                    tmpdir, unpack_inputs=False, return_test_data=True
                )

                self.assertTrue(isinstance(reduction, str))
                criterion = nn.MSELoss(reduction=reduction)

                self.assertTrue(callable(tracin_constructor))
                tracin = tracin_constructor(
                    net,
                    train_dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )

                if mode == "self influence by checkpoints":
                    # this tests progress for computing self influence scores, when
                    # `outer_loop_by_checkpoints` is True. In this case, we should see a
                    # single outer progress bar over checkpoints, and for every
                    # checkpoints, a separate progress bar over batches
                    # pyre-fixme[16]: `object` has no attribute `self_influence`.
                    tracin.self_influence(
                        DataLoader(train_dataset, batch_size=batch_size),
                        show_progress=True,
                        outer_loop_by_checkpoints=True,
                    )

                    # We are showing nested progress bars for the `self_influence`
                    # method, with the outer progress bar over checkpoints, and
                    # the inner progress bar over batches. First, we check that
                    # the outer progress bar reaches 100% once
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            # pyre-fixme[16]: `object` has no attribute `get_name`.
                            f"Using {tracin.get_name()} to compute self influence. "
                            "Processing checkpoint: 100%"
                        ),
                        1,
                    )
                    # Second, we check that the inner progress bar reaches 100%
                    # once for each checkpoint in `tracin.checkpoints`
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            f"Using {tracin.get_name()} to compute self influence. "
                            "Processing batch: 100%"
                        ),
                        # pyre-fixme[16]: `object` has no attribute `checkpoints`.
                        len(tracin.checkpoints),
                    )
                elif mode == "self influence by batches":
                    # This tests progress for computing self influence scores, when
                    # `outer_loop_by_checkpoints` is False. In this case, we should see
                    # a single outer progress bar over batches.
                    tracin.self_influence(
                        DataLoader(train_dataset, batch_size=batch_size),
                        show_progress=True,
                        outer_loop_by_checkpoints=False,
                    )
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            f"Using {tracin.get_name()} to compute self influence. "
                            "Processing batch: 100%"
                        ),
                        1,
                    )
                elif mode == "influence":

                    # pyre-fixme[16]: `object` has no attribute `influence`.
                    tracin.influence(
                        (test_samples, test_labels),
                        k=None,
                        show_progress=True,
                    )
                    # Since the computation iterates once over training batches, we
                    # check that the progress bar over batches reaches 100% once
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            f"Using {tracin.get_name()} to compute influence "
                            "for training batches: 100%"
                        ),
                        1,
                    )
                elif mode == "k-most":

                    tracin.influence(
                        (test_samples, test_labels),
                        k=2,
                        proponents=True,
                        show_progress=True,
                    )

                    # Since the computation iterates once over training batches, we
                    # check that the progress bar over batches reaches 100% once, and
                    # that the message is specific for finding proponents.
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            f"Using {tracin.get_name()} to perform computation for "
                            "getting proponents. Processing training batches: 100%"
                        ),
                        1,
                    )
                    mock_stderr.seek(0)
                    mock_stderr.truncate(0)

                    tracin.influence(
                        (test_samples, test_labels),
                        k=2,
                        proponents=False,
                        show_progress=True,
                    )

                    # Since the computation iterates once over training batches, we
                    # check that the progress bar over batches reaches 100% once, and
                    # that the message is specific for finding opponents.
                    self._check_error_msg_multiplicity(
                        mock_stderr,
                        (
                            f"Using {tracin.get_name()} to perform computation for "
                            "getting opponents. Processing training batches: 100%"
                        ),
                        1,
                    )
                else:
                    raise Exception("unknown test mode")

                mock_stderr.seek(0)
                mock_stderr.truncate(0)
