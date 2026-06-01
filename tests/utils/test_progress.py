#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import io
import unittest
import unittest.mock

from captum._utils.progress import NullProgress, progress
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_nullprogress(self, mock_stderr) -> None:
        count = 0
        with NullProgress(["x", "y", "z"]) as np:
            for _ in np:
                for _ in NullProgress([1, 2, 3]):
                    count += 1

        self.assertEqual(count, 9)
        output = mock_stderr.getvalue()
        self.assertEqual(output, "")

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_nested_progress_tqdm(self, mock_stderr) -> None:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Skipping tqdm test, tqdm not available.")

        parent_data = ["x", "y", "z"]
        test_data = [1, 2, 3]
        with progress(parent_data, desc="parent progress") as parent:
            for item in parent:
                for _ in progress(test_data, desc=f"test progress {item}"):
                    pass
        output = mock_stderr.getvalue()
        self.assertIn("parent progress:", output)
        for item in parent_data:
            self.assertIn(f"test progress {item}:", output)

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_progress_tqdm(self, mock_stderr) -> None:
        try:
            import tqdm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("Skipping tqdm test, tqdm not available.")

        test_data = [1, 3, 5]

        progressed = progress(test_data, desc="test progress")
        assert list(progressed) == test_data
        assert "test progress: " in mock_stderr.getvalue()
