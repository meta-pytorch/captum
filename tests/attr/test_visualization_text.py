#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from captum.attr._utils.visualization import format_word_importances
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_format_word_importances_escapes_html_tokens(self) -> None:
        html = format_word_importances(
            ["hello", "entrar<!--#exec", "cmd", '"ls', "-->"],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        )

        self.assertNotIn("<!--#exec", html)
        self.assertNotIn("-->", html)
        self.assertIn("&lt;!--#exec", html)
        self.assertIn("&quot;ls", html)
        self.assertIn("--&gt;", html)
