#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import tempfile
from typing import Any

import torch
from captum._utils.av import AV
from captum.influence._utils.common import _load_flexible_state_dict
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_av_dataset_loads_with_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor = torch.randn(2, 3)
            AV.save(tmpdir, "model", "identifier", "layer", tensor, "0")
            dataset = AV.load(tmpdir, "model", "identifier", "layer")
            assert dataset is not None

            original_load = torch.load
            calls = []

            def load_with_call_record(*args: Any, **kwargs: Any) -> Any:
                calls.append(kwargs.get("weights_only"))
                return original_load(*args, **kwargs)

            try:
                torch.load = load_with_call_record  # type: ignore[method-assign]
                loaded = dataset[0]
            finally:
                torch.load = original_load  # type: ignore[method-assign]

            self.assertTrue(torch.equal(loaded, tensor))
            self.assertEqual(calls, [True])

    def test_flexible_state_dict_loads_with_weights_only(self) -> None:
        model = torch.nn.Linear(2, 1)
        state_dict = model.state_dict()
        calls = []

        original_load = torch.load

        def load_with_call_record(*args: Any, **kwargs: Any) -> Any:
            calls.append(kwargs.get("weights_only"))
            return state_dict

        try:
            torch.load = load_with_call_record  # type: ignore[method-assign]
            learning_rate = _load_flexible_state_dict(model, "unused.pt")
        finally:
            torch.load = original_load  # type: ignore[method-assign]

        self.assertEqual(learning_rate, 1.0)
        self.assertEqual(calls, [True])
