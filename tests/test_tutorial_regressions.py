#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from pathlib import Path
from typing import Any


TUTORIALS_PATH = Path(__file__).resolve().parents[1] / "tutorials"


def _notebook_source(name: str) -> str:
    with open(TUTORIALS_PATH / name, encoding="utf-8") as notebook_file:
        notebook: dict[str, Any] = json.load(notebook_file)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def test_titanic_tutorial_models_return_logits_for_cross_entropy_loss() -> None:
    training_source = _notebook_source("Titanic_Basic_Interpret.ipynb")
    assert "nn.CrossEntropyLoss()" in training_source

    for notebook_name in (
        "Titanic_Basic_Interpret.ipynb",
        "Distributed_Attribution.ipynb",
    ):
        source = _notebook_source(notebook_name)

        assert "nn.Softmax" not in source
        assert "self.softmax" not in source
