#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
import warnings
from pathlib import Path
from typing import Any

import torch

TUTORIALS_PATH = Path(__file__).resolve().parents[1] / "tutorials"


def _notebook_source(name: str) -> str:
    with open(TUTORIALS_PATH / name, encoding="utf-8") as notebook_file:
        notebook: dict[str, Any] = json.load(notebook_file)
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _notebook_code_cells(name: str) -> list[str]:
    with open(TUTORIALS_PATH / name, encoding="utf-8") as notebook_file:
        notebook: dict[str, Any] = json.load(notebook_file)
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


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


def test_forecast_components_tutorial_runs_and_its_checks_hold() -> None:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    namespace: dict[str, Any] = {"__name__": "__tutorial__"}
    with warnings.catch_warnings():
        # plt.show() is a no-op with the Agg backend
        warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
        for cell_source in _notebook_code_cells("Forecast_Components_Interpret.ipynb"):
            exec(
                compile(cell_source, "Forecast_Components_Interpret", "exec"), namespace
            )
    plt.close("all")

    # the head's bases are orthogonal and the dense layer converts exactly
    assert max(namespace["orthogonality"].values()) < 1e-10
    assert namespace["conversion_error"] < 1e-10
    # safe selection never worsens the validation RMSE of the converted dense model
    assert namespace["chosen"]["rmse"] <= namespace["r0"]

    # the components add up to the forecast
    out = namespace["out"]
    torch.testing.assert_close(
        out["trend"] + out["seasonal"] + out["residual"], out["forecast"]
    )

    # Integrated Gradients is complete for every component scalar
    for component in ("trend", "seasonal", "residual"):
        attr = namespace["attributions"][component]
        gap = namespace["gaps"][component]
        assert attr.shape == namespace["x_explain"].shape
        torch.testing.assert_close(
            attr.sum(dim=(1, 2)),
            gap,
            rtol=1e-4,
            atol=1e-6 * gap.abs().max().item(),
        )
