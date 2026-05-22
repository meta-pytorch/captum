#!/usr/bin/env python3

# pyre-strict

import json
from pathlib import Path
from typing import Any, Callable

import torch


TUTORIALS_PATH = Path(__file__).resolve().parents[1] / "tutorials"


def _notebook_cells(name: str) -> list[str]:
    with open(TUTORIALS_PATH / name, encoding="utf-8") as notebook_file:
        notebook: dict[str, Any] = json.load(notebook_file)
    return ["".join(cell.get("source", [])) for cell in notebook["cells"]]


def _exec_bert_answer_span_cell(
    notebook_name: str,
    predict: Callable[..., Any],
) -> dict[str, Any]:
    answer_cells = [
        cell for cell in _notebook_cells(notebook_name) if "valid_span_mask" in cell
    ]
    assert len(answer_cells) == 1
    namespace: dict[str, Any] = {
        "torch": torch,
        "predict": predict,
        "input_ids": torch.tensor([[0, 1, 2]]),
        "token_type_ids": None,
        "position_ids": None,
        "attention_mask": None,
        "question": "question?",
        "all_tokens": ["zero", "one", "two"],
    }

    exec(answer_cells[0], namespace)
    return namespace


def test_bert_squad_tutorials_select_valid_answer_spans() -> None:
    # The highest start score is after the highest end score. The old notebook
    # logic selected those positions independently and produced an empty span.
    start_scores = torch.tensor([[0.0, 0.0, 10.0]])
    end_scores = torch.tensor([[9.0, 0.0, 0.0]])

    part1_namespace = _exec_bert_answer_span_cell(
        "Bert_SQUAD_Interpret.ipynb",
        lambda *args, **kwargs: (start_scores, end_scores),
    )
    part2_namespace = _exec_bert_answer_span_cell(
        "Bert_SQUAD_Interpret2.ipynb",
        lambda *args, **kwargs: (start_scores, end_scores, None),
    )

    for namespace in (part1_namespace, part2_namespace):
        assert namespace["start_idx"] <= namespace["end_idx"]
        assert namespace["end_idx"] == 2
