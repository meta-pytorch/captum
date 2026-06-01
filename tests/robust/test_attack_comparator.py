#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import collections
from typing import Dict, List, Tuple, Union

import torch
from captum.robust import AttackComparator, FGSM
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel, BasicModel_MultiLayer
from torch import Tensor


def float_metric(model_out: Tensor, target: int) -> Tensor:
    return model_out[:, target]


ModelResult = collections.namedtuple("ModelResult", "accuracy output")


def tuple_metric(model_out: Tensor, target: int, named_tuple: bool = False):
    _, pred = torch.max(model_out, dim=1)
    acc = (pred == target).float()
    output = model_out[:, target]
    if named_tuple:
        return ModelResult(
            accuracy=acc.item() if acc.numel() == 1 else acc,
            output=output.item() if output.numel() == 1 else output,
        )
    return (acc, output)


def drop_column_perturb(inp: Tensor, column: int) -> Tensor:
    mask = torch.ones_like(inp)
    mask[:, column] = 0.0
    return mask * inp


def text_preproc_fn(inp: List[str]) -> Tensor:
    return torch.tensor([float(ord(elem[0])) for elem in inp]).unsqueeze(0)


def batch_text_preproc_fn(inp: List[List[str]]) -> Tensor:
    return torch.cat([text_preproc_fn(elem) for elem in inp])


def string_perturb(inp: List[str]) -> List[str]:
    return ["a" + elem for elem in inp]


def string_batch_perturb(inp: List[List[str]]) -> List[List[str]]:
    return [string_perturb(elem) for elem in inp]


class SamplePerturb:
    def __init__(self) -> None:
        self.count = 0

    def perturb(self, inp: Tensor) -> Tensor:
        mask = torch.ones_like(inp)
        mask[:, self.count % mask.shape[1]] = 0.0
        self.count += 1
        return mask * inp


class Test(BaseTest):
    def test_attack_comparator_basic(self) -> None:
        model = BasicModel()
        inp = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        attack_comp = AttackComparator(
            forward_func=lambda x: model(x)
            + torch.tensor([[0.000001, 0.0, 0.0, 0.0, 0.0]]),
            metric=tuple_metric,
        )
        attack_comp.add_attack(
            drop_column_perturb,
            name="first_column_perturb",
            attack_kwargs={"column": 0},
        )
        attack_comp.add_attack(
            drop_column_perturb,
            name="last_column_perturb",
            attack_kwargs={"column": -1},
        )
        attack_comp.add_attack(
            FGSM(model),
            attack_kwargs={"epsilon": 0.5},
            additional_attack_arg_names=["target"],
        )
        batch_results = attack_comp.evaluate(inp, target=0, named_tuple=True)
        expected_first_results = {
            "Original": (1.0, 1.0),
            "first_column_perturb": {"mean": (0.0, 0.0)},
            "last_column_perturb": {"mean": (1.0, 1.0)},
            "FGSM": {"mean": (1.0, 1.0)},
        }
        self._compare_results(batch_results, expected_first_results)

        alt_inp = torch.tensor([[1.0, 2.0, -3.0, 4.0, -5.0]])

        second_batch_results = attack_comp.evaluate(alt_inp, target=4, named_tuple=True)
        expected_second_results = {
            "Original": (0.0, -5.0),
            "first_column_perturb": {"mean": (0.0, -5.0)},
            "last_column_perturb": {"mean": (0.0, 0.0)},
            "FGSM": {"mean": (0.0, -4.5)},
        }
        self._compare_results(second_batch_results, expected_second_results)

        expected_summary_results = {
            "Original": {"mean": (0.5, -2.0)},
            "first_column_perturb": {"mean": (0.0, -2.5)},
            "last_column_perturb": {"mean": (0.5, 0.5)},
            "FGSM": {"mean": (0.5, -1.75)},
        }
        self._compare_results(attack_comp.summary(), expected_summary_results)

    def test_attack_comparator_with_preproc(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = ["abc", "zyd", "ghi"]
        attack_comp = AttackComparator(
            forward_func=model, metric=tuple_metric, preproc_fn=text_preproc_fn
        )
        attack_comp.add_attack(
            SamplePerturb().perturb,
            name="Sequence Column Perturb",
            num_attempts=5,
            apply_before_preproc=False,
        )
        attack_comp.add_attack(
            string_perturb,
            name="StringPerturb",
            apply_before_preproc=True,
        )
        batch_results = attack_comp.evaluate(
            text_inp, target=0, named_tuple=True, perturbations_per_eval=3
        )
        expected_first_results = {
            "Original": (0.0, 1280.0),
            "Sequence Column Perturb": {
                "mean": (0.0, 847.2),
                "max": (0.0, 892.0),
                "min": (0.0, 792.0),
            },
            "StringPerturb": {"mean": (0.0, 1156.0)},
        }
        self._compare_results(batch_results, expected_first_results)

        expected_summary_results = {
            "Original": {"mean": (0.0, 1280.0)},
            "Sequence Column Perturb Mean Attempt": {"mean": (0.0, 847.2)},
            "Sequence Column Perturb Min Attempt": {"mean": (0.0, 792.0)},
            "Sequence Column Perturb Max Attempt": {"mean": (0.0, 892.0)},
            "StringPerturb": {"mean": (0.0, 1156.0)},
        }
        self._compare_results(attack_comp.summary(), expected_summary_results)

    def test_attack_comparator_with_additional_args(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = [["abc", "zyd", "ghi"], ["mnop", "qrs", "Tuv"]]
        additional_forward_args = torch.ones((2, 3)) * -97
        attack_comp = AttackComparator(
            forward_func=model, metric=tuple_metric, preproc_fn=batch_text_preproc_fn
        )
        attack_comp.add_attack(
            SamplePerturb().perturb,
            name="Sequence Column Perturb",
            num_attempts=5,
            apply_before_preproc=False,
        )
        attack_comp.add_attack(
            string_batch_perturb,
            name="StringPerturb",
            apply_before_preproc=True,
        )
        batch_results = attack_comp.evaluate(
            text_inp,
            additional_forward_args=additional_forward_args,
            target=0,
            named_tuple=True,
            perturbations_per_eval=2,
        )
        expected_first_results = {
            "Original": ([0.0, 0.0], [116.0, 52.0]),
            "Sequence Column Perturb": {
                "mean": ([0.0, 0.0], [-1.0, -1.0]),
                "max": ([0.0, 0.0], [-1.0, -1.0]),
                "min": ([0.0, 0.0], [-1.0, -1.0]),
            },
            "StringPerturb": {"mean": ([0.0, 0.0], [2.0, 2.0])},
        }
        self._compare_results(batch_results, expected_first_results)
        expected_summary_results = {
            "Original": {
                "mean": (0.0, 84.0),
            },
            "Sequence Column Perturb Mean Attempt": {"mean": (0.0, -1.0)},
            "Sequence Column Perturb Min Attempt": {"mean": (0.0, -1.0)},
            "Sequence Column Perturb Max Attempt": {"mean": (0.0, -1.0)},
            "StringPerturb": {"mean": (0.0, 2.0)},
        }
        self._compare_results(attack_comp.summary(), expected_summary_results)

        attack_comp.reset()
        self.assertEqual(len(attack_comp.summary()), 0)

    def _compare_results(
        self, obtained: Union[Dict, Tuple, Tensor], expected: Union[Dict, Tuple, Tensor]
    ) -> None:
        if isinstance(expected, dict):
            self.assertIsInstance(obtained, dict)
            for key in expected:
                self._compare_results(obtained[key], expected[key])
        elif isinstance(expected, tuple):
            self.assertIsInstance(obtained, tuple)
            for i in range(len(expected)):
                self._compare_results(obtained[i], expected[i])
        else:
            assertTensorAlmostEqual(self, obtained, expected)
