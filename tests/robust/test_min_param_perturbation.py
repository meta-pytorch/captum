#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import cast, List

import torch
from captum.robust import MinParamPerturbation
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel, BasicModel_MultiLayer
from torch import Tensor


def inp_subtract(inp: Tensor, ind: int = 0, add_arg: int = 0) -> Tensor:
    inp_repeat = 1.0 * inp
    inp_repeat[0][ind] -= add_arg
    return inp_repeat


def add_char(inp: List[str], ind: int = 0, char_val: int = 0) -> List[str]:
    list_copy = list(inp)
    list_copy[ind] = chr(122 - char_val) + list_copy[ind]
    return list_copy


def add_char_batch(inp: List[List[str]], ind: int, char_val: int) -> List[List[str]]:
    return [add_char(elem, ind, char_val) for elem in inp]


def text_preproc_fn(inp: List[str]) -> Tensor:
    return torch.tensor([float(ord(elem[0])) for elem in inp]).unsqueeze(0)


def batch_text_preproc_fn(inp: List[List[str]]) -> Tensor:
    return torch.cat([text_preproc_fn(elem) for elem in inp])


def alt_correct_fn(model_out: Tensor, target: int, threshold: float) -> bool:
    if all(model_out[:, target] > threshold):
        return True
    return False


class Test(BaseTest):
    def test_minimal_pert_basic_linear(self) -> None:
        model = BasicModel()
        inp = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        minimal_pert = MinParamPerturbation(
            forward_func=lambda x: model(x)
            + torch.tensor([[0.000001, 0.0, 0.0, 0.0, 0.0]]),
            attack=inp_subtract,
            arg_name="add_arg",
            arg_min=0.0,
            arg_max=1000.0,
            arg_step=1.0,
        )
        target_inp, pert = minimal_pert.evaluate(
            inp, target=0, attack_kwargs={"ind": 0}
        )
        self.assertAlmostEqual(cast(float, pert), 2.0)
        assertTensorAlmostEqual(
            self, target_inp, torch.tensor([[0.0, -9.0, 9.0, 1.0, -3.0]])
        )

    def test_minimal_pert_basic_binary(self) -> None:
        model = BasicModel()
        inp = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        minimal_pert = MinParamPerturbation(
            forward_func=lambda x: model(x)
            + torch.tensor([[0.000001, 0.0, 0.0, 0.0, 0.0]]),
            attack=inp_subtract,
            arg_name="add_arg",
            arg_min=0.0,
            arg_max=1000.0,
            arg_step=1.0,
            mode="binary",
        )
        target_inp, pert = minimal_pert.evaluate(
            inp,
            target=0,
            attack_kwargs={"ind": 0},
            perturbations_per_eval=10,
        )
        self.assertAlmostEqual(cast(float, pert), 2.0)
        assertTensorAlmostEqual(
            self, target_inp, torch.tensor([[0.0, -9.0, 9.0, 1.0, -3.0]])
        )

    def test_minimal_pert_preproc(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = ["abc", "zyd", "ghi"]
        minimal_pert = MinParamPerturbation(
            forward_func=model,
            attack=add_char,
            arg_name="char_val",
            arg_min=0,
            arg_max=26,
            arg_step=1,
            preproc_fn=text_preproc_fn,
            apply_before_preproc=True,
        )
        target_inp, pert = minimal_pert.evaluate(
            text_inp, target=1, attack_kwargs={"ind": 1}
        )
        self.assertEqual(pert, None)
        self.assertEqual(target_inp, None)

    def test_minimal_pert_alt_correct(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = ["abc", "zyd", "ghi"]
        minimal_pert = MinParamPerturbation(
            forward_func=model,
            attack=add_char,
            arg_name="char_val",
            arg_min=0,
            arg_max=26,
            arg_step=1,
            preproc_fn=text_preproc_fn,
            apply_before_preproc=True,
            correct_fn=alt_correct_fn,
            num_attempts=5,
        )
        expected_list = ["abc", "ezyd", "ghi"]

        target_inp, pert = minimal_pert.evaluate(
            text_inp,
            target=1,
            attack_kwargs={"ind": 1},
            correct_fn_kwargs={"threshold": 1200},
            perturbations_per_eval=5,
        )
        self.assertEqual(pert, 21)
        self.assertListEqual(target_inp, expected_list)

        target_inp_single, pert_single = minimal_pert.evaluate(
            text_inp,
            target=1,
            attack_kwargs={"ind": 1},
            correct_fn_kwargs={"threshold": 1200},
        )
        self.assertEqual(pert_single, 21)
        self.assertListEqual(target_inp_single, expected_list)

    def test_minimal_pert_additional_forward_args(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = [["abc", "zyd", "ghi"], ["abc", "uyd", "ghi"]]
        additional_forward_args = torch.ones((2, 3)) * -97

        model = BasicModel_MultiLayer()
        minimal_pert = MinParamPerturbation(
            forward_func=model,
            attack=add_char_batch,
            arg_name="char_val",
            arg_min=0,
            arg_max=26,
            arg_step=1,
            preproc_fn=batch_text_preproc_fn,
            apply_before_preproc=True,
            correct_fn=alt_correct_fn,
        )
        expected_list = [["abc", "uzyd", "ghi"], ["abc", "uuyd", "ghi"]]

        target_inp, pert = minimal_pert.evaluate(
            text_inp,
            target=1,
            attack_kwargs={"ind": 1},
            correct_fn_kwargs={"threshold": 100},
            perturbations_per_eval=15,
            additional_forward_args=(additional_forward_args,),
        )
        self.assertEqual(pert, 5)
        self.assertListEqual(target_inp, expected_list)

        target_inp_single, pert_single = minimal_pert.evaluate(
            text_inp,
            target=1,
            attack_kwargs={"ind": 1},
            correct_fn_kwargs={"threshold": 100},
            additional_forward_args=(additional_forward_args,),
        )
        self.assertEqual(pert_single, 5)
        self.assertListEqual(target_inp_single, expected_list)

    def test_minimal_pert_tuple_test(self) -> None:
        model = BasicModel_MultiLayer()
        text_inp = (
            [["abc", "zyd", "ghi"], ["abc", "uyd", "ghi"]],
            torch.ones((2, 3)) * -97,
        )

        model = BasicModel_MultiLayer()
        minimal_pert = MinParamPerturbation(
            forward_func=lambda x: model(*x),
            attack=lambda x, ind, char_val: (add_char_batch(x[0], ind, char_val), x[1]),
            arg_name="char_val",
            arg_min=0,
            arg_max=26,
            arg_step=1,
            preproc_fn=lambda x: (batch_text_preproc_fn(x[0]), x[1]),
            apply_before_preproc=True,
            correct_fn=alt_correct_fn,
        )
        expected_list = [["abc", "uzyd", "ghi"], ["abc", "uuyd", "ghi"]]

        target_inp, pert = minimal_pert.evaluate(
            text_inp,
            target=1,
            attack_kwargs={"ind": 1},
            correct_fn_kwargs={"threshold": 100},
            perturbations_per_eval=15,
        )
        self.assertEqual(pert, 5)
        self.assertListEqual(target_inp[0], expected_list)
