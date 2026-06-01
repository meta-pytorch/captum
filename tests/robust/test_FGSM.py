#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.robust import FGSM
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel_MultiLayer,
)
from torch import Tensor
from torch.nn import CrossEntropyLoss


class Test(BaseTest):
    def test_attack_nontargeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        self._FGSM_assert(model, input, 1, 0.1, [[2.0, -8.9, 9.0, 1.0, -3.0]])

    def test_attack_targeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]])
        self._FGSM_assert(
            model, input, 3, 0.2, [[9.0, 10.0, -6.0, -1.2]], targeted=True
        )

    def test_attack_multiinput(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        self._FGSM_assert(
            model,
            (input1, input2),
            0,
            0.25,
            ([[3.75, -1.0], [2.75, 10.0]], [[2.25, -5.0], [-2.0, 1.0]]),
        )

    def test_attack_label_list(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        self._FGSM_assert(
            model,
            (input1, input2),
            [0, 1],
            0.1,
            ([[3.9, -1.0], [3.0, 9.9]], [[2.1, -5.0], [-2.0, 1.1]]),
        )

    def test_attack_label_tensor(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        labels = torch.tensor([0, 1])
        self._FGSM_assert(
            model,
            (input1, input2),
            labels,
            0.1,
            ([[4.1, -1.0], [3.0, 10.1]], [[1.9, -5.0], [-2.0, 0.9]]),
            targeted=True,
        )

    def test_attack_label_tuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        labels = (0, 1)
        self._FGSM_assert(
            model,
            input,
            labels,
            0.1,
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -3.9], [10.0, 5.0]]],
        )

    def test_attack_label_listtuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        labels: List[Tuple[int, ...]] = [(1, 1), (0, 1)]
        self._FGSM_assert(
            model,
            input,
            labels,
            0.1,
            [[[4.0, 2.0], [-1.0, -1.9]], [[3.0, -3.9], [10.0, 5.0]]],
        )

    def test_attack_additional_inputs(self) -> None:
        model = BasicModel_MultiLayer()
        add_input = torch.tensor([[-1.0, 2.0, 2.0]], requires_grad=True)
        input = torch.tensor([[1.0, 6.0, -3.0]], requires_grad=True)
        self._FGSM_assert(
            model, input, 0, 0.2, [[0.8, 5.8, -3.2]], additional_inputs=(add_input,)
        )
        self._FGSM_assert(
            model, input, 0, 0.2, [[0.8, 5.8, -3.2]], additional_inputs=add_input
        )

    def test_attack_loss_defined(self) -> None:
        model = BasicModel_MultiLayer()
        add_input = torch.tensor([[-1.0, 2.0, 2.0]])
        input = torch.tensor([[1.0, 6.0, -3.0]])
        labels = torch.tensor([0])
        loss_func = CrossEntropyLoss(reduction="none")
        adv = FGSM(model, loss_func)
        perturbed_input = adv.perturb(
            input, 0.2, labels, additional_forward_args=(add_input,)
        )
        assertTensorAlmostEqual(
            self, perturbed_input, [[1.0, 6.0, -3.0]], delta=0.01, mode="max"
        )

    def test_attack_bound(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]])
        self._FGSM_assert(
            model,
            input,
            3,
            0.2,
            [[5.0, 5.0, -5.0, -1.2]],
            targeted=True,
            lower_bound=-5.0,
            upper_bound=5.0,
        )

    def test_attack_masked_tensor(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]], requires_grad=True)
        mask = torch.tensor([[1, 0, 0, 1, 1]])
        self._FGSM_assert(
            model, input, 1, 0.1, [[2.0, -9.0, 9.0, 1.0, -3.0]], mask=mask
        )

    def test_attack_masked_multiinput(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        mask1 = torch.tensor([[1, 0], [1, 0]])
        mask2 = torch.tensor([[0, 0], [0, 0]])
        self._FGSM_assert(
            model,
            (input1, input2),
            0,
            0.25,
            ([[3.75, -1.0], [2.75, 10.0]], [[2.0, -5.0], [-2.0, 1.0]]),
            mask=(mask1, mask2),
        )

    def test_attack_masked_loss_defined(self) -> None:
        model = BasicModel_MultiLayer()
        add_input = torch.tensor([[-1.0, 2.0, 2.0]])
        input = torch.tensor([[1.0, 6.0, -3.0]])
        labels = torch.tensor([0])
        mask = torch.tensor([[0, 0, 1]])
        loss_func = CrossEntropyLoss(reduction="none")
        adv = FGSM(model, loss_func)
        perturbed_input = adv.perturb(
            input, 0.2, labels, additional_forward_args=(add_input,), mask=mask
        )
        assertTensorAlmostEqual(
            self, perturbed_input, [[1.0, 6.0, -3.0]], delta=0.01, mode="max"
        )

    def test_attack_masked_bound(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]])
        mask = torch.tensor([[1, 0, 1, 0]])
        self._FGSM_assert(
            model,
            input,
            3,
            0.2,
            [[5.0, 5.0, -5.0, -1.0]],
            targeted=True,
            lower_bound=-5.0,
            upper_bound=5.0,
            mask=mask,
        )

    def _FGSM_assert(
        self,
        model: Callable,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: Any,
        epsilon: float,
        answer: Union[List, Tuple[List, ...]],
        targeted: bool = False,
        additional_inputs: Any = None,
        lower_bound: float = float("-inf"),
        upper_bound: float = float("inf"),
        mask: Optional[TensorOrTupleOfTensorsGeneric] = None,
    ) -> None:
        adv = FGSM(model, lower_bound=lower_bound, upper_bound=upper_bound)
        perturbed_input = adv.perturb(
            inputs, epsilon, target, additional_inputs, targeted, mask
        )
        if isinstance(perturbed_input, Tensor):
            assertTensorAlmostEqual(
                self, perturbed_input, answer, delta=0.01, mode="max"
            )
        else:
            for i in range(len(perturbed_input)):
                assertTensorAlmostEqual(
                    self, perturbed_input[i], answer[i], delta=0.01, mode="max"
                )
