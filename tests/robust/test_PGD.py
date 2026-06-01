#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import torch
from captum.robust import PGD
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel_MultiLayer,
)
from torch.nn import CrossEntropyLoss


class Test(BaseTest):
    def test_attack_nontargeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.25, 0.1, 2, 4)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[2.0, -9.0, 9.0, 1.0, -2.8]],
            delta=0.01,
            mode="max",
        )

    def test_attack_targeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]], requires_grad=True)
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.2, 0.1, 3, 3, targeted=True)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[9.0, 10.0, -6.0, -1.2]],
            delta=0.01,
            mode="max",
        )

    def test_attack_l2norm(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]], requires_grad=True)
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.2, 0.1, 3, 2, targeted=True, norm="L2")
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[9.0, 10.0, -6.2, -1.0]],
            delta=0.01,
            mode="max",
        )

    def test_attack_multiinput(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        adv = PGD(model)
        perturbed_input = adv.perturb((input1, input2), 0.25, 0.1, 3, 0, norm="L2")
        answer = ([[3.75, -1.0], [2.75, 10.0]], [[2.25, -5.0], [-2.0, 1.0]])
        for i in range(len(perturbed_input)):
            assertTensorAlmostEqual(
                self,
                perturbed_input[i],
                answer[i],
                delta=0.01,
                mode="max",
            )

    def test_attack_3dimensional_input(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.25, 0.1, 3, (0, 1))
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -3.75], [10.0, 5.0]]],
            delta=0.01,
            mode="max",
        )

    def test_attack_loss_defined(self) -> None:
        model = BasicModel_MultiLayer()
        add_input = torch.tensor([[-1.0, 2.0, 2.0]])
        input = torch.tensor([[1.0, 6.0, -3.0]])
        labels = torch.tensor([0])
        loss_func = CrossEntropyLoss(reduction="none")
        adv = PGD(model, loss_func)
        perturbed_input = adv.perturb(
            input, 0.25, 0.1, 3, labels, additional_forward_args=(add_input,)
        )
        assertTensorAlmostEqual(
            self, perturbed_input, [[1.0, 6.0, -3.0]], delta=0.01, mode="max"
        )

    def test_attack_random_start(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.25, 0.1, 0, 4, random_start=True)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[2.0, -9.0, 9.0, 1.0, -3.0]],
            delta=0.25,
            mode="max",
        )
        perturbed_input = adv.perturb(
            input, 0.25, 0.1, 0, 4, norm="L2", random_start=True
        )
        norm = torch.norm((perturbed_input - input).squeeze()).numpy()
        self.assertLessEqual(norm, 0.25)

    def test_attack_masked_nontargeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        mask = torch.tensor([[1, 1, 0, 0, 0]])
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.25, 0.1, 2, 4, mask=mask)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[2.0, -9.0, 9.0, 1.0, -3.0]],
            delta=0.01,
            mode="max",
        )

    def test_attack_masked_targeted(self) -> None:
        model = BasicModel()
        input = torch.tensor([[9.0, 10.0, -6.0, -1.0]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0]])
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.2, 0.1, 3, 3, targeted=True, mask=mask)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[9.0, 10.0, -6.0, -1.0]],
            delta=0.01,
            mode="max",
        )

    def test_attack_masked_multiinput(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        mask1 = torch.tensor([[1, 1], [0, 0]])
        mask2 = torch.tensor([[0, 1], [0, 1]])
        adv = PGD(model)
        perturbed_input = adv.perturb(
            (input1, input2), 0.25, 0.1, 3, 0, norm="L2", mask=(mask1, mask2)
        )
        answer = ([[3.75, -1.0], [3.0, 10.0]], [[2.0, -5.0], [-2.0, 1.0]])
        for i in range(len(perturbed_input)):
            assertTensorAlmostEqual(
                self,
                perturbed_input[i],
                answer[i],
                delta=0.01,
                mode="max",
            )

    def test_attack_masked_random_start(self) -> None:
        model = BasicModel()
        input = torch.tensor([[2.0, -9.0, 9.0, 1.0, -3.0]])
        mask = torch.tensor([[1, 0, 1, 0, 1]])
        adv = PGD(model)
        perturbed_input = adv.perturb(
            input, 0.25, 0.1, 0, 4, random_start=True, mask=mask
        )
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[2.0, -9.0, 9.0, 1.0, -3.0]],
            delta=0.25,
            mode="max",
        )
        perturbed_input = adv.perturb(
            input, 0.25, 0.1, 0, 4, norm="L2", random_start=True, mask=mask
        )
        norm = torch.norm((perturbed_input - input).squeeze()).numpy()
        self.assertLessEqual(norm, 0.25)

    def test_attack_masked_3dimensional_input(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        mask = torch.tensor([[[1, 0], [0, 1]], [[1, 0], [1, 1]]])
        adv = PGD(model)
        perturbed_input = adv.perturb(input, 0.25, 0.1, 3, (0, 1), mask=mask)
        assertTensorAlmostEqual(
            self,
            perturbed_input,
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]],
            delta=0.01,
            mode="max",
        )

    def test_attack_masked_loss_defined(self) -> None:
        model = BasicModel_MultiLayer()
        add_input = torch.tensor([[-1.0, 2.0, 2.0]])
        input = torch.tensor([[1.0, 6.0, -3.0]])
        mask = torch.tensor([[0, 1, 0]])
        labels = torch.tensor([0])
        loss_func = CrossEntropyLoss(reduction="none")
        adv = PGD(model, loss_func)
        perturbed_input = adv.perturb(
            input, 0.25, 0.1, 3, labels, additional_forward_args=(add_input,), mask=mask
        )
        assertTensorAlmostEqual(
            self, perturbed_input, [[1.0, 6.0, -3.0]], delta=0.01, mode="max"
        )
