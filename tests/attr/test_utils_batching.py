#!/usr/bin/env python3

# pyre-strict

from typing import cast, List, Tuple

import torch
from captum.attr._utils.batching import (
    _batched_generator,
    _batched_operator,
    _tuple_splice_range,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from torch import Tensor


class Test(BaseTest):
    def test_tuple_splice_range(self) -> None:
        test_tuple = (
            torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            "test",
            torch.tensor([[6, 7, 8], [0, 1, 2], [3, 4, 5]]),
        )
        spliced_tuple = _tuple_splice_range(test_tuple, 1, 3)
        assertTensorAlmostEqual(self, spliced_tuple[0], [[3, 4, 5], [6, 7, 8]])
        self.assertEqual(spliced_tuple[1], "test")
        assertTensorAlmostEqual(self, spliced_tuple[2], [[0, 1, 2], [3, 4, 5]])

    def test_tuple_splice_range_3d(self) -> None:
        test_tuple = (
            torch.tensor([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [6, 7, 8]]]),
            "test",
        )
        spliced_tuple = _tuple_splice_range(test_tuple, 1, 2)
        assertTensorAlmostEqual(self, spliced_tuple[0], [[[6, 7, 8], [6, 7, 8]]])
        self.assertEqual(spliced_tuple[1], "test")

    def test_batched_generator(self) -> None:
        def sample_operator(
            inputs: Tuple[Tensor, ...],
            additional_forward_args: Tuple[Tensor, int],
            target_ind: int,
            scale: int,
        ) -> Tuple[Tensor, Tensor, int]:
            inputs_sum = cast(Tensor, sum(inputs))
            add_args_sum = cast(Tensor, sum(additional_forward_args))
            return (
                scale * inputs_sum,
                scale * add_args_sum,
                target_ind,
            )

        array1 = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        array2 = [[6, 7, 8], [0, 1, 2], [3, 4, 5]]
        array3 = [[0, 1, 2], [0, 0, 0], [0, 0, 0]]
        inp1, inp2, inp3 = (
            torch.tensor(array1),
            torch.tensor(array2),
            torch.tensor(array3),
        )
        for index, (inp, add, targ) in enumerate(
            _batched_generator((inp1, inp2), (inp3, 5), 7, 1)
        ):
            assertTensorAlmostEqual(self, inp[0], [array1[index]])
            assertTensorAlmostEqual(self, inp[1], [array2[index]])
            assertTensorAlmostEqual(self, add[0], [array3[index]])
            self.assertEqual(add[1], 5)
            self.assertEqual(targ, 7)

    def test_batched_operator_0_bsz(self) -> None:
        inp1 = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        with self.assertRaises(AssertionError):
            _batched_operator(lambda x: x, inputs=inp1, internal_batch_size=0)

    def test_batched_operator(self) -> None:
        def _sample_operator(
            inputs: Tuple[Tensor, ...],
            additional_forward_args: Tuple[Tensor, ...],
            target_ind: List[int],
            scale: float,
        ) -> Tuple[Tensor, Tensor]:
            inputs_sum = cast(Tensor, sum(inputs))
            add_args_sum = cast(Tensor, sum(additional_forward_args))
            return (
                scale * inputs_sum,
                scale * add_args_sum + target_ind[0],
            )

        inp1 = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        inp2 = torch.tensor([[6, 7, 8], [0, 1, 2], [3, 4, 5]])
        inp3 = torch.tensor([[0, 1, 2], [0, 0, 0], [0, 0, 0]])
        batched_result = _batched_operator(
            _sample_operator,
            inputs=(inp1, inp2),
            additional_forward_args=(inp3),
            target_ind=[0, 1, 2],
            scale=2.0,
            internal_batch_size=1,
        )
        assertTensorAlmostEqual(
            self, batched_result[0], [[12, 16, 20], [6, 10, 14], [18, 22, 26]]
        )
        assertTensorAlmostEqual(
            self, batched_result[1], [[0, 2, 4], [1, 1, 1], [2, 2, 2]]
        )
