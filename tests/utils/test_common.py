#!/usr/bin/env python3

# pyre-strict

from typing import cast, List, Tuple

import torch
from captum._utils.common import (
    _format_feature_mask,
    _get_max_feature_index,
    _reduce_list,
    _select_targets,
    _sort_key_list,
    parse_version,
    safe_div,
)
from captum.attr._utils.common import get_total_features_from_mask
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)


class Test(BaseTest):
    def test_safe_div_number_denom(self) -> None:
        num = torch.tensor(4.0)
        assert safe_div(num, 2) == 2.0
        assert safe_div(num, 0, 2) == 2.0
        assert safe_div(num, 2.0) == 2.0
        assert safe_div(num, 0.0, 2.0) == 2.0

    def test_safe_div_tensor_denom(self) -> None:
        num = torch.tensor([4.0, 6.0])

        exp = torch.tensor([2.0, 3.0])
        assert (safe_div(num, torch.tensor([2.0, 2.0])) == exp).all()

        # tensor default denom
        assert (safe_div(num, torch.tensor([0.0, 0.0]), torch.tensor(2.0)) == exp).all()
        assert (
            safe_div(
                num,
                torch.tensor([0.0, 0.0]),
                torch.tensor([2.0, 2.0]),
            )
            == exp
        ).all()

        # float default denom
        assert (safe_div(num, torch.tensor([0.0, 0.0]), 2.0) == exp).all()

    def test_reduce_list_tensors(self) -> None:
        tensors = [torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])]
        reduced = _reduce_list(tensors)
        assertTensorAlmostEqual(self, reduced, [[3, 4, 5], [0, 1, 2]])

    def test_reduce_list_tuples(self) -> None:
        tensors = [
            (torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])),
            (torch.tensor([[3, 4, 5]]), torch.tensor([[0, 1, 2]])),
        ]
        reduced = _reduce_list(tensors)
        assertTensorAlmostEqual(self, reduced[0], [[3, 4, 5], [3, 4, 5]])
        assertTensorAlmostEqual(self, reduced[1], [[0, 1, 2], [0, 1, 2]])

    def test_sort_key_list(self) -> None:
        key_list = [
            torch.device("cuda:13"),
            torch.device("cuda:17"),
            torch.device("cuda:10"),
            torch.device("cuda:0"),
        ]
        device_index_list = [0, 10, 13, 17]
        sorted_keys = _sort_key_list(key_list, device_index_list)
        for i in range(len(key_list)):
            self.assertEqual(sorted_keys[i].index, device_index_list[i])

    def test_sort_key_list_incomplete(self) -> None:
        key_list = [torch.device("cuda:10"), torch.device("cuda:0")]
        device_index_list = [0, 10, 13, 17]
        sorted_keys = _sort_key_list(key_list, device_index_list)
        for i in range(len(key_list)):
            self.assertEqual(sorted_keys[i].index, device_index_list[i])

    def test_select_target_2d(self) -> None:
        output_tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        assertTensorAlmostEqual(self, _select_targets(output_tensor, 1), [2, 5, 8])
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, torch.tensor(0)), [1, 4, 7]
        )
        assertTensorAlmostEqual(
            self,
            _select_targets(output_tensor, torch.tensor([1, 2, 0])),
            [2, 6, 7],
        )
        assertTensorAlmostEqual(
            self, _select_targets(output_tensor, [1, 2, 0]), [2, 6, 7]
        )

        # Verify error is raised if too many dimensions are provided.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, (1, 2))

    def test_select_target_3d(self) -> None:
        output_tensor = torch.tensor(
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[9, 8, 7], [6, 5, 4], [3, 2, 1]]]
        )
        assertTensorAlmostEqual(self, _select_targets(output_tensor, (0, 1)), [2, 8])
        assertTensorAlmostEqual(
            self,
            _select_targets(
                output_tensor, cast(List[Tuple[int, ...]], [(0, 1), (2, 0)])
            ),
            [2, 3],
        )

        # Verify error is raised if list is longer than number of examples.
        with self.assertRaises(AssertionError):
            _select_targets(
                output_tensor, cast(List[Tuple[int, ...]], [(0, 1), (2, 0), (3, 2)])
            )

        # Verify error is raised if too many dimensions are provided.
        with self.assertRaises(AssertionError):
            _select_targets(output_tensor, (1, 2, 3))

    def test_format_feature_mask_of_tensor(self) -> None:
        formatted_inputs = (torch.tensor([[0.0, 0.0], [0.0, 0.0]]),)
        tensor_mask = torch.tensor([[0, 1]])
        formatted_tensor_mask = _format_feature_mask(tensor_mask, formatted_inputs)

        self.assertEqual(type(formatted_tensor_mask), tuple)
        assertTensorTuplesAlmostEqual(self, formatted_tensor_mask, (tensor_mask,))

    def test_format_feature_mask_of_tuple(self) -> None:
        formatted_inputs = (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        )

        tuple_mask = (
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([[4, 5], [6, 6]]),
        )
        formatted_tuple_mask = _format_feature_mask(tuple_mask, formatted_inputs)

        self.assertEqual(type(formatted_tuple_mask), tuple)
        assertTensorTuplesAlmostEqual(self, formatted_tuple_mask, tuple_mask)

    def test_format_feature_mask_of_none(self) -> None:
        formatted_inputs = (
            torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
            torch.tensor([]),  # empty tensor
            torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )

        expected_mask = (
            torch.tensor([[0, 1]]),
            torch.tensor([]),
            torch.tensor([[2, 3, 4]]),
        )
        formatted_none_mask = _format_feature_mask(None, formatted_inputs)

        self.assertEqual(type(formatted_none_mask), tuple)
        assertTensorTuplesAlmostEqual(self, formatted_none_mask, expected_mask)

    def test_get_max_feature_index(self) -> None:
        mask = (
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([]),
            torch.tensor([[4, 5], [6, 100]]),
            torch.tensor([[0, 1], [2, 3]]),
        )

        assert _get_max_feature_index(mask) == 100

    def test_mask_unique_elem(self) -> None:
        res = get_total_features_from_mask((torch.tensor([0, 0, 0]),))
        self.assertEqual(res, 1)
        res = get_total_features_from_mask((torch.tensor([0, 0, 4]),))
        self.assertEqual(res, 2)
        res = get_total_features_from_mask(
            (torch.tensor([0, 0, 4]), torch.tensor([0, 4, 5]))
        )
        self.assertEqual(res, 3)


class TestParseVersion(BaseTest):
    def test_parse_version_dev(self) -> None:
        version_str = "2.3.0.dev20240311 "
        output = parse_version(version_str)
        self.assertEqual(output, (2, 3, 0))

    def test_parse_version_post(self) -> None:
        version_str = "1.3.0.post2"
        output = parse_version(version_str)
        self.assertEqual(output, (1, 3, 0))

    def test_parse_version_1_12_0(self) -> None:
        version_str = "1.13.0"
        output = parse_version(version_str)
        self.assertEqual(output, (1, 13, 0))

    def test_parse_version_1_12_2(self) -> None:
        version_str = "1.13.1"
        output = parse_version(version_str)
        self.assertEqual(output, (1, 13, 1))

    def test_parse_version_2_0(self) -> None:
        version_str = "2.0.0"
        output = parse_version(version_str)
        self.assertEqual(output, (2, 0, 0))

    def test_parse_version_1_13(self) -> None:
        version_str = "1.13"
        output = parse_version(version_str)
        self.assertEqual(output, (1, 13))
