# pyre-strict
from typing import cast, Dict, List, Tuple, Union

from captum.attr._utils.baselines import ProductBaselines

# from parameterized import parameterized
from captum.testing.helpers import BaseTest


class TestProductBaselines(BaseTest):
    def test_list(self) -> None:
        baseline_values = [
            [1, 2, 3],
            [4, 5, 6, 7],
            [8, 9],
        ]

        baselines = ProductBaselines(baseline_values)

        baseline_sample = baselines()

        self.assertIsInstance(baseline_sample, list)
        for sample_val, vals in zip(baseline_sample, baseline_values):
            self.assertIn(sample_val, vals)

    def test_dict(self) -> None:
        baseline_values = {
            "f1": [1, 2, 3],
            "f2": [4, 5, 6, 7],
            "f3": [8, 9],
        }

        baselines = ProductBaselines(
            cast(Dict[Union[str, Tuple[str, ...]], List[int]], baseline_values)
        )

        baseline_sample = baselines()

        self.assertIsInstance(baseline_sample, dict)
        baseline_sample = cast(Dict[str, int], baseline_sample)

        for sample_key, sample_val in baseline_sample.items():
            self.assertIn(sample_val, baseline_values[sample_key])

    def test_dict_tuple_key(self) -> None:
        baseline_values: Dict[
            Union[str, Tuple[str, ...]], List[Union[Tuple[int, str], int]]
        ] = {
            ("f1", "f2"): [(1, "1"), (2, "2"), (3, "3")],
            "f3": [4, 5],
        }

        baselines = ProductBaselines(baseline_values)

        baseline_sample = baselines()

        self.assertIsInstance(baseline_sample, dict)
        baseline_sample = cast(Dict[str, Union[str, int]], baseline_sample)

        self.assertEqual(len(baseline_sample), 3)

        self.assertIn(
            (baseline_sample["f1"], baseline_sample["f2"]),
            baseline_values[("f1", "f2")],
        )
        self.assertIn(baseline_sample["f3"], baseline_values["f3"])
