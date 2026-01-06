#!/usr/bin/env python3

# pyre-strict
import copy
import random
import unittest

from typing import Any, Callable, Generator, Tuple, TypeVar, Union

import numpy as np
import torch
from captum.log import patch_methods
from torch import Tensor

ReturnType = TypeVar("ReturnType")


def deep_copy_args(func: Callable[..., ReturnType]) -> Callable[..., ReturnType]:
    def copy_args(*args: Any, **kwargs: Any) -> ReturnType:
        return func(
            *(copy.deepcopy(x) for x in args),
            **{k: copy.deepcopy(v) for k, v in kwargs.items()},
        )

    return copy_args


def assertTensorAlmostEqual(
    test: unittest.TestCase,
    # pyre-fixme[2]: Parameter must be annotated.
    actual,
    # pyre-fixme[2]: Parameter must be annotated.
    expected,
    delta: float = 0.0001,
    mode: str = "sum",
) -> None:
    assert isinstance(actual, torch.Tensor), (
        "Actual parameter given for " "comparison must be a tensor."
    )
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)
    assert (
        actual.shape == expected.shape
    ), f"Expected tensor with shape: {expected.shape}. Actual shape {actual.shape}."
    actual = actual.cpu()
    expected = expected.cpu()
    if mode == "sum":
        test.assertAlmostEqual(
            torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
        )
    elif mode == "max":
        # if both tensors are empty, they are equal but there is no max
        if actual.numel() == expected.numel() == 0:
            return

        if actual.size() == torch.Size([]):
            test.assertAlmostEqual(
                torch.max(torch.abs(actual - expected)).item(), 0.0, delta=delta
            )
        else:
            for index, (input, ref) in enumerate(zip(actual, expected)):
                almost_equal = abs(input - ref) <= delta
                if hasattr(almost_equal, "__iter__"):
                    almost_equal = almost_equal.all()
                assert (
                    almost_equal
                ), "Values at index {}, {} and {}, differ more than by {}".format(
                    index, input, ref, delta
                )
    else:
        raise ValueError("Mode for assertion comparison must be one of `max` or `sum`.")


def assertTensorTuplesAlmostEqual(
    test: unittest.TestCase,
    # pyre-fixme[2]: Parameter must be annotated.
    actual,
    # pyre-fixme[2]: Parameter must be annotated.
    expected,
    delta: float = 0.0001,
    mode: str = "sum",
) -> None:
    if isinstance(expected, tuple):
        assert len(actual) == len(
            expected
        ), f"the length of actual {len(actual)} != expected {len(expected)}"

        for i in range(len(expected)):
            assertTensorAlmostEqual(test, actual[i], expected[i], delta, mode)
    else:
        assertTensorAlmostEqual(test, actual, expected, delta, mode)


def assertTupleOfListOfTensorsAlmostEqual(
    test: unittest.TestCase,
    # pyre-fixme[2]: Parameter must be annotated.
    actual,
    # pyre-fixme[2]: Parameter must be annotated.
    expected,
    delta: float = 0.0001,
    mode: str = "sum",
) -> None:
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and isinstance(expected, tuple), (
            "Both actual and expected must be tuples, got "
            f"{type(actual)} and {type(expected)}"
        )
        assert len(actual) == len(
            expected
        ), f"Tuple lengths differ: {len(actual)} != {len(expected)}"
        for i, (actual_list, expected_list) in enumerate(zip(actual, expected)):
            assert isinstance(actual_list, list) and isinstance(expected_list, list), (
                f"Elements at index {i} must be lists, got "
                f"{type(actual_list)} and {type(expected_list)}"
            )
            assert len(actual_list) == len(expected_list), (
                "List lengths at tuple index "
                f"{i} differ: {len(actual_list)} != {len(expected_list)}"
            )
            for a_tensor, e_tensor in zip(actual_list, expected_list):
                assertTensorAlmostEqual(test, a_tensor, e_tensor, delta, mode)
    else:
        assertTensorAlmostEqual(test, actual, expected, delta, mode)


def assertAttributionComparision(
    test: unittest.TestCase,
    attributions1: Union[Tensor, Tuple[Tensor, ...]],
    attributions2: Union[Tensor, Tuple[Tensor, ...]],
) -> None:
    for attribution1, attribution2 in zip(attributions1, attributions2):
        for attr_row1, attr_row2 in zip(attribution1, attribution2):
            assertTensorAlmostEqual(test, attr_row1, attr_row2, 0.05, "max")


def assert_delta(test: unittest.TestCase, delta: Tensor) -> None:
    delta_condition = (delta.abs() < 0.00001).all()
    test.assertTrue(
        delta_condition,
        "The sum of attribution values {} for relu layer is not "
        "nearly equal to the difference between the endpoint for "
        "some samples".format(delta),
    )


def set_all_random_seeds(seed: int = 1234) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def lcg(
    a: int = 16843009, b: int = 3014898611, m: int = 1 << 32
) -> Generator[int, None, None]:
    """Linear congruential generator"""
    x = 1
    while True:
        x = (a * x + b) % m
        yield x


def rand_like(a: Tensor) -> Tensor:
    """Random tensors (for dependency-free version-agnostic reproducibility).
    PyTorch does not guarantee reproducible numbers across PyTorch releases,
    individual commits, or different platforms. See:
    https://pytorch.org/docs/stable/notes/randomness.html"""
    g = lcg()
    nums = [next(g) / (1 << 32) for _ in range(a.numel())]
    return torch.tensor(nums, dtype=a.dtype, device=a.device).reshape(a.shape)


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed. Since many torch
    initializations are random, this ensures that tests run deterministically.
    """

    def setUp(self) -> None:
        set_all_random_seeds(1234)
        patch_methods(self)


def extracted_features_equal(a: Any, b: Any) -> bool:
    """
    Recursively checks if two extracted feature structures are equal.
    The structures can be:
      - torch.Tensor
      - list of torch.Tensor
      - tuple of (torch.Tensor or list of torch.Tensor)
    Args:
        a: First extracted feature (tensor, list, or tuple).
        b: Second extracted feature (tensor, list, or tuple).
    Returns:
        bool: True if the structures are equal, False otherwise.
    """
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        return torch.equal(a, b)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(torch.equal(x, y) for x, y in zip(a, b))
    elif isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(extracted_features_equal(x, y) for x, y in zip(a, b))
    else:
        return False
