#!/usr/bin/env python3
import random
import unittest

import numpy as np
import torch


def assertArraysAlmostEqual(inputArr, refArr, delta=0.05):
    for index, (input, ref) in enumerate(zip(inputArr, refArr)):
        assert (
            abs(input - ref) <= delta
        ), "Values at index {}, {} and {}, \
            differ more than by {}".format(
            index, input, ref, delta
        )


def assertTensorAlmostEqual(test, tensor, expected):
    test.assertAlmostEqual(
        torch.sum(torch.abs(tensor - torch.tensor(expected))), 0.0, delta=0.0001
    )


class BaseTest(unittest.TestCase):
    """
    This class provides a basic framework for all Captum tests by providing
    a set up fixture, which sets a fixed random seed. Since many torch
    initializations are random, this ensures that tests run deterministically.
    """

    def setUp(self):
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True


class BaseGPUTest(BaseTest):
    """
    This class provides a basic framework for all Captum tests requiring
    CUDA and available GPUs to run appropriately, such as tests for
    DataParallel models. If CUDA is not available, these tests are skipped.
    """

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            raise unittest.SkipTest("Skipping GPU test since CUDA not available.")
