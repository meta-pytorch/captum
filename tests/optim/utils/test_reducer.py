#!/usr/bin/env python3
import unittest

import captum.optim._utils.reducer as reducer
import numpy as np
import torch
from captum.testing.helpers.basic import BaseTest


class FakeReductionAlgorithm(object):
    """
    Fake reduction algorithm for testing
    """

    def __init__(self, n_components=3, **kwargs) -> None:
        self.n_components = n_components
        self.components_ = np.ones((2, 64))

    def fit_transform(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, 0:3, ...]


class TestChannelReducer(BaseTest):
    def test_channelreducer_pytorch(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch swap_2nd_and_last_dims test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input)
        self.assertEqual(test_output.size(0), 1)
        self.assertEqual(test_output.size(1), 3)
        self.assertEqual(test_output.size(2), 224)
        self.assertEqual(test_output.size(3), 224)

    def test_channelreducer_pytorch_dim_three(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch swap_2nd_and_last_dims test"
            )

        test_input = torch.randn(32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input)
        self.assertEqual(list(test_output.shape), [3, 224, 224])

    def test_channelreducer_pytorch_pca(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch swap_2nd_and_last_dims PCA test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, reduction_alg="PCA")

        test_output = c_reducer.fit_transform(test_input)
        self.assertEqual(list(test_output.shape), [1, 3, 224, 224])

    def test_channelreducer_pytorch_custom_alg(self) -> None:
        test_input = torch.randn(1, 32, 224, 224).abs()
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        test_output = c_reducer.fit_transform(test_input)
        self.assertEqual(list(test_output.shape), [1, 3, 224, 224])

    def test_channelreducer_pytorch_custom_alg_components(self) -> None:
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        components = c_reducer.components
        self.assertTrue(torch.is_tensor(components))

    def test_channel_reducer_pytorch_custom_alg_cuda_input_cpu_reducer(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA tests due to not supporting CUDA.")
        test_input = torch.randn(1, 32, 224, 224).abs().cuda()
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3, reduction_alg=reduction_alg, max_iter=100
        )
        test_output = c_reducer.fit_transform(test_input)
        self.assertTrue(test_output.is_cuda)

    def test_channel_reducer_pytorch_custom_alg_cuda_input_cuda_reducer(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA tests due to not supporting CUDA.")
        test_input = torch.randn(1, 32, 224, 224).abs().cuda()
        reduction_alg = FakeReductionAlgorithm
        c_reducer = reducer.ChannelReducer(
            n_components=3,
            reduction_alg=reduction_alg,
            max_iter=100,
        )
        test_output = c_reducer.fit_transform(test_input)
        self.assertTrue(test_output.is_cuda)

    def test_channelreducer_pytorch_components(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch swap_2nd_and_last_dims test"
            )

        test_input = torch.randn(1, 32, 224, 224).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input)
        components = c_reducer.components
        self.assertTrue(torch.is_tensor(components))
        self.assertTrue(torch.is_tensor(test_output))

    def test_channelreducer_noreshape_pytorch(self) -> None:
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch no reshape test"
            )

        test_input = torch.randn(1, 224, 224, 32).abs()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        test_output = c_reducer.fit_transform(test_input, swap_2nd_and_last_dims=False)
        self.assertEqual(list(test_output.shape), [1, 224, 224, 3])

    def test_channelreducer_error(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping ChannelReducer CUDA test due to not supporting CUDA."
            )
        try:
            import sklearn  # noqa: F401

        except (ImportError, AssertionError):
            raise unittest.SkipTest(
                "Module sklearn not found, skipping ChannelReducer"
                + " PyTorch no reshape test"
            )

        test_input = torch.randn(1, 224, 224, 32).abs().cuda()
        c_reducer = reducer.ChannelReducer(n_components=3, max_iter=100)
        with self.assertRaises(TypeError):
            c_reducer.fit_transform(test_input, swap_2nd_and_last_dims=False)


class TestPosNeg(BaseTest):
    def test_posneg(self) -> None:
        x = torch.ones(1, 3, 224, 224) - 2
        self.assertGreater(
            torch.sum(reducer.posneg(x) >= 0).item(), torch.sum(x >= 0).item()
        )
