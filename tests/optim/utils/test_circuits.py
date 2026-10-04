#!/usr/bin/env python3
import unittest

import captum.optim._utils.circuits as circuits
import torch
from captum.optim.models import googlenet
from captum.testing.helpers.basic import BaseTest
from packaging import version


class TestGetExpandedWeights(BaseTest):
    def test_get_expanded_weights(self) -> None:
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        output_tensor = circuits.extract_expanded_weights(
            model, model.mixed3a, model.mixed3b
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(list(output_tensor.shape), [480, 256, 28, 28])

    def test_get_expanded_weights_crop_int(self) -> None:
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        output_tensor = circuits.extract_expanded_weights(
            model, model.mixed3a, model.mixed3b, 5
        )
        self.assertEqual(list(output_tensor.shape), [480, 256, 5, 5])

    def test_get_expanded_weights_crop_two_int(self) -> None:
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        output_tensor = circuits.extract_expanded_weights(
            model, model.mixed3a, model.mixed3b, (5, 5)
        )
        self.assertEqual(list(output_tensor.shape), [480, 256, 5, 5])

    def test_get_expanded_nonlinear_top_connections(self) -> None:
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        output_tensor = circuits.extract_expanded_weights(
            model, model.pool3, model.mixed4a, 5
        )
        self.assertEqual(list(output_tensor.shape), [508, 480, 5, 5])

        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            norm_func = torch.norm
        else:
            norm_func = torch.linalg.norm

        top_connected_neurons = torch.argsort(
            torch.stack(
                [
                    -norm_func(output_tensor[i, 379, :, :])
                    for i in range(output_tensor.shape[0])
                ]
            )
        )[:10].tolist()

        expected_list = [50, 437, 96, 398, 434, 423, 408, 436, 424, 168]
        self.assertEqual(top_connected_neurons, expected_list)

    def test_get_expanded_weights_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping Circuits CUDA test due to not supporting CUDA."
            )
        model = googlenet(pretrained=True, use_linear_modules_only=True).cuda()
        output_tensor = circuits.extract_expanded_weights(
            model, model.mixed3a, model.mixed3b
        )
        self.assertTrue(torch.is_tensor(output_tensor))
        self.assertEqual(list(output_tensor.shape), [480, 256, 28, 28])
        self.assertTrue(output_tensor.is_cuda)
