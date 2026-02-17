#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Any, Tuple, Union

import torch
from captum._utils.typing import TensorLikeList, TensorOrTupleOfTensorsGeneric
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel_ConvNet_One_Conv
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = 1.0 * torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        ex = [
            [
                [
                    [0.0, 0.0, 4.0, 4.0],
                    [0.0, 0.0, 12.0, 8.0],
                    [28.0, 84.0, 97.5, 65.0],
                    [28.0, 56.0, 65.0, 32.5],
                ]
            ]
        ]
        self._guided_grad_cam_test_assert(net, net.relu1, inp, ex)

    def test_simple_multi_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex = [
            [
                [
                    [14.5, 29.0, 38.0, 19.0],
                    [29.0, 58.0, 76.0, 38.0],
                    [65.0, 130.0, 148.0, 74.0],
                    [32.5, 65.0, 74.0, 37.0],
                ]
            ]
        ]
        self._guided_grad_cam_test_assert(net, net.conv1, (inp, inp2), (ex, ex))

    def test_simple_multi_input_relu_input(self) -> None:
        net = BasicModel_ConvNet_One_Conv(inplace=True)
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex = [
            [
                [
                    [14.5, 29.0, 38.0, 19.0],
                    [29.0, 58.0, 76.0, 38.0],
                    [65.0, 130.0, 148.0, 74.0],
                    [32.5, 65.0, 74.0, 37.0],
                ]
            ]
        ]
        self._guided_grad_cam_test_assert(
            net, net.relu1, (inp, inp2), (ex, ex), attribute_to_layer_input=True
        )

    def test_simple_multi_input_conv_inplace(self) -> None:
        net = BasicModel_ConvNet_One_Conv(inplace=True)
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones((1, 1, 4, 4))
        ex = [
            [
                [
                    [14.5, 29.0, 38.0, 19.0],
                    [29.0, 58.0, 76.0, 38.0],
                    [65.0, 130.0, 148.0, 74.0],
                    [32.5, 65.0, 74.0, 37.0],
                ]
            ]
        ]
        self._guided_grad_cam_test_assert(net, net.conv1, (inp, inp2), (ex, ex))

    def test_improper_dims_multi_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones(1)
        ex = [
            [
                [
                    [14.5, 29.0, 38.0, 19.0],
                    [29.0, 58.0, 76.0, 38.0],
                    [65.0, 130.0, 148.0, 74.0],
                    [32.5, 65.0, 74.0, 37.0],
                ]
            ]
        ]
        self._guided_grad_cam_test_assert(net, net.conv1, (inp, inp2), (ex, []))

    def test_improper_method_multi_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16, dtype=torch.float).view(1, 1, 4, 4)
        inp2 = torch.ones(1)
        self._guided_grad_cam_test_assert(
            net, net.conv1, (inp, inp2), ([], []), interpolate_mode="made_up_nonlinear"
        )

    def _guided_grad_cam_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        expected: Union[Tensor, TensorLikeList, Tuple[TensorLikeList, ...]],
        additional_input: Any = None,
        interpolate_mode: str = "nearest",
        attribute_to_layer_input: bool = False,
    ) -> None:
        guided_gc = GuidedGradCam(model, target_layer)
        self.assertFalse(guided_gc.multiplies_by_inputs)
        attributions = guided_gc.attribute(
            test_input,
            target=0,
            additional_forward_args=additional_input,
            interpolate_mode=interpolate_mode,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        if isinstance(test_input, tuple):
            for i in range(len(test_input)):
                assertTensorAlmostEqual(
                    self,
                    attributions[i],
                    expected[i],
                    delta=0.01,
                )
        else:
            assertTensorAlmostEqual(
                self,
                attributions,
                expected,
                delta=0.01,
            )


if __name__ == "__main__":
    unittest.main()
