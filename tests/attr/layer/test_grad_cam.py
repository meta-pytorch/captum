#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import Any, Dict, Optional, Tuple, Union

import torch
from captum._utils.typing import TensorLikeList
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorTuplesAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_non_conv(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._grad_cam_test_assert(net, net.linear0, inp, [[400.0]])

    def test_simple_multi_input_non_conv(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]], requires_grad=True)
        self._grad_cam_test_assert(net, net.multi_relu, inp, ([[21.0]], [[21.0]]))

    def test_simple_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        self._grad_cam_test_assert(
            net, net.conv1, inp, [[[[11.25, 13.5], [20.25, 22.5]]]]
        )

    def test_simple_input_conv_split_channels(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        expected_result = [
            [
                [[-3.7500, 3.0000], [23.2500, 30.0000]],
                [[15.0000, 10.5000], [-3.0000, -7.5000]],
            ]
        ]
        self._grad_cam_test_assert(
            net,
            net.conv1,
            inp,
            expected_activation=expected_result,
            attr_dim_summation=False,
        )

    def test_simple_input_conv_no_grad(self) -> None:
        net = BasicModel_ConvNet_One_Conv()

        # this way we deactivate require_grad. Some models explicitly
        # do that before interpreting the model.
        for param in net.parameters():
            param.requires_grad = False

        inp = torch.arange(16).view(1, 1, 4, 4).float()
        self._grad_cam_test_assert(
            net, net.conv1, inp, [[[[11.25, 13.5], [20.25, 22.5]]]]
        )

    def test_simple_input_conv_relu(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        self._grad_cam_test_assert(net, net.relu1, inp, [[[[0.0, 4.0], [28.0, 32.5]]]])

    def test_simple_input_conv_without_final_relu(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        # Adding negative value to test final relu is not applied by default
        inp[0, 0, 1, 1] = -4.0
        inp.requires_grad_()
        self._grad_cam_test_assert(
            net, net.conv1, inp, 0.5625 * inp, attribute_to_layer_input=True
        )

    def test_simple_input_conv_fc_with_final_relu(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        # Adding negative value to test final relu is applied
        inp[0, 0, 1, 1] = -4.0
        inp.requires_grad_()
        exp = 0.5625 * inp
        exp[0, 0, 1, 1] = 0.0
        self._grad_cam_test_assert(
            net,
            net.conv1,
            inp,
            exp,
            attribute_to_layer_input=True,
            relu_attributions=True,
        )

    def test_simple_multi_input_conv(self) -> None:
        net = BasicModel_ConvNet_One_Conv()
        inp = torch.arange(16).view(1, 1, 4, 4).float()
        inp2 = torch.ones((1, 1, 4, 4))
        self._grad_cam_test_assert(
            net, net.conv1, (inp, inp2), [[[[14.5, 19.0], [32.5, 37.0]]]]
        )

    def _grad_cam_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_activation: Union[
            TensorLikeList,
            Tuple[TensorLikeList, ...],
            Tensor,
            Tuple[Tensor, ...],
        ],
        additional_input: Any = None,
        attribute_to_layer_input: bool = False,
        relu_attributions: bool = False,
        attr_dim_summation: bool = True,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        layer_gc = LayerGradCam(model, target_layer)
        self.assertFalse(layer_gc.multiplies_by_inputs)
        attributions = layer_gc.attribute(
            test_input,
            target=0,
            additional_forward_args=additional_input,
            attribute_to_layer_input=attribute_to_layer_input,
            relu_attributions=relu_attributions,
            attr_dim_summation=attr_dim_summation,
            grad_kwargs=grad_kwargs,
        )
        assertTensorTuplesAlmostEqual(
            self, attributions, expected_activation, delta=0.01
        )

    def test_relu_gradcam_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]], requires_grad=True)
        gradcam = LayerGradCam(net, net.relu)
        attributions = gradcam.attribute(
            inputs=inp,
            target=0,
            grad_kwargs={"materialize_grads": True},
        )
        self.assertEqual(len(attributions), 1)
        self.assertEqual(list(attributions[0].shape), [1])
        self.assertAlmostEqual(attributions[0].sum(), 0)


if __name__ == "__main__":
    unittest.main()
