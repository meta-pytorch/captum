#!/usr/bin/env python3
import unittest

import torch
from captum.optim.models import googlenet
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version
from tests.optim.helpers.models import check_layer_in_model


class TestInceptionV1(BaseTest):
    def test_load_inceptionv1_with_redirected_relu(self) -> None:
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=True)
        self.assertTrue(check_layer_in_model(model, RedirectedReluLayer))

    def test_load_inceptionv1_no_redirected_relu(self) -> None:
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.ReLU))

    def test_load_inceptionv1_linear(self) -> None:
        model = googlenet(pretrained=True, use_linear_modules_only=True)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertFalse(check_layer_in_model(model, torch.nn.ReLU))
        self.assertFalse(check_layer_in_model(model, torch.nn.MaxPool2d))
        self.assertTrue(check_layer_in_model(model, SkipLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.AvgPool2d))

    def test_transform_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_inceptionv1_transform_warning(self) -> None:
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = googlenet(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_transform_bgr_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True, bgr_transform=True)
        output = model._transform_input(x)
        expected_output = x[:, [2, 1, 0]] * 255 - 117
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_load_and_forward_basic_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=True)
        try:
            model(x)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_load_and_forward_diff_sizes_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 512, 512).clamp(0, 1)
        x2 = torch.randn(1, 3, 383, 511).clamp(0, 1)
        model = googlenet(pretrained=True)
        try:
            model(x)
            model(x2)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_forward_aux_inceptionv1(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet(pretrained=False, aux_logits=True)
        outputs = model(x)
        self.assertEqual(len(outputs), 3)

    def test_inceptionv1_forward_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 forward CUDA test due to not"
                + " supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = googlenet(pretrained=True).cuda()
        outputs = model(x)
        self.assertTrue(outputs.is_cuda)
        self.assertEqual(list(outputs.shape), [1, 1008])

    def test_inceptionv1_load_and_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 load & JIT test"
                + " due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=True)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(list(outputs.shape), [1, 1008])

    def test_inceptionv1_load_and_jit_module_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 load & JIT with no"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet(pretrained=True, replace_relus_with_redirectedrelu=False)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(list(outputs.shape), [1, 1008])
