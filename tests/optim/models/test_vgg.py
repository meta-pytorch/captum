import unittest

import torch

from captum.optim.models import vgg16
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version
from tests.optim.helpers.models import check_layer_in_model


class TestVGG16(BaseTest):
    def test_load_vgg16_with_redirected_relu(self) -> None:
        model = vgg16(pretrained=True, replace_relus_with_redirectedrelu=True)
        self.assertTrue(check_layer_in_model(model, RedirectedReluLayer))

    def test_load_vgg16_no_redirected_relu(self) -> None:
        model = vgg16(pretrained=True, replace_relus_with_redirectedrelu=False)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.ReLU))

    def test_load_vgg16_linear(self) -> None:
        model = vgg16(pretrained=True, use_linear_modules_only=True)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertFalse(check_layer_in_model(model, torch.nn.ReLU))
        self.assertFalse(check_layer_in_model(model, torch.nn.MaxPool2d))
        self.assertTrue(check_layer_in_model(model, SkipLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.AvgPool2d))

    def test_transform_vgg16(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = vgg16(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - torch.tensor(
            [123.68, 116.779, 103.939], device=x.device
        ).view(3, 1, 1)
        expected_output = expected_output[:, [2, 1, 0]]  # RGB to BGR
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_vgg16_transform_warning(self) -> None:
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = vgg16(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_load_and_forward_basic_vgg16(self) -> None:
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = vgg16(pretrained=True)
        try:
            model(x)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_load_and_forward_diff_sizes_vgg16(self) -> None:
        x = torch.randn(1, 3, 512, 512).clamp(0, 1)
        x2 = torch.randn(1, 3, 383, 511).clamp(0, 1)
        model = vgg16(pretrained=True)
        try:
            model(x)
            model(x2)
            test = True
        except Exception:
            test = False
        self.assertTrue(test)

    def test_vgg16_forward_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test for VGG16.")
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = vgg16(pretrained=True).cuda()
        outputs = model(x)
        self.assertTrue(outputs.is_cuda)
        self.assertEqual(outputs.shape[0], 1)

    def test_vgg16_load_and_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest("Skipping JIT test due to torch version.")
        x = torch.zeros(1, 3, 224, 224)
        model = vgg16(pretrained=True)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(outputs.shape[0], 1)

    def test_vgg16_load_and_jit_module_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest("Skipping JIT test due to torch version.")
        x = torch.zeros(1, 3, 224, 224)
        model = vgg16(pretrained=True, replace_relus_with_redirectedrelu=False)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual(outputs.shape[0], 1)
