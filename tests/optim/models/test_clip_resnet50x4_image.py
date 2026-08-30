#!/usr/bin/env python3
import unittest

import torch
from captum.optim.models import clip_resnet50x4_image
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version
from tests.optim.helpers.models import check_layer_in_model


class TestCLIPResNet50x4Image(BaseTest):
    def test_load_clip_resnet50x4_image_with_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 Image due to insufficient"
                + " Torch version."
            )
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=True
        )
        self.assertTrue(check_layer_in_model(model, RedirectedReluLayer))

    def test_load_clip_resnet50x4_image_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 Image RedirectedRelu test"
                + " due to insufficient Torch version."
            )
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.ReLU))

    def test_load_clip_resnet50x4_image_linear(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained CLIP ResNet 50x4 Image linear test due to"
                + " insufficient Torch version."
            )
        model = clip_resnet50x4_image(pretrained=True, use_linear_modules_only=True)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertFalse(check_layer_in_model(model, torch.nn.ReLU))
        self.assertTrue(check_layer_in_model(model, SkipLayer))

    def test_clip_resnet50x4_image_transform(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping CLIP ResNet 50x4 Image internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 288, 288).clamp(0, 1)
        model = clip_resnet50x4_image(pretrained=True)
        output = model._transform_input(x)
        expected_output = x.clone() - torch.tensor(
            [0.48145466, 0.4578275, 0.40821073]
        ).view(3, 1, 1)
        expected_output = expected_output / torch.tensor(
            [0.26862954, 0.26130258, 0.27577711]
        ).view(3, 1, 1)
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_clip_resnet50x4_image_transform_warning(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping CLIP ResNet 50x4 Image internal transform warning test due"
                + " to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 288, 288) * -1, torch.ones(3, 288, 288) * 2], dim=0
        )
        model = clip_resnet50x4_image(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_clip_resnet50x4_image_load_and_forward(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 Image forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(pretrained=True, use_attnpool=True)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 640])
        self.assertTrue(model.use_attnpool)

    def test_untrained_clip_resnet50x4_image_load_and_forward(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic untrained CLIP ResNet 50x4 Image forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(pretrained=False, use_attnpool=True)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 640])
        self.assertTrue(model.use_attnpool)

    def test_clip_resnet50x4_image_warning(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Image transform input"
                + " warning test due to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 288, 288) * -1, torch.ones(3, 288, 288) * 2], dim=0
        )
        model = clip_resnet50x4_image(pretrained=True)
        with self.assertWarns(UserWarning):
            _ = model._transform_input(x)

    def test_clip_resnet50x4_image_use_attnpool_false(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 Image use_attnpool"
                + " forward due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(pretrained=True, use_attnpool=False)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 2560, 9, 9])
        self.assertFalse(model.use_attnpool)

    def test_clip_resnet50x4_image_use_attnpool_false_size_128(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 Image use_attnpool"
                + " forward with 128x128 input due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 128, 128)
        model = clip_resnet50x4_image(pretrained=True, use_attnpool=False)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 2560, 4, 4])
        self.assertFalse(model.use_attnpool)

    def test_clip_resnet50x4_image_forward_cuda(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Image forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Image forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.zeros(1, 3, 288, 288).cuda()
        model = clip_resnet50x4_image(pretrained=True, use_attnpool=True).cuda()
        output = model(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual(list(output.shape), [1, 640])
        self.assertTrue(model.use_attnpool)

    def test_clip_resnet50x4_image_jit_module_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Image load & JIT module with"
                + " no redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=False, use_attnpool=True
        )
        jit_model = torch.jit.script(model)
        output = jit_model(x)
        self.assertEqual(list(output.shape), [1, 640])
        self.assertTrue(model.use_attnpool)

    def test_clip_resnet50x4_image_jit_module_with_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Image load & JIT module with"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 288, 288)
        model = clip_resnet50x4_image(
            pretrained=True, replace_relus_with_redirectedrelu=True, use_attnpool=True
        )
        jit_model = torch.jit.script(model)
        output = jit_model(x)
        self.assertEqual(list(output.shape), [1, 640])
        self.assertTrue(model.use_attnpool)
