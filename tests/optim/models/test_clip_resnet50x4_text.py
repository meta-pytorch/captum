#!/usr/bin/env python3
import unittest

import torch
from captum.optim.models import clip_resnet50x4_text
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version


class TestCLIPResNet50x4Text(BaseTest):
    def test_clip_resnet50x4_text_logit_scale(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 Text logit scale test due"
                + " to insufficient Torch version."
            )
        model = clip_resnet50x4_text(pretrained=True)
        expected_logit_scale = torch.tensor(4.605170249938965)
        assertTensorAlmostEqual(self, model.logit_scale, expected_logit_scale)

    def test_clip_resnet50x4_text_load_and_forward(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained CLIP ResNet 50x4 Text forward test due to"
                + " insufficient Torch version."
            )
        # Start & End tokens: 49405, 49406
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)])
        x = x[None, :].long()
        model = clip_resnet50x4_text(pretrained=True)
        output = model(x)
        self.assertEqual(list(output.shape), [1, 640])

    def test_clip_resnet50x4_text_forward_cuda(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Text forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Text forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)]).cuda()
        x = x[None, :].long()
        model = clip_resnet50x4_text(pretrained=True).cuda()
        output = model(x)

        self.assertTrue(output.is_cuda)
        self.assertEqual(list(output.shape), [1, 640])

    def test_clip_resnet50x4_text_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
            raise unittest.SkipTest(
                "Skipping pretrained CLIP ResNet 50x4 Text load & JIT module"
                + " test due to insufficient Torch version."
            )
        x = torch.cat([torch.tensor([49405, 49406]), torch.zeros(77 - 2)])
        x = x[None, :].long()
        model = clip_resnet50x4_text(pretrained=True)
        jit_model = torch.jit.script(model)
        output = jit_model(x)
        self.assertEqual(list(output.shape), [1, 640])
