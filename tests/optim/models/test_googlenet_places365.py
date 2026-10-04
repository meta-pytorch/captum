#!/usr/bin/env python3
import unittest

import torch
from captum.optim.models import googlenet_places365
from captum.optim.models._common import RedirectedReluLayer, SkipLayer
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version
from tests.optim.helpers.models import check_layer_in_model


class TestInceptionV1Places365(BaseTest):
    def test_load_inceptionv1_places365_with_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 due to insufficient"
                + " Torch version."
            )
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=True
        )
        self.assertTrue(check_layer_in_model(model, RedirectedReluLayer))

    def test_load_inceptionv1_places365_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 RedirectedRelu test"
                + " due to insufficient Torch version."
            )
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.ReLU))

    def test_load_inceptionv1_places365_linear(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping load pretrained InceptionV1 Places365 linear test due to"
                + " insufficient Torch version."
            )
        model = googlenet_places365(pretrained=True, use_linear_modules_only=True)
        self.assertFalse(check_layer_in_model(model, RedirectedReluLayer))
        self.assertFalse(check_layer_in_model(model, torch.nn.ReLU))
        self.assertFalse(check_layer_in_model(model, torch.nn.MaxPool2d))
        self.assertTrue(check_layer_in_model(model, SkipLayer))
        self.assertTrue(check_layer_in_model(model, torch.nn.AvgPool2d))

    def test_inceptionv1_places365_transform(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InceptionV1 Places365 internal transform test due to"
                + " insufficient Torch version."
            )
        x = torch.randn(1, 3, 224, 224).clamp(0, 1)
        model = googlenet_places365(pretrained=True)
        output = model._transform_input(x)
        expected_output = x * 255 - torch.tensor(
            [116.7894, 112.6004, 104.0437], device=x.device
        ).view(3, 1, 1)
        expected_output = expected_output[:, [2, 1, 0]]
        assertTensorAlmostEqual(self, output, expected_output, 0)

    def test_inceptionv1_places365_transform_warning(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InceptionV1 Places365 internal transform warning test due"
                + " to insufficient Torch version."
            )
        x = torch.stack(
            [torch.ones(3, 112, 112) * -1, torch.ones(3, 112, 112) * 2], dim=0
        )
        model = googlenet_places365(pretrained=True)
        with self.assertWarns(UserWarning):
            model._transform_input(x)

    def test_inceptionv1_places365_load_and_forward(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping basic pretrained InceptionV1 Places365 forward test due to"
                + " insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=True)
        outputs = model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_load_and_forward_diff_sizes(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward with different"
                + " sized inputs test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 512, 512)
        x2 = torch.zeros(1, 3, 383, 511)
        model = googlenet_places365(pretrained=True)

        outputs = model(x)
        outputs2 = model(x2)

        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)
        self.assertEqual([list(o.shape) for o in outputs2], [[1, 365]] * 3)

    def test_inceptionv1_places365_forward_no_aux(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 with aux logits forward"
                + " test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=False, aux_logits=False)
        outputs = model(x)
        self.assertEqual(list(outputs.shape), [1, 365])

    def test_inceptionv1_places365_forward_cuda(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward CUDA test due to"
                + " insufficient Torch version."
            )
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 forward CUDA test due to"
                + " not supporting CUDA."
            )
        x = torch.zeros(1, 3, 224, 224).cuda()
        model = googlenet_places365(pretrained=True).cuda()
        outputs = model(x)

        self.assertTrue(outputs[0].is_cuda)
        self.assertTrue(outputs[1].is_cuda)
        self.assertTrue(outputs[2].is_cuda)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_load_and_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 load & JIT module test"
                + " due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(pretrained=True)
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)

    def test_inceptionv1_places365_jit_module_no_redirected_relu(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest(
                "Skipping pretrained InceptionV1 Places365 load & JIT module with no"
                + " redirected relu test due to insufficient Torch version."
            )
        x = torch.zeros(1, 3, 224, 224)
        model = googlenet_places365(
            pretrained=True, replace_relus_with_redirectedrelu=False
        )
        jit_model = torch.jit.script(model)
        outputs = jit_model(x)
        self.assertEqual([list(o.shape) for o in outputs], [[1, 365]] * 3)
