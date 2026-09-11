#!/usr/bin/env python3
import unittest
from typing import List

import captum.optim as opt
import torch
from captum.testing.helpers.basic import BaseTest
from captum.testing.helpers.basic_models import BasicModel_ConvNet_Optim
from packaging import version


class TestInputOptimization(BaseTest):
    def test_input_optimization_init(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InputOptimization init test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss_fn = opt.loss.ChannelActivation(model.layer, 1)
        transform = torch.nn.Identity()
        image_param = opt.images.NaturalImage()
        obj = opt.InputOptimization(
            model, loss_function=loss_fn, input_param=image_param, transform=transform
        )

        self.assertEqual(model, obj.model)
        self.assertEqual(image_param, obj.input_param)
        self.assertEqual(transform, obj.transform)
        self.assertEqual(loss_fn, obj.loss_function)
        self.assertEqual(list(image_param.parameters()), list(obj.parameters()))

    def test_input_optimization_custom_optimize(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InputOptimization custom optimze test due to insufficient"
                + " Torch version."
            )
        model = BasicModel_ConvNet_Optim()
        loss_fn = opt.loss.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)

        stop_criteria = opt.optimization.n_steps(512, show_progress=False)
        optimizer = torch.optim.Adam(obj.parameters(), lr=0.02)

        history: List[torch.Tensor] = []
        step = 0
        try:
            while stop_criteria(step, obj, history, optimizer):
                optimizer.zero_grad()
                loss_value = -1.0 * obj.loss().mean()
                history.append(loss_value.clone().detach())
                loss_value.backward()
                optimizer.step()
                step += 1
        finally:
            obj.cleanup()
        history = torch.stack(history)
        self.assertIsInstance(history, torch.Tensor)

    def test_input_optimization(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InputOptimization test due to insufficient Torch version."
            )
        model = BasicModel_ConvNet_Optim()
        loss_fn = opt.loss.ChannelActivation(model.layer, 0)
        obj = opt.InputOptimization(model, loss_function=loss_fn)
        n_steps = 25
        history = obj.optimize(opt.optimization.n_steps(n_steps, show_progress=False))
        self.assertTrue(history[0] > history[-1])
        self.assertTrue(len(history) == n_steps)

    def test_input_optimization_param(self) -> None:
        """Test for optimizing param without model"""
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping InputOptimization test due to insufficient Torch version."
            )
        img_param = opt.images.NaturalImage()
        loss_fn = opt.loss.ChannelActivation(img_param, 0)
        # Use torch.nn.Identity as placeholder for non-model optimization
        obj = opt.InputOptimization(torch.nn.Identity(), loss_fn, img_param)
        n_steps = 5
        history = obj.optimize(opt.optimization.n_steps(n_steps, show_progress=False))
        self.assertTrue(history[0] > history[-1])
        self.assertTrue(len(history) == n_steps)
