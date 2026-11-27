#!/usr/bin/env python3
from collections import OrderedDict
from typing import cast, List, Optional

import captum.optim._core.output_hook as output_hook
import torch
from captum.optim.models import googlenet
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest


def _count_forward_hooks(
    module: torch.nn.Module, hook_fn_name: Optional[str] = None
) -> int:
    """
    Count the number of active forward hooks on the specified module instance.

    Args:

        module (nn.Module): The model module instance to count the number of
            forward hooks on.
        name (str, optional): Optionally only count specific forward hooks based on
            their function's __name__ attribute.
            Default: None

    Returns:
        num_hooks (int): The number of active hooks in the specified module.
    """

    num_hooks: List[int] = [0]

    def _count_hooks(m: torch.nn.Module, name: Optional[str] = None) -> None:
        if hasattr(m, "_forward_hooks"):
            if m._forward_hooks != OrderedDict():
                dict_items = list(m._forward_hooks.items())
                for i, fn in dict_items:
                    if hook_fn_name is None or fn.__name__ == name:
                        num_hooks[0] += 1

    def _count_child_hooks(
        target_module: torch.nn.Module,
        hook_name: Optional[str] = None,
    ) -> None:

        for name, child in target_module._modules.items():
            if child is not None:
                _count_hooks(child, hook_name)
                _count_child_hooks(child, hook_name)

    _count_child_hooks(module, hook_fn_name)
    _count_hooks(module, hook_fn_name)
    return num_hooks[0]


class TestModuleOutputsHook(BaseTest):
    def test_init_single_target(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertEqual(len(hook_module.hooks), len(target_modules))

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(outputs, hook_module.outputs)
        self.assertEqual(list(hook_module.targets), target_modules)
        self.assertFalse(hook_module.is_ready)

    def test_init_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertEqual(len(hook_module.hooks), len(target_modules))

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(outputs, hook_module.outputs)
        self.assertEqual(list(hook_module.targets), target_modules)
        self.assertFalse(hook_module.is_ready)

    def test_init_multiple_targets_remove_hooks(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]

        hook_module = output_hook.ModuleOutputsHook(target_modules)

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, len(target_modules))

        hook_module.remove_hooks()

        n_hooks = _count_forward_hooks(model, "module_outputs_forward_hook")
        self.assertEqual(n_hooks, 0)

    def test_reset_outputs_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]
        test_input = torch.randn(1, 3, 4, 4)

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertFalse(hook_module.is_ready)

        _ = model(test_input)

        self.assertTrue(hook_module.is_ready)

        outputs_dict = hook_module.outputs
        i = 0
        for target, activations in outputs_dict.items():
            self.assertEqual(target, target_modules[i])
            assertTensorAlmostEqual(self, activations, test_input)
            i += 1

        hook_module._reset_outputs()

        self.assertFalse(hook_module.is_ready)

        expected_outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(hook_module.outputs, expected_outputs)

    def test_consume_outputs_multiple_targets(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]
        test_input = torch.randn(1, 3, 4, 4)

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertFalse(hook_module.is_ready)

        _ = model(test_input)

        self.assertTrue(hook_module.is_ready)

        test_outputs_dict = hook_module.outputs
        self.assertIsInstance(test_outputs_dict, dict)
        self.assertEqual(len(test_outputs_dict), len(target_modules))

        i = 0
        for target, activations in test_outputs_dict.items():
            self.assertEqual(target, target_modules[i])
            assertTensorAlmostEqual(self, activations, test_input)
            i += 1

        test_output = hook_module.consume_outputs()

        self.assertFalse(hook_module.is_ready)

        i = 0
        for target, activations in test_output.items():
            self.assertEqual(target, target_modules[i])
            assertTensorAlmostEqual(self, activations, test_input)
            i += 1

        expected_outputs = dict.fromkeys(target_modules, None)
        self.assertEqual(hook_module.outputs, expected_outputs)

    def test_consume_outputs_warning(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        target_modules = [model[0], model[1]]
        test_input = torch.randn(1, 3, 4, 4)

        hook_module = output_hook.ModuleOutputsHook(target_modules)
        self.assertFalse(hook_module.is_ready)

        _ = model(test_input)

        self.assertTrue(hook_module.is_ready)

        hook_module._reset_outputs()

        self.assertFalse(hook_module.is_ready)

        with self.assertWarns(Warning):
            _ = hook_module.consume_outputs()


class TestActivationFetcher(BaseTest):
    def test_activation_fetcher_simple_model(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())

        catch_activ = output_hook.ActivationFetcher(model, targets=[model[0]])
        test_input = torch.randn(1, 3, 224, 224)
        activ_out = catch_activ(test_input)

        self.assertIsInstance(activ_out, dict)
        self.assertEqual(len(activ_out), 1)
        activ = activ_out[model[0]]
        assertTensorAlmostEqual(self, activ, test_input)

    def test_activation_fetcher_single_target(self) -> None:
        model = googlenet(pretrained=True)

        catch_activ = output_hook.ActivationFetcher(model, targets=[model.mixed4d])
        activ_out = catch_activ(torch.zeros(1, 3, 224, 224))

        self.assertIsInstance(activ_out, dict)
        self.assertEqual(len(activ_out), 1)
        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(cast(torch.Tensor, m4d_activ).shape), [1, 528, 14, 14])

    def test_activation_fetcher_multiple_targets(self) -> None:
        model = googlenet(pretrained=True)

        catch_activ = output_hook.ActivationFetcher(
            model, targets=[model.mixed4d, model.mixed5b]
        )
        activ_out = catch_activ(torch.zeros(1, 3, 224, 224))

        self.assertIsInstance(activ_out, dict)
        self.assertEqual(len(activ_out), 2)

        m4d_activ = activ_out[model.mixed4d]
        self.assertEqual(list(cast(torch.Tensor, m4d_activ).shape), [1, 528, 14, 14])

        m5b_activ = activ_out[model.mixed5b]
        self.assertEqual(list(cast(torch.Tensor, m5b_activ).shape), [1, 1024, 7, 7])
