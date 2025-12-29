#!/usr/bin/env python3

# pyre-strict

from enum import Enum
from typing import Any, Callable, cast, Dict, Optional, Tuple, Type

import torch
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._models.base import _set_deep_layer_value
from captum.attr._utils.attribution import Attribution, InternalAttribution
from captum.testing.attr.helpers.gen_test_utils import (
    gen_test_name,
    get_target_layer,
    parse_test_config,
    should_create_generated_test,
)
from captum.testing.attr.helpers.test_config import config
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import deep_copy_args
from torch.nn import Module

"""
Tests in this file are dynamically generated based on the config
defined in captum/testing/attr/helpers/test_config.py. To add new test cases,
read the documentation in test_config.py and add cases based on the
schema described there.
"""


class HookRemovalMode(Enum):
    """
    Defines modes for hook removal tests:
    `normal` - Verifies no hooks remain after running an attribution method
    normally
    `incorrect_target_or_neuron` - Verifies no hooks remain after an incorrect
    target and neuron_selector are provided, which causes an assertion error
    in the algorithm.
    `invalid_module` - Verifies no hooks remain after an invalid module
    is executed, which causes an assertion error in model execution.
    """

    normal = 1
    incorrect_target_or_neuron = 2
    invalid_module = 3


class ErrorModule(Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, *args: Any, **kwargs: Any) -> None:
        raise AssertionError("Raising error on execution")


class HookRemovalMeta(type):
    """
    Attribution is computed either normally or with the changes based on the
    mode, which cause an error. Once attribution is calculated, test verifies
    that no forward, backward or forward pre hooks remain on any modules.
    """

    def __new__(
        cls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]
    ) -> "HookRemovalMeta":
        created_tests: Dict[Tuple[Type[Attribution], HookRemovalMode], bool] = {}
        for test_config in config:
            (
                algorithms,
                model,
                args,
                layer,
                noise_tunnel,
                _,
            ) = parse_test_config(test_config)

            for algorithm in algorithms:
                if not should_create_generated_test(algorithm):
                    continue
                for mode in HookRemovalMode:
                    if mode is HookRemovalMode.invalid_module and layer is None:
                        continue
                    # Only one test per algorithm and mode is necessary
                    if (algorithm, mode) in created_tests:
                        continue

                    test_method = cls.make_single_hook_removal_test(
                        algorithm,
                        model,
                        layer,
                        args,
                        noise_tunnel,
                        mode,
                    )
                    test_name = gen_test_name(
                        "test_hook_removal_" + mode.name,
                        cast(str, test_config["name"]),
                        algorithm,
                        noise_tunnel,
                    )

                    if test_name in attrs:
                        raise AssertionError(
                            "Trying to overwrite existing test with name: %r"
                            % test_name
                        )
                    attrs[test_name] = test_method
                    created_tests[(algorithm, mode)] = True
        return super(HookRemovalMeta, cls).__new__(cls, name, bases, attrs)

    # Arguments are deep copied to ensure tests are independent and are not affected
    # by any modifications within a previous test.
    @classmethod
    @deep_copy_args
    def make_single_hook_removal_test(
        cls,
        algorithm: Type[Attribution],
        model: Module,
        layer: Optional[str],
        args: Dict[str, Any],
        noise_tunnel: bool,
        mode: HookRemovalMode,
    ) -> Callable[..., None]:
        """
        This method creates a single hook removal test for the given
        algorithm and parameters.
        """

        def hook_removal_test_assert(self: "TestHookRemoval") -> None:
            attr_method: Attribution
            expect_error = False
            if layer is not None:
                if mode is HookRemovalMode.invalid_module:
                    expect_error = True
                    if isinstance(layer, list):
                        _set_deep_layer_value(model, layer[0], ErrorModule())
                    else:
                        _set_deep_layer_value(model, layer, ErrorModule())
                target_layer = get_target_layer(model, layer)
                internal_algorithm = cast(Type[InternalAttribution[Module]], algorithm)
                attr_method = internal_algorithm(model, target_layer)
            else:
                attr_method = algorithm(model)

            if noise_tunnel:
                attr_method = NoiseTunnel(attr_method)

            if mode is HookRemovalMode.incorrect_target_or_neuron:
                # Overwriting target and neuron index arguments to
                # incorrect values.
                if "target" in args:
                    args["target"] = (9999,) * 20
                    expect_error = True
                if "neuron_selector" in args:
                    args["neuron_selector"] = (9999,) * 20
                    expect_error = True

            if expect_error:
                with self.assertRaises(AssertionError):
                    attr_method.attribute(**args)
            else:
                attr_method.attribute(**args)

            def check_leftover_hooks(module: Module) -> None:
                self.assertEqual(len(module._forward_hooks), 0)
                self.assertEqual(len(module._backward_hooks), 0)
                self.assertEqual(len(module._forward_pre_hooks), 0)

            model.apply(check_leftover_hooks)

        return hook_removal_test_assert


class TestHookRemoval(BaseTest, metaclass=HookRemovalMeta):
    pass
