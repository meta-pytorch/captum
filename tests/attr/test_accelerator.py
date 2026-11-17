#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import copy
import os
from typing import Any, Callable, cast, Dict, Optional, Tuple, Type

import torch
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.attribution import Attribution, InternalAttribution
from captum.testing.attr.helpers.gen_test_utils import (
    gen_test_name,
    get_target_layer,
    parse_test_config,
    should_create_generated_test,
)
from captum.testing.attr.helpers.test_config import config
from captum.testing.helpers.basic import (
    assertTensorTuplesAlmostEqual,
    BaseTest,
    deep_copy_args,
)
from torch import Tensor
from torch.nn import Module

"""
Device-generic version of the `cpu_cuda` mode in test_data_parallel.py.
Every attribution config in captum/testing/attr/helpers/test_config.py is run on
CPU and on an accelerator and the results are compared. The accelerator is chosen
from, in order: the CAPTUM_TEST_DEVICE env var (e.g. "tpu", "cuda", "mps"),
torch.accelerator.current_accelerator(), or CUDA if available. Tests are only
generated when a non-CPU device is found.
"""


def _accelerator_device() -> Optional[torch.device]:
    env = os.environ.get("CAPTUM_TEST_DEVICE")
    if env:
        return torch.device(env)
    acc = getattr(torch, "accelerator", None)
    if acc is not None and acc.is_available():
        dev = acc.current_accelerator()
        if dev is not None and dev.type != "cpu":
            return dev
    if torch.cuda.is_available() and torch.cuda.device_count() != 0:
        return torch.device("cuda")
    return None


DEVICE: Optional[torch.device] = _accelerator_device()


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return value


class AcceleratorMeta(type):
    def __new__(
        metacls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]
    ) -> "AcceleratorMeta":
        assert DEVICE is not None
        for test_config in config:
            (
                algorithms,
                model,
                args,
                layer,
                noise_tunnel,
                baseline_distr,
            ) = parse_test_config(test_config)
            delta = test_config["dp_delta"] if "dp_delta" in test_config else 0.0001

            for algorithm in algorithms:
                if not should_create_generated_test(algorithm):
                    continue
                test_method = metacls.make_single_test(
                    algorithm, model, layer, args, delta, noise_tunnel
                )
                test_name = gen_test_name(
                    "test_cpu_vs_" + DEVICE.type,
                    cast(str, test_config["name"]),
                    algorithm,
                    noise_tunnel,
                )
                if test_name in attrs:
                    raise AssertionError(
                        "Trying to overwrite existing test with name: %r" % test_name
                    )
                attrs[test_name] = test_method

        return super(AcceleratorMeta, metacls).__new__(metacls, name, bases, attrs)

    @classmethod
    @deep_copy_args
    def make_single_test(
        metacls,
        algorithm: Type[Attribution],
        model: Module,
        target_layer: Optional[str],
        args: Dict[str, Any],
        delta: float,
        noise_tunnel: bool,
    ) -> Callable[[BaseTest], None]:
        def accelerator_test_assert(self: BaseTest) -> None:
            device = cast(torch.device, DEVICE)
            dev_args = {key: _move(val, device) for key, val in args.items()}
            dev_model = copy.deepcopy(model).to(device)

            attr_cpu: Attribution
            attr_dev: Attribution
            if target_layer:
                internal = cast(Type[InternalAttribution[Module]], algorithm)
                attr_cpu = internal(model, get_target_layer(model, target_layer))
                attr_dev = internal(
                    dev_model, get_target_layer(dev_model, target_layer)
                )
            else:
                attr_cpu = algorithm(model)
                attr_dev = algorithm(dev_model)
            if noise_tunnel:
                attr_cpu = NoiseTunnel(attr_cpu)
                attr_dev = NoiseTunnel(attr_dev)

            if attr_cpu.has_convergence_delta():
                out_cpu, delta_cpu = attr_cpu.attribute(
                    return_convergence_delta=True, **args
                )
                self.setUp()
                out_dev, delta_dev = attr_dev.attribute(
                    return_convergence_delta=True, **dev_args
                )
                assertTensorTuplesAlmostEqual(
                    self,
                    delta_cpu,
                    _move(delta_dev, torch.device("cpu")),
                    mode="max",
                    delta=delta,
                )
            else:
                out_cpu = attr_cpu.attribute(**args)
                self.setUp()
                out_dev = attr_dev.attribute(**dev_args)

            if isinstance(out_cpu, list):
                for i in range(len(out_cpu)):
                    assertTensorTuplesAlmostEqual(
                        self,
                        out_cpu[i],
                        _move(out_dev[i], torch.device("cpu")),
                        mode="max",
                        delta=delta,
                    )
            else:
                assertTensorTuplesAlmostEqual(
                    self,
                    out_cpu,
                    _move(out_dev, torch.device("cpu")),
                    mode="max",
                    delta=delta,
                )

        return accelerator_test_assert


if DEVICE is not None and DEVICE.type != "cpu":

    class AcceleratorCompareTest(BaseTest, metaclass=AcceleratorMeta):
        @classmethod
        def setUpClass(cls) -> None:
            # Accelerators that run fp32 matmuls at reduced precision by default
            # (TPU, TF32 on Ampere+) differ from CPU by ~1e-3, above dp_delta.
            cls._prev_precision = torch.get_float32_matmul_precision()
            torch.set_float32_matmul_precision("highest")

        @classmethod
        def tearDownClass(cls) -> None:
            torch.set_float32_matmul_precision(cls._prev_precision)
