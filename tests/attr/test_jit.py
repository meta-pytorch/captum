#!/usr/bin/env python3

# pyre-strict

from __future__ import annotations

import unittest
from enum import Enum
from typing import Any, Callable, cast, Dict, Tuple, Type

import torch
from captum._utils.common import (
    _format_additional_forward_args,
    _format_tensor_into_tuples,
)
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.lime import Lime
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.occlusion import Occlusion
from captum.attr._core.saliency import Saliency
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._utils.attribution import Attribution
from captum.testing.attr.helpers.gen_test_utils import (
    gen_test_name,
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

JIT_SUPPORTED = [
    IntegratedGradients,
    FeatureAblation,
    FeaturePermutation,
    GradientShap,
    InputXGradient,
    Occlusion,
    Saliency,
    ShapleyValueSampling,
    Lime,
    KernelShap,
]

"""
Tests in this file are dynamically generated based on the config
defined in tests/attr/helpers/test_config.py. To add new test cases,
read the documentation in test_config.py and add cases based on the
schema described there.
"""


class JITCompareMode(Enum):
    """
    Defines modes for JIT tests:
    `cpu_jit_trace` - Compares results of running the test case with a standard model
        on CPU with the result of JIT tracing the model and computing attributions
    `cpu_jit_script` - Compares results of running the test case with a standard model
        on CPU with the result of JIT scripting the model and computing attributions
    `data_parallel_jit_trace` - Compares results of running the test case with a
        standard model on CPU with the result of JIT tracing the model wrapped in
        DataParallel and computing attributions
    `data_parallel_jit_script` - Compares results of running the test case with a
        standard model on CPU with the result of JIT scripting the model wrapped
        in DataParallel and computing attributions
    """

    cpu_jit_trace = 1
    cpu_jit_script = 2
    data_parallel_jit_trace = 3
    data_parallel_jit_script = 3


class JITMeta(type):
    def __new__(
        metacls,
        name: str,
        bases: Tuple[Type[Any], ...],
        attrs: Dict[str, Callable[..., None]],
    ) -> JITMeta:
        for test_config in config:
            (
                algorithms,
                model,
                args,
                layer,
                noise_tunnel,
                baseline_distr,
            ) = parse_test_config(test_config)
            for algorithm in algorithms:
                if not should_create_generated_test(algorithm):
                    continue
                if algorithm in JIT_SUPPORTED:
                    for mode in JITCompareMode:
                        # Creates test case corresponding to each algorithm and
                        # JITCompareMode
                        test_method = metacls.make_single_jit_test(
                            algorithm, model, args, noise_tunnel, baseline_distr, mode
                        )
                        test_name = gen_test_name(
                            "test_jit_" + mode.name,
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

        return super(JITMeta, metacls).__new__(metacls, name, bases, attrs)

    # Arguments are deep copied to ensure tests are independent and are not affected
    # by any modifications within a previous test.
    @classmethod
    @deep_copy_args
    def make_single_jit_test(
        metacls,
        algorithm: Type[Attribution],
        model: Module,
        args: Dict[str, Any],
        noise_tunnel: bool,
        baseline_distr: bool,
        mode: JITCompareMode,
    ) -> Callable[[BaseTest], None]:
        """
        This method creates a single JIT test for the given algorithm and parameters.
        """

        def jit_test_assert(self: BaseTest) -> None:
            model_1 = model
            attr_args = args
            if (
                mode is JITCompareMode.data_parallel_jit_trace
                or JITCompareMode.data_parallel_jit_script
            ):
                if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
                    raise unittest.SkipTest(
                        "Skipping GPU test since CUDA not available."
                    )
                # Construct cuda_args, moving all tensor inputs in args to CUDA device
                cuda_args = {}
                for key in args:
                    if isinstance(args[key], Tensor):
                        cuda_args[key] = args[key].cuda()
                    elif isinstance(args[key], tuple):
                        cuda_args[key] = tuple(
                            elem.cuda() if isinstance(elem, Tensor) else elem
                            for elem in args[key]
                        )
                    else:
                        cuda_args[key] = args[key]
                attr_args = cuda_args
                model_1 = model_1.cuda()

            # Initialize models based on JITCompareMode
            if (
                mode is JITCompareMode.cpu_jit_script
                or JITCompareMode.data_parallel_jit_script
            ):
                model_2 = torch.jit.script(model_1)  # type: ignore
            elif (
                mode is JITCompareMode.cpu_jit_trace
                or JITCompareMode.data_parallel_jit_trace
            ):
                formatted_inputs = _format_tensor_into_tuples(args["inputs"])
                additional_args: Tuple[Any, ...] = (
                    _format_additional_forward_args(args["additional_forward_args"])
                    if "additional_forward_args" in args
                    and args["additional_forward_args"] is not None
                    else ()
                )
                if formatted_inputs is not None:
                    all_inps = formatted_inputs + additional_args
                else:
                    all_inps = additional_args
                model_2 = torch.jit.trace(model_1, all_inps)  # type: ignore
            else:
                raise AssertionError("JIT compare mode type is not valid.")

            attr_method_1 = algorithm(model_1)
            attr_method_2 = algorithm(model_2)

            if noise_tunnel:
                attr_method_1 = NoiseTunnel(attr_method_1)
                attr_method_2 = NoiseTunnel(attr_method_2)
            if attr_method_1.has_convergence_delta():
                attributions_1, delta_1 = attr_method_1.attribute(
                    return_convergence_delta=True, **attr_args
                )
                self.setUp()
                attributions_2, delta_2 = attr_method_2.attribute(
                    return_convergence_delta=True, **attr_args
                )
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, mode="max"
                )
                assertTensorTuplesAlmostEqual(self, delta_1, delta_2, mode="max")
            else:
                attributions_1 = attr_method_1.attribute(**attr_args)
                self.setUp()
                attributions_2 = attr_method_2.attribute(**attr_args)
                assertTensorTuplesAlmostEqual(
                    self, attributions_1, attributions_2, mode="max"
                )

        return jit_test_assert


if torch.cuda.is_available() and torch.cuda.device_count() != 0:

    class JITTest(BaseTest, metaclass=JITMeta):
        pass
