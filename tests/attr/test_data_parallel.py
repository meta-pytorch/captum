#!/usr/bin/env python3

# pyre-strict
import copy
import os
from enum import Enum
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type

import torch
import torch.distributed as dist
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from captum.attr._core.layer.layer_lrp import LayerLRP
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
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
Tests in this file are dynamically generated based on the config
defined in tests/attr/helpers/test_config.py. To add new test cases,
read the documentation in test_config.py and add cases based on the
schema described there.
"""

# Distributed Data Parallel env setup
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"
dist.init_process_group(backend="gloo", rank=0, world_size=1)


class DataParallelCompareMode(Enum):
    """
    Defines modes for DataParallel tests:
    `cpu_cuda` - Compares results when running attribution method on CPU vs GPU / CUDA
    `data_parallel_default` - Compares results when running attribution method on GPU
        with DataParallel
    `data_parallel_alt_dev_ids` - Compares results when running attribution method on
        GPU with DataParallel, but with an alternate device ID ordering (not default)
    """

    cpu_cuda = 1
    data_parallel_default = 2
    data_parallel_alt_dev_ids = 3
    dist_data_parallel = 4


class DataParallelMeta(type):
    def __new__(
        metacls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]
    ) -> "DataParallelMeta":
        for test_config in config:
            (
                algorithms,
                model,
                args,
                layer,
                noise_tunnel,
                baseline_distr,
            ) = parse_test_config(test_config)
            dp_delta = test_config["dp_delta"] if "dp_delta" in test_config else 0.0001

            for algorithm in algorithms:
                if not should_create_generated_test(algorithm):
                    continue
                for mode in DataParallelCompareMode:
                    # Creates test case corresponding to each algorithm and
                    # DataParallelCompareMode
                    test_method = metacls.make_single_dp_test(
                        algorithm,
                        model,
                        layer,
                        args,
                        dp_delta,
                        noise_tunnel,
                        baseline_distr,
                        mode,
                    )
                    test_name = gen_test_name(
                        "test_dp_" + mode.name,
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

        return super(DataParallelMeta, metacls).__new__(metacls, name, bases, attrs)

    # Arguments are deep copied to ensure tests are independent and are not affected
    # by any modifications within a previous test.
    @classmethod
    @deep_copy_args
    def make_single_dp_test(
        metacls,
        algorithm: Type[Attribution],
        model: Module,
        target_layer: Optional[str],
        args: Dict[str, Any],
        dp_delta: float,
        noise_tunnel: bool,
        baseline_distr: bool,
        mode: DataParallelCompareMode,
    ) -> Callable[[BaseTest], None]:
        """
        This method creates a single Data Parallel / GPU test for the given
        algorithm and parameters.
        """

        def data_parallel_test_assert(self: BaseTest) -> None:
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

            cuda_model = copy.deepcopy(model).cuda()
            # Set up test arguments based on DataParallelCompareMode
            model_1, model_2, args_1, args_2, alt_device_ids = _get_dp_test_args(
                cuda_model, model, cuda_args, args, mode
            )

            # Construct attribution methods
            attr_method_1, attr_method_2 = _get_dp_attr_methods(
                algorithm,
                target_layer,
                model_1,
                model_2,
                cuda_model,
                alt_device_ids,
                noise_tunnel,
            )
            if attr_method_1.has_convergence_delta():
                attributions_1, delta_1 = attr_method_1.attribute(
                    return_convergence_delta=True, **args_1
                )
                self.setUp()
                attributions_2, delta_2 = attr_method_2.attribute(
                    return_convergence_delta=True, **args_2
                )
                if isinstance(attributions_1, list):
                    for i in range(len(attributions_1)):
                        assertTensorTuplesAlmostEqual(
                            self,
                            attributions_1[i],
                            attributions_2[i],
                            mode="max",
                            delta=dp_delta,
                        )
                else:
                    assertTensorTuplesAlmostEqual(
                        self, attributions_1, attributions_2, mode="max", delta=dp_delta
                    )
                assertTensorTuplesAlmostEqual(
                    self, delta_1, delta_2, mode="max", delta=dp_delta
                )
            else:
                attributions_1 = attr_method_1.attribute(**args_1)
                self.setUp()
                attributions_2 = attr_method_2.attribute(**args_2)
                if isinstance(attributions_1, list):
                    for i in range(len(attributions_1)):
                        assertTensorTuplesAlmostEqual(
                            self,
                            attributions_1[i],
                            attributions_2[i],
                            mode="max",
                            delta=dp_delta,
                        )
                else:
                    assertTensorTuplesAlmostEqual(
                        self, attributions_1, attributions_2, mode="max", delta=dp_delta
                    )

        return data_parallel_test_assert


def _get_dp_test_args(
    cuda_model: Module,
    model: Module,
    cuda_args: Dict[str, Any],
    args: Dict[str, Any],
    mode: DataParallelCompareMode,
) -> Tuple[Module, Module, Dict[str, Any], Dict[str, Any], Optional[List[int]]]:
    # Initialize models based on DataParallelCompareMode
    alt_device_ids = None
    if mode is DataParallelCompareMode.cpu_cuda:
        model_1, model_2 = model, cuda_model
        args_1, args_2 = args, cuda_args
    elif mode is DataParallelCompareMode.data_parallel_default:
        model_1, model_2 = (
            cuda_model,
            torch.nn.parallel.DataParallel(cuda_model),
        )
        args_1, args_2 = cuda_args, cuda_args
    elif mode is DataParallelCompareMode.data_parallel_alt_dev_ids:
        alt_device_ids = [0] + list(range(torch.cuda.device_count() - 1, 0, -1))
        model_1, model_2 = (
            cuda_model,
            torch.nn.parallel.DataParallel(cuda_model, device_ids=alt_device_ids),
        )
        args_1, args_2 = cuda_args, cuda_args
    elif mode is DataParallelCompareMode.dist_data_parallel:

        model_1, model_2 = (
            cuda_model,
            torch.nn.parallel.DistributedDataParallel(
                cuda_model, device_ids=[0], output_device=0
            ),
        )
        args_1, args_2 = cuda_args, cuda_args
    else:
        raise AssertionError("DataParallel compare mode type is not valid.")

    return model_1, model_2, args_1, args_2, alt_device_ids


def _get_dp_attr_methods(
    algorithm: Type[Attribution],
    target_layer: Optional[str],
    model_1: Module,
    model_2: Module,
    cuda_model: Module,
    alt_device_ids: Optional[List[int]],
    noise_tunnel: bool,
) -> Tuple[Attribution, Attribution]:
    attr_method_1: Attribution
    attr_method_2: Attribution
    if target_layer:
        internal_algorithm = cast(Type[InternalAttribution[Module]], algorithm)
        attr_method_1 = internal_algorithm(
            model_1, get_target_layer(model_1, target_layer)
        )
        # cuda_model is used to obtain target_layer since DataParallel
        # adds additional wrapper.
        # model_2 is always either the CUDA model itself or DataParallel
        if alt_device_ids is None:
            attr_method_2 = internal_algorithm(
                model_2, get_target_layer(cuda_model, target_layer)
            )
        else:
            # LayerDeepLift and LayerDeepLiftShap do not take device ids
            # as a parameter, since they must always have the DataParallel
            # model object directly.
            # Some neuron methods and GuidedGradCAM also require the
            # model and cannot take a forward function.
            if issubclass(
                internal_algorithm,
                (
                    LayerDeepLift,
                    LayerDeepLiftShap,
                    LayerLRP,
                    NeuronDeepLift,
                    NeuronDeepLiftShap,
                    NeuronDeconvolution,
                    NeuronGuidedBackprop,
                    GuidedGradCam,
                ),
            ):
                attr_method_2 = internal_algorithm(
                    model_2,
                    get_target_layer(cuda_model, target_layer),  # type: ignore
                )
            else:
                attr_method_2 = internal_algorithm(
                    model_2.forward,
                    get_target_layer(cuda_model, target_layer),
                    device_ids=alt_device_ids,
                )
    else:
        attr_method_1 = algorithm(model_1)
        attr_method_2 = algorithm(model_2)
    if noise_tunnel:
        attr_method_1 = NoiseTunnel(attr_method_1)
        attr_method_2 = NoiseTunnel(attr_method_2)
    return attr_method_1, attr_method_2


if torch.cuda.is_available() and torch.cuda.device_count() != 0:

    class DataParallelTest(BaseTest, metaclass=DataParallelMeta):
        @classmethod
        def tearDownClass(cls) -> None:
            if torch.distributed.is_initialized():
                dist.destroy_process_group()
