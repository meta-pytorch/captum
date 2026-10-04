#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import typing
from typing import Any, cast, Dict, List, Tuple, Type, Union

from captum.attr._core.lime import Lime
from captum.attr._models.base import _get_deep_layer_name
from captum.attr._utils.attribution import Attribution
from torch.nn import Module


def gen_test_name(
    prefix: str, test_name: str, algorithm: Type[Attribution], noise_tunnel: bool
) -> str:
    # Generates test name for dynamically generated tests
    return (
        prefix
        + "_"
        + test_name
        + "_"
        + algorithm.__name__
        + ("NoiseTunnel" if noise_tunnel else "")
    )


def parse_test_config(
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    test_config: Dict,
) -> Tuple[List[Type[Attribution]], Module, Dict[str, Any], Module, bool, bool]:
    algorithms = cast(List[Type[Attribution]], test_config["algorithms"])
    model = test_config["model"]
    args = cast(Dict[str, Any], test_config["attribute_args"])
    layer = test_config["layer"] if "layer" in test_config else None
    noise_tunnel = (
        test_config["noise_tunnel"] if "noise_tunnel" in test_config else False
    )
    baseline_distr = (
        test_config["baseline_distr"] if "baseline_distr" in test_config else False
    )
    return algorithms, model, args, layer, noise_tunnel, baseline_distr  # type: ignore


def should_create_generated_test(algorithm: Type[Attribution]) -> bool:
    if issubclass(algorithm, Lime):
        try:
            import sklearn  # noqa: F401

            assert (
                sklearn.__version__ >= "0.23.0"
            ), "Must have sklearn version 0.23.0 or higher to use "
            "sample_weight in Lasso regression."
            return True
        except (ImportError, AssertionError):
            return False
    return True


@typing.overload
def get_target_layer(model: Module, layer_name: str) -> Module: ...


@typing.overload
def get_target_layer(model: Module, layer_name: List[str]) -> List[Module]: ...


def get_target_layer(
    model: Module, layer_name: Union[str, List[str]]
) -> Union[Module, List[Module]]:
    if isinstance(layer_name, str):
        return _get_deep_layer_name(model, layer_name)
    else:
        return [
            _get_deep_layer_name(model, single_layer_name)
            for single_layer_name in layer_name
        ]
