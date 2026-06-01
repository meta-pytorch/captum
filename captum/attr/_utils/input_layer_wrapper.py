#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import inspect
from typing import List

import torch.nn as nn
from torch import Tensor


class InputIdentity(nn.Module):
    def __init__(self, input_name: str) -> None:
        r"""
        The identity operation

        Args:
            input_name (str)
                The name of the input this layer is associated to. For debugging
                purposes.
        """
        super().__init__()
        self.input_name = input_name

    def forward(self, x: Tensor) -> Tensor:
        return x


class ModelInputWrapper(nn.Module):
    def __init__(self, module_to_wrap: nn.Module) -> None:
        r"""
        This is a convenience class. This wraps a model via first feeding the
        model's inputs to separate layers (one for each input) and then feeding
        the (unmodified) inputs to the underlying model (`module_to_wrap`). Each
        input is fed through an `InputIdentity` layer/module. This class does
        not change how you feed inputs to your model, so feel free to use your
        model as you normally would.

        To access a wrapped input layer, simply access it via the `input_maps`
        ModuleDict, e.g. to get the corresponding module for input "x", simply
        provide/write `my_wrapped_module.input_maps["x"]`

        This is done such that one can use layer attribution methods on inputs.
        Which should allow you to use mix layers with inputs with these
        attribution methods. This is especially useful multimodal models which
        input discrete features (mapped to embeddings, such as text) and regular
        continuous feature vectors.

        Notes:
        - Since inputs are mapped with the identity, attributing to the
          input/feature can be done with either the input or output of the
          layer, e.g. attributing to an input/feature doesn't depend on whether
          attribute_to_layer_input is True or False for
          LayerIntegratedGradients.
        - Please refer to the multimodal tutorial or unit tests
          (test/attr/test_layer_wrapper.py) for an example.

        Args:
            module_to_wrap (nn.Module):
                The model/module you want to wrap
        """
        super().__init__()
        self.module = module_to_wrap

        # ignore self
        self.arg_name_list: List[str] = inspect.getfullargspec(
            module_to_wrap.forward
        ).args[1:]
        self.input_maps = nn.ModuleDict(
            {arg_name: InputIdentity(arg_name) for arg_name in self.arg_name_list}
        )

    def forward(self, *args: object, **kwargs: object) -> object:
        args_list = list(args)
        for idx, (arg_name, arg) in enumerate(zip(self.arg_name_list, args_list)):
            args_list[idx] = self.input_maps[arg_name](arg)

        for arg_name in kwargs.keys():
            kwargs[arg_name] = self.input_maps[arg_name](kwargs[arg_name])

        return self.module(*tuple(args_list), **kwargs)
