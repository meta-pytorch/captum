# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import List, Tuple

import torch
from captum._utils.gradient import compute_gradients
from captum.testing.helpers.basic_models import BasicModel, BasicModel5_MultiArgs
from torch import Tensor
from torch.nn import Module


def get_basic_config() -> Tuple[Module, Tensor, Tensor, None]:
    input = torch.tensor([1.0, 2.0, 3.0, 0.0, -1.0, 7.0], requires_grad=True).T
    # manually percomputed gradients
    grads = torch.tensor([-0.0, -0.0, -0.0, 1.0, 1.0, -0.0])
    return BasicModel(), input, grads, None


def get_multiargs_basic_config() -> (
    Tuple[Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[List[int], int]]
):
    model = BasicModel5_MultiArgs()
    additional_forward_args = ([2, 3], 1)
    inputs = (
        torch.tensor([[1.5, 2.0, 34.3], [3.4, 1.2, 2.0]], requires_grad=True),
        torch.tensor([[3.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True),
    )
    grads = compute_gradients(
        model, inputs, additional_forward_args=additional_forward_args
    )
    return model, inputs, grads, additional_forward_args


def get_multiargs_basic_config_large() -> (
    Tuple[Module, Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[List[int], int]]
):
    model = BasicModel5_MultiArgs()
    additional_forward_args = ([2, 3], 1)
    inputs = (
        torch.tensor(
            [[10.5, 12.0, 34.3], [43.4, 51.2, 32.0]], requires_grad=True
        ).repeat_interleave(3, dim=0),
        torch.tensor(
            [[1.0, 3.5, 23.2], [2.3, 1.2, 0.3]], requires_grad=True
        ).repeat_interleave(3, dim=0),
    )
    grads = compute_gradients(
        model, inputs, additional_forward_args=additional_forward_args
    )
    return model, inputs, grads, additional_forward_args
