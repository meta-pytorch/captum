# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Tuple

import torch
from torch import Tensor


def create_inps_and_base_for_deeplift_neuron_layer_testing() -> (
    Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
):
    x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

    b1 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    b2 = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)

    inputs = (x1, x2)
    baselines = (b1, b2)

    return inputs, baselines


def create_inps_and_base_for_deepliftshap_neuron_layer_testing() -> (
    Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]
):
    x1 = torch.tensor([[-10.0, 1.0, -5.0]], requires_grad=True)
    x2 = torch.tensor([[3.0, 3.0, 1.0]], requires_grad=True)

    b1 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
    )
    b2 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True
    )

    inputs = (x1, x2)
    baselines = (b1, b2)

    return inputs, baselines
