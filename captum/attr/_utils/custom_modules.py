#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import torch.nn as nn
from torch import Tensor


class Addition_Module(nn.Module):
    """Custom addition module that uses multiple inputs to assure correct relevance
    propagation. Any addition in a forward function needs to be replaced with the
    module before using LRP."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        return x1 + x2
