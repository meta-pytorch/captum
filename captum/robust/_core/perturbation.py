#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable


class Perturbation:
    r"""
    All perturbation and attack algorithms extend this class. It enforces
    its child classes to extend and override core `perturb` method.
    """

    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    # pyre-fixme[13]: Attribute `perturb` is never initialized.
    perturb: Callable
    r"""
    This method computes and returns the perturbed input for each input tensor.
    Deriving classes are responsible for implementing its logic accordingly.

    Specific adversarial attack algorithms that extend this class take relevant
    arguments.

    Args:

        inputs (Tensor or tuple[Tensor, ...]): Input for which adversarial attack
                    is computed. It can be provided as a single tensor or
                    a tuple of multiple tensors. If multiple input tensors
                    are provided, the batch sizes must be aligned across all
                    tensors.

    Returns:

        - **perturbed inputs** (*Tensor* or *tuple[Tensor, ...]*):
                    Perturbed input for each
                    input tensor. The perturbed inputs have the same shape and
                    dimensionality as the inputs.
                    If a single tensor is provided as inputs, a single tensor
                    is returned. If a tuple is provided for inputs, a tuple of
                    corresponding sized tensors is returned.
    """

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def __call__(self, *args, **kwargs):
        return self.perturb(*args, **kwargs)
