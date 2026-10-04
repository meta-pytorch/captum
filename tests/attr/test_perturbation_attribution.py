#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import torch
from captum.attr._utils.attribution import PerturbationAttribution
from captum.testing.helpers import BaseTest
from torch import Tensor


# pyrefly: ignore [invalid-inheritance]
class PerturbationAttributionTest(BaseTest):
    def test_undecorated_attribute_is_rejected(self) -> None:
        with self.assertRaises(TypeError):

            class Undecorated(PerturbationAttribution):
                def attribute(self, inputs: Tensor) -> Tensor:
                    return inputs

    def test_no_grad_decorated_attribute_is_accepted(self) -> None:
        class Decorated(PerturbationAttribution):
            @torch.no_grad()
            def attribute(self, inputs: Tensor) -> Tensor:
                return inputs

        self.assertTrue(issubclass(Decorated, PerturbationAttribution))
