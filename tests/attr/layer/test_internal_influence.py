#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
import unittest
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.typing import BaselineType
from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.testing.helpers.basic import assertTensorTuplesAlmostEqual, BaseTest
from captum.testing.helpers.basic_models import (
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_input_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(net, net.linear0, inp, [[3.9, 3.9, 3.9]])

    def test_simple_input_multi_internal_inf(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(
            net,
            net.multi_relu,
            inp,
            ([[0.9, 1.0, 1.0, 1.0]], [[0.9, 1.0, 1.0, 1.0]]),
            attribute_to_layer_input=True,
        )

    def test_simple_linear_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.9, 1.0, 1.0, 1.0]]
        )

    def test_simple_relu_input_internal_inf_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.relu, inp, ([0.9, 1.0, 1.0, 1.0],), attribute_to_layer_input=True
        )

    def test_simple_linear_internal_inf_inplace(self) -> None:
        net = BasicModel_MultiLayer(inplace=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.9, 1.0, 1.0, 1.0]]
        )

    def test_simple_relu_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(net, net.relu, inp, [[1.0, 1.0, 1.0, 1.0]])

    def test_simple_output_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._internal_influence_test_assert(net, net.linear2, inp, [[1.0, 0.0]])

    def test_simple_with_baseline_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0]])
        base = torch.tensor([[0.0, -20.0, 0.0]])
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8]], base
        )

    def test_simple_multi_input_linear2_internal_inf(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            [[1.0, 0.0]],
            additional_args=(4,),
        )

    def test_simple_multi_input_relu_internal_inf(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            [[1.0, 1.0, 1.0, 1.0]],
            additional_args=(inp3, 5),
        )

    def test_simple_multi_input_batch_relu_internal_inf(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 80.0, 0.0]])
        inp2 = torch.tensor([[0.0, 6.0, 14.0], [0.0, 20.0, 0.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 20.0, 0.0]])
        self._internal_influence_test_assert(
            net,
            net.model.linear1,
            (inp1, inp2),
            [[0.95, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            additional_args=(inp3, 5),
        )

    def test_multiple_linear_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor(
            [
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 100.0, 0.0],
            ],
            requires_grad=True,
        )
        self._internal_influence_test_assert(
            net,
            net.linear1,
            inp,
            [
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
                [0.9, 1.0, 1.0, 1.0],
            ],
        )

    def test_multiple_with_baseline_internal_inf(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 80.0, 0.0], [30.0, 30.0, 0.0]], requires_grad=True)
        base = torch.tensor(
            [[0.0, -20.0, 0.0], [-20.0, -20.0, 0.0]], requires_grad=True
        )
        self._internal_influence_test_assert(
            net, net.linear1, inp, [[0.7, 0.8, 0.8, 0.8], [0.5, 0.6, 0.6, 0.6]], base
        )

    def test_simple_input_internal_inf_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._internal_influence_test_assert(
            net,
            net.multi_relu,
            inp,
            ([[0.9, 1.0, 1.0, 1.0]], [[0.9, 1.0, 1.0, 1.0]]),
            attribute_to_layer_input=True,
            grad_kwargs={"materialize_grads": True},
        )

    def _internal_influence_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_activation: Union[
            float,
            List[List[float]],
            Tuple[List[float], ...],
            Tuple[List[List[float]], ...],
        ],
        baseline: BaselineType = None,
        additional_args: Any = None,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        for internal_batch_size in [None, 5, 20]:
            int_inf = InternalInfluence(model, target_layer)
            self.assertFalse(int_inf.multiplies_by_inputs)
            attributions = int_inf.attribute(
                test_input,
                baselines=baseline,
                target=0,
                n_steps=500,
                method="riemann_trapezoid",
                additional_forward_args=additional_args,
                internal_batch_size=internal_batch_size,
                attribute_to_layer_input=attribute_to_layer_input,
                grad_kwargs=grad_kwargs,
            )
            assertTensorTuplesAlmostEqual(
                self, attributions, expected_activation, delta=0.01, mode="max"
            )


if __name__ == "__main__":
    unittest.main()
