#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import unittest
from typing import List, Tuple

import torch
from captum._utils.gradient import (
    apply_gradient_requirements,
    compute_gradients,
    compute_layer_gradients_and_eval,
    undo_gradient_requirements,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel5_MultiArgs,
    BasicModel6_MultiTensor,
    BasicModel_MultiLayer,
)
from packaging import version


class Test(BaseTest):
    def test_apply_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (torch.tensor([[5.0]]), test_tensor, torch.tensor([[7.0]]))
        out_mask = apply_gradient_requirements(test_tensor_tuple)
        for i in range(len(test_tensor_tuple)):
            self.assertTrue(test_tensor_tuple[i].requires_grad)
            self.assertEqual(out_mask[i], initial_grads[i])

    def test_undo_gradient_reqs(self) -> None:
        initial_grads = [False, True, False]
        test_tensor = torch.tensor([[6.0]], requires_grad=True)
        test_tensor.grad = torch.tensor([[7.0]])
        test_tensor_tuple = (
            torch.tensor([[6.0]], requires_grad=True),
            test_tensor,
            torch.tensor([[7.0]], requires_grad=True),
        )
        undo_gradient_requirements(test_tensor_tuple, initial_grads)
        for i in range(len(test_tensor_tuple)):
            self.assertEqual(test_tensor_tuple[i].requires_grad, initial_grads[i])

    def test_gradient_basic(self) -> None:
        model = BasicModel()
        input = torch.tensor([[5.0]], requires_grad=True)
        input.grad = torch.tensor([[9.0]])
        grads = compute_gradients(model, input)[0]
        assertTensorAlmostEqual(self, grads, [[0.0]], delta=0.01, mode="max")
        # Verify grad attribute is not altered
        assertTensorAlmostEqual(self, input.grad, [[9.0]], delta=0.0, mode="max")

    def test_gradient_basic_2(self) -> None:
        model = BasicModel()
        input = torch.tensor([[-3.0]], requires_grad=True)
        input.grad = torch.tensor([[14.0]])
        grads = compute_gradients(model, input)[0]
        assertTensorAlmostEqual(self, grads, [[1.0]], delta=0.01, mode="max")
        # Verify grad attribute is not altered
        assertTensorAlmostEqual(self, input.grad, [[14.0]], delta=0.0, mode="max")

    def test_gradient_multiinput(self) -> None:
        model = BasicModel6_MultiTensor()
        input1 = torch.tensor([[-3.0, -5.0]], requires_grad=True)
        input2 = torch.tensor([[-5.0, 2.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2))
        assertTensorAlmostEqual(self, grads[0], [[0.0, 1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[0.0, 1.0]], delta=0.01, mode="max")

    def test_gradient_additional_args(self) -> None:
        model = BasicModel4_MultiArgs()
        input1 = torch.tensor([[10.0]], requires_grad=True)
        input2 = torch.tensor([[8.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), additional_forward_args=(2,))
        assertTensorAlmostEqual(self, grads[0], [[1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[-0.5]], delta=0.01, mode="max")

    def test_gradient_additional_args_2(self) -> None:
        model = BasicModel5_MultiArgs()
        input1 = torch.tensor([[-10.0]], requires_grad=True)
        input2 = torch.tensor([[6.0]], requires_grad=True)
        grads = compute_gradients(
            model, (input1, input2), additional_forward_args=([3, -4],)
        )
        assertTensorAlmostEqual(self, grads[0], [[0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads[1], [[4.0]], delta=0.01, mode="max")

    def test_gradient_target_int(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, 5.0]], requires_grad=True)
        grads0 = compute_gradients(model, (input1, input2), target_ind=0)
        grads1 = compute_gradients(model, (input1, input2), target_ind=1)
        assertTensorAlmostEqual(self, grads0[0], [[1.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads0[1], [[-1.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads1[0], [[0.0, 0.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, grads1[1], [[0.0, 0.0]], delta=0.01, mode="max")

    def test_gradient_target_list(self) -> None:
        model = BasicModel2()
        input1 = torch.tensor([[4.0, -1.0], [3.0, 10.0]], requires_grad=True)
        input2 = torch.tensor([[2.0, -5.0], [-2.0, 1.0]], requires_grad=True)
        grads = compute_gradients(model, (input1, input2), target_ind=[0, 1])
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[1.0, 0.0], [0.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            grads[1],
            [[-1.0, 0.0], [0.0, -1.0]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_target_tuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        grads = compute_gradients(model, input, target_ind=(0, 1))[0]
        assertTensorAlmostEqual(
            self,
            grads,
            [[[0.0, 0.0], [0.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_target_listtuple(self) -> None:
        model = BasicModel()
        input = torch.tensor(
            [[[4.0, 2.0], [-1.0, -2.0]], [[3.0, -4.0], [10.0, 5.0]]], requires_grad=True
        )
        target: List[Tuple[int, ...]] = [(1, 1), (0, 1)]
        grads = compute_gradients(model, input, target_ind=target)[0]
        assertTensorAlmostEqual(
            self,
            grads,
            [[[0.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [0.0, 0.0]]],
            delta=0.01,
            mode="max",
        )

    def test_gradient_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[1.0, 6.0, -3.0]], requires_grad=True)
        grads = compute_gradients(model, input, target_ind=0)[0]
        assertTensorAlmostEqual(self, grads, [[3.0, 3.0, 3.0]], delta=0.01, mode="max")

    def test_layer_gradient_linear0(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, -11.0, 23.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear0, input, target_ind=0
        )
        assertTensorAlmostEqual(
            self, grads[0], [[4.0, 4.0, 4.0]], delta=0.01, mode="max"
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[5.0, -11.0, 23.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_linear1(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_linear1_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear1, input, target_ind=1
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_relu_input_inplace(self) -> None:
        model = BasicModel_MultiLayer(inplace=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.relu, input, target_ind=1, attribute_to_layer_input=True
        )
        assertTensorAlmostEqual(
            self,
            grads[0],
            [[0.0, 1.0, 1.0, 1.0]],
            delta=0.01,
            mode="max",
        )
        assertTensorAlmostEqual(
            self,
            eval[0],
            [[-2.0, 9.0, 9.0, 9.0]],
            delta=0.01,
            mode="max",
        )

    def test_layer_gradient_output(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model, model.linear2, input, target_ind=1
        )
        assertTensorAlmostEqual(self, grads[0], [[0.0, 1.0]], delta=0.01, mode="max")
        assertTensorAlmostEqual(self, eval[0], [[26.0, 28.0]], delta=0.01, mode="max")

    def test_layer_gradient_ignores_non_tensor_layer_outputs(self) -> None:
        class TensorAndMetadataLayer(torch.nn.Module):
            def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, None]:
                return 2 * input, None

        class TensorAndMetadataModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = TensorAndMetadataLayer()

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                layer_output, _ = self.layer(input)
                return 3 * layer_output[:, 0] - 2 * layer_output[:, 1]

        model = TensorAndMetadataModel()
        input = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

        grads, eval = compute_layer_gradients_and_eval(model, model.layer, input)

        assertTensorAlmostEqual(
            self, grads[0], [[3.0, -2.0], [3.0, -2.0]], delta=0.0, mode="max"
        )
        assertTensorAlmostEqual(
            self, eval[0], [[2.0, 4.0], [6.0, 8.0]], delta=0.0, mode="max"
        )

    def test_layer_gradient_lstm_output_with_state_tuple(self) -> None:
        class LSTMModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(3, 2, 1, batch_first=True)
                self.linear = torch.nn.Linear(2, 1, bias=False)
                self.linear.weight = torch.nn.Parameter(torch.tensor([[3.0, -2.0]]))

            def forward(self, input: torch.Tensor) -> torch.Tensor:
                output, _ = self.lstm(input)
                return self.linear(output[:, -1, :]).squeeze(1)

        model = LSTMModel().eval()
        input = torch.randn(4, 5, 3, requires_grad=True)

        grads, eval = compute_layer_gradients_and_eval(model, model.lstm, input)

        expected_grads = torch.zeros_like(eval[0])
        expected_grads[:, -1, :] = model.linear.weight
        assertTensorAlmostEqual(self, grads[0], expected_grads, delta=0.0, mode="max")
        self.assertEqual(eval[0].shape, (4, 5, 2))

    def test_layer_gradient_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )

        model = BasicModel_MultiLayer(multi_input_module=True)
        input = torch.tensor([[5.0, 2.0, 1.0]], requires_grad=True)
        grads, eval = compute_layer_gradients_and_eval(
            model,
            [model.linear1, model.relu],
            input,
            target_ind=1,
            grad_kwargs={"materialize_grads": True},
        )
        assertTensorAlmostEqual(
            self, grads[0][0], [[0.0, 1.0, 1.0, 1.0]], delta=0, mode="max"
        )
        assertTensorAlmostEqual(
            self, eval[0][0], [[-2.0, 9.0, 9.0, 9.0]], delta=0, mode="max"
        )
