#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import torch
import torch.nn as nn
from captum._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_with_sample_wise_trick,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicLinearModel2,
    BasicLinearModel_Multilayer,
)


class Test(BaseTest):
    def test_jacobian_scores_single_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float().reshape(1, 5))

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], a)

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], a)

    def test_jacobian_scores_single_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.ones(5).unsqueeze(0)
        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((a, a)))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((a, a)))

    def test_jacobian_scores_single_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((a, 2 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35]]))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((a, 2 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35]]))

    def test_jacobian_scores_single_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.ones(5).unsqueeze(0)

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((2 * a, 4 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.cat((2 * a, 4 * a)))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))

    def test_jacobian_scores_batch_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float().reshape(1, 5))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], a[0:1])
        assertTensorAlmostEqual(self, grads[0][1], a[1:2])

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], a[0:1])
        assertTensorAlmostEqual(self, grads[0][1], a[1:2])

    def test_jacobian_scores_batch_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], a[0])))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], a[1])))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], a[0])))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], a[1])))

    def test_jacobian_scores_batch_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], 2 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], 2 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70]]))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((a[0], 2 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((a[1], 2 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70]]))

    def test_jacobian_scores_batch_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))

        grads = _compute_jacobian_wrt_params(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a[0], 4 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((2 * a[1], 4 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70], [20, 70]]))

        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(model, (a,))
        assertTensorAlmostEqual(self, grads[0][0], torch.stack((2 * a[0], 4 * a[0])))
        assertTensorAlmostEqual(self, grads[1][0], torch.Tensor([[10, 35], [10, 35]]))
        assertTensorAlmostEqual(self, grads[0][1], torch.stack((2 * a[1], 4 * a[1])))
        assertTensorAlmostEqual(self, grads[1][1], torch.Tensor([[20, 70], [20, 70]]))

    def test_jacobian_loss_single_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).view(1, 5).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([9])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)
        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a)

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )
        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a)

    def test_jacobian_loss_single_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([[9, 38]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.cat((2 * (10 - 9) * a, 2 * (35 - 38) * a))
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )
        assertTensorAlmostEqual(
            self, grads[0][0], torch.cat((2 * (10 - 9) * a, 2 * (35 - 38) * a))
        )

    def test_jacobian_loss_batch_scalar(self) -> None:
        model = BasicLinearModel2(5, 1)
        model.linear.weight = nn.Parameter(torch.arange(0, 5).float().reshape(1, 5))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9], [18]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)

        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a[0:1])
        assertTensorAlmostEqual(self, grads[0][1], 2 * (20 - 18) * a[1:2])

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )

        assertTensorAlmostEqual(self, grads[0][0], 2 * (10 - 9) * a[0:1])
        assertTensorAlmostEqual(self, grads[0][1], 2 * (20 - 18) * a[1:2])

    def test_jacobian_loss_batch_vector(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_single_scalar_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 1)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(1, 3).view(1, 2).float())

        a = torch.ones(5).unsqueeze(0)
        label = torch.Tensor([[78]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)
        assertTensorAlmostEqual(
            self, grads[0][0], torch.cat((2 * (80 - 78) * a, 2 * 2 * (80 - 78) * a))
        )
        assertTensorAlmostEqual(
            self, grads[1][0], 2 * (80 - 78) * torch.Tensor([[10, 35]])
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )
        assertTensorAlmostEqual(
            self, grads[0][0], torch.cat((2 * (80 - 78) * a, 2 * 2 * (80 - 78) * a))
        )
        assertTensorAlmostEqual(
            self, grads[1][0], 2 * (80 - 78) * torch.Tensor([[10, 35]])
        )

    def test_jacobian_loss_batch_vector_multilayer(self) -> None:
        model = BasicLinearModel_Multilayer(5, 2, 2)
        model.linear1.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        model.linear2.weight = nn.Parameter(torch.arange(0, 4).view(2, 2).float())

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[33, 124], [69, 256]])

        loss_fn = nn.MSELoss(reduction="none")
        grads = _compute_jacobian_wrt_params(model, (a,), label, loss_fn)
        assertTensorAlmostEqual(
            self,
            grads[0][0],
            torch.stack(
                (
                    2 * (0 * (35 - 33) + 2 * (125 - 124)) * a[0],
                    2 * (1 * (35 - 33) + 3 * (125 - 124)) * a[0],
                )
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[1][0],
            torch.Tensor(
                [
                    [2 * (35 - 33) * 10, 2 * (35 - 33) * 35],
                    [2 * (125 - 124) * 10, 2 * (125 - 124) * 35],
                ]
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[0][1],
            torch.stack(
                (
                    2 * (0 * (70 - 69) + 2 * (250 - 256)) * a[1],
                    2 * (1 * (70 - 69) + 3 * (250 - 256)) * a[1],
                )
            ),
        )
        assertTensorAlmostEqual(
            self,
            grads[1][1],
            torch.Tensor(
                [
                    [2 * (70 - 69) * 10 * 2, 2 * (70 - 69) * 35 * 2],
                    [2 * (250 - 256) * 10 * 2, 2 * (250 - 256) * 35 * 2],
                ]
            ),
        )

        loss_fn = nn.MSELoss(reduction="sum")
        grads_h = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, loss_fn
        )
        assertTensorAlmostEqual(self, grads_h[0][0], grads[0][0])
        assertTensorAlmostEqual(self, grads_h[1][0], grads[1][0])
        assertTensorAlmostEqual(self, grads_h[0][1], grads[0][1])
        assertTensorAlmostEqual(self, grads_h[1][1], grads[1][1])

    def test_jacobian_loss_custom_correct(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return (out - label).pow(2)

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])
        grads = _compute_jacobian_wrt_params(model, (a,), label, my_loss)

        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_custom_wrong(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.sum((out - label).pow(2))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params(model, (a,), label, my_loss)

    def test_jacobian_loss_custom_correct_sample_wise_trick(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return torch.sum((out - label).pow(2))

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])
        grads = _compute_jacobian_wrt_params_with_sample_wise_trick(
            model, (a,), label, my_loss  # type: ignore
        )
        assertTensorAlmostEqual(
            self, grads[0][0], torch.stack((2 * (10 - 9) * a[0], 2 * (35 - 38) * a[0]))
        )
        assertTensorAlmostEqual(
            self, grads[0][1], torch.stack((2 * (20 - 18) * a[1], 2 * (70 - 74) * a[1]))
        )

    def test_jacobian_loss_custom_wrong_sample_wise_trick(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())

        def my_loss(out, label):
            return (out - label).pow(2)

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params_with_sample_wise_trick(
                model, (a,), label, my_loss  # type: ignore
            )

    def test_jacobian_loss_wrong_reduction_sample_wise_trick(self) -> None:
        model = BasicLinearModel2(5, 2)
        model.linear.weight = nn.Parameter(torch.arange(0, 10).view(2, 5).float())
        loss_fn = nn.MSELoss(reduction="none")

        a = torch.stack((torch.ones(5), torch.ones(5) * 2))
        label = torch.Tensor([[9, 38], [18, 74]])

        with self.assertRaises(AssertionError):
            _compute_jacobian_wrt_params_with_sample_wise_trick(
                model, (a,), label, loss_fn
            )
