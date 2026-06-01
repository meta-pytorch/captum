# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import tempfile
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction
from captum.influence._core.influence_function import NaiveInfluenceFunction
from captum.influence._core.tracincp import TracInCP
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _isSorted,
    _wrap_model_in_dataparallel,
    build_test_name_func,
    CoefficientNet,
    DataInfluenceConstructor,
    IdentityDataset,
    RangeDataset,
)
from parameterized import parameterized
from torch import Tensor


class TestTracInRegression(BaseTest):
    def _test_tracin_regression_setup(
        self, tmpdir: str, features: int, use_gpu: bool = False
    ) -> Tuple[RangeDataset, Dict[str, Any]]:  # fixme (return type)
        low = 1
        high = 17
        dataset = RangeDataset(low, high, features, use_gpu)
        net = CoefficientNet(in_features=features)

        checkpoint_name = "-".join(["checkpoint-reg", "0" + ".pt"])
        torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        for i, weight in enumerate(weights):
            net.fc1.weight.data.fill_(weight)
            net_adjusted = _wrap_model_in_dataparallel(net) if use_gpu else net
            checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
            torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

        # pyre-fixme[61]: `net_adjusted` is undefined, or not always defined.
        return dataset, net_adjusted  # type: ignore

    use_gpu_list = (
        [True, False]
        if torch.cuda.is_available() and torch.cuda.device_count() != 0
        else [False]
    )

    param_list: List[Tuple[Optional[str], DataInfluenceConstructor, str, int, bool]] = (
        []
    )
    for use_gpu in use_gpu_list:
        for dim in [1, 20]:
            for mode, reduction, constructor in [
                (
                    "check_idx",
                    "none",
                    DataInfluenceConstructor(TracInCP, name="TracInCP_all_layers"),
                ),
                (
                    "check_idx",
                    "none",
                    DataInfluenceConstructor(
                        TracInCP,
                        name="TracInCP_fc1",
                        layers=["module.fc1"] if use_gpu else ["fc1"],
                    ),
                ),
                (
                    "sample_wise_trick",
                    None,
                    DataInfluenceConstructor(TracInCP, name="TracInCP_fc1"),
                ),
                (
                    "check_idx",
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFast, name="TracInCPFast_last_fc_layer"
                    ),
                ),
                (
                    "check_idx",
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj, name="TracInCPFast_last_fc_layer"
                    ),
                ),
                (
                    "check_idx",
                    "mean",
                    DataInfluenceConstructor(
                        TracInCPFast, name="TracInCPFast_last_fc_layer"
                    ),
                ),
                (
                    "check_idx",
                    "mean",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj, name="TracInCPFastRandProj_last_fc_layer"
                    ),
                ),
                (
                    "check_idx",
                    "sum",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj,
                        name="TracInCPFastRandProj1DimensionalProjection_last_fc_layer",
                        projection_dim=1,
                    ),
                ),
                (
                    "check_idx",
                    "mean",
                    DataInfluenceConstructor(
                        TracInCPFast,
                        name="TracInCPFastDuplicateLossFn",
                        duplicate_loss_fn=True,
                    ),
                ),  # add a test where `duplicate_loss_fn` is True
                (
                    "check_idx",
                    "mean",
                    DataInfluenceConstructor(
                        TracInCPFastRandProj,
                        name="TracInCPFastRandProjDuplicateLossFn",
                        duplicate_loss_fn=True,
                    ),  # add a test where `duplicate_loss_fn` is True
                ),
            ]:
                if not (mode == "sample_wise_trick" and use_gpu):
                    param_list.append((reduction, constructor, mode, dim, use_gpu))

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func
    # ($parameter$args_to_skip = ["reduction"])` to decorator factory
    # `parameterized.parameterized.expand`.
    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(args_to_skip=["reduction"]),
    )
    def test_tracin_regression(
        self,
        reduction: Optional[str],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
        mode: str,
        features: int,
        use_gpu: bool,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_regression_setup(
                tmpdir,
                features,
                use_gpu,
            )  # and not mode == 'sample_wise_trick'

            # check influence scores of training data

            train_inputs = dataset.samples
            train_labels = dataset.labels

            test_inputs = (
                torch.arange(17, 33, dtype=torch.float).unsqueeze(1).repeat(1, features)
            )

            if use_gpu:
                test_inputs = test_inputs.cuda()

            test_labels = test_inputs

            self.assertTrue(callable(tracin_constructor))

            if mode == "check_idx":

                assert isinstance(reduction, str)
                criterion = nn.MSELoss(reduction=reduction)

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )

                # pyre-fixme[16]: `object` has no attribute `influence`.
                train_scores = tracin.influence((train_inputs, train_labels))
                idx, _ = tracin.influence(
                    (train_inputs, train_labels), k=len(dataset), proponents=True
                )
                # check that top influence is one with maximal value
                # (and hence gradient)
                for i in range(len(idx)):
                    self.assertEqual(idx[i][0], 15)

                # check influence scores of test data
                test_scores = tracin.influence((test_inputs, test_labels))
                idx, _ = tracin.influence(
                    (test_inputs, test_labels), k=len(test_inputs), proponents=True
                )
                # check that top influence is one with maximal value
                # (and hence gradient)
                for i in range(len(idx)):
                    self.assertTrue(_isSorted(idx[i]))

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=True,
                )

                train_scores = tracin.influence((train_inputs, train_labels))
                train_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    (train_inputs, train_labels)
                )
                assertTensorAlmostEqual(
                    self, train_scores, train_scores_sample_wise_trick
                )

                test_scores = tracin.influence((test_inputs, test_labels))
                test_scores_sample_wise_trick = tracin_sample_wise_trick.influence(
                    (test_inputs, test_labels)
                )
                assertTensorAlmostEqual(
                    self, test_scores, test_scores_sample_wise_trick
                )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func()`
    # to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        [
            (
                "sum",
                DataInfluenceConstructor(TracInCP, sample_wise_grads_per_batch=True),
            ),
            ("sum", DataInfluenceConstructor(TracInCPFast)),
            ("sum", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("mean", DataInfluenceConstructor(TracInCPFast)),
            ("mean", DataInfluenceConstructor(TracInCPFastRandProj)),
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_regression_1D_numerical(
        self,
        reduction: str,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
    ) -> None:

        low = 1
        high = 17
        features = 1
        dataset = RangeDataset(low, high, features)
        net = CoefficientNet()
        criterion = nn.MSELoss(reduction=reduction)
        batch_size = 4
        weights = [0.4379, 0.1653, 0.5132, 0.3651, 0.9992]

        train_inputs = dataset.samples
        train_labels = dataset.labels

        with tempfile.TemporaryDirectory() as tmpdir:

            for i, weight in enumerate(weights):
                net.fc1.weight.data.fill_(weight)
                checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
                torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

            self.assertTrue(callable(tracin_constructor))
            tracin = tracin_constructor(
                net,
                dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # pyre-fixme[16]: `object` has no attribute `influence`.
            train_scores = tracin.influence((train_inputs, train_labels), k=None)

            r"""
            Derivation for gradient / resulting TracIn score:

            For each checkpoint: $y = Wx,$ and $loss = (y - label)^2.$ Recall for this
            test case, there is no activation on y. For this example, $label = x.$

            Fast Rand Proj gives $\nabla_W loss = \nabla_y loss (x^T).$ We have $x$ and
            y as scalars so we can simply multiply. So then,

            \[\nabla_y loss * x = 2(y-x)*x = 2(Wx -x)*x = 2x^2 (w - 1).\]

            And we simply multiply these for x, x'. In this case, $x, x' \in [1..16]$.
            """
            for i in range(train_scores.shape[0]):
                for j in range(len(train_scores[0])):
                    _weights = torch.Tensor(weights)
                    num = 2 * (i + 1) * (i + 1) * (_weights - 1)
                    num *= 2 * (j + 1) * (j + 1) * (_weights - 1)
                    assertTensorAlmostEqual(
                        self, torch.sum(num), train_scores[i][j], delta=0.1
                    )

    def _test_tracin_identity_regression_setup(
        self, tmpdir: str
    ) -> Tuple[IdentityDataset, CoefficientNet]:
        num_features = 7
        dataset = IdentityDataset(num_features)
        net = CoefficientNet()

        num_checkpoints = 5

        for i in range(num_checkpoints):
            net.fc1.weight.data = torch.rand((1, num_features)) * 100
            checkpoint_name = "-".join(["checkpoint-reg", str(i) + ".pt"])
            torch.save(net.state_dict(), os.path.join(tmpdir, checkpoint_name))

        return dataset, net

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func()`
    # to decorator factory `parameterized.parameterized.expand`
    @parameterized.expand(
        [
            ("check_idx", "none", DataInfluenceConstructor(TracInCP)),
            ("check_idx", "none", DataInfluenceConstructor(TracInCP, layers=["fc1"])),
            ("sample_wise_trick", None, DataInfluenceConstructor(TracInCP)),
            (
                "sample_wise_trick",
                None,
                DataInfluenceConstructor(TracInCP, layers=["fc1"]),
            ),
            ("check_idx", "sum", DataInfluenceConstructor(TracInCPFast)),
            ("check_idx", "sum", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("check_idx", "mean", DataInfluenceConstructor(TracInCPFast)),
            ("check_idx", "mean", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("check_idx", "none", DataInfluenceConstructor(NaiveInfluenceFunction)),
            (
                "check_idx",
                "none",
                DataInfluenceConstructor(
                    ArnoldiInfluenceFunction,
                    arnoldi_tol=1e-8,  # needs to be small to avoid empty arnoldi basis
                    hessian_reg=2e-3,
                ),
            ),
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_identity_regression(
        self,
        mode: str,
        reduction: Optional[str],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
    ) -> None:
        """
        This test uses a linear model with positive coefficients, where input feature
        matrix is the identity matrix.  Since the dot product between 2 different
        training instances is always 0, when calculating influence scores on the
        training data, only self influence scores will be nonzero.  Since the linear
        model has positive coefficients, self influence scores will be positive.
        Thus, the training instance with the largest influence on another training
        instance is itself.
        """

        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_identity_regression_setup(tmpdir)

            train_inputs = dataset.samples
            train_labels = dataset.labels

            self.assertTrue(callable(tracin_constructor))

            if mode == "check_idx":

                assert isinstance(reduction, str)
                criterion = nn.MSELoss(reduction=reduction)

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                )

                # check influence scores of training data

                # pyre-fixme[16]: `object` has no attribute `influence`.
                train_scores = tracin.influence((train_inputs, train_labels))
                idx, _ = tracin.influence(
                    (train_inputs, train_labels), k=len(dataset), proponents=True
                )

                # check that top influence for an instance is itself
                for i in range(len(idx)):
                    self.assertEqual(idx[i][0], i)

            if mode == "sample_wise_trick":

                criterion = nn.MSELoss(reduction="none")

                tracin = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=False,
                )

                # With sample-wise trick
                criterion = nn.MSELoss(reduction="sum")
                tracin_sample_wise_trick = tracin_constructor(
                    net,
                    dataset,
                    tmpdir,
                    batch_size,
                    criterion,
                    sample_wise_grads_per_batch=True,
                )

                train_scores = tracin.influence((train_inputs, train_labels))
                train_scores_tracin_sample_wise_trick = (
                    tracin_sample_wise_trick.influence((train_inputs, train_labels))
                )
                assertTensorAlmostEqual(
                    self, train_scores, train_scores_tracin_sample_wise_trick
                )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func()`
    # to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        [
            ("none", "none", DataInfluenceConstructor(TracInCP)),
            (
                "mean",
                "mean",
                DataInfluenceConstructor(TracInCP, sample_wise_grads_per_batch=True),
            ),
            ("sum", "sum", DataInfluenceConstructor(TracInCPFast)),
            ("mean", "mean", DataInfluenceConstructor(TracInCPFast)),
            ("sum", "sum", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("mean", "mean", DataInfluenceConstructor(TracInCPFastRandProj)),
            ("none", "none", DataInfluenceConstructor(NaiveInfluenceFunction)),
            # (
            #    "none",
            #    "none",
            #    DataInfluenceConstructor(ArnoldiInfluenceFunction, arnoldi_tol=1e-9),
            #    # need to set `arnoldi_tol` small. otherwise, arnoldi iteration
            #    # terminates early and get 'Arnoldi basis is empty' exception.
            # ),
        ],
        name_func=build_test_name_func(),
    )
    def test_tracin_constant_test_loss_fn(
        self,
        reduction: Optional[str],
        test_reduction: Optional[str],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
    ) -> None:
        """
        All implementations of `TracInCPBase` can accept `test_loss_fn` in
        initialization, which sets the loss function applied to test examples, which
        can thus be different from the loss function applied to training examples.
        This test passes `test_loss_fn` to be a constant function. Then, the influence
        scores should all be 0, because gradients w.r.t. `test_loss_fn` will all be 0.
        It re-uses the dataset and model from `test_tracin_identity_regression`.

        The reduction for `loss_fn` and `test_loss_fn` initialization arguments is
        the same for all parameterized tests, for simplicity, and also because for
        `TracInCP`, both loss functions must both be reduction loss functions (i.e.
        reduction is "mean" or "sum"), or both be per-example loss functions (i.e.
        reduction is "none"). Recall that for `TracInCP`, the
        `sample_wise_grads_per_batch` initialization argument determines which of
        those cases holds.
        """
        with tempfile.TemporaryDirectory() as tmpdir:

            batch_size = 4

            dataset, net = self._test_tracin_identity_regression_setup(tmpdir)

            train_inputs = dataset.samples
            train_labels = dataset.labels

            self.assertTrue(callable(tracin_constructor))

            assert isinstance(reduction, str)
            criterion = nn.MSELoss(reduction=reduction)

            # the output of `net`, i.e. `input` for the loss functions below, is a
            # batch_size x 1 2D tensor
            if test_reduction == "none":
                # loss function returns 1D tensor of all 0's, so is constant
                def test_loss_fn(input: Tensor, target: int) -> Tensor:
                    return input.squeeze() * 0.0

            elif test_reduction in ["sum", "mean"]:
                # loss function returns scalar tensor of all 0's, so is constant
                def test_loss_fn(input: Tensor, target: int) -> Tensor:
                    return input.mean() * 0.0

            tracin = tracin_constructor(
                net,
                dataset,
                tmpdir,
                batch_size,
                criterion,
                # pyre-fixme[61]: `test_loss_fn` is undefined, or not always defined.
                test_loss_fn=test_loss_fn,
            )

            # check influence scores of training data. they should all be 0
            # pyre-fixme[16]: `object` has no attribute `influence`.
            train_scores = tracin.influence((train_inputs, train_labels), k=None)
            assertTensorAlmostEqual(self, train_scores, torch.zeros(train_scores.shape))
