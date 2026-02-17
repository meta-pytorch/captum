# pyre-unsafe
import tempfile
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from captum.influence._core.influence_function import NaiveInfluenceFunction
from captum.influence._utils.common import (
    _custom_functional_call,
    _flatten_params,
    _functional_call,
    _unflatten_params_factory,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
)
from captum.testing.helpers.influence.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    ExplicitDataset,
    get_random_model_and_data,
    is_gpu,
    Linear,
    UnpackDataset,
)
from parameterized import parameterized
from torch.utils.data import DataLoader

# TODO: for some unknown reason, this test does not work
# on `cuda_data_parallel` setting. We need to investigate why.
# Use a local version of setting list for these two tests for now
# since we have changed the default setting list to includes all options.
# (This is also used in many other tests, which also needs to be unified later).
gpu_settings_list = (
    ["", "cuda"]
    if torch.cuda.is_available() and torch.cuda.device_count() != 0
    else [""]
)


class TestNaiveInfluence(BaseTest):
    def setUp(self) -> None:
        super().setUp()

    @parameterized.expand(
        [
            (param_shape,)
            for param_shape in [
                [(2, 3), (4, 5)],
                [(3, 2), (4, 2), (1, 5)],
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_flatten_unflattener(self, param_shapes: List[Tuple[int, ...]]) -> None:
        # unflatten and flatten should be inverses of each other. check this holds.
        _unflatten_params = _unflatten_params_factory(param_shapes)
        params = tuple(torch.randn(shape) for shape in param_shapes)
        assertTensorTuplesAlmostEqual(
            self,
            params,
            _unflatten_params(_flatten_params(params)),
            delta=1e-4,
            mode="max",
        )

    @parameterized.expand(
        [
            (
                reduction,
                influence_constructor,
                delta,
                mode,
                unpack_inputs,
                gpu_setting,
            )
            for reduction in ["none", "sum", "mean"]
            for gpu_setting in gpu_settings_list
            for (influence_constructor, delta) in [
                (
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        layers=(
                            ["module.linear"]
                            if gpu_setting == "cuda_dataparallel"
                            else ["linear"]
                        ),
                        projection_dim=None,
                        # letting projection_dim is None means no projection is done,
                        # in which case exact influence is returned
                        show_progress=False,
                    ),
                    1e-3,
                ),
                (
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        layers=None,
                        # this tests that not specifyiing layers still works
                        projection_dim=None,
                        show_progress=False,
                        name="NaiveInfluenceFunction_all_layers",
                    ),
                    1e-3,
                ),
            ]
            for mode in [
                "influence",
                "self_influence",
            ]
            for unpack_inputs in [
                False,
                True,
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_matches_linear_regression(
        self,
        reduction: str,
        influence_constructor: Callable,
        delta: float,
        mode: str,
        unpack_inputs: bool,
        gpu_setting: Optional[str],
    ) -> None:
        """
        this tests that `NaiveInfluence`, the simplest implementation, agree with the
        analytically calculated solution for influence and self-influence for a model
        where we can calculate that solution - linear regression trained with squared
        error loss.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                hessian_samples,
                hessian_labels,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                return_test_data=True,
                gpu_setting=gpu_setting,
                return_hessian_data=True,
                model_type="trained_linear",
            )

            train_dataset = DataLoader(train_dataset, batch_size=5)

            use_gpu = is_gpu(gpu_setting)
            hessian_dataset = (
                ExplicitDataset(hessian_samples, hessian_labels, use_gpu)
                if not unpack_inputs
                else UnpackDataset(hessian_samples, hessian_labels, use_gpu)
            )
            hessian_dataset = DataLoader(hessian_dataset, batch_size=5)

            criterion = nn.MSELoss(reduction=reduction)
            batch_size = None

            # set `sample_grads_per_batch` based on `reduction` to be compatible
            sample_wise_grads_per_batch = False if reduction == "none" else True

            influence = influence_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
                sample_wise_grads_per_batch=sample_wise_grads_per_batch,
                hessian_dataset=hessian_dataset,
            )

            # since the model is a linear regression model trained with MSE loss, we
            # can calculate the hessian and per-example parameter gradients
            # analytically
            tensor_hessian_samples = (
                hessian_samples
                if not unpack_inputs
                else torch.cat(hessian_samples, dim=1)
            )
            # hessian at optimal parameters is 2 * X'X, where X is the feature matrix
            # of the examples used for calculating the hessian.
            # this is based on https://math.stackexchange.com/questions/2864585/hessian-on-linear-least-squares-problem # noqa: E501
            # and multiplying by 2, since we optimize squared error,
            # not 1/2 squared error.
            hessian = torch.matmul(tensor_hessian_samples.T, tensor_hessian_samples) * 2
            hessian = hessian + (
                torch.eye(len(hessian)).to(device=hessian.device) * 1e-4
            )

            hessian_inverse = torch.linalg.pinv(hessian, rcond=1e-4)

            # gradient for an example is 2 * features * error

            # compute train gradients
            tensor_train_samples = torch.cat(
                [torch.cat(batch[:-1], dim=1) for batch in train_dataset], dim=0
            )
            train_predictions = torch.cat(
                [net(*batch[:-1]) for batch in train_dataset], dim=0
            )
            train_labels = torch.cat([batch[-1] for batch in train_dataset], dim=0)
            train_gradients = (
                (train_predictions - train_labels) * tensor_train_samples * 2
            )

            # compute test gradients
            tensor_test_samples = (
                test_samples if not unpack_inputs else torch.cat(test_samples, dim=1)
            )
            test_predictions = (
                net(test_samples) if not unpack_inputs else net(*test_samples)
            )
            test_gradients = (test_predictions - test_labels) * tensor_test_samples * 2

            if mode == "influence":
                # compute pairwise influences, analytically
                analytical_train_test_influences = torch.matmul(
                    torch.matmul(test_gradients, hessian_inverse), train_gradients.T
                )
                # compute pairwise influences using influence implementation
                influence_train_test_influences = influence.influence(
                    _format_batch_into_tuple(test_samples, test_labels, unpack_inputs)
                )
                # check error
                assertTensorAlmostEqual(
                    self,
                    influence_train_test_influences,
                    analytical_train_test_influences,
                    delta=delta,
                    mode="max",
                )
            elif mode == "self_influence":
                # compute self influence, analytically
                analytical_self_influences = torch.diag(
                    torch.matmul(
                        torch.matmul(train_gradients, hessian_inverse),
                        train_gradients.T,
                    )
                )
                # compute pairwise influences using influence implementation
                influence_self_influences = influence.self_influence(train_dataset)
                # check error
                assertTensorAlmostEqual(
                    self,
                    influence_self_influences,
                    analytical_self_influences,
                    delta=delta,
                    mode="max",
                )
            else:
                raise Exception("unknown test mode")

    @parameterized.expand(
        [(_custom_functional_call,), (_functional_call,)],
        name_func=build_test_name_func(),
    )
    def test_functional_call(self, method) -> None:
        """
        tests `influence._utils.common._functional_call` for a simple case where the
        model and loss are linear regression and squared error.  `method` can either be
        `_custom_functional_call`, which uses the custom implementation that is used
        if pytorch does not provide one, or `_functional_call`, which uses a pytorch
        implementation if available.
        """
        # get linear model and a batch
        batch_size = 25
        num_features = 5
        batch_samples = torch.normal(0, 1, (batch_size, num_features))
        batch_labels = torch.normal(0, 1, (batch_size, 1))
        net = Linear(num_features)

        # get the analytical gradient wrt to model parameters
        batch_predictions = net(batch_samples)
        analytical_grad = 2 * torch.sum(
            (batch_predictions - batch_labels) * batch_samples, dim=0
        )

        # get gradient as computed using `_functional_call`
        param = net.linear.weight.detach().clone().requires_grad_(True)
        _batch_predictions = method(net, {"linear.weight": param}, (batch_samples,))
        loss = torch.sum((_batch_predictions - batch_labels) ** 2)
        actual_grad = torch.autograd.grad(loss, param)[0][0]

        # they should be the same
        assertTensorAlmostEqual(
            self, actual_grad, analytical_grad, delta=1e-3, mode="max"
        )
