# pyre-unsafe
import tempfile
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from captum.influence._core.arnoldi_influence_function import (
    _parameter_arnoldi,
    _parameter_distill,
    ArnoldiInfluenceFunction,
)
from captum.influence._core.influence_function import NaiveInfluenceFunction
from captum.influence._utils.common import (
    _eig_helper,
    _flatten_params,
    _top_eigen,
    _unflatten_params_factory,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    ExplicitDataset,
    generate_assymetric_matrix_given_eigenvalues,
    generate_symmetric_matrix_given_eigenvalues,
    get_random_model_and_data,
    is_gpu,
    UnpackDataset,
)
from parameterized import parameterized
from torch import Tensor
from torch.utils.data import DataLoader


class TestArnoldiInfluence(BaseTest):
    @parameterized.expand(
        [
            (dim, rank)
            for (dim, rank) in [
                (5, 2),
                (10, 5),
                (20, 15),
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_top_eigen(self, dim: int, rank: int) -> None:
        # generate symmetric matrix of specific rank and check can recover it using
        # the eigenvalues / eigenvectors returned by `_top_eigen`
        R = torch.randn(dim, rank)
        H = torch.matmul(R, R.T)
        ls, vs = _top_eigen(H, rank, 1e-5, 1e-5)
        assertTensorAlmostEqual(self, vs @ torch.diag(ls) @ vs.T, H, 1e-2, "max")

    @parameterized.expand(
        [
            (symmetric, eigenvalues, k, arnoldi_dim, params_shapes)
            for symmetric in [True, False]
            for (eigenvalues, k, arnoldi_dim, params_shapes, test_name) in [
                (
                    10 ** torch.linspace(-2, 2, 100),
                    10,
                    50,
                    [(4, 10), (15, 3), (3, 5)],
                    "standard",
                ),
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["eigenvalues", "params_shapes"]),
    )
    def test_parameter_arnoldi(
        self,
        symmetric: bool,
        eigenvalues: Tensor,
        k: int,
        arnoldi_dim: int,
        params_shapes: List[Tuple],
    ) -> None:
        """
        This performs the tests of https://github.com/google-research/jax-influence/blob/74bd321156b5445bb35b9594568e4eaaec1a76a3/jax_influence/arnoldi_test.py#L96 # noqa: E501
        See `_test_parameter_arnoldi_and_distill` documentation for 'arnoldi'
        mode for details.
        """
        self._test_parameter_arnoldi_and_distill(
            "arnoldi", symmetric, eigenvalues, k, arnoldi_dim, params_shapes
        )

    @parameterized.expand(
        [
            (symmetric, eigenvalues, k, arnoldi_dim, params_shapes)
            for symmetric in [True, False]
            for (eigenvalues, k, arnoldi_dim, params_shapes, test_name) in [
                (
                    10 ** torch.linspace(-2, 2, 100),
                    10,
                    50,
                    [(4, 10), (15, 3), (3, 5)],
                    "standard",
                ),
            ]
        ],
        name_func=build_test_name_func(args_to_skip=["eigenvalues", "params_shapes"]),
    )
    def test_parameter_distill(
        self,
        symmetric: bool,
        eigenvalues: Tensor,
        k: int,
        arnoldi_dim: int,
        params_shapes: List[Tuple],
    ) -> None:
        """
        This performs the tests of https://github.com/google-research/jax-influence/blob/74bd321156b5445bb35b9594568e4eaaec1a76a3/jax_influence/arnoldi_test.py#L116 # noqa: E501
        See `_test_parameter_arnoldi_and_distill` documentation for
        'distill' mode for details.
        """
        self._test_parameter_arnoldi_and_distill(
            "distill", symmetric, eigenvalues, k, arnoldi_dim, params_shapes
        )

    def _test_parameter_arnoldi_and_distill(
        self,
        mode: str,
        symmetric: bool,
        eigenvalues: Tensor,
        k: int,
        arnoldi_dim: int,
        param_shape: List[Tuple],
    ) -> None:
        """
        This is a helper with 2 modes. For both modes, it first generates a matrix
        with `A` with specified eigenvalues.

        When mode is 'arnoldi', it checks that `_parameter_arnoldi` is correct.
        In particular, it checks that the top-`k` eigenvalues of the restriction
        of `A` to a Krylov subspace (the `H` returned by `_parameter_arnoldi`)
        agree with those of the original matrix. This is a property we expect of the
        Arnoldi iteration that `_parameter_arnoldi` implements.

        When mode is 'distill', it checks that `_parameter_distill` is correct. In
        particular, it checks that the eigenvectors corresponding to the top
        eigenvalues it returns agree with the top eigenvectors of `A`. This is the
        property we require of `distill`, because we use the top eigenvectors (and
        eigenvalues) of (implicitly-defined) `A` to calculate a low-rank approximation
        of its inverse.
        """
        # generate matrix `A` with specified eigenvalues
        A = (
            generate_symmetric_matrix_given_eigenvalues(eigenvalues)
            if symmetric
            else generate_assymetric_matrix_given_eigenvalues(eigenvalues)
        )

        # create the matrix-vector multiplication function that `_parameter_arnoldi`
        # expects that represents multiplication by `A`.
        # since the vector actually needs to be a tuple of tensors, we
        # specify the dimensions of that tuple of tensors. the function then
        # flattens the vector, multiplies it by the generated matrix, and then
        # unflattens the result
        _unflatten_params = _unflatten_params_factory(param_shape)

        def _param_matmul(params: Tuple[Tensor]):
            return _unflatten_params(torch.matmul(A, _flatten_params(params)))

        # generate `b` and call `_parameter_arnoldi`
        b = tuple(torch.randn(shape) for shape in param_shape)
        qs, H = _parameter_arnoldi(
            _param_matmul,
            b,
            arnoldi_dim,
            1e-3,
            torch.device("cpu"),
            False,
        )

        assertTensorAlmostEqual(
            self,
            _flatten_params(_unflatten_params(_flatten_params(b))),
            _flatten_params(b),
            1e-5,
            "max",
        )

        # compute the eigenvalues / eigenvectors of `A` and `H`. we use `eig` since
        # each matrix may not be symmetric. since `eig` does not sort by eigenvalues,
        # need to manually do it. also get rid of last column of H, since
        # it is not part of the decomposition
        vs_A, ls_A = _eig_helper(A)
        vs_H, ls_H = _eig_helper(H[:-1])

        if mode == "arnoldi":
            # compare the top-`k` eigenvalue of the two matrices
            assertTensorAlmostEqual(self, vs_H[-k:], vs_A[-k:], 1e-3, "max")
        elif mode == "distill":
            # use `distill` to compute top-`k` eigenvectors of `H` in the original
            # basis. then check if they are actually eigenvectors
            vs_H_standard, ls_H_standard = _parameter_distill(qs, H, k, 0, 0)

            for l_H_standard, v_A in zip(ls_H_standard[-k:], vs_A[-k:]):
                l_H_standard_flattened = _flatten_params(l_H_standard)  # .real
                expected = v_A * l_H_standard_flattened
                actual = torch.matmul(A, l_H_standard_flattened)
                # tol copied from original code
                assert torch.norm(expected - actual) < 1e-2

            # check that the top-`k` eigenvalues of `A` as computed by
            # `_parameters_distill` are similar to those computed on `A` directly
            for v_H_standard, v_A in zip(vs_H_standard[-k:], vs_A[-k:]):
                # tol copied from original code
                assert abs(v_H_standard - v_A) < 5

            if False:
                # code from original paper does not do this test, so skip for now
                # use `distill`` to get top-`k` eigenvectors of `H` in the original
                # basis, and compare with the top-`k` eigenvectors of `A`. need to
                # flatten those from `distill` to compare
                _, ls_H_standard = _parameter_distill(qs, H, k, 0, 0)
                for l_H_standard, l_A in zip(ls_H_standard, ls_A):
                    # print(l_A)
                    # print(flatten_unflattener.flatten(l_H_standard).real)
                    l_H_standard_flattened /= torch.norm(l_H_standard_flattened)
                    assertTensorAlmostEqual(
                        self,
                        _flatten_params(l_H_standard).real,
                        l_A.real,
                        1e-2,
                        "max",
                    )

    # TODO: for some unknown reason, this test and the test below does not work
    # on `cuda_data_parallel` setting. We need to investigate why.
    # Use a local version of setting list for these two tests for now
    # since we have changed the default setting list to includes all options.
    # (This is also used in many other tests, which also needs to be unified later).
    gpu_setting_list = (
        ["", "cuda"]
        if torch.cuda.is_available() and torch.cuda.device_count() != 0
        else [""]
    )

    @parameterized.expand(
        [
            (
                influence_constructor_1,
                influence_constructor_2,
                delta,
                mode,
                unpack_inputs,
                gpu_setting,
            )
            for gpu_setting in gpu_setting_list
            for (influence_constructor_1, influence_constructor_2, delta) in [
                # compare implementations, when considering only 1 layer
                (
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_dataparallel"
                            else ["linear1"]
                        ),
                        projection_dim=5,
                        show_progress=False,
                        name="NaiveInfluenceFunction_linear1",
                    ),
                    DataInfluenceConstructor(
                        ArnoldiInfluenceFunction,
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_dataparallel"
                            else ["linear1"]
                        ),
                        arnoldi_dim=50,
                        arnoldi_tol=1e-5,  # set low enough so that arnoldi subspace
                        # is large enough
                        projection_dim=5,
                        show_progress=False,
                        name="ArnoldiInfluenceFunction_linear1",
                    ),
                    1e-2,
                ),
                # compare implementations, when considering all layers
                (
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        layers=None,
                        projection_dim=5,
                        show_progress=False,
                        name="NaiveInfluenceFunction_all_layers",
                    ),
                    DataInfluenceConstructor(
                        ArnoldiInfluenceFunction,
                        layers=None,
                        arnoldi_dim=50,
                        arnoldi_tol=1e-5,  # set low enough so that arnoldi subspace
                        # is large enough
                        projection_dim=5,
                        show_progress=False,
                        name="ArnoldiInfluenceFunction_all_layers",
                    ),
                    1e-2,
                ),
            ]
            for mode in [
                # we skip the 'intermediate_quantities' mode, as
                # `NaiveInfluenceFunction` and `ArnoldiInfluenceFunction` return
                # intermediate quantities lying in different coordinate systems,
                # which cannot be expected to be the same.
                "self_influence",
                "influence",
            ]
            for unpack_inputs in [
                False,
                True,
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_compare_implementations_trained_NN_model_and_data(
        self,
        influence_constructor_1: Callable,
        influence_constructor_2: Callable,
        delta: float,
        mode: str,
        unpack_inputs: bool,
        gpu_setting: Optional[str],
    ) -> None:
        """
        this compares 2 influence implementations on a trained 2-layer NN model.
        the implementations we compare are `NaiveInfluenceFunction` and
        `ArnoldiInfluenceFunction`. because the model is trained, calculations
        are more numerically stable, so that we can project to a higher dimension (5).
        """
        self._test_compare_implementations(
            "trained_NN",
            influence_constructor_1,
            influence_constructor_2,
            delta,
            mode,
            unpack_inputs,
            gpu_setting,
        )

    # this compares `ArnoldiInfluenceFunction` and `NaiveInfluenceFunction` on randomly
    # generated data. because these implementations are numerically equivalent, we
    # can also compare the intermediate quantities. we do not compare with
    # `NaiveInfluence` because on randomly generated data, it is not comparable,
    # conceptually, with the other implementations, due to numerical issues.

    @parameterized.expand(
        [
            (
                influence_constructor_1,
                influence_constructor_2,
                delta,
                mode,
                unpack_inputs,
                gpu_setting,
            )
            for gpu_setting in gpu_setting_list
            for (influence_constructor_1, influence_constructor_2, delta) in [
                (
                    DataInfluenceConstructor(
                        NaiveInfluenceFunction,
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_dataparallel"
                            else ["linear1"]
                        ),
                        show_progress=False,
                        projection_dim=1,
                    ),
                    DataInfluenceConstructor(
                        ArnoldiInfluenceFunction,
                        layers=(
                            ["module.linear1"]
                            if gpu_setting == "cuda_dataparallel"
                            else ["linear1"]
                        ),
                        show_progress=False,
                        arnoldi_dim=50,
                        arnoldi_tol=1e-6,
                        projection_dim=1,
                    ),
                    1e-2,
                ),
            ]
            for mode in [
                # we skip the 'intermediate_quantities' mode, as
                # `NaiveInfluenceFunction` and `ArnoldiInfluenceFunction` return
                # intermediate quantities lying in different coordinate systems,
                # which cannot be expected to be the same.
                "self_influence",
                "influence",
            ]
            for unpack_inputs in [
                False,
                True,
            ]
        ],
        name_func=build_test_name_func(),
    )
    def test_compare_implementations_random_model_and_data(
        self,
        influence_constructor_1: Callable,
        influence_constructor_2: Callable,
        delta: float,
        mode: str,
        unpack_inputs: bool,
        gpu_setting: Optional[str],
    ) -> None:
        """
        this compares 2 influence implementations on a trained 2-layer NN model.
        the implementations we compare are `NaiveInfluenceFunction` and
        `ArnoldiInfluenceFunction`. because the model is not trained, calculations
        are not numerically stable, and so we can only project to a low dimension (2).
        """
        self._test_compare_implementations(
            "random",
            influence_constructor_1,
            influence_constructor_2,
            delta,
            mode,
            unpack_inputs,
            gpu_setting,
        )

    def _test_compare_implementations(
        self,
        model_type: str,
        influence_constructor_1: Callable,
        influence_constructor_2: Callable,
        delta: float,
        mode: str,
        unpack_inputs: bool,
        gpu_setting: Optional[str],
    ) -> None:
        """
        checks that 2 implementations of `InfluenceFunctionBase` return the same
        output, where the output is either self influence scores, or influence scores,
        as determined by the `mode` input. this is a helper used by other tests. the
        implementations are compared using the same data, but the model and saved
        checkpoints can be different, and is specified using the `model_type` argument.
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
                model_type=model_type,
            )

            train_dataset = DataLoader(train_dataset, batch_size=5)

            use_gpu = is_gpu(gpu_setting)
            hessian_dataset = (
                ExplicitDataset(hessian_samples, hessian_labels, use_gpu)
                if not unpack_inputs
                else UnpackDataset(hessian_samples, hessian_labels, use_gpu)
            )
            hessian_dataset = DataLoader(hessian_dataset, batch_size=5)

            criterion = nn.MSELoss(reduction="none")
            batch_size = None

            influence_1 = influence_constructor_1(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
                hessian_dataset=hessian_dataset,
            )

            influence_2 = influence_constructor_2(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
                hessian_dataset=hessian_dataset,
            )

            if mode == "self_influence":
                # compare self influence scores
                assertTensorAlmostEqual(
                    self,
                    influence_1.self_influence(train_dataset),
                    influence_2.self_influence(train_dataset),
                    delta=delta,
                    mode="sum",
                )
            elif mode == "intermediate_quantities":
                # compare intermediate quantities
                assertTensorAlmostEqual(
                    self,
                    influence_1.compute_intermediate_quantities(train_dataset),
                    influence_2.compute_intermediate_quantities(train_dataset),
                    delta=delta,
                    mode="max",
                )
            elif mode == "influence":
                # compare influence scores
                assertTensorAlmostEqual(
                    self,
                    influence_1.influence(
                        _format_batch_into_tuple(
                            test_samples, test_labels, unpack_inputs
                        )
                    ),
                    influence_2.influence(
                        _format_batch_into_tuple(
                            test_samples, test_labels, unpack_inputs
                        )
                    ),
                    delta=delta,
                    mode="max",
                )
            else:
                raise Exception("unknown test mode")
