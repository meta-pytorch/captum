# pyre-strict

import tempfile
from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from captum.influence._core.tracincp import TracInCP
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.influence.common import (
    _format_batch_into_tuple,
    build_test_name_func,
    DataInfluenceConstructor,
    get_random_model_and_data,
    GPU_SETTING_LIST,
    is_gpu,
)
from parameterized import parameterized


class TestTracInGetKMostInfluential(BaseTest):
    param_list: List[
        Tuple[str, DataInfluenceConstructor, bool, bool, int, int, str, bool]
    ] = []
    for batch_size, k in [(4, 7), (7, 4), (40, 5), (5, 40), (40, 45)]:
        for unpack_inputs in [True, False]:
            for proponents in [True, False]:
                for gpu_setting in GPU_SETTING_LIST:
                    for reduction, constr, aggregate in [
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP, name="TracInCP_all_layers"
                            ),
                            False,
                        ),
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP, name="TracInCP_all_layers"
                            ),
                            True,
                        ),
                        (
                            "none",
                            DataInfluenceConstructor(
                                TracInCP,
                                name="linear2",
                                layers=(
                                    ["module.linear2"]
                                    if gpu_setting == "cuda_data_parallel"
                                    else ["linear2"]
                                ),
                            ),
                            False,
                        ),
                    ]:
                        if not (
                            "sample_wise_grads_per_batch" in constr.kwargs
                            and constr.kwargs["sample_wise_grads_per_batch"]
                            and is_gpu(gpu_setting)
                        ):
                            param_list.append(
                                (
                                    reduction,
                                    constr,
                                    unpack_inputs,
                                    proponents,
                                    batch_size,
                                    k,
                                    gpu_setting,
                                    aggregate,
                                )
                            )

    # pyre-fixme[56]: Pyre was not able to infer the type of argument
    # `captum.testing.helpers.influence.common.build_test_name_func()`
    # to decorator factory `parameterized.parameterized.expand`.
    @parameterized.expand(
        param_list,
        name_func=build_test_name_func(),
    )
    def test_tracin_k_most_influential(
        self,
        reduction: str,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        tracin_constructor: Callable,
        unpack_inputs: bool,
        proponents: bool,
        batch_size: int,
        k: int,
        gpu_setting: Optional[str],
        aggregate: bool,
    ) -> None:
        """
        This test constructs a random BasicLinearNet, and checks that the proponents
        obtained by calling `influence` and sorting are equal to the proponents
        obtained by calling `_k_most_influential`.  Those calls are made through
        the calls to wrapper method `influence`.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            (
                net,
                train_dataset,
                test_samples,
                test_labels,
            ) = get_random_model_and_data(
                tmpdir,
                unpack_inputs,
                True,
                gpu_setting,
            )

            self.assertTrue(isinstance(reduction, str))
            self.assertTrue(callable(tracin_constructor))

            criterion = nn.MSELoss(reduction=reduction)

            tracin = tracin_constructor(
                net,
                train_dataset,
                tmpdir,
                batch_size,
                criterion,
            )

            # pyre-fixme[16]: `object` has no attribute `influence`.
            train_scores = tracin.influence(
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=None,
                aggregate=aggregate,
            )
            sort_idx = torch.argsort(train_scores, dim=1, descending=proponents)[:, 0:k]
            idx, _train_scores = tracin.influence(
                _format_batch_into_tuple(test_samples, test_labels, unpack_inputs),
                k=k,
                proponents=proponents,
                aggregate=aggregate,
            )
            for i in range(len(idx)):
                # check that idx[i] is correct
                assertTensorAlmostEqual(
                    self,
                    train_scores[i, idx[i]],
                    train_scores[i, sort_idx[i]],
                    delta=0.0,
                    mode="max",
                )
                # check that _train_scores[i] is correct
                assertTensorAlmostEqual(
                    self,
                    _train_scores[i],
                    train_scores[i, sort_idx[i]],
                    delta=0.001,
                    mode="max",
                )
