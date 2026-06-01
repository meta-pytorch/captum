#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import typing
from typing import Any, Callable, cast, List, Optional, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import (
    Attribution,
    DeepLift,
    FeatureAblation,
    IntegratedGradients,
    Saliency,
)
from captum.metrics import infidelity, infidelity_perturb_func_decorator
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel2,
    BasicModel4_MultiArgs,
    BasicModel_ConvNet_One_Conv,
    BasicModel_MultiLayer,
)
from torch import Tensor
from torch.nn import Module


@infidelity_perturb_func_decorator(False)
def _local_perturb_func_default(
    inputs: TensorOrTupleOfTensorsGeneric,
) -> TensorOrTupleOfTensorsGeneric:
    # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got `Tensor`.
    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
    #  `TensorOrTupleOfTensorsGeneric`.
    return _local_perturb_func(inputs)[1]


@typing.overload
# pyre-ignore[43]: The implementation of `_local_perturb_func` does not accept all
#  possible arguments of overload defined on line `43`.
def _local_perturb_func(
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-ignore[43]: The implementation of `_local_perturb_func` does not accept all
#  possible arguments of overload defined on line `51`.
def _local_perturb_func(inputs: Tensor) -> Tuple[Tensor, Tensor]: ...


def _local_perturb_func(
    inputs: TensorOrTupleOfTensorsGeneric,
) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Union[Tensor, Tuple[Tensor, ...]]]:
    input2 = None
    if isinstance(inputs, tuple):
        input1 = inputs[0]
        input2 = inputs[1]
    else:
        input1 = cast(Tensor, inputs)

    perturb1 = 0.0009 * torch.ones_like(input1)
    if input2 is None:
        return perturb1, input1 - perturb1

    perturb2 = 0.0121 * torch.ones_like(input2)
    return (perturb1, perturb2), (input1 - perturb1, input2 - perturb2)


@infidelity_perturb_func_decorator(True)
def _global_perturb_func1_default(
    inputs: TensorOrTupleOfTensorsGeneric,
) -> TensorOrTupleOfTensorsGeneric:
    # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got `Tensor`.
    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
    #  `TensorOrTupleOfTensorsGeneric`.
    return _global_perturb_func1(inputs)[1]


@typing.overload
# pyre-fixme[43]: The implementation of `_global_perturb_func1` does not accept all
#  possible arguments of overload defined on line `74`.
def _global_perturb_func1(
    inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `_global_perturb_func1` does not accept all
#  possible arguments of overload defined on line `70`.
def _global_perturb_func1(inputs: Tensor) -> Tuple[Tensor, Tensor]: ...


# sensitivity-N, N = #input features
def _global_perturb_func1(
    inputs: TensorOrTupleOfTensorsGeneric,
) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Union[Tensor, Tuple[Tensor, ...]]]:
    input2 = None
    if isinstance(inputs, tuple):
        input1 = inputs[0]
        input2 = inputs[1]
    else:
        input1 = cast(Tensor, inputs)
    pert1 = torch.ones(input1.shape)
    if input2 is None:
        return pert1, torch.zeros(input1.shape)

    pert2 = torch.ones(input2.shape)
    return (pert1, pert2), (torch.zeros(input1.shape), torch.zeros(input2.shape))


class Test(BaseTest):
    def test_basic_infidelity_single(self) -> None:
        input1 = torch.tensor([3.0])
        input2 = torch.tensor([1.0])
        inputs = (input1, input2)
        expected = torch.zeros(1)

        self.basic_model_assert(BasicModel2(), inputs, expected)

    def test_basic_infidelity_multiple(self) -> None:
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        expected = torch.zeros(3)

        infid = self.basic_model_assert(BasicModel2(), inputs, expected)
        infid_w_common_func = self.basic_model_assert(
            BasicModel2(),
            inputs,
            expected,
            perturb_func=_local_perturb_func_default,
            multiply_by_inputs=False,
        )
        assertTensorAlmostEqual(self, infid, infid_w_common_func)

    def test_basic_infidelity_multiple_with_batching(self) -> None:
        input1 = torch.tensor([3.0] * 20)
        input2 = torch.tensor([1.0] * 20)
        expected = torch.zeros(20)

        infid1 = self.basic_model_assert(
            BasicModel2(),
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=21,
        )
        infid2 = self.basic_model_assert(
            BasicModel2(),
            (input1, input2),
            expected,
            n_perturb_samples=5,
            max_batch_size=60,
        )
        assertTensorAlmostEqual(self, infid1, infid2, delta=0.01, mode="max")

    def test_basic_infidelity_additional_forward_args1(self) -> None:
        model = BasicModel4_MultiArgs()

        input1 = torch.tensor([[1.5, 2.0, 3.3]])
        input2 = torch.tensor([[3.0, 3.5, 2.2]])

        args = torch.tensor([[1.0, 3.0, 4.0]])
        ig = IntegratedGradients(model)

        infidelity1 = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            torch.zeros(1),
            additional_args=args,
            n_perturb_samples=1,
            max_batch_size=1,
            perturb_func=_global_perturb_func1,
        )

        infidelity2 = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            torch.zeros(1),
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=2,
            perturb_func=_global_perturb_func1,
        )

        infidelity2_w_custom_pert_func = self.basic_model_global_assert(
            ig,
            model,
            (input1, input2),
            torch.zeros(1),
            additional_args=args,
            n_perturb_samples=5,
            max_batch_size=2,
            perturb_func=_global_perturb_func1_default,
        )
        assertTensorAlmostEqual(self, infidelity1, infidelity2, 0.0)
        assertTensorAlmostEqual(self, infidelity2_w_custom_pert_func, infidelity2, 0.0)

    def test_classification_infidelity_convnet_multi_targets(self) -> None:
        model = BasicModel_ConvNet_One_Conv()
        dl = DeepLift(model)

        input = torch.stack([torch.arange(1, 17).float()] * 20, dim=0).view(20, 1, 4, 4)

        self.infidelity_assert(
            model,
            dl.attribute(input, target=torch.tensor([1] * 20))  # type: ignore[has-type]
            / input,
            input,
            torch.zeros(20),
            target=torch.tensor([1] * 20),
            multi_input=False,
            n_perturb_samples=500,
            max_batch_size=120,
        )

    def test_classification_infidelity_tpl_target(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.arange(1.0, 13.0).view(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        sa = Saliency(model)

        infid1 = self.infidelity_assert(
            model,
            sa.attribute(
                input, target=targets, additional_forward_args=additional_forward_args
            ),
            input,
            torch.zeros(4),
            additional_args=additional_forward_args,
            target=targets,
            multi_input=False,
        )

        infid2 = self.infidelity_assert(
            model,
            sa.attribute(
                input, target=targets, additional_forward_args=additional_forward_args
            ),
            input,
            torch.zeros(4),
            additional_args=additional_forward_args,
            target=targets,
            max_batch_size=2,
            multi_input=False,
        )
        assertTensorAlmostEqual(self, infid1, infid2, delta=1e-05, mode="max")

    def test_classification_infidelity_tpl_target_w_baseline(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.arange(1.0, 13.0).view(4, 3)
        baseline = torch.ones(4, 3)
        additional_forward_args = (torch.arange(1, 13).view(4, 3).float(), True)
        targets: List[Tuple[int, ...]] = [(0, 1, 1), (0, 1, 1), (1, 1, 1), (0, 1, 1)]
        ig = IntegratedGradients(model)

        def perturbed_func2(inputs: Tensor, baselines: Tensor) -> Tuple[Tensor, Tensor]:
            return torch.ones(baselines.shape), baselines

        @infidelity_perturb_func_decorator(True)
        def perturbed_func3(inputs: Tensor, baselines: Tensor) -> Tensor:
            return baselines

        attr, delta = ig.attribute(  # type: ignore[has-type]
            input,
            target=targets,
            additional_forward_args=additional_forward_args,
            baselines=baseline,
            return_convergence_delta=True,
        )

        infid = self.infidelity_assert(
            model,
            attr,
            input,
            torch.tensor([0.10686, 0.0, 0.0, 0.0]),
            additional_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            multi_input=False,
            n_perturb_samples=3,
            perturb_func=perturbed_func3,
        )

        infid2 = self.infidelity_assert(
            model,
            attr,
            input,
            torch.tensor([0.10686, 0.0, 0.0, 0.0]),
            additional_args=additional_forward_args,
            baselines=baseline,
            target=targets,
            multi_input=False,
            n_perturb_samples=3,
            perturb_func=perturbed_func2,
        )

        assertTensorAlmostEqual(self, infid, delta * delta)
        assertTensorAlmostEqual(self, infid, infid2)

    def test_basic_infidelity_multiple_with_normalize(self) -> None:
        input1 = torch.tensor([3.0] * 3)
        input2 = torch.tensor([1.0] * 3)
        inputs = (input1, input2)
        expected = torch.zeros(3)

        model = BasicModel2()
        ig = IntegratedGradients(model)
        attrs = ig.attribute(inputs)  # type: ignore[has-type]
        scaled_attrs = tuple(attr * 100 for attr in attrs)

        infid = self.infidelity_assert(model, attrs, inputs, expected, normalize=True)
        scaled_infid = self.infidelity_assert(
            model,
            scaled_attrs,
            inputs,
            expected,
            normalize=True,
        )

        # scaling attr should not change normalized infidelity
        assertTensorAlmostEqual(self, infid, scaled_infid)

    def test_sensitivity_n_ig(self) -> None:
        model = BasicModel_MultiLayer()
        ig = IntegratedGradients(model)
        self.basic_multilayer_sensitivity_n(ig, model)

    def test_sensitivity_n_fa(self) -> None:
        model = BasicModel_MultiLayer()
        fa = FeatureAblation(model)
        self.basic_multilayer_sensitivity_n(fa, model)

    def basic_multilayer_sensitivity_n(
        self, attr_algo: Attribution, model: Module
    ) -> None:
        # sensitivity-2
        def _global_perturb_func2(input: Tensor) -> Tuple[Tensor, Tensor]:
            pert = torch.tensor([[0, 1, 1], [1, 1, 0], [1, 0, 1]]).float()
            return pert, (1 - pert) * input

        # sensitivity-1
        def _global_perturb_func3(input: Tensor) -> Tuple[Tensor, Tensor]:
            pert = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float()
            return pert, (1 - pert) * input

        @infidelity_perturb_func_decorator(True)
        def _global_perturb_func3_custom(input: Tensor) -> Tensor:
            return _global_perturb_func3(input)[1]

        input = torch.tensor([[1.0, 2.5, 3.3]])

        # infidelity for sensitivity-1
        infid = self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            torch.zeros(1),
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func3,
        )

        infid_w_default = self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            torch.zeros(1),
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func3_custom,
        )
        assertTensorAlmostEqual(self, infid, infid_w_default)

        # infidelity for sensitivity-2
        self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            torch.zeros(1),
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func2,
        )

        # infidelity for sensitivity-3
        self.basic_model_global_assert(
            attr_algo,
            model,
            input,
            torch.zeros(1),
            additional_args=None,
            target=0,
            n_perturb_samples=3,
            max_batch_size=None,
            perturb_func=_global_perturb_func1,
        )

    def basic_model_assert(
        self,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: Tensor,
        n_perturb_samples: Optional[int] = 10,
        max_batch_size: Optional[int] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        perturb_func: Optional[Callable] = _local_perturb_func,
        multiply_by_inputs: Optional[bool] = False,
        normalize: Optional[bool] = False,
    ) -> Tensor:
        ig = IntegratedGradients(model)
        if multiply_by_inputs:
            attrs = cast(
                TensorOrTupleOfTensorsGeneric,
                tuple(
                    attr / input
                    for input, attr in zip(
                        inputs, ig.attribute(inputs)  # type: ignore[has-type]
                    )
                ),
            )
        else:
            attrs = ig.attribute(inputs)  # type: ignore[has-type]

        return self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
            perturb_func=perturb_func,
            normalize=normalize,
        )

    def basic_model_global_assert(
        self,
        attr_algo: Attribution,
        model: Module,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: Tensor,
        additional_args: Optional[Any] = None,
        target: Optional[TargetType] = None,
        n_perturb_samples: Optional[int] = 10,
        max_batch_size: Optional[int] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        perturb_func: Optional[Callable] = _global_perturb_func1,
        normalize: Optional[bool] = False,
    ) -> Tensor:
        attrs = attr_algo.attribute(
            inputs, additional_forward_args=additional_args, target=target
        )
        infid = self.infidelity_assert(
            model,
            attrs,
            inputs,
            expected,
            additional_args=additional_args,
            perturb_func=perturb_func,
            target=target,
            n_perturb_samples=n_perturb_samples,
            max_batch_size=max_batch_size,
            normalize=normalize,
        )
        return infid

    def infidelity_assert(
        self,
        model: Module,
        attributions: TensorOrTupleOfTensorsGeneric,
        inputs: TensorOrTupleOfTensorsGeneric,
        expected: Tensor,
        additional_args: Optional[Any] = None,
        baselines: Optional[BaselineType] = None,
        n_perturb_samples: Optional[int] = 10,
        target: Optional[TargetType] = None,
        max_batch_size: Optional[int] = None,
        multi_input: Optional[bool] = True,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        perturb_func: Optional[Callable] = _local_perturb_func,
        normalize: Optional[bool] = False,
        **kwargs: Any,
    ) -> Tensor:
        infid = infidelity(
            model,
            perturb_func,
            inputs,
            attributions,
            additional_forward_args=additional_args,
            target=target,
            baselines=baselines,
            n_perturb_samples=n_perturb_samples,
            max_examples_per_batch=max_batch_size,
            normalize=normalize,
        )
        assertTensorAlmostEqual(self, infid, expected, 0.05)
        return infid
