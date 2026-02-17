#!/usr/bin/env python3

# pyre-strict

import functools
import inspect
from typing import Any, Dict, List, Tuple, Type

import torch
from captum._utils.gradient import _forward_layer_eval
from captum.attr import (
    DeepLift,
    DeepLiftShap,
    FeatureAblation,
    GradientShap,
    InputXGradient,
    IntegratedGradients,
    LayerDeepLift,
    LayerDeepLiftShap,
    LayerFeatureAblation,
    LayerGradientShap,
    LayerGradientXActivation,
    LayerIntegratedGradients,
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.testing.helpers.basic import assertTensorTuplesAlmostEqual, BaseTest
from captum.testing.helpers.basic_models import (
    BasicModel,
    BasicModel_MultiLayer_TrueMultiInput,
    MixedKwargsAndArgsModule,
)
from torch import Tensor
from torch.nn import Module

layer_methods_to_test_with_equiv = [
    # layer_method, equiv_method, whether or not to use multiple layers
    (LayerIntegratedGradients, IntegratedGradients, [True, False]),
    (LayerGradientXActivation, InputXGradient, [True, False]),
    (LayerFeatureAblation, FeatureAblation, [False]),
    (LayerDeepLift, DeepLift, [False]),
    (LayerDeepLiftShap, DeepLiftShap, [False]),
    (LayerGradientShap, GradientShap, [False]),
    # TODO: add other algorithms here
]


class InputLayerMeta(type):
    def __new__(
        metacls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]
    ) -> "InputLayerMeta":
        for (
            layer_method,
            equiv_method,
            multi_layers,
        ) in layer_methods_to_test_with_equiv:
            for multi_layer in multi_layers:
                test_name = (
                    f"test_{layer_method.__name__}"
                    + f"_{equiv_method.__name__}_{multi_layer}"
                )
                attrs[test_name] = (
                    lambda self, layer_method=layer_method, equiv_method=equiv_method, multi_layer=multi_layer: self.layer_method_with_input_layer_patches(  # noqa: E501
                        layer_method, equiv_method, multi_layer
                    )
                )

        return super(InputLayerMeta, metacls).__new__(metacls, name, bases, attrs)


class TestInputLayerWrapper(BaseTest, metaclass=InputLayerMeta):
    def test_forward_layer_eval_on_mixed_args_kwargs_module(self) -> None:
        x = torch.randn(10, 5)
        y = torch.randn(10, 5)

        model = MixedKwargsAndArgsModule()

        self.forward_eval_layer_with_inputs_helper(model, {"x": x})
        self.forward_eval_layer_with_inputs_helper(model, {"x": x, "y": y})

    def layer_method_with_input_layer_patches(
        self,
        layer_method_class: Type[Any],
        equiv_method_class: Type[Any],
        multi_layer: bool,
    ) -> None:
        model = BasicModel_MultiLayer_TrueMultiInput() if multi_layer else BasicModel()

        input_names = ["x1", "x2", "x3", "x4"] if multi_layer else ["input"]
        model = ModelInputWrapper(model)

        layers = [model.input_maps[inp] for inp in input_names]
        layer_method = layer_method_class(
            model, layer=layers if multi_layer else layers[0]
        )
        equivalent_method = equiv_method_class(model)

        inputs = tuple(torch.rand(5, 3) for _ in input_names)
        baseline = tuple(torch.zeros(5, 3) for _ in input_names)

        args = inspect.getfullargspec(equivalent_method.attribute.__wrapped__).args

        args_to_use = [inputs]
        if "baselines" in args:
            args_to_use += [baseline]

        a1 = layer_method.attribute(*args_to_use, target=0)
        a2 = layer_method.attribute(
            *args_to_use, target=0, attribute_to_layer_input=True
        )

        real_attributions = equivalent_method.attribute(*args_to_use, target=0)

        if isinstance(a1, list):
            a1 = tuple(a1)
        elif not isinstance(a1, tuple):
            a1 = (a1,)

        if isinstance(a2, list):
            a2 = tuple(a2)
        elif not isinstance(a2, tuple):
            a2 = (a2,)

        if not isinstance(real_attributions, tuple):
            real_attributions = (real_attributions,)

        assertTensorTuplesAlmostEqual(self, a1, a2)
        assertTensorTuplesAlmostEqual(self, a1, real_attributions)

    def forward_eval_layer_with_inputs_helper(
        self, model: Module, inputs_to_test: Dict[str, Tensor]
    ) -> None:
        # hard coding for simplicity
        # 0 if using args, 1 if using kwargs
        #   => no 0s after first 1 (left to right)
        #
        # used to test utilization of args/kwargs
        use_args_or_kwargs = [
            [[0], [1]],
            [
                [0, 0],
                [0, 1],
                [1, 1],
            ],
        ]

        model = ModelInputWrapper(model)

        def forward_func(
            *args: Tensor, args_or_kwargs: List[int] | None = None
        ) -> Tensor:
            assert args_or_kwargs is not None
            # convert to args or kwargs to test *args and **kwargs wrapping behavior
            new_args = []
            new_kwargs = {}
            for args_or_kwarg, name, inp in zip(
                args_or_kwargs, inputs_to_test.keys(), args
            ):
                if args_or_kwarg:
                    new_kwargs[name] = inp
                else:
                    new_args.append(inp)
            return model(*new_args, **new_kwargs)

        for args_or_kwargs in use_args_or_kwargs[len(inputs_to_test) - 1]:
            with self.subTest(args_or_kwargs=args_or_kwargs):
                inputs = _forward_layer_eval(
                    functools.partial(forward_func, args_or_kwargs=args_or_kwargs),
                    inputs=tuple(inputs_to_test.values()),
                    layer=[model.input_maps[name] for name in inputs_to_test.keys()],
                )

                inputs_with_attrib_to_inp = _forward_layer_eval(
                    functools.partial(forward_func, args_or_kwargs=args_or_kwargs),
                    inputs=tuple(inputs_to_test.values()),
                    layer=[model.input_maps[name] for name in inputs_to_test.keys()],
                    attribute_to_layer_input=True,
                )

                for i1, i2, i3 in zip(
                    inputs, inputs_with_attrib_to_inp, inputs_to_test.values()
                ):
                    self.assertTrue((i1[0] == i2[0]).all())
                    self.assertTrue((i1[0] == i3).all())
