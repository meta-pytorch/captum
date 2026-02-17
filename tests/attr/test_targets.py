#!/usr/bin/env python3

# pyre-strict


from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union

import torch
from captum._utils.common import _format_additional_forward_args
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.lime import Lime
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._utils.attribution import Attribution, InternalAttribution
from captum.testing.attr.helpers.gen_test_utils import (
    gen_test_name,
    get_target_layer,
    parse_test_config,
    should_create_generated_test,
)
from captum.testing.attr.helpers.test_config import config
from captum.testing.helpers.basic import (
    assertTensorTuplesAlmostEqual,
    BaseTest,
    deep_copy_args,
)
from captum.testing.helpers.basic_models import BasicModel_MultiLayer
from torch import Tensor
from torch.nn import Module

"""
Tests in this file are dynamically generated based on the config
defined in tests/attr/helpers/test_config.py. To add new test cases,
read the documentation in test_config.py and add cases based on the
schema described there.
"""


class TargetsMeta(type):
    """
    Target tests created in TargetsMeta apply to any test case with targets being a
    list or tensor.
    Attribution of each example is computed independently with the appropriate target
    and compared to the corresponding result of attributing to a batch with a tensor
    / list of targets.
    """

    def __new__(
        metacls, name: str, bases: Tuple[Type[Any], ...], attrs: Dict[str, Any]
    ) -> "TargetsMeta":
        for test_config in config:
            (
                algorithms,
                model,
                args,
                layer,
                noise_tunnel,
                baseline_distr,
            ) = parse_test_config(test_config)
            target_delta = (
                test_config["target_delta"] if "target_delta" in test_config else 0.0001
            )

            if "target" not in args or not isinstance(args["target"], (list, Tensor)):
                continue

            for algorithm in algorithms:
                # FeaturePermutation requires a batch of inputs
                # so skipping tests
                if issubclass(
                    algorithm, FeaturePermutation
                ) or not should_create_generated_test(algorithm):
                    continue
                test_method = metacls.make_single_target_test(
                    algorithm,
                    model,
                    layer,
                    args,
                    target_delta,
                    noise_tunnel,
                    baseline_distr,
                )
                test_name = gen_test_name(
                    "test_target",
                    cast(str, test_config["name"]),
                    algorithm,
                    noise_tunnel,
                )

                if test_name in attrs:
                    raise AssertionError(
                        "Trying to overwrite existing test with name: %r" % test_name
                    )
                attrs[test_name] = test_method
        return super(TargetsMeta, metacls).__new__(metacls, name, bases, attrs)

    # Arguments are deep copied to ensure tests are independent and are not affected
    # by any modifications within a previous test.
    @classmethod
    @deep_copy_args
    def make_single_target_test(
        metacls,
        algorithm: Type[Attribution],
        model: Module,
        layer: Optional[str],
        args: Dict[str, Any],
        target_delta: float,
        noise_tunnel: bool,
        baseline_distr: bool,
    ) -> Callable[..., None]:
        """
        This method creates a single target test for the given algorithm and parameters.
        """

        target_layer: Optional[Module] = (
            get_target_layer(model, layer) if layer is not None else None
        )
        # Obtains initial arguments to replace with each example
        # individually.
        original_inputs: Union[Tensor, Tuple[Tensor, ...]] = args["inputs"]
        original_targets: Union[List[Any], Tensor] = args["target"]
        original_additional_forward_args: Optional[Tuple[Any, ...]] = (
            _format_additional_forward_args(args["additional_forward_args"])
            if "additional_forward_args" in args
            else None
        )
        num_examples: int = (
            len(original_inputs)
            if isinstance(original_inputs, Tensor)
            else len(original_inputs[0])
        )
        replace_baselines: bool = "baselines" in args and not baseline_distr
        original_baselines: Any = None
        if replace_baselines:
            original_baselines = args["baselines"]

        def target_test_assert(self: "TestTargets") -> None:
            attr_method: Attribution
            if target_layer:
                internal_algorithm = cast(Type[InternalAttribution[Module]], algorithm)
                attr_method = internal_algorithm(model, target_layer)
            else:
                attr_method = algorithm(model)

            if noise_tunnel:
                attr_method = NoiseTunnel(attr_method)
            attributions_orig = attr_method.attribute(**args)
            self.setUp()
            for i in range(num_examples):
                args["target"] = (
                    original_targets[i]
                    if len(original_targets) == num_examples
                    else original_targets
                )
                args["inputs"] = (
                    original_inputs[i : i + 1]
                    if isinstance(original_inputs, Tensor)
                    else tuple(
                        original_inp[i : i + 1] for original_inp in original_inputs
                    )
                )
                if original_additional_forward_args is not None:
                    args["additional_forward_args"] = tuple(
                        (
                            single_add_arg[i : i + 1]
                            if isinstance(single_add_arg, Tensor)
                            else single_add_arg
                        )
                        for single_add_arg in original_additional_forward_args
                    )
                if replace_baselines:
                    if isinstance(original_inputs, Tensor):
                        args["baselines"] = original_baselines[i : i + 1]
                    elif isinstance(original_baselines, tuple):
                        args["baselines"] = tuple(
                            (
                                single_baseline[i : i + 1]
                                if isinstance(single_baseline, Tensor)
                                else single_baseline
                            )
                            for single_baseline in original_baselines
                        )
                # Since Lime methods compute attributions for a batch
                # sequentially, random seed should not be reset after
                # each example after the first.
                if not issubclass(algorithm, Lime):
                    self.setUp()
                single_attr = attr_method.attribute(**args)
                current_orig_attributions = (
                    attributions_orig[i : i + 1]
                    if isinstance(attributions_orig, Tensor)
                    else tuple(
                        single_attrib[i : i + 1] for single_attrib in attributions_orig
                    )
                )
                assertTensorTuplesAlmostEqual(
                    self,
                    current_orig_attributions,
                    single_attr,
                    delta=target_delta,
                    mode="max",
                )
                if (
                    not issubclass(algorithm, Lime)
                    and len(original_targets) == num_examples
                ):
                    # If original_targets contained multiple elements, then
                    # we also compare with setting targets to a list with
                    # a single element.
                    args["target"] = original_targets[i : i + 1]
                    self.setUp()
                    single_attr_target_list = attr_method.attribute(**args)
                    assertTensorTuplesAlmostEqual(
                        self,
                        current_orig_attributions,
                        single_attr_target_list,
                        delta=target_delta,
                        mode="max",
                    )

        return target_test_assert


class TestTargets(BaseTest, metaclass=TargetsMeta):
    def test_simple_target_missing_error(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        attr = IntegratedGradients(net)
        with self.assertRaises(AssertionError):
            attr.attribute(inp)  # type: ignore[has-type]

    def test_multi_target_error(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.zeros((1, 3))
        attr = IntegratedGradients(net)
        with self.assertRaises(AssertionError):
            attr.attribute(  # type: ignore[has-type]
                inp, additional_forward_args=(None, True), target=(1, 0)
            )
