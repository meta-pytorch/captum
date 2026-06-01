#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import math
from enum import Enum
from typing import Any, Callable, cast, Dict, Generator, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _format_additional_forward_args,
    _reduce_list,
)
from captum._utils.typing import TargetType
from captum.log import log_usage
from captum.robust._core.perturbation import Perturbation
from torch import Tensor


def drange(
    min_val: Union[int, float], max_val: Union[int, float], step_val: Union[int, float]
) -> Generator[Union[int, float], None, None]:
    curr = min_val
    # pyre-fixme[58]: `>` is not supported for operand types `Union[float, int]` and
    #  `int`.
    while curr < max_val:
        yield curr
        # pyre-fixme[58]: `+` is not supported for operand types `Union[float, int]`
        #  and `Union[float, int]`.
        curr += step_val


def default_correct_fn(model_out: Tensor, target: TargetType) -> bool:
    assert (
        isinstance(model_out, Tensor) and model_out.ndim == 2
    ), "Model output must be a 2D tensor to use default correct function;"
    " otherwise custom correct function must be provided"
    target_tensor = torch.tensor(target) if not isinstance(target, Tensor) else target
    return all(torch.argmax(model_out, dim=1) == target_tensor)


class MinParamPerturbationMode(Enum):
    LINEAR = 0
    BINARY = 1


class MinParamPerturbation:
    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        forward_func: Callable,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        attack: Union[Callable, Perturbation],
        arg_name: str,
        arg_min: Union[int, float],
        arg_max: Union[int, float],
        arg_step: Union[int, float],
        mode: str = "linear",
        num_attempts: int = 1,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        preproc_fn: Optional[Callable] = None,
        apply_before_preproc: bool = False,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        correct_fn: Optional[Callable] = None,
    ) -> None:
        r"""
        Identifies minimal perturbation based on target variable which causes
        misclassification (or other incorrect prediction) of target input.

        More specifically, given a perturbation parametrized by a single value
        (e.g. rotation by angle or mask percentage of top features based on
        attribution results), MinParamPerturbation helps identify the minimum value
        which leads to misclassification (or other model output change) with the
        corresponding perturbed input.

        Args:
            forward_func (Callable or torch.nn.Module): This can either be an instance
                of pytorch model or any modification of a model's forward
                function.

            attack (Perturbation or Callable): This can either be an instance
                of a Captum Perturbation / Attack
                or any other perturbation or attack function such
                as a torchvision transform.
                Perturb function must take additional argument (var_name) used for
                minimal perturbation search.

            arg_name (str): Name of argument / variable paramterizing attack, must be
                kwarg of attack. Examples are num_dropout or stdevs

            arg_min (int, float): Minimum value of target variable

            arg_max (int, float): Maximum value of target variable
                (not included in range)

            arg_step (int, float): Minimum interval for increase of target variable.

            mode (str, optional): Mode for search of minimum attack value;
                either ``linear`` for linear search on variable, or ``binary`` for
                binary search of variable
                Default: ``linear``

            num_attempts (int, optional): Number of attempts or trials with
                given variable. This should only be set to > 1 for non-deterministic
                perturbation / attack functions
                Default: ``1``

            preproc_fn (Callable, optional): Optional method applied to inputs. Output
                of preproc_fn is then provided as input to model, in addition to
                additional_forward_args provided to evaluate.
                Default: ``None``

            apply_before_preproc (bool, optional): Defines whether attack should be
                applied before or after preproc function.
                Default: ``False``

            correct_fn (Callable, optional): This determines whether the perturbed input
                leads to a correct or incorrect prediction. By default, this function
                is set to the standard classification test for correctness
                (comparing argmax of output with target), which requires model output to
                be a 2D tensor, returning True if all batch examples are correct and
                false otherwise. Setting this method allows
                any custom behavior defining whether the perturbation is successful
                at fooling the model. For non-classification use cases, a custom
                function must be provided which determines correctness.

                The first argument to this function must be the model out;
                any additional arguments should be provided through
                ``correct_fn_kwargs``.

                This function should have the following signature::

                    def correct_fn(model_out: Tensor, **kwargs: Any) -> bool

                Method should return a boolean if correct (True) and incorrect (False).
                Default: ``None`` (applies standard correct_fn for classification)
        """
        self.forward_func = forward_func
        self.attack = attack
        self.arg_name = arg_name
        self.arg_min = arg_min
        self.arg_max = arg_max
        self.arg_step = arg_step
        assert self.arg_max > (
            self.arg_min
            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[float,
            #  int]`.
            + self.arg_step
        ), "Step size cannot be smaller than range between min and max"

        self.num_attempts = num_attempts
        self.preproc_fn = preproc_fn
        self.apply_before_preproc = apply_before_preproc
        # pyre-fixme[4]: Attribute must be annotated.
        self.correct_fn = cast(
            # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
            Callable,
            correct_fn if correct_fn is not None else default_correct_fn,
        )

        assert (
            mode.upper() in MinParamPerturbationMode.__members__
        ), f"Provided perturb mode {mode} is not valid - must be linear or binary"
        self.mode = MinParamPerturbationMode[mode.upper()]

    def _evaluate_batch(
        self,
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List[<element type>]` to avoid runtime subscripting errors.
        input_list: List,
        additional_forward_args: Optional[Tuple[object, ...]],
        correct_fn_kwargs: Optional[Dict[str, Any]],
        target: TargetType,
    ) -> Optional[int]:
        if additional_forward_args is None:
            additional_forward_args = ()

        all_kwargs = {}
        if target is not None:
            all_kwargs["target"] = target
        if correct_fn_kwargs is not None:
            all_kwargs.update(correct_fn_kwargs)

        if len(input_list) == 1:
            model_out = self.forward_func(input_list[0], *additional_forward_args)
            out_metric = self.correct_fn(model_out, **all_kwargs)
            return 0 if not out_metric else None
        else:
            batched_inps = _reduce_list(input_list)
            model_out = self.forward_func(batched_inps, *additional_forward_args)
            current_count = 0
            for i in range(len(input_list)):
                batch_size = (
                    input_list[i].shape[0]
                    if isinstance(input_list[i], Tensor)
                    else input_list[i][0].shape[0]
                )
                out_metric = self.correct_fn(
                    model_out[current_count : current_count + batch_size], **all_kwargs
                )
                if not out_metric:
                    return i
                current_count += batch_size
            return None

    def _apply_attack(
        self,
        inputs: Any,
        preproc_input: Any,
        attack_kwargs: Optional[Dict[str, Any]],
        param: Union[int, float],
    ) -> Tuple[Any, Any]:
        if attack_kwargs is None:
            attack_kwargs = {}
        if self.apply_before_preproc:
            attacked_inp = self.attack(
                inputs, **attack_kwargs, **{self.arg_name: param}
            )
            preproc_attacked_inp = (
                self.preproc_fn(attacked_inp) if self.preproc_fn else attacked_inp
            )
        else:
            attacked_inp = self.attack(
                preproc_input, **attack_kwargs, **{self.arg_name: param}
            )
            preproc_attacked_inp = attacked_inp
        return preproc_attacked_inp, attacked_inp

    def _linear_search(
        self,
        inputs: Any,
        preproc_input: Any,
        attack_kwargs: Optional[Dict[str, Any]],
        additional_forward_args: Optional[Tuple[object, ...]],
        expanded_additional_args: Any,
        correct_fn_kwargs: Optional[Dict[str, Any]],
        target: TargetType,
        perturbations_per_eval: int,
    ) -> Tuple[Any, Optional[Union[int, float]]]:
        input_list = []
        attack_inp_list = []
        param_list = []

        for param in drange(self.arg_min, self.arg_max, self.arg_step):
            for _ in range(self.num_attempts):
                preproc_attacked_inp, attacked_inp = self._apply_attack(
                    inputs, preproc_input, attack_kwargs, param
                )

                input_list.append(preproc_attacked_inp)
                param_list.append(param)
                attack_inp_list.append(attacked_inp)

                if len(input_list) == perturbations_per_eval:
                    successful_ind = self._evaluate_batch(
                        input_list,
                        expanded_additional_args,
                        correct_fn_kwargs,
                        target,
                    )
                    if successful_ind is not None:
                        return (
                            attack_inp_list[successful_ind],
                            param_list[successful_ind],
                        )
                    input_list = []
                    param_list = []
                    attack_inp_list = []
        if len(input_list) > 0:
            final_add_args = _expand_additional_forward_args(
                additional_forward_args, len(input_list)
            )
            successful_ind = self._evaluate_batch(
                input_list,
                final_add_args,
                correct_fn_kwargs,
                target,
            )
            if successful_ind is not None:
                return (
                    attack_inp_list[successful_ind],
                    param_list[successful_ind],
                )
        return None, None

    def _binary_search(
        self,
        inputs: Any,
        preproc_input: Any,
        attack_kwargs: Optional[Dict[str, Any]],
        additional_forward_args: Optional[Tuple[object, ...]],
        expanded_additional_args: Any,
        correct_fn_kwargs: Optional[Dict[str, Any]],
        target: TargetType,
        perturbations_per_eval: int,
    ) -> Tuple[Any, Optional[Union[int, float]]]:
        min_range = self.arg_min
        max_range = self.arg_max
        min_so_far = None
        min_input = None
        # pyre-fixme[58]: `<` is not supported for operand types `Union[float, int]`
        #  and `int`.
        while max_range > min_range:
            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[float,
            #  int]`.
            # pyre-fixme[58]: `//` is not supported for operand types `int` and
            #  `Union[float, int]`.
            mid_step = ((max_range - min_range) // self.arg_step) // 2

            # pyre-fixme[6]: For 1st argument expected `int` but got `Union[float,
            #  int]`.
            # pyre-fixme[58]: `<` is not supported for operand types `int` and
            #  `Union[float, int]`.
            if mid_step == 0 and min_range + self.arg_step < max_range:
                mid_step = 1
            # pyre-fixme[58]: `*` is not supported for operand types `int` and
            #  `Union[float, int]`.
            mid = min_range + (mid_step * self.arg_step)

            input_list = []
            param_list = []
            attack_inp_list = []
            attack_success = False

            for i in range(self.num_attempts):
                preproc_attacked_inp, attacked_inp = self._apply_attack(
                    inputs, preproc_input, attack_kwargs, mid
                )

                input_list.append(preproc_attacked_inp)
                param_list.append(mid)
                attack_inp_list.append(attacked_inp)

                if len(input_list) == perturbations_per_eval or i == (
                    self.num_attempts - 1
                ):
                    additional_args = expanded_additional_args
                    if len(input_list) != perturbations_per_eval:
                        additional_args = _expand_additional_forward_args(
                            additional_forward_args, len(input_list)
                        )

                    successful_ind = self._evaluate_batch(
                        input_list,
                        additional_args,
                        correct_fn_kwargs,
                        target,
                    )
                    if successful_ind is not None:
                        attack_success = True
                        max_range = mid
                        if min_so_far is None or min_so_far > mid:
                            min_so_far = mid
                            min_input = attack_inp_list[successful_ind]
                        break

                    input_list = []
                    param_list = []
                    attack_inp_list = []

            if math.isclose(min_range, mid):
                break

            if not attack_success:
                min_range = mid

        return min_input, min_so_far

    @log_usage(part_of_slo=False)
    def evaluate(
        self,
        inputs: Any,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        additional_forward_args: Optional[Tuple] = None,
        target: TargetType = None,
        perturbations_per_eval: int = 1,
        attack_kwargs: Optional[Dict[str, Any]] = None,
        correct_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Optional[Union[int, float]]]:
        r"""
        This method evaluates the model at each perturbed input and identifies
        the minimum perturbation that leads to an incorrect model prediction.

        It is recommended to provide a single input (batch size = 1) when using
        this to identify a minimal perturbation for the chosen example. If a
        batch of examples is provided, the default correct function identifies
        the minimal perturbation for at least 1 example in the batch to be
        misclassified. A custom correct_fn can be provided to customize
        this behavior and define correctness for the batch.

        Args:

            inputs (Any): Input for which minimal perturbation
                    is computed. It can be provided as a tensor, tuple of tensors,
                    or any raw input type (e.g. PIL image or text string).
                    This input is provided directly as input to preproc function
                    as well as any attack applied before preprocessing. If no
                    pre-processing function is provided,
                    this input is provided directly to the main model and all attacks.

            additional_forward_args (Any, optional): If the forward function
                    requires additional arguments other than the preprocessing
                    outputs (or inputs if preproc_fn is None), this argument
                    can be provided. It must be either a single additional
                    argument of a Tensor or arbitrary (non-tuple) type or a
                    tuple containing multiple additional arguments including
                    tensors or any arbitrary python types. These arguments
                    are provided to forward_func in order following the
                    arguments in inputs.
                    For a tensor, the first dimension of the tensor must
                    correspond to the number of examples. For all other types,
                    the given argument is used for all forward evaluations.
                    Default: ``None``
            target (TargetType): Target class for classification. This is required if
                using the default ``correct_fn``.

            perturbations_per_eval (int, optional): Allows perturbations of multiple
                    attacks to be grouped and evaluated in one call of forward_fn
                    Each forward pass will contain a maximum of
                    perturbations_per_eval * #examples samples.
                    For DataParallel models, each batch is split among the
                    available devices, so evaluations on each available
                    device contain at most
                    (perturbations_per_eval * #examples) / num_devices
                    samples.
                    In order to apply this functionality, the output of preproc_fn
                    (or inputs itself if no preproc_fn is provided) must be a tensor
                    or tuple of tensors.
                    Default: ``1``
            attack_kwargs (dict, optional): Optional dictionary of keyword
                    arguments provided to attack function
            correct_fn_kwargs (dict, optional): Optional dictionary of keyword
                    arguments provided to correct function

        Returns:

            Tuple of (perturbed_inputs, param_val) if successful
            else Tuple of (None, None)

            - **perturbed inputs** (Any):
                   Perturbed input (output of attack) which results in incorrect
                   prediction.
            - param_val (int, float)
                    Param value leading to perturbed inputs causing misclassification

        Examples::

        >>> def gaussian_noise(inp: Tensor, std: float) -> Tensor:
        >>>     return inp + std*torch.randn_like(inp)

        >>> min_pert = MinParamPerturbation(forward_func=resnet18,
                                           attack=gaussian_noise,
                                           arg_name="std",
                                           arg_min=0.0,
                                           arg_max=2.0,
                                           arg_step=0.01,
                                        )
        >>> for images, labels in dataloader:
        >>>     noised_image, min_std = min_pert.evaluate(inputs=images, target=labels)

        """
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        expanded_additional_args = (
            _expand_additional_forward_args(
                additional_forward_args, perturbations_per_eval
            )
            if perturbations_per_eval > 1
            else additional_forward_args
        )
        preproc_input = inputs if not self.preproc_fn else self.preproc_fn(inputs)

        if self.mode is MinParamPerturbationMode.LINEAR:
            search_fn = self._linear_search
        elif self.mode is MinParamPerturbationMode.BINARY:
            search_fn = self._binary_search
        else:
            raise NotImplementedError(
                "Chosen MinParamPerturbationMode is not supported!"
            )

        return search_fn(
            inputs,
            preproc_input,
            attack_kwargs,
            additional_forward_args,
            expanded_additional_args,
            correct_fn_kwargs,
            target,
            perturbations_per_eval,
        )
