#!/usr/bin/env python3

# pyre-strict
import typing
from enum import Enum
from functools import reduce
from inspect import signature
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    overload,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
    TupleOrTensorOrBoolGeneric,
)

from torch import device, Tensor

from torch.futures import Future
from torch.nn import Module


def parse_version(v: str) -> Tuple[int, ...]:
    """
    Parse version strings into tuples for comparison.

    Versions should be in the form of "<major>.<minor>.<patch>", "<major>.<minor>",
    or "<major>". The "dev", "post" and other letter portions of the given version will
    be ignored.

    Args:

        v (str): A version string.

    Returns:
        version_tuple (tuple[int]): A tuple of integer values to use for version
            comparison.
    """
    version_list = [n for n in v.split(".") if n.isdigit()]
    assert version_list != []
    return tuple(map(int, version_list))


class ExpansionTypes(Enum):
    repeat = 1
    repeat_interleave = 2


def safe_div(
    numerator: Tensor,
    denom: Union[Tensor, int, float],
    default_denom: Union[Tensor, int, float] = 1.0,
) -> Tensor:
    r"""
    A simple utility function to perform `numerator / denom`
    if the statement is undefined => result will be `numerator / default_denorm`
    """
    if isinstance(denom, (int, float)):
        return numerator / (denom if denom != 0 else default_denom)

    # convert default_denom to tensor if it is float
    if not torch.is_tensor(default_denom):
        default_denom = torch.tensor(
            default_denom, dtype=denom.dtype, device=denom.device
        )

    return numerator / torch.where(denom != 0, denom, default_denom)


@typing.overload
def _is_tuple(inputs: Tuple[Tensor, ...]) -> Literal[True]: ...


@typing.overload
def _is_tuple(inputs: Tensor) -> Literal[False]: ...


@typing.overload
def _is_tuple(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> bool: ...


def _is_tuple(inputs: Union[Tensor, Tuple[Tensor, ...]]) -> bool:
    return isinstance(inputs, tuple)


def _validate_target(num_samples: int, target: TargetType) -> None:
    if isinstance(target, list) or (
        isinstance(target, torch.Tensor) and torch.numel(target) > 1
    ):
        assert num_samples == len(target), (
            "The number of samples provied in the"
            "input {} does not match with the number of targets. {}".format(
                num_samples, len(target)
            )
        )


def _validate_input(
    inputs: Tuple[Tensor, ...],
    baselines: Tuple[Union[Tensor, int, float], ...],
    draw_baseline_from_distrib: bool = False,
) -> None:
    assert len(inputs) == len(baselines), (
        "Input and baseline must have the same "
        "dimensions, baseline has {} features whereas input has {}.".format(
            len(baselines), len(inputs)
        )
    )

    for input, baseline in zip(inputs, baselines):
        if draw_baseline_from_distrib:
            assert (
                isinstance(baseline, (int, float))
                or input.shape[1:] == baseline.shape[1:]
            ), (
                "The samples in input and baseline batches must have"
                " the same shape or the baseline corresponding to the"
                " input tensor must be a scalar."
                " Found baseline: {} and input: {} ".format(baseline, input)
            )
        else:
            assert (
                isinstance(baseline, (int, float))
                or input.shape == baseline.shape
                or baseline.shape[0] == 1
            ), (
                "Baseline can be provided as a tensor for just one input and"
                " broadcasted to the batch or input and baseline must have the"
                " same shape or the baseline corresponding to each input tensor"
                " must be a scalar. Found baseline: {} and input: {}".format(
                    baseline, input
                )
            )


def _zeros(inputs: Tuple[Tensor, ...]) -> Tuple[int, ...]:
    r"""
    Takes a tuple of tensors as input and returns a tuple that has the same
    length as `inputs` with each element as the integer 0.
    """
    return tuple(0 if input.dtype is not torch.bool else False for input in inputs)


def _format_baseline(
    baselines: BaselineType, inputs: Tuple[Tensor, ...]
) -> Tuple[Union[Tensor, int, float], ...]:
    if baselines is None:
        return _zeros(inputs)

    if not isinstance(baselines, tuple):
        baselines = (baselines,)

    for baseline in baselines:
        assert isinstance(
            baseline, (torch.Tensor, int, float)
        ), "baseline input argument must be either a torch.Tensor or a number \
            however {} detected".format(
            type(baseline)
        )

    return baselines


def _is_mask_valid(mask: Tensor, inp: Tensor) -> bool:
    """
    Checks whether the mask is valid for the given input.
    """
    if mask.dim() > inp.dim():
        return False

    for mask_d, inp_d in zip(mask.shape[::-1], inp.shape[::-1]):
        if mask_d != 1 and mask_d != inp_d:
            return False

    return True


def _format_feature_mask(
    feature_mask: Union[None, Tensor, Tuple[Tensor, ...]],
    inputs: Tuple[Tensor, ...],
    start_idx: int = 0,
) -> Tuple[Tensor, ...]:
    """
    Format a feature mask into a tuple of tensors.
    The `inputs` should be correctly formatted first
    If `feature_mask` is None, assign each non-batch dimension with a consecutive
    integer from `start_idx`.
    If `feature_mask` is a tensor, wrap it in a tuple.
    """
    if feature_mask is None:
        formatted_mask = []
        current_num_features = start_idx
        for inp in inputs:
            # the following can handle empty tensor where numel is 0
            # empty tensor will be added to the feature mask
            num_features = torch.numel(inp[0:1])

            formatted_mask.append(
                current_num_features
                + torch.reshape(
                    torch.arange(num_features, device=inp.device),
                    inp[0:1].shape,
                )
            )
            current_num_features += num_features
        formatted_mask = tuple(formatted_mask)

    else:
        formatted_mask = _format_tensor_into_tuples(feature_mask)

    return formatted_mask


@overload
def _format_tensor_into_tuples(inputs: None) -> None: ...


@overload
def _format_tensor_into_tuples(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
) -> Tuple[Tensor, ...]: ...


def _format_tensor_into_tuples(
    inputs: Union[None, Tensor, Tuple[Tensor, ...]],
) -> Union[None, Tuple[Tensor, ...]]:
    if inputs is None:
        return None
    if not isinstance(inputs, tuple):
        assert isinstance(inputs, torch.Tensor), (
            "`inputs` must be a torch.Tensor or a tuple[torch.Tensor] "
            f"but found: {type(inputs)}"
        )
        inputs = (inputs,)
    return inputs


def _format_inputs(inputs: Any, unpack_inputs: bool = True) -> Any:
    return (
        inputs
        if (isinstance(inputs, tuple) or isinstance(inputs, list)) and unpack_inputs
        else (inputs,)
    )


def _format_float_or_tensor_into_tuples(
    inputs: Union[float, Tensor, Tuple[Union[float, Tensor], ...]],
) -> Tuple[Union[float, Tensor], ...]:
    if not isinstance(inputs, tuple):
        assert isinstance(
            inputs, (torch.Tensor, float)
        ), "`inputs` must have type float or torch.Tensor but {} found: ".format(
            type(inputs)
        )
        inputs = (inputs,)
    return inputs


@overload
def _format_additional_forward_args(
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    additional_forward_args: Union[Tensor, Tuple],
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
) -> Tuple: ...


@overload
def _format_additional_forward_args(  # type: ignore
    additional_forward_args: Optional[object],
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
) -> Union[None, Tuple]: ...


def _format_additional_forward_args(
    additional_forward_args: Optional[object],
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
) -> Union[None, Tuple]:
    if additional_forward_args is not None and not isinstance(
        additional_forward_args, tuple
    ):
        additional_forward_args = (additional_forward_args,)
    return additional_forward_args


def _expand_additional_forward_args(
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    additional_forward_args: Union[None, Tuple],
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
) -> Union[None, Tuple]:
    def _expand_tensor_forward_arg(
        additional_forward_arg: Tensor,
        n_steps: int,
        expansion_type: ExpansionTypes = ExpansionTypes.repeat,
    ) -> Tensor:
        if len(additional_forward_arg.size()) == 0:
            return additional_forward_arg
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([additional_forward_arg] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return additional_forward_arg.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    if additional_forward_args is None:
        return None

    return tuple(
        (
            _expand_tensor_forward_arg(additional_forward_arg, n_steps, expansion_type)
            if isinstance(additional_forward_arg, torch.Tensor)
            else additional_forward_arg
        )
        for additional_forward_arg in additional_forward_args
    )


def _expand_target(
    target: TargetType,
    n_steps: int,
    expansion_type: ExpansionTypes = ExpansionTypes.repeat,
) -> TargetType:
    if isinstance(target, list):
        if expansion_type == ExpansionTypes.repeat:
            return target * n_steps
        elif expansion_type == ExpansionTypes.repeat_interleave:
            expanded_target = []
            for i in target:
                expanded_target.extend([i] * n_steps)
            return cast(Union[List[Tuple[int, ...]], List[int]], expanded_target)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    elif isinstance(target, torch.Tensor) and torch.numel(target) > 1:
        if expansion_type == ExpansionTypes.repeat:
            return torch.cat([target] * n_steps, dim=0)
        elif expansion_type == ExpansionTypes.repeat_interleave:
            return target.repeat_interleave(n_steps, dim=0)
        else:
            raise NotImplementedError(
                "Currently only `repeat` and `repeat_interleave`"
                " expansion_types are supported"
            )

    return target


def _expand_feature_mask(
    feature_mask: Union[Tensor, Tuple[Tensor, ...]], n_samples: int
) -> Tuple[Tensor, ...]:
    is_feature_mask_tuple = _is_tuple(feature_mask)
    feature_mask = _format_tensor_into_tuples(feature_mask)
    feature_mask_new = tuple(
        (
            feature_mask_elem.repeat_interleave(n_samples, dim=0)
            if feature_mask_elem.size(0) > 1
            else feature_mask_elem
        )
        for feature_mask_elem in feature_mask
    )
    return _format_output(is_feature_mask_tuple, feature_mask_new)  # type: ignore


def _expand_and_update_baselines(
    inputs: Tuple[Tensor, ...],
    n_samples: int,
    # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
    #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    kwargs: dict,
    draw_baseline_from_distrib: bool = False,
) -> None:
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def get_random_baseline_indices(bsz, baseline):
        num_ref_samples = baseline.shape[0]
        return np.random.choice(num_ref_samples, n_samples * bsz).tolist()

    # expand baselines to match the sizes of input
    if "baselines" not in kwargs:
        return

    baselines = kwargs["baselines"]
    baselines = _format_baseline(baselines, inputs)
    _validate_input(
        inputs, baselines, draw_baseline_from_distrib=draw_baseline_from_distrib
    )

    if draw_baseline_from_distrib:
        bsz = inputs[0].shape[0]
        baselines = tuple(
            (
                baseline[get_random_baseline_indices(bsz, baseline)]
                if isinstance(baseline, torch.Tensor)
                else baseline
            )
            for baseline in baselines
        )
    else:
        baselines = tuple(
            (
                baseline.repeat_interleave(n_samples, dim=0)
                if isinstance(baseline, torch.Tensor)
                and baseline.shape[0] == input.shape[0]
                and baseline.shape[0] > 1
                else baseline
            )
            for input, baseline in zip(inputs, baselines)
        )
    # update kwargs with expanded baseline
    kwargs["baselines"] = baselines


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
#  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
def _expand_and_update_additional_forward_args(n_samples: int, kwargs: dict) -> None:
    if "additional_forward_args" not in kwargs:
        return
    additional_forward_args = kwargs["additional_forward_args"]
    additional_forward_args = _format_additional_forward_args(additional_forward_args)
    if additional_forward_args is None:
        return
    additional_forward_args = _expand_additional_forward_args(
        additional_forward_args,
        n_samples,
        expansion_type=ExpansionTypes.repeat_interleave,
    )
    # update kwargs with expanded baseline
    kwargs["additional_forward_args"] = additional_forward_args


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
#  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
def _expand_and_update_target(n_samples: int, kwargs: dict) -> None:
    if "target" not in kwargs:
        return
    target = kwargs["target"]
    target = _expand_target(
        target, n_samples, expansion_type=ExpansionTypes.repeat_interleave
    )
    # update kwargs with expanded baseline
    kwargs["target"] = target


# pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
#  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
def _expand_and_update_feature_mask(n_samples: int, kwargs: dict) -> None:
    if "feature_mask" not in kwargs:
        return

    feature_mask = kwargs["feature_mask"]
    if feature_mask is None:
        return

    feature_mask = _expand_feature_mask(feature_mask, n_samples)
    kwargs["feature_mask"] = feature_mask


@typing.overload
def _format_output(
    is_inputs_tuple: Literal[True],
    output: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]: ...


@typing.overload
def _format_output(
    is_inputs_tuple: Literal[False],
    output: Tuple[Tensor, ...],
) -> Tensor: ...


@typing.overload
def _format_output(
    is_inputs_tuple: bool, output: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]: ...


def _format_output(
    is_inputs_tuple: bool, output: Tuple[Tensor, ...]
) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""
    In case input is a tensor and the output is returned in form of a
    tuple we take the first element of the output's tuple to match the
    same shape signatues of the inputs
    """
    assert isinstance(output, tuple), "Output must be in shape of a tuple"
    assert is_inputs_tuple or len(output) == 1, (
        "The input is a single tensor however the output isn't."
        "The number of output tensors is: {}".format(len(output))
    )
    # pyre-fixme[7]: Expected `Union[Tensor, typing.Tuple[Tensor, ...]]` but got
    #  `Union[tuple[Tensor], Tensor]`.
    return output if is_inputs_tuple else output[0]


@typing.overload
def _format_outputs(
    is_multiple_inputs: Literal[False],
    outputs: List[Tuple[Tensor, ...]],
) -> Union[Tensor, Tuple[Tensor, ...]]: ...


@typing.overload
def _format_outputs(
    is_multiple_inputs: Literal[True],
    outputs: List[Tuple[Tensor, ...]],
) -> List[Union[Tensor, Tuple[Tensor, ...]]]: ...


@typing.overload
def _format_outputs(
    is_multiple_inputs: bool, outputs: List[Tuple[Tensor, ...]]
) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]: ...


def _format_outputs(
    is_multiple_inputs: bool, outputs: List[Tuple[Tensor, ...]]
) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
    assert isinstance(outputs, list), "Outputs must be a list"
    assert is_multiple_inputs or len(outputs) == 1, (
        "outputs should contain multiple inputs or have a single output"
        f"however the number of outputs is: {len(outputs)}"
    )

    return (
        [_format_output(len(output) > 1, output) for output in outputs]
        if is_multiple_inputs
        else _format_output(len(outputs[0]) > 1, outputs[0])
    )


# pyre-fixme[24] Callable requires 2 arguments
def _construct_future_forward(original_forward: Callable) -> Callable:
    # pyre-fixme[3] return type not specified
    def future_forward(*args: Any, **kwargs: Any):
        # pyre-fixme[29]: `typing.Type[torch.futures.Future]` is not a function.
        fut: torch.futures.Future[Tensor] = torch.futures.Future()
        fut.set_result(original_forward(*args, **kwargs))
        return fut

    return future_forward


def _run_forward(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_func: Callable,
    inputs: Any,
    target: TargetType = None,
    additional_forward_args: Optional[object] = None,
) -> Union[Tensor, Future[Tensor]]:
    forward_func_args = signature(forward_func).parameters
    if len(forward_func_args) == 0:
        output = forward_func()
        return output if target is None else _select_targets(output, target)

    # make everything a tuple so that it is easy to unpack without
    # using if-statements
    inputs = _format_inputs(inputs)
    additional_forward_args = _format_additional_forward_args(additional_forward_args)

    output = forward_func(
        *(
            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            #  tuples: `*inputs, *additional_forward_args`.
            (*inputs, *additional_forward_args)
            if additional_forward_args is not None
            else inputs
        )
    )
    if isinstance(output, torch.futures.Future):
        return output.then(lambda x: _select_targets(x.value(), target))
    return _select_targets(output, target)


def _select_targets(output: Tensor, target: TargetType) -> Tensor:
    """
    IMPORTANT:
    please avoid patching this function. The existing behavior is very
    unpredictable. We should be more opinionated about the type and format of
    the target so that we can stop supporting some unpredictable cases.
    Or better, we should encourage users to wrapping their forward function to
    return the attr targets themselves, instead of passing target.

    This legacy function behaves based on
    - the type of target
    - if the target has the length of the output

    If the target is int or scalar tensor, the target is seen as the
    index of the last dimensions of every example in the output. E.g., if the
    output is of shape (Batch, ..., X, Y), the selected output will be (Batch, ..., X)

    If the target is tuple[int], the target is seens as the last indices of every
    example in the output. E.g., if the
    output is of shape (Batch, ..., X, Y, Z) and the target is tuple(y, z),
    the selected output will be (Batch, ..., X)

    If the target is a non-scalar tensor, it must be a 1D tensor of the output length
    and the output must be a 2D tensor. The target is then seen as the indices of the
    2nd dim of the output. E.g., if the output is of shape (Batch, X) and the target is
    in shape (X,), the selected output will be (Batch,)

    If the target is a list[int], it must has the same length as the output. The output
    must be a 2D tensor and each int element of the target is seen as the 2nd dim of it.
    E.g., if the output is of shape (Batch, X) and the target is [x1, x2, ...],
    the selected output will be (Batch,)

    If the target is a list[tuple], it must has the same length as the output. Each
    tuple element of the target is seen as the leading dim behind the batch dim
    of the output. E.g., if the output is of shape (Batch, X, Y, Z, ...) and
    the target is [(x1, y1), (x2, y2), ...], the selected output
    will be in shape (Batch, Z, ...)
    """
    if target is None:
        return output

    num_examples = output.shape[0]
    dims = len(output.shape)
    device = output.device
    if isinstance(target, (int, tuple)):
        return _verify_select_column(output, target)
    elif isinstance(target, torch.Tensor):
        if torch.numel(target) == 1 and isinstance(target.item(), int):
            return _verify_select_column(output, cast(int, target.item()))
        elif len(target.shape) == 1 and torch.numel(target) == num_examples:
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(output, 1, target.reshape(len(output), 1)).squeeze(-1)
        else:
            raise AssertionError(
                "Tensor target dimension %r is not valid. %r"
                % (target.shape, output.shape)
            )
    elif isinstance(target, list):
        assert len(target) == num_examples, "Target list length does not match output!"
        if isinstance(target[0], int):
            assert dims == 2, "Output must be 2D to select tensor of targets."
            return torch.gather(
                output, 1, torch.tensor(target, device=device).reshape(len(output), 1)
            ).squeeze(-1)
        elif isinstance(target[0], tuple):
            return torch.stack(
                [
                    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type
                    #  parameter.
                    output[(i,) + cast(Tuple, targ_elem)]
                    for i, targ_elem in enumerate(target)
                ]
            )
        else:
            raise AssertionError(
                f"Target element type {type(target[0])} in list is not valid."
            )
    else:
        raise AssertionError(f"Target type {type(target)} is not valid.")


# pyre-fixme[24]: Generic type `slice` expects 3 type parameters.
def _contains_slice(target: Union[int, Tuple[Union[int, slice], ...]]) -> bool:
    if isinstance(target, tuple):
        for index in target:
            if isinstance(index, slice):
                return True
        return False
    return isinstance(target, slice)


def _verify_select_column(
    output: Tensor,
    # pyre-fixme[24]: Generic type `slice` expects 3 type parameters.
    target: Union[int, Tuple[Union[int, slice], ...]],
) -> Tensor:
    target = (target,) if isinstance(target, int) else target
    assert (
        len(target) <= len(output.shape) - 1
    ), "Cannot choose target column with output shape %r." % (output.shape,)
    return output[(slice(None), *target)]


def _verify_select_neuron(
    layer_output: Tuple[Tensor, ...],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    selector: Union[int, Tuple[Union[int, slice], ...], Callable],
) -> Tensor:
    if callable(selector):
        # pyre-fixme[7]: Expected `Tensor` but got `object`.
        return selector(layer_output if len(layer_output) > 1 else layer_output[0])

    assert len(layer_output) == 1, (
        "Cannot select neuron index from layer with multiple tensors,"
        "consider providing a neuron selector function instead."
    )

    selected_neurons = _verify_select_column(layer_output[0], selector)
    if _contains_slice(selector):
        return selected_neurons.reshape(selected_neurons.shape[0], -1).sum(1)
    return selected_neurons


def _extract_device(
    module: Module,
    hook_inputs: Union[None, Tensor, Tuple[Tensor, ...]],
    hook_outputs: Union[None, Tensor, Tuple[Tensor, ...]],
) -> device:
    params = list(module.parameters())
    if (
        (hook_inputs is None or len(hook_inputs) == 0)
        and (hook_outputs is None or len(hook_outputs) == 0)
        and len(params) == 0
    ):
        raise RuntimeError(
            """Unable to extract device information for the module
            {}. Both inputs and outputs to the forward hook and
            `module.parameters()` are empty.
            The reason that the inputs to the forward hook are empty
            could be due to the fact that the arguments to that
            module {} are all named and are passed as named
            variables to its forward function.
            """.format(
                module, module
            )
        )
    if hook_inputs is not None and len(hook_inputs) > 0:
        return hook_inputs[0].device
    if hook_outputs is not None and len(hook_outputs) > 0:
        return hook_outputs[0].device

    return params[0].device


def _reduce_list(
    val_list: Sequence[TupleOrTensorOrBoolGeneric],
    # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
    #  `typing.List[<element type>]` to avoid runtime subscripting errors.
    red_func: Callable[[List], Any] = torch.cat,
) -> TupleOrTensorOrBoolGeneric:
    """
    Applies reduction function to given list. If each element in the list is
    a Tensor, applies reduction function to all elements of the list, and returns
    the output Tensor / value. If each element is a boolean, apply any method (or).
    If each element is a tuple/list, applies reduction
    function to corresponding elements of each tuple/list in the list, and returns
    tuple of reduction function outputs with length matching the length of tuple
    val_list[0]. It is assumed that all tuples/lists in the list have the same length
    and red_func can be applied to all elements in each corresponding position.
    """
    assert len(val_list) > 0, "Cannot reduce empty list!"
    if isinstance(val_list[0], torch.Tensor):
        first_device = cast(Tensor, val_list[0]).device
        return red_func(
            [elem.to(first_device) for elem in cast(List[Tensor], val_list)]
        )
    elif isinstance(val_list[0], bool):
        # pyre-fixme[7]: Expected `TupleOrTensorOrBoolGeneric` but got `bool`.
        return any(val_list)
    elif isinstance(val_list[0], (tuple, list)):
        final_out = []
        # pyre-fixme[6]: For 1st argument expected `pyre_extensions.ReadOnly[Sized]`
        #  but got `TupleOrTensorOrBoolGeneric`.
        for i in range(len(val_list[0])):
            final_out.append(
                # pyre-fixme[16]: `bool` has no attribute `__getitem__`.
                _reduce_list([val_elem[i] for val_elem in val_list], red_func)
            )
    else:
        raise AssertionError(
            "Elements to be reduced can only be"
            "either Tensors or tuples/lists containing Tensors."
        )
    # pyre-fixme[7]: Expected `TupleOrTensorOrBoolGeneric` but got `Tuple[Any, ...]`.
    return tuple(final_out)


def _sort_key_list(
    keys: List[device], device_ids: Union[None, List[int]] = None
) -> List[device]:
    """
    Sorts list of torch devices (keys) by given index list, device_ids. If keys
    contains only one device, then the list is returned unchanged. If keys
    contains a device for which the id is not contained in device_ids, then
    an error is returned. This method is used to identify the order of DataParallel
    batched devices, given the device ID ordering.
    """
    if len(keys) == 1:
        return keys
    id_dict: Dict[int, device] = {}
    assert device_ids is not None, "Device IDs must be provided with multiple devices."
    for key in keys:
        if key.index in id_dict:
            raise AssertionError("Duplicate CUDA Device ID identified in device list.")
        id_dict[key.index] = key

    out_list = [
        id_dict[device_id]
        for device_id in filter(lambda device_id: device_id in id_dict, device_ids)
    ]

    assert len(out_list) == len(keys), "Given Device ID List does not match"
    "devices with computed tensors."

    return out_list


def _flatten_tensor_or_tuple(inp: TensorOrTupleOfTensorsGeneric) -> Tensor:
    if isinstance(inp, Tensor):
        return inp.flatten()
    return torch.cat([single_inp.flatten() for single_inp in inp])


def _get_module_from_name(model: Module, layer_name: str) -> Any:
    r"""
    Returns the module (layer) object, given its (string) name
    in the model.

    Args:
            name (str): Module or nested modules name string in self.model

    Returns:
            The module (layer) in self.model.
    """

    return reduce(getattr, layer_name.split("."), model)


def _register_backward_hook(
    module: Module,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    hook: Callable,
    attr_obj: Union[object, None],
) -> List[torch.utils.hooks.RemovableHandle]:
    grad_out: Dict[device, Tensor] = {}

    def forward_hook(
        module: Module,
        inp: Union[Tensor, Tuple[Tensor, ...]],
        out: Union[Tensor, Tuple[Tensor, ...]],
    ) -> None:

        def output_tensor_hook(output_grad: Tensor) -> None:
            grad_out[output_grad.device] = output_grad

        if isinstance(out, tuple):
            assert (
                len(out) == 1
            ), "Backward hooks not supported for module with >1 output"
            out[0].register_hook(output_tensor_hook)
        else:
            out.register_hook(output_tensor_hook)

    def pre_hook(module: Module, inp: Union[Tensor, Tuple[Tensor, ...]]) -> Tensor:
        def input_tensor_hook(
            input_grad: Tensor,
        ) -> Union[None, Tensor, Tuple[Tensor, ...]]:
            if len(grad_out) == 0:
                return None
            hook_out = hook(module, input_grad, grad_out[input_grad.device])

            if hook_out is not None:
                return hook_out[0] if isinstance(hook_out, tuple) else hook_out
            return None

        if isinstance(inp, tuple):
            assert (
                len(inp) == 1
            ), "Backward hooks not supported for module with >1 input"
            inp[0].register_hook(input_tensor_hook)
            return inp[0].clone()
        else:
            inp.register_hook(input_tensor_hook)
            return inp.clone()

    return [
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(forward_hook),
    ]


def _get_max_feature_index(feature_mask: Tuple[Tensor, ...]) -> int:
    """
    Returns the max feature mask index
    The feature mask should be formatted to tuple of tensors at first.

    Note: This util is commonly used to identify the number of features (max_index + 1),
    as we expect user to be resposible to ensure consecutive feature mask indices from 0
    """

    return int(max(torch.max(mask).item() for mask in feature_mask if mask.numel()))


def _get_feature_idx_to_tensor_idx(
    formatted_feature_mask: Tuple[Tensor, ...],
) -> Dict[int, List[int]]:
    """
    For a given tuple of tensors, return dict of tensor values to list of tensor
    indices they appear in.
    """
    feature_idx_to_tensor_idx: Dict[int, List[int]] = {}
    for i, mask in enumerate(formatted_feature_mask):
        for feature_idx in torch.unique(mask):
            if feature_idx.item() not in feature_idx_to_tensor_idx:
                feature_idx_to_tensor_idx[feature_idx.item()] = []
            feature_idx_to_tensor_idx[feature_idx.item()].append(i)
    return feature_idx_to_tensor_idx


def _maybe_expand_parameters(
    perturbations_per_eval: int,
    formatted_inputs: Tuple[Tensor, ...],
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    formatted_additional_forward_args: Union[None, Tuple],
    target: TargetType,
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
) -> Tuple[Tuple[Tensor, ...], Union[None, Tuple], TargetType]:
    """
    When passing in multiple perturbed inputs in one forward pass, we also
    need to expand additional forward args and target.
    """
    additional_args_repeated: object
    if perturbations_per_eval > 1:
        # Repeat features and additional args for batch size.
        all_features_repeated = tuple(
            torch.cat([formatted_inputs[j]] * perturbations_per_eval, dim=0)
            for j in range(len(formatted_inputs))
        )
        additional_args_repeated = (
            _expand_additional_forward_args(
                formatted_additional_forward_args, perturbations_per_eval
            )
            if formatted_additional_forward_args is not None
            else None
        )
        target_repeated = _expand_target(target, perturbations_per_eval)
    else:
        all_features_repeated = formatted_inputs
        additional_args_repeated = formatted_additional_forward_args
        target_repeated = target

    return all_features_repeated, additional_args_repeated, target_repeated
