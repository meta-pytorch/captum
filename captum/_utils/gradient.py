#!/usr/bin/env python3

# pyre-strict
import threading
import typing
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
from captum._utils.common import (
    _reduce_list,
    _run_forward,
    _sort_key_list,
    _verify_select_neuron,
)
from captum._utils.sample_gradient import SampleGradientWrapper
from captum._utils.typing import (
    ModuleOrModuleList,
    SliceIntType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from torch import device, Tensor
from torch.nn import Module


def apply_gradient_requirements(
    inputs: Tuple[Tensor, ...], warn: bool = True
) -> List[bool]:
    """
    Iterates through tuple on input tensors and sets requires_grad to be true on
    each Tensor, and ensures all grads are set to zero. To ensure that the input
    is returned to its initial state, a list of flags representing whether or not
     a tensor originally required grad is returned.
    """
    assert isinstance(
        inputs, (tuple, list)
    ), "Inputs should be wrapped in a tuple or list prior to preparing for gradients"
    grad_required = []
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        grad_required.append(input.requires_grad)
        inputs_dtype = input.dtype
        # Note: torch 1.2 doesn't support is_complex for dtype that's why we check
        # on the existance of is_complex method.
        if not inputs_dtype.is_floating_point and not (
            hasattr(inputs_dtype, "is_complex") and inputs_dtype.is_complex
        ):
            if warn:
                warnings.warn(
                    """Input Tensor %d has a dtype of %s.
                    Gradients cannot be activated
                    for these data types."""
                    % (index, str(inputs_dtype)),
                    stacklevel=2,
                )
        elif not input.requires_grad:
            if warn:
                warnings.warn(
                    "Input Tensor %d did not already require gradients, "
                    "required_grads has been set automatically." % index,
                    stacklevel=2,
                )
            input.requires_grad_()
    return grad_required


def undo_gradient_requirements(
    inputs: Tuple[Tensor, ...], grad_required: List[bool]
) -> None:
    """
    Iterates through list of tensors, zeros each gradient, and sets required
    grad to false if the corresponding index in grad_required is False.
    This method is used to undo the effects of prepare_gradient_inputs, making
    grads not required for any input tensor that did not initially require
    gradients.
    """

    assert isinstance(
        inputs, tuple
    ), "Inputs should be wrapped in a tuple prior to preparing for gradients."
    assert len(inputs) == len(
        grad_required
    ), "Input tuple length should match gradient mask."
    for index, input in enumerate(inputs):
        assert isinstance(input, torch.Tensor), "Given input is not a torch.Tensor"
        if not grad_required[index]:
            input.requires_grad_(False)


def compute_gradients(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        # _run_forward may return future of Tensor,
        # but we don't support it here now
        # And it will fail before here.
        outputs = cast(Tensor, outputs)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs)
    return grads


def _neuron_gradients(
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    saved_layer: Dict[device, Tuple[Tensor, ...]],
    key_list: List[device],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
) -> Tuple[Tensor, ...]:
    with torch.autograd.set_grad_enabled(True):
        gradient_tensors = []
        for key in key_list:
            current_out_tensor = _verify_select_neuron(
                saved_layer[key], gradient_neuron_selector
            )
            gradient_tensors.append(
                torch.autograd.grad(
                    (
                        torch.unbind(current_out_tensor)
                        if current_out_tensor.numel() > 1
                        else current_out_tensor
                    ),
                    inputs,
                )
            )
        _total_gradients = _reduce_list(gradient_tensors, sum)
    return _total_gradients


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_eval` does not accept all
#  possible arguments of overload defined on line `170`.
def _forward_layer_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: List[Module],
    additional_forward_args: Optional[object] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    grad_enabled: bool = False,
) -> List[Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_eval` does not accept all
#  possible arguments of overload defined on line `158`.
def _forward_layer_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Optional[object] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    grad_enabled: bool = False,
) -> Tuple[Tensor, ...]: ...


def _forward_layer_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: ModuleOrModuleList,
    additional_forward_args: Optional[object] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    grad_enabled: bool = False,
) -> Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]:
    return _forward_layer_eval_with_neuron_grads(
        forward_fn,
        inputs,
        # pyre-fixme[6]: For 3rd argument expected `Module` but got
        #  `ModuleOrModuleList`.
        layer,
        additional_forward_args=additional_forward_args,
        gradient_neuron_selector=None,
        grad_enabled=grad_enabled,
        device_ids=device_ids,
        attribute_to_layer_input=attribute_to_layer_input,
    )


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_distributed_eval` does not
#  accept all possible arguments of overload defined on line `203`.
def _forward_layer_distributed_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Any,
    layer: ModuleOrModuleList,
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: Literal[False] = False,
    require_layer_grads: bool = False,
) -> Dict[Module, Dict[device, Tuple[Tensor, ...]]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_distributed_eval` does not
#  accept all possible arguments of overload defined on line `216`.
def _forward_layer_distributed_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Any,
    layer: ModuleOrModuleList,
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    attribute_to_layer_input: bool = False,
    *,
    forward_hook_with_return: Literal[True],
    require_layer_grads: bool = False,
) -> Tuple[Dict[Module, Dict[device, Tuple[Tensor, ...]]], Tensor]: ...


def _forward_layer_distributed_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Any,
    layer: ModuleOrModuleList,
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    attribute_to_layer_input: bool = False,
    forward_hook_with_return: bool = False,
    require_layer_grads: bool = False,
) -> Union[
    Tuple[Dict[Module, Dict[device, Tuple[Tensor, ...]]], Tensor],
    Dict[Module, Dict[device, Tuple[Tensor, ...]]],
]:
    r"""
    A helper function that allows to set a hook on model's `layer`, run the forward
    pass and returns intermediate layer results, stored in a dictionary,
    and optionally also the output of the forward function. The keys in the
    dictionary are the device ids and the values are corresponding intermediate layer
    results, either the inputs or the outputs of the layer depending on whether we set
    `attribute_to_layer_input` to True or False.
    This is especially useful when we execute forward pass in a distributed setting,
    using `DataParallel`s for example.
    """
    saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]] = defaultdict(dict)
    lock = threading.Lock()
    # pyre-fixme[9]: all_layers has type `List[Module]`; used as
    #  `Union[List[Variable[ModuleOrModuleList <: [Module, List[Module]]]],
    #  Variable[ModuleOrModuleList <: [Module, List[Module]]]]`.
    all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer

    # Set a forward hook on specified module and run forward pass to
    # get layer output tensor(s).
    # For DataParallel models, each partition adds entry to dictionary
    # with key as device and value as corresponding Tensor.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def hook_wrapper(original_module):
        # pyre-fixme[53]: Captured variable `lock` is not annotated.
        # pyre-fixme[53]: Captured variable `original_module` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def forward_hook(module, inp, out=None):
            eval_tsrs = inp if attribute_to_layer_input else out
            is_eval_tuple_or_list = isinstance(eval_tsrs, (tuple, list))

            if not is_eval_tuple_or_list:
                eval_tsrs = (eval_tsrs,)
            if require_layer_grads:
                apply_gradient_requirements(eval_tsrs, warn=False)
            with lock:
                # Note that cloning behaviour of `eval_tsr` is different
                # when `forward_hook_with_return` is set to True. This is because
                # otherwise `backward()` on the last output layer won't execute.
                if forward_hook_with_return:
                    saved_layer[original_module][eval_tsrs[0].device] = eval_tsrs
                    if not is_eval_tuple_or_list:
                        eval_tsrs_to_return = eval_tsrs[0].clone()
                    elif isinstance(eval_tsrs, list):
                        eval_tsrs_to_return = [
                            eval_tsr.clone() for eval_tsr in eval_tsrs
                        ]
                    else:
                        eval_tsrs_to_return = tuple(
                            eval_tsr.clone() for eval_tsr in eval_tsrs
                        )
                    return eval_tsrs_to_return
                else:
                    saved_layer[original_module][eval_tsrs[0].device] = tuple(
                        eval_tsr.clone() for eval_tsr in eval_tsrs
                    )

        return forward_hook

    all_hooks = []
    try:
        for single_layer in all_layers:
            if attribute_to_layer_input:
                all_hooks.append(
                    single_layer.register_forward_pre_hook(hook_wrapper(single_layer))
                )
            else:
                all_hooks.append(
                    single_layer.register_forward_hook(hook_wrapper(single_layer))
                )
        output = _run_forward(
            forward_fn,
            inputs,
            target=target_ind,
            additional_forward_args=additional_forward_args,
        )
        # _run_forward may return future of Tensor,
        # but we don't support it here now
        # And it will fail before here.
        output = cast(Tensor, output)
    finally:
        for hook in all_hooks:
            hook.remove()

    if len(saved_layer) == 0:
        raise AssertionError("Forward hook did not obtain any outputs for given layer")

    if forward_hook_with_return:
        return saved_layer, output
    return saved_layer


def _gather_distributed_tensors(
    saved_layer: Dict[device, Tuple[Tensor, ...]],
    device_ids: Union[None, List[int]] = None,
    key_list: Union[None, List[device]] = None,
) -> Tuple[Tensor, ...]:
    r"""
    A helper function to concatenate intermediate layer results stored on
    different devices in `saved_layer`. `saved_layer` is a dictionary that
    contains `device_id` as a key and intermediate layer results (either
    the input or the output of the layer) stored on the device corresponding to
    the key.
    `key_list` is a list of devices in appropriate ordering for concatenation
    and if not provided, keys are sorted based on device ids.

    If only one key exists (standard model), key list simply has one element.
    """
    if key_list is None:
        key_list = _sort_key_list(list(saved_layer.keys()), device_ids)
    return _reduce_list([saved_layer[device_id] for device_id in key_list])


def _extract_device_ids(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    saved_layer: Dict[Module, Dict[device, Tuple[Tensor, ...]]],
    device_ids: Union[None, List[int]],
) -> Union[None, List[int]]:
    r"""
    A helper function to extract device_ids from `forward_function` in case it is
    provided as part of a `DataParallel` model or if is accessible from
    `forward_fn`.
    In case input device_ids is not None, this function returns that value.
    """
    # Multiple devices / keys implies a DataParallel model, so we look for
    # device IDs if given or available from forward function
    # (DataParallel model object).
    if (
        max(len(saved_layer[single_layer]) for single_layer in saved_layer) > 1
        and device_ids is None
    ):
        if (
            hasattr(forward_fn, "device_ids")
            and cast(Any, forward_fn).device_ids is not None
        ):
            device_ids = cast(Any, forward_fn).device_ids
        else:
            raise AssertionError(
                "Layer tensors are saved on multiple devices, however unable to access"
                " device ID list from the `forward_fn`. Device ID list must be"
                " accessible from `forward_fn`. For example, they can be retrieved"
                " if `forward_fn` is a model of type `DataParallel`. It is used"
                " for identifying device batch ordering."
            )
    return device_ids


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_eval_with_neuron_grads` does
#  not accept all possible arguments of overload defined on line `378`.
def _forward_layer_eval_with_neuron_grads(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Optional[object] = None,
    *,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
    grad_enabled: bool = False,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_eval_with_neuron_grads` does
#  not accept all possible arguments of overload defined on line `405`.
def _forward_layer_eval_with_neuron_grads(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: List[Module],
    additional_forward_args: Optional[object] = None,
    gradient_neuron_selector: None = None,
    grad_enabled: bool = False,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> List[Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `_forward_layer_eval_with_neuron_grads` does
#  not accept all possible arguments of overload defined on line `392`.
def _forward_layer_eval_with_neuron_grads(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: Module,
    additional_forward_args: Optional[object] = None,
    gradient_neuron_selector: None = None,
    grad_enabled: bool = False,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Tuple[Tensor, ...]: ...


def _forward_layer_eval_with_neuron_grads(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    layer: ModuleOrModuleList,
    additional_forward_args: Optional[object] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    gradient_neuron_selector: Union[
        None, int, Tuple[Union[int, slice], ...], Callable
    ] = None,
    grad_enabled: bool = False,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tensor, ...],
    List[Tuple[Tensor, ...]],
]:
    """
    This method computes forward evaluation for a particular layer using a
    forward hook. If a gradient_neuron_selector is provided, then gradients with
    respect to that neuron in the layer output are also returned.

    These functionalities are combined due to the behavior of DataParallel models
    with hooks, in which hooks are executed once per device. We need to internally
    combine the separated tensors from devices by concatenating based on device_ids.
    Any necessary gradients must be taken with respect to each independent batched
    tensor, so the gradients are computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel models
    can be found in the PyTorch data parallel documentation. We maintain the separate
    evals in a dictionary protected by a lock, analogous to the gather implementation
    for the core PyTorch DataParallel implementation.
    """
    grad_enabled = True if gradient_neuron_selector is not None else grad_enabled

    with torch.autograd.set_grad_enabled(grad_enabled):
        saved_layer = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
    device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)
    # Identifies correct device ordering based on device ids.
    # key_list is a list of devices in appropriate ordering for concatenation.
    # If only one key exists (standard model), key list simply has one element.
    key_list = _sort_key_list(list(next(iter(saved_layer.values())).keys()), device_ids)
    if gradient_neuron_selector is not None:
        assert isinstance(
            layer, Module
        ), "Cannot compute neuron gradients for multiple layers simultaneously!"
        inp_grads = _neuron_gradients(
            inputs, saved_layer[layer], key_list, gradient_neuron_selector
        )
        return (
            _gather_distributed_tensors(saved_layer[layer], key_list=key_list),
            inp_grads,
        )
    else:
        if isinstance(layer, Module):
            return _gather_distributed_tensors(saved_layer[layer], key_list=key_list)
        else:
            return [
                _gather_distributed_tensors(saved_layer[curr_layer], key_list=key_list)
                for curr_layer in layer
            ]


@typing.overload
# pyre-fixme[43]: The implementation of `compute_layer_gradients_and_eval` does not
#  accept all possible arguments of overload defined on line `486`.
def compute_layer_gradients_and_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    layer: Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    *,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    gradient_neuron_selector: Union[int, Tuple[Union[int, slice], ...], Callable],
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    output_fn: Union[None, Callable] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `compute_layer_gradients_and_eval` does not
#  accept all possible arguments of overload defined on line `502`.
def compute_layer_gradients_and_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    layer: List[Module],
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    gradient_neuron_selector: None = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    output_fn: Union[None, Callable] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]]: ...


@typing.overload
# pyre-fixme[43]: The implementation of `compute_layer_gradients_and_eval` does not
#  accept all possible arguments of overload defined on line `517`.
def compute_layer_gradients_and_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    layer: Module,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    gradient_neuron_selector: None = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    output_fn: Union[None, Callable] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]]: ...


def compute_layer_gradients_and_eval(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    forward_fn: Callable,
    layer: ModuleOrModuleList,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Optional[object] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    gradient_neuron_selector: Union[
        None, int, Tuple[Union[int, slice], ...], Callable
    ] = None,
    device_ids: Union[None, List[int]] = None,
    attribute_to_layer_input: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    output_fn: Union[None, Callable] = None,
    grad_kwargs: Optional[Dict[str, Any]] = None,
) -> Union[
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[Tuple[Tensor, ...], Tuple[Tensor, ...], Tuple[Tensor, ...]],
    Tuple[List[Tuple[Tensor, ...]], List[Tuple[Tensor, ...]]],
]:
    r"""
    Computes gradients of the output with respect to a given layer as well
    as the output evaluation of the layer for an arbitrary forward function
    and given input.

    For data parallel models, hooks are executed once per device ,so we
    need to internally combine the separated tensors from devices by
    concatenating based on device_ids. Any necessary gradients must be taken
    with respect to each independent batched tensor, so the gradients are
    computed and combined appropriately.

    More information regarding the behavior of forward hooks with DataParallel
    models can be found in the PyTorch data parallel documentation. We maintain
    the separate inputs in a dictionary protected by a lock, analogous to the
    gather implementation for the core PyTorch DataParallel implementation.

    NOTE: To properly handle inplace operations, a clone of the layer output
    is stored. This structure inhibits execution of a backward hook on the last
    module for the layer output when computing the gradient with respect to
    the input, since we store an intermediate clone, as
    opposed to the true module output. If backward module hooks are necessary
    for the final module when computing input gradients, utilize
    _forward_layer_eval_with_neuron_grads instead.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        layer:      Layer for which gradients / output will be evaluated.
        inputs:     Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        output_fn:  An optional function that is applied to the layer inputs or
                    outputs depending whether the `attribute_to_layer_input` is
                    set to `True` or `False`
        args:       Additional input arguments that forward function requires.
                    It takes an empty tuple (no additional arguments) if no
                    additional arguments are required
        grad_kwargs: Additional keyword arguments for torch.autograd.grad


    Returns:
        tuple[**gradients**, **evals**]:
        - **gradients**:
            Gradients of output with respect to target layer output.
        - **evals**:
            Target layer output for given input.
    """
    with torch.autograd.set_grad_enabled(True):
        # saved_layer is a dictionary mapping device to a tuple of
        # layer evaluations on that device.
        saved_layer, output = _forward_layer_distributed_eval(
            forward_fn,
            inputs,
            layer,
            target_ind=target_ind,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
            forward_hook_with_return=True,
            require_layer_grads=True,
        )
        assert output[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )

        device_ids = _extract_device_ids(forward_fn, saved_layer, device_ids)

        # Identifies correct device ordering based on device ids.
        # key_list is a list of devices in appropriate ordering for concatenation.
        # If only one key exists (standard model), key list simply has one element.
        key_list = _sort_key_list(
            list(next(iter(saved_layer.values())).keys()), device_ids
        )
        all_outputs: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        if isinstance(layer, Module):
            all_outputs = _reduce_list(
                [
                    (
                        saved_layer[layer][device_id]
                        if output_fn is None
                        else output_fn(saved_layer[layer][device_id])
                    )
                    for device_id in key_list
                ]
            )
        else:
            all_outputs = [
                _reduce_list(
                    [
                        (
                            saved_layer[single_layer][device_id]
                            if output_fn is None
                            else output_fn(saved_layer[single_layer][device_id])
                        )
                        for device_id in key_list
                    ]
                )
                for single_layer in layer
            ]
        # pyre-fixme[9]: all_layers has type `List[Module]`; used as
        #  `Union[List[Variable[ModuleOrModuleList <: [Module, List[Module]]]],
        #  Variable[ModuleOrModuleList <: [Module, List[Module]]]]`.
        all_layers: List[Module] = [layer] if isinstance(layer, Module) else layer
        grad_inputs = tuple(
            layer_tensor
            for single_layer in all_layers
            for device_id in key_list
            for layer_tensor in saved_layer[single_layer][device_id]
        )
        saved_grads = torch.autograd.grad(
            outputs=torch.unbind(output),
            inputs=grad_inputs,
            **grad_kwargs or {},
        )

        offset = 0
        all_grads: List[Tuple[Tensor, ...]] = []
        for single_layer in all_layers:
            num_tensors = len(next(iter(saved_layer[single_layer].values())))
            curr_saved_grads = [
                saved_grads[i : i + num_tensors]
                for i in range(
                    offset, offset + len(key_list) * num_tensors, num_tensors
                )
            ]
            offset += len(key_list) * num_tensors
            if output_fn is not None:
                curr_saved_grads = [
                    output_fn(curr_saved_grad) for curr_saved_grad in curr_saved_grads
                ]

            all_grads.append(_reduce_list(curr_saved_grads))

        layer_grads: Union[Tuple[Tensor, ...], List[Tuple[Tensor, ...]]]
        layer_grads = all_grads
        if isinstance(layer, Module):
            layer_grads = all_grads[0]

        if gradient_neuron_selector is not None:
            assert isinstance(
                layer, Module
            ), "Cannot compute neuron gradients for multiple layers simultaneously!"
            inp_grads = _neuron_gradients(
                inputs, saved_layer[layer], key_list, gradient_neuron_selector
            )
            return (
                cast(Tuple[Tensor, ...], layer_grads),
                cast(Tuple[Tensor, ...], all_outputs),
                inp_grads,
            )
    return layer_grads, all_outputs  # type: ignore


def construct_neuron_grad_fn(
    layer: Module,
    neuron_selector: Union[
        int,
        Tuple[Union[int, SliceIntType], ...],
        Callable[[Union[Tensor, Tuple[Tensor, ...]]], Tensor],
    ],
    device_ids: Union[None, List[int]] = None,
    attribute_to_neuron_input: bool = False,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
) -> Callable:
    def grad_fn(
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        forward_fn: Callable,
        inputs: TensorOrTupleOfTensorsGeneric,
        target_ind: TargetType = None,
        additional_forward_args: Optional[object] = None,
    ) -> Tuple[Tensor, ...]:
        _, grads = _forward_layer_eval_with_neuron_grads(
            forward_fn,
            inputs,
            layer,
            additional_forward_args,
            gradient_neuron_selector=neuron_selector,
            device_ids=device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
        )
        return grads

    return grad_fn


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _extract_parameters_from_layers(layer_modules):
    layer_parameters = []
    if layer_modules is not None:
        layer_parameters = [
            parameter
            for layer_module in layer_modules
            for parameter in layer_module.parameters()
        ]
        assert (
            len(layer_parameters) > 0
        ), "No parameters are available for modules for provided input `layers`"
    return layer_parameters


def _compute_jacobian_wrt_params(
    model: Module,
    inputs: Tuple[Any, ...],
    labels: Optional[Tensor] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]] = None,
    layer_modules: Optional[List[Module]] = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method is equivalent to calculating the
    gradient for every individual example in the minibatch.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (tuple[Any, ...]): The minibatch for which the forward pass is computed.
                It is unpacked before passing to `model`, so it must be a tuple.  The
                individual elements of `inputs` can be anything.
        labels (Tensor, optional): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable, optional): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='none'`.
        layer_modules (List[torch.nn.Module], optional): A list of PyTorch modules
                 w.r.t. which jacobian gradients are computed.
    Returns:
        grads (tuple[Tensor, ...]): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    with torch.autograd.set_grad_enabled(True):
        out = model(*inputs)
        assert out.dim() != 0, "Please ensure model output has at least one dimension."

        if labels is not None and loss_fn is not None:
            loss = loss_fn(out, labels)
            if hasattr(loss_fn, "reduction"):
                msg0 = "Please ensure loss_fn.reduction is set to `none`"
                assert loss_fn.reduction == "none", msg0  # type: ignore
            else:
                msg1 = (
                    "Loss function is applying a reduction. Please ensure "
                    f"Output shape: {out.shape} and Loss shape: {loss.shape} "
                    "are matching."
                )
                assert loss.dim() != 0, msg1
                assert out.shape[0] == loss.shape[0], msg1
            out = loss

        if layer_modules is not None:
            layer_parameters = _extract_parameters_from_layers(layer_modules)
        grads_list = [
            torch.autograd.grad(
                outputs=out[i],
                inputs=cast(
                    Union[Tensor, Sequence[Tensor]],
                    # pyre-fixme[61]: `layer_parameters` is undefined, or not always
                    #  defined.
                    model.parameters() if layer_modules is None else layer_parameters,
                ),
                grad_outputs=torch.ones_like(out[i]),
                retain_graph=True,
            )
            for i in range(out.shape[0])
        ]
        grads = tuple([torch.stack(x) for x in zip(*grads_list)])

        return tuple(grads)


def _compute_jacobian_wrt_params_with_sample_wise_trick(
    model: Module,
    inputs: Tuple[Any, ...],
    labels: Optional[Tensor] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = "sum",
    layer_modules: Optional[List[Module]] = None,
) -> Tuple[Any, ...]:
    r"""
    Computes the Jacobian of a batch of test examples given a model, and optional
    loss function and target labels. This method uses sample-wise gradients per
    batch trick to fully vectorize the Jacobian calculation. Currently, only
    linear and conv2d layers are supported.

    User must `add_hooks(model)` before calling this function.

    Args:
        model (torch.nn.Module): The trainable model providing the forward pass
        inputs (tuple[Any, ...]): The minibatch for which the forward pass is computed.
                It is unpacked before passing to `model`, so it must be a tuple.  The
                individual elements of `inputs` can be anything.
        labels (Tensor, optional): Labels for input if computing a loss function.
        loss_fn (torch.nn.Module or Callable, optional): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='sum'` or
                `reduction='mean'`.
        reduction_type (str, optional): The type of reduction applied. If a loss_fn is
                passed, this should match `loss_fn.reduction`. Else if gradients are
                being computed on direct model outputs (scores), then 'sum' should be
                used.
                Defaults to 'sum'.
        layer_modules (torch.nn.Module, optional): A list of PyTorch modules w.r.t.
                 which jacobian gradients are computed.

    Returns:
        grads (tuple[Tensor, ...]): Returns the Jacobian for the minibatch as a
                tuple of gradients corresponding to the tuple of trainable parameters
                returned by `model.parameters()`. Each object grads[i] references to the
                gradients for the parameters in the i-th trainable layer of the model.
                Each grads[i] object is a tensor with the gradients for the `inputs`
                batch. For example, grads[i][j] would reference the gradients for the
                parameters of the i-th layer, for the j-th member of the minibatch.
    """
    with torch.autograd.set_grad_enabled(True):
        inputs = tuple(inp.clone() for inp in inputs)
        apply_gradient_requirements(inputs)
        sample_grad_wrapper = SampleGradientWrapper(model, layer_modules)
        try:
            sample_grad_wrapper.add_hooks()

            out = model(*inputs)
            assert (
                out.dim() != 0
            ), "Please ensure model output has at least one dimension."

            if labels is not None and loss_fn is not None:
                loss = loss_fn(out, labels)
                # TODO: allow loss_fn to be Callable
                if (isinstance(loss_fn, Module) or callable(loss_fn)) and hasattr(
                    loss_fn, "reduction"
                ):
                    reduction = loss_fn.reduction  # type: ignore
                    msg0 = (
                        "Please ensure that loss_fn.reduction is set to `sum` or `mean`"
                    )

                    assert reduction != "none", msg0
                    msg1 = (
                        f"loss_fn.reduction ({reduction}) does not match"
                        f"reduction type ({reduction_type}). Please ensure they are"
                        " matching."
                    )
                    assert reduction == reduction_type, msg1
                msg2 = (
                    "Please ensure custom loss function is applying either a "
                    "sum or mean reduction."
                )
                assert out.shape != loss.shape, msg2

                if reduction_type != "sum" and reduction_type != "mean":
                    raise ValueError(
                        f"{reduction_type} is not a valid value for reduction_type. "
                        "Must be either 'sum' or 'mean'."
                    )
                out = loss

            sample_grad_wrapper.compute_param_sample_gradients(
                out,
                # pyre-fixme[6]: In call `SampleGradientWrapper.
                # compute_param_sample_gradients`, for argument `loss_mode`,
                # expected `str` but got `Optional[str]`.
                loss_mode=reduction_type,  # type: ignore
            )
            if layer_modules is not None:
                layer_parameters = _extract_parameters_from_layers(layer_modules)
            grads = tuple(
                param.sample_grad  # type: ignore
                for param in (
                    model.parameters()
                    if layer_modules is None
                    # pyre-fixme[61]: `layer_parameters` is undefined, or not always
                    #  defined.
                    else layer_parameters
                )
                if hasattr(param, "sample_grad")
            )
        finally:
            sample_grad_wrapper.remove_hooks()

        return grads
