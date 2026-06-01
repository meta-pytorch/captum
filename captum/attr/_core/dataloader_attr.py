#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from collections import defaultdict
from copy import copy
from typing import Callable, cast, Dict, Iterable, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _format_baseline,
    _format_feature_mask,
    _format_output,
    _format_tensor_into_tuples,
    _get_max_feature_index,
    _run_forward,
)
from captum._utils.typing import BaselineType
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._utils.attribution import Attribution
from torch import Tensor


class InputRole:
    need_attr = 0
    need_forward = 1
    no_forward = 2


SUPPORTED_METHODS = {FeatureAblation}


# default reducer wehn reduce is None. Simply concat the outputs by the batch dimension
def _concat_tensors(accum: Optional[Tensor], cur_output: Tensor, _) -> Tensor:
    return cur_output if accum is None else torch.cat([accum, cur_output])


def _create_perturbation_mask(
    perturbed_feature_indices: Tensor,  # 1D tensor of one-hot feature indices
    feature_mask: Tuple[Tensor, ...],
    feature_idx_to_mask_idx: Dict[int, List[int]],
) -> Tuple[Union[Tensor, None], ...]:
    """
    Create binary mask for inputs based on perturbed one-hot feature indices
    Use None if no perturbation is needed for the corresponding input
    """

    # a set of input/mask indices that need perturbation
    perturbation_mask_indices = set()
    for i, v in enumerate(perturbed_feature_indices.tolist()):
        # value 0 means the feature has been perturbed
        if not v:
            perturbation_mask_indices |= set(feature_idx_to_mask_idx[i])

    # create binary mask for inputs & set it to None if no perturbation is needed
    perturbation_mask = tuple(
        perturbed_feature_indices[mask_elem] if i in perturbation_mask_indices else None
        for i, mask_elem in enumerate(feature_mask)
    )

    return perturbation_mask


def _perturb_inputs(
    inputs: Iterable[object],
    input_roles: Tuple[int],
    baselines: Tuple[Union[int, float, Tensor], ...],
    perturbation_mask: Tuple[Union[Tensor, None], ...],
) -> Tuple[object, ...]:
    """
    Perturb inputs based on perturbation mask and baselines
    """

    perturbed_inputs = []
    attr_inp_count = 0

    for inp, role in zip(inputs, input_roles):
        if role != InputRole.need_attr:
            perturbed_inputs.append(inp)
            continue

        pert_mask = perturbation_mask[attr_inp_count]

        # no perturbation is needed for this input
        if pert_mask is None:
            perturbed_inputs.append(inp)
        else:
            baseline = baselines[attr_inp_count]

            perturbed_inp = cast(Tensor, inp) * pert_mask + baseline * (1 - pert_mask)
            perturbed_inputs.append(perturbed_inp)

        attr_inp_count += 1

    perturbed_inputs = tuple(perturbed_inputs)

    return perturbed_inputs


def _convert_output_shape(
    unique_attr: Tensor,
    attr_inputs: Tuple[Tensor, ...],
    feature_mask: Tuple[Tensor, ...],
) -> Tuple[Tensor, ...]:
    """
    Convert the shape of a single tensor of unique feature attributionto
    to match the shape of the inputs returned by dataloader
    """

    # unique_attr in shape(*output_dims, n_features)
    output_dims = unique_attr.shape[:-1]
    n_features = unique_attr.shape[-1]

    attr = []

    for inp, mask in zip(attr_inputs, feature_mask):
        # input in shape(batch_size, *inp_feature_dims)
        # attribute in shape(*output_dims, *inp_feature_dims)
        # pyre-fixme[60]: Concatenation not yet support for multiple variadic
        #  tuples: `*output_dims, *inp.shape[slice(1, None, None)]`.
        attr_shape = (*output_dims, *inp.shape[1:])

        expanded_feature_indices = mask.expand(attr_shape)

        if len(inp.shape) > 2:
            # exclude batch_size & last of actual value
            extra_inp_dims = list(inp.shape[1:-1])

            # unsqueeze unqiue_attr to have same number of dims as inp
            # (*output_dims, 1..., 1, n_features)
            # then broadcast to (*output_dims, *inp.shape[1:-1], n_features)
            n_extra_dims = len(extra_inp_dims)
            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            #  tuples: `*output_dims, *(1).__mul__(n_extra_dims)`.
            unsqueezed_shape = (*output_dims, *(1,) * n_extra_dims, n_features)
            # pyre-fixme[60]: Concatenation not yet support for multiple variadic
            #  tuples: `*output_dims, *extra_inp_dims`.
            expanded_shape = (*output_dims, *extra_inp_dims, n_features)
            expanded_unqiue_attr = unique_attr.reshape(unsqueezed_shape).expand(
                expanded_shape
            )
        else:
            expanded_unqiue_attr = unique_attr

        # gather from (*output_dims, *inp.shape[1:-1], n_features)
        inp_attr = torch.gather(expanded_unqiue_attr, -1, expanded_feature_indices)
        attr.append(inp_attr)

    return tuple(attr)


class DataLoaderAttribution(Attribution):
    r"""
    Decorate a perturbation-based attribution algorthm to make it work with dataloaders.
    The decorated instance will calculate attribution in the
    same way as configured in the original attribution instance, but it will provide a
    new "attribute" function which accepts a pytorch "dataloader" instance as the input
    instead of a single batched "tensor" and supports customizing a "reduce" function to
    determine how the forward return of each iteration of the dataloader should be
    aggregated to single metric tensor to attribute. This would
    be specially useful to attribute against some corpus-wise metrics,
    e.g., Precision & Recall.
    """

    attr_method: Attribution

    def __init__(self, attr_method: Attribution) -> None:
        r"""
        Args:
            attr_method (Attribution): An instance of any attribution algorithm
                        of type `Attribution`. E.g. Integrated Gradients,
                        Conductance or Saliency.
        """

        assert (
            type(attr_method) in SUPPORTED_METHODS
        ), f"DataloaderAttribution does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method = copy(attr_method)

        self.attr_method.forward_func = self._forward_with_dataloader

    def _forward_with_dataloader(
        self,
        batched_perturbed_feature_indices: Tensor,
        dataloader: torch.utils.data.DataLoader,
        input_roles: Tuple[int],
        baselines: Tuple[Union[int, float, Tensor], ...],
        feature_mask: Tuple[Tensor, ...],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        reduce: Callable,
        to_metric: Optional[Callable[[Tensor], Tensor]],
        show_progress: bool,
        feature_idx_to_mask_idx: Dict[int, List[int]],
    ) -> Tensor:
        """
        Wrapper of the original given forward_func to be used in the attribution method
        It iterates over the dataloader with the given forward_func
        """

        # batched_perturbed_feature_indices in shape(n_perturb, n_features)
        # n_perturb is not always the same as perturb_per_pass if not enough perturb
        perturbation_mask_list: List[Tuple[Union[Tensor, None], ...]] = [
            _create_perturbation_mask(
                perturbed_feature_indices,
                feature_mask,
                feature_idx_to_mask_idx,
            )
            for perturbed_feature_indices in batched_perturbed_feature_indices
        ]

        # each perturbation needs an accum state
        accum_states = [None for _ in range(len(perturbation_mask_list))]

        # tranverse the dataloader
        for inputs in dataloader:
            # for each batch read from the dataloader,
            # apply every perturbation based on perturbations_per_pass
            for i, perturbation_mask in enumerate(perturbation_mask_list):
                perturbed_inputs = _perturb_inputs(
                    inputs, input_roles, baselines, perturbation_mask
                )

                # due to explicitly defined roles
                # we can keep inputs in their original order
                # regardless of if they need attr
                # instead of using additional_forward_inputs
                forward_inputs = tuple(
                    _
                    for _, role in zip(perturbed_inputs, input_roles)
                    if role != InputRole.no_forward
                )

                output = _run_forward(
                    self.forward_func,
                    forward_inputs,
                )

                accum_states[i] = reduce(accum_states[i], output, perturbed_inputs)

        accum_states = cast(List[Tensor], accum_states)
        accum_results: List[Tensor] = [
            to_metric(accum) if to_metric else accum for accum in accum_states
        ]

        assert all(type(r) is Tensor for r in accum_results), (
            "Accumulated metrics for attribution must be a Tensor,"
            f"received: {next(r for r in accum_results if type(r) is not Tensor)}"
        )

        # shape(n_perturb * output_dims[0], *output_dims[1:])
        # the underneath attr method needs to support forward_func output's
        # 1st dim to grow with perturb_per_eval
        batched_accum = torch.stack(accum_results, dim=0)
        return batched_accum

    def attribute(
        self,
        dataloader: torch.utils.data.DataLoader,
        input_roles: Optional[Tuple[int, ...]] = None,
        baselines: BaselineType = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        reduce: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        to_metric: Optional[Callable] = None,
        perturbations_per_pass: int = 1,
        show_progress: bool = False,
        return_input_shape: bool = True,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        r"""
        Args:

            dataloader (torch.Dataloader): the dataloader to attribute, which should
                        return a tuple of consistent size for every iteration
            input_roles (tuple[int, ...], optional): a tuple of integers to define the
                        role of each element returned from the dataloader. It should
                        have the same size as the return of the dataloader.
                        The available roles are:

                        0: the element is passed to forward_func and needs attribution.
                        It must be a tensor.
                        1: the element is excluded for forward_func. A typical example
                        is the label.
                        2: the element is passed to forward_func but does not need
                        attribution. Like additional_forward_args

            baselines (Union[Tensor, tuple[Tensor, ...]], optional): same as the
                        baseline in attribute. The same baseline will be
                        applied to the entire dataloader. The first dimension is
                        assumed to be batch size and it must be 1. Baselines should only
                        be specififed for the dataloader's returns that need
                        attribution (role = 0)

            feature_mask (Union[Tensor, tuple[Tensor, ...]], optional): same as the
                        feature_mask in attribute. The same feature_mask will be
                        applied to the entire dataloader. The first dimension is
                        assumed to be batch size and it must be 1. Mask should only
                        be specififed for the dataloader's returns that need
                        attribution (role = 0)
            reduce (Callable, optional): a function to accumulate the forward output of
                        each iteration of the dataloader. The function signature is:
                        ``reduce(accum, current_output, current_inputs) -> accum``,
                        where:

                        accum (Any): accumulated states, can be any type
                        current_output (Tensor): current output tensor from forward_func
                        current_inputs (tuple[Any,...]): current inputs from dataloader

            to_metric (Callable, optional): an optional function to further convert
                        accumulated results through "reduce" after tranversing the whole
                        dataloader to a single tensor of metrics to calculate
                        attribution against. The function signature is:
                        ``to_metric(accum) -> metric``, where:

                        accum (Any): accumulated state from reduce function
                        metric (Tensor): final result to be attributed, must be a Tensor

                        If None, will directly attribute w.r.t the reduced ``accum``
            perturbations_per_pass (int, optional) the number perturbations to execute
                        concurrently in each traverse of the dataloader. The number of
                        traverses needed is
                        ceil(n_perturbations / perturbations_per_pass).

                        This argument offers control of the trade-off between memory
                        and efficiency. If the dataloader involves slow operations like
                        remote request or file I/O, multiple traversals can be
                        inefficient. On the other hand, each perturbation needs to
                        store its accumulated outputs of the reduce
                        function until the end of the data traverse.
            return_input_shape (bool, optional): if True, returns the attribution
                        following the input shapes given by the dataloader.
                        Otherwise, returns a single tensor for the attributions of
                        all the features, where the last dimension
                        is the number of features.

        Returns:
            **attributions** :
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Attribution with respect to each input feature.
                        if return_input_shape is True, attributions will be
                        the same size as the given dataloader's returns that need
                        attribution (role = 0), with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
                        If return_input_shape is False, a single tensor is returned
                        where each index of the last dimension represents a feature
        """
        inputs = cast(Union[Tensor, Tuple[Tensor, ...]], next(iter(dataloader)))
        is_inputs_tuple = True

        inputs_tuple: Tuple[Tensor, ...]
        if type(inputs) is list:
            # support list as it is a common return type for dataloader in torch
            inputs_tuple = tuple(inputs)
        elif type(inputs) is not tuple:
            is_inputs_tuple = False
            inputs_tuple = _format_tensor_into_tuples(inputs)

        if input_roles:
            assert len(input_roles) == len(inputs_tuple), (
                "input_roles must have the same size as the return of the dataloader,",
                f"length of input_roles is {len(input_roles)} ",
                f"whereas the length of dataloader return is {len(inputs_tuple)}",
            )

            assert any(role == InputRole.need_attr for role in input_roles), (
                "input_roles must contain at least one element need attribution"
                f"({InputRole.need_attr}), received input_roles: {input_roles}"
            )
        else:
            # by default, assume every element in the dataloader needs attribution
            input_roles = tuple(InputRole.need_attr for _ in inputs_tuple)

        attr_inputs = tuple(
            inp
            for role, inp in zip(input_roles, inputs_tuple)
            if role == InputRole.need_attr
        )

        baselines = _format_baseline(baselines, attr_inputs)

        assert len(attr_inputs) == len(baselines), (
            "Baselines must have the same size as the return of the dataloader ",
            "that need attribution",
            f"length of baseline is {len(baselines)} ",
            'whereas the length of dataloader return with role "0" is',
            f" {len(inputs_tuple)}",
        )

        for i, baseline in enumerate(baselines):
            if isinstance(baseline, Tensor):
                assert baseline.size(0) == 1, (
                    "If the baseline is a tensor, "
                    "its 1st dim of baseline must be 1 so it can be broadacasted to "
                    "any batch of the dataloader:"
                    f"baselines[{i}].shape = {baseline.shape}"
                )

        feature_mask = _format_feature_mask(feature_mask, attr_inputs)

        assert len(attr_inputs) == len(feature_mask), (
            "Feature mask must have the same size as the return of the dataloader ",
            "that need attribution",
            f"length of feature_mask is {len(feature_mask)} ",
            'whereas the length of dataloader return with role "0"',
            f" is {len(inputs_tuple)}",
        )

        for i, each_mask in enumerate(feature_mask):
            assert each_mask.size(0) == 1, (
                "The 1st dim of feature_mask must be 1 so it can be broadcasted to "
                "any batch of the dataloader:"
                f"feature_mask[{i}].shape = {each_mask.shape}"
            )

        # map to retrieve masks contain a given feature index
        feature_idx_to_mask_idx = defaultdict(list)
        for i, mask in enumerate(feature_mask):
            unqiue_feature_indices = torch.unique(mask).tolist()
            for feature_idx in unqiue_feature_indices:
                feature_idx_to_mask_idx[feature_idx].append(i)

        max_feature_idx = _get_max_feature_index(feature_mask)
        n_features = max_feature_idx + 1

        if reduce is None:
            reduce = _concat_tensors

        # onehot tensor for feature indices
        feature_indices = torch.ones((1, n_features), device=attr_inputs[0].device)

        # unique_attr in shape(*output_dims, n_features)
        unique_attr = self.attr_method.attribute(
            feature_indices,
            perturbations_per_eval=perturbations_per_pass,
            additional_forward_args=(
                dataloader,
                input_roles,
                baselines,
                feature_mask,
                reduce,
                to_metric,
                show_progress,
                feature_idx_to_mask_idx,
            ),
        )

        if not return_input_shape:
            return unique_attr
        else:
            attr = _convert_output_shape(
                unique_attr,
                attr_inputs,
                feature_mask,
            )

            return _format_output(is_inputs_tuple, attr)

    # pyre-fixme[24] Generic type `Callable` expects 2 type parameters.
    def attribute_future(self) -> Callable:
        r"""
        This method is not implemented for DataLoaderAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for DataLoaderAttribution"
        )
