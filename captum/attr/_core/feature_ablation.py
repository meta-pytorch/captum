#!/usr/bin/env python3

# pyre-strict

import logging
import math
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _get_feature_idx_to_tensor_idx,
    _is_tuple,
    _maybe_expand_parameters,
    _run_forward,
)
from captum._utils.exceptions import FeatureAblationFutureError
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    get_total_features_from_mask,
)
from captum.log import log_usage
from torch import dtype, Tensor
from torch.futures import collect_all, Future

from tqdm.auto import tqdm

IterableType = TypeVar("IterableType")

logger: logging.Logger = logging.getLogger(__name__)


class FeatureAblation(PerturbationAttribution):
    r"""
    A perturbation based approach to computing attribution, involving
    replacing each input feature with a given baseline / reference, and
    computing the difference in output. By default, each scalar value within
    each input tensor is taken as a feature and replaced independently. Passing
    a feature mask, allows grouping features to be ablated together. This can
    be used in cases such as images, where an entire segment or region
    can be ablated, measuring the importance of the segment (feature group).
    Each input scalar in the group will be given the same attribution value
    equal to the change in target as a result of ablating the entire feature
    group.

    The forward function can either return a scalar per example or a tensor
    of a fixed sized tensor (or scalar value) for the full batch, i.e. the
    output does not grow as the batch size increase. If the output is fixed
    we consider this model to be an "aggregation" of the inputs. In the fixed
    sized output mode we require `perturbations_per_eval == 1` and the
    `feature_mask` to be either `None` or for all of them to have 1 as their
    first dimension (i.e. a feature mask requires to be applied to all inputs).
    """

    def __init__(
        self, forward_func: Callable[..., Union[int, float, Tensor, Future[Tensor]]]
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.use_weights = False

        # only used when perturbations_per_eval > 1, where the 1st dim of forward_func's
        # output must grow as the input batch size. If forward's output is aggregated,
        # we cannot expand the input to include more perturbations in one call.
        # If it's False, we will force the validation by comparing the outpus of
        # the original input and the modified input whose batch size expanded based on
        # perturbations_per_eval. Set the flag to True if the output of the modified
        # input grow as expected. Once it turns to True, we will assume the model's
        # behavior stays consistent and no longer check again
        self._is_output_shape_valid = False

        # Considering the case when we permute multiple input tensors at once
        # through `feature_mask`, we disregard the feature group if the 0th
        # dim of *any* input tensor in the group is less than
        # `_min_examples_per_batch_grouped` if defined.
        # If *all* input tensors in the group are empty, we also skip the feature/
        # feature group (not parameterized by `_min_examples_per_batch_grouped`).
        self._min_examples_per_batch_grouped: Optional[int] = None

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:
            inputs (Tensor or tuple[Tensor, ...]): Input for which ablation
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        multiple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference value which replaces each
                        feature when ablated.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or
                          broadcastable to match the dimensions of inputs

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.
                        Default: None
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
                        this is usually the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary.
                        For general 2D outputs, targets can be either:

                        - a single integer or a tensor containing a single
                          integer, which is applied to all input examples

                        - a list of integers or a 1D tensor, with length matching
                          the number of examples in inputs (dim 0). Each integer
                          is applied as the target for the corresponding example.

                        For outputs with > 2 dimensions, targets can be either:

                        - A single tuple, which contains #output_dims - 1
                          elements. This target index is applied to all examples.

                        - A list of tuples with length equal to the number of
                          examples in inputs (dim 0), and each tuple containing
                          #output_dims - 1 elements. Each tuple is applied as the
                          target for the corresponding example.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which should be ablated together. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Each tensor
                        should contain integers in the range 0 to num_features
                        - 1, and indices corresponding to the same feature should
                        have the same value.
                        If the forward function returns a single scalar per batch,
                        we enforce that the first dimension of each mask must be 1,
                        since attributions are returned batch-wise rather than per
                        example, so the attributions must correspond to the
                        same features (indices) in each input example.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature, which
                        is ablated independently by default.
                        Default: None
            perturbations_per_eval (int, optional): Allows ablation of multiple
                        features to be processed simultaneously in one call to
                        forward_fn.
                        Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples samples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain at most
                        (perturbations_per_eval * #examples) / num_devices
                        samples.
                        If the forward function's number of outputs does not
                        change as the batch size grows (e.g. if it outputs a
                        scalar value), you must set perturbations_per_eval to 1
                        and use a single feature mask to describe the features
                        for all examples in the batch.
                        Default: 1
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
            **kwargs (Any, optional): Any additional arguments used by child
                        classes of FeatureAblation (such as Occlusion) to construct
                        ablations. These arguments are ignored when using
                        FeatureAblation directly.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The attributions with respect to each input feature.
                        If the forward function returns
                        a scalar value per example, attributions will be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If the forward function returns a scalar per batch, then
                        attribution tensor(s) will have first dimension 1 and
                        the remaining dimensions will match the input.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple of tensors is provided for inputs, a
                        tuple of corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining FeatureAblation interpreter
            >>> ablator = FeatureAblation(net)
            >>> # Computes ablation attribution, ablating each of the 16
            >>> # scalar input independently.
            >>> attr = ablator.attribute(input, target=1)

            >>> # Alternatively, we may want to ablate features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and ablating them together.
            >>> # This can be done by creating a feature mask as follows, which
            >>> # defines the feature groups, e.g.:
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 0 | 0 | 1 | 1 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # | 2 | 2 | 3 | 3 |
            >>> # +---+---+---+---+
            >>> # With this mask, all inputs with the same value are ablated
            >>> # simultaneously, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = ablator.attribute(input, target=1, feature_mask=feature_mask)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)

        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        formatted_additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        formatted_feature_mask = _format_feature_mask(feature_mask, formatted_inputs)

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be an integer and at least 1."
        with torch.no_grad():
            attr_progress = None
            if show_progress:
                attr_progress = self._attribute_progress_setup(
                    formatted_inputs,
                    formatted_feature_mask,
                    **kwargs,
                    perturbations_per_eval=perturbations_per_eval,
                )
                attr_progress.update(0)

            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval: Union[Tensor, Future[Tensor]] = _run_forward(
                self.forward_func,
                formatted_inputs,
                target,
                formatted_additional_forward_args,
            )
            if attr_progress is not None:
                attr_progress.update()

            total_attrib: List[Tensor] = []
            weights: List[Tensor] = []
            flattened_initial_eval: Tensor
            n_outputs: int
            attrib_type: dtype

            if isinstance(initial_eval, torch.Future):
                raise AssertionError(
                    "when using the attribute function, initial_eval should have "
                    f"non-Future type rather than {type(initial_eval)}"
                )

            (
                total_attrib,
                weights,
                initial_eval,
                flattened_initial_eval,
                n_outputs,
                attrib_type,
            ) = self._process_initial_eval(
                initial_eval,
                formatted_inputs,
            )

            total_attrib, weights = self._attribute_with_cross_tensor_feature_masks(
                formatted_inputs,
                formatted_additional_forward_args,
                target,
                baselines,
                formatted_feature_mask,
                attr_progress,
                flattened_initial_eval,
                initial_eval,
                n_outputs,
                total_attrib,
                weights,
                attrib_type,
                perturbations_per_eval,
                **kwargs,
            )

        if attr_progress is not None:
            attr_progress.close()

        return cast(
            TensorOrTupleOfTensorsGeneric,
            self._generate_result(total_attrib, weights, is_inputs_tuple),
        )

    def _attribute_with_cross_tensor_feature_masks(
        self,
        formatted_inputs: Tuple[Tensor, ...],
        formatted_additional_forward_args: Optional[Tuple[object, ...]],
        target: TargetType,
        baselines: BaselineType,
        formatted_feature_mask: Tuple[Tensor, ...],
        attr_progress: Optional[tqdm],
        flattened_initial_eval: Tensor,
        initial_eval: Tensor,
        n_outputs: int,
        total_attrib: List[Tensor],
        weights: List[Tensor],
        attrib_type: dtype,
        perturbations_per_eval: int,
        **kwargs: Any,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        feature_idx_to_tensor_idx = self._get_feature_idx_to_tensor_idx(
            formatted_feature_mask, **kwargs
        )
        all_feature_idxs = list(feature_idx_to_tensor_idx.keys())

        (all_features_repeated, additional_args_repeated, target_repeated) = (
            _maybe_expand_parameters(
                perturbations_per_eval,
                formatted_inputs,
                formatted_additional_forward_args,
                target,
            )
        )
        num_examples = formatted_inputs[0].shape[0]

        current_additional_args: object
        if isinstance(baselines, tuple):
            reshaped = False
            reshaped_baselines: list[Union[Tensor, int, float]] = []
            for baseline in baselines:
                if isinstance(baseline, Tensor):
                    reshaped = True
                    reshaped_baselines.append(
                        baseline.reshape((1,) + tuple(baseline.shape))
                    )
                else:
                    reshaped_baselines.append(baseline)
            baselines = tuple(reshaped_baselines) if reshaped else baselines
        for i in range(0, len(all_feature_idxs), perturbations_per_eval):
            current_feature_idxs = all_feature_idxs[i : i + perturbations_per_eval]
            current_num_ablated_features = min(
                perturbations_per_eval, len(current_feature_idxs)
            )

            if self._should_skip_inputs_and_warn(
                current_feature_idxs,
                feature_idx_to_tensor_idx,
                formatted_inputs,
            ):
                continue

            # Store appropriate inputs and additional args based on batch size.
            if current_num_ablated_features != perturbations_per_eval:
                current_additional_args = (
                    _expand_additional_forward_args(
                        formatted_additional_forward_args, current_num_ablated_features
                    )
                    if formatted_additional_forward_args is not None
                    else None
                )
                current_target = _expand_target(target, current_num_ablated_features)
                expanded_inputs = tuple(
                    feature_repeated[0 : current_num_ablated_features * num_examples]
                    for feature_repeated in all_features_repeated
                )
            else:
                current_additional_args = additional_args_repeated
                current_target = target_repeated
                expanded_inputs = all_features_repeated

            current_inputs, current_masks = (
                self._construct_ablated_input_across_tensors(
                    expanded_inputs,
                    formatted_feature_mask,
                    baselines,
                    current_feature_idxs,
                    feature_idx_to_tensor_idx,
                    current_num_ablated_features,
                    **kwargs,
                )
            )

            # modified_eval has (n_feature_perturbed * n_outputs) elements
            # shape:
            #   agg mode: (*initial_eval.shape)
            #   non-agg mode:
            #     (feature_perturbed * batch_size, *initial_eval.shape[1:])
            modified_eval = _run_forward(
                self.forward_func,
                current_inputs,
                current_target,
                current_additional_args,
            )

            if attr_progress is not None:
                attr_progress.update()

            assert not isinstance(modified_eval, torch.Future), (
                "when use_futures is True, modified_eval should have "
                f"non-Future type rather than {type(modified_eval)}"
            )

            total_attrib, weights = self._process_ablated_out_full(
                modified_eval,
                current_masks,
                flattened_initial_eval,
                initial_eval,
                current_inputs,
                n_outputs,
                num_examples,
                total_attrib,
                weights,
                attrib_type,
                perturbations_per_eval,
            )
        return total_attrib, weights

    def _get_feature_idx_to_tensor_idx(
        self, formatted_feature_mask: Tuple[Tensor, ...], **kwargs: Any
    ) -> Dict[int, List[int]]:
        return _get_feature_idx_to_tensor_idx(formatted_feature_mask)

    def _should_skip_inputs_and_warn(
        self,
        current_feature_idxs: List[int],
        feature_idx_to_tensor_idx: Dict[int, List[int]],
        formatted_inputs: Tuple[Tensor, ...],
    ) -> bool:
        should_skip = False
        all_empty = True
        tensor_idx_list = []
        for feature_idx in current_feature_idxs:
            tensor_idx_list += feature_idx_to_tensor_idx[feature_idx]
        for tensor_idx in set(tensor_idx_list):
            if all_empty and torch.numel(formatted_inputs[tensor_idx]) != 0:
                all_empty = False
            if self._min_examples_per_batch_grouped is not None and (
                formatted_inputs[tensor_idx].shape[0]
                < cast(int, self._min_examples_per_batch_grouped)
            ):
                should_skip = True
                break
        if should_skip:
            logger.warning(
                f"Skipping feature group {current_feature_idxs} since it contains "
                f"at least one input tensor with 0th dim less than "
                f"{self._min_examples_per_batch_grouped}"
            )
            return True
        if all_empty:
            logger.info(
                f"Skipping feature group {current_feature_idxs} since all "
                f"input tensors are empty"
            )
            return True
        return False

    def _construct_ablated_input_across_tensors(
        self,
        inputs: Tuple[Tensor, ...],
        input_mask: Tuple[Tensor, ...],
        baselines: BaselineType,
        feature_idxs: List[int],
        feature_idx_to_tensor_idx: Dict[int, List[int]],
        current_num_ablated_features: int,
        **kwargs: Any,
    ) -> Tuple[Tuple[Tensor, ...], Tuple[Optional[Tensor], ...]]:
        ablated_inputs = []
        current_masks: List[Optional[Tensor]] = []
        tensor_idxs = {
            tensor_idx
            for sublist in (
                feature_idx_to_tensor_idx[feature_idx] for feature_idx in feature_idxs
            )
            for tensor_idx in sublist
        }

        for i, input_tensor in enumerate(inputs):
            if i not in tensor_idxs:
                ablated_inputs.append(input_tensor)
                current_masks.append(None)
                continue
            tensor_mask = []
            ablated_input = input_tensor.clone()
            baseline = baselines[i] if isinstance(baselines, tuple) else baselines
            for j, feature_idx in enumerate(feature_idxs):
                original_input_size = (
                    input_tensor.shape[0] // current_num_ablated_features
                )
                start_idx = j * original_input_size
                end_idx = (j + 1) * original_input_size

                mask = (input_mask[i] == feature_idx).to(input_tensor.device).long()
                if mask.ndim == 0:
                    mask = mask.reshape((1,) * input_tensor.dim())
                tensor_mask.append(mask)

                assert baseline is not None, "baseline must be provided"
                ablated_feature = input_tensor[start_idx:end_idx] * (1 - mask).to(
                    input_tensor.dtype
                ) + (baseline * mask.to(input_tensor.dtype))
                ablated_input = ablated_input.to(ablated_feature.dtype)
                ablated_input[start_idx:end_idx] = ablated_feature
            current_masks.append(torch.stack(tensor_mask, dim=0))
            ablated_inputs.append(ablated_input)

        return tuple(ablated_inputs), tuple(current_masks)

    def _initial_eval_to_processed_initial_eval_fut(
        self, initial_eval: Future[Tensor], formatted_inputs: Tuple[Tensor, ...]
    ) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor, int, dtype]:
        try:
            initial_eval_processed = initial_eval.value()
            if not isinstance(initial_eval_processed, Tensor):
                raise AssertionError(
                    "initial_eval_to_processed_initial_eval_fut: "
                    "initial_eval should be a Tensor"
                )
            result = self._process_initial_eval(
                initial_eval_processed, formatted_inputs
            )

        except FeatureAblationFutureError as e:
            raise FeatureAblationFutureError(
                "initial_eval_to_processed_initial_eval_fut func failed"
            ) from e
        return result

    @log_usage(part_of_slo=True)
    def attribute_future(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Future[TensorOrTupleOfTensorsGeneric]:
        r"""
        Almost the same as the attribute function, except that it requires a
        forward function that returns a Future, and it returns a Future.
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        formatted_additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        formatted_feature_mask = _format_feature_mask(feature_mask, formatted_inputs)

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Perturbations per evaluation must be an integer and at least 1."
        with torch.no_grad():
            attr_progress = None
            if show_progress:
                attr_progress = self._attribute_progress_setup(
                    formatted_inputs,
                    formatted_feature_mask,
                    **kwargs,
                    perturbations_per_eval=perturbations_per_eval,
                )
                attr_progress.update(0)

            # Computes initial evaluation with all features, which is compared
            # to each ablated result.
            initial_eval: Union[Tensor, Future[Tensor]] = _run_forward(
                self.forward_func,
                formatted_inputs,
                target,
                formatted_additional_forward_args,
            )

            if attr_progress is not None:
                attr_progress.update()

            processed_initial_eval_fut: Optional[
                Future[Tuple[List[Tensor], List[Tensor], Tensor, Tensor, int, dtype]]
            ] = None

            if not isinstance(initial_eval, torch.Future):
                raise AssertionError(
                    "when using attribute_future, initial_eval should have "
                    f"Future type rather than {type(initial_eval)}"
                )

            processed_initial_eval_fut = initial_eval.then(
                lambda initial_eval: self._initial_eval_to_processed_initial_eval_fut(
                    initial_eval,
                    formatted_inputs,
                )
            )

            return cast(
                Future[TensorOrTupleOfTensorsGeneric],
                self._attribute_with_cross_tensor_feature_masks_future(
                    formatted_inputs=formatted_inputs,
                    formatted_additional_forward_args=formatted_additional_forward_args,  # noqa: E501 line too long
                    target=target,
                    baselines=baselines,
                    formatted_feature_mask=formatted_feature_mask,
                    attr_progress=attr_progress,
                    processed_initial_eval_fut=processed_initial_eval_fut,
                    is_inputs_tuple=is_inputs_tuple,
                    perturbations_per_eval=perturbations_per_eval,
                ),
            )

    def _attribute_with_cross_tensor_feature_masks_future(
        self,
        formatted_inputs: Tuple[Tensor, ...],
        formatted_additional_forward_args: Optional[Tuple[object, ...]],
        target: TargetType,
        baselines: BaselineType,
        formatted_feature_mask: Tuple[Tensor, ...],
        attr_progress: Optional[tqdm],
        processed_initial_eval_fut: Future[
            Tuple[List[Tensor], List[Tensor], Tensor, Tensor, int, dtype]
        ],
        is_inputs_tuple: bool,
        perturbations_per_eval: int,
        **kwargs: Any,
    ) -> Future[Union[Tensor, Tuple[Tensor, ...]]]:
        feature_idx_to_tensor_idx = self._get_feature_idx_to_tensor_idx(
            formatted_feature_mask, **kwargs
        )
        all_feature_idxs = list(feature_idx_to_tensor_idx.keys())

        (all_features_repeated, additional_args_repeated, target_repeated) = (
            _maybe_expand_parameters(
                perturbations_per_eval,
                formatted_inputs,
                formatted_additional_forward_args,
                target,
            )
        )
        num_examples = formatted_inputs[0].shape[0]

        current_additional_args: object
        if isinstance(baselines, tuple):
            reshaped = False
            reshaped_baselines: list[Union[Tensor, int, float]] = []
            for baseline in baselines:
                if isinstance(baseline, Tensor):
                    reshaped = True
                    reshaped_baselines.append(
                        baseline.reshape((1,) + tuple(baseline.shape))
                    )
                else:
                    reshaped_baselines.append(baseline)
            baselines = tuple(reshaped_baselines) if reshaped else baselines

        all_modified_eval_futures: List[Future[Tuple[List[Tensor], List[Tensor]]]] = []
        for i in range(0, len(all_feature_idxs), perturbations_per_eval):
            current_feature_idxs = all_feature_idxs[i : i + perturbations_per_eval]
            current_num_ablated_features = min(
                perturbations_per_eval, len(current_feature_idxs)
            )

            if self._should_skip_inputs_and_warn(
                current_feature_idxs,
                feature_idx_to_tensor_idx,
                formatted_inputs,
            ):
                continue

            # Store appropriate inputs and additional args based on batch size.
            if current_num_ablated_features != perturbations_per_eval:
                current_additional_args = (
                    _expand_additional_forward_args(
                        formatted_additional_forward_args, current_num_ablated_features
                    )
                    if formatted_additional_forward_args is not None
                    else None
                )
                current_target = _expand_target(target, current_num_ablated_features)
                expanded_inputs = tuple(
                    feature_repeated[0 : current_num_ablated_features * num_examples]
                    for feature_repeated in all_features_repeated
                )
            else:
                current_additional_args = additional_args_repeated
                current_target = target_repeated
                expanded_inputs = all_features_repeated

            current_inputs, current_masks = (
                self._construct_ablated_input_across_tensors(
                    expanded_inputs,
                    formatted_feature_mask,
                    baselines,
                    current_feature_idxs,
                    feature_idx_to_tensor_idx,
                    current_num_ablated_features,
                    **kwargs,
                )
            )

            # modified_eval has (n_feature_perturbed * n_outputs) elements
            # shape:
            #   agg mode: (*initial_eval.shape)
            #   non-agg mode:
            #     (feature_perturbed * batch_size, *initial_eval.shape[1:])
            modified_eval = _run_forward(
                self.forward_func,
                current_inputs,
                current_target,
                current_additional_args,
            )

            if attr_progress is not None:
                attr_progress.update()

            if not isinstance(modified_eval, torch.Future):
                raise AssertionError(
                    "when using attribute_future, modified_eval should have "
                    f"Future type rather than {type(modified_eval)}"
                )

            # Need to collect both initial eval and modified_eval
            eval_futs: Future[
                List[
                    Future[
                        Union[
                            Tuple[
                                List[Tensor],
                                List[Tensor],
                                Tensor,
                                Tensor,
                                int,
                                dtype,
                            ],
                            Tensor,
                        ]
                    ]
                ]
            ] = collect_all(
                [
                    processed_initial_eval_fut,
                    modified_eval,
                ]
            )

            ablated_out_fut: Future[Tuple[List[Tensor], List[Tensor]]] = eval_futs.then(
                lambda eval_futs, current_inputs=current_inputs, current_mask=current_masks, i=i: self._eval_fut_to_ablated_out_fut_cross_tensor(  # type: ignore # noqa: E501 line too long
                    eval_futs=eval_futs,
                    current_inputs=current_inputs,
                    current_mask=current_mask,
                    perturbations_per_eval=perturbations_per_eval,
                    num_examples=num_examples,
                )
            )

            all_modified_eval_futures.append(ablated_out_fut)

        if attr_progress is not None:
            attr_progress.close()

        return self._generate_async_result_cross_tensor(
            all_modified_eval_futures,
            is_inputs_tuple,
        )

    def _fut_tuple_to_accumulate_fut_list_cross_tensor(
        self,
        total_attrib: List[Tensor],
        weights: List[Tensor],
        fut_tuple: Future[Tuple[List[Tensor], List[Tensor]]],
    ) -> None:
        try:
            # process_ablated_out_* already accumlates the total attribution.
            # Just get the latest value
            attribs, this_weights = fut_tuple.value()
            total_attrib[:] = attribs
            weights[:] = this_weights
        except FeatureAblationFutureError as e:
            raise FeatureAblationFutureError(
                "_fut_tuple_to_accumulate_fut_list_cross_tensor failed"
            ) from e

    def _attribute_progress_setup(
        self,
        formatted_inputs: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        perturbations_per_eval: int,
        **kwargs: Any,
    ) -> tqdm:
        total_forwards = math.ceil(
            get_total_features_from_mask(feature_mask) / perturbations_per_eval
        )
        total_forwards += 1  # add 1 for the initial eval
        attr_progress = progress(
            desc=f"{self.get_name()} attribution", total=total_forwards
        )
        return attr_progress

    def _generate_async_result_cross_tensor(
        self,
        futs: List[Future[Tuple[List[Tensor], List[Tensor]]]],
        is_inputs_tuple: bool,
    ) -> Future[Union[Tensor, Tuple[Tensor, ...]]]:
        accumulate_fut_list: List[Future[None]] = []
        total_attrib: List[Tensor] = []
        weights: List[Tensor] = []

        for fut_tuple in futs:
            accumulate_fut_list.append(
                fut_tuple.then(
                    lambda fut_tuple: self._fut_tuple_to_accumulate_fut_list_cross_tensor(  # noqa: E501 line too long
                        total_attrib, weights, fut_tuple
                    )
                )
            )

        result_fut = collect_all(accumulate_fut_list).then(
            lambda x: self._generate_result(
                total_attrib,
                weights,
                is_inputs_tuple,
            )
        )

        return result_fut

    def _eval_fut_to_ablated_out_fut_cross_tensor(
        self,
        eval_futs: Future[List[Future[List[object]]]],
        current_inputs: Tuple[Tensor, ...],
        current_mask: Tuple[Optional[Tensor], ...],
        perturbations_per_eval: int,
        num_examples: int,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        try:
            modified_eval = cast(Tensor, eval_futs.value()[1].value())
            initial_eval_tuple = cast(
                Tuple[
                    List[Tensor],
                    List[Tensor],
                    Tensor,
                    Tensor,
                    int,
                    dtype,
                ],
                eval_futs.value()[0].value(),
            )
            if len(initial_eval_tuple) != 6:
                raise AssertionError(
                    "eval_fut_to_ablated_out_fut_cross_tensor: "
                    "initial_eval_tuple should have 6 elements: "
                    "total_attrib, weights, initial_eval, "
                    "flattened_initial_eval, n_outputs, attrib_type "
                )
            if not isinstance(modified_eval, Tensor):
                raise AssertionError(
                    "_eval_fut_to_ablated_out_fut_cross_tensor: "
                    "modified eval should be a Tensor"
                )
            (
                total_attrib,
                weights,
                initial_eval,
                flattened_initial_eval,
                n_outputs,
                attrib_type,
            ) = initial_eval_tuple
            total_attrib, weights = self._process_ablated_out_full(
                modified_eval=modified_eval,
                inputs=current_inputs,
                current_mask=current_mask,
                perturbations_per_eval=perturbations_per_eval,
                num_examples=num_examples,
                initial_eval=initial_eval,
                flattened_initial_eval=flattened_initial_eval,
                n_outputs=n_outputs,
                total_attrib=total_attrib,
                weights=weights,
                attrib_type=attrib_type,
            )
        except FeatureAblationFutureError as e:
            raise FeatureAblationFutureError(
                "_eval_fut_to_ablated_out_fut_cross_tensor func failed"
            ) from e
        return total_attrib, weights

    def _parse_forward_out(self, forward_output: Tensor) -> Tensor:
        """
        A temp wrapper for global _run_forward util to force forward output
        type assertion & conversion.
        Remove after the strict logic is supported by all attr classes
        """
        if isinstance(forward_output, Tensor):
            return forward_output

        output_type = type(forward_output)
        assert output_type is int or output_type is float, (
            "the return of forward_func must be a tensor, int, or float,"
            f" received: {forward_output}"
        )

        # using python built-in type as torch dtype
        # int -> torch.int64, float -> torch.float64
        # ref: https://github.com/pytorch/pytorch/pull/21215
        return torch.tensor(forward_output, dtype=cast(dtype, output_type))

    def _process_initial_eval(
        self,
        initial_eval: Tensor,
        inputs: TensorOrTupleOfTensorsGeneric,
    ) -> Tuple[List[Tensor], List[Tensor], Tensor, Tensor, int, dtype]:
        initial_eval = self._parse_forward_out(initial_eval)

        # number of elements in the output of forward_func
        n_outputs = initial_eval.numel() if isinstance(initial_eval, Tensor) else 1

        # flatten eval outputs into 1D (n_outputs)
        # add the leading dim for n_feature_perturbed
        flattened_initial_eval = initial_eval.reshape(1, -1)

        # Initialize attribution totals and counts
        attrib_type = flattened_initial_eval.dtype

        total_attrib = [
            # attribute w.r.t each output element
            torch.zeros(
                (n_outputs,) + input.shape[1:],
                dtype=attrib_type,
                device=input.device,
            )
            for input in inputs
        ]

        # Weights are used in cases where ablations may be overlapping.
        weights = []
        if self.use_weights:
            weights = [
                torch.zeros((n_outputs,) + input.shape[1:], device=input.device).float()
                for input in inputs
            ]

        return (
            total_attrib,
            weights,
            initial_eval,
            flattened_initial_eval,
            n_outputs,
            attrib_type,
        )

    def _process_ablated_out_full(
        self,
        modified_eval: Tensor,
        current_mask: Tuple[Optional[Tensor], ...],
        flattened_initial_eval: Tensor,
        initial_eval: Tensor,
        inputs: TensorOrTupleOfTensorsGeneric,
        n_outputs: int,
        num_examples: int,
        total_attrib: List[Tensor],
        weights: List[Tensor],
        attrib_type: dtype,
        perturbations_per_eval: int,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        modified_eval = self._parse_forward_out(modified_eval)
        # if perturbations_per_eval > 1, the output shape must grow with
        # input and not be aggregated
        current_batch_size = inputs[0].shape[0]

        # number of perturbation, which is not the same as
        # perturbations_per_eval when not enough features to perturb
        n_perturb = current_batch_size / num_examples
        if perturbations_per_eval > 1 and not self._is_output_shape_valid:

            current_output_shape = modified_eval.shape

            # use initial_eval as the forward of perturbations_per_eval = 1
            initial_output_shape = initial_eval.shape

            assert (
                # check if the output is not a scalar
                current_output_shape
                and initial_output_shape
                # check if the output grow in same ratio, i.e., not agg
                and current_output_shape[0] == n_perturb * initial_output_shape[0]
            ), (
                "When perturbations_per_eval > 1, forward_func's output "
                "should be a tensor whose 1st dim grow with the input "
                f"batch size: when input batch size is {num_examples}, "
                f"the output shape is {initial_output_shape}; "
                f"when input batch size is {current_batch_size}, "
                f"the output shape is {current_output_shape}"
            )

            self._is_output_shape_valid = True

        # reshape the leading dim for n_feature_perturbed
        # flatten each feature's eval outputs into 1D of (n_outputs)
        modified_eval = modified_eval.reshape(-1, n_outputs)
        # eval_diff in shape (n_feature_perturbed, n_outputs)
        eval_diff = flattened_initial_eval - modified_eval
        eval_diff_shape = eval_diff.shape

        if self.use_weights:
            for weight, mask in zip(weights, current_mask):
                if mask is not None:
                    weight += mask.float().sum(dim=0)
        for i, mask in enumerate(current_mask):
            if mask is None or inputs[i].numel() == 0:
                continue
            eval_diff = eval_diff.reshape(
                eval_diff_shape + (inputs[i].dim() - 1) * (1,)
            )
            eval_diff = eval_diff.to(total_attrib[i].device)
            total_attrib[i] += (eval_diff * mask.to(attrib_type)).sum(dim=0)

        return total_attrib, weights

    def _generate_result(
        self,
        total_attrib: List[Tensor],
        weights: List[Tensor],
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        # Divide total attributions by counts and return formatted attributions
        if self.use_weights:
            attrib = tuple(
                single_attrib.float() / weight
                for single_attrib, weight in zip(total_attrib, weights)
            )
        else:
            attrib = tuple(total_attrib)
        return _format_output(is_inputs_tuple, attrib)
