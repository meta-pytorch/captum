#!/usr/bin/env python3

# pyre-strict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.feature_ablation import FeatureAblation
from captum.log import log_usage
from torch import Tensor
from torch.futures import Future


def _permute_feature(x: Tensor, feature_mask: Tensor) -> Tensor:
    n = x.size(0)
    assert n > 1, "cannot permute features with batch_size = 1"

    perm = torch.randperm(n)
    no_perm = torch.arange(n)
    while (perm == no_perm).all():
        perm = torch.randperm(n)

    return (x[perm] * feature_mask.to(dtype=x.dtype)) + (
        x * feature_mask.bitwise_not().to(dtype=x.dtype)
    )


class FeaturePermutation(FeatureAblation):
    r"""
    A perturbation based approach to compute attribution, which
    takes each input feature, permutes the feature values within a batch,
    and computes the difference between original and shuffled outputs for
    the given batch. This difference signifies the feature importance
    for the permuted feature.

    Example pseudocode for the algorithm is as follows::

        perm_feature_importance(batch):
            importance = dict()
            baseline_error = error_metric(model(batch), batch_labels)
            for each feature:
                permute this feature across the batch
                error = error_metric(model(permuted_batch), batch_labels)
                importance[feature] = baseline_error - error
                "un-permute" the feature across the batch

            return importance

    It should be noted that the `error_metric` must be called in the
    `forward_func`. You do not need to have an error metric, e.g. you
    could simply return the logits (the model output), but this may or may
    not provide a meaningful attribution.

    This method, unlike other attribution methods, requires a batch
    of examples to compute attributions and cannot be performed on a single example.

    By default, each scalar value within
    each input tensor is taken as a feature and shuffled independently. Passing
    a feature mask allows grouping features to be shuffled together (including
    features defined across different input tensors).
    Each input scalar in the group will be given the same attribution value
    equal to the change in target as a result of shuffling the entire feature
    group.

    The forward function can either return a scalar per example, or a single
    scalar for the full batch. If a single scalar is returned for the batch,
    `perturbations_per_eval` must be 1, and the returned attributions will have
    first dimension 1, corresponding to feature importance across all
    examples in the batch.

    More information can be found in the permutation feature
    importance algorithm description here:
    https://christophm.github.io/interpretable-ml-book/feature-importance.html
    """

    def __init__(
        self,
        forward_func: Callable[..., Union[int, float, Tensor, Future[Tensor]]],
        perm_func: Callable[[Tensor, Tensor], Tensor] = _permute_feature,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                any modification of it.
            perm_func (Callable, optional): A function that accepts a batch of
                inputs and a feature mask, and "permutes" the feature using
                feature mask across the batch. This defaults to a function
                which applies a random permutation, this argument only needs
                to be provided if a custom permutation behavior is desired.
                Default: `_permute_feature`
        """
        FeatureAblation.__init__(self, forward_func=forward_func)
        self.perm_func = perm_func
        # Considering the case when we permute multiple input tensors at once
        # through `feature_mask`, we disregard the feature group if the 0th
        # dim of *any* input tensor in the group is less than
        # `_min_examples_per_batch_grouped`.
        self._min_examples_per_batch_grouped = 2

    # suppressing error caused by the child class not having a matching
    # signature to the parent
    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This function is almost equivalent to
        :func:`FeatureAblation.attribute <captum.attr.FeatureAblation.attribute>`. The
        main difference is the way ablated examples are generated. Specifically they
        are generated through the ``perm_func``, as we set the baselines for
        :func:`FeatureAblation.attribute <captum.attr.FeatureAblation.attribute>` to
        ``None``.


        Args:
                inputs (Tensor or tuple[Tensor, ...]): Input for which
                            permutation attributions are computed. If
                            forward_func takes a single tensor as input, a
                            single input tensor should be provided.  If
                            forward_func takes multiple tensors as input, a
                            tuple of the input tensors should be provided. It is
                            assumed that for all given input tensors, dimension
                            0 corresponds to the number of examples (aka batch
                            size), and if multiple input tensors are provided,
                            the examples must be aligned appropriately.
                target (int, tuple, Tensor, or list, optional): Output indices for
                            which difference is computed (for classification cases,
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
                            Each tensor should be the same size as the
                            corresponding input or broadcastable to match the
                            input tensor. Each tensor should contain integers in
                            the range 0 to num_features - 1, and indices
                            corresponding to the same feature should have the
                            same value.
                            The first dimension of each mask must be 1, as we require
                            to have the same group of features for each input sample.

                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature.
                            Default: None
                perturbations_per_eval (int, optional): Allows permutations
                            of multiple features to be processed simultaneously
                            in one call to forward_fn.  Each forward pass will
                            contain a maximum of perturbations_per_eval * #examples
                            samples.  For DataParallel models, each batch is
                            split among the available devices, so evaluations on
                            each available device contain at most
                            (perturbations_per_eval * #examples) / num_devices
                            samples.
                            If the forward function returns a single scalar per batch,
                            perturbations_per_eval must be set to 1.
                            Default: 1
                show_progress (bool, optional): Displays the progress of computation.
                            It will try to use tqdm if available for advanced features
                            (e.g. time estimation). Otherwise, it will fallback to
                            a simple output of progress.
                            Default: False
                **kwargs (Any, optional): Any additional arguments used by child
                            classes of :class:`.FeatureAblation` (such as
                            :class:`.Occlusion`) to construct ablations. These
                            arguments are ignored when using FeatureAblation directly.
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
                        returned. If a tuple of tensors is provided for inputs,
                        a tuple of corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 10 x 4 x 4
            >>> input = torch.randn(10, 4, 4)
            >>> # Defining FeaturePermutation interpreter
            >>> feature_perm = FeaturePermutation(net)
            >>> # Computes permutation attribution, shuffling each of the 16
            >>> # scalar input independently.
            >>> attr = feature_perm.attribute(input, target=1)

            >>> # Alternatively, we may want to permute features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and shuffling them together.
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
            >>> # With this mask, all inputs with the same value are shuffled
            >>> # simultaneously, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = feature_perm.attribute(input, target=1,
            >>>                               feature_mask=feature_mask)
        """
        # Remove baselines from kwargs if provided so we don't specify this field
        # twice in the FeatureAblation.attribute call below.
        if isinstance(kwargs, dict) and "baselines" in kwargs:
            del kwargs["baselines"]
        return FeatureAblation.attribute.__wrapped__(
            self,
            inputs,
            baselines=None,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
            **kwargs,
        )

    def attribute_future(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> Future[TensorOrTupleOfTensorsGeneric]:
        """
        Similar to attribute(), but supports async forward functions.
        """
        if isinstance(kwargs, dict) and "baselines" in kwargs:
            del kwargs["baselines"]
        return FeatureAblation.attribute_future.__wrapped__(
            self,
            inputs,
            baselines=None,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
            **kwargs,
        )

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
        current_masks: List[Optional[Tensor]] = []
        tensor_idxs = {
            tensor_idx
            for sublist in (
                feature_idx_to_tensor_idx[feature_idx] for feature_idx in feature_idxs
            )
            for tensor_idx in sublist
        }
        permuted_inputs = []
        for i, input_tensor in enumerate(inputs):
            if i not in tensor_idxs:
                current_masks.append(None)
                permuted_inputs.append(input_tensor)
                continue
            tensor_mask = []
            permuted_input = input_tensor.clone()
            for j, feature_idx in enumerate(feature_idxs):
                original_input_size = (
                    input_tensor.shape[0] // current_num_ablated_features
                )
                start_idx = j * original_input_size
                end_idx = (j + 1) * original_input_size

                mask = (input_mask[i] == feature_idx).to(input_tensor.device).bool()
                if mask.ndim == 0:
                    mask = mask.reshape((1,) * input_tensor.dim())
                tensor_mask.append(mask)
                permuted_input[start_idx:end_idx] = self.perm_func(
                    input_tensor[start_idx:end_idx], mask
                )
            current_masks.append(torch.stack(tensor_mask, dim=0))
            permuted_inputs.append(permuted_input)

        return tuple(permuted_inputs), tuple(current_masks)
