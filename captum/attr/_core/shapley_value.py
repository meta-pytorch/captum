#!/usr/bin/env python3

# pyre-strict

import itertools
import math
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_feature_mask,
    _format_output,
    _format_tensor_into_tuples,
    _get_max_feature_index,
    _is_mask_valid,
    _is_tuple,
    _run_forward,
)
from captum._utils.exceptions import ShapleyValueFutureError
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.common import (
    _find_output_mode_and_verify,
    _format_input_baseline,
    _tensorize_baseline,
)
from captum.log import log_usage
from torch import dtype, Size, Tensor
from torch.futures import collect_all, Future


def _all_perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for perm in itertools.permutations(range(num_features)):
        yield perm


def _perm_generator(num_features: int, num_samples: int) -> Iterable[Sequence[int]]:
    for _ in range(num_samples):
        yield torch.randperm(num_features).tolist()


def _shape_feature_mask(
    feature_mask: Tuple[Tensor, ...], inputs: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    """
    ensure feature_mask has the same number of dims as the inputs
    i.e., prepend dummy dims of 1 to the masks that broadcastable to inputs
    """
    mask_list = []
    for i, (mask, inp) in enumerate(zip(feature_mask, inputs)):
        assert _is_mask_valid(mask, inp), (
            f"the shape of feature mask (index {i}) is invalid,"
            f"input shape: {inp.shape}, feature mask shape {mask.shape}"
        )
        if mask.dim() < inp.dim():
            mask = mask.reshape((1,) * (inp.dim() - mask.dim()) + tuple(mask.shape))

        mask_list.append(mask)

    return tuple(mask_list)


class ShapleyValueSampling(PerturbationAttribution):
    """
    A perturbation based approach to compute attribution, based on the concept
    of Shapley Values from cooperative game theory. This method involves taking
    a random permutation of the input features and adding them one-by-one to the
    given baseline. The output difference after adding each feature corresponds
    to its attribution, and these difference are averaged when repeating this
    process n_samples times, each time choosing a new random permutation of
    the input features.

    By default, each scalar value within
    the input tensors are taken as a feature and added independently. Passing
    a feature mask, allows grouping features to be added together. This can
    be used in cases such as images, where an entire segment or region
    can be grouped together, measuring the importance of the segment
    (feature group). Each input scalar in the group will be given the same
    attribution value equal to the change in output as a result of adding back
    the entire feature group.

    More details regarding Shapley Value sampling can be found in these papers:
    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    https://pdfs.semanticscholar.org/7715/bb1070691455d1fcfc6346ff458dbca77b2c.pdf
    """

    def __init__(self, forward_func: Callable[..., Union[int, float, Tensor]]) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it. The forward function can either
                        return a scalar per example, or a single scalar for the
                        full batch. If a single scalar is returned for the batch,
                        `perturbations_per_eval` must be 1, and the returned
                        attributions will have first dimension 1, corresponding to
                        feature importance across all examples in the batch.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.permutation_generator = _perm_generator

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

                inputs (Tensor or tuple[Tensor, ...]): Input for which Shapley value
                            sampling attributions are computed. If forward_func takes
                            a single tensor as input, a single input tensor should
                            be provided.
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
                              exactly the same dimensions as inputs or the first
                              dimension is one and the remaining dimensions match
                              with inputs.

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
                            features which should be added together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Values across
                            all tensors should be integers in the range 0 to
                            num_features - 1, and indices corresponding to the same
                            feature should have the same value.
                            Note that features are grouped across tensors, so
                            if the same index is used in different tensors, those
                            features are still grouped and added simultaneously.
                            If the forward function returns a single scalar per batch,
                            we enforce that the first dimension of each mask must be 1,
                            since attributions are returned batch-wise rather than per
                            example, so the attributions must correspond to the
                            same features (indices) in each input example.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature
                            Default: None
                n_samples (int, optional): The number of feature permutations
                            tested.
                            Default: `25` if `n_samples` is not provided.
                perturbations_per_eval (int, optional): Allows multiple ablations
                            to be processed simultaneously in one call to forward_fn.
                            Each forward pass will contain a maximum of
                            perturbations_per_eval * #examples samples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain at most
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
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)
            >>> # Defining ShapleyValueSampling interpreter
            >>> svs = ShapleyValueSampling(net)
            >>> # Computes attribution, taking random orderings
            >>> # of the 16 features and computing the output change when adding
            >>> # each feature. We average over 200 trials (random permutations).
            >>> attr = svs.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we may want to add features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and adding them together.
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
            >>> # With this mask, all inputs with the same value are added
            >>> # together, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])
            >>> attr = svs.attribute(input, target=1, feature_mask=feature_mask)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        formatted_feature_mask = _format_feature_mask(feature_mask, inputs_tuple)
        reshaped_feature_mask = _shape_feature_mask(
            formatted_feature_mask, inputs_tuple
        )

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Ablations per evaluation must be at least 1."

        with torch.no_grad():
            baselines = _tensorize_baseline(inputs_tuple, baselines)
            num_examples = inputs_tuple[0].shape[0]

            total_features = _get_max_feature_index(reshaped_feature_mask) + 1

            if show_progress:
                attr_progress = progress(
                    desc=f"{self.get_name()} attribution",
                    total=self._get_n_evaluations(
                        total_features, n_samples, perturbations_per_eval
                    )
                    + 1,  # add 1 for the initial eval
                )
                attr_progress.update(0)

            initial_eval = self._strict_run_forward(
                self.forward_func, baselines, target, additional_forward_args
            )

            if show_progress:
                attr_progress.update()

            agg_output_mode = _find_output_mode_and_verify(
                initial_eval,
                num_examples,
                perturbations_per_eval,
                reshaped_feature_mask,
                allow_multi_outputs=True,
            )

            # Initialize attribution totals and counts
            output_shape = initial_eval.shape

            # attr shape (*output_shape, *input_feature_shape)
            total_attrib = [
                torch.zeros(
                    tuple(output_shape) + tuple(input.shape[1:]),
                    dtype=torch.float,
                    device=inputs_tuple[0].device,
                )
                for input in inputs_tuple
            ]

            iter_count = 0
            # Iterate for number of samples, generate a permutation of the features
            # and evalute the incremental increase for each feature.
            for feature_permutation in self.permutation_generator(
                total_features, n_samples
            ):
                iter_count += 1
                prev_results = initial_eval
                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_masks,
                    current_feat_list,
                ) in self._perturbation_generator(
                    inputs_tuple,
                    additional_forward_args,
                    target,
                    baselines,
                    reshaped_feature_mask,
                    feature_permutation,
                    perturbations_per_eval,
                ):
                    if sum(torch.sum(mask).item() for mask in current_masks) == 0:
                        warnings.warn(
                            "Feature mask is missing some integers between 0 and "
                            "num_features, for optimal performance, make sure each"
                            " consecutive integer corresponds to a feature.",
                            stacklevel=1,
                        )
                    # modified_eval dimensions: 1D tensor with length
                    # equal to #num_examples * #features in batch
                    modified_eval = self._strict_run_forward(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )
                    if show_progress:
                        attr_progress.update()
                    if agg_output_mode:
                        eval_diff = (modified_eval - prev_results).to(
                            inputs_tuple[0].device
                        )
                        prev_results = modified_eval
                    else:
                        # when perturb_per_eval > 1, every num_examples stands for
                        # one perturb. Since the perturbs are from a consecutive
                        # perumuation, each diff of a perturb is its eval minus
                        # the eval of the previous perturb
                        all_eval = torch.cat((prev_results, modified_eval), dim=0)
                        eval_diff = (
                            all_eval[num_examples:] - all_eval[:-num_examples]
                        ).to(inputs_tuple[0].device)
                        prev_results = all_eval[-num_examples:]
                    for j in current_feat_list:
                        # format eval_diff to shape
                        # (n_perturb, *output_shape, 1,.. 1)
                        # where n_perturb may not be perturb_per_eval
                        # Append n_input_feature dim of 1 to make the tensor
                        # have the same dim as the mask tensor.
                        formatted_eval_diff = eval_diff.reshape(
                            (-1,)
                            + tuple(output_shape)
                            + (len(inputs_tuple[j].shape) - 1) * (1,)
                        )

                        # mask in shape (n_perturb, *mask_shape_broadcastable_to_input)
                        # reshape to
                        # (
                        #     n_perturb,
                        #     *broadcastable_to_output_shape
                        #     *broadcastable_to_input_feature_shape
                        # )
                        cur_mask = current_masks[j]
                        cur_mask = cur_mask.reshape(
                            tuple(cur_mask.shape[:2])
                            + (len(output_shape) - 1) * (1,)
                            + tuple(cur_mask.shape[2:])
                        )

                        # aggregate n_perturb
                        cur_attr = (formatted_eval_diff * cur_mask.float()).sum(dim=0)

                        # (*output_shape, *input_feature_shape)
                        total_attrib[j] += cur_attr
            if show_progress:
                attr_progress.close()

            # Divide total attributions by number of random permutations and return
            # formatted attributions.
            attrib = tuple(
                tensor_attrib_total / iter_count for tensor_attrib_total in total_attrib
            )
            formatted_attr = _format_output(is_inputs_tuple, attrib)
        return cast(TensorOrTupleOfTensorsGeneric, formatted_attr)

    def attribute_future(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> Future[TensorOrTupleOfTensorsGeneric]:
        r"""
        This method is not implemented for ShapleyValueSampling.
        """
        is_inputs_tuple = _is_tuple(inputs)
        inputs_tuple, baselines = _format_input_baseline(inputs, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        formatted_feature_mask = _format_feature_mask(feature_mask, inputs_tuple)
        reshaped_feature_mask = _shape_feature_mask(
            formatted_feature_mask, inputs_tuple
        )

        assert (
            isinstance(perturbations_per_eval, int) and perturbations_per_eval >= 1
        ), "Ablations per evaluation must be at least 1."

        with torch.no_grad():
            baselines = _tensorize_baseline(inputs_tuple, baselines)
            num_examples = inputs_tuple[0].shape[0]

            total_features = _get_max_feature_index(reshaped_feature_mask) + 1

            if show_progress:
                attr_progress = progress(
                    desc=f"{self.get_name()} attribution",
                    total=self._get_n_evaluations(
                        total_features, n_samples, perturbations_per_eval
                    )
                    + 1,  # add 1 for the initial eval
                )
                attr_progress.update(0)

            initial_eval: Future[Tensor] = self._strict_run_forward_future(
                self.forward_func, baselines, target, additional_forward_args
            )

            if show_progress:
                attr_progress.update()

            prev_result_tuple: Future[
                Tuple[Tensor, Tensor, Size, List[Tensor], bool]
            ] = initial_eval.then(
                lambda inp=initial_eval: self._initial_eval_to_prev_results_tuple(  # type: ignore # noqa: E501 line too long
                    inp,
                    num_examples,
                    perturbations_per_eval,
                    reshaped_feature_mask,
                    inputs_tuple,
                )
            )

            iter_count = 0
            # Iterate for number of samples, generate a permutation of the features
            # and evalute the incremental increase for each feature.
            for feature_permutation in self.permutation_generator(
                total_features, n_samples
            ):
                prev_result_tuple = prev_result_tuple.then(
                    lambda inp=prev_result_tuple: self._set_prev_results_to_initial_eval(inp)  # type: ignore # noqa: E501 line too long
                )

                iter_count += 1
                for (
                    current_inputs,
                    current_add_args,
                    current_target,
                    current_masks,
                    _,
                ) in self._perturbation_generator(
                    inputs_tuple,
                    additional_forward_args,
                    target,
                    baselines,
                    reshaped_feature_mask,
                    feature_permutation,
                    perturbations_per_eval,
                ):
                    if sum(torch.sum(mask).item() for mask in current_masks) == 0:
                        warnings.warn(
                            "Feature mask is missing some integers between 0 and "
                            "num_features, for optimal performance, make sure each"
                            " consecutive integer corresponds to a feature.",
                            stacklevel=1,
                        )
                    # modified_eval dimensions: 1D tensor with length
                    # equal to #num_examples * #features in batch
                    modified_eval = self._strict_run_forward_future(
                        self.forward_func,
                        current_inputs,
                        current_target,
                        current_add_args,
                    )
                    if show_progress:
                        attr_progress.update()

                    assert isinstance(modified_eval, torch.Future), (
                        "when using futures method, modified_eval should have "
                        f"Future type rather than {type(modified_eval)}"
                    )
                    eval_futs: Future[
                        List[
                            Future[
                                Union[
                                    Tuple[Tensor, Tensor, Size, List[Tensor], bool],
                                    Tensor,
                                ]
                            ]
                        ]
                    ] = collect_all([prev_result_tuple, modified_eval])

                    prev_result_tuple = eval_futs.then(
                        lambda evals=eval_futs, masks=current_masks: self._eval_fut_to_prev_results_tuple(  # type: ignore # noqa: E501 line too long
                            evals, num_examples, inputs_tuple, masks
                        )
                    )

            if show_progress:
                attr_progress.close()

            # Divide total attributions by number of random permutations and return
            # formatted attributions.
            formatted_attr: Future[Union[Tensor, tuple[Tensor, ...]]] = (
                prev_result_tuple.then(
                    lambda inp=prev_result_tuple: self._prev_result_tuple_to_formatted_attr(  # type: ignore # noqa: E501 line too long
                        inp, iter_count, is_inputs_tuple
                    )
                )
            )
        return cast(Future[TensorOrTupleOfTensorsGeneric], formatted_attr)

    def _initial_eval_to_prev_results_tuple(
        self,
        initial_eval: Future[Tensor],
        num_examples: int,
        perturbations_per_eval: int,
        reshaped_feature_mask: TensorOrTupleOfTensorsGeneric,
        inputs_tuple: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor, Size, List[Tensor], bool]:
        """Since the initial eval is a Future, it is easier to bundle the prev_result,
        agg_output_mode, output_shape, and total_attrib together
        as Shapley Value Feature Attributions are being calculated"""
        try:
            initial_eval_processed = initial_eval.value()
            prev_result = initial_eval_processed
            if not isinstance(initial_eval_processed, Tensor):
                raise AssertionError(
                    "initial_eval_to_processed_initial_eval_fut: "
                    "initial_eval should be a Tensor"
                )
            agg_output_mode = _find_output_mode_and_verify(
                initial_eval_processed,
                num_examples,
                perturbations_per_eval,
                reshaped_feature_mask,
                allow_multi_outputs=True,
            )
            output_shape = initial_eval_processed.shape
            total_attrib: List[Tensor] = [
                torch.zeros(
                    tuple(output_shape) + tuple(input.shape[1:]),
                    dtype=torch.float,
                    device=inputs_tuple[0].device,
                )
                for input in inputs_tuple
            ]
            result = (
                initial_eval_processed,
                prev_result,
                output_shape,
                total_attrib,
                agg_output_mode,
            )
        except ShapleyValueFutureError as e:
            raise ShapleyValueFutureError(
                "_initial_eval_to_prev_results_tuple func failed"
            ) from e
        return result

    def _set_prev_results_to_initial_eval(
        self,
        processed_initial_eval: Future[Tuple[Tensor, Tensor, Size, List[Tensor], bool]],
    ) -> Tuple[Tensor, Tensor, Size, List[Tensor], bool]:
        """At the beginning of each feature permutation, the prev_results is
        reset to the initial eval, and this method helps set that up"""
        (initial_eval, prev_results, output_shape, total_attrib, agg_output_mode) = (
            processed_initial_eval.value()
        )
        prev_results = initial_eval
        return (initial_eval, prev_results, output_shape, total_attrib, agg_output_mode)

    def _eval_fut_to_prev_results_tuple(
        self,
        eval_futs: Future[
            List[
                Union[
                    Future[Tuple[Tensor, Tensor, Size, List[Tensor], bool]],
                    Future[Tensor],
                ]
            ]
        ],
        num_examples: int,
        inputs_tuple: Tuple[Tensor, ...],
        current_masks: Tuple[Tensor, ...],
    ) -> Tuple[Tensor, Tensor, Size, List[Tensor], bool]:
        """Helper method responsible for calculating
        eval differences between the modified eval and prev_results
        Tensor and storing them in total_attrib. Returns prev_results_tuple
        with modified total_attrib and prev_results"""
        prev_results_tuple = eval_futs.value()[0].value()
        modified_eval = eval_futs.value()[1].value()
        if not isinstance(modified_eval, Tensor) or not isinstance(
            prev_results_tuple, tuple
        ):
            raise ShapleyValueFutureError(
                "_eval_fut_to_prev_results_tuple func failed due to type mismatch"
            )
        (
            initial_eval,
            prev_results,
            output_shape,
            total_attrib,
            agg_output_mode,
        ) = prev_results_tuple
        if agg_output_mode:
            eval_diff = (modified_eval - prev_results).to(inputs_tuple[0].device)
            prev_results = modified_eval
        else:
            # when perturb_per_eval > 1, every num_examples stands for
            # one perturb. Since the perturbs are from a consecutive
            # perumuation, each diff of a perturb is its eval minus
            # the eval of the previous perturb

            all_eval = torch.cat((prev_results, modified_eval), dim=0)
            eval_diff = (all_eval[num_examples:] - all_eval[:-num_examples]).to(
                inputs_tuple[0].device
            )
            prev_results = all_eval[-num_examples:]

        for j in range(len(total_attrib)):
            # format eval_diff to shape
            # (n_perturb, *output_shape, 1,.. 1)
            # where n_perturb may not be perturb_per_eval
            # Append n_input_feature dim of 1 to make the tensor
            # have the same dim as the mask tensor.
            formatted_eval_diff = eval_diff.reshape(
                (-1,) + tuple(output_shape) + (len(inputs_tuple[j].shape) - 1) * (1,)
            )

            # mask in shape (n_perturb, *mask_shape_broadcastable_to_input)
            # reshape to
            # (
            #     n_perturb,
            #     *broadcastable_to_output_shape
            #     *broadcastable_to_input_feature_shape
            # )
            cur_mask = current_masks[j]
            cur_mask = cur_mask.reshape(
                tuple(cur_mask.shape[:2])
                + (len(output_shape) - 1) * (1,)
                + tuple(cur_mask.shape[2:])
            )

            # aggregate n_perturb
            cur_attr = (formatted_eval_diff * cur_mask.float()).sum(dim=0)
            # (*output_shape, *input_feature_shape)
            total_attrib[j] += cur_attr

        result = (
            initial_eval,
            prev_results,
            output_shape,
            total_attrib,
            agg_output_mode,
        )
        return result

    def _prev_result_tuple_to_formatted_attr(
        self,
        prev_result_tuple: Future[
            Tuple[Tensor, Tensor, Tuple[int], List[Tensor], bool]
        ],
        iter_count: int,
        is_inputs_tuple: bool,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Helper method to format total_attrib, which is a
        list of tensors, into formatted attributions, which
        are either a single tensor or a tuple of tensors"""

        (
            _,
            _,
            _,
            total_attrib,
            _,
        ) = prev_result_tuple.value()
        attrib = tuple(
            tensor_attrib_total / iter_count for tensor_attrib_total in total_attrib
        )
        formatted_attr = _format_output(is_inputs_tuple, attrib)
        return formatted_attr

    def _update_current_tensors(
        self,
        current_tensors: Tuple[Tensor, ...],
        input_tensors: Tuple[Tensor, ...],
        feature_index: int,
        mask: Tuple[Tensor, ...],
        feat_tensor_index_map: Dict[int, List[int]],
    ) -> Tuple[Tensor, ...]:
        feat_list = feat_tensor_index_map[feature_index]
        output_tensors = []
        for i in range(len(current_tensors)):
            if i in feat_list:
                output_tensors.append(
                    current_tensors[i]
                    * (~(mask[i] == feature_index)).to(current_tensors[i].dtype)
                    + input_tensors[i]
                    * (mask[i] == feature_index).to(input_tensors[i].dtype)
                )

            else:
                output_tensors.append(current_tensors[i])
        return tuple(output_tensors)

    def _construct_selected_mask(
        self,
        feature_index: int,
        mask: Tuple[Tensor, ...],
        empty_mask: Tuple[Tensor, ...],
        feat_tensor_index_map: Dict[int, List[int]],
        device: torch.device,
    ) -> Tuple[Tensor, ...]:
        feat_list = feat_tensor_index_map[feature_index]
        output_mask = []
        for i in range(len(mask)):
            if i in feat_list:
                output_mask.append((mask[i] == feature_index).to(device).unsqueeze(0))
            else:
                output_mask.append(empty_mask[i])
        return tuple(output_mask)

    def _perturbation_generator(
        self,
        inputs: Tuple[Tensor, ...],
        additional_args: Optional[Tuple[object, ...]],
        target: TargetType,
        baselines: Tuple[Tensor, ...],
        input_masks: Tuple[Tensor, ...],
        feature_permutation: Sequence[int],
        perturbations_per_eval: int,
    ) -> Iterable[
        Tuple[Tuple[Tensor, ...], object, TargetType, Tuple[Tensor, ...], Set[int]]
    ]:
        """
        This method is a generator which yields each perturbation to be evaluated
        including inputs, additional_forward_args, targets, and mask.
        """
        # current_tensors starts at baselines and includes each additional feature as
        # added based on the permutation order.
        current_tensors = baselines
        current_tensors_list = []
        current_mask_list = []
        current_feat_list = set()

        # Compute repeated additional args and targets
        additional_args_repeated = (
            _expand_additional_forward_args(additional_args, perturbations_per_eval)
            if additional_args is not None
            else None
        )
        feat_tensor_index_map = defaultdict(list)
        for i in range(len(input_masks)):
            for elem in input_masks[i].view(-1):
                feat_tensor_index_map[elem.item()].append(i)
        empty_masks = tuple(torch.zeros_like(elem).unsqueeze(0) for elem in input_masks)

        target_repeated = _expand_target(target, perturbations_per_eval)
        for i in range(len(feature_permutation)):
            current_tensors = self._update_current_tensors(
                current_tensors=current_tensors,
                input_tensors=inputs,
                feature_index=feature_permutation[i],
                mask=input_masks,
                feat_tensor_index_map=feat_tensor_index_map,
            )
            current_tensors_list.append(current_tensors)
            current_mask_list.append(
                self._construct_selected_mask(
                    feature_index=feature_permutation[i],
                    mask=input_masks,
                    empty_mask=empty_masks,
                    feat_tensor_index_map=feat_tensor_index_map,
                    device=inputs[0].device,
                )
            )
            current_feat_list.update(feat_tensor_index_map[feature_permutation[i]])

            if len(current_tensors_list) == perturbations_per_eval:
                if len(current_tensors_list) > 1:
                    combined_inputs = tuple(
                        torch.cat(aligned_tensors, dim=0)
                        for aligned_tensors in zip(*current_tensors_list)
                    )
                    combined_masks = tuple(
                        torch.cat(aligned_masks, dim=0)
                        for aligned_masks in zip(*current_mask_list)
                    )
                else:
                    combined_inputs = current_tensors_list[0]
                    combined_masks = current_mask_list[0]
                yield (
                    combined_inputs,
                    additional_args_repeated,
                    target_repeated,
                    combined_masks,
                    current_feat_list,
                )
                current_tensors_list = []
                current_mask_list = []
                current_feat_list = set()

        # Create batch with remaining evaluations, may not be a complete batch
        # (= perturbations_per_eval)
        if len(current_tensors_list) != 0:
            additional_args_repeated = (
                _expand_additional_forward_args(
                    additional_args, len(current_tensors_list)
                )
                if additional_args is not None
                else None
            )
            target_repeated = _expand_target(target, len(current_tensors_list))
            combined_inputs = tuple(
                torch.cat(aligned_tensors, dim=0)
                for aligned_tensors in zip(*current_tensors_list)
            )
            combined_masks = tuple(
                torch.cat(aligned_masks, dim=0)
                for aligned_masks in zip(*current_mask_list)
            )
            yield (
                combined_inputs,
                additional_args_repeated,
                target_repeated,
                combined_masks,
                current_feat_list,
            )

    def _get_n_evaluations(
        self, total_features: int, n_samples: int, perturbations_per_eval: int
    ) -> int:
        """return the total number of forward evaluations needed"""
        return math.ceil(total_features / perturbations_per_eval) * n_samples

    def _strict_run_forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """
        A temp wrapper for global _run_forward util to force forward output
        type assertion & conversion.
        Remove after the strict logic is supported by all attr classes
        """
        forward_output = _run_forward(*args, **kwargs)
        if isinstance(forward_output, Tensor):
            # format scalar to shape (1) so we can always assume non-empty output_shape
            if not forward_output.shape:
                forward_output = forward_output.reshape(1)

            return forward_output

        output_type = type(forward_output)
        assert output_type is int or output_type is float, (
            "the return of forward_func must be a tensor, int, or float,"
            f" received: {forward_output}"
        )

        # using python built-in type as torch dtype
        # int -> torch.int64, float -> torch.float64
        # ref: https://github.com/pytorch/pytorch/pull/21215
        return torch.tensor([forward_output], dtype=cast(dtype, output_type))

    def _strict_run_forward_future(self, *args: Any, **kwargs: Any) -> Future[Tensor]:
        """
        A temp wrapper for global _run_forward util to force
        forward outputtype assertion & conversion, but takes
        into account the Future tensor type
        """

        def process_strict_run_forward(fut: Future[Tensor]) -> Tensor:
            output = fut.value()
            if isinstance(output, Tensor):
                # format scalar to shape (1) so we can always
                # assume non-empty output_shape
                if not output.shape:
                    output = output.reshape(1)
                return output
            output_type = type(output)
            assert output_type is int or output_type is float, (
                "the return of forward_func must be a Future of tensor, int, or float,"
                f" received: {output_type}"
            )
            output = torch.tensor([output], dtype=cast(dtype, output_type))
            return output

        forward_output = _run_forward(*args, **kwargs)
        assert isinstance(forward_output, torch.Future), (
            "The return type of forward_func must be a Future"
            f" received: {type(forward_output)}"
        )

        return_output = forward_output.then(process_strict_run_forward)
        return return_output


class ShapleyValues(ShapleyValueSampling):
    """
    A perturbation based approach to compute attribution, based on the concept
    of Shapley Values from cooperative game theory. This method involves taking
    each permutation of the input features and adding them one-by-one to the
    given baseline. The output difference after adding each feature corresponds
    to its attribution, and these difference are averaged over all possible
    random permutations of the input features.

    By default, each scalar value within
    the input tensors are taken as a feature and added independently. Passing
    a feature mask, allows grouping features to be added together. This can
    be used in cases such as images, where an entire segment or region
    can be grouped together, measuring the importance of the segment
    (feature group). Each input scalar in the group will be given the same
    attribution value equal to the change in output as a result of adding back
    the entire feature group.

    More details regarding Shapley Values can be found in these papers:
    https://apps.dtic.mil/dtic/tr/fulltext/u2/604084.pdf
    https://www.sciencedirect.com/science/article/pii/S0305054808000804
    https://pdfs.semanticscholar.org/7715/bb1070691455d1fcfc6346ff458dbca77b2c.pdf

    NOTE: The method implemented here is very computationally intensive, and
    should only be used with a very small number of features (e.g. < 7).
    This implementation simply extends ShapleyValueSampling and
    evaluates all permutations, leading to a total of n * n! evaluations for n
    features. Shapley values can alternatively be computed with only 2^n
    evaluations, and we plan to add this approach in the future.
    """

    def __init__(self, forward_func: Callable[..., Union[int, float, Tensor]]) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it. The forward function can either
                        return a scalar per example, or a single scalar for the
                        full batch. If a single scalar is returned for the batch,
                        `perturbations_per_eval` must be 1, and the returned
                        attributions will have first dimension 1, corresponding to
                        feature importance across all examples in the batch.
        """
        ShapleyValueSampling.__init__(self, forward_func)
        self.permutation_generator = _all_perm_generator

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        NOTE: The feature_mask argument differs from other perturbation based
        methods, since feature indices can overlap across tensors. See the
        description of the feature_mask argument below for more details.

        Args:

                inputs (Tensor or tuple[Tensor, ...]): Input for which Shapley value
                            sampling attributions are computed. If forward_func takes
                            a single tensor as input, a single input tensor should
                            be provided.
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
                              exactly the same dimensions as inputs or the first
                              dimension is one and the remaining dimensions match
                              with inputs.

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
                            features which should be added together. feature_mask
                            should contain the same number of tensors as inputs.
                            Each tensor should
                            be the same size as the corresponding input or
                            broadcastable to match the input tensor. Values across
                            all tensors should be integers in the range 0 to
                            num_features - 1, and indices corresponding to the same
                            feature should have the same value.
                            Note that features are grouped across tensors
                            (unlike feature ablation and occlusion), so
                            if the same index is used in different tensors, those
                            features are still grouped and added simultaneously.
                            If the forward function returns a single scalar per batch,
                            we enforce that the first dimension of each mask must be 1,
                            since attributions are returned batch-wise rather than per
                            example, so the attributions must correspond to the
                            same features (indices) in each input example.
                            If None, then a feature mask is constructed which assigns
                            each scalar within a tensor as a separate feature
                            Default: None
                perturbations_per_eval (int, optional): Allows multiple ablations
                            to be processed simultaneously in one call to forward_fn.
                            Each forward pass will contain a maximum of
                            perturbations_per_eval * #examples samples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain at most
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
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.


        Examples::

            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>> # Generating random input with size 2 x 4 x 4
            >>> input = torch.randn(2, 4, 4)

            >>> # We may want to add features in groups, e.g.
            >>> # grouping each 2x2 square of the inputs and adding them together.
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
            >>> # With this mask, all inputs with the same value are added
            >>> # together, and the attribution for each input in the same
            >>> # group (0, 1, 2, and 3) per example are the same.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # With only 4 features, it is feasible to compute exact
            >>> # Shapley Values. These can be computed as follows:
            >>> sv = ShapleyValues(net)
            >>> attr = sv.attribute(input, target=1, feature_mask=feature_mask)
        """
        if feature_mask is None:
            total_features = sum(
                torch.numel(inp[0]) for inp in _format_tensor_into_tuples(inputs)
            )
        else:
            total_features = (
                int(max(torch.max(single_mask).item() for single_mask in feature_mask))
                + 1
            )

        if total_features >= 10:
            warnings.warn(
                "You are attempting to compute Shapley Values with at least 10 "
                "features, which will likely be very computationally expensive."
                "Consider using Shapley Value Sampling instead.",
                stacklevel=1,
            )

        return super().attribute.__wrapped__(
            self,
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            perturbations_per_eval=perturbations_per_eval,
            show_progress=show_progress,
        )

    def _get_n_evaluations(
        self, total_features: int, n_samples: int, perturbations_per_eval: int
    ) -> int:
        """return the total number of forward evaluations needed"""
        return math.ceil(total_features / perturbations_per_eval) * math.factorial(
            total_features
        )
