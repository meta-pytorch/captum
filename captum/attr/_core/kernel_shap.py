#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Any, Callable, cast, Generator, List, Optional, Tuple, Union

import torch
from captum._utils.common import _reduce_list, _run_forward
from captum._utils.models.linear_model import SkLearnLinearRegression
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.lime import construct_feature_mask, Lime, LimeBase
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import _format_input_baseline
from captum.log import log_usage
from torch import Tensor
from torch.distributions.categorical import Categorical


class KernelShap(Lime):
    r"""
    Kernel SHAP is a method that uses the LIME framework to compute
    Shapley Values. Setting the loss function, weighting kernel and
    regularization terms appropriately in the LIME framework allows
    theoretically obtaining Shapley Values more efficiently than
    directly computing Shapley Values.

    More information regarding this method and proof of equivalence
    can be found in the original paper here:
    https://arxiv.org/abs/1705.07874

    In Captum, missing features are represented by the values provided in
    `baselines`. This means Captum's KernelShap explains the model output
    relative to the chosen baseline values, rather than sampling missing
    features from a background distribution. `BaselineKernelShap` is an alias
    for this class that makes these baseline-substitution semantics explicit.
    """

    def __init__(self, forward_func: Callable[..., Union[int, float, Tensor]]) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        Lime.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=self.kernel_shap_perturb_generator,
        )
        self.inf_weight = 1000000.0

    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above,
        training an interpretable model based on KernelSHAP and returning a
        representation of the interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME / KernelShap
        is generally used for sample-based interpretability, training a separate
        interpretable model to explain a model's prediction on each individual example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. Note that provided similarity and perturbation functions will be
        provided each example separately (first dimension = 1) in this case.
        If forward_fn returns a scalar per batch (e.g. loss), attributions will
        still be computed using a single interpretable model for the full batch.
        In this case, similarity and perturbation functions will be provided the
        same original input containing the full batch.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which KernelShap
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0. Since missing features are replaced by
                        these values, the returned attributions explain the
                        model relative to the provided baselines rather than a
                        background data distribution.
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
                        which surrogate model is trained
                        (for classification cases,
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
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same
                        interpretable feature. feature_mask
                        should contain the same number of tensors as inputs.
                        Each tensor should
                        be the same size as the corresponding input or
                        broadcastable to match the input tensor. Values across
                        all tensors should be integers in the range 0 to
                        num_interp_features - 1, and indices corresponding to the
                        same feature should have the same value.
                        Note that features are grouped across tensors
                        (unlike feature ablation and occlusion), so
                        if the same index is used in different tensors, those
                        features are still grouped and added simultaneously.
                        If None, then a feature mask is constructed which assigns
                        each scalar within a tensor as a separate feature.
                        Default: None
            n_samples (int, optional): The number of samples of the original
                        model used to train the surrogate interpretable model.
                        Default: `50` if `n_samples` is not provided.
            perturbations_per_eval (int, optional): Allows multiple samples
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
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each interp-
                        retable feature from the trained surrogate model, or
                        whether the returned attributions match the input shape.
                        When return_input_shape is True, the return type of attribute
                        matches the input shape, with each element containing the
                        coefficient of the corresponding interpretable feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions. If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpretable model, with length
                        num_interp_features.
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The attributions with respect to each input feature.
                        If return_input_shape = True, attributions will be
                        the same size as the provided inputs, with each value
                        providing the coefficient of the corresponding
                        interpretale feature.
                        If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
                        num_interp_features.
        Examples::
            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4
            >>> input = torch.randn(1, 4, 4)

            >>> # Defining KernelShap interpreter
            >>> ks = KernelShap(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = ks.attribute(input, target=1, n_samples=200)

            >>> # Alternatively, we can group each 2x2 square of the inputs
            >>> # as one 'interpretable' feature and perturb them together.
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
            >>> # With this mask, all inputs with the same value are set to their
            >>> # baseline value, when the corresponding binary interpretable
            >>> # feature is set to 0.
            >>> # The attributions can be calculated as follows:
            >>> # feature mask has dimensions 1 x 4 x 4
            >>> feature_mask = torch.tensor([[[0,0,1,1],[0,0,1,1],
            >>>                             [2,2,3,3],[2,2,3,3]]])

            >>> # Computes KernelSHAP attributions with feature mask.
            >>> attr = ks.attribute(input, target=1, feature_mask=feature_mask)
        """
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )
        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = torch.tensor((num_interp_features - 1)) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            show_progress=show_progress,
        )

    def attribute_future(self) -> None:
        r"""
        This method is not implemented for KernelShap.
        """
        raise NotImplementedError("attribute_future is not implemented for KernelShap")

    def kernel_shap_similarity_kernel(
        self,
        _,
        __,
        interpretable_sample: Tensor,
        **kwargs: object,
    ) -> Tensor:
        assert (
            "num_interp_features" in kwargs
        ), "Must provide num_interp_features to use default similarity kernel"
        num_selected_features = int(interpretable_sample.sum(dim=1).item())
        num_features = kwargs["num_interp_features"]
        if num_selected_features == 0 or num_selected_features == num_features:
            # weight should be theoretically infinite when
            # num_selected_features = 0 or num_features
            # enforcing that trained linear model must satisfy
            # end-point criteria. In practice, it is sufficient to
            # make this weight substantially larger so setting this
            # weight to 1000000 (all other weights are 1).
            similarities = self.inf_weight
        else:
            similarities = 1.0
        return torch.tensor([similarities])

    def kernel_shap_perturb_generator(
        self,
        original_inp: Union[Tensor, Tuple[Tensor, ...]],
        **kwargs: object,
    ) -> Generator[Tensor, None, None]:
        r"""
        Perturbations are sampled by the following process:
         - Choose k (number of selected features), based on the distribution
                p(k) = (M - 1) / (k * (M - k))

            where M is the total number of features in the interpretable space

         - Randomly select a binary vector with k ones, each sample is equally
            likely. This is done by generating a random vector of normal
            values and thresholding based on the top k elements.

         Since there are M choose k vectors with k ones, this weighted sampling
         is equivalent to applying the Shapley kernel for the sample weight,
         defined as:
         k(M, k) = (M - 1) / (k * (M - k) * (M choose k))
        """
        assert (
            "num_select_distribution" in kwargs and "num_interp_features" in kwargs
        ), (
            "num_select_distribution and num_interp_features are necessary"
            " to use kernel_shap_perturb_func"
        )
        if isinstance(original_inp, Tensor):
            device = original_inp.device
        else:
            device = original_inp[0].device
        num_features = cast(int, kwargs["num_interp_features"])
        yield torch.ones(1, num_features, device=device, dtype=torch.long)
        yield torch.zeros(1, num_features, device=device, dtype=torch.long)
        while True:
            num_selected_features = cast(
                Categorical, kwargs["num_select_distribution"]
            ).sample()
            rand_vals = torch.randn(1, num_features)
            threshold = torch.kthvalue(
                rand_vals, num_features - num_selected_features
            ).values.item()
            yield (rand_vals > threshold).to(device=device).long()


BaselineKernelShap = KernelShap


class RandomBaselineKernelShap(KernelShap):
    r"""
    Random Baseline Kernel SHAP is a perturbation-based method that computes
    Random Baseline Shapley (RBShap) values using the Kernel SHAP weighted
    linear surrogate. RBShap averages Baseline Shapley values over a
    distribution of baselines. For a coalition of selected features `S`,
    the value function is:

    .. math::
        v(S) = E_{b \sim D}[f(x_S; b_{\bar{S}})]

    Captum's `KernelShap` / `BaselineKernelShap` uses one fixed baseline value
    to represent missing features. `RandomBaselineKernelShap` instead uses a
    baseline distribution and averages model evaluations over baseline samples
    for each sampled coalition before fitting the same Kernel SHAP surrogate
    model. This corresponds to the Random Baseline Shapley definition described
    in "The Many Shapley Values for Model Explanation":
    https://arxiv.org/abs/1908.08474
    """

    def __init__(self, forward_func: Callable[..., Union[int, float, Tensor]]) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model=SkLearnLinearRegression(),
            similarity_func=self.kernel_shap_similarity_kernel,
            perturb_func=self.kernel_shap_perturb_generator,
            perturb_interpretable_space=True,
            from_interp_rep_transform=self.random_baseline_from_interp_rep_transform,
            to_interp_rep_transform=None,
        )
        self.inf_weight = 1000000.0

    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        feature_mask: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        n_samples: int = 25,
        perturbations_per_eval: int = 1,
        return_input_shape: bool = True,
        show_progress: bool = False,
        n_baseline_samples: Optional[int] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using Random Baseline Kernel SHAP.
        It trains an interpretable weighted linear model using the Kernel SHAP
        sampling distribution and returns the learned coefficients.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME / KernelShap
        is generally used for sample-based interpretability, training a separate
        interpretable model to explain a model's prediction on each individual
        example.

        A batch of inputs can also be provided as inputs, similar to
        other perturbation-based attribution methods. In this case, if forward_fn
        returns a scalar per example, attributions will be computed for each
        example independently, with a separate interpretable model trained for each
        example. The same baseline distribution is used for each example.

        The number of interpretable features is determined from the provided
        feature mask, or if none is provided, from the default feature mask,
        which considers each scalar input as a separate feature. It is
        generally recommended to provide a feature mask which groups features
        into a small number of interpretable features / components (e.g.
        superpixels in images).

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which Random
                        Baseline Kernel SHAP is computed. If forward_func takes
                        a single tensor as input, a single input tensor should be
                        provided. If forward_func takes multiple tensors as input,
                        a tuple of input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor):
                        Baselines define the reference distribution used to
                        represent missing features. Tensor baselines should have
                        first dimension equal to the number of baseline samples,
                        and all remaining dimensions must match the corresponding
                        input tensor's dimensions starting from dimension 1.
                        If multiple input tensors are provided, tensor baselines
                        may either share the same first dimension or have first
                        dimension 1, in which case that baseline is broadcast
                        across baseline samples. Scalar baselines are treated as
                        a degenerate one-sample baseline distribution.
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which surrogate model is trained
                        (for classification cases,
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
                        arguments in inputs. For a tensor, the first dimension
                        of the tensor must correspond to the number of examples.
                        For all other types, the given argument is used for all
                        forward evaluations. Note that attributions are not
                        computed with respect to these arguments.
                        Default: None
            feature_mask (Tensor or tuple[Tensor, ...], optional):
                        feature_mask defines a mask for the input, grouping
                        features which correspond to the same interpretable
                        feature. feature_mask should contain the same number of
                        tensors as inputs. Each tensor should be the same size as
                        the corresponding input or broadcastable to match the
                        input tensor. Values across all tensors should be
                        integers in the range 0 to num_interp_features - 1, and
                        indices corresponding to the same feature should have the
                        same value. Note that features are grouped across
                        tensors, so if the same index is used in different
                        tensors, those features are still grouped and added
                        simultaneously. If None, then a feature mask is
                        constructed which assigns each scalar within a tensor as
                        a separate feature.
                        Default: None
            n_samples (int, optional): The number of sampled coalitions of the
                        original model used to train the surrogate interpretable
                        model.
                        Default: 25
            perturbations_per_eval (int, optional): Allows multiple sampled
                        coalitions to be processed simultaneously in one call to
                        forward_fn. Each forward pass will contain a maximum of
                        perturbations_per_eval * #examples * #baseline_samples
                        samples.
                        Default: 1
            return_input_shape (bool, optional): Determines whether the returned
                        tensor(s) only contain the coefficients for each
                        interpretable feature from the trained surrogate model,
                        or whether the returned attributions match the input
                        shape. When return_input_shape is True, the return type
                        of attribute matches the input shape, with each element
                        containing the coefficient of the corresponding
                        interpretable feature. All elements with the same value
                        in the feature mask will contain the same coefficient in
                        the returned attributions. If return_input_shape is
                        False, a 1D tensor is returned, containing only the
                        coefficients of the trained interpretable model, with
                        length num_interp_features.
                        Default: True
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced
                        features (e.g. time estimation). Otherwise, it will
                        fallback to a simple output of progress.
                        Default: False
            n_baseline_samples (int, optional): The number of baseline samples
                        to draw with replacement from the baseline distribution
                        for each sampled coalition. If None, all provided
                        baselines are used for each coalition, giving the exact
                        average over the finite baseline distribution.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        The attributions with respect to each input feature.
                        If return_input_shape = True, attributions will be the
                        same size as the provided inputs, with each value
                        providing the coefficient of the corresponding
                        interpretable feature. If return_input_shape is False,
                        a 1D tensor is returned, containing only the coefficients
                        of the trained interpretable model, with length
                        num_interp_features.
        Examples::
            >>> # SimpleClassifier takes a single input tensor of size Nx4x4,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()

            >>> # Generating random input with size 1 x 4 x 4 and a
            >>> # distribution of 10 baseline examples.
            >>> input = torch.randn(1, 4, 4)
            >>> baselines = torch.randn(10, 4, 4)

            >>> # Defining Random Baseline Kernel SHAP interpreter
            >>> rbks = RandomBaselineKernelShap(net)
            >>> attr = rbks.attribute(input, baselines, target=1, n_samples=200)
        """
        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )
        num_baselines = self._validate_baseline_distribution(
            formatted_inputs, formatted_baselines
        )
        assert (
            n_baseline_samples is None
            or isinstance(n_baseline_samples, int)
            and n_baseline_samples > 0
        ), (
            "n_baseline_samples must be None or a positive integer, "
            f"received {n_baseline_samples}."
        )
        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if formatted_inputs[0].shape[0] > 1:
            return self._attribute_batch(
                inputs,
                formatted_inputs,
                formatted_baselines,
                target,
                additional_forward_args,
                feature_mask,
                n_samples,
                perturbations_per_eval,
                return_input_shape,
                show_progress,
                n_baseline_samples,
            )

        num_features_list = torch.arange(num_interp_features, dtype=torch.float)
        denom = num_features_list * (num_interp_features - num_features_list)
        probs = torch.tensor((num_interp_features - 1)) / denom
        probs[0] = 0.0
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=formatted_baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            num_select_distribution=Categorical(probs),
            num_baselines=num_baselines,
            n_baseline_samples=n_baseline_samples,
            show_progress=show_progress,
        )

    def _attribute_batch(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        formatted_inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType,
        additional_forward_args: Optional[object],
        feature_mask: Tuple[Tensor, ...],
        n_samples: int,
        perturbations_per_eval: int,
        return_input_shape: bool,
        show_progress: bool,
        n_baseline_samples: Optional[int],
    ) -> TensorOrTupleOfTensorsGeneric:
        bsz = formatted_inputs[0].shape[0]
        test_output = _run_forward(
            self.forward_func, inputs, target, additional_forward_args
        )
        assert isinstance(test_output, Tensor) and torch.numel(test_output) == bsz, (
            "RandomBaselineKernelShap supports batched inputs only when "
            "forward_func returns one scalar per example. For a single scalar "
            "per input batch, compute attributions for one input batch at a time."
        )

        is_inputs_tuple = isinstance(inputs, tuple)
        output_list: List[TensorOrTupleOfTensorsGeneric] = []
        for (
            curr_inps,
            curr_target,
            curr_additional_args,
            curr_feature_mask,
        ) in _batch_example_iterator(
            bsz,
            formatted_inputs,
            target,
            additional_forward_args,
            feature_mask,
        ):
            output_list.append(
                self.attribute(
                    curr_inps if is_inputs_tuple else curr_inps[0],
                    baselines=baselines,
                    target=curr_target,
                    additional_forward_args=curr_additional_args,
                    feature_mask=(
                        curr_feature_mask if is_inputs_tuple else curr_feature_mask[0]
                    ),
                    n_samples=n_samples,
                    perturbations_per_eval=perturbations_per_eval,
                    return_input_shape=return_input_shape,
                    show_progress=show_progress,
                    n_baseline_samples=n_baseline_samples,
                )
            )

        return _reduce_list(output_list)

    def _validate_baseline_distribution(
        self,
        inputs: Tuple[Tensor, ...],
        baselines: Tuple[Union[Tensor, int, float], ...],
    ) -> int:
        assert len(inputs) == len(
            baselines
        ), "Must provide the same number of baseline entries as input tensors."
        baseline_counts: List[int] = []
        for input, baseline in zip(inputs, baselines):
            if isinstance(baseline, Tensor):
                assert input.dim() == baseline.dim(), (
                    "Baseline tensors must have the same number of dimensions "
                    "as their corresponding input tensor. Found baseline: "
                    f"{baseline.shape} and input: {input.shape}."
                )
                assert input.shape[1:] == baseline.shape[1:], (
                    "Baseline tensor dimensions after the first dimension must "
                    "match the corresponding input tensor. Found baseline: "
                    f"{baseline.shape} and input: {input.shape}."
                )
                baseline_counts.append(baseline.shape[0])

        num_baselines = max(baseline_counts) if baseline_counts else 1
        for count in baseline_counts:
            assert count in (1, num_baselines), (
                "All baseline tensors must have the same first dimension, or "
                "first dimension 1 so they can be broadcast across baseline "
                f"samples. Found counts: {baseline_counts}."
            )
        return num_baselines

    def _get_model_input_multiplier(self, **kwargs: Any) -> int:
        n_baseline_samples = cast(Optional[int], kwargs["n_baseline_samples"])
        if n_baseline_samples is not None:
            return n_baseline_samples
        return cast(int, kwargs["num_baselines"])

    def _aggregate_model_outputs(
        self,
        model_out: Tensor,
        num_interp_inputs: int,
        model_input_multiplier: int,
        **kwargs: Any,
    ) -> Tensor:
        return model_out.view(num_interp_inputs, model_input_multiplier).mean(dim=1)

    def random_baseline_from_interp_rep_transform(
        self,
        curr_sample: Tensor,
        original_inputs: TensorOrTupleOfTensorsGeneric,
        **kwargs: Any,
    ) -> TensorOrTupleOfTensorsGeneric:
        assert (
            "feature_mask" in kwargs
        ), "Must provide feature_mask to use Random Baseline Kernel SHAP"
        assert (
            "baselines" in kwargs
        ), "Must provide baselines to use Random Baseline Kernel SHAP"
        if isinstance(original_inputs, Tensor):
            baseline_indices = self._sample_baseline_indices(original_inputs, **kwargs)
            return self._replace_missing_with_random_baselines(
                curr_sample,
                original_inputs,
                cast(Tensor, kwargs["feature_mask"]),
                cast(Union[Tensor, int, float], kwargs["baselines"]),
                baseline_indices,
            )

        baseline_indices = self._sample_baseline_indices(original_inputs[0], **kwargs)
        baselines = cast(Tuple[Union[Tensor, int, float], ...], kwargs["baselines"])
        feature_mask = cast(Tuple[Tensor, ...], kwargs["feature_mask"])
        return cast(
            TensorOrTupleOfTensorsGeneric,
            tuple(
                self._replace_missing_with_random_baselines(
                    curr_sample,
                    original_inputs[i],
                    feature_mask[i],
                    baselines[i],
                    baseline_indices,
                )
                for i in range(len(feature_mask))
            ),
        )

    def _sample_baseline_indices(
        self,
        input: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        num_baselines = cast(int, kwargs["num_baselines"])
        n_baseline_samples = cast(Optional[int], kwargs["n_baseline_samples"])
        if n_baseline_samples is None:
            return torch.arange(num_baselines, device=input.device)
        return torch.randint(num_baselines, (n_baseline_samples,), device=input.device)

    def _replace_missing_with_random_baselines(
        self,
        curr_sample: Tensor,
        original_input: Tensor,
        feature_mask: Tensor,
        baseline: Union[Tensor, int, float],
        baseline_indices: Tensor,
    ) -> Tensor:
        assert (
            original_input.shape[0] == 1
        ), "RandomBaselineKernelShap expects one input example at a time."
        sample_count = baseline_indices.numel()
        binary_mask = curr_sample[0][feature_mask].bool()
        expanded_input = original_input.expand(
            (sample_count,) + tuple(original_input.shape[1:])
        )
        if isinstance(baseline, Tensor):
            baseline = baseline.to(
                device=original_input.device, dtype=original_input.dtype
            )
            if baseline.shape[0] == 1:
                selected_baselines = baseline.expand(
                    (sample_count,) + tuple(baseline.shape[1:])
                )
            else:
                selected_baselines = baseline.index_select(0, baseline_indices)
        else:
            selected_baselines = torch.full(
                (sample_count,) + tuple(original_input.shape[1:]),
                baseline,
                device=original_input.device,
                dtype=original_input.dtype,
            )
        return torch.where(binary_mask, expanded_input, selected_baselines)
