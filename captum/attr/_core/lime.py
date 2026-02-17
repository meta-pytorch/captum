#!/usr/bin/env python3

# pyre-strict
import inspect
import math
import typing
import warnings
from collections.abc import Iterator
from typing import Any, Callable, cast, List, Literal, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _flatten_tensor_or_tuple,
    _format_output,
    _format_tensor_into_tuples,
    _get_max_feature_index,
    _is_tuple,
    _reduce_list,
    _run_forward,
)
from captum._utils.models.linear_model import SkLearnLasso
from captum._utils.models.model import Model
from captum._utils.progress import progress
from captum._utils.typing import BaselineType, TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _format_input_baseline,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import CosineSimilarity
from torch.utils.data import DataLoader, TensorDataset


class LimeBase(PerturbationAttribution):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    LimeBase provides a generic framework to train a surrogate interpretable model.
    This differs from most other attribution methods, since the method returns a
    representation of the interpretable model (e.g. coefficients of the linear model).
    For a similar interface to other perturbation-based attribution methods, please use
    the Lime child class, which defines specific transformations for the interpretable
    model.

    LimeBase allows sampling points in either the interpretable space or the original
    input space to train the surrogate model. The interpretable space is a feature
    vector used to train the surrogate interpretable model; this feature space is often
    of smaller dimensionality than the original feature space in order for the surrogate
    model to be more interpretable.

    If sampling in the interpretable space, a transformation function must be provided
    to define how a vector sampled in the interpretable space can be transformed into
    an example in the original input space. If sampling in the original input space, a
    transformation function must be provided to define how the input can be transformed
    into its interpretable vector representation.

    More details regarding LIME can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(
        self,
        forward_func: Callable[..., Union[int, float, Tensor]],
        interpretable_model: Model,
        similarity_func: Callable[
            ...,
            Union[float, Tensor],
        ],
        perturb_func: Callable[..., object],
        perturb_interpretable_space: bool,
        from_interp_rep_transform: Optional[
            Callable[..., Union[Tensor, Tuple[Tensor, ...]]]
        ],
        to_interp_rep_transform: Optional[Callable[..., Tensor]],
    ) -> None:
        r"""

        Args:


            forward_func (Callable): The forward function of the model or any
                    modification of it. If a batch is provided as input for
                    attribution, it is expected that forward_func returns a scalar
                    representing the entire batch.
            interpretable_model (Model): Model object to train interpretable model.
                    A Model object provides a `fit` method to train the model,
                    given a dataloader, with batches containing three tensors:

                    - interpretable_inputs: Tensor
                      [2D num_samples x num_interp_features],
                    - expected_outputs: Tensor [1D num_samples],
                    - weights: Tensor [1D num_samples]

                    The model object must also provide a `representation` method to
                    access the appropriate coefficients or representation of the
                    interpretable model after fitting.
                    Some predefined interpretable linear models are provided in
                    captum._utils.models.linear_model including wrappers around
                    SkLearn linear models as well as SGD-based PyTorch linear
                    models.

                    Note that calling fit multiple times should retrain the
                    interpretable model, each attribution call reuses
                    the same given interpretable model object.
            similarity_func (Callable): Function which takes a single sample
                    along with its corresponding interpretable representation
                    and returns the weight of the interpretable sample for
                    training interpretable model. Weight is generally
                    determined based on similarity to the original input.
                    The original paper refers to this as a similarity kernel.

                    The expected signature of this callable is:

                    >>> similarity_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_interpretable_input:
                    >>>        Tensor [2D 1 x num_interp_features],
                    >>>    **kwargs: Any
                    >>> ) -> float or Tensor containing float scalar

                    perturbed_input and original_input will be the same type and
                    contain tensors of the same shape (regardless of whether or not
                    the sampling function returns inputs in the interpretable
                    space). original_input is the same as the input provided
                    when calling attribute.

                    All kwargs passed to the attribute method are
                    provided as keyword arguments (kwargs) to this callable.
            perturb_func (Callable): Function which returns a single
                    sampled input, generally a perturbation of the original
                    input, which is used to train the interpretable surrogate
                    model. Function can return samples in either
                    the original input space (matching type and tensor shapes
                    of original input) or in the interpretable input space,
                    which is a vector containing the intepretable features.
                    Alternatively, this function can return a generator
                    yielding samples to train the interpretable surrogate
                    model, and n_samples perturbations will be sampled
                    from this generator.

                    The expected signature of this callable is:

                    >>> perturb_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    **kwargs: Any
                    >>> ) -> Tensor, tuple[Tensor, ...], or
                    >>>    generator yielding tensor or tuple[Tensor, ...]

                    All kwargs passed to the attribute method are
                    provided as keyword arguments (kwargs) to this callable.

                    Returned sampled input should match the input type (Tensor
                    or Tuple of Tensor and corresponding shapes) if
                    perturb_interpretable_space = False. If
                    perturb_interpretable_space = True, the return type should
                    be a single tensor of shape 1 x num_interp_features,
                    corresponding to the representation of the
                    sample to train the interpretable model.

                    All kwargs passed to the attribute method are
                    provided as keyword arguments (kwargs) to this callable.
            perturb_interpretable_space (bool): Indicates whether
                    perturb_func returns a sample in the interpretable space
                    (tensor of shape 1 x num_interp_features) or a sample
                    in the original space, matching the format of the original
                    input. Once sampled, inputs can be converted to / from
                    the interpretable representation with either
                    to_interp_rep_transform or from_interp_rep_transform.
            from_interp_rep_transform (Callable): Function which takes a
                    single sampled interpretable representation (tensor
                    of shape 1 x num_interp_features) and returns
                    the corresponding representation in the input space
                    (matching shapes of original input to attribute).

                    This argument is necessary if perturb_interpretable_space
                    is True, otherwise None can be provided for this argument.

                    The expected signature of this callable is:

                    >>> from_interp_rep_transform(
                    >>>    curr_sample: Tensor [2D 1 x num_interp_features]
                    >>>    original_input: Tensor or Tuple of Tensors,
                    >>>    **kwargs: Any
                    >>> ) -> Tensor or tuple[Tensor, ...]

                    Returned sampled input should match the type of original_input
                    and corresponding tensor shapes.

                    All kwargs passed to the attribute method are
                    provided as keyword arguments (kwargs) to this callable.

            to_interp_rep_transform (Callable): Function which takes a
                    sample in the original input space and converts to
                    its interpretable representation (tensor
                    of shape 1 x num_interp_features).

                    This argument is necessary if perturb_interpretable_space
                    is False, otherwise None can be provided for this argument.

                    The expected signature of this callable is:

                    >>> to_interp_rep_transform(
                    >>>    curr_sample: Tensor or Tuple of Tensors,
                    >>>    original_input: Tensor or Tuple of Tensors,
                    >>>    **kwargs: Any
                    >>> ) -> Tensor [2D 1 x num_interp_features]

                    curr_sample will match the type of original_input
                    and corresponding tensor shapes.

                    All kwargs passed to the attribute method are
                    provided as keyword arguments (kwargs) to this callable.
        """
        PerturbationAttribution.__init__(self, forward_func)
        self.interpretable_model = interpretable_model
        self.similarity_func = similarity_func
        self.perturb_func = perturb_func
        self.perturb_interpretable_space = perturb_interpretable_space
        self.from_interp_rep_transform = from_interp_rep_transform
        self.to_interp_rep_transform = to_interp_rep_transform

        if self.perturb_interpretable_space:
            assert (
                self.from_interp_rep_transform is not None
            ), "Must provide transform from interpretable space to original input space"
            " when sampling from interpretable space."
        else:
            assert (
                self.to_interp_rep_transform is not None
            ), "Must provide transform from original input space to interpretable space"

    @log_usage(part_of_slo=True)
    @torch.no_grad()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        target: TargetType = None,
        additional_forward_args: Optional[Tuple[object, ...]] = None,
        n_samples: int = 50,
        perturbations_per_eval: int = 1,
        show_progress: bool = False,
        **kwargs: object,
    ) -> Tensor:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to the inputs of the model using the approach described above.
        It trains an interpretable model and returns a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

        A batch of inputs can be provided as inputs only if forward_func
        returns a single value per batch (e.g. loss).
        The interpretable feature representation should still have shape
        1 x num_interp_features, corresponding to the interpretable
        representation for the full batch, and perturbations_per_eval
        must be set to 1.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
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
                        correspond to the number of examples. For all other types,
                        the given argument is used for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
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
            show_progress (bool, optional): Displays the progress of computation.
                        It will try to use tqdm if available for advanced features
                        (e.g. time estimation). Otherwise, it will fallback to
                        a simple output of progress.
                        Default: False
            **kwargs (Any, optional): Any additional arguments necessary for
                        sampling and transformation functions (provided to
                        constructor).
                        Default: None

        Returns:
            **interpretable model representation**:
            - **interpretable model representation** (*Any*):
                    A representation of the interpretable model trained. The return
                    type matches the return type of train_interpretable_model_func.
                    For example, this could contain coefficients of a
                    linear surrogate model.

        Examples::

            >>> # SimpleClassifier takes a single input tensor of
            >>> # float features with size N x 5,
            >>> # and returns an Nx3 tensor of class probabilities.
            >>> net = SimpleClassifier()
            >>>
            >>> # We will train an interpretable model with the same
            >>> # features by simply sampling with added Gaussian noise
            >>> # to the inputs and training a model to predict the
            >>> # score of the target class.
            >>>
            >>> # For interpretable model training, we will use sklearn
            >>> # linear model in this example. We have provided wrappers
            >>> # around sklearn linear models to fit the Model interface.
            >>> # Any arguments provided to the sklearn constructor can also
            >>> # be provided to the wrapper, e.g.:
            >>> # SkLearnLinearModel("linear_model.Ridge", alpha=2.0)
            >>> from captum._utils.models.linear_model import SkLearnLinearModel
            >>>
            >>>
            >>> # Define similarity kernel (exponential kernel based on L2 norm)
            >>> def similarity_kernel(
            >>>     original_input: Tensor,
            >>>     perturbed_input: Tensor,
            >>>     perturbed_interpretable_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         # kernel_width will be provided to attribute as a kwarg
            >>>         kernel_width = kwargs["kernel_width"]
            >>>         l2_dist = torch.norm(original_input - perturbed_input)
            >>>         return torch.exp(- (l2_dist**2) / (kernel_width**2))
            >>>
            >>>
            >>> # Define sampling function
            >>> # This function samples in original input space
            >>> def perturb_func(
            >>>     original_input: Tensor,
            >>>     **kwargs)->Tensor:
            >>>         return original_input + torch.randn_like(original_input)
            >>>
            >>> # For this example, we are setting the interpretable input to
            >>> # match the model input, so the to_interp_rep_transform
            >>> # function simply returns the input. In most cases, the interpretable
            >>> # input will be different and may have a smaller feature set, so
            >>> # an appropriate transformation function should be provided.
            >>>
            >>> def to_interp_transform(curr_sample, original_inp,
            >>>                                      **kwargs):
            >>>     return curr_sample
            >>>
            >>> # Generating random input with size 1 x 5
            >>> input = torch.randn(1, 5)
            >>> # Defining LimeBase interpreter
            >>> lime_attr = LimeBase(net,
                                     SkLearnLinearModel("linear_model.Ridge"),
                                     similarity_func=similarity_kernel,
                                     perturb_func=perturb_func,
                                     perturb_interpretable_space=False,
                                     from_interp_rep_transform=None,
                                     to_interp_rep_transform=to_interp_transform)
            >>> # Computes interpretable model, returning coefficients of linear
            >>> # model.
            >>> attr_coefs = lime_attr.attribute(input, target=1, kernel_width=1.1)
        """
        inp_tensor = cast(Tensor, inputs) if isinstance(inputs, Tensor) else inputs[0]
        device = inp_tensor.device

        interpretable_inps = []
        similarities = []
        outputs = []

        curr_model_inputs = []
        expanded_additional_args = None
        expanded_target = None
        gen_perturb_func = self._get_perturb_generator_func(inputs, **kwargs)

        if show_progress:
            attr_progress = progress(
                total=math.ceil(n_samples / perturbations_per_eval),
                desc=f"{self.get_name()} attribution",
            )
            attr_progress.update(0)

        batch_count = 0
        for _ in range(n_samples):
            try:
                interpretable_inp, curr_model_input = gen_perturb_func()
            except StopIteration:
                warnings.warn(
                    "Generator completed prior to given n_samples iterations!",
                    stacklevel=1,
                )
                break
            batch_count += 1
            interpretable_inps.append(interpretable_inp)
            curr_model_inputs.append(curr_model_input)

            curr_sim = self.similarity_func(
                inputs, curr_model_input, interpretable_inp, **kwargs
            )
            similarities.append(
                curr_sim.flatten()
                if isinstance(curr_sim, Tensor)
                else torch.tensor([curr_sim], device=device)
            )

            if len(curr_model_inputs) == perturbations_per_eval:
                if expanded_additional_args is None:
                    expanded_additional_args = _expand_additional_forward_args(
                        additional_forward_args, len(curr_model_inputs)
                    )
                if expanded_target is None:
                    expanded_target = _expand_target(target, len(curr_model_inputs))

                model_out = self._evaluate_batch(
                    curr_model_inputs,
                    expanded_target,
                    expanded_additional_args,
                    device,
                )

                if show_progress:
                    attr_progress.update()

                outputs.append(model_out)

                curr_model_inputs = []

        if len(curr_model_inputs) > 0:
            expanded_additional_args = _expand_additional_forward_args(
                additional_forward_args, len(curr_model_inputs)
            )
            expanded_target = _expand_target(target, len(curr_model_inputs))
            model_out = self._evaluate_batch(
                curr_model_inputs,
                expanded_target,
                expanded_additional_args,
                device,
            )
            if show_progress:
                attr_progress.update()
            outputs.append(model_out)

        if show_progress:
            attr_progress.close()

        # Argument 1 to "cat" has incompatible type
        # "list[Tensor | tuple[Tensor, ...]]";
        # expected "tuple[Tensor, ...] | list[Tensor]"  [arg-type]
        combined_interp_inps = torch.cat(interpretable_inps).float()  # type: ignore
        combined_outputs = (
            torch.cat(outputs) if len(outputs[0].shape) > 0 else torch.stack(outputs)
        ).float()
        combined_sim = (
            torch.cat(similarities)
            if len(similarities[0].shape) > 0
            else torch.stack(similarities)
        ).float()
        dataset = TensorDataset(combined_interp_inps, combined_outputs, combined_sim)
        self.interpretable_model.fit(DataLoader(dataset, batch_size=batch_count))
        return self.interpretable_model.representation()

    def _get_perturb_generator_func(
        self, inputs: TensorOrTupleOfTensorsGeneric, **kwargs: Any
    ) -> Callable[
        [], Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
    ]:
        perturb_generator: Optional[Iterator[TensorOrTupleOfTensorsGeneric]]
        perturb_generator = None
        if inspect.isgeneratorfunction(self.perturb_func):
            perturb_generator = self.perturb_func(inputs, **kwargs)

        def generate_perturbation() -> (
            Tuple[TensorOrTupleOfTensorsGeneric, TensorOrTupleOfTensorsGeneric]
        ):
            if perturb_generator:
                curr_sample = next(perturb_generator)
            else:
                curr_sample = self.perturb_func(inputs, **kwargs)

            if self.perturb_interpretable_space:
                interpretable_inp = curr_sample
                curr_model_input = self.from_interp_rep_transform(  # type: ignore
                    curr_sample, inputs, **kwargs
                )
            else:
                curr_model_input = curr_sample
                interpretable_inp = self.to_interp_rep_transform(  # type: ignore
                    curr_sample, inputs, **kwargs
                )

            return interpretable_inp, curr_model_input  # type: ignore

        return generate_perturbation

    def attribute_future(self) -> None:
        r"""
        This method is not implemented for LimeBase.
        """
        raise NotImplementedError(
            "LimeBase does not support attribution of future samples."
        )

    def _evaluate_batch(
        self,
        curr_model_inputs: List[TensorOrTupleOfTensorsGeneric],
        expanded_target: TargetType,
        expanded_additional_args: object,
        device: torch.device,
    ) -> Tensor:
        model_out = _run_forward(
            self.forward_func,
            _reduce_list(curr_model_inputs),
            expanded_target,
            expanded_additional_args,
        )
        if isinstance(model_out, Tensor):
            assert model_out.numel() == len(curr_model_inputs), (
                "Number of outputs is not appropriate, must return "
                "one output per perturbed input"
            )
        if isinstance(model_out, Tensor):
            return model_out.flatten()
        return torch.tensor([model_out], device=device)

    def has_convergence_delta(self) -> bool:
        return False

    @property
    def multiplies_by_inputs(self) -> bool:
        return False


# Default transformations and methods
# for Lime child implementation.


def default_from_interp_rep_transform(
    curr_sample: Tensor,
    original_inputs: TensorOrTupleOfTensorsGeneric,
    **kwargs: Any,
) -> TensorOrTupleOfTensorsGeneric:

    assert (
        "feature_mask" in kwargs
    ), "Must provide feature_mask to use default interpretable representation transform"
    assert (
        "baselines" in kwargs
    ), "Must provide baselines to use default interpretable representation transform"
    feature_mask: TensorOrTupleOfTensorsGeneric = kwargs["feature_mask"]
    if isinstance(feature_mask, Tensor):
        binary_mask = curr_sample[0][feature_mask].bool()
        return (
            binary_mask.to(original_inputs.dtype) * original_inputs
            + (~binary_mask).to(original_inputs.dtype) * kwargs["baselines"]
        )
    else:
        binary_mask = tuple(
            curr_sample[0][feature_mask[j]].bool() for j in range(len(feature_mask))
        )

        return cast(
            TensorOrTupleOfTensorsGeneric,
            tuple(
                binary_mask[j].to(original_inputs[j].dtype) * original_inputs[j]
                + (~binary_mask[j]).to(original_inputs[j].dtype)
                * kwargs["baselines"][j]
                for j in range(len(feature_mask))
            ),
        )


def get_exp_kernel_similarity_function(
    distance_mode: str = "cosine",
    kernel_width: float = 1.0,
) -> Callable[..., float]:
    r"""
    This method constructs an appropriate similarity function to compute
    weights for perturbed sample in LIME. Distance between the original
    and perturbed inputs is computed based on the provided distance mode,
    and the distance is passed through an exponential kernel with given
    kernel width to convert to a range between 0 and 1.

    The callable returned can be provided as the similarity_fn for
    Lime or LimeBase.

    Args:

        distance_mode (str, optional): Distance mode can be either "cosine" or
                    "euclidean" corresponding to either cosine distance
                    or Euclidean distance respectively. Distance is computed
                    by flattening the original inputs and perturbed inputs
                    (concatenating tuples of inputs if necessary) and computing
                    distances between the resulting vectors.
                    Default: "cosine"
        kernel_width (float, optional):
                    Kernel width for exponential kernel applied to distance.
                    Default: 1.0

    Returns:

        *Callable*:
        - **similarity_fn** (*Callable*):
            Similarity function. This callable can be provided as the
            similarity_fn for Lime or LimeBase.
    """

    def default_exp_kernel(
        original_inp: TensorOrTupleOfTensorsGeneric,
        perturbed_inp: TensorOrTupleOfTensorsGeneric,
        __: Any,
        **kwargs: Any,
    ) -> float:
        flattened_original_inp = _flatten_tensor_or_tuple(original_inp).float()
        flattened_perturbed_inp = _flatten_tensor_or_tuple(perturbed_inp).float()
        if distance_mode == "cosine":
            cos_sim = CosineSimilarity(dim=0)
            distance = 1 - cos_sim(flattened_original_inp, flattened_perturbed_inp)
        elif distance_mode == "euclidean":
            distance = torch.norm(flattened_original_inp - flattened_perturbed_inp)
        else:
            raise ValueError("distance_mode must be either cosine or euclidean.")
        return math.exp(-1 * (distance**2) / (2 * (kernel_width**2)))

    return default_exp_kernel


def default_perturb_func(
    original_inp: TensorOrTupleOfTensorsGeneric, **kwargs: object
) -> Tensor:
    assert (
        "num_interp_features" in kwargs
    ), "Must provide num_interp_features to use default interpretable sampling function"
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device

    probs = torch.ones(1, cast(int, kwargs["num_interp_features"])) * 0.5
    return torch.bernoulli(probs).to(device=device).long()


def construct_feature_mask(
    feature_mask: Union[None, Tensor, Tuple[Tensor, ...]],
    formatted_inputs: Tuple[Tensor, ...],
) -> Tuple[Tuple[Tensor, ...], int]:
    feature_mask_tuple: Tuple[Tensor, ...]
    if feature_mask is None:
        feature_mask_tuple, num_interp_features = _construct_default_feature_mask(
            formatted_inputs
        )
    else:
        feature_mask_tuple = _format_tensor_into_tuples(feature_mask)
        min_interp_features = int(
            min(
                torch.min(single_mask).item()
                for single_mask in feature_mask_tuple
                if single_mask.numel()
            )
        )
        if min_interp_features != 0:
            warnings.warn(
                "Minimum element in feature mask is not 0, shifting indices to"
                " start at 0.",
                stacklevel=2,
            )
            feature_mask_tuple = tuple(
                single_mask - min_interp_features for single_mask in feature_mask_tuple
            )

        num_interp_features = _get_max_feature_index(feature_mask_tuple) + 1
    return feature_mask_tuple, num_interp_features


class Lime(LimeBase):
    r"""
    Lime is an interpretability method that trains an interpretable surrogate model
    by sampling points around a specified input example and using model evaluations
    at these points to train a simpler interpretable 'surrogate' model, such as a
    linear model.

    Lime provides a more specific implementation than LimeBase in order to expose
    a consistent API with other perturbation-based algorithms. For more general
    use of the LIME framework, consider using the LimeBase class directly and
    defining custom sampling and transformation to / from interpretable
    representation functions.

    Lime assumes that the interpretable representation is a binary vector,
    corresponding to some elements in the input being set to their baseline value
    if the corresponding binary interpretable feature value is 0 or being set
    to the original input value if the corresponding binary interpretable
    feature value is 1. Input values can be grouped to correspond to the same
    binary interpretable feature using a feature mask provided when calling
    attribute, similar to other perturbation-based attribution methods.

    One example of this setting is when applying Lime to an image classifier.
    Pixels in an image can be grouped into super-pixels or segments, which
    correspond to interpretable features, provided as a feature_mask when
    calling attribute. Sampled binary vectors convey whether a super-pixel
    is on (retains the original input values) or off (set to the corresponding
    baseline value, e.g. black image). An interpretable linear model is trained
    with input being the binary vectors and outputs as the corresponding scores
    of the image classifier with the appropriate super-pixels masked based on the
    binary vector. Coefficients of the trained surrogate
    linear model convey the importance of each super-pixel.

    More details regarding LIME can be found in the original paper:
    https://arxiv.org/abs/1602.04938
    """

    def __init__(
        self,
        forward_func: Callable[..., Union[int, float, Tensor]],
        interpretable_model: Optional[Model] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        similarity_func: Optional[Callable] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        perturb_func: Optional[Callable] = None,
    ) -> None:
        r"""

        Args:


            forward_func (Callable): The forward function of the model or any
                    modification of it
            interpretable_model (Model, optional): Model object to train
                    interpretable model.

                    This argument is optional and defaults to SkLearnLasso(alpha=0.01),
                    which is a wrapper around the Lasso linear model in SkLearn.
                    This requires having sklearn version >= 0.23 available.

                    Other predefined interpretable linear models are provided in
                    captum._utils.models.linear_model.

                    Alternatively, a custom model object must provide a `fit` method to
                    train the model, given a dataloader, with batches containing
                    three tensors:

                    - interpretable_inputs: Tensor
                      [2D num_samples x num_interp_features],
                    - expected_outputs: Tensor [1D num_samples],
                    - weights: Tensor [1D num_samples]

                    The model object must also provide a `representation` method to
                    access the appropriate coefficients or representation of the
                    interpretable model after fitting.

                    Note that calling fit multiple times should retrain the
                    interpretable model, each attribution call reuses
                    the same given interpretable model object.
            similarity_func (Callable, optional): Function which takes a single sample
                    along with its corresponding interpretable representation
                    and returns the weight of the interpretable sample for
                    training the interpretable model.
                    This is often referred to as a similarity kernel.

                    This argument is optional and defaults to a function which
                    applies an exponential kernel to the cosine distance between
                    the original input and perturbed input, with a kernel width
                    of 1.0.

                    A similarity function applying an exponential
                    kernel to cosine / euclidean distances can be constructed
                    using the provided get_exp_kernel_similarity_function in
                    captum.attr._core.lime.

                    Alternately, a custom callable can also be provided.
                    The expected signature of this callable is:

                    >>> def similarity_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_input: Tensor or tuple[Tensor, ...],
                    >>>    perturbed_interpretable_input:
                    >>>        Tensor [2D 1 x num_interp_features],
                    >>>    **kwargs: Any
                    >>> ) -> float or Tensor containing float scalar

                    perturbed_input and original_input will be the same type and
                    contain tensors of the same shape, with original_input
                    being the same as the input provided when calling attribute.

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask).
            perturb_func (Callable, optional): Function which returns a single
                    sampled input, which is a binary vector of length
                    num_interp_features, or a generator of such tensors.

                    This function is optional, the default function returns
                    a binary vector where each element is selected
                    independently and uniformly at random. Custom
                    logic for selecting sampled binary vectors can
                    be implemented by providing a function with the
                    following expected signature:

                    >>> perturb_func(
                    >>>    original_input: Tensor or tuple[Tensor, ...],
                    >>>    **kwargs: Any
                    >>> ) -> Tensor [Binary 2D Tensor 1 x num_interp_features]
                    >>>  or generator yielding such tensors

                    kwargs includes baselines, feature_mask, num_interp_features
                    (integer, determined from feature mask).

        """
        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()

        if perturb_func is None:
            perturb_func = default_perturb_func

        LimeBase.__init__(
            self,
            forward_func,
            interpretable_model,
            similarity_func,
            perturb_func,
            True,
            default_from_interp_rep_transform,
            None,
        )

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
        training an interpretable model and returning a representation of the
        interpretable model.

        It is recommended to only provide a single example as input (tensors
        with first dimension or batch size = 1). This is because LIME is generally
        used for sample-based interpretability, training a separate interpretable
        model to explain a model's prediction on each individual example.

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

            inputs (Tensor or tuple[Tensor, ...]): Input for which LIME
                        is computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference value which replaces each
                        feature when the corresponding interpretable feature
                        is set to 0.
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
                        coefficient of the corresponding interpretale feature.
                        All elements with the same value in the feature mask
                        will contain the same coefficient in the returned
                        attributions.
                        If forward_func returns a single element per batch, then the
                        first dimension of each tensor will be 1, and the remaining
                        dimensions will have the same shape as the original input
                        tensor.
                        If return_input_shape is False, a 1D
                        tensor is returned, containing only the coefficients
                        of the trained interpreatable models, with length
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

            >>> # Defining Lime interpreter
            >>> lime = Lime(net)
            >>> # Computes attribution, with each of the 4 x 4 = 16
            >>> # features as a separate interpretable feature
            >>> attr = lime.attribute(input, target=1, n_samples=200)

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

            >>> # Computes interpretable model and returning attributions
            >>> # matching input shape.
            >>> attr = lime.attribute(input, target=1, feature_mask=feature_mask)
        """
        return self._attribute_kwargs(
            inputs=inputs,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            feature_mask=feature_mask,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            return_input_shape=return_input_shape,
            show_progress=show_progress,
        )

    def attribute_future(self) -> None:
        return super().attribute_future()

    def _attribute_kwargs(  # type: ignore
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
        **kwargs: object,
    ) -> TensorOrTupleOfTensorsGeneric:
        is_inputs_tuple = _is_tuple(inputs)
        formatted_inputs, baselines = _format_input_baseline(inputs, baselines)
        bsz = formatted_inputs[0].shape[0]

        feature_mask, num_interp_features = construct_feature_mask(
            feature_mask, formatted_inputs
        )

        if num_interp_features > 10000:
            warnings.warn(
                "Attempting to construct interpretable model with > 10000 features."
                "This can be very slow or lead to OOM issues. Please provide a feature"
                "mask which groups input features to reduce the number of interpretable"
                "features. ",
                stacklevel=1,
            )

        coefs: Tensor
        if bsz > 1:
            test_output = _run_forward(
                self.forward_func, inputs, target, additional_forward_args
            )
            if isinstance(test_output, Tensor) and torch.numel(test_output) > 1:
                if torch.numel(test_output) == bsz:
                    warnings.warn(
                        "You are providing multiple inputs for Lime / Kernel SHAP "
                        "attributions. This trains a separate interpretable model "
                        "for each example, which can be time consuming. It is "
                        "recommended to compute attributions for one example at a "
                        "time.",
                        stacklevel=1,
                    )
                    output_list = []
                    for (
                        curr_inps,
                        curr_target,
                        curr_additional_args,
                        curr_baselines,
                        curr_feature_mask,
                    ) in _batch_example_iterator(
                        bsz,
                        formatted_inputs,
                        target,
                        additional_forward_args,
                        baselines,
                        feature_mask,
                    ):
                        coefs = super().attribute.__wrapped__(
                            self,
                            inputs=curr_inps if is_inputs_tuple else curr_inps[0],
                            target=curr_target,
                            additional_forward_args=curr_additional_args,
                            n_samples=n_samples,
                            perturbations_per_eval=perturbations_per_eval,
                            baselines=(
                                curr_baselines if is_inputs_tuple else curr_baselines[0]
                            ),
                            feature_mask=(
                                curr_feature_mask
                                if is_inputs_tuple
                                else curr_feature_mask[0]
                            ),
                            num_interp_features=num_interp_features,
                            show_progress=show_progress,
                            **kwargs,
                        )
                        if return_input_shape:
                            output_list.append(
                                self._convert_output_shape(
                                    curr_inps,
                                    curr_feature_mask,
                                    coefs,
                                    num_interp_features,
                                    is_inputs_tuple,
                                )
                            )
                        else:
                            output_list.append(coefs.reshape(1, -1))  # type: ignore

                    return _reduce_list(output_list)
                else:
                    raise AssertionError(
                        "Invalid number of outputs, forward function should return a"
                        "scalar per example or a scalar per input batch."
                    )
            else:
                assert perturbations_per_eval == 1, (
                    "Perturbations per eval must be 1 when forward function"
                    "returns single value per batch!"
                )

        coefs = super().attribute.__wrapped__(
            self,
            inputs=inputs,
            target=target,
            additional_forward_args=additional_forward_args,
            n_samples=n_samples,
            perturbations_per_eval=perturbations_per_eval,
            baselines=baselines if is_inputs_tuple else baselines[0],
            feature_mask=feature_mask if is_inputs_tuple else feature_mask[0],
            num_interp_features=num_interp_features,
            show_progress=show_progress,
            **kwargs,
        )
        if return_input_shape:
            # pyre-fixme[7]: Expected `TensorOrTupleOfTensorsGeneric` but got
            #  `Tuple[Tensor, ...]`.
            return self._convert_output_shape(
                formatted_inputs,
                feature_mask,
                coefs,
                num_interp_features,
                is_inputs_tuple,
                leading_dim_one=(bsz > 1),
            )
        else:
            return coefs

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[True],
        leading_dim_one: bool = False,
    ) -> Tuple[Tensor, ...]: ...

    @typing.overload
    def _convert_output_shape(  # type: ignore
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: Literal[False],
        leading_dim_one: bool = False,
    ) -> Tensor: ...

    @typing.overload
    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
        leading_dim_one: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]: ...

    def _convert_output_shape(
        self,
        formatted_inp: Tuple[Tensor, ...],
        feature_mask: Tuple[Tensor, ...],
        coefs: Tensor,
        num_interp_features: int,
        is_inputs_tuple: bool,
        leading_dim_one: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        coefs = coefs.flatten()
        attr = [
            torch.zeros_like(single_inp, dtype=torch.float)
            for single_inp in formatted_inp
        ]
        for tensor_ind in range(len(formatted_inp)):
            for single_feature in range(num_interp_features):
                attr[tensor_ind] += (
                    coefs[single_feature].item()
                    * (feature_mask[tensor_ind] == single_feature).float()
                )
        if leading_dim_one:
            for i in range(len(attr)):
                attr[i] = attr[i][0:1]
        return _format_output(is_inputs_tuple, tuple(attr))
