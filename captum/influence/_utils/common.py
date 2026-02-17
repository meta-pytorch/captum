#!/usr/bin/env python3

# pyre-strict
import warnings
from functools import reduce
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import torch
import torch.nn as nn
from captum._utils.common import _get_module_from_name, parse_version
from captum._utils.gradient import (
    _compute_jacobian_wrt_params,
    _compute_jacobian_wrt_params_with_sample_wise_trick,
)
from captum._utils.progress import progress
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from captum.influence._core.influence_function import (
        InfluenceFunctionBase,
        IntermediateQuantitiesInfluenceFunction,
    )
    from captum.influence._core.tracincp import TracInCP, TracInCPBase


def _tensor_batch_dot(t1: Tensor, t2: Tensor) -> Tensor:
    r"""
    Computes pairwise dot product between two tensors

    Args:
        Tensors t1 and t2 are feature vectors with dimension (batch_size_1, *) and
        (batch_size_2,  *). The * dimensions must match in total number of elements.

    Returns:
        Tensor with shape (batch_size_1, batch_size_2) containing the pairwise dot
        products. For example, Tensor[i][j] would be the dot product between
        t1[i] and t2[j].
    """

    msg = (
        "Please ensure each batch member has the same feature dimension. "
        f"First input has {torch.numel(t1) / t1.shape[0]} features, and "
        f"second input has {torch.numel(t2) / t2.shape[0]} features."
    )
    assert torch.numel(t1) / t1.shape[0] == torch.numel(t2) / t2.shape[0], msg

    return torch.mm(
        t1.view(t1.shape[0], -1),
        t2.view(t2.shape[0], -1).T,
    )


def _gradient_dot_product(
    input_grads: Tuple[Tensor, ...], src_grads: Tuple[Tensor, ...]
) -> Tensor:
    r"""
    Computes the dot product between the gradient vector for a model on an input batch
    and src batch, for each pairwise batch member. Gradients are passed in as a tuple
    corresponding to the trainable parameters returned by model.parameters(). Output
    corresponds to a tensor of size (inputs_batch_size, src_batch_size) with all
    pairwise dot products.
    """

    assert len(input_grads) == len(src_grads), "Mismatching gradient parameters."

    iterator = zip(input_grads, src_grads)
    total = _tensor_batch_dot(*next(iterator))
    for input_grad, src_grad in iterator:
        total += _tensor_batch_dot(input_grad, src_grad)

    return total


def _jacobian_loss_wrt_inputs(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Union[Module, Callable],
    out: Tensor,
    targets: Tensor,
    vectorize: bool,
    reduction_type: str,
) -> Tensor:
    r"""
    Often, we have a loss function that computes a per-sample loss given a 1D tensor
    input, and we want to calculate the jacobian of the loss w.r.t. that input.  For
    example, the input could be a length K tensor specifying the probability a given
    sample belongs to each of K possible classes, and the loss function could be
    cross-entropy loss. This function performs that calculation, but does so for a
    *batch* of inputs. We create this helper function for two reasons: 1) to handle
    differences between Pytorch versiosn for vectorized jacobian calculations, and
    2) this function does not accept the aforementioned per-sample loss function.
    Instead, it accepts a "reduction" loss function that *reduces* the per-sample loss
    for a batch into a single loss. Using a "reduction" loss improves speed.
    We will allow this reduction to either be the mean or sum of the per-sample losses,
    and this function provides an uniform way to handle different possible reductions,
    and also check if the reduction used is valid. Regardless of the reduction used,
    this function returns the jacobian for the per-sample loss (for each sample in the
    batch).

    Args:
        loss_fn (torch.nn.Module, Callable): The loss function. If a library
                defined loss function is provided, it would be expected to be a
                torch.nn.Module. If a custom loss is provided, it can be either type,
                but must behave as a library loss function would if `reduction='sum'`
                or `reduction='mean'`.
        out (Tensor): This is a tensor that represents the batch of inputs to
                `loss_fn`. In practice, this will be the output of a model; this is
                why this argument is named `out`. `out` is a 2D tensor of shape
                (batch size, model output dimensionality). We will call `loss_fn` via
                `loss_fn(out, targets)`.
        targets (Tensor): The labels for the batch of inputs.
        vectorize (bool): Flag to use experimental vectorize functionality for
                `torch.autograd.functional.jacobian`.
        reduction_type (str): The type of reduction used by `loss_fn`. If `loss_fn`
                has the "reduction" attribute, we will check that they match. Can
                only be "mean" or "sum".

    Returns:
        jacobians (Tensor): Returns the jacobian of the per-sample loss (implicitly
                defined by `loss_fn` and `reduction_type`) w.r.t each sample
                in the batch represented by `out`. This is a 2D tensor, where the
                first dimension is the batch dimension.
    """
    if reduction_type != "sum" and reduction_type != "mean":
        raise ValueError(
            f"`{reduction_type}` is not a valid value for reduction_type. "
            "Must be either 'sum' or 'mean'."
        )

    # TODO: allow loss_fn to be Callable
    if isinstance(loss_fn, Module) and hasattr(loss_fn, "reduction"):
        msg = (
            f"loss_fn.reduction `{loss_fn.reduction}` does not match"
            f"reduction type `{reduction_type}`. Please ensure they are"
            " matching."
        )
        assert loss_fn.reduction == reduction_type, msg

    input_jacobians = torch.autograd.functional.jacobian(
        lambda out: loss_fn(out, targets), out
    )

    if reduction_type == "mean":
        input_jacobians = input_jacobians * len(input_jacobians)

    return input_jacobians


def _load_flexible_state_dict(model: Module, path: str) -> float:
    r"""
    Helper to load pytorch models. This function attempts to find compatibility for
    loading models that were trained on different devices / with DataParallel but are
    being loaded in a different environment.

    Assumes that the model has been saved as a state_dict in some capacity. This can
    either be a single state dict, or a nesting dictionary which contains the model
    state_dict and other information.

    Args:

        model (torch.nn.Module): The model for which to load a checkpoint
        path (str): The filepath to the checkpoint

    The module state_dict is modified in-place, and the learning rate is returned.
    """

    checkpoint = torch.load(path)

    learning_rate = checkpoint.get("learning_rate", 1.0)
    # can get learning rate from optimizer state_dict?

    if "module." in next(iter(checkpoint)):
        if isinstance(model, nn.DataParallel):
            model.load_state_dict(checkpoint)
        else:
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint)
            model = model.module
    else:
        if isinstance(model, nn.DataParallel):
            model = model.module
            model.load_state_dict(checkpoint)
            model = nn.DataParallel(model)
        else:
            model.load_state_dict(checkpoint)

    return learning_rate


def _get_k_most_influential_helper(
    influence_src_dataloader: DataLoader,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    influence_batch_fn: Callable,
    inputs: Any,
    k: int = 5,
    proponents: bool = True,
    show_progress: bool = False,
    desc: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    r"""
    Helper function that computes the quantities returned by
    `TracInCPBase._get_k_most_influential`, using a specific implementation that is
    constant memory.

    Args:
        influence_src_dataloader (DataLoader): The DataLoader, representing training
                data, for which we want to compute proponents / opponents.
        influence_batch_fn (Callable): A callable that will be called via
                `influence_batch_fn(inputs, batch)`, where `batch` is a batch
                in the `influence_src_dataloader` argument.
        inputs (any): This argument represents the test batch, and can be of any type.
                It is passed as the first argument to `influence_batch_fn`, and thus
                needs to be compatible with it. It is not necessarily the test batch
                itself, but can be some quantity derived from it, i.e. its jacobians.
        k (int, optional): The number of proponents or opponents to return per test
                instance.
                Default: 5
        proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                or opponents (`proponents=False`)
                Default: True
        show_progress (bool, optional): To compute the proponents (or opponents)
                for the batch of examples, we perform computation for each batch in
                training dataset `influence_src_dataloader`, If `show_progress` is
                true, the progress of this computation will be displayed. In
                particular, the number of batches for which the computation has
                been performed will be displayed. It will try to use tqdm if
                available for advanced features (e.g. time estimation). Otherwise,
                it will fallback to a simple output of progress.
                Default: False
        desc (str, optional): If `show_progress` is true, this is the description to
                show when displaying progress. If `desc` is none, no description is
                shown.
                Default: None

    Returns:
        (indices, influence_scores): `indices` is a torch.long Tensor that contains the
                indices of the proponents (or opponents) for each test example. Its
                dimension is `(inputs_batch_size, k)`, where `inputs_batch_size` is the
                number of examples in `inputs`. For example, if `proponents==True`,
                `indices[i][j]` is the index of the example in training dataset
                `influence_src_dataloader` with the k-th highest influence score for
                the j-th example in `inputs`. `indices` is a `torch.long` tensor so that
                it can directly be used to index other tensors. Each row of
                `influence_scores` contains the influence scores for a different test
                example, in sorted order. In particular, `influence_scores[i][j]` is
                the influence score of example `indices[i][j]` in training dataset
                `influence_src_dataloader` on example `i` in the test batch represented
                by `inputs` and `targets`.
    """
    # For each test instance, maintain the best indices and corresponding distances
    # initially, these will be empty
    topk_indices = torch.Tensor().long()
    topk_tracin_scores = torch.Tensor()

    multiplier = 1.0 if proponents else -1.0

    # needed to map from relative index in a batch fo index within entire `dataloader`
    num_instances_processed = 0

    # if show_progress, create progress bar
    total: Optional[int] = None
    data_iterator: Union[Iterable[object], DataLoader] = influence_src_dataloader
    if show_progress:
        try:
            total = len(influence_src_dataloader)
        except AttributeError:
            pass
        data_iterator = progress(
            cast(Iterable[object], influence_src_dataloader),
            desc=desc,
            total=total,
        )

    for batch in data_iterator:

        # calculate tracin_scores for the batch
        batch_tracin_scores = influence_batch_fn(inputs, batch)
        batch_tracin_scores *= multiplier

        # get the top-k indices and tracin_scores for the batch
        batch_size = batch_tracin_scores.shape[1]
        batch_topk_tracin_scores, batch_topk_indices = torch.topk(
            batch_tracin_scores, min(batch_size, k), dim=1
        )
        batch_topk_indices = batch_topk_indices + num_instances_processed
        num_instances_processed += batch_size

        # combine the top-k for the batch with those for previously seen batches
        topk_indices = torch.cat(
            [topk_indices.to(batch_topk_indices.device), batch_topk_indices], dim=1
        )
        topk_tracin_scores = torch.cat(
            [
                topk_tracin_scores.to(batch_topk_tracin_scores.device),
                batch_topk_tracin_scores,
            ],
            dim=1,
        )

        # retain only the top-k in terms of tracin_scores
        topk_tracin_scores, topk_argsort = torch.topk(
            topk_tracin_scores, min(k, topk_indices.shape[1]), dim=1
        )
        topk_indices = torch.gather(topk_indices, dim=1, index=topk_argsort)

    # if seeking opponents, we were actually keeping track of negative tracin_scores
    topk_tracin_scores *= multiplier

    return topk_indices, topk_tracin_scores


class _DatasetFromList(Dataset):
    def __init__(self, _l: List[Any]) -> None:
        self._l = _l

    def __getitem__(self, i: int) -> Any:
        return self._l[i]

    def __len__(self) -> int:
        return len(self._l)


def _format_inputs_dataset(
    inputs_dataset: Union[Tuple[Any, ...], DataLoader],
) -> DataLoader:
    # if `inputs_dataset` is not a `DataLoader`, turn it into one.
    # `_DatasetFromList` turns a list into a `Dataset` where `__getitem__`
    # returns an element in the list, and using it to construct a `DataLoader`
    # with `batch_size=None` gives a `DataLoader` that yields a single batch.
    if not isinstance(inputs_dataset, DataLoader):
        inputs_dataset = DataLoader(
            _DatasetFromList([inputs_dataset]), shuffle=False, batch_size=None
        )
    return inputs_dataset


def _self_influence_by_batches_helper(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    self_influence_batch_fn: Callable,
    instance_name: str,
    inputs_dataset: Union[Tuple[Any, ...], DataLoader],
    show_progress: bool = False,
) -> Tensor:
    """
    Computes self influence scores for the examples in `inputs_dataset`, which is
    either a single batch or a Pytorch `DataLoader` that yields batches. The self
    influence scores for a single batch are computed using the
    `self_influence_batch_fn` input. Note that if `inputs_dataset` is a single batch,
    this will call `model` on that single batch, where `model` is the model used to
    compute self influence scores by `self_influence_batch_fn`, and if `inputs_dataset`
    yields batches, this will call `model` on each batch that is yielded. Therefore,
    please ensure that for both cases, the batch(es) that `model` is called
    with are not too large, so that there will not be an out-of-memory error. This
    implementation performs an outer iteration over all batches that
    `inputs_dataset` represents, and an inner iteration over checkpoints. The pros
    of this implementation are that showing the progress of the computation is
    straightforward.

    Args:
        self_influence_batch_fn (Callable): This is the function that computes self
                influence scores for a single batch.
        instance_name (str): This is the name of the implementation class that
                `self_influence_batch_fn` is a method of. This is used for displaying
                warning messages.
        batches (tuple or DataLoader): Either a single tuple of any, or a
                `DataLoader`, where each batch yielded is a tuple of any. In
                either case, the tuple represents a single batch, where the last
                element is assumed to be the labels for the batch. That is,
                `model(*batch[0:-1])` produces the output for `model`,
                and `batch[-1]` are the labels, if any. This is the same
                assumption made for each batch yielded by training dataset
                `train_dataset`. Please see documentation for the
                `train_dataset` argument to `TracInCP.__init__` for
                more details on the assumed structure of a batch.
        show_progress (bool, optional): Computation of self influence scores can
                take a long time if `inputs_dataset` represents many examples. If
                `show_progress`is true, the progress of this computation will be
                displayed. In particular, the number of batches for which self
                influence scores have been computed will be displayed. It will try
                to use tqdm if available for advanced features (e.g. time
                estimation). Otherwise, it will fallback to a simple output of
                progress.
                Default: False

    Returns:
        self_influence_scores (Tensor): This is a 1D tensor containing the self
                influence scores of all examples in `inputs_dataset`, regardless of
                whether it represents a single batch or a `DataLoader` that yields
                batches.
    """
    # If `inputs_dataset` is not a `DataLoader`, turn it into one.
    inputs_dataset = _format_inputs_dataset(inputs_dataset)
    inputs_dataset_iterator: Union[Iterable[object], DataLoader] = inputs_dataset

    # If `show_progress` is true, create a progress bar that keeps track of how
    # many batches have been processed
    if show_progress:
        # First, try to determine length of progress bar if possible, with a
        # default of `None`
        inputs_dataset_len = None
        try:
            inputs_dataset_len = len(inputs_dataset)
        except TypeError:
            warnings.warn(
                "Unable to determine the number of batches in `inputs_dataset`. "
                "Therefore, if showing the progress of the computation of self "
                "influence scores, only the number of batches processed can be "
                "displayed, and not the percentage completion of the computation, "
                "nor any time estimates.",
                stacklevel=1,
            )
        # then create the progress bar
        inputs_dataset_iterator = progress(
            inputs_dataset,
            desc=f"Using {instance_name} to compute self influence. Processing batch",
            total=inputs_dataset_len,
        )

    # To compute self influence scores for each batch, we use
    # `_self_influence_by_checkpoints`, which can accept a tuple representing a
    # single batch as the `inputs_dataset` argument (as well as a DataLoader).
    # Because we are already displaying progress in terms of number of batches
    # processed in this method, we will not show progress for the call to
    # `_self_influence_by_checkpoints`.
    return torch.cat(
        [
            self_influence_batch_fn(batch, show_progress=False)
            for batch in inputs_dataset_iterator
        ]
    )


def _check_loss_fn(
    influence_instance: Union["TracInCPBase", "InfluenceFunctionBase"],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]],
    loss_fn_name: str,
    sample_wise_grads_per_batch: bool = True,
) -> str:
    """
    This checks whether `loss_fn` satisfies the requirements assumed of all
    implementations of `TracInCPBase`. It works regardless of whether the
    implementation has the `sample_wise_grads_per_batch` attribute.
    It returns the reduction type of the loss_fn. If `sample_wise_grads_per_batch`
    if not provided, we assume the implementation does not have that attribute.
    """
    # if `loss_fn` is `None`, there is nothing to check. then, the reduction type is
    # only used by `_compute_jacobian_wrt_params_with_sample_wise_trick`, where
    # reduction type should be "sum" if `loss_fn` is `None`.
    if loss_fn is None:
        return "sum"

    # perhaps since `Module` is an implementation of `Callable`, this has redundancy
    assert isinstance(loss_fn, Module) or callable(loss_fn)

    reduction_type = "none"

    # If we are able to access the reduction used by `loss_fn`, we check whether
    # the reduction is compatible with `sample_wise_grads_per_batch`, if it has the
    # attribute.
    if hasattr(loss_fn, "reduction"):
        reduction = loss_fn.reduction  # type: ignore
        if sample_wise_grads_per_batch:
            assert reduction in [
                "sum",
                "mean",
            ], (
                'reduction for `loss_fn` must be "sum" or "mean" when '
                "`sample_wise_grads_per_batch` is True (i.e. the default value) "
            )
            reduction_type = str(reduction)
        else:
            assert reduction == "none", (
                'reduction for `loss_fn` must be "none" when '
                "`sample_wise_grads_per_batch` is False"
            )
    else:
        # if we are unable to access the reduction used by `loss_fn`, we warn
        # the user about the assumptions we are making regarding the reduction
        # used by `loss_fn`
        if sample_wise_grads_per_batch:
            warnings.warn(
                f"Since `{loss_fn_name}`` has no 'reduction' attribute, and "
                "`sample_wise_grads_per_batch` is True, the implementation assumes "
                f"that `{loss_fn_name}` is a 'reduction' loss function that reduces "
                f"the per-example losses by taking their *sum*. If `{loss_fn_name}` "
                "instead reduces the per-example losses by taking their mean, "
                f'please set the reduction attribute of `{loss_fn_name}` to "mean", '
                f'i.e. `{loss_fn_name}.reduction = "mean"`. Note that if '
                "`sample_wise_grads_per_batch` is True, the implementation "
                "assumes the reduction is either a sum or mean reduction.",
                stacklevel=1,
            )
            reduction_type = "sum"
        else:
            warnings.warn(
                f'Since `{loss_fn_name}` has no "reduction" attribute, and '
                "`sample_wise_grads_per_batch` is False, the implementation "
                f'assumes that `{loss_fn_name}` is a "per-example" loss function (see '
                f"documentation for `{loss_fn_name}` for details).  Please ensure "
                "that this is the case.",
                stacklevel=1,
            )

    return reduction_type


def _set_active_parameters(model: Module, layers: List[str]) -> List[Module]:
    """
    sets relevant parameters, as indicated by `layers`, to have `requires_grad=True`,
    and returns relevant modules.
    """
    assert isinstance(layers, List), "`layers` should be a list!"
    assert len(layers) > 0, "`layers` cannot be empty!"
    assert isinstance(layers[0], str), "`layers` should contain str layer names."
    layer_modules = [_get_module_from_name(model, layer) for layer in layers]
    for layer, layer_module in zip(layers, layer_modules):
        for name, param in layer_module.named_parameters():
            if not param.requires_grad:
                warnings.warn(
                    "Setting required grads for layer: {}, name: {}".format(
                        ".".join(layer), name
                    ),
                    stacklevel=1,
                )
                param.requires_grad = True
    return layer_modules


# pyre-fixme[3]: Return type must be annotated.
def _progress_bar_constructor(
    influence_inst: "InfluenceFunctionBase",
    inputs_dataset: DataLoader,
    quantities_name: str,
    dataset_name: str = "inputs_dataset",
):
    # Try to determine length of progress bar if possible, with a default
    # of `None`.
    inputs_dataset_len = None
    try:
        inputs_dataset_len = len(inputs_dataset)
    except TypeError:
        warnings.warn(
            f"Unable to determine the number of batches in "
            f"`{dataset_name}`. Therefore, if showing the progress "
            f"of the computation of {quantities_name}, "
            "only the number of batches processed can be "
            "displayed, and not the percentage completion of the computation, "
            "nor any time estimates.",
            stacklevel=1,
        )

    return progress(
        inputs_dataset,
        desc=(
            f"Using {influence_inst.get_name()} to compute {quantities_name}. "
            "Processing batch"
        ),
        total=inputs_dataset_len,
    )


def _params_to_names(params: Iterable[nn.Parameter], model: nn.Module) -> List[str]:
    """
    Given an iterable of parameters, `params` of a model, `model`, returns the names of
    the parameters from the perspective of `model`. This is useful if, given
    parameters for which we do not know the name, want to pass them as a dict
    to a function of those parameters, i.e. `torch.nn.utils._stateless`.
    """
    param_id_to_name = {
        id(param): param_name for (param_name, param) in model.named_parameters()
    }
    return [param_id_to_name[id(param)] for param in params]


def _flatten_params(_params: Tuple[Tensor, ...]) -> Tensor:
    """
    Given a tuple of tensors, which is how Pytorch represents parameters of a model,
    flattens it into a single tensor. This is useful if we want to do matrix operations
    on the parameters of a model, i.e. invert its Hessian, or compute dot-product of
    parameter-gradients. Note that flattening and then passing to standard linear
    algebra operations may not be the most efficient way to perform them.
    """
    return torch.cat([_param.view(-1) for _param in _params])


# pyre-fixme[3]: Return type must be annotated.
def _unflatten_params_factory(
    param_shapes: Union[List[Tuple[int, ...]], Tuple[Tensor, ...]],
):
    """
    returns a function which is the inverse of `_flatten_params`
    """

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _unflatten_params(flattened_params):
        params = []
        offset = 0
        for shape in param_shapes:
            length = 1
            for s in shape:
                length *= s
            params.append(flattened_params[offset : offset + length].view(shape))
            offset += length
        return tuple(params)

    return _unflatten_params


def _influence_batch_intermediate_quantities_influence_function(
    influence_inst: "IntermediateQuantitiesInfluenceFunction",
    test_batch: Tuple[Any, ...],
    train_batch: Tuple[Any, ...],
) -> Tensor:
    """
    computes influence of a test batch on a train batch, for implementations of
    `IntermediateQuantitiesInfluenceFunction`
    """
    return torch.matmul(
        influence_inst.compute_intermediate_quantities(test_batch),
        influence_inst.compute_intermediate_quantities(train_batch).T,
    )


def _influence_helper_intermediate_quantities_influence_function(
    influence_inst: "IntermediateQuantitiesInfluenceFunction",
    inputs_dataset: Union[Tuple[Any, ...], DataLoader],
    show_progress: bool,
) -> Tensor:
    """
    Helper function that computes influence scores for implementations of
    `NaiveInfluenceFunction` which implement the `compute_intermediate_quantities`
    method returning "embedding" vectors, so that the influence score of one example
    on another is the dot-product of their vectors.
    """
    # If `inputs_dataset` is not a `DataLoader`, turn it into one.
    inputs_dataset = _format_inputs_dataset(inputs_dataset)

    inputs_intermediate_quantities = influence_inst.compute_intermediate_quantities(
        inputs_dataset,
        show_progress=show_progress,
        test=True,
    )

    train_dataloader = influence_inst.train_dataloader
    if show_progress:
        train_dataloader = _progress_bar_constructor(
            influence_inst, train_dataloader, "train_dataset", "influence scores"
        )

    return torch.cat(
        [
            torch.matmul(
                inputs_intermediate_quantities,
                influence_inst.compute_intermediate_quantities(batch).T,
            )
            for batch in train_dataloader
        ],
        dim=1,
    )


def _self_influence_helper_intermediate_quantities_influence_function(
    influence_inst: "IntermediateQuantitiesInfluenceFunction",
    inputs_dataset: Optional[Union[Tuple[Any, ...], DataLoader]],
    show_progress: bool,
) -> Tensor:
    """
    Helper function that computes self-influence scores for implementations of
    `NaiveInfluenceFunction` which implement the `compute_intermediate_quantities`
    method returning "embedding" vectors, so that the self-influence score of an
    example is the squared norm of its vector.
    """

    inputs_dataset = (
        inputs_dataset
        if inputs_dataset is not None
        else influence_inst.train_dataloader
    )

    # If `inputs_dataset` is not a `DataLoader`, turn it into one.
    inputs_dataset = _format_inputs_dataset(inputs_dataset)

    if show_progress:
        inputs_dataset = _progress_bar_constructor(
            influence_inst, inputs_dataset, "inputs_dataset", "self influence scores"
        )

    return torch.cat(
        [
            torch.sum(
                influence_inst.compute_intermediate_quantities(
                    batch,
                    show_progress=False,
                )
                ** 2,
                dim=1,
            )
            for batch in inputs_dataset
        ]
    )


# pyre-fixme[3]: Return type must be annotated.
def _eig_helper(H: Tensor):
    """
    wrapper around `torch.linalg.eig` that sorts eigenvalues / eigenvectors by
    ascending eigenvalues, like `torch.linalg.eigh`, and returns the real component
    (since `H` is never complex, there should never be a complex component. however,
    `torch.linalg.eig` always returns a complex tensor, which in this case would
    actually have no complex component)
    """
    ls, vs = torch.linalg.eig(H)
    ls, vs = ls.real, vs.real

    ls_argsort = torch.argsort(ls)
    vs = vs[:, ls_argsort]
    ls = ls[ls_argsort]
    return ls, vs


def _top_eigen(
    H: Tensor, k: Optional[int], hessian_reg: float, hessian_inverse_tol: float
) -> Tuple[Tensor, Tensor]:
    """
    This is a wrapper around `torch.linalg.eig` that performs some pre /
    post-processing to make it suitable for computing the low-rank
    "square root" of a matrix, i.e. given square matrix H, find tall and
    skinny L such that LL' approximates H. This function returns eigenvectors (as the
    columns of a matrix Q) and corresponding eigenvectors (as diagonal entries in
    a matrix V), and we can then let L=QV^{1/2}Q'.  However, doing so requires the
    eigenvalues in V to be positive.  Thus, this function does pre-processing (adds
    an entry to the diagonal of H) and post-processing (returns only the top-k
    eigenvectors / eigenvalues where the eigenvalues are above a positive tolerance)
    to encourage and guarantee, respectively, that the returned eigenvalues be
    positive.  The pre-processing shifts the eigenvalues up by a constant, and the
    post-processing effectively replaces H with the most similar matrix (in terms of
    Frobenius norm) whose eigenvalues are above the tolerance, see
    https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/.

    Args:
        H (Tensor): a 2D square Tensor for which the top eigenvectors / eigenvalues
                will be computed.
        k (int): how many eigenvectors / eigenvalues to return (before dropping pairs
                whose eigenvalue is below the tolerance).
        hessian_reg (float): We add an entry to the diagonal of `H` to encourage it to
                be positive definite. This is that entry.
        hessian_inverse_tol (float): To compute the "square root" of `H` using the top
                eigenvectors / eigenvalues, the eigenvalues should be positive, and
                furthermore if above a tolerance, the inversion will be more
                numerically stable. Therefore, we only return eigenvectors /
                eigenvalues where the eigenvalue is above a tolerance. This argument
                specifies that tolerance.

    Returns:
        (eigenvalues, eigenvectors) (tuple of tensors): Mimicking the output of
                `torch.linalg.eigh`, `eigenvalues` is a 1D tensor of the top-k
                eigenvalues of the regularized `H` that are additionally above
                `hessian_inverse_tol`, and `eigenvectors` is a 2D tensor whose columns
                contain the corresponding eigenvectors. The eigenvalues are in
                ascending order.
    """
    # add regularization to hopefully make H positive definite
    H = H + (torch.eye(len(H)).to(device=H.device) * hessian_reg)

    # find eigvectors / eigvals of H
    # ls are eigenvalues, in ascending order
    # columns of vs are corresponding eigenvectors
    ls, vs = _eig_helper(H)

    # despite adding regularization to the hessian, it may still not be positive
    # definite. we can get rid of negative eigenvalues, but for numerical stability
    # can get rid of eigenvalues below a tolerance
    keep = ls > hessian_inverse_tol

    ls = ls[keep]
    vs = vs[:, keep]

    # only keep the top `k` eigvals / eigvectors
    if not (k is None):
        ls = ls[-k:]
        vs = vs[:, -k:]

    # `torch.linalg.eig` is not deterministic in that you can multiply an eigenvector
    # by -1, and it is still an eigenvector. to make eigenvectors deterministic,
    # we multiply an eigenvector according to some rule that flips if you multiply
    # the eigenvector by -1. in this case, that rule is whether the sum of the
    # entries of the eigenvector are > 0
    rule = torch.sum(vs, dim=0) > 0  # entries are 0/1
    rule_multiplier = (2 * rule) - 1  # entries are -1/1
    vs = vs * rule_multiplier.unsqueeze(0)

    return ls, vs


class KMostInfluentialResults(NamedTuple):
    """
    This namedtuple stores the results of using the `influence` method. This method
    is implemented by all subclasses of `TracInCPBase` to calculate
    proponents / opponents. The `indices` field stores the indices of the
    proponents / opponents for each example in the test batch. For example, if finding
    opponents, `indices[i][j]` stores the index in the training data of the example
    with the `j`-th highest influence score on the `i`-th example in the test batch.
    Similarly, the `influence_scores` field stores the actual influence scores, so that
    `influence_scores[i][j]` is the influence score of example `indices[i][j]` in the
    training data on example `i` of the test batch. Please see `TracInCPBase.influence`
    for more details.
    """

    indices: Tensor
    influence_scores: Tensor


def _influence_route_to_helpers(
    influence_instance: Union["TracInCPBase", "InfluenceFunctionBase"],
    inputs: Union[Tuple[Any, ...], DataLoader],
    k: Optional[int] = None,
    proponents: bool = True,
    # pyre-fixme[2]: Parameter must be annotated.
    **kwargs,
) -> Union[Tensor, KMostInfluentialResults]:
    """
    This is a helper function called by `TracInCPBase` and `InfluenceFunctionBase`
    implementations. Those methods share a common logic in that they assume
    an instance of their respective classes implement 2 private methods
    (``_influence`, `_get_k_most_influential`), and the logic of
    which private method to call is common, as described in the documentation of the
    `influence` method. The arguments and return values of this function are the exact
    same as the `influence` method. Note that `influence_instance` refers to the
    instance for which the `influence` method was called.
    """
    if k is None:
        return influence_instance._influence(inputs, **kwargs)
    else:
        return influence_instance._get_k_most_influential(
            inputs,
            k,
            proponents,
            **kwargs,
        )


def _parameter_dot(
    params_1: Tuple[Tensor, ...], params_2: Tuple[Tensor, ...]
) -> Tensor:
    """
    returns the dot-product of 2 tensors, represented as tuple of tensors.
    """
    return torch.tensor(
        sum(
            torch.sum(param_1 * param_2)
            for (param_1, param_2) in zip(params_1, params_2)
        )
    )


def _parameter_add(
    params_1: Tuple[Tensor, ...], params_2: Tuple[Tensor, ...]
) -> Tuple[Tensor, ...]:
    """
    returns the sum of 2 tensors, represented as tuple of tensors.
    """
    return tuple(param_1 + param_2 for (param_1, param_2) in zip(params_1, params_2))


def _parameter_multiply(
    params: Tuple[Tensor, ...], c: Union[Tensor, float]
) -> Tuple[Tensor, ...]:
    """
    multiplies all tensors in a tuple of tensors by a given scalar
    """
    return tuple(param * c for param in params)


# pyre-fixme[2]: Parameter must be annotated.
def _parameter_to(params: Tuple[Tensor, ...], **to_kwargs) -> Tuple[Tensor, ...]:
    """
    applies the `to` method to all tensors in a tuple of tensors
    """
    return tuple(param.to(**to_kwargs) for param in params)


def _parameter_linear_combination(
    paramss: List[Tuple[Tensor, ...]], cs: Tensor
) -> Tuple[Tensor, ...]:
    """
    scales each parameter (tensor of tuples) in a list by the corresponding scalar in a
    1D tensor of the same length, and sums up the scaled parameters
    """
    assert len(cs.shape) == 1
    result = _parameter_multiply(paramss[0], cs[0])
    for params, c in zip(paramss[1:], cs[1:]):
        result = _parameter_add(result, _parameter_multiply(params, c))
    return result


def _compute_jacobian_sample_wise_grads_per_batch(
    influence_inst: Union["TracInCP", "InfluenceFunctionBase"],
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = "none",
) -> Tuple[Tensor, ...]:
    """
    `TracInCP`, `InfluenceFunction`, and `ArnoldiInfluenceFunction` all compute
    jacobians, depending on their `sample_wise_grads_per_batch` attribute. this helper
    wraps that logic.
    """

    if influence_inst.sample_wise_grads_per_batch:
        return _compute_jacobian_wrt_params_with_sample_wise_trick(
            influence_inst.model,
            inputs,
            targets,
            loss_fn,
            reduction_type,
            influence_inst.layer_modules,
        )
    return _compute_jacobian_wrt_params(
        influence_inst.model,
        inputs,
        targets,
        loss_fn,
        influence_inst.layer_modules,
    )


# pyre-fixme[3]: Return type must be annotated.
def _compute_batch_loss_influence_function_base(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]],
    input: Any,
    target: Any,
    reduction_type: str,
):
    """
    In implementations of `InfluenceFunctionBase`, we need to compute the total loss
    for a batch given `loss_fn`, whose reduction can either be 'none', 'sum', or
    'mean', and whose output requires different scaling based on the reduction. This
    helper houses that common logic, and returns the total loss for a batch given the
    predictions (`inputs`) and labels (`targets`) for it. We compute the total loss
    in order to compute the Hessian.
    """
    if loss_fn is not None:
        _loss = loss_fn(input, target)
    else:
        # following convention of `_compute_jacobian_wrt_params`, is no loss function is
        # provided, the quantity backpropped is the output of the forward function.
        assert reduction_type == "none"
        _loss = input

    if reduction_type == "none":
        # if loss_fn is a "reduction='none'" loss function, need to sum
        # up the per-example losses.
        return torch.sum(_loss)
    elif reduction_type == "mean":
        # in this case, we want the total loss for the batch, and should
        # multiply the mean loss for the batch by the batch size. however,
        # we can only infer the batch size if `_output` is a Tensor, and
        # we assume the 0-th dimension to be the batch dimension.
        if isinstance(input, Tensor):
            multiplier = input.shape[0]
        else:
            multiplier = 1
            msg = (
                "`loss_fn` was inferred to behave as a `reduction='mean'` "
                "loss function. however, the batch size of batches could not "
                "be inferred. therefore, the total loss of a batch, which is "
                "needed to compute the Hessian, is approximated as the output "
                "of `loss_fn` for the batch. if this approximation is not "
                "accurate, please change `loss_fn` to behave as a "
                "`reduction='sum'` loss function, or a `reduction='none'` "
                "and set `sample_grads_per_batch` to false."
            )
            warnings.warn(
                msg,
                stacklevel=1,
            )
        return _loss * multiplier
    elif reduction_type == "sum":
        return _loss
    else:
        # currently, only support `reduction_type` to be
        # 'none', 'sum', or 'mean' for
        # `InfluenceFunctionBase` implementations
        raise Exception


# pyre-fixme[2]: Parameter must be annotated.
def _set_attr(obj, names, val) -> None:
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)


# pyre-fixme[2]: Parameter must be annotated.
def _del_attr(obj, names) -> None:
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _model_make_functional(model, param_names, params):
    params = tuple([param.detach().requires_grad_() for param in params])

    for param_name in param_names:
        _del_attr(model, param_name.split("."))

    return params


# pyre-fixme[2]: Parameter must be annotated.
def _model_reinsert_params(model, param_names, params, register: bool = False) -> None:
    for param_name, param in zip(param_names, params):
        _set_attr(
            model,
            param_name.split("."),
            torch.nn.Parameter(param) if register else param,
        )


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _custom_functional_call(model, d, features):
    param_names, params = zip(*list(d.items()))
    _params = _model_make_functional(model, param_names, params)
    _model_reinsert_params(model, param_names, params)
    out = model(*features)
    _model_reinsert_params(model, param_names, _params, register=True)
    return out


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _functional_call(model: Module, d: Dict[str, Tensor], features):
    """
    Makes a call to `model.forward`, which is treated as a function of the parameters
    in `d`, a dict from parameter name to parameter, instead of as a function of
    `features`, the argument that is unpacked to `model.forward` (i.e.
    `model.forward(*features)`).  Depending on what version of PyTorch is available,
    we either use our own implementation, or directly use `torch.nn.utils.stateless`
    or `torch.func.functional_call`.  Put another way, this function mimics the latter
    two implementations, using our own when the PyTorch version is too old.
    """
    import torch

    version = parse_version(torch.__version__)
    if version < (1, 12, 0):
        return _custom_functional_call(model, d, features)
    elif version >= (1, 12, 0) and version < (2, 0, 0):
        import torch.nn.utils.stateless

        return torch.nn.utils.stateless.functional_call(model, d, features)
    else:
        import torch.func

        return torch.func.functional_call(model, d, features)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _dataset_fn(dataloader, batch_fn, reduce_fn, *batch_fn_args, **batch_fn_kwargs):
    """
    Applies `batch_fn` to each batch in `dataloader`, reducing the results using
    `reduce_fn`.  This is useful for computing Hessians and Hessian-vector
    products over an entire dataloader, and is used by both `NaiveInfluenceFunction`
    and `ArnoldiInfluenceFunction`.
    """
    _dataloader = iter(dataloader)

    # pyre-fixme[53]: Captured variable `batch_fn` is not annotated.
    # pyre-fixme[53]: Captured variable `reduce_fn` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    def _reduce_fn(_result, _batch):
        return reduce_fn(_result, batch_fn(_batch, *batch_fn_args, **batch_fn_kwargs))

    result = batch_fn(next(_dataloader), *batch_fn_args, **batch_fn_kwargs)
    return reduce(_reduce_fn, _dataloader, result)
