# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import functools
from abc import abstractmethod
from operator import add
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from captum._utils.gradient import _extract_parameters_from_layers
from captum.influence._core.influence import DataInfluence
from captum.influence._utils.common import (
    _check_loss_fn,
    _compute_batch_loss_influence_function_base,
    _compute_jacobian_sample_wise_grads_per_batch,
    _dataset_fn,
    _flatten_params,
    _format_inputs_dataset,
    _functional_call,
    _get_k_most_influential_helper,
    _influence_batch_intermediate_quantities_influence_function,
    _influence_helper_intermediate_quantities_influence_function,
    _influence_route_to_helpers,
    _load_flexible_state_dict,
    _params_to_names,
    _progress_bar_constructor,
    _self_influence_helper_intermediate_quantities_influence_function,
    _set_active_parameters,
    _top_eigen,
    _unflatten_params_factory,
    KMostInfluentialResults,
)
from captum.log import log_usage
from torch import device, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class InfluenceFunctionBase(DataInfluence):
    r"""
    `InfluenceFunctionBase` is a base class for implementations which compute the
    influence score as defined in the paper "Understanding Black-box Predictions via
    Influence Functions" (https://arxiv.org/pdf/1703.04730.pdf). This "infinitesimal"
    influence score approximately answers the question if a given training example
    were infinitesimally down-weighted and the model re-trained to optimality, how much
    would the loss on a given test example change. Mathematically, the aforementioned
    influence score is given by :math`\nabla_\theta L(x)' H^{-1} \nabla_\theta L(z)`,
    where :math`\nabla_\theta L(x)` is the gradient of the loss, considering only
    training example :math`x` with respect to (a subset of) model parameters
    :math`\theta`, :math`\nabla_\theta L(z)` is the analogous quantity for a test
    example :math`z`, and :math`H` is the Hessian of the (subset of) model parameters
    at a given model checkpoint. "Subset of model parameters" refers to the parameters
    specified by the `layers` initialization argument; for computational purposes,
    we may only consider the gradients / Hessian involving parameters in a subset of
    the model's layers. This is a commonly-taken approach in the research literature.

    There can be multiple implementations of this class, because although the paper
    defines a particular "infinitesimal" kind of influence score, there can be multiple
    ways to compute it, each with different levels of accuracy / scalability.
    """

    def __init__(
        self,
        model: Module,
        train_dataset: Union[Dataset, DataLoader],
        checkpoint: str,
        checkpoints_load_func: Callable[
            [Module, str], float
        ] = _load_flexible_state_dict,
        layers: Optional[List[str]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        hessian_dataset: Optional[Union[Dataset, DataLoader]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
    ) -> None:
        """
        Args:
            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            train_dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader):
                    In the `influence` method, we either compute the influence score of
                    training examples on examples in a test batch, or self influence
                    scores for those training examples, depending on which mode is used.
                    This argument represents the training dataset containing those
                    training examples. In order to compute those influence scores, we
                    will create a Pytorch DataLoader yielding batches of training
                    examples that is then used for processing. If this argument is
                    already a Pytorch Dataloader, that DataLoader can be directly
                    used for processing. If it is instead a Pytorch Dataset, we will
                    create a DataLoader using it, with batch size specified by
                    `batch_size`. For efficiency purposes, the batch size of the
                    DataLoader used for processing should be as large as possible, but
                    not too large, so that certain intermediate quantities created
                    from a batch still fit in memory. Therefore, if
                    `train_dataset` is a Dataset, `batch_size` should be large.
                    If `train_dataset` was already a DataLoader to begin with,
                    it should have been constructed to have a large batch size. It is
                    assumed that the Dataloader (regardless of whether it is created
                    from a Pytorch Dataset or not) yields tuples. For a `batch` that is
                    yielded, of length `L`, it is assumed that the forward function of
                    `model` accepts `L-1` arguments, and the last element of `batch` is
                    the label. In other words, `model(*batch[:-1])` gives the output of
                    `model`, and `batch[-1]` are the labels for the batch.
            checkpoint (str): The path to the checkpoint used to compute influence
                    scores.
            checkpoints_load_func (Callable, optional): The function to load a saved
                    checkpoint into a model to update its parameters, and get the
                    learning rate if it is saved. By default uses a utility to load a
                    model saved as a state dict.
                    Default: _load_flexible_state_dict
            layers (list[str] or None, optional): A list of layer names for which
                    gradients should be computed. If `layers` is None, gradients will
                    be computed for all layers. Otherwise, they will only be computed
                    for the layers specified in `layers`.
                    Default: None
            loss_fn (Callable, optional): The loss function applied to model. There
                    are two options for the return type of `loss_fn`. First, `loss_fn`
                    can be a "per-example" loss function - returns a 1D Tensor of
                    losses for each example in a batch. `nn.BCELoss(reduction="none")`
                    would be an "per-example" loss function. Second, `loss_fn` can be
                    a "reduction" loss function that reduces the per-example losses,
                    in a batch, and returns a single scalar Tensor. For this option,
                    the reduction must be the *sum* or the *mean* of the per-example
                    losses. For instance, `nn.BCELoss(reduction="sum")` is acceptable.
                    Note for the first option, the `sample_wise_grads_per_batch`
                    argument must be False, and for the second option,
                    `sample_wise_grads_per_batch` must be True.  Also note that for
                    the second option, if `loss_fn` has no "reduction" attribute,
                    the implementation assumes that the reduction is the *sum* of the
                    per-example losses.  If this is not the case, i.e. the reduction
                    is the *mean*, please set the "reduction" attribute of `loss_fn`
                    to "mean", i.e. `loss_fn.reduction = "mean"`.
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `train_dataset` and `hessian_dataset`, if they are
                    of type `Dataset`. `batch_size` should be chosen as large as
                    possible so that a backwards pass on a batch still fits in memory.
                    If `train_dataset` and `hessian_dataset`are both of type
                    `DataLoader`, then `batch_size` is ignored as an argument.
                    Default: 1
            hessian_dataset (Dataset or Dataloader, optional): The influence score and
                    self-influence scores this implementation calculates are defined in
                    terms of the Hessian, i.e. the second-derivative of the model
                    parameters. This argument provides the dataset used for calculating
                    the Hessian. It should be smaller than `train_dataset`, which
                    is the dataset whose examples we want the influence of. If not
                    provided or none, it will be assumed to be the same as
                    `train_dataset`.
                    Default: None
            test_loss_fn (Callable, optional): In some cases, one may want to use a
                    separate loss functions for training examples, i.e. those in
                    `train_dataset`, and for test examples, i.e. those
                    represented by the `inputs` and `targets` arguments to the
                    `influence` method. For example, if one wants to calculate the
                    influence score of a training example on a test example's
                    prediction for a fixed class, `test_loss_fn` could map from the
                    logits for all classes to the logits for a fixed class.
                    `test_loss_fn` needs satisfy the same constraints as `loss_fn`.
                    Thus, the same checks that we apply to `loss_fn` are also applied
                    to `test_loss_fn`, if the latter is provided. Note that the
                    constraints on both `loss_fn` and `test_loss_fn` both depend on
                    `sample_wise_grads_per_batch`. This means `loss_fn` and
                    `test_loss_fn` must either both be "per-example"  loss functions,
                    or both be "reduction" loss functions. If not provided, the loss
                    function for test examples is assumed to be the same as the loss
                    function for training examples, i.e. `loss_fn`.
                    Default: None
            sample_wise_grads_per_batch (bool, optional): PyTorch's native gradient
                    computations w.r.t. model parameters aggregates the results for a
                    batch and does not allow to access sample-wise gradients w.r.t.
                    model parameters. This forces us to iterate over each sample in
                    the batch if we want sample-wise gradients which is computationally
                    inefficient. We offer an implementation of batch-wise gradient
                    computations w.r.t. to model parameters which is computationally
                    more efficient. This implementation can be enabled by setting the
                    `sample_wise_grad_per_batch` argument to `True`, and should be
                    enabled if and only if the `loss_fn` argument is a "reduction" loss
                    function. For example, `nn.BCELoss(reduction="sum")` would be a
                    valid `loss_fn` if this implementation is enabled (see
                    documentation for `loss_fn` for more details). Note that our
                    current implementation enables batch-wise gradient computations
                    only for a limited number of PyTorch nn.Modules: Conv2D and Linear.
                    This list will be expanded in the near future.  Therefore, please
                    do not enable this implementation if gradients will be computed
                    for other kinds of layers.
                    Default: False
        """

        self.model = model

        self.checkpoint = checkpoint

        self.checkpoints_load_func = checkpoints_load_func
        # actually load the checkpoint
        checkpoints_load_func(model, checkpoint)
        self.loss_fn = loss_fn
        # If test_loss_fn not provided, it's assumed to be same as loss_fn
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_loss_fn = loss_fn if test_loss_fn is None else test_loss_fn
        self.sample_wise_grads_per_batch = sample_wise_grads_per_batch
        self.batch_size = batch_size

        if not isinstance(train_dataset, DataLoader):
            assert isinstance(batch_size, int), (
                "since the `train_dataset` argument was a `Dataset`, "
                "`batch_size` must be an int."
            )
            # pyre-fixme[4]: Attribute must be annotated.
            self.train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)
        else:
            self.train_dataloader = train_dataset

        if hessian_dataset is None:
            # pyre-fixme[4]: Attribute must be annotated.
            self.hessian_dataloader = self.train_dataloader
        elif not isinstance(hessian_dataset, DataLoader):
            assert isinstance(batch_size, int), (
                "since the `shared_dataset` argument was a `Dataset`, "
                "`batch_size` must be an int."
            )
            self.hessian_dataloader = DataLoader(
                hessian_dataset, batch_size, shuffle=False
            )
        else:
            self.hessian_dataloader = hessian_dataset

        # we check the loss functions in `InfluenceFunctionBase` rather than
        # individually in its child classes because we assume all its implementations
        # have the same requirements on loss functions, i.e. the type of reductions
        # supported. furthermore, these checks are done using a helper function that
        # handles all implementations with a `sample_wise_grads_per_batch`
        # initialization argument.

        # we save the reduction type for both `loss_fn` and `test_loss_fn` because
        # 1) if `sample_wise_grads_per_batch` is true, the reduction type is needed
        # to compute per-example gradients, and 2) regardless, reduction type for
        # `loss_fn` is needed to compute the Hessian.

        # check `loss_fn`
        self.reduction_type: str = _check_loss_fn(
            self, loss_fn, "loss_fn", sample_wise_grads_per_batch
        )
        # check `test_loss_fn` if it was provided
        self.test_reduction_type: str = ""
        if not (test_loss_fn is None):
            self.test_reduction_type = _check_loss_fn(
                self, test_loss_fn, "test_loss_fn", sample_wise_grads_per_batch
            )
        else:
            self.test_reduction_type = self.reduction_type

        self.layer_modules: Optional[List[Module]] = None
        if not (layers is None):
            self.layer_modules = _set_active_parameters(model, layers)

    @abstractmethod
    def self_influence(
        self,
        inputs_dataset: Optional[Union[Tuple[Any, ...], DataLoader]] = None,
        show_progress: bool = False,
    ) -> Tensor:
        """
        Computes self influence scores for the examples in `inputs_dataset`, which is
        either a single batch or a Pytorch `DataLoader` that yields batches. Therefore,
        the computed self influence scores are *not* for the examples in training
        dataset `train_dataset` (unlike when computing self influence scores using the
        `influence` method). Note that if `inputs_dataset` is a single batch, this
        will call `model` on that single batch, and if `inputs_dataset` yields
        batches, this will call `model` on each batch that is yielded. Therefore,
        please ensure that for both cases, the batch(es) that `model` is called
        with are not too large, so that there will not be an out-of-memory error.

        Args:
            inputs_dataset (tuple or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`,
                    and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset`.
            show_progress (bool, optional): Computation of self influence scores can
                    take a long time if `inputs_dataset` represents many examples. If
                    `show_progress` is true, the progress of this computation will be
                    displayed. In more detail, this computation will iterate over all
                    checkpoints (provided as the `checkpoints` initialization argument)
                    in an outer loop, and iterate over all batches that
                    `inputs_dataset` represents in an inner loop. Therefore, the
                    total number of (checkpoint, batch) combinations that need to be
                    iterated over is
                    (# of checkpoints x # of batches that `inputs_dataset` represents).
                    If `show_progress` is True, the total progress of both the outer
                    iteration over checkpoints and the inner iteration over batches is
                    displayed. It will try to use tqdm if available for advanced
                    features (e.g. time estimation). Otherwise, it will fallback to a
                    simple output of progress.
                    Default: False

        Returns:
            self_influence_scores (Tensor): This is a 1D tensor containing the self
                    influence scores of all examples in `inputs_dataset`, regardless of
                    whether it represents a single batch or a `DataLoader` that yields
                    batches.
        """
        pass

    @abstractmethod
    def _get_k_most_influential(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        k: int = 5,
        proponents: bool = True,
        show_progress: bool = False,
    ) -> KMostInfluentialResults:
        r"""
        Args:

            inputs (tuple): `inputs` is the test batch and is a tuple of
                    any, where the last element is assumed to be the labels for the
                    batch. That is, `model(*batch[0:-1])` produces the output for
                    `model`, and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset` - please see its documentation in `__init__` for
                    more details on the assumed structure of a batch.
            k (int, optional): The number of proponents or opponents to return per test
                    example.
                    Default: 5
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`)
                    Default: True
            show_progress (bool, optional): To compute the proponents (or opponents)
                    for the batch of examples, we perform computation for each batch in
                    training dataset `train_dataset`, If `show_progress` is
                    true, the progress of this computation will be displayed. In
                    particular, the number of batches for which the computation has
                    been performed will be displayed. It will try to use tqdm if
                    available for advanced features (e.g. time estimation). Otherwise,
                    it will fallback to a simple output of progress.
                    Default: False

        Returns:
            (indices, influence_scores) (namedtuple): `indices` is a torch.long Tensor
                    that contains the indices of the proponents (or opponents) for each
                    test example. Its dimension is `(inputs_batch_size, k)`, where
                    `inputs_batch_size` is the number of examples in `inputs`. For
                    example, if `proponents==True`, `indices[i][j]` is the index of the
                    example in training dataset `train_dataset` with the
                    k-th highest influence score for the j-th example in `inputs`.
                    `indices` is a `torch.long` tensor so that it can directly be used
                    to index other tensors. Each row of `influence_scores` contains the
                    influence scores for a different test example, in sorted order. In
                    particular, `influence_scores[i][j]` is the influence score of
                    example `indices[i][j]` in training dataset `train_dataset`
                    on example `i` in the test dataset represented by `inputs`.
        """
        pass

    @abstractmethod
    def _influence(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        show_progress: bool = False,
    ) -> Tensor:
        r"""
        Args:

            inputs (tuple[Any, ...]): A batch of examples. Does not represent labels,
                    which are passed as `targets`. The assumption is that
                    `model(*inputs)` produces the predictions for the batch.
            targets (Tensor, optional): If computing influence scores on a loss
                    function, these are the labels corresponding to the batch
                    `inputs`.
                    Default: None

        Returns:
            influence_scores (Tensor): Influence scores over the entire
                    training dataset `train_dataset`. Dimensionality is
                    (inputs_batch_size, src_dataset_size). For example:
                    influence_scores[i][j] = the influence score for the j-th training
                    example to the i-th input example.
        """
        pass

    @abstractmethod
    def influence(  # type: ignore[override]
        self,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        inputs: Tuple,
        k: Optional[int] = None,
        proponents: bool = True,
        show_progress: bool = False,
    ) -> Union[Tensor, KMostInfluentialResults]:
        r"""
        This is the key method of this class, and can be run in 2 different modes,
        where the mode that is run depends on the arguments passed to this method:

        - influence score mode: This mode is used if `k` is None. This mode computes
          the influence score of every example in training dataset `train_dataset`
          on every example in the test dataset represented by `inputs`.
        - k-most influential mode: This mode is used if `k` is not None, and an int.
          This mode computes the proponents or opponents of every example in the
          test dataset represented by `inputs`. In particular, for each test example in
          the test dataset, this mode computes its proponents (resp. opponents),
          which are the indices in the training dataset `train_dataset` of the
          training examples with the `k` highest (resp. lowest) influence scores on the
          test example. Proponents are computed if `proponents` is True. Otherwise,
          opponents are computed. For each test example, this method also returns the
          actual influence score of each proponent (resp. opponent) on the test
          example.

        Args:

            inputs (tuple): `inputs` is the test batch and is a tuple of
                    any, where the last element is assumed to be the labels for the
                    batch. That is, `model(*batch[0:-1])` produces the output for
                    `model`, and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset` - please see its documentation in `__init__` for
                    more details on the assumed structure of a batch.
            k (int, optional): If not provided or `None`, the influence score mode will
                    be run. Otherwise, the k-most influential mode will be run,
                    and `k` is the number of proponents / opponents to return per
                    example in the test dataset.
                    Default: None
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`), if running in k-most influential
                    mode.
                    Default: True
            show_progress (bool, optional): For all modes, computation of results
                    requires "training dataset computations": computations for each
                    batch in the training dataset `train_dataset`, which may
                    take a long time. If `show_progress` is true, the progress of
                    "training dataset computations" will be displayed. In particular,
                    the number of batches for which computations have been performed
                    will be displayed. It will try to use tqdm if available for
                    advanced features (e.g. time estimation). Otherwise, it will
                    fallback to a simple output of progress.
                    Default: False

        Returns:
            The return value of this method depends on which mode is run.

            - influence score mode: if this mode is run (`k` is None), returns a 2D
              tensor `influence_scores` of shape `(input_size, train_dataset_size)`,
              where `input_size` is the number of examples in the test dataset, and
              `train_dataset_size` is the number of examples in training dataset
              `train_dataset`. In other words, `influence_scores[i][j]` is the
              influence score of the `j`-th example in `train_dataset` on the `i`-th
              example in the test dataset.
            - k-most influential mode: if this mode is run (`k` is an int), returns
              a namedtuple `(indices, influence_scores)`. `indices` is a 2D tensor of
              shape `(input_size, k)`, where `input_size` is the number of examples in
              the test dataset. If computing proponents (resp. opponents),
              `indices[i][j]` is the index in training dataset `train_dataset` of the
              example with the `j`-th highest (resp. lowest) influence score (out of
              the examples in `train_dataset`) on the `i`-th example in the test
              dataset. `influence_scores` contains the corresponding influence scores.
              In particular, `influence_scores[i][j]` is the influence score of example
              `indices[i][j]` in `train_dataset` on example `i` in the test dataset
              represented by `inputs`.
        """
        pass


class IntermediateQuantitiesInfluenceFunction(InfluenceFunctionBase):
    """
    Implementations of this class all implement the `compute_intermediate_quantities`
    method, which computes the "embedding" vectors for all examples in a test dataset.
    These embedding vectors are assumed to have the following properties:
    - the influence score of one example on another example, as calculated by the
      implementation, is the dot-product of their respective embeddings.
    - the self influence score of an example is the squared norm of its embedding.
    """

    @abstractmethod
    # pyre-fixme[3]: Return type must be annotated.
    def compute_intermediate_quantities(
        self,
        inputs_dataset: Union[Tuple[Any, ...], DataLoader],
        aggregate: bool = False,
        show_progress: bool = False,
        return_on_cpu: bool = True,
        test: bool = False,
    ):
        pass


# pyre-fixme[3]: Return type must be annotated.
def _flatten_forward_factory(
    model: nn.Module,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]],
    reduction_type: str,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    unflatten_fn: Callable,
    param_names: List[str],
):
    """
    Given a model, loss function, reduction type of the loss, function that unflattens
    1D tensor input into a tuple of tensors, the name of each tensor in that tuple,
    each of which represents a parameter of `model`, and returns a factory. The factory
    accepts a batch, and returns a function whose input is the parameters represented
    by `param_names`, and output is the total loss of the model with those parameters,
    calculated on the batch. The parameter input to the returned function is assumed to
    be *flattened* via the inverse of `unflatten_fn`, which takes a tuple of tensors to
    a 1D tensor. This returned function, accepting a single flattened 1D parameter, is
    useful for computing the parameter gradient involving the batch as a 1D tensor, and
    the Hessian involving the batch as a 2D tensor. Both quantities are needed to
    calculate the kind of influence scores returned by implementations of
    `InfluenceFunctionBase`.
    """

    # this is the factory that accepts a batch
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def flatten_forward_factory_given_batch(batch):

        # this is the function that factory returns, which is a function of flattened
        # parameters
        # pyre-fixme[53]: Captured variable `batch` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def flattened_forward(flattened_params):
            # as everywhere else, the all but the last elements of a batch are
            # assumed to correspond to the features, i.e. input to forward function
            features, labels = tuple(batch[0:-1]), batch[-1]

            _output = _functional_call(
                model, dict(zip(param_names, unflatten_fn(flattened_params))), features
            )

            # compute the total loss for the batch, adjusting the output of
            # `loss_fn` based on `reduction_type`
            return _compute_batch_loss_influence_function_base(
                loss_fn, _output, labels, reduction_type
            )

        return flattened_forward

    return flatten_forward_factory_given_batch


# pyre-fixme[3]: Return type must be annotated.
def _compute_dataset_func(
    inputs_dataset: Union[Tuple[Tensor, ...], DataLoader],
    model: Module,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]],
    reduction_type: str,
    layer_modules: Optional[List[Module]],
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    f: Callable,
    show_progress: bool,
    # pyre-fixme[2]: Parameter must be annotated.
    **f_kwargs,
):
    """
    This function is used to compute higher-order functions of a given model's loss
    over a given dataset, using the model's current parameters. For example, that
    higher-order function `f` could be the Hessian, or a Hessian-vector product.
    This function uses the factory returned by `_flatten_forward_factory`, which given
    a batch, returns the loss for the batch as a function of flattened parameters.
    In particular, for each batch in `inputs_dataset`, this function uses the factory
    to obtain `flattened_forward`, which returns the loss for `model`, using the batch.
    `flattened_forward`, as well as the flattened parameters for `model`, are used by
    argument `f`, a higher-order function, to compute a batch-specific quantity.
    For example, `f` could compute the Hessian via `torch.autograd.functional.hessian`,
    or compute a Hessian-vector product via `torch.autograd.functional.hvp`. Additional
    arguments besides `flattened_forward` and the flattened parameters, i.e. the vector
    in Hessian-vector products, can be passed via named arguments.
    """
    # extract the parameters in a tuple
    params = tuple(
        model.parameters()
        if layer_modules is None
        else _extract_parameters_from_layers(layer_modules)
    )

    # construct functions that can flatten / unflatten tensors, and get
    # names of each param in `params`.
    # Both are needed for calling `_flatten_forward_factory`
    _unflatten_params = _unflatten_params_factory(
        tuple([param.shape for param in params])
    )
    param_names = _params_to_names(params, model)

    # prepare factory
    factory_given_batch = _flatten_forward_factory(
        model,
        loss_fn,
        reduction_type,
        _unflatten_params,
        param_names,
    )

    # the function returned by the factor is evaluated at a *flattened* version of
    # params, so need to create that
    flattened_params = _flatten_params(params)

    # define function of a single batch
    # pyre-fixme[53]: Captured variable `factory_given_batch` is not annotated.
    # pyre-fixme[53]: Captured variable `flattened_params` is not annotated.
    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def batch_f(batch):
        flattened_forward = factory_given_batch(batch)  # accepts flattened params
        return f(flattened_forward, flattened_params, **f_kwargs)

    # sum up results of `batch_f`
    if show_progress:
        # pyre-fixme[9]: inputs_dataset has type `Union[DataLoader[typing.Any],
        #  typing.Tuple[Tensor, ...]]`; used as `tqdm[Tensor]`.
        inputs_dataset = tqdm(inputs_dataset, desc="processing `hessian_dataset` batch")

    return _dataset_fn(inputs_dataset, batch_f, add)


def _get_dataset_embeddings_intermediate_quantities_influence_function(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    batch_embeddings_fn: Callable,
    inputs_dataset: DataLoader,
    aggregate: bool,
) -> Tensor:
    """
    given `batch_embeddings_fn`, which produces the embeddings for a given batch,
    returns either the embeddings for an entire dataset (if `aggregate` is false),
    or the sum of the embeddings for an entire dataset (if `aggregate` is true).
    """
    # if aggregate is false, we concatenate the embeddings for all batches
    if not aggregate:
        return torch.cat(
            [batch_embeddings_fn(batch) for batch in inputs_dataset], dim=0
        )
    else:
        # if aggregate is True, we return the sum of all embeddings for all
        # batches. we do this by summing over each batch, and then summing over all
        # batches.
        inputs_dataset_iter = iter(inputs_dataset)

        batch = next(inputs_dataset_iter)
        total_embedding = torch.sum(batch_embeddings_fn(batch), dim=0)

        for batch in inputs_dataset_iter:
            total_embedding += torch.sum(batch_embeddings_fn(batch), dim=0)

        # we unsqueeze because regardless of aggregate, the returned tensor should
        # be 2D.
        return total_embedding.unsqueeze(0)


class NaiveInfluenceFunction(IntermediateQuantitiesInfluenceFunction):
    r"""
    This is a computationally-inefficient implementation that computes the type of
    "infinitesimal" influence scores defined in the paper "Understanding Black-box
    Predictions via Influence Functions" by Koh et al
    (https://arxiv.org/pdf/1703.04730.pdf). The computational bottleneck in computing
    infinitesimal influence scores is computing inverse Hessian-vector products, as can
    be seen from its definition in `InfluenceFunctionBase`. This implementation is
    inefficient / naive in that it explicitly forms the Hessian :math`H`, unlike other
    implementations which compute inverse Hessian-vector products without explicitly
    forming the Hessian. The purpose of this implementation is to have a way to
    generate the "ground-truth" influence scores, to which other implementations,
    which are more efficient but return only approximations of the influence score, can
    be compared.

    This implementation computes a low-rank approximation of the inverse Hessian, i.e.
    a tall and skinny (with width k) matrix :math`R` such that
    :math`H^{-1} \approx RR'`, where k is small. In particular, let :math`L` be the
    matrix of width k whose columns contain the top-k eigenvectors of :math`H`, and let
    :math`V` be the k by k matrix whose diagonals contain the corresponding eigenvalues.
    This implementation lets :math`R=LV^{-1}L'`. Thus, the core computational step is
    computing the top-k eigenvalues / eigenvectors.

    This low-rank approximation is useful for several reasons:
    - It avoids numerical issues associated with inverting small eigenvalues.
    - Since the influence score is given by
      :math`\nabla_\theta L(x)' H^{-1} \nabla_\theta L(z)`, which is approximated by
      :math`(\nabla_\theta L(x)' R) (\nabla_\theta L(z)' R)`, we can compute an
      "influence embedding" for a given example :math`x`, :math`\nabla_\theta L(x)' R`,
      such that the influence score of one example on another is approximately the
      dot-product of their respective embeddings.

    This implementation is "naive" in that it computes the top-k eigenvalues /
    eigenvectors by explicitly forming the Hessian, converting it to a 2D tensor,
    computing its eigenvectors / eigenvalues, and then sorting. See documentation of the
    `_retrieve_projections_naive_influence_function` method for more details.
    """

    def __init__(
        self,
        model: Module,
        train_dataset: Union[Dataset, DataLoader],
        checkpoint: str,
        checkpoints_load_func: Callable[
            [Module, str], float
        ] = _load_flexible_state_dict,
        layers: Optional[List[str]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        hessian_dataset: Optional[Union[Dataset, DataLoader]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: int = 50,
        seed: int = 42,
        hessian_reg: float = 1e-6,
        hessian_inverse_tol: float = 1e-5,
        projection_on_cpu: bool = True,
        show_progress: bool = False,
    ) -> None:
        """
        Args:
            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            train_dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader):
                    In the `influence` method, we either compute the influence score of
                    training examples on examples in a test batch, or self influence
                    scores for those training examples, depending on which mode is used.
                    This argument represents the training dataset containing those
                    training examples. In order to compute those influence scores, we
                    will create a Pytorch DataLoader yielding batches of training
                    examples that is then used for processing. If this argument is
                    already a Pytorch Dataloader, that DataLoader can be directly
                    used for processing. If it is instead a Pytorch Dataset, we will
                    create a DataLoader using it, with batch size specified by
                    `batch_size`. For efficiency purposes, the batch size of the
                    DataLoader used for processing should be as large as possible, but
                    not too large, so that certain intermediate quantities created
                    from a batch still fit in memory. Therefore, if
                    `train_dataset` is a Dataset, `batch_size` should be large.
                    If `train_dataset` was already a DataLoader to begin with,
                    it should have been constructed to have a large batch size. It is
                    assumed that the Dataloader (regardless of whether it is created
                    from a Pytorch Dataset or not) yields tuples. For a `batch` that is
                    yielded, of length `L`, it is assumed that the forward function of
                    `model` accepts `L-1` arguments, and the last element of `batch` is
                    the label. In other words, `model(*batch[:-1])` gives the output of
                    `model`, and `batch[-1]` are the labels for the batch.
            checkpoint (str): The path to the checkpoint used to compute influence
                    scores.
            checkpoints_load_func (Callable, optional): The function to load a saved
                    checkpoint into a model to update its parameters, and get the
                    learning rate if it is saved. By default uses a utility to load a
                    model saved as a state dict.
                    Default: _load_flexible_state_dict
            layers (list[str] or None, optional): A list of layer names for which
                    gradients should be computed. If `layers` is None, gradients will
                    be computed for all layers. Otherwise, they will only be computed
                    for the layers specified in `layers`.
                    Default: None
            loss_fn (Callable, optional): The loss function applied to model. For now,
                    we require it to be a "reduction='none'" loss function. For
                    example, `BCELoss(reduction='none')` would be acceptable, but
                    `BCELoss(reduction='sum')` would not.
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `train_dataset` and `hessian_dataset`, if they are
                    of type `Dataset`. `batch_size` should be chosen as large as
                    possible so that a backwards pass on a batch still fits in memory.
                    If `train_dataset` and `hessian_dataset`are both of type
                    `DataLoader`, then `batch_size` is ignored as an argument.
                    Default: 1
            hessian_dataset (Dataset or Dataloader, optional): The influence score and
                    self-influence scores this implementation calculates are defined in
                    terms of the Hessian, i.e. the second-derivative of the model
                    parameters. This argument provides the dataset used for calculating
                    the Hessian. It should be smaller than `train_dataset`, which
                    is the dataset whose examples we want the influence of. If not
                    provided or none, it will be assumed to be the same as
                    `train_dataset`.
                    Default: None
            test_loss_fn (Callable, optional): In some cases, one may want to use a
                    separate loss functions for training examples, i.e. those in
                    `train_dataset`, and for test examples, i.e. those
                    represented by the `inputs` and `targets` arguments to the
                    `influence` method. For example, if one wants to calculate the
                    influence score of a training example on a test example's
                    prediction for a fixed class, `test_loss_fn` could map from the
                    logits for all classes to the logits for a fixed class.
                    `test_loss_fn` needs satisfy the same constraints as `loss_fn`.
                    Thus, the same checks that we apply to `loss_fn` are also applied
                    to `test_loss_fn`, if the latter is provided. Note that the
                    constraints on both `loss_fn` and `test_loss_fn` both depend on
                    `sample_wise_grads_per_batch`. This means `loss_fn` and
                    `test_loss_fn` must either both be "per-example"  loss functions,
                    or both be "reduction" loss functions. If not provided, the loss
                    function for test examples is assumed to be the same as the loss
                    function for training examples, i.e. `loss_fn`.
                    Default: None
            sample_wise_grads_per_batch (bool, optional): PyTorch's native gradient
                    computations w.r.t. model parameters aggregates the results for a
                    batch and does not allow to access sample-wise gradients w.r.t.
                    model parameters. This forces us to iterate over each sample in
                    the batch if we want sample-wise gradients which is computationally
                    inefficient. We offer an implementation of batch-wise gradient
                    computations w.r.t. to model parameters which is computationally
                    more efficient. This implementation can be enabled by setting the
                    `sample_wise_grad_per_batch` argument to `True`, and should be
                    enabled if and only if the `loss_fn` argument is a "reduction" loss
                    function. For example, `nn.BCELoss(reduction="sum")` would be a
                    valid `loss_fn` if this implementation is enabled (see
                    documentation for `loss_fn` for more details). Note that our
                    current implementation enables batch-wise gradient computations
                    only for a limited number of PyTorch nn.Modules: Conv2D and Linear.
                    This list will be expanded in the near future.  Therefore, please
                    do not enable this implementation if gradients will be computed
                    for other kinds of layers.
                    Default: False
            projection_dim (int, optional): This implementation produces a low-rank
                    approximation of the (inverse) Hessian. This is the rank of that
                    approximation, and also corresponds to the dimension of the
                    "influence embeddings" produced by the
                    `compute_intermediate_quantities` method.
                    Default: 50
            seed (int, optional): This implementation has a source of randomness - the
                    initialization basis to the Arnoldi iteration. This seed is used
                    to make that randomness reproducible.
                    Default: 42
            hessian_reg (float, optional): We add an entry to the hessian's diagonal
                    entries before computing its eigenvalues / eigenvectors.
                    This is that entry.
                    Default: 1e-6
            hessian_inverse_tol: (float) The tolerance to use when computing the
                    pseudo-inverse of the (square root of) hessian.
                    Default: 1e-6
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
                    Default: True
            show_progress (bool, optional): This implementation explicitly computes the
                    Hessian over batches in `hessian_dataloader` (and sums them) which
                    can take a long time. If `show_progress` is true, the number of
                    batches for which the Hessian has been computed will be displayed.
                    It will try to use tqdm if available for advanced features (e.g.
                    time estimation). Otherwise, it will fallback to a simple output of
                    progress.
                    Default: False
        """
        InfluenceFunctionBase.__init__(
            self,
            model,
            train_dataset,
            checkpoint,
            checkpoints_load_func,
            layers,
            loss_fn,
            batch_size,
            hessian_dataset,
            test_loss_fn,
            sample_wise_grads_per_batch,
        )

        self.projection_dim = projection_dim
        torch.manual_seed(seed)  # for reproducibility

        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device: device = next(model.parameters()).device

        self.R: Tensor = self._retrieve_projections_naive_influence_function(
            self.hessian_dataloader,
            projection_on_cpu,
            show_progress,
        )

    def _retrieve_projections_naive_influence_function(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
    ) -> Tensor:
        r"""
        Returns the matrix `R` described in the documentation for
        `NaiveInfluenceFunction`. In short, `R` has the property that
        :math`H^{-1} \approx RR'`, where `H` is the Hessian. Since this is a "naive"
        implementation, it does so by explicitly forming the Hessian, converting
        it to a 2D tensor, and computing its eigenvectors / eigenvalues, before
        filtering out some eigenvalues and then inverting them. The returned matrix
        `R` represents a set of parameters in parameter space. Since the Hessian
        is obtained by first flattening the parameters, each column of `R` corresponds
        to a *flattened* parameter in parameter space.

        Args:
            dataloader (DataLoader): The returned matrix `R` gives a low-rank
                    approximation of the Hessian `H`. This dataloader defines the
                    dataset used to compute the Hessian that is being approximated.
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
            show_progress (bool): Computing the Hessian that is being approximated
                    requires summing up the Hessians computed using different batches,
                    which may take a long time. If `show_progress` is true, the number
                    of batches that have been processed will be displayed. It will try
                    to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
                    progress.

        Returns:
            R (Tensor): Tall and skinny tensor with width `projection_dim`
                    (initialization argument). Each column corresponds to a flattened
                    parameter in parameter-space. `R` has the property that
                    :math`H^{-1} \approx RR'`.
        """
        # compute the hessian using the dataloader. hessian is always computed using
        # the training loss function. H is 2D, with each column / row corresponding to
        # a different parameter. we cannot directly use
        # `torch.autograd.functional.hessian`, because it does not return a 2D tensor.
        # instead, to compute H, we first create a function that accepts *flattened*
        # model parameters (i.e. a 1D tensor), and outputs the loss of `self.model`,
        # using those parameters, aggregated over `dataloader`. this function is then
        # passed to `torch.autograd.functional.hessian`. because its input is 1D, the
        # resulting hessian is 2D, as desired. all this functionality is handled by
        # `_compute_dataset_func`.
        H = _compute_dataset_func(
            dataloader,
            self.model,
            self.loss_fn,
            self.reduction_type,
            self.layer_modules,
            torch.autograd.functional.hessian,
            show_progress,
        )

        # H is approximately `vs @ torch.diag(ls) @ vs.T``, using eigendecomposition
        ls, vs = _top_eigen(
            H, self.projection_dim, self.hessian_reg, self.hessian_inverse_tol
        )

        # if no positive eigenvalues exist, we cannot compute a low-rank
        # approximation of the square root of the hessian H, so raise exception
        if len(ls) == 0:
            raise Exception(
                "Hessian has no positive "
                "eigenvalues, so cannot take its square root."
            )

        # `R` is `vs @ torch.diag(ls ** -0.5)`, since H^{-1} is approximately
        #  `vs @ torch.diag(ls ** -1) @ vs.T`
        # see https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix#Matrix_inverse_via_eigendecomposition # noqa: E501
        # for details, which mentions that discarding small eigenvalues (as done in
        # `_top_eigen`) reduces noisiness of the inverse.
        # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
        ls = (1.0 / ls) ** 0.5
        return (ls.unsqueeze(0) * vs).to(
            device=torch.device("cpu") if projection_on_cpu else self.model_device
        )

    def compute_intermediate_quantities(
        self,
        inputs_dataset: Union[Tuple[Any, ...], DataLoader],
        aggregate: bool = False,
        show_progress: bool = False,
        return_on_cpu: bool = True,
        test: bool = False,
    ) -> Tensor:
        r"""
        Computes "embedding" vectors for all examples in a single batch, or a
        `Dataloader` that yields batches. These embedding vectors are constructed so
        that the influence score of a training example on a test example is simply the
        dot-product of their corresponding vectors. In both cases, a batch should be
        small enough so that a backwards pass for a batch does not lead to
        out-of-memory errors.

        In more detail, the embedding vector for an example `x` is
        :math`\nabla_\theta L(x)' R`, where :math`R` is as defined in this class'
        description. The embeddings for a batch of examples are computed by assembling
        :math`\nabla_\theta L(x)` for all examples `x` in the batch as rows in a 2D
        tensor, and right-multiplying by `R`.

        If `aggregate` is True, the *sum* of the vectors for all examples is returned,
        instead of the vectors for each example. This can be useful for computing the
        influence of a given training example on the total loss over a validation
        dataset, because due to properties of the dot-product, this influence is the
        dot-product of the training example's vector with the sum of the vectors in the
        validation dataset. Also, by doing the sum aggregation within this method as
        opposed to outside of it (by computing all vectors for the validation dataset,
        then taking the sum) allows memory usage to be reduced.

        Args:
            inputs_dataset (Tuple, or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`, and
                    and `batch[-1]` are the labels, if any. Here, `model` is model
                    provided in initialization. This is the same assumption made for
                    each batch yielded by training dataset `train_dataset`.
            aggregate (bool): Whether to return the sum of the vectors for all
                    examples, as opposed to vectors for each example.
            show_progress (bool, optional): Computation of vectors can take a long
                    time if `inputs_dataset` represents many examples. If
                    `show_progress`is true, the progress of this computation will be
                    displayed. In particular, the number of batches for which
                    vectors have been computed will be displayed. It will try to
                    use tqdm if available for advanced features (e.g. time estimation).
                    Otherwise, it will fallback to a simple output of progress.
                    Default: False
            return_on_cpu (bool, optional): Whether to return the vectors on the cpu.
                    If None or False, is set to the device that the model is on.
                    Default: None
            test (bool, optional): Whether to compute the vectors using the loss
                    function `test_loss_fn` provided in initialization (instead of
                    `loss_fn`). This argument does not matter if `test_loss_fn` was
                    not provided, as in this case, `test_loss_fn` and `loss_fn` are the
                    same.

        Returns:
            intermediate_quantities (Tensor): This is a 2D tensor with shape
                    `(N, projection_dim)`, where `N` is the total number of examples in
                    `inputs_dataset`, and `projection_dim` was provided in
                    initialization. Each row contains the vector for a different
                    example.
        """
        # if `inputs_dataset` is not a `DataLoader`, turn it into one.
        inputs_dataset = _format_inputs_dataset(inputs_dataset)

        if show_progress:
            inputs_dataset = _progress_bar_constructor(
                self, inputs_dataset, "inputs_dataset", "intermediate quantities"
            )

        # infer model / data device through model
        return_device: device = (
            torch.device("cpu") if return_on_cpu else self.model_device
        )

        # as described in the description for `NaiveInfluenceFunction`, the embedding
        # for an example `x` is :math`\nabla_\theta L(x)' R`.
        # `_basic_computation_naive_influence_function` returns a 2D tensor where
        # each row is :math`\nabla_\theta L(x)'` for a different example `x` in a
        # batch. therefore, we right-multiply its output with `R` to get the embeddings
        # for a batch, and then concatenate the per-batch embeddings to get embeddings
        # for the entire dataset.

        # choose the correct loss function and reduction type based on `test`
        loss_fn = self.test_loss_fn if test else self.loss_fn
        reduction_type: str = self.test_reduction_type if test else self.reduction_type

        # define a helper function that returns the embeddings for a batch
        # pyre-fixme[53]: Captured variable `loss_fn` is not annotated.
        def get_batch_embeddings(batch: Tuple[Tensor, ...]) -> Tensor:
            # if `self.R` is on cpu, and `self.model_device` was not cpu, this implies
            # `self.R` was too large to fit in gpu memory, and we should do the matrix
            # multiplication of the batch jacobians with `self.R` separately for each
            # column of `self.R`, to avoid moving the entire `self.R` to gpu all at
            # once and running out of gpu memory
            batch_jacobians = _basic_computation_naive_influence_function(
                self, batch[0:-1], batch[-1], loss_fn, reduction_type
            )
            if self.R.device == torch.device(
                "cpu"
            ) and self.model_device != torch.device("cpu"):
                return torch.stack(
                    [
                        torch.matmul(batch_jacobians, R_col.to(batch_jacobians.device))
                        for R_col in self.R.T
                    ],
                    dim=1,
                ).to(return_device)
            else:
                return torch.matmul(batch_jacobians, self.R).to(device=return_device)

        # using `get_batch_embeddings` and a helper, return all the vectors or their
        # sum, depending on `aggregate`
        return _get_dataset_embeddings_intermediate_quantities_influence_function(
            get_batch_embeddings,
            inputs_dataset,
            aggregate,
        )

    @log_usage(part_of_slo=True, skip_self_logging=True)
    def influence(  # type: ignore[override]
        self,
        # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
        inputs: Tuple,
        k: Optional[int] = None,
        proponents: bool = True,
        show_progress: bool = False,
    ) -> Union[Tensor, KMostInfluentialResults]:
        """
        This is the key method of this class, and can be run in 2 different modes,
        where the mode that is run depends on the arguments passed to this method:

        - influence score mode: This mode is used if `k` is None. This mode computes
          the influence score of every example in training dataset `train_dataset`
          on every example in the test batch represented by `inputs`.
        - k-most influential mode: This mode is used if `k` is not None, and an int.
          This mode computes the proponents or opponents of every example in the
          test batch represented by `inputs`. In particular, for each test example in
          the test batch, this mode computes its proponents (resp. opponents),
          which are the indices in the training dataset `train_dataset` of the
          training examples with the `k` highest (resp. lowest) influence scores on the
          test example. Proponents are computed if `proponents` is True. Otherwise,
          opponents are computed. For each test example, this method also returns the
          actual influence score of each proponent (resp. opponent) on the test
          example.

        Args:

            inputs (tuple): `inputs` is the test batch and is a tuple of
                    any, where the last element is assumed to be the labels for the
                    batch. That is, `model(*batch[0:-1])` produces the output for
                    `model`, and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset` - please see its documentation in `__init__` for
                    more details on the assumed structure of a batch.
            k (int, optional): If not provided or `None`, the influence score mode will
                    be run. Otherwise, the k-most influential mode will be run,
                    and `k` is the number of proponents / opponents to return per
                    example in the test batch.
                    Default: None
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`), if running in k-most influential
                    mode.
                    Default: True
            show_progress (bool, optional): For all modes, computation of results
                    requires "training dataset computations": computations for each
                    batch in the training dataset `train_dataset`, which may
                    take a long time. If `show_progress` is true, the progress of
                    "training dataset computations" will be displayed. In particular,
                    the number of batches for which computations have been performed
                    will be displayed. It will try to use tqdm if available for
                    advanced features (e.g. time estimation). Otherwise, it will
                    fallback to a simple output of progress.
                    Default: False

        Returns:
            The return value of this method depends on which mode is run.

            - influence score mode: if this mode is run (`k` is None), returns a 2D
              tensor `influence_scores` of shape `(input_size, train_dataset_size)`,
              where `input_size` is the number of examples in the test dataset, and
              `train_dataset_size` is the number of examples in training dataset
              `train_dataset`. In other words, `influence_scores[i][j]` is the
              influence score of the `j`-th example in `train_dataset` on the `i`-th
              example in the test batch.
            - k-most influential mode: if this mode is run (`k` is an int), returns
              a namedtuple `(indices, influence_scores)`. `indices` is a 2D tensor of
              shape `(input_size, k)`, where `input_size` is the number of examples in
              the test batch. If computing proponents (resp. opponents),
              `indices[i][j]` is the index in training dataset `train_dataset` of the
              example with the `j`-th highest (resp. lowest) influence score (out of
              the examples in `train_dataset`) on the `i`-th example in the test
              batch. `influence_scores` contains the corresponding influence scores.
              In particular, `influence_scores[i][j]` is the influence score of example
              `indices[i][j]` in `train_dataset` on example `i` in the test batch
              represented by `inputs`.
        """

        return _influence_route_to_helpers(
            self,
            inputs,
            k,
            proponents,
            show_progress=show_progress,
        )

    def _get_k_most_influential(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        k: int = 5,
        proponents: bool = True,
        show_progress: bool = False,
    ) -> KMostInfluentialResults:
        r"""
        Args:

            inputs (tuple): `inputs` is the test batch and is a tuple of
                    any, where the last element is assumed to be the labels for the
                    batch. That is, `model(*batch[0:-1])` produces the output for
                    `model`, and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset` - please see its documentation in `__init__` for
                    more details on the assumed structure of a batch.
            k (int, optional): The number of proponents or opponents to return per test
                    example.
                    Default: 5
            proponents (bool, optional): Whether seeking proponents (`proponents=True`)
                    or opponents (`proponents=False`)
                    Default: True
            show_progress (bool, optional): To compute the proponents (or opponents)
                    for the batch of examples, we perform computation for each batch in
                    training dataset `train_dataset`, If `show_progress` is
                    true, the progress of this computation will be displayed. In
                    particular, the number of batches for which the computation has
                    been performed will be displayed. It will try to use tqdm if
                    available for advanced features (e.g. time estimation). Otherwise,
                    it will fallback to a simple output of progress.
                    Default: False

        Returns:
            (indices, influence_scores) (namedtuple): `indices` is a torch.long Tensor
                    that contains the indices of the proponents (or opponents) for each
                    test example. Its dimension is `(inputs_batch_size, k)`, where
                    `inputs_batch_size` is the number of examples in `inputs`. For
                    example, if `proponents==True`, `indices[i][j]` is the index of the
                    example in training dataset `train_dataset` with the
                    k-th highest influence score for the j-th example in `inputs`.
                    `indices` is a `torch.long` tensor so that it can directly be used
                    to index other tensors. Each row of `influence_scores` contains the
                    influence scores for a different test example, in sorted order. In
                    particular, `influence_scores[i][j]` is the influence score of
                    example `indices[i][j]` in training dataset `train_dataset`
                    on example `i` in the test dataset represented by `inputs`.
        """
        desc = (
            None
            if not show_progress
            else (
                (
                    f"Using {self.get_name()} to perform computation for "
                    f'getting {"proponents" if proponents else "opponents"}. '
                    "Processing training batches"
                )
            )
        )
        return KMostInfluentialResults(
            *_get_k_most_influential_helper(
                self.train_dataloader,
                functools.partial(
                    _influence_batch_intermediate_quantities_influence_function, self
                ),
                inputs,
                k,
                proponents,
                show_progress,
                desc,
            )
        )

    def _influence(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        show_progress: bool = False,
    ) -> Tensor:
        r"""
        Args:

            inputs (tuple): `inputs` is the test batch and is a tuple of
                    any, where the last element is assumed to be the labels for the
                    batch. That is, `model(*batch[0:-1])` produces the output for
                    `model`, and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset` - please see its documentation in `__init__` for
                    more details on the assumed structure of a batch.
            show_progress (bool, optional): To compute the influence of examples in
                    training dataset `train_dataset`, we compute the influence
                    of each batch. If `show_progress` is true, the progress of this
                    computation will be displayed. In particular, the number of batches
                    for which influence has been computed will be displayed. It will
                    try to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
                    progress.
                    Default: False

        Returns:
            influence_scores (Tensor): Influence scores over the entire
                    training dataset `train_dataset`. Dimensionality is
                    (inputs_batch_size, src_dataset_size). For example:
                    influence_scores[i][j] = the influence score for the j-th training
                    example to the i-th example in the test dataset.
        """
        # turn inputs and targets into a dataset. inputs has already been processed
        # so that it should always be unpacked
        inputs_dataset = _format_inputs_dataset(inputs)
        return _influence_helper_intermediate_quantities_influence_function(
            self, inputs_dataset, show_progress
        )

    @log_usage(part_of_slo=True, skip_self_logging=True)
    def self_influence(
        self,
        inputs_dataset: Optional[Union[Tuple[Any, ...], DataLoader]] = None,
        show_progress: bool = False,
    ) -> Tensor:
        """
        Computes self influence scores for the examples in `inputs_dataset`, which is
        either a single batch or a Pytorch `DataLoader` that yields batches. Therefore,
        the computed self influence scores are *not* for the examples in training
        dataset `train_dataset` (unlike when computing self influence scores using the
        `influence` method). Note that if `inputs_dataset` is a single batch, this
        will call `model` on that single batch, and if `inputs_dataset` yields
        batches, this will call `model` on each batch that is yielded. Therefore,
        please ensure that for both cases, the batch(es) that `model` is called
        with are not too large, so that there will not be an out-of-memory error.

        Implementation-wise, the self-influence score for an example is simply the
        squared norm of the example's "embedding" vector. Therefore, the implementation
        leverages `compute_intermediate_quantities`.

        Args:
            inputs_dataset (tuple or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`,
                    and `batch[-1]` are the labels, if any. This is the same
                    assumption made for each batch yielded by training dataset
                    `train_dataset`.
                    Default: None
            show_progress (bool, optional): Computation of self influence scores can
                    take a long time if `inputs_dataset` represents many examples. If
                    `show_progress`is true, the progress of this computation will be
                    displayed. In particular, the number of batches for which
                    self influence scores have been computed will be displayed. It will
                    try to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
                    progress.
                    Default: False
        """
        return _self_influence_helper_intermediate_quantities_influence_function(
            self, inputs_dataset, show_progress
        )


def _basic_computation_naive_influence_function(
    influence_inst: InfluenceFunctionBase,
    inputs: Tuple[Any, ...],
    targets: Optional[Tensor] = None,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = None,
) -> Tensor:
    """
    This computes the per-example parameter gradients for a batch, flattened into a
    2D tensor where the first dimension is batch dimension. This is used by
    `NaiveInfluenceFunction` which computes embedding vectors for each example by
    projecting their parameter gradients.
    """
    # `jacobians` contains one tensor for each parameter we compute jacobians for.
    # the first dimension of each tensor is the batch dimension, and the remaining
    # dimensions correspond to the parameter, so that for the tensor corresponding
    # to parameter `p`, its shape is `(batch_size, *p.shape)`
    jacobians = _compute_jacobian_sample_wise_grads_per_batch(
        influence_inst, inputs, targets, loss_fn, reduction_type
    )

    return torch.stack(
        [
            _flatten_params(tuple(jacobian[i] for jacobian in jacobians))
            for i in range(len(next(iter(jacobians))))
        ],
        dim=0,
    )
