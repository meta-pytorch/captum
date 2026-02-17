#!/usr/bin/env python3

# pyre-strict

import glob
import warnings
from abc import abstractmethod
from os.path import join
from typing import (
    Any,
    Callable,
    cast,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

import torch
from captum._utils.av import AV
from captum._utils.progress import NullProgress, progress
from captum.influence._core.influence import DataInfluence
from captum.influence._utils.common import (
    _check_loss_fn,
    _compute_jacobian_sample_wise_grads_per_batch,
    _format_inputs_dataset,
    _get_k_most_influential_helper,
    _gradient_dot_product,
    _influence_route_to_helpers,
    _load_flexible_state_dict,
    _self_influence_by_batches_helper,
    _set_active_parameters,
    KMostInfluentialResults,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


r"""

Note: methods starting with "_" are protected, not private, and can be overridden in
child classes.  They are not part of the API.

Implements abstract DataInfluence class and provides implementation details for
influence computation based on the logic provided in TracIn paper
(https://arxiv.org/abs/2002.08484).

The TracIn paper proposes an idealized notion of influence which can be represented by
the total amount a training example reduces loss for a test example via a training
process such as stochastic gradient descent. As this idealized notion of influence is
impractical to compute, the TracIn paper proposes instead to compute an influence
score, which uses a first-order approximation for the change in loss for a test example
by a training example, which is accumulated across saved model checkpoints. This
influence score is accumulated via a summed dot-product of gradient vectors for the
scores/loss of a test and training example.
"""

"""
TODO: Support for checkpoint type. Currently only supports model parameters as saved
checkpoints. Can use enum or string.

Potential implementation from design doc:
checkpoint_type (Enum = [Parameters | Loss_Grad]): For performance,
                saved / loaded checkpoints can be either model parameters, or
                gradient of the loss function on an input w.r.t parameters.
"""


class TracInCPBase(DataInfluence):
    """
    To implement the `influence` method, classes inheriting from `TracInCPBase` will
    separately implement the private `_self_influence`, `_get_k_most_influential`,
    and `_influence` methods. The public `influence` method is a wrapper for these
    private methods.
    """

    def __init__(
        self,
        model: Module,
        train_dataset: Union[Dataset, DataLoader],
        checkpoints: Union[str, List[str], Iterator[str]],
        checkpoints_load_func: Callable[
            [Module, str], float
        ] = _load_flexible_state_dict,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
    ) -> None:
        r"""
        Args:

            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            train_dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader):
                    In the `influence` method, we compute the influence score of
                    training examples on examples in a test batch.
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
            checkpoints (str, list[str], or Iterator): Either the directory of the
                    path to store and retrieve model checkpoints, a list of
                    filepaths with checkpoints from which to load, or an iterator which
                    returns objects from which to load checkpoints.
            checkpoints_load_func (Callable, optional): The function to load a saved
                    checkpoint into a model to update its parameters, and get the
                    learning rate if it is saved. By default uses a utility to load a
                    model saved as a state dict.
                    Default: _load_flexible_state_dict
            loss_fn (Callable, optional): The loss function applied to model.
                    Default: None
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `train_dataset`, if it is a Dataset.
                    `batch_size` should be chosen as large as possible so that certain
                    intermediate quantities created from a batch still fit in memory.
                    Specific implementations of `TracInCPBase` will detail the size of
                    the intermediate quantities. `batch_size` must be an int if
                    `train_dataset` is a Dataset. If `train_dataset`
                    is a DataLoader, then `batch_size` is ignored as an argument.
                    Default: 1
            test_loss_fn (Callable, optional): In some cases, one may want to use a
                    separate loss functions for training examples, i.e. those in
                    `train_dataset`, and for test examples, i.e. those
                    represented by the `inputs` and `targets` arguments to the
                    `influence` method. For example, if one wants to calculate the
                    influence score of a training example on a test example's
                    prediction for a fixed class, `test_loss_fn` could map from the
                    logits for all classes to the logits for a fixed class.
                    `test_loss_fn` needs to satisfy the same constraints as `loss_fn`.
                    If not provided, the loss function for test examples is assumed to
                    be the same as the loss function for training examples, i.e.
                    `loss_fn`.
                    Default: None
        """

        self.model: Module = model
        self.checkpoints = checkpoints  # type: ignore
        self._checkpoints: List[str] = self.checkpoints
        self.checkpoints_load_func = checkpoints_load_func
        self.loss_fn = loss_fn
        # If test_loss_fn not provided, it's assumed to be same as loss_fn
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_loss_fn = loss_fn if test_loss_fn is None else test_loss_fn
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

        self.train_dataloader_len: Optional[int] = None
        try:

            # since we will calculate the number of batches in
            # `self.train_dataloader` whenever we use progress bar, calculate
            # it once in initialization, for re-use.
            self.train_dataloader_len = len(self.train_dataloader)
        except TypeError:
            warnings.warn(
                "Unable to determine the number of batches in training dataset "
                "`train_dataset`. Therefore, if showing the progress of computations, "
                "only the number of batches processed can be displayed, and not the "
                "percentage completion of the computation, nor any time estimates.",
                stacklevel=1,
            )

    @property
    def checkpoints(self) -> List[str]:
        return self._checkpoints

    @checkpoints.setter
    def checkpoints(self, checkpoints: Union[str, List[str], Iterator[str]]) -> None:
        if isinstance(checkpoints, str):
            self._checkpoints = AV.sort_files(glob.glob(join(checkpoints, "*")))
        elif isinstance(checkpoints, List) and isinstance(checkpoints[0], str):
            self._checkpoints = AV.sort_files(checkpoints)
        else:
            self._checkpoints = list(checkpoints)  # cast to avoid mypy error

        if len(self._checkpoints) <= 0:
            raise ValueError(
                f"Invalid checkpoints provided for TracIn class: {checkpoints}!"
            )

    @abstractmethod
    def self_influence(
        self,
        inputs: Optional[Union[Tuple[Any, ...], DataLoader]] = None,
        show_progress: bool = False,
    ) -> Tensor:
        """
        If `inputs` is not specified calculates the self influence
        scores for the training dataset `train_dataset`. Otherwise, computes
        self influence scores for the examples in `inputs`,
        which is either a single batch or a Pytorch `DataLoader` that yields
        batches. Therefore, in this case, the computed self influence scores
        are *not* for the examples in training dataset `train_dataset`.
        Note that if `inputs` is a single batch, this
        will call `model` on that single batch, and if `inputs` yields
        batches, this will call `model` on each batch that is yielded. Therefore,
        please ensure that for both cases, the batch(es) that `model` is called
        with are not too large, so that there will not be an out-of-memory error.

        Args:
            inputs (tuple or DataLoader, optional): This specifies the
                    dataset for which self influence scores will be computed.
                    Either a single tuple of any, or a `DataLoader`, where each
                    batch yielded is a tuple of type any. In either case, the tuple
                    represents a single batch, where the last element is assumed to
                    be the labels for the batch. That is, `model(*batch[0:-1])`
                    produces the output for `model`, and `batch[-1]` are the labels,
                    if any. This is the same assumption made for each batch yielded
                    by training dataset `train_dataset`. Please see documentation for
                    the `train_dataset` argument to `TracInCP.__init__` for
                    more details on the assumed structure of a batch. If not provided
                    or `None`, self influence scores will be computed for training
                    dataset `train_dataset`, which yields batches satisfying the
                    above assumptions.
                    Default: None.
            show_progress (bool, optional): Computation of self influence scores can
                    take a long time if `inputs` represents many examples. If
                    `show_progress` is true, the progress of this computation will be
                    displayed. In more detail, this computation will iterate over all
                    checkpoints (provided as the `checkpoints` initialization argument)
                    in an outer loop, and iterate over all batches that
                    `inputs` represents in an inner loop. Therefore, the
                    total number of (checkpoint, batch) combinations that need to be
                    iterated over is
                    (# of checkpoints x # of batches that `inputs` represents).
                    If `show_progress` is True, the total progress of both the outer
                    iteration over checkpoints and the inner iteration over batches is
                    displayed. It will try to use tqdm if available for advanced
                    features (e.g. time estimation). Otherwise, it will fallback to a
                    simple output of progress.
                    Default: False

        Returns:
            self_influence_scores (Tensor): This is a 1D tensor containing the self
                    influence scores of all examples in `inputs`, regardless of
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
                    on example `i` in the test batch represented by `inputs`.
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
                    example to the i-th example in the test batch.
        """
        pass

    @abstractmethod
    def influence(  # type: ignore[override]
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        k: Optional[int] = None,
        proponents: bool = True,
        unpack_inputs: bool = True,
        show_progress: bool = False,
    ) -> Union[Tensor, KMostInfluentialResults]:
        r"""
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
            unpack_inputs (bool, optional): If True and `inputs` is a tuple or list,
                    the elements are unpacked as separate positional arguments to the
                    model. If False, `inputs` is wrapped in a tuple and passed as a
                    single argument.
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
              where `input_size` is the number of examples in the test batch, and
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
              dataset. `influence_scores` contains the corresponding influence scores.
              In particular, `influence_scores[i][j]` is the influence score of example
              `indices[i][j]` in `train_dataset` on example `i` in the test batch
              represented by `inputs`.
        """
        pass

    @classmethod
    def get_name(cls: Type["TracInCPBase"]) -> str:
        r"""
        Create readable class name.  Due to the nature of the names of `TracInCPBase`
        subclasses, simplies returns the class name.  For example, for a class called
        TracInCP, we return the string TracInCP.

        Returns:
            name (str): a readable class name
        """
        return cls.__name__


class TracInCP(TracInCPBase):
    def __init__(
        self,
        model: Module,
        train_dataset: Union[Dataset, DataLoader],
        checkpoints: Union[str, List[str], Iterator[str]],
        checkpoints_load_func: Callable[
            [Module, str], float
        ] = _load_flexible_state_dict,
        layers: Optional[List[str]] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        sample_wise_grads_per_batch: bool = False,
    ) -> None:
        r"""
        Args:

            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            train_dataset (torch.utils.data.Dataset or torch.utils.data.DataLoader):
                    In the `influence` method, we compute the influence score of
                    training examples on examples in a test batch.
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
            checkpoints (str, list[str], or Iterator): Either the directory of the
                    path to store and retrieve model checkpoints, a list of
                    filepaths with checkpoints from which to load, or an iterator which
                    returns objects from which to load checkpoints.
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
                    Default: None
            batch_size (int or None, optional): Batch size of the DataLoader created to
                    iterate through `train_dataset`, if it is a Dataset.
                    `batch_size` should be chosen as large as possible so that certain
                    intermediate quantities created from a batch still fit in memory.
                    Specific implementations of `TracInCPBase` will detail the size of
                    the intermediate quantities. `batch_size` must be an int if
                    `train_dataset` is a Dataset. If `train_dataset`
                    is a DataLoader, then `batch_size` is ignored as an argument.
                    Default: 1
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

        TracInCPBase.__init__(
            self,
            model,
            train_dataset,
            checkpoints,
            checkpoints_load_func,
            loss_fn,
            batch_size,
            test_loss_fn,
        )

        self.sample_wise_grads_per_batch = sample_wise_grads_per_batch

        # check `loss_fn`
        self.reduction_type: str = _check_loss_fn(
            self, loss_fn, "loss_fn", sample_wise_grads_per_batch
        )
        # check `test_loss_fn` if it was provided
        self.test_reduction_type: str = (
            self.reduction_type
            if test_loss_fn is None
            else _check_loss_fn(
                self, test_loss_fn, "test_loss_fn", sample_wise_grads_per_batch
            )
        )

        r"""
        TODO: Either restore model state after done (would have to place functionality
        within influence to restore after every influence call)? or make a copy so that
        changes to grad_requires aren't persistent after using TracIn.
        """
        self.layer_modules: Optional[List[Module]] = None
        if layers is not None:
            self.layer_modules = _set_active_parameters(model, layers)

    @log_usage(part_of_slo=True)
    def influence(  # type: ignore[override]
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        k: Optional[int] = None,
        proponents: bool = True,
        show_progress: bool = False,
        aggregate: bool = False,
    ) -> Union[Tensor, KMostInfluentialResults]:
        r"""
        This is the key method of this class, and can be run in 2 different modes,
        where the mode that is run depends on the arguments passed to this method.
        Below, we describe the 2 modes, when `aggregate` is false:

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

        When `aggregate` is True, this method computes "aggregate" influence scores,
        which for a given training example, is the *sum* of its influence scores over
        all examples in the test dataset. Below, we describe the 2 modes, when
        `aggregate` is True:

        - influence score mode: This mode is used if `k` is None. This mode computes
          the aggregate influence score of each example in training dataset
          `train_dataset` on the test dataset.
        - k-most influential mode: This mode is used if `k` is not None, and an int.
          This mode computes the "aggregate" proponents (resp. opponents), which are
          the indices in the training dataset `train_dataset` of the examples with the
          `k` highest (resp. lowest) aggregate influence scores on the test dataset.
          Proponents are computed if `proponents` is True. Otherwise, opponents are
          computed. This method also returns the actual aggregate influence scores
          of each proponent (resp. opponent) on the test dataset.

        Args:

            inputs (Tuple, or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`, and
                    and `batch[-1]` are the labels, if any. Here, `model` is model
                    provided in initialization. This is the same assumption made for
                    each batch yielded by training dataset `train_dataset`. Please see
                    documentation for the `train_dataset` argument to
                    `TracInCPFastRandProj.__init__` for more details on the assumed
                    structure of a batch.
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
            aggregate (bool, optional): If true, return "aggregate" influence scores or
                    examples with the highest / lowest aggregate influence scores on
                    the test dataset, depending on the mode.

        Returns:
            The return value of this method depends on which mode is run, and whether
            `aggregate` is True of False.

            Below are the return values for the 2 modes, when `aggregate` is False:

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

            Below are the return values for the 2 modes, when `aggregate` is True:

            - influence score mode: if this mode is run (`k` is None), returns a 2D
              tensor `influence_scores` of shape `(1, train_dataset_size)`, where
              `influence_scores[0][j] is the aggregate influence score of the `j`-th
              example in `train_dataset` on the test dataset.
            - k-most influential mode: if this mode is run (`k` is an int), returns a
              namedtuple `(indices, influence_scores)`. `indices` is a 2D tensor of
              shape `(1, k)`. If computing proponents (resp. opponents),
              `indices[0][j]` is the index in training dataset `train_dataset` of the
              example with the `j`-th highest (resp. lowest) aggregate influence score
              on the test dataset. `influence_scores` contains the corresponding
              aggregate influence scores. In particular, `influence_scores[0][j]` is
              the aggregate influence score of example `indices[0][j]` on the test
              dataset.
        """

        assert inputs is not None, (
            "`inputs` argument is required."
            "If you wish to calculate self influence scores,"
            " please use the `self_influence` method instead."
        )
        return _influence_route_to_helpers(
            self,
            inputs,
            k,
            proponents,
            show_progress=show_progress,
            aggregate=aggregate,
        )

    def _sum_jacobians(
        self,
        inputs: DataLoader,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        reduction_type: Optional[str] = None,
    ) -> Tuple[Tensor, ...]:
        """
        sums the jacobians of all examples in `inputs`. result is of the
        same format as layer_jacobians, but the batch dimension has size 1
        """
        inputs_iter = iter(inputs)

        inputs_batch = next(inputs_iter)

        def get_batch_contribution(inputs_batch: Tuple[Any, ...]) -> Tuple[Tensor, ...]:
            _input_jacobians = self._basic_computation_tracincp(
                inputs_batch[0:-1],
                inputs_batch[-1],
                loss_fn,
                reduction_type,
            )

            return tuple(
                torch.sum(jacobian, dim=0).unsqueeze(0) for jacobian in _input_jacobians
            )

        inputs_jacobians = get_batch_contribution(inputs_batch)

        for inputs_batch in inputs_iter:
            inputs_batch_jacobians = get_batch_contribution(inputs_batch)
            inputs_jacobians = tuple(
                [
                    inputs_jacobian + inputs_batch_jacobian
                    for (inputs_jacobian, inputs_batch_jacobian) in zip(
                        inputs_jacobians, inputs_batch_jacobians
                    )
                ]
            )

        return inputs_jacobians

    def _concat_jacobians(
        self,
        inputs: DataLoader,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        reduction_type: Optional[str] = None,
    ) -> Tuple[Tensor, ...]:
        all_inputs_batch_jacobians = [
            self._basic_computation_tracincp(
                inputs_batch[0:-1],
                inputs_batch[-1],
                loss_fn,
                reduction_type,
            )
            for inputs_batch in inputs
        ]

        return tuple(
            torch.cat(all_inputs_batch_jacobian, dim=0)
            for all_inputs_batch_jacobian in zip(*all_inputs_batch_jacobians)
        )

    @log_usage(part_of_slo=True)
    def compute_intermediate_quantities(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        aggregate: bool = False,
    ) -> Tensor:
        """
        Computes "embedding" vectors for all examples in a single batch, or a
        `Dataloader` that yields batches. These embedding vectors are constructed so
        that the influence score of a training example on a test example is simply the
        dot-product of their corresponding vectors. Allowing a `DataLoader`
        yielding batches to be passed in (as opposed to a single batch) gives the
        potential to improve efficiency, because we load each checkpoint only once in
        this method call. Thus if a `DataLoader` yielding batches is passed in, this
        reduces the total number of times each checkpoint is loaded for a dataset,
        compared to if a single batch is passed in. The reason we do not just increase
        the batch size is that for large models, large batches do not fit in memory.

        If `aggregate` is True, the *sum* of the vectors for all examples is returned,
        instead of the vectors for each example. This can be useful for computing the
        influence of a given training example on the total loss over a validation
        dataset, because due to properties of the dot-product, this influence is the
        dot-product of the training example's vector with the sum of the vectors in the
        validation dataset. Also, by doing the sum aggregation within this method as
        opposed to outside of it (by computing all vectors for the validation dataset,
        then taking the sum) allows memory usage to be reduced.

        Args:
            inputs (Tuple, or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`, and
                    and `batch[-1]` are the labels, if any. Here, `model` is model
                    provided in initialization. This is the same assumption made for
                    each batch yielded by training dataset `train_dataset`.
            aggregate (bool): Whether to return the sum of the vectors for all
                    examples, as opposed to vectors for each example.

        Returns:
            intermediate_quantities (Tensor): A tensor of dimension
                    (N, D * C). Here, N is the total number of examples in
                    `inputs` if `aggregate` is False, and 1, otherwise (so that
                    a 2D tensor is always returned). C is the number of checkpoints
                    passed as the `checkpoints` argument of `TracInCP.__init__`, and
                    each row represents the vector for an example. Regarding D: Let I
                    be the dimension of the output of the last fully-connected layer
                    times the dimension of the input of the last fully-connected layer.
                    If `self.projection_dim` is specified in initialization,
                    D = min(I * C, `self.projection_dim` * C). Otherwise, D = I * C.
                    In summary, if `self.projection_dim` is None, the dimension of each
                    vector will be determined by the size of the input and output of
                    the last fully-connected layer of `model`. Otherwise,
                    `self.projection_dim` must be an int, and random projection will be
                    performed to ensure that the vector is of dimension no more than
                    `self.projection_dim` * C. `self.projection_dim` corresponds to
                    the variable d in the top of page 15 of the TracIn paper:
                    https://arxiv.org/pdf/2002.08484.pdf.
        """
        f_inputs: DataLoader = _format_inputs_dataset(inputs)

        def get_checkpoint_contribution(checkpoint: str) -> Tensor:
            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)
            # get jacobians as tuple of tensors
            if aggregate:
                inputs_jacobians = self._sum_jacobians(
                    f_inputs,
                    self.loss_fn,
                    self.reduction_type,
                )
            else:
                inputs_jacobians = self._concat_jacobians(
                    f_inputs,
                    self.loss_fn,
                    self.reduction_type,
                )
            # flatten into single tensor
            return learning_rate * torch.cat(
                [
                    input_jacobian.flatten(start_dim=1)
                    for input_jacobian in inputs_jacobians
                ],
                dim=1,
            )

        return torch.cat(
            [
                get_checkpoint_contribution(checkpoint)
                for checkpoint in self.checkpoints
            ],
            dim=1,
        )

    def _influence_batch_tracincp(
        self,
        input_checkpoint_jacobians: List[Tuple[Any, ...]],
        train_batch: Tuple[Any, ...],
    ) -> Tensor:
        """
        computes influence scores for a single training batch.
        `input_checkpoint_jacobians` is the output of
        `_basic_computation_tracincp` applied to the test batch, for each checkpoint,
        computed by `_get_checkpoint_jacobians`.
        """

        def get_checkpoint_contribution(
            input_jacobians: Tuple[Tensor, ...], checkpoint: str
        ) -> Tensor:

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            return (
                _gradient_dot_product(
                    input_jacobians,
                    self._basic_computation_tracincp(
                        train_batch[0:-1],
                        train_batch[-1],
                        self.loss_fn,
                        self.reduction_type,
                    ),
                )
                * learning_rate
            )

        batch_tracin_scores = get_checkpoint_contribution(
            input_checkpoint_jacobians[0], self.checkpoints[0]
        )

        for input_jacobians, checkpoint in zip(
            input_checkpoint_jacobians[1:], self.checkpoints[1:]
        ):
            batch_tracin_scores += get_checkpoint_contribution(
                input_jacobians, checkpoint
            )

        return batch_tracin_scores

    def _get_checkpoint_jacobians(
        self,
        inputs_dataset: DataLoader,
        aggregate: bool,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
    ) -> List[Tuple[Tensor, ...]]:
        """
        computes the jacobians of all examples in `inputs_dataset`, for all
        checkpoints. if `aggregate` is True, the jacobians for examples are summed.
        returns a list where each element corresponds to a checkpoint. this logic is
        separated into a helper function because it is used by both `_influence` and
        `_get_k_most_influential`.
        """
        inputs_checkpoint_jacobians = []
        for checkpoint in self.checkpoints:
            self.checkpoints_load_func(self.model, checkpoint)
            if aggregate:
                inputs_checkpoint_jacobians.append(
                    self._sum_jacobians(inputs_dataset, loss_fn, self.reduction_type)
                )
            else:
                inputs_checkpoint_jacobians.append(
                    self._concat_jacobians(inputs_dataset, loss_fn, self.reduction_type)
                )
        return inputs_checkpoint_jacobians

    def _influence(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        show_progress: bool = False,
        aggregate: bool = False,
    ) -> Tensor:
        r"""
        Computes the influence of examples in training dataset `train_dataset`
        on the examples in the test dataset represented by `inputs`.
        This implementation does not require knowing the number of training examples
        in advance. Instead, the number of training examples is inferred from the
        output of `self._basic_computation_tracincp`.

        Args:

            inputs_dataset (Tuple, or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`, and
                    and `batch[-1]` are the labels, if any. Here, `model` is model
                    provided in initialization. This is the same assumption made for
                    each batch yielded by training dataset `train_dataset`.
            show_progress (bool, optional): To compute the influence of examples in
                    training dataset `train_dataset`, we compute the influence
                    of each batch. If `show_progress` is true, the progress of this
                    computation will be displayed. In particular, the number of batches
                    for which influence has been computed will be displayed. It will
                    try to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
                    progress.
                    Default: False
            aggregate (bool): Whether to return "aggregate" influence scores (see their
                    definition in `influence`).
                    Default: False

        Returns:
            influence_scores (Tensor): If `aggregate` is False, influence scores are
            returned as a 2D tensor whose shape is `(input_size, train_dataset_size)`,
            where `input_size` is the number of examples in the test dataset, and
            `train_dataset_size` is the number of examples in
            training dataset `train_dataset`. For example:
            `influence_scores[i][j]` is the influence score for the j-th training
            example to the i-th example in the test dataset. If `aggregate` is True,
            "aggregate" influence scores are returned as a 2D tensor whose shape is
            `(1, train_dataset_size)`. For example: `influence_scores[0][j]` is the
            aggregate influence score of the j-th training example on the test dataset.
        """
        # If `inputs` is not a `DataLoader`, turn it into one.
        inputs = _format_inputs_dataset(inputs)

        train_dataloader = self.train_dataloader
        data_iterable: Union[Iterable[Tuple[object, ...]], DataLoader] = (
            train_dataloader
        )
        if show_progress:
            data_iterable = progress(
                cast(Iterable[Tuple[object, ...]], train_dataloader),
                desc=(
                    f"Using {self.get_name()} to compute "
                    "influence for training batches"
                ),
                total=self.train_dataloader_len,
            )

        # create list of the outputs of `_basic_computation_tracincp`, for each
        # checkpoint, which are jacobians
        inputs_checkpoint_jacobians = self._get_checkpoint_jacobians(
            inputs, aggregate, self.test_loss_fn
        )

        return torch.cat(
            [
                self._influence_batch_tracincp(inputs_checkpoint_jacobians, batch)
                for batch in data_iterable
            ],
            dim=1,
        )

    def _get_k_most_influential(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        k: int = 5,
        proponents: bool = True,
        show_progress: bool = False,
        aggregate: bool = False,
    ) -> KMostInfluentialResults:
        r"""
        Args:

            inputs (Tuple, or DataLoader): Either a single tuple of any, or a
                    `DataLoader`, where each batch yielded is a tuple of any. In
                    either case, the tuple represents a single batch, where the last
                    element is assumed to be the labels for the batch. That is,
                    `model(*batch[0:-1])` produces the output for `model`, and
                    and `batch[-1]` are the labels, if any. Here, `model` is model
                    provided in initialization. This is the same assumption made for
                    each batch yielded by training dataset `train_dataset`.
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
            aggregate (bool): Whether to return with the highest / lowest "aggregate"
                    influence scores (see their definition in `influence`).

        Returns:
            (indices, influence_scores) (namedtuple): If `aggregate` is False,
                    `indices` is a 2D tensor of shape `(input_size, k)`, where
                    `input_size` is the number of examples in the test dataset. If
                    computing proponents (resp. opponents), `indices[i][j]` is the
                    index in training dataset `train_dataset` of the example with the
                    `j`-th highest (resp. lowest) influence score (out of the examples
                    in `train_dataset`) on the `i`-th example in the test dataset.
                    `influence_scores` contains the corresponding influence scores. In
                    particular, `influence_scores[i][j]` is the influence score of
                    example `indices[i][j]` in `train_dataset` on example `i` in the
                    test dataset represented by `inputs`. If `aggregate` is True,
                    `indices` is a 2D tensor of shape `(1, k)`. If computing proponents
                    (resp. opponents), `indices[0][j]` is the index in training dataset
                    `train_dataset` of the example with the `j`-th highest (resp.
                    lowest) aggregate influence score on the test dataset.
                    `influence_scores` contains the corresponding aggregate influence
                    scores. In particular, `influence_scores[0][j]` is the aggregate
                    influence score of example `indices[0][j]` on the test dataset.
        """
        # If `inputs` is not a `DataLoader`, turn it into one.
        inputs = _format_inputs_dataset(inputs)

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

        # create list of the outputs of `_basic_computation_tracincp`, for each
        # checkpoint, which are jacobians
        inputs_checkpoint_jacobians = self._get_checkpoint_jacobians(
            inputs, aggregate, self.test_loss_fn
        )

        return KMostInfluentialResults(
            *_get_k_most_influential_helper(
                self.train_dataloader,
                self._influence_batch_tracincp,
                inputs_checkpoint_jacobians,
                k,
                proponents,
                show_progress,
                desc,
            )
        )

    def _self_influence_by_checkpoints(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        show_progress: bool = False,
    ) -> Tensor:
        """
        Computes self influence scores for the examples in `inputs`, which is
        either a single batch or a Pytorch `DataLoader` that yields batches. Therefore,
        the computed self influence scores are *not* for the examples in training
        dataset `train_dataset` (unlike when computing self influence scores using the
        `influence` method). Note that if `inputs` is a single batch, this
        will call `model` on that single batch, and if `inputs` yields
        batches, this will call `model` on each batch that is yielded. Therefore,
        please ensure that for both cases, the batch(es) that `model` is called
        with are not too large, so that there will not be an out-of-memory error. This
        implementation performs an outer iteration over checkpoints, and an inner
        iteration over all batches that `inputs` represents. The pros of this
        implementation are that the checkpoints do not need to be loaded too many
        times.

        Args:
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
                    take a long time if `inputs` represents many examples. If
                    `show_progress` is true, the progress of this computation will be
                    displayed. In more detail, this computation will iterate over all
                    checkpoints (provided as the `checkpoints` initialization argument)
                    in an outer loop, and iterate over all batches that
                    `inputs` represents in an inner loop. Thus if
                    `show_progress` is True, the progress of both the outer
                    iteration and the inner iterations will be displayed. To show
                    progress, it will try to use tqdm if available for advanced
                    features (e.g. time estimation). Otherwise, it will fallback to a
                    simple output of progress.
                    Default: False

        Returns:
            self_influence_scores (Tensor): This is a 1D tensor containing the self
                    influence scores of all examples in `inputs`, regardless of
                    whether it represents a single batch or a `DataLoader` that yields
                    batches.
        """
        # If `inputs` is not a `DataLoader`, turn it into one.
        inputs = _format_inputs_dataset(inputs)

        # If `show_progress` is true, create an outer progress bar that keeps track of
        # how many checkpoints have been processed
        if show_progress:
            # Try to determine length of inner progress bar if possible, with a default
            # of `None`.
            inputs_len: Optional[int] = None
            try:
                inputs_len = len(inputs)
            except TypeError:
                warnings.warn(
                    "Unable to determine the number of batches in `inputs`. "
                    "Therefore, if showing the progress of the computation of self "
                    "influence scores, only the number of batches processed can be "
                    "displayed, and not the percentage completion of the computation, "
                    "nor any time estimates.",
                    stacklevel=1,
                )

        def calculate_via_vector_norm(layer_jacobian: Tensor) -> Tensor:
            # Helper to efficiently calculate vector norm if pytorch version permits.
            return (
                torch.linalg.vector_norm(
                    layer_jacobian,
                    dim=list(range(1, len(layer_jacobian.shape))),
                )
                ** 2
            )

        def get_checkpoint_contribution(checkpoint: str) -> Tensor:
            # This function returns a 1D tensor representing the contribution to the
            # self influence score for the given checkpoint, for all batches in
            # `inputs`. The length of the 1D tensor is the total number of
            # examples in `inputs`.
            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            # This will store a list of the contribution of the self influence score
            # from each batch. Each element is a 1D tensor of length batch_size - the
            # batch size of each batch in `inputs` (they do not need to be all
            # the same)
            checkpoint_contribution = []

            _inputs: Union[DataLoader, Iterable[Tuple[Tensor, ...]]] = inputs
            # If `show_progress` is true, create an inner progress bar that keeps track
            # of how many batches have been processed for the current checkpoint
            if show_progress:
                _inputs = progress(
                    inputs,
                    desc=(
                        f"Using {self.get_name()} to compute self "
                        "influence. Processing batch"
                    ),
                    total=inputs_len,
                )

            for batch in _inputs:

                layer_jacobians = self._basic_computation_tracincp(
                    cast(Tuple[Tensor, ...], batch)[0:-1],
                    cast(Tuple[Tensor, ...], batch)[-1],
                    self.loss_fn,
                    self.reduction_type,
                )

                # Note that all variables in this function are for an entire batch.
                # Each `layer_jacobian` in `layer_jacobians` corresponds to a different
                # layer. `layer_jacobian` is the jacobian w.r.t to a given layer's
                # parameters. If the given layer's parameters are of shape *, then
                # `layer_jacobian` is of shape (batch_size, *). For each layer, we need
                # the squared jacobian for each example. So we square the jacobian and
                # sum over all dimensions except the 0-th (the batch dimension). We then
                # sum the contribution over all layers.  We use the optimized
                # torch.linalg.vector_norm as opposed to the explicit flatten.

                checkpoint_contribution.append(
                    torch.sum(
                        torch.stack(
                            [
                                calculate_via_vector_norm(layer_jacobian)
                                for layer_jacobian in layer_jacobians
                            ],
                            dim=0,
                        ),
                        dim=0,
                    )
                    * learning_rate
                )

            # We concatenate the contributions from each batch into a single 1D tensor,
            # which represents the contributions for all batches in `inputs`

            return torch.cat(checkpoint_contribution, dim=0)

        if show_progress:
            checkpoints_progress = progress(
                desc=(
                    f"Using {self.get_name()} to compute self "
                    "influence. Processing checkpoint"
                ),
                total=len(self.checkpoints),
                mininterval=0.0,
            )
        else:
            checkpoints_progress = NullProgress()
        with checkpoints_progress:
            batches_self_tracin_scores = get_checkpoint_contribution(
                self.checkpoints[0]
            )
            checkpoints_progress.update()
            # The self influence score for all examples is the sum of contributions from
            # each checkpoint
            for checkpoint in self.checkpoints[1:]:
                batches_self_tracin_scores += get_checkpoint_contribution(checkpoint)
                checkpoints_progress.update()

        return batches_self_tracin_scores

    @log_usage(part_of_slo=True)
    def self_influence(
        self,
        inputs: Optional[Union[Tuple[Any, ...], DataLoader]] = None,
        show_progress: bool = False,
        outer_loop_by_checkpoints: bool = False,
    ) -> Tensor:
        """
        Computes self influence scores for the examples in `inputs`, which is
        either a single batch or a Pytorch `DataLoader` that yields batches.
        If `inputs` is not specified or `None` calculates self influence
        score for the training dataset `train_dataset`. Note that if `inputs`
        is a single batch, this will call `model` on that single batch, and if
        `inputs` yields batches, this will call `model` on each batch that is
        yielded. Therefore, please ensure that for both cases, the batch(es) that
        `model` is called with are not too large, so that there will not be an
        out-of-memory error.
        Internally, this computation requires iterating both over the batches in
        `inputs`, as well as different model checkpoints. There are two ways
        this iteration can be done. If `outer_loop_by_checkpoints` is False, the outer
        iteration will be over batches, and the inner iteration will be over
        checkpoints. This has the pro that displaying the progress of the computation
        is more intuitive, involving displaying the number of batches for which self
        influence scores have been computed. If `outer_loop_by_checkpoints` is True,
        the outer iteration will be over checkpoints, and the inner iteration will be
        over batches. This has the pro that the checkpoints do not need to be loaded
        for each batch. For large models, loading checkpoints can be time-intensive.

        Args:
            inputs (tuple or DataLoader, optional): This specifies the
                    dataset for which self influence scores will be computed.
                    Either a single tuple of any, or a `DataLoader`, where each
                    batch yielded is a tuple of type any. In either case, the tuple
                    represents a single batch, where the last element is assumed to
                    be the labels for the batch. That is, `model(*batch[0:-1])`
                    produces the output for `model`, and `batch[-1]` are the labels,
                    if any. This is the same assumption made for each batch yielded
                    by training dataset `train_dataset`. Please see documentation for
                    the `train_dataset` argument to `TracInCP.__init__` for
                    more details on the assumed structure of a batch. If not provided
                    or `None`, self influence scores will be computed for training
                    dataset `train_dataset`, which yields batches satisfying the
                    above assumptions.
                    Default: None.
            show_progress (bool, optional): Computation of self influence scores can
                    take a long time if `inputs` represents many examples. If
                    `show_progress`is true, the progress of this computation will be
                    displayed. In more detail, if `outer_loop_by_checkpoints` is False,
                    this computation will iterate over all batches in an outer loop.
                    Thus if `show_progress` is True, the number of batches for which
                    self influence scores have been computed will be displayed. If
                    `outer_loop_by_checkpoints` is True, this computation will iterate
                    over all checkpoints (provided as the `checkpoints` initialization
                    argument) in an outer loop, and iterate over all batches that
                    `inputs` represents in an inner loop. Thus if
                    `show_progress` is True, the progress of both the outer
                    iteration and the inner iterations will be displayed. To show
                    progress, it will try to use tqdm if available for advanced
                    features (e.g. time estimation). Otherwise, it will fallback to a
                    simple output of progress.
                    Default: False
            outer_loop_by_checkpoints (bool, optional): If performing an outer
                    iteration over checkpoints; see method description for more
                    details.
                    Default: False
        """
        inputs = inputs if inputs is not None else self.train_dataloader
        if outer_loop_by_checkpoints:
            return self._self_influence_by_checkpoints(inputs, show_progress)
        return _self_influence_by_batches_helper(
            self._self_influence_by_checkpoints,
            self.get_name(),
            inputs,
            show_progress,
        )

    def _basic_computation_tracincp(
        self,
        inputs: Tuple[Any, ...],
        targets: Optional[Tensor] = None,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        reduction_type: Optional[str] = None,
    ) -> Tuple[Tensor, ...]:
        """
        For instances of TracInCP, computation of influence scores or self influence
        scores repeatedly calls this function for different checkpoints
        and batches. In particular, this function computes the jacobian of a loss
        function w.r.t. parameters in the `layers` initialization argument.

        Args:

            inputs (tuple[Any, ...]): A batch of examples, which could be a training
                    batch or test batch, depending which method is the caller. Does not
                    represent labels, which are passed as `targets`. The assumption is
                    that `model(*inputs)` produces the predictions for the batch.
            targets (tensor or None): If computing influence scores on a loss function,
                    these are the labels corresponding to the batch `inputs`.
                    Default: none
            loss_fn (Callable, optional): The loss function to use when computing the
                    jacobian.
            reduction_type (str, optional): The reduction type of `loss_fn`. This
                    argument is only used if `sample_wise_grads_per_batch` was true in
                    initialization.
        """
        return _compute_jacobian_sample_wise_grads_per_batch(
            self, inputs, targets, loss_fn, reduction_type
        )
