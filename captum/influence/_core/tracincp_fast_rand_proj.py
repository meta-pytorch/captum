#!/usr/bin/env python3

# pyre-strict

import threading
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch
from captum._utils.common import _get_module_from_name, _sort_key_list
from captum._utils.gradient import _gather_distributed_tensors
from captum._utils.progress import NullProgress, progress
from captum.influence._core.tracincp import (
    _influence_route_to_helpers,
    KMostInfluentialResults,
    TracInCPBase,
)
from captum.influence._utils.common import (
    _check_loss_fn,
    _format_inputs_dataset,
    _get_k_most_influential_helper,
    _jacobian_loss_wrt_inputs,
    _load_flexible_state_dict,
    _self_influence_by_batches_helper,
    _tensor_batch_dot,
)
from captum.influence._utils.nearest_neighbors import (
    AnnoyNearestNeighbors,
    NearestNeighbors,
)
from captum.log import log_usage
from torch import device, Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

r"""
Implements abstract DataInfluence class and also provides implementation details for
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


class TracInCPFast(TracInCPBase):
    r"""
    In Appendix F, Page 14 of the TracIn paper, they show that the calculation
    of the influence score of between a test example x' and a training example x,
    can be computed much more quickly than naive back-propagation in the special
    case when considering only gradients in the last fully-connected layer. This class
    computes influence scores for that special case. Note that the computed
    influence scores are exactly the same as when naive back-propagation is used -
    there is no loss in accuracy.

    In more detail regarding the influence score computation: let :math`x`
    and :math`\nabla_y f(y)` be the input and output-gradient of the last
    fully-connected layer, respectively, for a training example. Similarly, let
    :math`x'` and :math`\nabla_{y'} f(y')` be the corresponding quantities for
    a test example. Then, the influence score of the training example on the test
    example is the sum of the contribution from each checkpoint. The contribution from
    a given checkpoint is :math`(x^T x')(\nabla_y f(y)^T \nabla_{y'} f(y'))`.

    """

    def __init__(
        self,
        model: Module,
        final_fc_layer: Union[Module, str],
        train_dataset: Union[Dataset, DataLoader],
        # pyre-fixme[24]: Generic type `Iterator` expects 1 type parameter.
        checkpoints: Union[str, List[str], Iterator],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        checkpoints_load_func: Callable = _load_flexible_state_dict,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        vectorize: bool = False,
    ) -> None:
        r"""
        Args:

            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            final_fc_layer (torch.nn.Module): The last fully connected layer in
                    the network for which gradients will be approximated via fast random
                    projection method.
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
            loss_fn (Callable, optional): The loss function applied to model. `loss_fn`
                    must be a "reduction" loss function that reduces the per-example
                    losses in a batch, and returns a single scalar Tensor. Furthermore,
                    the reduction must be the *sum* or the *mean* of the per-example
                    losses. For instance, `nn.BCELoss(reduction="sum")` is acceptable.
                    Also note that if `loss_fn` has no "reduction" attribute,
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
                    to `test_loss_fn`, if the latter is provided. If not provided, the
                    loss function for test examples is assumed to be the same as the
                    loss function for training examples, i.e. `loss_fn`.
                    Default: None
            vectorize (bool, optional): Flag to use experimental vectorize functionality
                    for `torch.autograd.functional.jacobian`.
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

        self.vectorize = vectorize

        # TODO: restore prior state
        self.final_fc_layer = cast(Module, final_fc_layer)
        for param in self.final_fc_layer.parameters():
            param.requires_grad = True

        assert loss_fn is not None, "loss function must not be none"

        # check `loss_fn`
        # pyre-fixme[4]: Attribute must be annotated.
        self.reduction_type = _check_loss_fn(self, loss_fn, "loss_fn")
        # check `test_loss_fn` if it was provided
        # pyre-fixme[4]: Attribute must be annotated.
        self.test_reduction_type = (
            self.reduction_type
            if test_loss_fn is None
            else _check_loss_fn(self, test_loss_fn, "test_loss_fn")
        )

    @property
    def final_fc_layer(self) -> Module:
        # pyre-fixme[16]: `TracInCPFast` has no attribute `_final_fc_layer`.
        return self._final_fc_layer

    @final_fc_layer.setter
    def final_fc_layer(self, layer: Union[Module, str]) -> None:
        if isinstance(layer, str):
            try:
                self._final_fc_layer = _get_module_from_name(self.model, layer)
                if not isinstance(self._final_fc_layer, Module):
                    raise Exception("No module found for final_fc_layer")
            except Exception as ex:
                raise ValueError(
                    f'Invalid final_fc_layer str: "{layer}" provided!'
                ) from ex
        else:
            self._final_fc_layer = layer

    @log_usage(part_of_slo=True)
    def influence(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        k: Optional[int] = None,
        proponents: bool = True,
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

            inputs (tuple or DataLoader): `inputs` is the test batch and is a tuple of
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
              batch. `influence_scores` contains the corresponding influence scores.
              In particular, `influence_scores[i][j]` is the influence score of example
              `indices[i][j]` in `train_dataset` on example `i` in the test batch
              represented by `inputs`.
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
        )

    # pyre-fixme[3]: Return type must be annotated.
    def _influence_batch_tracincp_fast(
        self,
        test_batch: Tuple[Any, ...],
        train_batch: Tuple[Any, ...],
    ):
        """
        computes influence scores for a single training batch, when only considering
        gradients in the last fully-connected layer, using the computation trick
        described in the `TracInCPFast` class description.
        """

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def get_checkpoint_contribution(checkpoint):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)

            input_jacobians, input_layer_inputs = _basic_computation_tracincp_fast(
                self,
                test_batch[0:-1],
                test_batch[-1],
                self.test_loss_fn,
                self.test_reduction_type,
            )

            src_jacobian, src_layer_input = _basic_computation_tracincp_fast(
                self,
                train_batch[0:-1],
                train_batch[-1],
                self.loss_fn,
                self.reduction_type,
            )
            return (
                _tensor_batch_dot(
                    input_jacobians, src_jacobian
                )  # shape is (test batch size, training batch size), containing x^T x'
                # for every example x in the training batch and example x' in the test
                # batch
                * _tensor_batch_dot(input_layer_inputs, src_layer_input)
                # shape is (test batch size, training batch size), containing
                # (\nabla_y f(y)^T \nabla_{y'} f(y')) for every label y in the training
                # batch and label y' in the test batch
                * learning_rate
            )

        batch_tracin_scores = get_checkpoint_contribution(self.checkpoints[0])

        for checkpoint in self.checkpoints[1:]:
            batch_tracin_scores += get_checkpoint_contribution(checkpoint)

        return batch_tracin_scores

    def _influence(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        show_progress: bool = False,
    ) -> Tensor:
        r"""
        Computes the influence of examples in training dataset `train_dataset`
        on the examples in the test batch represented by `inputs`.
        This implementation does not require knowing the number of training examples
        in advance. Instead, the number of training examples is inferred from the
        output of `_basic_computation_tracincp_fast`.

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
            influence_scores (Tensor): Influence scores from the `TracInCPFast` method.
            Its shape is `(input_size, train_dataset_size)`, where `input_size`
            is the number of examples in the test batch, and
            `train_dataset_size` is the number of examples in
            training dataset `train_dataset`. For example:
            `influence_scores[i][j]` is the influence score for the j-th training
            example to the i-th example in the test batch.
        """

        train_dataloader = self.train_dataloader
        train_dataloader_iterable: Union[DataLoader, Iterable[Tuple[object, ...]]] = (
            train_dataloader
        )

        if show_progress:
            train_dataloader_iterable = progress(
                cast(Iterable[Tuple[object, ...]], train_dataloader),
                desc=(
                    f"Using {self.get_name()} to compute "
                    "influence for training batches"
                ),
                total=self.train_dataloader_len,
            )

        return torch.cat(
            [
                self._influence_batch_tracincp_fast(inputs, batch)
                for batch in train_dataloader_iterable
            ],
            dim=1,
        )

    def _get_k_most_influential(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
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
                self._influence_batch_tracincp_fast,
                inputs,
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
            inputs_len = None
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

        # pyre-fixme[53]: Captured variable `inputs_len` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def get_checkpoint_contribution(checkpoint):
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

            _inputs = inputs
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

                batch_jacobian, batch_layer_input = _basic_computation_tracincp_fast(
                    self,
                    batch[0:-1],
                    batch[-1],
                    self.loss_fn,
                    self.reduction_type,
                )

                checkpoint_contribution.append(
                    # pyre-fixme[58]: `**` is not supported for operand types
                    #  `Tensor` and `int`.
                    torch.sum(batch_jacobian**2, dim=1)
                    # pyre-fixme[58]: `**` is not supported for operand types
                    #  `Tensor` and `int`.
                    * torch.sum(batch_layer_input**2, dim=1)
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
        is a single batch, this will call `model` on that single batch,
        and if `inputs` yields batches, this will call `model`
        on each batch that is yielded. Therefore, please ensure that for both cases,
        the batch(es) that `model` is called with are not too large, so that
        there will not be an out-of-memory error.
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


def _basic_computation_tracincp_fast(
    influence_instance: TracInCPFast,
    inputs: Tuple[Any, ...],
    targets: Tensor,
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    loss_fn: Optional[Union[Module, Callable]] = None,
    reduction_type: Optional[str] = None,
) -> Tuple[Tensor, Tensor]:
    """
    For instances of TracInCPFast and children classes, computation of influence scores
    or self influence scores repeatedly calls this function for different checkpoints
    and batches. These computations involve a loss function. If `test` is True, the
    loss function is `self.loss_fn`. If `test` is False, the loss function is
    `self.test_loss_fn`. These two attributes were set in initialization, with
    `self.loss_fn` equal to the `loss_fn` initialization argument, and
    `self.test_loss_fn` equal to the `test_loss_fn` initialization argument if it was
    provided, and `loss_fn` otherwise.

    Args:

        influence_instance (TracInCPFast): A instance of TracInCPFast or its children.
                We assume `influence_instance` has a `loss_fn` attribute, i.e. the loss
                function applied to the output of the last fully-connected layer, as
                well as a `reduction_type` attribute, which indicates whether `loss_fn`
                reduces the per-example losses by using their mean or sum. The
                `reduction_type` attribute must either be "mean" or "sum".
        inputs (tuple[Any, ...]): A batch of examples, which could be a training batch
                or test batch, depending which method is the caller. Does not
                represent labels, which are passed as `targets`. The assumption is
                that `model(*inputs)` produces the predictions for the batch.
        targets (Tensor): If computing influence scores on a loss function,
                these are the labels corresponding to the batch `inputs`.
        loss_fn (Callable, optional): The loss function to use when computing the
                jacobian.
        reduction_type (str, optional): The reduction type of `loss_fn`. This argument
                is only used if `sample_wise_grads_per_batch` was true in
                initialization of `influence_instance`.

    Returns:
        (input_jacobians, layer_inputs) (tuple): `input_jacobians` is a 2D tensor,
                where each row is the jacobian of the loss, with respect to the
                *output* of the last fully-connected layer. `layer_inputs` is a 1D
                tensor, where each row is the *input* to the last fully-connected
                layer. For both, the length is the number of examples in the batch
                represented by `inputs` and `targets`.
    """
    layer_inputs: Dict[device, Tuple[Tensor, ...]] = defaultdict()
    lock = threading.Lock()

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def hook_wrapper(original_module):
        # pyre-fixme[53]: Captured variable `lock` is not annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def _capture_inputs(layer, input, output) -> None:
            r"""Save activations into layer_inputs in forward pass"""
            with lock:
                is_eval_tuple = isinstance(input, tuple)
                if is_eval_tuple:
                    layer_inputs_val = tuple(inp.detach() for inp in input)
                else:
                    layer_inputs_val = input.detach()
                layer_inputs[layer_inputs_val[0].device] = layer_inputs_val

        return _capture_inputs

    assert isinstance(influence_instance.final_fc_layer, Module)
    handle = influence_instance.final_fc_layer.register_forward_hook(
        hook_wrapper(influence_instance.final_fc_layer)
    )
    out = influence_instance.model(*inputs)

    assert loss_fn is not None, "loss function is required"
    assert reduction_type in [
        "sum",
        "mean",
    ], 'reduction_type must be either "mean" or "sum"'
    input_jacobians = _jacobian_loss_wrt_inputs(
        loss_fn,
        out,
        targets,
        influence_instance.vectorize,
        reduction_type,
    )
    handle.remove()

    device_ids = cast(
        Union[None, List[int]],
        (
            influence_instance.model.device_ids
            if hasattr(influence_instance.model, "device_ids")
            else None
        ),
    )
    key_list = _sort_key_list(list(layer_inputs.keys()), device_ids)

    _layer_inputs = _gather_distributed_tensors(layer_inputs, key_list=key_list)[0]

    assert len(input_jacobians.shape) == 2

    return input_jacobians, _layer_inputs


class TracInCPFastRandProj(TracInCPFast):
    r"""
    A version of TracInCPFast which is optimized for "interactive" calls to
    `influence` for the purpose of calculating proponents / opponents, or
    influence scores. "Interactive" means there will be multiple calls to
    `influence`, with each call for a different batch of test examples, and
    subsequent calls rely on the results of previous calls. The implementation in
    this class has been optimized so that each call to `influence` is fast, so that
    it can be used for interactive analysis. This class should only be used for
    interactive use cases. It should not be used if `influence` will only be
    called once, because to enable fast calls to `influence`, time and memory
    intensive preprocessing is required in `__init__`. Furthermore, it should not
    be used to calculate self influence scores - `TracInCPFast` should be used
    instead for that purpose. To enable interactive analysis, this implementation
    computes and saves "embedding" vectors for all training examples in
    `train_dataset`. Crucially, the influence score of a training
    example on a test example is simply the dot-product of their corresponding
    vectors, and proponents / opponents can be found by first storing vectors for
    training examples in a nearest-neighbor data structure, and then finding the
    nearest-neighbors for a test example in terms of dot-product (see appendix F
    of the TracIn paper). This class should only be used if calls to `influence`
    to obtain proponents / opponents or influence scores will be made in an
    "interactive" manner, and there is sufficient memory to store vectors for the
    entire `train_dataset`. This is because in order to enable interactive
    analysis, this implementation incures overhead in `__init__` to setup the
    nearest-neighbors data structure, which is both time and memory intensive, as
    vectors corresponding to all training examples needed to be stored. To reduce
    memory usage, this implementation enables random projections of those vectors.
    Note that the influence scores computed with random projections are less
    accurate, though correct in expectation.

    In more detail regarding the "embedding" vectors - the influence of a training
    example on a test example, when only considering gradients in the last
    fully-connected layer, the sum of the contribution from each checkpoint. The
    contribution from a given checkpoint is
    :math`(x^T x')(\nabla_y f(y)^T \nabla_{y'} f(y'))`, using the notation in the
    description of `TracInCPFast`. As is, this is not a dot-product of 2 vectors.
    However, we can rewrite that contribution as
    :math`(x \nabla_y f(y)^T) \dot (x' f(y')^T)`. Both terms in this
    product are 2D matrices, as they are outer products, and the "product" is actually
    a dot-product, treating both matrices as vectors. Therefore, for a given
    checkpoint, its contribution to the "embedding" of an example is just the
    outer-product :math`(x \nabla_y f(y)^T)`, flattened. Furthemore, to reduce the
    dimension of this contribution, we can right-multiply and
    left-multiply the outer-product with two separate projection matrices. These
    transform :math`\nabla_y f(y)` and :math`x` to lower dimensional vectors. While
    the dimension of these two lower dimensional vectors do not necessarily need to
    be the same, in our implementation, we let them be the same, both equal to the
    square root of the desired projection dimension. Finally, the embedding of an
    example is the concatenation of the contributions from each checkpoint.
    """

    def __init__(
        self,
        model: Module,
        final_fc_layer: Module,
        train_dataset: Union[Dataset, DataLoader],
        # pyre-fixme[24]: Generic type `Iterator` expects 1 type parameter.
        checkpoints: Union[str, List[str], Iterator],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        checkpoints_load_func: Callable = _load_flexible_state_dict,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]] = None,
        batch_size: Union[int, None] = 1,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_loss_fn: Optional[Union[Module, Callable]] = None,
        vectorize: bool = False,
        nearest_neighbors: Optional[NearestNeighbors] = None,
        projection_dim: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        r"""
        Args:

            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            final_fc_layer (torch.nn.Module): The last fully connected layer in
                    the network for which gradients will be approximated via fast random
                    projection method.
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
            loss_fn (Callable, optional): The loss function applied to model. `loss_fn`
                    must be a "reduction" loss function that reduces the per-example
                    losses in a batch, and returns a single scalar Tensor. Furthermore,
                    the reduction must be the *sum* of the per-example losses. For
                    instance, `nn.BCELoss(reduction="sum")` is acceptable, but
                    `nn.BCELoss(reduction="mean")` is *not* acceptable.
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
                    to `test_loss_fn`, if the latter is provided. If not provided, the
                    loss function for test examples is assumed to be the same as the
                    loss function for training examples, i.e. `loss_fn`.
            vectorize (bool): Flag to use experimental vectorize functionality
                    for `torch.autograd.functional.jacobian`.
                    Default: False
            nearest_neighbors (NearestNeighbors, optional): The NearestNeighbors
                    instance for finding nearest neighbors. If None, defaults to
                    `AnnoyNearestNeighbors(n_trees=10)`.
                    Default: None
            projection_dim (int, optional): Each example will be represented in
                    the nearest neighbors data structure with a vector. This vector
                    is the concatenation of several "checkpoint vectors", each of which
                    is computed using a different checkpoint in the `checkpoints`
                    argument. If `projection_dim` is an int, it represents the
                    dimension we will project each "checkpoint vector" to, so that the
                    vector for each example will be of dimension at most
                    `projection_dim` * C, where C is the number of checkpoints.
                    Regarding the dimension of each vector, D: Let I be the dimension
                    of the output of the last fully-connected layer times the dimension
                    of the input of the last fully-connected layer. If `projection_dim`
                    is not `None`, then D = min(I * C, `projection_dim` * C).
                    Otherwise, D = I * C. In summary, if `projection_dim` is None, the
                    dimension of this vector will be determined by the size of the
                    input and output of the last fully-connected layer of `model`, and
                    the number of checkpoints. Otherwise, `projection_dim` must be an
                    int, and random projection will be performed to ensure that the
                    vector is of dimension no more than `projection_dim` * C.
                    `projection_dim` corresponds to the variable d in the top of page
                    15 of the TracIn paper: https://arxiv.org/abs/2002.08484.
                    Default: None
            seed (int, optional): Because this implementation chooses a random
                    projection, its output is random. Setting this seed specifies the
                    random seed when choosing the random projection.
                    Default: 0
        """

        TracInCPFast.__init__(
            self,
            model,
            final_fc_layer,
            train_dataset,
            checkpoints,
            checkpoints_load_func,
            loss_fn,
            batch_size,
            test_loss_fn,
            vectorize,
        )

        warnings.warn(
            (
                "WARNING: Using this implementation stores quantities related to the "
                "entire `train_dataset` in memory, and may results in running "
                "out of memory. If this happens, consider using %s instead, for which "
                "each call to `influence` to compute influence scores or proponents "
                "will be slower, but may avoid running out of memory."
            )
            % "`TracInCPFast`",
            stacklevel=1,
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.nearest_neighbors = (
            AnnoyNearestNeighbors() if nearest_neighbors is None else nearest_neighbors
        )

        self.projection_dim = projection_dim

        torch.manual_seed(seed)  # for reproducibility
        # pyre-fixme[4]: Attribute must be annotated.
        self.projection_quantities = self._set_projections_tracincp_fast_rand_proj(
            self.train_dataloader,
        )

        # pyre-fixme[4]: Attribute must be annotated.
        self.src_intermediate_quantities = (
            self._get_intermediate_quantities_tracincp_fast_rand_proj(
                self.train_dataloader,
                self.projection_quantities,
            )
        )

        self._process_src_intermediate_quantities_tracincp_fast_rand_proj(
            self.src_intermediate_quantities,
        )

    def _influence(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
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

        Returns:
            influence_scores (Tensor): Influence scores from the `TracInCPFastRandProj`
            method. Its shape is `(input_size, train_dataset_size)`, where `input_size`
            is the number of examples in the test batch, and
            `train_dataset_size` is the number of examples in
            training dataset `train_dataset`. For example:
            `influence_scores[i][j]` is the influence score for the j-th training
            example to the i-th example in the test batch.
        """
        # TODO: after D35721609 lands, use helper function
        # `TracInCP._influence_rand_proj` here to avoid duplicated logic
        input_projections = self._get_intermediate_quantities_tracincp_fast_rand_proj(
            inputs,
            self.projection_quantities,
            test=True,
        )

        src_projections = self.src_intermediate_quantities

        return torch.matmul(input_projections, src_projections.T)

    def _get_k_most_influential(  # type: ignore[override]
        self,
        inputs: Tuple[Any, ...],
        k: int = 5,
        proponents: bool = True,
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
        input_projections = self._get_intermediate_quantities_tracincp_fast_rand_proj(
            inputs,
            self.projection_quantities,
            test=True,
        )
        multiplier = 1 if proponents else -1

        input_projections *= multiplier

        indices, distances = self.nearest_neighbors.get_nearest_neighbors(
            input_projections, k
        )

        distances *= multiplier

        return KMostInfluentialResults(indices, distances)

    @log_usage(part_of_slo=True)
    def self_influence(
        self,
        inputs: Optional[Union[Tuple[Any, ...], DataLoader]] = None,
        show_progress: bool = False,
        outer_loop_by_checkpoints: bool = False,
    ) -> Tensor:
        """
        NOT IMPLEMENTED - no need to implement `TracInCPFastRandProj.self_influence`,
        as `TracInCPFast.self_influence` is sufficient - the latter does not benefit
        from random projections, since no quantities associated with a training
        example are stored (other than its self influence score)

        Computes self influence scores for a single batch or a Pytorch `DataLoader`
        that yields batches. Note that if `inputs` is a single batch, this
        will call `model` on that single batch, and if `inputs` yields
        batches, this will call `model` on each batch that is yielded. Therefore,
        please ensure that for both cases, the batch(es) that `model` is called
        with are not too large, so that there will not be an out-of-memory error.

        Args:
            inputs (tuple or DataLoader): Either a single tuple of any, or a
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
                    and all batches that `inputs` represents. Therefore, the
                    total number of (checkpoint, batch) combinations that need to be
                    iterated over is
                    (# of checkpoints x # of batches that `inputs` represents).
                    If `show_progress` is True, the total number of such combinations
                    that have been iterated over is displayed. It will try to use tqdm
                    if available for advanced features (e.g. time estimation).
                    Otherwise, it will fallback to a simple output of progress.
                    Default: False
            outer_loop_by_checkpoints (bool, optional): If performing an outer
                    iteration over checkpoints; see method description for more
                    details.
                    Default: False

        Returns:
            self_influence_scores (Tensor): This is a 1D tensor containing the self
                    influence scores of all examples in `inputs`, regardless of
                    whether it represents a single batch or a `DataLoader` that yields
                    batches.
        """
        warnings.warn(
            (
                "WARNING: If calculating self influence scores, when only considering "
                "gradients with respect to the last fully-connected layer, "
                "`TracInCPFastRandProj` should not be used. Instead, please use "
                "`TracInCPFast`. This is because when calculating self influence "
                "scores, no quantities associated with a training example are stored "
                "so that memory-saving benefit of the random projections used by "
                "`TracInCPFastRandProj`needed. Further considering the fact that "
                "random projections results only in approximate self influence "
                "scores, there is no reason to use `TracInCPFastRandProj` when "
                "calculating self influence scores."
            ),
            stacklevel=1,
        )
        raise NotImplementedError

    @log_usage(part_of_slo=True)
    def influence(  # type: ignore[override]
        self,
        inputs: Optional[Tuple[Any, ...]] = None,
        k: int = 5,
        proponents: bool = True,
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
              batch. `influence_scores` contains the corresponding influence scores.
              In particular, `influence_scores[i][j]` is the influence score of example
              `indices[i][j]` in `train_dataset` on example `i` in the test batch
              represented by `inputs`.
        """
        assert inputs is not None, (
            "`inputs` argument is required."
            "`TracInCPFastRandProj` does not support computing self influence scores"
            "Even if it did, one would use the `self_influence` method."
        )
        return _influence_route_to_helpers(
            self,
            inputs,
            k,
            proponents,
        )

    def _set_projections_tracincp_fast_rand_proj(
        self,
        dataloader: DataLoader,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        returns the variables `jacobian_projection` and `layer_input_projection`
        if needed, based on `self.projection_dim`. The two variables are
        used by `self._get_intermediate_quantities_fast_rand_proj`. They are both None
        if projection is not needed, due to the intermediate quantities (see the
        `_get_intermediate_quantities_fast_rand_proj` method for details) being no
        greater than `self.projection_dim` * C even without projection, where C is the
        number of checkpoints in the `checkpoints` argument to
        `TracInCPFastRandProj.__init__`.

        Args:

            dataloader (DataLoader): determining the projection requires knowing the
                    dimensionality of the last layer's parameters (`jacobian_dim`
                    below) and its input (`layer_input_dim` below). These are
                    determined by passing a batch to `model`. `dataloader`
                    provides that batch.

        Returns:
            jacobian_projection (Tensor or None): Projection matrix to apply to
                    Jacobian of last layer to reduce its dimension, if needed.
                    None otherwise.
            input_projection (Tensor or None): Projection matrix to apply to input of
                    last layer to reduce its dimension, if needed. None otherwise.
        """
        # figure out projection dimensions, if needed

        projection_dim = self.projection_dim

        projection_quantities = None

        if not (projection_dim is None):

            # figure out original dimensions by looking at data, passing through network
            self.checkpoints_load_func(self.model, next(iter(self.checkpoints)))

            batch = next(iter(dataloader))
            batch_jacobians, batch_layer_inputs = _basic_computation_tracincp_fast(
                self,
                batch[0:-1],
                batch[-1],
                self.loss_fn,
                self.reduction_type,
            )

            jacobian_dim = batch_jacobians.shape[
                1
            ]  # this is the dimension of the output of the last fully-connected layer
            layer_input_dim = batch_layer_inputs.shape[
                1
            ]  # this is the dimension of the input of the last fully-connected layer
            device = batch_jacobians.device
            dtype = batch_jacobians.dtype

            # choose projection if needed
            # without projection, the dimension of the intermediate quantities returned
            # by `_get_intermediate_quantities_fast_rand_proj` will be
            # `jacobian_dim` * `layer_input_dim` * number of checkpoints
            # this is because for each checkpoint, we compute a "partial" intermediate
            # quantity, and the intermediate quantity is the concatenation of the
            # "partial" intermediate quantities, and the dimension of each "partial"
            # intermediate quantity, without projection, is `jacobian_dim` *
            # `layer_input_dim`. However, `projection_dim` refers to the maximum
            # allowable dimension of the "partial" intermediate quantity. Therefore,
            # we only project if `jacobian_dim` * `layer_input_dim` > `projection_dim`.
            # `projection_dim` corresponds to the variable d in the top of page 15 of
            # the TracIn paper: https://arxiv.org/abs/2002.08484.
            if jacobian_dim * layer_input_dim > projection_dim:
                jacobian_projection_dim = min(int(projection_dim**0.5), jacobian_dim)
                layer_input_projection_dim = min(
                    int(projection_dim**0.5), layer_input_dim
                )
                jacobian_projection = torch.normal(
                    torch.zeros(jacobian_dim, jacobian_projection_dim),
                    1.0 / jacobian_projection_dim**0.5,
                )
                layer_input_projection = torch.normal(
                    torch.zeros(layer_input_dim, layer_input_projection_dim),
                    1.0 / layer_input_projection_dim**0.5,
                )

                projection_quantities = jacobian_projection.to(
                    device=device, dtype=dtype
                ), layer_input_projection.to(device=device, dtype=dtype)

        return projection_quantities

    def _process_src_intermediate_quantities_tracincp_fast_rand_proj(
        self,
        src_intermediate_quantities: torch.Tensor,
    ) -> None:
        """
        Assumes `self._get_intermediate_quantities_tracin_fast_rand_proj` returns
        vector representations for each example, and that influence between a
        training and test example is obtained by taking the dot product of their
        vector representations. In this case, given a test example, its proponents
        can be found by storing the vector representations for training examples
        into a data structure enablng fast largest-dot-product computation. This
        method creates that data structure. This method has side effects.

        Args:

            src_intermediate_quantities (Tensor): the output of the
                    `_get_intermediate_quantities_tracin_fast_rand_proj` function when
                    applied to training dataset `train_dataset`. This
                    output is the vector representation of all training examples.
                    The dot product between the representation of a training example
                    and the representation of a test example gives the influence score
                    of the training example on the test example.
        """
        self.nearest_neighbors.setup(src_intermediate_quantities)

    def _get_intermediate_quantities_tracincp_fast_rand_proj(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
        projection_quantities: Optional[Tuple[torch.Tensor, torch.Tensor]],
        test: bool = False,
    ) -> torch.Tensor:
        r"""
        This method computes vectors that can be used to compute influence. (see
        Appendix F, page 15). Crucially, the influence score between a test example
        and a training example is simply the dot product of their respective
        vectors. This means that the training example with the largest influence score
        on a given test example can be found using a nearest-neighbor (more
        specifically, largest dot-product) data structure.

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
            projection_quantities (tuple or None): Is either the two tensors defining
                    the randomized projections to apply, or None, which means no
                    projection is to be applied.
            test (bool): If True, the intermediate quantities are computed using
                    `self.test_loss_fn`. Otherwise, they are computed using
                    `self.loss_fn`.
                    Default: False

        Returns:
            intermediate_quantities (Tensor): A tensor of dimension
                    (N, D * C), where N is total number of examples in `dataloader`, C
                    is the number of checkpoints passed as the `checkpoints` argument
                    of `TracInCPFastRandProj.__init__`, and each row represents the
                    vector for an example. Regarding D: Let I be the dimension of the
                    output of the last fully-connected layer times the dimension of the
                    input of the last fully-connected layer. If `self.projection_dim`
                    is specified in initialization,
                    D = min(I * C, `self.projection_dim` * C). Otherwise, D = I * C.
                    In summary, if `self.projection_dim` is None, the dimension of each
                    vector will be determined by the size of the input and output of
                    the last fully-connected layer of `model`. Otherwise,
                    `self.projection_dim` must be an int, and random projection will be
                    performed to ensure that the vector is of dimension no more than
                    `self.projection_dim` * C. `self.projection_dim` corresponds to
                    the variable d in the top of page 15 of the TracIn paper:
                    https://arxiv.org/abs/2002.08484.
        """
        # if `inputs` is not a `DataLoader`, turn it into one.
        inputs = _format_inputs_dataset(inputs)

        # internally, whether `projection_quantities` is None determines whether
        # any projection will be applied to reduce the dimension of the "embedding"
        # vectors. If projection will be applied, there are actually 2 different
        # projection matrices - one to project the `input_jacobians`, and one to
        # project the `layer_inputs`. See below for details of those two quantities.
        # here, we extract the corresponding projection matrices for those two
        # quantities, if doing projection. Note that the same projections are used
        # for each checkpoint.
        project = False
        if projection_quantities is not None:
            project = True
            jacobian_projection, layer_input_projection = projection_quantities

        # for each checkpoint, we will populate a list containing the contribution of
        # the checkpoint for each batch
        # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
        #  `typing.List[<element type>]` to avoid runtime subscripting errors.
        checkpoint_contributions: List[Union[List, Tensor]] = [
            [] for _ in self.checkpoints
        ]

        # the "embedding" vector is the concatenation of contributions from each
        # checkpoint, which we compute one by one
        for j, checkpoint in enumerate(self.checkpoints):

            assert (
                checkpoint is not None
            ), "None returned from `checkpoints`, cannot load."

            learning_rate = self.checkpoints_load_func(self.model, checkpoint)
            learning_rate_root = learning_rate**0.5

            # after loading a checkpoint, we compute the contribution of that
            # checkpoint, for *all* batches (instead of a single batch). this enables
            # increased efficiency.
            for batch in inputs:

                # compute `input_jacobians` and `layer_inputs`, for a given checkpoint
                # using a helper function. `input_jacobians` is a 2D tensor,
                # where each row is the jacobian of the loss, with respect to the
                # *output* of the last fully-connected layer. `layer_inputs` is a 2D
                # tensor, where each row is the *input* to the last fully-connected
                # layer. For both, the length is the number of examples in `batch`
                input_jacobians, layer_inputs = _basic_computation_tracincp_fast(
                    self,
                    batch[0:-1],
                    batch[-1],
                    self.test_loss_fn,
                    self.test_reduction_type,
                )

                # if doing projection, project those two quantities
                if project:

                    # pyre-fixme[61]: `jacobian_projection` is undefined, or not
                    #  always defined.
                    input_jacobians = torch.matmul(input_jacobians, jacobian_projection)

                    # pyre-fixme[61]: `layer_input_projection` is undefined, or not
                    #  always defined.
                    layer_inputs = torch.matmul(layer_inputs, layer_input_projection)

                # for an example, the contribution to the "embedding" vector from each
                # checkpoint is the outer product of its `input_jacobian` and its
                # `layer_input`, flattened to a 1D tensor. here, we perform this
                # for the entire batch. we append the contribution to a list containing
                # the contribution of all batches, from the checkpoint.
                # pyre-fixme[24]: Generic type `list` expects 1 type parameter, use
                #  `typing.List[<element type>]` to avoid runtime subscripting errors.
                cast(list, checkpoint_contributions[j]).append(
                    torch.matmul(
                        torch.unsqueeze(
                            input_jacobians, 2
                        ),  # size is (batch_size, output_size, 1)
                        torch.unsqueeze(
                            layer_inputs, 1
                        ),  # size is (batch_size, 1, input_size)
                    ).flatten(
                        start_dim=1
                    )  # matmul does a batched matrix multiplication to return a 3D
                    # tensor. each element along the batch (0-th) dimension is the
                    # matrix product of a (output_size, 1) and (1, input_size) tensor
                    # in other words, each element is an outer product, and the matmul
                    # is just doing a batched outer product. this is what we want, as
                    # the contribution to the "embedding" for an example is the outer
                    # product of the last layer's input and the gradient of its output.
                    # finally, we flatten the 3rd dimension so that the contribution to
                    # the embedding for this checkpoint is a 2D tensor, i.e. each
                    # example's contribution to the embedding is a 1D tensor.
                    * learning_rate_root
                )

            # once we have computed the contribution from each batch, for a given
            # checkpoint, we concatenate them along the batch dimension to get a
            # single 2D tensor for that checkpoint
            checkpoint_contributions[j] = torch.cat(
                checkpoint_contributions[j], dim=0  # type: ignore
            )

        # finally, we concatenate along the checkpoint dimension, to get a tensor of
        # shape (batch_size, projection_dim * number of checkpoints)
        # each row in this result is the "embedding" vector for an example in `batch`
        return torch.cat(checkpoint_contributions, dim=1)  # type: ignore

    @log_usage(part_of_slo=True)
    def compute_intermediate_quantities(
        self,
        inputs: Union[Tuple[Any, ...], DataLoader],
    ) -> Tensor:
        """
        Computes "embedding" vectors for all examples in a single batch, or a
        `Dataloader` that yields batches. These embedding vectors are constructed so
        that the influence score of a training example on a test example is simply the
        dot-product of their corresponding vectors. Please see the documentation for
        `TracInCPFastRandProj.__init__` for more details. Allowing a `DataLoader`
        yielding batches to be passed in (as opposed to a single batch) gives the
        potential to improve efficiency, because we load each checkpoint only once in
        this method call. Thus if a `DataLoader` yielding batches is passed in, this
        reduces the total number of times each checkpoint is loaded for a dataset,
        compared to if a single batch is passed in. The reason we do not just increase
        the batch size is that for large models, large batches do not fit in memory.

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

        Returns:
            intermediate_quantities (Tensor): A tensor of dimension
                    (N, D * C), where N is total number of examples in
                    `inputs`, C is the number of checkpoints passed as the
                    `checkpoints` argument of `TracInCPFastRandProj.__init__`, and each
                    row represents the vector for an example. Regarding D: Let I be the
                    dimension of the output of the last fully-connected layer times the
                    dimension of the input of the last fully-connected layer. If
                    `self.projection_dim` is specified in initialization,
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
        return self._get_intermediate_quantities_tracincp_fast_rand_proj(
            inputs, self.projection_quantities
        )
