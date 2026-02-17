# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
import functools
from typing import Callable, List, Optional, Tuple, Union

import torch
from captum._utils.gradient import _extract_parameters_from_layers
from captum.influence._core.influence_function import (
    _get_dataset_embeddings_intermediate_quantities_influence_function,
    InfluenceFunctionBase,
    IntermediateQuantitiesInfluenceFunction,
)
from captum.influence._utils.common import (
    _compute_batch_loss_influence_function_base,
    _compute_jacobian_sample_wise_grads_per_batch,
    _dataset_fn,
    _format_inputs_dataset,
    _functional_call,
    _get_k_most_influential_helper,
    _influence_batch_intermediate_quantities_influence_function,
    _influence_helper_intermediate_quantities_influence_function,
    _influence_route_to_helpers,
    _load_flexible_state_dict,
    _parameter_add,
    _parameter_dot,
    _parameter_linear_combination,
    _parameter_multiply,
    _parameter_to,
    _params_to_names,
    _progress_bar_constructor,
    _self_influence_helper_intermediate_quantities_influence_function,
    _top_eigen,
    KMostInfluentialResults,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def _parameter_arnoldi(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    hvp: Callable,
    b: Tuple[Tensor, ...],
    n: int,
    tol: float,
    projection_device: torch.device,
    show_progress: bool,
) -> Tuple[List[Tuple[Tensor, ...]], Tensor]:
    r"""
    Given `hvp`, a function which computes the Hessian-vector product of an arbitrary
    vector `v` with an implicitly-defined Hessian matrix `A`, performs the Arnoldi
    iteration for `A` for `n` iterations.  (We use `A`, not `H` to refer to the
    Hessian, unlike elsewhere, because `H` is already used in the below explanation
    of the Arnoldi iteration.)

    For more details on the Arnoldi iteration, please see Trefethen and Bau, Chp 33.
    Running Arnoldi iteration for n iterations gives a basis for the Krylov subspace
    spanned by :math`\{b, Ab,..., A^{n-1}b\}`, as well as a `n+1` by `n` matrix
    :math`H_n` which is upper Hessenberg (all entries below the diagonal, except those
    adjoining it, are 0), whose first n rows represent the restriction of `A` to the
    Krylov subspace, using the basis. Here, `b` is an arbitrary initialization basis
    vector. The basis is assembled into a `D` by `n+1` matrix, where the last
    column is a "correction factor", i.e. not part of the basis, denoted
    :math`Q_{n+1}`. Letting :math`Q_n` denote the matrix with the first n columns of
    :math`Q_{n+1}`, the following equality is satisfied: :math`A=Q_{n+1} H_n Q_n'`.

    In this implementation, `v` is not actually a vector, but instead a tuple of
    tensors, because `hvp` being a Hessian-vector product, `v` lies in parameter-space,
    which Pytorch represents as tuples of tensors. This implementation avoids
    flattening `v` to a 1D tensor, which leads to scalability gains.

    Args:
        hvp (Callable): A callable that accepts an arbitrary tuple of tensors
                `v`, which represents a parameter, and returns
                `Av`, i.e. the multiplication of `v` with an implicitly defined matrix
                `A` of compatible dimension, which in practice is a Hessian-vector
                product.
        b (tensor): The Arnoldi iteration requires an initialization basis to
                construct the basis, typically randomly chosen. This is that basis,
                and is a tuple of tensors. We assume that the device of `b` is the same
                as the required device of input `v` to `hvp`. For example, if `hvp`
                computes HVP using a model that is on the GPU, then `b` should also be
                on the GPU.
        n (int): The number of iterations to run the iteration for.
        tol (float, optional): After many iterations, the already-obtained
                basis vectors may already approximately span the Krylov subspace,
                in which case the addition of additional basis vectors involves
                normalizing a vector with a small norm. These vectors are not
                necessary to include in the basis and furthermore, their small norm
                leads to numerical issues. Therefore we stop the Arnoldi iteration
                when the addition of additional vectors involves normalizing a
                vector with norm below a certain threshold. This argument specifies
                that threshold.
                Default: 1e-4
        projection_device (torch.device) The returned quantities (which will be used
                to define a projection of parameter-gradients, hence the name) are
                potentially memory intensive, because they represent a basis of a
                subspace in the space of parameters, which are potentially
                high-dimensional. Therefore we need to be careful of out-of-memory
                GPU errors. This argument represents the device where the returned
                quantities should be stored, and its choice requires balancing
                speed with GPU memory.
        show_progress (bool): If true, the progress of the iteration (i.e. number of
                basis vectors already determined) will be displayed. It will try to
                use tqdm if available for advanced features (e.g. time estimation).
                Otherwise, it will fallback to a simple output of progress.

    Returns:
        qs (list of tuple of tensors): A list of tuple of tensors, whose first `n`
                elements contain a basis for the Krylov subspace.
        H (tensor): A tensor with shape `(n+1, n)` whose first `n` rows represent
                the restriction of `A` to the Krylov subspace.
    """
    # because the HVP is the computational bottleneck, we always do HVP on
    # the same device as the model, which is assumed to be the device `b` is on
    computation_device = next(iter(b)).device

    # all entries of `b` have the same dtype, and so can be used to determine dtype
    # of `H`
    H = torch.zeros(n + 1, n, dtype=next(iter(b)).dtype).to(device=projection_device)
    qs = [
        _parameter_to(
            _parameter_multiply(
                b, torch.div(1.0, torch.pow(_parameter_dot(b, b), 0.5))
            ),
            device=projection_device,
        )
    ]

    iterates = range(1, n + 1)
    if show_progress:
        iterates = tqdm(iterates, desc="Running Arnoldi Iteration for step")

    for k in iterates:
        v = _parameter_to(
            hvp(_parameter_to(qs[k - 1], device=computation_device)),
            device=projection_device,
        )

        for i in range(k):
            H[i, k - 1] = _parameter_dot(qs[i], v)
            v = _parameter_add(v, _parameter_multiply(qs[i], -H[i, k - 1]))
        H[k, k - 1] = torch.pow(_parameter_dot(v, v), 0.5)

        if H[k, k - 1] < tol:
            break
        qs.append(_parameter_multiply(v, torch.div(1.0, H[k, k - 1])))

    # pyre-fixme[61]: `k` is undefined, or not always defined.
    return qs[:k], H[:k, : k - 1]


def _parameter_distill(
    qs: List[Tuple[Tensor, ...]],
    H: Tensor,
    k: Optional[int],
    hessian_reg: float,
    hessian_inverse_tol: float,
) -> Tuple[Tensor, List[Tuple[Tensor, ...]]]:
    """
    This takes the output of `_parameter_arnoldi`, and extracts the top-k eigenvalues
    / eigenvectors of the matrix that `_parameter_arnoldi` found the Krylov subspace
    for. In this documentation, we will refer to that matrix by `A`.

    Args:
        qs (list of tuple of tensors): A list of tuple of tensors, whose first `N`
                elements contain a basis for the Krylov subspace.
        H (tensor): A tensor with shape `(N+1, N)` whose first `N` rows represent
                the restriction of `A` to the Krylov subspace.
        k (int): The number of top eigenvalues / eigenvectors to return. Note that the
                actual number returned may be less, due to filtering based on
                `hessian_inverse_tol`.
        hessian_reg (float): hessian_reg (float): We add an entry to the diagonal of
                `H` to encourage it to be positive definite. This is that entry.
        hessian_inverse_tol (float): To compute the "square root" of `H` using the top
                eigenvectors / eigenvalues, the eigenvalues should be positive, and
                furthermore if above a tolerance, the inversion will be more
                numerically stable. Therefore, we only return eigenvectors /
                eigenvalues where the eigenvalue is above a tolerance. This argument
                specifies that tolerance. We do not compute the square root in this
                function, but assume the output of this function will be used for
                computing it, hence the need for this argument.

    Returns:
        (eigenvalues, eigenvectors) (tensor, list of tuple of tensors): `eigenvalues`
                is a 1D tensor of the top eigenvalues of `A`. Note that due to
                filtering based on `hessian_inverse_tol`, the actual number of
                eigenvalues may be less than `k`. The eigenvalues are in ascending
                order, mimicking the convention of `torch.linalg.eigh`. `eigenvectors`
                are the corresponding eigenvectors. Since `A` represents the Hessian
                of parameters, with the parameters represented as a tuple of tensors,
                the eigenvectors, because they represent parameters, are also
                tuples of tensors. Therefore, `eigenvectors` is a list of tuple of
                tensors.
    """
    # get rid of last basis of qs, last column of H, since they are not part of
    # the decomposition
    qs = qs[:-1]
    H = H[:-1]

    # if arnoldi basis is empty, raise exception
    if len(qs) == 0:
        raise Exception(
            "Arnoldi basis is empty. Consider increasing the `arnoldi_tol` argument"
        )

    # ls, vs are the top eigenvalues / eigenvectors.  however, the eigenvectors are
    # expressed as coordinates using the Krylov subspace basis, qs (each column of vs
    # represents a different eigenvector).
    ls, vs = _top_eigen(H, k, hessian_reg, hessian_inverse_tol)

    # if no positive eigenvalues exist, we cannot compute a low-rank
    # approximation of the square root of the hessian H, so raise exception
    if vs.shape[1] == 0:
        raise Exception(
            "Restriction of Hessian to Krylov subspace has no positive "
            "eigenvalues, so cannot take its square root."
        )

    # we want to express the top eigenvectors as coordinates using the standard basis.
    # each column of vs represents a different eigenvector, expressed as coordinates
    # using the Krylov subspace basis.  to express the eigenvector using the standard
    # basis, we use it as the coefficients in a linear combination with the Krylov
    # subspace basis, qs.
    vs_standard = [_parameter_linear_combination(qs, v) for v in vs.T]

    return ls, vs_standard


class ArnoldiInfluenceFunction(IntermediateQuantitiesInfluenceFunction):
    r"""
    This is a computationally-efficient implementation that computes the type of
    "infinitesimal" influence scored defined in the paper "Understanding Black-box
    Predictions via Influence Functions" by Koh et al
    (https://arxiv.org/pdf/1703.04730.pdf). This implementation does *not* follow
    the approach in that paper, however. Instead, it follows an implementation that is
    several orders of magnitudes faster, described in the paper "Scaling Up Influence
    Functions" by Schioppa et al (https://arxiv.org/pdf/2112.03052.pdf).

    This implementation computes a low-rank approximation of the inverse Hessian, i.e.
    a tall and skinny (with width k) matrix :math`R` such that
    :math`H^{-1} \approx RR'`, where k is small. In particular, let :math`V` be the
    matrix of width k whose columns contain the top-k eigenvectors of :math`H`, and let
    :math`S` be the k by k matrix whose diagonals contain the corresponding eigenvalues.
    This implementation lets :math`R=VS^{-0.5}`. Thus, the core computational step is
    computing the top-k eigenvalues / eigenvectors.

    This approximation is useful for several reasons:
    - It avoids numerical issues associated with inverting small eigenvalues
    - Since the influence score is given by
      :math`\nabla_\theta L(x)' H^{-1} \nabla_\theta L(z)`, which is approximated by
      :math`(\nabla_\theta L(x)' R) (\nabla_\theta L(z)' R)`, we can compute an
      "influence embedding" for a given example :math`x`, :math`\nabla_\theta L(x)' R`,
      such that the influence score of one example on another is approximately the
      dot-product of their respective embeddings.
    - Even for large models, we can store `R` in memory, provided k is small. This
      means influence embeddings (and thus influence scores) can be efficiently
      computed by doing a backwards pass to compute :math`\nabla_\theta L(x)` and then
      multiplying by :math`R'`. This is orders of magnitude faster than the previous
      LISSA approach of Koh et al, which to compute the influence score involving a
      given example, need to compute Hessian-vector products involving on the order
      of 10^4 examples.

    The key novelty of the approach by Schioppa et al is that it uses the Arnoldi
    iteration to find the top-k eigenvalues / eigenvectors of the Hessian without
    explicitly forming the Hessian. In more detail, the approach first runs the
    Arnoldi iteration, which only requires the ability to compute Hessian-vector
    products, to find a Krylov subspace of moderate dimension, i.e. 200. It then finds
    the top-k eigevalues / eigenvectors of the restriction of the Hessian to the
    subspace, where k is small, i.e. 50. Finally, it expresses the eigenvectors in
    the original basis. This approach for finding the top-k eigenvalues / eigenvectors
    is justified by the property of the Arnoldi iteration, that the Krylov subspace
    it returns tends to contain the top eigenvectors.

    This implementation require some computation time `__init__`, where it
    runs the Arnoldi iteration to calculate `R`. This computation is linear in
    `arnoldi_dim` as well as the size of `hessian_dataset`. After that initial
    overhead, calculation of influence scores is quick, only requiring a backwards pass
    and multiplication, per example.

    Unlike `NaiveInfluenceFunction`, this implementation does not flatten any
    parameters, as the 2D Hessian is never formed, and Pytorch's Hessian-vector
    implementation (`torch.autograd.functional.hvp`) allows the input and output
    vector to be a tuple of tensors. Avoiding flattening / unflattening parameters
    brings scalability gains.
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
        loss_fn: Optional[Union[Module, Callable[[Tensor, Tensor], Tensor]]] = None,
        batch_size: Union[int, None] = 1,
        hessian_dataset: Optional[Union[Dataset, DataLoader]] = None,
        test_loss_fn: Optional[
            Union[Module, Callable[[Tensor, Tensor], Tensor]]
        ] = None,
        sample_wise_grads_per_batch: bool = False,
        projection_dim: int = 50,
        seed: int = 0,
        arnoldi_dim: int = 200,
        arnoldi_tol: float = 1e-1,
        hessian_reg: float = 1e-3,
        hessian_inverse_tol: float = 1e-4,
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
                    If not provided, the loss function for test examples is assumed to
                    be the same as the loss function for training examples, i.e.
                    `loss_fn`.
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
            arnoldi_dim (int, optional): Calculating the low-rank approximation of the
                    (inverse) Hessian requires approximating the Hessian's top
                    eigenvectors / eigenvalues. This is done by first computing a
                    Krylov subspace via the Arnoldi iteration, and then finding the top
                    eigenvectors / eigenvalues of the restriction of the Hessian to the
                    Krylov subspace. Because only the top eigenvectors / eigenvalues
                    computed in the restriction will be similar to those in the full
                    space, `arnoldi_dim` should be chosen to be larger than
                    `projection_dim`. In the paper, they often choose `projection_dim`
                    to be between 10 and 100, and `arnoldi_dim` to be 200. Please see
                    the paper as well as Trefethen and Bau, Chapters 33-34 for more
                    details on the Arnoldi iteration.
                    Default: 200
            arnoldi_tol (float, optional): After many iterations, the already-obtained
                    basis vectors may already approximately span the Krylov subspace,
                    in which case the addition of additional basis vectors involves
                    normalizing a vector with a small norm. These vectors are not
                    necessary to include in the basis and furthermore, their small norm
                    leads to numerical issues. Therefore we stop the Arnoldi iteration
                    when the addition of additional vectors involves normalizing a
                    vector with norm below a certain threshold. This argument specifies
                    that threshold.
                    Default: 1e-4
            hessian_reg (float, optional): After computing the basis for the Krylov
                    subspace, the restriction of the Hessian to the subspace may not be
                    positive definite, which is required, as we compute a low-rank
                    approximation of its square root via eigen-decomposition.
                    `hessian_reg` adds an entry to the diagonals of the restriction of
                    the Hessian to encourage it to be positive definite. This argument
                    specifies that entry. Note that the regularized Hessian (i.e. with
                    `hessian_reg` added to its diagonals) does not actually need to be
                    positive definite - it just needs to have at least 1 positive
                    eigenvalue.
                    Default: 1e-3
            hessian_inverse_tol: (float) The tolerance to use when computing the
                    pseudo-inverse of the (square root of) hessian, restricted to the
                    Krylov subspace.
                    Default: 1e-4
            projection_on_cpu (bool, optional): Whether to move the projection,
                    i.e. low-rank approximation of the inverse Hessian, to cpu, to save
                    gpu memory.
                    Default: True
            show_progress (bool, optional): In initialization, the Arnoldi iteration
                    and the subroutine it uses (calculating Hessian-vector products
                    over batches in `hessian_dataset`) can take a long time. If
                    `show_progress` is true, the progress of both computations
                    (number of steps in Arnoldi iteration, number of batches processed
                    in computing Hessian-vector products) will be displayed. It will
                    try to use tqdm if available for advanced features (e.g. time
                    estimation). Otherwise, it will fallback to a simple output of
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

        self.arnoldi_dim = arnoldi_dim
        self.arnoldi_tol = arnoldi_tol
        self.hessian_reg = hessian_reg
        self.hessian_inverse_tol = hessian_inverse_tol

        # infer the device the model is on.  all parameters are assumed to be on the
        # same device
        self.model_device: torch.device = next(model.parameters()).device

        self.R: List[Tuple[Tensor, ...]] = (
            self._retrieve_projections_arnoldi_influence_function(
                self.hessian_dataloader,
                projection_on_cpu,
                show_progress,
            )
        )

    def _retrieve_projections_arnoldi_influence_function(
        self,
        dataloader: DataLoader,
        projection_on_cpu: bool,
        show_progress: bool,
    ) -> List[Tuple[Tensor, ...]]:
        """

        Returns the `R` described in the documentation for
        `ArnoldiInfluenceFunction`. The returned `R` represents a set of
        parameters in parameter space. However, since this implementation does *not*
        flatten parameters, each of those parameters is represented as a tuple of
        tensors. Therefore, `R` is represented as a list of tuple of tensors, and
        can be viewed as a linear function that takes in a tuple of tensors
        (representing a parameter), and returns a vector, where the i-th entry is
        the dot-product (as it would be defined over tuple of tensors) of the parameter
        (i.e. the input to the linear function) with the i-th entry of `R`.

        Can specify that projection should always be saved on cpu. if so, gradients are
        always moved to same device as projections before multiplying (moving
        projections to gpu when multiplying would defeat the purpose of moving them to
        cpu to save gpu memory).

        Returns:
            R (list of tuple of tensors): List of tuple of tensors of length
                    `projection_dim` (initialization argument). Each element
                    corresponds to a parameter in parameter-space, is represented as a
                    tuple of tensors, and together, define a projection that can be
                    applied to parameters (represented as tuple of tensors).
        """
        # create function that computes hessian-vector product, given a vector
        # represented as a tuple of tensors

        # first figure out names of params that require gradients. this is need to
        # create that function, as it replaces params based on their names
        params = tuple(
            self.model.parameters()
            if self.layer_modules is None
            else _extract_parameters_from_layers(self.layer_modules)
        )
        # the same position in `params` and `param_names` correspond to each other
        param_names = _params_to_names(params, self.model)

        # get factory that given a batch, returns a function that given params as
        # tuple of tensors, returns loss over the batch
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def tensor_tuple_loss_given_batch(batch):
            # pyre-fixme[53]: Captured variable `param_names` is not annotated.
            # pyre-fixme[53]: Captured variable `batch` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def tensor_tuple_loss(*params):
                # `params` is a tuple of tensors, and assumed to be order specified by
                # `param_names`
                features, labels = tuple(batch[0:-1]), batch[-1]

                _output = _functional_call(
                    self.model, dict(zip(param_names, params)), features
                )

                # compute the total loss for the batch, adjusting the output of
                # `self.loss_fn` based on `self.reduction_type`
                return _compute_batch_loss_influence_function_base(
                    self.loss_fn, _output, labels, self.reduction_type
                )

            return tensor_tuple_loss

        # define function that given batch and vector, returns HVP of loss using the
        # batch and vector
        # pyre-fixme[53]: Captured variable `params` is not annotated.
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def batch_HVP(batch, v):
            tensor_tuple_loss = tensor_tuple_loss_given_batch(batch)
            return torch.autograd.functional.hvp(tensor_tuple_loss, params, v=v)[1]

        # define function that returns HVP of loss over `dataloader`, given a
        # specified vector
        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def HVP(v):
            _hvp = None

            _dataloader = dataloader
            if show_progress:
                _dataloader = tqdm(
                    dataloader, desc="processing `hessian_dataset` batch"
                )

            # the HVP of loss using the entire `dataloader` is the sum of the
            # per-batch HVP's
            return _dataset_fn(_dataloader, batch_HVP, _parameter_add, v)

            for batch in _dataloader:
                hvp = batch_HVP(batch, v)
                if _hvp is None:
                    _hvp = hvp
                else:
                    _hvp = _parameter_add(_hvp, hvp)
            return _hvp

        # now that can compute the hessian-vector product (of loss over `dataloader`),
        # can perform arnoldi iteration

        # we always perform the HVP computations on the device where the model is.
        # effectively this means we do the computations on gpu if available. this
        # is necessary because the HVP is computationally expensive.

        # get initial random vector, and place it on the same device as the model.
        # `_parameter_arnoldi` needs to know which device the model is on, and
        # will infer it through the device of this random vector
        b = _parameter_to(
            tuple(torch.randn_like(param) for param in params),
            device=self.model_device,
        )

        # perform the arnoldi iteration, see its documentation for what its return
        # values are.  note that `H` is *not* the Hessian.
        qs, H = _parameter_arnoldi(
            HVP,
            b,
            self.arnoldi_dim,
            self.arnoldi_tol,
            torch.device("cpu") if projection_on_cpu else self.model_device,
            show_progress,
        )

        # `ls`` and `vs`` are (approximately) the top eigenvalues / eigenvectors of the
        # matrix used (implicitly) to compute Hessian-vector products by the `HVP`
        # input to `_parameter_arnoldi`. this matrix is the Hessian of the loss,
        # summed over the examples in `dataloader`. note that because the vectors in
        # the Hessian-vector product are actually tuples of tensors representing
        # parameters, `vs`` is a list of tuples of tensors.  note that here, `H` is
        # *not* the Hessian (`qs` and `H` together define the Krylov subspace of the
        # Hessian)

        ls, vs = _parameter_distill(
            qs, H, self.projection_dim, self.hessian_reg, self.hessian_inverse_tol
        )

        # if `vs` were a 2D tensor whose columns contain the top eigenvectors of the
        # aforementioned hessian, then `R` would be `vs @ torch.diag(ls ** -0.5)`, i.e.
        # scaling each column of `vs` by the corresponding entry in `ls ** -0.5`.
        # however, since `vs` is instead a list of tuple of tensors, `R` should be
        # a list of tuple of tensors, where each entry in the list is scaled by the
        # corresponding entry in `ls ** 0.5`, which we first compute.
        ls = torch.pow(torch.div(1.0, ls), 0.5)

        # then, scale each entry in `vs` by the corresponding entry in `ls ** 0.5`
        # since each entry in `vs` is a tuple of tensors, we use a helper function
        # that takes in a tuple of tensors, and a scalar, and multiplies every tensor
        # by the scalar.
        return [_parameter_multiply(v, l) for (v, l) in zip(vs, ls)]

    def compute_intermediate_quantities(
        self,
        inputs_dataset: Union[Tuple[Tensor, ...], DataLoader],
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
        description. Each element of `R` and :math`\nabla_\theta L(x)` lie in
        parameter-space. Therefore, if parameter-space were 1D, so that `R` were
        a 2D tensor whose columns are different elements in parameter-space, we would
        compute the embeddings for a batch by assembling :math`\nabla_\theta L(x)` for
        all examples `x` in the batch as rows in a 2D "batch parameter-gradient"
        tensor, and right-multiplying by `R`. However, parameter-space in this
        implementation is actually a tuple of tensors. So we do the analogous
        computation given this representation of parameter-space.

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
            return_on_cpu (bool, optional): Whether to return the vectors on the cpu
                    (or if not, the gpu). If None, is set to the device that the model
                    is on.
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

        # infer model / data device through model. return device is same as that of
        # model unless explicitly specified
        return_device: Optional[torch.device] = None
        if return_on_cpu is None:
            return_device = self.model_device
        else:
            return_device = torch.device("cpu") if return_on_cpu else self.model_device

        # choose the correct loss function and reduction type based on `test`
        loss_fn: Optional[Union[Module, Callable[[Tensor, Tensor], Tensor]]] = (
            self.test_loss_fn if test else self.loss_fn
        )
        reduction_type: str = self.test_reduction_type if test else self.reduction_type

        # define a helper function that returns the embeddings for a batch
        def get_batch_embeddings(batch: List[Tensor]) -> Tensor:
            # get gradient
            features, labels = tuple(batch[0:-1]), batch[-1]
            # `jacobians`` is a tensor of tuples. unlike parameters, however, the first
            # dimension is a batch dimension
            jacobians: Tuple[Tensor, ...] = (
                _compute_jacobian_sample_wise_grads_per_batch(
                    self, features, labels, loss_fn, reduction_type
                )
            )

            # `jacobians`` contains the per-example parameters for a batch. this
            # function takes in `params`, a tuple of tensors representing a single
            # parameter setting, and for each example, computes the dot-product of its
            # per-example parameter with `params`. in other words, given `params`,
            # representing a basis vector, this function returns the coordinate of
            # each example in the batch along that basis. note that `jacobians` and
            # `params` are both tuple of tensors, with the same length. however, a
            # tensor in `jacobians` always has dimension 1 greater than the
            # corresponding tensor in `params`, because the tensors in `jacobians` have
            # a batch dimension (the 1st). to do this computation, the naive way would
            # be to convert `jacobians` to a list of tuple of tensors, and use
            # `_parameter_dot` to take the dot-product of each element in the list
            # with `params` to get a 1D tensor whose length is the batch size. however,
            # we can do the same computation without actually creating that list of
            # tuple of tensors by using broadcasting.
            def get_batch_coordinate(params: Tuple[Tensor, ...]) -> Tensor:
                batch_coordinate = None
                for _jacobians, param in zip(jacobians, params):
                    example_coordinate = torch.sum(
                        _jacobians * param.to(device=self.model_device).unsqueeze(0),
                        dim=tuple(range(1, len(_jacobians.shape))),
                    )
                    if batch_coordinate is None:
                        batch_coordinate = example_coordinate
                    else:
                        batch_coordinate += example_coordinate
                return (
                    batch_coordinate.to(device=return_device)
                    if batch_coordinate is not None
                    else torch.tensor(0).to(device=return_device)
                )

            # to get the embedding for the batch, we get the coordinates for the batch
            # corresponding to one parameter in `R`. We do this for every parameter in
            # `R`, and then concatenate.
            return torch.stack(
                [get_batch_coordinate(params) for params in self.R],
                dim=1,
            )

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
        inputs: Tuple[Tensor, ...],
        k: Optional[int] = None,
        proponents: bool = True,
        show_progress: bool = False,
    ) -> Union[Tensor, KMostInfluentialResults]:
        """
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
        return _influence_route_to_helpers(
            self,
            inputs,
            k,
            proponents,
            show_progress=show_progress,
        )

    def _get_k_most_influential(
        self,
        inputs: Union[Tuple[Tensor, ...], DataLoader],
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
        inputs: Union[Tuple[Tensor, ...], DataLoader],
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

    def self_influence(
        self,
        inputs_dataset: Optional[Union[Tuple[Tensor, ...], DataLoader]] = None,
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
