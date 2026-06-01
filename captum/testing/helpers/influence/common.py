# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import inspect
import os
import unittest
from functools import partial
from inspect import isfunction
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.influence import DataInfluence
from captum.influence._core.arnoldi_influence_function import ArnoldiInfluenceFunction
from captum.influence._core.influence_function import NaiveInfluenceFunction
from captum.influence._core.tracincp_fast_rand_proj import (
    TracInCPFast,
    TracInCPFastRandProj,
)
from parameterized import parameterized
from parameterized.parameterized import param
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


# pyre-fixme[2]: Parameter must be annotated.
def _isSorted(x, key=lambda x: x, descending=True) -> bool:
    if descending:
        return all(key(x[i]) >= key(x[i + 1]) for i in range(len(x) - 1))
    else:
        return all(key(x[i]) <= key(x[i + 1]) for i in range(len(x) - 1))


# pyre-fixme[2]: Parameter must be annotated.
def _wrap_model_in_dataparallel(net) -> Module:
    alt_device_ids = [0] + list(range(torch.cuda.device_count() - 1, 0, -1))
    net = net.cuda()
    return torch.nn.DataParallel(net, device_ids=alt_device_ids)


def _move_sample_list_to_cuda(samples: List[Tensor]) -> List[Tensor]:
    return [s.cuda() for s in samples]


class ExplicitDataset(Dataset):
    def __init__(
        self,
        samples: Tensor,
        labels: Tensor,
        use_gpu: bool = False,
    ) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = self.samples.cuda()
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return (self.samples[idx], self.labels[idx])


class UnpackDataset(Dataset):
    def __init__(
        self,
        samples: List[Tensor],
        labels: Tensor,
        use_gpu: bool = False,
    ) -> None:
        # pyre-fixme[4]: Attribute must be annotated.
        self.samples, self.labels = samples, labels
        if use_gpu:
            self.samples = _move_sample_list_to_cuda(self.samples)
            self.labels = self.labels.cuda()

    def __len__(self) -> int:
        return len(self.samples[0])

    # pyre-fixme[3]: Return type must be annotated.
    def __getitem__(self, idx: int):
        """
        The signature of the returning item is: List[List], where the contents
        are: [sample_0, sample_1, ...] + [labels] (two lists concacenated).
        """
        return [lst[idx] for lst in self.samples] + [self.labels[idx]]


class IdentityDataset(ExplicitDataset):
    def __init__(
        self,
        num_features: int,
        use_gpu: bool = False,
    ) -> None:
        self.samples: Tensor = torch.diag(torch.ones(num_features))
        self.labels: Tensor = torch.zeros(num_features).unsqueeze(1)
        if use_gpu:
            self.samples = self.samples.cuda()
            self.labels = self.labels.cuda()


class RangeDataset(ExplicitDataset):
    def __init__(
        self,
        low: int,
        high: int,
        num_features: int,
        use_gpu: bool = False,
    ) -> None:
        self.samples: Tensor = (
            torch.arange(start=low, end=high, dtype=torch.float)
            .repeat(num_features, 1)
            .transpose(1, 0)
        )
        self.labels: Tensor = torch.arange(
            start=low, end=high, dtype=torch.float
        ).unsqueeze(1)
        if use_gpu:
            self.samples = self.samples.cuda()
            self.labels = self.labels.cuda()


class BinaryDataset(ExplicitDataset):
    def __init__(self, use_gpu: bool = False) -> None:
        self.samples: Tensor = F.normalize(
            torch.stack(
                (
                    torch.Tensor([1, 1]),
                    torch.Tensor([2, 1]),
                    torch.Tensor([1, 2]),
                    torch.Tensor([1, 5]),
                    torch.Tensor([0.01, 1]),
                    torch.Tensor([5, 1]),
                    torch.Tensor([1, 0.01]),
                    torch.Tensor([-1, -1]),
                    torch.Tensor([-2, -1]),
                    torch.Tensor([-1, -2]),
                    torch.Tensor([-1, -5]),
                    torch.Tensor([-5, -1]),
                    torch.Tensor([1, -1]),
                    torch.Tensor([2, -1]),
                    torch.Tensor([1, -2]),
                    torch.Tensor([1, -5]),
                    torch.Tensor([0.01, -1]),
                    torch.Tensor([5, -1]),
                    torch.Tensor([-1, 1]),
                    torch.Tensor([-2, 1]),
                    torch.Tensor([-1, 2]),
                    torch.Tensor([-1, 5]),
                    torch.Tensor([-5, 1]),
                    torch.Tensor([-1, 0.01]),
                )
            )
        )
        self.labels: Tensor = torch.cat(
            (
                torch.Tensor([1]).repeat(12, 1),
                torch.Tensor([-1]).repeat(12, 1),
            )
        )
        super().__init__(self.samples, self.labels, use_gpu)


class CoefficientNet(nn.Module):
    def __init__(self, in_features: int = 1) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 1, bias=False)
        self.fc1.weight.data.fill_(0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        return x


class BasicLinearNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_nodes: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, input: Tensor) -> Tensor:
        x = torch.tanh(self.linear1(input))
        return torch.tanh(self.linear2(x))


class MultLinearNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_nodes: int,
        out_features: int,
        num_inputs: int,
    ) -> None:
        super().__init__()
        self.pre = nn.Linear(in_features * num_inputs, in_features)
        self.linear1 = nn.Linear(in_features, hidden_nodes)
        self.linear2 = nn.Linear(hidden_nodes, out_features)

    def forward(self, *inputs: Tensor) -> Tensor:
        """
        The signature of inputs is a Tuple of Tensor,
        where the Tensor has the dimensions [num_inputs x in_features].
        It first concacenates the list and a linear layer to reduce the
        dimension.
        """
        inputs = self.pre(torch.cat(inputs, dim=1))
        x = torch.tanh(self.linear1(inputs))
        return torch.tanh(self.linear2(x))


class Linear(nn.Module):
    """
    a wrapper around `nn.Linear`, with purpose being to have an analogue to
    `UnpackLinear`, with both's only parameter being 'linear'. "infinitesimal"
    influence (i.e. that calculated by `InfluenceFunctionBase` implementations) for
    this simple module can be analytically calculated, so its purpose is for testing
    those implementations.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=False)

    def forward(self, input: Tensor) -> Tensor:
        return self.linear(input)


class UnpackLinear(nn.Module):
    """
    the analogue of `Linear` which unpacks inputs, serving the same purpose.
    """

    def __init__(self, in_features: int, num_inputs: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * num_inputs, 1, bias=False)

    def forward(self, *inputs: Tensor) -> Tensor:
        return self.linear(torch.cat(inputs, dim=1))


def get_random_data(
    in_features: int,
    out_features: int,
    num_examples: int,
    use_gpu: bool,
    unpack_inputs: bool,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    returns train_dataset, test_dataset and hessian_dataset constructed from
    random labels and random features, with features having shape
    [num_examples x num_features] and labels having shape [num_examples].

    Note: the random labels and features for different dataset needs to be
    generated together.
    Otherwise, some tests will fail (https://fburl.com/testinfra/737jnpip)
    """

    num_train = 32
    num_hessian = 22  # this needs to be high to prevent numerical issues
    num_inputs = 2 if unpack_inputs else 1

    labels = torch.normal(1, 2, (num_examples, out_features)).double()
    all_samples = [
        torch.normal(0, 1, (num_examples, in_features)).double()
        for _ in range(num_inputs)
    ]

    train_dataset = (
        UnpackDataset(
            [samples[:num_train] for samples in all_samples],
            labels[:num_train],
            use_gpu,
        )
        if unpack_inputs
        else ExplicitDataset(all_samples[0][:num_train], labels[:num_train], use_gpu)
    )

    hessian_dataset = (
        UnpackDataset(
            [samples[:num_hessian] for samples in all_samples],
            labels[:num_hessian],
            use_gpu,
        )
        if unpack_inputs
        else ExplicitDataset(
            all_samples[0][:num_hessian], labels[:num_hessian], use_gpu
        )
    )

    test_dataset = (
        UnpackDataset(
            [samples[num_train:] for samples in all_samples],
            labels[num_train:],
            use_gpu,
        )
        if unpack_inputs
        else ExplicitDataset(all_samples[0][num_train:], labels[num_train:], use_gpu)
    )
    return (train_dataset, hessian_dataset, test_dataset)


def _adjust_model(model: Module, gpu_setting: Optional[str]) -> Module:
    """
    Given a model, returns a copy of the model on GPU based on the provided
    `gpu_setting`.
    Or returns the original model on CPU if no valid setting is provided.

    Two valid settings are supported for now:
        - `'cuda'`: returned model is on gpu
        - `'cuda_data_parallel``: returned model is a `DataParallel` model,
        and on gpu

    The need to differentiate between `'cuda'` and `'cuda_data_parallel'`
    is that sometimes we may want to test a model that is on gpu, but is *not*
    wrapped in `DataParallel`.
    """
    if gpu_setting == "cuda_data_parallel":
        return _wrap_model_in_dataparallel(model)
    elif gpu_setting == "cuda":
        return model.cuda()
    else:
        return model


def is_gpu(gpu_setting: Optional[str]) -> bool:
    """
    Returns whether the model should be on gpu based on the given `gpu_setting` str.
    """
    return gpu_setting == "cuda_data_parallel" or gpu_setting == "cuda"


# pyre-fixme[3]: Return type must be annotated.
def get_random_model_and_data(
    # pyre-fixme[2]: Parameter must be annotated.
    tmpdir,
    # pyre-fixme[2]: Parameter must be annotated.
    unpack_inputs,
    return_test_data: bool = True,
    gpu_setting: Optional[str] = None,
    return_hessian_data: bool = False,
    model_type: str = "random",
):
    """
    returns a model, training data, and optionally data for computing the hessian
    (needed for `InfluenceFunctionBase` implementations) as features / labels, and
    optionally test data as features / labels.

    the data is always generated the same way. however depending on `model_type`,
    a different model and checkpoints are returned.
    - `model_type='random'`: the model is a 2-layer NN, and several checkpoints are
    generated
    - `model_type='trained_linear'`: the model is a linear model, and assumed to be
    eventually trained to optimality. therefore, we find the optimal parameters, and
    save a single checkpoint containing them. the training is done using the Hessian
    data, because the purpose of training the model is so that the Hessian is positive
    definite. since the Hessian is calculated using the Hessian data, it should be
    used for training. since it is trained to optimality using the Hessian data, we can
    guarantee that the Hessian is positive definite, so that different
    implementations of `InfluenceFunctionBase` can be more easily compared. (if the
    Hessian is not positive definite, we drop eigenvectors corresponding to negative
    eigenvalues. since the eigenvectors dropped in `ArnoldiInfluence` differ from those
    in `NaiveInfluenceFunction` due to the formers' use of Arnoldi iteration, we should
    only use models / data whose Hessian is positive definite, so that no eigenvectors
    are dropped). in short, this model / data are suitable for comparing different
    `InfluenceFunctionBase` implementations.
    - `model_type='trained_NN'`: the model is a 2-layer NN, and trained (not
    necessarily) to optimality using the Hessian data. since it is trained, for same
    reasons as for `model_type='trained_linear`, different implementations of
    `InfluenceFunctionBase` can be more easily compared, due to lack of numerical
    issues.

    `gpu_setting` specify whether the model is on gpu and whether it is a `DataParallel`
    model. More details in the `_adjust_model_for_gpu` API.
    """
    in_features, hidden_nodes = 5, 4
    num_inputs = 2
    use_gpu = is_gpu(gpu_setting)

    # generate data. regardless the model, the data is always generated the same way
    # the only exception is if the `model_type` is 'trained_linear', i.e. a simple
    # linear regression model. this is a simple model, and for simplicity, the
    # number of `out_features` is 1 in this case.
    if model_type == "trained_linear":
        out_features = 1
    else:
        out_features = 3

    num_samples = 50

    train_dataset, hessian_dataset, test_dataset = get_random_data(
        in_features, out_features, num_samples, use_gpu, unpack_inputs
    )

    net: Module  # Union[BasicLinearNet, MultLinearNet, Linear, UnpackLinear]
    if model_type == "random":
        net = (
            BasicLinearNet(in_features, hidden_nodes, out_features)
            if not unpack_inputs
            else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
        ).double()

        # generate checkpoints randomly
        num_checkpoints = 5

        for i in range(num_checkpoints):
            net.linear1.weight.data = torch.normal(  # type: ignore
                3, 4, (hidden_nodes, in_features)
            ).double()
            net.linear2.weight.data = torch.normal(  # type: ignore
                5, 6, (out_features, hidden_nodes)
            ).double()
            if unpack_inputs:
                net.pre.weight.data = torch.normal(  # type: ignore
                    3, 4, (in_features, in_features * num_inputs)
                ).double()
            checkpoint_name = "-".join(["checkpoint-reg", str(i + 1) + ".pt"])
            net_adjusted = _adjust_model(net, gpu_setting)
            torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

    elif model_type == "trained_linear":
        net = (
            Linear(in_features)
            if not unpack_inputs
            else UnpackLinear(in_features, num_inputs)
        ).double()

        # regardless of `unpack_inputs`, the model is a linear regression, so that
        # we can get the optimal trained parameters via least squares

        # turn input into a single tensor for use by least squares
        tensor_hessian_samples = (
            hessian_dataset.samples  # type: ignore
            if not unpack_inputs
            else torch.cat(hessian_dataset.samples, dim=1)  # type: ignore
        )

        # run least squares to get optimal trained parameters
        theta = torch.linalg.lstsq(
            hessian_dataset.labels,  # type: ignore
            tensor_hessian_samples,
        ).solution
        # the first `n` rows of `theta` contains the least squares solution, where
        # `n` is the number of features in `tensor_hessian_samples`
        theta = theta[: tensor_hessian_samples.shape[1]]

        # save that trained parameter as a checkpoint
        checkpoint_name = "checkpoint-final.pt"
        net.linear.weight.data = theta.contiguous()  # type: ignore
        net_adjusted = _adjust_model(net, gpu_setting)
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

    elif model_type == "trained_NN":
        net = (
            BasicLinearNet(in_features, hidden_nodes, out_features)
            if not unpack_inputs
            else MultLinearNet(in_features, hidden_nodes, out_features, num_inputs)
        ).double()

        net_adjusted = _adjust_model(net, gpu_setting)

        # train model using several optimization steps on Hessian data
        batch = next(iter(DataLoader(hessian_dataset, batch_size=len(hessian_dataset))))  # type: ignore # noqa: E501 line too long

        optimizer = torch.optim.Adam(net.parameters())
        num_steps = 200
        criterion = nn.MSELoss(reduction="sum")
        for _ in range(num_steps):
            optimizer.zero_grad()
            output = net_adjusted(*batch[:-1])
            loss = criterion(output, batch[-1])
            loss.backward()
            optimizer.step()

        # save that trained parameter as a checkpoint
        checkpoint_name = "checkpoint-final.pt"
        torch.save(net_adjusted.state_dict(), os.path.join(tmpdir, checkpoint_name))

    training_data = (
        # pyre-fixme[61]: `net_adjusted` is undefined, or not always defined.
        net_adjusted,
        train_dataset,
    )

    hessian_data = (hessian_dataset.samples, hessian_dataset.labels)  # type: ignore

    test_data = (test_dataset.samples, test_dataset.labels)  # type: ignore

    if return_test_data:
        if not return_hessian_data:
            return (*training_data, *test_data)
        else:
            return (*training_data, *hessian_data, *test_data)
    else:
        if not return_hessian_data:
            return training_data
        else:
            return (*training_data, *hessian_data)


# pyre-fixme[3]: Return type must be annotated.
def generate_symmetric_matrix_given_eigenvalues(
    eigenvalues: Union[Tensor, List[float]],
):
    """
    following https://github.com/google-research/jax-influence/blob/74bd321156b5445bb35b9594568e4eaaec1a76a3/jax_influence/test_utils.py#L123  # noqa: E501
    generate symmetric random matrix with specified eigenvalues.  this is used in
    `TestArnoldiInfluence._test_parameter_arnoldi_and_distill` either to check that
    `_parameter_arnoldi` does return the top eigenvalues of a symmetric random matrix,
    or that `_parameter_distill` does return the eigenvectors corresponding to the top
    eigenvalues of that symmetric random matrix.
    """
    # generate random matrix, then apply gram-schmidt to get random orthonormal basis
    D = len(eigenvalues)

    Q, _ = torch.linalg.qr(torch.randn((D, D)))
    return torch.matmul(Q, torch.matmul(torch.diag(torch.tensor(eigenvalues)), Q.T))


def generate_assymetric_matrix_given_eigenvalues(
    eigenvalues: Union[Tensor, List[float]],
) -> Tensor:
    """
    following https://github.com/google-research/jax-influence/blob/74bd321156b5445bb35b9594568e4eaaec1a76a3/jax_influence/test_utils.py#L105 # noqa: E501
    generate assymetric random matrix with specified eigenvalues. this is used in
    `TestArnoldiInfluence._test_parameter_arnoldi_and_distill` either to check that
    `_parameter_arnoldi` does return the top eigenvalues of a assymmetric random
    matrix, or that `_parameter_distill` does return the eigenvectors corresponding to
    the top eigenvalues of that assymmetric random matrix.
    """
    # the matrix M, given eigenvectors Q and eigenvalues L, should satisfy MQ = QL
    # or equivalently, Q'M' = LQ'.
    D = len(eigenvalues)
    Q_T = torch.randn((D, D))

    return torch.linalg.solve(
        Q_T, torch.matmul(torch.diag(torch.tensor(eigenvalues)), Q_T)
    ).T


class DataInfluenceConstructor:
    name: str = ""
    # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
    #  `typing.Type[<base type>]` to avoid runtime subscripting errors.
    data_influence_class: type

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `type` expects 1 type parameter, use
        #  `typing.Type[<base type>]` to avoid runtime subscripting errors.
        data_influence_class: type,
        name: Optional[str] = None,
        duplicate_loss_fn: bool = False,
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> None:
        """
        if `duplicate_loss_fn` is True, will explicitly pass the provided `loss_fn` as
        the `test_loss_fn` when constructing the TracInCPBase instance
        """
        self.data_influence_class = data_influence_class
        self.name = name if name else data_influence_class.__name__
        self.duplicate_loss_fn = duplicate_loss_fn
        # pyre-fixme[4]: Attribute must be annotated.
        self.kwargs = kwargs

    def __repr__(self) -> str:
        return self.name

    def __name__(self) -> str:
        return self.name

    def __call__(
        self,
        net: Module,
        dataset: Union[Dataset, DataLoader],
        tmpdir: str,
        batch_size: Union[int, None],
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        loss_fn: Optional[Union[Module, Callable]],
        # pyre-fixme[2]: Parameter must be annotated.
        **kwargs,
    ) -> DataInfluence:
        constructor_kwargs = self.kwargs.copy()
        constructor_kwargs.update(kwargs)
        # if `self.duplicate_loss_fn`, explicitly pass in `loss_fn` as `test_loss_fn`
        # when constructing the instance. Doing so should not affect the behavior of
        # the returned tracincp instance, since if `test_loss_fn` is not passed in,
        # the constructor sets `test_loss_fn` to be the same as `loss_fn`
        if self.duplicate_loss_fn:
            constructor_kwargs["test_loss_fn"] = loss_fn
        if self.data_influence_class is TracInCPFastRandProj:
            self.check_annoy()
        if self.data_influence_class in [TracInCPFast, TracInCPFastRandProj]:
            return self.data_influence_class(
                net,
                list(net.children())[-1],
                dataset,
                tmpdir,
                loss_fn=loss_fn,
                batch_size=batch_size,
                **constructor_kwargs,
            )
        elif self.data_influence_class in [
            NaiveInfluenceFunction,
            ArnoldiInfluenceFunction,
        ]:
            # for these implementations, only a single checkpoint is needed, not
            # a directory containing several checkpoints. therefore, given
            # directory `tmpdir`, we do not pass it directly to the constructor,
            # but instead find 1 checkpoint in it, and pass that to the
            # constructor
            checkpoint_name = sorted(os.listdir(tmpdir))[-1]
            checkpoint = os.path.join(tmpdir, checkpoint_name)

            return self.data_influence_class(
                net,
                dataset,
                checkpoint,
                loss_fn=loss_fn,
                batch_size=batch_size,
                **constructor_kwargs,
            )
        else:
            return self.data_influence_class(
                net,
                dataset,
                tmpdir,
                batch_size=batch_size,
                loss_fn=loss_fn,
                **constructor_kwargs,
            )

    def check_annoy(self) -> None:
        try:
            import annoy  # noqa
        except ImportError:
            raise unittest.SkipTest(
                (
                    f"Skipping tests for {self.data_influence_class.__name__}, "
                    "because it requires the Annoy module."
                )
            )


def generate_test_name(
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    testcase_func: Callable,
    param_num: str,
    # pyre-fixme[11]: Annotation `param` is not defined as a type.
    param: param,
    args_to_skip: Optional[List[str]] = None,
) -> str:
    """
    Creates human readable names for parameterized tests
    """

    if args_to_skip is None:
        args_to_skip = []
    param_strs = []

    func_param_names = list(inspect.signature(testcase_func).parameters)
    # skip the first 'self' parameter
    if func_param_names[0] == "self":
        func_param_names = func_param_names[1:]

    for i, arg in enumerate(param.args):
        if func_param_names[i] in args_to_skip:
            continue
        if isinstance(arg, bool):
            if arg:
                param_strs.append(func_param_names[i])
        elif isfunction(arg):
            param_strs.append(arg.__name__)
        else:
            args_str = str(arg)
            if args_str.isnumeric():
                param_strs.append(func_param_names[i])
            param_strs.append(args_str)
    return "%s_%s_%s" % (
        testcase_func.__name__,
        param_num,
        parameterized.to_safe_name("_".join(param_strs)),
    )


# pyre-fixme[24]: Generic type `partial` expects 1 type parameter.
# Should be partial[str] but will cause TypeError: 'type' object is not subscriptable
def build_test_name_func(args_to_skip: Optional[List[str]] = None) -> partial:
    """
    Returns function to generate human readable names for parameterized tests
    """

    return partial(generate_test_name, args_to_skip=args_to_skip)


def _format_batch_into_tuple(
    # pyre-fixme[24]: Generic type `tuple` expects at least 1 type parameter.
    inputs: Union[Tuple, Tensor],
    targets: Tensor,
    unpack_inputs: bool,
) -> Tuple[Union[Tensor, Tuple[Any, ...]], Tensor]:
    if unpack_inputs:
        return (*inputs, targets)
    else:
        return (inputs, targets)


GPU_SETTING_LIST = (
    ["", "cuda", "cuda_data_parallel"]
    if torch.cuda.is_available() and torch.cuda.device_count() != 0
    else [""]
)
