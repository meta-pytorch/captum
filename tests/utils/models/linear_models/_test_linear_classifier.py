# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import argparse
import random
from typing import cast, Optional

import captum._utils.models.linear_model.model as pytorch_model_module
import numpy as np
import numpy.typing as npt
import sklearn.datasets as datasets
import torch
from captum.testing.helpers.evaluate_linear_model import evaluate
from torch.utils.data import DataLoader, TensorDataset


# pyre-fixme[3]: Return type must be annotated.
def sklearn_dataset_to_loaders(
    # pyre-fixme[2]: Parameter must be annotated.
    data,
    train_prop: float = 0.7,
    batch_size: int = 64,
    num_workers: int = 4,
    shuffle: bool = False,
    one_hot: bool = False,
):
    xs, ys = data
    if one_hot and ys.dtype != float:
        oh = np.zeros((ys.size, ys.max() + 1))
        oh[np.arange(ys.size), ys] = 1
        ys = oh

    dataset = TensorDataset(torch.FloatTensor(xs), torch.FloatTensor(ys))

    lens = [int(train_prop * len(xs))]
    lens += [len(xs) - lens[0]]
    train_dset, val_dset = torch.utils.data.random_split(dataset, lens)

    train_loader = DataLoader(
        train_dset,
        batch_size=min(batch_size, lens[0]),
        shuffle=shuffle,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dset,
        batch_size=min(batch_size, lens[1]),
        num_workers=num_workers,
        shuffle=False,
    )

    return train_loader, val_loader, xs.shape[1], xs.shape[0]


# pyre-fixme[3]: Return type must be annotated.
def compare_to_sk_learn(
    max_epoch: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_prop: float,
    sklearn_model_type: str,
    pytorch_model_type: str,
    norm_type: Optional[str],
    objective: str,
    alpha: float,
    init_scheme: str = "zeros",
):
    if "LinearRegression" not in sklearn_model_type:
        sklearn_classifier = getattr(pytorch_model_module, sklearn_model_type)(
            alpha=alpha
        )
    else:
        sklearn_classifier = getattr(pytorch_model_module, sklearn_model_type)()

    pytorch_classifier = getattr(pytorch_model_module, pytorch_model_type)(
        norm_type=args.norm_type,
    )

    sklearn_stats = sklearn_classifier.fit(
        train_data=train_loader,
        norm_input=args.norm_sklearn,
    )
    pytorch_stats = pytorch_classifier.fit(
        train_data=train_loader,
        max_epoch=max_epoch,
        init_scheme=init_scheme,
        alpha=alpha,
    )

    sklearn_stats.update(evaluate(val_loader, sklearn_classifier))
    pytorch_stats.update(evaluate(val_loader, pytorch_classifier))

    train_stats_pytorch = evaluate(train_loader, pytorch_classifier)
    train_stats_sklearn = evaluate(train_loader, sklearn_classifier)

    o_pytorch = {"l2": train_stats_pytorch["l2"]}
    o_sklearn = {"l2": train_stats_sklearn["l2"]}

    pytorch_h = pytorch_classifier.representation()
    sklearn_h = sklearn_classifier.representation()
    if objective == "ridge":
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
        o_pytorch["l2_reg"] = alpha * pytorch_h.norm(p=2, dim=-1)
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
        o_sklearn["l2_reg"] = alpha * sklearn_h.norm(p=2, dim=-1)
    elif objective == "lasso":
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
        o_pytorch["l1_reg"] = alpha * pytorch_h.norm(p=1, dim=-1)
        # pyre-fixme[6]: For 2nd argument expected `Tensor` but got `float`.
        o_sklearn["l1_reg"] = alpha * sklearn_h.norm(p=1, dim=-1)

    rel_diff = cast(
        npt.NDArray,
        # pyre-fixme[6]: For 1st argument expected `int` but got `Union[int, Tensor]`.
        (sum(o_sklearn.values()) - sum(o_pytorch.values())),
    ) / abs(sum(o_sklearn.values()))
    return (
        {
            "objective_rel_diff": rel_diff.tolist(),
            "objective_pytorch": o_pytorch,
            "objective_sklearn": o_sklearn,
        },
        sklearn_stats,
        pytorch_stats,
    )


# pyre-fixme[2]: Parameter must be annotated.
def main(args) -> None:
    if args.seed:
        torch.manual_seed(0)
        random.seed(0)

    assert args.norm_type in [None, "layer_norm", "batch_norm"]

    print(
        "dataset,num_samples,dimensionality,objective_diff,objective_pytorch,"
        + "objective_sklearn,pytorch_time,sklearn_time,pytorch_l2_val,sklearn_l2_val"
    )
    for dataset in args.datasets:
        dataset_fn = getattr(datasets, dataset)
        data = dataset_fn(return_X_y=True)

        (
            train_loader,
            val_loader,
            in_features,
            num_samples,
        ) = sklearn_dataset_to_loaders(
            data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=args.shuffle,
            one_hot=args.one_hot,
        )

        similarity, sklearn_stats, pytorch_stats = compare_to_sk_learn(
            alpha=args.alpha,
            max_epoch=args.max_epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            train_prop=args.training_prop,
            pytorch_model_type=args.pytorch_model_type,
            sklearn_model_type=args.sklearn_model_type,
            norm_type=args.norm_type,
            init_scheme=args.init_scheme,
            objective=args.objective,
        )

        print(
            f"{dataset},{num_samples},{in_features},{similarity['objective_rel_diff']},"
            + f"{similarity['objective_pytorch']},{similarity['objective_sklearn']},"
            + f"{pytorch_stats['train_time']},{sklearn_stats['train_time']},"
            + f"{pytorch_stats['l2']},{sklearn_stats['l2']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train & test linear model with SGD + compare to sklearn"
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=[
            "load_boston",
            "load_breast_cancer",
            "load_diabetes",
            "fetch_california_housing",
        ],
    )
    parser.add_argument("--initial_lr", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shuffle", default=False, action="store_true")
    parser.add_argument("--one_hot", default=False, action="store_true")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--training_prop", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sklearn_model_type", type=str, default="Lasso")
    parser.add_argument("--pytorch_model_type", type=str, default="SGDLasso")
    parser.add_argument("--init_scheme", type=str, default="xavier")
    parser.add_argument("--norm_sklearn", default=False, action="store_true")
    parser.add_argument("--objective", type=str, default="lasso")
    args: argparse.Namespace = parser.parse_args()
    main(args)
