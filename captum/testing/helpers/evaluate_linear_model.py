#!/usr/bin/env python3
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import cast, Dict

import torch
from captum._utils.models.linear_model.model import LinearModel
from torch import Tensor
from torch.utils.data import DataLoader


def evaluate(test_data: DataLoader, classifier: LinearModel) -> Dict[str, Tensor]:
    classifier.eval()

    l1_loss = 0.0
    l2_loss = 0.0
    n = 0
    l2_losses = []
    with torch.no_grad():
        for data in test_data:
            if len(data) == 2:
                x, y = data
                w = None
            else:
                x, y, w = data

            out = classifier(x)

            y = y.view(x.shape[0], -1)
            assert y.shape == out.shape

            if w is None:
                l1_loss += (out - y).abs().sum(0).to(dtype=torch.float64)
                l2_loss += ((out - y) ** 2).sum(0).to(dtype=torch.float64)
                l2_losses.append(((out - y) ** 2).to(dtype=torch.float64))
            else:
                l1_loss += (
                    (w.view(-1, 1) * (out - y)).abs().sum(0).to(dtype=torch.float64)
                )
                l2_loss += (
                    (w.view(-1, 1) * ((out - y) ** 2)).sum(0).to(dtype=torch.float64)
                )
                l2_losses.append(
                    (w.view(-1, 1) * ((out - y) ** 2)).to(dtype=torch.float64)
                )

            n += x.shape[0]

    l2_losses = torch.cat(l2_losses, dim=0)
    assert n > 0

    # just to double check
    assert ((l2_losses.mean(0) - l2_loss / n).abs() <= 0.1).all()

    classifier.train()
    return {"l1": cast(Tensor, l1_loss / n), "l2": cast(Tensor, l2_loss / n)}
