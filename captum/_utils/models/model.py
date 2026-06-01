#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from torch import Tensor
from torch.utils.data import DataLoader


class Model(ABC):
    r"""
    Abstract Class to describe the interface of a trainable model to be used
    within the algorithms of captum.

    Please note that this is an experimental feature.
    """

    @abstractmethod
    def fit(
        self,
        train_data: DataLoader,
        **kwargs: object,
    ) -> Optional[Dict[str, Union[int, float, Tensor]]]:
        r"""
        Override this method to actually train your model.

        The specification of the dataloader will be supplied by the algorithm
        you are using within captum. This will likely be a supervised learning
        task, thus you should expect batched (x, y) pairs or (x, y, w) triples.

        Args:
            train_data (DataLoader):
                The data to train on

        Returns:
            Optional statistics about training, e.g.  iterations it took to
            train, training loss, etc.
        """
        pass

    @abstractmethod
    def representation(self) -> Tensor:
        r"""
        Returns the underlying representation of the interpretable model. For a
        linear model this is simply a tensor (the concatenation of weights
        and bias). For something slightly more complicated, such as a decision
        tree, this could be the nodes of a decision tree.

        Returns:
            A Tensor describing the representation of the model.
        """
        pass

    @abstractmethod
    def __call__(
        self, x: TensorOrTupleOfTensorsGeneric
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Predicts with the interpretable model.

        Args:
            x (TensorOrTupleOfTensorsGeneric)
                A batched input of tensor(s) to the model to predict
        Returns:
            The prediction of the input as a TensorOrTupleOfTensorsGeneric.
        """
        pass
