#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from abc import ABC, abstractmethod
from typing import Any, Type

from torch.nn import Module
from torch.utils.data import Dataset


class DataInfluence(ABC):
    r"""
    An abstract class to define model data influence skeleton.
    """

    def __init_(self, model: Module, train_dataset: Dataset, **kwargs: Any) -> None:
        r"""
        Args:
            model (torch.nn.Module): An instance of pytorch model.
            train_dataset (torch.utils.data.Dataset): PyTorch Dataset that is
                    used to create a PyTorch Dataloader to iterate over the dataset and
                    its labels. This is the dataset for which we will be seeking for
                    influential instances. In most cases this is the training dataset.
            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.
        """
        # pyre-fixme[16]: `DataInfluence` has no attribute `model`.
        self.model = model
        # pyre-fixme[16]: `DataInfluence` has no attribute `train_dataset`.
        self.train_dataset = train_dataset

    @abstractmethod
    def influence(self, inputs: Any = None, **kwargs: Any) -> Any:
        r"""
        Args:
            inputs (Any): Batch of examples for which influential
                    instances are computed. They are passed to the forward_func. If
                    `inputs` if a tensor or tuple of tensors, the first dimension
                    of a tensor corresponds to the batch dimension.
            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.

        Returns:
            influences (Any): We do not add restrictions on the return type for now,
                    though this may change in the future.
        """
        pass

    @classmethod
    def get_name(cls: Type["DataInfluence"]) -> str:
        r"""
        Create readable class name.  Due to the nature of the names of `TracInCPBase`
        subclasses, simply returns the class name.  For example, for a class called
        TracInCP, we return the string TracInCP.

        Returns:
            name (str): a readable class name
        """
        return cls.__name__
