#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import glob
import os
from typing import Callable, Iterator

from torch import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset


class CustomIterableDataset(IterableDataset):
    r"""
    An auxiliary class for iterating through a dataset.
    """

    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def __init__(self, transform_filename_to_tensor: Callable, path: str) -> None:
        r"""
        Args:
            transform_filename_to_tensor (Callable): Function to read a data
                        file from path and return a tensor from that file.
            path (str): Path to dataset files. This can be either a path to a
                        directory or a file where input examples are stored.
        """
        # pyre-fixme[4]: Attribute must be annotated.
        self.file_itr = None
        self.path = path

        if os.path.isdir(self.path):
            self.file_itr = glob.glob(self.path + "*")

        self.transform_filename_to_tensor = transform_filename_to_tensor

    def __iter__(self) -> Iterator[Tensor]:
        r"""
        Returns:
            iter (Iterator[Tensor]): A map from a function that
                processes a list of file path(s) to a list of Tensors.
        """
        if self.file_itr is not None:
            return map(self.transform_filename_to_tensor, self.file_itr)
        else:
            return self.transform_filename_to_tensor(self.path)


def dataset_to_dataloader(dataset: Dataset, batch_size: int = 64) -> DataLoader:
    r"""
    An auxiliary function that creates torch DataLoader from torch Dataset
    using input `batch_size`.

    Args:
        dataset (Dataset): A torch dataset that allows to iterate over
            the batches of examples.
        batch_size (int, optional): Batch size of for each tensor in the
            iteration.

    Returns:
        dataloader_iter (DataLoader): a DataLoader for data iteration.
    """

    return DataLoader(dataset, batch_size=batch_size)
