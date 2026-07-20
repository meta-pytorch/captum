#!/usr/bin/env python3import

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from typing import cast, Iterable

import torch
from captum.concept._core.concept import Concept
from captum.concept._utils.data_iterator import dataset_to_dataloader
from captum.testing.helpers import BaseTest
from torch.utils.data import IterableDataset


class CustomIterableDataset(IterableDataset):
    r"""
    An auxiliary class for iterating through an image dataset.
    """

    def __init__(self, get_tensor_from_filename_func, path) -> None:
        r"""
        Args:

            path (str): Path to dataset files
        """

        self.path = path
        self.file_itr = ["x"] * 2
        self.get_tensor_from_filename_func = get_tensor_from_filename_func

    def get_tensor_from_filename(self, filename):

        return self.get_tensor_from_filename_func(filename)

    def __iter__(self):

        mapped_itr = map(self.get_tensor_from_filename, self.file_itr)

        return mapped_itr


# pyrefly: ignore [invalid-inheritance]
class Test(BaseTest):
    def test_create_concepts_from_images(self) -> None:
        def get_tensor_from_filename(filename):
            return torch.rand(3, 224, 224)

        # Striped
        concepts_path = "./dummy/concepts/striped/"
        dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
        striped_iter = dataset_to_dataloader(dataset)

        self.assertEqual(
            len(cast(CustomIterableDataset, striped_iter.dataset).file_itr), 2
        )

        concept = Concept(id=0, name="striped", data_iter=striped_iter)

        for data in cast(Iterable, concept.data_iter):
            self.assertEqual(data.shape[1:], torch.Size([3, 224, 224]))

        # Random
        concepts_path = "./dummy/concepts/random/"
        dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
        random_iter = dataset_to_dataloader(dataset)
        self.assertEqual(
            len(cast(CustomIterableDataset, random_iter.dataset).file_itr), 2
        )

        concept = Concept(id=1, name="random", data_iter=random_iter)
        for data in cast(Iterable, concept.data_iter):
            self.assertEqual(data.shape[1:], torch.Size([3, 224, 224]))
