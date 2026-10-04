# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor


class NearestNeighbors(ABC):
    r"""
    An abstract class to define a nearest neighbors data structure. Classes
    implementing this interface are intended for computing proponents / opponents in
    certain implementations of `TracInCPBase`. In particular, it is for use in
    implementations which compute proponents / opponents of a test instance by
    1) storing representations of training instances within a nearest neighbors data
    structure, and 2) finding within that structure the nearest neighbor of the
    representation of a test instance. The assumption is that the data structure
    stores the tensors passed to the `setup` method, which we refer to as the "stored
    tensors". If this class is used to find proponents / opponents, the nearest
    neighbors of a tensor should be the stored tensors that have the largest
    dot-product with the query.
    """

    @abstractmethod
    def get_nearest_neighbors(
        self, query: torch.Tensor, k: int
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Given a `query`, a tensor of shape (N, *), returns the nearest neighbors in the
        "stored tensors" (see above). `query` represents a batch of N tensors, each
        of common but arbitrary shape *. We always assume the 0-th dimension indexes
        the batch. In use cases of this class for computing proponents / opponents,
        the nearest neighbors of a tensor should be the stored tensors with the largest
        dot-product with the tensor, and the tensors in `query` will all be 1D,
        so that `query` is 2D.

        Args:
            query (Tensor): tensor representing the batch of tensors for which k-nearest
                    neighbors are desired. `query` is of shape (N, *), where N is the
                    size of the batch, i.e. the 0-th dimension of `query` indexes the
                    batch. * denotes an arbitrary shape, so that each tensor in the
                    batch can be of a common, but arbitrary shape.
            k (int): The number of nearest neighbors to return.

        Returns:
            results (tuple): A tuple of `(indices, distances)` is returned. `indices`
                    is a 2D tensor where `indices[i,j]` is the index (within the
                    "stored tensors" passed to the `setup` method) of the `j`-th
                    nearest neighbor of the `i`-th instance in query, and
                    `distances[i,j]` is the corresponding distance. `indices` should
                    be of dtype `torch.long` so that it can be used to index torch
                    tensors.
        """
        pass

    @abstractmethod
    def setup(self, data: torch.Tensor) -> None:
        r"""
        `data` denotes the "stored tensors". These are the tensors within which we
        want to find the nearest neighbors to each tensor in a batch of tensors, via a
        call to the`get_nearest_neighbors` method. Before we can call it, however,
        we need to first store the stored tensors, by doing processing that indexes
        the stored tensors in a form that enables nearest-neighbors computation.
        This method does that preprocessing, and is assumed to be called before any
        call to `get_nearest_neighbors`. For example, this method might put the
        stored tensors in a K-d tree. The tensors in the "stored tensors" can be of a
        common, but arbitrary shape, denoted *, so that `data` is of shape (N, *),
        where N is the number of tensors in the stored tensors. Therefore, the 0-th
        dimension indexes the tensors in the stored tensors.

        Args:
            data (Tensor): A tensor of shape (N, *) representing the stored tensors.
                    The 0-th dimension indexes the tensors in the stored tensors,
                    so that `data[i]` is the tensor with index `i`. The nearest
                    neighbors of a query will be referred to by their index.
        """
        pass


class AnnoyNearestNeighbors(NearestNeighbors):
    """
    This is an implementation of `NearestNeighbors` that uses the Annoy module. At a
    high level, Annoy finds nearest neighbors by constructing binary trees in which
    vectors reside at leaf nodes. Vectors near each other will tend to be in the same
    leaf node. See https://tinyurl.com/2p89sb2h and https://github.com/spotify/annoy
    for more details. Annoy has 1 key parameter: the number of trees to construct.
    Increasing the number of trees leads to more accurate results, but longer time to
    create the trees and memory usage. As mentioned in the `NearestNeighbors`
    documentation, for the use case of computing proponents / opponents, the nearest
    neighbors returned should be those with the largest dot product with the query
    vector. The term "vector" is used here because Annoy stores 1D vectors. However
    in our wrapper around Annoy, we will allow the stored tensors to be of a common
    but arbitrary shape *, and flatten them before storing in the Annoy data structure.
    """

    def __init__(self, num_trees: int = 10) -> None:
        """
        Args:
            num_trees (int): The number of trees to use. Increasing this number gives
                    more accurate computation of nearest neighbors, but requires longer
                    setup time to create the trees, as well as memory.
        """
        try:
            import annoy  # noqa
        except ImportError:
            raise ValueError(
                (
                    "Using `AnnoyNearestNeighbors` requires installing the annoy "
                    "module. If pip is installed, this can be done with "
                    "`pip install --user annoy`."
                )
            )

        self.num_trees = num_trees

    def setup(self, data: torch.Tensor) -> None:
        """
        `data` denotes the "stored tensors". These are the tensors within which we
        want to find the nearest neighbors to a query tensor, via a call to the
        `get_nearest_neighbors` method. Before we can call `get_nearest_neighbors`,
        we need to first store the stored tensors, by doing processing that indexes
        the stored tensors in a form that enables nearest-neighbors computation.
        This method does that preprocessing, and is assumed to be called before any
        call to `get_nearest_neighbors`. In particular, it creates the trees used to
        index the stored tensors. This index is built to enable computation of
        vectors that have the largest dot-product with the query tensors. The tensors
        in the "stored tensors" can be of a common, but arbitrary shape, denoted *, so
        that `data` is of shape (N, *), where N is the number of tensors in the stored
        tensors. Therefore, the 0-th dimension indexes the tensors in the stored
        tensors.

        Args:
            data (Tensor): A tensor of shape (N, *) representing the stored tensors.
                    The 0-th dimension indexes the tensors in the stored tensors,
                    so that `data[i]` is the tensor with index `i`. The nearest
                    neighbors of a query will be referred to by their index.
        """
        import annoy

        data = data.view((len(data), -1))
        projection_dim = data.shape[1]
        # pyre-fixme[16]: `AnnoyNearestNeighbors` has no attribute `knn_index`.
        # pyre-fixme[16]: Module `annoy` has no attribute `AnnoyIndex`.
        self.knn_index = annoy.AnnoyIndex(projection_dim, "dot")
        for i, projection in enumerate(data):
            self.knn_index.add_item(i, projection)
        self.knn_index.build(self.num_trees)

    def get_nearest_neighbors(
        self, query: torch.Tensor, k: int
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Given a `query`, a tensor of shape (N, *), returns the nearest neighbors in the
        "stored tensors" (see above). `query` represents a batch of N tensors, each
        of common but arbitrary shape *. We always assume the 0-th dimension indexes
        the batch. In use cases of this class for computing proponents / opponents,
        the nearest neighbors of a tensor should be the stored tensors with the largest
        dot-product with the tensor, and the tensors in `query` will all be 1D,
        so that `query` is 2D. This implementation returns the stored tensors
        that have the largest dot-product with the query tensor, and does not constrain
        the tensors in `query` or in the stored tensors to be 1D. If tensors are of
        dimension greater than 1D, their dot-product will be defined to be the
        dot-product of the flattened version of tensors.

        Args:
            query (Tensor): tensor representing the batch of tensors for which k-nearest
                    neighbors are desired. `query` is of shape (N, *), where N is the
                    size of the batch, i.e. the 0-th dimension of `query` indexes the
                    batch. * denotes an arbitrary shape, so that each tensor in the
                    batch can be of a common, but arbitrary shape.
            k (int): The number of nearest neighbors to return.

        Returns:
            results (tuple): A tuple of `(indices, distances)` is returned. `indices`
                    is a 2D tensor where `indices[i,j]` is the index (within the
                    "stored tensors" passed to the `setup` method) of the `j`-th
                    nearest neighbor of the `i`-th instance in query, and
                    `distances[i,j]` is the corresponding distance. `indices` should
                    be of dtype `torch.long` so that it can be used to index torch
                    tensors.
        """
        query = query.view((len(query), -1))
        indices_and_distances = [
            # pyre-fixme[16]: `AnnoyNearestNeighbors` has no attribute `knn_index`.
            self.knn_index.get_nns_by_vector(instance, k, include_distances=True)
            for instance in query
        ]
        indices, distances = zip(*indices_and_distances)
        indices = torch.Tensor(indices).type(torch.long)
        distances = torch.Tensor(distances)
        return indices, distances
