#!/usr/bin/env python3

# pyre-strict
import itertools
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import cast, List, Sized, Tuple

import torch
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric

from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _format_output,
    _validate_input,
)
from captum.log.dummy_log import log_usage


class Partition(Sized):
    def __init__(
        self,
        borders: None | List[Tuple[int, int]] = None,
        elements: None | torch.Tensor = None,
        size: int = -1,
    ):
        self.borders = borders
        self.elements = elements
        self.size = size

        self._mask = None

    def generate_mask(self, shape, device):
        # generates a mask for a partition (False indicates membership)
        if self._mask is not None:
            return self._mask
        self._mask = torch.zeros(shape, dtype=torch.bool, device=device)

        # non-contiguous case
        if self.elements is not None:
            self._mask[tuple(self.elements.T)] = True

        # contiguous case
        elif self.borders is not None:
            slices = list(slice(lo, hi) for (lo, hi) in self.borders)
            self._mask[slices] = True

        return self._mask

    def __len__(self):
        return self.size


@dataclass(eq=False)
class Mutant:
    partitions: List[Partition]
    data: torch.Tensor

    # eagerly create the underlying mutant data
    def __init__(self, partitions: List[Partition], data: torch.Tensor, neutral, shape):

        # A bitmap in the shape of the input indicating membership to
        # a partition in this mutant
        mask = torch.zeros(shape, dtype=torch.bool, device=data.device)
        for part in partitions:
            mask |= part.generate_mask(mask.shape, data.device)

        self.partitions = partitions
        self.data = torch.where(mask, data, neutral)

    def __len__(self):
        return len(self.partitions)


def _powerset(s):
    return (
        list(combo) for r in range(len(s) + 1) for combo in itertools.combinations(s, r)
    )


def _apply_responsibility(fi, part, responsibility):
    distributed = responsibility / len(part)
    mask = part.generate_mask(fi.shape, fi.device)

    return torch.where(mask, distributed, fi)


def _calculate_responsibility(
    subject_partition: Partition,
    mutants: List[Mutant],
    consistent_mutants: List[Mutant],
) -> float:
    recovery_set = {frozenset(m.partitions) for m in consistent_mutants}

    valid_witnesses = []
    for m in mutants:
        if subject_partition in m.partitions:
            continue
        W = m.partitions

        W_set = frozenset(W)
        W_plus_P_set = frozenset([subject_partition] + W)

        # W alone does NOT recover, but W ∪ {P} DOES recover.
        if (W_set not in recovery_set) and (W_plus_P_set in recovery_set):
            valid_witnesses.append(W)

    if not valid_witnesses:
        return 0.0

    k = min(len(w) for w in valid_witnesses)
    return 1.0 / (1.0 + float(k))


def _generate_indices(ts):
    # return a tensor containing all indices in the input shape
    return torch.tensor(
        tuple(itertools.product(*(range(s) for s in ts.shape))),
        dtype=torch.long,
        device=ts.device,
    )


class ReX(PerturbationAttribution):
    """
    A perturbation-based approach to computing attribution, derived from the
    Halpern-Pearl definition of Actual Causality[1].

    ReX conducts a recursive search on the input to find areas that are
    most responsible[3] for a models prediction. ReX splits an input into "partitions",
    and masks combinations of these partitions with baseline (neutral) values
    to form "mutants".

    Intuitively, where masking a partition never changes a models
    prediction, that partition is not responsible for the output. Conversely,
    where some combination of masked partitions changes the prediction, each
    partition has some  responsibility. Specifically, their responsibility is 1/(1+k)
    where k is the minimal number of *other* masked partitions required to create
    a dependence on a partition.

    Responsible partitions are recursively searched to refine responsibility estimates,
    and results are (optionally) merged to produce the final attribution map.


    [1] - Cause: https://www.cs.cornell.edu/home/halpern/papers/modified-HPdef.pdf
    [2] - ReX paper: https://arxiv.org/pdf/2411.08875
    [3] - Responsibility and Blame; https://arxiv.org/pdf/cs/0312038
    """

    def __init__(self, forward_func):
        r"""
        Args:
            forward_func (Callable): The function to be explained. *Must* return
                a scalar for which the equality operator is defined.
        """
        PerturbationAttribution.__init__(self, forward_func)

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        baselines: BaselineType = 0,
        search_depth: int = 10,
        n_partitions: int = 4,
        n_searches: int = 5,
        assume_locality: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:
            inputs (Tensor or tuple[Tensor, ...]): An input or tuple of inputs
                to be explained. Each input must be of the shape expected by
                the forward_func. Where multiple examples are provided, they
                must be listed in a tuple.

            baselines (Tensor or tuple[Tensor, ...]): A neutral values to be used
                as occlusion values. Where a scalar or tensor is provided, they
                are broadcast to the input shape. Where tuples are provided,
                they are paired element-wise, and must match the structure of
                the input.

            search_depth (int, optional): The maximum depth to which ReX will refine
                responsibility estimates for causes.

            n_partitions (optional): The maximum number of partitions to be made out of
                the input at each search step. At least 1, and no larger than the
                partition size. Where ``contiguous partitioning`` is set to False,
                partitions are created using previous attribution maps as heuristics.

            n_searches (int, optional): The number of times the search is to be ran.

            assume_locality (int, optional): Where True, partitioning is contiguous and
                attribution maps are merged after each search. Otherwise,
                partitioning is initially random, then uses the previous attribution
                map as a heuristic for further searches, returning the result of the
                final search.
        """

        inputs, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inputs, baselines)

        self._n_partitions: int = n_partitions
        self._max_depth: int = search_depth
        self._n_searches: int = n_searches
        self._assume_locality: bool = assume_locality

        is_input_tuple = isinstance(inputs, tuple)
        is_baseline_tuple = isinstance(baselines, tuple)

        attributions = []

        # broadcast baselines, explain
        if is_input_tuple and is_baseline_tuple:
            for input, baseline in zip(inputs, baselines):
                attributions.append(self._explain(input, baseline))
        elif is_input_tuple and not is_baseline_tuple:
            for input in inputs:
                attributions.append(self._explain(input, baselines))
        else:
            attributions.append(self._explain(inputs, baselines))

        return cast(
            TensorOrTupleOfTensorsGeneric,
            _format_output(is_input_tuple, tuple(attributions)),
        )

    @torch.no_grad()
    def _explain(self, input, baseline) -> torch.Tensor:
        self._device = input.device
        self._shape = input.shape
        self._size = input.numel()

        prediction = self.forward_func(input)

        prev_attribution = torch.full_like(
            input, 0.0, dtype=torch.float32, device=self._device
        )
        attribution = torch.full_like(
            input, 1.0 / self._size, dtype=torch.float32, device=self._device
        )

        initial_partition = Partition(
            borders=list((0, top) for top in self._shape),
            elements=_generate_indices(input),
            size=self._size,
        )

        for i in range(1, self._n_searches + 1):
            Q: deque = deque()
            Q.append((initial_partition, 0))

            while Q:
                prev_part, depth = Q.popleft()
                partitions = (
                    self._contiguous_partition(prev_part, depth)
                    if self._assume_locality
                    else self._partition(prev_part, attribution)
                )

                mutants = [
                    Mutant(part, input, baseline, self._shape)
                    for part in _powerset(partitions)
                ]
                consistent_mutants = [
                    mut for mut in mutants if self.forward_func(mut.data) == prediction
                ]

                for part in partitions:
                    resp = _calculate_responsibility(part, mutants, consistent_mutants)
                    attribution = _apply_responsibility(attribution, part, resp)

                    if resp == 1 and len(part) > 1 and self._max_depth > depth:
                        Q.append((part, depth + 1))

            # take average of responsibility maps
            if self._assume_locality:
                prev_attribution += (1 / i) * (attribution - prev_attribution)
                attribution = prev_attribution

        return attribution.clone().detach()

    def _partition(
        self, part: Partition, responsibility: torch.Tensor
    ) -> List[Partition]:
        # shuffle candidate indices (randomize tiebreakers)
        perm = torch.randperm(len(part), device=self._device)

        assert (
            part is not None
        ), "Partitioning strategy changed mid search. Contact developer"
        population = part.elements[perm]  # type: ignore

        weights = responsibility[tuple(population.T)]

        if torch.sum(weights, dim=None) == 0:
            weights = torch.ones_like(weights, device=self._device) / len(weights)
        target_weight = torch.sum(weights) / self._n_partitions

        # sort for greedy selection
        idx = torch.argsort(weights, descending=False)
        weight_sorted, pop_sorted = weights[idx], population[idx]

        # cumulative sum of weights / weight per bucket rounded down gives us bucket ids
        eps = torch.finfo(weight_sorted.dtype).eps
        c = weight_sorted.cumsum(0) + eps
        bin_id = torch.div(c, target_weight, rounding_mode="floor").clamp_min(0).long()

        # count elements in each bucket, and split input accordingly
        _, counts = torch.unique_consecutive(bin_id, return_counts=True)
        groups = torch.split(pop_sorted, counts.tolist())

        partitions = [Partition(elements=g, size=len(g)) for g in groups]
        return partitions

    def _contiguous_partition(self, part, depth):
        ndim = len(self._shape)
        split_dim = -1

        # find a dimension we can split
        dmin, dmax = max(self._shape), 0
        for i in range(ndim):
            candidate_dim = (i + depth) % ndim
            dmin, dmax = tuple(part.borders[candidate_dim])

            if dmax - dmin > 1:
                split_dim = candidate_dim
                break

        if split_dim == -1:
            return [part]
        n_splits = min((dmax - dmin), self._n_partitions) - 1

        # drop splits randomly
        split_points = random.sample(range(dmin + 1, dmax), n_splits)
        split_borders = sorted(set([dmin, *split_points, dmax]))

        bins = []
        for i in range(len(split_borders) - 1):
            new_borders = list(part.borders)
            new_borders[split_dim] = (split_borders[i], split_borders[i + 1])

            bins.append(
                Partition(
                    borders=tuple(new_borders),
                    size=math.prod(hi - lo for (lo, hi) in new_borders),
                )
            )

        return bins

    def multiplies_by_inputs(self):
        return False

    def has_convergence_delta(self):
        return True
