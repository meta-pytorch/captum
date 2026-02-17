#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from captum.attr._utils.stat import (
    Count,
    Max,
    Mean,
    Min,
    MSE,
    Stat,
    StatValue,
    StdDev,
    Sum,
    Var,
)
from captum.log import log_usage
from torch import Tensor


class Summarizer:
    r"""
    This class simply wraps over a given a set of SummarizerSingleTensor's in order
    to summarise multiple input tensors.

    Basic usage:

    >>>from captum.attr.aggregator import Summarizer
    >>>from captum.attr._utils.stats import Mean, StdDev
    >>>
    >>>attrib = torch.tensor([1, 2, 3, 4, 5])
    >>>
    >>>summ = Summarizer([Mean(), StdDev(0])
    >>>summ.update(attrib)
    >>>
    >>>print(summ.summary['mean'])
    """

    _stats: List[Stat]
    _summary_stats_indicies: List[int]

    @log_usage(part_of_slo=False)
    def __init__(self, stats: List[Stat]) -> None:
        r"""
        Args:
            stats (List[Stat]):
                The list of statistics you wish to track
        """
        self._summarizers: List[SummarizerSingleTensor] = []
        self._is_inputs_tuple: Optional[bool] = None
        self._stats, self._summary_stats_indicies = _reorder_stats(stats)

    def _copy_stats(self) -> List[Stat]:
        import copy

        return copy.deepcopy(self._stats)

    def update(self, x: Union[float, Tensor, Tuple[Union[float, Tensor], ...]]) -> None:
        r"""
        Calls `update` on each `Stat` object within the summarizer

        Args:
            x (Tensor or Tuple[Tensor, ...]):
                The input(s) you wish to summarize
        """
        if self._is_inputs_tuple is None:
            self._is_inputs_tuple = isinstance(x, tuple)
        else:
            # we want input to be consistently a single input or a tuple
            assert not (self._is_inputs_tuple ^ isinstance(x, tuple))

        from captum._utils.common import _format_float_or_tensor_into_tuples

        x = _format_float_or_tensor_into_tuples(x)

        for i, inp in enumerate(x):
            if i >= len(self._summarizers):
                # _summarizers[i] is a new SummarizerSingleTensor, which
                # aims to summarize input i (i.e. x[i])
                #
                # Thus, we must copy our stats, as otherwise
                # in the best case the statistics for each input will be mangled
                # and in the worst case we will run into an error due to different
                # dimensionality in the input tensors tensors (i.e.
                # x[i].shape != x[j].shape for some pair i, j)
                stats = self._copy_stats()
                self._summarizers.append(
                    SummarizerSingleTensor(
                        stats=stats, summary_stats_indices=self._summary_stats_indicies
                    )
                )
            if not isinstance(inp, torch.Tensor):
                inp = torch.tensor(inp, dtype=torch.float)
            self._summarizers[i].update(inp)

    @property
    def summary(
        self,
    ) -> Optional[
        Union[Dict[str, Optional[StatValue]], List[Dict[str, Optional[StatValue]]]]
    ]:
        r"""
        Effectively calls `get` on each `Stat` object within this object for each input

        Returns:
            A dict or list of dict: mapping from the Stat
            object's `name` to the associated value of `get`
        """
        if len(self._summarizers) == 0:
            return None

        temp = [summ.summary for summ in self._summarizers]
        return temp if self._is_inputs_tuple else temp[0]


def _reorder_stats(stats: List[Stat]) -> Tuple[List[Stat], List[int]]:
    # We want to want to store two things:
    # 1. A mapping from a Stat to Stat object (self._stat_to_stat):
    #    This is to retrieve an existing Stat object for dependency
    #    resolution, e.g.  Mean needs the Count stat - we want to
    #    retrieve it in O(1)
    #
    # 2. All of the necessary stats, in the correct order,
    #    to perform an update for each Stat (self.stats) trivially

    # As a reference, the dependency graph for our stats is as follows:
    # StdDev(x) -> Var(x) -> MSE -> Mean -> Count, for all valid x
    #
    # Step 1:
    #    Ensure we have all the necessary stats
    #    i.e. ensure we have the dependencies
    # Step 2:
    #    Figure out the order to update them
    dep_order = [StdDev, Var, MSE, Mean, Count]

    # remove dupe stats
    stats_set = set(stats)
    summary_stats = set(stats)

    from collections import defaultdict

    stats_by_module: Dict[Type[Stat], List[Stat]] = defaultdict(list)
    for stat in stats_set:
        stats_by_module[stat.__class__].append(stat)

    # StdDev is an odd case since it is parameterized, thus
    # for each StdDev(order) we must ensure there is an associated Var(order)
    for std_dev in stats_by_module[StdDev]:
        stat_to_add = Var(order=std_dev.order)  # type: ignore
        stats_set.add(stat_to_add)
        stats_by_module[stat_to_add.__class__].append(stat_to_add)

    # For the other modules (deps[1:n-1]): if i exists =>
    # we want to ensure i...n-1 exists
    for i, dep in enumerate(dep_order[1:]):
        if dep in stats_by_module:
            stats_set.update([mod() for mod in dep_order[i + 1 :]])
            break

    # Step 2: get the correct order
    # NOTE: we are sorting via a given topological order
    sort_order: Dict[Type[Stat], int] = {mod: i for i, mod in enumerate(dep_order)}
    sort_order[Min] = -1
    sort_order[Max] = -1
    sort_order[Sum] = -1

    stats = list(stats_set)
    stats.sort(key=lambda x: sort_order[x.__class__], reverse=True)

    # get the summary stat indices
    summary_stat_indexs = []
    for i, stat in enumerate(stats):
        if stat in summary_stats:
            summary_stat_indexs.append(i)
    return stats, summary_stat_indexs


class SummarizerSingleTensor:
    r"""
    A simple class that summarizes a single tensor. The basic functionality
    of this class is two operations .update and .summary

    If possible use `Summarizer` instead.
    """

    _stats: List[Stat]
    _stat_to_stat: Dict[Stat, Stat]
    _summary_stats: List[Stat]

    def __init__(self, stats: List[Stat], summary_stats_indices: List[int]) -> None:
        r"""
        Args:
            stats (list[Stat]): A list of all the Stat objects that
                need to be updated. This must be in the appropriate order for
                updates (see `_reorder_stats`)
            summary_stats (list[int]): A list of indicies, referencing `stats`,
                which are the stats you want to show in the .summary property. This
                does not require any specific order.
        """
        self._stats = stats
        self._stat_to_stat = {stat: stat for stat in self._stats}
        self._summary_stats = [stats[i] for i in summary_stats_indices]

        for stat in stats:
            stat._other_stats = self
            stat.init()

    def update(self, x: Tensor) -> None:
        r"""
        Updates the summary of a given tensor `x`

        Args:
            x (Tensor):
                The tensor to summarize
        """
        for stat in self._stats:
            stat.update(x)

    def get(self, stat: Stat) -> Optional[Stat]:
        r"""
        Retrieves `stat` from cache if this summarizer contains it.

        Note that `Stat` has it's hash/equality method overridden, such
        that an object with the same class and parameters will have the
        same hash. Thus, if you call `get` with a `Stat`, an associated
        `Stat` with the same class and parameters belonging to this object
        will be retrieved if it exists.

        If no such object is retrieved then `None` is returned.

        Args:
            stat (Stat):
                The stat to retrieve
        Returns:
            Stat
                The cached stat object or `None`
        """
        if stat not in self._stat_to_stat:
            return None

        return self._stat_to_stat[stat]

    @property
    def summary(self) -> Dict[str, Optional[StatValue]]:
        """
        Returns:
            Optional[Dict[str, Optional[Tensor]]]
                The cached stat object
        """
        return {stat.name: stat.get() for stat in self._summary_stats}
