#!/usr/bin/env python3

# pyre-strict
from collections import defaultdict
from typing import Any, cast, Dict, Generic, List, Optional, TypeVar, Union

from captum._utils.common import _format_tensor_into_tuples
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr._utils.stat import Stat, StatValue
from captum.attr._utils.summarizer import Summarizer
from captum.log import log_usage
from torch import Tensor

KeyType = TypeVar("KeyType")


class ClassSummarizer(Summarizer, Generic[KeyType]):
    r"""
    Used to keep track of summaries for associated classes. The
    classes/labels can be of any type that are supported by `dict`.

    This also keeps track of an aggregate of all class summaries.
    """

    @log_usage(part_of_slo=False)
    def __init__(self, stats: List[Stat]) -> None:
        Summarizer.__init__.__wrapped__(self, stats)
        self.summaries: Dict[KeyType, Summarizer] = defaultdict(
            lambda: Summarizer(stats=stats)
        )

    def update(  # type: ignore
        self,
        x: TensorOrTupleOfTensorsGeneric,
        labels: TargetType = None,
    ) -> None:
        r"""
        Updates the stats of the summarizer, optionally associated to classes.

        This accepts either a single tensor to summarise or a tuple of tensors.

        Args:
            x (Tensor or tuple[Tensor, ...]):
                The input tensor to be summarised. The first
                dimension of this input must be associated to
                the batch size of the inputs.
            labels (int, tuple, Tensor, or list, optional):
                The associated labels for `x`. If Any, we
                assume `labels` represents the label for all inputs in `x`.

                If this is None we simply aggregate the total summary.
        """
        if labels is None:
            super().update(x)
            return

        # pyre-fixme[9]: x has type `TensorOrTupleOfTensorsGeneric`; used as
        #  `Tuple[Tensor, ...]`.
        x = _format_tensor_into_tuples(x)

        num_labels = 1

        labels_typed: Union[List[Any], Tensor]
        if isinstance(labels, list) or isinstance(labels, Tensor):
            labels_typed = labels
            num_labels = len(labels)  # = labels.size(0) if tensor
        else:
            labels_typed = [labels]

        # mypy doesn't realise I have made the int a list
        if len(labels_typed) > 1:
            for x_i in x:
                assert x_i.size(0) == num_labels, (
                    "batch size does not equal amount of labels; "
                    "please ensure length of labels is equal to 1 "
                    "or to the `batch_size` corresponding to the "
                    "number of examples in the input(s)"
                )

        batch_size = x[0].size(0)

        for i in range(batch_size):
            tensors_to_summarize = tuple(tensor[i] for tensor in x)
            tensors_to_summarize_copy = tuple(tensor[i].clone() for tensor in x)
            label = labels_typed[0] if len(labels_typed) == 1 else labels_typed[i]

            self.summaries[cast(KeyType, label)].update(tensors_to_summarize)
            super().update(tensors_to_summarize_copy)

    @property
    def class_summaries(
        self,
    ) -> Dict[
        KeyType,
        Union[
            None, Dict[str, Optional[StatValue]], List[Dict[str, Optional[StatValue]]]
        ],
    ]:
        r"""
        Returns:
             The summaries for each class.
        """
        return {key: value.summary for key, value in self.summaries.items()}
