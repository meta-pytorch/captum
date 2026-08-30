import sys
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from torch import distributions, Tensor
from torch.nn import Module
from torch.optim import Optimizer

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol


ParametersForOptimizers = Iterable[Union[Tensor, Dict[str, Tensor]]]


class HasLoss(Protocol):
    def loss(self) -> Tensor: ...


class Parameterized(Protocol):
    parameters: ParametersForOptimizers


class Objective(Parameterized, HasLoss):
    def cleanup(self) -> None:
        pass


ModuleOutputMapping = Dict[Module, Optional[Tensor]]
StopCriteria = Callable[[int, Objective, Iterable[Tensor], Optimizer], bool]
LossFunction = Callable[[ModuleOutputMapping], Tensor]
SingleTargetLossFunction = Callable[[Tensor], Tensor]

NumSeqOrTensorOrProbDistType = Union[
    Sequence[int],
    Sequence[float],
    Tensor,
    distributions.distribution.Distribution,
]
IntSeqOrIntType = Union[List[int], Tuple[int], Tuple[int, int], int]
TupleOfTensorsOrTensorType = Union[Tuple[Tensor, ...], Tensor]
