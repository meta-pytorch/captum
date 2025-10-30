#!/usr/bin/env python3

# pyre-strict

from collections import UserDict
from typing import List, Literal, Optional, overload, Protocol, Tuple, TypeVar, Union

from torch import Tensor
from torch.nn import Module

TensorOrTupleOfTensorsGeneric = TypeVar(
    "TensorOrTupleOfTensorsGeneric", Tensor, Tuple[Tensor, ...]
)
TupleOrTensorOrBoolGeneric = TypeVar(
    "TupleOrTensorOrBoolGeneric", Tuple[Tensor, ...], Tensor, bool
)
PassThroughOutputType = TypeVar("PassThroughOutputType")
ModuleOrModuleList = TypeVar("ModuleOrModuleList", Module, List[Module])
TargetType = Union[None, int, Tuple[int, ...], Tensor, List[Tuple[int, ...]], List[int]]
BaselineTupleType = Union[None, Tuple[Union[Tensor, int, float], ...]]
BaselineType = Union[None, Tensor, int, float, BaselineTupleType]

TensorLikeList1D = List[float]
TensorLikeList2D = List[TensorLikeList1D]
TensorLikeList3D = List[TensorLikeList2D]
TensorLikeList4D = List[TensorLikeList3D]
TensorLikeList5D = List[TensorLikeList4D]
TensorLikeList = Union[
    TensorLikeList1D,
    TensorLikeList2D,
    TensorLikeList3D,
    TensorLikeList4D,
    TensorLikeList5D,
]

try:
    # Subscripted slice syntax is not supported in previous Python versions,
    # falling back to slice type.
    SliceIntType = slice[int, int, int]
except TypeError:
    # pyre-ignore[24]: Generic type `slice` expects 3 type parameters.
    SliceIntType = slice  # type: ignore

BatchEncodingType = UserDict[Union[int, str], object]


class TokenizerLike(Protocol):
    """A protocol for tokenizer-like objects that can be used with Captum
    LLM attribution methods."""

    @overload
    def encode(
        self, text: str, add_special_tokens: bool = ..., return_tensors: None = ...
    ) -> List[int]: ...

    @overload
    def encode(
        self,
        text: str,
        add_special_tokens: bool = ...,
        return_tensors: Literal["pt"] = ...,
    ) -> Tensor: ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Tensor]: ...

    def decode(self, token_ids: Tensor) -> str: ...

    @overload
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]: ...
    @overload
    def convert_ids_to_tokens(self, token_ids: int) -> str: ...

    def convert_ids_to_tokens(
        self, token_ids: Union[List[int], int]
    ) -> Union[List[str], str]: ...

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]: ...

    def convert_tokens_to_ids(
        self, tokens: Union[List[str], str]
    ) -> Union[List[int], int]: ...

    def __call__(
        self,
        text: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> BatchEncodingType: ...
