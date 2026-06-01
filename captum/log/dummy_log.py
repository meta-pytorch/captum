#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from functools import wraps
from inspect import iscoroutinefunction
from types import TracebackType
from typing import Any, List, Optional, Union


def log(*args: Any, **kwargs: Any) -> None:
    pass


class TimedLog:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "TimedLog":
        return self

    def __exit__(
        self,
        exception_type: Optional[BaseException],
        exception_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> bool:
        return exception_value is not None


# pyre-fixme[3]: Return type must be annotated.
def log_usage(*log_args: Any, **log_kwargs: Any):

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def _log_usage(func):
        if iscoroutinefunction(func):

            @wraps(func)
            # pyre-fixme[53]: Captured variable `func` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            async def async_wrapper(*args, **kwargs):
                return await func(*args, **kwargs)

            return async_wrapper
        else:

            @wraps(func)
            # pyre-fixme[53]: Captured variable `func` is not annotated.
            # pyre-fixme[3]: Return type must be annotated.
            # pyre-fixme[2]: Parameter must be annotated.
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

    return _log_usage


def set_environment(env: Union[None, List[str], str]) -> None:
    pass


def disable_detailed_logging() -> None:
    pass


def patch_methods(_, patch_log: bool = True) -> None:
    pass
