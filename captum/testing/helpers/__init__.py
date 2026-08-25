#!/usr/bin/env python3


try:
    from captum.testing.helpers.fb.internal_base import (  # type: ignore
        FbBaseTest as BaseTest,
    )

except ImportError:
    # tests/helpers/__init__.py:13: error: Incompatible import of "_BaseTest"
    # (imported name has type "type[BaseTest]", local name has type
    # "type[FbBaseTest]")  [assignment]
    from captum.testing.helpers.basic import BaseTest  # type: ignore

__all__ = [
    "BaseTest",
]
