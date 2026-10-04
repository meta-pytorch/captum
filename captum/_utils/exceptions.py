# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


class FeatureAblationFutureError(Exception):
    """This custom error is raised when an error
    occurs within the callback chain of a
    FeatureAblation attribution call"""

    pass


class ShapleyValueFutureError(Exception):
    """This custom error is raised when an error
    occurs within the callback chain of a
    ShapleyValue attribution call"""

    pass
