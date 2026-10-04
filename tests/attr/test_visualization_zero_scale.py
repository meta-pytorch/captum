#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import numpy.typing as npt
import pytest
from captum.attr._utils.visualization import (
    visualize_image_attr,
    visualize_image_attr_multiple,
)
from matplotlib import pyplot as plt

ZERO_ATTRIBUTION_WARNING = "No non-zero attribution values found"


@pytest.mark.parametrize(
    "attr, sign",
    [
        (np.zeros((4, 4, 3)), "positive"),
        (-np.ones((4, 4, 3)), "positive"),
        (np.ones((4, 4, 3)), "negative"),
    ],
)
def test_visualize_image_attr_handles_zero_signed_attributions(
    attr: npt.NDArray,
    sign: str,
) -> None:
    original_image = np.zeros((4, 4, 3))

    with pytest.warns(UserWarning, match=ZERO_ATTRIBUTION_WARNING):
        fig, _ = visualize_image_attr(
            attr,
            original_image,
            method="heat_map",
            sign=sign,
            show_colorbar=True,
            use_pyplot=False,
        )

    plt.close(fig)


def test_visualize_image_attr_multiple_handles_zero_signed_attributions() -> None:
    attr = -np.ones((4, 4, 3))
    original_image = np.zeros((4, 4, 3))

    with pytest.warns(UserWarning, match=ZERO_ATTRIBUTION_WARNING):
        fig, _ = visualize_image_attr_multiple(
            attr,
            original_image,
            methods=["original_image", "masked_image", "heat_map"],
            signs=["all", "positive", "positive"],
            show_colorbar=True,
            use_pyplot=False,
        )

    plt.close(fig)
