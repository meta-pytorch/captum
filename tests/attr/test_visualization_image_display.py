#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
from captum.attr._utils.visualization import visualize_image_attr
from matplotlib import pyplot as plt


def test_visualize_image_attr_normalizes_out_of_range_float_images() -> None:
    attr = np.ones((2, 2, 3))
    original_image = np.array(
        [
            [[-1.0, 0.0, 1.0], [2.0, -0.5, 0.5]],
            [[1.5, -1.5, 0.25], [0.75, 1.25, -0.25]],
        ]
    )

    fig, ax = visualize_image_attr(
        attr,
        original_image,
        method="original_image",
        sign="positive",
        use_pyplot=False,
    )
    rendered_image = np.asarray(ax.images[0].get_array())

    assert rendered_image.min() == 0
    assert rendered_image.max() == 255
    plt.close(fig)
