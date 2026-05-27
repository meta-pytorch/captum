#!/usr/bin/env python3

# pyre-strict

import numpy as np
import numpy.typing as npt
from captum.attr import visualization
from captum.testing.helpers import BaseTest
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class Test(BaseTest):
    def _attr(self) -> npt.NDArray:
        return np.array([[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]])

    def _image(self) -> npt.NDArray:
        return np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

    def test_visualize_image_attr_default_return_value(self) -> None:
        fig, ax = visualization.visualize_image_attr(
            self._attr(), method="heat_map", use_pyplot=False
        )

        self.assertIsInstance(fig, Figure)
        self.assertIsInstance(ax, Axes)

    def test_visualize_image_attr_return_numpy_heatmap(self) -> None:
        image_array = visualization.visualize_image_attr(
            self._attr(),
            method="heat_map",
            sign="all",
            use_pyplot=False,
            return_numpy=True,
        )

        self.assertEqual(image_array.shape, (2, 3, 4))
        self.assertEqual(image_array.dtype, np.uint8)

    def test_visualize_image_attr_return_numpy_image_methods(self) -> None:
        masked_image = visualization.visualize_image_attr(
            self._attr(),
            self._image(),
            method="masked_image",
            sign="absolute_value",
            use_pyplot=False,
            return_numpy=True,
        )
        alpha_scaled_image = visualization.visualize_image_attr(
            self._attr(),
            self._image(),
            method="alpha_scaling",
            sign="absolute_value",
            use_pyplot=False,
            return_numpy=True,
        )

        self.assertEqual(masked_image.shape, (2, 3, 3))
        self.assertEqual(alpha_scaled_image.shape, (2, 3, 4))

    def test_visualize_image_attr_return_numpy_blended_heatmap(self) -> None:
        image_array = visualization.visualize_image_attr(
            self._attr(),
            self._image(),
            method="blended_heat_map",
            sign="absolute_value",
            alpha_overlay=0.25,
            use_pyplot=False,
            return_numpy=True,
        )

        self.assertEqual(image_array.shape, (2, 3, 4))
        self.assertEqual(image_array.dtype, np.uint8)
        self.assertAlmostEqual(image_array[0, 0, 3] / 255.0, 0.25, delta=1 / 255)
