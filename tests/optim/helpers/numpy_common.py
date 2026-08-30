from typing import List

import numpy as np


def weights_to_heatmap_2d(
    array: np.ndarray,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> np.ndarray:
    """
    Create a color heatmap of an input weight array.
    By default red represents excitatory values,
    blue represents inhibitory values, and white represents
    no excitation or inhibition.

    Args:
        weight (array):  A 2d array to create the heatmap from.
        colors (List of strings):  A list of strings containing color
        hex values to use for coloring the heatmap.
    Returns:
        *array*:  A weight heatmap.
    """

    assert array.ndim == 2
    assert len(colors) == 5

    def get_color(x: str) -> np.ndarray:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return np.array([hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])])

    def color_scale(x: np.ndarray) -> np.ndarray:
        if x < 0:
            x = -x
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[1])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[1]) + x * get_color(colors[0])
        else:
            if x < 0.5:
                x = x * 2
                return (1 - x) * get_color(colors[2]) + x * get_color(colors[3])
            else:
                x = (x - 0.5) * 2
                return (1 - x) * get_color(colors[3]) + x * get_color(colors[4])

    return np.stack([np.stack([color_scale(x) for x in a]) for a in array]).transpose(
        2, 0, 1
    )
