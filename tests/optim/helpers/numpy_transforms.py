import math
from typing import cast, List, no_type_check, Optional, Tuple, Union

import numpy as np
from captum.optim._utils.typing import IntSeqOrIntType


class BlendAlpha:
    """
    NumPy version of the BlendAlpha transform.

    Args:
        background (array, optional):  An NCHW image array to be used as the
            Alpha channel's background.
    """

    def __init__(self, background: Optional[np.ndarray] = None) -> None:
        super().__init__()
        self.background = background

    def blend_alpha(self, x: np.ndarray) -> np.ndarray:
        """
        Blend the Alpha channel into the RGB channels.

        Args:
            x (array): RGBA image array to blend into an RGB image array.
        Returns:
            blended (array): RGB image array.
        """
        assert x.shape[1] == 4
        assert x.ndim == 4
        rgb, alpha = x[:, :3, ...], x[:, 3:4, ...]
        background = (
            self.background
            if self.background is not None
            else np.random.randn(*rgb.shape)
        )
        blended = alpha * rgb + (1 - alpha) * background
        return blended


class RandomSpatialJitter:
    """
    NumPy version of the RandomSpatialJitter transform.

    Args:
        translate (int):  The amount to translate the H and W dimensions
            of an CHW or NCHW array.
    """

    def __init__(self, translate: int) -> None:
        super().__init__()
        self.pad_range = translate

    def translate_array(self, x: np.ndarray, insets) -> np.ndarray:
        x = np.pad(x, (self.pad_range, self.pad_range), "reflect")
        x = np.roll(x, (self.pad_range - insets[1]), axis=0)
        x = np.roll(x, (self.pad_range - insets[0]), axis=1)

        h_crop = x.shape[0] - (self.pad_range * 2)
        w_crop = x.shape[1] - (self.pad_range * 2)
        sw, sh = x.shape[1] // 2 - (w_crop // 2), x.shape[0] // 2 - (h_crop // 2)
        x = x[..., sh : sh + h_crop, sw : sw + w_crop]
        return x

    def jitter(self, x: np.ndarray) -> np.ndarray:
        insets = (
            np.random.randint(high=self.pad_range),
            np.random.randint(high=self.pad_range),
        )
        return self.translate_array(x, insets)


class CenterCrop:
    """
    Center crop a specified amount from a tensor.

    Arguments:
        size (int or sequence of int): Number of pixels to center crop away.
        pixels_from_edges (bool, optional): Whether to treat crop size values
            as the number of pixels from the tensor's edge, or an exact shape
            in the center.
        offset_left (bool, optional): If the cropped away sides are not
            equal in size, offset to the left. Default is set to False
            for offseting to the right. This parameter is only valid when
            pixels_from_edges is False.
    """

    def __init__(
        self,
        size: IntSeqOrIntType = 0,
        pixels_from_edges: bool = False,
        offset_left: bool = False,
    ) -> None:
        super(CenterCrop, self).__init__()
        self.crop_vals = size
        self.pixels_from_edges = pixels_from_edges
        self.offset_left = offset_left

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Center crop an input.
        Arguments:
            input (array): Input to center crop.
        Returns:
            tensor (array): A center cropped tensor.
        """

        return center_crop(
            input, self.crop_vals, self.pixels_from_edges, self.offset_left
        )


@no_type_check
def center_crop(
    input: np.ndarray,
    crop_vals: IntSeqOrIntType,
    pixels_from_edges: bool = False,
    offset_left: bool = False,
) -> np.ndarray:
    """
    Center crop a specified amount from a array.
    Arguments:
        input (array):  A CHW or NCHW image array to center crop.
        crop_vals (int, sequence, int): Number of pixels to center crop away.
        pixels_from_edges (bool, optional): Whether to treat crop size values
            as the number of pixels from the array's edge, or an exact shape
            in the center.
        offset_left (bool, optional): If the cropped away sides are not
            equal in size, offset to the left. Default is set to False
            for offseting to the right. This parameter is only valid when
            pixels_from_edges is False.
    Returns:
        *array*:  A center cropped array.
    """

    assert input.ndim == 3 or input.ndim == 4
    crop_vals = [crop_vals] if not hasattr(crop_vals, "__iter__") else crop_vals
    crop_vals = cast(Union[List[int], Tuple[int], Tuple[int, int]], crop_vals)
    assert len(crop_vals) == 1 or len(crop_vals) == 2
    crop_vals = crop_vals * 2 if len(crop_vals) == 1 else crop_vals

    if input.ndim == 4:
        h, w = input.shape[2], input.shape[3]
    if input.ndim == 3:
        h, w = input.shape[1], input.shape[2]

    if pixels_from_edges:
        h_crop = h - crop_vals[0]
        w_crop = w - crop_vals[1]
        sw, sh = w // 2 - (w_crop // 2), h // 2 - (h_crop // 2)
        x = input[..., sh : sh + h_crop, sw : sw + w_crop]
    else:
        h_crop = h - int(math.ceil((h - crop_vals[0]) / 2.0))
        w_crop = w - int(math.ceil((w - crop_vals[1]) / 2.0))
        if h % 2 == 0 and crop_vals[0] % 2 != 0 or h % 2 != 0 and crop_vals[0] % 2 == 0:
            h_crop = h_crop + 1 if offset_left else h_crop
        if w % 2 == 0 and crop_vals[1] % 2 != 0 or w % 2 != 0 and crop_vals[1] % 2 == 0:
            w_crop = w_crop + 1 if offset_left else w_crop
        x = input[..., h_crop - crop_vals[0] : h_crop, w_crop - crop_vals[1] : w_crop]
    return x


class ToRGB:
    """
    NumPy version of the ToRGB transform
    """

    @staticmethod
    def klt_transform() -> np.ndarray:
        KLT = [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
        transform = np.array(KLT, dtype=float)
        transform = transform / np.linalg.norm(transform, axis=0).max()
        return transform

    @staticmethod
    def i1i2i3_transform() -> np.ndarray:
        i1i2i3_matrix = [
            [1 / 3, 1 / 3, 1 / 3],
            [1 / 2, 0, -1 / 2],
            [-1 / 4, 1 / 2, -1 / 4],
        ]
        return np.array(i1i2i3_matrix, dtype=float)

    def __init__(self, transform: Union[str, np.ndarray] = "klt") -> None:
        super().__init__()
        assert isinstance(transform, str) or isinstance(transform, np.ndarray)
        if isinstance(transform, np.ndarray):
            assert list(transform.shape) == [3, 3]
            self.transform = transform
        elif transform == "klt":
            self.transform = ToRGB.klt_transform()
        elif transform == "i1i2i3":
            self.transform = ToRGB.i1i2i3_transform()
        else:
            raise ValueError(
                "transform has to be either 'klt', 'i1i2i3', or a matrix array."
            )

    def to_rgb(self, x: np.ndarray, inverse: bool = False) -> np.ndarray:
        """
        Args:
            x (array):  A CHW or NCHW RGB or RGBA image array.
            inverse (bool, optional):  Whether to recorrelate or decorrelate colors.
                Default is set to False.
        Returns:
            *array*:  An array with it's colors recorrelated or decorrelated.
        """

        assert x.ndim == 3 or x.ndim == 4

        # alpha channel is taken off...
        if x.ndim == 3:
            has_alpha = x.shape[0] == 4
            h, w = x.shape[1], x.shape[2]
        elif x.ndim == 4:
            has_alpha = x.shape[1] == 4
            h, w = x.shape[2], x.shape[3]
        if has_alpha:
            if x.ndim == 3:
                x, alpha_channel = x[:3], x[3:]
            elif x.ndim == 4:
                x, alpha_channel = x[:, :3], x[:, 3:]
            assert x.ndim == alpha_channel.ndim  # ensure we "keep_dim"

        if x.ndim == 3:
            flat = x.reshape(x.shape[1], h * w)
        elif x.ndim == 4:
            flat = x.reshape(x.shape[0], x.shape[1], h * w)

        if inverse:
            correct = np.linalg.inv(self.transform) @ flat
        else:
            correct = self.transform @ flat

        if x.ndim == 3:
            chw = correct.reshape(x.shape[1], h, w)
        elif x.ndim == 4:
            chw = correct.reshape(x.shape[0], x.shape[1], h, w)

        # ...alpha channel is concatenated on again.
        if has_alpha:
            d = 0 if x.ndim == 3 else 1
            chw = np.concatenate([chw, alpha_channel], d)

        return chw
