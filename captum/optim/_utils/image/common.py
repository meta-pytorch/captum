import math
from typing import Callable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.optim._utils.reducer import posneg
from packaging import version

try:
    from PIL import Image
except (ImportError, AssertionError):
    print("The Pillow/PIL library is required to use Captum's Optim library")


def make_grid_image(
    tiles: Union[torch.Tensor, List[torch.Tensor]],
    images_per_row: int = 4,
    padding: int = 2,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Make grids from NCHW Image tensors in a way similar to torchvision.utils.make_grid,
    but without any channel duplication or creation behaviour.

    Args:

        tiles (torch.Tensor or list of torch.Tensor): A stack of NCHW image tensors or
            a list of NCHW image tensors to create a grid from.
        images_per_row (int, optional): The number of rows to use for the grid image.
            Default: ``4``
        padding (int, optional): The amount of padding between images in the grid
            images.
            padding: ``2``
        pad_value (float, optional): The value to use for the padding.
            Default: ``0.0``

    Returns:
        grid_img (torch.Tensor): The full NCHW grid image.
    """
    assert padding >= 0 and images_per_row >= 1
    if isinstance(tiles, (list, tuple)):
        assert all([t.device == tiles[0].device for t in tiles])
        assert all([t.dim() == 4 for t in tiles])
        tiles = torch.cat(tiles, 0)
    assert tiles.dim() == 4

    B, C, H, W = tiles.shape

    x_rows = min(images_per_row, B)
    y_rows = int(math.ceil(float(B) / x_rows))

    base_height = ((H + padding) * y_rows) + padding
    base_width = ((W + padding) * x_rows) + padding

    grid_img = torch.ones(1, C, base_height, base_width, device=tiles.device)
    grid_img = grid_img * pad_value

    n = 0
    for y in range(y_rows):
        for x in range(x_rows):
            if n >= B:
                break
            y_idx = ((H + padding) * y) + padding
            x_idx = ((W + padding) * x) + padding
            grid_img[..., y_idx : y_idx + H, x_idx : x_idx + W] = tiles[n : n + 1]
            n += 1
    return grid_img


def show(
    x: torch.Tensor,
    figsize: Optional[Tuple[int, int]] = None,
    scale: float = 255.0,
    images_per_row: Optional[int] = None,
    padding: int = 2,
    pad_value: float = 0.0,
) -> None:
    """
    Show CHW & NCHW tensors as an image.

    Alias: ``captum.optim.images.show``

    Args:

        x (torch.Tensor): The tensor you want to display as an image.
        figsize (tuple of int, optional): The height & width to use for displaying the
            ``ImageTensor`` figure, in the format of: (height, width).
            Default: ``None``
        scale (float, optional): Value to multiply the input tensor by so that
            it's value range is [0-255] for display.
            Default: ``255.0``
        images_per_row (int, optional): The number of images per row to use for the
            grid image. Default is set to ``None`` for no grid image creation.
            Default: ``None``
        padding (int, optional): The amount of padding between images in the grid
            images. This parameter only has an effect if ``images_per_row`` is not
            ``None``.
            Default: ``2``
        pad_value (float, optional): The value to use for the padding. This parameter
            only has an effect if ``images_per_row`` is not ``None``.
            Default: ``0.0``
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    if images_per_row is not None:
        x = make_grid_image(
            x, images_per_row=images_per_row, padding=padding, pad_value=pad_value
        )[0, ...]
    else:
        x = torch.cat([t[0] for t in x.split(1)], dim=2) if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(x.numpy().astype(np.uint8))
    plt.axis("off")
    plt.show()


def save_tensor_as_image(
    x: torch.Tensor,
    filename: str,
    scale: float = 255.0,
    mode: Optional[str] = None,
    images_per_row: Optional[int] = None,
    padding: int = 2,
    pad_value: float = 0.0,
) -> None:
    """
    Save RGB & RGBA image tensors with a shape of CHW or NCHW as images.

    Alias: ``captum.optim.images.save_tensor_as_image``

    Args:

        x (torch.Tensor): The tensor you want to save as an image.
        filename (str): The filename to use when saving the image.
        scale (float, optional): Value to multiply the input tensor by so that
            it's value range is [0-255] for saving.
            Default: ``255.0``
        mode (str, optional): A PIL / Pillow supported colorspace. Default is
            set to None for automatic RGB / RGBA detection and usage.
            Default: ``None``
        images_per_row (int, optional): The number of images per row to use for the
            grid image. Default is set to None for no grid image creation.
            Default: ``None``
        padding (int, optional): The amount of padding between images in the grid
            images. This parameter only has an effect if ``images_per_row`` is not
            ``None``.
            Default: ``2``
        pad_value (float, optional): The value to use for the padding. This parameter
            only has an effect if ``images_per_row`` is not ``None``.
            Default: ``0.0``
    """

    if x.dim() not in [3, 4]:
        raise ValueError(
            f"Incompatible number of dimensions. x.dim() = {x.dim()}; should be 3 or 4."
        )
    if images_per_row is not None:
        x = make_grid_image(
            x, images_per_row=images_per_row, padding=padding, pad_value=pad_value
        )[0, ...]
    else:
        x = torch.cat([t[0] for t in x.split(1)], dim=2) if x.dim() == 4 else x
    x = x.clone().cpu().detach().permute(1, 2, 0) * scale
    if mode is None:
        mode = "RGB" if x.shape[2] == 3 else "RGBA"
    im = Image.fromarray(x.numpy().astype(np.uint8), mode=mode)
    im.save(filename)


def get_neuron_pos(
    H: int, W: int, x: Optional[int] = None, y: Optional[int] = None
) -> Tuple[int, int]:
    """
    Args:

        H (int): The h position to use.
        W (int): The w position to use.
        x (int, optional): Optionally specify and exact x location of the neuron. If
            set to ``None``, then the center x location will be used.
            Default: ``None``
        y (int, optional): Optionally specify and exact y location of the neuron. If
            set to ``None``, then the center y location will be used.
            Default: ``None``

    Return:
        Tuple[_x, _y] (Tuple[int, int]): The x and y dimensions of the neuron.
    """
    if x is None:
        _x = W // 2
    else:
        assert x < W
        _x = x

    if y is None:
        _y = H // 2
    else:
        assert y < H
        _y = y
    return _x, _y


def _dot_cossim(
    x: torch.Tensor,
    y: torch.Tensor,
    cossim_pow: float = 0.0,
    dim: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes product between dot product and cosine similarity of two tensors along
    a specified dimension.

    Args:

        x (torch.Tensor): The tensor that you wish to compute the cosine similarity
            for in relation to tensor y.
        y (torch.Tensor): The tensor that you wish to compute the cosine similarity
            for in relation to tensor x.
        cossim_pow (float, optional): The desired cosine similarity power to use.
            Default: ``0.0``
        dim (int, optional): The target dimension for computing cosine similarity.
            Default: ``1``
        eps (float, optional): If cossim_pow is greater than zero, the desired
            epsilon value to use for cosine similarity calculations.
            Default: ``1e-8``

    Returns:
        tensor (torch.Tensor): Dot cosine similarity between x and y, along the
            specified dim.
    """

    dot = torch.sum(x * y, dim)
    if cossim_pow == 0:
        return dot
    return dot * torch.clamp(torch.cosine_similarity(x, y, eps=eps), 0.1) ** cossim_pow


# Handle older versions of PyTorch
# Defined outside of function in order to support JIT
_torch_norm = (
    torch.linalg.norm
    if version.parse(torch.__version__) >= version.parse("1.7.0")
    else torch.norm
)


def hue_to_rgb(
    angle: float, device: torch.device = torch.device("cpu"), warp: bool = True
) -> torch.Tensor:
    """
    Create an RGB unit vector based on a hue of the input angle.

    Args:

        angle (float): The hue angle to create an RGB color for.
        device (torch.device, optional): The device to create the angle color tensor
            on.
            Default: ``torch.device("cpu")``
        warp (bool, optional): Whether or not to make colors more distinguishable.
            Default: ``True``

    Returns:
        color_vec (torch.Tensor): A color vector.
    """

    angle = angle - 360 * (angle // 360)
    colors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.7071, 0.7071, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.7071, 0.7071],
            [0.0, 0.0, 1.0],
            [0.7071, 0.0, 0.7071],
        ],
        device=device,
    )

    idx = math.floor(angle / 60)
    d = (angle - idx * 60) / 60

    if warp:
        # Idea from: https://github.com/tensorflow/lucid/pull/193
        d = (
            math.sin(d * math.pi / 2)
            if idx % 2 == 0
            else 1 - math.sin((1 - d) * math.pi / 2)
        )

    vec = (1 - d) * colors[idx] + d * colors[(idx + 1) % 6]
    return vec / _torch_norm(vec)


def nchannels_to_rgb(
    x: torch.Tensor, warp: bool = True, eps: float = 1e-4
) -> torch.Tensor:
    """
    Convert an NCHW image with n channels into a 3 channel RGB image.

    Args:

        x (torch.Tensor): NCHW image tensor to transform into RGB image.
        warp (bool, optional): Whether or not to make colors more distinguishable.
            Default: ``True``
        eps (float, optional): An optional epsilon value.
            Default: ``1e-4``

    Returns:
        tensor (torch.Tensor): An NCHW RGB image tensor.
    """

    assert x.dim() == 4

    if (x < 0).any():
        x = posneg(x.permute(0, 2, 3, 1), -1).permute(0, 3, 1, 2)

    rgb = torch.zeros(1, 3, x.size(2), x.size(3), device=x.device)
    num_channels = x.size(1)
    for i in range(num_channels):
        rgb_angle = hue_to_rgb(360 * i / num_channels, device=x.device, warp=warp)
        rgb = rgb + (x[:, i][:, None, :, :] * rgb_angle[None, :, None, None])

    rgb = rgb + (
        torch.ones(1, 1, x.size(2), x.size(3), device=x.device)
        * (torch.sum(x, 1) - torch.max(x, 1)[0])[:, None]
    )
    rgb = rgb / (eps + _torch_norm(rgb, dim=1, keepdim=True))
    return rgb * _torch_norm(x, dim=1, keepdim=True)


def weights_to_heatmap_2d(
    weight: torch.Tensor,
    colors: List[str] = ["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"],
) -> torch.Tensor:
    """
    Create a color heatmap of an input weight tensor. By default red represents
    excitatory values, blue represents inhibitory values, and white represents
    no excitation or inhibition.

    Args:

        weight (torch.Tensor): A 2d tensor to create the heatmap from.
        colors (list of str, optional): A list of 5 strings containing hex triplet
            (six digit), three-byte hexadecimal color values to use for coloring
            the heatmap.
            Default: ``["0571b0", "92c5de", "f7f7f7", "f4a582", "ca0020"]``

    Returns:
        color_tensor (torch.Tensor): A weight heatmap.
    """

    assert weight.dim() == 2
    assert len(colors) == 5
    assert all([len(c) == 6 for c in colors])

    def get_color(x: str, device: torch.device = torch.device("cpu")) -> torch.Tensor:
        def hex2base10(x: str) -> float:
            return int(x, 16) / 255.0

        return torch.tensor(
            [hex2base10(x[0:2]), hex2base10(x[2:4]), hex2base10(x[4:6])], device=device
        )

    color_list = [get_color(c, weight.device) for c in colors]
    x = weight.expand((3, weight.shape[0], weight.shape[1])).permute(1, 2, 0)

    color_tensor = (
        (x >= 0) * (x < 0.5) * ((1 - x * 2) * color_list[2] + x * 2 * color_list[3])
        + (x >= 0)
        * (x >= 0.5)
        * ((1 - (x - 0.5) * 2) * color_list[3] + (x - 0.5) * 2 * color_list[4])
        + (x < 0)
        * (x > -0.5)
        * ((1 - (-x * 2)) * color_list[2] + (-x * 2) * color_list[1])
        + (x < 0)
        * (x <= -0.5)
        * ((1 - (-x - 0.5) * 2) * color_list[1] + (-x - 0.5) * 2 * color_list[0])
    ).permute(2, 0, 1)
    return color_tensor


def _create_new_vector(
    x: torch.Tensor,
    vec: torch.Tensor,
    activation_fn: Optional[
        Callable[[torch.Tensor], torch.Tensor]
    ] = torch.nn.functional.relu,
    move_channel_dim_to_final_dim: bool = True,
) -> torch.Tensor:
    """
    Create a vector using a given set of activations and another vector.
    This function is intended for use in CLIP related loss objectives.

    https://distill.pub/2021/multimodal-neurons/
    https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py
    The einsum equation: "ijkl,j->ikl", used by the paper's associated code is the
    same thing as: "[..., C] @ vec", where vec has a shape of 'C'.

    Args:

        x (torch.Tensor): A set of 2d or 4d activations.
        vec (torch.Tensor): A 1D direction vector to use, with a compatible shape for
            computing the matrix product of the activations. See torch.matmul for
            See torch.matmul for more details on compatible shapes:
            https://pytorch.org/docs/stable/generated/torch.matmul.html
            By default, ``vec`` is expected to share the same size as the channel or
            feature dimension of the activations.
        activation_fn (Callable, optional): An optional activation function to
            apply to the activations before computing the matrix product. If set
            to None, then no activation function will be used.
            Default: ``torch.nn.functional.relu``
        move_channel_dim_to_final_dim (bool, optional): Whether or not to move the
            channel dimension to the last dimension before computing the matrix
            product.
            Default: ``True``

    Returns
        x (torch.Tensor): A vector created from the input activations and the
            stored vector.
    """
    assert x.device == vec.device
    assert x.dim() > 1 and vec.dim() == 1
    if activation_fn:
        x = activation_fn(x)
    if x.dim() > 2:
        if move_channel_dim_to_final_dim:
            permute_vals = [0] + list(range(x.dim()))[2:] + [1]
            x = x.permute(*permute_vals)
        mean_vals = list(range(1, x.dim() - 1))
        return torch.mean(x @ vec, mean_vals)
    else:
        return (x @ vec)[:, None]
