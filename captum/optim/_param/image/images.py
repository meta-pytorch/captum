from types import MethodType
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

try:
    from PIL import Image
except (ImportError, AssertionError):
    print("The Pillow/PIL library is required to use Captum's Optim library")

from captum.optim._param.image.transforms import SymmetricPadding, ToRGB
from captum.optim._utils.image.common import save_tensor_as_image, show

TORCH_VERSION = torch.__version__


class ImageTensor(torch.Tensor):
    r"""
    A subclass of :class:`torch.Tensor` that provides functions for easy loading,
    saving, and displaying image tensors.

    Alias: ``captum.optim.ImageTensor``

    Example using file path or URL::

        >>> image_tensor = opt.images.ImageTensor.load(<path/to/image_file>)
        >>> image_tensor.export(filename="image_tensor.jpg")  # Save image(s)
        >>> image_tensor.show()  # Displays image(s) via Matplotlib

    Example using ``torch.Tensor``::

        >>> image_tensor = torch.randn(1, 3, 224, 224)
        >>> image_tensor = opt.images.ImageTensor(image_tensor)

    Example using ``np.ndarray``::

        >>> image_tensor = np.random.rand(1, 3, 224, 224)
        >>> image_tensor = opt.images.ImageTensor(image_tensor)
    """

    @staticmethod
    def __new__(
        cls: Type["ImageTensor"],
        x: Union[List, np.ndarray, torch.Tensor] = [],
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:

            x (list or np.ndarray or torch.Tensor): A list, NumPy array, or PyTorch
                tensor to create an ``ImageTensor`` from.

        Returns:
           x (ImageTensor): An ``ImageTensor`` instance.
        """
        if (
            isinstance(x, torch.Tensor)
            and x.is_cuda
            or isinstance(x, torch.Tensor)
            and x.dtype != torch.float32
        ):
            x.show = MethodType(cls.show, x)
            x.export = MethodType(cls.export, x)
            return x
        else:
            return super().__new__(cls, x, *args, **kwargs)

    @classmethod
    def load(cls, path: str, scale: float = 255.0, mode: str = "RGB") -> "ImageTensor":
        """
        Load an image file from a URL or local filepath directly into an
        ``ImageTensor``.

        Args:

            path (str): A URL or filepath to an image.
            scale (float, optional): The image scale to use.
                Default: ``255.0``
            mode (str, optional): The image loading mode / colorspace to use.
                Default: ``"RGB"``

        Returns:
           x (ImageTensor): An `ImageTensor` instance.
        """
        if path.startswith("https://") or path.startswith("http://"):
            headers = {"User-Agent": "Captum"}
            response = requests.get(path, stream=True, headers=headers)
            img = Image.open(response.raw)
        else:
            img = Image.open(path)
        img_np = np.array(img.convert(mode)).astype(np.float32)
        return cls(img_np.transpose(2, 0, 1) / scale)

    @classmethod
    def open(cls, path: str, scale: float = 255.0, mode: str = "RGB") -> "ImageTensor":
        r"""Alias for :func:`load`."""
        return cls.load(path=path, scale=scale, mode=mode)

    def __repr__(self) -> str:
        prefix = "ImageTensor("
        indent = len(prefix)
        tensor_str = torch._tensor_str._tensor_str(self, indent)
        suffixes = []
        if self.device.type != torch._C._get_default_device() or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        ):
            suffixes.append("device='" + str(self.device) + "'")
        return torch._tensor_str._add_suffixes(
            prefix + tensor_str, suffixes, indent, force_newline=self.is_sparse
        )

    @classmethod
    def __torch_function__(
        cls: Type["ImageTensor"],
        func: Callable,
        types: List[Type[torch.Tensor]],
        args: Tuple = (),
        kwargs: dict = None,
    ) -> torch.Tensor:
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

    def show(
        self,
        figsize: Optional[Tuple[int, int]] = None,
        scale: float = 255.0,
        images_per_row: Optional[int] = None,
        padding: int = 2,
        pad_value: float = 0.0,
    ) -> None:
        """
        Display image(s) in the ``ImageTensor`` instance using
        :func:`captum.optim.show`.

        Args:

            figsize (tuple of int, optional): The height & width to use for displaying
                the ``ImageTensor`` figure, in the format of: (height, width).
                Default: ``None``
            scale (float, optional): Value to multiply the ``ImageTensor`` by so that
                it's value range is [0-255] for display.
                Default: ``255.0``
            images_per_row (int, optional): The number of images per row to use for the
                grid image. Default is set to ``None`` for no grid image creation.
                Default: ``None``
            padding (int, optional): The amount of padding between images in the grid
                images. This parameter only has an effect if ``images_per_row`` is not
                ``None``.
                Default: ``2``
            pad_value (float, optional): The value to use for the padding. This
                parameter only has an effect if ``images_per_row`` is not None.
                Default: ``0.0``
        """
        show(
            self,
            figsize=figsize,
            scale=scale,
            images_per_row=images_per_row,
            padding=padding,
            pad_value=pad_value,
        )

    def export(
        self,
        filename: str,
        scale: float = 255.0,
        mode: Optional[str] = None,
        images_per_row: Optional[int] = None,
        padding: int = 2,
        pad_value: float = 0.0,
    ) -> None:
        """
        Save image(s) in the `ImageTensor` instance as an image file, using
        :func:`captum.optim.save_tensor_as_image`.

        Args:

            filename (str): The filename to use when saving the ``ImageTensor`` as an
                image file.
            scale (float, optional): Value to multiply the ``ImageTensor`` by so that
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
            pad_value (float, optional): The value to use for the padding. This
                parameter only has an effect if ``images_per_row`` is not ``None``.
                Default: ``0.0``
        """
        save_tensor_as_image(
            self,
            filename=filename,
            scale=scale,
            mode=mode,
            images_per_row=images_per_row,
            padding=padding,
            pad_value=pad_value,
        )


class InputParameterization(torch.nn.Module):
    def forward(self) -> torch.Tensor:
        raise NotImplementedError


class ImageParameterization(InputParameterization):
    r"""The base class for all Image Parameterizations"""

    pass


class FFTImage(ImageParameterization):
    """
    Parameterize an image using inverse real 2D FFT

    Example::

        >>> fft_image = opt.images.FFTImage(size=(224, 224))
        >>> output_image = fft_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])

    Example for using an initialization tensor::

        >>> init = torch.randn(1, 3, 224, 224)
        >>> fft_image = opt.images.FFTImage(init=init)
        >>> output_image = fft_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])
    """

    __constants__ = ["size"]

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (tuple of int): The height & width dimensions to use for the
                parameterized output image tensor, in the format of: (height, width).
            channels (int, optional): The number of channels to use for each image.
                Default: ``3``
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: ``1``
            init (torch.Tensor, optional): Optionally specify a CHW or NCHW tensor to
                use instead of creating one.
                Default: ``None``
        """
        super().__init__()
        if init is None:
            assert len(size) == 2
            self.size = size
        else:
            assert init.dim() == 3 or init.dim() == 4
            if init.dim() == 3:
                init = init.unsqueeze(0)
            self.size = (init.size(2), init.size(3))
        self.torch_rfft, self.torch_irfft, self.torch_fftfreq = self._get_fft_funcs()

        frequencies = self._rfft2d_freqs(*self.size)
        scale = 1.0 / torch.max(
            frequencies,
            torch.full_like(frequencies, 1.0 / (max(self.size[0], self.size[1]))),
        )
        scale = scale * ((self.size[0] * self.size[1]) ** (1 / 2))
        spectrum_scale = scale[None, :, :, None]

        if init is None:
            coeffs_shape = (
                batch,
                channels,
                self.size[0],
                self.size[1] // 2 + 1,
                2,
            )
            random_coeffs = torch.randn(
                coeffs_shape
            )  # names=["C", "H_f", "W_f", "complex"]
            fourier_coeffs = random_coeffs / 50
        else:
            spectrum_scale = spectrum_scale.to(init.device)
            fourier_coeffs = self.torch_rfft(init) / spectrum_scale

        self.register_buffer("spectrum_scale", spectrum_scale)
        self.fourier_coeffs = nn.Parameter(fourier_coeffs)

    def _rfft2d_freqs(self, height: int, width: int) -> torch.Tensor:
        """
        Computes 2D spectrum frequencies.

        Args:

            height (int): The h dimension of the 2d frequency scale.
            width (int): The w dimension of the 2d frequency scale.

        Returns:
            tensor (torch.Tensor): A 2d frequency scale tensor.
        """

        fy = self.torch_fftfreq(height)[:, None]
        fx = self.torch_fftfreq(width)[: width // 2 + 1]
        return torch.sqrt((fx * fx) + (fy * fy))

    @torch.jit.export
    def _torch_irfftn(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            x = torch.view_as_complex(x)
        return torch.fft.irfftn(x, s=self.size)  # type: ignore

    def _get_fft_funcs(self) -> Tuple[Callable, Callable, Callable]:
        """
        Support older versions of PyTorch. This function ensures that the same FFT
        operations are carried regardless of whether your PyTorch version has the
        torch.fft update.

        Returns:
            fft_functions (tuple of callable): A list of FFT functions to use for
                irfft, rfft, and fftfreq operations.
        """

        if version.parse(TORCH_VERSION) > version.parse("1.7.0"):
            if version.parse(TORCH_VERSION) <= version.parse("1.8.0"):
                global torch
                import torch.fft

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.view_as_real(torch.fft.rfftn(x, s=self.size))

            torch_irfftn = self._torch_irfftn

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                return torch.fft.fftfreq(v, d)

        else:

            def torch_rfft(x: torch.Tensor) -> torch.Tensor:
                return torch.rfft(x, signal_ndim=2)

            def torch_irfftn(x: torch.Tensor) -> torch.Tensor:
                return torch.irfft(x, signal_ndim=2)[
                    :, :, : self.size[0], : self.size[1]
                ]

            def torch_fftfreq(v: int, d: float = 1.0) -> torch.Tensor:
                """PyTorch version of np.fft.fftfreq"""
                results = torch.empty(v)
                s = (v - 1) // 2 + 1
                results[:s] = torch.arange(0, s)
                results[s:] = torch.arange(-(v // 2), 0)
                return results * (1.0 / (v * d))

        return torch_rfft, torch_irfftn, torch_fftfreq

    def forward(self) -> torch.Tensor:
        """
        Returns:
            output (torch.Tensor): A spatially recorrelated NCHW tensor.
        """

        scaled_spectrum = self.fourier_coeffs * self.spectrum_scale
        output = self.torch_irfft(scaled_spectrum)
        if torch.jit.is_scripting():
            return output
        return output.refine_names("B", "C", "H", "W")


class PixelImage(ImageParameterization):
    """
    Parameterize a simple pixel image tensor that requires no additional transforms.

    Example::

        >>> pixel_image = opt.images.PixelImage(size=(224, 224))
        >>> output_image = pixel_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])

    Example for using an initialization tensor::

        >>> init = torch.randn(1, 3, 224, 224)
        >>> pixel_image = opt.images.PixelImage(init=init)
        >>> output_image = pixel_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])
    """

    def __init__(
        self,
        size: Tuple[int, int] = None,
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:

            size (tuple of int): The height & width dimensions to use for the
                parameterized output image tensor, in the format of: (height, width).
            channels (int, optional): The number of channels to use for each image.
                Default: ``3``
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: ``1``
            init (torch.Tensor, optional): Optionally specify a CHW or NCHW tensor to
                use instead of creating one.
                Default: ``None``
        """
        super().__init__()
        if init is None:
            assert size is not None and channels is not None and batch is not None
            init = torch.randn([batch, channels, size[0], size[1]]) / 10 + 0.5
        else:
            assert init.dim() == 3 or init.dim() == 4
            if init.dim() == 3:
                init = init.unsqueeze(0)
        self.image = nn.Parameter(init)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            output (torch.Tensor): An NCHW tensor.
        """
        if torch.jit.is_scripting():
            return self.image
        return self.image.refine_names("B", "C", "H", "W")


class LaplacianImage(ImageParameterization):
    """
    Parameterize an image tensor with a laplacian pyramid.

    Example::

        >>> laplacian_image = opt.images.LaplacianImage(size=(224, 224))
        >>> output_image = laplacian_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])

    Example for using an initialization tensor::

        >>> init = torch.randn(1, 3, 224, 224)
        >>> laplacian_image = opt.images.LaplacianImage(init=init)
        >>> output_image = laplacian_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([1, 3, 224, 224])
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
        power: float = 0.1,
        scale_list: List[float] = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    ) -> None:
        """
        Args:

            size (tuple of int): The height & width dimensions to use for the
                parameterized output image tensor, in the format of: (height, width).
            channels (int, optional): The number of channels to use for each image.
                Default: ``3``
            batch (int, optional): The number of images to stack along the batch
                dimension.
                Default: ``1``
            init (torch.Tensor, optional): Optionally specify a CHW or NCHW tensor to
                use instead of creating one.
                Default: ``None``
            power (float, optional): The desired power value to use.
                Default: ``0.1``
            scale_list (list of float, optional): The desired list of scale values to
                use in the laplacian pyramid. The height & width dimensions specified
                in ``size`` or used in the ``init`` tensor should be divisible by every
                scale value in the scale list with no remainder left over. The default
                ``scale_list`` values are set to work with a ``size`` of
                ``(224, 224)``.
                Default: ``[1.0, 2.0, 4.0, 8.0, 16.0, 32.0]``
        """
        super().__init__()
        if init is not None:
            assert init.dim() in [3, 4]
            init = init.unsqueeze(0) if init.dim() == 3 else init
            size = list(init.shape[2:])

        tensor_params, scaler = [], []
        for scale in scale_list:
            assert size[0] % scale == 0 and size[1] % scale == 0, (
                "The chosen image height & width dimensions"
                + " must be divisible by all scale values "
                + " with no remainder left over."
            )

            h, w = int(size[0] // scale), int(size[1] // scale)
            if init is None:
                x = torch.randn([batch, channels, h, w]) / 10
            else:
                x = F.interpolate(init.clone(), size=(h, w), mode="bilinear")
                x = x / 10
            upsample = torch.nn.Upsample(scale_factor=scale, mode="nearest")
            x = x * (scale**power) / (max(scale_list) ** power)
            x = torch.nn.Parameter(x)
            tensor_params.append(x)
            scaler.append(upsample)
        self.tensor_params = torch.nn.ParameterList(tensor_params)
        self.scaler = scaler

    def forward(self) -> torch.Tensor:
        """
        Returns:
            output (torch.Tensor): An NCHW tensor created from a laplacian pyramid.
        """
        A = []
        for xi, upsamplei in zip(self.tensor_params, self.scaler):
            A.append(upsamplei(xi))
        output = sum(A) + 0.5

        if torch.jit.is_scripting():
            return output
        return output.refine_names("B", "C", "H", "W")


class SimpleTensorParameterization(ImageParameterization):
    """
    Parameterize a simple tensor with or without it requiring grad.
    Compared to PixelImage, this parameterization has no specific shape requirements
    and does not wrap inputs in nn.Parameter.

    This parameterization can for example be combined with StackImage for batch
    dimensions that both require and don't require gradients.

    This parameterization can also be combined with nn.ModuleList as workaround for
    TorchScript / JIT not supporting nn.ParameterList. SharedImage uses this module
    internally for this purpose.
    """

    def __init__(self, tensor: torch.Tensor = None) -> None:
        """
        Args:

            tensor (torch.Tensor): The tensor to return every time this module is
                called.
        """
        super().__init__()
        assert isinstance(tensor, torch.Tensor)
        self.tensor = tensor

    def forward(self) -> torch.Tensor:
        """
        Returns:
            tensor (torch.Tensor): The tensor stored during initialization.
        """
        return self.tensor


class SharedImage(ImageParameterization):
    """
    Share some image parameters across the batch to increase spatial alignment,
    by using interpolated lower resolution tensors.
    This is sort of like a laplacian pyramid but more general.

    Offsets are similar to phase in Fourier transforms, and can be applied to
    any dimension.

    Mordvintsev, et al., "Differentiable Image Parameterizations", Distill, 2018.
    https://distill.pub/2018/differentiable-parameterizations/

    Example::

        >>> fft_image = opt.images.FFTImage(size=(224, 224), batch=2)
        >>> shared_shapes = ((1, 3, 64, 64), (4, 3, 32, 32))
        >>> shared_image = opt.images.SharedImage(shared_shapes, fft_image)
        >>> output_image = shared_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([2, 3, 224, 224])
    """

    __constants__ = ["offset"]

    def __init__(
        self,
        shapes: Union[Tuple[Tuple[int]], Tuple[int]] = None,
        parameterization: ImageParameterization = None,
        offset: Union[int, Tuple[int], Tuple[Tuple[int]], None] = None,
    ) -> None:
        """
        Args:

            shapes (list of int or list of list of int): The shapes of the shared
                tensors to use for creating the nn.Parameter tensors.
            parameterization (ImageParameterization): An image parameterization
                instance.
            offset (int or list of int or list of list of int, optional): The offsets
                to use for the shared tensors.
                Default: ``None``
        """
        super().__init__()
        assert shapes is not None
        A = []
        shared_shapes = [shapes] if type(shapes[0]) is not tuple else shapes
        for shape in shared_shapes:
            assert len(shape) >= 2 and len(shape) <= 4
            shape = ([1] * (4 - len(shape))) + list(shape)
            batch, channels, height, width = shape
            shape_param = torch.nn.Parameter(
                torch.randn([batch, channels, height, width])
            )
            A.append(SimpleTensorParameterization(shape_param))
        self.shared_init = torch.nn.ModuleList(A)
        self.parameterization = parameterization
        self.offset = self._get_offset(offset, len(A)) if offset is not None else None

    def _get_offset(self, offset: Union[int, Tuple[int]], n: int) -> List[List[int]]:
        """
        Given offset values, return a list of offsets for _apply_offset to use.

        Args:

            offset (int or list of int or list of list of int, optional): The offsets
                to use for the shared tensors.
            n (int): The number of tensors needing offset values.

        Returns:
            offset (List[List[int]]): A list of offset values.
        """
        if type(offset) is tuple or type(offset) is list:
            if type(offset[0]) is tuple or type(offset[0]) is list:
                assert len(offset) == n and all(len(t) == 4 for t in offset)
            else:
                assert len(offset) >= 1 and len(offset) <= 4
                offset = [([0] * (4 - len(offset))) + list(offset)] * n
        else:
            offset = [[offset] * 4] * n
        offset = [list(v) for v in offset]
        assert all([all([type(o) is int for o in v]) for v in offset])
        return offset

    @torch.jit.ignore
    def _apply_offset(self, x_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply list of offsets to list of tensors.

        Args:

            x_list (list of torch.Tensor): list of tensors to offset.

        Returns:
            A (list of torch.Tensor): list of offset tensors.
        """

        A: List[torch.Tensor] = []
        for x, offset in zip(x_list, self.offset):
            assert x.dim() == 4
            size = list(x.size())

            offset_pad = (
                [[abs(offset[0])] * 2]
                + [[abs(offset[1])] * 2]
                + [[abs(offset[2])] * 2]
                + [[abs(offset[3])] * 2]
            )

            x = SymmetricPadding.apply(x, offset_pad)

            for o, s in zip(offset, range(x.dim())):
                x = torch.roll(x, shifts=o, dims=s)

            x = x[: size[0], : size[1], : size[2], : size[3]]
            A.append(x)
        return A

    def _interpolate_bilinear(
        self,
        x: torch.Tensor,
        size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Perform interpolation without any warnings.

        Args:

            x (torch.Tensor): The NCHW tensor to resize.
            size (tuple of int): The desired output size to resize the input to, with
                a format of: [height, width].

        Returns:
            x (torch.Tensor): A resized NCHW tensor.
        """
        assert x.dim() == 4
        assert len(size) == 2

        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        return x

    def _interpolate_trilinear(
        self,
        x: torch.Tensor,
        size: Tuple[int, int, int],
    ) -> torch.Tensor:
        """
        Perform interpolation without any warnings.

        Args:

            x (torch.Tensor): The NCHW tensor to resize.
            size (tuple of int): The desired output size to resize the input to, with
                a format of: [channels, height, width].

        Returns:
            x (torch.Tensor): A resized NCHW tensor.
        """
        x = x.unsqueeze(0)
        assert x.dim() == 5
        x = F.interpolate(
            x,
            size=size,
            mode="trilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        return x.squeeze(0)

    def _interpolate_tensor(
        self, x: torch.Tensor, batch: int, channels: int, height: int, width: int
    ) -> torch.Tensor:
        """
        Linear interpolation for 4D, 5D, and 6D tensors. If the batch dimension needs
        to be resized, we move it's location temporarily for F.interpolate.

        Args:

            x (torch.Tensor): The tensor to resize.
            batch (int): The batch size to resize the tensor to.
            channels (int): The channel size to resize the tensor to.
            height (int): The height to resize the tensor to.
            width (int): The width to resize the tensor to.

        Returns:
            tensor (torch.Tensor): A resized tensor.
        """

        if x.size(1) == channels:
            size = (height, width)
            x = self._interpolate_bilinear(x, size=size)
        else:
            size = (channels, height, width)
            x = self._interpolate_trilinear(x, size=size)
        if x.size(0) != batch:
            x = x.permute(1, 0, 2, 3)
            x = self._interpolate_trilinear(x, size=(batch, x.size(2), x.size(3)))
            x = x.permute(1, 0, 2, 3)
        return x

    def forward(self) -> torch.Tensor:
        """
        Returns:
            output (torch.Tensor): An NCHW image parameterization output.
        """
        image = self.parameterization()
        x = [
            self._interpolate_tensor(
                shared_tensor(),
                image.size(0),
                image.size(1),
                image.size(2),
                image.size(3),
            )
            for shared_tensor in self.shared_init
        ]
        if self.offset is not None:
            x = self._apply_offset(x)
        output = image + torch.cat(x, 0).sum(0, keepdim=True)

        if torch.jit.is_scripting():
            return output
        return output.refine_names("B", "C", "H", "W")


class StackImage(ImageParameterization):
    """
    Stack multiple NCHW image parameterizations along their batch dimensions.

    Example::

        >>> fft_image_1 = opt.images.FFTImage(size=(224, 224), batch=1)
        >>> fft_image_2 = opt.images.FFTImage(size=(224, 224), batch=1)
        >>> stack_image = opt.images.StackImage([fft_image_1, fft_image_2])
        >>> output_image = stack_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([2, 3, 224, 224])

    Example with ``ImageParameterization`` & ``torch.Tensor``::

        >>> fft_image = opt.images.FFTImage(size=(224, 224), batch=1)
        >>> tensor_image = torch.randn(1, 3, 224, 224)
        >>> stack_image = opt.images.StackImage([fft_image, tensor_image])
        >>> output_image = stack_image()
        >>> print(output_image.required_grad)
        True
        >>> print(output_image.shape)
        torch.Size([2, 3, 224, 224])
    """

    __constants__ = ["dim", "output_device"]

    def __init__(
        self,
        parameterizations: List[Union[ImageParameterization, torch.Tensor]],
        dim: int = 0,
        output_device: Optional[torch.device] = None,
    ) -> None:
        """
        Args:

            parameterizations (list of ImageParameterization and torch.Tensor): A list
                of image parameterizations and tensors to concatenate across a
                specified dimension.
            dim (int, optional): Optionally specify the dim to concatenate
                parameterization outputs on. Default is set to the batch dimension.
                Default: ``0``
            output_device (torch.device, optional): If the parameterizations are on
                different devices, then their outputs will be moved to the device
                specified by this variable. Default is set to ``None`` with the
                expectation that all parameterization outputs are on the same device.
                Default: ``None``
        """
        super().__init__()
        assert len(parameterizations) > 0
        assert isinstance(parameterizations, (list, tuple))
        assert all(
            [
                isinstance(param, (ImageParameterization, torch.Tensor))
                for param in parameterizations
            ]
        )
        parameterizations = [
            SimpleTensorParameterization(p) if isinstance(p, torch.Tensor) else p
            for p in parameterizations
        ]
        self.parameterizations = torch.nn.ModuleList(parameterizations)
        self.dim = dim
        self.output_device = output_device

    def forward(self) -> torch.Tensor:
        """
        Returns:
            image (torch.Tensor): A set of NCHW image parameterization outputs stacked
                along the batch dimension.
        """
        P = []
        for image_param in self.parameterizations:
            img = image_param()
            if self.output_device is not None:
                img = img.to(self.output_device, dtype=img.dtype)
            P.append(img)

        assert P[0].dim() == 4
        assert all([im.shape == P[0].shape for im in P])
        assert all([im.device == P[0].device for im in P])

        image = torch.cat(P, dim=self.dim)
        if torch.jit.is_scripting():
            return image
        return image.refine_names("B", "C", "H", "W")


class NaturalImage(ImageParameterization):
    r"""Outputs an optimizable input image wrapped in :class:`.ImageTensor`.

    By convention, single images are CHW and float32s in [0, 1].
    The underlying parameterization can be decorrelated via a
    :class:`captum.optim.transforms.ToRGB` transform.
    When used with the (default) :class:`.FFTImage` parameterization, this results in
    a fully uncorrelated image parameterization. :-)

    If a model requires a normalization step, such as normalizing imagenet RGB values,
    or rescaling to [0, 255], it can perform those steps with the provided transforms
    or inside its module class.

    Example::

        >>> image = opt.images.NaturalImage(size=(224, 224), channels=3, batch=1)
        >>> image_tensor = image()
        >>> print(image_tensor.required_grad)
        True
        >>> print(image_tensor.shape)
        torch.Size([1, 3, 224, 224])

    Example for using an initialization tensor::

        >>> init = torch.randn(1, 3, 224, 224)
        >>> image = opt.images.NaturalImage(init=init)
        >>> image_tensor = image()
        >>> print(image_tensor.required_grad)
        True
        >>> print(image_tensor.shape)
        torch.Size([1, 3, 224, 224])

    Example for using a parameterization::

        >>> fft_image = opt.images.FFTImage(size=(224, 224), channels=3, batch=1)
        >>> image = opt.images.NaturalImage(parameterization=fft_image)
        >>> image_tensor = image()
        >>> print(image_tensor.required_grad)
        True
        >>> print(image_tensor.shape)
        torch.Size([1, 3, 224, 224])
    """

    def __init__(
        self,
        size: Tuple[int, int] = (224, 224),
        channels: int = 3,
        batch: int = 1,
        init: Optional[torch.Tensor] = None,
        parameterization: ImageParameterization = FFTImage,
        squash_func: Optional[Callable[[torch.Tensor], torch.Tensor]] = torch.sigmoid,
        decorrelation_module: Optional[nn.Module] = ToRGB(transform="klt"),
        decorrelate_init: bool = True,
    ) -> None:
        """
        Args:

            size (tuple of int, optional): The height and width to use for the
                nn.Parameter image tensor, in the format of: (height, width).
                This parameter is not used if the given ``parameterization`` is an
                instance.
                Default: ``(224, 224)``
            channels (int, optional): The number of channels to use when creating the
                nn.Parameter tensor. This parameter is not used if the given
                ``parameterization`` is an instance.
                Default: ``3``
            batch (int, optional): The number of channels to use when creating the
                nn.Parameter tensor. This parameter is not used if the given
                ``parameterization`` is an instance.
                Default: ``1``
            init (torch.Tensor, optional): Optionally specify a tensor to use instead
                of creating one from random noise. This parameter is not used if the
                given ``parameterization`` is an instance. Set to ``None`` for random
                init.
                Default: ``None``
            parameterization (ImageParameterization, optional): An image
                parameterization class, or instance of an image parameterization class.
                Default: :class:`.FFTImage`
            squash_func (callable, optional): The squash function to use after color
                recorrelation. A function, lambda function, or callable class instance.
                Any provided squash function should take a single input tensor and
                return a single output tensor. If set to ``None``, then
                :class:`torch.nn.Identity` will be used to make it a non op.
                Default: :func:`torch.sigmoid`
            decorrelation_module (nn.Module, optional): A module instance that
                recorrelates the colors of an input image. Custom modules can make use
                of the ``decorrelate_init`` parameter by having a second ``inverse``
                parameter in their forward functions that performs the inverse
                operation when it is set to ``True`` (see :class:`.ToRGB` for an
                example). Set to ``None`` for no recorrelation.
                Default: :class:`.ToRGB`
            decorrelate_init (bool, optional): Whether or not to apply color
                decorrelation to the init tensor input. This parameter is not used if
                the given ``parameterization`` is an instance or if init is ``None``.
                Default: ``True``

        Attributes:

            parameterization (ImageParameterization): The given image parameterization
                instance given when initializing ``NaturalImage``.
                Default: :class:`.FFTImage`
            decorrelation_module (torch.nn.Module): The given decorrelation module
                instance given when initializing ``NaturalImage``.
                Default: :class:`.ToRGB`
        """
        super().__init__()
        if not isinstance(parameterization, ImageParameterization):
            # Verify uninitialized class is correct type
            assert issubclass(parameterization, ImageParameterization)
        else:
            assert isinstance(parameterization, ImageParameterization)

        self.decorrelate = decorrelation_module
        if init is not None and not isinstance(parameterization, ImageParameterization):
            assert init.dim() == 3 or init.dim() == 4
            if decorrelate_init and self.decorrelate is not None:
                init = (
                    init.refine_names("B", "C", "H", "W")
                    if init.dim() == 4
                    else init.refine_names("C", "H", "W")
                )
                init = self.decorrelate(init, inverse=True).rename(None)

        self.squash_func = squash_func or torch.nn.Identity()
        if not isinstance(parameterization, ImageParameterization):
            parameterization = parameterization(
                size=size, channels=channels, batch=batch, init=init
            )
        self.parameterization = parameterization

    @torch.jit.ignore
    def _to_image_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrap ImageTensor in torch.jit.ignore for JIT support.

        Args:

            x (torch.Tensor): An input tensor.

        Returns:
            x (ImageTensor): An instance of ``ImageTensor`` with the input tensor.
        """
        return ImageTensor(x)

    def forward(self) -> ImageTensor:
        """
        Returns:
            image_tensor (ImageTensor): The parameterization output wrapped in
                :class:`.ImageTensor`, that has optionally had its colors
                recorrelated.
        """
        image = self.parameterization()
        if self.decorrelate is not None:
            image = self.decorrelate(image)
        image = image.rename(None)  # TODO: the world is not yet ready
        return self._to_image_tensor(self.squash_func(image))


__all__ = [
    "ImageTensor",
    "InputParameterization",
    "ImageParameterization",
    "FFTImage",
    "PixelImage",
    "LaplacianImage",
    "SharedImage",
    "StackImage",
    "NaturalImage",
]
