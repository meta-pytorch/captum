import math
from inspect import signature
from typing import cast, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from captum.optim._core.output_hook import ActivationFetcher
from captum.optim._utils.typing import ModuleOutputMapping, TupleOfTensorsOrTensorType


def get_model_layers(model: nn.Module) -> List[str]:
    """
    Return a list of hookable layers for the target model.

    Args:

        model (nn.Module): A PyTorch model or module instance to collect layers from.

    Returns:
        model_layers (list of str): A list of hookable layers in the model.
    """
    layers = []

    def get_layers(net: nn.Module, prefix: str = "") -> None:
        for name, layer in net.named_children():
            delimiter = "." if prefix else ""
            name_str = (
                f"{prefix}[{name}]"
                if str(name).isdigit()
                else f"{prefix}{delimiter}{name}"
            )
            layers.append(name_str)
            get_layers(layer, prefix=name_str)

    get_layers(model)
    return layers


class RedirectedReLU(torch.autograd.Function):
    """
    A workaround when there is no gradient flow from an initial random input.
    ReLU layers will block the gradient flow during backpropagation when their
    input is less than 0. This means that it can be impossible to visualize a
    target without allowing negative values to pass through ReLU layers during
    backpropagation.
    See:
    https://github.com/tensorflow/lucid/blob/master/lucid/misc/redirected_relu_grad.py
    """

    @staticmethod
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.save_for_backward(input_tensor)
        return input_tensor.clamp(min=0)

    @staticmethod
    def backward(self, grad_output: torch.Tensor) -> torch.Tensor:
        (input_tensor,) = self.saved_tensors
        relu_grad = grad_output.clone()
        relu_grad[input_tensor < 0] = 0
        if torch.equal(relu_grad, torch.zeros_like(relu_grad)):
            # Let "wrong" gradients flow if gradient is completely 0
            return grad_output.clone()
        return relu_grad


class RedirectedReluLayer(nn.Module):
    """
    Class for applying RedirectedReLU
    """

    @torch.jit.ignore
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): A tensor to pass through RedirectedReLU.

        Returns:
            x (torch.Tensor): The output of RedirectedReLU.
        """
        return RedirectedReLU.apply(input)


def replace_layers(
    model: nn.Module,
    layer1: Type[nn.Module],
    layer2: Type[nn.Module],
    transfer_vars: bool = False,
    **kwargs,
) -> None:
    """
    Replace all target layers with new layers inside the specified model,
    possibly with the same initialization variables.

    Example::

        >>> model = opt.models.googlenet(pretrained=True)
        >>> # Replace MaxPool2d layers with their AvgPool2d equivalents
        >>> opt.models.replace_layers(model, nn.MaxPool2d, nn.AvgPool2d, True)

    Args:

        model (nn.Module): A PyTorch model instance.
        layer1 (Type[nn.Module]): The layer class that you want to transfer
            initialization variables from.
        layer2 (Type[nn.Module]): The layer class to create with the variables
            from ``layer1``.
        transfer_vars (bool, optional): Whether or not to try and copy
            initialization variables from ``layer1`` instances to the replacement
            ``layer2`` instances.
            Default: ``False``
        kwargs (Any, optional): Any additional variables to use when creating
            the new layer.
    """

    for name, child in model._modules.items():
        if isinstance(child, layer1):
            if transfer_vars:
                new_layer = _transfer_layer_vars(child, layer2, **kwargs)
            else:
                new_layer = layer2(**kwargs)
            setattr(model, name, new_layer)
        elif child is not None:
            replace_layers(child, layer1, layer2, transfer_vars, **kwargs)


def _transfer_layer_vars(
    layer1: nn.Module, layer2: Type[nn.Module], **kwargs
) -> nn.Module:
    """
    Given a layer instance, create a new layer instance of another class
    with the same initialization variables as the original layer.

    Args:

        layer1 (nn.Module): A layer instance that you want to transfer
            initialization variables from.
        layer2 (nn.Module): The layer class to create with the variables
            from of layer1.
        kwargs (Any, optional): Any additional variables to use when creating
            the new layer.

    Returns:
        layer2 instance (nn.Module): An instance of layer2 with the initialization
            variables that it shares with layer1, and any specified additional
            initialization variables.
    """

    l2_vars = list(signature(layer2.__init__).parameters.values())
    l2_vars = [
        str(l2_vars[i]).split()[0]
        for i in range(len(l2_vars))
        if str(l2_vars[i]) != "self"
    ]
    l2_vars = [p.split(":")[0] if ":" in p and "=" not in p else p for p in l2_vars]
    l2_vars = [p.split("=")[0] if "=" in p and ":" not in p else p for p in l2_vars]
    layer2_vars: Dict = {k: [] for k in dict.fromkeys(l2_vars).keys()}

    layer1_vars = {k: v for k, v in vars(layer1).items() if not k.startswith("_")}
    shared_vars = {k: v for k, v in layer1_vars.items() if k in layer2_vars}
    new_vars = dict(item for d in (shared_vars, kwargs) for item in d.items())
    return layer2(**new_vars)


class Conv2dSame(nn.Conv2d):
    """
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions.
    torch.nn.Conv2d with padding='same' can be used when the stride is equal to 1.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        """
        See nn.Conv2d for more details on the possible arguments:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

        Args:

           in_channels (int): The expected number of channels in the input tensor.
           out_channels (int): The desired number of channels in the output tensor.
           kernel_size (int or tuple of int): The desired kernel size to use.
           stride (int or tuple of int, optional): The desired stride for the
               cross-correlation.
               Default: ``1``
           padding (int or tuple of int, optional): This value is always set to 0.
               Default: ``0``
           dilation (int or tuple of int, optional): The desired spacing between the
               kernel points.
               Default: ``1``
           groups (int, optional): Number of blocked connections from input channels
               to output channels. Both in_channels and out_channels must be divisible
               by groups.
               Default: ``1``
           bias (bool, optional): Whether or not to apply a learnable bias to the
               output.
               Default: ``True``
        """
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """
        Calculate the required padding for a dimension.

        Args:

            i (int): The specific size of the tensor dimension requiring padding.
            k (int): The size of the Conv2d weight dimension.
            s (int): The Conv2d stride value for the dimension.
            d (int): The Conv2d dilation value for the dimension.

        Returns:
            padding_vale (int): The calculated padding value.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): The input tensor to apply 2D convolution to.

        Returns
            x (torch.Tensor): The input tensor after the 2D convolution was applied.
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        pad_h = self._calc_same_pad(i=ih, k=kh, s=self.stride[0], d=self.dilation[0])
        pad_w = self._calc_same_pad(i=iw, k=kw, s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


def collect_activations(
    model: nn.Module,
    targets: Union[nn.Module, List[nn.Module]],
    model_input: TupleOfTensorsOrTensorType = torch.zeros(1, 3, 224, 224),
) -> ModuleOutputMapping:
    """
    Collect target activations for a model.

    Example::

        >>> model = opt.models.googlenet(pretrained=True)
        >>> target = model.mixed4c  # Target layer
        >>> activ_dict = opt.models.collect_activations(model, target)
        >>> activations = activ_dict[target]  # Get activations from dict

    Args:

        model (nn.Module): A PyTorch model instance.
        targets (nn.Module or list of nn.Module): One or more layer targets for the
            given model.
        model_input (torch.Tensor or tuple of torch.Tensor, optional): Optionally
            provide an input tensor to use when collecting the target activations.
            Default: ``torch.zeros(1, 3, 224, 224)``

    Returns:
        activ_dict (dict[nn.Module, torch.Tensor]): A dictionary of collected
            activations where the keys are the target layers.
    """
    if not isinstance(targets, (list, tuple)):
        targets = [targets]
    targets = list(dict.fromkeys(targets))
    catch_activ = ActivationFetcher(model, targets)
    activ_dict = catch_activ(model_input)
    return activ_dict


class SkipLayer(torch.nn.Module):
    """
    This layer is made to take the place of any layer that needs to be skipped over
    during the forward pass. Use cases include removing nonlinear activation layers
    like ReLU for circuits research.

    This layer works almost exactly the same way that :class:`torch.nn.Identity` does,
    except it also ignores any additional arguments passed to the forward function.
    Any layer replaced by SkipLayer must have the same input and output shapes.

    See nn.Identity for more details:
    https://pytorch.org/docs/stable/generated/torch.nn.Identity.html
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args:

            args (Any, optional): Any argument. Arguments will be safely ignored.
            kwargs (Any, optional) Any keyword argument. Arguments will be safely
                ignored.
        """
        super().__init__()

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor or tuple of torch.Tensor): The input tensor or tensors.
            args (Any): Any argument. Arguments will be safely ignored.
            kwargs (Any) Any keyword argument. Arguments will be safely ignored.

        Returns:
            x (torch.Tensor): The unmodified input tensor.
        """
        return x


def skip_layers(
    model: nn.Module, layers: Union[List[Type[nn.Module]], Type[nn.Module]]
) -> None:
    """
    This function is a wrapper function for
    replace_layers and replaces the target layer
    with layers that do nothing.
    This is useful for removing the nonlinear ReLU
    layers when creating expanded weights.

    Args:

        model (nn.Module): A PyTorch model instance.
        layers (nn.Module or list of nn.Module): The layer class type to replace in the
            model.
    """
    if not isinstance(layers, (tuple, list)):
        layers = cast(Type[nn.Module], layers)
        replace_layers(model, layers, SkipLayer)
    else:
        layers = cast(List[Type[nn.Module]], layers)
        for target_layer in layers:
            replace_layers(model, target_layer, SkipLayer)


class MaxPool2dRelaxed(torch.nn.Module):
    """
    A relaxed pooling layer, that's useful for calculating attributions of spatial
    positions. Noise in the gradient is reduced by the continuous relaxation of the
    gradient of models using this layer.

    This layer is meant to be combined with forward-mode AD, so that the class
    attributions of spatial posititions can be estimated using the rate at which
    increasing the neuron affects the output classes.

    This layer peforms a :class:`torch.nn.MaxPool2d` operation on the input, while
    using an equivalent :class:`torch.nn.AvgPool2d` layer to compute the gradient.
    This means that the forward pass returns ``nn.MaxPool2d(input)`` while the
    backward pass uses ``nn.AvgPool2d(input)``.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/

    The Lucid equivalent of this class can be found here:
    https://github.com/
    tensorflow/lucid/blob/master/lucid/optvis/overrides/smoothed_maxpool_grad.py

    An additional Lucid reference implementation can be found here:
    https://colab.research.google.com/github/tensorflow/
    lucid/blob/master/notebooks/building-blocks/AttrSpatial.ipynb
    """

    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Optional[Union[int, Tuple[int, int]]] = None,
        padding: Union[int, Tuple[int, int]] = 0,
        ceil_mode: bool = False,
    ) -> None:
        """
        Args:

            kernel_size (int or tuple of int): The size of the window to perform max
                and average pooling with. Either a single int to use for both the
                height & width or a tuple of 2 integers in format of: (height, width).
            stride (int or tuple of int, optional): The stride window size to use.
                Either a single int to use for both the height & width or a tuple of 2
                integers in format of: (height, width).
                Default: ``None``
            padding (int or tuple of int): The amount of zero padding to add to both
                sides in the ``nn.MaxPool2d`` & ``nn.AvgPool2d`` modules. Either a
                single int to use for both the height & width or a tuple of 2 integers
                in format of: (height, width).
                Default: ``0``
            ceil_mode (bool, optional): Whether to use ceil or floor for creating the
                output shape.
                Default: ``False``
        """
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )
        self.avgpool = torch.nn.AvgPool2d(
            kernel_size=kernel_size, stride=stride, padding=padding, ceil_mode=ceil_mode
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to run the pooling operations on.

        Returns:
            x (torch.Tensor): A max pooled x tensor with gradient of an equivalent avg
                pooled tensor.
        """
        return self.maxpool(x.detach()) + self.avgpool(x) - self.avgpool(x.detach())
