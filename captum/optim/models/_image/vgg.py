from typing import cast, List, Optional, Type, Union
from warnings import warn

import torch
import torch.nn as nn

from captum.optim.models._common import RedirectedReluLayer, SkipLayer

GS_SAVED_WEIGHTS_URL = (
    "https://pytorch-tutorial-assets.s3.amazonaws.com/captum/vgg16_caffe_features.pth"
)

VGG16_LAYERS: List[Union[int, str]] = (
    [64, 64, "P", 128, 128, "P"]
    + [256] * 3
    + ["P"]
    + list([512] * 3 + ["P"]) * 2  # type: ignore
)


def vgg16(
    pretrained: bool = False,
    progress: bool = True,
    model_path: Optional[str] = None,
    **kwargs,
) -> "VGG":
    r"""
    The VGG-16 model Caffe that the Oxford Visual Geometry Group trained for the
    ImageNet ILSVRC-2014 Challenge.
    https://arxiv.org/abs/1409.1556
    http://www.robots.ox.ac.uk/~vgg/research/very_deep/
    https://gist.github.com/ksimonyan/3785162f95cd2d5fee77#file-readme-md

    Args:

        pretrained (bool, optional): If ``True``, returns a model pre-trained on ImageNet.
            Default: ``False``
        progress (bool, optional): If ``True``, displays a progress bar of the download to
            stderr
            Default: ``True``
        model_path (str, optional): Optional path for VGG model file.
            Default: ``None``
        replace_relus_with_redirectedrelu (bool, optional): If ``True``, return
            pretrained model with :class:`.RedirectedReLU` in place of
            :class:`torch.nn.ReLU` layers.
            Default: *``True``* when pretrained is True otherwise *``False``*
        use_linear_modules_only (bool, optional): If ``True``, return pretrained
            model with all nonlinear layers replaced with linear equivalents.
            Default: ``False``
        out_features (int, optional): Number of output features in the model used for
            training.
            Default: ``1000``
        transform_input (bool, optional): If ``True``, preprocesses the input according
            to the method with which it was trained on ImageNet.
            Default: ``False``
        scale_input (bool, optional): If True and transform_input is True, scale the
            input range from [0, 1] to [0, 255] in the internal preprocessing.
            Default: ``True``
        classifier_logits (bool, optional): If True, adds the classifier component of
            the model.
            Default: *``False``* when pretrained is True otherwise set to *``True``*

        Returns:
            model (nn.Module): A VGG-16 model instance.
    """

    if "layers" not in kwargs:
        kwargs["layers"] = VGG16_LAYERS

    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "scale_input" not in kwargs:
            kwargs["scale_input"] = True
        if "classifier_logits" not in kwargs:
            kwargs["classifier_logits"] = False
        if "replace_relus_with_redirectedrelu" not in kwargs:
            kwargs["replace_relus_with_redirectedrelu"] = False
        if "use_linear_modules_only" not in kwargs:
            kwargs["use_linear_modules_only"] = False
        if "out_features" not in kwargs:
            kwargs["out_features"] = 1000

        model = VGG(**kwargs)

        if model_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
            )
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    return VGG(**kwargs)


class VGG(nn.Module):
    __constants__ = ["transform_input", "scale_input", "classifier_logits"]

    def __init__(
        self,
        layers: List[Union[int, str]] = VGG16_LAYERS,
        out_features: int = 1000,
        transform_input: bool = True,
        scale_input: bool = True,
        classifier_logits: bool = False,
        replace_relus_with_redirectedrelu: bool = False,
        use_linear_modules_only: bool = False,
    ) -> None:
        """
        Args:

            layers (list of int and str): A list of numbers corresponding to layer
                channel sizes, along with the letter 'P' to denote pooling layers.
                Default: ``VGG16_LAYERS``
            out_features (int, optional): Number of output features in the model used
                for training.
                Default: ``1000``
            transform_input (bool, optional): If ``True``, preprocesses the input
                according to the method with which it was trained on ImageNet.
                Default: ``True``
            scale_input (bool, optional): If ``True`` and ``transform_input`` is ``True``, scale
                the input range from [0, 1] to [0, 255] in the internal preprocessing.
                Default: ``True``
            replace_relus_with_redirectedrelu (bool, optional): If ``True``, return
                pretrained model with :class:`.RedirectedReLU` in place of ReLU layers.
                Default: ``False``
            use_linear_modules_only (bool, optional): If ``True``, return pretrained model
                with all nonlinear layers replaced with linear equivalents.
                Default: ``False``
            classifier_logits (bool, optional): If ``True``, adds the classifier component
                of the model.
                Default: ``False``
        """
        super().__init__()
        self.classifier_logits = classifier_logits
        self.transform_input = transform_input
        self.scale_input = scale_input

        if use_linear_modules_only:
            activ = SkipLayer
            pool = nn.AvgPool2d
        else:
            if replace_relus_with_redirectedrelu:
                activ = RedirectedReluLayer
            else:
                activ = nn.ReLU
            pool = nn.MaxPool2d

        self.features = _buildSequential(layers, activ, pool)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        if self.classifier_logits:
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                activ(),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                activ(),
                nn.Dropout(),
                nn.Linear(4096, out_features),
            )

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to normalize and scale the values of.

        Returns:
            x (torch.Tensor): A transformed tensor.
        """
        if self.transform_input:
            assert x.dim() == 3 or x.dim() == 4
            if x.min() < 0.0 or x.max() > 1.0 and self.scale_input:
                warn("Model input has values outside of the range [0, 1].")
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x * 255 if self.scale_input else x
            x = x - torch.tensor([123.68, 116.779, 103.939], device=x.device).view(
                3, 1, 1
            )
            x = x[:, [2, 1, 0]]  # RGB to BGR
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor for the model.

        Returns:
            x (torch.Tensor): The model output tensor.
        """
        x = self._transform_input(x)
        x = self.features(x)
        x = self.avgpool(x)
        if self.classifier_logits:
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x


def _buildSequential(
    channel_list: List[Union[int, str]],
    activ: Type[nn.Module] = nn.ReLU,
    p_layer: Type[nn.Module] = nn.MaxPool2d,
) -> nn.Sequential:
    """
    Build the feature component of VGG models, based on the make_layers helper function
    from: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

    Args:

        channel_list (list of int and str): The list of layer channels and pool layer
            locations to use for creating the feature model.
        activ (Type[nn.Module]): The type of activation layer to use for the feature
            model.
            Default: :class:`torch.nn.ReLU`
        p_layer (Type[nn.Module]): The type of pooling layer to use for the feature
            model.
            Default: :class:`torch.nn.MaxPool2d`

    Returns:
        features (nn.Sequential): The full feature model for a VGG model.
    """
    layers: List[nn.Module] = []
    in_channels: int = 3
    for c in channel_list:
        if c == "P" or c == "p":
            layers += [p_layer(kernel_size=2, stride=2)]
        else:
            c = cast(int, c)
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, activ()]
            in_channels = c
    return nn.Sequential(*layers)
