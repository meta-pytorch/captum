from typing import Any, Optional, Type
from warnings import warn

import torch
import torch.nn as nn
from captum.optim.models._common import RedirectedReluLayer, SkipLayer

GS_SAVED_WEIGHTS_URL = (
    "https://pytorch.s3.amazonaws.com/models/captum/clip_resnet50x4_image.pt"
)


def clip_resnet50x4_image(
    pretrained: bool = False,
    progress: bool = True,
    model_path: Optional[str] = None,
    **kwargs: Any,
) -> "CLIP_ResNet50x4Image":
    """
    The visual portion of OpenAI's ResNet 50x4 CLIP model from 'Learning Transferable
    Visual Models From Natural Language Supervision': https://arxiv.org/abs/2103.00020

    This model can be combined with the CLIP ResNet 50x4 Text model to create the full
    CLIP ResNet 50x4 model.

    Note that the model was trained on inputs with a shape of: [B, 3, 288, 288].

    Example::

        >>> model = opt.models.clip_resnet50x4_image(pretrained=True)
        >>> output = model(torch.zeros(1, 3, 288, 288))

    See here for more details:
    https://github.com/openai/CLIP
    https://github.com/mlfoundations/open_clip

    Args:

        pretrained (bool, optional): If ``True``, returns a pre-trained model.
            Default: ``False``
        progress (bool, optional): If ``True``, displays a progress bar of the download
            to stderr.
            Default: ``True``
        model_path (str, optional): Optional path for the model file.
            Default: ``None``
        replace_relus_with_redirectedrelu (bool, optional): If ``True``, return
            pretrained model with Redirected ReLU in place of ReLU layers.
            Default: *``True``* when ``pretrained`` is ``True`` otherwise *``False``*
        use_linear_modules_only (bool, optional): If ``True``, return model
            with all nonlinear layers replaced with linear equivalents.
            Default: ``False``
        transform_input (bool, optional): If ``True``, preprocesses the input according
            to the method with which it was trained.
            Default: *``True``* when ``pretrained`` is ``True`` otherwise *``False``*
        use_attnpool (bool, optional): Whether or not to use the final
            ``AttentionPool2d`` layer in the forward function. If set to ``True``,
            model inputs are required to have a shape of: [B, 3, 288, 288] or
            [3, 288, 288].
            Default: ``False``

    Returns:
        model (CLIP_ResNet50x4Image): An instance of a CLIP ResNet 50x4 model's
            image portion.
    """
    if pretrained:
        if "transform_input" not in kwargs:
            kwargs["transform_input"] = True
        if "replace_relus_with_redirectedrelu" not in kwargs:
            kwargs["replace_relus_with_redirectedrelu"] = True
        if "use_linear_modules_only" not in kwargs:
            kwargs["use_linear_modules_only"] = False
        if "use_attnpool" not in kwargs:
            kwargs["use_attnpool"] = False

        model = CLIP_ResNet50x4Image(**kwargs)

        if model_path is None:
            state_dict = torch.hub.load_state_dict_from_url(
                GS_SAVED_WEIGHTS_URL, progress=progress, check_hash=False
            )
        else:
            state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    return CLIP_ResNet50x4Image(**kwargs)


class CLIP_ResNet50x4Image(nn.Module):
    """
    The visual portion of OpenAI's ResNet 50x4 CLIP model from 'Learning Transferable
    Visual Models From Natural Language Supervision': https://arxiv.org/abs/2103.00020
    """

    __constants__ = ["transform_input", "use_attnpool"]

    def __init__(
        self,
        transform_input: bool = False,
        replace_relus_with_redirectedrelu: bool = False,
        use_linear_modules_only: bool = False,
        use_attnpool: bool = True,
    ) -> None:
        """
        Args:

            replace_relus_with_redirectedrelu (bool, optional): If ``True``, return
                model with Redirected ReLU in place of ReLU layers.
                Default: False
            use_linear_modules_only (bool, optional): If ``True``, return model with
                all nonlinear layers replaced with linear equivalents.
                Default: ``False``
            transform_input (bool, optional): If ``True``, preprocesses the input
                according to the method with which it was trained on.
                Default: ``False``
            use_attnpool (bool, optional): Whether or not to use the final
                ``AttentionPool2d`` layer in the forward function. If set to ``True``,
                model inputs are required to have a shape of: [B, 3, 288, 288] or
                [3, 288, 288].
                Default: ``True``
        """
        super().__init__()
        if use_linear_modules_only:
            activ = SkipLayer
        else:
            if replace_relus_with_redirectedrelu:
                activ = RedirectedReluLayer
            else:
                activ = nn.ReLU

        self.transform_input = transform_input
        self.use_attnpool = use_attnpool

        # Stem layers
        self.conv1 = nn.Conv2d(3, 40, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(40)
        self.relu1 = activ()
        self.conv2 = nn.Conv2d(40, 40, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(40)
        self.relu2 = activ()
        self.conv3 = nn.Conv2d(40, 80, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(80)
        self.relu3 = activ()
        self.avgpool = nn.AvgPool2d(2)

        # Residual layers
        self.layer1 = self._build_layer(80, 80, blocks=4, stride=1, activ=activ)
        self.layer2 = self._build_layer(320, 160, blocks=6, stride=2, activ=activ)
        self.layer3 = self._build_layer(640, 320, blocks=10, stride=2, activ=activ)
        self.layer4 = self._build_layer(1280, 640, blocks=6, stride=2, activ=activ)

        # Attention Pooling
        self.attnpool = AttentionPool2d(9, 2560, out_features=640, num_heads=40)

    def _build_layer(
        self,
        inplanes: int = 80,
        planes: int = 80,
        blocks: int = 4,
        stride: int = 1,
        activ: Type[nn.Module] = nn.ReLU,
    ) -> nn.Module:
        """
        Residual layer creation helper function.

        Args:

            inplanes (int, optional): The number of input channels / features to use
                for the first layer.
                Default: ``80``
            planes (int, optional): The number of output channels / features to use
                for the first layer. This variable is then multiplied by 4 to get the
                number of input channels / features to use for the subsequent layers.
                Default: ``80``
            blocks (int, optional): The number of Bottleneck layers to create.
                Default: ``4``
            stride (int, optional): The stride value to use for the Bottleneck layers.
                Default: ``1``
            activ (type of nn.Module, optional): The nn.Module class type to use for
                activation layers.
                Default: ``nn.ReLU``

        Returns:
            residual_layer (nn.Sequential): A full residual layer instance.
        """
        layers = [Bottleneck(inplanes, planes, stride, activ=activ)]
        for _ in range(blocks - 1):
            layers += [Bottleneck(planes * 4, planes, activ=activ)]
        return nn.Sequential(*layers)

    def _transform_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to normalize the values of.

        Returns:
            x (torch.Tensor): A normalized tensor.
        """
        assert x.dim() == 3 or x.dim() == 4
        if self.transform_input:
            if x.min() < 0.0 or x.max() > 1.0:
                warn("Model input has values outside of the range [0, 1].")
            x = x.unsqueeze(0) if x.dim() == 3 else x
            x = x - torch.tensor(
                [0.48145466, 0.4578275, 0.40821073], device=x.device
            ).view(3, 1, 1)
            x = x / torch.tensor(
                [0.26862954, 0.26130258, 0.27577711], device=x.device
            ).view(3, 1, 1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to run through the model.

        Returns:
            x (torch.Tensor): The model output.
        """
        x = self._transform_input(x)

        # Stem layers
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Attention Pooling
        if self.use_attnpool:
            x = self.attnpool(x)
        return x


class Bottleneck(nn.Module):
    def __init__(
        self,
        inplanes: int = 80,
        planes: int = 80,
        stride: int = 1,
        activ: Type[nn.Module] = nn.ReLU,
    ) -> None:
        """
        Args:

            inplanes (int, optional): The number of input channels / features to use
                for the first layer.
                Default: ``80``
            planes (int, optional): The number of output channels / features to use
                for the subsequent layers.
                Default: ``80``
            stride (int, optional): The stride value to use for the Bottleneck layers.
                Default: ``1``
            activ (type of nn.Module, optional): The nn.Module class type to use for
                activation layers.
                Default: ``nn.ReLU``
        """
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = activ()

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = activ()

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu3 = activ()

        if stride > 1 or inplanes != planes * 4:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(inplanes, planes * 4, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * 4),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to run through the module.

        Returns:
            x (torch.Tensor): The module output.
        """
        assert x.dim() == 4
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x.clone()

        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.avgpool(x)

        x = self.bn3(self.conv3(x)) + identity
        x = self.relu3(x)
        return x


class AttentionPool2d(nn.Module):
    def __init__(
        self,
        spacial_size: int = 9,
        in_features: int = 2560,
        out_features: int = 640,
        num_heads: int = 40,
    ) -> None:
        """
        Args:

            spacial_size (int, optional): The desired size to user for the positional
                embedding.
                Default: ``9``
            in_features (int, optional): The desired input size for the nn.Linear
                layers.
                Default: ``2560``
            out_features (int, optional): The desired output size for the nn.Linear
                layers.
                Default: ``640``
            num_heads (int, optional): The number of heads to use.
                Default: ``40``
        """
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_size**2 + 1, in_features) / in_features**0.5
        )
        self.k_proj = nn.Linear(in_features, in_features)
        self.q_proj = nn.Linear(in_features, in_features)
        self.v_proj = nn.Linear(in_features, in_features)
        self.c_proj = nn.Linear(in_features, out_features)
        self.num_heads = num_heads

    @torch.jit.ignore
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:

            x (torch.Tensor): An input tensor to run through the module.

        Returns:
            x (torch.Tensor): The module output.
        """
        assert x.dim() == 4
        x = x.reshape(*x.shape[:2], -1).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :]
        return torch.nn.functional.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]
            ),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )[0][0]
