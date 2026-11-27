from typing import Callable, Optional

import torch
import torch.nn as nn
from captum.optim._param.image.transforms import center_crop
from captum.optim._utils.typing import IntSeqOrIntType, TupleOfTensorsOrTensorType
from captum.optim.models import collect_activations


def extract_expanded_weights(
    model: nn.Module,
    target1: nn.Module,
    target2: nn.Module,
    crop_shape: Optional[IntSeqOrIntType] = None,
    model_input: TupleOfTensorsOrTensorType = torch.zeros(1, 3, 224, 224),
    crop_func: Optional[Callable] = center_crop,
) -> torch.Tensor:
    """
    Extract meaningful weight interactions from between neurons which aren’t
    literally adjacent in a neural network, or where the weights aren’t directly
    represented in a single weight tensor.

    Example::

        >>> # Load InceptionV1 model with nonlinear layers replaced by
        >>> # their linear equivalents
        >>> linear_model = opt.models.googlenet(
        >>>     pretrained=True, use_linear_modules_only=True
        >>> ).eval()
        >>> # Extract weight interactions between target layers
        >>> W_3a_3b = opt.circuits.extract_expanded_weights(
        >>>     linear_model, linear_model.mixed3a, linear_model.mixed3b, 5
        >>> )
        >>> # Display results for channel 147 of mixed3a and channel 379 of
        >>> # mixed3b, in human readable format
        >>> W_3a_3b_hm = opt.weights_to_heatmap_2d(
        >>>     W_3a_3b[379, 147, ...] / W_3a_3b[379, ...].max()
        >>> )
        >>> opt.show(W_3a_3b_hm)

    Voss, et al., "Visualizing Weights", Distill, 2021.
    See: https://distill.pub/2020/circuits/visualizing-weights/

    Args:

        model (nn.Module): The reference to PyTorch model instance.
        target1 (nn.Module): The starting target layer. Must be below the layer
            specified for ``target2``.
        target2 (nn.Module): The end target layer. Must be above the layer
            specified for ``target1``.
        crop_shape (int, list of int, or tuple of int, optional): Specify the exact
            output size to crop out. Set to ``None`` for no cropping.
            Default: ``None``
        model_input (torch.Tensor or tuple of torch.Tensor, optional): The input to use
            with the specified model.
            Default: ``torch.zeros(1, 3, 224, 224)``
        crop_func (Callable, optional): Specify a function to crop away the padding
            from the output weights.
            Default: :func:`.center_crop`

    Returns:
        tensor (torch.Tensor): A tensor containing the expanded weights in the form
            of: (target2 output channels, target1 output channels, height, width)
    """
    if isinstance(model_input, torch.Tensor):
        model_input = model_input.to(next(model.parameters()).device)
    elif isinstance(model_input, tuple):
        model_input = tuple(
            tensor.to(next(model.parameters()).device) for tensor in model_input
        )
    activations = collect_activations(model, [target1, target2], model_input)
    activ1 = activations[target1]
    activ2 = activations[target2]

    if activ2.dim() == 4:
        t_offset_h, t_offset_w = (activ2.size(2) - 1) // 2, (activ2.size(3) - 1) // 2
        t_center = activ2[:, :, t_offset_h, t_offset_w]
    elif activ2.dim() == 2:
        t_center = activ2

    A = []
    for i in range(activ2.size(1)):
        x = torch.autograd.grad(
            outputs=t_center[:, i],
            inputs=[activ1],
            grad_outputs=torch.ones_like(t_center[:, i]),
            retain_graph=True,
        )[0]
        A.append(x)
    expanded_weights = torch.cat(A, 0)

    if crop_shape is not None and crop_func is not None:
        expanded_weights = crop_func(expanded_weights, crop_shape)
    return expanded_weights


__all__ = ["extract_expanded_weights"]
