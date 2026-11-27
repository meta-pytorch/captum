from typing import Type

import torch


def check_layer_in_model(model: torch.nn.Module, layer: Type[torch.nn.Module]) -> bool:
    for _, child in model._modules.items():
        if child is None:
            continue
        if isinstance(child, layer) or check_layer_in_model(child, layer):
            return True
    return False
