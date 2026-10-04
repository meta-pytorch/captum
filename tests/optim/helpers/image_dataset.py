from typing import List, Tuple

import torch


class ImageTestDataset(torch.utils.data.Dataset):
    """
    Create a simple tensor dataset for testing image dataset classes
    and functions.

    Args:
        tensors (list):  A list of tensors to use in the dataset.
    """

    def __init__(self, tensors: List[torch.Tensor]) -> None:
        assert all(t.size(0) == 1 for t in tensors if t.dim() == 4)

        def t_squeeze(x: torch.Tensor) -> torch.Tensor:
            return x.squeeze(0) if x.dim() == 4 else x

        tensors = [t_squeeze(t) for t in tensors]
        self.tensors = tensors

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.tensors[idx], 0

    def __len__(self) -> int:
        return len(self.tensors)
