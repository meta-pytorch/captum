#!/usr/bin/env python3

# pyre-strict
import torch
import torch.nn as nn


class SigmoidModel(nn.Module):
    """
    Model architecture from:
    https://medium.com/coinmonks/create-a-neural-network-in
        -pytorch-and-make-your-life-simpler-ec5367895199
    """

    def __init__(self, num_in: int, num_hidden: int, num_out: int) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lin1 = self.lin1(input)
        lin2 = self.lin2(self.relu1(lin1))
        return self.sigmoid(lin2)


class SoftmaxModel(nn.Module):
    """
    Model architecture from:
    https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    def __init__(
        self, num_in: int, num_hidden: int, num_out: int, inplace: bool = False
    ) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.relu1 = nn.ReLU(inplace=inplace)
        self.relu2 = nn.ReLU(inplace=inplace)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lin1 = self.relu1(self.lin1(input))
        lin2 = self.relu2(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)


class SigmoidDeepLiftModel(nn.Module):
    """
    Model architecture from:
    https://medium.com/coinmonks/create-a-neural-network-in
        -pytorch-and-make-your-life-simpler-ec5367895199
    """

    def __init__(self, num_in: int, num_hidden: int, num_out: int) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden, bias=False)
        self.lin2 = nn.Linear(num_hidden, num_out, bias=False)
        self.lin1.weight = nn.Parameter(torch.ones(num_hidden, num_in))
        self.lin2.weight = nn.Parameter(torch.ones(num_out, num_hidden))
        self.relu1 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lin1 = self.lin1(input)
        lin2 = self.lin2(self.relu1(lin1))
        return self.sigmoid(lin2)


class SoftmaxDeepLiftModel(nn.Module):
    """
    Model architecture from:
    https://adventuresinmachinelearning.com/pytorch-tutorial-deep-learning/
    """

    def __init__(self, num_in: int, num_hidden: int, num_out: int) -> None:
        super().__init__()
        self.num_in = num_in
        self.num_hidden = num_hidden
        self.num_out = num_out
        self.lin1 = nn.Linear(num_in, num_hidden)
        self.lin2 = nn.Linear(num_hidden, num_hidden)
        self.lin3 = nn.Linear(num_hidden, num_out)
        self.lin1.weight = nn.Parameter(torch.ones(num_hidden, num_in))
        self.lin2.weight = nn.Parameter(torch.ones(num_hidden, num_hidden))
        self.lin3.weight = nn.Parameter(torch.ones(num_out, num_hidden))
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        lin1 = self.relu1(self.lin1(input))
        lin2 = self.relu2(self.lin2(lin1))
        lin3 = self.lin3(lin2)
        return self.softmax(lin3)
