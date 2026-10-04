# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import tempfile
from typing import List

import torch
import torch.nn as nn
from captum.influence._core.similarity_influence import (
    cosine_similarity,
    euclidean_distance,
    SimilarityInfluence,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from torch import Tensor
from torch.utils.data import Dataset


class BasicLinearNet(nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(num_features, 5, bias=False)
        self.fc1.weight.data.fill_(0.02)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 1, bias=False)
        self.fc2.weight.data.fill_(0.02)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x


class RangeDataset(Dataset):
    def __init__(self, low: int, high: int, num_features: int) -> None:
        self.samples: Tensor = (
            torch.arange(start=low, end=high, dtype=torch.float)
            .repeat(num_features, 1)
            .transpose(1, 0)
        )

    def __len__(self) -> int:
        return len(self.samples)

    # pyrefly: ignore [bad-override-param-name]
    def __getitem__(self, idx: int) -> Tensor:
        return self.samples[idx]


# pyrefly: ignore [invalid-inheritance]
class Test(BaseTest):
    def test_cosine_with_zeros(self) -> None:
        a = torch.cat((torch.zeros((1, 3, 16, 16)), torch.rand((1, 3, 16, 16))))
        b = torch.rand((2, 3, 16, 16))
        similarity = cosine_similarity(a, b)
        self.assertFalse(torch.any(torch.isnan(similarity)))

    def test_correct_influences_standard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_features = 4
            low, high = 0, 16
            batch_size = high // 2
            mymodel = BasicLinearNet(num_features)
            mydata = RangeDataset(low, high, num_features)
            layers = []
            for name, _module in mymodel.named_modules():
                layers.append(name)
            # pyre-fixme[35]: Target cannot be annotated.
            layers: List[str] = list(filter(None, layers))
            testlayers = layers[1:]

            sim = SimilarityInfluence(
                mymodel,
                testlayers,
                mydata,
                tmpdir,
                "linear",
                batch_size=batch_size,
                similarity_metric=euclidean_distance,
                similarity_direction="min",
            )
            inputs = torch.stack((mydata[1], mydata[8], mydata[14]))
            influences = sim.influence(inputs, top_k=3)

            self.assertEqual(len(influences), len(testlayers))
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[1]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[2]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )

    def test_correct_influences_batch_single(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_features = 4
            low, high = 0, 16
            batch_size = 1
            mymodel = BasicLinearNet(num_features)
            mydata = RangeDataset(low, high, num_features)
            layers = []
            for name, _module in mymodel.named_modules():
                layers.append(name)
            # pyre-fixme[35]: Target cannot be annotated.
            layers: List[str] = list(filter(None, layers))
            testlayers = layers[1:]

            sim = SimilarityInfluence(
                mymodel,
                testlayers,
                mydata,
                tmpdir,
                "linear",
                batch_size=batch_size,
                similarity_metric=euclidean_distance,
                similarity_direction="min",
            )
            inputs = torch.stack((mydata[1], mydata[8], mydata[14]))
            influences = sim.influence(inputs, top_k=3)

            self.assertEqual(len(influences), len(testlayers))
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[1]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[2]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )

    def test_correct_influences_batch_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_features = 4
            low, high = 0, 16
            batch_size = 12
            mymodel = BasicLinearNet(num_features)
            mydata = RangeDataset(low, high, num_features)
            layers = []
            for name, _module in mymodel.named_modules():
                layers.append(name)
            # pyre-fixme[35]: Target cannot be annotated.
            layers: List[str] = list(filter(None, layers))
            testlayers = layers[1:]

            sim = SimilarityInfluence(
                mymodel,
                testlayers,
                mydata,
                tmpdir,
                "linear",
                batch_size=batch_size,
                similarity_metric=euclidean_distance,
                similarity_direction="min",
            )
            inputs = torch.stack((mydata[1], mydata[8], mydata[14]))
            influences = sim.influence(inputs, top_k=3)

            self.assertEqual(len(influences), len(testlayers))
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[1]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )
            assertTensorAlmostEqual(
                self,
                torch.sum(influences[layers[2]][0], 1),
                torch.sum(torch.Tensor([[1, 0, 2], [8, 7, 9], [14, 15, 13]]), 1),
            )

    def test_zero_activations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_features = 4
            low, high = 0, 16
            batch_size = high // 2
            mymodel = BasicLinearNet(num_features)
            mydata = RangeDataset(low, high, num_features)
            layers = []
            for name, _module in mymodel.named_modules():
                layers.append(name)
            # pyre-fixme[35]: Target cannot be annotated.
            layers: List[str] = list(filter(None, layers))
            testlayers = layers[1:]

            sim1 = SimilarityInfluence(
                mymodel, testlayers, mydata, tmpdir, "linear", batch_size=batch_size
            )
            inputs = torch.stack((mydata[1], mydata[8], mydata[14]))
            influences = sim1.influence(inputs, top_k=3)
            self.assertEqual(len(influences), len(layers[1:]) + 1)  # zero_acts included
            self.assertTrue("zero_acts-fc2" in influences)
