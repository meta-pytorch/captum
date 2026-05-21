#!/usr/bin/env python3

# pyre-strict

import torch
from captum.concept._utils.classifier import _predict_classes
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_predict_classes_binary_scores(self) -> None:
        scores = torch.tensor([[-2.0], [3.0], [-0.1]])
        classes = torch.tensor([4, 9])

        predictions = _predict_classes(scores, classes)

        self.assertTrue(torch.equal(predictions, torch.tensor([4, 9, 4])))

    def test_predict_classes_multiclass_scores(self) -> None:
        scores = torch.tensor([[0.1, 0.7, 0.2], [5.0, 3.0, 4.0]])
        classes = torch.tensor([2, 4, 6])

        predictions = _predict_classes(scores, classes)

        self.assertTrue(torch.equal(predictions, torch.tensor([4, 2])))
