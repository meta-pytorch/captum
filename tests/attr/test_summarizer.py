#!/usr/bin/env python3

# pyre-strict
import torch
from captum.attr import CommonStats, Summarizer
from captum.testing.helpers import BaseTest


class Test(BaseTest):
    def test_single_input(self) -> None:
        size = (2, 3)
        summarizer = Summarizer(stats=CommonStats())
        for _ in range(10):
            attrs = torch.randn(size)
            summarizer.update(attrs)

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertTrue(isinstance(summ, dict))

        for k in summ:
            self.assertTrue(summ[k].size() == size)

    def test_multi_input(self) -> None:
        size1 = (10, 5, 5)
        size2 = (3, 5)

        summarizer = Summarizer(stats=CommonStats())
        for _ in range(10):
            a1 = torch.randn(size1)
            a2 = torch.randn(size2)
            summarizer.update((a1, a2))

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertTrue(len(summ) == 2)
        self.assertTrue(isinstance(summ[0], dict))
        self.assertTrue(isinstance(summ[1], dict))

        for k in summ[0]:
            self.assertTrue(summ[0][k].size() == size1)
            self.assertTrue(summ[1][k].size() == size2)
