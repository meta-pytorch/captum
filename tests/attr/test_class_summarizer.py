#!/usr/bin/env python3

# pyre-strict
from typing import List, Optional, Tuple, Union

import torch
from captum.attr import ClassSummarizer, CommonStats
from captum.testing.helpers import BaseTest
from torch import Tensor


class Test(BaseTest):
    def class_test(
        self,
        data: List[Tuple[Tuple[Tensor, ...], object]],
        classes: List[object],
        x_sizes: Tuple[Tuple[int, ...], ...],
    ) -> None:
        summarizer = ClassSummarizer(stats=CommonStats())
        for x, y in data:
            summarizer.update(x, y)

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertIsInstance(summ, list)
        for s, size in zip(summ, x_sizes):
            self.assertIsInstance(s, dict)
            for key in s:
                self.assertEqual(s[key].size(), size)

        self.assertIsNotNone(summarizer.class_summaries)
        all_classes = torch.zeros(len(classes))
        class_summaries = summarizer.class_summaries
        all_keys = set(class_summaries.keys())
        for i, clazz in enumerate(classes):
            self.assertTrue(clazz in class_summaries)
            all_keys.remove(clazz)
            all_classes[i] = 1

            summ = class_summaries[clazz]
            self.assertIsNotNone(summ)
            self.assertIsInstance(summ, list)

            for s, size in zip(summ, x_sizes):
                self.assertIsInstance(s, dict)
                for key in s:
                    self.assertEqual(s[key].size(), size)

        self.assertEqual(len(all_keys), 0)
        self.assertEqual(all_classes.sum(), len(classes))

    def test_classes(self) -> None:
        sizes_to_test = [
            # ((1,),),
            ((3, 2, 10, 3), (1,)),
            # ((20,),),
        ]
        list_of_classes: List[List[object]] = [
            list(range(100)),
            ["%d" % i for i in range(100)],
            list(range(300, 400)),
        ]
        for batch_size in [None, 1, 4]:
            for sizes, classes in zip(sizes_to_test, list_of_classes):

                def create_batch_labels(
                    batch_idx: int,
                    batch_size: Optional[int] = batch_size,
                    classes: List[object] = classes,
                ) -> Union[object, List[object]]:
                    if batch_size is None:
                        # batch_size = 1
                        return classes[batch_idx]

                    return classes[
                        batch_idx * batch_size : (batch_idx + 1) * batch_size
                    ]

                bs = 1 if batch_size is None else batch_size
                num_batches = len(classes) // bs
                sizes_plus_batch = tuple((bs,) + si for si in sizes)

                data = [
                    (
                        tuple(torch.randn(si) for si in sizes_plus_batch),
                        create_batch_labels(batch_idx),
                    )
                    for batch_idx in range(num_batches)
                ]
                with self.subTest(
                    batch_size=batch_size, sizes=sizes_plus_batch, classes=classes
                ):
                    self.class_test(data, classes, sizes)

    def test_no_class(self) -> None:
        size = (30, 20)
        summarizer = ClassSummarizer(stats=CommonStats())
        for _ in range(10):
            x = torch.randn(size)
            summarizer.update(x)

        summ = summarizer.summary
        self.assertIsNotNone(summ)
        self.assertIsInstance(summ, dict)
        for key in summ:
            self.assertTrue(summ[key].size() == size)

        self.assertIsNotNone(summarizer.class_summaries)
        self.assertIsInstance(summarizer.class_summaries, dict)
        self.assertEqual(len(summarizer.class_summaries), 0)

    def test_single_label(self) -> None:
        size = (4, 3, 2, 1)
        data = torch.randn((100,) + size)

        single_labels = [1, "apple"]

        for label in single_labels:
            summarizer = ClassSummarizer(stats=CommonStats())
            summarizer.update(data, label)
            summ1 = summarizer.summary
            summ2 = summarizer.class_summaries
            self.assertIsNotNone(summ1)
            self.assertIsNotNone(summ2)

            self.assertIsInstance(summ1, list)
            self.assertTrue(len(summ1) == 1)

            self.assertIsInstance(summ2, dict)
            self.assertTrue(label in summ2)
            self.assertTrue(len(summ1) == len(summ2[label]))
            for key in summ1[0].keys():
                self.assertTrue((summ1[0][key] == summ2[label][0][key]).all())
