#!/usr/bin/env python3

# pyre-strict

from typing import Any, Callable, List, Tuple

import torch
from captum.attr._core.feature_permutation import _permute_feature, FeaturePermutation
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual, set_all_random_seeds
from captum.testing.helpers.basic_models import BasicModelWithSparseInputs
from torch import Tensor


class Test(BaseTest):
    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    def construct_future_forward(self, original_forward: Callable) -> Callable:
        def future_forward(*args: Any, **kwargs: Any) -> torch.futures.Future[Tensor]:
            # pyre-fixme[29]: `typing.Type[torch.futures.Future]` is not a function.
            fut: torch.futures.Future[Tensor] = torch.futures.Future()
            fut.set_result(original_forward(*args, **kwargs))
            return fut

        return future_forward

    def _check_features_are_permuted(
        self, inp: Tensor, perm_inp: Tensor, mask: Tensor
    ) -> None:
        permuted_features = mask.expand_as(inp[0])
        unpermuted_features = permuted_features.bitwise_not()

        self.assertTrue(inp.dtype == perm_inp.dtype)
        self.assertTrue(inp.shape == perm_inp.shape)
        self.assertTrue(
            (inp[:, permuted_features] != perm_inp[:, permuted_features]).any()
        )
        self.assertTrue(
            (inp[:, unpermuted_features] == perm_inp[:, unpermuted_features]).all()
        )

    def _check_perm_fn_with_mask(self, inp: Tensor, mask: Tensor) -> None:
        perm_inp = _permute_feature(inp, mask)
        self._check_features_are_permuted(inp, perm_inp, mask)

    def test_perm_fn_single_feature(self) -> None:
        batch_size = 2
        sizes_to_test: List[Tuple[int, ...]] = [(10,), (4, 5), (3, 4, 5)]
        for inp_size in sizes_to_test:
            inp = torch.randn((batch_size,) + inp_size)
            flat_mask = torch.zeros_like(inp[0]).flatten().bool()

            num_features = inp.numel() // batch_size
            for i in range(num_features):
                flat_mask[i] = 1
                self._check_perm_fn_with_mask(inp, flat_mask.view_as(inp[0]))
                flat_mask[i] = 0

    def test_perm_fn_broadcastable_masks(self) -> None:
        batch_size = 5
        inp_size = (3, 20, 30)

        inp = torch.randn((batch_size,) + inp_size)

        # To be broadcastable dimensions have
        # match from end to beginning, by equalling 1 or the dim.
        #
        # If a dimension is missing then it must be the
        # last dim provided (from right to left). The missing
        # dimensions are implied to be = 1
        #
        # Here I write them explicitly for clarity
        mask_sizes: List[Tuple[int, ...]] = [
            # dims = 1
            (1, 20, 30),
            (3, 1, 30),
            (3, 20, 1),
            (1, 1, 30),
            (1, 20, 1),
            # missing
            (1,),  # empty set (all features)
            (30,),
            (20, 30),
            (3, 20, 30),
        ]

        for mask_size in mask_sizes:
            mask = torch.randint(0, 2, mask_size).bool()
            self.assertTrue(mask.shape == mask_size)

            self._check_perm_fn_with_mask(inp, mask)

    def test_single_input(self) -> None:
        batch_size = 2
        input_size = (6,)
        constant_value = 10000

        def forward_func(x: Tensor) -> Tensor:
            return x.sum(dim=-1)

        feature_importance = FeaturePermutation(forward_func=forward_func)

        inp = torch.randn((batch_size,) + input_size)

        inp[:, 0] = constant_value
        zeros = torch.zeros_like(inp[:, 0])
        attribs = feature_importance.attribute(
            inp,
        )
        self.assertTrue(attribs.squeeze(0).size() == (batch_size,) + input_size)
        assertTensorAlmostEqual(self, attribs[:, 0], zeros, delta=0.05, mode="max")
        self.assertTrue((attribs[:, 1 : input_size[0]].abs() > 0).all())

    def test_simple_input_with_min_examples_in_group(self) -> None:
        def forward_func(x: Tensor) -> Tensor:
            return x.sum(dim=-1)

        feature_importance = FeaturePermutation(forward_func=forward_func)
        inp = torch.tensor([[1.0, 2.0]])
        assertTensorAlmostEqual(
            self,
            feature_importance.attribute(inp),
            torch.tensor([[0.0, 0.0]]),
            delta=0.0,
        )
        assertTensorAlmostEqual(
            self,
            feature_importance.attribute(torch.tensor([])),
            torch.tensor([0.0]),
            delta=0.0,
        )

        feature_importance._min_examples_per_batch_grouped = 1
        with self.assertRaises(AssertionError):
            feature_importance.attribute(inp)

    def test_simple_input_custom_mask_with_min_examples_in_group(self) -> None:
        def forward_func(x1: Tensor, x2: Tensor) -> Tensor:
            return x1.sum(dim=-1)

        feature_importance = FeaturePermutation(forward_func=forward_func)
        inp = (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor(([3.0, 4.0], [5.0, 6.0])),
        )
        mask = (
            torch.tensor([0, 0]),
            torch.tensor([[0, 0], [0, 0]]),
        )
        assertTensorAlmostEqual(
            self,
            feature_importance.attribute(inp, feature_mask=mask)[0],
            torch.tensor([[0.0, 0.0]]),
            delta=0.0,
        )

        feature_importance._min_examples_per_batch_grouped = 1
        with self.assertRaises(AssertionError):
            feature_importance.attribute(inp, feature_mask=mask)

    def test_single_input_with_future(
        self,
    ) -> None:
        batch_size = 2
        input_size = (6,)
        constant_value = 10000

        def forward_func(x: Tensor) -> Tensor:
            return x.sum(dim=-1)

        feature_importance = FeaturePermutation(
            forward_func=self.construct_future_forward(forward_func),
        )

        inp = torch.randn((batch_size,) + input_size)

        inp[:, 0] = constant_value
        zeros = torch.zeros_like(inp[:, 0])
        attribs = feature_importance.attribute_future(
            inp,
        )

        self.assertTrue(type(attribs) is torch.Future)
        attribs = attribs.wait()

        self.assertTrue(attribs.squeeze(0).size() == (batch_size,) + input_size)
        assertTensorAlmostEqual(self, attribs[:, 0], zeros, delta=0.05, mode="max")
        self.assertTrue((attribs[:, 1 : input_size[0]].abs() > 0).all())

    def test_multi_input(
        self,
    ) -> None:
        batch_size = 20
        inp1_size = (5, 2)
        inp2_size = (5, 3)

        labels: Tensor = torch.randn(batch_size)

        def forward_func(*x: Tensor) -> Tensor:
            y = torch.zeros(x[0].shape[0:2])
            for xx in x:
                y += xx[:, :, 0] * xx[:, :, 1]
            y = y.sum(dim=-1)

            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            return torch.mean((y - labels) ** 2)

        feature_importance = FeaturePermutation(forward_func=forward_func)

        inp = (
            torch.randn((batch_size,) + inp1_size),
            torch.randn((batch_size,) + inp2_size),
        )

        feature_mask = (
            torch.arange(inp[0][0].numel()).view_as(inp[0][0]).unsqueeze(0),
            torch.arange(inp[0][0].numel(), inp[0][0].numel() + inp[1][0].numel())
            .view_as(inp[1][0])
            .unsqueeze(0),
        )

        inp[1][:, :, 1] = 4
        attribs = feature_importance.attribute(
            inp,
            feature_mask=feature_mask,
        )

        self.assertTrue(isinstance(attribs, tuple))
        self.assertTrue(len(attribs) == 2)

        self.assertTrue(attribs[0].squeeze(0).size() == inp1_size)
        self.assertTrue(attribs[1].squeeze(0).size() == inp2_size)

        self.assertTrue((attribs[1][:, :, 1] == 0).all())
        self.assertTrue((attribs[1][:, :, 2] == 0).all())

        self.assertTrue((attribs[0] != 0).all())
        self.assertTrue((attribs[1][:, :, 0] != 0).all())

    def test_multi_input_group_across_input_tensors(
        self,
    ) -> None:
        batch_size = 20
        inp1_size = (5, 2)
        inp2_size = (5, 3)

        labels: Tensor = torch.randn(batch_size)

        def forward_func(*x: Tensor) -> Tensor:
            y = torch.zeros(x[0].shape[0:2])
            for xx in x:
                y += xx[:, :, 0] * xx[:, :, 1]
            y = y.sum(dim=-1)

            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            return torch.mean((y - labels) ** 2)

        feature_importance = FeaturePermutation(forward_func=forward_func)

        inp = (
            torch.randn((batch_size,) + inp1_size),
            torch.randn((batch_size,) + inp2_size),
        )
        # Group all features together
        feature_mask = tuple(
            torch.zeros_like(inp_tensor[0]).unsqueeze(0) for inp_tensor in inp
        )
        attribs = feature_importance.attribute(inp, feature_mask=feature_mask)

        self.assertTrue(isinstance(attribs, tuple))
        self.assertTrue(len(attribs) == 2)

        self.assertTrue(attribs[0].squeeze(0).size() == inp1_size)
        self.assertTrue(attribs[1].squeeze(0).size() == inp2_size)

        first_elem_first_attrib = attribs[0].flatten()[0]
        first_elem_second_attrib = attribs[1].flatten()[0]
        self.assertTrue(torch.all(attribs[0] == first_elem_first_attrib))
        self.assertTrue(torch.all(attribs[0] == first_elem_second_attrib))
        self.assertEqual(first_elem_first_attrib, first_elem_second_attrib)

    def test_multi_input_with_future(
        self,
    ) -> None:
        batch_size = 20
        inp1_size = (5, 2)
        inp2_size = (5, 3)

        labels: Tensor = torch.randn(batch_size)

        def forward_func(*x: Tensor) -> Tensor:
            y = torch.zeros(x[0].shape[0:2])
            for xx in x:
                y += xx[:, :, 0] * xx[:, :, 1]
            y = y.sum(dim=-1)

            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            return torch.mean((y - labels) ** 2)

        feature_importance = FeaturePermutation(
            forward_func=self.construct_future_forward(forward_func)
        )

        inp = (
            torch.randn((batch_size,) + inp1_size),
            torch.randn((batch_size,) + inp2_size),
        )

        feature_mask = (
            torch.arange(inp[0][0].numel()).view_as(inp[0][0]).unsqueeze(0),
            torch.arange(inp[0][0].numel(), inp[0][0].numel() + inp[1][0].numel())
            .view_as(inp[1][0])
            .unsqueeze(0),
        )

        inp[1][:, :, 1] = 4

        attribs = feature_importance.attribute_future(
            inp,
            feature_mask=feature_mask,
        )
        self.assertTrue(type(attribs) is torch.Future)
        attribs = attribs.wait()

        self.assertTrue(isinstance(attribs, tuple))
        self.assertTrue(len(attribs) == 2)

        self.assertTrue(attribs[0].squeeze(0).size() == inp1_size)
        self.assertTrue(attribs[1].squeeze(0).size() == inp2_size)

        self.assertTrue((attribs[1][:, :, 1] == 0).all())
        self.assertTrue((attribs[1][:, :, 2] == 0).all())

        self.assertTrue((attribs[0] != 0).all())
        self.assertTrue((attribs[1][:, :, 0] != 0).all())

    def test_multiple_perturbations_per_eval(
        self,
    ) -> None:
        perturbations_per_eval = 4
        batch_size = 2
        input_size = (4,)

        inp = torch.randn((batch_size,) + input_size)

        def forward_func(x: Tensor) -> Tensor:
            return 1 - x

        target = 1

        feature_importance = FeaturePermutation(forward_func=forward_func)

        attribs = feature_importance.attribute(
            inp, perturbations_per_eval=perturbations_per_eval, target=target
        )

        self.assertTrue(attribs.size() == (batch_size,) + input_size)

        for i in range(inp.size(1)):
            if i == target:
                continue
            assertTensorAlmostEqual(
                self, attribs[:, i], torch.zeros_like(attribs[:, i])
            )

        y = forward_func(inp)
        actual_diff = torch.stack([(y[0] - y[1])[target], (y[1] - y[0])[target]])
        assertTensorAlmostEqual(self, attribs[:, target], actual_diff)

    def test_multiple_perturbations_per_eval_with_futures(
        self,
    ) -> None:
        perturbations_per_eval = 4
        batch_size = 2
        input_size = (4,)

        inp = torch.randn((batch_size,) + input_size)

        def forward_func(x: Tensor) -> Tensor:
            return 1 - x

        target = 1

        feature_importance = FeaturePermutation(
            forward_func=self.construct_future_forward(forward_func)
        )

        attribs = feature_importance.attribute_future(
            inp,
            perturbations_per_eval=perturbations_per_eval,
            target=target,
        )
        self.assertTrue(type(attribs) is torch.Future)
        attribs = attribs.wait()

        self.assertTrue(attribs.size() == (batch_size,) + input_size)

        for i in range(inp.size(1)):
            if i == target:
                continue
            assertTensorAlmostEqual(
                self, attribs[:, i], torch.zeros_like(attribs[:, i])
            )

        y = forward_func(inp)
        actual_diff = torch.stack([(y[0] - y[1])[target], (y[1] - y[0])[target]])
        assertTensorAlmostEqual(self, attribs[:, target], actual_diff)

    def test_broadcastable_masks(
        self,
    ) -> None:
        # integration test to ensure that
        # permutation function works with custom masks
        def forward_func(x: Tensor) -> Tensor:
            return x.view(x.shape[0], -1).sum(dim=-1)

        batch_size = 2
        inp = torch.randn((batch_size,) + (3, 4, 4))

        feature_importance = FeaturePermutation(forward_func=forward_func)

        masks = [
            torch.tensor([0]),
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[[0, 1, 2, 3], [3, 3, 4, 5], [6, 6, 4, 6], [7, 8, 9, 10]]]),
        ]
        for mask in masks:

            attribs = feature_importance.attribute(
                inp,
                feature_mask=mask,
            )
            self.assertTrue(attribs is not None)
            self.assertTrue(attribs.shape == inp.shape)

            fm = mask.expand_as(inp[0])

            features = set(mask.flatten())
            for feature in features:
                m = (fm == feature).bool()
                attribs_for_feature = attribs[:, m]
                assertTensorAlmostEqual(
                    self,
                    attribs_for_feature[0],
                    -attribs_for_feature[1],
                    delta=0.05,
                    mode="max",
                )

    def test_broadcastable_masks_with_future(
        self,
    ) -> None:
        # integration test to ensure that
        # permutation function works with custom masks
        def forward_func(x: Tensor) -> Tensor:
            return x.view(x.shape[0], -1).sum(dim=-1)

        batch_size = 2
        inp = torch.randn((batch_size,) + (3, 4, 4))

        feature_importance = FeaturePermutation(
            forward_func=self.construct_future_forward(forward_func)
        )

        masks = [
            torch.tensor([0]),
            torch.tensor([[0, 1, 2, 3]]),
            torch.tensor([[[0, 1, 2, 3], [3, 3, 4, 5], [6, 6, 4, 6], [7, 8, 9, 10]]]),
        ]

        results = []
        for mask in masks:
            attribs_future = feature_importance.attribute_future(
                inp,
                feature_mask=mask,
            )
            results.append(attribs_future)
            self.assertTrue(attribs_future is not None)

        for idx in range(len(results)):
            attribs = results[idx].wait()
            self.assertTrue(attribs is not None)
            self.assertTrue(attribs.shape == inp.shape)

            fm = masks[idx].expand_as(inp[0])

            features = set(masks[idx].flatten())
            for feature in features:
                m = (fm == feature).bool()
                attribs_for_feature = attribs[:, m]
                assertTensorAlmostEqual(
                    self,
                    attribs_for_feature[0],
                    -attribs_for_feature[1],
                    delta=0.05,
                    mode="max",
                )

    def test_empty_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        inp2 = torch.tensor([])

        # test empty sparse tensor
        feature_importance = FeaturePermutation(model)
        attr1, attr2 = feature_importance.attribute(
            (inp1, inp2),
        )
        self.assertEqual(attr1.shape, (1, 3))
        self.assertEqual(attr2.shape, (1,))

    def test_sparse_features(self) -> None:
        model = BasicModelWithSparseInputs()
        inp1 = torch.tensor([[1.0, -2.0, 3.0], [2.0, -1.0, 3.0]])
        # Length of sparse index list may not match # of examples
        inp2 = torch.tensor([1, 7, 2, 4, 5, 3, 6])

        feature_importance = FeaturePermutation(model)

        set_all_random_seeds(1234)
        total_attr1, total_attr2 = feature_importance.attribute(
            (inp1, inp2),
        )
        for _ in range(50):
            attr1, attr2 = feature_importance.attribute(
                (inp1, inp2),
            )
            total_attr1 += attr1
            total_attr2 += attr2
        total_attr1 /= 50
        total_attr2 /= 50
        self.assertEqual(total_attr2.shape, (1,))
        assertTensorAlmostEqual(self, total_attr1, torch.zeros_like(total_attr1))
        assertTensorAlmostEqual(self, total_attr2, [-6.0], delta=0.2)
