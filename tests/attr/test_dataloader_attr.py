#!/usr/bin/env fbpython

# pyre-strict
import math
from typing import Callable, cast, Dict, List, Optional, Tuple, Union
from unittest.mock import Mock, patch

import torch
from captum.attr._core.dataloader_attr import DataLoaderAttribution, InputRole
from captum.attr._core.feature_ablation import FeatureAblation
from captum.testing.helpers.basic import (
    assertAttributionComparision,
    assertTensorAlmostEqual,
    BaseTest,
)
from parameterized import parameterized
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


def sum_forward(*inps: Tensor) -> Tensor:
    flat_inps = [torch.flatten(inp, start_dim=1) for inp in inps]
    return torch.cat(flat_inps, dim=1).sum(1)


class Linear(torch.nn.Module):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n, 1)

    def forward(self, *inps: Tensor) -> Tensor:
        flat_inps = [torch.flatten(inp, start_dim=1) for inp in inps]
        return self.linear(torch.cat(flat_inps, dim=1))


mock_dataset = TensorDataset(
    # iD feature
    torch.tensor(
        [
            [0.0, 0.1],
            [0.3, 0.4],
            [0.6, 0.7],
            [0.9, 1.0],
            [1.2, 1.3],
        ]
    ),
    # 2D feature
    torch.tensor(
        [
            [[0.1, 0.2], [0.3, 0.2]],
            [[0.4, 0.5], [0.3, 0.2]],
            [[0.8, 0.1], [0.2, 0.5]],
            [[1.1, 0.7], [0.1, 0.7]],
            [[0.6, 1.4], [1.2, 0.4]],
        ]
    ),
    # scalar feature or label
    torch.tensor(
        [
            [0],
            [1],
            [0],
            [0],
            [1],
        ]
    ),
)


class Test(BaseTest):
    @parameterized.expand(
        [
            (sum_forward,),
            (Linear(7),),
        ]
    )
    def test_dl_attr(self, forward: Callable[..., Tensor]) -> None:
        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        dataloader = DataLoader(mock_dataset, batch_size=2)

        dl_attributions = dl_fa.attribute(dataloader)

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            batch_attr = fa.attribute(tuple(batch))
            attr_list.append(batch_attr)

        expected_attr = tuple(
            torch.cat(feature_attrs, dim=0) for feature_attrs in zip(*attr_list)
        )

        assertAttributionComparision(self, dl_attributions, expected_attr)

    @parameterized.expand(
        [
            (sum_forward,),
            (Linear(7),),
        ]
    )
    def test_dl_attr_with_mask(self, forward: Callable[..., Tensor]) -> None:
        # FeatureAblation does not support grouping across tensors for now
        # add such test cases after support grouping across tensors in FeatureAblation
        masks = (
            torch.tensor([[0, 0]]),
            torch.tensor([[[1, 2], [3, 2]]]),
            torch.tensor([[4]]),
        )

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        dataloader = DataLoader(mock_dataset, batch_size=2)

        dl_attributions = dl_fa.attribute(dataloader, feature_mask=masks)

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            batch_attr = fa.attribute(tuple(batch), feature_mask=masks)
            attr_list.append(batch_attr)

        expected_attr = tuple(
            torch.cat(feature_attrs, dim=0) for feature_attrs in zip(*attr_list)
        )

        assertAttributionComparision(self, dl_attributions, expected_attr)

    @parameterized.expand(
        [
            (sum_forward,),
            (Linear(7),),
        ]
    )
    def test_dl_attr_with_baseline(self, forward: Callable[..., Tensor]) -> None:
        baselines = (
            torch.tensor([[0, -1]]),
            1,
            0.1,
        )

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        dataloader = DataLoader(mock_dataset, batch_size=2)

        dl_attributions = dl_fa.attribute(dataloader, baselines=baselines)

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            batch_attr = fa.attribute(tuple(batch), baselines=baselines)
            attr_list.append(batch_attr)

        expected_attr = tuple(
            torch.cat(feature_attrs, dim=0) for feature_attrs in zip(*attr_list)
        )

        assertAttributionComparision(self, dl_attributions, expected_attr)

    def test_dl_attr_with_reduce_and_to_metric(self) -> None:
        forward = sum_forward
        func_call_counts: Dict[str, int] = {
            "reduce": 0,
            "to_metric": 0,
        }

        def reduce(
            accum: Optional[Dict[str, Union[int, Tensor]]],
            cur_output: Tensor,
            cur_inputs: Tuple[Tensor, ...],
        ) -> Dict[str, Union[int, Tensor]]:
            func_call_counts["reduce"] += 1

            if accum is None:
                accum = {"sum": cur_output.sum(), "count": len(cur_output)}
            else:
                accum["sum"] = accum["sum"] + cur_output.sum()
                accum["count"] = int(accum["count"]) + len(cur_output)

            return accum

        def to_metric(accum: Dict[str, Union[int, Tensor]]) -> Tensor:
            func_call_counts["to_metric"] += 1

            self.assertEqual(isinstance(accum, dict), True)
            return torch.tensor(
                [
                    cast(Tensor, accum["sum"]) / accum["count"],
                    accum["sum"],
                ]
            )

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        batch_size = 2
        dataloader = DataLoader(mock_dataset, batch_size=batch_size)

        dl_attribution = dl_fa.attribute(
            dataloader,
            reduce=reduce,
            to_metric=to_metric,
            return_input_shape=False,
        )

        n_iters = len(dataloader)

        n_features = 7
        # after support other attr methods, this can be diff from n_features
        n_perturbations = 7
        n_passes = n_perturbations + 1  # +1 for base forward without perturbation
        n_outputs = 2  # [mean, sum]

        self.assertEqual(func_call_counts["reduce"], n_iters * n_passes)
        self.assertEqual(func_call_counts["to_metric"], n_passes)

        expected_attr_shape = (n_outputs, n_features)

        self.assertEqual(type(dl_attribution), Tensor)
        dl_attribution = cast(Tensor, dl_attribution)
        self.assertEqual(dl_attribution.shape, expected_attr_shape)

    @parameterized.expand(
        [
            ([0, 0, 0],),
            ([0, 1, 0],),
            ([0, 1, 1],),
            ([0, 1, 2],),
            ([0, 2, 2],),
        ]
    )
    def test_dl_attr_with_input_roles(self, input_roles: List[int]) -> None:
        n_inputs: int = len(input_roles)
        n_forward_inputs: int = sum(1 for r in input_roles if r != InputRole.no_forward)
        n_attr_inputs: int = sum(1 for r in input_roles if r == InputRole.need_attr)

        def reduce(
            accum: Optional[Tensor], cur_output: Tensor, cur_inputs: Tuple[Tensor, ...]
        ) -> Tensor:
            # all inputs from dataloader should be given to reduce
            self.assertEqual(len(cur_inputs), n_inputs)

            return cur_output if accum is None else torch.cat([accum, cur_output])

        def forward(*forward_inputs: Tensor) -> Tensor:
            # inputs of InputRole.no_forward should not be passed to forward
            self.assertEqual(len(forward_inputs), n_forward_inputs)
            return sum_forward(*forward_inputs)

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        batch_size = 2
        dataloader = DataLoader(mock_dataset, batch_size=batch_size)

        dl_attributions = dl_fa.attribute(
            dataloader,
            input_roles=tuple(input_roles),
            reduce=reduce,
        )

        # only inputs needs
        self.assertEqual(len(dl_attributions), n_attr_inputs)

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            attr_inputs = tuple(
                _ for _, role in zip(batch, input_roles) if role == InputRole.need_attr
            )
            additional_forward_args = tuple(
                _
                for _, role in zip(batch, input_roles)
                if role == InputRole.need_forward
            )

            batch_attr = fa.attribute(
                attr_inputs, additional_forward_args=additional_forward_args
            )
            attr_list.append(batch_attr)

        expected_attr = tuple(
            torch.cat(feature_attrs, dim=0) for feature_attrs in zip(*attr_list)
        )

        assertAttributionComparision(self, dl_attributions, expected_attr)

    def test_dl_attr_not_return_input_shape(self) -> None:
        forward = sum_forward
        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        dataloader = DataLoader(mock_dataset, batch_size=2)

        dl_attribution = dl_fa.attribute(dataloader, return_input_shape=False)

        expected_attr_shape = (len(mock_dataset), 7)

        self.assertEqual(type(dl_attribution), Tensor)
        dl_attribution = cast(Tensor, dl_attribution)
        self.assertEqual(dl_attribution.shape, expected_attr_shape)

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            batch_attr = fa.attribute(tuple(batch))
            attr_list.append(batch_attr)

        expected_attr = torch.cat(
            [
                # flatten feature dim
                torch.cat(feature_attrs, dim=0).flatten(start_dim=1)
                for feature_attrs in zip(*attr_list)
            ],
            dim=1,
        )

        assertTensorAlmostEqual(self, dl_attribution, expected_attr)

    def test_dl_attr_with_mask_not_return_input_shape(self) -> None:
        forward = sum_forward
        masks = (
            torch.tensor([[0, 0]]),
            torch.tensor([[[1, 2], [3, 2]]]),
            torch.tensor([[4]]),
        )

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        dataloader = DataLoader(mock_dataset, batch_size=2)

        dl_attribution = dl_fa.attribute(
            dataloader, feature_mask=masks, return_input_shape=False
        )

        expected_attr_shape = (len(mock_dataset), 5)

        self.assertEqual(type(dl_attribution), Tensor)
        dl_attribution = cast(Tensor, dl_attribution)
        self.assertEqual(dl_attribution.shape, expected_attr_shape)

    @parameterized.expand([(2,), (3,), (4,)])
    def test_dl_attr_with_perturb_per_pass(self, perturb_per_pass: int) -> None:
        forward = sum_forward

        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)

        mock_dl_iter = Mock(wraps=DataLoader.__iter__)

        with patch.object(DataLoader, "__iter__", lambda self: mock_dl_iter(self)):
            dataloader = DataLoader(mock_dataset, batch_size=2)

            dl_attributions = dl_fa.attribute(
                dataloader, perturbations_per_pass=perturb_per_pass
            )

        n_features = 7
        # 2 extra iter calls: get one input for format; get unperturbed output
        n_iter_overhead = 2

        self.assertEqual(
            mock_dl_iter.call_count,
            math.ceil(n_features / perturb_per_pass) + n_iter_overhead,
        )

        # default reduce of DataLoaderAttribution works the same as concat all batches
        attr_list = []
        for batch in dataloader:
            batch_attr = fa.attribute(tuple(batch))
            attr_list.append(batch_attr)

        expected_attr = tuple(
            torch.cat(feature_attrs, dim=0) for feature_attrs in zip(*attr_list)
        )

        assertAttributionComparision(self, dl_attributions, expected_attr)

    def test_futures_not_implemented(self) -> None:
        forward = sum_forward
        fa = FeatureAblation(forward)
        dl_fa = DataLoaderAttribution(fa)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = dl_fa.attribute_future()
        self.assertEqual(attributions, None)
