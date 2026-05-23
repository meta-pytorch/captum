import itertools
import math
import random
import statistics

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from captum.attr._core.rex import ReX

from captum.testing.helpers.basic import BaseTest
from parameterized import parameterized


def visualize_tensor(tensor, cmap="viridis"):
    arr = tensor.detach().cpu().numpy()
    plt.imshow(arr, cmap=cmap)
    plt.colorbar()
    plt.show()


class Test(BaseTest):
    # rename for convenience
    ts = torch.tensor

    depth_opts = range(4, 10)
    n_partition_opts = range(4, 7)
    n_search_opts = range(10, 15)
    assume_locality_opts = [True, False]

    all_options = list(
        itertools.product(
            depth_opts, n_partition_opts, n_search_opts, assume_locality_opts
        )
    )

    def _generate_gaussian_pdf(self, shape, mean):
        k = len(shape)

        cov = 0.1 * torch.eye(k) * statistics.mean(shape)
        dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

        grids = torch.meshgrid(
            *[torch.arange(n, dtype=torch.float64) for n in shape], indexing="ij"
        )
        coords = torch.stack(grids, dim=-1).reshape(-1, k)

        pdf_vals = torch.exp(dist.log_prob(coords))
        return pdf_vals.reshape(*shape)

    @parameterized.expand(
        [
            # inputs:                       baselines:
            (ts([1, 2, 3]), ts([[2, 3], [3, 4]])),
            ((ts([1]), ts([2]), ts([3])), (ts([1]), ts([2]))),
            ((ts([1])), ()),
            ((), ts([1])),
        ]
    )
    def test_input_baseline_mismatch_throws(self, input, baseline):
        rex = ReX(lambda x: 1 / 0)  # dummy forward, should be unreachable
        with self.assertRaises(AssertionError):
            rex.attribute(input, baseline)

    @parameterized.expand(
        [
            (ts([1, 2, 3]), 0),
            (ts([[1, 2, 3], [4, 5, 6]]), 0),
            (ts([1, 2, 3, 4]), ts([0, 0, 0, 0])),
            (ts([[1, 2], [1, 2]]), ts([[0, 0], [0, 0]])),
            (ts([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), 0),
            ((ts([1, 2]), ts([3, 4]), ts([5, 6])), (0, 0, 0)),
            (
                (ts([1, 2]), ts([3, 4]), ts([5, 6])),
                (ts([0, 0]), ts([0, 0]), ts([0, 0])),
            ),
            ((ts([1, 2]), ts([3, 4])), (ts([0, 0]), ts([0, 0]))),
        ]
    )
    def test_valid_input_baseline(self, input, baseline):
        for o in self.all_options:
            rex = ReX(lambda x: True)

            attributions = rex.attribute(input, baseline, *o)[0]

            inp_unwrapped = input
            if isinstance(input, tuple):
                inp_unwrapped = input[0]

            # Forward_func returns a constant, no responsibility in input
            self.assertEqual(torch.sum(attributions), 0)
            self.assertEqual(attributions.size(), inp_unwrapped.size())

    @parameterized.expand(
        [
            # input                                  # selected_idx
            (ts([1, 2, 3]), 0),
            (ts([[1, 2], [3, 4]]), (0, 1)),
            (ts([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]), (0, 1, 0)),
        ]
    )
    def test_selector_function(self, input, idx):
        for o in self.all_options:
            rex = ReX(lambda x: x[idx])

            attributions = rex.attribute(input, 0, *o)[0]
            self.assertEqual(
                attributions[idx], 1, f"expected 1 at {idx} but found {attributions}"
            )

            attributions[idx] = 0
            self.assertEqual(torch.sum(attributions), 0)

    @parameterized.expand(
        [
            # input shape                             # important idx
            ((4, 4), (0, 0)),
            ((12, 12, 12), (1, 2, 1)),
            ((12, 12, 12, 6), (1, 1, 4, 1)),
            ((1920, 1080), (1, 1)),  # image-like
        ]
    )
    def test_selector_function_large_input(self, input_shape, idx):
        rex = ReX(lambda x: x[idx])

        input = torch.ones(*input_shape)
        attributions = rex.attribute(
            input, 0, n_partitions=2, search_depth=10, n_searches=3
        )[0]
        self.assertGreater(attributions[idx], 0)
        attributions[idx] = 0
        self.assertLess(torch.sum(attributions), 1)

    @parameterized.expand(
        [
            # input shape                           # lhs_idx   # rhs_idx
            ((2, 4), (0, 2), (1, 3))
        ]
    )
    def test_boolean_or(self, input_shape, lhs_idx, rhs_idx):
        for o in self.all_options:
            rex = ReX(lambda x: max(x[lhs_idx], x[rhs_idx]))
            input = torch.ones(input_shape)

            attributions = rex.attribute(input, 0, *o)[0]

            self.assertEqual(attributions[lhs_idx], 1.0, f"{attributions}")
            self.assertEqual(attributions[rhs_idx], 1.0, f"{attributions}")

            attributions[lhs_idx] = 0
            attributions[rhs_idx] = 0
            self.assertLess(torch.sum(attributions), 1, f"{attributions}")

    @parameterized.expand(
        [
            # input shape                           # lhs_idx   # rhs_idx
            ((2, 4), (0, 2), (0, 3))
        ]
    )
    def test_boolean_and(self, input_shape, lhs_idx, rhs_idx):
        for i, o in enumerate(self.all_options):
            rex = ReX(lambda x: min(x[lhs_idx], x[rhs_idx]))
            input = torch.ones(input_shape)

            attributions = rex.attribute(input, 0, *o)[0]

            self.assertEqual(attributions[lhs_idx], 0.5, f"{attributions}, {i}, {o}")
            self.assertEqual(attributions[rhs_idx], 0.5, f"{attributions}, {i}, {o}")

            attributions[lhs_idx] = 0
            attributions[rhs_idx] = 0
            self.assertLess(torch.sum(attributions), 1, f"{attributions}")

    @parameterized.expand(
        [
            # shape                         # mean
            ((30, 30),),
            ((50, 50),),
            ((100, 100),),
        ]
    )
    def test_gaussian_recovery(self, shape):
        random.seed()
        eps = 1e-12

        p = torch.zeros(shape)
        for _ in range(3):
            center = self.ts([int(random.random() * dim) for dim in shape])
            p += self._generate_gaussian_pdf(shape, center)

        p += eps
        p = p / torch.sum(p)

        thresh = math.sqrt(torch.mean(p))

        def _forward(inp):
            return 1 if torch.sum(inp) > thresh else 0

        rex = ReX(_forward)
        for b in self.n_partition_opts:
            attributions = rex.attribute(
                p,
                0,
                n_partitions=b,
                search_depth=10,
                n_searches=25,
                assume_locality=True,
            )[0]

            attributions += eps
            attrib_norm = attributions / torch.sum(attributions)

            # visualize_tensor(p)
            # visualize_tensor(attrib_norm)
            # visualize_tensor(p - attrib_norm)

            mid = 0.5 * (p + attrib_norm)
            jsd = 0.5 * F.kl_div(p.log(), mid, reduction="sum") + 0.5 * F.kl_div(
                attrib_norm.log(), mid, reduction="sum"
            )

            self.assertLess(jsd, 0.1)
