#!/usr/bin/env python3
import unittest

import captum.optim._utils.image.atlas as atlas
import torch
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version


class TestNormalizeGrid(BaseTest):
    def test_normalize_grid(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping normalize grid test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()

        xy_grid = atlas.normalize_grid(xy_grid)

        xy_grid_expected = torch.tensor(
            [
                [0.0000, 0.0000],
                [0.1250, 0.1250],
                [0.2500, 0.2500],
                [0.3750, 0.3750],
                [0.5000, 0.5000],
                [0.6250, 0.6250],
                [0.7500, 0.7500],
                [0.8750, 0.8750],
                [1.0000, 1.0000],
            ]
        )

        assertTensorAlmostEqual(self, xy_grid, xy_grid_expected)

    def test_normalize_grid_max_percentile(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping normalize grid test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()

        xy_grid = atlas.normalize_grid(xy_grid, max_percentile=0.85)

        xy_grid_expected = torch.tensor(
            [
                [0.0000, 0.0000],
                [0.1326, 0.1326],
                [0.2653, 0.2653],
                [0.3979, 0.3979],
                [0.5306, 0.5306],
                [0.6632, 0.6632],
                [0.7958, 0.7958],
                [0.9285, 0.9285],
                [1.0000, 1.0000],
            ]
        )

        assertTensorAlmostEqual(self, xy_grid, xy_grid_expected, 0.001)

    def test_normalize_grid_min_percentile(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping normalize grid test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()

        xy_grid = atlas.normalize_grid(xy_grid, min_percentile=0.5)

        xy_grid_expected = torch.tensor(
            [
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0000, 0.0000],
                [0.0893, 0.0893],
                [0.3169, 0.3169],
                [0.5446, 0.5446],
                [0.7723, 0.7723],
                [1.0000, 1.0000],
            ]
        )

        assertTensorAlmostEqual(self, xy_grid, xy_grid_expected, 0.001)

    def test_normalize_grid_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping normalize grid CUDA test due to not supporting CUDA."
            )
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping normalize grid CUDA test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float().cuda()

        xy_grid = atlas.normalize_grid(xy_grid)

        xy_grid_expected = torch.tensor(
            [
                [0.0000, 0.0000],
                [0.1250, 0.1250],
                [0.2500, 0.2500],
                [0.3750, 0.3750],
                [0.5000, 0.5000],
                [0.6250, 0.6250],
                [0.7500, 0.7500],
                [0.8750, 0.8750],
                [1.0000, 1.0000],
            ]
        )

        self.assertTrue(xy_grid.is_cuda)
        assertTensorAlmostEqual(self, xy_grid.cpu(), xy_grid_expected)


class TestCalcGridIndices(BaseTest):
    def test_calc_grid_indices(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping calc grid indices test due to insufficient Torch version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        indices = atlas.calc_grid_indices(xy_grid, grid_size=(2, 2))

        expected_indices = [
            [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([4])],
            [torch.tensor([4]), torch.tensor([4, 5, 6, 7, 8])],
        ]

        for list1, list2 in zip(indices, expected_indices):
            for t1, t2 in zip(list1, list2):
                assertTensorAlmostEqual(self, t1, t2)

    def test_calc_grid_indices_extent(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping calc grid indices extent test due to insufficient Torch"
                + " version."
            )
        xy_grid = torch.arange(0, 2 * 5 * 5).view(5 * 5, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        indices = atlas.calc_grid_indices(
            xy_grid, grid_size=(1, 1), x_extent=(1.0, 2.0), y_extent=(1.0, 2.0)
        )
        assertTensorAlmostEqual(self, indices[0][0], torch.tensor([24]), 0)

    def test_calc_grid_indices_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping calc grid indices CUDA test due to not supporting CUDA."
            )
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping calc grid indices CUDA test due to insufficient Torch"
                + " version."
            )
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float().cuda()
        xy_grid = atlas.normalize_grid(xy_grid)
        indices = atlas.calc_grid_indices(xy_grid, grid_size=(2, 2))

        expected_indices = [
            [torch.tensor([0, 1, 2, 3, 4]), torch.tensor([4])],
            [torch.tensor([4]), torch.tensor([4, 5, 6, 7, 8])],
        ]

        for list1, list2 in zip(indices, expected_indices):
            for t1, t2 in zip(list1, list2):
                self.assertTrue(t1.is_cuda)
                assertTensorAlmostEqual(self, t1.cpu(), t2)


class TestComputeAvgCellSamples(BaseTest):
    def test_compute_avg_cell_samples(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping compute_avg_cell_samples test due to insufficient Torch"
                + " version."
            )
        grid_size = (2, 2)
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        grid_indices = atlas.calc_grid_indices(xy_grid, grid_size=grid_size)

        vecs, vec_coords = atlas.compute_avg_cell_samples(
            grid_indices, raw_activ, grid_size=grid_size, min_density=2
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)

    def test_compute_avg_cell_samples_assertion_error(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping compute_avg_cell_samples assertion test due to insufficient"
                + " Torch version."
            )

        grid_size = (2, 2)
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        xy_grid = atlas.normalize_grid(xy_grid)
        grid_indices = atlas.calc_grid_indices(xy_grid, grid_size=grid_size)

        with self.assertRaises(AssertionError):
            vecs, vec_coords = atlas.compute_avg_cell_samples(
                grid_indices, raw_activ, grid_size=grid_size, min_density=50
            )

    def test_compute_avg_cell_samples_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping compute_avg_cell_samples CUDA test due to not supporting"
                + " CUDA."
            )
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping compute_avg_cell_samples CUDA test due to insufficient"
                + " Torch version."
            )
        grid_size = (2, 2)
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float().cuda()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float().cuda()
        xy_grid = atlas.normalize_grid(xy_grid)
        grid_indices = atlas.calc_grid_indices(xy_grid, grid_size=grid_size)

        vecs, vec_coords = atlas.compute_avg_cell_samples(
            grid_indices, raw_activ, grid_size=grid_size, min_density=2
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        self.assertTrue(vecs.is_cuda)
        assertTensorAlmostEqual(self, vecs.cpu(), expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlasVectors(BaseTest):
    def test_create_atlas_vectors(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas vectors test due to insufficient Torch version."
            )
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float()
        vecs, vec_coords = atlas.create_atlas_vectors(
            xy_grid, raw_activ, grid_size=(2, 2), min_density=2, normalize=True
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)

    def test_create_atlas_vectors_diff_grid_sizes(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas vectors test due to insufficient Torch version."
            )
        grid_size = (3, 2)
        raw_activ = torch.arange(0, 4 * 5 * 4).view(5 * 4, 4).float()
        xy_grid = torch.arange(0, 2 * 5 * 4).view(5 * 4, 2).float()

        vecs, vec_coords = atlas.create_atlas_vectors(
            xy_grid, raw_activ, grid_size=grid_size, min_density=4, normalize=True
        )

        expected_vecs = torch.tensor(
            [[12.0, 13.0, 14.0, 15.0], [64.0, 65.0, 66.0, 67.0]]
        )
        expected_coords = [(0, 0, 7), (2, 1, 7)]

        assertTensorAlmostEqual(self, vecs, expected_vecs)
        self.assertEqual(vec_coords, expected_coords)

    def test_create_atlas_vectors_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping create atlas vectors CUDA test due to not supporting CUDA."
            )
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas vectors CUDA test due to insufficient Torch"
                + " version."
            )
        raw_activ = torch.arange(0, 4 * 3 * 3).view(3 * 3, 4).float().cuda()
        xy_grid = torch.arange(0, 2 * 3 * 3).view(3 * 3, 2).float().cuda()
        vecs, vec_coords = atlas.create_atlas_vectors(
            xy_grid, raw_activ, grid_size=(2, 2), min_density=2, normalize=True
        )

        expected_vecs = torch.tensor([[8.0, 9.0, 10.0, 11.0], [24.0, 25.0, 26.0, 27.0]])
        expected_coords = [(0, 0, 5), (1, 1, 5)]

        self.assertTrue(vecs.is_cuda)
        assertTensorAlmostEqual(self, vecs.cpu(), expected_vecs)
        self.assertEqual(vec_coords, expected_coords)


class TestCreateAtlas(BaseTest):
    def test_create_atlas_square_grid_size(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas canvas test due to insufficient Torch version."
            )
        grid_size = (2, 2)
        img_list = [torch.zeros(1, 3, 4, 4)] * 2
        vec_coords = [(0, 0), (1, 1)]

        atlas_canvas = atlas.create_atlas(img_list, vec_coords, grid_size=grid_size)

        c_pattern = torch.hstack((torch.ones(4, 4), torch.zeros(4, 4)))
        expected_canvas = torch.stack(
            [torch.vstack((c_pattern, c_pattern.flip(1)))] * 3, 0
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, atlas_canvas, expected_canvas, 0)

    def test_create_atlas_tensor_stack(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas canvas tensor stack test due to insufficient"
                + " Torch version."
            )
        grid_size = (2, 2)
        img_stack = torch.stack([torch.zeros(3, 4, 4)] * 2, dim=0)
        vec_coords = [(0, 0), (1, 1)]

        atlas_canvas = atlas.create_atlas(img_stack, vec_coords, grid_size=grid_size)

        c_pattern = torch.hstack((torch.ones(4, 4), torch.zeros(4, 4)))
        expected_canvas = torch.stack(
            [torch.vstack((c_pattern, c_pattern.flip(1)))] * 3, 0
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, atlas_canvas, expected_canvas, 0)

    def test_create_atlas_test_diff_grid_sizes(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas canvas test due to insufficient Torch version."
            )
        grid_size = (3, 2)
        img_list = [torch.zeros(1, 3, 4, 4)] * 2
        vec_coords = [(0, 0), (2, 1)]

        atlas_canvas = atlas.create_atlas(img_list, vec_coords, grid_size=grid_size)

        c_pattern = torch.hstack(
            (torch.ones(4, 4), torch.ones(4, 4), torch.zeros(4, 4))
        )
        expected_canvas = torch.stack(
            [torch.vstack((c_pattern, c_pattern.flip(1)))] * 3, 0
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, atlas_canvas, expected_canvas, 0)

    def test_create_atlas_zeros(self) -> None:
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create atlas canvas test due to insufficient Torch version."
            )
        grid_size = (2, 2)
        img_list = [torch.ones(1, 3, 4, 4)] * 2
        vec_coords = [(0, 0), (1, 1)]

        atlas_canvas = atlas.create_atlas(
            img_list, vec_coords, grid_size=grid_size, base_tensor=torch.zeros
        )

        c_pattern = torch.hstack((torch.zeros(4, 4), torch.ones(4, 4)))
        expected_canvas = torch.stack(
            [torch.vstack((c_pattern, c_pattern.flip(1)))] * 3, 0
        ).unsqueeze(0)
        assertTensorAlmostEqual(self, atlas_canvas, expected_canvas, 0)

    def test_create_atlas_square_grid_size_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping create atlas CUDA test due to not supporting CUDA."
            )
        if version.parse(torch.__version__) < version.parse("1.7.0"):
            raise unittest.SkipTest(
                "Skipping create_atlas canvas CUDA due to insufficient Torch version."
            )
        grid_size = (2, 2)
        img_list = [torch.zeros(1, 3, 4, 4).cuda()] * 2
        vec_coords = [(0, 0), (1, 1)]

        atlas_canvas = atlas.create_atlas(img_list, vec_coords, grid_size=grid_size)

        c_pattern = torch.hstack((torch.ones(4, 4), torch.zeros(4, 4)))
        expected_canvas = torch.stack(
            [torch.vstack((c_pattern, c_pattern.flip(1)))] * 3, 0
        ).unsqueeze(0)

        self.assertTrue(atlas_canvas.is_cuda)
        assertTensorAlmostEqual(self, atlas_canvas.cpu(), expected_canvas, 0)
