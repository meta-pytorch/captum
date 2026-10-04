#!/usr/bin/env python3
import captum.optim._utils.image.dataset as dataset_utils
import torch
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from tests.optim.helpers.image_dataset import ImageTestDataset


class TestImageCov(BaseTest):
    def test_image_cov_3_channels(self) -> None:
        test_input = torch.cat(
            [
                torch.ones(1, 1, 4, 4) * 0.1,
                torch.ones(1, 1, 4, 4) * 0.2,
                torch.ones(1, 1, 4, 4) * 0.3,
            ],
            1,
        )

        test_output = dataset_utils.image_cov(test_input)
        expected_output = torch.tensor(
            [
                [
                    [0.0073, 0.0067, 0.0067],
                    [0.0067, 0.0067, 0.0067],
                    [0.0067, 0.0067, 0.0073],
                ]
            ]
        )
        self.assertEqual(list(test_output.shape), [3, 3])
        assertTensorAlmostEqual(self, test_output, expected_output[0], delta=0.001)

    def test_image_cov_3_channels_batch_5(self) -> None:
        test_input = torch.cat(
            [
                torch.ones(5, 1, 4, 4) * 0.1,
                torch.ones(5, 1, 4, 4) * 0.2,
                torch.ones(5, 1, 4, 4) * 0.3,
            ],
            1,
        )

        test_output = dataset_utils.image_cov(test_input)
        expected_output = torch.tensor(
            [
                [0.0365, 0.0333, 0.0335],
                [0.0333, 0.0333, 0.0333],
                [0.0335, 0.0333, 0.0365],
            ]
        )
        self.assertEqual(list(test_output.shape), [3, 3])
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.001)

    def test_image_cov_2_channels(self) -> None:
        test_input = torch.randn(1, 2, 5, 5)
        test_output = dataset_utils.image_cov(test_input)
        self.assertEqual(list(test_output.shape), [2, 2])

    def test_image_cov_4_channels(self) -> None:
        test_input = torch.randn(1, 4, 5, 5)
        test_output = dataset_utils.image_cov(test_input)
        self.assertEqual(list(test_output.shape), [4, 4])


class TestDatasetCovMatrix(BaseTest):
    def test_dataset_cov_matrix(self) -> None:
        num_tensors = 100

        def create_tensor() -> torch.Tensor:
            return torch.cat(
                [
                    torch.ones(1, 224, 224) * 0.9,
                    torch.ones(1, 224, 224) * 0.5,
                    torch.ones(1, 224, 224) * 0.4,
                ],
                0,
            )

        dataset_tensors = [create_tensor() for x in range(num_tensors)]
        test_dataset = ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, num_workers=0, shuffle=False
        )
        output_mtx = dataset_utils.dataset_cov_matrix(dataset_loader)
        expected_mtx = torch.tensor(
            [
                [0.0467, 0.0467, 0.0467],
                [0.0467, 0.0467, 0.0467],
                [0.0467, 0.0467, 0.0467],
            ]
        )
        assertTensorAlmostEqual(self, output_mtx, expected_mtx, delta=0.001)


class TestCovMatrixToKLT(BaseTest):
    def test_cov_matrix_to_klt(self) -> None:
        test_input = torch.tensor(
            [
                [0.0477, 0.0415, 0.0280],
                [0.0415, 0.0425, 0.0333],
                [0.0280, 0.0333, 0.0419],
            ]
        )
        output_mtx = dataset_utils.cov_matrix_to_klt(test_input)
        expected_mtx = torch.tensor(
            [
                [-0.2036, 0.0750, 0.0249],
                [-0.2024, 0.0158, -0.0358],
                [-0.1749, -0.1056, 0.0124],
            ]
        )
        assertTensorAlmostEqual(self, output_mtx, expected_mtx, delta=0.001)


class TestDatasetKLTMatrix(BaseTest):
    def test_dataset_klt_matrix(self) -> None:
        num_tensors = 100

        def create_tensor() -> torch.Tensor:
            return torch.cat(
                [
                    torch.ones(1, 224, 224) * 0.2,
                    torch.ones(1, 224, 224) * 0.9,
                    torch.ones(1, 224, 224) * 0.3,
                ],
                0,
            )

        dataset_tensors = [create_tensor() for x in range(num_tensors)]
        test_dataset = ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, num_workers=0, shuffle=False
        )

        klt_transform = dataset_utils.dataset_klt_matrix(dataset_loader)

        expected_mtx = torch.tensor(
            [
                [-0.3091, 0.0023, 0.0004],
                [-0.3091, -0.0005, -0.0012],
                [-0.3091, -0.0018, 0.0008],
            ]
        )
        assertTensorAlmostEqual(self, klt_transform, expected_mtx, delta=0.001)

    def test_dataset_klt_matrix_randn(self) -> None:
        num_tensors = 100

        def create_tensor() -> torch.Tensor:
            return torch.randn(1, 3, 224, 224).clamp(0, 1)

        dataset_tensors = [create_tensor() for x in range(num_tensors)]
        test_dataset = ImageTestDataset(dataset_tensors)
        dataset_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=10, num_workers=0, shuffle=False
        )

        klt_transform = dataset_utils.dataset_klt_matrix(dataset_loader)
        self.assertEqual(list(klt_transform.shape), [3, 3])
