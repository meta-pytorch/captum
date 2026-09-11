#!/usr/bin/env python3
import unittest

import captum.optim._utils.image.common as common
import torch
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from packaging import version


class TestGetNeuronPos(unittest.TestCase):
    def test_get_neuron_pos_hw(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_xy(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, 5)

        self.assertEqual(x, 5)
        self.assertEqual(y, 5)

    def test_get_neuron_pos_x_none(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, 5, None)

        self.assertEqual(x, 5)
        self.assertEqual(y, H // 2)

    def test_get_neuron_pos_none_y(self) -> None:
        W, H = 128, 128
        x, y = common.get_neuron_pos(H, W, None, 5)

        self.assertEqual(x, W // 2)
        self.assertEqual(y, 5)


class TestDotCossim(BaseTest):
    def test_dot_cossim_cossim_pow_0(self) -> None:
        x = torch.arange(0, 1 * 3 * 4 * 4).view(1, 3, 4, 4).float()
        y = torch.roll(x.clone(), shifts=(1, 2, 2, 1), dims=(0, 1, 2, 3))
        test_output = common._dot_cossim(x, y, cossim_pow=0.0)

        expected_output = torch.tensor(
            [
                [
                    [1040.0, 968.0, 1094.0, 1226.0],
                    [1604.0, 1508.0, 1658.0, 1814.0],
                    [1112.0, 944.0, 1070.0, 1202.0],
                    [1676.0, 1484.0, 1634.0, 1790.0],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output)

    def test_dot_cossim_cossim_pow_4(self) -> None:
        x = torch.arange(0, 1 * 3 * 4 * 4).view(1, 3, 4, 4).float()
        y = torch.roll(x.clone(), shifts=(1, 2, 2, 1), dims=(0, 1, 2, 3))
        test_output = common._dot_cossim(x, y, cossim_pow=4.0)

        expected_output = torch.tensor(
            [
                [
                    [101.9391, 89.0743, 124.8861, 168.7577],
                    [314.2930, 282.3505, 352.6324, 432.1260],
                    [133.2007, 80.3036, 114.3202, 156.4043],
                    [365.9309, 266.5905, 335.3027, 413.3354],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.001)


class TestHueToRGB(BaseTest):
    def test_hue_to_rgb_n_groups_4_warp_true(self) -> None:
        n_groups = 4
        channels = list(range(n_groups))
        test_outputs = []
        for ch in channels:
            output = common.hue_to_rgb(360 * ch / n_groups)
            test_outputs.append(output)
        test_outputs = torch.stack(test_outputs)
        expected_outputs = torch.tensor(
            [
                [1.0000, 0.0000, 0.0000],
                [0.5334, 0.8459, 0.0000],
                [0.0000, 0.7071, 0.7071],
                [0.5334, 0.0000, 0.8459],
            ]
        )
        assertTensorAlmostEqual(self, test_outputs, expected_outputs)

    def test_hue_to_rgb_n_groups_4_warp_false(self) -> None:
        n_groups = 4
        channels = list(range(n_groups))
        test_outputs = []
        for ch in channels:
            output = common.hue_to_rgb(360 * ch / n_groups, warp=False)
            test_outputs.append(output)
        test_outputs = torch.stack(test_outputs)
        expected_outputs = torch.tensor(
            [
                [1.0000, 0.0000, 0.0000],
                [0.3827, 0.9239, 0.0000],
                [0.0000, 0.7071, 0.7071],
                [0.3827, 0.0000, 0.9239],
            ]
        )
        assertTensorAlmostEqual(self, test_outputs, expected_outputs)

    def test_hue_to_rgb_n_groups_3_warp_true(self) -> None:
        n_groups = 3
        channels = list(range(n_groups))
        test_outputs = []
        for ch in channels:
            output = common.hue_to_rgb(360 * ch / n_groups)
            test_outputs.append(output)
        test_outputs = torch.stack(test_outputs)
        expected_outputs = torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )
        assertTensorAlmostEqual(self, test_outputs, expected_outputs, delta=0.0)

    def test_hue_to_rgb_n_groups_2_warp_true(self) -> None:
        n_groups = 2
        channels = list(range(n_groups))
        test_outputs = []
        for ch in channels:
            output = common.hue_to_rgb(360 * ch / n_groups)
            test_outputs.append(output)
        test_outputs = torch.stack(test_outputs)
        expected_outputs = torch.tensor(
            [[1.0000, 0.0000, 0.0000], [0.0000, 0.7071, 0.7071]]
        )
        assertTensorAlmostEqual(self, test_outputs, expected_outputs)

    def test_hue_to_rgb_n_groups_2_warp_false(self) -> None:
        n_groups = 2
        channels = list(range(n_groups))
        test_outputs = []
        for ch in channels:
            output = common.hue_to_rgb(360 * ch / n_groups, warp=False)
            test_outputs.append(output)
        test_outputs = torch.stack(test_outputs)
        expected_outputs = torch.tensor(
            [[1.0000, 0.0000, 0.0000], [0.0000, 0.7071, 0.7071]]
        )
        assertTensorAlmostEqual(self, test_outputs, expected_outputs)


class TestNChannelsToRGB(BaseTest):
    def test_nchannels_to_rgb_collapse(self) -> None:
        test_input = torch.arange(0, 1 * 4 * 4 * 4).view(1, 4, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=True)
        expected_output = torch.tensor(
            [
                [
                    [
                        [30.3782, 31.5489, 32.7147, 33.8773],
                        [35.0379, 36.1975, 37.3568, 38.5163],
                        [39.6765, 40.8378, 42.0003, 43.1642],
                        [44.3296, 45.4967, 46.6655, 47.8360],
                    ],
                    [
                        [31.1266, 32.0951, 33.0678, 34.0451],
                        [35.0270, 36.0137, 37.0051, 38.0011],
                        [39.0015, 40.0063, 41.0152, 42.0282],
                        [43.0449, 44.0654, 45.0894, 46.1167],
                    ],
                    [
                        [41.1375, 41.8876, 42.6646, 43.4656],
                        [44.2882, 45.1304, 45.9901, 46.8658],
                        [47.7561, 48.6597, 49.5754, 50.5023],
                        [51.4394, 52.3859, 53.3411, 54.3044],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_collapse_warp_false(self) -> None:
        test_input = torch.arange(0, 1 * 4 * 4 * 4).view(1, 4, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=False)
        expected_output = torch.tensor(
            [
                [
                    [
                        [27.0349, 28.1947, 29.3453, 30.4887],
                        [31.6266, 32.7605, 33.8914, 35.0201],
                        [36.1474, 37.2737, 38.3995, 39.5252],
                        [40.6511, 41.7772, 42.9039, 44.0312],
                    ],
                    [
                        [31.8525, 32.8600, 33.8708, 34.8851],
                        [35.9034, 36.9257, 37.9522, 38.9828],
                        [40.0175, 41.0561, 42.0987, 43.1451],
                        [44.1951, 45.2486, 46.3054, 47.3655],
                    ],
                    [
                        [42.8781, 43.6494, 44.4480, 45.2710],
                        [46.1162, 46.9813, 47.8644, 48.7640],
                        [49.6786, 50.6069, 51.5477, 52.5000],
                        [53.4629, 54.4355, 55.4172, 56.4071],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_increase(self) -> None:
        test_input = torch.arange(0, 1 * 2 * 4 * 4).view(1, 2, 4, 4).float()
        test_output = common.nchannels_to_rgb(test_input, warp=True)
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0000, 1.8388, 3.4157, 4.8079],
                        [6.0713, 7.2442, 8.3524, 9.4137],
                        [10.4405, 11.4414, 12.4226, 13.3886],
                        [14.3428, 15.2878, 16.2253, 17.1568],
                    ],
                    [
                        [11.3136, 11.9711, 12.5764, 13.1697],
                        [13.7684, 14.3791, 15.0039, 15.6425],
                        [16.2941, 16.9572, 17.6306, 18.3131],
                        [19.0037, 19.7013, 20.4051, 21.1145],
                    ],
                    [
                        [11.3136, 11.9711, 12.5764, 13.1697],
                        [13.7684, 14.3791, 15.0039, 15.6425],
                        [16.2941, 16.9572, 17.6306, 18.3131],
                        [19.0037, 19.7013, 20.4051, 21.1145],
                    ],
                ]
            ]
        )
        assertTensorAlmostEqual(self, test_output, expected_output, delta=0.005)

    def test_nchannels_to_rgb_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping nchannels_to_rgb CUDA test due to not supporting CUDA."
            )
        test_input = torch.randn(1, 6, 224, 224).cuda()
        test_output = common.nchannels_to_rgb(test_input)
        self.assertTrue(test_output.is_cuda)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])

    def test_nchannels_to_rgb_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.8.0"):
            raise unittest.SkipTest(
                "Skipping nchannels_to_rgb JIT module test due to insufficient Torch"
                + " version."
            )
        test_input = torch.randn(1, 6, 224, 224)
        jit_nchannels_to_rgb = torch.jit.script(common.nchannels_to_rgb)
        test_output = jit_nchannels_to_rgb(test_input)
        self.assertEqual(list(test_output.size()), [1, 3, 224, 224])


class TestWeightsToHeatmap2D(BaseTest):
    def test_weights_to_heatmap_2d(self) -> None:
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x)

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)

    def test_weights_to_heatmap_2d_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping weights_to_heatmap_2d CUDA test due to not supporting CUDA."
            )
        x = torch.ones(5, 4)
        x[0:1, 0:4] = x[0:1, 0:4] * 0.2
        x[1:2, 0:4] = x[1:2, 0:4] * 0.8
        x[2:3, 0:4] = x[2:3, 0:4] * 0.0
        x[3:4, 0:4] = x[3:4, 0:4] * -0.2
        x[4:5, 0:4] = x[4:5, 0:4] * -0.8

        x_out = common.weights_to_heatmap_2d(x.cuda())

        x_out_expected = torch.tensor(
            [
                [
                    [0.9639, 0.9639, 0.9639, 0.9639],
                    [0.8580, 0.8580, 0.8580, 0.8580],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8102, 0.8102, 0.8102, 0.8102],
                    [0.2408, 0.2408, 0.2408, 0.2408],
                ],
                [
                    [0.8400, 0.8400, 0.8400, 0.8400],
                    [0.2588, 0.2588, 0.2588, 0.2588],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.8902, 0.8902, 0.8902, 0.8902],
                    [0.5749, 0.5749, 0.5749, 0.5749],
                ],
                [
                    [0.7851, 0.7851, 0.7851, 0.7851],
                    [0.2792, 0.2792, 0.2792, 0.2792],
                    [0.9686, 0.9686, 0.9686, 0.9686],
                    [0.9294, 0.9294, 0.9294, 0.9294],
                    [0.7624, 0.7624, 0.7624, 0.7624],
                ],
            ]
        )
        assertTensorAlmostEqual(self, x_out, x_out_expected, delta=0.01)
        self.assertTrue(x_out.is_cuda)


class TestMakeGridImage(BaseTest):
    def test_make_grid_image_single_tensor(self) -> None:
        test_input = torch.ones(6, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_tensor_list(self) -> None:
        test_input = [torch.ones(1, 1, 2, 2) for i in range(6)]
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_fewer_tiles(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=3, padding=1, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_padding(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=2, pad_value=0.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 10, 10])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value(self) -> None:
        test_input = torch.ones(4, 1, 2, 2)
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest(
                "Skipping make_image_grid CUDA test due to not supporting" + " CUDA."
            )
        test_input = torch.ones(4, 1, 2, 2).cuda()
        test_output = common.make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        self.assertTrue(test_output.is_cuda)
        assertTensorAlmostEqual(self, test_output, expected_output, 0)

    def test_make_grid_image_single_tensor_pad_value_jit_module(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.10.0"):
            raise unittest.SkipTest(
                "Skipping make_image_grid JIT module test due to"
                + "  insufficient Torch version."
            )
        test_input = torch.ones(4, 1, 2, 2)
        jit_make_grid_image = torch.jit.script(common.make_grid_image)
        test_output = jit_make_grid_image(
            test_input, images_per_row=2, padding=1, pad_value=5.0
        )
        expected_output = torch.tensor(
            [
                [
                    [
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5.0],
                        [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                    ]
                ]
            ]
        )
        self.assertEqual(list(expected_output.shape), [1, 1, 7, 7])
        assertTensorAlmostEqual(self, test_output, expected_output, 0)


class TestCreateNewVector(BaseTest):
    def test_create_new_vector_one_hot(self) -> None:
        x = torch.arange(0, 1 * 3 * 5 * 5).view(1, 3, 5, 5).float()
        vec = torch.tensor([0, 1, 0]).float()
        out = common._create_new_vector(x, vec)
        self.assertEqual(out.item(), 37.0)

    def test_create_new_vector_one_hot_batch(self) -> None:
        x = torch.arange(0, 4 * 3 * 5 * 5).view(4, 3, 5, 5).float()
        vec = torch.tensor([0, 1, 0]).float()
        out = common._create_new_vector(x, vec)
        self.assertEqual(out.tolist(), [37.0, 112.0, 187.0, 262.0])

    def test_create_new_vector(self) -> None:
        x = torch.arange(0, 1 * 3 * 5 * 5).view(1, 3, 5, 5).float()
        vec = torch.tensor([1, 1, 1]).float()
        out = common._create_new_vector(x, vec)
        self.assertEqual(out.item(), 111.0)

    def test_create_new_vector_activation_fn(self) -> None:
        x = torch.arange(0, 1 * 3 * 5 * 5).view(1, 3, 5, 5).float()
        x = x - x.mean()
        vec = torch.tensor([1, 0, 1]).float()
        out = common._create_new_vector(x, vec, activation_fn=torch.nn.functional.relu)
        self.assertEqual(out.item(), 25.0)

    def test_create_new_vector_no_activation_fn(self) -> None:
        x = torch.arange(0, 1 * 3 * 5 * 5).view(1, 3, 5, 5).float()
        x = x - x.mean()
        vec = torch.tensor([1, 1, 1]).float()
        out = common._create_new_vector(x, vec, activation_fn=None)
        self.assertEqual(out.item(), 0.0)

    def test_create_new_vector_channels_last(self) -> None:
        x = torch.arange(0, 4 * 5 * 5 * 3).view(4, 5, 5, 3).float()
        vec = torch.tensor([0, 1, 0]).float()
        out = common._create_new_vector(x, vec, move_channel_dim_to_final_dim=False)
        self.assertEqual(out.tolist(), [37.0, 112.0, 187.0, 262.0])

    def test_create_new_vector_dim_2(self) -> None:
        x = torch.arange(0, 1 * 3).view(1, 3).float()
        vec = torch.tensor([0, 1, 0]).float()
        out = common._create_new_vector(x, vec)
        self.assertEqual(list(out.shape), [1, 1])
        self.assertEqual(out.item(), 1.0)
