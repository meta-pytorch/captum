#!/usr/bin/env python3
import operator
import unittest
from typing import Any, List, Type, Union

import captum.optim._core.loss as opt_loss
import torch
from captum.optim.models import collect_activations
from captum.testing.helpers.basic import assertTensorAlmostEqual, BaseTest
from captum.testing.helpers.basic_models import BasicModel_ConvNet_Optim
from packaging import version

CHANNEL_ACTIVATION_0_LOSS = 1.3
CHANNEL_ACTIVATION_1_LOSS = 1.3


def get_loss_value(
    model: torch.nn.Module,
    loss: opt_loss.Loss,
    model_input: Union[List[int], torch.Tensor] = [1, 3, 1, 1],
) -> torch.Tensor:
    """
    Collect target activations and pass them through a composable loss instance.

    Args:

        model (nn.Module): A PyTorch model instance.
        loss (Loss): A composable loss instance that uses targets from the provided
            model instance.
        model_input (list of int or torch.Tensor): A list of integers to use for the
            shape of the model input, or a tensor to use as the model input.
            Default: [1, 3, 1, 1]

    Returns:
        loss (torch.Tensor): The target activations run through the loss objectives.
    """
    if isinstance(model_input, (list, tuple)):
        model_input = torch.ones(*model_input)
    else:
        assert isinstance(model_input, torch.Tensor)
    module_outputs = collect_activations(model, loss.target, model_input)
    return loss(module_outputs).detach()


class TestModuleOP(BaseTest):
    def test_module_op_loss_unary_op(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping ModuleOP unary op test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        composed_loss = opt_loss.module_op(loss, None, operator.neg)

        expected_name = "ChannelActivation"
        self.assertEqual(composed_loss.__name__, expected_name)
        output = get_loss_value(model, composed_loss)
        expected = -torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS]).sum().item()
        self.assertEqual(output, expected)

    def test_module_op_loss_num_add(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping ModuleOP loss add num test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        composed_loss = opt_loss.module_op(loss, 1.0, operator.add)

        expected_name = "ChannelActivation"
        self.assertEqual(composed_loss.__name__, expected_name)
        output = get_loss_value(model, composed_loss)
        expected = torch.tensor([CHANNEL_ACTIVATION_0_LOSS]) + 1.0
        self.assertEqual(output, expected.item())

    def test_module_op_loss_loss_add(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping ModuleOP Loss add Loss test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss1 = opt_loss.ChannelActivation(model.layer, 0)
        loss2 = opt_loss.ChannelActivation(model.layer, 1)
        composed_loss = opt_loss.module_op(loss1, loss2, operator.add)

        expected_name = "Compose(ChannelActivation, ChannelActivation)"
        self.assertEqual(composed_loss.__name__, expected_name)
        output = get_loss_value(model, composed_loss)
        expected = (
            torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_0_LOSS])
            .sum()
            .item()
        )
        self.assertEqual(output, expected)

    def test_module_op_loss_pow_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            loss = opt_loss.ChannelActivation(model.layer, 0)
            opt_loss.module_op(loss, "string", operator.pow)  # type: ignore

    def test_round(self) -> None:
        round_value = 0
        model = BasicModel_ConvNet_Optim()
        loss = round(opt_loss.ChannelActivation(model.layer, 0), round_value)
        expected = torch.round(
            torch.tensor(CHANNEL_ACTIVATION_0_LOSS), decimals=round_value
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), expected.item(), places=6
        )


class TestRModuleOP(BaseTest):
    def test_module_op_loss_num_div(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        composed_loss = opt_loss.rmodule_op(loss, 1.0, operator.pow)

        output = get_loss_value(model, composed_loss)
        self.assertEqual(output, 1.0**CHANNEL_ACTIVATION_0_LOSS)

    def test_rmodule_op_loss_pow_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            loss = opt_loss.ChannelActivation(model.layer, 0)
            opt_loss.rmodule_op(loss, "string", operator.pow)  # type: ignore


class TestDeepDream(BaseTest):
    def test_deepdream(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.DeepDream(model.layer)
        expected = torch.as_tensor(
            [[[CHANNEL_ACTIVATION_0_LOSS**2]], [[CHANNEL_ACTIVATION_1_LOSS**2]]]
        )[None, :]
        assertTensorAlmostEqual(self, get_loss_value(model, loss), expected, mode="max")

    def test_deepdream_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        loss = opt_loss.DeepDream(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(
            self, output, model_input[batch_index : batch_index + 1] ** 2, delta=0.0
        )


class TestLayerActivation(BaseTest):
    def test_layer_activation(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.LayerActivation(model.layer)
        output = get_loss_value(model, loss)
        expected = torch.as_tensor(
            [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS]
        )
        expected = expected[None, :, None, None]

        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            delta = 1.0e-5
        else:
            delta = 0.0
        assertTensorAlmostEqual(self, output, expected, delta=delta)

    def test_layer_activation_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        loss = opt_loss.LayerActivation(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(
            self, output, model_input[batch_index : batch_index + 1], delta=0.0
        )

    def test_layer_activation_batch_index_negative(self) -> None:
        model = torch.nn.Identity()
        batch_index = -2
        loss = opt_loss.LayerActivation(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(
            self, output, model_input[batch_index : batch_index + 1], delta=0.0
        )


class TestChannelActivation(BaseTest):
    def test_channel_activation_init(self) -> None:
        model = torch.nn.Identity()
        channel_index = 5
        loss = opt_loss.ChannelActivation(model, channel_index=channel_index)
        self.assertEqual(loss.channel_index, channel_index)

    def test_channel_activation_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), CHANNEL_ACTIVATION_0_LOSS, places=6
        )

    def test_channel_activation_1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 1)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), CHANNEL_ACTIVATION_1_LOSS, places=6
        )

    def test_channel_index_activation_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        channel_index = 2
        loss = opt_loss.ChannelActivation(
            model, channel_index=channel_index, batch_index=batch_index
        )

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(
            self,
            output,
            model_input[batch_index : batch_index + 1, channel_index],
            delta=0.0,
        )


class TestNeuronActivation(BaseTest):
    def test_neuron_activation_init(self) -> None:
        model = torch.nn.Identity()
        channel_index = 5
        loss = opt_loss.NeuronActivation(model, channel_index=channel_index)
        self.assertEqual(loss.channel_index, channel_index)
        self.assertIsNone(loss.x)
        self.assertIsNone(loss.y)

    def test_neuron_activation_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.NeuronActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), CHANNEL_ACTIVATION_0_LOSS, places=6
        )

    def test_neuron_activation_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        channel_index = 2
        loss = opt_loss.NeuronActivation(
            model, channel_index=channel_index, batch_index=batch_index
        )

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(
            self,
            output,
            model_input[batch_index : batch_index + 1, channel_index, 2:3, 2:3],
            delta=0.0,
        )


class TestTotalVariation(BaseTest):
    def test_total_variation(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.TotalVariation(model.layer)
        self.assertAlmostEqual(get_loss_value(model, loss).item(), 0.0)

    def test_total_variation_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        loss = opt_loss.TotalVariation(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        self.assertEqual(output.item(), 360.0)


class TestL1(BaseTest):
    def test_l1_init(self) -> None:
        model = torch.nn.Identity()
        loss = opt_loss.L1(model)
        self.assertEqual(loss.constant, 0.0)

    def test_l1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.L1(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
            places=6,
        )

    def test_l1_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        loss = opt_loss.L1(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        self.assertEqual(output.item(), 8400.0)


class TestL2(BaseTest):
    def test_l2_init(self) -> None:
        model = torch.nn.Identity()
        loss = opt_loss.L2(model)
        self.assertEqual(loss.constant, 0.0)
        self.assertEqual(loss.eps, 1e-6)

    def test_l2(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.L2(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            (CHANNEL_ACTIVATION_0_LOSS**2 + CHANNEL_ACTIVATION_1_LOSS**2) ** 0.5,
            places=5,
        )

    def test_l2_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        loss = opt_loss.L2(model, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        self.assertAlmostEqual(output.item(), 987.9017944335938, places=3)


class TestDiversity(BaseTest):
    def test_diversity(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.Diversity(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss, model_input=[2, 3, 1, 1]).item(),
            -1,
        )


class TestActivationInterpolation(BaseTest):
    def test_activation_interpolation_0_1(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping Activation Interpolation test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationInterpolation(
            target1=model.layer,
            channel_index1=0,
            target2=model.layer,
            channel_index2=1,
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss, model_input=[2, 3, 1, 1]).item(),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS,
            places=6,
        )


class TestAlignment(BaseTest):
    def test_alignment(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.Alignment(model.layer)
        self.assertAlmostEqual(
            get_loss_value(model, loss, model_input=[2, 3, 1, 1]).item(), 0.0
        )


class TestDirection(BaseTest):
    def test_direction_init(self) -> None:
        model = torch.nn.Identity()
        vec = torch.ones(2) * 0.5
        loss = opt_loss.Direction(model, vec=vec)
        self.assertEqual(list(loss.vec.shape), [1, 2, 1, 1])
        assertTensorAlmostEqual(self, loss.vec, vec.reshape((1, -1, 1, 1)), delta=0.0)
        self.assertEqual(loss.cossim_pow, 0.0)

    def test_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(2)
        loss = opt_loss.Direction(model.layer, vec=torch.ones(2))
        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS])
        dot = torch.sum(vec.reshape((1, -1, 1, 1)) * b.reshape((1, -1, 1, 1)), 1)
        self.assertAlmostEqual(get_loss_value(model, loss).item(), dot.item(), places=6)

    def test_direction_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        vec = torch.tensor([0, 1, 0]).float()
        loss = opt_loss.Direction(model, vec=vec, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)

        expected = torch.tensor(
            [
                [
                    [100.0, 101.0, 102.0, 103.0, 104.0],
                    [105.0, 106.0, 107.0, 108.0, 109.0],
                    [110.0, 111.0, 112.0, 113.0, 114.0],
                    [115.0, 116.0, 117.0, 118.0, 119.0],
                    [120.0, 121.0, 122.0, 123.0, 124.0],
                ]
            ]
        )
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        assertTensorAlmostEqual(self, output, expected, delta=0.0)


class TestNeuronDirection(BaseTest):
    def test_neuron_direction_init(self) -> None:
        model = torch.nn.Identity()
        vec = torch.ones(2) * 0.5
        loss = opt_loss.NeuronDirection(model, vec=vec)
        self.assertIsNone(loss.x)
        self.assertIsNone(loss.y)
        self.assertIsNone(loss.channel_index)
        self.assertEqual(loss.cossim_pow, 0.0)
        self.assertEqual(list(loss.vec.shape), [1, 2, 1, 1])
        assertTensorAlmostEqual(self, loss.vec, vec.reshape((1, -1, 1, 1)), delta=0.0)

    def test_neuron_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(2)
        loss = opt_loss.NeuronDirection(model.layer, vec=vec)
        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS])
        dot = torch.sum(b * vec)
        self.assertAlmostEqual(get_loss_value(model, loss).item(), dot.item(), places=6)

    def test_neuron_direction_channel_index(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(2)
        loss = opt_loss.NeuronDirection(model.layer, vec=vec, channel_index=0)

        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS])
        dot = torch.sum(b * vec)
        self.assertAlmostEqual(get_loss_value(model, loss).item(), dot.item(), places=6)

    def test_neuron_direction_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        vec = torch.tensor([0, 1, 0]).float()
        loss = opt_loss.NeuronDirection(model, vec=vec, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        self.assertEqual(output.item(), 112.0)


class TestAngledNeuronDirection(BaseTest):
    def test_neuron_activation_init(self) -> None:
        model = torch.nn.Identity()
        vec = torch.ones(1, 2) * 0.5
        loss = opt_loss.AngledNeuronDirection(
            model,
            vec=vec,
        )
        self.assertEqual(loss.eps, 1.0e-4)
        self.assertEqual(loss.cossim_pow, 4.0)
        self.assertIsNone(loss.x)
        self.assertIsNone(loss.y)
        self.assertIsNone(loss.vec_whitened)
        assertTensorAlmostEqual(self, loss.vec, vec, delta=0.0)

    def test_angled_neuron_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(1, 2)
        loss = opt_loss.AngledNeuronDirection(model.layer, vec=vec, cossim_pow=0)
        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_0_LOSS])
        dot = torch.sum(b * vec).item()
        output = torch.sum(get_loss_value(model, loss))
        self.assertAlmostEqual(output.item(), dot, places=6)

    def test_angled_neuron_direction_whitened(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(1, 2)
        loss = opt_loss.AngledNeuronDirection(
            model.layer,
            vec=vec,
            vec_whitened=torch.ones(2, 2),
            cossim_pow=0,
        )
        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_0_LOSS])
        dot = torch.sum(vec * b).item() * 2
        output = torch.sum(get_loss_value(model, loss))
        self.assertAlmostEqual(output.item(), dot, places=6)

    def test_angled_neuron_direction_cossim_pow_4(self) -> None:
        model = BasicModel_ConvNet_Optim()
        cossim_pow = 4.0
        vec = torch.ones(1, 2)
        loss = opt_loss.AngledNeuronDirection(
            model.layer, vec=vec, cossim_pow=cossim_pow
        )
        a = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_0_LOSS])[
            None, :
        ]

        dot = torch.mean(a * vec)
        cossims = dot / (1.0e-4 + torch.sqrt(torch.sum(a**2)))
        dot = dot * torch.clamp(cossims, min=0.1) ** cossim_pow

        output = get_loss_value(model, loss).item()
        self.assertAlmostEqual(output, dot.item(), places=6)

    def test_angled_neuron_direction_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        vec = torch.tensor([1, 0, 1]).float()
        loss = opt_loss.AngledNeuronDirection(model, vec=vec, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 3 * 5 * 5).view(5, 3, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))
        self.assertEqual(output.item(), 1.5350958108901978)


class TestTensorDirection(BaseTest):
    def test_tensor_init(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(1, 1, 1, 1)
        loss = opt_loss.TensorDirection(model.layer, vec=vec)
        self.assertEqual(loss.cossim_pow, 0.0)
        assertTensorAlmostEqual(self, loss.vec, vec, delta=0.0)

    def test_tensor_direction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.ones(1, 1, 1, 1)
        loss = opt_loss.TensorDirection(model.layer, vec=vec)
        b = torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS])
        dot = torch.sum(b[None, :, None, None] * vec).item()
        self.assertAlmostEqual(get_loss_value(model, loss).item(), dot, places=6)

    def test_tensor_direction_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 1
        vec = torch.tensor([1, 0, 1, 0]).float().reshape((1, -1, 1, 1))
        loss = opt_loss.TensorDirection(model, vec=vec, batch_index=batch_index)

        model_input = torch.arange(0, 5 * 1 * 5 * 5).view(5, 1, 5, 5).float()
        output = get_loss_value(model, loss, model_input)
        self.assertEqual(output.item(), 74.0)


class TestActivationWeights(BaseTest):
    def test_neuron_activation_init(self) -> None:
        model = torch.nn.Identity()
        weights = torch.zeros(1)
        loss = opt_loss.ActivationWeights(model, weights=weights)
        self.assertIsNone(loss.x)
        self.assertIsNone(loss.y)
        self.assertIsNone(loss.wx)
        self.assertIsNone(loss.wy)
        self.assertFalse(loss.neuron)
        assertTensorAlmostEqual(self, loss.weights, weights, delta=0.0)

    def test_activation_weights_0(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationWeights(model.layer, weights=torch.zeros(1))
        assertTensorAlmostEqual(
            self, get_loss_value(model, loss), torch.zeros(1, 2, 1, 1), mode="max"
        )

    def test_activation_weights_1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationWeights(
            model.layer, weights=torch.ones(1), neuron=True
        )
        assertTensorAlmostEqual(
            self,
            get_loss_value(model, loss),
            [CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS],
            mode="max",
        )

    def test_activation_weights_neuron_1(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ActivationWeights(
            model.layer, weights=torch.ones(1), neuron=True, x=0, y=0, wx=1, wy=1
        )
        assertTensorAlmostEqual(
            self,
            get_loss_value(model, loss),
            torch.as_tensor([CHANNEL_ACTIVATION_0_LOSS, CHANNEL_ACTIVATION_1_LOSS])[
                None, :, None, None
            ],
            mode="max",
        )


class _OverrideAbstractFunctions:
    """
    Context manager for testing classes with abstract functions.

    Examples::
        >>> # Overriding the abstract methods in BaseLoss
        >>> with _OverrideAbstractFunctions(path.to.classtype):
        >>>     # Do stuff with <path.to.classtype>
    """

    def __init__(self, class_type: Type) -> None:
        """
        Args:

            class_type (type): The path to the library class type.
        """
        self.class_type = class_type

    def __enter__(self) -> None:
        self.abstract_methods = self.class_type.__abstractmethods__
        self.class_type.__abstractmethods__ = frozenset()

    def __exit__(self, *args: Any) -> None:
        self.class_type.__abstractmethods__ = self.abstract_methods


class TestLoss(BaseTest):
    def test_loss_init(self) -> None:
        with _OverrideAbstractFunctions(opt_loss.Loss):
            loss = opt_loss.Loss()  # type: ignore
            self.assertIsNone(loss.target)
            self.assertEqual(loss.__name__, "Loss")
            self.assertEqual(opt_loss.Loss.__name__, "Loss")


class TestBaseLoss(BaseTest):
    def test_subclass(self) -> None:
        self.assertTrue(issubclass(opt_loss.BaseLoss, opt_loss.Loss))

    def test_base_loss_init(self) -> None:
        model = torch.nn.Identity()
        with _OverrideAbstractFunctions(opt_loss.BaseLoss):
            loss = opt_loss.BaseLoss(model)  # type: ignore
            self.assertEqual(loss._batch_index, (None, None))
            self.assertEqual(loss.batch_index, (None, None))
            self.assertEqual(loss._target, model)
            self.assertEqual(loss.target, model)
            self.assertEqual(loss.__name__, "BaseLoss")
            self.assertEqual(opt_loss.BaseLoss.__name__, "BaseLoss")

    def test_base_loss_batch_index(self) -> None:
        model = torch.nn.Identity()
        batch_index = 5
        with _OverrideAbstractFunctions(opt_loss.BaseLoss):
            loss = opt_loss.BaseLoss(model, batch_index=batch_index)  # type: ignore
            self.assertEqual(loss._batch_index, (batch_index, batch_index + 1))
            self.assertEqual(loss.batch_index, (batch_index, batch_index + 1))

    def test_base_loss_target_list(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        targets = [model[0], model[1]]
        with _OverrideAbstractFunctions(opt_loss.BaseLoss):
            loss = opt_loss.BaseLoss(targets)  # type: ignore
            self.assertEqual(loss._target, targets)
            self.assertEqual(loss.target, targets)


class TestL2Mean(BaseTest):
    def test_l2mean_init(self) -> None:
        model = torch.nn.Identity()
        loss = opt_loss.L2Mean(model)
        self.assertEqual(loss.constant, 0.5)
        self.assertIsNone(loss.channel_index)

    def test_l2mean_constant(self) -> None:
        model = BasicModel_ConvNet_Optim()
        constant = 0.5
        loss = opt_loss.L2Mean(model.layer, constant=constant)
        output = get_loss_value(model, loss).item()

        expected = (CHANNEL_ACTIVATION_0_LOSS - constant) ** 2
        self.assertAlmostEqual(output, expected, places=6)

    def test_l2mean_channel_index(self) -> None:
        model = BasicModel_ConvNet_Optim()
        constant = 0.0
        loss = opt_loss.L2Mean(model.layer, channel_index=0, constant=constant)
        output = get_loss_value(model, loss).item()

        expected = (CHANNEL_ACTIVATION_0_LOSS - constant) ** 2
        self.assertAlmostEqual(output, expected, places=6)


class TestVectorLoss(BaseTest):
    def test_vectorloss_init(self) -> None:
        model = torch.nn.Identity()
        vec = torch.tensor([0, 1]).float()
        loss = opt_loss.VectorLoss(model, vec=vec)
        assertTensorAlmostEqual(self, loss.vec, vec, delta=0.0)
        self.assertTrue(loss.move_channel_dim_to_final_dim)
        self.assertEqual(loss.activation_fn, torch.nn.functional.relu)

    def test_vectorloss_single_channel(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.tensor([0, 1]).float()
        loss = opt_loss.VectorLoss(model.layer, vec=vec)
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        self.assertAlmostEqual(output, CHANNEL_ACTIVATION_1_LOSS, places=6)

    def test_vectorloss_multiple_channels(self) -> None:
        model = BasicModel_ConvNet_Optim()
        vec = torch.tensor([1, 1]).float()
        loss = opt_loss.VectorLoss(model.layer, vec=vec)
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        self.assertAlmostEqual(output, CHANNEL_ACTIVATION_1_LOSS * 2, places=6)


class TestFacetLoss(BaseTest):
    def test_facetloss_init(self) -> None:
        model = torch.nn.Sequential(torch.nn.Identity(), torch.nn.Identity())
        vec = torch.tensor([0, 1, 0]).float()
        facet_weights = torch.ones([1, 2, 1, 1]) * 1.5
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0],
            vec=vec,
            facet_weights=facet_weights,
        )
        assertTensorAlmostEqual(self, loss.vec, vec, delta=0.0)
        assertTensorAlmostEqual(self, loss.facet_weights, facet_weights, delta=0.0)

    def test_facetloss_single_channel(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore
        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([0, 1, 0]).float()
        facet_weights = torch.ones([1, 2, 6, 6]) * 1.5
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
        )
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        expected = (CHANNEL_ACTIVATION_0_LOSS * 2) * 1.5
        self.assertAlmostEqual(output, expected / 10.0, places=6)

    def test_facetloss_multi_channel(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore

        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([1, 1, 1]).float()
        facet_weights = torch.ones([1, 2, 6, 6]) * 2.0
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
        )
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        self.assertAlmostEqual(output, 1.560000, places=6)

    def test_facetloss_strength(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore
        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([0, 1, 0]).float()
        facet_weights = torch.ones([1, 2, 6, 6]) * 1.5
        strength = 0.5
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
            strength=strength,
        )
        self.assertEqual(loss.strength, strength)
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        self.assertAlmostEqual(output, 0.1950000, places=6)

    def test_facetloss_strength_batch(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore
        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([0, 1, 0]).float()
        facet_weights = torch.ones([1, 2, 6, 6]) * 1.5
        strength = [0.1, 5.05]
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
            strength=strength,
        )
        self.assertEqual(loss.strength, strength)
        output = get_loss_value(model, loss, model_input=[4, 3, 6, 6])
        self.assertAlmostEqual(output, 4.017000198364258, places=6)

    def test_facetloss_2d_weights(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore
        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([0, 1, 0]).float()
        facet_weights = torch.ones([1, 2]) * 1.5
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
        )
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        expected = (CHANNEL_ACTIVATION_0_LOSS * 2) * 1.5
        self.assertAlmostEqual(output, expected / 10.0, places=6)

    def test_facetloss_resize_4d(self) -> None:
        layer = torch.nn.Conv2d(2, 3, 1, bias=True)
        layer.weight.data.fill_(0.1)  # type: ignore
        layer.bias.data.fill_(1)  # type: ignore

        model = torch.nn.Sequential(BasicModel_ConvNet_Optim(), layer)

        vec = torch.tensor([1, 1, 1]).float()
        facet_weights = torch.ones([1, 2, 12, 12]) * 2.0
        loss = opt_loss.FacetLoss(
            ultimate_target=model[1],
            layer_target=model[0].layer,
            vec=vec,
            facet_weights=facet_weights,
        )
        output = get_loss_value(model, loss, model_input=[1, 3, 6, 6]).item()
        self.assertAlmostEqual(output, 1.560000, places=6)


class TestCompositeLoss(BaseTest):
    def test_subclass(self) -> None:
        self.assertTrue(issubclass(opt_loss.CompositeLoss, opt_loss.BaseLoss))

    def test_negative(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = -opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), -CHANNEL_ACTIVATION_0_LOSS, places=6
        )

    def test_addition(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = (
            opt_loss.ChannelActivation(model.layer, 0)
            + opt_loss.ChannelActivation(model.layer, 1)
            + 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS + CHANNEL_ACTIVATION_1_LOSS + 1,
            places=6,
        )

    def test_radd(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = 1.0 + opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS + 1.0,
            places=6,
        )

    def test_subtraction(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = (
            opt_loss.ChannelActivation(model.layer, 0)
            - opt_loss.ChannelActivation(model.layer, 1)
            - 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS - CHANNEL_ACTIVATION_1_LOSS - 1,
        )

    def test_rsub(self) -> None:
        if version.parse(torch.__version__) <= version.parse("1.6.0"):
            raise unittest.SkipTest(
                "Skipping CompositeLoss rsub test due to insufficient Torch"
                + " version."
            )
        model = BasicModel_ConvNet_Optim()
        loss = 1.0 - opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            1.0 - CHANNEL_ACTIVATION_0_LOSS,
        )

    def test_multiplication_loss_type(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) * opt_loss.ChannelActivation(
            model.layer, 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS * CHANNEL_ACTIVATION_0_LOSS,
            places=5,
        )

    def test_multiplication(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) * 10
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), CHANNEL_ACTIVATION_0_LOSS * 10, places=5
        )

    def test_multiplication_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            opt_loss.ChannelActivation(model.layer, 0) * "string"  # type: ignore

    def test_rmul(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = 10 * opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), 10 * CHANNEL_ACTIVATION_0_LOSS, places=5
        )

    def test_rmul_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            "string" * opt_loss.ChannelActivation(model.layer, 0)  # type: ignore

    def test_division_loss_type(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) / opt_loss.ChannelActivation(
            model.layer, 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS / CHANNEL_ACTIVATION_0_LOSS,
        )

    def test_division(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) / 10
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(), CHANNEL_ACTIVATION_0_LOSS / 10
        )

    def test_division_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            opt_loss.ChannelActivation(model.layer, 0) / "string"  # type: ignore

    def test_rdiv(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = 10.0 / opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            10.0 / CHANNEL_ACTIVATION_0_LOSS,
            places=6,
        )

    def test_rdiv_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            "string" / opt_loss.ChannelActivation(model.layer, 0)  # type: ignore

    def test_pow_loss_type(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) ** opt_loss.ChannelActivation(
            model.layer, 1
        )
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS**CHANNEL_ACTIVATION_0_LOSS,
            places=6,
        )

    def test_pow(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = opt_loss.ChannelActivation(model.layer, 0) ** 2
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            CHANNEL_ACTIVATION_0_LOSS**2,
            places=6,
        )

    def test_pow_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            opt_loss.ChannelActivation(model.layer, 0) ** "string"  # type: ignore

    def test_rpow(self) -> None:
        model = BasicModel_ConvNet_Optim()
        loss = 2.0 ** opt_loss.ChannelActivation(model.layer, 0)
        self.assertAlmostEqual(
            get_loss_value(model, loss).item(),
            2.0**CHANNEL_ACTIVATION_0_LOSS,
            places=6,
        )

    def test_rpow_error(self) -> None:
        model = BasicModel_ConvNet_Optim()
        with self.assertRaises(TypeError):
            "string" ** opt_loss.ChannelActivation(model.layer, 0)  # type: ignore


class TestSumLossList(BaseTest):
    def test_sum_loss_list(self) -> None:
        n_batch = 400
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list)
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(out, float(n_batch))

    def test_sum_loss_list_compose_add(self) -> None:
        n_batch = 400
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list) + opt_loss.LayerActivation(model)
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(out, float(n_batch + 1.0))

    def test_sum_loss_list_sum(self) -> None:
        n_batch = 100
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list, torch.sum)
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(out.item(), 30000.0)

    def test_sum_loss_list_identity(self) -> None:
        n_batch = 100
        model = torch.nn.Identity()
        loss_fn_list = [opt_loss.LayerActivation(model) for i in range(n_batch)]
        loss_fn = opt_loss.sum_loss_list(loss_fn_list, torch.nn.Identity())
        out = get_loss_value(model, loss_fn, [n_batch, 3, 1, 1])
        self.assertEqual(list(out.shape), [n_batch, 3, 1, 1])
        self.assertEqual(out.sum().item(), 30000.0)


class TestDefaultLossSummarize(BaseTest):
    def test_default_loss_summarize(self) -> None:
        x = torch.arange(0, 1 * 3 * 5 * 5).view(1, 3, 5, 5).float()
        output = opt_loss.default_loss_summarize(x)
        self.assertEqual(output.item(), -37.0)
