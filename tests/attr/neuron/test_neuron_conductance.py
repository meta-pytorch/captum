#!/usr/bin/env python3

# pyre-strict

import unittest
from typing import Any, Callable, cast, List, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.neuron.neuron_conductance import NeuronConductance
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_simple_conductance_input_linear2(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]], requires_grad=True)
        self._conductance_input_test_assert(
            net, net.linear2, inp, (0,), [0.0, 390.0, 0.0]
        )

    def test_simple_conductance_input_linear2_wo_mult_by_inputs(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[100.0, 100.0, 100.0]], requires_grad=True)
        self._conductance_input_test_assert(
            net,
            net.linear2,
            inp,
            (0,),
            [3.96, 3.96, 3.96],
            multiply_by_inputs=False,
        )

    def test_simple_conductance_input_linear1(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_input_test_assert(net, net.linear1, inp, 0, [0.0, 90.0, 0.0])

    def test_simple_conductance_input_linear1_selector_fn(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._conductance_input_test_assert(
            net, net.linear1, inp, lambda x: x[:, 0], [0.0, 90.0, 0.0]
        )

    def test_simple_conductance_input_relu(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[0.0, 70.0, 30.0]], requires_grad=True)
        self._conductance_input_test_assert(net, net.relu, inp, (3,), [0.0, 70.0, 30.0])

    def test_simple_conductance_multi_input_linear2(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 0.0]])
        inp2 = torch.tensor([[0.0, 10.0, 0.0]])
        inp3 = torch.tensor([[0.0, 5.0, 0.0]])
        self._conductance_input_test_assert(
            net,
            net.model.linear2,
            (inp1, inp2, inp3),
            (0,),
            ([[0.0, 156.0, 0.0]], [[0.0, 156.0, 0.0]], [[0.0, 78.0, 0.0]]),
            (4,),
        )

    def test_simple_conductance_multi_input_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0]])
        self._conductance_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (3,),
            ([[0.0, 50.0, 5.0]], [[0.0, 20.0, 25.0]]),
            (inp3, 5),
        )

    def test_simple_conductance_multi_input_batch_relu(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[0.0, 10.0, 1.0], [0.0, 0.0, 10.0]])
        inp2 = torch.tensor([[0.0, 4.0, 5.0], [0.0, 0.0, 10.0]])
        inp3 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        self._conductance_input_test_assert(
            net,
            net.model.relu,
            (inp1, inp2),
            (3,),
            (
                [[0.0, 50.0, 5.0], [0.0, 0.0, 50.0]],
                [[0.0, 20.0, 25.0], [0.0, 0.0, 50.0]],
            ),
            (inp3, 5),
        )

    def test_layer_tuple_selector_fn(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])
        self._conductance_input_test_assert(
            net, net.multi_relu, inp, lambda x: x[0][:, 1], [0.0, 6.0, 0.0]
        )

    def test_matching_conv2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(2, 1, 10, 10)
        self._conductance_input_sum_test_assert(net, net.conv2, inp, 0.0)

        # trying different baseline
        self._conductance_input_sum_test_assert(net, net.conv2, inp, 0.000001)

    def test_matching_relu2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        baseline = 20 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.relu2, inp, baseline)

    def test_matching_relu2_with_scalar_base_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(3, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.relu2, inp, 0.0)

    def test_matching_pool2_multi_input_conductance(self) -> None:
        net = BasicModel_ConvNet()
        inp = 100 * torch.randn(1, 1, 10, 10)
        baseline = 20 * torch.randn(1, 1, 10, 10, requires_grad=True)
        self._conductance_input_sum_test_assert(net, net.pool2, inp, baseline)

    def test_matching_layer_tuple_selector_fn(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])

        lc = LayerConductance(net, net.multi_relu)
        layer_attr = lc.attribute(  # type: ignore[has-type]
            inp, target=0, n_steps=500, method="gausslegendre"
        )
        nc = NeuronConductance(net, net.multi_relu)
        for i in range(len(layer_attr)):
            for j in range(layer_attr[i].shape[1]):
                neuron_attr = nc.attribute(
                    inp,
                    lambda x, i=i, j=j: x[i][:, j],
                    target=0,
                    n_steps=500,
                    method="gausslegendre",
                )
                self.assertAlmostEqual(
                    neuron_attr.sum().item(),
                    layer_attr[i][0][j].item(),
                    delta=0.005,
                )

    def test_relu_neuron_conductance_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )

        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 6.0, 0.0]])

        lc = LayerConductance(net, net.multi_relu)
        layer_attr = lc.attribute(  # type: ignore[has-type]
            inp, target=0, n_steps=500, method="gausslegendre"
        )
        nc = NeuronConductance(net, net.multi_relu)
        for i in range(len(layer_attr)):
            for j in range(layer_attr[i].shape[1]):
                neuron_attr = nc.attribute(
                    inp,
                    lambda x, i=i, j=j: x[i][:, j],
                    target=0,
                    n_steps=500,
                    method="gausslegendre",
                    grad_kwargs={"materialize_grads": True},
                )
                self.assertAlmostEqual(
                    neuron_attr.sum().item(),
                    layer_attr[i][0][j].item(),
                    delta=0.005,
                )

    def _conductance_input_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        test_neuron: Union[int, Tuple[int, ...], Callable],
        expected_input_conductance: Union[List[float], Tuple[List[List[float]], ...]],
        additional_input: Any = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        for internal_batch_size in (None, 5, 20):
            cond = NeuronConductance(
                model,
                target_layer,
                multiply_by_inputs=multiply_by_inputs,
            )
            self.assertEqual(cond.multiplies_by_inputs, multiply_by_inputs)
            attributions = cond.attribute(
                test_input,
                test_neuron,
                target=0,
                n_steps=500,
                method="gausslegendre",
                additional_forward_args=additional_input,
                internal_batch_size=internal_batch_size,
            )
            if isinstance(expected_input_conductance, tuple):
                for i in range(len(expected_input_conductance)):
                    for j in range(len(expected_input_conductance[i])):
                        assertTensorAlmostEqual(
                            self,
                            attributions[i][j : j + 1].squeeze(0),
                            expected_input_conductance[i][j],
                            delta=0.1,
                            mode="max",
                        )
            else:
                if isinstance(attributions, Tensor):
                    assertTensorAlmostEqual(
                        self,
                        attributions.squeeze(0),
                        expected_input_conductance,
                        delta=0.1,
                        mode="max",
                    )
                else:
                    raise AssertionError(
                        "Attributions not returning a Tensor when expected."
                    )

    def _conductance_input_sum_test_assert(
        self,
        model: Module,
        target_layer: Module,
        test_input: TensorOrTupleOfTensorsGeneric,
        test_baseline: BaselineType = None,
    ) -> None:
        layer_cond = LayerConductance(model, target_layer)
        attributions = cast(
            Tensor,
            layer_cond.attribute(  # type: ignore[has-type]
                test_input,
                baselines=test_baseline,
                target=0,
                n_steps=500,
                method="gausslegendre",
            ),
        )
        neuron_cond = NeuronConductance(model, target_layer)
        attr_shape = cast(Tuple[int, ...], attributions.shape)
        for i in range(attr_shape[1]):
            for j in range(attr_shape[2]):
                for k in range(attr_shape[3]):
                    neuron_vals = neuron_cond.attribute(
                        test_input,
                        (i, j, k),
                        baselines=test_baseline,
                        target=0,
                        n_steps=500,
                    )
                    for n in range(attributions.shape[0]):
                        self.assertAlmostEqual(
                            torch.sum(neuron_vals[n]).item(),
                            attributions[n, i, j, k].item(),
                            delta=0.005,
                        )


if __name__ == "__main__":
    unittest.main()
