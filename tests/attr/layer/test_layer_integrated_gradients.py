#!/usr/bin/env python3

# pyre-strict
import unittest
from typing import Any, cast, List, Tuple, Union

import torch
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._models.base import (
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
)
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicEmbeddingModel,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_TrueMultiInput,
)
from packaging import version
from torch import Tensor
from torch.nn import Module


class Test(BaseTest):
    def test_compare_with_emb_patching(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1]])
        input3 = torch.tensor([[2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_emb_patching_wo_mult_by_inputs(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1]])
        input3 = torch.tensor([[2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1,
            baseline1,
            additional_args=(input2, input3),
            multiply_by_inputs=False,
        )

    def test_compare_with_emb_patching_batch(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1], [3, 1, 1, 0]])
        baseline1 = torch.tensor([[0, 0, 0, 0]])
        # these ones will be use as an additional forward args
        input2 = torch.tensor([[0, 2, 4, 1], [2, 3, 5, 7]])
        input3 = torch.tensor([[3, 5, 6, 7], [2, 3, 0, 1]])

        self._assert_compare_with_emb_patching(
            input1, baseline1, additional_args=(input2, input3)
        )

    def test_compare_with_layer_conductance_attr_to_outputs(self) -> None:
        model = BasicModel_MultiLayer()
        input = torch.tensor([[50.0, 50.0, 50.0]], requires_grad=True)
        self._assert_compare_with_layer_conductance(model, input)

    def test_compare_with_layer_conductance_attr_to_inputs(self) -> None:
        # Note that Layer Conductance and Layer Integrated Gradients (IG) aren't
        # exactly the same. Layer IG computes partial derivative of the output
        # with respect to the layer and sums along the straight line. While Layer
        # Conductance also computes the same partial derivatives it doesn't use
        # the straight line but a path defined by F(i) - F(i - 1).
        # However, in some cases when that path becomes close to a straight line,
        # Layer IG and Layer Conductance become numerically very close.
        model = BasicModel_MultiLayer()
        input = torch.tensor([[50.0, 50.0, 50.0]], requires_grad=True)
        self._assert_compare_with_layer_conductance(model, input, True)

    def test_multiple_tensors_compare_with_expected(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        self._assert_compare_with_expected(
            net,
            net.multi_relu,
            inp,
            ([[90.0, 100.0, 100.0, 100.0]], [[90.0, 100.0, 100.0, 100.0]]),
        )

    def test_lstm_layer_output_with_state_tuple(self) -> None:
        class LSTMModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.lstm = torch.nn.LSTM(7, 3, 1, batch_first=True)
                self.linear = torch.nn.Linear(3, 2)

            def forward(self, input: Tensor) -> Tensor:
                output, _ = self.lstm(input)
                return self.linear(output[:, -1, :])

        model = LSTMModel().eval()
        inputs = torch.randn(4, 5, 7)
        layer_ig = LayerIntegratedGradients(model, model.lstm)

        attributions, delta = layer_ig.attribute(  # type: ignore[has-type]
            inputs, target=0, return_convergence_delta=True
        )

        self.assertEqual(attributions.shape, (4, 5, 3))
        self.assertEqual(delta.shape, (4,))

    def test_tensor_layer_output_with_non_tensor_metadata(self) -> None:
        class TensorAndMetadataLayer(Module):
            def forward(self, input: Tensor) -> Tuple[Tensor, None]:
                return 2 * input, None

        class TensorAndMetadataModel(Module):
            def __init__(self) -> None:
                super().__init__()
                self.layer = TensorAndMetadataLayer()

            def forward(self, input: Tensor) -> Tensor:
                layer_output, _ = self.layer(input)
                return 3 * layer_output[:, 0] - 2 * layer_output[:, 1]

        model = TensorAndMetadataModel()
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        baselines = torch.zeros_like(inputs)
        layer_ig = LayerIntegratedGradients(model, model.layer)

        attributions, delta = layer_ig.attribute(  # type: ignore[has-type]
            inputs,
            baselines,
            return_convergence_delta=True,
        )

        assertTensorAlmostEqual(
            self,
            attributions,
            [[6.0, -8.0], [18.0, -16.0]],
            delta=1e-5,
            mode="max",
        )
        assertTensorAlmostEqual(self, delta, [0.0, 0.0], delta=1e-5, mode="max")

    def test_multiple_layers_single_inputs(self) -> None:
        input1 = torch.tensor([[2, 5, 0, 1], [3, 1, 1, 0]])
        input2 = torch.tensor([[0, 2, 4, 1], [2, 3, 5, 7]])
        input3 = torch.tensor([[3, 5, 6, 7], [2, 3, 0, 1]])

        inputs = (input1, input2, input3)
        baseline = tuple(torch.zeros_like(inp) for inp in inputs)

        self._assert_compare_with_emb_patching(
            inputs,
            baseline,
            multiple_emb=True,
            additional_args=None,
        )

    def test_multiple_layers_multiple_inputs_shared_input(self) -> None:
        input1 = torch.randn(5, 3)
        input2 = torch.randn(5, 3)
        input3 = torch.randn(5, 3)
        inputs = (input1, input2, input3)
        baseline = tuple(torch.zeros_like(inp) for inp in inputs)

        net = BasicModel_MultiLayer_TrueMultiInput()

        # pyre-fixme[6]: For 2nd argument expected `ModuleOrModuleList` but got
        #  `List[Union[BasicModel_MultiLayer, BasicModel_MultiLayer_MultiInput]]`.
        lig = LayerIntegratedGradients(net, layer=[net.m1, net.m234])
        ig = IntegratedGradients(net)

        # test layer inputs
        attribs_inputs = lig.attribute(  # type: ignore[has-type]
            inputs, baseline, target=0, attribute_to_layer_input=True
        )
        attribs_inputs_regular_ig = ig.attribute(  # type: ignore[has-type]
            inputs, baseline, target=0
        )

        self.assertIsInstance(attribs_inputs, list)
        self.assertEqual(len(attribs_inputs), 2)
        self.assertIsInstance(attribs_inputs[0], Tensor)
        self.assertIsInstance(attribs_inputs[1], tuple)
        self.assertEqual(len(attribs_inputs[1]), 3)

        assertTensorTuplesAlmostEqual(
            self,
            # last input for second layer is first input =>
            # add the attributions
            (attribs_inputs[0] + attribs_inputs[1][-1],)  # type: ignore
            + attribs_inputs[1][0:-1],  # type: ignore
            attribs_inputs_regular_ig,
            delta=1e-5,
        )

        # test layer outputs
        attribs = lig.attribute(inputs, baseline, target=0)  # type: ignore[has-type]
        ig = IntegratedGradients(lambda x, y: x + y)
        attribs_ig = ig.attribute(  # type: ignore[has-type]
            (net.m1(input1), net.m234(input2, input3, input1, 1)),
            (net.m1(baseline[0]), net.m234(baseline[1], baseline[2], baseline[1], 1)),
            target=0,
        )

        assertTensorTuplesAlmostEqual(self, attribs, attribs_ig, delta=1e-5)

    def test_multiple_layers_multiple_input_outputs(self) -> None:
        # test with multiple layers, where one layer accepts multiple inputs
        input1 = torch.randn(5, 3)
        input2 = torch.randn(5, 3)
        input3 = torch.randn(5, 3)
        input4 = torch.randn(5, 3)
        inputs = (input1, input2, input3, input4)
        baseline = tuple(torch.zeros_like(inp) for inp in inputs)

        net = BasicModel_MultiLayer_TrueMultiInput()

        # pyre-fixme[6]: For 2nd argument expected `ModuleOrModuleList` but got
        #  `List[Union[BasicModel_MultiLayer, BasicModel_MultiLayer_MultiInput]]`.
        lig = LayerIntegratedGradients(net, layer=[net.m1, net.m234])
        ig = IntegratedGradients(net)

        # test layer inputs
        attribs_inputs = lig.attribute(  # type: ignore[has-type]
            inputs, baseline, target=0, attribute_to_layer_input=True
        )
        attribs_inputs_regular_ig = ig.attribute(  # type: ignore[has-type]
            inputs, baseline, target=0
        )

        self.assertIsInstance(attribs_inputs, list)
        self.assertEqual(len(attribs_inputs), 2)
        self.assertIsInstance(attribs_inputs[0], Tensor)
        self.assertIsInstance(attribs_inputs[1], tuple)
        self.assertEqual(len(attribs_inputs[1]), 3)

        assertTensorTuplesAlmostEqual(
            self,
            (attribs_inputs[0],) + attribs_inputs[1],  # type: ignore
            attribs_inputs_regular_ig,
            delta=1e-7,
        )

        # test layer outputs
        attribs = lig.attribute(inputs, baseline, target=0)  # type: ignore[has-type]
        ig = IntegratedGradients(lambda x, y: x + y)
        attribs_ig = ig.attribute(  # type: ignore[has-type]
            (net.m1(input1), net.m234(input2, input3, input4, 1)),
            (net.m1(baseline[0]), net.m234(baseline[1], baseline[2], baseline[3], 1)),
            target=0,
        )

        assertTensorTuplesAlmostEqual(self, attribs, attribs_ig, delta=1e-7)

    def test_multiple_tensors_compare_with_exp_wo_mult_by_inputs(self) -> None:
        net = BasicModel_MultiLayer(multi_input_module=True)
        inp = torch.tensor([[0.0, 100.0, 0.0]])
        base = torch.tensor([[0.0, 0.0, 0.0]])
        target_layer = net.multi_relu
        layer_ig = LayerIntegratedGradients(net, target_layer)
        layer_ig_wo_mult_by_inp = LayerIntegratedGradients(
            net, target_layer, multiply_by_inputs=False
        )
        layer_act = LayerActivation(net, target_layer)
        attributions = layer_ig.attribute(inp, target=0)  # type: ignore[has-type]
        attributions_wo_mult_by_inp = layer_ig_wo_mult_by_inp.attribute(  # type: ignore
            inp, target=0
        )
        inp_minus_baseline_activ = tuple(
            inp_act - base_act
            for inp_act, base_act in zip(
                layer_act.attribute(inp),  # type: ignore[has-type]
                layer_act.attribute(base),  # type: ignore[has-type]
            )
        )
        assertTensorTuplesAlmostEqual(
            self,
            tuple(
                attr_wo_mult * inp_min_base
                for attr_wo_mult, inp_min_base in zip(
                    attributions_wo_mult_by_inp, inp_minus_baseline_activ
                )
            ),
            attributions,
        )

    def test_simple_multi_gradient_activation_with_unused_layer(self) -> None:
        if version.parse(torch.__version__) < version.parse("2.1.0"):
            raise unittest.SkipTest(
                "Skipping unused layed gradient test since it is not supported "
                "by torch version < 2.1"
            )

        model = BasicModel_MultiLayer(multi_input_module=True)
        test_input = torch.tensor([[3.0, 4.0, 0.0]], requires_grad=True)
        # pyre-fixme[6]: For 2nd argument expected `ModuleOrModuleList` but got
        #  `List[Union[ReLU, Linear]]`.
        layer_ig = LayerIntegratedGradients(model, [model.linear1, model.relu])
        attributions = cast(
            List[Tensor],
            layer_ig.attribute(  # type: ignore[has-type]
                inputs=test_input, target=0, grad_kwargs={"materialize_grads": True}
            ),
        )
        self.assertEqual(len(attributions), 2)
        self.assertEqual(list(attributions[0].shape), [1, 4])
        self.assertEqual(list(attributions[1].shape), [1, 4])

    def _assert_compare_with_layer_conductance(
        self, model: Module, input: Tensor, attribute_to_layer_input: bool = False
    ) -> None:
        lc = LayerConductance(model, cast(Module, model.linear2))
        # For large number of steps layer conductance and layer integrated gradients
        # become very close
        attribution, delta = lc.attribute(  # type: ignore[has-type]
            input,
            target=0,
            n_steps=1500,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        lig = LayerIntegratedGradients(model, cast(Module, model.linear2))
        attributions2, delta2 = lig.attribute(  # type: ignore[has-type]
            input,
            target=0,
            n_steps=1500,
            return_convergence_delta=True,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        assertTensorAlmostEqual(
            self, attribution, attributions2, delta=0.01, mode="max"
        )
        assertTensorAlmostEqual(self, delta, delta2, delta=0.5, mode="max")

    def _assert_compare_with_emb_patching(
        self,
        input: Union[Tensor, Tuple[Tensor, ...]],
        baseline: Union[Tensor, Tuple[Tensor, ...]],
        additional_args: Union[None, Tuple[Tensor, ...]],
        multiply_by_inputs: bool = True,
        multiple_emb: bool = False,
    ) -> None:
        model = BasicEmbeddingModel(nested_second_embedding=True)
        if multiple_emb:
            module_list: List[Module] = [model.embedding1, model.embedding2]
            lig = LayerIntegratedGradients(
                model,
                module_list,
                multiply_by_inputs=multiply_by_inputs,
            )
        else:
            lig = LayerIntegratedGradients(
                model, model.embedding1, multiply_by_inputs=multiply_by_inputs
            )

        attributions, delta = lig.attribute(  # type: ignore[has-type]
            input,
            baselines=baseline,
            additional_forward_args=additional_args,
            return_convergence_delta=True,
        )

        # now let's interpret with standard integrated gradients and
        # the embeddings for monkey patching
        e1 = configure_interpretable_embedding_layer(model, "embedding1")
        e1_input_emb = e1.indices_to_embeddings(input[0] if multiple_emb else input)
        e1_baseline_emb = e1.indices_to_embeddings(
            baseline[0] if multiple_emb else baseline
        )

        input_emb = e1_input_emb
        baseline_emb = e1_baseline_emb
        e2 = None
        if multiple_emb:
            e2 = configure_interpretable_embedding_layer(model, "embedding2")
            e2_input_emb = e2.indices_to_embeddings(*input[1:])
            e2_baseline_emb = e2.indices_to_embeddings(*baseline[1:])

            input_emb = (e1_input_emb, e2_input_emb)
            baseline_emb = (e1_baseline_emb, e2_baseline_emb)

        ig = IntegratedGradients(model, multiply_by_inputs=multiply_by_inputs)
        attributions_with_ig, delta_with_ig = ig.attribute(  # type: ignore[has-type]
            input_emb,
            baselines=baseline_emb,
            additional_forward_args=additional_args,
            target=0,
            return_convergence_delta=True,
        )
        remove_interpretable_embedding_layer(model, e1)
        if e2 is not None:
            remove_interpretable_embedding_layer(model, e2)

        self.assertEqual(
            isinstance(attributions_with_ig, tuple), isinstance(attributions, list)
        )

        self.assertTrue(
            isinstance(attributions_with_ig, tuple)
            if multiple_emb
            else not isinstance(attributions_with_ig, tuple)
        )

        # convert to tuple for comparison
        if not isinstance(attributions_with_ig, tuple):
            attributions = (attributions,)
            attributions_with_ig = (attributions_with_ig,)
        else:
            # convert list to tuple
            self.assertIsInstance(attributions, list)
            attributions = tuple(attributions)

        for attr_lig, attr_ig in zip(attributions, attributions_with_ig):
            self.assertEqual(cast(Tensor, attr_lig).shape, cast(Tensor, attr_ig).shape)
            assertTensorAlmostEqual(self, attr_lig, attr_ig, delta=0.05, mode="max")

        if multiply_by_inputs:
            assertTensorAlmostEqual(self, delta, delta_with_ig, delta=0.05, mode="max")

    def _assert_compare_with_expected(
        self,
        model: Module,
        target_layer: Module,
        test_input: Union[Tensor, Tuple[Tensor, ...]],
        expected_ig: Tuple[List[List[float]], ...],
        additional_input: Any = None,
    ) -> None:
        layer_ig = LayerIntegratedGradients(model, target_layer)
        attributions = layer_ig.attribute(  # type: ignore[has-type]
            test_input, target=0, additional_forward_args=additional_input
        )
        assertTensorTuplesAlmostEqual(self, attributions, expected_ig, delta=0.01)
