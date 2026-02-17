#!/usr/bin/env python3

# pyre-strict

import io
import unittest
import unittest.mock
from typing import Any, Callable, List, Tuple, Union

import torch
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.kernel_shap import KernelShap
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
    set_all_random_seeds,
)
from captum.testing.helpers.basic_models import (
    BasicLinearModel,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
)
from torch import Tensor


class Test(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        try:
            import sklearn  # noqa: F401

            assert (
                sklearn.__version__ >= "0.23.0"
            ), "Must have sklearn version 0.23.0 or higher"
        except (ImportError, AssertionError):
            raise unittest.SkipTest("Skipping KernelShap tests, sklearn not available.")

    def test_linear_kernel_shap(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        baseline = torch.tensor([[10.0, 20.0, 10.0]], requires_grad=True)

        self._kernel_shap_test_assert(
            net,
            inp,
            [[40.0, 120.0, 80.0]],
            n_samples=500,
            baselines=baseline,
            expected_coefs=[[40.0, 120.0, 80.0]],
        )

    def test_simple_kernel_shap(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._kernel_shap_test_assert(
            net,
            inp,
            [[76.66666, 196.66666, 116.66666]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
        )

    def test_simple_kernel_shap_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._kernel_shap_test_assert(
            net,
            inp,
            [[275.0, 275.0, 115.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
            expected_coefs=[[275.0, 115.0]],
        )

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_kernel_shap_with_show_progress(
        self, mock_stderr: io.StringIO
    ) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._kernel_shap_test_assert(
                net,
                inp,
                [[76.66666, 196.66666, 116.66666]],
                perturbations_per_eval=(bsz,),
                n_samples=500,
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Kernel Shap attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    def test_simple_kernel_shap_with_baselines(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        self._kernel_shap_test_assert(
            net,
            inp,
            [[248.0, 248.0, 104.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
        )

    def test_simple_batch_kernel_shap(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._kernel_shap_test_assert(
            net,
            inp,
            [[7.0, 32.5, 10.5], [76.66666, 196.66666, 116.66666]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=2000,
        )

    def test_simple_batch_kernel_shap_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[2.0, 10.0, 3.0], [20.0, 50.0, 30.0]], requires_grad=True)
        self._kernel_shap_test_assert(
            net,
            inp,
            [[39.5, 39.5, 10.5], [275.0, 275.0, 115.0]],
            feature_mask=torch.tensor([[0, 0, 1], [1, 1, 0]]),
            perturbations_per_eval=(1, 2, 3),
            n_samples=100,
            expected_coefs=[[39.5, 10.5], [115.0, 275.0]],
        )

    def test_multi_input_kernel_shap_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0]])
        expected = (
            [[90.0, 0.0, 0.0]],
            [[78.0, 0.0, 198.0]],
            [[0.0, 398.0, 38.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=2000,
        )

    def test_multi_input_kernel_shap_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 0, 0]])
        expected = (
            [[255.0, 595.0, 255.0]],
            [[255.0, 595.0, 0.0]],
            [[255.0, 255.0, 255.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline = (
            [[184.0, 580.0, 184.0]],
            [[184.0, 580.0, -12.0]],
            [[184.0, 184.0, 184.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    def test_multi_input_kernel_shap_with_empty_input(self) -> None:
        net = BasicLinearModel()
        inp1 = torch.tensor([[23.0, 0.0, 0.0, 23.0, 0.0, 0.0, 23.0]])
        inp2 = torch.tensor([[]])  # empty input
        mask1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        mask2 = torch.tensor([[]], dtype=torch.long)  # empty mask
        expected: Tuple[List[List[float]], ...] = (
            [[-8.0, 0.0, 0.0, -2.0, 0.0, 0.0, -8.0]],
            [[]],
        )
        # no mask
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2),
            expected,
            n_samples=2000,
            expected_coefs=[[-8.0, 0.0, 0.0, -2.0, 0.0, 0.0, -8.0]],
        )
        # with mask
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2),
            expected,
            n_samples=2000,
            expected_coefs=[[-8.0, 0.0, 0.0, -2.0, 0.0, 0.0, -8.0]],
            feature_mask=(mask1, mask2),
        )

    def test_multi_input_batch_kernel_shap_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected = (
            [[90.0, 0.0, 0.0], [78.0, 198.0, 118.0]],
            [[78.0, 0.0, 198.0], [0.0, 398.0, 0.0]],
            [[0.0, 398.0, 38.0], [0.0, 38.0, 0.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=2500,
            expected_coefs=[
                [90.0, 0.0, 0.0, 78.0, 0.0, 198.0, 0.0, 398.0, 38.0],
                [78.0, 198.0, 118.0, 0.0, 398.0, 0.0, 0.0, 38.0, 0.0],
            ],
        )

    def test_multi_input_batch_kernel_shap(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected = (
            [[1088.6666, 1088.6666, 1088.6666], [255.0, 595.0, 255.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 595.0, 0.0]],
            [[76.6666, 1088.6666, 156.6666], [255.0, 255.0, 255.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            n_samples=300,
        )
        expected_with_baseline = (
            [[1040.0, 1040.0, 1040.0], [184.0, 580.0, 184.0]],
            [[52.0, 1040.0, 132.0], [184.0, 580.0, -12.0]],
            [[52.0, 1040.0, 132.0], [184.0, 184.0, 184.0]],
        )
        self._kernel_shap_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
        )

    # Remaining tests are for cases where forward function returns a scalar
    # as either a float, integer, 0d tensor or 1d tensor.
    def test_single_kernel_shap_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_kernel_shap_assert(
            lambda inp: torch.sum(net(inp)).item()
        )

    def test_single_kernel_shap_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_kernel_shap_assert(lambda inp: torch.sum(net(inp)))

    def test_single_kernel_shap_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_kernel_shap_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_single_kernel_shap_scalar_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_kernel_shap_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def _single_input_scalar_kernel_shap_assert(
        self, func: Callable[..., Union[Tensor, float, int]]
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._kernel_shap_test_assert(
            func,
            inp,
            [[79.0, 79.0, 21.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
        )

    def test_multi_inp_kernel_shap_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_kernel_shap_assert(lambda *inp: torch.sum(net(*inp)))

    def test_multi_inp_kernel_shap_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_kernel_shap_assert(
            lambda *inp: torch.sum(net(*inp)).reshape(1)
        )

    def test_multi_inp_kernel_shap_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_kernel_shap_assert(
            lambda *inp: int(torch.sum(net(*inp)).item())
        )

    def test_multi_inp_kernel_shap_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_kernel_shap_assert(
            lambda *inp: torch.sum(net(*inp)).item()
        )

    def test_futures_not_implemented(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        kernel_shap = KernelShap(net)
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = kernel_shap.attribute_future()  # type: ignore
        self.assertEqual(attributions, None)

    def _multi_input_scalar_kernel_shap_assert(
        self, func: Callable[..., Union[Tensor, float, int]]
    ) -> None:
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [20.0, 10.0, 13.0]])
        mask1 = torch.tensor([[1, 1, 1]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2]])
        expected = (
            [[3850.6666, 3850.6666, 3850.6666]],
            [[306.6666, 3850.6666, 410.6666]],
            [[306.6666, 3850.6666, 410.6666]],
        )

        self._kernel_shap_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
            n_samples=1500,
        )

    def _kernel_shap_test_assert(
        self,
        model: Callable[..., Union[int, float, Tensor]],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr: Union[
            List[List[float]], Tuple[List[List[float]], ...], List[List[List[float]]]
        ],
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
        expected_coefs: Union[None, List[float], List[List[float]]] = None,
        show_progress: bool = False,
    ) -> None:
        for batch_size in perturbations_per_eval:
            kernel_shap = KernelShap(model)
            attributions = kernel_shap.attribute(
                test_input,
                target=target,
                feature_mask=feature_mask,
                additional_forward_args=additional_input,
                baselines=baselines,
                perturbations_per_eval=batch_size,
                n_samples=n_samples,
                show_progress=show_progress,
            )

            assertTensorTuplesAlmostEqual(
                self, attributions, expected_attr, delta=delta, mode="max"
            )

            if expected_coefs is not None:
                set_all_random_seeds(1234)
                # Test with return_input_shape = False
                attributions = kernel_shap.attribute(
                    test_input,
                    target=target,
                    feature_mask=feature_mask,
                    additional_forward_args=additional_input,
                    baselines=baselines,
                    perturbations_per_eval=batch_size,
                    n_samples=n_samples,
                    return_input_shape=False,
                    show_progress=show_progress,
                )
                assertTensorAlmostEqual(
                    self, attributions, expected_coefs, delta=delta, mode="max"
                )


if __name__ == "__main__":
    unittest.main()
