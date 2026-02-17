#!/usr/bin/env python3

# pyre-strict

import io
import unittest
import unittest.mock
from functools import partial
from typing import Any, Callable, cast, Generator, List, Optional, Tuple, Union

import torch
from captum._utils.models.linear_model import SGDLasso, SkLearnLasso
from captum._utils.models.model import Model
from captum._utils.typing import BaselineType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.lime import get_exp_kernel_similarity_function, Lime, LimeBase
from captum.attr._utils.batching import _batch_example_iterator
from captum.attr._utils.common import (
    _construct_default_feature_mask,
    _format_input_baseline,
    _format_tensor_into_tuples,
)
from captum.testing.helpers.basic import (
    assertTensorAlmostEqual,
    assertTensorTuplesAlmostEqual,
    BaseTest,
)
from captum.testing.helpers.basic_models import (
    BasicLinearModel,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModelBoolInput,
)
from torch import Tensor


def alt_perturb_func(
    original_inp: TensorOrTupleOfTensorsGeneric, **kwargs: Any
) -> TensorOrTupleOfTensorsGeneric:
    if isinstance(original_inp, Tensor):
        device = original_inp.device
    else:
        device = original_inp[0].device

    feature_mask = kwargs["feature_mask"]

    probs = torch.ones(1, kwargs["num_interp_features"]) * 0.5
    curr_sample = torch.bernoulli(probs).to(device=device)

    if isinstance(original_inp, Tensor):
        binary_mask = curr_sample[0][feature_mask]
        return binary_mask * original_inp + (1 - binary_mask) * kwargs["baselines"]
    else:
        binary_mask_tuple = tuple(
            curr_sample[0][feature_mask[j]] for j in range(len(feature_mask))
        )

        return cast(
            TensorOrTupleOfTensorsGeneric,
            tuple(
                binary_mask_tuple[j] * original_inp[j]
                + (1 - binary_mask_tuple[j]) * kwargs["baselines"][j]
                for j in range(len(feature_mask))
            ),
        )


def alt_perturb_generator(
    original_inp: TensorOrTupleOfTensorsGeneric, **kwargs: Any
) -> Generator[TensorOrTupleOfTensorsGeneric, None, None]:
    while True:
        yield alt_perturb_func(original_inp, **kwargs)


def alt_to_interp_rep(
    curr_sample: TensorOrTupleOfTensorsGeneric,
    original_input: TensorOrTupleOfTensorsGeneric,
    **kwargs: Any,
) -> Tensor:
    binary_vector = torch.zeros(1, kwargs["num_interp_features"])
    feature_mask = kwargs["feature_mask"]
    for i in range(kwargs["num_interp_features"]):
        curr_total = 1
        if isinstance(curr_sample, Tensor):
            if (
                torch.sum(
                    torch.abs(
                        (feature_mask == i).float() * (curr_sample - original_input)
                    )
                )
                > 0.001
            ):
                curr_total = 0
        else:
            sum_diff = cast(
                Tensor,
                sum(
                    torch.sum(torch.abs((mask == i).float() * (sample - inp)))
                    for inp, sample, mask in zip(
                        original_input, curr_sample, feature_mask
                    )
                ),
            )
            if sum_diff > 0.001:
                curr_total = 0
        binary_vector[0][i] = curr_total
    return binary_vector


class Test(BaseTest):
    def setUp(self) -> None:
        super().setUp()
        try:
            import sklearn  # noqa: F401

            assert (
                sklearn.__version__ >= "0.23.0"
            ), "Must have sklearn version 0.23.0 or higher to use "
            "sample_weight in Lasso regression."
        except (ImportError, AssertionError):
            raise unittest.SkipTest("Skipping Lime tests, sklearn not available.")

    def test_simple_lime(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[73.3716, 193.3349, 113.3349]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
            expected_coefs_only=[[73.3716, 193.3349, 113.3349]],
            test_generator=True,
        )

    def test_simple_lime_sgd_model(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        interpretable_model = SGDLasso()
        interpretable_model.fit = partial(  # type: ignore
            interpretable_model.fit, initial_lr=0.1, max_epoch=500
        )
        self._lime_test_assert(
            net,
            inp,
            [[73.3716, 193.3349, 113.3349]],
            n_samples=1000,
            expected_coefs_only=[[73.3716, 193.3349, 113.3349]],
            interpretable_model=interpretable_model,
        )

    def test_simple_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[271.0, 271.0, 111.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
            expected_coefs_only=[[271.0, 111.0]],
        )

    def test_simple_lime_with_baselines(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]])
        self._lime_test_assert(
            net,
            inp,
            [[244.0, 244.0, 100.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=4,
            perturbations_per_eval=(1, 2, 3),
            expected_coefs_only=[[244.0, 100.0]],
            test_generator=True,
        )

    def test_simple_lime_boolean(self) -> None:
        net = BasicModelBoolInput()
        inp = torch.tensor([[True, False, True]])
        self._lime_test_assert(
            net,
            inp,
            [[31.42, 31.42, 30.90]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            perturbations_per_eval=(1, 2, 3),
            test_generator=True,
        )

    def test_simple_lime_boolean_with_baselines(self) -> None:
        net = BasicModelBoolInput()
        inp = torch.tensor([[True, False, True]])
        self._lime_test_assert(
            net,
            inp,
            [[-36.0, -36.0, 0.0]],
            feature_mask=torch.tensor([[0, 0, 1]]),
            baselines=True,
            perturbations_per_eval=(1, 2, 3),
            test_generator=True,
        )

    @unittest.mock.patch("sys.stderr", new_callable=io.StringIO)
    def test_simple_lime_with_show_progress(
        self, mock_stderr: unittest.mock.Mock
    ) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0]], requires_grad=True)

        # test progress output for each batch size
        for bsz in (1, 2, 3):
            self._lime_test_assert(
                net,
                inp,
                [[73.3716, 193.3349, 113.3349]],
                perturbations_per_eval=(bsz,),
                n_samples=500,
                test_generator=True,
                show_progress=True,
            )
            output = mock_stderr.getvalue()

            # to test if progress calculation aligns with the actual iteration
            # all perturbations_per_eval should reach progress of 100%
            assert (
                "Lime attribution: 100%" in output
            ), f"Error progress output: {repr(output)}"

            mock_stderr.seek(0)
            mock_stderr.truncate(0)

    def test_simple_batch_lime(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0], [10.0, 14.0, 4.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[73.4450, 193.5979, 113.4363], [32.11, 48.00, 11.00]],
            perturbations_per_eval=(1, 2, 3),
            n_samples=800,
            expected_coefs_only=[[73.4450, 193.5979, 113.4363], [32.11, 48.00, 11.00]],
        )

    def test_simple_batch_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer()
        inp = torch.tensor([[20.0, 50.0, 30.0], [10.0, 14.0, 4.0]], requires_grad=True)
        self._lime_test_assert(
            net,
            inp,
            [[271.0, 271.0, 111.0], [32.11, 48.00, 11.00]],
            feature_mask=torch.tensor([[0, 0, 1], [0, 1, 2]]),
            perturbations_per_eval=(1, 2, 3),
            n_samples=600,
            expected_coefs_only=[[271.0, 111.0, 0.0], [32.11, 48.00, 11.00]],
            test_generator=True,
        )

    def test_multi_input_lime_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0]])
        expected = (
            [[87, 0, 0]],
            [[75, 0, 195]],
            [[0, 395, 35]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=2000,
            expected_coefs_only=[[87, 0, 0, 75, 0, 195, 0, 395, 35]],
        )

    def test_multi_input_lime_with_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 0, 0]])
        expected = (
            [[251.0, 591.0, 251.0]],
            [[251.0, 591.0, 0.0]],
            [[251.0, 251.0, 251.0]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            n_samples=500,
            expected_coefs_only=[[251.0, 591.0, 0.0]],
        )
        expected_with_baseline = (
            [[180, 576.0, 180]],
            [[180, 576.0, -8.0]],
            [[180, 180, 180]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
            n_samples=500,
            expected_coefs_only=[[180, 576.0, -8.0]],
            test_generator=True,
        )

    def test_multi_input_lime_with_empty_input(self) -> None:
        net = BasicLinearModel()
        inp1 = torch.tensor([[23.0, 0.0, 0.0, 23.0, 0.0, 0.0, 23.0]])
        inp2 = torch.tensor([[]])  # empty input
        mask1 = torch.tensor([[0, 1, 2, 3, 4, 5, 6]])
        mask2 = torch.tensor([[]], dtype=torch.long)  # empty mask
        expected: Tuple[List[List[float]], ...] = (
            [[-4.0, 0, 0, 0, 0, 0, -4.0]],
            [[]],
        )
        # no mask
        self._lime_test_assert(
            net,
            (inp1, inp2),
            expected,
            n_samples=2000,
            expected_coefs_only=[[-4.0, 0, 0, 0, 0, 0, -4.0]],
        )
        # with mask
        self._lime_test_assert(
            net,
            (inp1, inp2),
            expected,
            n_samples=2000,
            expected_coefs_only=[[-4.0, 0, 0, 0, 0, 0, -4.0]],
            feature_mask=(mask1, mask2),
        )

    def test_multi_input_batch_lime_without_mask(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 0.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 0.0, 50.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [0.0, 10.0, 0.0]])
        expected = (
            [[87.8777, 0.0000, 0.0000], [75.8461, 195.6842, 115.3390]],
            [[74.7283, 0.0000, 195.1708], [0.0000, 395.3823, 0.0000]],
            [[0.0000, 395.5216, 35.5530], [0.0000, 35.1349, 0.0000]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            n_samples=1000,
            expected_coefs_only=[
                [87.8777, 0.0, 0.0, 74.7283, 0.0, 195.1708, 0.0, 395.5216, 35.5530],
                [
                    75.8461,
                    195.6842,
                    115.3390,
                    0.0000,
                    395.3823,
                    0.0000,
                    0.0000,
                    35.1349,
                    0.0000,
                ],
            ],
            delta=1.2,
        )

    def test_multi_input_batch_lime(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        inp1 = torch.tensor([[23.0, 100.0, 0.0], [20.0, 50.0, 30.0]])
        inp2 = torch.tensor([[20.0, 50.0, 30.0], [0.0, 100.0, 0.0]])
        inp3 = torch.tensor([[0.0, 100.0, 10.0], [2.0, 10.0, 3.0]])
        mask1 = torch.tensor([[1, 1, 1], [0, 1, 0]])
        mask2 = torch.tensor([[0, 1, 2]])
        mask3 = torch.tensor([[0, 1, 2], [0, 0, 0]])
        expected = (
            [[1086.2802, 1086.2802, 1086.2802], [250.8907, 590.9789, 250.8907]],
            [[73.2166, 1086.2802, 152.6888], [250.8907, 590.9789, 0.0000]],
            [[73.2166, 1086.2802, 152.6888], [250.8907, 250.8907, 250.8907]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
        )
        expected_with_baseline = (
            [[1036.4233, 1036.4233, 1036.4233], [180.3035, 575.8969, 180.3035]],
            [[48.2441, 1036.4233, 128.3161], [180.3035, 575.8969, -8.3229]],
            [[48.2441, 1036.4233, 128.3161], [180.3035, 180.3035, 180.3035]],
        )
        self._lime_test_assert(
            net,
            (inp1, inp2, inp3),
            expected_with_baseline,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            baselines=(2, 3.0, 4),
            perturbations_per_eval=(1, 2, 3),
            expected_coefs_only=[
                [48.2441, 1036.4233, 128.3161],
                [180.3035, 575.8969, -8.3229],
            ],
            n_samples=500,
            test_generator=True,
        )

    # Remaining tests are for cases where forward function returns a scalar
    # as either a float, integer, 0d tensor or 1d tensor.
    def test_single_lime_scalar_float(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(lambda inp: torch.sum(net(inp)).item())

    def test_single_lime_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(lambda inp: torch.sum(net(inp)))

    def test_single_lime_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(
            lambda inp: torch.sum(net(inp)).reshape(1)
        )

    def test_single_lime_scalar_int(self) -> None:
        net = BasicModel_MultiLayer()
        self._single_input_scalar_lime_assert(
            lambda inp: int(torch.sum(net(inp)).item())
        )

    def _single_input_scalar_lime_assert(
        self, func: Callable[[Tensor], Union[Tensor, float, int]]
    ) -> None:
        inp = torch.tensor([[2.0, 10.0, 3.0]], requires_grad=True)
        mask = torch.tensor([[0, 0, 1]])

        self._lime_test_assert(
            func,
            inp,
            [[75.0, 75.0, 17.0]],
            feature_mask=mask,
            perturbations_per_eval=(1,),
            target=None,
            expected_coefs_only=[[75.0, 17.0]],
            n_samples=700,
        )

    def test_multi_inp_lime_scalar_tensor_0d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(lambda *inp: torch.sum(net(*inp)))

    def test_multi_inp_lime_scalar_tensor_1d(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(
            lambda *inp: torch.sum(net(*inp)).reshape(1)
        )

    def test_multi_inp_lime_scalar_tensor_int(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(
            lambda *inp: int(torch.sum(net(*inp)).item())
        )

    def test_multi_inp_lime_scalar_float(self) -> None:
        net = BasicModel_MultiLayer_MultiInput()
        self._multi_input_scalar_lime_assert(lambda *inp: torch.sum(net(*inp)).item())

    def test_futures_not_implemented(self) -> None:
        net = BasicLinearModel()
        # no mask
        lime = Lime(
            net,
            similarity_func=get_exp_kernel_similarity_function("cosine", 10.0),
            interpretable_model=(SkLearnLasso(alpha=1.0)),
        )
        attributions = None
        with self.assertRaises(NotImplementedError):
            attributions = lime.attribute_future()  # type: ignore
        self.assertEqual(attributions, None)

    def _multi_input_scalar_lime_assert(
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
            [[305.5, 3850.6666, 410.1]],
            [[305.5, 3850.6666, 410.1]],
        )

        self._lime_test_assert(
            func,
            (inp1, inp2, inp3),
            expected,
            additional_input=(1,),
            feature_mask=(mask1, mask2, mask3),
            perturbations_per_eval=(1,),
            target=None,
            n_samples=1500,
            expected_coefs_only=[[305.5, 3850.6666, 410.1]],
            delta=1.5,
            batch_attr=True,
            test_generator=True,
        )

    def _lime_test_assert(
        self,
        model: Callable[..., Union[int, float, Tensor]],
        test_input: TensorOrTupleOfTensorsGeneric,
        expected_attr: Any,
        expected_coefs_only: Union[None, List[List[Union[int, float]]], Tensor] = None,
        feature_mask: Union[None, TensorOrTupleOfTensorsGeneric] = None,
        additional_input: Any = None,
        perturbations_per_eval: Tuple[int, ...] = (1,),
        baselines: BaselineType = None,
        target: Union[None, int] = 0,
        n_samples: int = 100,
        delta: float = 1.0,
        batch_attr: bool = False,
        test_generator: bool = False,
        show_progress: bool = False,
        interpretable_model: Optional[Model] = None,
    ) -> None:
        for batch_size in perturbations_per_eval:
            lime = Lime(
                model,
                similarity_func=get_exp_kernel_similarity_function("cosine", 10.0),
                interpretable_model=(
                    interpretable_model
                    if interpretable_model
                    else SkLearnLasso(alpha=1.0)
                ),
            )
            attributions = lime.attribute(
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
            if expected_coefs_only is not None:
                # Test with return_input_shape = False
                attributions = lime.attribute(
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
                    self, attributions, expected_coefs_only, delta=delta, mode="max"
                )

                lime_alt = LimeBase(
                    model,
                    (
                        interpretable_model
                        if interpretable_model
                        else SkLearnLasso(alpha=1.0)
                    ),
                    get_exp_kernel_similarity_function("euclidean", 1000.0),
                    alt_perturb_generator if test_generator else alt_perturb_func,
                    False,
                    None,
                    alt_to_interp_rep,
                )

                # Test with equivalent sampling in original input space
                formatted_inputs, baselines = _format_input_baseline(
                    test_input, baselines
                )
                if feature_mask is None:
                    (
                        formatted_feature_mask,
                        num_interp_features,
                    ) = _construct_default_feature_mask(formatted_inputs)
                else:
                    formatted_feature_mask = _format_tensor_into_tuples(feature_mask)
                    num_interp_features = int(
                        max(
                            torch.max(single_mask).item()
                            for single_mask in feature_mask
                            if single_mask.numel()
                        )
                        + 1
                    )
                if batch_attr:
                    attributions = lime_alt.attribute(
                        test_input,
                        target=target,
                        feature_mask=(
                            formatted_feature_mask
                            if isinstance(test_input, tuple)
                            else formatted_feature_mask[0]
                        ),
                        additional_forward_args=additional_input,
                        baselines=baselines,
                        perturbations_per_eval=batch_size,
                        n_samples=n_samples,
                        num_interp_features=num_interp_features,
                        show_progress=show_progress,
                    )
                    assertTensorAlmostEqual(
                        self, attributions, expected_coefs_only, delta=delta, mode="max"
                    )
                    return

                bsz = formatted_inputs[0].shape[0]
                for (
                    curr_inps,
                    curr_target,
                    curr_additional_args,
                    curr_baselines,
                    curr_feature_mask,
                    expected_coef_single,
                ) in _batch_example_iterator(
                    bsz,
                    test_input,
                    target,
                    additional_input,
                    baselines if isinstance(test_input, tuple) else baselines[0],
                    (
                        formatted_feature_mask
                        if isinstance(test_input, tuple)
                        else formatted_feature_mask[0]
                    ),
                    expected_coefs_only,
                ):
                    attributions = lime_alt.attribute(
                        curr_inps,
                        target=curr_target,
                        feature_mask=curr_feature_mask,
                        additional_forward_args=curr_additional_args,
                        baselines=curr_baselines,
                        perturbations_per_eval=batch_size,
                        n_samples=n_samples,
                        num_interp_features=num_interp_features,
                        show_progress=show_progress,
                    )
                    assertTensorAlmostEqual(
                        self,
                        attributions,
                        expected_coef_single,
                        delta=delta,
                        mode="max",
                    )


if __name__ == "__main__":
    unittest.main()
