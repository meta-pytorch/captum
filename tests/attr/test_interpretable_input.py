#!/usr/bin/env python3

# pyre-strict

from typing import Dict, List, Literal, Optional, overload, Tuple, Union

import numpy as np
import PIL.Image
import torch
from captum._utils.typing import BatchEncodingType
from captum.attr._utils.interpretable_input import (
    ImageMaskInput,
    TextTemplateInput,
    TextTokenInput,
)
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from parameterized import parameterized
from torch import Tensor


class DummyTokenizer:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    unk_idx: int

    def __init__(self, vocab_list: List[str]) -> None:
        self.token_to_id = {v: i for i, v in enumerate(vocab_list)}
        self.id_to_token = vocab_list
        self.unk_idx = len(vocab_list) + 1

    @overload
    def encode(
        self, text: str, add_special_tokens: bool = ..., return_tensors: None = ...
    ) -> List[int]: ...

    @overload
    def encode(
        self,
        text: str,
        add_special_tokens: bool = ...,
        return_tensors: Literal["pt"] = ...,
    ) -> Tensor: ...

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = "pt",
    ) -> Union[List[int], Tensor]:
        assert return_tensors == "pt"
        return torch.tensor([self.convert_tokens_to_ids(text.split(" "))])

    @overload
    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]: ...
    @overload
    def convert_ids_to_tokens(self, token_ids: int) -> str: ...

    def convert_ids_to_tokens(
        self, token_ids: Union[List[int], int]
    ) -> Union[List[str], str]:
        if isinstance(token_ids, int):
            return (
                self.id_to_token[token_ids]
                if token_ids < len(self.id_to_token)
                else "[UNK]"
            )
        return [
            (self.id_to_token[i] if i < len(self.id_to_token) else "[UNK]")
            for i in token_ids
        ]

    @overload
    def convert_tokens_to_ids(self, tokens: str) -> int: ...
    @overload
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]: ...

    def convert_tokens_to_ids(
        self, tokens: Union[List[str], str]
    ) -> Union[List[int], int]:
        if isinstance(tokens, str):
            return (
                self.token_to_id[tokens] if tokens in self.token_to_id else self.unk_idx
            )
        return [
            (self.token_to_id[t] if t in self.token_to_id else self.unk_idx)
            for t in tokens
        ]

    def decode(self, token_ids: Tensor) -> str:
        raise NotImplementedError

    def __call__(
        self,
        text: Optional[Union[str, List[str], List[List[str]]]] = None,
        add_special_tokens: bool = True,
        return_offsets_mapping: bool = False,
    ) -> BatchEncodingType:
        raise NotImplementedError


class TestTextTemplateInput(BaseTest):
    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
            ),
        ]
    )
    def test_input(
        self, template: str, values: Union[List[str], Dict[str, str]]
    ) -> None:
        tt_input = TextTemplateInput(template, values)

        expected_tensor = torch.tensor([[1.0] * 4])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        self.assertEqual(tt_input.to_model_input(), "a b c d e f")

        perturbed_tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b  d e ")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], ["w", "x", "y", "z"]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": "w", "arg2": "x", "arg3": "y", "arg4": "z"},
            ),
        ]
    )
    def test_input_with_baselines(
        self,
        template: str,
        values: Union[List[str], Dict[str, str]],
        baselines: Union[List[str], Dict[str, str]],
    ) -> None:
        perturbed_tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        # single instance baselines
        tt_input = TextTemplateInput(template, values, baselines=baselines)
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b x d e z")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], [0, 1, 0, 1]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": 0, "arg2": 1, "arg3": 0, "arg4": 1},
            ),
        ]
    )
    def test_input_with_mask(
        self,
        template: str,
        values: Union[List[str], Dict[str, str]],
        mask: Union[List[int], Dict[str, int]],
    ) -> None:
        tt_input = TextTemplateInput(template, values, mask=mask)

        expected_tensor = torch.tensor([[1.0] * 2])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        self.assertEqual(tt_input.to_model_input(), "a b c d e f")

        perturbed_tensor = torch.tensor([[1.0, 0.0]])
        self.assertEqual(tt_input.to_model_input(perturbed_tensor), "a b  d e ")

    @parameterized.expand(
        [
            ("{} b {} {} e {}", ["a", "c", "d", "f"], [0, 1, 0, 1]),
            (
                "{arg1} b {arg2} {arg3} e {arg4}",
                {"arg1": "a", "arg2": "c", "arg3": "d", "arg4": "f"},
                {"arg1": 0, "arg2": 1, "arg3": 0, "arg4": 1},
            ),
        ]
    )
    def test_format_attr(
        self,
        template: str,
        values: Union[List[str], Dict[str, str]],
        mask: Union[List[int], Dict[str, int]],
    ) -> None:
        tt_input = TextTemplateInput(template, values, mask=mask)

        attr = torch.tensor([[0.1, 0.2]])

        assertTensorAlmostEqual(
            self, tt_input.format_attr(attr), torch.tensor([[0.1, 0.2, 0.1, 0.2]])
        )


class TestTextTokenInput(BaseTest):
    def test_input(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])
        tt_input = TextTokenInput("a c d", tokenizer)

        expected_tensor = torch.tensor([[1.0] * 3])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        expected_model_inp = torch.tensor([[0, 2, tokenizer.unk_idx]])
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)

        perturbed_tensor = torch.tensor([[1.0, 0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 0, 0]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

    def test_input_with_baselines(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])

        # int baselines
        tt_input = TextTokenInput("a c d", tokenizer, baselines=1)

        perturbed_tensor = torch.tensor([[1.0, 0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 1, 1]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

        # str baselines
        tt_input = TextTokenInput("a c d", tokenizer, baselines="b")
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

    def test_input_with_skip_tokens(self) -> None:
        tokenizer = DummyTokenizer(["a", "b", "c"])

        # int skip tokens
        tt_input = TextTokenInput("a c d", tokenizer, skip_tokens=[0])

        expected_tensor = torch.tensor([[1.0] * 2])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)

        expected_model_inp = torch.tensor([[0, 2, tokenizer.unk_idx]])
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)

        perturbed_tensor = torch.tensor([[0.0, 0.0]])
        expected_perturbed_inp = torch.tensor([[0, 0, 0]])
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )

        # str skip tokens
        tt_input = TextTokenInput("a c d", tokenizer, skip_tokens=["a"])
        assertTensorAlmostEqual(self, tt_input.to_tensor(), expected_tensor)
        assertTensorAlmostEqual(self, tt_input.to_model_input(), expected_model_inp)
        assertTensorAlmostEqual(
            self, tt_input.to_model_input(perturbed_tensor), expected_perturbed_inp
        )


class TestImageMaskInput(BaseTest):
    def _create_test_image(
        self,
        width: int = 10,
        height: int = 10,
        color: Tuple[int, int, int] = (255, 0, 0),
    ) -> PIL.Image.Image:
        """Helper method to create a test PIL image."""
        img_array = np.full((height, width, 3), color, dtype=np.uint8)
        return PIL.Image.fromarray(img_array)

    def _simple_processor(self, image: PIL.Image.Image) -> Dict[str, Tensor]:
        """Simple test processor that converts image to tensor."""
        img_array = np.array(image)
        return {"pixel_values": torch.from_numpy(img_array).float()}

    def test_init_without_mask(self) -> None:
        # Setup: create test image and processor
        image = self._create_test_image()

        # Execute: create ImageMaskInput without mask
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
        )

        # Assert: verify n_itp_features is 1 when no mask provided
        # When mask is None, a dummy mask with all zeros is created
        self.assertEqual(mm_input.n_itp_features, 1)
        self.assertEqual(mm_input.mask_id_to_idx, {0: 0})
        self.assertIsNotNone(mm_input.mask)
        # Verify dummy mask has all zeros
        self.assertTrue(torch.all(mm_input.mask == 0))
        # Verify dummy mask shape matches image size (height, width)
        self.assertEqual(mm_input.mask.shape, (image.size[1], image.size[0]))

    def test_init_with_mask(self) -> None:
        # Setup: create test image and mask with 2 segments
        image = self._create_test_image()
        mask = torch.zeros((10, 10), dtype=torch.int32)
        mask[:, 5:] = 1  # Split horizontally into 2 segments

        # Execute: create ImageMaskInput with mask
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
        )

        # Assert: verify n_itp_features matches number of unique mask values
        self.assertEqual(mm_input.n_itp_features, 2)
        self.assertEqual(len(mm_input.mask_id_to_idx), 2)
        self.assertIn(0, mm_input.mask_id_to_idx)
        self.assertIn(1, mm_input.mask_id_to_idx)

    def test_init_with_non_continuous_mask_ids(self) -> None:
        # Setup: create mask with non-continuous IDs (e.g., 5, 10, 15)
        image = self._create_test_image(width=15, height=10)
        mask = torch.zeros((10, 15), dtype=torch.int32)
        mask[:, 5:10] = 5
        mask[:, 10:] = 10

        # Execute: create ImageMaskInput
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
        )

        # Assert: verify mask_id_to_idx creates continuous mapping
        self.assertEqual(mm_input.n_itp_features, 3)
        self.assertEqual(len(mm_input.mask_id_to_idx), 3)
        # Verify all mask IDs are mapped to continuous indices 0, 1, 2
        mapped_indices = set(mm_input.mask_id_to_idx.values())
        self.assertEqual(mapped_indices, {0, 1, 2})

    def test_to_tensor_without_mask(self) -> None:
        # Setup: create ImageMaskInput without mask
        image = self._create_test_image()
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
        )

        # Execute: convert to tensor
        result = mm_input.to_tensor()

        # Assert: verify tensor shape and values for single feature
        expected = torch.tensor([[1.0]])
        assertTensorAlmostEqual(self, result, expected)

    def test_to_tensor_with_mask(self) -> None:
        # Setup: create ImageMaskInput with 3 segments
        image = self._create_test_image(width=15)
        mask = torch.zeros((10, 15), dtype=torch.int32)
        mask[:, 5:10] = 1
        mask[:, 10:] = 2

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
        )

        # Execute: convert to tensor
        result = mm_input.to_tensor()

        # Assert: verify tensor has correct number of features
        expected = torch.tensor([[1.0, 1.0, 1.0]])
        assertTensorAlmostEqual(self, result, expected)

    def test_to_model_input_without_perturbation(self) -> None:
        # Setup: create ImageMaskInput
        image = self._create_test_image()
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
        )

        # Execute: get model input without perturbation
        result = mm_input.to_model_input()

        # Assert: verify returns original model inputs
        self.assertIn("pixel_values", result)
        assertTensorAlmostEqual(
            self, result["pixel_values"], mm_input.original_model_inputs["pixel_values"]
        )

    def test_to_model_input_with_perturbation_no_mask_present(self) -> None:
        # Setup: create red image without mask
        image = self._create_test_image(color=(255, 0, 0))
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            baseline=(255, 255, 255),  # white baseline
        )

        # Execute: perturb with feature present (value 1)
        perturbed_tensor = torch.tensor([[1.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: image should remain red (unchanged)
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        self.assertTrue(np.all(img_array[:, :, 0] == 255))
        self.assertTrue(np.all(img_array[:, :, 1] == 0))
        self.assertTrue(np.all(img_array[:, :, 2] == 0))

    def test_to_model_input_with_perturbation_no_mask_absent(self) -> None:
        # Setup: create red image without mask
        image = self._create_test_image(color=(255, 0, 0))
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            baseline=(255, 255, 255),  # white baseline
        )

        # Execute: perturb with feature absent (value 0)
        perturbed_tensor = torch.tensor([[0.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: entire image should be white (baseline)
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        self.assertTrue(np.all(img_array == 255))

    def test_to_model_input_with_mask_partial_perturbation(self) -> None:
        # Setup: create image with 2 segments (left red, right green)
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:, :5] = [255, 0, 0]  # Left half red
        img_array[:, 5:] = [0, 255, 0]  # Right half green
        image = PIL.Image.fromarray(img_array)

        mask = torch.zeros((10, 10), dtype=torch.int32)
        mask[:, 5:] = 1  # Right half is segment 1

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
            baseline=(255, 255, 255),  # white baseline
        )

        # Execute: perturb to keep left segment (0) but remove right segment (1)
        perturbed_tensor = torch.tensor([[1.0, 0.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: left half should be red, right half should be white
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        # Left half should be red
        self.assertTrue(np.all(img_array[:, :5, 0] == 255))
        self.assertTrue(np.all(img_array[:, :5, 1] == 0))
        # Right half should be white (baseline)
        self.assertTrue(np.all(img_array[:, 5:] == 255))

    def test_to_model_input_with_custom_baselines(self) -> None:
        # Setup: create image with custom baseline color
        image = self._create_test_image(color=(255, 0, 0))
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            baseline=(0, 128, 255),  # Custom blue-ish baseline
        )

        # Execute: perturb to remove feature
        perturbed_tensor = torch.tensor([[0.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: image should have custom baseline color
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        self.assertTrue(np.all(img_array[:, :, 0] == 0))
        self.assertTrue(np.all(img_array[:, :, 1] == 128))
        self.assertTrue(np.all(img_array[:, :, 2] == 255))

    def test_format_pixel_attr_with_mask(self) -> None:
        # Setup: create ImageMaskInput with 2 segments
        image = self._create_test_image(width=10, height=5)
        mask = torch.zeros((5, 10), dtype=torch.int32)
        mask[:, 5:] = 1  # Split horizontally

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
        )

        # Execute: format attribution with different values for each segment
        attr = torch.tensor([[0.3, 0.7]])
        result = mm_input.format_pixel_attr(attr)

        # Assert: left half should have 0.3, right half should have 0.7
        self.assertEqual(result.shape, (1, 5, 10))
        assertTensorAlmostEqual(
            self, result[0, :, :5], torch.full((5, 5), 0.3)
        )  # Left half
        assertTensorAlmostEqual(
            self, result[0, :, 5:], torch.full((5, 5), 0.7)
        )  # Right half

    def test_format_pixel_attr_with_non_continuous_mask(self) -> None:
        # Setup: create mask with non-continuous IDs
        image = self._create_test_image(width=15, height=5)
        mask = torch.zeros((5, 15), dtype=torch.int32)
        mask[:, 5:10] = 10
        mask[:, 10:] = 20

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
        )

        # Execute: format attribution
        attr = torch.tensor([[0.1, 0.2, 0.3]])
        result = mm_input.format_pixel_attr(attr)

        # Assert: verify correct attribution values for each segment
        self.assertEqual(result.shape, (1, 5, 15))
        # Find which continuous index maps to which mask ID
        idx_0 = mm_input.mask_id_to_idx[0]
        idx_10 = mm_input.mask_id_to_idx[10]
        idx_20 = mm_input.mask_id_to_idx[20]

        # Verify each segment has its corresponding attribution
        segment_0_value = attr[0, idx_0].item()
        segment_10_value = attr[0, idx_10].item()
        segment_20_value = attr[0, idx_20].item()

        self.assertTrue(torch.all(result[0, :, :5] == segment_0_value))
        self.assertTrue(torch.all(result[0, :, 5:10] == segment_10_value))
        self.assertTrue(torch.all(result[0, :, 10:] == segment_20_value))

    # Tests for mask_list functionality

    def test_init_mask_list_ignores_mask(self) -> None:
        # Setup: provide both mask and mask_list
        image = self._create_test_image(width=10, height=10)
        # mask has 3 segments
        mask = torch.zeros((10, 10), dtype=torch.int32)
        mask[:, 3:7] = 1
        mask[:, 7:] = 2
        # mask_list has 2 masks
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:, :5] = True
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[:, 5:] = True

        # Execute: create ImageMaskInput with both mask and mask_list
        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask=mask,
            mask_list=[mask1, mask2],
        )

        # Assert: mask_list takes precedence, so n_itp_features should be 2
        self.assertEqual(mm_input.n_itp_features, 2)
        self.assertEqual(len(mm_input.mask_list), 2)

    def test_to_tensor_with_mask_list(self) -> None:
        # Setup: create ImageMaskInput with 3 masks
        image = self._create_test_image(width=15, height=10)
        mask1 = torch.zeros((10, 15), dtype=torch.bool)
        mask1[:, :5] = True
        mask2 = torch.zeros((10, 15), dtype=torch.bool)
        mask2[:, 5:10] = True
        mask3 = torch.zeros((10, 15), dtype=torch.bool)
        mask3[:, 10:] = True

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask_list=[mask1, mask2, mask3],
        )

        # Execute: convert to tensor
        result = mm_input.to_tensor()

        # Assert: verify tensor has correct number of features
        expected = torch.tensor([[1.0, 1.0, 1.0]])
        assertTensorAlmostEqual(self, result, expected)

    def test_to_model_input_with_mask_list(self) -> None:
        # Setup: create image with 2 halves (left red, right green)
        img_array = np.zeros((10, 10, 3), dtype=np.uint8)
        img_array[:, :5] = [255, 0, 0]  # Left half red
        img_array[:, 5:] = [0, 255, 0]  # Right half green
        image = PIL.Image.fromarray(img_array)

        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:, :5] = True  # Left half
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[:, 5:] = True  # Right half

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask_list=[mask1, mask2],
            baseline=(255, 255, 255),
        )

        # Execute: keep left half (0), remove right half (1)
        perturbed_tensor = torch.tensor([[1.0, 0.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: left half should be red, right half should be white
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        # Left half should be red
        self.assertTrue(np.all(img_array[:, :5, 0] == 255))
        self.assertTrue(np.all(img_array[:, :5, 1] == 0))
        # Right half should be white (baseline)
        self.assertTrue(np.all(img_array[:, 5:] == 255))

    def test_to_model_input_with_mask_list_overlapping(self) -> None:
        # Setup: create red image with overlapping masks
        image = self._create_test_image(color=(255, 0, 0))
        mask1 = torch.zeros((10, 10), dtype=torch.bool)
        mask1[:, :7] = True  # Left 7 columns
        mask2 = torch.zeros((10, 10), dtype=torch.bool)
        mask2[:, 3:] = True  # Right 7 columns (overlap at columns 3-6)

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask_list=[mask1, mask2],
            baseline=(255, 255, 255),
        )

        # Execute: remove first feature (mask1), keep second (mask2)
        perturbed_tensor = torch.tensor([[0.0, 1.0]])
        result = mm_input.to_model_input(perturbed_tensor)

        # Assert: left 7 columns (covered by mask1) should be white
        # even though columns 3-6 are also in mask2 (but mask1 sets them to baseline)
        img_array = result["pixel_values"].numpy().astype(np.uint8)
        self.assertTrue(np.all(img_array[:, :7] == 255))

    def test_format_pixel_attr_with_mask_list(self) -> None:
        # Setup: create ImageMaskInput with 2 non-overlapping masks
        image = self._create_test_image(width=10, height=5)
        mask1 = torch.zeros((5, 10), dtype=torch.bool)
        mask1[:, :5] = True  # Left half
        mask2 = torch.zeros((5, 10), dtype=torch.bool)
        mask2[:, 5:] = True  # Right half

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask_list=[mask1, mask2],
        )

        # Execute: format attribution
        attr = torch.tensor([[0.3, 0.7]])
        result = mm_input.format_pixel_attr(attr)

        # Assert: left half should have 0.3, right half should have 0.7
        self.assertEqual(result.shape, (1, 5, 10))
        assertTensorAlmostEqual(
            self, result[0, :, :5], torch.full((5, 5), 0.3)
        )  # Left half
        assertTensorAlmostEqual(
            self, result[0, :, 5:], torch.full((5, 5), 0.7)
        )  # Right half

    def test_format_pixel_attr_with_mask_list_overlapping(self) -> None:
        # Setup: create ImageMaskInput with overlapping masks
        image = self._create_test_image(width=10, height=5)
        mask1 = torch.zeros((5, 10), dtype=torch.bool)
        mask1[:, :7] = True  # Left 7 columns
        mask2 = torch.zeros((5, 10), dtype=torch.bool)
        mask2[:, 3:] = True  # Right 7 columns (overlap at columns 3-6)

        mm_input = ImageMaskInput(
            processor_fn=self._simple_processor,
            image=image,
            mask_list=[mask1, mask2],
        )

        # Execute: format attribution with values 0.3 for mask1, 0.5 for mask2
        attr = torch.tensor([[0.3, 0.5]])
        result = mm_input.format_pixel_attr(attr)

        # Assert: overlapping region should have summed attribution
        self.assertEqual(result.shape, (1, 5, 10))
        # Columns 0-2: only mask1 (0.3)
        assertTensorAlmostEqual(self, result[0, :, :3], torch.full((5, 3), 0.3))
        # Columns 3-6: both masks (0.3 + 0.5 = 0.8)
        assertTensorAlmostEqual(self, result[0, :, 3:7], torch.full((5, 4), 0.8))
        # Columns 7-9: only mask2 (0.5)
        assertTensorAlmostEqual(self, result[0, :, 7:], torch.full((5, 3), 0.5))
