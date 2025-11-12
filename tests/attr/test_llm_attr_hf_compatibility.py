#!/usr/bin/env python3

import warnings
from typing import cast, Dict, Optional, Type

import torch
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.llm_attr import (
    _convert_ids_to_pretty_tokens,
    _convert_ids_to_pretty_tokens_fallback,
    LLMAttribution,
)
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import PerturbationAttribution
from captum.attr._utils.interpretable_input import TextTemplateInput
from captum.testing.helpers import BaseTest
from parameterized import parameterized, parameterized_class
from torch import Tensor

HAS_HF = True
try:
    # pyre-ignore[21]: Could not find a module corresponding to import `transformers`
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    HAS_HF = False


@parameterized_class(
    ("device", "use_cached_outputs"),
    (
        [("cpu", True), ("cpu", False), ("cuda", True), ("cuda", False)]
        if torch.cuda.is_available()
        else [("cpu", True), ("cpu", False)]
    ),
)
class TestLLMAttrHFCompatibility(BaseTest):
    # pyre-fixme[13]: Attribute `device` is never initialized.
    device: str
    # pyre-fixme[13]: Attribute `use_cached_outputs` is never initialized.
    use_cached_outputs: bool

    def setUp(self) -> None:
        if not HAS_HF:
            self.skipTest("transformers package not found, skipping tests")
        super().setUp()

    # pyre-fixme[56]: Pyre was not able to infer the type of argument `comprehension
    @parameterized.expand(
        [
            (
                AttrClass,
                n_samples,
            )
            for AttrClass, n_samples in zip(
                (FeatureAblation, ShapleyValueSampling, ShapleyValues),  # AttrClass
                (None, 1000, None),  # n_samples
            )
        ]
    )
    def test_llm_attr_hf_compatibility(
        self,
        AttrClass: Type[PerturbationAttribution],
        n_samples: Optional[int],
    ) -> None:
        attr_kws: Dict[str, int] = {}
        if n_samples is not None:
            attr_kws["n_samples"] = n_samples

        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        llm = AutoModelForCausalLM.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )

        llm.to(self.device)  # type: ignore[arg-type]
        llm.eval()
        llm_attr = LLMAttribution(AttrClass(llm), tokenizer)

        inp = TextTemplateInput("{} b {} {} e {}", ["a", "c", "d", "f"])
        res = llm_attr.attribute(
            inp,
            "m n o p q",
            use_cached_outputs=self.use_cached_outputs,
            # pyre-fixme[6]: In call `LLMAttribution.attribute`,
            # for 4th positional argument, expected
            # `Optional[typing.Callable[..., typing.Any]]` but got `int`.
            **attr_kws,  # type: ignore
        )
        self.assertEqual(res.seq_attr.shape, (4,))
        self.assertEqual(res.input_tokens, ["a", "c", "d", "f"])
        self.assertEqual(res.seq_attr.device.type, self.device)
        self.assertEqual(cast(Tensor, res.token_attr).device.type, self.device)


class TestTokenizerHFCompatibility(BaseTest):
    def setUp(self) -> None:
        if not HAS_HF:
            self.skipTest("transformers package not found, skipping tests")
        super().setUp()

    @parameterized.expand([(True,), (False,)])
    def test_tokenizer_pretty_print(self, add_special_tokens: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        txt = (
            'One two three\n😍\n😂\n😸\n😍\n😂\n😸\n😍\n\'😂\n😸😂\n😍😍😍😍😍\n😂:\n"😸"\n😂'
            "\n�\n\nథஐૹৣआΔΘϖ\n"
        )
        special_tokens_pretty = [
            "<s>",
            "One",
        ]
        no_special_tokens_pretty = [
            "One",
        ]
        expected_tokens_tail_pretty = [
            "two",
            "three",
            "\\n",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "\\n",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "\\n",
            "😸",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "\\n",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "\\n",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "\\n",
            "😸",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "\\n",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "\\n",
            "'",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "\\n",
            "😸",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "\\n",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "😍 [OVERLAP]",
            "\\n",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            ":",
            "\\n",
            '"',
            "😸",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            "😸 [OVERLAP]",
            '"',
            "\\n",
            "😂",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "😂 [OVERLAP]",
            "\\n",
            "�",
            "\\n",
            "\\n",
            "థ",
            "థ [OVERLAP]",
            "థ [OVERLAP]",
            "ஐ",
            "ஐ [OVERLAP]",
            "ஐ [OVERLAP]",
            "ૹ",
            "ૹ [OVERLAP]",
            "ૹ [OVERLAP]",
            "ৣ",
            "ৣ [OVERLAP]",
            "ৣ [OVERLAP]",
            "आ",
            "Δ",
            "Θ",
            "ϖ",
            "ϖ [OVERLAP]",
            "\\n",
        ]
        ids = tokenizer.encode(txt, add_special_tokens=add_special_tokens)
        head_pretty = (
            special_tokens_pretty if add_special_tokens else no_special_tokens_pretty
        )
        with warnings.catch_warnings():
            if add_special_tokens:
                # This particular tokenizer adds a token for the space after <s> when
                # we encode the decoded ids in _convert_ids_to_pretty_tokens
                warnings.filterwarnings(
                    "ignore", category=UserWarning, message=".* Skipping this token."
                )
            self.assertEqual(
                _convert_ids_to_pretty_tokens(ids, tokenizer),
                head_pretty + expected_tokens_tail_pretty,
            )

    @parameterized.expand([(True,), (False,)])
    def test_tokenizer_pretty_print_fallback(self, add_special_tokens: bool) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            "hf-internal-testing/tiny-random-LlamaForCausalLM"
        )
        txt = "Running and jumping and climbing:\nMeow meow meow"
        ids = tokenizer.encode(txt, add_special_tokens=add_special_tokens)

        special_tokens_pretty = ["<s>", "Running"]
        no_special_tokens_pretty = ["Running"]
        expected_tokens_tail_pretty = [
            "and",
            "jump",
            "ing",
            "and",
            "clim",
            "bing",
            ":",
            "\\n",
            "Me",
            "ow",
            "me",
            "ow",
            "me",
            "ow",
        ]
        head_pretty = (
            special_tokens_pretty if add_special_tokens else no_special_tokens_pretty
        )
        self.assertEqual(
            _convert_ids_to_pretty_tokens_fallback(ids, tokenizer),
            head_pretty + expected_tokens_tail_pretty,
        )
