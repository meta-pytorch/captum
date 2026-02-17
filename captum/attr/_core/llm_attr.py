# pyre-strict

import warnings
from abc import ABC
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from textwrap import dedent, shorten
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import torch
from captum._utils.typing import TokenizerLike
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.lime import Lime
from captum.attr._core.remote_provider import RemoteLLMProvider
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._utils.attribution import (
    Attribution,
    GradientAttribution,
    PerturbationAttribution,
)
from captum.attr._utils.interpretable_input import (
    ImageMaskInput,
    InterpretableInput,
    TextTemplateInput,
    TextTokenInput,
)
from captum.attr._utils.visualization import draw_mask_border, draw_mask_legend

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes, Figure
from torch import nn, Tensor

DEFAULT_GEN_ARGS: Dict[str, Any] = {
    "max_new_tokens": 25,
    "do_sample": False,
    "temperature": None,
    "top_p": None,
}


@dataclass
class LLMAttributionResult:
    """
    Data class for the return result of LLMAttribution,
    which includes the necessary properties of the attribution.
    It also provides utilities to help present and plot the result in different forms.
    """

    input_tokens: List[str]
    output_tokens: List[str]
    # pyre-ignore[13]: initialized via a property setter
    _seq_attr: Tensor
    _token_attr: Optional[Tensor] = None
    _output_probs: Optional[Tensor] = None
    inp: Optional[InterpretableInput] = None

    def __init__(
        self,
        *,
        input_tokens: List[str],
        output_tokens: List[str],
        seq_attr: npt.ArrayLike,
        token_attr: Optional[npt.ArrayLike] = None,
        output_probs: Optional[npt.ArrayLike] = None,
        inp: Optional[InterpretableInput] = None,
    ) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.seq_attr = seq_attr
        self.token_attr = token_attr
        self.output_probs = output_probs

        # optionally link to the InterpretableInput
        # to support input type specific utils
        # For future scaling, a better design may be inheritence,
        # customized Result class for specifc Input, e.g., ImageMaskLLMAttributionResult
        self.inp = inp

    @property
    def seq_attr(self) -> Tensor:
        return self._seq_attr

    @seq_attr.setter
    def seq_attr(self, seq_attr: npt.ArrayLike) -> None:
        if isinstance(seq_attr, Tensor):
            self._seq_attr = seq_attr
        else:
            self._seq_attr = torch.tensor(seq_attr)
        # IDEA: in the future we might want to support higher dim seq_attr
        # (e.g. attention w.r.t. multiple layers, gradients w.r.t. different classes)
        assert len(self._seq_attr.shape) == 1, "seq_attr must be a 1D tensor"
        assert (
            len(self.input_tokens) == self._seq_attr.shape[0]
        ), "seq_attr and input_tokens must have the same length"

    @property
    def token_attr(self) -> Optional[Tensor]:
        return self._token_attr

    @token_attr.setter
    def token_attr(self, token_attr: Optional[npt.ArrayLike]) -> None:
        if token_attr is None:
            self._token_attr = None
        elif isinstance(token_attr, Tensor):
            self._token_attr = token_attr
        else:
            self._token_attr = torch.tensor(token_attr)

        if self._token_attr is not None:
            # IDEA: in the future we might want to support higher dim seq_attr
            assert len(self._token_attr.shape) == 2, "token_attr must be a 2D tensor"
            assert self._token_attr.shape == (
                len(self.output_tokens),
                len(self.input_tokens),
            ), dedent(
                f"""\
                Expect token_attr to have shape
                {len(self.output_tokens), len(self.input_tokens)},
                got {self._token_attr.shape}
                """
            )

    @property
    def output_probs(self) -> Optional[Tensor]:
        return self._output_probs

    @output_probs.setter
    def output_probs(self, output_probs: Optional[npt.ArrayLike]) -> None:
        if output_probs is None:
            self._output_probs = None
        elif isinstance(output_probs, Tensor):
            self._output_probs = output_probs
        else:
            self._output_probs = torch.tensor(output_probs)

        if self._output_probs is not None:
            assert (
                len(self._output_probs.shape) == 1
            ), "output_probs must be a 1D tensor"
            assert (
                len(self.output_tokens) == self._output_probs.shape[0]
            ), "seq_attr and input_tokens must have the same length"

    @property
    def seq_attr_dict(self) -> Dict[str, float]:
        return {k: v for v, k in zip(self.seq_attr.cpu().tolist(), self.input_tokens)}

    def plot_token_attr(
        self, show: bool = False
    ) -> Union[None, Tuple["Figure", "Axes"]]:
        """
        Generate a matplotlib plot for visualising the attribution
        of the output tokens.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False


        Returns:
            None or tuple[Figure, Axes]: If show is True, displays the plot and returns
                None. If show is False, returns a tuple of (figure, axes) for further
                customization.
        """

        if self.token_attr is None:
            raise ValueError(
                "token_attr is None (no token-level attribution was performed), please "
                "use plot_seq_attr instead for the sequence-level attribution plot"
            )
        token_attr = self.token_attr.cpu()

        # maximum absolute attribution value
        # used as the boundary of normalization
        # always keep 0 as the mid point to differentiate pos/neg attr
        max_abs_attr_val = token_attr.abs().max().item()

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        # Hide the grid
        ax.grid(False)

        # Plot the heatmap
        data = token_attr.numpy()

        fig.set_size_inches(
            max(data.shape[1] * 1.3, 6.4), max(data.shape[0] / 2.5, 4.8)
        )

        im = ax.imshow(
            data,
            vmax=max_abs_attr_val,
            vmin=-max_abs_attr_val,
            cmap=self._get_plot_color_map(),
            aspect="auto",
        )
        fig.set_facecolor("white")

        # Create colorbar
        cbar = fig.colorbar(im, ax=ax)  # type: ignore
        cbar.ax.set_ylabel("Token Attribution", rotation=-90, va="bottom")

        # Show all ticks and label them with the respective list entries.
        shortened_tokens = [
            shorten(repr(t)[1:-1], width=50, placeholder="...")
            for t in self.input_tokens
        ]
        ax.set_xticks(np.arange(data.shape[1]), labels=shortened_tokens)
        ax.set_yticks(
            np.arange(data.shape[0]),
            labels=[repr(token)[1:-1] for token in self.output_tokens],
        )

        # Let the horizontal axes labeling appear on top.
        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")

        # Loop over the data and create a `Text` for each "pixel".
        # Change the text's color depending on the data.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                color = "black" if 0.2 < im.norm(val) < 0.8 else "white"
                im.axes.text(
                    j,
                    i,
                    "%.4f" % val,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                )

        if show:
            plt.show()
            return None  # mypy wants this
        else:
            return fig, ax

    def plot_seq_attr(self, show: bool = False) -> Union[None, Tuple["Figure", "Axes"]]:
        """
        Generate a matplotlib plot for visualising the attribution
        of the output sequence.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False


        Returns:
            None or tuple[Figure, Axes]: If show is True, displays the plot and returns
                None. If show is False, returns a tuple of (figure, axes) for further
                customization.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        data = self.seq_attr.cpu().numpy()

        fig.set_size_inches(max(data.shape[0] / 2, 6.4), max(data.shape[0] / 4, 4.8))

        shortened_tokens = [
            shorten(repr(t)[1:-1], width=50, placeholder="...")
            for t in self.input_tokens
        ]
        ax.set_xticks(range(data.shape[0]), labels=shortened_tokens)

        ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

        plt.setp(
            ax.get_xticklabels(),
            rotation=-30,
            ha="right",
            rotation_mode="anchor",
        )

        fig.set_facecolor("white")

        # pos bar
        ax.bar(
            range(data.shape[0]),
            [max(v, 0) for v in data],
            align="center",
            color="#4772b3",
        )
        # neg bar
        ax.bar(
            range(data.shape[0]),
            [min(v, 0) for v in data],
            align="center",
            color="#d0365b",
        )

        ax.set_ylabel("Sequence Attribution", rotation=90, va="bottom")

        if show:
            plt.show()
            return None  # mypy wants this
        else:
            return fig, ax

    def plot_image_heatmap(
        self,
        show: bool = False,
        target_token_pos: Union[int, tuple[int, int], None] = None,
        border_width: int = 2,
        show_legends: bool = True,
    ) -> Union[None, Tuple["Figure", "Axes"]]:
        """
        Plot the image in the input with the overlay of salience based on
        the attribution. Only available for certain multi-modality input types.

        Args:
            show (bool): whether to show the plot directly or return the figure and axis
                Default: False
            target_token_pos (int or tuple[int, int] or None): target token positions.
                Compute salience w.r.t the target tokens of the specified positions.
                If None, use the sequence attribuition. If int, use the attribution of
                the token at the given index. If tuple[int, int], like (m, n), use the
                summed token attribution of tokens from m to n (noninclusive)
                Default: None
            border_width (int): Width of the border around each mask segment in pixels.
                Set to 0 to disable borders. Only used when input has mask_list.
                Default: 2
            show_legends (bool): If True, display the mask id for each segment at its
                centroid. Only used when input has mask_list.
                Default: True


        Returns:
            None or tuple[Figure, Axes]: If show is True, displays the plot and returns
                None. If show is False, returns a tuple of (figure, axes) for further
                customization.
        """

        inp = self.inp
        if not isinstance(inp, ImageMaskInput):
            raise ValueError("plot_image_heatmap is only available for ImageMaskInput")

        import matplotlib.pyplot as plt

        if target_token_pos is None:
            attr = self.seq_attr
        else:
            if self.token_attr is None:
                raise ValueError(
                    "token_attr is None (no token-level attribution was performed), "
                    "please use target_token_pos=None for sequence-level attribution"
                )

            if isinstance(target_token_pos, int):
                attr = self.token_attr[target_token_pos]
            else:
                from_pos, to_pos = target_token_pos
                attr = self.token_attr[from_pos:to_pos].sum(dim=0)

        fig, ax = plt.subplots()
        ax.imshow(np.array(inp.image).mean(axis=2), cmap="gray")

        # Get pixel-level attribution using format_pixel_attr
        pixel_attr = inp.format_pixel_attr(attr.unsqueeze(0))
        pixel_attr = pixel_attr.squeeze(0).cpu().numpy()

        max_abs_attr_val = np.abs(pixel_attr).max()
        alpha = 0.8
        heatmap = ax.imshow(
            pixel_attr,
            vmax=max_abs_attr_val,
            vmin=-max_abs_attr_val,
            cmap=self._get_plot_color_map(),
            alpha=alpha,
        )

        cbar = fig.colorbar(heatmap, ax=ax)
        cbar.ax.set_ylabel("Attribution", rotation=-90, va="bottom")

        # Draw borders and legends on top if mask_list is available
        if hasattr(inp, "get_mask_list"):
            mask_list = [m.numpy().astype(bool) for m in inp.get_mask_list()]
            mask_ids = list(inp.mask_id_to_idx.keys())
            cmap = self._get_plot_color_map()

            # Get attribution values for border color calculation
            attr_np = attr.cpu().numpy()

            for i, mask in enumerate(mask_list):
                if not mask.any():
                    continue

                if border_width > 0:
                    # Calculate border color as a darker version of the salience color
                    norm_val = (
                        (attr_np[i] / max_abs_attr_val + 1) / 2
                        if max_abs_attr_val > 0
                        else 0.5
                    )
                    rgba = np.array(cmap(norm_val))
                    # Create darker version by multiplying RGB by 0.6
                    border_color = np.array([*(rgba[:3] * 0.7), alpha])
                    draw_mask_border(ax, mask, border_width, border_color=border_color)
                if show_legends:
                    draw_mask_legend(ax, mask, label=str(mask_ids[i]))

        fig.set_facecolor("white")
        ax.axis("off")

        if show:
            plt.show()
            return None
        else:
            return fig, ax

    def _get_plot_color_map(self) -> mcolors.LinearSegmentedColormap:
        # stays unified with the color convention used in Captum
        # https://github.com/meta-pytorch/captum/blob/master/captum/attr/_utils/visualization.py
        return mcolors.LinearSegmentedColormap.from_list(
            "RdWhGn", ["red", "white", "green"]
        )


def _clean_up_pretty_token(token: str) -> str:
    """Remove newlines and leading/trailing whitespace from token."""
    return token.replace("\n", "\\n").strip()


def _encode_with_offsets(
    txt: str,
    tokenizer: TokenizerLike,
    add_special_tokens: bool = True,
    **kwargs: Any,
) -> Tuple[List[int], List[Tuple[int, int]]]:
    enc = tokenizer(
        txt,
        return_offsets_mapping=True,
        add_special_tokens=add_special_tokens,
        **kwargs,
    )
    input_ids = cast(List[int], enc["input_ids"])
    offset_mapping = cast(List[Tuple[int, int]], enc["offset_mapping"])
    assert len(input_ids) == len(offset_mapping), (
        f"{len(input_ids)} != {len(offset_mapping)}: {txt} -> "
        f"{input_ids}, {offset_mapping}"
    )
    # For the case where offsets are not set properly (the end and start are
    # equal for all tokens - fall back on the start of the next span in the
    # offset mapping)
    offset_mapping_corrected = []
    for i, (start, end) in enumerate(offset_mapping):
        if start == end:
            if (i + 1) < len(offset_mapping):
                end = offset_mapping[i + 1][0]
            else:
                end = len(txt)
        offset_mapping_corrected.append((start, end))
    return input_ids, offset_mapping_corrected


def _convert_ids_to_pretty_tokens(
    ids: Tensor,
    tokenizer: TokenizerLike,
) -> List[str]:
    """
    Convert ids to tokens without ugly unicode characters (e.g., Ġ). See:
    https://github.com/huggingface/transformers/issues/4786 and
    https://discuss.huggingface.co/t/bpe-tokenizers-and-spaces-before-words/475/2

    This is the preferred function over tokenizer.convert_ids_to_tokens() for
    user-facing data.

    Quote from links:
    > Spaces are converted in a special character (the Ġ) in the tokenizer prior to
    > BPE splitting mostly to avoid digesting spaces since the standard BPE algorithm
    > used spaces in its process
    """
    txt = tokenizer.decode(ids)
    input_ids: Optional[List[int]] = None
    # Don't add special tokens (they're either already there, or we don't want them)
    input_ids, offset_mapping = _encode_with_offsets(
        txt, tokenizer, add_special_tokens=False
    )

    pretty_tokens = []
    end_prev = -1
    idx = 0
    for i, offset in enumerate(offset_mapping):
        start, end = offset
        if input_ids[i] != ids[idx]:
            # When the re-encoded string doesn't match the original encoding we skip
            # this token and hope for the best, falling back on a naive method. This
            # can happen when a tokenizer might add a token that corresponds to
            # a space only when add_special_tokens=False.
            warnings.warn(
                f"(i={i}, idx={idx}) input_ids[i] {input_ids[i]} != ids[idx] "
                f"{ids[idx]} (corresponding to text: {repr(txt[start:end])}). "
                "Skipping this token.",
                stacklevel=2,
            )
            continue
        pretty_tokens.append(
            _clean_up_pretty_token(txt[start:end])
            + (" [OVERLAP]" if end_prev > start else "")
        )
        end_prev = end
        idx += 1
    if len(pretty_tokens) != len(ids):
        warnings.warn(
            f"Pretty tokens length {len(pretty_tokens)} != ids length {len(ids)}! "
            "Falling back to naive decoding logic.",
            stacklevel=2,
        )
        return _convert_ids_to_pretty_tokens_fallback(ids, tokenizer)
    return pretty_tokens


def _convert_ids_to_pretty_tokens_fallback(
    ids: Tensor, tokenizer: TokenizerLike
) -> List[str]:
    """
    Fallback function that naively handles logic when multiple ids map to one string.
    """
    pretty_tokens = []
    idx = 0
    while idx < len(ids):
        decoded = tokenizer.decode(ids[idx])
        decoded_pretty = _clean_up_pretty_token(decoded)
        # Handle case where single token (e.g. unicode) is split into multiple IDs
        # NOTE: This logic will fail if a tokenizer splits a token into 3+ IDs
        if decoded.strip() == "�" and tokenizer.encode(decoded) != [ids[idx]]:
            # ID at idx is split, ensure next token is also from a split
            decoded_next = tokenizer.decode(ids[idx + 1])
            if decoded_next.strip() == "�" and tokenizer.encode(decoded_next) != [
                ids[idx + 1]
            ]:
                # Both tokens are from a split, combine them
                decoded = tokenizer.decode(ids[idx : idx + 2])
                pretty_tokens.append(decoded_pretty)
                pretty_tokens.append(decoded_pretty + " [OVERLAP]")
            else:
                # Treat tokens as separate
                pretty_tokens.append(decoded_pretty)
                pretty_tokens.append(_clean_up_pretty_token(decoded_next))
            idx += 2
        else:
            # Just a normal token
            idx += 1
            pretty_tokens.append(decoded_pretty)
    return pretty_tokens


class BaseLLMAttribution(Attribution, ABC):
    """Base class for LLM Attribution methods"""

    SUPPORTED_INPUTS: Tuple[Type[InterpretableInput], ...]
    SUPPORTED_METHODS: Tuple[Type[Attribution], ...]

    model: nn.Module
    tokenizer: TokenizerLike
    device: torch.device

    def __init__(
        self,
        attr_method: Attribution,
        tokenizer: TokenizerLike,
    ) -> None:
        assert isinstance(
            attr_method, self.SUPPORTED_METHODS
        ), f"{self.__class__.__name__} does not support {type(attr_method)}"

        super().__init__(attr_method.forward_func)

        # alias, we really need a model and don't support wrapper functions
        # coz we need call model.forward, model.generate, etc.
        self.model: nn.Module = cast(nn.Module, self.forward_func)

        self.tokenizer: TokenizerLike = tokenizer
        self.device: torch.device = (
            cast(torch.device, self.model.device)
            if hasattr(self.model, "device")
            else next(self.model.parameters()).device
        )

    def _get_target_tokens(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        gen_args: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"LLMAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None
            assert hasattr(self.model, "generate") and callable(self.model.generate), (
                "The model does not have recognizable generate function."
                "Target must be given for attribution"
            )
            generate_func = cast(Callable[..., Tensor], self.model.generate)

            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            model_inp = self._format_model_input(inp.to_model_input())
            input_token_len = model_inp["input_ids"].size(1)
            output_tokens = generate_func(**model_inp, **gen_args)
            target_tokens = output_tokens[0][input_token_len:]
        else:
            assert gen_args is None, "gen_args must be None when target is given"

            if isinstance(target, str):
                # skip the leading special token bos
                # but add_special_tokens may also skip tailing tokens like eos
                # will it be a problem to the reliability of attr scores?
                # in Llama4, <|eot|> is appended even with add_special_tokens=False
                # this api also limits us to hf
                # https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PythonBackend.encode.add_special_tokens
                encoded = self.tokenizer.encode(target, add_special_tokens=False)
                target_tokens = torch.tensor(encoded)
            elif isinstance(target, torch.Tensor):
                target_tokens = target
            else:
                raise TypeError(
                    "target must either be str or Tensor, but the type of target is "
                    "{}".format(type(target))
                )
        return target_tokens

    def _format_model_input(
        self, model_input: Union[str, Tensor, Mapping]
    ) -> dict[str, Any]:
        """
        Modern LLMs usually expect a series of inputs, primarily including input_ids
        This fun ensures the model input from InterpretableInput to be a dict-like
        - convert str to tokenized tensor input_ids
        - for tensor, assume it is input_ids. Wrap in a dict
        - for other dict-like, assume they are processed correctly by any processor.
          E.g., BatchFeature returned by transformers processor. Convert to dict
        """
        # return tensor(1, n_tokens)
        if isinstance(model_input, str):
            model_input = self.tokenizer.encode(model_input, return_tensors="pt")

        if isinstance(model_input, Tensor):
            input_ids = model_input.to(self.device)
            return {"input_ids": input_ids}

        assert isinstance(
            model_input, Mapping
        ), f"Invalid model input. {type(model_input)}"

        return {**model_input}


class LLMAttribution(BaseLLMAttribution):
    """
    Attribution class for large language models. It wraps a perturbation-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm, but it will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    SUPPORTED_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
        Lime,
        KernelShap,
    )
    SUPPORTED_PER_TOKEN_ATTR_METHODS = (
        FeatureAblation,
        ShapleyValueSampling,
        ShapleyValues,
    )
    SUPPORTED_INPUTS = (TextTemplateInput, TextTokenInput, ImageMaskInput)

    def __init__(
        self,
        attr_method: PerturbationAttribution,
        tokenizer: TokenizerLike,
        attr_target: str = "log_prob",  # TODO: support callable attr_target
    ) -> None:
        """
        Args:
            attr_method (Attribution): Instance of a supported perturbation attribution
                    Supported methods include FeatureAblation, ShapleyValueSampling,
                    ShapleyValues, Lime, and KernelShap. Lime and KernelShap do not
                    support per-token attribution and will only return attribution
                    for the full target sequence.
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
            attr_target (str): attribute towards log probability or probability.
                    Available values ["log_prob", "prob"]
                    Default: "log_prob"
        """

        super().__init__(attr_method, tokenizer)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method: PerturbationAttribution = copy(attr_method)
        self.include_per_token_attr: bool = isinstance(
            attr_method, self.SUPPORTED_PER_TOKEN_ATTR_METHODS
        )

        self.attr_method.forward_func = self._forward_func

        assert attr_target in (
            "log_prob",
            "prob",
        ), "attr_target should be either 'log_prob' or 'prob'"
        self.attr_target = attr_target

    def _forward_func_by_seq(
        self,
        perturbed_tensor: Union[None, Tensor],
        inp: InterpretableInput,
        target_tokens: Tensor,
    ) -> Tensor:
        """
        LLM wrapper's forward function that process the concatenated input and target
        in one run. The result logits are often not the same as ones from
        the actual auto-regression generation process which produces the target string
        token by token, due to modern LLMs' internal mechanisms like cache.
        But it's a reasonable approximation for efficiency since it only
        calls the underlying model forward once regardless of the sequence length.
        In contrast, use _forward_func_by_tokens to simulate more authentic
        generation process.
        """
        perturbed_input = self._format_model_input(inp.to_model_input(perturbed_tensor))

        input_ids = perturbed_input["input_ids"]
        target_token_tensor = target_tokens.unsqueeze(0).to(input_ids.device)
        combined_ids = torch.cat([input_ids, target_token_tensor], dim=1)

        model_inp = {
            **perturbed_input,
            "input_ids": combined_ids,
            "attention_mask": torch.ones(
                [1, combined_ids.shape[1]], dtype=torch.long, device=combined_ids.device
            ),
        }

        outputs = self.model.forward(**model_inp)
        logits = outputs.logits

        # Llama4 returns a 4D tensor (1, 1, seq_len, vocab_size), though the doc says 3D
        # the 2nd dim may be n_returns for Speculative Decoding / Medusa-style heads
        # assume the 2nd dim must be 1
        if logits.dim() == 4:
            logits = logits[:, 0]

        input_len = input_ids.shape[1]
        target_len = target_tokens.shape[0]

        # Extract logits for the positions where we predict target tokens
        # Get logits for all target positions at once: shape (1, target_len, vocab_size)
        # logits[i] predicts token[i+1], so need input_len-1 to input_len+target_len-2
        target_logits = logits[:, input_len - 1 : input_len - 1 + target_len]
        log_probs = torch.nn.functional.log_softmax(target_logits, dim=2)

        # Gather log probs for the actual target tokens: shape (target_len,)
        token_log_probs = log_probs[0, torch.arange(target_len), target_tokens].detach()
        total_log_prob = token_log_probs.sum()
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        if self.include_per_token_attr:
            target_log_probs = torch.cat(
                [total_log_prob.unsqueeze(0), token_log_probs], dim=0
            ).unsqueeze(0)
        else:
            target_log_probs = total_log_prob
        target_probs = torch.exp(target_log_probs)

        return target_probs if self.attr_target != "log_prob" else target_log_probs

    def _forward_func_by_tokens(
        self,
        perturbed_tensor: Union[None, Tensor],
        inp: InterpretableInput,
        target_tokens: Tensor,
        use_cached_outputs: bool = False,
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
    ) -> Tensor:
        """
        LLM wrapper's forward function that decode token one by one.
        This method best authentically replicate the actual generation process of
        how a model will produce the target string in practice.
        But it's slow to re-generate target tokens one by one, since each token means
        calling the underneath model forward once, while _forward_func_by_seq is
        a more efficient approximation that concatenate all target token and forward
        in one shot
        """
        perturbed_input = self._format_model_input(inp.to_model_input(perturbed_tensor))
        init_model_inp = perturbed_input

        model_inp = {**init_model_inp}

        # model's forward function kwargs modifications
        # we assume the model should extends GenerationMixin
        # need trace its generate fn to understand how to set args for model forward
        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L2252

        input_ids = model_inp["input_ids"]
        if "attention_mask" not in model_inp:
            model_inp["attention_mask"] = torch.ones(
                [1, input_ids.shape[1]], dtype=torch.long, device=input_ids.device
            )

        if use_cached_outputs:
            model_inp["cache_position"] = torch.arange(
                input_ids.shape[1], dtype=torch.int64, device=input_ids.device
            )
            model_inp["use_cache"] = True
        else:
            model_inp["use_cache"] = False

        log_prob_list: List[Tensor] = []
        outputs = None
        for target_token in target_tokens:
            if use_cached_outputs:
                if outputs is not None:
                    # nn.Module typing suggests non-base attributes are modules or
                    # tensors
                    _update_model_kwargs_for_generation = cast(
                        Callable[..., Dict[str, object]],
                        self.model._update_model_kwargs_for_generation,
                    )
                    model_inp = _update_model_kwargs_for_generation(  # type: ignore
                        outputs, model_inp
                    )
                # nn.Module typing suggests non-base attributes are modules or tensors
                prep_inputs_for_generation = cast(
                    Callable[..., Dict[str, object]],
                    self.model.prepare_inputs_for_generation,
                )
                model_inputs = prep_inputs_for_generation(**model_inp)  # type: ignore
                outputs = self.model.forward(**model_inputs)
            else:
                # Update attention mask to adapt to input size change
                input_ids = model_inp["input_ids"]
                attention_mask = torch.ones(
                    [1, model_inp["input_ids"].shape[1]],
                    dtype=torch.long,
                    device=input_ids.device,
                )
                model_inp["attention_mask"] = attention_mask
                outputs = self.model.forward(**model_inp)

            logits = outputs.logits

            # Llama4 returns a 4D tensor (1, 1, 744, 202048), though the doc says 3D
            # https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/llama4#transformers.Llama4ForConditionalGeneration.forward
            # the 2nd dim may be n_returns for Speculative Decoding / Medusa-style heads
            # assume the 2nd dim must be 1
            if logits.dim() == 4:
                logits = logits[:, 0]

            new_token_logits = logits[:, -1]
            log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)
            log_prob_list.append(log_probs[0][target_token].detach())

            model_inp["input_ids"] = torch.cat(
                (
                    model_inp["input_ids"],
                    torch.tensor([[target_token]]).to(self.device),
                ),
                dim=1,
            )

        total_log_prob = torch.sum(torch.stack(log_prob_list), dim=0)
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        if self.include_per_token_attr:
            target_log_probs = torch.stack(
                [total_log_prob, *log_prob_list], dim=0
            ).unsqueeze(0)
        else:
            target_log_probs = total_log_prob
        target_probs = torch.exp(target_log_probs)

        if _inspect_forward:
            prompt = self.tokenizer.decode(init_model_inp["input_ids"][0])
            response = self.tokenizer.decode(target_tokens)

            # callback for externals to inspect (prompt, response, seq_prob)
            _inspect_forward(prompt, response, target_probs[0].tolist())

        return target_probs if self.attr_target != "log_prob" else target_log_probs

    def _forward_func(
        self,
        perturbed_tensor: Union[None, Tensor],
        inp: InterpretableInput,
        target_tokens: Tensor,
        use_cached_outputs: bool = False,
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
        forward_in_tokens: bool = True,
    ) -> Tensor:
        if forward_in_tokens:
            return self._forward_func_by_tokens(
                perturbed_tensor,
                inp,
                target_tokens,
                use_cached_outputs,
                _inspect_forward,
            )

        return self._forward_func_by_seq(
            perturbed_tensor,
            inp,
            target_tokens,
        )

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        num_trials: int = 1,
        gen_args: Optional[Dict[str, Any]] = None,
        use_cached_outputs: bool = True,
        # internal callback hook can be used for logging
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
        forward_in_tokens: bool = True,
        **kwargs: Any,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            num_trials (int, optional): number of trials to run. Return is the average
                    attributions over all the trials.
                    Defaults: 1.
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_new_tokens": 25, "do_sample": False,
                    "temperature": None, "top_p": None}
                    Defaults: None
            use_cached_outputs (bool, optional): whether to use cached outputs when
                    generating tokens in sequence. Only support huggingface
                    GenerationMixin, since this functionality has to depend on the
                    actual APIs of the model
                    Defaults: True.
            forward_in_tokens (bool, optional): whether to use token-by-token forward
                    or sequence-level forward. When True, it decodes tokens one by one
                    to replicate the actual generation process authentically. When
                    False, it concatenates the input and target tokens and forwards
                    them in one pass, which is more efficient but may produce slightly
                    different logits due to modern LLMs' internal mechanisms like cache.
                    Defaults: True.
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): Attribution result. token_attr will be None
                    if attr method is Lime or KernelShap.
        """
        target_tokens = self._get_target_tokens(
            inp,
            target,
            gen_args=gen_args,
        )

        attr = torch.zeros(
            [
                1 + len(target_tokens) if self.include_per_token_attr else 1,
                inp.n_itp_features,
            ],
            dtype=torch.float,
            device=self.device,
        )

        for _ in range(num_trials):
            attr_input = inp.to_tensor().to(self.device)

            cur_attr = self.attr_method.attribute(
                attr_input,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    use_cached_outputs,
                    _inspect_forward,
                    forward_in_tokens,
                ),
                **kwargs,
            )

            # temp necessary due to FA & Shapley's different return shape of multi-task
            # FA will flatten output shape internally (n_output_token, n_itp_features)
            # Shapley will keep output shape (batch, n_output_token, n_input_features)
            cur_attr = cur_attr.reshape(attr.shape)

            attr += cur_attr

        attr = attr / num_trials

        attr = inp.format_attr(attr)

        return LLMAttributionResult(
            seq_attr=attr[0],
            token_attr=(
                attr[1:] if self.include_per_token_attr else None
            ),  # shape(n_output_token, n_input_features)
            input_tokens=inp.values,
            output_tokens=_convert_ids_to_pretty_tokens(target_tokens, self.tokenizer),
            inp=inp,
        )

    def attribute_future(self) -> Callable[[], LLMAttributionResult]:
        r"""
        This method is not implemented for LLMAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for LLMAttribution"
        )


class LLMGradientAttribution(BaseLLMAttribution):
    """
    Attribution class for large language models. It wraps a gradient-based
    attribution algorthm to produce commonly interested attribution
    results for the use case of text generation.
    The wrapped instance will calculate attribution in the
    same way as configured in the original attribution algorthm,
    with respect to the log probabilities of each
    generated token and the whole sequence. It will provide a
    new "attribute" function which accepts text-based inputs
    and returns LLMAttributionResult
    """

    SUPPORTED_METHODS = (
        LayerGradientShap,
        LayerGradientXActivation,
        LayerIntegratedGradients,
    )
    SUPPORTED_INPUTS = (TextTokenInput,)

    def __init__(
        self,
        attr_method: GradientAttribution,
        tokenizer: TokenizerLike,
    ) -> None:
        """
        Args:
            attr_method (Attribution): instance of a supported perturbation attribution
                    class created with the llm model that follows huggingface style
                    interface convention
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
        """
        super().__init__(attr_method, tokenizer)

        # shallow copy is enough to avoid modifying original instance
        self.attr_method: GradientAttribution = copy(attr_method)
        self.attr_method.forward_func = GradientForwardFunc(self)

    def attribute(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        gen_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> LLMAttributionResult:
        """
        Args:
            inp (InterpretableInput): input prompt for which attributions are computed
            target (str or Tensor, optional): target response with respect to
                    which attributions are computed. If None, it uses the model
                    to generate the target based on the input and gen_args.
                    Default: None
            gen_args (dict, optional): arguments for generating the target. Only used if
                    target is not given. When None, the default arguments are used,
                    {"max_new_tokens": 25, "do_sample": False,
                    "temperature": None, "top_p": None}
                    Defaults: None
            **kwargs (Any): any extra keyword arguments passed to the call of the
                    underlying attribute function of the given attribution instance

        Returns:

            attr (LLMAttributionResult): attribution result
        """
        target_tokens = self._get_target_tokens(
            inp,
            target,
            gen_args=gen_args,
        )

        attr_inp = inp.to_tensor().to(self.device)

        attr_list = []
        for cur_target_idx, _ in enumerate(target_tokens):
            # attr in shape(batch_size, input+output_len, emb_dim)
            attr = self.attr_method.attribute(
                attr_inp,
                additional_forward_args=(
                    inp,
                    target_tokens,
                    cur_target_idx,
                ),
                **kwargs,
            ).detach()
            attr = cast(Tensor, attr)

            # will have the attr for previous output tokens
            # cut to shape(batch_size, inp_len, emb_dim)
            if cur_target_idx:
                attr = attr[:, :-cur_target_idx]

            # the author of IG uses sum
            # https://github.com/ankurtaly/Integrated-Gradients/blob/master/BertModel/bert_model_utils.py#L350
            attr = attr.sum(-1)

            attr_list.append(attr)

        # assume inp batch only has one instance
        # to shape(n_output_token, ...)
        attr = torch.cat(attr_list, dim=0)

        # grad attr method do not care the length of features in interpretable format
        # it attributes to all the elements of the output of the specified layer
        # so we need special handling for the inp type which don't care all the elements
        if isinstance(inp, TextTokenInput) and inp.itp_mask is not None:
            itp_mask = inp.itp_mask.to(attr.device)
            itp_mask = itp_mask.expand_as(attr)
            attr = attr[itp_mask].view(attr.size(0), -1)

        # for all the gradient methods we support in this class
        # the seq attr is the sum of all the token attr if the attr_target is log_prob,
        # shape(n_input_features)
        seq_attr = attr.sum(0)

        return LLMAttributionResult(
            seq_attr=seq_attr,
            token_attr=attr,  # shape(n_output_token, n_input_features)
            input_tokens=inp.values,
            output_tokens=_convert_ids_to_pretty_tokens(target_tokens, self.tokenizer),
        )

    def attribute_future(self) -> Callable[[], LLMAttributionResult]:
        r"""
        This method is not implemented for LLMGradientAttribution.
        """
        raise NotImplementedError(
            "attribute_future is not implemented for LLMGradientAttribution"
        )


class GradientForwardFunc(nn.Module):
    """
    A wrapper class for the forward function of a model in LLMGradientAttribution
    """

    def __init__(self, attr: LLMGradientAttribution) -> None:
        super().__init__()
        self.attr = attr
        self.model: nn.Module = attr.model

    def forward(
        self,
        perturbed_tensor: Tensor,
        inp: InterpretableInput,
        target_tokens: Tensor,  # 1D tensor of target token ids
        cur_target_idx: int,  # current target index
    ) -> Tensor:
        # TODO: support model that needs more than just input_ids
        perturbed_input = self.attr._format_model_input(
            inp.to_model_input(perturbed_tensor)
        )["input_ids"]

        if cur_target_idx:
            # the input batch size can be expanded by attr method
            output_token_tensor = (
                target_tokens[:cur_target_idx]
                .unsqueeze(0)
                .expand(perturbed_input.size(0), -1)
                .to(self.attr.device)
            )
            new_input_tensor = torch.cat([perturbed_input, output_token_tensor], dim=1)
        else:
            new_input_tensor = perturbed_input

        output_logits = self.model(new_input_tensor)

        new_token_logits = output_logits.logits[:, -1]
        log_probs = torch.nn.functional.log_softmax(new_token_logits, dim=1)

        target_token = target_tokens[cur_target_idx]
        token_log_probs = log_probs[..., target_token]

        # the attribution target is limited to the log probability
        return token_log_probs


class _PlaceholderModel:
    """
    Simple placeholder model that can be used with
    RemoteLLMAttribution without needing a real model.
    This can be acheived by `lambda *_:0` but BaseLLMAttribution expects
    `device`, so creating this class to set the device.
    """

    def __init__(self) -> None:
        self.device: Union[torch.device, str] = torch.device("cpu")

    def __call__(self, *args: Any, **kwargs: Any) -> int:
        return 0


class RemoteLLMAttribution(LLMAttribution):
    """
    Attribution class for large language models
    that are hosted remotely and offer logprob APIs.
    """

    placeholder_model = _PlaceholderModel()

    def __init__(
        self,
        attr_method: PerturbationAttribution,
        tokenizer: TokenizerLike,
        provider: RemoteLLMProvider,
        attr_target: str = "log_prob",
    ) -> None:
        """
        Args:
            attr_method: Instance of a supported perturbation attribution class
            tokenizer (Tokenizer): tokenizer of the llm model used in the attr_method
            provider: Remote LLM provider that implements the RemoteLLMProvider protocol
            attr_target: attribute towards log probability or probability.
                    Available values ["log_prob", "prob"]
                    Default: "log_prob"
        """
        super().__init__(
            attr_method=attr_method,
            tokenizer=tokenizer,
            attr_target=attr_target,
        )

        self.provider = provider
        self.attr_method.forward_func = self._remote_forward_func

    def _get_target_tokens(
        self,
        inp: InterpretableInput,
        target: Union[str, torch.Tensor, None] = None,
        gen_args: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """
        Get the target tokens for the remote LLM provider.
        """
        assert isinstance(
            inp, self.SUPPORTED_INPUTS
        ), f"RemoteLLMAttribution does not support input type {type(inp)}"

        if target is None:
            # generate when None with remote provider
            assert hasattr(self.provider, "generate") and callable(
                self.provider.generate
            ), (
                "The provider does not have generate function"
                " for generating target sequence."
                "Target must be given for attribution"
            )
            if not gen_args:
                gen_args = DEFAULT_GEN_ARGS

            model_inp = self._format_remote_model_input(inp.to_model_input())
            target_str = self.provider.generate(model_inp, **gen_args)
            target_tokens = self.tokenizer.encode(
                target_str, return_tensors="pt", add_special_tokens=False
            )[0]

        else:
            target_tokens = super()._get_target_tokens(inp, target, gen_args)

        return target_tokens

    def _format_remote_model_input(self, model_input: Union[str, Tensor]) -> str:
        """
        Format the model input for the remote LLM provider.
        Convert tokenized tensor to str
        to make RemoteLLMAttribution work with model inputs of both
        raw text and text token tensors
        """
        # return str input
        if isinstance(model_input, Tensor):
            return self.tokenizer.decode(model_input.flatten())
        return model_input

    def _remote_forward_func(
        self,
        perturbed_tensor: Union[None, Tensor],
        inp: InterpretableInput,
        target_tokens: Tensor,
        use_cached_outputs: bool = False,
        _inspect_forward: Optional[Callable[[str, str, List[float]], None]] = None,
        # forward_in_tokens here is for compatibility only
        # for remote, depend on the underlying LLM API
        # may not be able to support generate the exact
        # target tokens one by one (forced decoding)
        # VLLMProvider for now only support concat sequence forward
        forward_in_tokens: bool = False,
    ) -> Tensor:
        """
        Forward function for the remote LLM provider.

        Raises:
            ValueError: If the number of token logprobs doesn't match expected length
        """
        perturbed_input = self._format_remote_model_input(
            inp.to_model_input(perturbed_tensor)
        )

        target_str: str = self.tokenizer.decode(target_tokens)

        target_token_probs = self.provider.get_logprobs(
            input_prompt=perturbed_input,
            target_str=target_str,
            tokenizer=self.tokenizer,
        )

        if len(target_token_probs) != target_tokens.size()[0]:
            raise ValueError(
                f"Number of token logprobs from provider ({len(target_token_probs)}) "
                f"does not match expected target "
                f"token length ({target_tokens.size()[0]})"
            )

        log_prob_list: List[Tensor] = list(map(torch.tensor, target_token_probs))

        total_log_prob = torch.sum(torch.stack(log_prob_list), dim=0)
        # 1st element is the total prob, rest are the target tokens
        # add a leading dim for batch even we only support single instance for now
        if self.include_per_token_attr:
            target_log_probs = torch.stack(
                [total_log_prob, *log_prob_list], dim=0
            ).unsqueeze(0)
        else:
            target_log_probs = total_log_prob
        target_probs = torch.exp(target_log_probs)

        if _inspect_forward:
            prompt = perturbed_input
            response = self.tokenizer.decode(target_tokens)

            # callback for externals to inspect (prompt, response, seq_prob)
            _inspect_forward(prompt, response, target_probs[0].tolist())

        return target_probs if self.attr_target != "log_prob" else target_log_probs
