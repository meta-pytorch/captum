# pyre-strict
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import PIL.Image
import torch
from captum._utils.typing import TokenizerLike
from captum.attr._utils.visualization import draw_mask_border, draw_mask_legend
from torch import Tensor

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes, Figure


def _scatter_itp_attr_by_mask(
    itp_attr: Tensor,
    input_shape: Tuple[int, ...],
    mask: Tensor,
) -> Tensor:
    """
    Scatter the attribution of the interpretable features to the model input shape
    by mask, if the interpretable features are the mask groups of the raw
    input elements,
    """

    # itp_attr in shape(*output_dims, n_itp_features)
    output_dims = itp_attr.shape[:-1]
    n_itp_features = itp_attr.shape[-1]

    # input_shape in shape(batch_size, *inp_feature_dims)
    # attribute in shape(*output_dims, *inp_feature_dims)
    # Current limitation in pyre with multiple variadic tuples
    attr_shape = (*output_dims, *input_shape[1:])  # pyre-ignore[60]

    expanded_feature_indices = mask.expand(attr_shape)

    if len(input_shape) > 2:
        # exclude batch_size & last of actual value
        extra_inp_dims = list(input_shape[1:-1])

        # unsqueeze itp_attr to have same number of dims as input
        # (*output_dims, 1..., 1, n_itp_features)
        # then broadcast to (*output_dims, *inp.shape[1:-1], n_itp_features)
        n_extra_dims = len(extra_inp_dims)
        # Current limitation in pyre with multiple variadic tuples
        unsqueezed_shape = (  # pyre-ignore[60]
            *output_dims,
            *(1,) * n_extra_dims,
            n_itp_features,
        )
        expanded_shape = (  # pyre-ignore[60]
            *output_dims,
            *extra_inp_dims,
            n_itp_features,
        )
        expanded_itp_attr = itp_attr.reshape(unsqueezed_shape).expand(expanded_shape)
    else:
        expanded_itp_attr = itp_attr

    # gather index must be long
    # may support int32 soon https://github.com/pytorch/pytorch/pull/151822
    if expanded_feature_indices.dtype != torch.long:
        expanded_feature_indices = expanded_feature_indices.long()

    # gather from (*output_dims, *inp.shape[1:-1], n_itp_features)
    attr = torch.gather(expanded_itp_attr, -1, expanded_feature_indices)

    return attr


class InterpretableInput(ABC):
    """
    InterpretableInput is an adapter for different kinds of model inputs to
    work in Captum's attribution methods. Generally, attribution methods of Captum
    assume the inputs are numerical PyTorch tensors whose 1st dimension must be batch
    size and each index in the rest of dimensions is an interpretable feature. But this
    is not always true in practice. First, the model may take inputs of formats other
    than tensor that also require attributions. For example, a model with encapsulated
    tokenizer can directly take string as input. Second, what is considered as
    an interpretable feature always depends on the actual application and the user's
    desire. For example, the interpretable feature of an image tensor can either be
    each pixel or some segments. For text, users may see the entire string as one
    interpretable feature or view each word as one interpretable feature. This class
    provides a place to define what is the actual model input and the corresponding
    interpretable format for attribution, and the transformation between them.
    It serves as a common interface to be used inthe attribution methods to make
    Captum understand how to perturb various inputs.

    The concept Interpretable Input mainly comes from the following two papers:

    `"Why Should I Trust You?": Explaining the Predictions of Any Classifier
    <https://arxiv.org/abs/1602.04938>`_

    `A Unified Approach to Interpreting Model Predictions
    <https://arxiv.org/abs/1705.07874>`_

    which is also referred to as interpretable representation or simplified
    input. It can be represented as a mapping function:

    .. math::
        x = h_x(x')

    where :math:`x` is the model input, which can be anything that the model consumes;
    :math:`x'` is the interpretable input used in the attribution algorithms
    (it must be a PyTorch tensor in Captum), which is often
    binary indicating the “presence” or “absence”; :math:`h_x` is the
    transformer. It is supposed to work with perturbation-based attribution methods,
    but if :math:`h_x` is differentiable, it may also be used
    in gradient-based methods.

    InterpretableInput is the abstract class defining the interface. Captum provides
    the child implementations for some common input formats,
    like text and sparse features. Users can inherit this
    class to create other types of customized input.

    (We expect to support InterpretableInput in all attribution methods, but it
    is only allowed in certain attribution classes like LLMAttribution for now.)
    """

    n_itp_features: int
    values: List[str]

    @abstractmethod
    def to_tensor(self) -> Tensor:
        """
        Return the interpretable representation of this input as a tensor

        Returns:

            itp_tensor (Tensor): interpretable tensor
        """
        pass

    @abstractmethod
    def to_model_input(self, perturbed_tensor: Optional[Tensor] = None) -> Any:
        """
        Get the (perturbed) input in the format required by the model
        based on the given (perturbed) interpretable representation.

        Args:

            perturbed_tensor (Tensor, optional): tensor of the interpretable
                    representation of this input. If it is None, assume the
                    interpretable representation is pristine and return the
                    original model input
                    Default: None.


        Returns:

            model_input (Any): model input passed to the forward function
        """
        pass

    def format_attr(self, itp_attr: Tensor) -> Tensor:
        """
        Format the attribution of the interpretable feature if needed.
        The way of formatting depends on the specific interpretable input type.
        A common use is if the interpretable features are the mask groups of the raw
        input elements, the attribution of the interpretable features can be scattered
        back to the model input shape.

        Args:

                itp_attr (Tensor): attributions of the interpretable features

        Returns:

                attr (Tensor): formatted attribution
        """
        return itp_attr


class TextTemplateInput(InterpretableInput):
    """
    TextTemplateInput is an implementation of InterpretableInput for text inputs, whose
    interpretable features are certain segments (e.g., words, phrases) of the text.
    It takes a template string (or function) to define the feature segmentats
    of the input text. Its input format to the model will be the completed text,
    while its interpretable representation will be a binary tensor of the number of
    the segment features whose values indicates if the feature is
    “presence” or “absence”.

    Args:

        template (str or Callable): template string or function that takes
                the text segments and format them into the text input for the model
        values (List[str] or Dict[str, str]): the values of the segments. it is
                the input to the template.
        baselines (List[str] or Dict[str, str] or Callable or None, optional): the
                baseline values for the segment features. If it is None, emptry string
                will be used as the baseline.
                Default: None
        mask (List[int] or Dict[str, int] or None, optional): the mask to group the
                segment features. It must be in the same format as the values
                and assign each segment a mask index. Segments with the same
                index will be seen as a single interpretable feature, which means
                they must be perturbed together and end with same attributions.
                Default: None

    Examples::

        >>> text_inp = TextTemplateInput(
        >>>     template="{} feels {} right now",
        >>>     values=["He", "depressed"],
        >>>     baselines=["It", "neutral"],
        >>> )
        >>>
        >>> text_inp.to_tensor()
        >>> # torch.tensor([[1, 1]])
        >>>
        >>> text_inp.to_model_input(torch.tensor([[0, 1]]))
        >>> # "It feels depressed right now"

    """

    values: List[str]
    dict_keys: List[str]
    baselines: Union[List[str], Callable[[], Union[List[str], Dict[str, str]]]]
    n_features: int
    n_itp_features: int
    format_fn: Callable[..., str]
    mask: Union[List[int], Dict[str, int], None]
    formatted_mask: List[int]

    def __init__(
        self,
        template: Union[str, Callable[..., str]],
        values: Union[List[str], Dict[str, str]],
        baselines: Union[
            List[str],
            Dict[str, str],
            Callable[[], Union[List[str], Dict[str, str]]],
            None,
        ] = None,
        mask: Union[List[int], Dict[str, int], None] = None,
    ) -> None:
        # convert values dict to list
        if isinstance(values, dict):
            dict_keys = list(values.keys())
            values = [values[k] for k in dict_keys]
        else:
            assert isinstance(
                values, list
            ), f"the values must be either a list or a dict, received: {type(values)}"
            dict_keys = []

        self.values = values
        self.dict_keys = dict_keys

        n_features = len(values)

        if baselines is None:
            # default baseline is to remove the element
            baselines = [""] * len(values)
        elif not callable(baselines):
            if dict_keys:
                assert isinstance(baselines, dict), (
                    "if values is a dict, the baselines must also be a dict "
                    "or a callable which return a dict, "
                    f"received: {type(baselines)}"
                )

                # convert dict to list
                baselines = [baselines[k] for k in dict_keys]
            else:
                assert isinstance(baselines, list), (
                    "if values is a list, the baselines must also be a list "
                    "or a callable which return a list, "
                    f"received: {type(baselines)}"
                )

        self.baselines = baselines

        if mask is None:
            n_itp_features = n_features
        else:
            if self.dict_keys:
                assert isinstance(mask, dict), (
                    "if values is dict, the mask must also be a dict, "
                    f"received: {type(mask)}"
                )

                # convert dict to list
                mask = [mask[k] for k in self.dict_keys]

            mask_ids = set(mask)
            mask_id_to_idx = {mid: i for i, mid in enumerate(mask_ids)}

            # internal compressed mask of continuous interpretable indices from 0
            # cannot replace original mask of ids for grouping across values externally
            self.formatted_mask = [mask_id_to_idx[mid] for mid in mask]

            n_itp_features = len(mask_ids)

        # number of raw features and intepretable features
        self.n_features = n_features
        self.n_itp_features = n_itp_features

        # preserve template before modification
        self.template = template
        if isinstance(template, str):
            template = template.format
        else:
            assert callable(template), (
                "the template must be either a string or a callable, "
                f"received: {type(template)}"
            )
            template = template
        self.format_fn = template

        self.mask = mask

    def to_tensor(self) -> torch.Tensor:
        # Interpretable representation in shape(1, n_itp_features)
        return torch.tensor([[1.0] * self.n_itp_features])

    def to_model_input(self, perturbed_tensor: Optional[Tensor] = None) -> str:
        values = list(self.values)  # clone

        if perturbed_tensor is not None:
            if callable(self.baselines):
                # a placeholder for advanced baselines
                # TODO: support callable baselines
                baselines = self.baselines()
                if self.dict_keys:
                    assert isinstance(baselines, dict), (
                        "if values is a dict and the baselines is a callable"
                        f"it must return a dict, received: {type(baselines)}"
                    )
                    baselines = [baselines[k] for k in self.dict_keys]
                else:
                    assert isinstance(baselines, list), (
                        "if values is a list and the baselines is a callable"
                        f"it must return a list, received: {type(baselines)}"
                    )
            else:
                baselines = self.baselines

            for i in range(len(values)):
                itp_idx = i
                if self.mask:
                    itp_idx = self.formatted_mask[i]

                itp_val = perturbed_tensor[0][itp_idx]

                if not itp_val:
                    values[i] = baselines[i]

        if self.dict_keys:
            dict_values = dict(zip(self.dict_keys, values))
            input_str = self.format_fn(**dict_values)
        else:
            input_str = self.format_fn(*values)

        return input_str

    def format_attr(self, itp_attr: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            return itp_attr

        device = itp_attr.device

        formatted_attr = _scatter_itp_attr_by_mask(
            itp_attr,  # shape(*output_dims, n_itp_features)
            (1, self.n_features),
            torch.tensor([self.formatted_mask], device=device),
        )
        return formatted_attr


class TextTokenInput(InterpretableInput):
    """
    TextTokenInput is an implementation of InterpretableInput for text inputs, whose
    interpretable features are the tokens of the text with respect to a given tokenizer.
    It is initiated with the string form of the input text and the corresponding
    tokenizer. Its input format to the model will be the tokenized id tensor,
    while its interpretable representation will be a binary tensor of the tokens
    whose values indicates if the token is “presence” or “absence”.

    Args:

        text (str): text string for the model
        tokenizer (Tokenizer): tokenizer of the language model
        baselines (int or str, optional): the
                baseline value for the tokens. It can be a string of the baseline token
                or an integer of the baseline token id. Common choices include unknown
                token or padding token. The default value is 0, which
                is commonly used for unknown token.
                Default: 0
        skip_tokens (List[int] or List[str], optional): the tokens to skip in the
                the input's interpretable representation. Use this argument to define
                uninterested tokens, commonly like special tokens, e.g., sos, and unk.
                It can be a list of strings of the tokens or a list of integers of the
                token ids.
                Default: None

    Examples::

        >>> text_inp = TextTokenInput("This is a test.", tokenizer)
        >>>
        >>> text_inp.to_tensor()
        >>> # the shape dependens on the tokenizer
        >>> # assuming it is broken into ["<s>", "This", "is", "a", "test", "."],
        >>> # torch.tensor([[1, 6]])
        >>>
        >>> text_inp.to_model_input(torch.tensor([[0, 1]]))
        >>> # torch.tensor([[1, 6]])

    """

    inp_tensor: Tensor
    itp_tensor: Tensor
    itp_mask: Optional[Tensor]
    values: List[str]
    tokenizer: TokenizerLike
    n_itp_features: int
    baselines: int

    def __init__(
        self,
        text: str,
        tokenizer: TokenizerLike,
        baselines: Union[int, str] = 0,  # usually UNK
        skip_tokens: Union[List[int], List[str], None] = None,
    ) -> None:
        inp_tensor = tokenizer.encode(text, return_tensors="pt")

        # input tensor into the model of token ids
        self.inp_tensor = inp_tensor
        # tensor of interpretable token ids
        self.itp_tensor = inp_tensor
        # interpretable mask
        self.itp_mask = None

        if skip_tokens:
            if isinstance(skip_tokens[0], str):
                skip_tokens = cast(List[str], skip_tokens)
                skip_tokens = tokenizer.convert_tokens_to_ids(skip_tokens)
                assert isinstance(skip_tokens, list)

            skip_token_set = set(skip_tokens)
            itp_mask = torch.zeros_like(inp_tensor)
            itp_mask.map_(inp_tensor, lambda _, v: v not in skip_token_set)
            itp_mask = itp_mask.bool()

            itp_tensor = inp_tensor[itp_mask].unsqueeze(0)

            self.itp_tensor = itp_tensor
            self.itp_mask = itp_mask

        self.skip_tokens = skip_tokens

        # features values, the tokens
        self.values = tokenizer.convert_ids_to_tokens(self.itp_tensor[0].tolist())
        self.tokenizer = tokenizer
        self.n_itp_features = len(self.values)

        self.baselines = (
            baselines
            if type(baselines) is int
            else tokenizer.convert_tokens_to_ids([baselines])[0]  # type: ignore
        )

    def to_tensor(self) -> torch.Tensor:
        # return the perturbation indicator as interpretable tensor instead of token ids
        return torch.ones_like(self.itp_tensor)

    def to_model_input(self, perturbed_tensor: Optional[Tensor] = None) -> Tensor:
        if perturbed_tensor is None:
            return self.inp_tensor

        device = perturbed_tensor.device

        perturb_mask = perturbed_tensor != 1

        # perturb_per_eval or gradient based can expand the batch dim
        expand_shape = (perturbed_tensor.size(0), -1)

        perturb_itp_tensor = self.itp_tensor.expand(*expand_shape).clone().to(device)
        perturb_itp_tensor[perturb_mask] = self.baselines

        # if no iterpretable mask, the interpretable tensor is the input tensor
        if self.itp_mask is None:
            return perturb_itp_tensor

        itp_mask = self.itp_mask.expand(*expand_shape).to(device)
        perturb_inp_tensor = self.inp_tensor.expand(*expand_shape).clone().to(device)

        perturb_inp_tensor[itp_mask] = perturb_itp_tensor.view(-1)

        return perturb_inp_tensor

    def format_attr(self, itp_attr: Tensor) -> Tensor:
        return itp_attr


class ImageMaskInput(InterpretableInput):
    """
    ImageMaskInput is an implementation of InterpretableInput for the image, whose
    interpretable features are certain image segments (e.g., groups of pixels).
    It takes an image and its corresponding segmentation masks .
    Its interpretable representation will be a binary tensor of the number of
    the image segment features whose values indicates if the image segment is
    “presence” or “absence”. By default, its model input will be the masked out image.
    But it also accepts an optional processor function, which converts the image into
    the model inputs if the model does not directly consume images.
    A typical example is MM LLM, which requires the image to be processed with
    other texts into the tokenized multi-modality tensors.

    Args:

        image (PIL.Image.Image): an opened PIL image file.
        mask (Tensor, optional): the mask to group the image pixels into
                segment features. It must be in the same shape as the image size
                and assign each pixel a mask index.
                Pixels with the same index will be seen as a single
                interpretable feature, which means they must be perturbed together
                and end with same attributions. When mask is None, the entire image is
                considered as one interpretable feature.
                Default: None
        mask_list (List[Tensor], optional): a list of binary masks to group
                the image pixels as interpretable segment features. Each mask indicates
                one feature and must be in the same shape as the image size. Value True
                means the pixel is included in the feature. Compared to mask, mask_list
                allows multiple features to be defined on the same pixel. If mask_list
                is not None, mask will be ignored.
                Default: None
        baseline (Tuple[int, int, int], optional): the baseline RGB value for
                the “absent” image pixels.
                Default: (255, 255, 255)
        processor_fn (Callable, optional): function convert the image into
                the model input. A common example is the multi-modality LLM processor
                function which take an input image to encode with any text prompt
                and outputs the inputs for the LLM

    Examples::

        >>> def processor_fn(image):
        >>>    messages = [
        >>>        {
        >>>            "role": "user",
        >>>            "content": [
        >>>                {"type": "image"},
        >>>                {
        >>>                    "type": "text",
        >>>                    "text": "Please describe the image in detail.",
        >>>                },
        >>>            ],
        >>>        }
        >>>    ]
        >>>
        >>>    prompt = processor.apply_chat_template(
        >>>        messages, add_generation_prompt=True
        >>>    )
        >>>
        >>>    return processor(
        >>>        text=prompt,
        >>>        images=image,
        >>>        return_tensors="pt",
        >>>    ).to(model.device)
        >>>
        >>> image = Image.open("test.jpg")
        >>>
        >>> # Split horizontally: left half = 0, right half = 1
        >>> mask = torch.zeros(image.size[::-1], dtype=torch.int32)
        >>> mask[:, image.size[0] // 2:] = 1
        >>>
        >>> image_mask_inp = ImageMaskInput(
        >>>     image=image,
        >>>     mask=mask,
        >>>     processor_fn=processor_fn,
        >>> )
        >>>
        >>> text_inp.to_tensor()
        >>> # torch.tensor([[1, 1]])
        >>>
        >>> text_inp.to_model_input(torch.tensor([[0, 1]]))
        >>> # model inputs where the right half of the image is masked out

    """

    image: PIL.Image.Image
    mask: Tensor
    mask_list: List[Tensor]
    baseline: Tuple[int, int, int]
    processor_fn: Callable[[PIL.Image.Image], Any]
    n_itp_features: int
    original_model_inputs: Any
    mask_id_to_idx: Dict[int, int]
    values: List[str]

    def __init__(
        self,
        image: PIL.Image.Image,
        mask: Optional[Tensor] = None,
        mask_list: Optional[List[Tensor]] = None,
        baseline: Tuple[int, int, int] = (255, 255, 255),
        processor_fn: Callable[[PIL.Image.Image], Any] = lambda x: x,
    ) -> None:
        super().__init__()

        self.processor_fn = processor_fn
        self.image = image
        self.baseline = baseline

        image_shape = (image.size[1], image.size[0])  # (height, width)

        if mask_list is not None:
            # Validate that all masks in mask_list have the correct shape
            for i, m in enumerate(mask_list):
                assert m.shape == image_shape, (
                    f"mask_list[{i}] shape {m.shape} must match "
                    f"image shape {image_shape}"
                )

            mask_ids = list(range(len(mask_list)))
            # Create a dummy mask for compatibility
            mask = torch.empty(0)
        else:
            # Create a dummy mask if None is provided
            if mask is None:
                # Create a mask with all zeros (entire image as one segment)
                mask = torch.zeros(image_shape, dtype=torch.int32)
            else:
                # Validate that mask size matches image size
                assert (
                    mask.shape == image_shape
                ), f"mask shape {mask.shape} must match image shape {image_shape}"

            mask_ids = torch.unique(mask).cpu().tolist()

            # dummy mask_list
            mask_list = []

        self.mask = mask
        self.mask_list = mask_list
        self.n_itp_features = len(mask_ids)
        self.mask_id_to_idx = {int(mid): i for i, mid in enumerate(mask_ids)}

        # temporarily for compatibility with AttributionResult
        # which use the values for plot legends
        self.values = [f"image_feature_{mid}" for mid in mask_ids]

        self.original_model_inputs = processor_fn(image)

    def to_tensor(self) -> Tensor:
        return torch.tensor([[1.0] * self.n_itp_features])

    def to_model_input(self, perturbed_tensor: Optional[Tensor] = None) -> Any:
        if perturbed_tensor is None:
            return self.original_model_inputs

        img_array = np.array(self.image)

        if self.mask_list:
            for itp_idx, feature_mask in enumerate(self.mask_list):
                if perturbed_tensor[0][itp_idx] == 0:
                    mask_positions = feature_mask.bool()
                    img_array[mask_positions] = self.baseline
        else:
            for mask_id, itp_idx in self.mask_id_to_idx.items():
                if perturbed_tensor[0][itp_idx] == 0:
                    mask_positions = self.mask == mask_id
                    img_array[mask_positions] = self.baseline

        perturbed_image = PIL.Image.fromarray(img_array.astype("uint8"))

        return self.processor_fn(perturbed_image)

    def get_mask_list(self) -> List[Tensor]:
        """
        Get the list of binary masks for each interpretable feature.
        If mask_list is provided, return it directly. Otherwise, create
        a list of binary masks from the mask tensor.

        Returns:
            List[Tensor]: list of binary masks, one for each interpretable feature
        """
        return self.mask_list or [
            self.mask == mask_id for mask_id in self.mask_id_to_idx.keys()
        ]

    def format_attr(self, itp_attr: Tensor) -> Tensor:
        """
        Attribution for interpretable image segments
        Returns: 2D tensor of shape (1, n_itp_features)
        """
        return itp_attr

    def format_pixel_attr(self, itp_attr: Tensor) -> Tensor:
        """
        Attribution for image pixels.
        When mask_list is used, attributions from overlapping features are summed.
        Returns: 3D tensor of shape (1, height, width)
        """
        device = itp_attr.device
        image_shape = (self.image.size[1], self.image.size[0])  # (height, width)

        if self.mask_list:
            # For mask_list, sum attributions from overlapping features
            output_dims = itp_attr.shape[:-1]
            attr_shape = (*output_dims, *image_shape)
            formatted_attr = torch.zeros(attr_shape, device=device)

            for itp_idx, feature_mask in enumerate(self.mask_list):
                # Get attribution value for this feature
                # itp_attr shape: (*output_dims, n_itp_features)
                feature_attr = itp_attr[..., itp_idx]
                # Expand to match image shape and add where mask is True
                mask_bool = feature_mask.bool().to(device)
                # Expand feature_attr to broadcast with mask
                expanded_attr = feature_attr.unsqueeze(-1).unsqueeze(-1)
                formatted_attr = formatted_attr + expanded_attr * mask_bool

            return formatted_attr
        else:
            # Map mask IDs to continuous indices
            formatted_mask = torch.zeros_like(self.mask, device=device)
            for mask_id, itp_idx in self.mask_id_to_idx.items():
                formatted_mask[self.mask == mask_id] = itp_idx

            formatted_attr = _scatter_itp_attr_by_mask(
                itp_attr,
                (1, *image_shape),
                formatted_mask.unsqueeze(0),
            )
            return formatted_attr

    def plot_mask_overlay(
        self, show: bool = False, border_width: int = 0, show_legends: bool = True
    ) -> Union[None, Tuple["Figure", "Axes"]]:
        """
        util to help visualize the mask segementation

        Args:
            show: If True, display the plot immediately using plt.show().
                  If False, return the figure and axes objects.
            border_width: Width of the border around each segment in pixels.
                          Set to 0 to disable borders. Default is 0.
            show_legends: If True, display the mask id for each segment at its
                          centroid. Default is True.
        """

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

        ax.imshow(self.image)
        fig.set_facecolor("white")

        # random colors for all interpretable features
        opacity = 0.6
        colors = [
            np.array([*np.random.random(3), opacity])
            for _ in range(self.n_itp_features)
        ]

        mask_list = self.get_mask_list()

        for mask, color, mid in zip(mask_list, colors, self.mask_id_to_idx.keys()):
            mask_np = mask.numpy().astype(bool)
            h, w = mask_np.shape
            mask_image = mask_np.reshape(h, w, 1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

            # Add border inside each segment using erosion
            if border_width > 0:
                border_color = np.array([*color[:3], 0.8])
                draw_mask_border(ax, mask_np, border_width, border_color)

            # Calculate centroid and display mask id
            if show_legends:
                draw_mask_legend(ax, mask_np, label=str(mid))

        ax.axis("off")

        if show:
            plt.show()
            return None
        else:
            return fig, ax
