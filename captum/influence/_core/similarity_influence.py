#!/usr/bin/env python3

# pyre-strict

import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import captum._utils.common as common
import torch
from captum._utils.av import AV
from captum.attr import LayerActivation
from captum.influence._core.influence import DataInfluence
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset

r"""
Additional helper functions to calculate similarity metrics.
"""


def euclidean_distance(test: Tensor, train: Tensor) -> Tensor:
    r"""
    Calculates the pairwise euclidean distance for batches of feature vectors.
    Tensors test and train have shape (batch_size_1, *), and (batch_size_2, *).
    Returns pairwise euclidean distance Tensor of shape (batch_size_1, batch_size_2).
    """
    similarity = torch.cdist(
        test.view(test.shape[0], -1).unsqueeze(0),
        train.view(train.shape[0], -1).unsqueeze(0),
    ).squeeze(0)
    return similarity


def cosine_similarity(test: Tensor, train: Tensor, replace_nan: int = 0) -> Tensor:
    r"""
    Calculates the pairwise cosine similarity for batches of feature vectors.
    Tensors test and train have shape (batch_size_1, *), and (batch_size_2, *).
    Returns pairwise cosine similarity Tensor of shape (batch_size_1, batch_size_2).
    """
    test = test.view(test.shape[0], -1)
    train = train.view(train.shape[0], -1)

    test_norm = torch.linalg.norm(test, ord=2, dim=1, keepdim=True)
    train_norm = torch.linalg.norm(train, ord=2, dim=1, keepdim=True)

    test = torch.where(test_norm != 0.0, test / test_norm, Tensor([replace_nan]))
    train = torch.where(train_norm != 0.0, train / train_norm, Tensor([replace_nan])).T

    similarity = torch.mm(test, train)
    return similarity


r"""
Implements abstract DataInfluence class and provides implementation details for
similarity metric-based influence computation. Similarity metrics can be used to compare
intermediate or final activation vectors of a model for different sets of input. Then,
these can be used to draw conclusions about influential instances.

Some standard similarity metrics such as dot product similarity or euclidean distance
are provided, but the user can provide any custom similarity metric as well.
"""


class SimilarityInfluence(DataInfluence):
    def __init__(
        self,
        module: Module,
        layers: Union[str, List[str]],
        influence_src_dataset: Dataset,
        activation_dir: str,
        model_id: str = "",
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        similarity_metric: Callable = cosine_similarity,
        similarity_direction: str = "max",
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        r"""
        Args:
            module (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            layers (str or list[str]): The fully qualified layer(s) for which the
                    activation vectors are computed.
            influence_src_dataset (torch.utils.data.Dataset): PyTorch Dataset that is
                    used to create a PyTorch Dataloader to iterate over the dataset and
                    its labels. This is the dataset for which we will be seeking for
                    influential instances. In most cases this is the training dataset.
            activation_dir (str): The directory of the path to store
                    and retrieve activation computations. Best practice would be to use
                    an absolute path.
            model_id (str): The name/version of the model for which layer
                    activations are being computed. Activations will be stored and
                    loaded under the subdirectory with this name if provided.
            similarity_metric (Callable): This is a callable function that computes a
                    similarity metric between two representations. For example, the
                    representations pair could be from the training and test sets.

                    This function must adhere to certain standards. The inputs should be
                    torch Tensors with shape (batch_size_i/j, feature dimensions). The
                    output Tensor should have shape (batch_size_i, batch_size_j) with
                    scalar values corresponding to the similarity metric used for each
                    pairwise combination from the two batches.

                    For example, suppose we use `batch_size_1 = 16` for iterating
                    through `influence_src_dataset`, and for the `inputs` argument
                    we pass in a Tensor with 3 examples, i.e. batch_size_2 = 3. Also,
                    suppose that our inputs and intermediate activations throughout the
                    model will have dimension (N, C, H, W). Then, the feature dimensions
                    should be flattened within this function. For example::

                        >>> av_test.shape
                        torch.Size([3, N, C, H, W])
                        >>> av_src.shape
                        torch.Size([16, N, C, H, W])
                        >>> av_test = torch.view(av_test.shape[0], -1)
                        >>> av_test.shape
                        torch.Size([3, N x C x H x W])

                    and similarly for av_src. The similarity_metric should then use
                    these flattened tensors to return the pairwise similarity matrix.
                    For example, `similarity_metric(av_test, av_src)` should return a
                    tensor of shape (3, 16).

            similarity_direction (str): Specifies the direction for selecting
                    top-k similar examples. If set to "max", the examples with
                    highest similarity scores are selected (most similar). If
                    set to "min", the examples with lowest similarity scores
                    are selected (least similar / most distant).
                    Default: "max"
            batch_size (int): Batch size for iterating through `influence_src_dataset`.
            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.
        """
        self.module = module
        # pyre-fixme[4]: Attribute must be annotated.
        self.layers = [layers] if isinstance(layers, str) else layers
        self.influence_src_dataset = influence_src_dataset
        self.activation_dir = activation_dir
        self.model_id = model_id
        self.batch_size = batch_size

        if similarity_direction == "max" or similarity_direction == "min":
            # pyre-fixme[4]: Attribute must be annotated.
            self.similarity_direction = similarity_direction
        else:
            raise ValueError(
                f"{similarity_direction} is not a valid value. "
                "Must be either 'max' or 'min'"
            )

        if similarity_metric is cosine_similarity:
            if "replace_nan" in kwargs:
                # pyre-fixme[4]: Attribute must be annotated.
                self.replace_nan = kwargs["replace_nan"]
            else:
                self.replace_nan = -2 if self.similarity_direction == "max" else 2
            similarity_metric = partial(cosine_similarity, replace_nan=self.replace_nan)

        self.similarity_metric = similarity_metric

        # pyre-fixme[4]: Attribute must be annotated.
        self.influence_src_dataloader = DataLoader(
            influence_src_dataset, batch_size, shuffle=False
        )

    def influence(  # type: ignore[override]
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        top_k: int = 1,
        additional_forward_args: Optional[Any] = None,
        load_src_from_disk: bool = True,
        **kwargs: Any,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        #  `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Dict:
        r"""
        Args:
            inputs (Tensor or tuple[Tensor, ...]): Batch of examples for which
                    influential instances are computed. They are passed to the
                    forward_func. The first dimension in `inputs` tensor or tuple
                    of tensors corresponds to the batch size. A tuple of tensors
                    is only passed in if thisis the input form that `module` accepts.
            top_k (int): The number of top-matching activations to return
            additional_forward_args (Any, optional): Additional arguments that will be
                    passed to forward_func after inputs.
            load_src_from_disk (bool): Loads activations for `influence_src_dataset`
                    where possible. Setting to False would force regeneration of
                    activations.
            load_input_from_disk (bool): Regenerates activations for inputs by default
                    and removes previous `inputs` activations that are flagged with
                    `inputs_id`. Setting to True will load prior matching inputs
                    activations. Note that this could lead to unexpected behavior if
                    `inputs_id` is not configured properly and activations are loaded
                    for a different, prior `inputs`.
            inputs_id (str): Used to identify inputs for loading activations.

            **kwargs: Additional key-value arguments that are necessary for specific
                    implementation of `DataInfluence` abstract class.

        Returns:

            influences (dict): Returns the influential instances retrieved from
                    `influence_src_dataset` for each test example represented through a
                    tensor or a tuple of tensor in `inputs`. Returned influential
                    examples are represented as dict, with keys corresponding to
                    the layer names passed in `layers`. Each value in the dict is a
                    tuple containing the indices and values for the top k similarities
                    from `influence_src_dataset` by the chosen metric. The first value
                    in the tuple corresponds to the indices corresponding to the top k
                    most similar examples, and the second value is the similarity score.
                    The batch dimension corresponds to the batch dimension of `inputs`.
                    If inputs.shape[0] == 5, then dict[`layer_name`][0].shape[0] == 5.
                    These tensors will be of shape (inputs.shape[0], top_k).
        """
        inputs_batch_size = (
            inputs[0].shape[0] if isinstance(inputs, tuple) else inputs.shape[0]
        )

        influences: Dict[str, Any] = {}

        layer_AVDatasets = AV.generate_dataset_activations(
            self.activation_dir,
            self.module,
            self.model_id,
            self.layers,
            DataLoader(self.influence_src_dataset, self.batch_size, shuffle=False),
            identifier="src",
            load_from_disk=load_src_from_disk,
            return_activations=True,
        )

        assert layer_AVDatasets is not None and not isinstance(
            layer_AVDatasets, AV.AVDataset
        )

        layer_modules = [
            common._get_module_from_name(self.module, layer) for layer in self.layers
        ]
        test_activations = LayerActivation(self.module, layer_modules).attribute(
            inputs, additional_forward_args
        )

        minmax = self.similarity_direction == "max"

        # av_inputs shape: (inputs_batch_size, *) e.g. (inputs_batch_size, N, C, H, W)
        # av_src shape: (self.batch_size, *) e.g. (self.batch_size, N, C, H, W)
        test_activations = (
            test_activations if len(self.layers) > 1 else [test_activations]
        )
        for i, (layer, layer_AVDataset) in enumerate(
            zip(self.layers, layer_AVDatasets)
        ):
            topk_val, topk_idx = torch.Tensor(), torch.Tensor().long()
            zero_acts = torch.Tensor().long()

            av_inputs = test_activations[i]
            src_loader = DataLoader(layer_AVDataset)
            for j, av_src in enumerate(src_loader):
                av_src = av_src.squeeze(0)

                similarity = self.similarity_metric(av_inputs, av_src)
                msg = (
                    "Output of custom similarity does not meet required dimensions. "
                    f"Your output has shape {similarity.shape}.\nPlease ensure the "
                    "output shape matches (inputs_batch_size, src_dataset_batch_size), "
                    f"which should be {(inputs_batch_size, self.batch_size)}."
                )
                assert similarity.shape == (inputs_batch_size, av_src.shape[0]), msg
                if hasattr(self, "replace_nan"):
                    idx = (similarity == self.replace_nan).nonzero()
                    zero_acts = torch.cat((zero_acts, idx))

                r"""
                TODO: For models that can have tuples as activations, we should
                allow similarity metrics to accept tuples, support topk selection.
                """

                topk_batch = min(top_k, self.batch_size)
                values, indices = torch.topk(
                    similarity, topk_batch, dim=1, largest=minmax
                )
                indices += int(j * self.batch_size)

                topk_val = torch.cat((topk_val, values), dim=1)
                topk_idx = torch.cat((topk_idx, indices), dim=1)

                # can modify how often to sort for efficiency? minor
                sort_idx = torch.argsort(topk_val, dim=1, descending=minmax)
                topk_val = torch.gather(topk_val, 1, sort_idx[:, :top_k])
                topk_idx = torch.gather(topk_idx, 1, sort_idx[:, :top_k])

            influences[layer] = (topk_idx, topk_val)

            if torch.numel(zero_acts != 0):
                zero_warning = (
                    f"Layer {layer} has zero-vector activations for some inputs. This "
                    "may cause undefined behavior for cosine similarity. The indices "
                    "for the offending inputs will be included under the key "
                    f"'zero_acts-{layer}' in the output dictionary. Indices are "
                    "returned as a tensor with [inputs_idx, src_dataset_idx] pairs "
                    "which may have corrupted similarity scores."
                )
                warnings.warn(
                    zero_warning,
                    RuntimeWarning,
                    stacklevel=1,
                )
                key = "-".join(["zero_acts", layer])
                influences[key] = zero_acts

        return influences
