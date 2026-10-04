#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import warnings
from functools import reduce

import torch
from torch.nn import Module


class InterpretableEmbeddingBase(Module):
    r"""
    Since some embedding vectors, e.g. word are created and assigned in
    the embedding layers of Pytorch models we need a way to access
    those layers, generate the embeddings and subtract the baseline.
    To do so, we separate embedding layers from the model, compute the
    embeddings separately and do all operations needed outside of the model.
    The original embedding layer is being replaced by
    `InterpretableEmbeddingBase` layer which passes already
    precomputed embedding vectors to the layers below.
    """

    # pyre-fixme[2]: Parameter must be annotated.
    def __init__(self, embedding, full_name) -> None:
        Module.__init__(self)
        # pyre-fixme[4]: Attribute must be annotated.
        self.num_embeddings = getattr(embedding, "num_embeddings", None)
        # pyre-fixme[4]: Attribute must be annotated.
        self.embedding_dim = getattr(embedding, "embedding_dim", None)

        # pyre-fixme[4]: Attribute must be annotated.
        self.embedding = embedding
        # pyre-fixme[4]: Attribute must be annotated.
        self.full_name = full_name

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def forward(self, *inputs, **kwargs):
        r"""
        The forward function of a wrapper embedding layer that takes and returns
        embedding layer. It allows embeddings to be created outside of the model
        and passes them seamlessly to the preceding layers of the model.

        Args:

           *inputs (Any, optional): A sequence of inputs arguments that the
                   forward function takes. Since forward functions can take any
                   type and number of arguments, this will ensure that we can
                   execute the forward pass using interpretable embedding layer.
                   Note that if inputs are specified, it is assumed that the first
                   argument is the embedding tensor generated using the
                   `self.embedding` layer using all input arguments provided in
                   `inputs` and `kwargs`.
           **kwargs (Any, optional): Similar to `inputs` we want to make sure
                   that our forward pass supports arbitrary number and type of
                   key-value arguments. If `inputs` is not provided, `kwargs` must
                   be provided and the first argument corresponds to the embedding
                   tensor generated using the `self.embedding`. Note that we make
                   here an assumption here that `kwargs` is an ordered dict which
                   is new in python 3.6 and is not guaranteed that it will
                   consistently remain that way in the newer versions. In case
                   current implementation doesn't work for special use cases,
                   it is encouraged to override `InterpretableEmbeddingBase` and
                   address those specifics in descendant classes.

        Returns:

           embedding_tensor (Tensor):
                   Returns a tensor which is the same as first argument passed
                   to the forward function.
                   It passes pre-computed embedding tensors to lower layers
                   without any modifications.
        """
        assert len(inputs) > 0 or len(kwargs) > 0, (
            "No input arguments are provided to `InterpretableEmbeddingBase`."
            "Input embedding tensor has to be provided as first argument to forward "
            "function either through inputs argument or kwargs."
        )
        return inputs[0] if len(inputs) > 0 else list(kwargs.values())[0]

    # pyre-fixme[3]: Return type must be annotated.
    # pyre-fixme[2]: Parameter must be annotated.
    def indices_to_embeddings(self, *input, **kwargs):
        r"""
        Maps indices to corresponding embedding vectors. E.g. word embeddings

        Args:

            *input (Any, optional): This can be a tensor(s) of input indices or any
                    other variable necessary to comput the embeddings. A typical
                    example of input indices are word or token indices.
            **kwargs (Any, optional): Similar to `input` this can be any sequence
                    of key-value arguments necessary to compute final embedding
                    tensor.
        Returns:

            tensor:
            A tensor of word embeddings corresponding to the
            indices specified in the input
        """
        return self.embedding(*input, **kwargs)


class TokenReferenceBase:
    r"""
    A base class for creating reference (aka baseline) tensor for a sequence of
    tokens. A typical example of such token is `PAD`. Users need to provide the
    index of the reference token in the vocabulary as an argument to
    `TokenReferenceBase` class.
    """

    def __init__(self, reference_token_idx: int = 0) -> None:
        self.reference_token_idx = reference_token_idx

    # pyre-fixme[2]: Parameter must be annotated.
    def generate_reference(self, sequence_length, device: torch.device) -> torch.Tensor:
        r"""
        Generated reference tensor of given `sequence_length` using
        `reference_token_idx`.

        Args:
            sequence_length (int): The length of the reference sequence
            device (torch.device): The device on which the reference tensor will
                          be created.
        Returns:

            tensor:
            A sequence of reference token with shape:
                          [sequence_length]
        """
        return torch.tensor([self.reference_token_idx] * sequence_length, device=device)


# pyre-fixme[3]: Return type must be annotated.
# pyre-fixme[2]: Parameter must be annotated.
def _get_deep_layer_name(obj, layer_names):
    r"""
    Traverses through the layer names that are separated by
    dot in order to access the embedding layer.
    """
    return reduce(getattr, layer_names.split("."), obj)


# pyre-fixme[2]: Parameter must be annotated.
def _set_deep_layer_value(obj, layer_names, value) -> None:
    r"""
    Traverses through the layer names that are separated by
    dot in order to access the embedding layer and update its value.
    """
    layer_names = layer_names.split(".")
    setattr(reduce(getattr, layer_names[:-1], obj), layer_names[-1], value)


def configure_interpretable_embedding_layer(
    model: Module, embedding_layer_name: str = "embedding"
) -> InterpretableEmbeddingBase:
    r"""
    This method wraps a model's embedding layer with an interpretable embedding
    layer that allows us to access the embeddings through their indices.

    Args:

        model (torch.nn.Module): An instance of PyTorch model that contains embeddings.
        embedding_layer_name (str, optional): The name of the embedding layer
                    in the `model` that we would like to make interpretable.

    Returns:

        interpretable_emb (InterpretableEmbeddingBase): An instance of
                    `InterpretableEmbeddingBase` embedding layer that wraps model's
                    embedding layer that is being accessed through
                    `embedding_layer_name`.

    Examples::

                >>> # Let's assume that we have a DocumentClassifier model that
                >>> # has a word embedding layer named 'embedding'.
                >>> # To make that layer interpretable we need to execute the
                >>> # following command:
                >>> net = DocumentClassifier()
                >>> interpretable_emb = configure_interpretable_embedding_layer(net,
                >>>    'embedding')
                >>> # then we can use interpretable embedding to convert our
                >>> # word indices into embeddings.
                >>> # Let's assume that we have the following word indices
                >>> input_indices = torch.tensor([1, 0, 2])
                >>> # we can access word embeddings for those indices with the command
                >>> # line stated below.
                >>> input_emb = interpretable_emb.indices_to_embeddings(input_indices)
                >>> # Let's assume that we want to apply integrated gradients to
                >>> # our model and that target attribution class is 3
                >>> ig = IntegratedGradients(net)
                >>> attribution = ig.attribute(input_emb, target=3)
                >>> # after we finish the interpretation we need to remove
                >>> # interpretable embedding layer with the following command:
                >>> remove_interpretable_embedding_layer(net, interpretable_emb)

    """
    embedding_layer = _get_deep_layer_name(model, embedding_layer_name)
    assert (
        embedding_layer.__class__ is not InterpretableEmbeddingBase
    ), "InterpretableEmbeddingBase has already been configured for layer {}".format(
        embedding_layer_name
    )
    warnings.warn(
        "In order to make embedding layers more interpretable they will "
        "be replaced with an interpretable embedding layer which wraps the "
        "original embedding layer and takes word embedding vectors as inputs of "
        "the forward function. This allows us to generate baselines for word "
        "embeddings and compute attributions for each embedding dimension. "
        "The original embedding layer must be set "
        "back by calling `remove_interpretable_embedding_layer` function "
        "after model interpretation is finished. ",
        stacklevel=1,
    )
    interpretable_emb = InterpretableEmbeddingBase(
        embedding_layer, embedding_layer_name
    )
    _set_deep_layer_value(model, embedding_layer_name, interpretable_emb)
    return interpretable_emb


def remove_interpretable_embedding_layer(
    model: Module, interpretable_emb: InterpretableEmbeddingBase
) -> None:
    r"""
    Removes interpretable embedding layer and sets back original
    embedding layer in the model.

    Args:

        model (torch.nn.Module): An instance of PyTorch model that contains embeddings
        interpretable_emb (InterpretableEmbeddingBase): An instance of
                    `InterpretableEmbeddingBase` that was originally created in
                    `configure_interpretable_embedding_layer` function and has
                    to be removed after interpretation is finished.

    Examples::

                >>> # Let's assume that we have a DocumentClassifier model that
                >>> # has a word embedding layer named 'embedding'.
                >>> # To make that layer interpretable we need to execute the
                >>> # following command:
                >>> net = DocumentClassifier()
                >>> interpretable_emb = configure_interpretable_embedding_layer(net,
                >>>    'embedding')
                >>> # then we can use interpretable embedding to convert our
                >>> # word indices into embeddings.
                >>> # Let's assume that we have the following word indices
                >>> input_indices = torch.tensor([1, 0, 2])
                >>> # we can access word embeddings for those indices with the command
                >>> # line stated below.
                >>> input_emb = interpretable_emb.indices_to_embeddings(input_indices)
                >>> # Let's assume that we want to apply integrated gradients to
                >>> # our model and that target attribution class is 3
                >>> ig = IntegratedGradients(net)
                >>> attribution = ig.attribute(input_emb, target=3)
                >>> # after we finish the interpretation we need to remove
                >>> # interpretable embedding layer with the following command:
                >>> remove_interpretable_embedding_layer(net, interpretable_emb)

    """
    _set_deep_layer_value(
        model, interpretable_emb.full_name, interpretable_emb.embedding
    )
