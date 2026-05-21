#!/usr/bin/env python3

# pyre-strict

import glob
import os
import re
import warnings
from typing import Any, List, Optional, Tuple, Union

import captum._utils.common as common
import torch
from captum.attr import LayerActivation
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class AV:
    r"""
    This class provides functionality to store and load activation vectors
    generated for pre-defined neural network layers.
    It also provides functionality to check if activation vectors already
    exist in the manifold and other auxiliary functions.

    This class also defines a torch `Dataset`, representing Activation Vectors,
    which enables lazy access to activation vectors and layer stored in the manifold.

    """

    r"""
        The name of the subfolder in the manifold where the activation vectors
        are stored.
    """

    class AVDataset(Dataset):
        r"""
        This dataset enables access to activation vectors for a given `model` stored
        under a pre-defined path.
        The iterator of this dataset returns a batch of data tensors.
        Additionally, subsets of the model activations can be loaded based on layer
        or identifier or num_id (representing batch number in source dataset).
        """

        def __init__(
            self,
            path: str,
            model_id: str,
            identifier: Optional[str] = None,
            layer: Optional[str] = None,
            num_id: Optional[str] = None,
        ) -> None:
            r"""
            Loads into memory the list of all activation file paths associated
            with the input `model_id`.

            Args:
                path (str): The path where the activation vectors
                        for the `layer` are stored.
                model_id (str): The name/version of the model for which layer
                        activations are being computed and stored.
                identifier (str or None): An optional identifier for the layer
                        activations. Can be used to distinguish between activations for
                        different training batches.
                layer (str or None): The layer for which the activation vectors
                        are computed.
                num_id (str): An optional string representing the batch number for
                    which the activation vectors are computed
            """

            # pyre-fixme[4]: Attribute must be annotated.
            self.av_filesearch = AV._construct_file_search(
                path, model_id, identifier, layer, num_id
            )

            files = glob.glob(self.av_filesearch)

            # pyre-fixme[4]: Attribute must be annotated.
            self.files = AV.sort_files(files)

        def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, ...]]:
            assert idx < len(self.files), "Layer index is out of bounds!"
            fl = self.files[idx]
            av = torch.load(fl, weights_only=True)
            return av

        def __len__(self) -> int:
            return len(self.files)

    AV_DIR_NAME: str = "av"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _assemble_model_dir(path: str, model_id: str) -> str:
        r"""
        Returns a directory path for the given source path `path` and `model_id.`
        This path is suffixed with the '/' delimiter.
        """
        return "/".join([path, AV.AV_DIR_NAME, model_id, ""])

    @staticmethod
    def _assemble_file_path(source_dir: str, identifier: str, layer: str) -> str:
        r"""
        Returns a full filepath given a source directory, layer, and required
        identifier. The source dir is not required to end with a "/" delimiter.
        """
        if not source_dir.endswith("/"):
            source_dir += "/"

        filepath = os.path.join(source_dir, identifier)

        filepath = os.path.join(filepath, layer)

        return filepath

    @staticmethod
    def _construct_file_search(
        source_dir: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> str:
        r"""
        Returns a search string that can be used by glob to search `source_dir/model_id`
        for the desired layer/identifier pair. Leaving `layer` as None will search ids
        over all layers, and leaving `identifier` as none will search layers over all
        ids.  Leaving both as none will return a path to glob for every activation.
        Assumes identifier is always specified when saving activations, so that
        activations live at source_dir/model_id/identifier/layer
        (and never source_dir/model_id/layer)
        """

        av_filesearch = AV._assemble_model_dir(source_dir, model_id)

        av_filesearch = os.path.join(
            av_filesearch, "*" if identifier is None else identifier
        )

        av_filesearch = os.path.join(av_filesearch, "*" if layer is None else layer)

        av_filesearch = os.path.join(
            av_filesearch, "*.pt" if num_id is None else "%s.pt" % num_id
        )

        return av_filesearch

    @staticmethod
    def exists(
        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> bool:
        r"""
        Verifies whether the model + layer activations exist
        under the path.

        Args:
            path (str): The path where the activation vectors
                    for the `model_id` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer activations.
                    Can be used to distinguish between activations for different
                    training batches. For example, the id could be a suffix composed of
                    a train/test label and numerical value, such as "-train-xxxxx".
                    The numerical id is often a monotonic sequence taken from datetime.
            layer (str or None): The layer for which the activation vectors are
                    computed.
            num_id (str): An optional string representing the batch number for which
                    the activation vectors are computed

        Returns:
            exists (bool): Indicating whether the activation vectors for the `layer`
                    and `identifier` (if provided) and num_id (if provided) were stored
                    in the manifold. If no `identifier` is provided, will return `True`
                    if any layer activation exists, whether it has an identifier or
                    not, and vice-versa.
        """
        av_dir = AV._assemble_model_dir(path, model_id)
        av_filesearch = AV._construct_file_search(
            path, model_id, identifier, layer, num_id
        )
        return os.path.exists(av_dir) and len(glob.glob(av_filesearch)) > 0

    @staticmethod
    def save(
        path: str,
        model_id: str,
        identifier: str,
        layers: Union[str, List[str]],
        act_tensors: Union[Tensor, List[Tensor]],
        num_id: str,
    ) -> None:
        r"""
        Saves the activation vectors `act_tensor` for the
        `layer` under the manifold `path`.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches. For example, the identifier could be
                    a suffix composed of a train/test label and numerical value, such
                    as "-src-abc".
                    Additionally, (abc) could be a unique identifying number. For
                    example, it is automatically created in
                    AV.generate_dataset_activations from batch index.
                    It assumes identifier is same for all layers if a list of
                    `layers` is provided.
            layers (str or list[str]): The layer(s) for which the activation vectors
                    are computed.
            act_tensors (tensor or list of tensor): A batch of activation vectors.
                    This must match the dimension of `layers`.
            num_id (str): string representing the batch number for which the activation
                    vectors are computed
        """
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(act_tensors, Tensor):
            act_tensors = [act_tensors]

        if len(layers) != len(act_tensors):
            raise ValueError("The dimension of `layers` and `act_tensors` must match!")

        av_dir = AV._assemble_model_dir(path, model_id)

        for i, layer in enumerate(layers):
            av_save_fl_path = os.path.join(
                AV._assemble_file_path(av_dir, identifier, layer), "%s.pt" % num_id
            )

            layer_dir = os.path.dirname(av_save_fl_path)
            if not os.path.exists(layer_dir):
                os.makedirs(layer_dir)
            torch.save(act_tensors[i], av_save_fl_path)

    @staticmethod
    def load(
        path: str,
        model_id: str,
        identifier: Optional[str] = None,
        layer: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> AVDataset:
        r"""
        Loads lazily the activation vectors for given `model_id` and
        `layer` saved under the `path`.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches.
            layer (str or None): The layer for which the activation vectors
                are computed.
            num_id (str): An optional string representing the batch number for which
                    the activation vectors are computed

        Returns:
            dataset (AV.AVDataset): AV.AVDataset that allows to iterate
                    over the activation vectors for given layer, identifier (if
                    provided), num_id (if provided).  Returning an AV.AVDataset as
                    opposed to a DataLoader constructed from it offers more
                    flexibility.  Raises RuntimeError if activation vectors are not
                    found.
        """

        av_save_dir = AV._assemble_model_dir(path, model_id)

        if os.path.exists(av_save_dir):
            avdataset = AV.AVDataset(path, model_id, identifier, layer, num_id)
            return avdataset
        else:
            raise RuntimeError(
                f"Activation vectors for model {model_id} was not found at path {path}"
            )

    @staticmethod
    def _manage_loading_layers(
        path: str,
        model_id: str,
        layers: Union[str, List[str]],
        load_from_disk: bool = True,
        identifier: Optional[str] = None,
        num_id: Optional[str] = None,
    ) -> List[str]:
        r"""
        Returns unsaved layers, and deletes saved layers if load_from_disk is False.

        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or list[str]): The layer(s) for which the activation vectors
                    are computed.
            load_from_disk (bool, optional): Whether or not to load from disk.
                Default: True
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches.
            num_id (str, optional): An optional string representing the batch number
                for which the activation vectors are computed.

        Returns:
            List of layer names for which activations should be generated
        """

        layers = [layers] if isinstance(layers, str) else layers
        unsaved_layers = []

        if load_from_disk:
            for layer in layers:
                if not AV.exists(path, model_id, identifier, layer, num_id):
                    unsaved_layers.append(layer)
        else:
            unsaved_layers = layers
            warnings.warn(
                "Overwriting activations: load_from_disk is set to False. Removing all "
                f"activations matching specified parameters {{path: {path}, "
                f"model_id: {model_id}, layers: {layers}, identifier: {identifier}}} "
                "before generating new activations.",
                stacklevel=1,
            )
            for layer in layers:
                files = glob.glob(
                    AV._construct_file_search(path, model_id, identifier, layer)
                )
                for filename in files:
                    os.remove(filename)

        return unsaved_layers

    @staticmethod
    def _compute_and_save_activations(
        path: str,
        model: Module,
        model_id: str,
        layers: Union[str, List[str]],
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        identifier: str,
        num_id: str,
        additional_forward_args: Optional[object] = None,
        load_from_disk: bool = True,
    ) -> None:
        r"""
        Computes layer activations for the given inputs and specified `layers`

        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            model (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or list[str]): The layer(s) for which the activation vectors
                    are computed.
            inputs (Tensor or tuple[Tensor, ...]): Batch of examples for
                    which influential instances are computed. They are passed to the
                    input `model`. The first dimension in `inputs` tensor or tuple of
                    tensors corresponds to the batch size.
            identifier (str or None): An optional identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches.
            num_id (str): An required string representing the batch number for which the
                    activation vectors are computed
            additional_forward_args (Any, optional): Additional arguments that will be
                    passed to `model` after inputs.
                    Default: None
            load_from_disk (bool): Forces function to regenerate activations if False.
                    Default: True
        """
        unsaved_layers = AV._manage_loading_layers(
            path,
            model_id,
            layers,
            load_from_disk,
            identifier,
            num_id,
        )
        layer_modules = [
            common._get_module_from_name(model, layer) for layer in unsaved_layers
        ]
        if len(unsaved_layers) > 0:
            layer_act = LayerActivation(model, layer_modules)
            new_activations = layer_act.attribute.__wrapped__(  # type: ignore
                layer_act, inputs, additional_forward_args
            )
            AV.save(path, model_id, identifier, unsaved_layers, new_activations, num_id)

    @staticmethod
    def _unpack_data(data: Union[Any, Tuple[Any, Any]]) -> Any:
        r"""
        Helper to extract input from labels when getting items from a Dataset. Assumes
        that data is either a single value, or a tuple containing two elements.
        The input could itself be a Tuple containing multiple values. If your
        dataset returns a Tuple with more than 2 elements, please reformat it such that
        all inputs are formatted into a tuple stored at the first position.
        """
        if isinstance(data, tuple) or isinstance(data, list):
            data = data[0]
        return data

    r"""TODO:
    1. Can propagate saving labels along with activations.
    2. Use of additional_forward_args when sourcing from dataset?
    """

    @staticmethod
    def generate_dataset_activations(
        path: str,
        model: Module,
        model_id: str,
        layers: Union[str, List[str]],
        dataloader: DataLoader,
        identifier: str = "default",
        load_from_disk: bool = True,
        return_activations: bool = False,
    ) -> Optional[Union[AVDataset, List[AVDataset]]]:
        r"""
        Computes layer activations for a source dataset and specified `layers`. Assumes
        that the dataset returns a single value, or a tuple containing two elements
        (see AV._unpack_data).

        Args:
            path (str): The path where the activation vectors
                    for the `layer` are stored.
            module (torch.nn.Module): An instance of pytorch model. This model should
                    define all of its layers as attributes of the model.
            model_id (str): The name/version of the model for which layer activations
                    are being computed and stored.
            layers (str or list[str]): The layer(s) for which the activation vectors
                    are computed.
            dataloader (torch.utils.data.DataLoader): DataLoader that yields Dataset
                    for which influential instances are computed. They are passed to
                    input `model`.
            identifier (str or None): An identifier for the layer
                    activations. Can be used to distinguish between activations for
                    different training batches.
                    Default: "default"
            load_from_disk (bool): Forces function to regenerate activations if False.
                    Default: True
            return_activations (bool, optional): Whether to return the activations.
                    Default: False
        Returns: If `return_activations == True`, returns a single `AVDataset` if
                    `layers` is a str, otherwise, a list of `AVDataset`s of the length
                    of `layers`, where each element corresponds to a layer.  In either
                    case, `AVDataset`'s represent the activations for a single layer,
                    over the entire `dataloader`.  If `return_activations == False`,
                    does not return anything.

        """

        unsaved_layers = AV._manage_loading_layers(
            path,
            model_id,
            layers,
            load_from_disk,
            identifier,
        )
        if len(unsaved_layers) > 0:
            for i, data in enumerate(dataloader):
                AV._compute_and_save_activations(
                    path,
                    model,
                    model_id,
                    layers,
                    AV._unpack_data(data),
                    identifier,
                    str(i),
                )

        if not return_activations:
            return None
        if isinstance(layers, str):
            return AV.load(path, model_id, identifier, layers)
        else:
            return [AV.load(path, model_id, identifier, layer) for layer in layers]

    @staticmethod
    def sort_files(files: List[str]) -> List[str]:
        r"""
        Utility for sorting files based on natural sorting instead of the default
        lexigraphical sort.
        """

        # pyre-fixme[3]: Return type must be annotated.
        # pyre-fixme[2]: Parameter must be annotated.
        def split_alphanum(s):
            r"""
            Splits string into a list of strings and numbers
                "z23a" -> ["z", 23, "a"]
            """

            return [int(x) if x.isdigit() else x for x in re.split("([0-9]+)", s)]

        return sorted(files, key=split_alphanum)
