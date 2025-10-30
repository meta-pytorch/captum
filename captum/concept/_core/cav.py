#!/usr/bin/env python3

# pyre-strict

import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from captum.concept._core.concept import Concept
from captum.concept._utils.common import concepts_to_str


class CAV:
    r"""
    Concept Activation Vector (CAV) is a vector orthogonal to the decision
    boundary of a classifier which distinguishes between activation
    vectors produced by different concepts.
    More details can be found in the paper:
        https://arxiv.org/abs/1711.11279
    """

    def __init__(
        self,
        concepts: List[Concept],
        layer: str,
        stats: Optional[Dict[str, Any]] = None,
        save_path: str = "./cav/",
        model_id: str = "default_model_id",
    ) -> None:
        r"""
        This class encapsulates the instances of CAVs objects, saves them in
        and loads them from the disk (storage).

        Args:
            concepts (list[Concept]): a List of Concept objects. Only their
                        names will be saved and loaded.
            layer (str): The layer where concept activation vectors are
                        computed using a predefined classifier.
            stats (dict, optional): a dictionary that retains information about
                        the CAV classifier such as CAV weights and accuracies.
                        Ex.: stats = {"weights": weights, "classes": classes,
                                      "accs": accs}, where "weights" are learned
                        model parameters, "classes" are a list of classes used
                        by the model to generate the "weights" and "accs"
                        the classifier training or validation accuracy.
            save_path (str, optional): The path where the CAV objects are stored.
            model_id (str, optional): A unique model identifier associated with
                        this CAV instance.
        """

        self.concepts = concepts
        self.layer = layer
        self.stats = stats
        self.save_path = save_path
        self.model_id = model_id

    @staticmethod
    def assemble_save_path(
        path: str, model_id: str, concepts: List[Concept], layer: str
    ) -> str:
        r"""
        A utility method for assembling filename and its path, from
        a concept list and a layer name.

        Args:
            path (str): A path to be concatenated with the concepts key and
                    layer name.
            model_id (str): A unique model identifier associated with input
                    `layer` and `concepts`
            concepts (list[Concept]): A list of concepts that are concatenated
                    together and used as a concept key using their ids. These
                    concept ids are retrieved from TCAV s`Concept` objects.
            layer (str): The name of the layer for which the activations are
                    computed.

        Returns:
            cav_path(str): A string containing the path where the computed CAVs
                    will be stored.
                    For example, given:
                        concept_ids = [0, 1, 2]
                        concept_names = ["striped", "random_0", "random_1"]
                        layer = "inception4c"
                        path = "/cavs",
                    the resulting save path will be:
                        "/cavs/default_model_id/0-1-2-inception4c.pkl"

        """

        file_name = concepts_to_str(concepts) + "-" + layer + ".pkl"
        return os.path.join(path, model_id, file_name)

    def save(self) -> None:
        r"""
        Saves a dictionary of the CAV computed values into a pickle file in the
        location returned by the "assemble_save_path" static methods. The
        dictionary contains the concept names list, the layer name for which
        the activations are computed for, the stats dictionary which contains
        information about the classifier train/eval statistics such as the
        weights and training accuracies. Ex.:

        save_dict = {
            "concept_ids": [0, 1, 2],
            "concept_names": ["striped", "random_0", "random_1"],
            "layer": "inception4c",
            "stats": {"weights": weights, "classes": classes, "accs": accs}
        }

        """

        save_dict = {
            "concept_ids": [c.id for c in self.concepts],
            "concept_names": [c.name for c in self.concepts],
            "layer": self.layer,
            "stats": self.stats,
        }

        cavs_path = CAV.assemble_save_path(
            self.save_path, self.model_id, self.concepts, self.layer
        )
        torch.save(save_dict, cavs_path)

    @staticmethod
    def create_cav_dir_if_missing(save_path: str, model_id: str) -> None:
        r"""
        A utility function for creating the directories where the CAVs will
        be stored. CAVs are saved in a folder under named by `model_id`
        under `save_path`.
        Args:
            save_path (str): A root path where the CAVs will be stored
            model_id (str): A unique model identifier associated with the
                    CAVs. A folder named `model_id` is created under
                    `save_path`. The CAVs are later stored there.
        """
        cav_model_id_path = os.path.join(save_path, model_id)
        if not os.path.exists(cav_model_id_path):
            os.makedirs(cav_model_id_path)

    @staticmethod
    def load(
        cavs_path: str, model_id: str, concepts: List[Concept], layer: str
    ) -> Optional["CAV"]:
        r"""
        Loads CAV dictionary from a pickle file for given input
        `layer` and `concepts`.

        Args:
            cavs_path (str): The root path where the cavs are stored
                    in the storage (on the disk).
                    Ex.: "/cavs"
            model_id (str): A unique model identifier associated with the
                    CAVs. There exist a folder named `model_id` under
                    `cavs_path` path. The CAVs are loaded from this folder.
            concepts (list[Concept]): A List of concepts for which
                    we would like to load the cavs.
            layer (str): The layer name. Ex.: "inception4c". In case of nested
                    layers we use dots to specify the depth / hierarchy.
                    Ex.: "layer.sublayer.subsublayer"

        Returns:
            cav(CAV): An instance of a CAV class, containing the respective CAV
                    score per concept and layer. An example of a path where the
                    cavs are loaded from is:
                    "/cavs/default_model_id/0-1-2-inception4c.pkl"
        """

        cavs_path = CAV.assemble_save_path(cavs_path, model_id, concepts, layer)

        if os.path.exists(cavs_path):
            ctx: AbstractContextManager[None, None]
            if hasattr(torch.serialization, "safe_globals"):
                safe_globals = [
                    # pyre-ignore[16]: Module `numpy.core.multiarray` has no attribute
                    # `_reconstruct`
                    np.core.multiarray._reconstruct,  # type: ignore[attr-defined]
                    np.ndarray,
                    np.dtype,
                ]
                if hasattr(np, "dtypes"):
                    safe_globals.extend(
                        [
                            np.dtypes.UInt32DType,  # type: ignore
                            np.dtypes.Int32DType,  # type: ignore
                        ]
                    )
                ctx = torch.serialization.safe_globals(safe_globals)
            else:
                # safe globals not in existence in this version of torch yet. Use a
                # dummy context manager instead
                ctx = nullcontext()
            with ctx:
                save_dict = torch.load(cavs_path)

            concept_names = save_dict["concept_names"]
            concept_ids = save_dict["concept_ids"]
            concepts = [
                Concept(concept_id, concept_name, None)
                for concept_id, concept_name in zip(concept_ids, concept_names)
            ]
            cav = CAV(concepts, save_dict["layer"], save_dict["stats"])

            return cav

        return None
