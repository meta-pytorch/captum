#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import glob
import os
import tempfile
import unittest
from collections import defaultdict, OrderedDict
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    Iterator,
    List,
    Set,
    Tuple,
    Union,
)

import torch
from captum._utils.av import AV
from captum._utils.common import _get_module_from_name
from captum.concept._core.concept import Concept
from captum.concept._core.tcav import TCAV
from captum.concept._utils.classifier import Classifier
from captum.concept._utils.common import concepts_to_str
from captum.concept._utils.data_iterator import dataset_to_dataloader
from captum.testing.helpers import BaseTest
from captum.testing.helpers.basic import assertTensorAlmostEqual
from captum.testing.helpers.basic_models import BasicModel_ConvNet
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset


class CustomClassifier(Classifier):
    r"""
    Wrapps a custom linear Classifier that is necessary for the
    impementation of Concept Activation Vectors (TCAVs), as described
    in the paper:
        https://arxiv.org/pdf/1711.11279.pdf

    This class simulates the output of a Linear Classifier such as
    sklearn without actually using it.

    """

    def __init__(self) -> None:
        Classifier.__init__(self)

    def train_and_eval(
        self, dataloader: DataLoader, **kwargs: Any
    ) -> Union[Dict[str, Tensor], None]:
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)
        inputs = torch.cat(inputs)
        labels = torch.cat(labels)
        # update concept ids aka classes
        # pyre-fixme[16]: `CustomClassifier` has no attribute `_classes`.
        self._classes = list(OrderedDict.fromkeys([label.item() for label in labels]))

        # Training is skipped for performance and indepenence of sklearn reasons
        _, x_test, _, y_test = train_test_split(inputs, labels)

        # A tensor with dimensions n_inputs x (1 - test_split) x n_concepts
        # should be returned here.

        # Assemble a list with size inputs.shape[0], divided in 4 quarters
        # [0, 0, 0, ... | 1, 1, 1, ... | 0, 0, 0, ... | 1, 1, 1, ... ]
        pred = [0] * x_test.shape[0]

        # Store the shape of 1/4 of inputs.shape[0] (sh_4) and use it
        sh_4 = x_test.shape[0] / 4
        for i in range(1, 4, 2):

            from_ = round(i * sh_4)
            to_ = round((i + 1) * sh_4)

            pred[from_:to_] = [1] * (round((i + 1) * sh_4) - round(i * sh_4))

        y_pred = torch.tensor(pred)
        score = y_pred == y_test
        accs = score.float().mean()

        # A hack to mock weights for two different layer
        # pyre-fixme[16]: `CustomClassifier` has no attribute `num_features`.
        self.num_features = input.shape[1]

        return {"accs": accs}

    def weights(self) -> Tensor:
        # pyre-fixme[16]: `CustomClassifier` has no attribute `num_features`.
        if self.num_features != 16:
            return torch.randn(2, self.num_features)

        return torch.tensor(
            [
                [
                    -0.2167,
                    -0.0809,
                    -0.1235,
                    -0.2450,
                    0.2954,
                    0.5409,
                    -0.2587,
                    -0.3428,
                    0.2486,
                    -0.0123,
                    0.2737,
                    0.4876,
                    -0.1133,
                    0.1616,
                    -0.2016,
                    -0.0413,
                ],
                [
                    -0.2167,
                    -0.0809,
                    -0.1235,
                    -0.2450,
                    0.2954,
                    0.5409,
                    -0.2587,
                    -0.3428,
                    0.2486,
                    -0.0123,
                    0.2737,
                    0.4876,
                    -0.1133,
                    0.2616,
                    -0.2016,
                    -0.0413,
                ],
            ],
            dtype=torch.float64,
        )

    def classes(self) -> List[int]:
        # pyre-fixme[16]: `CustomClassifier` has no attribute `_classes`.
        return self._classes


class CustomClassifier_WO_Returning_Metrics(CustomClassifier):
    def __init__(self) -> None:
        CustomClassifier.__init__(self)

    def train_and_eval(
        self, dataloader: DataLoader, **kwargs: Any
    ) -> Union[Dict[str, Tensor], None]:
        CustomClassifier.train_and_eval(self, dataloader)
        return None


class CustomClassifier_W_Flipped_Class_Id(CustomClassifier):
    def __init__(self) -> None:
        CustomClassifier.__init__(self)

    def weights(self) -> Tensor:
        _weights = CustomClassifier.weights(self)
        _weights[0], _weights[1] = _weights[1], _weights[0].clone()
        return _weights

    def classes(self) -> List[int]:
        _classes = CustomClassifier.classes(self)
        _classes[0], _classes[1] = _classes[1], _classes[0]
        return _classes


class CustomIterableDataset(IterableDataset):
    r"""
    Auxiliary class for iterating through an image dataset.
    """

    def __init__(
        self,
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        get_tensor_from_filename_func: Callable,
        path: str,
        num_samples: int = 100,
    ) -> None:
        r"""
        Args:

            path (str): Path to dataset files
        """

        self.path = path
        self.file_itr: List[str] = ["x"] * num_samples
        self.get_tensor_from_filename_func = get_tensor_from_filename_func

    def get_tensor_from_filename(self, filename: str) -> Tensor:

        return self.get_tensor_from_filename_func(filename)

    def __iter__(self) -> Iterator[Tensor]:

        mapped_itr = map(self.get_tensor_from_filename, self.file_itr)

        return mapped_itr


def train_test_split(
    x_list: Tensor, y_list: Union[Tensor, List[int]], test_split: float = 0.33
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    z_list = list(zip(x_list, y_list))
    # Split
    test_size = int(test_split * len(z_list))
    z_test, z_train = z_list[:test_size], z_list[test_size:]
    x_test, y_test = zip(*z_test)
    x_train, y_train = zip(*z_train)

    x_train = torch.stack(x_train)
    x_test = torch.stack(x_test)
    y_train = torch.stack(y_train)
    y_test = torch.stack(y_test)

    y_train[: len(y_train) // 2] = 0
    y_train[len(y_train) // 2 :] = 1

    y_test[: len(y_test) // 2] = 0
    y_test[len(y_test) // 2 :] = 1

    return x_train, x_test, y_train, y_test


def get_tensor_from_filename(filename: str) -> Tensor:

    file_tensor = (
        torch.tensor(
            [
                [
                    [
                        0.4963,
                        0.7682,
                        0.0885,
                        0.1320,
                        0.3074,
                        0.6341,
                        0.4901,
                        0.8964,
                        0.4556,
                        0.6323,
                    ],
                    [
                        0.3489,
                        0.4017,
                        0.0223,
                        0.1689,
                        0.2939,
                        0.5185,
                        0.6977,
                        0.8000,
                        0.1610,
                        0.2823,
                    ],
                    [
                        0.6816,
                        0.9152,
                        0.3971,
                        0.8742,
                        0.4194,
                        0.5529,
                        0.9527,
                        0.0362,
                        0.1852,
                        0.3734,
                    ],
                    [
                        0.3051,
                        0.9320,
                        0.1759,
                        0.2698,
                        0.1507,
                        0.0317,
                        0.2081,
                        0.9298,
                        0.7231,
                        0.7423,
                    ],
                    [
                        0.5263,
                        0.2437,
                        0.5846,
                        0.0332,
                        0.1387,
                        0.2422,
                        0.8155,
                        0.7932,
                        0.2783,
                        0.4820,
                    ],
                    [
                        0.8198,
                        0.9971,
                        0.6984,
                        0.5675,
                        0.8352,
                        0.2056,
                        0.5932,
                        0.1123,
                        0.1535,
                        0.2417,
                    ],
                    [
                        0.7262,
                        0.7011,
                        0.2038,
                        0.6511,
                        0.7745,
                        0.4369,
                        0.5191,
                        0.6159,
                        0.8102,
                        0.9801,
                    ],
                    [
                        0.1147,
                        0.3168,
                        0.6965,
                        0.9143,
                        0.9351,
                        0.9412,
                        0.5995,
                        0.0652,
                        0.5460,
                        0.1872,
                    ],
                    [
                        0.0340,
                        0.9442,
                        0.8802,
                        0.0012,
                        0.5936,
                        0.4158,
                        0.4177,
                        0.2711,
                        0.6923,
                        0.2038,
                    ],
                    [
                        0.6833,
                        0.7529,
                        0.8579,
                        0.6870,
                        0.0051,
                        0.1757,
                        0.7497,
                        0.6047,
                        0.1100,
                        0.2121,
                    ],
                ]
            ]
        )
        * 100
    )

    return file_tensor


def get_inputs_tensor() -> Tensor:

    input_tensor = torch.tensor(
        [
            [
                [
                    [
                        -1.1258e00,
                        -1.1524e00,
                        -2.5058e-01,
                        -4.3388e-01,
                        8.4871e-01,
                        6.9201e-01,
                        -3.1601e-01,
                        -2.1152e00,
                        3.2227e-01,
                        -1.2633e00,
                    ],
                    [
                        3.4998e-01,
                        3.0813e-01,
                        1.1984e-01,
                        1.2377e00,
                        1.1168e00,
                        -2.4728e-01,
                        -1.3527e00,
                        -1.6959e00,
                        5.6665e-01,
                        7.9351e-01,
                    ],
                    [
                        5.9884e-01,
                        -1.5551e00,
                        -3.4136e-01,
                        1.8530e00,
                        7.5019e-01,
                        -5.8550e-01,
                        -1.7340e-01,
                        1.8348e-01,
                        1.3894e00,
                        1.5863e00,
                    ],
                    [
                        9.4630e-01,
                        -8.4368e-01,
                        -6.1358e-01,
                        3.1593e-02,
                        -4.9268e-01,
                        2.4841e-01,
                        4.3970e-01,
                        1.1241e-01,
                        6.4079e-01,
                        4.4116e-01,
                    ],
                    [
                        -1.0231e-01,
                        7.9244e-01,
                        -2.8967e-01,
                        5.2507e-02,
                        5.2286e-01,
                        2.3022e00,
                        -1.4689e00,
                        -1.5867e00,
                        -6.7309e-01,
                        8.7283e-01,
                    ],
                    [
                        1.0554e00,
                        1.7784e-01,
                        -2.3034e-01,
                        -3.9175e-01,
                        5.4329e-01,
                        -3.9516e-01,
                        -4.4622e-01,
                        7.4402e-01,
                        1.5210e00,
                        3.4105e00,
                    ],
                    [
                        -1.5312e00,
                        -1.2341e00,
                        1.8197e00,
                        -5.5153e-01,
                        -5.6925e-01,
                        9.1997e-01,
                        1.1108e00,
                        1.2899e00,
                        -1.4782e00,
                        2.5672e00,
                    ],
                    [
                        -4.7312e-01,
                        3.3555e-01,
                        -1.6293e00,
                        -5.4974e-01,
                        -4.7983e-01,
                        -4.9968e-01,
                        -1.0670e00,
                        1.1149e00,
                        -1.4067e-01,
                        8.0575e-01,
                    ],
                    [
                        -9.3348e-02,
                        6.8705e-01,
                        -8.3832e-01,
                        8.9182e-04,
                        8.4189e-01,
                        -4.0003e-01,
                        1.0395e00,
                        3.5815e-01,
                        -2.4600e-01,
                        2.3025e00,
                    ],
                    [
                        -1.8817e00,
                        -4.9727e-02,
                        -1.0450e00,
                        -9.5650e-01,
                        3.3532e-02,
                        7.1009e-01,
                        1.6459e00,
                        -1.3602e00,
                        3.4457e-01,
                        5.1987e-01,
                    ],
                ]
            ],
            [
                [
                    [
                        -2.6133e00,
                        -1.6965e00,
                        -2.2824e-01,
                        2.7995e-01,
                        2.4693e-01,
                        7.6887e-02,
                        3.3801e-01,
                        4.5440e-01,
                        4.5694e-01,
                        -8.6537e-01,
                    ],
                    [
                        7.8131e-01,
                        -9.2679e-01,
                        -2.1883e-01,
                        -2.4351e00,
                        -7.2915e-02,
                        -3.3986e-02,
                        9.6252e-01,
                        3.4917e-01,
                        -9.2146e-01,
                        -5.6195e-02,
                    ],
                    [
                        -6.2270e-01,
                        -4.6372e-01,
                        1.9218e00,
                        -4.0255e-01,
                        1.2390e-01,
                        1.1648e00,
                        9.2337e-01,
                        1.3873e00,
                        -8.8338e-01,
                        -4.1891e-01,
                    ],
                    [
                        -8.0483e-01,
                        5.6561e-01,
                        6.1036e-01,
                        4.6688e-01,
                        1.9507e00,
                        -1.0631e00,
                        -7.7326e-02,
                        1.1640e-01,
                        -5.9399e-01,
                        -1.2439e00,
                    ],
                    [
                        -1.0209e-01,
                        -1.0335e00,
                        -3.1264e-01,
                        2.4579e-01,
                        -2.5964e-01,
                        1.1834e-01,
                        2.4396e-01,
                        1.1646e00,
                        2.8858e-01,
                        3.8660e-01,
                    ],
                    [
                        -2.0106e-01,
                        -1.1793e-01,
                        1.9220e-01,
                        -7.7216e-01,
                        -1.9003e00,
                        1.3068e-01,
                        -7.0429e-01,
                        3.1472e-01,
                        1.5739e-01,
                        3.8536e-01,
                    ],
                    [
                        9.6715e-01,
                        -9.9108e-01,
                        3.0161e-01,
                        -1.0732e-01,
                        9.9846e-01,
                        -4.9871e-01,
                        7.6111e-01,
                        6.1830e-01,
                        3.1405e-01,
                        2.1333e-01,
                    ],
                    [
                        -1.2005e-01,
                        3.6046e-01,
                        -3.1403e-01,
                        -1.0787e00,
                        2.4081e-01,
                        -1.3962e00,
                        -6.6144e-02,
                        -3.5836e-01,
                        -1.5616e00,
                        -3.5464e-01,
                    ],
                    [
                        1.0811e00,
                        1.3148e-01,
                        1.5735e00,
                        7.8143e-01,
                        -5.1107e-01,
                        -1.7137e00,
                        -5.1006e-01,
                        -4.7489e-01,
                        -6.3340e-01,
                        -1.4677e00,
                    ],
                    [
                        -8.7848e-01,
                        -2.0784e00,
                        -1.1005e00,
                        -7.2013e-01,
                        1.1931e-02,
                        3.3977e-01,
                        -2.6345e-01,
                        1.2805e00,
                        1.9395e-02,
                        -8.8080e-01,
                    ],
                ]
            ],
        ],
        requires_grad=True,
    )

    return input_tensor


def create_concept(concept_name: str, concept_id: int) -> Concept:

    concepts_path = "./dummy/concepts/" + concept_name + "/"
    dataset = CustomIterableDataset(get_tensor_from_filename, concepts_path)
    concept_iter = dataset_to_dataloader(dataset)

    concept = Concept(id=concept_id, name=concept_name, data_iter=concept_iter)

    return concept


def create_concepts() -> Dict[str, Concept]:

    # Function to create concept objects from a pre-set concept name list.

    concept_names = ["striped", "ceo", "random", "dotted"]

    concept_dict: Dict[str, Concept] = defaultdict()

    for c, concept_name in enumerate(concept_names):
        concept = create_concept(concept_name, c)
        concept_dict[concept_name] = concept

    return concept_dict


def find_concept_by_id(concepts: Set[Concept], id: int) -> Union[Concept, None]:
    for concept in concepts:
        if concept.id == id:
            return concept
    return None


def create_TCAV(
    save_path: str,
    classifier: Classifier,
    layers: Union[str, List[str]],
    attribute_to_layer_input: bool = False,
) -> TCAV:

    model = BasicModel_ConvNet()
    tcav = TCAV(
        model,
        layers,
        classifier=classifier,
        save_path=save_path,
        attribute_to_layer_input=attribute_to_layer_input,
    )
    return tcav


def init_TCAV(
    save_path: str,
    classifier: Classifier,
    layers: Union[str, List[str]],
    attribute_to_layer_input: bool = False,
) -> Tuple[TCAV, Dict[str, Concept]]:

    # Create Concepts
    concepts_dict = create_concepts()

    tcav = create_TCAV(
        save_path, classifier, layers, attribute_to_layer_input=attribute_to_layer_input
    )
    return tcav, concepts_dict


def remove_pkls(path: str) -> None:

    pkl_files = glob.glob(os.path.join(path, "*.pkl"))
    for pkl_file in pkl_files:
        os.remove(pkl_file)


class Test(BaseTest):
    r"""
    Class for testing the TCAV class through a sequence of operations:
    - Create the Concepts (random tensor generation simulation)
    - Create the TCAV class
    - Generate Activations
    - Compute the CAVs
    - Interpret (the images - simulated with random tensors)
    """

    def test_compute_cav_repeating_concept_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav = create_TCAV(tmpdirname, CustomClassifier(), "conv1")
            experimental_sets = [
                [create_concept("striped", 0), create_concept("random", 1)],
                [create_concept("ceo", 2), create_concept("striped2", 0)],
            ]
            with self.assertRaises(AssertionError):
                tcav.compute_cavs(experimental_sets)

    def test_compute_cav_repeating_concept_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav = create_TCAV(tmpdirname, CustomClassifier(), "conv1")
            experimental_sets = [
                [create_concept("striped", 0), create_concept("random", 1)],
                [create_concept("ceo", 2), create_concept("striped", 3)],
            ]
            cavs = tcav.compute_cavs(experimental_sets)
            self.assertTrue("0-1" in cavs.keys())
            self.assertTrue("2-3" in cavs.keys())

            self.assertEqual(cavs["0-1"]["conv1"].layer, "conv1")
            self.assertEqual(cavs["2-3"]["conv1"].layer, "conv1")

            self.assertEqual(cavs["0-1"]["conv1"].concepts[0].id, 0)
            self.assertEqual(cavs["0-1"]["conv1"].concepts[0].name, "striped")
            self.assertEqual(cavs["0-1"]["conv1"].concepts[1].id, 1)
            self.assertEqual(cavs["0-1"]["conv1"].concepts[1].name, "random")

            stats = cavs["0-1"]["conv1"].stats
            self.assertIsNotNone(stats)
            self.assertEqual(stats["classes"], [0, 1])  # type: ignore
            self.assertAlmostEqual(stats["accs"].item(), 0.4848, delta=0.001)  # type: ignore # noqa: E501 line too long
            self.assertEqual(list(stats["weights"].shape), [2, 128])  # type: ignore

            self.assertEqual(cavs["2-3"]["conv1"].concepts[0].id, 2)
            self.assertEqual(cavs["2-3"]["conv1"].concepts[0].name, "ceo")
            self.assertEqual(cavs["2-3"]["conv1"].concepts[1].id, 3)
            self.assertEqual(cavs["2-3"]["conv1"].concepts[1].name, "striped")

            stats = cavs["2-3"]["conv1"].stats
            self.assertIsNotNone(stats)
            self.assertEqual(stats["classes"], [2, 3])  # type: ignore
            self.assertAlmostEqual(stats["accs"].item(), 0.4848, delta=0.001)  # type: ignore # noqa: E501 line too long
            self.assertEqual(list(stats["weights"].shape), [2, 128])  # type: ignore

    def compute_cavs_interpret(
        self,
        experimental_sets: List[List[str]],
        force_train: bool,
        accs: float,
        sign_count: float,
        magnitude: float,
        processes: int = 1,
        remove_activation: bool = False,
        layers: Union[str, List[str]] = "conv2",
        attribute_to_layer_input: bool = False,
    ) -> None:
        classifier = CustomClassifier()
        self._compute_cavs_interpret(
            experimental_sets,
            force_train,
            accs,
            sign_count,
            magnitude,
            classifier,
            processes=processes,
            remove_activation=remove_activation,
            layers=layers,
            attribute_to_layer_input=attribute_to_layer_input,
        )

    def _compute_cavs_interpret(
        self,
        experimental_set_list: List[List[str]],
        force_train: bool,
        accs: Union[float, List[float]],
        sign_count: Union[float, List[float]],
        magnitude: Union[float, List[float]],
        classifier: Classifier,
        processes: int = 1,
        remove_activation: bool = False,
        layers: Union[str, List[str]] = "conv2",
        attribute_to_layer_input: bool = False,
    ) -> None:
        def wrap_in_list_if_not_already(
            input: Union[str, float, List[float], List[str]],
        ) -> Union[List[Union[float, str]], List[float], List[str]]:
            return (
                input
                if isinstance(input, list)
                else [
                    input,
                ]
            )

        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav, concept_dict = init_TCAV(
                tmpdirname,
                classifier,
                layers,
                attribute_to_layer_input=attribute_to_layer_input,
            )

            experimental_sets = self._create_experimental_sets(
                experimental_set_list, concept_dict
            )

            # Compute CAVs
            tcav.compute_cavs(
                experimental_sets,
                force_train=force_train,
                processes=processes,
            )
            concepts_key = concepts_to_str(experimental_sets[0])

            _layers: List[str] = cast(List[str], wrap_in_list_if_not_already(layers))
            _accs: List[float] = cast(List[float], wrap_in_list_if_not_already(accs))
            _sign_counts: List[float] = cast(
                List[float], wrap_in_list_if_not_already(sign_count)
            )
            _magnitudes: List[float] = cast(
                List[float], wrap_in_list_if_not_already(magnitude)
            )

            for layer, acc, sign_count, magnitude in zip(
                _layers, _accs, _sign_counts, _magnitudes
            ):
                stats = cast(Dict[str, Tensor], tcav.cavs[concepts_key][layer].stats)
                self.assertEqual(
                    stats["weights"].shape,
                    torch.Size([2, 16]),
                )

                if not isinstance(classifier, CustomClassifier_WO_Returning_Metrics):
                    self.assertAlmostEqual(
                        stats["accs"].item(),
                        acc,
                        delta=0.0001,
                    )

                # Provoking a CAV absence by deleting the .pkl files and one
                # activation
                if remove_activation:
                    remove_pkls(tmpdirname)
                    for fl in glob.glob(tmpdirname + "/av/" + layer + "/random-*-*"):
                        os.remove(fl)

                # Interpret
                inputs = 100 * get_inputs_tensor()
                scores = tcav.interpret(
                    inputs=inputs,
                    experimental_sets=experimental_sets,
                    target=0,
                    processes=processes,
                )
                self.assertAlmostEqual(
                    cast(float, scores[concepts_key][layer]["sign_count"][0].item()),
                    sign_count,
                    delta=0.0001,
                )

                self.assertAlmostEqual(
                    cast(float, scores[concepts_key][layer]["magnitude"][0].item()),
                    magnitude,
                    delta=0.0001,
                )

    def _create_experimental_sets(
        self, experimental_set_list: List[List[str]], concept_dict: Dict[str, Concept]
    ) -> List[List[Concept]]:
        experimental_sets = []
        for concept_set in experimental_set_list:
            concepts = []
            for concept in concept_set:
                self.assertTrue(concept in concept_dict)
                concepts.append(concept_dict[concept])
            experimental_sets.append(concepts)
        return experimental_sets

    # Init - Generate Activations
    def test_TCAV_1(self) -> None:

        # Create Concepts
        concepts_dict = create_concepts()
        for concept in concepts_dict.values():
            self.assertTrue(concept.data_iter is not None)
            # pyre-fixme[22]: The cast is redundant.
            data_iter = cast(DataLoader, concept.data_iter)
            self.assertEqual(
                len(cast(CustomIterableDataset, data_iter.dataset).file_itr), 100
            )
            self.assertTrue(concept.data_iter is not None)

            total_batches = 0
            for data in cast(Iterable[Tensor], concept.data_iter):
                total_batches += data.shape[0]
                self.assertEqual(data.shape[1:], torch.Size([1, 10, 10]))
            self.assertEqual(total_batches, 100)

    def test_TCAV_generate_all_activations(self) -> None:
        def forward_hook_wrapper(expected_act: Tensor) -> int:
            # pyre-fixme[2]: Parameter `module` must have a type other than `Any`.
            def forward_hook(module: Any, inp: Tensor, out=None) -> None:
                out = torch.reshape(out, (out.shape[0], -1))
                self.assertEqual(out.detach().shape[1:], expected_act.shape[1:])

            return forward_hook  # type: ignore

        with tempfile.TemporaryDirectory() as tmpdirname:
            layers = ["conv1", "conv2", "fc1", "fc2"]
            tcav, concept_dict = init_TCAV(
                tmpdirname, CustomClassifier(), layers=layers
            )
            tcav.concepts = set(concept_dict.values())

            # generating all activations for given layers and concepts
            tcav.generate_all_activations()

            # verify that all activations exist and have correct shapes
            for layer in layers:
                for _, concept in concept_dict.items():
                    self.assertTrue(
                        AV.exists(
                            tmpdirname, "default_model_id", concept.identifier, layer
                        )
                    )

                concept_meta: Dict[int, int] = defaultdict(int)
                for _, concept in concept_dict.items():
                    activations = AV.load(
                        tmpdirname, "default_model_id", concept.identifier, layer
                    )

                    # pyre-fixme[2]: Parameter `batch` has no type specified.
                    def batch_collate(batch) -> Tensor:
                        return torch.cat(batch)

                    self.assertTrue(concept.data_iter is not None)
                    assert not (activations is None)
                    for activation in cast(
                        # pyre-fixme[24]: Generic type `Iterable`
                        # expects 1 type parameter.
                        Iterable,
                        DataLoader(activations, collate_fn=batch_collate),
                    ):

                        concept_meta[concept.id] += activation.shape[0]

                        layer_module = _get_module_from_name(tcav.model, layer)

                        # pyre-fixme[24]: Generic type `Iterable`
                        # expects 1 type parameter.
                        for data in cast(Iterable, concept.data_iter):
                            hook = layer_module.register_forward_hook(
                                forward_hook_wrapper(activation)
                            )
                            tcav.model(data)
                            hook.remove()

                # asserting the length of entire dataset for each concept
                for concept_meta_i in concept_meta.values():
                    self.assertEqual(concept_meta_i, 100)

    def test_TCAV_multi_layer(self) -> None:
        concepts = [["striped", "random"], ["ceo", "random"]]
        layers = ["conv1", "conv2"]
        classifier = CustomClassifier()

        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav, concept_dict = init_TCAV(tmpdirname, classifier, layers)

            experimental_sets = self._create_experimental_sets(concepts, concept_dict)

            # Interpret
            inputs = 100 * get_inputs_tensor()
            scores = tcav.interpret(
                inputs=inputs,
                experimental_sets=experimental_sets,
                target=0,
                processes=3,
            )
            self.assertEqual(len(scores.keys()), len(experimental_sets))
            for _, tcavs in scores.items():
                for _, tcav_i in tcavs.items():
                    self.assertEqual(tcav_i["sign_count"].shape[0], 2)
                    self.assertEqual(tcav_i["magnitude"].shape[0], 2)

    def test_TCAV_interpret_preserves_experimental_set_order(self) -> None:
        class MultiConceptClassifier(CustomClassifier):
            def weights(self) -> Tensor:
                # pyre-fixme[16]: `MultiConceptClassifier` has no attribute
                #  `num_features`.
                return torch.ones(len(self.classes()), self.num_features)

        concepts = [
            ["striped", "random", "ceo"],
            ["ceo", "random"],
            ["striped", "dotted"],
            ["random", "striped", "dotted"],
        ]
        classifier = MultiConceptClassifier()

        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav, concept_dict = init_TCAV(tmpdirname, classifier, "conv2")
            experimental_sets = self._create_experimental_sets(concepts, concept_dict)

            scores = tcav.interpret(
                inputs=100 * get_inputs_tensor(),
                experimental_sets=experimental_sets,
                target=0,
                processes=1,
            )

            self.assertEqual(
                list(scores.keys()),
                [concepts_to_str(concepts) for concepts in experimental_sets],
            )

    # Force Train
    def test_TCAV_1_1_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=5,
        )

    def test_TCAV_1_1_a_wo_acc_metric(self) -> None:
        self._compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            -1.0,  # acc is not defined, this field will not be asserted
            0.5000,
            8.185208066890937e-09,
            CustomClassifier_WO_Returning_Metrics(),
            processes=2,
        )

    def test_TCAV_1_1_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"]], True, 0.4848, 0.5000, 8.185208066890937e-09
        )

    def test_TCAV_1_1_c(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"], ["striped", "ceo"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=6,
        )

    # Non-existing concept in the experimental set ("dotted")
    def test_TCAV_1_1_d(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=4,
        )

    # Force Train
    def test_TCAV_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=2,
        )

    def test_TCAV_0_1_attr_to_inputs(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=2,
            layers="relu2",
            attribute_to_layer_input=True,
        )

    # Do not Force Train
    def test_TCAV_0_0(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=2,
        )

    # Non-existing concept in the experimental set ("dotted"), do Not Force Train
    def test_TCAV_1_0_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=5,
        )

    # Do not Force Train, Missing Activation
    def test_TCAV_1_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=5,
            remove_activation=True,
        )

    # Do not run parallel:

    # Force Train
    def test_TCAV_x_1_1_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    def test_TCAV_x_1_1_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    def test_TCAV_x_1_1_c(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"], ["striped", "ceo"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    def test_TCAV_x_1_1_c_concept_order_changed(self) -> None:
        self.compute_cavs_interpret(
            [["random", "striped"], ["random", "ceo"], ["ceo", "striped"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    # Non-existing concept in the experimental set ("dotted")
    def test_TCAV_x_1_1_d(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            True,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    # Do not Force Train
    def test_TCAV_x_1_0_a(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    def test_TCAV_x_1_0_1_attr_to_inputs(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
            remove_activation=True,
            layers="relu2",
            attribute_to_layer_input=True,
        )

    # Non-existing concept in the experimental set ("dotted"), do Not Force Train
    def test_TCAV_x_1_0_b(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["dotted", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
        )

    # Do not Force Train, Missing Activation
    def test_TCAV_x_1_0_1(self) -> None:
        self.compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            processes=1,
            remove_activation=True,
        )

    def test_TCAV_x_1_0_1_w_flipped_class_id(self) -> None:
        self._compute_cavs_interpret(
            [["striped", "random"], ["ceo", "random"]],
            False,
            0.4848,
            0.5000,
            8.185208066890937e-09,
            CustomClassifier_W_Flipped_Class_Id(),
            processes=1,
        )

    # Testing TCAV with default classifier and experimental sets of varying lengths
    def test_exp_sets_with_diffent_lengths(self) -> None:
        try:
            import sklearn.svm  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("sklearn is not available.")
        # Create Concepts
        concepts_dict = create_concepts()

        # defining experimental sets of different length
        experimental_set_list = [["striped", "random"], ["ceo", "striped", "random"]]
        experimental_sets_diff_length = self._create_experimental_sets(
            experimental_set_list, concepts_dict
        )

        exp_sets_striped_random = self._create_experimental_sets(
            [["striped", "random"]], concepts_dict
        )
        exp_sets_ceo_striped_random = self._create_experimental_sets(
            [["ceo", "striped", "random"]], concepts_dict
        )
        striped_random_str = concepts_to_str(exp_sets_striped_random[0])
        ceo_striped_random_str = concepts_to_str(exp_sets_ceo_striped_random[0])

        model = BasicModel_ConvNet()
        model.eval()
        layers = ["conv1", "conv2", "fc1", "fc2"]
        inputs = torch.randn(5, 1, 10, 10)

        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav_diff_length = TCAV(
                model,
                layers,
                save_path=tmpdirname,
            )

            # computing tcav scores for `striped and random` set and
            # `ceo, striped and random` set at once using one `interpret`
            # call.
            interpret_diff_lengths = tcav_diff_length.interpret(
                inputs, experimental_sets=experimental_sets_diff_length, target=0
            )

            # computing tcav scores for striped and random
            interpret_striped_random = tcav_diff_length.interpret(
                inputs, experimental_sets=exp_sets_striped_random, target=0
            )

            # computing tcav scores for ceo, striped and random
            interpret_ceo_striped_random = tcav_diff_length.interpret(
                inputs, experimental_sets=exp_sets_ceo_striped_random, target=0
            )

            for combined, separate in zip(
                interpret_diff_lengths[striped_random_str].items(),
                interpret_striped_random[striped_random_str].items(),
            ):
                self.assertEqual(combined[0], separate[0])
                for c_tcav, s_tcav in zip(combined[1].items(), separate[1].items()):
                    self.assertEqual(c_tcav[0], s_tcav[0])
                    assertTensorAlmostEqual(self, c_tcav[1], s_tcav[1])

            for combined, separate in zip(
                interpret_diff_lengths[ceo_striped_random_str].items(),
                interpret_ceo_striped_random[ceo_striped_random_str].items(),
            ):
                self.assertEqual(combined[0], separate[0])
                for c_tcav, s_tcav in zip(combined[1].items(), separate[1].items()):
                    self.assertEqual(c_tcav[0], s_tcav[0])
                    assertTensorAlmostEqual(self, c_tcav[1], s_tcav[1])

    def test_model_ids_in_tcav(
        self,
    ) -> None:
        # creating concepts and mapping between concepts and their names
        concepts_dict = create_concepts()

        # defining experimental sets of different length
        experimental_set_list = [["striped", "random"], ["dotted", "random"]]
        experimental_sets = self._create_experimental_sets(
            experimental_set_list, concepts_dict
        )
        model = BasicModel_ConvNet()
        model.eval()
        layer = "conv2"
        inputs = 100 * get_inputs_tensor()

        with tempfile.TemporaryDirectory() as tmpdirname:
            tcav1 = TCAV(
                model,
                layer,
                model_id="my_basic_model1",
                classifier=CustomClassifier(),
                save_path=tmpdirname,
            )

            interpret1 = tcav1.interpret(
                inputs, experimental_sets=experimental_sets, target=0
            )

            tcav2 = TCAV(
                model,
                layer,
                model_id="my_basic_model2",
                classifier=CustomClassifier(),
                save_path=tmpdirname,
            )
            interpret2 = tcav2.interpret(
                inputs, experimental_sets=experimental_sets, target=0
            )

            # testing that different folders were created for two different
            # ids of the model
            self.assertTrue(
                AV.exists(
                    tmpdirname,
                    "my_basic_model1",
                    concepts_dict["striped"].identifier,
                    layer,
                )
            )
            self.assertTrue(
                AV.exists(
                    tmpdirname,
                    "my_basic_model2",
                    concepts_dict["striped"].identifier,
                    layer,
                )
            )
            for interpret1_elem, interpret2_elem in zip(interpret1, interpret2):
                for interpret1_sub_elem, interpret2_sub_elem in zip(
                    interpret1[interpret1_elem], interpret2[interpret2_elem]
                ):
                    assertTensorAlmostEqual(
                        self,
                        interpret1[interpret1_elem][interpret1_sub_elem]["sign_count"],
                        interpret2[interpret2_elem][interpret2_sub_elem]["sign_count"],
                        0.0,
                    )
                    assertTensorAlmostEqual(
                        self,
                        interpret1[interpret1_elem][interpret1_sub_elem]["magnitude"],
                        interpret2[interpret2_elem][interpret2_sub_elem]["magnitude"],
                        0.0,
                    )
                    self.assertEqual(interpret1_sub_elem, interpret2_sub_elem)

                self.assertEqual(interpret1_elem, interpret2_elem)
