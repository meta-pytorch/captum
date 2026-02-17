#!/usr/bin/env python3

# pyre-strict

from collections import defaultdict
from typing import Any, cast, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from captum._utils.av import AV
from captum._utils.common import _format_tensor_into_tuples, _get_module_from_name
from captum._utils.typing import TargetType, TensorOrTupleOfTensorsGeneric
from captum.attr import LayerActivation, LayerAttribution, LayerGradientXActivation
from captum.concept._core.cav import CAV
from captum.concept._core.concept import Concept, ConceptInterpreter
from captum.concept._utils.classifier import Classifier, DefaultClassifier
from captum.concept._utils.common import concepts_to_str
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset


class LabelledDataset(Dataset):
    """
    A torch Dataset whose __getitem__ returns both a batch of activation vectors,
    as well as a batch of labels associated with those activation vectors.
    It is used to train a classifier in train_tcav
    """

    def __init__(self, datasets: List[AV.AVDataset], labels: List[int]) -> None:
        """
        Creates the LabelledDataset given a list of K Datasets, and a length K
        list of integer labels representing K different concepts.
        The assumption is that the k-th Dataset of datasets is associated with
        the k-th element of labels.
        The LabelledDataset is the concatenation of the K Datasets in datasets.
        However, __get_item__ not only returns a batch of activation vectors,
        but also a batch of labels indicating which concept that batch of
        activation vectors is associated with.

        Args:

            datasets (list[Dataset]): The k-th element of datasets is a Dataset
                    representing activation vectors associated with the k-th
                    concept
            labels (list[int]): The k-th element of labels is the integer label
                    associated with the k-th concept
        """
        assert len(datasets) == len(
            labels
        ), "number of datasets does not match the number of concepts"

        from itertools import accumulate

        offsets = [0] + list(accumulate(map(len, datasets), (lambda x, y: x + y)))
        self.length: int = offsets[-1]
        self.datasets = datasets
        self.labels = labels
        self.lowers: List[int] = offsets[:-1]
        self.uppers: List[int] = offsets[1:]

    def _i_to_k(self, i: int) -> int:

        left, right = 0, len(self.uppers)
        while left < right:
            mid = (left + right) // 2
            if self.lowers[mid] <= i and i < self.uppers[mid]:
                return mid
            if i >= self.uppers[mid]:
                left = mid
            else:
                right = mid
        return -1

    def __getitem__(self, i: int) -> Tuple[Union[Tensor, Tuple[Tensor, ...]], Tensor]:
        """
        Returns a batch of activation vectors, as well as a batch of labels
        indicating which concept the batch of activation vectors is associated
        with.

        Args:

            i (int): which (activation vector, label) batch in the dataset to
                    return
        Returns:
            inputs (Tensor): i-th batch in Dataset (representing activation
                    vectors)
            labels (Tensor): labels of i-th batch in Dataset
        """
        assert i < self.length
        k = self._i_to_k(i)
        inputs = self.datasets[k][i - self.lowers[k]]
        # pyre-fixme[16]: Item `tuple` of `Union[Tensor, Tuple[Tensor, ...]]` has no
        #  attribute `shape`.
        assert len(inputs.shape) == 2  # type: ignore

        # pyre-fixme[16]: Item `tuple` of `Union[Tensor, Tuple[Tensor, ...]]` has no
        #  attribute `size`.
        # pyre-fixme[16]: Item `tuple` of `Union[Tensor, Tuple[Tensor, ...]]` has no
        #  attribute `device`.
        labels = torch.tensor([self.labels[k]] * inputs.size(0), device=inputs.device)  # type: ignore # noqa: E501 line too long
        return inputs, labels

    def __len__(self) -> int:
        """
        returns the total number of batches in the labelled_dataset
        """
        return self.length


def train_cav(
    model_id: str,
    concepts: List[Concept],
    layers: Union[str, List[str]],
    classifier: Classifier,
    save_path: str,
    classifier_kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Optional[CAV]]]:
    r"""
    A helper function for parallel CAV computations that can be called
    from a python process.

    Please see the TCAV class documentation for further information.

    Args:

        model_id (str): A unique identifier for the PyTorch model for which
                we would like to load the layer activations and train a
                model in order to compute CAVs.
        concepts (list[Concept]): A list of Concept objects that are used
                to train a classifier and learn decision boundaries between
                those concepts for each layer defined in the `layers`
                argument.
        layers (str or list[str]): A list of layer names or a single layer
                name that is used to compute the activations of all concept
                examples per concept and train a classifier using those
                activations.
        classifier (Classifier): A custom classifier class, such as the
                Sklearn "linear_model" that allows us to train a model
                using the activation vectors extracted for a layer per concept.
                It also allows us to access trained weights of the classifier
                and the list of prediction classes.
        save_path (str): The path for storing Concept Activation
                Vectors (CAVs) and Activation Vectors (AVs).
        classifier_kwargs (dict): Additional named arguments that are passed to
                concept classifier's `train_and_eval` method.

    Returns:
        cavs (dict): A dictionary of CAV objects indexed by concept ids and
                layer names. It gives access to the weights of each concept
                in a given layer and model statistics such as accuracies
                that resulted in trained concept weights.
    """

    concepts_key = concepts_to_str(concepts)
    cavs: Dict[str, Dict[str, Optional[CAV]]] = defaultdict()
    cavs[concepts_key] = defaultdict()
    layers = [layers] if isinstance(layers, str) else layers
    for layer in layers:

        # Create data loader to initialize the trainer.
        datasets = [
            AV.load(save_path, model_id, concept.identifier, layer)
            for concept in concepts
        ]

        labels = [concept.id for concept in concepts]

        labelled_dataset = LabelledDataset(datasets, labels)

        # pyre-fixme[2]: Parameter must be annotated.
        def batch_collate(batch) -> Tuple[Tensor, Tensor]:
            inputs, labels = zip(*batch)
            return torch.cat(inputs), torch.cat(labels)

        dataloader = DataLoader(labelled_dataset, collate_fn=batch_collate)

        classifier_stats_dict = classifier.train_and_eval(
            dataloader, **classifier_kwargs
        )
        classifier_stats_dict = (
            {} if classifier_stats_dict is None else classifier_stats_dict
        )

        weights = classifier.weights()
        assert (
            weights is not None and len(weights) > 0
        ), "Model weights connot be None or empty"

        classes = classifier.classes()
        assert (
            classes is not None and len(classes) > 0
        ), "Classes cannot be None or empty"

        classes = (
            cast(torch.Tensor, classes).detach().numpy()
            if isinstance(classes, torch.Tensor)
            else classes
        )
        cavs[concepts_key][layer] = CAV(
            concepts,
            layer,
            {"weights": weights, "classes": classes, **classifier_stats_dict},
            save_path,
            model_id,
        )
        # Saving cavs on the disk
        # pyre-fixme[16]: `Optional` has no attribute `save`.
        cavs[concepts_key][layer].save()  # type: ignore

    return cavs


class TCAV(ConceptInterpreter):
    r"""
    This class implements ConceptInterpreter abstract class using an
    approach called Testing with Concept Activation Vectors (TCAVs),
    as described in the paper:
    https://arxiv.org/abs/1711.11279

    TCAV scores for a given layer, a list of concepts and input example
    are computed using the dot product between prediction's layer
    sensitivities for given input examples and Concept Activation Vectors
    (CAVs) in that same layer.

    CAVs are defined as vectors that are orthogonal to the classification boundary
    hyperplane that separate given concepts in a given layer from each other.
    For a given layer, CAVs are computed by training a classifier that uses the
    layer activation vectors for a set of concept examples as input examples and
    concept ids as corresponding input labels. Trained weights of
    that classifier represent CAVs.

    CAVs are represented as a learned weight matrix with the dimensionality
    C X F, where:
    F represents the number of input features in the classifier.
    C is the number of concepts used for the classification. Concept
    ids are used as labels for concept examples during the training.

    We can use any layer attribution algorithm to compute layer sensitivities
    of a model prediction.
    For example, the gradients of an output prediction w.r.t. the outputs of
    the layer.
    The CAVs and the Sensitivities (SENS) are used to compute the TCAV score:

    0. TCAV = CAV • SENS, a dot product between those two vectors

    The final TCAV score can be computed by aggregating the TCAV scores
    for each input concept based on the sign or magnitude of the tcav scores.

    1. sign_count_score = | TCAV > 0 | / | TCAV |
    2. magnitude_score = SUM(ABS(TCAV * (TCAV > 0))) / SUM(ABS(TCAV))
    """

    def __init__(
        self,
        model: Module,
        layers: Union[str, List[str]],
        model_id: str = "default_model_id",
        classifier: Optional[Classifier] = None,
        layer_attr_method: Optional[LayerAttribution] = None,
        attribute_to_layer_input: bool = False,
        save_path: str = "./cav/",
        **classifier_kwargs: Any,
    ) -> None:
        r"""
        Args:

            model (Module): An instance of pytorch model that is used to compute
                    layer activations and attributions.
            layers (str or list[str]): A list of layer name(s) that are
                    used for computing concept activations (cavs) and layer
                    attributions.
            model_id (str, optional): A unique identifier for the PyTorch `model`
                    passed as first argument to the constructor of TCAV class. It
                    is used to store and load activations for given input `model`
                    and associated `layers`.
            classifier (Classifier, optional): A custom classifier class, such as the
                    Sklearn "linear_model" that allows us to train a model
                    using the activation vectors extracted for a layer per concept.
                    It also allows us to access trained weights of the model
                    and the list of prediction classes.
            layer_attr_method (LayerAttribution, optional): An instance of a layer
                    attribution algorithm that helps us to compute model prediction
                    sensitivity scores.

                    Default: None
                    If `layer_attr_method` is None, we default it to gradients
                    for the layers using `LayerGradientXActivation` layer
                    attribution algorithm.
            attribute_to_layer_input (bool, optional): Indicates whether to
                    compute the attributions with respect to the layer input
                    or output. If `attribute_to_layer_input` is set to True
                    then the attributions will be computed with respect to
                    layer inputs, otherwise it will be computed with respect
                    to layer outputs.
                    Default: False
            save_path (str, optional): The path for storing CAVs and
                    Activation Vectors (AVs).
            classifier_kwargs (Any, optional): Additional arguments such as
                    `test_split_ratio` that are passed to concept `classifier`.

        Examples::
            >>>
            >>> # TCAV use example:
            >>>
            >>> # Define the concepts
            >>> stripes = Concept(0, "stripes", striped_data_iter)
            >>> random = Concept(1, "random", random_data_iter)
            >>>
            >>>
            >>> mytcav = TCAV(model=imagenet,
            >>>     layers=['inception4c', 'inception4d'])
            >>>
            >>> scores = mytcav.interpret(inputs, [[stripes, random]], target = 0)
            >>>
            For more thorough examples, please check out TCAV tutorial and test cases.
        """
        ConceptInterpreter.__init__(self, model)
        self.layers: List[str] = [layers] if isinstance(layers, str) else layers
        self.model_id = model_id
        self.concepts: Set[Concept] = set()
        self.classifier = classifier
        self.classifier_kwargs: Any = classifier_kwargs
        # pyre-fixme[8]: Attribute has type `Dict[str, Dict[str, CAV]]`; used as
        #  `DefaultDict[Variable[_KT], DefaultDict[Variable[_KT], Variable[_VT]]]`.
        self.cavs: Dict[str, Dict[str, CAV]] = defaultdict(lambda: defaultdict())
        if self.classifier is None:
            self.classifier = DefaultClassifier()
        if layer_attr_method is None:
            self.layer_attr_method: LayerAttribution = cast(
                LayerAttribution,
                LayerGradientXActivation(  # type: ignore
                    model,
                    # pyre-fixme[6]: For 2nd argument expected `ModuleOrModuleList`
                    #  but got `None`.
                    None,
                    multiply_by_inputs=False,
                ),
            )
        else:
            self.layer_attr_method = layer_attr_method

        assert model_id, (
            "`model_id` cannot be None or empty. Consider giving `model_id` "
            "a meaningful name or leave it unspecified. If model_id is unspecified we "
            "will use `default_model_id` as its default value."
        )

        self.attribute_to_layer_input: bool = attribute_to_layer_input
        self.save_path = save_path

        # Creates CAV save directory if it doesn't exist. It is created once in the
        # constructor before generating the CAVs.
        # It is assumed that `model_id` can be used as a valid directory name
        # otherwise `create_cav_dir_if_missing` will raise an error
        CAV.create_cav_dir_if_missing(self.save_path, model_id)

    def generate_all_activations(self) -> None:
        r"""
        Computes layer activations for all concepts and layers that are
        defined in `self.layers` and `self.concepts` instance variables.
        """
        for concept in self.concepts:
            self.generate_activation(self.layers, concept)

    def generate_activation(
        self, layers: Union[str, List[str]], concept: Concept
    ) -> None:
        r"""
        Computes layer activations for the specified `concept` and
        the list of layer(s) `layers`.

        Args:
            layers (str or list[str]): A list of layer names or a layer name
                    that is used to compute layer activations for the
                    specific `concept`.
            concept (Concept): A single Concept object that provides access
                    to concept examples using a data iterator.
        """
        layers = [layers] if isinstance(layers, str) else layers
        layer_modules = [_get_module_from_name(self.model, layer) for layer in layers]

        layer_act = LayerActivation(self.model, layer_modules)
        data_iter = concept.data_iter
        assert data_iter is not None, (
            "Data iterator for concept id:",
            "{} must be specified".format(concept.id),
        )
        for i, examples in enumerate(data_iter):
            activations = layer_act.attribute.__wrapped__(  # type: ignore
                layer_act,
                examples,
                attribute_to_layer_input=self.attribute_to_layer_input,
            )
            for activation, layer_name in zip(activations, layers):
                activation = torch.reshape(activation, (activation.shape[0], -1))
                AV.save(
                    self.save_path,
                    self.model_id,
                    concept.identifier,
                    layer_name,
                    activation.detach(),
                    str(i),
                )

    def generate_activations(self, concept_layers: Dict[Concept, List[str]]) -> None:
        r"""
        Computes layer activations for the concepts and layers specified in
        `concept_layers` dictionary.

        Args:
            concept_layers (dict[Concept, list[str]]): Dictionay that maps
                    Concept objects to a list of layer names to generate
                    the activations. Ex.: concept_layers =
                    {"striped": ['inception4c', 'inception4d']}
        """
        for concept in concept_layers:
            self.generate_activation(concept_layers[concept], concept)

    def load_cavs(
        self, concepts: List[Concept]
    ) -> Tuple[List[str], Dict[Concept, List[str]]]:
        r"""
        This function load CAVs as a dictionary of concept ids and
        layers. CAVs are stored in a directory located under
        `self.save_path` path, in .pkl files with the format:
        <self.save_path>/<concept_ids>-<layer_name>.pkl. Ex.:
        "/cavs/0-1-2-inception4c.pkl", where 0, 1 and 2 are concept ids.

        It returns a list of layers and a dictionary of concept-layers mapping
        for the concepts and layer that require CAV computation through training.
        This can happen if the CAVs aren't already pre-computed for a given list
        of concepts and layer.

        Args:

            concepts (list[Concept]): A list of Concept objects for which we want
                    to load the CAV.

        Returns:
            layers (list[layer]): A list of layers for which some CAVs still need
                    to be computed.
            concept_layers (dict[concept, layer]): A dictionay of concept-layers
                    mapping for which we need to perform CAV computation through
                    training.
        """

        concepts_key = concepts_to_str(concepts)

        layers = []
        concept_layers = defaultdict(list)

        for layer in self.layers:
            cav = CAV.load(self.save_path, self.model_id, concepts, layer)

            if cav is not None:
                self.cavs[concepts_key][layer] = cav

            # If CAV aren't loaded
            if (
                concepts_key not in self.cavs
                or layer not in self.cavs[concepts_key]
                or not self.cavs[concepts_key][layer]
            ):

                layers.append(layer)
                # For all concepts in this experimental_set
                for concept in concepts:
                    # Collect not activated layers for this concept
                    if not AV.exists(
                        self.save_path, self.model_id, layer, concept.identifier
                    ):
                        concept_layers[concept].append(layer)
        return layers, concept_layers

    def compute_cavs(
        self,
        experimental_sets: List[List[Concept]],
        force_train: bool = False,
        processes: Optional[int] = None,
    ) -> Dict[str, Dict[str, CAV]]:
        r"""
        This method computes CAVs for given `experiments_sets` and layers
        specified in `self.layers` instance variable. Internally, it
        trains a classifier and creates an instance of CAV class using the
        weights of the trained classifier for each experimental set.

        It also allows to compute the CAVs in parallel using python's
        multiprocessing API and the number of processes specified in
        the argument.

        Args:

            experimental_sets (list[list[Concept]]): A list of lists of concept
                    instances for which the cavs will be computed.
            force_train (bool, optional): A flag that indicates whether to
                    train the CAVs regardless of whether they are saved or not.
                    Default: False
            processes (int, optional): The number of processes to be created
                    when running in multi-processing mode. If processes > 0 then
                    CAV computation will be performed in parallel using
                    multi-processing, otherwise it will be performed sequentially
                    in a single process.
                    Default: None

        Returns:
            cavs (dict) : A mapping of concept ids and layers to CAV objects.
                    If CAVs for the concept_ids-layer pairs are present in the
                    data storage they will be loaded into the memory, otherwise
                    they will be computed using a training process and stored
                    in the data storage that can be configured using `save_path`
                    input argument.
        """

        # Update self.concepts with concepts
        for concepts in experimental_sets:
            self.concepts.update(concepts)

        concept_ids = []
        for concept in self.concepts:
            assert concept.id not in concept_ids, (
                "There is more than one instance "
                "of a concept with id {} defined in experimental sets. Please, "
                "make sure to reuse the same instance of concept".format(
                    str(concept.id)
                )
            )
            concept_ids.append(concept.id)

        if force_train:
            self.generate_all_activations()

        # List of layers per concept key (experimental_set item) to be trained
        concept_key_to_layers = defaultdict(list)

        for concepts in experimental_sets:

            concepts_key = concepts_to_str(concepts)

            # If not 'force_train', try to load a saved CAV
            if not force_train:
                layers, concept_layers = self.load_cavs(concepts)
                concept_key_to_layers[concepts_key] = layers
                # Generate activations for missing (concept, layers)
                self.generate_activations(concept_layers)
            else:
                concept_key_to_layers[concepts_key] = self.layers
        if processes is not None and processes > 1:
            pool = multiprocessing.Pool(processes)
            cavs_list = pool.starmap(
                train_cav,
                [
                    (
                        self.model_id,
                        concepts,
                        concept_key_to_layers[concepts_to_str(concepts)],
                        self.classifier,
                        self.save_path,
                        self.classifier_kwargs,
                    )
                    for concepts in experimental_sets
                ],
            )

            pool.close()
            pool.join()

        else:
            cavs_list = []
            for concepts in experimental_sets:
                cavs_list.append(
                    train_cav(
                        self.model_id,
                        concepts,
                        concept_key_to_layers[concepts_to_str(concepts)],
                        cast(Classifier, self.classifier),
                        self.save_path,
                        self.classifier_kwargs,
                    )
                )

        # list[Dict[concept, Dict[layer, list]]] => Dict[concept, Dict[layer, list]]
        for cavs in cavs_list:
            for c_key in cavs:
                self.cavs[c_key].update(cavs[c_key])  # type: ignore

        return self.cavs

    @log_usage(part_of_slo=True)
    def interpret(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        experimental_sets: List[List[Concept]],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        processes: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Dict[str, Dict[str, Tensor]]]:
        r"""
        This method computes magnitude and sign-based TCAV scores for each
        experimental sets in `experimental_sets` list.
        TCAV scores are computed using a dot product between layer attribution
        scores for specific predictions and CAV vectors.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Inputs for which predictions
                    are performed and attributions are computed.
                    If model takes a single tensor as
                    input, a single input tensor should be provided.
                    If model takes multiple tensors as
                    input, a tuple of the input tensors should be provided.
                    It is assumed that for all given input tensors,
                    dimension 0 corresponds to the number of examples
                    (aka batch size), and if multiple input tensors are
                    provided, the examples must be aligned appropriately.
            experimental_sets (list[list[Concept]]): A list of list of Concept
                    instances.
            target (int, tuple, Tensor, or list, optional): Output indices for
                    which attributions are computed (for classification cases,
                    this is usually the target class).
                    If the network returns a scalar value per example,
                    no target index is necessary.
                    For general 2D outputs, targets can be either:

                    - a single integer or a tensor containing a single
                        integer, which is applied to all input examples
                    - a list of integers or a 1D tensor, with length matching
                        the number of examples in inputs (dim 0). Each integer
                        is applied as the target for the corresponding example.

                    For outputs with > 2 dimensions, targets can be either:

                    - A single tuple, which contains #output_dims - 1
                        elements. This target index is applied to all examples.
                    - A list of tuples with length equal to the number of
                        examples in inputs (dim 0), and each tuple containing
                        #output_dims - 1 elements. Each tuple is applied as the
                        target for the corresponding example.

            additional_forward_args (Any, optional): Extra arguments that are passed to
                     model when computing the attributions for `inputs`
                     w.r.t. layer output.
                     Default: None
            processes (int, optional): The number of processes to be created. if
                    processes is larger than one then CAV computations will be
                    performed in parallel using the number of processes equal to
                    `processes`. Otherwise, CAV computations will be performed
                    sequential.
                    Default:None
            **kwargs (Any, optional): A list of arguments that are passed to layer
                    attribution algorithm's attribute method. This could be for
                    example `n_steps` in case of integrated gradients.
                    Default: None

        Returns:
            results (dict): A dictionary of sign and magnitude -based tcav scores
                    for each concept set per layer.
                    The order of TCAV scores in the resulting tensor for each
                    experimental set follows the order in which concepts
                    are passed in `experimental_sets` input argument.

        results example::
            >>> #
            >>> # scores =
            >>> # {'0-1':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.5800, 0.4200]),
            >>> #          'magnitude': tensor([0.6613, 0.3387])},
            >>> #      'inception4d':
            >>> #         {'sign_count': tensor([0.6200, 0.3800]),
            >>> #           'magnitude': tensor([0.7707, 0.2293])}}),
            >>> #  '0-2':
            >>> #     {'inception4c':
            >>> #         {'sign_count': tensor([0.6200, 0.3800]),
            >>> #          'magnitude': tensor([0.6806, 0.3194])},
            >>> #      'inception4d':
            >>> #         {'sign_count': tensor([0.6400, 0.3600]),
            >>> #          'magnitude': tensor([0.6563, 0.3437])}})})
            >>> #

        """
        assert "attribute_to_layer_input" not in kwargs, (
            "Please, set `attribute_to_layer_input` flag as a constructor "
            "argument to TCAV class. In that case it will be applied "
            "consistently to both layer activation and layer attribution methods."
        )
        self.compute_cavs(experimental_sets, processes=processes)

        # pyre-fixme[9]: scores has type `Dict[str, Dict[str, Dict[str, Tensor]]]`;
        #  used as `DefaultDict[Variable[_KT], DefaultDict[Variable[_KT],
        #  Variable[_VT]]]`.
        scores: Dict[str, Dict[str, Dict[str, Tensor]]] = defaultdict(
            lambda: defaultdict()
        )

        # Retrieves the lengths of the experimental sets so that we can sort
        # them by the length and compute TCAV scores in batches.
        exp_set_lens = np.array(
            [len(exp_set) for exp_set in experimental_sets], dtype=object
        )
        exp_set_lens_arg_sort = np.argsort(exp_set_lens)

        # compute offsets using sorted lengths using their indices
        exp_set_lens_sort = exp_set_lens[exp_set_lens_arg_sort]
        exp_set_offsets_bool = [False] + list(
            exp_set_lens_sort[:-1] == exp_set_lens_sort[1:]
        )
        exp_set_offsets = []
        for i, offset in enumerate(exp_set_offsets_bool):
            if not offset:
                exp_set_offsets.append(i)

        exp_set_offsets.append(len(exp_set_lens))

        # sort experimental sets using the length of the concepts in each set
        experimental_sets_sorted = np.array(experimental_sets, dtype=object)[
            exp_set_lens_arg_sort
        ]

        for layer in self.layers:
            layer_module = _get_module_from_name(self.model, layer)
            self.layer_attr_method.layer = layer_module
            attribs = self.layer_attr_method.attribute.__wrapped__(  # type: ignore
                self.layer_attr_method,  # self
                inputs,
                target=target,
                additional_forward_args=additional_forward_args,
                attribute_to_layer_input=self.attribute_to_layer_input,
                **kwargs,
            )

            attribs = _format_tensor_into_tuples(attribs)
            # n_inputs x n_features
            attribs = torch.cat(
                # pyre-fixme[16]: `None` has no attribute `__iter__`.
                [torch.reshape(attrib, (attrib.shape[0], -1)) for attrib in attribs],
                dim=1,
            )

            # n_experiments x n_concepts x n_features
            cavs = []
            classes = []
            for concepts in experimental_sets:
                concepts_key = concepts_to_str(concepts)
                cavs_stats = cast(Dict[str, Any], self.cavs[concepts_key][layer].stats)
                cavs.append(cavs_stats["weights"].float().detach().tolist())
                classes.append(cavs_stats["classes"])

            # sort cavs and classes using the length of the concepts in each set
            cavs_sorted = np.array(cavs, dtype=object)[exp_set_lens_arg_sort]
            classes_sorted = np.array(classes, dtype=object)[exp_set_lens_arg_sort]
            i = 0
            while i < len(exp_set_offsets) - 1:
                cav_subset = np.array(
                    cavs_sorted[exp_set_offsets[i] : exp_set_offsets[i + 1]],
                    dtype=object,
                ).tolist()
                classes_subset = classes_sorted[
                    exp_set_offsets[i] : exp_set_offsets[i + 1]
                ].tolist()

                # n_experiments x n_concepts x n_features
                cav_subset = torch.tensor(cav_subset)
                cav_subset = cav_subset.to(attribs.device)
                assert len(cav_subset.shape) == 3, (
                    "cav should have 3 dimensions: n_experiments x "
                    "n_concepts x n_features."
                )

                experimental_subset_sorted = experimental_sets_sorted[
                    exp_set_offsets[i] : exp_set_offsets[i + 1]
                ]
                self._tcav_sub_computation(
                    scores,
                    layer,
                    attribs,
                    cav_subset,
                    classes_subset,
                    experimental_subset_sorted,  # type: ignore
                )
                i += 1

        return scores

    def _tcav_sub_computation(
        self,
        scores: Dict[str, Dict[str, Dict[str, Tensor]]],
        layer: str,
        attribs: Tensor,
        cavs: Tensor,
        classes: List[List[int]],
        experimental_sets: List[List[Concept]],
    ) -> None:
        # n_inputs x n_concepts
        tcav_score = torch.matmul(attribs.float(), torch.transpose(cavs, 1, 2))
        assert len(tcav_score.shape) == 3, (
            "tcav_score should have 3 dimensions: n_experiments x "
            "n_inputs x n_concepts."
        )

        assert attribs.shape[0] == tcav_score.shape[1], (
            "attrib and tcav_score should have the same 1st and "
            "2nd dimensions respectively (n_inputs)."
        )
        # n_experiments x n_concepts
        sign_count_score = torch.mean((tcav_score > 0.0).float(), dim=1)

        magnitude_score = torch.mean(tcav_score, dim=1)

        for i, (cls_set, concepts) in enumerate(zip(classes, experimental_sets)):
            concepts_key = concepts_to_str(concepts)

            # sort classes / concepts in the order specified in concept_keys
            concept_ord = [concept.id for concept in concepts]
            class_ord = {cls_: idx for idx, cls_ in enumerate(cls_set)}

            new_ord = torch.tensor(
                [class_ord[cncpt] for cncpt in concept_ord], device=tcav_score.device
            )

            # sort based on classes
            scores[concepts_key][layer] = {
                "sign_count": torch.index_select(
                    sign_count_score[i, :], dim=0, index=new_ord
                ),
                "magnitude": torch.index_select(
                    magnitude_score[i, :], dim=0, index=new_ord
                ),
            }
