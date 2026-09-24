#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import random
import time
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.models.linear_model.model import LinearModel, SkLearnSGDClassifier
from captum._utils.models.linear_model.train import NormLayer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class Classifier(ABC):
    r"""
    An abstract class definition of any classifier that allows to train a model
    and access trained weights of that model.

    More specifically the classifier can, for instance, be trained on the
    activations of a particular layer. Below we can see an example a sklearn
    linear classifier wrapped by the `CustomClassifier` which extends `Classifier`
    abstract class.

    Example::

    >>> from sklearn import linear_model
    >>>
    >>> class CustomClassifier(Classifier):
    >>>
    >>> def __init__(self):
    >>>
    >>>     self.lm = linear_model.SGDClassifier(alpha=0.01, max_iter=1000,
    >>>                                          tol=1e-3)
    >>>
    >>> def train_and_eval(self, dataloader):
    >>>
    >>>     x_train, x_test, y_train, y_test = train_test_split(inputs, labels)
    >>>     self.lm.fit(x_train.detach().numpy(), y_train.detach().numpy())
    >>>
    >>>     preds = torch.tensor(self.lm.predict(x_test.detach().numpy()))
    >>>     return {'accs': (preds == y_test).float().mean()}
    >>>
    >>>
    >>> def weights(self):
    >>>
    >>>     if len(self.lm.coef_) == 1:
    >>>         # if there are two concepts, there is only one label.
    >>>         # We split it in two.
    >>>         return torch.tensor([-1 * self.lm.coef_[0], self.lm.coef_[0]])
    >>>     else:
    >>>         return torch.tensor(self.lm.coef_)
    >>>
    >>>
    >>> def classes(self):
    >>>     return self.lm.classes_
    >>>
    >>>

    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train_and_eval(
        self,
        dataloader: DataLoader,
        **kwargs: Any,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        # `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Union[Dict, None]:
        r"""
        This method is responsible for training a classifier using the data
        provided through `dataloader` input arguments. Based on the specific
        implementation, it may or may not return a statistics about model
        training and evaluation.

        Args:
            dataloader (dataloader): A dataloader that enables batch-wise access to
                    the inputs and corresponding labels. Dataloader allows us to
                    iterate over the dataset by loading the batches in lazy manner.
            kwargs  (dict): Named arguments that are used for training and evaluating
                    concept classifier.
                    Default: None
        Returns:
            stats (dict): a dictionary of statistics about the performance of the model.
                    For example the accuracy of the model on the test and/or
                    train dataset(s). The user may decide to return None or an
                    empty dictionary if they decide to not return any performance
                    statistics.
        """
        pass

    @abstractmethod
    def weights(self) -> Tensor:
        r"""
        This function returns a C x F tensor weights, where
        C is the number of classes and F is the number of features.

        Returns:
            weights (Tensor): A torch Tensor with the weights resulting from
                the model training.
        """
        pass

    @abstractmethod
    def classes(self) -> List[int]:
        r"""
        This function returns the list of all classes that are used by the
        classifier to train the model in the `train_and_eval` method.
        The order of returned classes has to match the same order used in
        the weights matrix returned by the `weights` method.

        Returns:
            classes (list): The list of classes used by the classifier to train
            the model in the `train_and_eval` method.
        """
        pass


class DefaultClassifier(Classifier):
    r"""
    A default Linear Classifier based on sklearn's SGDClassifier for
    learning decision boundaries between concepts.
    Note that default implementation slices input dataset into train and test
    splits and keeps them in memory.
    In case concept datasets are large, this can lead to out of memory and we
    recommend to provide a custom Classier that extends `Classifier` abstract
    class and handles large concept datasets accordingly.
    """

    def __init__(self) -> None:
        warnings.warn(
            "Using default classifier for TCAV which keeps input"
            " both train and test datasets in the memory. Consider defining"
            " your own classifier that doesn't rely heavily on memory, for"
            " large number of concepts, by extending"
            " `Classifer` abstract class",
            stacklevel=2,
        )
        self.lm: LinearModel = SkLearnSGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)

    def train_and_eval(
        self,
        dataloader: DataLoader,
        test_split_ratio: float = 0.33,
        **kwargs: Any,
        # pyre-fixme[24]: Generic type `dict` expects 2 type parameters, use
        # `typing.Dict[<key type>, <value type>]` to avoid runtime subscripting errors.
    ) -> Union[Dict, None]:
        r"""
         Implements Classifier::train_and_eval abstract method for small concept
         datsets provided by `dataloader`.
         It is assumed that when iterating over `dataloader` we can still
         retain the entire dataset in the memory.
         This method shuffles all examples randomly provided, splits them
         into train and test partitions and trains an SGDClassifier using sklearn
         library. Ultimately, it measures and returns model accuracy using test
         split of the dataset.

        Args:
            dataloader (dataloader): A dataloader that enables batch-wise access to
                    the inputs and corresponding labels. Dataloader allows us to
                    iterate over the dataset by loading the batches in lazy manner.
            test_split_ratio (float): The ratio of test split in the entire dataset
                    served by input data loader `dataloader`.

                    Default: 0.33
        Returns:
            stats (dict): a dictionary of statistics about the performance of the model.
                    In this case stats represents a dictionary of model accuracy
                    measured on the test split of the dataset.

        """
        inputs = []
        labels = []
        for input, label in dataloader:
            inputs.append(input)
            labels.append(label)

        # pyre-fixme[61]: `input` is undefined, or not always defined.
        device = "cpu" if input is None else input.device
        x_train, x_test, y_train, y_test = _train_test_split(
            torch.cat(inputs), torch.cat(labels), test_split=test_split_ratio
        )
        # error: Incompatible types in assignment (expression has type "str | Any",
        # variable has type "Tensor | Module")  [assignment]
        self.lm.device = device  # type: ignore
        self.lm.fit(DataLoader(TensorDataset(x_train, y_train)))

        predict = self.lm(x_test)
        classes = self.lm.classes()
        assert classes is not None

        predict = _predict_classes(predict, classes)
        score = predict.long() == y_test.long().cpu()

        accs = score.float().mean()

        return {"accs": accs}

    def weights(self) -> Tensor:
        r"""
        This function returns a C x F tensor weights, where
        C is the number of classes and F is the number of features.
        In case of binary classification, C = 2 otherwise it is > 2.

        Returns:
            weights (Tensor): A torch Tensor with the weights resulting from
                the model training.
        """
        assert self.lm.linear is not None, (
            "The weights cannot be obtained because no model was trained."
            "In order to train the model call `train_and_eval` method first."
        )
        weights = self.lm.representation()
        if weights.shape[0] == 1:
            # if there are two concepts, there is only one label. We split it in two.
            return torch.stack([-1 * weights[0], weights[0]])
        else:
            return weights

    def classes(self) -> List[int]:
        r"""
        This function returns the list of all classes that are used by the
        classifier to train the model in the `train_and_eval` method.
        The order of returned classes has to match the same order used in
        the weights matrix returned by the `weights` method.

        Returns:
            classes (list): The list of classes used by the classifier to train
            the model in the `train_and_eval` method.
        """
        return self.lm.classes().detach().numpy()  # type: ignore


def _predict_classes(predict: Tensor, classes: Tensor) -> Tensor:
    if len(classes) == 2 and predict.shape[1] == 1:
        predict = torch.cat([-predict, predict], dim=1)
    return classes[torch.argmax(predict, dim=1).cpu()]


def _train_test_split(
    x_list: Tensor, y_list: Tensor, test_split: float = 0.33
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # Shuffle
    z_list = list(zip(x_list, y_list))
    random.shuffle(z_list)
    # Split
    test_size = int(test_split * len(z_list))
    z_test, z_train = z_list[:test_size], z_list[test_size:]
    x_test, y_test = zip(*z_test)
    x_train, y_train = zip(*z_train)
    return (
        torch.stack(x_train),
        torch.stack(x_test),
        torch.stack(y_train),
        torch.stack(y_test),
    )


class FastCAVClassifier(DefaultClassifier):
    r"""Fast implementation of concept activation vectors calculation
    using mean differences. This implementation requires balanced classes.
    This implements the classifier proposed in the paper `FastCAV: Efficient
    Computation of Concept Activation Vectors for Explaining Deep Neural Networks
    <https://arxiv.org/abs/2505.17883>`_.

    This classifier provides an efficient alternative to other CAV classifiers.

    It is equivalent to an SVM when the following assumptions hold:

    - **Gaussian Distribution**: The activation vectors for both the random samples
      and the concept samples are assumed to follow independent multivariate
      Gaussian distributions.
    - **Equal Mixture**: The set of concept examples and the set of random examples
      are of equal size :math:`(∣D_c​∣=∣D_r​∣)`, resulting in a uniform mixture of
      the two Gaussian distributions.
    - **Isotropic Covariance**: The within-class covariance matrices are assumed to
      be isotropic, meaning they are proportional to the unit matrix. This is a
      critical assumption that makes the FastCAV solution equivalent to the solution
      of a Fisher discriminant analysis.
    - **High-Dimensionality**: The method is analyzed in the context of
      high-dimensional activation spaces, where the number of dimensions :math:`d` is
      significantly larger than the number of samples :math:`n` (:math:`d >> n`).
      In such spaces, the set of support vectors used by an SVM is likely to contain
      most of the training samples, making `the SVM solution approximate the Fisher
      discriminant solution
      <https://link.springer.com/article/10.1023/A:1018677409366>`_,
      and by extension, the FastCAV solution.

    Note that default implementation slices input dataset into train and test
    splits and keeps them in memory.
    In case concept datasets are large, this can lead to out of memory and we
    recommend to provide a custom Classier that extends `Classifier` abstract
    class and handles large concept datasets accordingly.

    Example:

    >>> import torchvision
    >>> from captum.concept import TCAV
    >>> from captum.concept._utils.classifier import FastCAVClassifier
    >>> from captum.attr import LayerIntegratedGradients
    >>>
    >>> model = torchvision.models.googlenet(pretrained=True)
    >>> model = model.eval()
    >>> clf = FastCAVClassifier()
    >>> layers=['inception4c', 'inception4d', 'inception4e']
    >>> mytcav = TCAV(model=model,
    >>>          layers=layers,
    >>>          classifier=clf,
    >>>          layer_attr_method = LayerIntegratedGradients(
    >>>            model, None, multiply_by_inputs=False))
    >>> # ...
    >>> # For a full workflow, follow `tutorials/TCAV_Image.ipynb` and
    >>> # replace the classifier.
    """

    def __init__(self) -> None:
        self.lm = FastCAVLinearModel()


class FastCAVLinearModel(LinearModel):
    """
    FastCAVLinearModel is a wrapper to convert `FastCAV` into a `LinearModel`.

    Args:
        **kwargs (Any): Additional keyword arguments passed to the
            `LinearModel` base class.

    Returns:
        None
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(train_fn=fastcav_train_linear_model, **kwargs)


def fastcav_train_linear_model(
    model: LinearModel,
    dataloader: DataLoader,
    construct_kwargs: Dict[str, Any],
    norm_input: bool = False,
    **fit_kwargs: Any,
) -> Dict[str, float]:
    r"""
    Trains a `captum.concept._utils.models.linear_model.LinearModel` using
    the `FastCAV` classifier. It follows closely the implementation of the
    `linea_model.train.sklearn_train_linear_model` function.

    This method consumes the entire dataloader to construct the dataset in
    memory for training.

    Please note that this assumes:

    1. The dataset can fit into memory.

    Args:
        model (LinearModel): The model to train.
        dataloader (DataLoader): The data to use. This will be exhausted and converted
            to single tensors. Do not use an infinite dataloader.
        construct_kwargs (dict): Arguments to pass to the FastCAV constructor.
            FastCAV currently does not support any additional parameters.
        norm_input (bool, optional): Whether or not to normalize the input.
            Default: False
        fit_kwargs (dict, optional): Other arguments to send to FastCAV's fit method.
            FastCAV does not support sample weights or other fit arguments.
            Default: None

    Returns:
        dict: A dictionary containing the train_time.
    """
    # Extract data from dataloader
    fast_classifier = FastCAV(**construct_kwargs)
    num_batches = 0
    xs: List[Tensor] = []
    ys: List[Tensor] = []
    ws: List[Tensor] = []
    for data in dataloader:
        if len(data) == 3:
            x, y, w = data
        else:
            assert len(data) == 2
            x, y = data
            w = None

        xs.append(x)
        ys.append(y)
        if w is not None:
            ws.append(w)
        num_batches += 1

    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    if len(ws) > 0:
        w = torch.cat(ws, dim=0)
    else:
        w = None

    if norm_input:
        mean, std = x.mean(0), x.std(0)
        x -= mean
        x /= std

    t1 = time.time()

    if len(w) > 0:
        warnings.warn(
            "Sample weight is not supported for FastCAV!"
            " Trained model without weighting inputs",
            stacklevel=1,
        )

    fast_classifier.fit(x, y, **fit_kwargs)

    t2 = time.time()

    # Convert weights to pytorch
    classes = torch.IntTensor(fast_classifier.classes_)

    # extract model device
    device = getattr(model, "device", "cpu")

    num_outputs = (
        fast_classifier.coef_.shape[0]  # type: ignore
        if fast_classifier.coef_.ndim > 1  # type: ignore
        else 1
    )  # type: ignore
    weight_values = torch.FloatTensor(fast_classifier.coef_).to(device)  # type: ignore
    bias_values = torch.FloatTensor([fast_classifier.intercept_]).to(  # type: ignore
        device  # type: ignore
    )  # type: ignore
    model._construct_model_params(
        norm_type=None,
        weight_values=weight_values.view(num_outputs, -1),
        bias_value=bias_values.squeeze().unsqueeze(0),
        classes=classes,
    )

    if norm_input:
        # pyre-fixme[61]: `mean` is undefined, or not always defined.
        # pyre-fixme[61]: `std` is undefined, or not always defined.
        model.norm = NormLayer(mean, std)

    return {"train_time": t2 - t1}


class FastCAV:
    r"""Fast implementation of concept activation vectors calculation
    using mean differences. This implementation requires balanced classes.

    This classifier provides an efficient alternative to other CAV classifiers.

    It is equal to an SVM when the following assumptions hold:

    - **Gaussian Distribution**: The activation vectors for both the random samples
      and the concept samples are assumed to follow independent multivariate
      Gaussian distributions.
    - **Equal Mixture**: The set of concept examples and the set of random examples
      are of equal size :math:`(∣D_c​∣=∣D_r​∣)`, resulting in a uniform mixture of
      the two Gaussian distributions.
    - **Isotropic Covariance**: The within-class covariance matrices are assumed to
      be isotropic, meaning they are proportional to the unit matrix. This is a
      critical assumption that makes the FastCAV solution equivalent to the solution
      of a Fisher discriminant analysis.
    - **High-Dimensionality**: The method is analyzed in the context of
      high-dimensional activation spaces, where the number of dimensions :math:`d` is
      significantly larger than the number of samples :math:`n` (:math:`d >> n`).
      In such spaces, the set of support vectors used by an SVM is likely to contain
      most of the training samples, making `the SVM solution approximate the Fisher
      discriminant solution
      <https://link.springer.com/article/10.1023/A:1018677409366>`_,
      and by extension, the FastCAV solution.

    For more details, see the paper:
    `FastCAV: Efficient Computation of Concept Activation Vectors for Explaining
    Deep Neural Networks <https://arxiv.org/abs/2505.17883>`_.

    Example::

    >>> from captum.concept._utils import classifier
    >>> fast_cav = classifier.FastCAV()
    >>> x = torch.randn(100, 20)  # 100 samples, 20 features
    >>> y = torch.randint(0, 2, (100,))  # Binary
    >>> fast_cav.fit(x, y)
    >>> predictions = fast_cav.predict(x)

    """

    def __init__(self, **kwargs) -> None:
        self.intercept_: Optional[torch.Tensor] = None
        self.coef_: Optional[torch.Tensor] = None
        self.mean: Optional[torch.Tensor] = None
        self.classes_: Optional[torch.Tensor] = None
        if kwargs:
            warnings.warn(
                "FastCAV does not support any additional parameters. "
                f"Ignoring provided parameters: {kwargs.keys()}.",
                stacklevel=2,
            )

    def fit(self, x: Tensor, y: Tensor) -> None:
        """
        Fits a binary linear classifier to obtain a Concept Activation Vector (CAV)
        using the mean difference between two classes.

        Args:
            x (Tensor): Input data of shape (n_samples, n_features).
            Training data for binary classification.
            y (Tensor): Binary target labels of shape (n_samples,).
            Labels should be 0 or 1. Classes should be balanced.

        Returns:
            None

        Note:
            Computes the linear concept boundary using the mean difference vector
            between the two classes. Converts inputs to PyTorch tensors if needed.

        Why balanced classes:
            Imbalanced classes will skew the computed CAV toward the majority class,
            leading to inaccurate results. FastCAV works best with balanced classes.
        """
        x = torch.as_tensor(x)
        y = torch.as_tensor(y)

        assert x.ndim == 2, "Input tensor must be 2D (batch_size, num_features)"
        assert y.ndim == 1, "Labels tensor must be 1D (batch_size,)"
        assert x.shape[0] == y.shape[0], "Input and labels must have same batch size"

        self.classes_ = torch.unique(y).int()
        assert len(self.classes_) == 2, "Only binary classification is supported"

        class_counts = torch.bincount(y)
        if torch.abs(class_counts[0] - class_counts[1]).float() / len(y) > 0.2:
            warnings.warn(
                "Classes are imbalanced (>20% difference). "
                "FastCAV works best with balanced classes."
            )

        with torch.no_grad():
            self.mean = x.mean(dim=0)
            self.coef_ = (
                (x[y == self.classes_[-1]] - self.mean).mean(dim=0).unsqueeze(0)
            )
            self.intercept_ = (-self.coef_ @ self.mean).unsqueeze(1)

    def predict(self, x: Tensor) -> Tensor:
        """
        Predicts the class labels for the given input tensor using the trained model.

        Args:
            x (Tensor): Input tensor of shape (n_samples, n_features) or (n_features,).
                If a 1D tensor is provided, it is treated as a single sample.

        Returns:
            Tensor: Predicted class labels as a tensor of shape (n_samples,).

        Raises:
            ValueError: If the model has not been trained (i.e., `coef_`, `intercept_`,
                or `classes_` is None).
        """
        if self.coef_ is None or self.intercept_ is None or self.classes_ is None:
            raise ValueError("Model not trained. Call fit() first.")

        x = torch.as_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with torch.no_grad():
            return torch.take(
                self.classes_,
                ((self.coef_ @ torch.as_tensor(x).T + self.intercept_) > 0).long(),
            ).T

    def classes(self) -> Tensor:
        """
        Returns the classes learned by the classifier.

        Returns:
            Tensor: A tensor containing the unique class labels identified during
            model training.

        Raises:
            ValueError: If the model has not been trained and `fit` has not been called.
        """
        if self.classes_ is None:
            raise ValueError("Please call `fit` to train the model first.")
        return self.classes_
