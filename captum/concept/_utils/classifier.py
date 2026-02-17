#!/usr/bin/env python3

# pyre-strict

import random
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Union

import torch
from captum._utils.models.linear_model import model
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
        self.lm = model.SkLearnSGDClassifier(alpha=0.01, max_iter=1000, tol=1e-3)

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

        predict = self.lm.classes()[torch.argmax(predict, dim=1).cpu()]  # type: ignore
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
