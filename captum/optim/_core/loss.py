import operator
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from captum.optim._utils.image.common import (
    _create_new_vector,
    _dot_cossim,
    get_neuron_pos,
)
from captum.optim._utils.typing import ModuleOutputMapping


class Loss(ABC):
    """
    Abstract Class to describe loss.
    Note: All Loss classes should expose self.target for hooking by
    InputOptimization
    """

    def __init__(self) -> None:
        super().__init__()
        self.__name__ = self.__class__.__name__

    @abstractproperty
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        pass

    @abstractmethod
    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        pass

    def __repr__(self) -> str:
        return self.__name__

    def __neg__(self) -> "CompositeLoss":
        return module_op(self, None, operator.neg)

    def __add__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.add)

    def __sub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.sub)

    def __mul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.mul)

    def __truediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.truediv)

    def __pow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return module_op(self, other, operator.pow)

    def __radd__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__add__(other)

    def __rsub__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__neg__().__add__(other)

    def __rmul__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return self.__mul__(other)

    def __rtruediv__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return rmodule_op(self, other, operator.truediv)

    def __rpow__(self, other: Union[int, float, "Loss"]) -> "CompositeLoss":
        return rmodule_op(self, other, operator.pow)

    def __round__(self, decimals: Optional[int] = 0) -> "CompositeLoss":
        return module_op(self, decimals, lambda x, d: torch.round(x, decimals=d))


def module_op(
    self: Loss, other: Union[None, int, float, Loss], math_op: Callable
) -> "CompositeLoss":
    """
    This is a general function for applying math operations to Losses

    Args:

        self (Loss): A Loss objective instance.
        other (int, float, Loss, or None): The Loss objective instance or number to
            use on the self Loss objective as part of a math operation. If math_op
            is a unary operation, then other should be set to None.
        math_op (Callable): A math operator to use on the Loss instance.

    Returns:
        loss (CompositeLoss): A CompositeLoss instance with the math operations
            created by the specified arguments.
    """
    if other is None and math_op == operator.neg:

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            """
            Pass collected activations through loss objective, and then apply a unary
            math op.

            Args:

                module (ModuleOutputMapping): A dict of captured activations with
                    nn.Modules as keys.

                Returns:
                    loss (torch.Tensor): The target activations after being run
                        through the loss objective, and the unary math_op.
            """
            return math_op(self(module))

        name = self.__name__
        target = self.target
    elif isinstance(other, (int, float)):

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            """
            Pass collected activations through the loss objective and then apply the
            math operations with numbers.

            Args:

                module (ModuleOutputMapping): A dict of captured activations with
                    nn.Modules as keys.

                Returns:
                    loss (torch.Tensor): The target activations after being run
                        through the loss objective, and then the math_op with a number.
            """
            return math_op(self(module), other)

        name = self.__name__
        target = self.target
    elif isinstance(other, Loss):
        # We take the mean of the output tensor to resolve shape mismatches
        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            """
            Pass collected activations through the loss objectives and then combine the
            outputs with a math operation.

            Args:

                module (ModuleOutputMapping): A dict of captured activations with
                    nn.Modules as keys.

                Returns:
                    loss (torch.Tensor): The target activations after being run
                        through the loss objectives, and then merged with the math_op.
            """
            return math_op(torch.mean(self(module)), torch.mean(other(module)))

        name = f"Compose({', '.join([self.__name__, other.__name__])})"

        # ToDo: Refine logic for self.target handling
        target = (self.target if isinstance(self.target, list) else [self.target]) + (
            other.target if isinstance(other.target, list) else [other.target]
        )

        # Filter out duplicate targets
        target = list(dict.fromkeys(target))
    else:
        raise TypeError(
            "Can only apply math operations with int, float or Loss. Received type "
            + str(type(other))
        )
    return CompositeLoss(loss_fn, name=name, target=target)


def rmodule_op(
    self: Loss, other: Union[int, float, Loss], math_op: Callable
) -> "CompositeLoss":
    """
    This is a general function for applying the "r" versions of math operations to
    Losses.
    """
    if isinstance(other, (int, float)):

        def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
            return math_op(other, self(module))

        name = self.__name__
        target = self.target
    elif isinstance(other, Loss):
        # This should never get called because __math_op__ will be called instead
        pass
    else:
        raise TypeError(
            "Can only apply math operations with int, float or Loss. Received type "
            + str(type(other))
        )
    return CompositeLoss(loss_fn, name=name, target=target)


class BaseLoss(Loss):
    """
    The base class used for all Loss objectives.
    """

    def __init__(
        self,
        target: Union[nn.Module, List[nn.Module]] = [],
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module or list of nn.Module): A target nn.Module or list of
                nn.Module.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set to
                ``None``, defaults to all activations in the batch. Index ranges should
                be in the format of: [start, end].
                Default: ``None``
        """
        super().__init__()
        self._target = target
        if batch_index is None:
            self._batch_index = (None, None)
        elif isinstance(batch_index, (list, tuple)):
            self._batch_index = tuple(batch_index)
        else:
            self._batch_index = (batch_index, batch_index + 1)
        assert all([isinstance(b, (int, type(None))) for b in self._batch_index])
        assert len(self._batch_index) == 2

    @property
    def target(self) -> Union[nn.Module, List[nn.Module]]:
        """
        Returns:
            target (nn.Module or list of nn.Module): A target nn.Module or list of
                nn.Module.
        """
        return self._target

    @property
    def batch_index(self) -> Tuple:
        """
        Returns:
            batch_index (tuple of int): A tuple of batch indices with a format
                of: (start, end).
        """
        return self._batch_index


class CompositeLoss(BaseLoss):
    """
    When math operations are performed using one or more loss objectives, this class
    is used to store and run those operations. Below we show examples of common
    CompositeLoss use cases.


    Using CompositeLoss with a unary op or with a binary op involving a Loss instance
    and a float or integer:

    .. code-block:: python

        def compose_single_loss(loss: opt.loss.Loss) -> opt.loss.CompositeLoss:
            def loss_fn(
                module: Dict[nn.Module, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                return loss(module)

            # Name of new composable loss instance
            name = loss.__name__
            # All targets being used in the composable loss instance
            target = loss.target
            return opt.loss.CompositeLoss(loss_fn, name=name, target=target)

    Using CompositeLoss with a binary op using two Loss instances:

    .. code-block:: python

        def compose_binary_loss(
            loss1: opt.loss.Loss, loss2: opt.loss.Loss
        ) -> opt.loss.CompositeLoss:
            def loss_fn(
                module: Dict[nn.Module, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                # Operation using 2 loss instances
                return loss1(module) + loss2(module)

            # Name of new composable loss instance
            name = "Compose(" + ", ".join([loss1.__name__, loss2.__name__]) + ")"

            # All targets being used in the composable loss instance
            target1 = loss1.target if type(loss1.target) is list else [loss1.target]
            target2 = loss2.target if type(loss2.target) is list else [loss2.target]
            target = target1 + target2

            # Remove duplicate targets
            target = list(dict.fromkeys(target))
            return opt.loss.CompositeLoss(loss_fn, name=name, target=target)

    Using CompositeLoss with a list of Loss instances:

    .. code-block:: python

        def compose_multiple_loss(loss: List[opt.loss.Loss]) -> opt.loss.CompositeLoss:
            def loss_fn(
                module: Dict[nn.Module, Optional[torch.Tensor]]
            ) -> torch.Tensor:
                loss_tensors = [loss_obj(module) for loss_obj in loss]
                # We can use any operation that combines the list of tensors into a
                # single tensor
                return sum(loss_tensors)

            # Name of new composable loss instance
            name = "Compose(" + ", ".join([obj.__name__ for obj in loss]) + ")"

            # All targets being used in the composable loss instance
            # targets will either be List[nn.Module] or nn.Module
            targets = [loss_obj.target for loss_obj in loss]
            # Flatten list of targets
            target = [
                o for l in [t if type(t) is list else [t] for t in targets] for o in l
            ]
            # Remove duplicate targets
            target = list(dict.fromkeys(target))
            return opt.loss.CompositeLoss(loss_fn, name=name, target=target)
    """

    def __init__(
        self,
        loss_fn: Callable,
        name: str = "",
        target: Union[nn.Module, List[nn.Module]] = [],
    ) -> None:
        """
        Args:

            loss_fn (Callable): A function that takes a dict of captured activations
                with nn.Modules as keys, and then passes those activations through loss
                objective(s) & math operations.
            name (str, optional): The name of all composable operations in the
                instance.
                Default: ``""``
            target (nn.Module or list of nn.Module): A target nn.Module or list of
                nn.Module.
        """
        super().__init__(target)
        self.__name__ = name
        self.loss_fn = loss_fn

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        """
        Pass collected activations through the loss function.

        Args:

            module (ModuleOutputMapping): A dict of captured activations with
                nn.Modules as keys.

        Returns:
            loss (torch.Tensor): The target activations after being run through the
                loss function.
        """
        return self.loss_fn(targets_to_values)


class LayerActivation(BaseLoss):
    """
    Maximize activations at the target layer.
    This is the most basic loss available and it simply returns the activations in
    their original form.
    """

    def __init__(
        self,
        target: nn.Module,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return activations


class ChannelActivation(BaseLoss):
    """
    Maximize activations at the target layer and target channel.
    This loss maximizes the activations of a target channel in a specified target
    layer, and can be useful to determine what features the channel is excited by.
    """

    def __init__(
        self,
        target: nn.Module,
        channel_index: int,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            channel_index (int): The index of the channel to optimize for.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set to
                ``None``, defaults to all activations in the batch. Index ranges should
                be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        # ensure channel_index is valid
        assert self.channel_index < activations.shape[1]
        # assume NCHW
        # NOTE: not necessarily true e.g. for Linear layers
        # assert len(activations.shape) == 4
        return activations[
            self.batch_index[0] : self.batch_index[1], self.channel_index, ...
        ]


class NeuronActivation(BaseLoss):
    """
    This loss maximizes the activations of a target neuron in the specified channel
    from the specified layer. This loss is useful for determining the type of features
    that excite a neuron, and thus is often used for circuits and neuron related
    research.
    """

    def __init__(
        self,
        target: nn.Module,
        channel_index: int,
        x: Optional[int] = None,
        y: Optional[int] = None,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            channel_index (int): The index of the channel to optimize for.
            x (int, optional): The x coordinate of the neuron to optimize for. If
                unspecified, defaults to center, or one unit left of center for even
                lengths.
                Default: ``None``
            y (int, optional): The y coordinate of the neuron to optimize for. If
                unspecified, defaults to center, or one unit up of center for even
                heights.
                Default: ``None``
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set to
                ``None``, defaults to all activations in the batch. Index ranges should
                be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.channel_index = channel_index
        self.x = x
        self.y = y

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations is not None
        assert self.channel_index < activations.shape[1]
        assert len(activations.shape) == 4  # assume NCHW
        _x, _y = get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )
        return activations[
            self.batch_index[0] : self.batch_index[1],
            self.channel_index,
            _x : _x + 1,
            _y : _y + 1,
        ]


class DeepDream(BaseLoss):
    """
    Maximize 'interestingness' at the target layer.
    Mordvintsev et al., 2015.
    https://github.com/google/deepdream

    This loss returns the squared layer activations. When combined with a negative
    mean loss summarization, this loss will create hallucinogenic visuals commonly
    referred to as 'Deep Dream'.

    DeepDream tries to increase the values of neurons proportional to the amount
    they are presently active. This is equivalent to maximizing the sum of the
    squares. If you remove the square, you'd be visualizing a direction of:
    ``[1,1,1,....]`` (which is same as :class:`.LayerActivation`).
    """

    def __init__(
        self,
        target: nn.Module,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return activations**2


class TotalVariation(BaseLoss):
    """
    Total variation denoising penalty for activations.
    See Mahendran, V. 2014. Understanding Deep Image Representations by Inverting Them.
    https://arxiv.org/abs/1412.0035
    This loss attempts to smooth / denoise the target by performing total variance
    denoising. The target is most often the image that’s being optimized. This loss is
    often used to remove unwanted visual artifacts.
    """

    def __init__(
        self,
        target: nn.Module,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        x_diff = activations[..., 1:, :] - activations[..., :-1, :]
        y_diff = activations[..., :, 1:] - activations[..., :, :-1]
        return torch.sum(torch.abs(x_diff)) + torch.sum(torch.abs(y_diff))


class L1(BaseLoss):
    """
    L1 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self,
        target: nn.Module,
        constant: float = 0.0,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            constant (float): Constant threshold to deduct from the activations.
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set to
                ``None``, defaults to all activations in the batch. Index ranges should
                be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.constant = constant

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return torch.abs(activations - self.constant).sum()


class L2(BaseLoss):
    """
    L2 norm of the target layer, generally used as a penalty.
    """

    def __init__(
        self,
        target: nn.Module,
        constant: float = 0.0,
        eps: float = 1e-6,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            constant (float): Constant threshold to deduct from the activations.
                Default: ``0.0``
            eps (float): Small value to add to L2 prior to sqrt.
                Default: ``1e-6``
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set to
                ``None``, defaults to all activations in the batch. Index ranges should
                be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.constant = constant
        self.eps = eps

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target][
            self.batch_index[0] : self.batch_index[1]
        ]
        activations = ((activations - self.constant) ** 2).sum()
        return torch.sqrt(self.eps + activations)


class Diversity(BaseLoss):
    """
    Use a cosine similarity penalty to extract features from a polysemantic neuron.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#diversity
    This loss helps break up polysemantic layers, channels, and neurons by encouraging
    diversity across the different batches. This loss is to be used along with a main
    loss.
    """

    def __init__(
        self,
        target: nn.Module,
        batch_index: Optional[List[int]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            batch_index (list of int, optional): The index range of activations to
                optimize. If set to ``None``, defaults to all activations in the batch.
                Index ranges should be in the format of: [start, end].
                Default: ``None``
        """
        if batch_index:
            assert isinstance(batch_index, (list, tuple))
            assert len(batch_index) == 2
        BaseLoss.__init__(self, target, batch_index)

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        batch, channels = activations.shape[:2]
        flattened = activations.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = nn.functional.normalize(grams, p=2, dim=(1, 2))
        return (
            -sum(
                [
                    sum([(grams[i] * grams[j]).sum() for j in range(batch) if j != i])
                    for i in range(batch)
                ]
            )
            / batch
        )


class ActivationInterpolation(BaseLoss):
    """
    Interpolate between two different layers & channels.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons
    This loss helps to interpolate or mix visualizations from two activations (layer or
    channel) by interpolating a linear sum between the two activations.
    """

    def __init__(
        self,
        target1: nn.Module = None,
        channel_index1: Optional[int] = None,
        target2: nn.Module = None,
        channel_index2: Optional[int] = None,
    ) -> None:
        """
        Args:

            target1 (nn.Module): The first layer, transform, or image parameterization
                instance to optimize the output for.
            channel_index1 (int, optional): Index of channel in first target to
                optimize. Default is set to ``None`` for all channels.
                Default: ``None``
            target2 (nn.Module): The second layer, transform, or image parameterization
                instance to optimize the output for.
            channel_index2 (int, optional): Index of channel in second target to
                optimize. Default is set to ``None`` for all channels.
                Default: ``None``
        """
        self.target_one = target1
        self.channel_index_one = channel_index1
        self.target_two = target2
        self.channel_index_two = channel_index2
        # Expose targets for InputOptimization
        BaseLoss.__init__(self, [target1, target2])

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations_one = targets_to_values[self.target_one]
        activations_two = targets_to_values[self.target_two]

        assert activations_one is not None and activations_two is not None
        # ensure channel indices are valid
        if self.channel_index_one:
            assert self.channel_index_one < activations_one.shape[1]
        if self.channel_index_two:
            assert self.channel_index_two < activations_two.shape[1]

        assert activations_one.size(0) == activations_two.size(0)

        if self.channel_index_one:
            activations_one = activations_one[:, self.channel_index_one]
        if self.channel_index_two:
            activations_two = activations_two[:, self.channel_index_two]
        B = activations_one.size(0)

        batch_weights = torch.arange(B, device=activations_one.device) / (B - 1)
        sum_tensor = torch.zeros(1, device=activations_one.device)
        for n in range(B):
            sum_tensor = (
                sum_tensor + ((1 - batch_weights[n]) * activations_one[n]).mean()
            )
            sum_tensor = sum_tensor + (batch_weights[n] * activations_two[n]).mean()
        return sum_tensor


class Alignment(BaseLoss):
    """
    Penalize the L2 distance between tensors in the batch to encourage visual
    similarity between them.
    Olah, Mordvintsev & Schubert, 2017.
    https://distill.pub/2017/feature-visualization/#Interaction-between-Neurons
    When interpolating between activations, it may be desirable to keep image landmarks
    in the same position for visual comparison. This loss helps to minimize L2 distance
    between neighbouring images.
    """

    def __init__(
        self,
        target: nn.Module,
        decay_ratio: float = 2.0,
        batch_index: Optional[List[int]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            decay_ratio (float): How much to decay penalty as images move apart in
                the batch.
                Default: ``2.0``
            batch_index (list of int, optional): The index range of activations to
                optimize. If set to ``None``, defaults to all activations in the batch.
                Index ranges should be in the format of: [start, end].
                Default: ``None``
        """
        if batch_index:
            assert isinstance(batch_index, (list, tuple))
            assert len(batch_index) == 2
        BaseLoss.__init__(self, target, batch_index)
        self.decay_ratio = decay_ratio

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        B = activations.size(0)

        sum_tensor = torch.zeros(1, device=activations.device)
        for d in [1, 2, 3, 4]:
            for i in range(B - d):
                a, b = i, i + d
                activ_a, activ_b = activations[a], activations[b]
                sum_tensor = sum_tensor + (
                    (activ_a - activ_b) ** 2
                ).mean() / self.decay_ratio ** float(d)

        return -sum_tensor


class Direction(BaseLoss):
    """
    Visualize a general direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    This loss helps to visualize a specific vector direction in a layer, by maximizing
    the alignment between the input vector and the layer’s activation vector. The
    dimensionality of the vector should correspond to the number of channels in the
    layer.
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        cossim_pow: float = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            vec (torch.Tensor): Vector representing direction to align to.
            cossim_pow (float, optional): The desired cosine similarity power to use.
                Default: ``0.0``
            batch_index (int, optional): The index of activations to optimize if
                optimizing a batch of activations. If set to ``None``, defaults to
                all activations in the batch.
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.vec = vec.reshape((1, -1, 1, 1))
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        assert activations.size(1) == self.vec.size(1)
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return _dot_cossim(self.vec, activations, cossim_pow=self.cossim_pow)


class NeuronDirection(BaseLoss):
    """
    Visualize a single (x, y) position for a direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    Extends Direction loss by focusing on visualizing a single neuron within the
    kernel.
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        x: Optional[int] = None,
        y: Optional[int] = None,
        channel_index: Optional[int] = None,
        cossim_pow: float = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            vec (torch.Tensor): Vector representing direction to align to.
            x (int, optional): The x coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit left of center for
                even lengths.
                Default: ``None``
            y (int, optional): The y coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit up of center for
                even heights.
                Default: ``None``
            channel_index (int): The index of the channel to optimize for. If set to
                ``None``, then all channels will be used.
                Default: ``None``
            cossim_pow (float, optional): The desired cosine similarity power to use.
                Default: ``0.0``
            batch_index (int, optional): The index of activations to optimize if
                optimizing a batch of activations. If set to ``None``, defaults to all
                activations in the batch.
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.vec = vec.reshape((1, -1, 1, 1))
        self.x = x
        self.y = y
        self.channel_index = channel_index
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        _x, _y = get_neuron_pos(
            activations.size(2), activations.size(3), self.x, self.y
        )
        activations = activations[
            self.batch_index[0] : self.batch_index[1], :, _x : _x + 1, _y : _y + 1
        ]
        if self.channel_index is not None:
            activations = activations[:, self.channel_index, ...][:, None, ...]
        return _dot_cossim(self.vec, activations, cossim_pow=self.cossim_pow)


class AngledNeuronDirection(BaseLoss):
    """
    Visualize a direction vector with an optional whitened activation vector to
    unstretch the activation space. Compared to the traditional Direction objectives,
    this objective places more emphasis on angle by optionally multiplying the dot
    product by the cosine similarity.

    When cossim_pow is equal to 0, this objective works as a euclidean
    neuron objective. When cossim_pow is greater than 0, this objective works as a
    cosine similarity objective. An additional whitened neuron direction vector
    can optionally be supplied to improve visualization quality for some models.

    More information on the algorithm this objective uses can be found here:
    https://github.com/tensorflow/lucid/issues/116

    This Lucid equivalents of this loss objective can be found here:
    https://github.com/tensorflow/lucid/blob/master/notebooks/activation-atlas/activation-atlas-simple.ipynb
    https://github.com/tensorflow/lucid/blob/master/notebooks/activation-atlas/class-activation-atlas.ipynb

    Like the Lucid equivalents, our implementation differs slightly from the
    associated research paper.

    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/
    """

    def __init__(
        self,
        target: torch.nn.Module,
        vec: torch.Tensor,
        vec_whitened: Optional[torch.Tensor] = None,
        cossim_pow: float = 4.0,
        x: Optional[int] = None,
        y: Optional[int] = None,
        eps: float = 1.0e-4,
        batch_index: Optional[int] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            vec (torch.Tensor): A neuron direction vector to use.
            vec_whitened (torch.Tensor, optional): A whitened neuron direction vector.
                If set to ``None``, then no whitened vec will be used.
                Default: ``None``
            cossim_pow (float, optional): The desired cosine similarity power to use.
            x (int, optional): The x coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit left of center for
                even lengths.
                Default: ``None``
            y (int, optional): The y coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit up of center for
                even heights.
                Default: ``None``
            eps (float, optional): If cossim_pow is greater than zero, the desired
                epsilon value to use for cosine similarity calculations.
                Default: ``1.0e-4``
            batch_index (int, optional): The index of activations to optimize if
                optimizing a batch of activations. If set to ``None``, defaults to all
                activations in the batch.
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.vec = vec.unsqueeze(0) if vec.dim() == 1 else vec
        self.vec_whitened = vec_whitened
        self.cossim_pow = cossim_pow
        self.eps = eps
        self.x = x
        self.y = y
        if self.vec_whitened is not None:
            assert self.vec_whitened.dim() == 2
        assert self.vec.dim() == 2

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        assert activations.dim() == 4 or activations.dim() == 2
        assert activations.shape[1] == self.vec.shape[1]
        if activations.dim() == 4:
            _x, _y = get_neuron_pos(
                activations.size(2), activations.size(3), self.x, self.y
            )
            activations = activations[..., _x, _y]

        vec = (
            torch.matmul(self.vec, self.vec_whitened)[0]
            if self.vec_whitened is not None
            else self.vec
        )
        if self.cossim_pow == 0:
            return activations * vec

        dot = torch.mean(activations * vec)
        cossims = dot / (self.eps + torch.sqrt(torch.sum(activations**2)))
        return dot * torch.clamp(cossims, min=0.1) ** self.cossim_pow


class TensorDirection(BaseLoss):
    """
    Visualize a tensor direction vector.
    Carter, et al., "Activation Atlas", Distill, 2019.
    https://distill.pub/2019/activation-atlas/#Aggregating-Multiple-Images
    Extends Direction loss by allowing batch-wise direction visualization.
    """

    def __init__(
        self,
        target: nn.Module,
        vec: torch.Tensor,
        cossim_pow: float = 0.0,
        batch_index: Optional[int] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            vec (torch.Tensor): Vector representing direction to align to.
            cossim_pow (float, optional): The desired cosine similarity power to use.
                Default: ``0.0``
            batch_index (int, optional): The index of activations to optimize if
                optimizing a batch of activations. If set to ``None``, defaults to all
                activations in the batch.
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        assert vec.dim() == 4
        self.vec = vec
        self.cossim_pow = cossim_pow

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]

        assert activations.dim() == 4

        H_direction, W_direction = self.vec.shape[2:]
        H_activ, W_activ = activations.shape[2:]

        H = (H_activ - H_direction) // 2
        W = (W_activ - W_direction) // 2

        activations = activations[
            self.batch_index[0] : self.batch_index[1],
            :,
            H : H + H_direction,
            W : W + W_direction,
        ]
        return _dot_cossim(self.vec, activations, cossim_pow=self.cossim_pow)


class ActivationWeights(BaseLoss):
    """
    Apply weights to channels, neurons, or spots in the target.
    This loss weighs specific channels or neurons in a given layer, via a weight
    vector.
    """

    def __init__(
        self,
        target: nn.Module,
        weights: torch.Tensor = None,
        neuron: bool = False,
        x: Optional[int] = None,
        y: Optional[int] = None,
        wx: Optional[int] = None,
        wy: Optional[int] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance to optimize the output of.
            weights (torch.Tensor): Weights to apply to targets.
            neuron (bool): Whether target is a neuron.
                Default: ``False``
            x (int, optional): The x coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit left of center for
                even lengths.
                Default: ``None``
            y (int, optional): The y coordinate of the neuron to optimize for. If
                set to ``None``, defaults to center, or one unit up of center for
                even heights.
                Default: ``None``
            wx (int, optional): Length of neurons to apply the weights to, along the
                x-axis. Set to ``None`` for the full length.
                Default: ``None``
            wy (int, optional): Length of neurons to apply the weights to, along the
                y-axis. Set to ``None`` for the full length.
                Default: ``None``
        """
        BaseLoss.__init__(self, target)
        self.x = x
        self.y = y
        self.wx = wx
        self.wy = wy
        self.weights = weights
        self.neuron = x is not None or y is not None or neuron
        assert (
            wx is None
            and wy is None
            or wx is not None
            and wy is not None
            and x is not None
            and y is not None
        )

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        if self.neuron:
            assert activations.dim() == 4
            if self.wx is None and self.wy is None:
                _x, _y = get_neuron_pos(
                    activations.size(2), activations.size(3), self.x, self.y
                )
                activations = (
                    activations[..., _x : _x + 1, _y : _y + 1].squeeze() * self.weights
                )
            else:
                activations = activations[
                    ..., self.y : self.y + self.wy, self.x : self.x + self.wx
                ] * self.weights.view(1, -1, 1, 1)
        else:
            activations = activations * self.weights.view(1, -1, 1, 1)
        return activations


class L2Mean(BaseLoss):
    """
    Simple L2Loss penalty where the mean is used instead of the square root of the
    sum.

    Used for CLIP models in https://distill.pub/2021/multimodal-neurons/ as per the
    supplementary code:
    https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py
    """

    def __init__(
        self,
        target: torch.nn.Module,
        channel_index: Optional[int] = None,
        constant: float = 0.5,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer, transform, or image parameterization
                instance.
            channel_index (int, optional): Optionally only target a specific channel.
                If set to ``None``, all channels with be used.
                Default: ``None``
            constant (float, optional): Constant value to deduct from the activations.
                Default: ``0.5``
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        self.constant = constant
        self.channel_index = channel_index

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target][
            self.batch_index[0] : self.batch_index[1]
        ]
        if self.channel_index is not None:
            activations = activations[:, self.channel_index : self.channel_index + 1]
        return ((activations - self.constant) ** 2).mean()


class VectorLoss(BaseLoss):
    """
    This objective is useful for optimizing towards channel directions. This can
    helpful for visualizing models like OpenAI's CLIP.

    This loss objective is similar to the Direction objective, except it computes the
    matrix product of the activations and vector, rather than the cosine similarity.
    In addition to optimizing towards channel directions, this objective can also
    perform a similar role to the ChannelActivation objective by using one-hot 1D
    vectors.

    See here for more details:
    https://distill.pub/2021/multimodal-neurons/
    https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py
    """

    def __init__(
        self,
        target: torch.nn.Module,
        vec: torch.Tensor,
        activation_fn: Optional[Callable] = torch.nn.functional.relu,
        move_channel_dim_to_final_dim: bool = True,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            target (nn.Module): A target layer instance.
            vec (torch.Tensor): A 1D channel vector with the same size as the
                channel / feature dimension of the target layer instance.
            activation_fn (callable, optional): An optional activation function to
                apply to the activations before computing the matrix product. If set
                to ``None``, then no activation function will be used.
                Default: ``torch.nn.functional.relu``
            move_channel_dim_to_final_dim (bool, optional): Whether or not to move the
                channel dimension to the last dimension before computing the matrix
                product. Set to ``False`` if the using the channels last format.
                Default: ``True``
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, target, batch_index)
        assert vec.dim() == 1
        self.vec = vec
        self.activation_fn = activation_fn
        self.move_channel_dim_to_final_dim = move_channel_dim_to_final_dim

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations = targets_to_values[self.target]
        activations = activations[self.batch_index[0] : self.batch_index[1]]
        return _create_new_vector(
            activations,
            vec=self.vec,
            activation_fn=self.activation_fn,
            move_channel_dim_to_final_dim=self.move_channel_dim_to_final_dim,
        ).mean()


class FacetLoss(BaseLoss):
    """
    The Facet loss objective used for Faceted Feature Visualization as described in:
    https://distill.pub/2021/multimodal-neurons/#faceted-feature-visualization
    https://github.com/openai/CLIP-featurevis/blob/master/example_facets.py

    The FacetLoss objective allows us to steer feature visualization towards a
    particular theme / concept. This is done by using the weights from linear probes
    trained on the lower layers of a model to discriminate between a certain theme or
    concept and generic natural images.
    """

    def __init__(
        self,
        vec: torch.Tensor,
        ultimate_target: torch.nn.Module,
        layer_target: Union[torch.nn.Module, List[torch.nn.Module]],
        facet_weights: torch.Tensor,
        strength: Optional[Union[float, List[float]]] = None,
        batch_index: Optional[Union[int, List[int]]] = None,
    ) -> None:
        """
        Args:

            vec (torch.Tensor): A 1D channel vector with the same size as the
                channel / feature dimension of ultimate_target.
            ultimate_target (nn.Module): The main target layer that we are
                visualizing targets from. This is normally the penultimate layer of
                the model.
            layer_target (nn.Module): A layer that we have facet_weights for. This
                target layer should be below the ``ultimate_target`` layer in the
                model.
            facet_weights (torch.Tensor): Weighting that steers the objective
                towards a particular theme or concept. These weight values should
                come from linear probes trained on ``layer_target``.
            strength (float, list of float, optional): A single float or list of floats
                to use for batch dimension weighting. If using a single value, then it
                will be applied to all batch dimensions equally. Otherwise a list of
                floats with a shape of: [start, end] should be used for
                :func:`torch.linspace` to calculate the step values in between. Default
                is set to ``None`` for no weighting.
                Default: ``None``
            batch_index (int or list of int, optional): The index or index range of
                activations to optimize if optimizing a batch of activations. If set
                to ``None``, defaults to all activations in the batch. Index ranges
                should be in the format of: [start, end].
                Default: ``None``
        """
        BaseLoss.__init__(self, [ultimate_target, layer_target], batch_index)
        self.ultimate_target = ultimate_target
        self.layer_target = layer_target
        assert vec.dim() == 1
        self.vec = vec
        if isinstance(strength, (tuple, list)):
            assert len(strength) == 2
        self.strength = strength
        assert facet_weights.dim() == 4 or facet_weights.dim() == 2
        self.facet_weights = facet_weights

    def __call__(self, targets_to_values: ModuleOutputMapping) -> torch.Tensor:
        activations_ultimate = targets_to_values[self.ultimate_target]
        activations_ultimate = activations_ultimate[
            self.batch_index[0] : self.batch_index[1]
        ]
        new_vec = _create_new_vector(activations_ultimate, self.vec)
        target_activations = targets_to_values[self.layer_target]

        layer_grad = torch.autograd.grad(
            outputs=new_vec,
            inputs=target_activations,
            grad_outputs=torch.ones_like(new_vec),
            retain_graph=True,
        )[0].detach()[self.batch_index[0] : self.batch_index[1]]
        layer = target_activations[self.batch_index[0] : self.batch_index[1]]

        flat_attr = layer * torch.nn.functional.relu(layer_grad)
        if self.facet_weights.dim() == 2 and flat_attr.dim() == 4:
            flat_attr = torch.sum(flat_attr, dim=(2, 3))

        if self.strength:
            if isinstance(self.strength, (tuple, list)):
                strength_t = torch.linspace(
                    self.strength[0],
                    self.strength[1],
                    steps=flat_attr.shape[0],
                    device=flat_attr.device,
                ).reshape(flat_attr.shape[0], *[1] * (flat_attr.dim() - 1))
            else:
                strength_t = self.strength
            flat_attr = strength_t * flat_attr

        if (
            self.facet_weights.dim() == 4
            and layer.dim() == 4
            and self.facet_weights.shape[2:] != layer.shape[2:]
        ):
            facet_weights = torch.nn.functional.interpolate(
                self.facet_weights, size=layer.shape[2:]
            )
        else:
            facet_weights = self.facet_weights

        return torch.sum(flat_attr * facet_weights)


def sum_loss_list(
    loss_list: List,
    to_scalar_fn: Callable[[torch.Tensor], torch.Tensor] = torch.mean,
) -> CompositeLoss:
    """
    Summarize a large number of losses without recursion errors. By default using 300+
    loss objectives for a single optimization task will result in exceeding Python's
    default maximum recursion depth limit. This function can be used to avoid the
    recursion depth limit for tasks such as summarizing a large list of loss objectives
    with the built-in sum() function.

    This function works similar to Lucid's optvis.objectives.Objective.sum() function.

    Args:

        loss_list (list): A list of loss objectives.
        to_scalar_fn (Callable): A function for converting loss objective outputs to
            scalar values, in order to prevent size mismatches. Set to
            :class:`torch.nn.Identity` for no reduction op.
            Default: :func:`torch.mean`

    Returns:
        loss_fn (CompositeLoss): A CompositeLoss instance containing all the loss
            functions from ``loss_list``.
    """

    def loss_fn(module: ModuleOutputMapping) -> torch.Tensor:
        """
        Pass collected activations through the list of loss objectives based on
        specified targets, and then apply a reduction op to reduce them to scalar
        before adding them together.

        Args:

            module (ModuleOutputMapping): A dict of captured activations with
                nn.Modules as keys.

        Returns:
            loss (torch.Tensor): The target activations after being run through the
                loss objectives, and then added together.
        """
        return sum([to_scalar_fn(loss(module)) for loss in loss_list])

    name = "Sum(" + ", ".join([loss.__name__ for loss in loss_list]) + ")"
    # Collect targets from losses
    target = [
        target
        for targets in [
            [loss.target] if not isinstance(loss.target, list) else loss.target
            for loss in loss_list
        ]
        for target in targets
    ]

    # Filter out duplicate targets
    target = list(dict.fromkeys(target))
    return CompositeLoss(loss_fn, name=name, target=target)


def default_loss_summarize(loss_value: torch.Tensor) -> torch.Tensor:
    """
    Helper function to summarize tensor outputs from loss objectives.

    default_loss_summarize applies :func:`torch.mean` to the loss tensor
    and negates it so that optimizing it maximizes the activations we
    are interested in.

    Args:

        loss_value (torch.Tensor): A tensor containing the loss values.

    Returns:
        loss_value (torch.Tensor): The loss_value's mean multiplied by -1.
    """
    return -1 * loss_value.mean()


__all__ = [
    "Loss",
    "BaseLoss",
    "CompositeLoss",
    "LayerActivation",
    "ChannelActivation",
    "NeuronActivation",
    "DeepDream",
    "TotalVariation",
    "L1",
    "L2",
    "Diversity",
    "ActivationInterpolation",
    "Alignment",
    "Direction",
    "NeuronDirection",
    "AngledNeuronDirection",
    "TensorDirection",
    "ActivationWeights",
    "L2Mean",
    "VectorLoss",
    "FacetLoss",
    "sum_loss_list",
    "default_loss_summarize",
]
