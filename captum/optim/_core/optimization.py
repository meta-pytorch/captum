import warnings
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn
import torch.optim as optim

try:
    from tqdm.auto import tqdm
except (ImportError, AssertionError):
    print(
        "The tqdm package is required to use captum.optim's"
        + " n_steps stop criteria with progress bar"
    )

from captum.optim._core.loss import default_loss_summarize
from captum.optim._core.output_hook import ModuleOutputsHook
from captum.optim._param.image.images import InputParameterization, NaturalImage
from captum.optim._param.image.transforms import RandomScale, RandomSpatialJitter
from captum.optim._utils.typing import (
    LossFunction,
    Objective,
    Parameterized,
    StopCriteria,
)


class InputOptimization(Objective, Parameterized):
    """
    Core function that optimizes an input to maximize a target (aka objective).
    This is similar to gradient-based methods for adversarial examples, such
    as :class:`FGSM <captum.robust.FGSM>`. The code for this was based on the
    implementation by the authors of Lucid. For more details, see the following:

      * https://github.com/tensorflow/lucid
      * https://distill.pub/2017/feature-visualization/

    Alias: ``captum.optim.InputOptimization``

    Example::

        >>> model = opt.models.googlenet(pretrained=True)
        >>> loss_fn = opt.loss.LayerActivation(model.mixed4c)
        >>> image = opt.images.NaturalImage(size=(224, 224))
        >>> transform = opt.transforms.TransformationRobustness()
        >>>
        >>> obj = opt.InputOptimization(model, loss_fn, image, transform)
        >>> history = obj.optimize(opt.optimization.n_steps(512))
        >>> image().show(figsize=(10, 10)) # Display results
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        loss_function: LossFunction,
        input_param: Optional[InputParameterization] = None,
        transform: Optional[nn.Module] = None,
    ) -> None:
        r"""
        Args:

            model (nn.Module, optional): The reference to PyTorch model instance. Set
                to ``None`` for no model instance.
            loss_function (Callable): The :mod:`Loss <.loss>` objective instance to
                minimize during optimization.
            input_param (InputParameterization, optional): A module that generates an
                input, consumed by the model. Example: An
                :mod:`ImageParameterization <captum.optim.images>` instance.
            transform (nn.Module, optional): A module that transforms or preprocesses
                the input before being passed to the model. Set to
                :class:`torch.nn.Identity` for no transforms.

        Instance variables that be used in the :func:`InputOptimization.optimize`
        function, custom optimization functions, and StopCriteria functions:

        Attributes:

            model (torch.nn.Module): The given model instance given when initializing
                ``InputOptimization``. If ``model`` was set to ``None`` during
                initialization, then an instance of :class:`torch.nn.Identity` will be
                returned.
            input_param (InputParameterization): The given input parameterization
                instance given when initializing ``InputOptimization``.
            loss_function (Loss): The composable :mod:`Loss <.loss>` instance given
                when initializing ``InputOptimization``.
            transform (torch.nn.Module): The given transform instance given when
                initializing ``InputOptimization``.
        """
        self.model = model or nn.Identity()
        # Grab targets from loss_function
        assert hasattr(loss_function, "target")
        if isinstance(loss_function.target, list):
            self.hooks = ModuleOutputsHook(loss_function.target)
        else:
            self.hooks = ModuleOutputsHook([loss_function.target])
        self.input_param = input_param or NaturalImage((224, 224))
        if isinstance(self.model, Iterable):
            param = next(self.model.parameters(), None)
            if param:
                self.input_param = self.input_param.to(param.device)
        self.transform = transform or torch.nn.Sequential(
            RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05)), RandomSpatialJitter(16)
        )
        self.loss_function = loss_function

    def loss(self) -> torch.Tensor:
        r"""Compute loss value for current iteration.

        Returns:
            tensor representing **loss**:
            - **loss** (torch.Tensor): Size of the tensor corresponds to the targets
                passed.
        """
        input_t = self.input_param()

        if self.transform:
            input_t = self.transform(input_t)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _unreachable = self.model(input_t)  # noqa: F841

        # consume_outputs return the captured values and resets the hook's state
        module_outputs = self.hooks.consume_outputs()
        loss_value = self.loss_function(module_outputs)
        return loss_value

    def cleanup(self) -> None:
        r"""Garbage collection, mainly removing hooks.
        This should only be run after optimize is finished running.
        """
        self.hooks.remove_hooks()

    # Targets are managed by ModuleOutputHooks; we mainly just want a convenient setter
    @property
    def targets(self) -> Iterable[nn.Module]:
        return self.hooks.targets

    @targets.setter
    def targets(self, value: Iterable[nn.Module]) -> None:
        self.hooks.remove_hooks()
        self.hooks = ModuleOutputsHook(value)

    def parameters(self) -> Iterable[nn.Parameter]:
        """
        Returns:
            parameters (iterable of torch.nn.Parameter): An iterable of parameters in
                the input parameterization.
        """
        return self.input_param.parameters()

    def optimize(
        self,
        stop_criteria: Optional[StopCriteria] = None,
        optimizer: Optional[optim.Optimizer] = None,
        loss_summarize_fn: Optional[Callable] = None,
        lr: float = 0.025,
    ) -> torch.Tensor:
        r"""Optimize input based on loss function and objectives.

        Args:

            stop_criteria (StopCriteria, optional): A function that is called
                every iteration and returns a bool that determines whether to stop the
                optimization.
                Default: :func:`n_steps(512) <.n_steps>`
            optimizer (torch.optim.Optimizer, optional): A ``torch.optim.Optimizer``
                instance to use for optimizing the input based on the loss function.
                Default: :class:`torch.optim.Adam`
            loss_summarize_fn (Callable, optional): The function to use for summarizing
                tensor outputs from loss functions.
                Default: :func:`.default_loss_summarize`
            lr (float, optional): If no optimizer is given, then lr is used as the
                learning rate for the Adam optimizer.
                Default: ``0.025``

        Returns:
            history (torch.Tensor): A stack of loss values per iteration. The size
                of the dimension on which loss values are stacked corresponds to
                the number of iterations.
        """
        stop_criteria = stop_criteria or n_steps(512)
        optimizer = optimizer or optim.Adam(self.parameters(), lr=lr)
        assert isinstance(optimizer, optim.Optimizer)
        loss_summarize_fn = loss_summarize_fn or default_loss_summarize

        history = []
        step = 0
        try:
            while stop_criteria(step, self, history, optimizer):
                optimizer.zero_grad()
                loss_value = loss_summarize_fn(self.loss())
                history.append(loss_value.clone().detach())
                loss_value.backward()
                optimizer.step()
                step += 1
        finally:
            self.cleanup()
        return torch.stack(history)


def n_steps(n: int, show_progress: bool = True) -> StopCriteria:
    """StopCriteria generator that uses number of steps as a stop criteria.

    Example::

        >>> stop_criteria = opt.optimization.n_steps(512, True)

    Args:

        n (int): Number of steps to run optimization.
        show_progress (bool, optional): Whether or not to show progress bar.
            Default: ``True``

    Returns:
        StopCriteria (Callable): A stop criteria function.
    """

    if show_progress:
        pbar = tqdm(total=n, unit=" step")

    def continue_while(
        step: int,
        obj: Objective,
        history: Iterable[torch.Tensor],
        optim: torch.optim.Optimizer,
    ) -> bool:
        if len(history) > 0:
            if show_progress:
                pbar.set_postfix(
                    {"Objective": f"{history[-1].mean():.1f}"}, refresh=False
                )
        if step < n:
            if show_progress:
                pbar.update()
            return True
        else:
            if show_progress:
                pbar.close()
            return False

    return continue_while


__all__ = [
    "InputOptimization",
    "n_steps",
]
