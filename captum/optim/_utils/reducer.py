from typing import Any, Callable, List, Union

import numpy as np
import torch.nn.functional as F

try:
    import sklearn.decomposition
    from sklearn.base import BaseEstimator
except (ImportError, AssertionError):
    print(
        "The sklearn library is required to use Captum's ChannelReducer"
        + " unless you supply your own reduction algorithm."
    )
import torch


class ChannelReducer:
    """
    The ChannelReducer class is a wrapper for PyTorch and NumPy based dimensionality
    reduction algorithms, like those from ``sklearn.decomposition`` (ex: NMF, PCA),
    ``sklearn.manifold`` (ex: TSNE), UMAP, and other libraries. This class handles
    things like reshaping, algorithm search by name (for scikit-learn only), and
    PyTorch tensor conversions to and from NumPy arrays.

    Example::

        >>> reducer = opt.reducer.ChannelReducer(2, "NMF")
        >>> x = torch.randn(1, 8, 128, 128).abs()
        >>> output = reducer.fit_transform(x)
        >>> print(output.shape)
        torch.Size([1, 2, 128, 128])

        >>> # reduction_alg attributes are easily accessible
        >>> print(reducer.components.shape)
        torch.Size([2, 8])

    Dimensionality reduction for the channel dimension of an input tensor.
    Olah, et al., "The Building Blocks of Interpretability", Distill, 2018.

    See here for more information: https://distill.pub/2018/building-blocks/

    Some of the possible algorithm choices:

     * https://scikit-learn.org/stable/modules/classes.html#module-sklearn.decomposition
     * https://scikit-learn.org/stable/modules/classes.html#module-sklearn.manifold
     * https://umap-learn.readthedocs.io/en/latest/

    Args:

        n_components (int, optional): The number of channels to reduce the target
            dimension to.
        reduction_alg (str or Callable, optional): The desired dimensionality
            reduction algorithm to use. The default ``reduction_alg`` is set to NMF
            from sklearn, which requires users to put inputs on CPU before passing them
            to :func:`ChannelReducer.fit_transform`. Name strings are only supported
            for ``sklearn.decomposition`` & ``sklearn.manifold`` class names.
            Default: ``NMF``
        **kwargs (Any, optional): Arbitrary keyword arguments used by the specified
            ``reduction_alg``.
    """

    def __init__(
        self, n_components: int = 3, reduction_alg: Any = "NMF", **kwargs
    ) -> None:
        if isinstance(reduction_alg, str):
            reduction_alg = self._get_reduction_algo_instance(reduction_alg)
            if reduction_alg is None:
                raise ValueError(
                    "Unknown sklearn dimensionality reduction method '%s'."
                    % reduction_alg
                )

        self.n_components = n_components
        self._reducer = reduction_alg(n_components=n_components, **kwargs)

    def _get_reduction_algo_instance(self, name: str) -> Union[None, Callable]:
        """
        Search through a library for a ``reduction_alg`` matching the provided str
        name.

        Args:

            name (str): The name of the reduction_alg to search for.

        Returns:
            reduction_alg (Callable or None): The ``reduction_alg`` if it was found,
                otherwise None.
        """
        if hasattr(sklearn.decomposition, name):
            obj = sklearn.decomposition.__getattribute__(name)
            if issubclass(obj, BaseEstimator):
                return obj
        elif hasattr(sklearn.manifold, name):
            obj = sklearn.manifold.__getattribute__(name)
            if issubclass(obj, BaseEstimator):
                return obj
        return None

    @classmethod
    def _apply_flat(cls, func: Callable, x: torch.Tensor) -> torch.Tensor:
        """
        Flatten inputs, run them through the reduction_alg, and then reshape them back
        to their original size using the resized dimension.

        Args:

            func (Callable): The ``reduction_alg`` transform function being used.
            x (torch.Tensor): The tensor being transformed and reduced.

        Returns:
            x (torch.Tensor): A transformed tensor.
        """
        orig_shape = x.shape
        try:
            return func(x.reshape([-1, x.shape[-1]])).reshape(
                list(orig_shape[:-1]) + [-1]
            )
        except TypeError:
            raise TypeError(
                "The provided input is incompatible with the reduction_alg. "
                "Try placing the input on CPU first via x.cpu()."
            )

    def fit_transform(
        self, x: torch.Tensor, swap_2nd_and_last_dims: bool = True
    ) -> torch.Tensor:
        """
        Perform dimensionality reduction on an input tensor using the specified
        ``reduction_alg``'s ``.fit_transform`` function.

        Args:

            x (torch.Tensor): A tensor to perform dimensionality reduction on.
            swap_2nd_and_last_dims (bool, optional): If ``True``, input channels are
                expected to be in the second dimension unless the input tensor has a
                shape of CHW. When reducing the channel dimension, this parameter
                should be set to ``True`` unless you are already using the channels
                last format.
                Default: ``True``.

        Returns:
            x (torch.Tensor): A tensor with one of it's dimensions reduced.
        """

        if x.dim() == 3 and swap_2nd_and_last_dims:
            x = x.permute(2, 1, 0)
        elif x.dim() > 3 and swap_2nd_and_last_dims:
            permute_vals = [0] + list(range(x.dim()))[2:] + [1]
            x = x.permute(*permute_vals)

        x_out = ChannelReducer._apply_flat(self._reducer.fit_transform, x)

        x_out = torch.as_tensor(x_out, device=x.device)

        if x.dim() == 3 and swap_2nd_and_last_dims:
            x_out = x_out.permute(2, 1, 0)
        elif x.dim() > 3 and swap_2nd_and_last_dims:
            permute_vals = (
                [0]
                + [x.dim() - 1]
                + list(range(x.dim()))[1 : len(list(range(x.dim()))) - 1]
            )
            x_out = x_out.permute(*permute_vals)

        return x_out

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            out = self.__dict__[name]
        elif name + "_" in self._reducer.__dict__:
            out = self._reducer.__dict__[name + "_"]
        if type(out) == np.ndarray:
            out = torch.as_tensor(out)
        return out

    def __dir__(self) -> List:
        dynamic_attrs = [
            name[:-1]
            for name in dir(self._reducer)
            if name[-1] == "_" and name[0] != "_"
        ]

        return (
            list(ChannelReducer.__dict__.keys())
            + list(self.__dict__.keys())
            + dynamic_attrs
        )


def posneg(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Hack that makes a matrix positive by concatenation in order to simulate one-sided
    NMF with regular NMF.

    Voss, et al., "Visualizing Weights", Distill, 2021.
    See: https://distill.pub/2020/circuits/visualizing-weights/

    Args:

        x (torch.Tensor): A tensor to make positive.
        dim (int, optional): The dimension to concatenate the two tensor halves at.
            Default: ``0``

    Returns:
        tensor (torch.Tensor): A positive tensor for one-sided dimensionality
            reduction.
    """

    return torch.cat([F.relu(x), F.relu(-x)], dim=dim)


__all__ = [
    "ChannelReducer",
    "posneg",
]
