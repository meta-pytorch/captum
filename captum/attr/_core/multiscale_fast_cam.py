#!/usr/bin/env python3
from typing import Any, Callable, List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from captum.log import log_usage

from ..._utils.typing import ModuleOrModuleList
from .._utils.attribution import GradientAttribution
from .layer.layer_activation import LayerActivation


class MultiscaleFastCam(GradientAttribution):
    r"""
    Compute saliency map using Saliency Map Order Equivalence (SMOE). This
    method first computes the layer activation, then passes activations through
    a nonlinear SMOE function.

    The recommended use case for FastCAM is to compute saliency maps for multiple
    layers with different scales in a deep network, then combine them to obtain
    a more meaningful saliency map for the original input. For details, please
    refer to the example in the docstring of `attribute()`.

    More details regrading FastCam can be found in the original paper:
    https://arxiv.org/abs/1911.11293
    """

    def __init__(
        self,
        forward_func: Callable,
        layers: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
    ) -> None:
        r"""
        Args:
            forward_func (callable): The forward function of the model or any
                          modification of it
            layers (torch.nn.Module or list(torch.nn.Module)): A list of layers
                          for which attributions.
                          are computed.
            device_ids (list(int)): Device ID list, necessary only if forward_func
                          applies a DataParallel model. This allows reconstruction of
                          intermediate outputs from batched results across devices.
                          If forward_func is given as the DataParallel model itself,
                          then it is not necessary to provide this argument.
        """
        GradientAttribution.__init__(self, forward_func)
        self.layer_act = LayerActivation(forward_func, layers, device_ids)
        self.layers = layers  # type: ModuleOrModuleList

    @log_usage()
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        scale: str = "smoe",
        norm: str = "gaussian",
        weights: List[float] = None,
        combine: bool = True,
        resize_mode: str = "bilinear",
        relu_attribution: bool = False,
        additional_forward_args: Any = None,
        attribute_to_layer_input: bool = False,
    ) -> Tuple[Tensor, ...]:
        r"""
        Args:

            inputs (tensor):  Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
            scale (str, optional): The choice of scale to pass through attributes.
                        The available options are:

                        - `smoe`: Saliency Map Order Equivalence, in which pixels
                        are approximating the scale of Gamma distribution. See paper
                        for more details.

                        - `std`: Standard Deviation of all the pixel values, with
                        respect to the inputs' channel dimension.

                        - `mean`: Mean of all the pixel values, with respect to
                        the inputs' channel dimension.

                        - `max`: Maximum of all the pixel values, with respect to
                        the inputs' channel dimension.

                        - `normal`: Normal Entropy Scale function
                        Default: `smoe`
            norm (str, callable, optional): The choice of normalization after scaling.
                        The available options are:

                        - `gamma`: Normalize via the Gamma distribution.

                        - `gaussian`: Normalize via the Gaussian distribution.

                        - `identity`: The identity function; no normalization.
                        Default: `gaussian`
            weights (list(float), optional): Weight of each layer of attribution. If
                        `None`, then each layer will have equal attribution to the final
                        combined saliency map.
                        Default: None
            combine (bool, optional): Return the combined attributes if true. If
                        False, return the weighted maps individually.
                        Default: True
            resize_mode (str, optional): An argument to interpolation method for
                        rescaling.
                        Default: `bilinear`
            relu_attribution (bool, optional): An option to pass final attributes
                        through a ReLU function before returning.
                        Default: False
            additional_forward_args (any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Note that currently it is assumed that either the input
                        or the output of internal layer, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False

        Returns:
            *tensor* or tuple of *tensors* of **attributions**:
            - **attributions** (*tensor* or tuple of *tensors*):
                        Depending on the value of `combine`. If `combine` is set to
                        True, then the number of attributions will be the batch size
                        of the inputs. If `combine` is set to False, then each input
                        will have a number of attributions, depending on the number
                        of layers passed in during object instantiation.
                        The attributions with respect to each input feature.
                        If the forward function returns
                        a scalar value per example, attributions will be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If the forward function returns a scalar per batch, then
                        attribution tensor(s) will have first dimension 1 and
                        the remaining dimensions will match the input.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple of tensors is provided for inputs, a
                        tuple of corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and have multiple convolutional layers. Suppose we combine the
            >>> # saliency map of three ReLU layers, which are
            >>> # instances of nn.conv2d.
            >>> # We will also use `smoe` scale and `gaussian` normalization,
            >>> # which often yields the best results.
            >>> net = ImageClassifier()
            >>> fastcam = MultiscaleFastCam(net, [net.conv1, net.conv2, net.conv3])
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # We set here that each layer we selected contributes equally to our
            >>> # combined saliency map.
            >>> attribution = fastcam.attribute(input, scale='smoe', norm='gaussian')
            >>> # If we want to obtain each layer's attribution individually, we
            >>> # set `combine=False`
            >>> attribution = fastcam.attribute(input, scale='smoe',
                                                norm='gaussian', combine=False)
        """
        # pick out functions
        self.scale_func = self.pick_scale_func(scale)
        self.norm_func = self.pick_norm_func(norm)
        if not weights:
            weights = np.ones(len(self.layers))

        layer_attrs = self.layer_act.attribute(
            inputs, additional_forward_args, attribute_to_layer_input
        )
        attributes = []
        for layer_attr in layer_attrs:
            scaled_attr = self.scale_func(layer_attr)
            normed_attr = self.norm_func(scaled_attr)
            attributes.append(normed_attr)
        attributes = tuple(attributes)

        # Combine
        bn, channels, height, width = inputs.shape
        combined_map = torch.zeros(
            (bn, 1, height, width),
            dtype=attributes[0].dtype,
            device=attributes[0].device,
        )
        weighted_maps = [[] for _ in range(bn)]  # type: List[List[Any]]
        for m, smap in enumerate(attributes):
            for i in range(bn):
                w = F.interpolate(
                    smap[i].unsqueeze(0).unsqueeze(0),
                    size=(height, width),
                    mode=resize_mode,
                    align_corners=False,
                ).squeeze()
                weighted_maps[i].append(w)
                combined_map[i] += w * weights[m]
        combined_map = combined_map / np.sum(weights)
        weighted_maps = torch.stack([torch.stack(wmaps) for wmaps in weighted_maps])

        if relu_attribution:
            combined_map = F.relu(combined_map)
            weighted_maps = F.relu(weighted_maps)

        if not combine:
            return weighted_maps
        return combined_map

    def pick_norm_func(self, norm):
        norm = norm.lower()
        if norm == "gamma":
            norm_func = self._compute_gamma_norm
        elif norm == "gaussian":
            norm_func = self._compute_gaussian_norm
        elif norm == "identity":

            def identity(x):
                return x.squeeze(1)

            norm_func = identity
        else:
            msg = (
                f"{norm} norming option not found or invalid. "
                + "Available options: [gamma, gaussian, identity]"
            )
            raise NameError(msg)
        return norm_func

    def pick_scale_func(self, scale):
        scale = scale.lower()
        if scale == "smoe":
            scale_func = self._compute_smoe_scale
        elif scale == "std":
            scale_func = self._compute_std_scale
        elif scale == "mean":
            scale_func = self._compute_mean_scale
        elif scale == "max":
            scale_func = self._compute_max_scale
        elif scale == "normal":
            scale_func = self._compute_normal_entropy_scale
        elif scale == "identity":

            def identity(x):
                return x

            scale_func = identity
        else:
            msg = (
                f"{scale} scaling option not found or invalid. "
                + "Available options: [smoe, std, mean, max, normal, identity]"
            )
            raise NameError(msg)
        return scale_func

    def _compute_smoe_scale(self, inputs):
        x = inputs + 1e-7
        m = x.mean(dim=1, keepdims=True)
        k = torch.log2(m) - torch.log2(x).mean(dim=1, keepdims=True)
        th = k * m
        return th

    def _compute_std_scale(self, inputs):
        return torch.std(inputs, dim=1, keepdim=True)

    def _compute_mean_scale(self, inputs):
        return torch.mean(inputs, dim=1, keepdim=True)

    def _compute_max_scale(self, inputs):
        return torch.max(inputs, dim=1, keepdim=True).values

    def _compute_normal_entropy_scale(self, inputs):
        c1 = torch.tensor(0.3989422804014327)  # 1.0/math.sqrt(2.0*math.pi)
        c2 = torch.tensor(1.4142135623730951)  # math.sqrt(2.0)
        c3 = torch.tensor(4.1327313541224930)

        def _compute_alpha(mean, std, a=0):
            return (a - mean) / std

        def _compute_pdf(eta):
            return c1 * torch.exp(-0.5 * eta.pow(2.0))

        def _compute_cdf(eta):
            e = torch.erf(eta / c2)
            return 0.5 * (1.0 + e) + 1e-7

        m = torch.mean(inputs, dim=1)
        s = torch.std(inputs, dim=1) + 1e-7
        a = _compute_alpha(m, s)
        pdf = _compute_pdf(a)
        cdf = _compute_cdf(a)
        Z = 1.0 - cdf
        T1 = torch.log(c3 * s * Z)
        T2 = (a * pdf) / (2.0 * Z)
        ent = T1 + T2
        return ent.unsqueeze(1)

    def _compute_gaussian_norm(self, inputs):
        b, _, h, w = inputs.size()
        x = inputs.reshape(b, h * w)
        m = x.mean(dim=1, keepdims=True)
        s = x.std(dim=1, keepdims=True)
        x = 0.5 * (1.0 + torch.erf((x - m) / (s * torch.sqrt(torch.tensor(2.0)))))
        x = x.reshape(b, h, w)
        return x

    def _compute_gamma_norm(self, inputs):
        def _gamma(z):
            x = torch.ones_like(z) * 0.99999999999980993
            for i in range(8):
                i1 = torch.tensor(i + 1.0)
                x = x + cheb[i] / (z + i1)
            t = z + 8.0 - 0.5
            y = two_pi * t.pow(z + 0.5) * torch.exp(-t) * x
            y = y / z
            return y

        def _lower_incl_gamma(s, x, _iter=8):
            _iter = _iter - 2
            gs = _gamma(s)
            L = x.pow(s) * gs * torch.exp(-x)
            gs *= s  # Gamma(s + 1)
            R = torch.reciprocal(gs) * torch.ones_like(x)
            X = x  # x.pow(1)
            for k in range(_iter):
                gs *= s + k + 1.0  # Gamma(s + k + 2)
                R += X / gs
                X = X * x  # x.pow(k+1)
            gs *= s + _iter + 1.0  # Gamma(s + iter + 2)
            R += X / gs
            return L * R

        def _trigamma(x):
            z = x + 1.0
            zz = z.pow(2.0)
            a = 0.2 - torch.reciprocal(7.0 * zz)
            b = 1.0 - a / zz
            c = 1.0 + b / (3.0 * z)
            d = 1.0 + c / (2.0 * z)
            e = d / z + torch.reciprocal(x.pow(2.0))
            return e

        def _k_update(k, s):
            nm = torch.log(k) - torch.digamma(k) - s
            dn = torch.reciprocal(k) - _trigamma(k)
            k2 = k - nm / dn
            return k2

        def _compute_ml_est(x, i=10):
            x = x + eps
            s = torch.log(x.mean(dim=1, keepdims=True))
            s = s - torch.log(x).mean(dim=1, keepdims=True)
            s3 = s - 3.0
            rt = torch.sqrt(s3.pow(2.0) + 24.0 * s)
            nm = 3.0 - s + rt
            dn = 12.0 * s
            k = nm / dn + eps
            for _ in range(i):
                k = _k_update(k, s)
            k = torch.clamp(k, eps, 18.0)
            th = torch.reciprocal(k) * torch.mean(x, dim=1, keepdims=True)
            return k, th

        cheb = torch.tensor(
            [
                676.5203681218851,
                -1259.1392167224028,
                771.32342877765313,
                -176.61502916214059,
                12.507343278686905,
                -0.13857109526572012,
                9.9843695780195716e-6,
                1.5056327351493116e-7,
            ]
        )
        eps = 1e-7
        two_pi = torch.tensor(np.sqrt(2.0 * np.pi))
        b, c, h, w = inputs.size()
        x = inputs.reshape(b, h * w)
        x = x - torch.min(x, dim=1, keepdims=True)[0] + 1e-7
        k, th = _compute_ml_est(x)
        x = (1.0 / _gamma(k)) * _lower_incl_gamma(k, x / th)
        x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
        output = x.reshape(b, h, w)
        return output
