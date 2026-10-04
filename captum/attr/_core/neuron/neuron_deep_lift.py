#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable, cast, Optional, Tuple, Union

from captum._utils.gradient import construct_neuron_grad_fn
from captum._utils.typing import (
    BaselineType,
    SliceIntType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._utils.attribution import GradientAttribution, NeuronAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class NeuronDeepLift(NeuronAttribution, GradientAttribution):
    r"""
    Implements DeepLIFT algorithm for the neuron based on the following paper:
    Learning Important Features Through Propagating Activation Differences,
    Avanti Shrikumar, et. al.
    https://arxiv.org/abs/1704.02685

    and the gradient formulation proposed in:
    Towards better understanding of gradient-based attribution methods for
    deep neural networks,  Marco Ancona, et.al.
    https://openreview.net/pdf?id=Sy21R9JAW

    This implementation supports only Rescale rule. RevealCancel rule will
    be supported in later releases.
    Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
    Integrated Gradients, it runs significantly faster than Integrated
    Gradients and is preferred for large datasets.

    Currently we only support a limited number of non-linear activations
    but the plan is to expand the list in the future.

    Note: As we know, currently we cannot access the building blocks,
    of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
    Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
    with performance similar to built-in ones using TorchScript.
    More details on how to build custom RNNs can be found here:
    https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
    """

    def __init__(
        self, model: Module, layer: Module, multiply_by_inputs: bool = True
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                        Attributions for a particular neuron for the input or output
                        of this layer are computed using the argument neuron_selector
                        in the attribute method.
                        Currently, it is assumed that the inputs or the outputs
                        of the layer, depending on which one is used for
                        attribution, can only be a single tensor.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of Neuron DeepLift, if `multiply_by_inputs`
                        is set to True, final sensitivity scores
                        are being multiplied by (inputs - baselines).
                        This flag applies only if `custom_attribution_func` is
                        set to None.
        """
        NeuronAttribution.__init__(self, model, layer)
        GradientAttribution.__init__(self, model)
        self._multiply_by_inputs = multiply_by_inputs

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        neuron_selector: Union[
            int,
            Tuple[Union[int, SliceIntType], ...],
            Callable[[Union[Tensor, Tuple[Tensor, ...]]], Tensor],
        ],
        baselines: BaselineType = None,
        additional_forward_args: Optional[object] = None,
        attribute_to_neuron_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attributions are computed. If model takes a
                        single tensor as input, a single input tensor should be
                        provided. If model takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            neuron_selector (int, Callable, tuple[int], or slice):
                        Selector for neuron
                        in given layer for which attribution is desired.
                        Neuron selector can be provided as:

                        - a single integer, if the layer output is 2D. This integer
                          selects the appropriate neuron column in the layer input
                          or output

                        - a tuple of integers or slice objects. Length of this
                          tuple must be one less than the number of dimensions
                          in the input / output of the given layer (since
                          dimension 0 corresponds to number of examples).
                          The elements of the tuple can be either integers or
                          slice objects (slice object allows indexing a
                          range of neurons rather individual ones).

                          If any of the tuple elements is a slice object, the
                          indexed output tensor is used for attribution. Note
                          that specifying a slice of a tensor would amount to
                          computing the attribution of the sum of the specified
                          neurons, and not the individual neurons independently.

                        - a callable, which should
                          take the target layer as input (single tensor or tuple
                          if multiple tensors are in layer) and return a neuron or
                          aggregate of the layer's neurons for attribution.
                          For example, this function could return the
                          sum of the neurons in the layer or sum of neurons with
                          activations in a particular range. It is expected that
                          this function returns either a tensor with one element
                          or a 1D tensor with length equal to batch_size (one scalar
                          per input example)

            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references.
                        Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          exactly the same dimensions as inputs or the first
                          dimension is one and the remaining dimensions match
                          with inputs.

                        - a single scalar, if inputs is a single tensor, which will
                          be broadcasted for each input value in input tensor.

                        - a tuple of tensors or scalars, the baseline corresponding
                          to each tensor in the inputs' tuple can be:

                          - either a tensor with matching dimensions to
                            corresponding tensor in the inputs' tuple
                            or the first dimension is one and the remaining
                            dimensions match with the corresponding
                            input tensor.

                          - or a scalar, corresponding to a tensor in the
                            inputs' tuple. This scalar value is broadcasted
                            for corresponding input tensor.

                        In the cases when `baselines` is not provided, we internally
                        use zero scalar corresponding to each input tensor.

                        Default: None
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided
                        to model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_neuron_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the neuron input
                        or output. If `attribute_to_neuron_input` is set to True
                        then the attributions will be computed with respect to
                        neuron's inputs, otherwise it will be computed with respect
                        to neuron's outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal neuron, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Computes attributions using Deeplift's rescale rule for
                particular neuron with respect to each input feature.
                Attributions will always be the same size as the provided
                inputs, with each value providing the attribution of the
                corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = NeuronDeepLift(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and neuron
            >>> # index (4,1,2).
            >>> attribution = dl.attribute(input, (4,1,2))
        """
        dl = DeepLift(cast(Module, self.forward_func), self.multiplies_by_inputs)
        dl.gradient_func = construct_neuron_grad_fn(
            self.layer,
            neuron_selector,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )

        # NOTE: using __wrapped__ to not log
        return dl.attribute.__wrapped__(  # type: ignore
            dl,  # self
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            custom_attribution_func=custom_attribution_func,
        )

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs


class NeuronDeepLiftShap(NeuronAttribution, GradientAttribution):
    r"""
    Extends NeuronAttribution and uses LayerDeepLiftShap algorithms and
    approximates SHAP values for given input `layer` and `neuron_selector`.
    For each input sample - baseline pair it computes DeepLift attributions
    with respect to inputs or outputs of given `layer` and `neuron_selector`
    averages resulting attributions across baselines. Whether to compute the
    attributions with respect to the inputs or outputs of the layer is defined
    by the input flag `attribute_to_layer_input`.
    More details about the algorithm can be found here:

    https://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions.pdf

    Note that the explanation model:
        1. Assumes that input features are independent of one another
        2. Is linear, meaning that the explanations are modeled through
            the additive composition of feature effects.

    Although, it assumes a linear model for each explanation, the overall
    model across multiple explanations can be complex and non-linear.
    """

    def __init__(
        self, model: Module, layer: Module, multiply_by_inputs: bool = True
    ) -> None:
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                        Attributions for a particular neuron for the input or output
                        of this layer are computed using the argument neuron_selector
                        in the attribute method.
                        Currently, only layers with a single tensor input and output
                        are supported.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of Neuron DeepLift Shap, if `multiply_by_inputs`
                        is set to True, final sensitivity scores
                        are being multiplied by (inputs - baselines).
                        This flag applies only if `custom_attribution_func` is
                        set to None.
        """
        NeuronAttribution.__init__(self, model, layer)
        GradientAttribution.__init__(self, model)
        self._multiply_by_inputs = multiply_by_inputs

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        neuron_selector: Union[
            int,
            Tuple[Union[int, SliceIntType], ...],
            Callable[[Union[Tensor, Tuple[Tensor, ...]]], Tensor],
        ],
        baselines: Union[
            TensorOrTupleOfTensorsGeneric, Callable[..., TensorOrTupleOfTensorsGeneric]
        ],
        additional_forward_args: Optional[object] = None,
        attribute_to_neuron_input: bool = False,
        custom_attribution_func: Union[None, Callable[..., Tuple[Tensor, ...]]] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer
                        attributions are computed. If model takes a
                        single tensor as input, a single input tensor should be
                        provided. If model takes multiple tensors as input,
                        a tuple of the input tensors should be provided. It is
                        assumed that for all given input tensors, dimension 0
                        corresponds to the number of examples (aka batch size),
                        and if multiple input tensors are provided, the examples
                        must be aligned appropriately.
            neuron_selector (int, Callable, tuple[int], or slice):
                        Selector for neuron
                        in given layer for which attribution is desired.
                        Neuron selector can be provided as:

                        - a single integer, if the layer output is 2D. This integer
                          selects the appropriate neuron column in the layer input
                          or output

                        - a tuple of integers or slice objects. Length of this
                          tuple must be one less than the number of dimensions
                          in the input / output of the given layer (since
                          dimension 0 corresponds to number of examples).
                          The elements of the tuple can be either integers or
                          slice objects (slice object allows indexing a
                          range of neurons rather individual ones).

                          If any of the tuple elements is a slice object, the
                          indexed output tensor is used for attribution. Note
                          that specifying a slice of a tensor would amount to
                          computing the attribution of the sum of the specified
                          neurons, and not the individual neurons independently.

                        - a callable, which should
                          take the target layer as input (single tensor or tuple
                          if multiple tensors are in layer) and return a neuron or
                          aggregate of the layer's neurons for attribution.
                          For example, this function could return the
                          sum of the neurons in the layer or sum of neurons with
                          activations in a particular range. It is expected that
                          this function returns either a tensor with one element
                          or a 1D tensor with length equal to batch_size (one scalar
                          per input example)

            baselines (Tensor, tuple[Tensor, ...], or Callable):
                        Baselines define reference samples that are compared with
                        the inputs. In order to assign attribution scores DeepLift
                        computes the differences between the inputs/outputs and
                        corresponding references. Baselines can be provided as:

                        - a single tensor, if inputs is a single tensor, with
                          the first dimension equal to the number of examples
                          in the baselines' distribution. The remaining dimensions
                          must match with input tensor's dimension starting from
                          the second dimension.

                        - a tuple of tensors, if inputs is a tuple of tensors,
                          with the first dimension of any tensor inside the tuple
                          equal to the number of examples in the baseline's
                          distribution. The remaining dimensions must match
                          the dimensions of the corresponding input tensor
                          starting from the second dimension.

                        - callable function, optionally takes `inputs` as an
                          argument and either returns a single tensor
                          or a tuple of those.

                        It is recommended that the number of samples in the baselines'
                        tensors is larger than one.
            additional_forward_args (Any, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided
                        to model in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_neuron_input (bool, optional): Indicates whether to
                        compute the attributions with respect to the neuron input
                        or output. If `attribute_to_neuron_input` is set to True
                        then the attributions will be computed with respect to
                        neuron's inputs, otherwise it will be computed with respect
                        to neuron's outputs.
                        Note that currently it is assumed that either the input
                        or the output of internal neuron, depending on whether we
                        attribute to the input or output, is a single tensor.
                        Support for multiple tensors will be added later.
                        Default: False
            custom_attribution_func (Callable, optional): A custom function for
                        computing final attribution scores. This function can take
                        at least one and at most three arguments with the
                        following signature:

                        - custom_attribution_func(multipliers)
                        - custom_attribution_func(multipliers, inputs)
                        - custom_attribution_func(multipliers, inputs, baselines)

                        In case this function is not provided, we use the default
                        logic defined as: multipliers * (inputs - baselines)
                        It is assumed that all input arguments, `multipliers`,
                        `inputs` and `baselines` are provided in tuples of same
                        length. `custom_attribution_func` returns a tuple of
                        attribution tensors that have the same length as the
                        `inputs`.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Computes attributions using Deeplift's rescale rule for
                        particular neuron with respect to each input feature.
                        Attributions will always be the same size as the provided
                        inputs, with each value providing the attribution of the
                        corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> # creates an instance of LayerDeepLift to interpret target
            >>> # class 1 with respect to conv4 layer.
            >>> dl = NeuronDeepLiftShap(net, net.conv4)
            >>> input = torch.randn(1, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for conv4 layer and neuron
            >>> # index (4,1,2).
            >>> attribution = dl.attribute(input, (4,1,2))
        """

        dl = DeepLiftShap(cast(Module, self.forward_func), self.multiplies_by_inputs)
        dl.gradient_func = construct_neuron_grad_fn(
            self.layer,
            neuron_selector,
            attribute_to_neuron_input=attribute_to_neuron_input,
        )

        # NOTE: using __wrapped__ to not log
        return dl.attribute.__wrapped__(  # type: ignore
            dl,  # self
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            custom_attribution_func=custom_attribution_func,
        )

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs
