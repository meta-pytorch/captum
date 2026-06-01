#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from typing import Callable, List, Optional, Tuple, Union

from captum._utils.gradient import construct_neuron_grad_fn
from captum._utils.typing import SliceIntType, TensorOrTupleOfTensorsGeneric
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.attribution import GradientAttribution, NeuronAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class NeuronIntegratedGradients(NeuronAttribution, GradientAttribution):
    r"""
    Approximates the integral of gradients for a particular neuron
    along the path from a baseline input to the given input.
    If no baseline is provided, the default baseline is the zero tensor.
    More details regarding the integrated gradient method can be found in the
    original paper here:
    https://arxiv.org/abs/1703.01365

    Note that this method is equivalent to applying integrated gradients
    where the output is the output of the identified neuron.
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: Module,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                        modification of it
            layer (torch.nn.Module): Layer for which attributions are computed.
                        Output size of attribute matches this layer's input or
                        output dimensions, depending on whether we attribute to
                        the inputs or outputs of the layer, corresponding to
                        attribution of each neuron in the input or output of
                        this layer.
                        Currently, it is assumed that the inputs or the outputs
                        of the layer, depending on which one is used for
                        attribution, can only be a single tensor.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                        applies a DataParallel model. This allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in
                        then that type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of Neuron Integrated Gradients,
                        if `multiply_by_inputs` is set to True, final
                        sensitivity scores are being multiplied
                        by (inputs - baselines).

        """
        NeuronAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)
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
        baselines: Union[None, Tensor, Tuple[Tensor, ...]] = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        attribute_to_neuron_input: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which neuron integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
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
                        Baselines define the starting point from which integral
                        is computed.
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
                        argument of a Tensor or arbitrary (non-tuple) type or a
                        tuple containing multiple additional arguments including
                        tensors or any arbitrary python types. These arguments
                        are provided to forward_func in order following the
                        arguments in inputs.
                        For a tensor, the first dimension of the tensor must
                        correspond to the number of examples. It will be
                        repeated for each of `n_steps` along the integrated
                        path. For all other types, the given argument is used
                        for all forward evaluations.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (int, optional): The number of steps used by the approximation
                        method. Default: 50.
            method (str, optional): Method for approximating the integral,
                        one of `riemann_right`, `riemann_left`, `riemann_middle`,
                        `riemann_trapezoid` or `gausslegendre`.
                        Default: `gausslegendre` if no method is provided.
            internal_batch_size (int, optional): Divides total #steps * #examples
                        data points into chunks of size at most internal_batch_size,
                        which are computed (forward / backward passes)
                        sequentially. internal_batch_size must be at least equal to
                        #examples.
                        For DataParallel models, each batch is split among the
                        available devices, so evaluations on each available
                        device contain internal_batch_size / num_devices examples.
                        If internal_batch_size is None, then all evaluations are
                        processed in one batch.
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

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Integrated gradients for particular neuron with
                        respect to each input feature.
                        Attributions will always be the same size as the provided
                        inputs, with each value providing the attribution of the
                        corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> neuron_ig = NeuronIntegratedGradients(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
            >>> # which indexes a particular neuron in the layer output.
            >>> # For this example, we choose the index (4,1,2).
            >>> # Computes neuron integrated gradients for neuron with
            >>> # index (4,1,2).
            >>> attribution = neuron_ig.attribute(input, (4,1,2))
        """
        ig = IntegratedGradients(self.forward_func, self.multiplies_by_inputs)
        ig.gradient_func = construct_neuron_grad_fn(
            self.layer, neuron_selector, self.device_ids, attribute_to_neuron_input
        )
        # NOTE: using __wrapped__ to not log
        # Return only attributions and not delta
        return ig.attribute.__wrapped__(  # type: ignore
            ig,  # self
            inputs,
            baselines,
            additional_forward_args=additional_forward_args,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
        )

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs
