#!/usr/bin/env python3

# pyre-strict
import warnings
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

import torch
from captum._utils.common import (
    _expand_additional_forward_args,
    _expand_target,
    _format_additional_forward_args,
    _format_output,
    _is_tuple,
    _verify_select_neuron,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import (
    BaselineType,
    SliceIntType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
from captum.attr._utils.approximation_methods import approximation_parameters
from captum.attr._utils.attribution import GradientAttribution, NeuronAttribution
from captum.attr._utils.batching import _batch_attribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _reshape_and_sum,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class NeuronConductance(NeuronAttribution, GradientAttribution):
    r"""
    Computes conductance with respect to particular hidden neuron. The
    returned output is in the shape of the input, showing the attribution
    / conductance of each input feature to the selected hidden layer neuron.
    The details of the approach can be found here:
    https://arxiv.org/abs/1805.12233
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
            layer (torch.nn.Module): Layer for which neuron attributions are computed.
                        Attributions for a particular neuron in the input or output
                        of this layer are computed using the argument neuron_selector
                        in the attribute method.
                        Currently, only layers with a single tensor input or output
                        are supported.
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

                        In case of Neuron Conductance,
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
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "riemann_trapezoid",
        internal_batch_size: Union[None, int] = None,
        attribute_to_neuron_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which neuron
                        conductance is computed. If forward_func takes a single
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

                        - a tuple of integers. Length of this
                          tuple must be one less than the number of dimensions
                          in the input / output of the given layer (since
                          dimension 0 corresponds to number of examples).
                          This can be used as long as the layer input / output
                          is a single tensor.

                        - a callable, which should
                          take the target layer as input (single tensor or tuple
                          if multiple tensors are in layer) and return a selected
                          neuron - output shape should be 1D with length equal to
                          batch_size (one scalar per input example)

                          NOTE: Callables applicable for neuron conductance are
                          less general than those of other methods and should
                          NOT aggregate values of the layer, only return a specific
                          output. This option should only be used in cases where the
                          layer input / output is a tuple of tensors, where the other
                          options would not suffice. This limitation is necessary since
                          neuron conductance, unlike other neuron methods, also utilizes
                          the gradient of output with respect to the intermedite neuron,
                          which cannot be computed for aggregations of multiple
                          intemediate neurons.
            baselines (scalar, Tensor, tuple of scalar, or Tensor, optional):
                        Baselines define the starting point from which integral
                        is computed and can be provided as:

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
            target (int, tuple, Tensor, or list, optional): Output indices for
                        which gradients are computed (for classification cases,
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
            grad_kwargs (dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None

        Returns:
            *Tensor* or *tuple[Tensor, ...]* of **attributions**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                        Conductance for
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
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> neuron_cond = NeuronConductance(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # To compute neuron attribution, we need to provide the neuron
            >>> # index for which attribution is desired. Since the layer output
            >>> # is Nx12x32x32, we need a tuple in the form (0..11,0..31,0..31)
            >>> # which indexes a particular neuron in the layer output.
            >>> # Computes neuron conductance for neuron with
            >>> # index (4,1,2).
            >>> attribution = neuron_cond.attribute(input, (4,1,2))
        """
        if callable(neuron_selector):
            warnings.warn(
                "The neuron_selector provided is a callable. Please ensure that this"
                " function only selects neurons from the given layer; aggregating"
                " or performing other operations on the tensor may lead to inaccurate"
                " results.",
                stacklevel=1,
            )
        is_inputs_tuple = _is_tuple(inputs)

        formatted_inputs, formatted_baselines = _format_input_baseline(
            inputs, baselines
        )
        _validate_input(formatted_inputs, formatted_baselines, n_steps, method)

        num_examples = formatted_inputs[0].shape[0]

        if internal_batch_size is not None:
            num_examples = formatted_inputs[0].shape[0]
            attrs = _batch_attribution(
                self,
                num_examples,
                internal_batch_size,
                n_steps,
                inputs=formatted_inputs,
                baselines=formatted_baselines,
                neuron_selector=neuron_selector,
                target=target,
                additional_forward_args=additional_forward_args,
                method=method,
                attribute_to_neuron_input=attribute_to_neuron_input,
            )
        else:
            attrs = self._attribute(
                inputs=formatted_inputs,
                neuron_selector=neuron_selector,
                baselines=formatted_baselines,
                target=target,
                additional_forward_args=additional_forward_args,
                n_steps=n_steps,
                method=method,
                attribute_to_neuron_input=attribute_to_neuron_input,
                grad_kwargs=grad_kwargs,
            )
        return cast(
            TensorOrTupleOfTensorsGeneric, _format_output(is_inputs_tuple, attrs)
        )

    def _attribute(
        self,
        inputs: Tuple[Tensor, ...],
        neuron_selector: Union[
            int,
            Tuple[Union[int, SliceIntType], ...],
            Callable[[Union[Tensor, Tuple[Tensor, ...]]], Tensor],
        ],
        baselines: Tuple[Union[Tensor, int, float], ...],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "riemann_trapezoid",
        attribute_to_neuron_input: bool = False,
        step_sizes_and_alphas: Union[None, Tuple[List[float], List[float]]] = None,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tensor, ...]:

        num_examples = inputs[0].shape[0]
        total_batch = num_examples * n_steps

        if step_sizes_and_alphas is None:
            # retrieve step size and scaling factor for specified approximation method
            step_sizes_func, alphas_func = approximation_parameters(method)
            step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)
        else:
            step_sizes, alphas = step_sizes_and_alphas

        # Compute scaled inputs from baseline to final input.
        scaled_features_tpl = tuple(
            torch.cat(
                [baseline + alpha * (input - baseline) for alpha in alphas], dim=0
            ).requires_grad_()
            for input, baseline in zip(inputs, baselines)
        )

        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # apply number of steps to additional forward args
        # currently, number of steps is applied only to additional forward arguments
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (#examples * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )
        expanded_target = _expand_target(target, n_steps)

        # Conductance Gradients - Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_eval, input_grads = compute_layer_gradients_and_eval(
            forward_fn=self.forward_func,
            layer=self.layer,
            inputs=scaled_features_tpl,
            target_ind=expanded_target,
            additional_forward_args=input_additional_args,
            gradient_neuron_selector=neuron_selector,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_neuron_input,
            grad_kwargs=grad_kwargs,
        )

        mid_grads = _verify_select_neuron(layer_gradients, neuron_selector)
        scaled_input_gradients = tuple(
            input_grad
            * mid_grads.reshape((total_batch,) + (1,) * (len(input_grad.shape) - 1))
            for input_grad in input_grads
        )

        # Mutliplies by appropriate step size.
        scaled_grads = tuple(
            scaled_input_gradient.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(scaled_input_gradient.device)
            for scaled_input_gradient in scaled_input_gradients
        )

        # Aggregates across all steps for each tensor in the input tuple
        total_grads = tuple(
            _reshape_and_sum(
                scaled_grad, n_steps, num_examples, tuple(input_grad.shape[1:])
            )
            for (scaled_grad, input_grad) in zip(scaled_grads, input_grads)
        )

        if self.multiplies_by_inputs:
            # computes attribution for each tensor in input tuple
            # attributions has the same dimensionality as inputs
            attributions = tuple(
                total_grad * (input - baseline)
                for total_grad, input, baseline in zip(total_grads, inputs, baselines)
            )
        else:
            attributions = total_grads

        return attributions

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs
