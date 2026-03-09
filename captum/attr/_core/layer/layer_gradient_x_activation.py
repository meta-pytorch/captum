#!/usr/bin/env python3

# pyre-strict
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

from captum._utils.common import (
    _format_additional_forward_args,
    _format_output,
    _format_tensor_into_tuples,
)
from captum._utils.gradient import compute_layer_gradients_and_eval
from captum._utils.typing import ModuleOrModuleList, TargetType
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module


class LayerGradientXActivation(LayerAttribution, GradientAttribution):
    r"""
    Computes element-wise product of gradient and activation for selected
    layer on given inputs.
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
        memory_efficient: bool = False,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                        modification of it
            layer (torch.nn.Module or list of torch.nn.Module): Layer or layers
                          for which attributions are computed.
                          Output size of attribute matches this layer's input or
                          output dimensions, depending on whether we attribute to
                          the inputs or outputs of the layer, corresponding to
                          attribution of each neuron in the input or output of
                          this layer. If multiple layers are provided, attributions
                          are returned as a list, each element corresponding to the
                          attributions of the corresponding layer.
            device_ids (list[int]): Device ID list, necessary only if forward_func
                        applies a DataParallel model. This allows reconstruction of
                        intermediate outputs from batched results across devices.
                        If forward_func is given as the DataParallel model itself,
                        then it is not necessary to provide this argument.
            multiply_by_inputs (bool, optional): Indicates whether to factor
                        model inputs' multiplier in the final attribution scores.
                        In the literature this is also known as local vs global
                        attribution. If inputs' multiplier isn't factored in,
                        then this type of attribution method is also called local
                        attribution. If it is, then that type of attribution
                        method is called global.
                        More detailed can be found here:
                        https://arxiv.org/abs/1711.06104

                        In case of layer gradient x activation, if `multiply_by_inputs`
                        is set to True, final sensitivity scores are being multiplied by
                        layer activations for inputs.

            memory_efficient (bool, optional): If True, offloads intermediate
                        activations and gradients to CPU during computation.
                        Default: False

        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids)
        GradientAttribution.__init__(self, forward_func)
        self._multiply_by_inputs = multiply_by_inputs
        self._memory_efficient = memory_efficient

    @property
    def multiplies_by_inputs(self) -> bool:
        return self._multiply_by_inputs

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]:
        r"""
        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which attributions
                        are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
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
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
            attribute_to_layer_input (bool, optional): Indicates whether to
                        compute the attribution with respect to the layer input
                        or output. If `attribute_to_layer_input` is set to True
                        then the attributions will be computed with respect to
                        layer input, otherwise it will be computed with respect
                        to layer output.
                        Default: False
            grad_kwargs (Dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None
        Returns:
            *Tensor* or *tuple[Tensor, ...]* or list of **attributions**:
            - **attributions** (*Tensor*, *tuple[Tensor, ...]*, or *list*):
                        Product of gradient and activation for each
                        neuron in given layer output.
                        Attributions will always be the same size as the
                        output of the given layer.
                        Attributions are returned in a tuple if
                        the layer inputs / outputs contain multiple tensors,
                        otherwise a single tensor is returned.
                        If multiple layers are provided, attributions
                        are returned as a list, each element corresponding to the
                        activations of the corresponding layer.


        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> layer_ga = LayerGradientXActivation(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer activation x gradient for class 3.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = layer_ga.attribute(input, 3)
        """
        inputs = _format_tensor_into_tuples(inputs)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )
        # Returns gradient of output with respect to
        # hidden layer and hidden layer evaluated at each input.
        layer_gradients, layer_evals = compute_layer_gradients_and_eval(
            self.forward_func,
            self.layer,
            inputs,
            target,
            additional_forward_args,
            device_ids=self.device_ids,
            attribute_to_layer_input=attribute_to_layer_input,
            grad_kwargs=grad_kwargs,
            offload_to_cpu=self._memory_efficient,
        )
        if isinstance(self.layer, Module):
            return _format_output(
                len(layer_evals) > 1,
                self.multiply_gradient_acts(
                    cast(Tuple[Tensor, ...], layer_gradients),
                    cast(Tuple[Tensor, ...], layer_evals),
                ),
            )
        else:
            results = []
            for i in range(len(self.layer)):
                grads_i = layer_gradients[i]
                evals_i = layer_evals[i]
                results.append(
                    _format_output(
                        len(evals_i) > 1,
                        self.multiply_gradient_acts(grads_i, evals_i),
                    )
                )
            return results

    def multiply_gradient_acts(
        self, gradients: Tuple[Tensor, ...], evals: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        return tuple(
            (
                single_gradient * single_eval
                if self.multiplies_by_inputs
                else single_gradient
            )
            for single_gradient, single_eval in zip(gradients, evals)
        )
