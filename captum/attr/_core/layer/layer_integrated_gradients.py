#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import functools
import warnings
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Literal,
    Optional,
    overload,
    Tuple,
    Union,
)

import torch
from captum._utils.common import (
    _extract_device,
    _format_additional_forward_args,
    _format_outputs,
)
from captum._utils.gradient import _forward_layer_eval, _run_forward
from captum._utils.typing import BaselineType, ModuleOrModuleList, TargetType
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._utils.attribution import GradientAttribution, LayerAttribution
from captum.attr._utils.common import (
    _format_input_baseline,
    _tensorize_baseline,
    _validate_input,
)
from captum.log import log_usage
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel.scatter_gather import scatter


class LayerIntegratedGradients(LayerAttribution, GradientAttribution):
    r"""
    Layer Integrated Gradients is a variant of Integrated Gradients that assigns
    an importance score to layer inputs or outputs, depending on whether we
    attribute to the former or to the latter one.

    Integrated Gradients is an axiomatic model interpretability algorithm that
    attributes / assigns an importance score to each input feature by approximating
    the integral of gradients of the model's output with respect to the inputs
    along the path (straight line) from given baselines / references to inputs.

    Baselines can be provided as input arguments to attribute method.
    To approximate the integral we can choose to use either a variant of
    Riemann sum or Gauss-Legendre quadrature rule.

    More details regarding the integrated gradients method can be found in the
    original paper:
    https://arxiv.org/abs/1703.01365
    """

    def __init__(
        self,
        forward_func: Callable[..., Tensor],
        layer: ModuleOrModuleList,
        device_ids: Union[None, List[int]] = None,
        multiply_by_inputs: bool = True,
    ) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or any
                        modification of it
            layer (ModuleOrModuleList): Layer or list of layers for which attributions
                        are computed. For each layer the output size of the attribute
                        matches this layer's input or output dimensions, depending on
                        whether we attribute to the inputs or outputs of the
                        layer, corresponding to the attribution of each neuron
                        in the input or output of this layer.

                        Please note that layers to attribute on cannot be
                        dependent on each other. That is, a subset of layers in
                        `layer` cannot produce the inputs for another layer.

                        For example, if your model is of a simple linked-list
                        based graph structure (think nn.Sequence), e.g. x -> l1
                        -> l2 -> l3 -> output. If you pass in any one of those
                        layers, you cannot pass in another due to the
                        dependence, e.g.  if you pass in l2 you cannot pass in
                        l1 or l3.

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

                        In case of layer integrated gradients, if `multiply_by_inputs`
                        is set to True, final sensitivity scores are being multiplied by
                        layer activations for inputs - layer activations for baselines.

        """
        LayerAttribution.__init__(self, forward_func, layer, device_ids=device_ids)
        GradientAttribution.__init__(self, forward_func)
        self.ig = IntegratedGradients(forward_func, multiply_by_inputs)

        if isinstance(layer, list) and len(layer) > 1:
            warnings.warn(
                "Multiple layers provided. Please ensure that each layer is"
                "**not** solely dependent on the outputs of"
                "another layer. Please refer to the documentation for more"
                "detail.",
                stacklevel=2,
            )

    def _make_gradient_func(
        self,
        num_outputs_cumsum: Tensor,
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Callable[..., Tuple[Tensor, ...]]:

        def _gradient_func(
            forward_fn: Callable[..., Tensor],
            inputs: Union[Tensor, Tuple[Tensor, ...]],
            target_ind: TargetType = None,
            additional_forward_args: Optional[object] = None,
        ) -> Tuple[Tensor, ...]:
            if self.device_ids is None or len(self.device_ids) == 0:
                scattered_inputs = (inputs,)
            else:
                # scatter method does not have a precise enough return type in its
                # stub, so suppress the type warning.
                scattered_inputs = scatter(  # type: ignore
                    # pyre-fixme[6]: For 1st argument expected `Tensor` but got
                    #  `Union[Tensor, typing.Tuple[Tensor, ...]]`.
                    inputs,
                    target_gpus=self.device_ids,
                )

            scattered_inputs_dict: Dict[
                torch.device, Union[Tensor, Tuple[Tensor, ...]]
            ] = {
                scattered_input[0].device: scattered_input
                for scattered_input in scattered_inputs
            }

            with torch.autograd.set_grad_enabled(True):

                def layer_forward_hook(
                    module: Module,
                    hook_inputs: Union[Tensor, Tuple[Tensor, ...]],
                    hook_outputs: Union[None, Tensor, Tuple[Tensor, ...]] = None,
                    layer_idx: int = 0,
                ) -> Union[Tensor, Tuple[Tensor, ...]]:
                    device = _extract_device(module, hook_inputs, hook_outputs)
                    is_layer_tuple = (
                        isinstance(hook_outputs, tuple)
                        # hook_outputs is None if attribute_to_layer_input == True
                        if hook_outputs is not None
                        else isinstance(hook_inputs, tuple)
                    )

                    if is_layer_tuple:
                        replacement_outputs = scattered_inputs_dict[device][
                            num_outputs_cumsum[layer_idx] : num_outputs_cumsum[
                                layer_idx + 1
                            ]
                        ]
                        if hook_outputs is not None and isinstance(
                            hook_outputs, (tuple, list)
                        ):
                            replacement_output_iter = iter(replacement_outputs)
                            replaced_outputs = [
                                (
                                    next(replacement_output_iter)
                                    if isinstance(hook_output, Tensor)
                                    else hook_output
                                )
                                for hook_output in hook_outputs
                            ]
                            return cast(
                                Union[Tensor, Tuple[Tensor, ...]],
                                (
                                    tuple(replaced_outputs)
                                    if isinstance(hook_outputs, tuple)
                                    else replaced_outputs
                                ),
                            )
                        return cast(
                            Union[Tensor, Tuple[Tensor, ...]],
                            replacement_outputs,
                        )

                    return scattered_inputs_dict[device][num_outputs_cumsum[layer_idx]]

                hooks = []
                try:

                    layers = self.layer
                    if not isinstance(layers, list):
                        layers = [self.layer]

                    for layer_idx, layer in enumerate(layers):
                        hook = None
                        # TODO:
                        # Allow multiple attribute_to_layer_input flags for
                        # each layer, i.e. attribute_to_layer_input[layer_idx]
                        if attribute_to_layer_input:
                            hook = layer.register_forward_pre_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )
                        else:
                            hook = layer.register_forward_hook(
                                functools.partial(
                                    layer_forward_hook, layer_idx=layer_idx
                                )
                            )

                        hooks.append(hook)

                    # the inputs is an empty tuple
                    # coz it is prepended into additional_forward_args
                    output = _run_forward(
                        self.forward_func, (), target_ind, additional_forward_args
                    )
                finally:
                    for hook in hooks:
                        if hook is not None:
                            hook.remove()

                # _run_forward may return future of Tensor,
                # but we don't support it here now
                # And it will fail before here.
                output = cast(Tensor, output)
                assert output[0].numel() == 1, (
                    "Target not provided when necessary, cannot"
                    " take gradient with respect to multiple outputs."
                )
                # torch.unbind(forward_out) is a list of scalar tensor tuples and
                # contains batch_size * #steps elements
                grads = torch.autograd.grad(
                    torch.unbind(output), inputs, **grad_kwargs or {}
                )
            return grads

        return _gradient_func

    @overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Optional[object],
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[False],
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]]: ...

    @overload
    @log_usage(part_of_slo=True)
    def attribute(  # type: ignore
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType,
        target: TargetType,
        additional_forward_args: Optional[object],
        n_steps: int,
        method: str,
        internal_batch_size: Union[None, int],
        return_convergence_delta: Literal[True],
        attribute_to_layer_input: bool,
        grad_kwargs: Optional[Dict[str, Any]],
    ) -> Tuple[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tensor,
    ]: ...

    @overload
    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]: ...

    @log_usage(part_of_slo=True)
    def attribute(
        self,
        inputs: Union[Tensor, Tuple[Tensor, ...]],
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Optional[object] = None,
        n_steps: int = 50,
        method: str = "gausslegendre",
        internal_batch_size: Union[None, int] = None,
        return_convergence_delta: bool = False,
        attribute_to_layer_input: bool = False,
        grad_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[
        Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
        Tuple[
            Union[Tensor, Tuple[Tensor, ...], List[Union[Tensor, Tuple[Tensor, ...]]]],
            Tensor,
        ],
    ]:
        r"""
        This method attributes the output of the model with given target index
        (in case it is provided, otherwise it assumes that output is a
        scalar) to layer inputs or outputs of the model, depending on whether
        `attribute_to_layer_input` is set to True or False, using the approach
        described above.

        In addition to that it also returns, if `return_convergence_delta` is
        set to True, integral approximation delta based on the completeness
        property of integrated gradients.

        Args:

            inputs (Tensor or tuple[Tensor, ...]): Input for which layer integrated
                        gradients are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples, and if multiple input tensors
                        are provided, the examples must be aligned appropriately.
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
            return_convergence_delta (bool, optional): Indicates whether to return
                        convergence delta or not. If `return_convergence_delta`
                        is set to True convergence delta will be returned in
                        a tuple following attributions.
                        Default: False
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
            grad_kwargs (Dict[str, Any], optional): Additional keyword
                        arguments for torch.autograd.grad.
                        Default: None

        Returns:
            **attributions** or 2-element tuple of **attributions**, **delta**:
            - **attributions** (*Tensor* or *tuple[Tensor, ...]*):
                Integrated gradients with respect to `layer`'s inputs
                or outputs. Attributions will always be the same size and
                dimensionality as the input or output of the given layer,
                depending on whether we attribute to the inputs or outputs
                of the layer which is decided by the input flag
                `attribute_to_layer_input`.

                For a single layer, attributions are returned in a tuple if
                the layer inputs / outputs contain multiple tensors,
                otherwise a single tensor is returned.

                For multiple layers, attributions will always be
                returned as a list. Each element in this list will be
                equivalent to that of a single layer output, i.e. in the
                case that one layer, in the given layers, inputs / outputs
                multiple tensors: the corresponding output element will be
                a tuple of tensors. The ordering of the outputs will be
                the same order as the layers given in the constructor.

            - **delta** (*Tensor*, returned if return_convergence_delta=True):
                The difference between the total approximated and true
                integrated gradients. This is computed using the property
                that the total sum of forward_func(inputs) -
                forward_func(baselines) must equal the total sum of the
                integrated gradient.
                Delta is calculated per example, meaning that the number of
                elements in returned delta tensor is equal to the number of
                examples in inputs.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> # It contains an attribute conv1, which is an instance of nn.conv2d,
            >>> # and the output of this layer has dimensions Nx12x32x32.
            >>> net = ImageClassifier()
            >>> lig = LayerIntegratedGradients(net, net.conv1)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes layer integrated gradients for class 3.
            >>> # attribution size matches layer output, Nx12x32x32
            >>> attribution = lig.attribute(input, target=3)
        """
        inps, baselines = _format_input_baseline(inputs, baselines)
        _validate_input(inps, baselines, n_steps, method)

        baselines = _tensorize_baseline(inps, baselines)
        additional_forward_args = _format_additional_forward_args(
            additional_forward_args
        )

        def flatten_tuple(tup: List[Tuple[Tensor, ...]]) -> Tuple[Tensor, ...]:
            return tuple(
                cast(
                    List[Tensor],
                    sum(
                        (
                            (
                                list(x)
                                if isinstance(x, (tuple, list))
                                else cast(List[Tensor], [x])
                            )
                            for x in tup
                        ),
                        [],
                    ),
                )
            )

        if self.device_ids is None:
            self.device_ids = getattr(self.forward_func, "device_ids", None)

        inputs_layer = _forward_layer_eval(
            self.forward_func,
            inps,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        input_layer_list: List[Tuple[Tensor, ...]]
        # if we have one output
        if not isinstance(self.layer, list):
            input_layer_list = [cast(Tuple[Tensor, ...], inputs_layer)]
        else:
            input_layer_list = inputs_layer

        num_outputs = [1 if isinstance(x, Tensor) else len(x) for x in input_layer_list]
        num_outputs_cumsum = torch.cumsum(
            torch.IntTensor([0] + num_outputs), dim=0  # type: ignore
        )
        inputs_layer = flatten_tuple(input_layer_list)

        baselines_layer = _forward_layer_eval(
            self.forward_func,
            baselines,
            self.layer,
            device_ids=self.device_ids,
            additional_forward_args=additional_forward_args,
            attribute_to_layer_input=attribute_to_layer_input,
        )
        baselines_layer = flatten_tuple(baselines_layer)

        # inputs -> these inputs are scaled

        self.ig.gradient_func = self._make_gradient_func(
            num_outputs_cumsum, attribute_to_layer_input, grad_kwargs
        )
        all_inputs = (
            (inps + additional_forward_args)
            if additional_forward_args is not None
            else inps
        )

        attributions = self.ig.attribute.__wrapped__(  # type: ignore
            self.ig,  # self
            inputs_layer,
            baselines=baselines_layer,
            target=target,
            additional_forward_args=all_inputs,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
        )

        # handle multiple outputs
        output: List[Tuple[Tensor, ...]] = [
            tuple(
                attributions[
                    int(num_outputs_cumsum[i]) : int(num_outputs_cumsum[i + 1])
                ]
            )
            for i in range(len(num_outputs))
        ]

        if return_convergence_delta:
            start_point, end_point = baselines, inps
            # computes approximation error based on the completeness axiom
            delta = self.compute_convergence_delta(
                attributions,
                start_point,
                end_point,
                additional_forward_args=additional_forward_args,
                target=target,
            )
            return _format_outputs(isinstance(self.layer, list), output), delta
        return _format_outputs(isinstance(self.layer, list), output)

    def has_convergence_delta(self) -> bool:
        return True

    @property
    def multiplies_by_inputs(self) -> bool:
        return self.ig.multiplies_by_inputs
