#!/usr/bin/env python3
import warnings
import torch
import torch.nn.functional as F

from .._utils.common import (
    format_input,
    format_baseline,
    _format_attributions,
    validate_input,
)
from .._utils.attribution import GradientBasedAttribution
from .._utils.gradient import apply_gradient_requirements, undo_gradient_requirements


# TODO: GradientBasedAttribution needs to be replaced with Attribution or
# OutputAttribution class
class DeepLift(GradientBasedAttribution):
    def __init__(self, model):
        r"""
        Args:

            model (nn.Module):  The reference to PyTorch model instance.
        """
        super().__init__(model)
        self.model = model
        self.forward_handles = []
        self.backward_handles = []

    def attribute(
        self, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        r""""
        Implements DeepLIFT algorithm based on the following paper:
        Learning Important Features Through Propagating Activation Differences,
        Avanti Shrikumar, et. al.
        https://arxiv.org/abs/1704.02685

        and the gradient formulation proposed in:
        Towards better understanding of gradient-based attribution methods for
        deep neural networks,  Marco Ancona, et.al.
        https://openreview.net/pdf?id=Sy21R9JAW

        This implementation supports only Rescale rule. RevealCancel rule will
        be supported in later releases.
        An addition to that, in order to keep the implementation cleaner, DeepLIFT
        for internal neurons and layers will be implemented in a separate file.
        Although DeepLIFT's(Rescale Rule) attribution quality is comparable with
        Integrated Gradients, it runs significantly faster than Integrated
        Gradients and is preferred for large datasets.

        Currently we only support a limited number of non-linear activations
        but the plan is to expand the list in the future.

        Note: As we know, currently we cannot access the building blocks,
        of PyTorch's built-in LSTM, RNNs and GRUs such as Tanh and Sigmoid.
        Nonetheless, it is possible to build custom LSTMs, RNNS and GRUs
        with performance similar to built-in ones with TorchScript.
        More details on how to build custom RNNs can be found here:
        https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/

        Args:

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        mutliple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (tensor or tuple of tensors, optional): Baselines define
                        reference samples which are compared with the inputs.
                        In order to assign attribution scores DeepLift computes
                        the differences between the inputs and references and
                        corresponding outputs.
                        If inputs is a single tensor, baselines must also be a
                        single tensor with exactly the same dimensions as inputs.
                        If inputs is a tuple of tensors, baselines must also be
                        a tuple of tensors, with matching dimensions to inputs.
                        Default: zero tensor for each input tensor
            target (int, optional):  Output index for which gradient is computed
                        (for classification cases, this is the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary. (Note: Tuples for multi
                        -dimensional output indices will be supported soon.)
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
        Returns:

            attributions (tensor or tuple of tensors): Attribution score
                        computed based on DeepLift rescale rule with respect
                        to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            delta (float): This is computed using the property that the total
                        sum of forward_func(inputs) - forward_func(baselines)
                        must equal the total sum of the attributions computed
                        based on Deeplift's rescale rule.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLift(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes deeplift attribution scores for class 3.
            >>> attribution, delta = dl.attribute(input, target=3)
        """

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        baselines = format_baseline(baselines, inputs)
        gradient_mask = apply_gradient_requirements(inputs)

        # set hooks for baselines
        self._traverse_modules(self.model, self._register_hooks, input_type="ref")
        # make forward pass and remove baseline hooks
        self.forward_func(*baselines)

        validate_input(inputs, baselines)

        # set hook for inputs
        self._traverse_modules(self.model, self._register_hooks)
        gradients = self.gradient_func(
            self.forward_func,
            inputs,
            target_ind=target,
            additional_forward_args=additional_forward_args,
        )
        attributions = tuple(
            (input - baseline) * gradient
            for input, baseline, gradient in zip(inputs, baselines, gradients)
        )

        # remove hooks from all activations
        self._remove_hooks()

        start_point, end_point = baselines, inputs

        # computes convergence error
        delta = self._compute_convergence_delta(
            attributions,
            start_point,
            end_point,
            additional_forward_args=additional_forward_args,
            target=target,
        )
        undo_gradient_requirements(inputs, gradient_mask)
        return _format_attributions(is_inputs_tuple, attributions), delta

    def _is_non_linear(self, module):
        module_name = module._get_name()
        return module_name in SUPPORTED_NON_LINEAR.keys()

    # we need forward hook to access and detach the inputs and outputs of a neuron
    def _forward_hook(self, module, inputs, outputs):
        input_attr_name = "input"
        output_attr_name = "output"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)

    def _forward_hook_ref(self, module, inputs, outputs):
        input_attr_name = "input_ref"
        output_attr_name = "output_ref"
        self._detach_tensors(input_attr_name, output_attr_name, module, inputs, outputs)
        # since it is a reference forward hook remove it from the module after
        # detaching the variables
        module.ref_handle.remove()
        # remove attribute `ref_handle`
        del module.ref_handle

    def _detach_tensors(
        self, input_attr_name, output_attr_name, module, inputs, outputs
    ):
        setattr(module, input_attr_name, tuple(input.detach() for input in inputs))
        setattr(module, output_attr_name, tuple(output.detach() for output in outputs))

    def _backward_hook(self, module, grad_input, grad_output, eps=1e-10):
        r"""
         `grad_input` is the gradient of the neuron with respect to its input
         `grad_output` is the gradient of the neuron with respect to its output
          we can override `grad_input` according to chain rule with.
         `grad_output` * delta_out / delta_in.

         """
        delta_in = tuple(
            inp - inp_ref for inp, inp_ref in zip(module.input, module.input_ref)
        )
        delta_out = tuple(
            out - out_ref for out, out_ref in zip(module.output, module.output_ref)
        )

        #modified_grads = [g_input for g_input in grad_input]

        # remove all the properies that we set for the inputs and output
        del module.input_ref
        del module.output_ref
        del module.input
        del module.output

        return tuple(
            SUPPORTED_NON_LINEAR[module._get_name()](
                module, delta_in, delta_out, list(grad_input), grad_output, eps=eps
            )
        )

    def _register_hooks(self, module, input_type="non_ref"):
        # TODO find a better way of checking if a module is a container or not
        module_fullname = str(type(module))
        has_already_hooks = len(module._backward_hooks) > 0
        if (
            "nn.modules.container" in module_fullname
            or has_already_hooks
            or not self._is_non_linear(module)
        ):
            return
        # adds forward hook to leaf nodes that are non-linear
        if input_type != "ref":
            forward_handle = module.register_forward_hook(self._forward_hook)
            backward_handle = module.register_backward_hook(self._backward_hook)
            self.forward_handles.append(forward_handle)
            self.backward_handles.append(backward_handle)
        else:
            handle = module.register_forward_hook(self._forward_hook_ref)
            ref_handle = "ref_handle"
            setattr(module, ref_handle, handle)

    def _traverse_modules(self, module, hook_fn, input_type="non_ref"):
        warnings.warn(
            """Setting forward, backward hooks and attributes on non-linear
               activations. The hooks and attributes will be removed
            after the attribution is finished"""
        )
        children = module.children()
        for child in children:
            self._traverse_modules(child, hook_fn, input_type)
            hook_fn(child, input_type)

    def _remove_hooks(self):
        for forward_handle in self.forward_handles:
            forward_handle.remove()
        for backward_handle in self.backward_handles:
            backward_handle.remove()

    def _has_convergence_delta(self):
        return True


class DeepLiftShap(DeepLift):
    def __init__(self, model):
        super().__init__(model)

    def attribute(
        self, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        r"""
        Extends DeepLift alogrithm and approximates SHAP values using Deeplift.
        More details about the algorithm can be found here:

        http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting
        -model-predictions.pdf

        Note that the model makes two major assumptions.
        It assumes that:
            1. Input features are independant
            2. The model is linear when explaining predictions for each input
        Although, it assumes a linear model for each input, the overall
        model can be complex and non-linear.

        Args:

            inputs (tensor or tuple of tensors):  Input for which
                        attributions are computed. If forward_func takes a single
                        tensor as input, a single input tensor should be provided.
                        If forward_func takes multiple tensors as input, a tuple
                        of the input tensors should be provided. It is assumed
                        that for all given input tensors, dimension 0 corresponds
                        to the number of examples (aka batch size), and if
                        mutliple input tensors are provided, the examples must
                        be aligned appropriately.
            baselines (tensor or tuple of tensors, optional): Baselines define
                        reference samples which are compared with the inputs.
                        In order to assign attribution scores DeepLift computes
                        the differences between the inputs and references and
                        corresponding outputs.
                        If inputs is a single tensor, baselines must also be a
                        single tensor. If inputs is a tuple of tensors, baselines
                        must also be a tuple of tensors. The first dimension in
                        baseline tensors defines the distribution from which we
                        randomly draw samples. All other dimensions starting after
                        the first dimension should match with the inputs'
                        dimensions after the first dimension. It is recommended that
                        the number of samples in the baselines' tensors is larger
                        than one.

                        Default: zero tensor for each input tensor
            target (int, optional):  Output index for which gradient is computed
                        (for classification cases, this is the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary. (Note: Tuples for multi
                        -dimensional output indices will be supported soon.)
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It must be either a single additional
                        argument of a Tensor or arbitrary (non-tuple) type or a tuple
                        containing multiple additional arguments including tensors
                        or any arbitrary python types. These arguments are provided to
                        forward_func in order, following the arguments in inputs.
                        Note that attributions are not computed with respect
                        to these arguments.
                        Default: None
        Returns:

            attributions (tensor or tuple of tensors): Attribution score
                        computed based on DeepLift rescale rule with respect
                        to each input feature. Attributions will always be
                        the same size as the provided inputs, with each value
                        providing the attribution of the corresponding input index.
                        If a single tensor is provided as inputs, a single tensor is
                        returned. If a tuple is provided for inputs, a tuple of
                        corresponding sized tensors is returned.
            delta (float): This is computed using the property that the total
                        sum of forward_func(inputs) - forward_func(baselines)
                        must equal the total sum of attributions computed
                        based on approximated SHAP values using Deeplift's
                        rescale rule.

        Examples::

            >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
            >>> # and returns an Nx10 tensor of class probabilities.
            >>> net = ImageClassifier()
            >>> dl = DeepLiftShap(net)
            >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
            >>> # Computes shap values using deeplift for class 3.
            >>> attribution, delta = dl.attribute(input, target=3)
        """
        def compute_mean(inp_bsz, base_bsz, attribution):
            # Average for multiple references
            attr_shape = (base_bsz, inp_bsz)
            if len(attribution.shape) > 1:
                attr_shape += (-1,)
            return torch.mean(attribution.view(attr_shape), axis=0, keepdim=False)

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs = format_input(inputs)
        baselines = format_baseline(baselines, inputs)

        # batch sizes
        inp_bsz = inputs[0].shape[0]
        base_bsz = baselines[0].shape[0]
        # match the sizes of inputs and baselines in case of multiple references
        # for a single input
        inputs = tuple(
            [
                input.repeat_interleave(base_bsz, dim=0).requires_grad_()
                for input, baseline in zip(inputs, baselines)
            ]
        )
        baselines = tuple(
            [
                baseline.repeat(
                    (inp_bsz,) + tuple([1] * (len(baseline.shape) - 1))
                ).requires_grad_()
                for baseline in baselines
            ]
        )

        attributions, delta = super().attribute(
            inputs,
            baselines,
            target=target,
            additional_forward_args=additional_forward_args,
        )

        attributions = tuple(
            compute_mean(inp_bsz, base_bsz, attribution) for attribution in attributions
        )

        start_point, end_point = baselines, inputs

        # computes convergence error
        delta = self._compute_convergence_delta(
            attributions,
            start_point,
            end_point,
            additional_forward_args=additional_forward_args,
            target=target,
            is_multi_baseline=True,
        )

        return _format_attributions(is_inputs_tuple, attributions), delta


def nonlinear(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    r"""
    grad_input: (dLoss / dprev_layer_out, dLoss / wij, dLoss / bij)
    grad_output: (dLoss / dlayer_out)
    https://github.com/pytorch/pytorch/issues/12331
    """
    # supported non-linear modules take only single tensor as input hence accessing
    # only the first element in `grad_input` and `grad_output`

    grad_input[0] = torch.where(
        delta_in[0] < eps, grad_input[0], grad_output[0] * delta_out[0] / delta_in[0]
    )
    print('grad_input: ', grad_input)
    return grad_input


def softmax(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    grad_input_unnorm = torch.where(
        delta_in[0] < eps, grad_input[0], grad_output[0] * delta_out[0] / delta_in[0]
    )
    # normalizing
    n = grad_input[0].shape[1]
    grad_input[0] = grad_input_unnorm - grad_input_unnorm.sum() * 1 / n
    return grad_input


def maxpool1d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool1d,
        F.max_unpool1d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool2d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool2d,
        F.max_unpool2d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool3d(module, delta_in, delta_out, grad_input, grad_output, eps=1e-10):
    return maxpool(
        module,
        F.max_pool3d,
        F.max_unpool3d,
        delta_in,
        delta_out,
        grad_input,
        grad_output,
        eps=eps,
    )


def maxpool(
    module,
    pool_func,
    unpool_func,
    delta_in,
    delta_out,
    grad_input,
    grad_output,
    eps=1e-10,
):
    # The forward function of maxpool takes only tensors not
    # a tuple hence accessing the first
    # element in the tuple of inputs, grad_input and grad_output
    _, indices = pool_func(
        module.input[0],
        module.kernel_size,
        module.stride,
        module.padding,
        module.dilation,
        module.ceil_mode,
        True,
    )
    unpool_grad_out_delta = unpool_func(
        grad_output[0] * delta_out[0],
        indices,
        module.kernel_size,
        module.stride,
        module.padding,
        list(module.input[0].shape),
    )

    grad_input[0] = torch.where(
        delta_in[0] < eps, grad_input[0], unpool_grad_out_delta / delta_in[0]
    )
    return grad_input


SUPPORTED_NON_LINEAR = {
    "ReLU": nonlinear,
    "Elu": nonlinear,
    "LeakyReLU": nonlinear,
    "Sigmoid": nonlinear,
    "Tanh": nonlinear,
    "Softplus": nonlinear,
    "MaxPool1d": maxpool1d,
    "MaxPool2d": maxpool2d,
    "MaxPool3d": maxpool3d,
    "Softmax": softmax,
}
