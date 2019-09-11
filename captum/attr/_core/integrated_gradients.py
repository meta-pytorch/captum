#!/usr/bin/env python3
import torch

from .._utils.approximation_methods import approximation_parameters
from .._utils.common import (
    validate_input,
    _format_input_baseline,
    _format_additional_forward_args,
    _format_attributions,
    _run_forward,
    _reshape_and_sum,
    _expand_additional_forward_args,
)
from .._utils.attribution import GradientBasedAttribution


class IntegratedGradients(GradientBasedAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (function): The forward function of the model or
                       any modification of it
        """
        super().__init__(forward_func)

    def attribute(
        self,
        inputs,
        baselines=None,
        target=None,
        additional_forward_args=None,
        n_steps=50,
        method="gausslegendre",
    ):
        r"""
            Approximates the integral of gradients along the path from a baseline input
            to the given input. If no baseline is provided, the default baseline
            is the zero tensor.
            More details regarding the integrated gradient method can be found in the
            original paper here:
            https://arxiv.org/abs/1703.01365

            Args:

                inputs (tensor or tuple of tensors):  Input for which integrated
                            gradients are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples, and if mutliple input tensors
                            are provided, the examples must be aligned appropriately.
                baselines (tensor or tuple of tensors, optional):  Baseline from which
                            integral is computed. If inputs is a single tensor,
                            baselines must also be a single tensor with exactly the same
                            dimensions as inputs. If inputs is a tuple of tensors,
                            baselines must also be a tuple of tensors, with matching
                            dimensions to inputs.
                            Default: zero tensor for each input tensor
                target (int, optional):  Output index for which gradient is computed
                            (for classification cases, this is the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary. (Note: Tuples for multi
                            -dimensional output indices will be supported soon.)
                            Default: None
                additional_forward_args (tuple, optional): If the forward function
                            requires additional arguments other than the inputs for
                            which attributions should not be computed, this argument
                            can be provided. It can contain a tuple of ND tensors or
                            any arbitrary python type of any shape.
                            In case of the ND tensor the first dimension of the
                            tensor must correspond to the batch size. It will be
                            repeated for each of `n_steps` along the integrated path
                            of integrated gradients.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                n_steps (tuple, optional): The number of steps used by the approximation
                            method. Default: 50.
                method (string, optional): Method for approximating the integral,
                            one of `riemann_right`, `riemann_left`, `riemann_middle`,
                            `riemann_trapezoid` or `gausslegendre`.
                            Default: `gausslegendre` if no method is provided.

            Return:

                attributions (tensor or tuple of tensors): Integrated gradients with
                            respect to each input feature. attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.
                delta (float): The difference between the total approximation to the
                            integrated gradient and total true integrated gradient.
                            This is computed using the property that the total sum of
                            forward_func(inputs) - forward_func(baselines) must equal
                            the total sum of the integrated gradient.

            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> net = ImageClassifier()
                >>> ig = IntegratedGradients(net)
                >>> input = torch.randn(2, 3, 32, 32, requires_grad=True)
                >>> # Computes integrated gradients for class 3.
                >>> attribution, delta = ig.attribute(input, target=3)
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        inputs, baselines = _format_input_baseline(inputs, baselines)

        validate_input(inputs, baselines, n_steps, method)

        # retrieve step size and scaling factor for specified approximation method
        step_sizes_func, alphas_func = approximation_parameters(method)
        step_sizes, alphas = step_sizes_func(n_steps), alphas_func(n_steps)

        # scale features and compute gradients. (batch size is abbreviated as bsz)
        # scaled_features' dim -> (bsz * #steps x inputs[0].shape[1:], ...)
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
        # currently, number of steps is applied only to additional forward arguemnts
        # that are nd-tensors. It is assumed that the first dimension is
        # the number of batches.
        # dim -> (bsz * #steps x additional_forward_args[0].shape[1:], ...)
        input_additional_args = (
            _expand_additional_forward_args(additional_forward_args, n_steps)
            if additional_forward_args is not None
            else None
        )

        # grads: dim -> (bsz * #steps x inputs[0].shape[1:], ...)
        grads = self.gradient_func(
            self.forward_func, scaled_features_tpl, target, input_additional_args
        )

        # flattening grads so that we can multipy it with step-size
        # calling contigous to avoid `memory whole` problems
        scaled_grads = [
            grad.contiguous().view(n_steps, -1)
            * torch.tensor(step_sizes).view(n_steps, 1).to(grad.device)
            for grad in grads
        ]

        # aggregates across all steps for each tensor in the input tuple
        # total_grads has the same dimentionality as inputs
        total_grads = [
            _reshape_and_sum(
                scaled_grad, n_steps, grad.shape[0] // n_steps, grad.shape[1:]
            )
            for (scaled_grad, grad) in zip(scaled_grads, grads)
        ]

        # computes attribution for each tensor in input tuple
        # attributions has the same dimentionality as inputs
        attributions = tuple(
            total_grad * (input - baseline)
            for total_grad, input, baseline in zip(total_grads, inputs, baselines)
        )

        start_point, end_point = baselines, inputs

        # computes approximation error based on the completeness axiom
        delta = self._compute_convergence_delta(
            attributions,
            start_point,
            end_point,
            additional_forward_args=additional_forward_args,
            target=target,
        )

        return _format_attributions(is_inputs_tuple, attributions), delta

    def _compute_convergence_delta(
        self,
        attributions,
        start_point,
        end_point,
        target=None,
        additional_forward_args=None,
    ):
        attr_sum = sum(attribution.sum().item() for attribution in attributions)
        start_point = (
            _run_forward(
                self.forward_func, start_point, target, additional_forward_args
            )
            .sum()
            .item()
        )
        end_point = (
            _run_forward(self.forward_func, end_point, target, additional_forward_args)
            .sum()
            .item()
        )

        return abs(attr_sum - (end_point - start_point))
