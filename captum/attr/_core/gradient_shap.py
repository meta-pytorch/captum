#!/usr/bin/env python3
import torch

import numpy as np

from .._utils.attribution import GradientBasedAttribution
from .._utils.common import _format_attributions
from .noise_tunnel import NoiseTunnel


class GradientShap(GradientBasedAttribution):
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
        baselines,
        n_samples=50,
        stdevs=0.0,
        random_seed=None,
        target=None,
        additional_forward_args=None,
    ):
        r"""
        Implements gradient shap based on the implementation from Shap's primary
        author. For reference, please, view:

        https://github.com/slundberg/shap
        deep-learning-example-with-gradientexplainer-tensorflowkeraspytorch-models

        A Unified Approach to Interpreting Model Predictions
        http://papers.nips.cc/paper/
        7062-a-unified-approach-to-interpreting-model-predictions

        GradientShap approximates shap values by computing the expectations of
        gradients by randomly sampling from the distribution of
        baselines/references.
        It makes an assumption that the input features are independant and that there
        is a linear relationship between current inputs and the baselines/references.
        Under those assumptions, shap value can be approximated as the
        expectation of gradients that are computed for randomly generated `n_samples`
        input samples after adding gaussian noise `n_samples` times to each input
        for different baselines/references.

        In some sense it can be viewed as an approximation of integrated gradients
        by computing the expectations of gradients for different baselines.

        Current implementation uses Smoothgrad from `NoiseTunnel` in order to
        randomly draw samples from the distribution of baselines, add noise to input
        samples and compute the expectation (smoothgrad).

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
                        integral is computed. It is recommended that the number
                        of samples in the baselines' tensors is larger than one.
                        If inputs is a tuple of tensors,
                        baselines must also be a tuple of tensors, with the
                        same number of tensors as the inputs.
                        Default: zero tensor for each input tensor
            target (int, optional):  Output index for which gradient is computed
                        (for classification cases, this is the target class).
                        If the network returns a scalar value per example,
                        no target index is necessary. (Note: Tuples for multi
                        -dimensional output indices will be supported soon.)
            additional_forward_args (tuple, optional): If the forward function
                        requires additional arguments other than the inputs for
                        which attributions should not be computed, this argument
                        can be provided. It can contain a tuple of ND tensors or
                        any arbitrary python type of any shape.
                        In case of the ND tensor the first dimension of the
                        tensor must correspond to the batch size. It will be
                        repeated for each `n_steps` for each randomly generated
                        input sample.
                        Note that the gradients are not computed with respect
                        to these arguments.
                        Default: None
            n_steps (tuple, optional): The number of steps used by the smoothgrad
                        method. Default: 50.


            Examples::

                >>> # ImageClassifier takes a single input tensor of images Nx3x32x32,
                >>> # and returns an Nx10 tensor of class probabilities.
                >>> net = ImageClassifier()
                >>> gradient_shap = GradientShap(net)
                >>> input = torch.randn(3, 3, 32, 32, requires_grad=True)
                >>> # choosing baselines randomly
                >>> baselines = torch.randn(20, 3, 32, 32)
                >>> # Computes gradient shap for the input
                >>> # Attribution size matches input size, 3x12x32x32
                >>> attribution = gradient_shap.attribute(input, baselines, target=5)

        """
        input_min_baseline_x_grad = InputBaselineXGradient(self.forward_func)

        nt = NoiseTunnel(input_min_baseline_x_grad)
        attributions = nt.attribute(
            inputs,
            nt_type="smoothgrad",
            n_samples=n_samples,
            stdevs=stdevs,
            draw_baseline_from_distrib=True,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
        )
        delta = self._compute_convergence_delta(
            attributions if isinstance(attributions, tuple) else (attributions,),
            baselines,
            inputs,
            additional_forward_args=additional_forward_args,
            target=target,
            is_multi_baseline=True,
        )
        return attributions, delta

    def has_convergence_delta(self):
        return True


class InputBaselineXGradient(GradientBasedAttribution):
    def __init__(self, forward_func):
        r"""
        Args:

            forward_func (function): The forward function of the model or
                       any modification of it
        """
        super().__init__(forward_func)

    def attribute(
        self, inputs, baselines=None, target=None, additional_forward_args=None
    ):
        def scale_inputs(input, baseline, rand_coefficient):
            num_elements_input = int(np.prod(input.shape[1:]))

            # expand and reshape the indices
            rand_coefficient = (
                rand_coefficient.repeat_interleave(num_elements_input, dim=0)
                .view(input.shape)
                .requires_grad_()
            )

            input_baseline_scaled = (
                rand_coefficient * input + (1 - rand_coefficient) * baseline
            )
            return input_baseline_scaled

        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = isinstance(inputs, tuple)

        rand_coefficient = torch.tensor(
            np.random.uniform(0.0, 1.0, inputs[0].shape[0]),
            device=inputs[0].device,
            dtype=inputs[0].dtype,
        )

        input_baseline_scaled = tuple(
            scale_inputs(input, baseline, rand_coefficient)
            for input, baseline in zip(inputs, baselines)
        )
        grads = self.gradient_func(
            self.forward_func, input_baseline_scaled, target, additional_forward_args
        )

        input_baseline_diffs = tuple(
            input - baseline for input, baseline in zip(inputs, baselines)
        )
        attributions = tuple(
            input_baseline_diff * grad
            for input_baseline_diff, grad in zip(input_baseline_diffs, grads)
        )
        return _format_attributions(is_inputs_tuple, attributions)

    def _has_convergence_delta(self):
        return False
