#!/usr/bin/env python3

# pyre-strict
from typing import Any, Dict, List

import torch
from captum.attr._core.deep_lift import DeepLift, DeepLiftShap
from captum.attr._core.feature_ablation import FeatureAblation
from captum.attr._core.feature_permutation import FeaturePermutation
from captum.attr._core.gradient_shap import GradientShap
from captum.attr._core.guided_backprop_deconvnet import Deconvolution, GuidedBackprop
from captum.attr._core.guided_grad_cam import GuidedGradCam
from captum.attr._core.input_x_gradient import InputXGradient
from captum.attr._core.integrated_gradients import IntegratedGradients
from captum.attr._core.kernel_shap import KernelShap
from captum.attr._core.layer.grad_cam import LayerGradCam
from captum.attr._core.layer.internal_influence import InternalInfluence
from captum.attr._core.layer.layer_activation import LayerActivation
from captum.attr._core.layer.layer_conductance import LayerConductance
from captum.attr._core.layer.layer_deep_lift import LayerDeepLift, LayerDeepLiftShap
from captum.attr._core.layer.layer_feature_ablation import LayerFeatureAblation
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_lrp import LayerLRP
from captum.attr._core.lime import Lime
from captum.attr._core.lrp import LRP
from captum.attr._core.neuron.neuron_conductance import NeuronConductance
from captum.attr._core.neuron.neuron_deep_lift import NeuronDeepLift, NeuronDeepLiftShap
from captum.attr._core.neuron.neuron_feature_ablation import NeuronFeatureAblation
from captum.attr._core.neuron.neuron_gradient import NeuronGradient
from captum.attr._core.neuron.neuron_gradient_shap import NeuronGradientShap
from captum.attr._core.neuron.neuron_guided_backprop_deconvnet import (
    NeuronDeconvolution,
    NeuronGuidedBackprop,
)
from captum.attr._core.neuron.neuron_integrated_gradients import (
    NeuronIntegratedGradients,
)
from captum.attr._core.occlusion import Occlusion
from captum.attr._core.saliency import Saliency
from captum.attr._core.shapley_value import ShapleyValueSampling
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper
from captum.testing.helpers.basic import set_all_random_seeds
from captum.testing.helpers.basic_models import (
    BasicModel_ConvNet,
    BasicModel_MultiLayer,
    BasicModel_MultiLayer_MultiInput,
    BasicModel_MultiLayer_TrueMultiInput,
    ReLULinearModel,
)

"""
This file defines a test configuration for attribution methods, particularly
defining valid input parameters for attribution methods. These test cases are
utilized for DataParallel tests, JIT tests, and target tests. Generally, these
tests follow a consistent structure of running the identified algorithm(s) in
two different way, e.g. with a DataParallel or JIT wrapped model versus a standard
model and verifying that the results match. New tests for additional model variants
or features can be built using this config.

The current schema for each test cases (each element in the list config) includes
the following information:
* "name": String defining name for test config
* "algorithms": List of algorithms (Attribution classes) which are applicable for
    the given test case
* "model": nn.Module model for given test
* "attribute_args": Arguments to be passed to attribute call of algorithm
* "layer": nn.Module corresponding to layer for Layer or Neuron attribution
* "noise_tunnel": True or False, based on whether to apply NoiseTunnel to the algorithm.
    If True, "attribute_args" corresponds to arguments for NoiseTunnel.attribute.
* "baseline_distr": True or False based on whether baselines in "attribute_args" are
    provided as a distribution or per-example.
* "target_delta": Delta for comparison in test_targets
* "dp_delta": Delta for comparison in test_data_parallel

To add tests for a new algorithm, simply add the algorithm to any existing test
case with applicable parameters by adding the algorithm to the corresponding
algorithms list. If the algorithm has particular arguments not covered by existing
test cases, add a new test case following the config schema described above. For
targets tests, ensure that the new test cases includes cases with tensor or list
targets. If the new algorithm works with JIT models, make sure to also
add the method to the whitelist in test_jit.

To create new tests for all methods, follow the same structure as test_jit,
test_targets, or test_data_parallel. Each of these iterates through the test
config and creates relevant test cases based on the config.
"""

# Set random seeds to ensure deterministic behavior
set_all_random_seeds(1234)

config: List[Dict[str, Any]] = [
    # Attribution Method Configs
    # Primary Methods (Generic Configs)
    {
        "name": "basic_single_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 1},
    },
    {
        "name": "basic_single_target_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 1,
            "enable_cross_tensor_attribution": True,
        },
    },
    {
        "name": "basic_multi_input",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
        "dp_delta": 0.001,
    },
    {
        "name": "basic_multi_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
    },
    {
        "name": "basic_multi_input_multi_target",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
        },
        "dp_delta": 0.0005,
    },
    {
        "name": "basic_multi_input_multi_target_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "enable_cross_tensor_attribution": True,
        },
        "dp_delta": 0.0005,
    },
    {
        "name": "basic_multiple_tuple_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_multiple_tuple_target_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "enable_cross_tensor_attribution": True,
        },
    },
    {
        "name": "basic_tensor_single_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {"inputs": torch.randn(4, 3), "target": torch.tensor([0])},
    },
    {
        "name": "basic_tensor_multi_target",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([1, 1, 0, 0]),
        },
    },
    {
        "name": "basic_tensor_multi_target_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
            FeaturePermutation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([1, 1, 0, 0]),
            "enable_cross_tensor_attribution": True,
        },
    },
    # Primary Configs with Baselines
    {
        "name": "basic_multiple_tuple_target_with_baselines",
        "algorithms": [
            IntegratedGradients,
            FeatureAblation,
            DeepLift,
            ShapleyValueSampling,
            Lime,
            KernelShap,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_multiple_tuple_target_with_baselines_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "enable_cross_tensor_attribution": True,
        },
    },
    {
        "name": "basic_tensor_single_target_with_baselines",
        "algorithms": [
            IntegratedGradients,
            FeatureAblation,
            DeepLift,
            ShapleyValueSampling,
            Lime,
            KernelShap,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0]),
        },
    },
    {
        "name": "basic_tensor_single_target_with_baselines_cross_tensor_attributions",
        "algorithms": [
            FeatureAblation,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0]),
            "enable_cross_tensor_attribution": True,
        },
    },
    # Primary Configs with Internal Batching
    {
        "name": "basic_multiple_tuple_target_with_internal_batching",
        "algorithms": [IntegratedGradients],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
        },
    },
    # NoiseTunnel
    {
        "name": "basic_multi_input_multi_target_nt",
        "algorithms": [
            IntegratedGradients,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            Saliency,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "nt_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
        "dp_delta": 0.01,
    },
    {
        "name": "basic_multiple_target_with_baseline_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [0, 1, 1, 0],
            "nt_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multiple_tuple_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "nt_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_single_tensor_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0]),
            "nt_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multi_tensor_target_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "nt_samples": 20,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    {
        "name": "basic_multi_tensor_target_batched_nt",
        "algorithms": [
            IntegratedGradients,
            Saliency,
            InputXGradient,
            FeatureAblation,
            DeepLift,
            GuidedBackprop,
            Deconvolution,
            LRP,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "nt_samples": 20,
            "nt_samples_batch_size": 2,
            "stdevs": 0.0,
        },
        "noise_tunnel": True,
    },
    # DeepLift SHAP
    {
        "name": "basic_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_input_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (torch.randn(4, 3), torch.randn(4, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [0, 1, 1, 0],
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_targe_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0]),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_dl_shap",
        "algorithms": [DeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
        },
        "baseline_distr": True,
    },
    # Gradient SHAP
    {
        "name": "basic_multi_inp_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "attribute_args": {
            "inputs": (torch.randn(6, 3), torch.randn(6, 3)),
            "baselines": (torch.randn(1, 3), torch.randn(1, 3)),
            "additional_forward_args": (torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "stdevs": 0.0,
            "n_samples": 2000,
        },
        "target_delta": 1.0,
        "dp_delta": 0.005,
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": [0, 1, 1, 0],
            "n_samples": 800,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multiple_tuple_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.15 * torch.randn(1, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "n_samples": 2000,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "dp_delta": 0.003,
        "baseline_distr": True,
    },
    {
        "name": "basic_single_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": torch.tensor([0]),
            "n_samples": 500,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_multi_tensor_target_with_single_baseline_grad_shap",
        "algorithms": [GradientShap],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(1, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "n_samples": 500,
            "stdevs": 0.0,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    # Perturbation-Specific Configs
    {
        "name": "conv_with_perturbations_per_eval",
        "algorithms": [
            FeatureAblation,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
        ],
        "model": BasicModel_ConvNet(),
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "target": 0,
            "perturbations_per_eval": 20,
        },
        "dp_delta": 0.008,
    },
    {
        "name": "basic_multiple_tuple_target_with_perturbations_per_eval",
        "algorithms": [
            FeatureAblation,
            ShapleyValueSampling,
            FeaturePermutation,
            Lime,
            KernelShap,
        ],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "conv_occlusion_with_perturbations_per_eval",
        "algorithms": [Occlusion],
        "model": BasicModel_ConvNet(),
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "perturbations_per_eval": 8,
            "sliding_window_shapes": (1, 4, 2),
            "target": 0,
        },
    },
    {
        "name": "basic_multi_input_with_perturbations_per_eval_occlusion",
        "algorithms": [Occlusion],
        "model": ReLULinearModel(),
        "attribute_args": {
            "inputs": (torch.randn(4, 3), torch.randn(4, 3)),
            "perturbations_per_eval": 2,
            "sliding_window_shapes": ((2,), (1,)),
        },
    },
    {
        "name": "basic_multiple_tuple_target_occlusion",
        "algorithms": [Occlusion],
        "model": BasicModel_MultiLayer(),
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "sliding_window_shapes": (2,),
        },
    },
    # Layer Attribution Method Configs
    {
        "name": "conv_layer_single_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
            GuidedGradCam,
        ],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {"inputs": 100 * torch.randn(4, 1, 10, 10), "target": 1},
    },
    {
        "name": "basic_layer_in_place",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(inplace=True),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_output",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "multi_relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_multi_input",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
    },
    {
        "name": "basic_layer_multiple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": [0, 1, 1, 0]},
    },
    {
        "name": "basic_layer_tensor_multiple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
        },
    },
    {
        "name": "basic_layer_multiple_tuple_target",
        "algorithms": [
            LayerConductance,
            LayerIntegratedGradients,
            LayerDeepLift,
            InternalInfluence,
            LayerFeatureAblation,
            LayerGradientXActivation,
            LayerGradCam,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
        },
    },
    {
        "name": "basic_layer_multiple_tuple_target_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
        },
    },
    {
        "name": "basic_layer_multi_input_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
            "internal_batch_size": 2,
        },
    },
    {
        "name": "basic_layer_multi_output_with_internal_batching",
        "algorithms": [LayerConductance, InternalInfluence, LayerIntegratedGradients],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "multi_relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 0,
            "internal_batch_size": 2,
        },
    },
    # Layer Perturbation
    {
        "name": "basic_layer_multi_input_with_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "basic_layer_multi_output_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "multi_relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 0,
            "perturbations_per_eval": 2,
        },
    },
    {
        "name": "conv_layer_with_perturbations_per_eval",
        "algorithms": [LayerFeatureAblation],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "target": 1,
            "perturbations_per_eval": 20,
        },
    },
    # Layer DeepLiftSHAP
    {
        "name": "relu_layer_multi_inp_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": ReLULinearModel(),
        "layer": "l3",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(2, 3), 6 * torch.randn(2, 3)),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_output_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "multi_relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(2, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_inp_multi_target_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(11, 3), 6 * torch.randn(11, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multiple_target_dl_shap",
        "algorithms": [LayerDeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "target": [0, 1, 1, 0],
        },
        "baseline_distr": True,
    },
    # Layer Gradient SHAP
    {
        "name": "relu_layer_multi_inp_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": ReLULinearModel(),
        "layer": "l3",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (2 * torch.randn(2, 3), 6 * torch.randn(2, 3)),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_output_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": "multi_relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(2, 3),
            "target": 0,
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_layer_multi_inp_multi_target_grad_shap",
        "algorithms": [LayerGradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (torch.randn(6, 3), torch.randn(6, 3)),
            "baselines": (torch.randn(2, 3), torch.randn(2, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "n_samples": 1000,
        },
        "baseline_distr": True,
        "target_delta": 0.6,
    },
    # Neuron Attribution Method Configs
    {
        "name": "basic_neuron",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {"inputs": torch.randn(4, 3), "neuron_selector": 3},
    },
    {
        "name": "conv_neuron",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "neuron_selector": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multi_input",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_selector": (3,),
        },
    },
    # Neuron Conductance (with target)
    {
        "name": "basic_neuron_single_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": 1,
            "neuron_selector": 3,
        },
    },
    {
        "name": "basic_neuron_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [0, 1, 1, 0],
            "neuron_selector": 3,
        },
    },
    {
        "name": "conv_neuron_single_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "target": 1,
            "neuron_selector": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multi_input_multi_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "neuron_selector": 3,
        },
    },
    {
        "name": "basic_neuron_tensor_multiple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": torch.tensor([0, 1, 1, 0]),
            "neuron_selector": 3,
        },
    },
    {
        "name": "basic_neuron_multiple_tuple_target",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "neuron_selector": 3,
        },
    },
    # Neuron Conductance with Internal Batching
    {
        "name": "basic_neuron_multiple_tuple_target_with_internal_batching",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "target": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "additional_forward_args": (None, True),
            "internal_batch_size": 2,
            "neuron_selector": 3,
        },
    },
    {
        "name": "basic_neuron_multi_input_multi_target_with_internal_batching",
        "algorithms": [NeuronConductance],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "target": [0, 1, 1, 0, 0, 1],
            "internal_batch_size": 2,
            "neuron_selector": 3,
        },
    },
    # Neuron Gradient SHAP
    {
        "name": "basic_neuron_grad_shap",
        "algorithms": [NeuronGradientShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": torch.randn(1, 3),
            "neuron_selector": 3,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    {
        "name": "basic_neuron_multi_inp_grad_shap",
        "algorithms": [NeuronGradientShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(6, 3), 5 * torch.randn(6, 3)),
            "baselines": (10 * torch.randn(1, 3), 5 * torch.randn(1, 3)),
            "additional_forward_args": (2 * torch.randn(6, 3), 5),
            "neuron_selector": 3,
        },
        "target_delta": 0.6,
        "baseline_distr": True,
    },
    # Neuron DeepLift SHAP
    {
        "name": "basic_neuron_dl_shap",
        "algorithms": [NeuronDeepLiftShap],
        "model": BasicModel_MultiLayer(),
        "layer": "relu",
        "attribute_args": {
            "inputs": torch.randn(4, 3),
            "baselines": 0.5 * torch.randn(6, 3),
            "neuron_selector": (3,),
        },
        "baseline_distr": True,
    },
    {
        "name": "basic_neuron_multi_input_dl_shap",
        "algorithms": [NeuronDeepLiftShap],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (torch.randn(4, 3), torch.randn(4, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_selector": 3,
        },
        "baseline_distr": True,
    },
    # Neuron Feature Ablation
    {
        "name": "conv_neuron_with_perturbations_per_eval",
        "algorithms": [NeuronFeatureAblation],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": torch.arange(400).view(4, 1, 10, 10).float(),
            "perturbations_per_eval": 20,
            "neuron_selector": (0, 1, 0),
        },
    },
    {
        "name": "basic_neuron_multiple_input_with_baselines_and_perturbations_per_eval",
        "algorithms": [NeuronFeatureAblation],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "baselines": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_selector": (3,),
            "perturbations_per_eval": 2,
        },
    },
    # Neuron Attribution with Functional Selector
    {
        "name": "basic_neuron_multi_input_function_selector",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.relu",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "neuron_selector": lambda x: torch.sum(x, 1),
        },
    },
    # Neuron Attribution with slice Selector
    {
        "name": "conv_neuron_slice_selector",
        "algorithms": [
            NeuronGradient,
            NeuronIntegratedGradients,
            NeuronGuidedBackprop,
            NeuronDeconvolution,
            NeuronDeepLift,
            NeuronFeatureAblation,
        ],
        "model": BasicModel_ConvNet(),
        "layer": "conv2",
        "attribute_args": {
            "inputs": 100 * torch.randn(4, 1, 10, 10),
            "neuron_selector": (slice(0, 2, 1), 1, slice(0, 2, 1)),
        },
    },
    # Layer Attribution with Multiple Layers
    {
        "name": "basic_activation_multi_layer_multi_output",
        "algorithms": [LayerActivation],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": ["multi_relu", "linear1", "linear0"],
        "attribute_args": {"inputs": torch.randn(4, 3)},
    },
    {
        "name": "basic_gradient_multi_layer_multi_output",
        "algorithms": [LayerGradientXActivation],
        "model": BasicModel_MultiLayer(multi_input_module=True),
        "layer": ["multi_relu", "linear1", "linear0"],
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_ig_multi_layer_multi_output",
        "algorithms": [LayerIntegratedGradients],
        "model": BasicModel_MultiLayer_TrueMultiInput(),
        "layer": ["m1", "m234"],
        "attribute_args": {
            "inputs": (
                torch.randn(5, 3),
                torch.randn(5, 3),
                torch.randn(5, 3),
                torch.randn(5, 3),
            ),
            "target": 0,
        },
    },
    {
        "name": "basic_layer_ig_multi_layer_multi_output_with_input_wrapper",
        "algorithms": [LayerIntegratedGradients],
        "model": ModelInputWrapper(BasicModel_MultiLayer_TrueMultiInput()),
        "layer": ["module.m1", "module.m234"],
        "attribute_args": {
            "inputs": (
                torch.randn(5, 3),
                torch.randn(5, 3),
                torch.randn(5, 3),
                torch.randn(5, 3),
            ),
            "target": 0,
        },
    },
    # Layer LRP
    {
        "name": "basic_layer_lrp",
        "algorithms": [
            LayerLRP,
        ],
        "model": BasicModel_MultiLayer(),
        "layer": "linear2",
        "attribute_args": {"inputs": torch.randn(4, 3), "target": 0},
    },
    {
        "name": "basic_layer_lrp_multi_input",
        "algorithms": [
            LayerLRP,
        ],
        "model": BasicModel_MultiLayer_MultiInput(),
        "layer": "model.linear1",
        "attribute_args": {
            "inputs": (10 * torch.randn(12, 3), 5 * torch.randn(12, 3)),
            "additional_forward_args": (2 * torch.randn(12, 3), 5),
            "target": 0,
        },
        "dp_delta": 0.0002,
    },
]
