#!/usr/bin/env python3

# pyre-strict
from captum.attr._core.dataloader_attr import DataLoaderAttribution
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
from captum.attr._core.layer.layer_feature_permutation import LayerFeaturePermutation
from captum.attr._core.layer.layer_gradient_shap import LayerGradientShap
from captum.attr._core.layer.layer_gradient_x_activation import LayerGradientXActivation
from captum.attr._core.layer.layer_integrated_gradients import LayerIntegratedGradients
from captum.attr._core.layer.layer_lrp import LayerLRP
from captum.attr._core.lime import Lime, LimeBase
from captum.attr._core.llm_attr import (
    LLMAttribution,
    LLMAttributionResult,
    LLMGradientAttribution,
    RemoteLLMAttribution,
)
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
from captum.attr._core.noise_tunnel import NoiseTunnel
from captum.attr._core.occlusion import Occlusion
from captum.attr._core.remote_provider import RemoteLLMProvider, VLLMProvider
from captum.attr._core.saliency import Saliency
from captum.attr._core.shapley_value import ShapleyValues, ShapleyValueSampling
from captum.attr._models.base import (
    configure_interpretable_embedding_layer,
    InterpretableEmbeddingBase,
    remove_interpretable_embedding_layer,
    TokenReferenceBase,
)
from captum.attr._utils import visualization
from captum.attr._utils.attribution import (
    Attribution,
    GradientAttribution,
    LayerAttribution,
    NeuronAttribution,
    PerturbationAttribution,
)
from captum.attr._utils.baselines import ProductBaselines
from captum.attr._utils.class_summarizer import ClassSummarizer
from captum.attr._utils.interpretable_input import (
    ImageMaskInput,
    InterpretableInput,
    TextTemplateInput,
    TextTokenInput,
)
from captum.attr._utils.stat import (
    CommonStats,
    Count,
    Max,
    Mean,
    Min,
    MSE,
    StatValue,
    StdDev,
    Sum,
    Var,
)
from captum.attr._utils.summarizer import Summarizer, SummarizerSingleTensor

__all__ = [
    "Attribution",
    "GradientAttribution",
    "PerturbationAttribution",
    "NeuronAttribution",
    "LayerAttribution",
    "IntegratedGradients",
    "DataLoaderAttribution",
    "DeepLift",
    "DeepLiftShap",
    "InputXGradient",
    "Saliency",
    "GuidedBackprop",
    "Deconvolution",
    "GuidedGradCam",
    "FeatureAblation",
    "FeaturePermutation",
    "Occlusion",
    "ShapleyValueSampling",
    "ShapleyValues",
    "LimeBase",
    "Lime",
    "LRP",
    "KernelShap",
    "LayerConductance",
    "LayerGradientXActivation",
    "LayerActivation",
    "LayerFeaturePermutation",
    "LayerFeatureAblation",
    "LLMAttribution",
    "LLMAttributionResult",
    "LLMGradientAttribution",
    "RemoteLLMAttribution",
    "RemoteLLMProvider",
    "VLLMProvider",
    "InternalInfluence",
    "InterpretableInput",
    "LayerGradCam",
    "LayerDeepLift",
    "LayerDeepLiftShap",
    "LayerGradientShap",
    "LayerIntegratedGradients",
    "LayerLRP",
    "NeuronConductance",
    "NeuronFeatureAblation",
    "NeuronGradient",
    "NeuronIntegratedGradients",
    "NeuronDeepLift",
    "NeuronDeepLiftShap",
    "NeuronGradientShap",
    "NeuronDeconvolution",
    "NeuronGuidedBackprop",
    "NoiseTunnel",
    "ProductBaselines",
    "GradientShap",
    "ImageMaskInput",
    "InterpretableEmbeddingBase",
    "TextTemplateInput",
    "TextTokenInput",
    "TokenReferenceBase",
    "visualization",
    "configure_interpretable_embedding_layer",
    "remove_interpretable_embedding_layer",
    "Summarizer",
    "CommonStats",
    "ClassSummarizer",
    "Mean",
    "StdDev",
    "MSE",
    "Var",
    "Min",
    "Max",
    "Sum",
    "Count",
    "StatValue",
    "SummarizerSingleTensor",
]
