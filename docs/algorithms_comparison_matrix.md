---
id: algorithms_comparison_matrix
title: Algorithm Comparison Matrix
---

# **Attribution Algorithm Comparison Matrix**

Please, scroll to the right for more details.
<table style="overflow-x: scroll; overflow: auto; display: block;" width="100%">
  <tr>
    <th style="padding: 30px;">Algorithm</th>
    <th style="padding: 30px;">Type</th>
    <th style="padding: 80px 100px;">Application</th>
    <th style="padding: 30px;">Space&nbsp;Complexity</th>
    <th style="padding: 30px;">Model&nbsp;Passes&nbsp;(Forward Only or Forward and Backward))</th>
    <th style="padding: 30px;">Number&nbsp;of&nbsp;Samples&nbsp;Passed through Model's Forward (and Backward) Passes</th>
    <th style="padding: 30px;">Requires&nbsp;Baseline&nbsp;aka Reference ?</th>
    <th style="padding: 80px 150px;">Description</th>
  </tr>
  <tr>
    <td><strong>Integrated Gradients˚^</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function.</td>
    <td>O(#steps * #examples * #features)</td>
    <td>Forward and Backward</td>
    <td>#steps * #examples</td>
    <td>Yes (Single Baseline Per Input Example)</td>
    <td>Approximates the integral of gradients along the path (straight line from baseline to input) sand multiplies with (input - baseline)</td>
  </tr>
  <tr>
    <td><strong>DeepLift˚^</strong></td>
    <td>Application</td>
    <td>Any model that can be represented as a differentiable function. NOTE: In our implementation we perform gradient overrides only for a small set of non-linearities. If your model has any kind of special non-linearities that aren't included in our list, we need to add that support separately. </td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
    <td>#examples</td>
    <td>Yes (Single Baseline Per Input Example)</td>
    <td>Explains differences in the non-linear activations' outputs in terms of the differences of the input from its corresponding reference. NOTE: Currently, only rescale rule is supported.</td>
  </tr>
  <tr>
    <td><strong>DeepLiftSHAP˚^</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function. NOTE: In our implementation we perform gradient overrides only for a small set of non-linearities. If your model has any kind of special non-linearities that aren't included in our list, we need to add that support separately.</td>
    <td>O(#examples * #features * #baselines)</td>
    <td>Forward and Backward</td>
     <td>#steps * #examples</td>
    <td>Yes (Multiple Baselines Per Input Example)</td>
    <td> An extension of DeepLift that approximates SHAP values. For each input example it considers a distribution of baselines and computes the expected value of the attributions based on DeepLift algorithm across all input-baseline pairs. NOTE: Currently, only rescale rule is supported. </td>
  </tr>
  <tr>
    <td><strong>GradientSHAP˚^</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function.</td>
    <td>O(#examples * # samples * #features + #baselines * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples * #samples</td>
    <td>Yes (Multiple Baselines Per Input Example)</td>
    <td> Approximates SHAP values based on the expected gradients. It adds gaussian noise to each input example #samples times, selects a random point between each sample and randomly drawn baseline from baselines' distribution, computes the gradient for it and multiples it with (input - baseline). Final SHAP values represent the expected values of gradients * (input - baseline) for each input example.</td>
  </tr>
  <tr>
    <td><strong>Input * Gradient</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function.</td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Multiplies model inputs with the gradients of the model outputs w.r.t. those inputs.</td>
  </tr>
  <tr>
    <td><strong>Saliency˚</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function.</td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>The gradients of the output w.r.t. inputs.</td>
  </tr>
  <tr>
    <td><strong>Guided BackProp˚ / DeconvNet˚</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function. NOTE: this algorithm makes sense to use if the model contains RELUs since it is based on the idea of overriding the gradients of inputs or outputs of any ReLU.</td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Computes the gradients of the model outputs w.r.t. its inputs. If there are any RELUs present in the model, their gradients will be overridden so that only positive gradients of the inputs (in case of Guided BackProp) and outputs (in case of deconvnet) are back-propagated.</td>
  </tr>
  <tr>
    <td><strong>Guided GradCam</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function. NOTE: this algorithm is designed primarily for CNNs.</td>
    <td>O(2 * #examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Computes the element-wise product of Guided BackProp and up-sampled positive GradCam attributions.</td>
  </tr>
  <tr>
    <td><strong>LayerGradCam</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function and has a convolutional layer. NOTE: this algorithm is designed primarily for CNNs.</td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Computes the gradients of model outputs w.r.t. selected input layer, averages them for each output channel and multiplies with the layer activations.</td>
  </tr>
  <tr>
    <td><strong>Layer Internal Influence</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function and has a convolutional layer. NOTE: this algorithm is designed primarily for CNNs.</td>
    <td>O(#steps * #examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#steps * #examples </td>
    <td>Yes (Single Baseline Per Input Example)</td>
    <td>Approximates the integral of gradients along the path from baseline to inputs for selected input layer. </td>
  </tr>
  <tr>
    <td><strong>Layer Conductance˚</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function and has a convolutional layer.</td>
    <td>O(#steps * #examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#steps * #examples </td>
    <td>Yes (Single Baseline Per Input Example)</td>
    <td>Decomposes integrated gradients via chain rule. It approximates the integral of gradients defined by a chain rule, described as the gradients of the output w.r.t. to the neurons multiplied by the gradients of the neurons w.r.t. the inputs, along the path from baseline to inputs. Finally, the latter is multiplied by (input - baseline).</td>
  </tr>
  <tr>
    <td><strong>Layer Gradient * Activation</strong></td>
    <td>Gradient</td>
    <td>Any model that can be represented as a differentiable function and has a convolutional layer.</td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Computes element-wise product of layer activations and the gradient of the output w.r.t. that layer.</td>
  </tr>
  <tr>
    <td><strong>Layer Activation</strong></td>
    <td> - </td>
    <td>Any neural network model. </td>
    <td>O(#examples * #features)</td>
    <td>Forward and Backward</td>
     <td>#examples </td>
    <td>No</td>
    <td>Computes the inputs or outputs of selected layer.</td>
  </tr>
  <tr>
    <td><strong>Feature Ablation˚^</strong></td>
    <td> Perturbation </td>
    <td>Any traditional or neural network model. </td>
    <td>O(#examples * #features * #perturbations_per_eval) </td>
    <td>Forward</td>
     <td>#examples * #features </td>
    <td>Yes (Single Baseline Per Input Example; Usually, zero baseline is used)</td>
    <td>Assigns an importance score to each input feature based on the magnitude changes in model output or loss when those features are replaced by a baseline (usually zeros) based on an input feature mask.</td>
  </tr>
  <tr>
    <td><strong>Feature Permutation</strong></td>
    <td> Perturbation </td>
    <td>Any traditional or neural network model. </td>
    <td>O(#examples * #features * #perturbations_per_eval)</td>
    <td>Forward</td>
     <td>#examples * #features </td>
    <td>No (Internally in our implementation permuted features for each batch are treated as baselines)</td>
    <td>Assigns an importance score to each input feature based on the magnitude changes in model output or loss when those features are permuted based on input feature mask. </td>
  </tr>
  <tr>
    <td><strong>Occlusion</strong></td>
    <td> Perturbation </td>
    <td> Any traditional or neural network model. NOTE: this algorithm has been primarily used for computer vision but could theoretically also be used for other applications as well. In addition to that this algorithm also requires strides which indicates the length of the steps required for sliding k-dimensional window.</td>
    <td>O(#examples * #features * #ablations_per_eval *  1 / #strides)</td>
    <td>Forward</td>
     <td>#examples * #features </td>
    <td>Yes (usually, zero baseline is used)</td>
    <td>Assigns an importance score to each input feature based on the magnitude changes in model output when those features are replaced by a baseline (usually zeros) using rectangular sliding windows and sliding strides. If a features is located in multiple hyper-rectangles the importance scores are averaged across those hyper-rectangles.</td>
  </tr>
  <tr>
    <td><strong>Shapely Value</strong></td>
    <td>Perturbation </td>
    <td>Any traditional or neural network model.</td>
    <td>O(#examples * #features * #perturbations_per_eval )</td>
    <td>Forward</td>
     <td>#examples * #features * #features! </td>
    <td>Yes (usually, zero baseline is used)</td>
    <td>Computes feature importances based on all permutations of all input features. It adds each feature for each permutation one-by-one to the baseline and computes the magnitudes of output changes for each feature which are ultimately being averaged across all permutations to estimate final attribution score. </td>
  </tr>
  <tr>
    <td><strong>Shapely Value Sampling</strong></td>
    <td>Perturbation </td>
    <td>Any traditional or neural network model.</td>
    <td>O(#examples * #features * #perturbations_per_eval )</td>
    <td>Forward</td>
     <td>#examples * #features * #samples</td>
    <td>Yes (usually, zero baseline is used)</td>
    <td>Similar to Shapely value, but instead of considering all feature permutations it considers only #samples random permutations.</td>
  </tr>
  <tr>
    <td><strong>NoiseTunnel</strong></td>
    <td> -  </td>
    <td>This can be used in combination with any above mentioned attribution algorithms</td>
    <td>Depends on the choice of above mentioned attribution algorithm. </td>
    <td>Forward or Forward and Backward - It depends on the choice of above mentioned attribution algorithm.</td>
     <td>#examples * #features * #samples</td>
    <td>Depends on the choice of above mentioned attribution algorithm. </td>
    <td>Depends on the choice of above mentioned attribution algorithm. | Adds gaussian noise to each input example #samples times, calls any above mentioned attribution algorithm for all #samples per example and aggregates / smoothens them based on different techniques for each input example. Supported smoothing techniques include: smoothgrad, vargrad, smoothgrad_sq.</td>
  </tr>
  <tr>
    <td><strong>ReX</strong></td>
    <td> Perturbation </td>
    <td>Any function returning a single value</td>
    <td> O(#partitions ^ #max_depth)  - user defined values </td>
    <td> Any function returning a single value </td>
     <td>O( #iterations (#partitions ^ #max_depth))</td>
    <td>Yes (strong assumption regarding neutral baseline) </td>
    <td>Perturbation based approach based on a recursive search over the input. By recursively occluding partitions of an input, ReX searches for partitions who's values have predictive value wrt. the output. </td>
  </tr>
</table>

**^ Including Layer Variant**

**˚ Including Neuron Variant**

<a href="/img/algorithms_comparison_matrix.png">Algorithm Comparison Matrix.png</a>
