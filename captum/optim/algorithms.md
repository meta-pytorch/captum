---
id: algorithms
title: Loss Objective Descriptions
---

## Loss Objectives

### LayerActivation
This is the most basic loss available and it simply returns the activations in their original form.

* Pros: Can potentially give a broad overview of a target layer.
* Cons: Not specific enough for most research tasks.

### ChannelActivation
This loss maximizes the activations of a target channel in a specified target layer, and can be useful to determine what features the channel is excited by.

* Pros: A good balance between neuron and layer activation.
* Cons: Can be very polysemantic in many cases.

### NeuronActivation:
This loss maximizes the activations of a target neuron in the specified channel from the specified layer. This loss is useful for determining the type of features that excite a neuron, and thus is often used for circuits and neuron related research.

* Pros: Extremely specific in what it targets, and thus the information obtained can be extremely useful.
* Cons: Sometimes you don’t want something overly specific. Neurons don’t scale well to larger image sizes when rendering.

### DeepDream
This loss returns the squared layer activations. When combined with a negative mean loss summarization, this loss will create hallucinogenic visuals commonly referred to as ‘Deep Dream’. 

* Pros: Can create visually interesting images.
* Cons: Not specific enough for most research tasks.

### TotalVariation
This loss attempts to smooth / denoise the target by performing total variance denoising. The target is most often the image that’s being optimized. This loss is often used to remove unwanted visual artifacts.

* Pros: Can remove unwanted visual artifacts.
* Cons: Can result in less sharp / more blurry visualization images.

### L1
Penalizes the l1 of the target layer activations.

* Pros: Can be used as a penalty, similar to L1 regularization.
* Cons:

### L2
Penalizes the l2 of the target layer activations.

* Pros: Can be used as a penalty, similar to L2 regularization.
* Cons:

### Diversity
This loss helps break up polysemantic layers, channels, and neurons by encouraging diversity across the different batches. This loss is to be used along with a main loss.

* Pros: Helps separate polysemantic features into different images.
* Cons: Can be extremely slow with large batch sizes. This loss really only works on targets that are polysemantic. 

### ActivationInterpolation
This loss helps to interpolate or mix visualizations from two activations (layer or channel) by interpolating a linear sum between the two activations.

* Pros: Can create visually interesting images, especially when used with Alignment.
* Cons: Interpolations may not be semantically useful beyond visual interest.

### Alignment
When interpolating between activations, it may be desirable to keep image landmarks in the same position for visual comparison. This loss helps to minimize L2 distance between neighbouring images. 

* Pros: Helps to make interpolated images more comparable.
* Cons: Resulting images may be less semantically representative of the channel/layer/neuron, since we are forcing images to also be visually aligned.

### Direction
This loss helps to visualize a specific vector direction in a layer, by maximizing the alignment between the input vector and the layer’s activation vector. The dimensionality of the vector should correspond to the number of channels in the layer.

* Pros: Szegedy et al. and Bau et al. respectively found that activations along random and basis directions could be semantically meaningful and this loss allows us to visualize these directions.
* Cons: Largely random and, as of now, no structured way to find meaningful directions.

### NeuronDirection
Extends Direction loss by focusing on visualizing a single neuron within the kernel.

* Pros: See Direction loss.
* Cons: See Direction loss.

### AngledNeuronDirection
This objective is similar to NeuronDirection, but it places more emphasis on the angle by optionally multiplying the dot product by the cosine similarity.

* Pros: More useful for visualizing activation atlas images.

### TensorDirection
Extends Direction loss by allowing batch-wise direction visualization.

Pros: See Direction loss.
Cons: See Direction loss.

### ActivationWeights
This loss weighs specific channels or neurons in a given layer, via a weight vector. 

* Pros: Allows for region and dimension specific weighting.
* Cons: Requires knowledge beforehand of the target region.

### L2Mean
A simple L2 penalty where the mean is used instead of the square root of the sum.

* Pros: It was found to work better for CLIP visualizations than the traditional L2 objective.
* Cons:

### VectorLoss

This loss objective is similar to the Direction objective, except it computes the matrix product of the activations and vector, rather than the cosine similarity. In addition to optimizing towards channel directions, this objective can also perform a similar role to the ChannelActivation objective by using one-hot 1D vectors.

* Pros:
* Cons:

### FacetLoss

The FacetLoss objective allows us to steer feature visualization towards a particular theme / concept. This is done by using the weights from linear probes trained on the lower layers of a model to discriminate between a certain theme or concept and generic natural images.

* Pros: Works on highly polysemantic / highly faceted targets where the Diversity objective fails due to lack of specificity.
* Cons: Requires training linear probes on the target layers using training images from the desired facet.
