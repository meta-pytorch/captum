captum-optim Module Overview
=================

About Optim
-----------------

The Optim module is a set tools for optimization based interpretability for neural networks. It is a continuation of the research work performed by the team behind the [tensorflow/lucid](https://github.com/tensorflow/lucid) library.


The Optim module is designed to be extremely customizable, as to avoid limitations in its research potential.

History
-----------------

The initial concept for the Optim module was devised by Ludwig Shubert, and then developed by Ben Egan and Swee Kiat Lim with help from Chris Olah & Narine Kokhlikyan. This is project began as an attempt to integrate the tensorflow/lucid library into the pytorch/captum library. It implements numerous papers on feature visualization and interpretability.


Optim Structure
-----------------

![](https://user-images.githubusercontent.com/10626398/177629584-33e7ff7c-a504-404e-a7ab-d8d786b7e25a.svg?sanitize=true)

The standard rendering process works like this for the forward pass, with loss objectives being able to target any of the steps:

* ``NaturalImage`` (``ImageParameterization`` ➔ ``ToRGB`` ➔ Squash Function ➔ ``ImageTensor``) ➔ Transforms ➔ Model


Parameterizations
-----------------

The default settings store image parameters in a fully decorrelated format where the spatial information and channel information is decorrelated. By preconditioning our optimizer with decorrelated data, we alter the loss landscape to make optimization significantly easier and decrease the presence of high frequency patterns. Parameterizations like these are also known as a differentiable image parameterizations.

![](https://user-images.githubusercontent.com/10626398/176753493-b90f4e18-0133-4dca-afd4-26e811aa965e.svg?sanitize=true)

* Decorrelated Data ➔ Recorrelate Spatial ➔ Recorrelate Color ➔ Squash Function ➔ Transforms ➔ Model

By default, recorrelation occurs entirely within the ``NaturalImage`` class.


Submodules
-----------------

**Reducer**: The reducer module makes it easy to perform dimensionality reduction with a wide array of algorithms like [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [UMAP](https://umap-learn.readthedocs.io/en/latest/), [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html), & [NMF](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html).

**Circuits**: The circuits module allows for the extraction of meaningful weight interactions from between neurons which aren’t literally adjacent in a neural network.

**Models**: The models module contains the model zoo of pretrained models along with various help functions and classes.

**Dataset**: The dataset module provides functions for calculating color correlation matrices of image datasets.


Docs
-----------------

The docs for the optim module can be found here.


Tutorials
-----------------

We also provide multiple tutorials covering a wide array of research for the optim module [here](https://github.com/ProGamerGov/captum/tree/master-optim/tutorials/optimvis).


FAQ
=================

**How do I know if my model is compatible with the Optim module?**

In general model layers need to be nn.Modules as functional layers don't support hooks and also cannot be replaced.
Please check out the 'Getting Started Model Preparation' tutorial notebook for more information.

**Are only RGB images supported or can I use a different color space?**

By default the rendering modules in Optim are setup for rendering RGB images, but they can easily support other [color spaces](https://en.wikipedia.org/wiki/Color_space) with a simple settings change. In the case of ``ToRGB``, you may have to provide a new 3x3 transform matrix for 3 channel color spaces. For color spaces using less than or greater than 3 channels, you will need to create a custom color recorrelation module to replace ``ToRGB``. New color correlation matrices can be created using the dataset module, or with your own custom algorithms.

**Why are my rendered visualizations poor quality or non-existent in outputs?**

There are a wide array of factors that dictate how well a model performs for rendering visualizations. Aspects like the model architecture, the training data used to train the model, the optimizer being used, and your Optim module settings like parameterizations & transforms all play an important role in creating visualizations.

ReLU layers will block the flow of gradients during the backward pass, if their inputs are less than 0. This can result in no visualizations being produced for the target, even if the model already performs well with other targets. To avoid this issue, you can ensure that all applicable ReLU layers have been replaced with Optim's ``RedirectedReLU`` layer (the ``replace_layers`` function makes this extremely easy to do!).

**Does the Optim module support JIT?**

For the most part, yes. Image parameterizations, transforms, and many of the helper classes & functions support JIT. The provided models also support JIT, but rendering JIT models with ``InputOptimizatization`` is not supported. The ``InputOptimizatization`` class itself does not support JIT either, but it does work with scripted image parameterizations and transforms. The loss objective system also does not support JIT. These limitations are due to the limitations with JIT supporting PyTorch hooks.

**What dtypes does the Optim module support?**

In addition to the default ``torch.float32`` dtype, the Optim module also easily support the other float dtypes. 

The ``FFTImage`` parameterization currently doesn't work with ``torch.float16`` or ``torch.bfloat16`` due to issues with PyTorch's support for ``torch.complex32``.


References
-----------------

* Feature Visualization: https://distill.pub/2017/feature-visualization/

* Differentiable Image Parameterizations: https://distill.pub/2018/differentiable-parameterizations/

* The Building Blocks of Interpretability: https://distill.pub/2018/building-blocks/

* Exploring Neural Networks with Activation Atlases: https://distill.pub/2019/activation-atlas/

* Understanding Deep Image Representations by Inverting Them: https://arxiv.org/abs/1412.0035

* Color information for region segmentation: https://www.sciencedirect.com/science/article/pii/0146664X80900477

* Thread: Circuits: https://distill.pub/2020/circuits/

  * Visualizing Weights: https://distill.pub/2020/circuits/visualizing-weights/

  * Weight Banding: https://distill.pub/2020/circuits/weight-banding/

  * Zoom in: https://distill.pub/2020/circuits/zoom-in/

  * Curve Detectors: https://distill.pub/2020/circuits/curve-detectors/

  * Frequency Edges: https://distill.pub/2020/circuits/frequency-edges/

* Multimodal Neurons in Artificial Neural Networks: https://distill.pub/2021/multimodal-neurons/
