<a href="https://captum.org">
  <img width="350" src="./captum_logo_lockup.svg" alt="Captum Logo" />
</a>

<hr/>

[![Conda](https://img.shields.io/conda/v/pytorch/captum.svg)](https://anaconda.org/pytorch/captum)
[![PyPI](https://img.shields.io/pypi/v/captum.svg)](https://pypi.org/project/captum)
[![CircleCI](https://circleci.com/gh/pytorch/captum.svg?style=shield)](https://circleci.com/gh/pytorch/captum)
[![Codecov](https://img.shields.io/codecov/c/github/pytorch/captum.svg)](https://codecov.io/github/pytorch/captum)

Captum is a model interpretability and understanding library for PyTorch.
Captum means comprehension in latin and contains general purpose implementations
of integrated gradients, saliency maps, smoothgrad, vargrad and others for
PyTorch models. It has quick integration for models built with domain-specific
libraries such as torchvision, torchtext, and others.

*Captum is currently in beta and under active development!*


#### Why Captum?

TODO: Add content here


#### Target Audience

TODO: Add content here


## Installation

**Installation Requirements**
- Python >= 3.6
- PyTorch >= 1.2


##### Installing the latest release

The latest release of Captum is easily installed either via
[Anaconda](https://www.anaconda.com/distribution/#download-section) (recommended):
```bash
conda install captum -c pytorch
```
or via `pip`:
```bash
pip install captum
```


##### Installing from latest master

If you'd like to try our bleeding edge features (and don't mind potentially
running into the occasional bug here or there), you can install the latest
master directly from GitHub:
```bash
pip install git+https://github.com/pytorch/captum.git
```

**Manual / Dev install**

Alternatively, you can do a manual install. For a basic install, run:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .
```

To customize the installation, you can also run the following variants of the
above:
* `pip install -e .[dev]`: Also installs all tools necessary for development
  (testing, linting, docs building; see [Contributing](#contributing) below).
* `pip install -e .[tutorials]`: Also installs all packages necessary for running the tutorial notebooks.

To execute unit tests from a manual install, run:
```bash
# running a single unit test
python -m unittest -v tests.attr.test_saliency
# running all unit tests
pytest -ra
```

## Getting Started
Captum allows to interpret/understand the predictions of PyTorch models by
looking at the features that contribute to decisions that the model makes. It also
helps to understand which neurons and layers are important for model predictions.

To do so, it uses state of the art model interpretability algorithms and
attributes contributions to each input of the model with respect to
different neurons / layers, both intermediate and final.

Now, let's apply some of those algorithms to a Toy model that we have created for
demonstration purposes.
For simplicity we can use the following simple architecture but users are welcome
to use any PyTorch model of their choice.
```
import torch
import torch.nn as nn
import numpy as np

from captum.attr import IntegratedGradients, NoiseTunnel, \
    NeuronConductance, LayerConductance

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()

            self.lin1 = nn.Linear(3, 3)
            self.lin1.weight = nn.Parameter(torch.eye(3))
            self.lin2 = nn.Linear(3, 4)
            self.lin2.weight = nn.Parameter(torch.ones(4, 3))
            self.lin2.bias = nn.Parameter(torch.tensor([-10.0, 1.0, 1.0, 1.0]))
            self.relu = nn.ReLU()
            self.lin3 = nn.Linear(4, 1)
            self.lin3.weight = nn.Parameter(torch.ones(1, 4))
            self.lin3.bias = nn.Parameter(torch.tensor([-3.0]))

        def forward(self, input):
            lin1 = self.lin1(input)
            lin2 = self.lin2(lin1)
            relu1 = self.relu(lin2)
            lin3 = self.lin3(relu1)
            return lin3
```
Let's create an instance of our model and set it in eval mode.
```
model = ToyModel()
model.eval()
```
As next, we would like to define a simple input and baseline tensors.
The latter is a tensor that belongs to the input space and carries
no predictive signal. Zero tensor can serve as a baseline for some tasks.

Afterwards, we will apply model interpretability algorithms on the network
mentioned above, in order to understand the importance of individual
neurons/layers and the parts of the input that play an important role in the
final prediction.

Let's fix random seeds in order to make computations deterministic
```
torch.manual_seed(123)
np.random.seed(123)
```

Now, let's define our input and baseline tensors. Baselines are used in some
interpretability algorithms such as Integrated Gradients, DeepLift,
GradientShap, etc.
```
input = torch.rand(2, 3)
baseline = torch.zeros(2, 3)
```
As next we will use `IntegratedGradients` algorithms to assign attribution
scores to each input feature with respect to final output.
```
ig = IntegratedGradients(model)
attributions, delta = ig.attribute(input, baseline)
print('IG Attributions: ', attributions, ' Approximation error: ', delta)
```
Output:
```
IG Attributions:  tensor([[0.8883, 1.5497, 0.7550],
                          [2.0657, 0.2219, 2.5996]])
Approximation error:  9.5367431640625e-07
```
The algorithm outputs an attribution score for each input element and an
approximation error that we would like to minimize. If the approximation error
is large, we can try larger number of integral approximation steps by setting
`n_steps` to a larger value.
Positive attribution score means that the input in that particular position positively
contributed to the final prediction and negative means the opposite.
The magnitude of the attribution score signifies the strength of the contribution.
Zero attribution score means no contribution from that particular feature.

Similarly, we can apply GradientShap, DeepLift and other attribution algorithms to the model.
```
gs = GradientShap(model)

# We define a distribution of baselines and draw `n_samples` from that
# distribution in order to estimate the expectations of gradients across all baselines
baseline_dist = torch.rand(100, 3)
attributions, delta = gs.attribute(input, baseline_dist, n_samples=50)
print('GradientShap Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
GradientShap Attributions:  tensor([[ 0.0159, -0.8478,  0.3028],
                                    [ 0.1546, -1.0068,  0.2770]])
Approximation error:  tensor(0.0462)

```
In order to smoothen and improve the quality of the attributions we can run
Integrated Gradients and other attribution methods through a Noise Tunnel.
Noise Tunnel allows to use Smoothgrad/Smoothgrad_sq and Vargrad techniques
with gaussian kernels to smoothen the gradients.
Here is en example how we can use NoseTunnel with Integrated Gradients.
```
ig = IntegratedGradients(model)
nt = NoiseTunnel(ig)
attributions, delta = nt.attribute(input, nt_type='smoothgrad', baselines=baseline)
print('IG + SmoothGrad Attributions: ', attributions, ' Approximation error: ', delta)
```
Output
```
IG + SmoothGrad Attributions:  tensor([[-1.2138,  0.6688,  0.7747],
                                       [1.3862,  0.7529,  2.2907]])
Approximation error:  0.07243824005126953

```

Now, let's look into the internals of our network and understand which layers
and neurons are important for the predictions.
Let's start with the neuron conductance. Neuron conductance helps us to identify
input features that are important for a particular neuron in a given
layer. In this case third neuron in the second layer.
```
nc = NeuronConductance(model, model.lin2)
attributions = nc.attribute(input, neuron_index=3, baselines=baseline)
print('Neuron Attributions: ', attributions)
```
Output
```
Neuron Attributions:  tensor([[0.2902, 0.5062, 0.2466],
                             [0.6748, 0.0725, 0.8492]])
```
Since neuron conductance does not have completeness property, delta is not computed
here. However, layer conductance has completeness property and we will compute for
the layer.

Layer conductance shows the importance of neurons for a layer and given input.
It doesn't attribute the contribution scores to the input features
but shows the importance of each neuron in selected layer.
```
lc = LayerConductance(model, model.lin1)
attributions, delta = lc.attribute(input, baselines=baseline)
print('Layer Attributions: ', attributions, ' Approximation error: ', delta)
```
Outputs
```
Layer Attributions:  tensor([[0.8883, 1.5497, 0.7550],
                            [2.0657, 0.2219, 2.5996]], grad_fn=<SumBackward1>)
Approximation error:  9.5367431640625e-07
```

## Contributing
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


## References

* [Axiomatic Attribution for Deep Networks, Mukund Sundararajan et al. 2017](https://arxiv.org/abs/1703.01365)
* [Did the Model Understand the Question? Pramod K. Mudrakarta, et al. 2018](https://www.aclweb.org/anthology/P18-1176)
* [Investigating the influence of noise and distractors on the interpretation of neural networks, Pieter-Jan Kindermans et al. 2016](https://arxiv.org/abs/1611.07270)
* [SmoothGrad: removing noise by adding noise, Daniel Smilkov et al. 2017](https://arxiv.org/abs/1706.03825)
* [Local Explanation Methods for Deep Neural Networks Lack Sensitivity to Parameter Values, Julius Adebayo et al. 2018](https://arxiv.org/abs/1810.03307)
* [Sanity Checks for Saliency Maps, Julius Adebayo et al. 2018](https://arxiv.org/abs/1810.03292)
* [How Important is a neuron?, Kedar Dhamdhere et al. 2018](https://arxiv.org/abs/1805.12233)
* [Learning Important Features Through Propagating Activation Differences, Avanti Shrikumar et al. 2017](https://arxiv.org/pdf/1704.02685.pdf)
* [Computationally Efficient Measures of Internal Neuron Importance, Avanti Shrikumar et al. 2018](https://arxiv.org/pdf/1807.09946.pdf)
* [A Unified Approach to Interpreting Model Predictions, Scott M. Lundberg et al. 2017](http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions)
* [Influence-Directed Explanations for Deep Convolutional Networks, Klas Leino et al. 2018](https://arxiv.org/pdf/1802.03788.pdf)
* [Towards better understanding of gradient-based attribution methods for deep neural networks, Marco Ancona et al. 2018](https://openreview.net/pdf?id=Sy21R9JAW)

## License
Captum is BSD licensed, as found in the [LICENSE](LICENSE) file.
