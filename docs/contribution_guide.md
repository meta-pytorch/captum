---
id: contribution_guidelines
title: The Captum Contribution Process
---

The Captum development process involves a healthy amount of open discussions between the core development team and the community.
Captum operates similarly to most open source projects on GitHub. However, if you've never contributed to an open source project before, here is the basic process.


1. **Figure out what you're going to work on.**
2. **Figure out the scope of your change and reach out for design comments or feedback on a GitHub issue.**
    * **If you want to contribute a new feature or algorithm, please, check out the section on Proposing new features and algorithms.**
    * We expect that all contributed features, algorithms and improvements follow these guidelines, analogous to existing methods in Captum:
        * Algorithm APIs should maintain as much similarity as possible with existing APIs to ensure ease-of-use when switching between algorithms.
        * Detailed documentation including a summary of the general algorithm, example usage, and descriptions of all parameters and returns as well as documentation of any limitations.
        * Explicit test cases for test models, similar to existing method test cases.
        * Test the algorithms and changes on real models and real datasets. Use the datasets and models mentioned in the benchmarking section or in Captum tutorials or propose new datasets in GitHub issue’s section and we will review them.
        * Support and tests for model wrappers such as DataParallel, DistributedDataParallel, and JIT.
        * Type hints and formatting as described in the general contributing guidelines.
3. **Code it out!**
4. **Open a Pull Request!**
5. **Iterate on the pull request until it's accepted!**

## Proposing New Features and Algorithms

New feature ideas are best discussed on a specific issue. Please include as much information as you can, including any accompanying data and your proposed solution. The Captum team and community frequently reviews new issues and comments where they think they can help.

While we would like to accept as many algorithms and features to Captum as possible, the core team is responsible for maintaining and supporting all functionalities in the future, so we need to ensure the package remains maintainable. Additionally, we also want to ensure that the supported methods cover the variety and breadth of methods in model interpretability, without overwhelming new users with many new or experimental methods. To balance these objectives, we have established guidelines and general criteria to help make decisions when considering new algorithms and features. The evaluation of any model interpretability algorithm can be subjective, especially since there are no general purpose qualitative and quantitative metrics measuring their quality. Hence, we provide these guidelines as a starting point and will utilize discussions in GitHub issues (https://github.com/pytorch/captum/issues) to ensure a transparent process while reviewing proposals.

Before contributing an algorithm to Captum, please review these guidelines and provide as much context as possible regarding which criteria apply to the proposed method.

1. **Usage / Citations and Impact on Model Interpretability Community**
    * A primary factor we look at when considering new methods is popularity of the method and impact on the model interpretability community. Our baseline for consideration is generally 20 citations or 100 forks on GitHub, but this is not an absolute requirement.  If an algorithm is newly published or has had less visibility in the community, this criteria may not be satisfied, but strengths in the remaining criteria could justify acceptance.
2. **Multimodality**
    * Since one of the core values of Captum is multimodality, we prefer algorithms that are generic so that they can be used for different types of model architectures and input types. If the algorithm is unimodal, let’s say, if it works only for text or vision models, it is important to clarify what type of vision or text models the implementation supports. Does it work for LSTMs only? In other words, it is important to discuss the scope of the algorithm and its impact.
3. **Benchmarking Results**
    * We also would like comparisons with existing algorithm benchmarks in terms of performance and visual interpretation, and strong results compared to baselines are a plus when considering inclusion in Captum.
        * **Performance benchmarking**
            * Please report runtime execution numbers of the algorithm in CPU and GPU environments. Describe the environment where the experiments were conducted. It is also encouraged to do performance comparison with existing baseline approaches and report those numbers as well.
        * **Visual interpretation**
            * Although visual interpretations can be deceptive, it is important to compare newly implemented algorithms with other state of the art approaches side by  side using well-known baseline models and datasets. For baseline models and datasets check out the section on [**Algorithm benchmarking on real datasets and models**](#algorithm-benchmarking-on-real-datasets-and-models) section
        * In addition to visual interpretations, if possible, for attribution algorithms we can also assess infidelity and sensitivity metrics provided in captum.metrics package.
4. **Axiomatic and Mathematically Sound**
    * Since evaluation and qualitative analysis of interpretability methods can sometimes be misleading, methods that are axiomatic or have strong theoretical justification are preferred.

If you think that proposed algorithm/feature satisfies many of the above criteria, please open an issue on GitHub (https://github.com/pytorch/captum/issues) to discuss the method with the core Captum team members before submitting a pull request. If the method seems suitable for inclusion in Captum, we will generally request a design document or proposal, explaining the proposed API and structure for the new algorithm. An example of a design proposal for LIME and Kernel SHAP can be found here (https://github.com/pytorch/captum/issues/467).

If an algorithm or feature adds only marginal improvements or does not meet most the criteria described above, then we would suggest including it into our AWESOME_LIST.MD (https://github.com/pytorch/captum/blob/master/AWESOME_LIST.md) instead of adding it to the core Captum library. In the future, if the algorithm gains more popularity and acceptance in the model interpretability community, we would be happy to accept a PR to add it to the Captum core library.

*Note that we reserve the right to decide not to include any algorithms that meet the above criteria, but we are unable to support in the long run.*


## Algorithm benchmarking on real datasets and models

**NLP**
- We provide a sample CNN-based model for sensitivity analysis. Although, currently, we do not provide a LSTM model in the tutorials, we strongly encourage you to test the model on a baseline LSTM model (e.g. the original LSTM model described inLong short-term memory (https://dl.acm.org/doi/10.1162/neco.1997.9.8.1735) for sentiment analysis) as well.
Besides that, it is encouraged to test the algorithms on Bert models as well. An example Bert Q&A model is available in the tutorial’s section.
https://captum.ai/tutorials/Bert_SQUAD_Interpret
https://captum.ai/tutorials/IMDB_TorchText_Interpret

**Vision**
- We provide a sample toy model for the CIFAR dataset and examples with a ResNet model.
https://captum.ai/tutorials/CIFAR_TorchVision_Interpret
https://captum.ai/tutorials/Resnet_TorchVision_Interpret
These would be great starting points for benchmarking.
We also encourage you to test your models on other well-known benchmarks such as MNIST digit and fashion
dataset, Inception and VGG models.

**Baseline MLP Classification Models**
- In terms of baseline MLP models and datasets we encourage you to use titanic dataset and the simple MLP model that we built in the following tutorial:
https://captum.ai/tutorials/Titanic_Basic_Interpret

**Baseline Regression models**
- Boston House prices dataset and model can be found here:
https://captum.ai/tutorials/House_Prices_Regression_Interpret

**Multimodal**
- You can use VQA model and dataset described here:
https://captum.ai/tutorials/Multimodal_VQA_Interpret


## Docstring style

Docstring is required for all public APIs to provide users the details of the arguments and returns. [Our API documentation](https://captum.ai/api/) is generated from the docstring. Captum adopts a customized docstring format modified on top of [Google style](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html). Specifically, each argument should be listed as `arg_name (type): description` in the `Args:` section. The argument typing convention:
- primitive types: `int`, `str`, `float`, `bool`
- common collection types: `list`, `tuple`, `dict`
    - [PEP 585](https://peps.python.org/pep-0585/#implementation) has deprecated the duplicate types: `List`, `Tuple`, `Dict`
    - element types: `list[int]`, `dict[int, str]`
- other foundamental types: `Any`, `Callable`, `Iterable`
- class types: `MyClass`, `external_lib.SomeClass`
- omit `torch` for common Pytorch types: `Tensor`, `nn.Module`
- use `or` and `,` for union types: `type1 or type2`, `type1, tyep2, or type3`
    - [PEP 604](https://peps.python.org/pep-0604/) proposes to use `|` to connect types: `type1 | type2`. We may consider migration later.
- append `optional` for argument with default value: `int, optional`
    - append default value to the end of the description: `Default: None`
    - Notice this is different with python's type hint `Optional[...]`, which indicate if the argument can be `None`
