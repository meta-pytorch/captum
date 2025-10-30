# Contributing to Captum

Thank you for your interest in contributing to Captum! We want to make contributing to Captum as easy and transparent as possible.
Before you begin writing code, it is important that you share your intention to contribute with the team, based on the type of contribution:


1. You want to propose and implement a new algorithm, add a new feature or fix a bug. This can be both code and documentation proposals.
    1. For all non-outstanding features, bug-fixes and algorithms in the Captum issue list (https://github.com/pytorch/captum/issues) please create an issue first.
    2. If the implementation requires API or any other major code changes (new files, packages or algorithms), we will likely request a design document to review and discuss the design and implementation before making changes. An example design document for LIME can be found here (https://github.com/pytorch/captum/issues/467).
    3. Once we agree that the plan looks good or confirmed that the change is small enough to not require a detailed design discussion, go ahead and implement it!

2. You want to implement a feature or bug-fix for an outstanding issue.

    1. Search for your issue in the Captum issue list (https://github.com/pytorch/captum/issues).
    2. Pick an issue and comment that you'd like to work on the feature or bug-fix.
    3. If you need more context on a particular issue, please ask and we’ll be happy to help.

Once you implement and test your feature or bug-fix, please submit a Pull Request to https://github.com/pytorch/captum (https://github.com/pytorch/pytorch).

This document covers some of the techical aspects of contributing to Captum. More details on what we are looking for in the contributions can be found in the [Contributing Guidelines](https://captum.ai/docs/contribution_guidelines).


## Development installation

To get the development installation with all the necessary dependencies for
linting, testing, and building the documentation, run the following:
```bash
git clone https://github.com/pytorch/captum.git
cd captum
pip install -e .[dev]
```


## Our Development Process

#### Code Style

Captum uses [ufmt](https://pypi.org/project/ufmt/) and  [flake8](https://github.com/PyCQA/flake8) to
enforce a common code style across the code base. ufmt and flake8 are installed easily via
pip using `pip install ufmt flake8`, and run locally by calling
```bash
ufmt format .
flake8 .
```
from the repository root.

We feel strongly that having a consistent code style is extremely important, so
Github Actions will fail on your PR if it does not adhere to the ufmt or flake8 formatting style.


#### Type Hints

Captum is fully typed using Python 3.10+
[type hints](https://www.python.org/dev/peps/pep-0484/).
We expect any contributions to also use proper type annotations, and we enforce
consistency of these in our continuous integration tests.

To type check your code locally, install [mypy](https://github.com/python/mypy),
which can be done with pip using `pip install "mypy>=0.760"`
Then run this script from the repository root:
```bash
./scripts/run_mypy.sh
```
Note that we expect mypy to have version 0.760 or higher, and when type checking, use PyTorch 1.10 or
higher due to fixes to the PyTorch type hints available. We also use the Literal feature which is
available only in Python 3.10 or above.

We also use [pyre](https://pyre-check.org/) for type checking. For contributors, the nightly version of
pyre is used which can be installed with pip `pip install pyre-check-nightly`. To run pyre, you can
use the following command from inside the repository root:
```bash
pyre check
```

#### Unit Tests

To run the unit tests, you can either use `pytest` (if installed):
```bash
pytest -ra
```
or python's `unittest`:
```bash
python -m unittest
```

To get coverage reports we recommend using the `pytest-cov` plugin:
```bash
pytest -ra --cov=. --cov-report term-missing
```


#### Documentation

Captum's website is also open source, and is part of this very repository (the
code can be found in the [website](/website/) folder).
It is built using [Docusaurus](https://docusaurus.io/), and consists of three
main elements:

1. The documentation in Docusaurus itself (if you know Markdown, you can
   already contribute!). This lives in the [docs](/docs/).
2. The API reference, auto-generated from the docstrings using
   [Sphinx](http://www.sphinx-doc.org), and embedded into the Docusaurus website.
   The sphinx .rst source files for this live in [sphinx/source](/sphinx/source/).
3. The Jupyter notebook tutorials, parsed by `nbconvert`, and embedded into the
   Docusaurus website. These live in [tutorials](/tutorials/).

To build the documentation you will need [Node](https://nodejs.org/en/) >= 8.x
and [Yarn](https://yarnpkg.com/en/) >= 1.5.

The following command will both build the docs and serve the site locally:
```bash
./scripts/build_docs.sh
```

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `master`.
2. If you have added code that should be tested, add unit tests.
   In other words, add unit tests.
3. If you have changed APIs, update the documentation. Make sure the
   documentation builds.
4. Ensure the test suite passes.
5. Make sure your code passes both `black` and `flake8` formatting checks.


## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.


## License

By contributing to Captum, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
