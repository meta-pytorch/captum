#!/usr/bin/env python3

# Welcome to the PyTorch Captum setup.py.
#
# Environment variables for feature toggles:
#
#   BUILD_INSIGHTS
#     enables Captum Insights build via yarn
#

import os
import re
import subprocess
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 10

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


# Allow for environment variable checks
def check_env_flag(name, default=""):
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


BUILD_INSIGHTS = check_env_flag("BUILD_INSIGHTS")
VERBOSE_SCRIPT = True
for arg in sys.argv:
    if arg == "-q" or arg == "--quiet":
        VERBOSE_SCRIPT = False


def report(*args):
    if VERBOSE_SCRIPT:
        print(*args)
    else:
        pass


INSIGHTS_REQUIRES = ["flask", "ipython", "ipywidgets", "jupyter", "flask-compress"]

INSIGHTS_FILE_SUBDIRS = [
    "insights/attr_vis/frontend/build",
    "insights/attr_vis/models",
    "insights/attr_vis/widget/static",
]

TUTORIALS_REQUIRES = INSIGHTS_REQUIRES + ["torchtext", "torchvision"]

TEST_REQUIRES = ["pytest", "pytest-cov", "parameterized", "flask", "flask-compress"]

REMOTE_REQUIRES = ["openai"]

DEV_REQUIRES = (
    TUTORIALS_REQUIRES
    + TEST_REQUIRES
    + REMOTE_REQUIRES
    + [
        "black",
        "flake8",
        "sphinx<8.2.0",
        "sphinx-autodoc-typehints",
        "sphinxcontrib-katex",
        "mypy>=0.760",
        "pyre-check-nightly==0.0.101750936314",
        "usort==1.0.2",
        "ufmt",
        "scikit-learn",
        "annoy",
        "click<8.2.0",
    ]
)

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "captum/__init__.py"), "r") as f:
    version_match = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    assert version_match is not None, "Unable to find version string."
    version = version_match.group(1)
    report("-- Building version " + version)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


# optionally build Captum Insights via yarn
def build_insights():
    report("-- Building Captum Insights")
    command = "./scripts/build_insights.sh"
    report("Running: " + command)
    subprocess.check_call(command)


# explore paths under root and subdirs to gather package files
def get_package_files(root, subdirs):
    paths = []
    for subroot in subdirs:
        paths.append(os.path.join(subroot, "*"))
        for path, dirs, _ in os.walk(os.path.join(root, subroot)):
            for d in dirs:
                paths.append(os.path.join(path, d, "*")[len(root) + 1 :])
    return paths


if __name__ == "__main__":

    if BUILD_INSIGHTS:
        build_insights()

    package_files = get_package_files("captum", INSIGHTS_FILE_SUBDIRS)

    setup(
        name="captum",
        version=version,
        description="Model interpretability for PyTorch",
        author="PyTorch Team",
        license="BSD-3",
        url="https://captum.ai",
        project_urls={
            "Documentation": "https://captum.ai",
            "Source": "https://github.com/pytorch/captum",
            "conda": "https://anaconda.org/pytorch/captum",
        },
        keywords=[
            "Model Interpretability",
            "Model Understanding",
            "Model Interpretability",
            "Model Understanding",
            "Feature Importance",
            "Neuron Importance",
            "Data Attribution",
            "Explainable AI",
            "PyTorch",
        ],
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
        ],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">={required_major}.{required_minor}".format(
            required_minor=REQUIRED_MINOR,
            required_major=REQUIRED_MAJOR,
        ),
        install_requires=[
            "matplotlib",
            "numpy",
            "packaging",
            "torch>=2.3",
            "tqdm",
        ],
        packages=find_packages(exclude=("tests", "tests.*")),
        extras_require={
            "dev": DEV_REQUIRES,
            "insights": INSIGHTS_REQUIRES,
            "test": TEST_REQUIRES,
            "tutorials": TUTORIALS_REQUIRES,
            "remote": REMOTE_REQUIRES,
        },
        package_data={"captum": package_files},
        data_files=[
            (
                "share/jupyter/nbextensions/jupyter-captum-insights",
                [
                    "captum/insights/attr_vis/frontend/widget/src/extension.js",
                    "captum/insights/attr_vis/frontend/widget/src/index.js",
                ],
            ),
            (
                "etc/jupyter/nbconfig/notebook.d",
                ["captum/insights/attr_vis/widget/jupyter-captum-insights.json"],
            ),
        ],
    )
