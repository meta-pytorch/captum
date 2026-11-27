#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build the Sphinx API docs and copy the rendered HTML into the
Docusaurus static/ directory so it is served at /api/.

Docusaurus serves anything under ``website/static/`` verbatim at the
site root, so HTML at ``website/static/api/foo.html`` is reachable at
``https://captum.ai/api/foo.html``. Under v1 the Sphinx HTML was
wrapped in Docusaurus' ``<header>``/``<footer>`` via
``wrapPagesHTML`` — v3 has no such hook, so we serve the raw Sphinx
output directly.
"""

import argparse
import os
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_INPUT_DIR = os.path.join(REPO_ROOT, "sphinx", "build", "html")
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "website", "static", "api")


def parse_sphinx(input_dir: str, output_dir: str) -> None:
    """Mirror ``input_dir`` (Sphinx build output) into ``output_dir``.

    We copy every file (HTML, CSS, JS, images, search index, etc.)
    so that intra-Sphinx relative links keep working when served from
    the Docusaurus static/api/ tree.
    """
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Sphinx build output not found: {input_dir!r}. "
            "Run `make html` inside sphinx/ first."
        )

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Copy Sphinx-built API HTML into website/static/api/ so it is "
            "served at /api/ by Docusaurus v3."
        )
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        metavar="path",
        default=DEFAULT_INPUT_DIR,
        help="Sphinx HTML build directory (default: sphinx/build/html).",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="path",
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: website/static/api).",
    )
    args = parser.parse_args()
    parse_sphinx(args.input_dir, args.output_dir)
