#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Build the Captum documentation site (Docusaurus v3 + Sphinx API docs).
# Run from the project root: ./scripts/build_docs.sh
#
# Output ends up in website/build/. The Sphinx API reference is rendered
# into website/static/api/ and served at /api/ by Docusaurus.
set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 [-h]

Build Captum documentation (Docusaurus v3 + Sphinx API docs).

  -h   Show this help and exit.

The script always produces a static build under website/build/.
EOF
  exit 1
}

while getopts 'h' flag; do
  case "${flag}" in
    h) usage ;;
    *) usage ;;
  esac
done

cd "$(dirname "$0")/.."

echo "-----------------------------------"
echo "Generating API reference via Sphinx"
echo "-----------------------------------"
(cd sphinx && make html)
python3 scripts/parse_sphinx.py        # → website/static/api/

echo "-----------------------------------"
echo "Converting tutorials to MDX"
echo "-----------------------------------"
python3 scripts/convert_ipynb_to_mdx.py --clean   # → docs/tutorials/

echo "-----------------------------------"
echo "Installing website dependencies"
echo "-----------------------------------"
cd website
yarn install --frozen-lockfile --ignore-scripts

echo "-----------------------------------"
echo "Building static site"
echo "-----------------------------------"
yarn build
