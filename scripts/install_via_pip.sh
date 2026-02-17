#!/bin/bash

set -e

PYTORCH_NIGHTLY=false
DEPLOY=false
CHOSEN_TORCH_VERSION=-1
CHOSEN_TRANSFORMERS_VERSION=-1

while getopts 'ndfv:t:' flag; do
  case "${flag}" in
    n) PYTORCH_NIGHTLY=true ;;
    d) DEPLOY=true ;;
    f) FRAMEWORKS=true ;;
    v) CHOSEN_TORCH_VERSION=${OPTARG};;
    t) CHOSEN_TRANSFORMERS_VERSION=${OPTARG};;
    *) echo "usage: $0 [-n] [-d] [-f] [-v version] [-t transformers_version]" >&2
       exit 1 ;;
    esac
  done

# NOTE: Only Debian variants are supported, since this script is only
# used by our tests on GitHub Actions. In the future we might generalize,
# but users should hopefully be using conda installs.

# install nodejs and yarn for insights build
sudo apt-get update
sudo apt install apt-transport-https ca-certificates
curl -sL https://deb.nodesource.com/setup_20.x | sudo -E bash -
curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | sudo apt-key add -
echo "deb https://dl.yarnpkg.com/debian/ stable main" | sudo tee /etc/apt/sources.list.d/yarn.list
sudo apt update
sudo apt install nodejs
sudo apt install yarn

# yarn needs terminal info
export TERM=xterm

# Remove all items from pip cache to avoid hash mismatch
pip cache purge

# upgrade pip
pip install --upgrade pip --progress-bar off

# install captum with dev deps
echo "[install_via_pip] pip install captum dev"
pip install -e .[dev] --progress-bar off

# echo "[install_via_pip] captum setup"
# BUILD_INSIGHTS=1 python setup.py develop

echo "[install_via_pip] pip install torch"
# install pytorch nightly if asked for
if [[ $PYTORCH_NIGHTLY == true ]]; then
  pip install --upgrade --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --progress-bar off
else
  # If no version is specified, upgrade to the latest release.
  if [[ $CHOSEN_TORCH_VERSION == -1 ]]; then
    pip install --upgrade torch --progress-bar off
  else
    pip install torch=="$CHOSEN_TORCH_VERSION" --progress-bar off
  fi
fi

echo "[install_via_pip] pip install transformers"
# install appropriate transformers version
# If no version is specified, upgrade to the latest release.
if [[ $CHOSEN_TRANSFORMERS_VERSION == -1 ]]; then
  pip install --upgrade transformers --progress-bar off
else
  pip install transformers=="$CHOSEN_TRANSFORMERS_VERSION" --progress-bar off
fi

echo "[install_via_pip] pip install deploy deps"
# install deployment bits if asked for
if [[ $DEPLOY == true ]]; then
  pip install beautifulsoup4 ipython nbconvert==5.6.1 --progress-bar off
fi
