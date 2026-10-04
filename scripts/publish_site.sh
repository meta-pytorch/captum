#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

usage() {
  echo "Usage: $0 [-b]"
  echo ""
  echo "Build and push updated Captum site. Will either update latest or bump stable version."
  echo ""
  echo "  -v    Build site for new library version. If not specified, will update latest."
  echo ""
  exit 1
}

VERSION=false

while getopts 'hv' option; do
  case "${option}" in
    h)
      usage
      ;;
    v)
      VERSION=${OPTARG}
      ;;
    *)
      usage
      ;;
  esac
done

# Command to strip out Algolia (search functionality) form siteConfig.js
# Algolia only indexes stable build, so we'll remove from older versions
REMOVE_ALGOLIA_CMD="import os, re; "
REMOVE_ALGOLIA_CMD+="c = open('siteConfig.js', 'r').read(); "
REMOVE_ALGOLIA_CMD+="out = re.sub('algolia: \{.+\},', '', c, flags=re.DOTALL); "
REMOVE_ALGOLIA_CMD+="f = open('siteConfig.js', 'w'); "
REMOVE_ALGOLIA_CMD+="f.write(out); "
REMOVE_ALGOLIA_CMD+="f.close(); "

# Current directory (needed for cleanup later)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make temporary directory
WORK_DIR=$(mktemp -d)
cd "${WORK_DIR}" || exit

# Clone both master & gh-pages branches
git clone git@github.com:pytorch/captum.git captum-master
git clone --branch gh-pages git@github.com:pytorch/captum.git captum-gh-pages

# A few notes about the script below:
# * Docusaurus versioning was designed to *only* version the markdown
#   files in the docs/ subdirectory. We are repurposing parts of Docusaurus
#   versioning, but snapshotting the entire site. Versions of the site are
#   stored in the versions/ subdirectory on gh-pages:
#
#   --gh-pages/
#     |-- api/
#     |-- css/
#     |-- docs/
#     |   ...
#     |-- versions/
#     |   |-- 1.0.1/
#     |   |-- 1.0.2/
#     |   |   ...
#     |   |-- latest/
#     |   ..
#     |-- versions.html
#
# * The stable version is in the top-level directory. It is also
#   placed into the versions/ subdirectory so that it does not need to
#   be built again when the version is augmented.
# * We want to support serving / building the Docusaurus site locally
#   without any versions. This means that we have to keep versions.js
#   outside of the website/ subdirectory.
# * We do not want to have a tracked file that contains all versions of
#   the site or the latest version. Instead, we determine this at runtime.
#   We use what's on gh-pages in the versions subdirectory as the
#   source of truth for available versions and use the latest tag on
#   the master branch as the source of truth for the latest version.

if [[ $VERSION == false ]]; then
  echo "-----------------------------------------"
  echo "Updating latest (master) version of site "
  echo "-----------------------------------------"

  # Populate _versions.json from existing versions; this is used
  # by versions.js & needed to build the site (note that we don't actually
  # use versions.js for latest build, but we do need versions.js
  # in website/pages in order to use docusaurus-versions)
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('captum-gh-pages/versions') if v != 'latest' and not v.startswith('.')]; "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > captum-master/website/_versions.json

  # Move versions.js to website subdirectory.
  # This is the page you see when click on version in navbar.
  cp captum-master/scripts/versions.js captum-master/website/pages/en/versions.js
  cd captum-master/website || exit

  # Build site, tagged with "latest" version; baseUrl set to /versions/latest/
  yarn
  yarn run version latest
  sed -i '' "s/baseUrl = '\/'/baseUrl = '\/versions\/latest\/'/g" siteConfig.js

  # disable search for non-stable version (can't use sed b/c of newline)
  python3 -c "$REMOVE_ALGOLIA_CMD"

  cd .. || exit
  ./scripts/build_docs.sh -b
  rm -rf website/build/captum/docs/next  # don't need this

  # Move built site to gh-pages (but keep old versions.js)
  cd "${WORK_DIR}" || exit
  cp captum-gh-pages/versions/latest/versions.html versions.html
  rm -rf captum-gh-pages/versions/latest
  mv captum-master/website/build/captum captum-gh-pages/versions/latest
  # versions.html goes both in top-level and under en/ (default language)
  cp versions.html captum-gh-pages/versions/latest/versions.html
  cp versions.html captum-gh-pages/versions/latest/en/versions.html

  # Push changes to gh-pages
  cd captum-gh-pages || exit
  git add .
  git commit -m 'Update latest version of site'
  git push

else
  echo "-----------------------------------------"
  echo "Building new version ($VERSION) of site "
  echo "-----------------------------------------"

  # Checkout master branch with specified tag
  cd captum-master || exit
  git fetch --tags
  git checkout "${VERSION}"

  # Populate _versions.json from existing versions; this contains a list
  # of versions present in gh-pages (excluding latest). This is then used
  # to populate versions.js (which forms the page that people see when they
  # click on version number in navbar).
  # Note that this script doesn't allow building a version of the site that
  # is already on gh-pages.
  CMD="import os, json; "
  CMD+="vs = [v for v in os.listdir('../captum-gh-pages/versions') if v != 'latest' and not v.startswith('.')]; "
  CMD+="assert '${VERSION}' not in vs, '${VERSION} is already on gh-pages.'; "
  CMD+="vs.append('${VERSION}'); "
  CMD+="print(json.dumps(vs))"
  python3 -c "$CMD" > website/_versions.json

  cp scripts/versions.js website/pages/en/versions.js

  # Set Docusaurus version
  cd website || exit
  yarn
  yarn run version stable

  # Build new version of site (this will be stable, default version)
  cd .. || exit
  ./scripts/build_docs.sh -b

  # Move built site to new folder (new-site) & carry over old versions
  # from existing gh-pages
  cd "${WORK_DIR}" || exit
  rm -rf captum-master/website/build/captum/docs/next  # don't need this
  mv captum-master/website/build/captum new-site
  mv captum-gh-pages/versions new-site/versions

  # Build new version of site (to be placed in versions/$VERSION/)
  # the only thing that changes here is the baseUrl (for nav purposes)
  # we build this now so that in the future, we can just bump version and not move
  # previous stable to versions
  cd captum-master/website || exit
  sed -i '' "s/baseUrl = '\/'/baseUrl = '\/versions\/${VERSION}\/'/g" siteConfig.js

  # disable search for non-stable version (can't use sed b/c of newline)
  python3 -c "$REMOVE_ALGOLIA_CMD"

  yarn run version "${VERSION}"
  cd .. || exit
  ./scripts/build_docs.sh -b
  rm -rf website/build/captum/docs/next  # don't need this
  rm -rf website/build/captum/docs/stable  # or this
  mv website/build/captum "../new-site/versions/${VERSION}"

  # Need to run script to update versions.js for previous versions in
  # new-site/versions with the newly built versions.js. Otherwise,
  # the versions.js for older versions in versions subdirectory
  # won't be up-to-date and will not have a way to navigate back to
  # newer versions. This is the only part of the old versions that
  # needs to be updated when a new version is built.
  cd "${WORK_DIR}" || exit
  python3 captum-master/scripts/update_versions_html.py -p "${WORK_DIR}"

  # Init as Git repo and push to gh-pages
  cd new-site || exit
  git init
  git add --all
  git commit -m "Publish version ${VERSION} of site"
  git push --force "https://github.com/pytorch/captum" master:gh-pages

fi

# Clean up
cd "${SCRIPT_DIR}" || exit
rm -rf "${WORK_DIR}"
