#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json

from bs4 import BeautifulSoup

BASE_URL = "/"


def updateVersionHTML(base_path, base_url=BASE_URL):
    with open(base_path + "/captum-master/website/_versions.json", "rb") as infile:
        versions = json.loads(infile.read())

    with open(base_path + "/new-site/versions.html", "rb") as infile:
        html = infile.read()

    versions.append("latest")

    def prepend_url(a_tag, base_url, version):
        href = a_tag.attrs["href"]
        if href.startswith("https://") or href.startswith("http://"):
            return href
        else:
            return "{base_url}versions/{version}{original_url}".format(
                base_url=base_url, version=version, original_url=href
            )

    for v in versions:
        soup = BeautifulSoup(html, "html.parser")

        # title
        title_link = soup.find("header").find("a")
        title_link.attrs["href"] = prepend_url(title_link, base_url, v)

        # nav
        nav_links = soup.find("nav").findAll("a")
        for link in nav_links:
            link.attrs["href"] = prepend_url(link, base_url, v)

        # version link
        t = soup.find("h2", {"class": "headerTitleWithLogo"}).find_next("a")
        t.string = v
        t.attrs["href"] = prepend_url(t, base_url, v)

        # output files
        with open(
            base_path + "/new-site/versions/{}/versions.html".format(v), "w"
        ) as outfile:
            outfile.write(str(soup))
        with open(
            base_path + "/new-site/versions/{}/en/versions.html".format(v), "w"
        ) as outfile:
            outfile.write(str(soup))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Fix links in version.html files for Docusaurus site."
            "This is used to ensure that the versions.js for older "
            "versions in versions subdirectory are up-to-date and "
            "will have a way to navigate back to newer versions."
        )
    )
    parser.add_argument(
        "-p",
        "--base_path",
        metavar="path",
        required=True,
        help="Input directory for rolling out new version of site.",
    )
    args = parser.parse_args()
    updateVersionHTML(args.base_path)
