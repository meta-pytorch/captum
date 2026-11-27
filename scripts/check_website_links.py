#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check internal links in the built website, resolved as GitHub Pages does.

Every ``href`` and ``src`` in every HTML file under the build directory that
points inside the site must resolve to a file. With ``--urls``, every listed
URL path (for example, every page of the previous site) must resolve too.

Usage: python3 scripts/check_website_links.py website/build [--base-url /captum/]
       [--urls urls.txt]
"""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse


class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)


def resolve(build_dir: Path, url_path: str) -> Path | None:
    """Return the file GitHub Pages would serve for ``url_path``, if any."""
    rel = unquote(url_path).lstrip("/")
    target = build_dir / rel
    if url_path.endswith("/") or not rel:
        candidates = [target / "index.html"]
    else:
        candidates = [
            target,
            target.with_name(target.name + ".html"),
            target / "index.html",
        ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def page_url(build_dir: Path, html_file: Path, base_url: str) -> str:
    rel = html_file.relative_to(build_dir).as_posix()
    return base_url + rel


def check_pages(build_dir: Path, base_url: str) -> list[str]:
    errors = []
    for html_file in sorted(build_dir.rglob("*.html")):
        parser = _LinkParser()
        parser.feed(html_file.read_text(errors="replace"))
        here = page_url(build_dir, html_file, base_url)
        for link in parser.links:
            parsed = urlparse(urljoin(here, link))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            if not parsed.path.startswith(base_url):
                errors.append(f"{here}: {link} is outside the base URL {base_url}")
                continue
            if resolve(build_dir, parsed.path[len(base_url) :]) is None:
                errors.append(f"{here}: broken link {link}")
    return errors


def check_urls(build_dir: Path, urls_file: Path) -> list[str]:
    errors = []
    for line in urls_file.read_text().splitlines():
        path = line.strip()
        if path and resolve(build_dir, path) is None:
            errors.append(f"missing URL {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("--base-url", default="/")
    parser.add_argument("--urls", type=Path, help="File with one URL path per line.")
    args = parser.parse_args()

    base_url = "/" + args.base_url.strip("/") + "/" if args.base_url.strip("/") else "/"
    errors = check_pages(args.build_dir, base_url)
    if args.urls:
        errors += check_urls(args.build_dir, args.urls)
    for error in errors:
        print(error)
    print(f"{len(errors)} problem(s) found")
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
