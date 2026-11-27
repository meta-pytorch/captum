# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

"""
Convert Captum tutorial Jupyter notebooks into Docusaurus v3 MDX pages.

Ported from BoTorch's ``scripts/convert_ipynb_to_mdx.py``. Differences:

* Captum notebooks live flat under ``tutorials/<id>.ipynb`` (not
  ``tutorials/<id>/<id>.ipynb`` like BoTorch).
* Output is written to ``docs/tutorials/<id>/index.md`` and served at
  ``/docs/tutorials/<id>``; ``docusaurus.config.js`` redirects the v1
  ``/tutorials/<id>`` URLs there.
* ``website/tutorials.json`` may contain nested categories whose items use
  a ``children`` array (the BERT case). We flatten those for conversion;
  the sidebar handles the nesting separately.
* When a notebook is missing from ``website/static/files/``, we copy the
  ``.ipynb`` and emit a ``.py`` via ``jupyter nbconvert --to script`` so
  the download buttons work.

Run from the repository root (``scripts/build_docs.sh`` does this), with the
packages that ``.github/workflows/website-deploy.yml`` installs:

    python3 scripts/convert_ipynb_to_mdx.py --clean
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import mdformat
import nbformat
import pandas as pd
from nbformat.notebooknode import NotebookNode

SCRIPTS_DIR: Path = Path(__file__).resolve().parent
LIB_DIR: Path = SCRIPTS_DIR.parent

WEBSITE_DIR: Path = LIB_DIR.joinpath("website")
DOCS_DIR: Path = LIB_DIR.joinpath("docs")
TUTORIALS_OUT_DIR: Path = DOCS_DIR.joinpath("tutorials")
TUTORIALS_NB_DIR: Path = LIB_DIR.joinpath("tutorials")
STATIC_FILES_DIR: Path = WEBSITE_DIR.joinpath("static", "files")

# Output extension. We deliberately emit ``.md`` (CommonMark) rather than
# ``.mdx`` so we never have to deal with MDX's strict JSX/ESM parsing of
# notebook content (model outputs frequently contain ``<unk>``-style tokens,
# stray braces, and other text that confuses MDX). With
# ``markdown.format: 'detect'`` Docusaurus parses ``.md`` as CommonMark, which
# happily passes through the small bits of HTML/markdown we emit.
OUTPUT_EXT = "md"

ORGANIZATION = "meta-pytorch"
PROJECT = "captum"

# Data display priority. Below lists the priority for displaying data from cell
# outputs. Cells can output many different items, and some will output a
# fallback display, e.g. text/plain if text/html is not working. The below
# priorities help ensure the output in the MDX file shows the best
# representation of the cell output.
priorities: list[str] = [
    "error",
    "text/markdown",
    "image/png",  # matplotlib output.
    "application/vnd.jupyter.widget-view+json",  # tqdm progress bars.
    "text/html",
    "stream",
    "text/plain",
]


def load_tutorial_metadata() -> list[dict[str, str]]:
    """Load and flatten tutorial metadata from ``website/tutorials.json``.

    The JSON groups tutorials by category. A category entry may itself be a
    nested group with a ``children`` list (used for BERT). Both flat and
    nested entries are flattened into a single list of leaf tutorials,
    each with at least ``id`` and ``title``.
    """
    tutorials_json_path = WEBSITE_DIR.joinpath("tutorials.json")
    with tutorials_json_path.open("r") as f:
        tutorials_data: dict[str, list[dict[str, Any]]] = json.load(f)

    leaves: list[dict[str, str]] = []
    for items in tutorials_data.values():
        for item in items:
            if "id" in item:
                leaves.append({"id": item["id"], "title": item["title"]})
            elif "children" in item:
                for child in item["children"]:
                    leaves.append({"id": child["id"], "title": child["title"]})
    return leaves


def load_notebook(path: Path) -> NotebookNode:
    """Load a Jupyter notebook into memory."""
    with path.open("r") as f:
        nb_str = f.read()
        nb = nbformat.reads(nb_str, nbformat.NO_CONVERT)
    return nb


def create_folders(notebook_id: str) -> tuple[Path, Path]:
    """Create the per-tutorial output folder + asset subfolders.

    Returns the folder containing ``index.md`` and the ``assets`` folder.
    """
    folder = TUTORIALS_OUT_DIR / notebook_id
    assets_folder = folder / "assets"
    img_folder = assets_folder / "img"
    plot_data_folder = assets_folder / "plot_data"
    folder.mkdir(parents=True, exist_ok=True)
    img_folder.mkdir(parents=True, exist_ok=True)
    plot_data_folder.mkdir(parents=True, exist_ok=True)
    return folder, assets_folder


def create_frontmatter(nb_metadata: dict[str, str]) -> str:
    """Render the ``---`` frontmatter block for the Markdown file.

    The generated page is gitignored, so "Edit this page" points at the source
    notebook instead of the ``editUrl`` default under ``docs/``.
    """
    title = json.dumps(nb_metadata["title"])
    edit_url = (
        f"https://github.com/{ORGANIZATION}/{PROJECT}/edit/master/"
        f"tutorials/{nb_metadata['id']}.ipynb"
    )
    # JSON strings are valid YAML double-quoted scalars.
    return (
        f"---\ntitle: {title}\nsidebar_label: {title}\n"
        f"custom_edit_url: {json.dumps(edit_url)}\n---\n\n"
    )


def get_current_git_tag() -> str | None:
    """Return the current git tag if HEAD is tagged, else None."""
    try:
        tag = (
            subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                stderr=subprocess.STDOUT,
                cwd=str(LIB_DIR),
            )
            .strip()
            .decode("utf-8")
        )
        return tag
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def create_buttons(nb_metadata: dict[str, str]) -> str:
    """GitHub + Colab links rendered as plain Markdown.

    Captum notebooks live flat under ``tutorials/<id>.ipynb``.
    """
    version = get_current_git_tag() or "master"
    github_path = (
        f"{ORGANIZATION}/{PROJECT}/blob/{version}/tutorials/{nb_metadata['id']}.ipynb"
    )
    github_url = f"https://github.com/{github_path}"
    colab_url = f"https://colab.research.google.com/github/{github_path}"
    download_url = f"/files/{nb_metadata['id']}.ipynb"
    return (
        f"[Open in GitHub]({github_url}) \u00b7 "
        f"[Run in Google Colab]({colab_url}) \u00b7 "
        f"[Download notebook]({download_url})\n\n"
    )


def handle_image_attachments(
    markdown: str,
    attachments: dict[str, dict[str, str]],
) -> str:
    """Inline base64 image attachments referenced via ``attachment:name``."""
    markdown_image_pattern = re.compile(
        r"""!\[([^\]]*)\]\(attachment:(.*?)\s*?(\".*\")?\)"""
    )
    searches = reversed(list(re.finditer(markdown_image_pattern, markdown)))
    for search in searches:
        alt_text, attachment_name, _ = search.groups()
        mime_type, base64 = next(iter(attachments[attachment_name].items()))
        start, end = search.span()
        markdown = (
            markdown[:start]
            + generate_img_base64_md(base64, mime_type, alt_text)
            + markdown[end:]
        )
    return markdown


def handle_image_paths_found_in_markdown(
    markdown: str,
    new_img_dir: Path,
    nb_path: Path,
) -> str:
    """Rewrite local image refs in markdown and copy assets next to the MDX."""
    markdown_image_pattern = re.compile(
        r"""!\[[^\]]*\]\((?!attachment:|https?://)([^\s)]+)(\s+"[^"]*")?\)"""
    )
    searches = list(re.finditer(markdown_image_pattern, markdown))
    if not searches:
        return markdown

    for search in reversed(searches):
        old_path_str, _ = search.groups()
        start, end = search.span(1)

        old_path = Path(old_path_str)
        name = old_path.name.strip()
        new_path = f"assets/img/{name}"

        markdown = markdown[:start] + new_path + markdown[end:]

        if old_path.is_absolute() and old_path.exists():
            old_img_path = old_path
        else:
            old_img_path = (nb_path.parent / old_path).resolve()
        if old_img_path.exists():
            new_img_path = str(new_img_dir / name)
            shutil.copy(str(old_img_path), new_img_path)
        else:
            print(
                f"  WARN: missing image {old_img_path} referenced from {nb_path.name}"
            )

    return markdown


def sanitize_mdx(mdx: str) -> str:
    """Strip constructs that break the CommonMark (``.md``) pages Docusaurus builds."""
    mdx = mdx.replace("<markdowncell>", "")
    mdx = mdx.replace("<TODO>", "")
    mdx = mdx.replace("<br>", "<br />")
    # Remove HTML comments — fine in notebooks, breaks MDX otherwise.
    mdx = re.sub("(<!--.*?-->)", "", mdx, flags=re.DOTALL)

    return mdx


def get_source(cell: NotebookNode) -> str:
    """Return cell source as a single string."""
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return source


def handle_markdown_cell(
    cell: NotebookNode,
    new_img_dir: Path,
    nb_path: Path,
) -> str:
    """Convert a markdown cell to MDX-ready text."""
    markdown = get_source(cell)
    markdown = handle_image_paths_found_in_markdown(markdown, new_img_dir, nb_path)
    markdown = handle_image_attachments(markdown, cell.get("attachments", {}))
    markdown = sanitize_mdx(markdown)
    mdx = mdformat.text(markdown, options={"wrap": 88}, extensions={"myst"})
    return f"{mdx}\n"


def handle_cell_input(cell: NotebookNode, language: str) -> str:
    """Wrap a code cell in a fenced code block."""
    cell_source = get_source(cell)
    return f"```{language}\n{cell_source}\n```\n\n"


def generate_img_base64_md(
    img_as_base64: int | str | NotebookNode,
    mime_type: int | str | NotebookNode,
    alt_text: str = "",
) -> str:
    """Build a markdown image tag from base64-encoded data."""
    return f"![{alt_text}](data:{mime_type};base64,{img_as_base64})"


def handle_image(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Convert embedded images (e.g. matplotlib PNGs) to markdown images."""
    output = []
    for value in values:
        index = int(value["index"])
        mime_type = value["mime_type"]
        img = value["data"]
        output.append((index, f"{generate_img_base64_md(img, mime_type)}\n\n"))
    return output


def handle_markdown(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Format markdown-typed cell outputs."""
    output = []
    for value in values:
        index = int(value["index"])
        markdown = str(value["data"])
        markdown = mdformat.text(markdown, options={"wrap": 88}, extensions={"myst"})
        output.append((index, f"{markdown}\n\n"))
    return output


def handle_pandas(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Convert pandas DataFrame HTML outputs to markdown tables."""
    output = []
    for value in values:
        index = int(value["index"])
        data = str(value["data"])
        if 'class="dataframe"' not in data:
            continue
        try:
            df = pd.read_html(io.StringIO(data), flavor="lxml")
        except ValueError:
            continue
        if len(df) == 0:
            continue
        md_df = df[0]
        for column in md_df.columns:
            if isinstance(column, str) and column.startswith("Unnamed"):
                md_df.rename(columns={column: ""}, inplace=True)
        if isinstance(md_df.index, pd.RangeIndex):
            mdx = md_df.to_markdown(index=False)
        else:
            mdx = md_df.to_markdown()
        if mdx is None:
            continue
        output.append((index, f"\n{sanitize_mdx(mdx)}\n\n"))
    return output


def _flatten_text(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Group adjacent plain-text outputs into ``text`` fenced code blocks."""
    output: list[tuple[int, str]] = []
    adjacent_outputs: list[str] = []
    previous_index: int = -1

    def flush() -> None:
        if not adjacent_outputs:
            return
        body = "\n".join(adjacent_outputs)
        # Use a 4-backtick fence so any 3-backtick content inside the output
        # cannot accidentally close the block.
        output.append((previous_index, f"````text\n{body}\n````\n\n"))

    for value in values:
        index = int(value["index"])
        data = str(value["data"])
        stripped = [line for line in data.splitlines() if line.strip()]
        if not stripped:
            continue
        joined = "\n".join(line for line in data.splitlines() if line)
        # Defensively kill 4-backtick sequences inside the body so the outer
        # fence above stays well-formed.
        joined = joined.replace("````", "\u200b````")
        if previous_index in [-1, index - 1]:
            adjacent_outputs.append(joined)
        else:
            flush()
            adjacent_outputs.clear()
            adjacent_outputs.append(joined)
        previous_index = index
    flush()
    return output


def handle_plain(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Render plain-text cell outputs as a fenced ``text`` code block."""
    return _flatten_text(values)


def handle_tqdm(
    values: list[dict[str, int | str | NotebookNode]],
) -> list[tuple[int, str]]:
    """Render tqdm progress lines as a fenced ``text`` code block."""
    return _flatten_text(values)


CELL_OUTPUTS_TO_PROCESS = dict[
    str,
    list[dict[str, int | str | NotebookNode]],
]


def aggregate_mdx(
    cell_outputs_to_process: CELL_OUTPUTS_TO_PROCESS,
) -> str:
    """Glue per-output-type renderings together preserving cell order."""
    processed_mdx: list[tuple[int, str]] = []
    for key, values in cell_outputs_to_process.items():
        if not values:
            continue
        if key == "image":
            processed_mdx.extend(handle_image(values))
        if key == "markdown":
            processed_mdx.extend(handle_markdown(values))
        if key == "pandas":
            processed_mdx.extend(handle_pandas(values))
        if key == "plain":
            processed_mdx.extend(handle_plain(values))
        if key == "tqdm":
            processed_mdx.extend(handle_tqdm(values))

    processed_mdx = sorted(processed_mdx, key=lambda item: item[0])
    mdx = "\n".join([item[1] for item in processed_mdx])
    return mdx


def prioritize_dtypes(
    cell_outputs: list[NotebookNode],
) -> list[list[str]]:
    """Pick the best mime type for each cell output (highest priority first)."""
    cell_output_dtypes = [
        (
            list(cell_output["data"].keys())
            if "data" in cell_output
            else [cell_output["output_type"]]
        )
        for cell_output in cell_outputs
    ]
    prioritized_cell_output_dtypes = [
        sorted(
            set(dtypes).intersection(set(priorities)),
            key=lambda dtype: priorities.index(dtype),
        )
        for dtypes in cell_output_dtypes
    ]
    return [[str(item) for item in items] for items in prioritized_cell_output_dtypes]


def aggregate_output_types(cell_outputs: list[NotebookNode]) -> CELL_OUTPUTS_TO_PROCESS:
    """Bucket each cell output into the appropriate per-type list."""
    prioritized_cell_output_dtypes = prioritize_dtypes(cell_outputs)

    cell_outputs_to_process: CELL_OUTPUTS_TO_PROCESS = {
        "image": [],
        "markdown": [],
        "pandas": [],
        "plain": [],
        "tqdm": [],
    }
    for i, cell_output in enumerate(cell_outputs):
        if not prioritized_cell_output_dtypes[i]:
            continue
        prioritized_data_dtype = prioritized_cell_output_dtypes[i][0]

        data = (
            cell_output["data"][prioritized_data_dtype]
            if "data" in cell_output
            else (
                cell_output["text"]
                if "text" in cell_output
                else cell_output.get("evalue", "")
            )
        )
        if prioritized_data_dtype.startswith("image"):
            cell_outputs_to_process["image"].append(
                {"index": i, "data": data, "mime_type": prioritized_data_dtype},
            )
        plain_check = prioritized_data_dtype in ["text/plain", "stream", "error"]
        if plain_check:
            if "name" in cell_output and cell_output["name"] == "stderr":
                pass
            if prioritized_data_dtype == "text/plain" and "matplotlib" in data:
                pass
            cell_outputs_to_process["plain"].append({"index": i, "data": data})
        if prioritized_data_dtype == "text/markdown":
            cell_outputs_to_process["markdown"].append({"index": i, "data": data})
        if isinstance(data, str) and "dataframe" in data:
            cell_outputs_to_process["pandas"].append({"index": i, "data": data})
        if prioritized_data_dtype == "application/vnd.jupyter.widget-view+json":
            data = cell_output["data"]["text/plain"]
            cell_outputs_to_process["tqdm"].append({"index": i, "data": data})

    return cell_outputs_to_process


def handle_cell_outputs(cell: NotebookNode) -> str:
    """Convert all outputs of a single code cell to MDX."""
    cell_outputs = cell.get("outputs", [])
    if not cell_outputs:
        return ""
    cell_outputs_to_process = aggregate_output_types(cell_outputs)
    return aggregate_mdx(cell_outputs_to_process)


def handle_code_cell(cell: NotebookNode) -> str:
    """Convert a code cell (input + outputs) to MDX."""
    cell_input_mdx = handle_cell_input(cell, "python")
    cell_output_mdx = handle_cell_outputs(cell)
    return cell_input_mdx + cell_output_mdx


def transform_notebook(path: Path, nb_metadata: dict[str, str]) -> str:
    """Convert a single notebook into ``docs/tutorials/<id>/index.md``."""
    folder, assets_folder = create_folders(nb_metadata["id"])
    img_folder = assets_folder / "img"
    nb = load_notebook(path)
    mdx = ""
    mdx += create_frontmatter(nb_metadata)
    mdx += create_buttons(nb_metadata)
    for cell in nb["cells"]:
        cell_type = cell["cell_type"]
        if cell_type == "markdown":
            mdx += handle_markdown_cell(cell, img_folder, path)
        if cell_type == "code":
            mdx += handle_code_cell(cell)

    save_path = folder / f"index.{OUTPUT_EXT}"
    # Strip trailing whitespace per line + ensure single trailing newline.
    cleaned = "\n".join(line.rstrip() for line in mdx.splitlines()).rstrip() + "\n"
    with save_path.open("w") as f:
        f.write(cleaned)
    return mdx


def clean_up_directories() -> None:
    """Wipe previously generated tutorial folders under docs/tutorials/."""
    if TUTORIALS_OUT_DIR.exists():
        for item in os.scandir(TUTORIALS_OUT_DIR):
            if item.is_dir():
                shutil.rmtree(item.path)


def ensure_static_files(notebook_id: str, src_ipynb: Path) -> None:
    """Make sure ``static/files/<id>.{ipynb,py}`` exist for the download buttons.

    Copies the source ipynb verbatim and uses ``jupyter nbconvert --to script``
    to derive the .py. Skips work that is already done.
    """
    STATIC_FILES_DIR.mkdir(parents=True, exist_ok=True)
    target_ipynb = STATIC_FILES_DIR / f"{notebook_id}.ipynb"
    target_py = STATIC_FILES_DIR / f"{notebook_id}.py"

    if not target_ipynb.exists():
        shutil.copy(str(src_ipynb), str(target_ipynb))
        print(f"  copied {src_ipynb.name} -> {target_ipynb.relative_to(LIB_DIR)}")

    if not target_py.exists():
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "script",
                "--output",
                notebook_id,
                "--output-dir",
                str(STATIC_FILES_DIR),
                str(target_ipynb),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print(f"  generated {target_py.relative_to(LIB_DIR)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert tutorial notebooks into mdx files."
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output from previously converted notebooks.",
    )
    args = parser.parse_args()

    tutorials_metadata = load_tutorial_metadata()
    print("--------------------------------------------")
    print("Converting tutorial notebooks into mdx files")
    print("--------------------------------------------")
    if args.clean:
        clean_up_directories()
    for metadata in tutorials_metadata:
        path = (TUTORIALS_NB_DIR / f"{metadata['id']}.ipynb").resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing tutorial notebook: {path}")
        print(f"{metadata['id']}")
        transform_notebook(path=path, nb_metadata=metadata)
        ensure_static_files(metadata["id"], path)
    print("")


if __name__ == "__main__":
    main()
