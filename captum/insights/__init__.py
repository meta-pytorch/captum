#!/usr/bin/env python3
# pyre-strict

"""
Captum Insights was retired after v0.8.0 and is no longer supported.

This stub module exists to provide a clear error message to users who
attempt to import the old ``captum.insights`` API (e.g.,
``AttributionVisualizer``) instead of an opaque ``ModuleNotFoundError``.

For visualization, use the supported matplotlib-based API::

    from captum.attr import visualization as viz

    # Image attributions
    fig, ax = viz.visualize_image_attr(attr, original_image)

    # Text attributions (renders HTML in Jupyter notebooks)
    viz.visualize_text(datarecords)

    # Timeseries attributions
    fig, ax = viz.visualize_timeseries_attr(attr, data)

See https://captum.ai/api/utilities for full documentation.
"""

from __future__ import annotations

from typing import NoReturn

_RETIRED_MESSAGE = (
    "Captum Insights was retired after v0.8.0 and is no longer available. "
    "The AttributionVisualizer class and its render()/serve() methods have "
    "been removed.\n\n"
    "For visualization, use the matplotlib-based API instead:\n\n"
    "    from captum.attr import visualization as viz\n\n"
    "    # Image attributions\n"
    "    fig, ax = viz.visualize_image_attr(attr, original_image)\n\n"
    "    # Text attributions (HTML in Jupyter notebooks)\n"
    "    viz.visualize_text(datarecords)\n\n"
    "    # Timeseries attributions\n"
    "    fig, ax = viz.visualize_timeseries_attr(attr, data)\n\n"
    "To view the old Captum Insights code, check out the v0.8.0 tag:\n"
    "    git checkout v0.8.0\n\n"
    "See https://captum.ai/api/utilities for documentation."
)


class _RetiredClass:
    """Placeholder that raises an informative error on instantiation."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ImportError(_RETIRED_MESSAGE)


AttributionVisualizer = _RetiredClass
Batch = _RetiredClass


def __getattr__(name: str) -> NoReturn:
    raise ImportError(
        f"cannot import name '{name}' from 'captum.insights'. " + _RETIRED_MESSAGE
    )
