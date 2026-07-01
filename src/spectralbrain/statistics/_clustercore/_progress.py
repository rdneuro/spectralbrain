"""Lightweight progress-bar shim for the vendored cluster cores.

Yields an ``advance(step=1)`` callable, matching the API the vendored modules
expect. Uses ``rich`` if available, otherwise is silent. Kept dependency-free so
the clustering cores import cleanly with only numpy/scipy present.
"""

from __future__ import annotations

import contextlib
from typing import Callable, Iterator, Optional


@contextlib.contextmanager
def progress_bar(description: str, total: Optional[int] = None,
                 disable: bool = False) -> Iterator[Callable[[int], None]]:
    """Context manager yielding an ``advance(step=1)`` callable."""
    if disable:
        yield lambda step=1: None
        return
    try:
        from rich.progress import (Progress, BarColumn, TextColumn,
                                    TimeRemainingColumn)
        with Progress(TextColumn("[progress.description]{task.description}"),
                      BarColumn(), TextColumn("{task.completed}/{task.total}"),
                      TimeRemainingColumn(), transient=True) as prog:
            task = prog.add_task(description, total=total)

            def advance(step: int = 1) -> None:
                prog.advance(task, step)

            yield advance
    except Exception:
        # rich unavailable or non-interactive: stay silent.
        yield lambda step=1: None
