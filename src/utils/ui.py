"""CLI presentation helpers: colour-with-meaning, progress bars, plan header,
and consistent error styling.

Rich is used for the progress bar, spinner (status), and summary tables.
A single module-level ``console`` instance is shared across the pipeline so
Rich's output never interleaves with plain ``print()`` calls on the same
stream.  Rich auto-detects non-TTY (piped / redirected output) and falls back
to plain sequential text — no extra configuration needed.

The legacy ``yellow()``, ``dim()``, ``bold()``, ``green()``, ``format_error()``
and ``print_plan()`` helpers are kept unchanged because they are used in
contexts that intentionally bypass Rich (e.g. stderr error formatting).
"""

from __future__ import annotations

import math
import os
import sys
import threading
import itertools
from collections.abc import Callable
from typing import Any, Self

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

# ---------------------------------------------------------------------------
# Shared Console singleton
# ---------------------------------------------------------------------------

# force_terminal=None lets Rich decide based on the real TTY state, so piped
# output falls back to plain text automatically.
console = Console()


def create_progress(c: Console = console) -> Progress:
    """Return a unified Rich Progress instance configured for the pipeline."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=60),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=c,
        transient=False,
    )

# ---------------------------------------------------------------------------
# Legacy ANSI helpers (kept for backward compat — used in format_error, etc.)
# ---------------------------------------------------------------------------

_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def color_enabled() -> bool:
    """Return True when ANSI color should be used (TTY and NO_COLOR unset)."""
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _paint(code: str, msg: str) -> str:
    return f"{code}{msg}{_RESET}" if color_enabled() else msg


def yellow(msg: str) -> str:
    """Yellow = warning semantics: skip/overwrite notices."""
    return _paint(_YELLOW, msg)


def dim(msg: str) -> str:
    """Wrap ``msg`` in dim styling (returns unchanged if color is off)."""
    return _paint(_DIM, msg)


def bold(msg: str) -> str:
    """Wrap ``msg`` in bold styling (returns unchanged if color is off)."""
    return _paint(_BOLD, msg)


def green(msg: str) -> str:
    """Wrap ``msg`` in green styling (returns unchanged if color is off)."""
    return _paint(_GREEN, msg)


def format_error(msg: str) -> str:
    """Consistent suggest-the-fix error styling."""
    lines = msg.splitlines()
    return "\n".join([bold("error: ") + lines[0], *lines[1:]])


def print_plan(
    model_dir: str,
    dataset: str,
    seeds: list[int],
    config_layers: list[tuple[str, str]],
) -> None:
    """Print the pre-run plan header before any compute (abortable)."""
    print(bold("=== Plan ==="))
    print(f"  model:    {model_dir}")
    print(f"  dataset:  {dataset}")
    print(f"  seeds:    {', '.join(map(str, seeds))}")
    print("  config layers (lowest -> highest precedence):")
    for name, source in config_layers:
        status = source if source else dim("absent")
        print(f"    1. {name}: {status}")
    print()


# ---------------------------------------------------------------------------
# Rich progress: eval bar
# ---------------------------------------------------------------------------

def chunked_predict_with_progress(
    model: Any,
    X: Any,
    label: str = "evaluating",
    chunks: int = 20,
    progress: Progress | None = None,
) -> Any:
    """predict() over chunks with a Rich progress bar.

    predict() is a pure read (contract), so chunking is safe; results are
    concatenated back into one array.

    When ``n < 5``, skips the progress bar entirely — prints a single
    "evaluating N flows..." line, runs predict, then prints "done".
    Rich auto-degrades to plain text when stdout is not a TTY.
    """
    import numpy as np

    n = len(X)
    if n == 0:
        return model.predict(X)

    # Very small runs: a bar just adds noise.
    if n < 5:
        console.print(f"[dim]{label} {n} flow(s)…[/dim]")
        result = model.predict(X)
        console.print(f"[dim]{label} done[/dim]")
        return result

    size = max(1, math.ceil(n / max(chunks, 1)))
    parts: list[Any] = []

    if progress is not None:
        task = progress.add_task(f"[bold]{label}[/bold]", total=chunks)
        for i in range(0, n, size):
            parts.append(model.predict(X[i : i + size]))
            progress.advance(task)
        progress.update(
            task,
            completed=chunks,
            total=chunks,
            description=f"[green]{label} completed[/green]",
        )
        return np.concatenate(parts)

    with create_progress() as p:
        task = p.add_task(f"[bold]{label}[/bold]", total=chunks)
        for i in range(0, n, size):
            parts.append(model.predict(X[i : i + size]))
            p.advance(task)
        p.update(
            task,
            completed=chunks,
            total=chunks,
            description=f"[green]{label} completed[/green]",
        )
        return np.concatenate(parts)


# ---------------------------------------------------------------------------
# Rich progress: opaque phases (fit)
# ---------------------------------------------------------------------------

def phase(
    label: str,
    fn: Callable[[], Any],
    progress: Progress | None = None,
) -> Any:
    """Run an opaque callable behind a Rich indeterminate progress task."""
    import time

    t0 = time.perf_counter()
    if progress is not None:
        task = progress.add_task(f"[bold]{label}[/bold]", total=None)
        try:
            res = fn()
            elapsed = time.perf_counter() - t0
            progress.update(
                task,
                completed=1,
                total=1,
                description=f"[green]{label} completed[/green] [dim]({elapsed:.2f}s)[/dim]",
            )
            return res
        except Exception:
            progress.update(
                task,
                completed=1,
                total=1,
                description=f"[red]{label} failed[/red]",
            )
            raise

    with create_progress() as p:
        task = p.add_task(f"[bold]{label}[/bold]", total=None)
        try:
            res = fn()
            elapsed = time.perf_counter() - t0
            p.update(
                task,
                completed=1,
                total=1,
                description=f"[green]{label} completed[/green] [dim]({elapsed:.2f}s)[/dim]",
            )
            return res
        except Exception:
            p.update(
                task,
                completed=1,
                total=1,
                description=f"[red]{label} failed[/red]",
            )
            raise


# ---------------------------------------------------------------------------
# Deprecated legacy classes — kept so existing imports don't break.
# Tests that assert on their exact stderr output are updated in test_cli_ux.py.
# ---------------------------------------------------------------------------

class ProgressBar:
    """Deprecated: thin wrapper kept for import compat. Use rich Progress directly."""

    def __init__(self, total: int, label: str = "", width: int = 24) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self._n = 0
        self._is_tty = sys.stderr.isatty()

    def update(self, amount: int = 1) -> None:
        self._n = min(self._n + amount, self.total)
        self.render()

    def render(self) -> None:
        frac = self._n / self.total
        filled = int(frac * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        line = f"\r{self.label} [{bar}] {self._n}/{self.total}"
        stream = sys.stderr
        stream.write(line)
        if self._n >= self.total or not self._is_tty:
            stream.write("\n")
        stream.flush()

    def finish(self) -> None:
        self._n = self.total
        self.render()


class Spinner:
    """Deprecated: thin wrapper kept for import compat. Use phase() instead."""

    FRAMES = itertools.cycle("|/-\\")

    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        while not self._stop.wait(0.1):
            frame = next(self.FRAMES)
            sys.stderr.write(f"\r{self.label} {frame}")
            sys.stderr.flush()

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        sys.stderr.write("\r" + " " * (len(self.label) + 3) + "\r")
        sys.stderr.flush()
