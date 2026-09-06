"""CLI presentation helpers (ticket 07): color-with-meaning, progress bars,
plan header, and consistent error styling per CONTEXT.md § CLI Design Language.

Stdlib-only, no external dependency. All colors auto-disable when stdout
is not a TTY or ``NO_COLOR`` is set.
"""

from __future__ import annotations

import itertools
import math
import os
import shutil
import sys
import threading
import time
from typing import Any, Callable

_YELLOW = "\033[33m"
_BLUE = "\033[34m"
_GREEN = "\033[32m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RESET = "\033[0m"


def color_enabled() -> bool:
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _paint(code: str, msg: str) -> str:
    return f"{code}{msg}{_RESET}" if color_enabled() else msg


def yellow(msg: str) -> str:
    """Yellow = warning semantics: skip/overwrite notices."""
    return _paint(_YELLOW, msg)


def dim(msg: str) -> str:
    return _paint(_DIM, msg)


def bold(msg: str) -> str:
    return _paint(_BOLD, msg)


def green(msg: str) -> str:
    return _paint(_GREEN, msg)


def format_error(msg: str) -> str:
    """Consistent suggest-the-fix error styling (ticket 02 language)."""
    lines = msg.splitlines()
    return "\n".join([bold("error: ") + lines[0], *lines[1:]])


def print_plan(
    model_dir: str,
    dataset: str,
    seeds: list[int],
    config_layers: list[tuple[str, str]],
) -> None:
    """Pre-run plan header: what will run, before any compute (abortable)."""
    print(bold("=== Plan ==="))
    print(f"  model:    {model_dir}")
    print(f"  dataset:  {dataset}")
    print(f"  seeds:    {', '.join(map(str, seeds))}")
    print("  config layers (lowest -> highest precedence):")
    for name, source in config_layers:
        status = source if source else dim("absent")
        print(f"    1. {name}: {status}")
    print()


class ProgressBar:
    """Determinate single-line progress bar: [####    ] 50/100."""

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
    """Indeterminate progress indicator for opaque phases (e.g. fit())."""

    FRAMES = itertools.cycle("|/-\\")

    def __init__(self, label: str) -> None:
        self.label = label
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Spinner":
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        while not self._stop.wait(0.1):
            frame = next(self.FRAMES)
            sys.stderr.write(f"\r{self.label} {frame}")
            sys.stderr.flush()

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        sys.stderr.write("\r" + " " * (len(self.label) + 3) + "\r")
        sys.stderr.flush()


def chunked_predict_with_progress(
    model: Any,
    X: Any,
    label: str = "evaluating",
    chunks: int = 20,
) -> Any:
    """predict() over chunks with a determinate progress bar.

    predict() is a pure read (contract), so chunking is safe; results are
    concatenated back into one array.
    """
    import numpy as np

    n = len(X)
    if n == 0:
        return model.predict(X)
    size = max(1, math.ceil(n / max(chunks, 1)))
    bar = ProgressBar(chunks, label=label)
    parts: list[Any] = []
    for i in range(0, n, size):
        parts.append(model.predict(X[i : i + size]))
        bar.update()
    bar.finish()
    return np.concatenate(parts)


def phase(label: str, fn: Callable[[], Any]) -> Any:
    """Run an opaque callable behind an indeterminate spinner."""
    with Spinner(label):
        return fn()


def terminal_width() -> int:
    return shutil.get_terminal_size((80, 24)).columns
