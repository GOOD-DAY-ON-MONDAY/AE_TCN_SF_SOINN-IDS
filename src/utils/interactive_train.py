"""Interactive ``train`` form for the builder shell.

When the user types ``train`` with no flags inside the builder REPL, this
module collects the missing arguments through an InquirerPy form:

* ``inquirer.select`` for a scrollable model menu (scans ``models/`` for
  directories containing a ``main.py``),
* ``inquirer.select`` for the dataset (``netml2020``, ``cicids2017``),
* ``inquirer.text`` for seeds (default ``42``) and an optional row limit
  (blank = full dataset).

The result is a token list equivalent to a fully-flagged
``train --model ... --dataset ...`` command line, so the shell routes it
through the *same* argparse parser as the one-shot CLI (no drift).
"""

from __future__ import annotations

import sys
from pathlib import Path

from rich.console import Console

_MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"
_DATASETS = ("netml2020", "cicids2017")

# Claude Code-inspired palette (mirrors src/utils/shell.py).
_TEAL = "#56B6C2"
_AMBER = "#E5C07B"


def discover_models(root: Path = _MODELS_ROOT) -> list[Path]:
    """Return directories under ``root`` that contain a ``main.py``.

    Delegates to the Runner-owned seam and relativizes for ``--model`` use.
    """
    from src.runner import discover_models as _canonical

    repo_root = Path(root).parent
    found: list[Path] = []
    for model_dir in _canonical(root):
        if "__pycache__" in model_dir.parts or any(
            part.startswith(".") for part in model_dir.parts
        ):
            continue
        try:
            found.append(model_dir.relative_to(repo_root))
        except ValueError:  # pragma: no cover - root outside repo
            found.append(model_dir)
    return found


def _parse_seeds(raw: str) -> list[int] | None:
    """Parse a comma/space separated seed string; ``None`` when empty/invalid."""
    raw = raw.strip().rstrip(",")
    if not raw:
        return None
    try:
        return [int(part) for part in raw.replace(",", " ").split()]
    except ValueError:
        return None


def _parse_limit(raw: str) -> int | None:
    """Parse the row-limit answer; blank means "full dataset" (``None``)."""
    raw = raw.strip()
    if not raw:
        return None
    try:
        limit = int(raw)
    except ValueError:
        return None
    return limit if limit > 0 else None


def run_interactive_train(console: Console) -> list[str] | None:
    """Run the interactive ``train`` form.

    Returns the equivalent CLI token list (e.g.
    ``["train", "--model", "models/baselines/SVM", "--dataset", "netml2020",
    "--seeds", "42", "--limit", "500"]``), or ``None`` if the user cancelled,
    no models exist, or the session is not interactive.
    """
    # Never block (or explode) when stdin is not a terminal: scripts, tests
    # and piped sessions fall back to the explicit-flag path.
    if not sys.stdin.isatty():
        console.print(
            f"[{_AMBER}]note:[/{_AMBER}] interactive form needs a TTY — "
            f"use [bold]train --model <dir> --dataset <ds>[/bold] instead."
        )
        return None

    models = discover_models()
    if not models:
        console.print(
            "[#E5C07B]warning:[/#E5C07B] no runnable models found under "
            "[bold]models/[/bold] (directories with a main.py)."
        )
        return None

    try:
        from InquirerPy import inquirer
        from InquirerPy.base.control import Choice
    except ImportError:
        console.print(
            "[#E5C07B]warning:[/#E5C07B] InquirerPy is not installed; "
            "use [bold]train --model <dir> --dataset <ds>[/bold] instead."
        )
        return None

    try:
        model = inquirer.select(
            message="Model:",
            choices=[Choice(value=str(m), name=str(m)) for m in models],
            qmark="",
            amark="",
            pointer="›",  # noqa: RUF001
            instruction="(↑/k down/↓ j to move, enter to select)",
        ).execute()

        dataset = inquirer.select(
            message="Dataset:",
            choices=list(_DATASETS),
            default=_DATASETS[0],
            qmark="",
            amark="",
            pointer="›",  # noqa: RUF001
            instruction="(↑/k down/↓ j to move, enter to select)",
        ).execute()

        seeds_raw: str = inquirer.text(
            message="Seeds (comma/space separated):",
            default="42",
            qmark="",
            amark="",
            validate=lambda text: (
                bool(_parse_seeds(text))
                or "Enter one or more integers (e.g. 42 or 1, 2, 3)."
            ),
        ).execute()

        limit_raw: str = inquirer.text(
            message="Row limit (leave blank for full data):",
            default="",
            qmark="",
            amark="",
            validate=lambda text: (
                text.strip() == ""
                or _parse_limit(text) is not None
                or "Enter a positive integer, or leave blank for full data."
            ),
        ).execute()
    except KeyboardInterrupt:
        console.print("[dim](form cancelled — command not run)[/dim]")
        return None

    tokens = ["train", "--model", model, "--dataset", dataset]
    seeds = _parse_seeds(seeds_raw)
    if seeds:
        tokens += ["--seeds", *[str(s) for s in seeds]]
    limit = _parse_limit(limit_raw)
    if limit is not None:
        tokens += ["--limit", str(limit)]
        console.print(
            f"[#E5C07B]smoke-test:[/#E5C07B] --limit {limit} caps the training "
            "data — never use for real experiments."
        )
    return tokens
