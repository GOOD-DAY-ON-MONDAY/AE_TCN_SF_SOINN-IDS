"""Runner-side logic for the model-comparison Runner.

Owns model discovery, dataset registry, contract validation (added in
ticket 04), and nearest-match error messages. Kept separate from
``src/train.py`` (the CLI layer) per CONTEXT.md.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from src.utils.config import REPO_ROOT

VALID_DATASETS = ("netml2020", "cicids2017")

MODELS_ROOT = REPO_ROOT / "models"

# A directory counts as a runnable model if it contains the fixed entry file.
ENTRY_FILE = "main.py"


class RunnerError(Exception):
    """Fast-fail Runner error with a suggest-the-fix message."""


def nearest_match(candidate: str, valid: list[str] | tuple[str, ...]) -> str | None:
    """Return the closest valid name to ``candidate``, or None if nothing is close."""
    matches = difflib.get_close_matches(candidate, list(valid), n=1, cutoff=0.4)
    return matches[0] if matches else None


def discover_models(models_root: str | Path | None = None) -> list[Path]:
    """Return runnable model directories (those containing ``main.py``).

    Searches two levels deep under ``models/``: ``models/<family>/<model>/``.
    """
    root = Path(models_root) if models_root is not None else MODELS_ROOT
    if not root.is_dir():
        return []
    found: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        # Family dir: look one level down for model dirs.
        for sub in sorted(entry.iterdir()):
            if sub.is_dir() and (sub / ENTRY_FILE).is_file():
                found.append(sub)
        # Also allow models placed directly under models/ (family-less).
        if (entry / ENTRY_FILE).is_file():
            found.append(entry)
    return found


def resolve_model(model_path: str | Path) -> Path:
    """Validate that ``model_path`` is a directory with an entry file.

    The directory path is the model's identity/display name. A typo'd or
    nonexistent path fails fast with the nearest matching model directory
    suggested.
    """
    path = Path(model_path)
    if path.is_dir() and (path / ENTRY_FILE).is_file():
        return path.resolve()

    # Search both the repo-wide models tree and the neighborhood of the
    # given path (so typos like `.../bilstmm` can be matched to `.../bilstm`).
    known = [str(p) for p in discover_models()]
    search_root = path.parent.parent if path.parent.parent.is_dir() else None
    if search_root is not None:
        known += [str(p) for p in discover_models(search_root)]
    seen: set[str] = set()
    known = [k for k in known if not (k in seen or seen.add(k))]
    hint = nearest_match(str(path), known)
    msg = f"Unknown model directory: '{path}'"
    if hint:
        msg += f"\nDid you mean '{hint}'?"
    if known:
        msg += "\nAvailable models:\n  " + "\n  ".join(known)
    raise RunnerError(msg)


def resolve_dataset(dataset: str) -> str:
    """Validate the dataset name, suggesting the nearest valid name on typo."""
    if dataset in VALID_DATASETS:
        return dataset
    hint = nearest_match(dataset, VALID_DATASETS)
    msg = f"Unknown dataset: '{dataset}' (valid: {', '.join(VALID_DATASETS)})"
    if hint:
        msg += f"\nDid you mean '{hint}'?"
    raise RunnerError(msg)
