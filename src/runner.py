"""Runner-side logic for the model-comparison Runner.

Owns model discovery, dataset registry, model loading/contract
validation (ticket 04), and nearest-match error messages. Kept
separate from ``src/train.py`` (the CLI layer) per CONTEXT.md.

Model contract (CONTEXT.md Round 1 Q1 / Round 2 Q1-Q4):
- ``main.py`` in the model dir exposes ``create_model(cfg)`` where
  ``cfg`` is the fully merged config namespace.
- Required methods: ``fit(X, y, X_val=None, y_val=None)`` (re-callable
  on the same object without resetting it), ``predict(X)`` returning
  hard integer class indices (pure read), ``save(path)``.
- ``supports_incremental = True`` additionally requires
  ``partial_fit(X, y)`` and ``predict_and_adapt(X)`` (contract-only in
  v1 — demo use, never the scientific loop).
"""

from __future__ import annotations

import difflib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

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


# ---------------------------------------------------------------------------
# Model loading + contract validation (ticket 04)
# ---------------------------------------------------------------------------


_REQUIRED_METHODS = ("fit", "predict", "save")
_INCREMENTAL_METHODS = ("partial_fit", "predict_and_adapt")


class ContractViolation(RunnerError):
    """Model does not satisfy the Runner contract; message names the fix."""


def load_model(model_dir: str | Path, cfg: Any) -> Any:
    """Load ``create_model(cfg)`` from ``<model_dir>/main.py`` and validate it."""
    model_dir = Path(model_dir)
    entry = model_dir / ENTRY_FILE
    if not entry.is_file():
        raise RunnerError(
            f"Model directory '{model_dir}' has no '{ENTRY_FILE}' entry file.\n"
            f"Fix: add '{ENTRY_FILE}' exposing create_model(cfg)."
        )
    spec = importlib.util.spec_from_file_location(
        f"runner_model_{abs(hash(str(model_dir)))}", entry
    )
    if spec is None or spec.loader is None:
        raise RunnerError(f"Could not import model entry file: '{entry}'")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "create_model"):
        raise ContractViolation(
            f"'{entry}' does not expose create_model(cfg).\n"
            "Fix: define create_model(cfg) returning the model object."
        )
    model = module.create_model(cfg)
    return validate_model(model, model_dir)


def validate_model(model: Any, origin: str = "<model>") -> Any:
    """Fail fast if ``model`` violates the contract; return it unchanged."""
    for name in _REQUIRED_METHODS:
        if not callable(getattr(model, name, None)):
            raise ContractViolation(
                f"Model '{origin}' is missing required method '{name}'.\n"
                f"Fix: implement {name}() per the Runner contract "
                f"(required: {', '.join(_REQUIRED_METHODS)})."
            )
    if getattr(model, "supports_incremental", False):
        for name in _INCREMENTAL_METHODS:
            if not callable(getattr(model, name, None)):
                raise ContractViolation(
                    f"Incremental model '{origin}' declares "
                    f"supports_incremental=True but is missing '{name}'.\n"
                    f"Fix: implement {name}(), or remove the "
                    "supports_incremental declaration."
                )
    return model


def check_predict_contract(model: Any, X: Any) -> None:
    """Post-fit runtime checks: hard integer indices + pure read.

    Called by the run loop (ticket 05) on the val/test arrays before
    metrics are computed.
    """
    out1 = model.predict(X)
    out2 = model.predict(X)
    arr1 = np.asarray(out1)
    if arr1.ndim != 1:
        raise ContractViolation(
            f"predict() returned shape {arr1.shape}; expected 1-D hard "
            "class indices (no probabilities, no 2-D outputs)."
        )
    if arr1.size > 0 and not np.issubdtype(arr1.dtype, np.integer):
        raise ContractViolation(
            f"predict() returned dtype {arr1.dtype}; the contract requires "
            "hard integer class indices. Convert inside the model "
            "(e.g. argmax over probabilities)."
        )
    if not np.array_equal(arr1, np.asarray(out2)):
        raise ContractViolation(
            "predict() is not a pure read: repeated calls with the same "
            "input returned different outputs. predict() must have no "
            "side effects (adaptation belongs in partial_fit / "
            "predict_and_adapt)."
        )
    return None
