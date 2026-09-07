"""Data-loading seam for the Runner.

Hands models plain numpy arrays ``X_train, y_train, X_val, y_val, X_test,
y_test``, already label-encoded via ``class_map`` from ``base_config.yaml``.
The Runner imports no deep-learning framework; framework conversion stays
inside model implementations.

Val is a stratified fraction held out from the training set
(``splitting.val_split``, seeded by ``splitting.random_seed``). Test is
externally blocked: both datasets have ``labels_available: false``, so the
seam returns ``None`` for the test halves and evaluation falls back to val.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np

from src.data.loader import encode_label


class DataSeamError(Exception):
    """Raised when the data seam receives inconsistent inputs."""


def stratified_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Hold out a stratified ``val_split`` fraction from (X, y).

    Deterministic for a given seed; preserves class distribution so every
    training class also appears in val.
    """
    if not 0.0 < val_split < 1.0:
        raise DataSeamError(f"val_split must be in (0, 1), got {val_split!r}")
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(y):
        cls_idx = np.flatnonzero(y == cls)
        cls_idx = rng.permutation(cls_idx)
        n_val = max(1, int(len(cls_idx) * val_split)) if len(cls_idx) > 1 else 0
        val_idx.extend(cls_idx[:n_val])
        train_idx.extend(cls_idx[n_val:])
    train_idx_arr = np.sort(np.asarray(train_idx, dtype=int))
    val_idx_arr = np.sort(np.asarray(val_idx, dtype=int))
    return X[train_idx_arr], y[train_idx_arr], X[val_idx_arr], y[val_idx_arr]


def encode_with_class_map(
    labels: list[str] | np.ndarray, class_map: Mapping[str, int]
) -> np.ndarray:
    """Label-encode string labels using ``class_map`` from base_config.yaml.

    Raises DataSeamError naming the first unknown label (e.g. a genuinely new
    zero-day class) instead of KeyError from the generic loader.
    """
    try:
        encoded, _ = encode_label(list(labels), dict(class_map))
    except KeyError as exc:
        raise DataSeamError(
            f"Label {exc} not present in class_map — zero-day/unknown "
            "classes must be handled by the zero-day protocol, not the "
            "training seam."
        ) from exc
    return encoded


def assemble_run_arrays(
    X: np.ndarray,
    y: np.ndarray,
    val_split: float,
    seed: int,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> SimpleNamespace:
    """Build the six-array seam contract for a run.

    Returns a namespace with ``X_train, y_train, X_val, y_val, X_test, y_test``.
    ``X_test``/``y_test`` are ``None`` while the labels_available blocker
    stands; otherwise they pass through unchanged (never split or shuffled).
    """
    X = np.asarray(X)
    y = np.asarray(y).reshape(-1)
    if len(X) != len(y):
        raise DataSeamError(
            f"X/y length mismatch: {len(X)} features vs {len(y)} labels"
        )
    X_train, y_train, X_val, y_val = stratified_val_split(X, y, val_split, seed)

    if (X_test is None) != (y_test is None):
        raise DataSeamError("X_test and y_test must be provided together or both None")
    if X_test is not None and len(np.asarray(X_test)) != len(np.asarray(y_test)):
        raise DataSeamError(
            f"X_test/y_test length mismatch: {len(X_test)} vs {len(y_test)}"
        )

    return SimpleNamespace(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
