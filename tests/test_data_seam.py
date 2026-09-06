"""Tests for the Runner data seam (ticket 03)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.seam import (
    DataSeamError,
    assemble_run_arrays,
    encode_with_class_map,
    stratified_val_split,
)


def _synthetic(class_counts: dict[int, int], dim: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X, y = [], []
    for cls, n in class_counts.items():
        X.append(rng.normal(size=(n, dim)) + cls)
        y.append(np.full(n, cls))
    return np.vstack(X), np.concatenate(y)


# netml2020-like shape: 21 classes, 121 features; cicids2017-like: 8 classes.
NETML_COUNTS = {c: 10 for c in range(21)}
CICIDS_COUNTS = {c: 12 for c in range(8)}


@pytest.mark.parametrize(
    "counts,dim",
    [(NETML_COUNTS, 121), (CICIDS_COUNTS, 121)],
    ids=["netml2020", "cicids2017"],
)
def test_seam_shapes_end_to_end(counts: dict, dim: int) -> None:
    X, y = _synthetic(counts, dim)
    arrays = assemble_run_arrays(X, y, val_split=0.15, seed=42)
    assert arrays.X_train.shape[1] == dim
    assert len(arrays.X_train) == len(arrays.y_train)
    assert len(arrays.X_val) == len(arrays.y_val)
    assert len(arrays.X_train) + len(arrays.X_val) == len(X)
    # External blocker: test halves are None until labels_available clears.
    assert arrays.X_test is None and arrays.y_test is None


@pytest.mark.parametrize("counts", [NETML_COUNTS, CICIDS_COUNTS])
def test_val_split_is_stratified(counts: dict) -> None:
    X, y = _synthetic(counts, 5)
    _, y_train, _, y_val = stratified_val_split(X, y, val_split=0.15, seed=42)
    train_classes = set(np.unique(y_train))
    val_classes = set(np.unique(y_val))
    assert val_classes == train_classes == set(counts)
    # floor-per-class biases the total upward slightly; keep a loose bound.
    assert len(y_val) == pytest.approx(0.15 * len(y), abs=15)
    assert len(y_val) < 0.25 * len(y)


def test_val_split_deterministic_per_seed() -> None:
    X, y = _synthetic({0: 20, 1: 20}, 3)
    a = stratified_val_split(X, y, 0.2, 42)
    b = stratified_val_split(X, y, 0.2, 42)
    assert all(np.array_equal(x1, x2) for x1, x2 in zip(a, b))


def test_seam_deterministic_per_seed_shapes() -> None:
    X, y = _synthetic(NETML_COUNTS, 121)
    a = assemble_run_arrays(X, y, 0.15, 42)
    b = assemble_run_arrays(X, y, 0.15, 42)
    assert np.array_equal(a.X_train, b.X_train)
    assert np.array_equal(a.X_val, b.X_val)


def test_invalid_val_split_raises() -> None:
    X, y = _synthetic({0: 10}, 3)
    with pytest.raises(DataSeamError, match="val_split"):
        stratified_val_split(X, y, 1.5, 42)


def test_mismatched_lengths_raise() -> None:
    X, y = _synthetic({0: 10}, 3)
    with pytest.raises(DataSeamError, match="length mismatch"):
        assemble_run_arrays(X, y[:-1], 0.15, 42)
    with pytest.raises(DataSeamError, match="together or both None"):
        assemble_run_arrays(X, y, 0.15, 42, X_test=X)
    with pytest.raises(DataSeamError, match="length mismatch"):
        assemble_run_arrays(X, y, 0.15, 42, X_test=X, y_test=y[:-1])


def test_test_data_passed_through_unchanged() -> None:
    X, y = _synthetic({0: 20, 1: 20}, 4)
    Xt = X[:5] + 100.0
    yt = np.array([0, 1, 0, 1, 0])
    arrays = assemble_run_arrays(X, y, 0.15, 42, X_test=Xt, y_test=yt)
    assert np.array_equal(arrays.X_test, Xt)
    assert np.array_equal(arrays.y_test, yt)


def test_encode_with_class_map() -> None:
    labels = ["benign", "DDoS", "benign"]
    encoded = encode_with_class_map(labels, {"DDoS": 0, "DoS": 1, "benign": 2})
    assert encoded.tolist() == [2, 0, 2]


def test_encode_unknown_zero_day_class_raises_specific_error() -> None:
    with pytest.raises(DataSeamError, match="not present in class_map"):
        encode_with_class_map(["benign", "NEW_MALWARE"], {"benign": 0})


def test_runner_never_imports_torch() -> None:
    """Grep guard: the Runner (src/ minus src/data loader) must not import torch."""
    src_root = Path(__file__).resolve().parents[1] / "src"
    offenders = [
        str(p.relative_to(src_root))
        for p in src_root.rglob("*.py")
        if "torch" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"Runner files import/reference torch: {offenders}"
