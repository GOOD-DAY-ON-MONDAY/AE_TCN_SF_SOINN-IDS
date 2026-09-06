"""Tests for the multi-seed loop and mean ± std aggregation (ticket 06)."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest

from src.runner import aggregate_rows, run_seeds
from tests.test_model_contract import REQUIRED, _Cfg, _main_py, _write_model


@pytest.fixture()
def stub_model(tmp_path: Path) -> Path:
    return _write_model(tmp_path, _main_py(REQUIRED))


@pytest.fixture()
def data() -> tuple:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 8)).astype(np.float32)
    y = rng.integers(0, 4, size=120)
    half = 100
    return X[:half], y[:half], X[half:], y[half:]


def test_run_seeds_produces_dirs_and_csv_rows(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    csv_path = tmp_path / "reports" / "comparison_results" / "all_runs.csv"
    report_root = tmp_path / "artifacts"

    rows = run_seeds(
        model_dir=stub_model,
        dataset="netml2020",
        seeds=[1, 2, 3],
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=report_root,
        csv_path=csv_path,
    )

    assert [r["seed"] for r in rows] == [1, 2, 3]
    for seed in (1, 2, 3):
        run_dir = report_root / stub_model.name / "netml2020" / f"seed_{seed}"
        assert (run_dir / "metrics.json").is_file()
    with open(csv_path, newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 3
    assert [r["seed"] for r in csv_rows] == ["1", "2", "3"]


def test_aggregation_math_on_fixed_stub_across_two_seeds(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    rows = run_seeds(
        model_dir=stub_model,
        dataset="netml2020",
        seeds=[1, 2],
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=tmp_path / "artifacts",
        csv_path=tmp_path / "all_runs.csv",
    )

    agg = aggregate_rows(rows)
    accs = [float(r["accuracy"]) for r in rows]
    expected_mean = sum(accs) / len(accs)
    expected_std = math.sqrt(sum((a - expected_mean) ** 2 for a in accs) / len(accs))
    mean, std = agg["accuracy"]
    assert mean == pytest.approx(expected_mean)
    assert std == pytest.approx(expected_std)
    # Deterministic stub model: same data -> identical predictions -> std 0.
    assert std == pytest.approx(0.0)


def test_aggregate_rows_single_row_has_zero_std() -> None:
    rows = [{
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
        "latency_ms": 2.0,
        "peak_mem_gb": 1.0,
        "train_time_s": 0.1,
    }]
    agg = aggregate_rows(rows)  # type: ignore[arg-type]
    for key in ("accuracy", "latency_ms", "train_time_s"):
        mean, std = agg[key]
        assert std == 0.0
        assert mean == pytest.approx(float(rows[0][key]))


def test_run_seeds_empty_list_raises(tmp_path: Path) -> None:
    from src.runner import RunnerError

    with pytest.raises(RunnerError):
        run_seeds(
            model_dir=tmp_path,
            dataset="netml2020",
            seeds=[],
            cfg=None,
            X_train=None,
            y_train=None,
            X_val=None,
            y_val=None,
        )
