"""Tests for the single-seed tracer run (ticket 05)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.runner import RUN_COLUMNS, run_single_seed
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


def test_full_run_produces_artifacts_and_csv_row(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    csv_path = tmp_path / "reports" / "comparison_results" / "all_runs.csv"
    report_root = tmp_path / "artifacts"

    row = run_single_seed(
        model_dir=stub_model,
        dataset="netml2020",
        seed=1,
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=report_root,
        csv_path=csv_path,
    )

    # --- seed directory layout ------------------------------------------
    run_dir = Path(row["run_dir"])
    assert run_dir == report_root / stub_model.name / "netml2020" / "seed_1"
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "confusion_matrix.png").is_file()
    assert (run_dir / "model_artifact").is_file()

    # --- metrics.json contents -------------------------------------------
    metrics = json.loads((run_dir / "metrics.json").read_text())
    for key in RUN_COLUMNS:
        assert key in metrics
    assert "confusion_matrix" in metrics
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == len(X_val)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["latency_ms"] >= 0.0      # ms/flow
    assert metrics["peak_mem_gb"] > 0.0      # process RSS, GB
    assert metrics["train_time_s"] >= 0.0

    # --- exactly one CSV row with the exact 15 columns --------------------
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert tuple(rows[0].keys()) == RUN_COLUMNS
    assert rows[0]["seed"] == "1"
    assert rows[0]["dataset"] == "netml2020"
    assert rows[0]["model"] == str(stub_model)


def test_second_run_appends_exactly_one_row(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    csv_path = tmp_path / "all_runs.csv"
    kwargs = dict(
        model_dir=stub_model,
        dataset="cicids2017",
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=tmp_path / "artifacts",
        csv_path=csv_path,
    )
    run_single_seed(seed=1, **kwargs)
    run_single_seed(seed=2, **kwargs)
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert [r["seed"] for r in rows] == ["1", "2"]


def test_evaluates_on_test_split_when_provided(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    row = run_single_seed(
        model_dir=stub_model,
        dataset="netml2020",
        seed=1,
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_val,
        y_test=y_val,
        report_root=tmp_path / "artifacts",
        csv_path=tmp_path / "all_runs.csv",
    )
    metrics = json.loads((Path(row["run_dir"]) / "metrics.json").read_text())
    assert sum(sum(r) for r in metrics["confusion_matrix"]) == len(X_val)


def test_rerun_same_seed_overwrites_run_dir(
    tmp_path: Path, stub_model: Path, data: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    X_train, y_train, X_val, y_val = data
    kwargs = dict(
        model_dir=stub_model,
        dataset="netml2020",
        seed=7,
        cfg=_Cfg(),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=tmp_path / "artifacts",
        csv_path=tmp_path / "all_runs.csv",
    )
    first = run_single_seed(**kwargs)
    (Path(first["run_dir"]) / "sentinel.txt").write_text("old")
    second = run_single_seed(**kwargs)
    assert first["run_dir"] == second["run_dir"]
    assert not (Path(second["run_dir"]) / "sentinel.txt").exists()
