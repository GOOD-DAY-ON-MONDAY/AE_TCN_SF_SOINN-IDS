"""Tests for CLI/UX polish (ticket 07)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.runner import run_single_seed
from src.utils.ui import (
    ProgressBar,
    Spinner,
    chunked_predict_with_progress,
    format_error,
    print_plan,
)
from tests.test_model_contract import REQUIRED, _Cfg, _main_py, _write_model


def test_plan_header_lists_model_dataset_seeds_and_layers(
    capsys: pytest.CaptureFixture,
) -> None:
    print_plan(
        model_dir="models/baseline/rf",
        dataset="netml2020",
        seeds=[1, 2, 3],
        config_layers=[
            ("configs/base_config.yaml", "always"),
            ("model.yaml (model dir)", "yes"),
            ("--config", ""),
        ],
    )
    out = capsys.readouterr().out
    assert "models/baseline/rf" in out
    assert "netml2020" in out
    assert "1, 2, 3" in out
    assert "base_config.yaml" in out
    assert "absent" in out  # --config layer not provided


def test_error_formatting_suggests_the_fix() -> None:
    styled = format_error("Unknown model: 'x'\nDid you mean 'y'?")
    assert styled.startswith("error: Unknown model: 'x'")
    assert "Did you mean 'y'?" in styled


def test_progress_bar_renders_counts(capsys: pytest.CaptureFixture) -> None:
    # ProgressBar is a deprecated wrapper — still exported for compat.
    bar = ProgressBar(4, label="evaluating")
    for _ in range(4):
        bar.update()
    bar.finish()
    out = capsys.readouterr().err
    assert "4/4" in out


def test_spinner_runs_and_clears() -> None:
    # Spinner is a deprecated wrapper — still exported for compat.
    with Spinner("training"):
        pass  # enter/exit without error


def test_chunked_predict_with_progress_concatenates(
    capsys: pytest.CaptureFixture,
) -> None:
    class Stub:
        def predict(self, X):
            import numpy as np

            return np.zeros(len(X), dtype=int)

    X = np.arange(10)
    out = chunked_predict_with_progress(Stub(), X, chunks=5)
    assert list(out) == [0] * 10
    # Rich writes to stdout (console).  In a non-TTY pytest context Rich
    # renders plain text; MofNCompleteColumn emits "5/5".
    captured = capsys.readouterr()
    assert "5/5" in captured.out or "5/5" in captured.err


def test_chunked_predict_tiny_skips_bar(
    capsys: pytest.CaptureFixture,
) -> None:
    """n < 5: no progress bar, just a one-liner."""

    class Stub:
        def predict(self, X):
            import numpy as np

            return np.ones(len(X), dtype=int)

    X = np.arange(3)
    out = chunked_predict_with_progress(Stub(), X, chunks=5)
    assert list(out) == [1, 1, 1]
    captured = capsys.readouterr()
    # Should print a plain "evaluating N flow(s)…" line, not a progress bar.
    all_out = captured.out + captured.err
    assert "evaluating" in all_out


def test_run_single_seed_prints_summary_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.chdir(tmp_path)
    stub_model = _write_model(tmp_path, _main_py(REQUIRED))
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8)).astype(np.float32)
    y = rng.integers(0, 4, size=60)
    run_single_seed(
        model_dir=stub_model,
        dataset="netml2020",
        seed=1,
        cfg=_Cfg(),
        X_train=X[:40],
        y_train=y[:40],
        X_val=X[40:],
        y_val=y[40:],
        report_root=tmp_path / "artifacts",
        csv_path=tmp_path / "all_runs.csv",
    )
    captured = capsys.readouterr()
    # Rich Panel prints to stdout; the panel title contains "Run summary".
    all_out = captured.out + captured.err
    assert "Run summary" in all_out
    assert "accuracy" in all_out
