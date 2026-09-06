"""Tests for the zero-day loop gating (ticket 08)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.runner import run_single_seed, run_zero_day_loop
from tests.test_model_contract import _main_py, _write_model

CLASS_MAP = {"benign": 0, "dos": 1, "zeroday": 2}


def _cfg(zero_day: list[str]) -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(random_seed=42),
        data=SimpleNamespace(
            netml2020=SimpleNamespace(
                class_map=CLASS_MAP, num_classes=3, feature_dim=8
            ),
        ),
        splitting=SimpleNamespace(
            val_split=0.15,
            random_seed=42,
            zero_day_classes=SimpleNamespace(
                netml2020=zero_day, cicids2017=[]
            ),
        ),
    )


def _incremental_model(tmp_path: Path) -> Path:
    src = _main_py(("fit", "predict", "save", "partial_fit", "predict_and_adapt"),
                   incremental=True)
    src = src.replace(
        "def partial_fit(self, X, y):\n        pass\n",
        "def partial_fit(self, X, y):\n"
        "        self.taught_ = list(y)\n",
    ).replace(
        "def predict(self, X):\n",
        "def predict(self, X):\n",
    )
    return _write_model(tmp_path, src, name="incr_model")


def _plain_model(tmp_path: Path) -> Path:
    return _write_model(tmp_path, _main_py(("fit", "predict", "save")),
                        name="plain_model")


def _data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 8)).astype(np.float32)
    y = rng.integers(0, 3, size=90)  # labels 0,1,2 — 2 is the zero-day class
    return X[:60], y[:60], X[60:], y[60:]


def test_loop_runs_protocol_for_incremental_model(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from src.runner import load_model

    model = load_model(_incremental_model(tmp_path), _cfg(["zeroday"]))
    X_train, y_train, X_val, y_val = _data()
    model.fit(X_known := X_val[y_val != 2], y_known := y_val[y_val != 2])
    result = run_zero_day_loop(
        model=model,
        cfg=_cfg(["zeroday"]),
        dataset="netml2020",
        X_withheld=X_val[y_val == 2],
        y_withheld=y_val[y_val == 2],
        X_known=X_val[y_val != 2],
        y_known=y_val[y_val != 2],
        run_dir=tmp_path / "zd",
    )
    assert result is not None
    assert result["n_withheld"] == int((y_val == 2).sum())
    assert result["used_predict_and_adapt"] is False
    assert "retention_prediction_stability" in result
    zd_json = json.loads((tmp_path / "zd" / "zero_day.json").read_text())
    assert zd_json == result


def test_loop_skips_for_non_incremental_model(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from src.runner import load_model

    model = load_model(_plain_model(tmp_path), _cfg(["zeroday"]))
    X_train, y_train, X_val, y_val = _data()
    model.fit(X_val[y_val != 2], y_val[y_val != 2])
    result = run_zero_day_loop(
        model=model,
        cfg=_cfg(["zeroday"]),
        dataset="netml2020",
        X_withheld=X_val[y_val == 2],
        y_withheld=y_val[y_val == 2],
        X_known=X_val[y_val != 2],
        y_known=y_val[y_val != 2],
    )
    assert result is None
    assert "not an incremental model" in capsys.readouterr().out


def test_loop_skips_when_no_zero_day_classes_configured(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    from src.runner import load_model

    model = load_model(_incremental_model(tmp_path), _cfg([]))
    X_train, y_train, X_val, y_val = _data()
    model.fit(X_val[y_val != 2], y_val[y_val != 2])
    result = run_zero_day_loop(
        model=model,
        cfg=_cfg([]),
        dataset="netml2020",
        X_withheld=X_val[y_val == 2],
        y_withheld=y_val[y_val == 2],
        X_known=X_val[y_val != 2],
        y_known=y_val[y_val != 2],
    )
    assert result is None
    assert "no zero-day classes configured" in capsys.readouterr().out


def test_predict_and_adapt_never_invoked_by_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.runner import load_model

    model_dir = _incremental_model(tmp_path)
    model = load_model(model_dir, _cfg(["zeroday"]))
    X_train, y_train, X_val, y_val = _data()
    model.fit(X_val[y_val != 2], y_val[y_val != 2])
    monkeypatch.setattr(
        type(model), "predict_and_adapt",
        lambda self, X: (_ for _ in ()).throw(AssertionError("predict_and_adapt called")),
        raising=False,
    )
    X_train, y_train, X_val, y_val = _data()
    run_zero_day_loop(
        model=model,
        cfg=_cfg(["zeroday"]),
        dataset="netml2020",
        X_withheld=X_val[y_val == 2],
        y_withheld=y_val[y_val == 2],
        X_known=X_val[y_val != 2],
        y_known=y_val[y_val != 2],
    )  # must not raise


def test_loop_seed_reproducible(tmp_path: Path) -> None:
    from src.runner import load_model

    model_dir = _incremental_model(tmp_path)
    X_train, y_train, X_val, y_val = _data()
    X_known, y_known = X_val[y_val != 2], y_val[y_val != 2]
    kwargs = dict(
        cfg=_cfg(["zeroday"]),
        dataset="netml2020",
        X_withheld=X_val[y_val == 2],
        y_withheld=y_val[y_val == 2],
        X_known=X_val[y_val != 2],
        y_known=y_val[y_val != 2],
    )
    m1 = load_model(model_dir, _cfg(["zeroday"]))
    m1.fit(X_known, y_known)
    m2 = load_model(model_dir, _cfg(["zeroday"]))
    m2.fit(X_known, y_known)
    r1 = run_zero_day_loop(model=m1, run_dir=None, **kwargs)
    r2 = run_zero_day_loop(model=m2, run_dir=None, **kwargs)
    assert r1 == r2


def test_run_single_seed_completes_with_skip_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    # Non-incremental model + configured zero-day classes: run completes,
    # loop prints the yellow skip notice.
    monkeypatch.chdir(tmp_path)
    model_dir = _plain_model(tmp_path)
    X_train, y_train, X_val, y_val = _data()
    row = run_single_seed(
        model_dir=model_dir,
        dataset="netml2020",
        seed=1,
        cfg=_cfg(["zeroday"]),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        report_root=tmp_path / "artifacts",
        csv_path=tmp_path / "all_runs.csv",
    )
    assert Path(row["run_dir"]).is_dir()
    assert "zero-day loop skipped" in capsys.readouterr().out
