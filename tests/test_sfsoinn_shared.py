"""Shared SF-SOINN module contract tests (ticket 04, TDD red)."""

from __future__ import annotations

import numpy as np


def _cfg(threshold: float = 5.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(
            zeroday_hunter=SimpleNamespace(similarity_threshold=threshold)
        ),
        training=SimpleNamespace(random_seed=42),
    )


def test_sfsoinn_incremental_flag_and_methods():
    from models._shared.sfsoinn import create_sfsoinn_model

    model = create_sfsoinn_model(_cfg())
    assert getattr(model, "supports_incremental", False) is True
    assert hasattr(model, "partial_fit")
    assert hasattr(model, "predict_and_adapt")


def test_sfsoinn_learns_new_class_without_breaking_old(tmp_path):
    from models._shared.sfsoinn import create_sfsoinn_model
    from src.runner import check_predict_contract

    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(20, 6)).astype(np.float32)
    y0 = np.zeros(20, dtype=int)
    X1 = rng.normal(loc=8.0, scale=0.5, size=(10, 6)).astype(np.float32)
    y1 = np.ones(10, dtype=int)
    model = create_sfsoinn_model(_cfg(threshold=5.0))
    model.fit(X0, y0)
    check_predict_contract(model, X0)
    before = model.predict(X0)
    assert (before == 0).mean() > 0.8
    model.partial_fit(X1, y1)
    after_old = model.predict(X0)
    assert (after_old == 0).mean() > 0.7
    model.save(tmp_path / "sfsoinn_artifact")
    assert (tmp_path / "sfsoinn_artifact").is_file()
