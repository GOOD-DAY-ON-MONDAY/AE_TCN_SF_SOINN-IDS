"""Shared TCN module contract tests (ticket 03, TDD red)."""

from __future__ import annotations

import numpy as np


def _cfg():
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(
            temporal_engine=SimpleNamespace(
                channels=[16, 16],
                kernel_size=3,
                dilations=[1, 2],
                dropout=0.0,
            )
        ),
        training=SimpleNamespace(random_seed=42),
    )


def test_tcn_transform_shape():
    from models._shared.tcn import TCNExtractor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12)).astype(np.float32)
    ext = TCNExtractor(in_dim=12, channels=[16, 16], kernel_size=3, dilations=[1, 2])
    Z = ext.transform(X)
    assert Z.shape[0] == 20
    assert Z.shape[1] == 16


def test_tcn_model_contract(tmp_path):
    from models._shared.tcn import create_tcn_model
    from src.runner import check_predict_contract

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 12)).astype(np.float32)
    y = rng.integers(0, 3, size=30)
    model = create_tcn_model(_cfg())
    model.fit(X, y)
    n_before = getattr(model, "n_fits_", 1)
    model.fit(X, y, X_val=X, y_val=y)
    assert model.n_fits_ == n_before + 1
    check_predict_contract(model, X)
    assert model.predict(X).dtype.kind in ("i", "u")
    model.save(tmp_path / "tcn_artifact")
    assert (tmp_path / "tcn_artifact").is_file()
