"""Shared PCA wrapper contract tests (ticket 02, TDD red)."""

from __future__ import annotations

import numpy as np


def _cfg(n_components: int = 8):
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(pca=SimpleNamespace(n_components=n_components)),
        training=SimpleNamespace(random_seed=42),
    )


def test_pca_transform_shape_and_deterministic():
    from models._shared.pca import PCACompressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12)).astype(np.float32)
    a = PCACompressor(n_components=8, random_state=42)
    b = PCACompressor(n_components=8, random_state=42)
    a.fit(X)
    b.fit(X)
    Za, Zb = a.transform(X), b.transform(X)
    assert Za.shape == (20, 8)
    np.testing.assert_allclose(Za, Zb, rtol=1e-5, atol=1e-6)


def test_pca_model_ignores_val_and_epochs(tmp_path):
    from models._shared.pca import create_pca_model
    from src.runner import check_predict_contract

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 12)).astype(np.float32)
    y = rng.integers(0, 3, size=30)
    model = create_pca_model(_cfg(n_components=8))
    model.fit(X, y)
    n_before = getattr(model, "n_fits_", 1)
    model.fit(X, y, X_val=X * 999, y_val=y)
    assert model.n_fits_ == n_before + 1
    check_predict_contract(model, X)
    assert model.predict(X).dtype.kind in ("i", "u")
    model.save(tmp_path / "pca_artifact")
    assert (tmp_path / "pca_artifact").is_file()
