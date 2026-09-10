"""Shared AE module contract tests (ticket 01, TDD red)."""

from __future__ import annotations

import numpy as np


def _cfg(latent_dim: int = 8):
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(denoising_gate=SimpleNamespace(latent_dim=latent_dim)),
        training=SimpleNamespace(random_seed=42),
    )


def test_ae_compressor_transform_shape():
    from models._shared.ae import AECompressor

    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 12)).astype(np.float32)
    comp = AECompressor(latent_dim=8, random_state=42, max_iter=5)
    comp.fit(X)
    Z = comp.transform(X)
    assert Z.shape == (20, 8)


def test_ae_model_contract_pure_and_recallable(tmp_path):
    from models._shared.ae import create_ae_model
    from src.runner import check_predict_contract

    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 12)).astype(np.float32)
    y = rng.integers(0, 3, size=30)
    model = create_ae_model(_cfg(latent_dim=8))
    model.fit(X, y)
    n_before = getattr(model, "n_fits_", 1)
    model.fit(X, y, X_val=X, y_val=y)
    assert model.n_fits_ == n_before + 1
    check_predict_contract(model, X)
    preds = model.predict(X)
    assert preds.dtype.kind in ("i", "u")
    model.save(tmp_path / "ae_artifact")
    assert (tmp_path / "ae_artifact").is_file()
