"""Shared AE module contract tests (ticket 03, real torch AE per ADR 0008)."""

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


def test_ae_reconstruction_loss_decreases_over_epochs():
    """Training longer must reduce reconstruction MSE (real learning)."""
    from models._shared.ae import AECompressor

    rng = np.random.default_rng(2)
    X = rng.normal(size=(128, 12)).astype(np.float32)
    losses = []
    for epochs in (1, 10):
        comp = AECompressor(latent_dim=8, random_state=42, max_iter=epochs)
        comp.fit(X)
        losses.append(comp.reconstruction_loss(X))
    assert losses[1] < losses[0]


def test_ae_per_seed_init_differs():
    from models._shared.ae import AECompressor

    X = np.random.default_rng(3).normal(size=(32, 12)).astype(np.float32)
    c1 = AECompressor(latent_dim=8, random_state=1, max_iter=1)
    c2 = AECompressor(latent_dim=8, random_state=2, max_iter=1)
    c1.fit(X)
    c2.fit(X)
    w1 = next(c1.net.parameters()).detach().cpu().numpy()
    w2 = next(c2.net.parameters()).detach().cpu().numpy()
    assert not np.allclose(w1, w2)


def test_ae_save_load_roundtrip_state_dict(tmp_path):
    import joblib

    from models._shared.ae import AECompressor

    X = np.random.default_rng(4).normal(size=(32, 12)).astype(np.float32)
    comp = AECompressor(latent_dim=8, random_state=42, max_iter=2)
    comp.fit(X)
    path = tmp_path / "ae_state.pt"
    comp.save(path)
    assert path.is_file()
    payload = joblib.load(path)
    assert "state_dict" in payload and "scaler" in payload
    restored = AECompressor(latent_dim=8, random_state=42, max_iter=2)
    restored.load(path)
    np.testing.assert_allclose(restored.transform(X), comp.transform(X), rtol=1e-5)


def test_ae_device_check_printed(capsys):
    from models._shared.ae import AECompressor

    X = np.random.default_rng(5).normal(size=(32, 12)).astype(np.float32)
    comp = AECompressor(latent_dim=8, random_state=42, max_iter=1)
    comp.fit(X)
    out = capsys.readouterr().out
    assert "AE" in out and "device" in out.lower()


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
