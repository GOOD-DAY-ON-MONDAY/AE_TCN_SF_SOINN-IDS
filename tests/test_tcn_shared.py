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


def test_tcn_training_changes_weights():
    """Real gradient training: conv weights must move during fit (ADR 0006)."""
    import torch

    from models._shared.tcn import TCNExtractor

    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 12)).astype(np.float32)
    y = rng.integers(0, 2, size=60)
    ext = TCNExtractor(
        in_dim=12, channels=[16, 16], kernel_size=3, dilations=[1, 2], epochs=2
    )
    before = {
        k: v.clone() for k, v in ext.net.state_dict().items()
    }
    ext.fit(X, y)
    after = ext.net.state_dict()
    changed = any(
        not torch.equal(before[k], after[k])
        for k in before
        if before[k].dtype.is_floating_point
    )
    assert changed, "TCN weights did not change during fit — still a stub"


def test_tcn_per_seed_init_differs():
    """Per-instance seeding: different random_state → different init weights."""
    import torch

    from models._shared.tcn import TCNExtractor

    e1 = TCNExtractor(in_dim=12, channels=[16], kernel_size=3, dilations=[1], random_state=1)
    e2 = TCNExtractor(in_dim=12, channels=[16], kernel_size=3, dilations=[1], random_state=2)
    s1 = e1.net.state_dict()
    s2 = e2.net.state_dict()
    assert not torch.equal(next(iter(s1.values())), next(iter(s2.values())))


def test_tcn_transform_pure_after_fit():
    """transform must be inference-only: repeated calls give identical output."""
    from models._shared.tcn import TCNExtractor

    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 12)).astype(np.float32)
    y = rng.integers(0, 2, size=40)
    ext = TCNExtractor(in_dim=12, channels=[16], kernel_size=3, dilations=[1], epochs=1)
    ext.fit(X, y)
    Z1 = ext.transform(X)
    Z2 = ext.transform(X)
    assert np.array_equal(Z1, Z2)
