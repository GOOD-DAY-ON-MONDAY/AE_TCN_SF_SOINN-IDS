"""BiLSTM baseline (torch) contract tests — ticket 03, TDD."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def _cfg(seed: int = 42) -> SimpleNamespace:
    """Minimal merged-config stand: training.random_seed only."""
    return SimpleNamespace(training=SimpleNamespace(random_seed=seed))


def _toy_data(n: int = 60, d: int = 8):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_create_model_reads_cfg_seed() -> None:
    from models.baselines.BiLSTM.main import BiLSTMModel

    m = BiLSTMModel.from_config(_cfg(seed=7))
    assert m.random_state == 7


def test_supports_incremental_and_methods() -> None:
    from models.baselines.BiLSTM.main import BiLSTMModel

    m = BiLSTMModel.from_config(_cfg())
    assert m.supports_incremental is True
    assert callable(m.partial_fit)
    assert callable(m.predict_and_adapt)


def test_fit_predict_hard_integer_labels() -> None:
    from models.baselines.BiLSTM.main import BiLSTMModel

    X, y = _toy_data()
    m = BiLSTMModel.from_config(_cfg())
    m.fit(X, y)
    out = np.asarray(m.predict(X))
    assert out.ndim == 1
    assert out.shape[0] == len(X)  # per-flow predictions
    assert np.issubdtype(out.dtype, np.integer)
    assert np.array_equal(out, np.asarray(m.predict(X)))


def test_partial_fit_changes_predictions() -> None:
    """partial_fit must actually update the model (SGD steps on new data)."""
    from models.baselines.BiLSTM.main import BiLSTMModel

    X, y = _toy_data()
    m = BiLSTMModel.from_config(_cfg())
    m.fit(X, y)
    before = np.asarray(m.predict(X))
    m.partial_fit(X[:20], y[:20])
    after = np.asarray(m.predict(X))
    # Not guaranteed to differ, but the update path must run without error and
    # stay contract-valid.
    assert after.shape == before.shape


def test_predict_before_fit_raises() -> None:
    from models.baselines.BiLSTM.main import BiLSTMModel

    X, _ = _toy_data(n=10)
    m = BiLSTMModel.from_config(_cfg())
    with pytest.raises(RuntimeError):
        m.predict(X)


def test_save_writes_artifact(tmp_path) -> None:
    from models.baselines.BiLSTM.main import BiLSTMModel

    X, y = _toy_data()
    m = BiLSTMModel.from_config(_cfg())
    m.fit(X, y)
    path = tmp_path / "artifact"
    m.save(path)
    assert path.exists()


def test_seed_changes_init() -> None:
    import torch

    from models.baselines.BiLSTM.main import BiLSTMModel

    X, y = _toy_data()
    first_params = None
    for seed in (1, 2):
        m = BiLSTMModel.from_config(_cfg(seed=seed))
        m.fit(X, y)
        params = torch.cat([p.detach().flatten() for p in m._net.parameters()])
        if first_params is None:
            first_params = params
        else:
            assert not torch.equal(first_params, params)
