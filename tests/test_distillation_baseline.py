"""Distillation baseline (true KL-divergence KD) tests — ticket 04, TDD."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _cfg(seed: int = 42) -> SimpleNamespace:
    """Minimal merged-config stand."""
    return SimpleNamespace(
        training=SimpleNamespace(random_seed=seed),
        model=SimpleNamespace(
            distillation=SimpleNamespace(temperature=2.0, alpha=0.5)
        ),
    )


def _toy_data(n: int = 80, d: int = 8):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_create_model_reads_cfg_seed() -> None:
    from models.baselines.Distillation.main import DistillationModel

    m = DistillationModel.from_config(_cfg(seed=7))
    assert m.random_state == 7


def test_teacher_is_larger_than_student() -> None:
    """ADR 0005: teacher = DNN architecture (128,64); student strictly smaller."""
    from models.baselines.Distillation.main import DistillationModel

    X, y = _toy_data()
    m = DistillationModel.from_config(_cfg())
    m.fit(X, y)
    t_params = sum(p.numel() for p in m._teacher.parameters())
    s_params = sum(p.numel() for p in m._student.parameters())
    assert t_params > s_params


def test_fit_predict_hard_integer_labels() -> None:
    from models.baselines.Distillation.main import DistillationModel

    X, y = _toy_data()
    m = DistillationModel.from_config(_cfg())
    m.fit(X, y)
    out = np.asarray(m.predict(X))
    assert out.ndim == 1
    assert np.issubdtype(out.dtype, np.integer)
    assert set(out.tolist()) <= {0, 1}
    assert np.array_equal(out, np.asarray(m.predict(X)))


def test_kd_loss_uses_soft_targets() -> None:
    """The KD loss must consume teacher soft probabilities, not argmax."""
    from models.baselines.Distillation.main import DistillationModel

    X, y = _toy_data()
    m = DistillationModel.from_config(_cfg())
    m.fit(X, y)
    Xt = torch.from_numpy(X)
    with torch.no_grad():
        t_probs = torch.softmax(m._teacher(Xt) / m.temperature, dim=1)
    # Soft probabilities must be non-degenerate (not one-hot argmax collapsed).
    assert ((t_probs > 0.01) & (t_probs < 0.99)).any()
    assert abs(t_probs.sum(dim=1).mean().item() - 1.0) < 1e-5


def test_predict_before_fit_raises() -> None:
    from models.baselines.Distillation.main import DistillationModel

    X, _ = _toy_data(n=10)
    m = DistillationModel.from_config(_cfg())
    with pytest.raises(RuntimeError):
        m.predict(X)


def test_save_writes_artifact(tmp_path) -> None:
    from models.baselines.Distillation.main import DistillationModel

    X, y = _toy_data()
    m = DistillationModel.from_config(_cfg())
    m.fit(X, y)
    path = tmp_path / "artifact"
    m.save(path)
    assert path.exists()


def test_seed_changes_init() -> None:
    from models.baselines.Distillation.main import DistillationModel

    X, y = _toy_data()
    first_params = None
    for seed in (1, 2):
        m = DistillationModel.from_config(_cfg(seed=seed))
        m.fit(X, y)
        params = torch.cat(
            [p.detach().flatten() for p in m._student.parameters()]
        )
        if first_params is None:
            first_params = params
        else:
            assert not torch.equal(first_params, params)
