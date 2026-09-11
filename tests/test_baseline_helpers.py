"""Baseline infra extensions tests (ticket 05, TDD red)."""

from __future__ import annotations

import numpy as np


def test_sequence_batcher_shapes():
    from models._shared.baseline_helpers import SequenceBatcher

    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 6)).astype(np.float32)
    b = SequenceBatcher(window_size=4, stride=2)
    S = b.to_sequences(X)
    assert S.shape[0] == 4
    assert S.shape[1:] == (4, 6)
    X_back = b.from_sequences(S)
    assert X_back.shape[1] == 6


def test_seed_torch_is_deterministic():
    import torch

    from models._shared.baseline_helpers import seed_torch

    seed_torch(7)
    a = torch.rand(4)
    seed_torch(7)
    b = torch.rand(4)
    assert torch.equal(a, b)
    seed_torch(8)
    c = torch.rand(4)
    assert not torch.equal(a, c)


def test_get_device_returns_torch_device():
    import torch

    from models._shared.baseline_helpers import get_device

    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "cuda")


def test_print_device_check_outputs_model_and_device(capsys):
    import torch

    from models._shared.baseline_helpers import print_device_check

    print_device_check("TestModel", torch.device("cpu"))
    out = capsys.readouterr().out
    assert "TestModel" in out
    assert "cpu" in out


def test_distillation_helper_trains_student(tmp_path):
    from sklearn.linear_model import LogisticRegression

    from models._shared.baseline_helpers import DistillationHelper

    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y = (X[:, 0] > 0).astype(int)
    teacher = LogisticRegression(max_iter=500).fit(X, y)
    helper = DistillationHelper(temperature=2.0, alpha=0.5, random_state=42)
    student = helper.distill(teacher, X, y)
    preds = student.predict(X)
    assert preds.shape == (40,)
    assert (preds == y).mean() > 0.7
