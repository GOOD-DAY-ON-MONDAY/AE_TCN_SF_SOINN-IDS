"""Baseline infra extensions — sequence, incremental, distillation helpers.

Torch-ready interfaces with sklearn tracer implementations so Baselines
stay green without new Runner plumbing.
"""

from __future__ import annotations

import numpy as np


class SequenceBatcher:
    """Sliding-window sequences for temporal baselines."""

    def __init__(self, window_size: int = 8, stride: int = 4):
        self.window_size = int(window_size)
        self.stride = int(stride)

    def to_sequences(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        starts = list(range(0, max(n - self.window_size + 1, 1), self.stride))
        if not starts:
            starts = [0]
        seqs = []
        for s in starts:
            chunk = X[s : s + self.window_size]
            if len(chunk) < self.window_size:
                pad = np.zeros(
                    (self.window_size - len(chunk), X.shape[1]), dtype=np.float32
                )
                chunk = np.concatenate([chunk, pad], axis=0)
            seqs.append(chunk)
        return np.stack(seqs).astype(np.float32)

    def from_sequences(self, S) -> np.ndarray:
        S = np.asarray(S, dtype=np.float32)
        return S.reshape(-1, S.shape[-1]).astype(np.float32)


class DistillationHelper:
    """Teacher-student training with temperature and weighting."""

    def __init__(
        self, temperature: float = 2.0, alpha: float = 0.5, random_state: int = 42
    ):
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.random_state = int(random_state)

    def soft_targets(self, teacher, X) -> np.ndarray:
        import numpy as _np

        proba = np.asarray(teacher.predict_proba(np.asarray(X)))
        logits = _np.log(_np.clip(proba, 1e-6, 1.0))
        tempered = logits / self.temperature
        tempered -= tempered.max(axis=1, keepdims=True)
        exp = _np.exp(tempered)
        return (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)

    def distill(self, teacher, X, y):
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X)
        y = np.asarray(y)
        soft = self.soft_targets(teacher, X)
        hard_labels = np.asarray(y)
        soft_labels = soft.argmax(axis=1)
        combined_X = np.concatenate([X, X], axis=0)
        combined_y = np.concatenate([hard_labels, soft_labels], axis=0)
        weights = np.concatenate(
            [
                np.full(len(X), 1.0 - self.alpha),
                np.full(len(X), self.alpha),
            ]
        )
        student = LogisticRegression(max_iter=500, random_state=self.random_state)
        student.fit(combined_X, combined_y, sample_weight=weights)
        return student
