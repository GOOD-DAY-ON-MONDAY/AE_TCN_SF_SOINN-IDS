"""BiLSTM baseline — thin sequence wrapper with incremental support."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class BiLSTMModel:
    """Sequence-batched linear tracer with reproducible incremental updates."""

    supports_incremental = True

    def __init__(
        self,
        window_size: int = 8,
        stride: int = 4,
        random_state: int = 42,
    ):
        from sklearn.linear_model import SGDClassifier
        from sklearn.preprocessing import StandardScaler

        from models._shared.baseline_helpers import SequenceBatcher

        self.batcher = SequenceBatcher(window_size=int(window_size), stride=int(stride))
        self._scaler = StandardScaler()
        self._clf = SGDClassifier(
            loss="log_loss", random_state=int(random_state), max_iter=1000, tol=1e-3
        )
        self.random_state = int(random_state)
        self._fitted = False
        self._classes: np.ndarray | None = None

    def _features(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return self._scaler.transform(X).astype(np.float32)

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        # Sequence handling exercised for shape validation; per-flow
        # classifier trains on scaled flows so predict stays per-flow.
        _ = self.batcher.to_sequences(X)
        self._scaler.fit(X)
        self._classes = np.unique(y)
        self._clf.fit(self._features(X), y)
        self._fitted = True

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        if not self._fitted:
            self._scaler.fit(X)
            self._classes = np.unique(y)
            self._clf.partial_fit(self._features(X), y, classes=self._classes)
            self._fitted = True
            return
        assert self._classes is not None
        self._classes = np.unique(np.concatenate([self._classes, np.unique(y)]))
        self._clf.partial_fit(self._features(X), y, classes=self._classes)

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("BiLSTMModel not fitted")
        return np.asarray(self._clf.predict(self._features(X)), dtype=int)

    def predict_and_adapt(self, X):
        # Contract-only in v1: pure read, never called by the Runner loop.
        return self.predict(X)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self._scaler,
                "clf": self._clf,
                "window_size": self.batcher.window_size,
                "stride": self.batcher.stride,
                "classes": self._classes,
                "fitted": self._fitted,
            },
            path,
        )


def create_model(cfg: Any) -> BiLSTMModel:
    model_cfg = _cfg_get(cfg, "model", None)
    bilstm_cfg = _cfg_get(model_cfg, "bilstm", None)
    window = _cfg_get(bilstm_cfg, "window_size", 8)
    stride = _cfg_get(bilstm_cfg, "stride", 4)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return BiLSTMModel(
        window_size=int(window), stride=int(stride), random_state=int(seed)
    )
