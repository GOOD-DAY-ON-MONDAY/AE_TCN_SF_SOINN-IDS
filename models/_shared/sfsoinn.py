"""Shared SF-SOINN module — reusable incremental clustering.

Tracer-bullet prototype-based behavior honoring incremental contract:
pure-read predict, controlled partial_fit, live predict_and_adapt.
"""

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


class SFSOINNCluster:
    """Prototype nodes with distance-gated growth."""

    def __init__(self, similarity_threshold: float = 0.5):
        self.threshold = float(similarity_threshold)
        self.prototypes: list[np.ndarray] = []
        self.labels: list[int] = []
        self.counts: list[int] = []

    def _nearest(self, x: np.ndarray) -> tuple[int, float]:
        if not self.prototypes:
            return -1, float("inf")
        dists = [float(np.linalg.norm(p - x)) for p in self.prototypes]
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        for label in sorted(set(y.tolist())):
            members = X[y == label]
            self.prototypes.append(members.mean(axis=0).astype(np.float32))
            self.labels.append(int(label))
            self.counts.append(len(members))

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y, strict=True):
            idx, dist = self._nearest(xi)
            if idx >= 0 and self.labels[idx] == int(yi) and dist <= self.threshold:
                c = self.counts[idx]
                self.prototypes[idx] = (
                    (self.prototypes[idx] * c + xi) / (c + 1)
                ).astype(np.float32)
                self.counts[idx] = c + 1
            else:
                self.prototypes.append(xi.astype(np.float32))
                self.labels.append(int(yi))
                self.counts.append(1)

    def predict(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        out = []
        for xi in X:
            idx, _ = self._nearest(xi)
            out.append(self.labels[idx] if idx >= 0 else 0)
        return np.asarray(out, dtype=int)


class SFSOINNModel:
    """Runner-contract incremental wrapper."""

    supports_incremental = True

    def __init__(self, similarity_threshold: float = 0.5):
        self.cluster = SFSOINNCluster(similarity_threshold=similarity_threshold)
        self.n_fits_ = 0
        self._fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        if not self._fitted:
            self.cluster.fit(X, y)
            self._fitted = True
        else:
            self.cluster.partial_fit(X, y)
        self.n_fits_ += 1

    def partial_fit(self, X, y):
        if not self._fitted:
            self.cluster.fit(X, y)
            self._fitted = True
        else:
            self.cluster.partial_fit(X, y)

    def predict(self, X):
        return self.cluster.predict(X)

    def predict_and_adapt(self, X):
        X = np.asarray(X, dtype=np.float32)
        out = []
        next_label = max(self.cluster.labels, default=-1) + 1
        for xi in X:
            idx, dist = self.cluster._nearest(xi)
            if idx < 0 or dist > self.cluster.threshold:
                self.cluster.prototypes.append(xi.astype(np.float32))
                self.cluster.labels.append(int(next_label))
                self.cluster.counts.append(1)
                out.append(int(next_label))
                next_label += 1
            else:
                out.append(self.cluster.labels[idx])
        return np.asarray(out, dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "prototypes": self.cluster.prototypes,
                "labels": self.cluster.labels,
                "counts": self.cluster.counts,
                "threshold": self.cluster.threshold,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_sfsoinn_model(cfg: Any) -> SFSOINNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    hunter = _cfg_get(model_cfg, "zeroday_hunter", None)
    thr = _cfg_get(hunter, "similarity_threshold", 0.5)
    return SFSOINNModel(similarity_threshold=float(thr))
