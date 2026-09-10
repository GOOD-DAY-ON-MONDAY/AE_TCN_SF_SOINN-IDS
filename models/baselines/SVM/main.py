"""SVM baseline — thin wrapper proving taxonomy split.

Standard linear classifier with no shared deep modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class SVMModel:
    """Non-incremental linear SVM exposing fit, predict, save."""

    def __init__(self, C: float = 1.0, max_iter: int = 10000):
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        self._clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=float(C), dual="auto", max_iter=int(max_iter)),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        self._clf.fit(X, y)

    def predict(self, X):
        return self._clf.predict(X)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)


def create_model(cfg: Any) -> SVMModel:
    model_cfg = _cfg_get(cfg, "model", None)
    C = _cfg_get(model_cfg, "C", None) or _cfg_get(cfg, "C", 1.0)
    max_iter = _cfg_get(model_cfg, "max_iter", None) or _cfg_get(cfg, "max_iter", 10000)
    return SVMModel(C=float(C), max_iter=int(max_iter))
