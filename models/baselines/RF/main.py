"""Random Forest baseline — thin ensemble wrapper."""

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


class RFModel:
    """Non-incremental forest exposing fit, predict, save."""

    def __init__(
        self, n_estimators: int = 200, random_state: int = 42, n_jobs: int = -1
    ):
        from sklearn.ensemble import RandomForestClassifier

        self._clf = RandomForestClassifier(
            n_estimators=int(n_estimators),
            random_state=int(random_state),
            n_jobs=int(n_jobs),
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


def create_model(cfg: Any) -> RFModel:
    model_cfg = _cfg_get(cfg, "model", None)
    n_est = _cfg_get(model_cfg, "n_estimators", 200)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return RFModel(n_estimators=int(n_est), random_state=int(seed))
