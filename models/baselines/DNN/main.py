"""DNN baseline — thin feedforward wrapper reusing harness concept."""

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


class DNNModel:
    """Non-incremental MLP exposing fit, predict, save."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        random_state: int = 42,
        max_iter: int = 200,
    ):
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self._clf = make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=tuple(hidden),
                random_state=int(random_state),
                max_iter=int(max_iter),
            ),
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


def create_model(cfg: Any) -> DNNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    dnn_cfg = _cfg_get(model_cfg, "dnn", None)
    hidden = _cfg_get(dnn_cfg, "hidden", (128, 64))
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return DNNModel(hidden=tuple(hidden), random_state=int(seed))
