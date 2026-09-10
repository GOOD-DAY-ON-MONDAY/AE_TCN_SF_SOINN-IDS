"""Distillation baseline — thin teacher-student wrapper."""

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


class DistillationModel:
    """Teacher MLP distilled into a lightweight student for inference."""

    supports_incremental = False

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        random_state: int = 42,
        max_iter: int = 200,
    ):
        from models._shared.baseline_helpers import DistillationHelper

        self.helper = DistillationHelper(
            temperature=float(temperature),
            alpha=float(alpha),
            random_state=int(random_state),
        )
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self._teacher = None
        self._student = None

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        from sklearn.neural_network import MLPClassifier

        X = np.asarray(X)
        y = np.asarray(y)
        self._teacher = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        self._teacher.fit(X, y)
        self._student = self.helper.distill(self._teacher, X, y)

    def predict(self, X):
        if self._student is None:
            raise RuntimeError("DistillationModel not fitted")
        return np.asarray(self._student.predict(np.asarray(X)), dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "teacher": self._teacher,
                "student": self._student,
                "temperature": self.helper.temperature,
                "alpha": self.helper.alpha,
            },
            path,
        )


def create_model(cfg: Any) -> DistillationModel:
    model_cfg = _cfg_get(cfg, "model", None)
    dist_cfg = _cfg_get(model_cfg, "distillation", None)
    temperature = _cfg_get(dist_cfg, "temperature", 2.0)
    alpha = _cfg_get(dist_cfg, "alpha", 0.5)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return DistillationModel(
        temperature=float(temperature), alpha=float(alpha), random_state=int(seed)
    )
