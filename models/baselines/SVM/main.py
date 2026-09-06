"""Minimal SVM baseline — tracer-bullet model for the Runner harness.

Implements the v1 contract: ``create_model(cfg)`` factory, ``fit(X, y, X_val=None,
y_val=None)`` (val ignored), ``predict(X)`` (hard integer indices, pure read), and
``save(path)`` (joblib). Not incremental: no ``supports_incremental``,
``partial_fit``, or ``predict_and_adapt``.

Uses LinearSVC (liblinear) wrapped in a StandardScaler pipeline, not kernel SVC:
libsvm's kernel matrix scales O(n^2) in memory (~100+ GB on the ~387k-row
netml2020 set), while liblinear scales ~linearly. C is read from cfg
(``cfg.model.<key>`` or ``cfg.<key>``) if present, else the sklearn default 1.0.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict or an attribute-style config node.

    Args:
        obj (Any): dict-like or attribute-like config node (or None).
        key (str): key/attribute name to read.
        default (Any): value returned when the key is absent.

    Returns:
        Any: the value at ``key``, or ``default``.
    """
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class SVMModel:
    """Non-incremental SVM baseline exposing fit / predict / save."""

    def __init__(self, C: float = 1.0):
        """Build a StandardScaler + LinearSVC pipeline.

        Args:
            C (float): LinearSVC inverse-regularization strength.
        """
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import LinearSVC

        # dual="auto" picks the primal solver since n_samples >> n_features.
        self._clf = make_pipeline(
            StandardScaler(),
            LinearSVC(C=C, dual="auto", max_iter=10000),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        # X_val/y_val accepted per the uniform signature; SVM has no early
        # stopping, so they are ignored.
        self._clf.fit(X, y)

    def predict(self, X):
        return self._clf.predict(X)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)


def create_model(cfg: Any) -> SVMModel:
    # Hyperparameters are optional; model.yaml need not exist.
    model_cfg = _cfg_get(cfg, "model", None)
    C = _cfg_get(model_cfg, "C", None) or _cfg_get(cfg, "C", 1.0)
    return SVMModel(C=float(C))
