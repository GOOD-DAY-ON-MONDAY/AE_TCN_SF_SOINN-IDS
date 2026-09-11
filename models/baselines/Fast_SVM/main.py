"""Fast Linear SVM baseline — SGD-based tracer-bullet model for the Runner harness.

Implements the v1 contract: ``create_model(cfg)`` factory, ``fit(X, y, X_val=None,
y_val=None)`` (val ignored), ``predict(X)`` (hard integer indices, pure read), and
``save(path)`` (joblib). Not incremental: no ``supports_incremental``,
``partial_fit``, or ``predict_and_adapt``.

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


class FastLinearSVMModel:
    """Non-incremental fast Linear SVM (SGDClassifier) exposing fit / predict / save."""

    def __init__(
        self, alpha: float = 0.0001, max_iter: int = 1000, random_state: int = 42
    ):
        """Build a StandardScaler + SGDClassifier(loss="hinge") pipeline.

        Args:
            alpha (float): L2 regularization penalty multiplier (inversely proportional to C).
            max_iter (int): Maximum number of passes over the training data.
            random_state (int): seed for SGD shuffling (injected per seed by
                the Runner via cfg.training.random_seed).
        """
        from sklearn.linear_model import SGDClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        # loss="hinge" trains a Linear Support Vector Machine using SGD in O(n) time.
        self._clf = make_pipeline(
            StandardScaler(),
            SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=alpha,
                max_iter=max_iter,
                random_state=int(random_state),
                n_jobs=-1,
            ),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """Fit the scaler + SVM pipeline; validation args ignored.

        Args:
            X: training feature matrix.
            y: training integer labels.
            X_val: unused validation features (signature compat).
            y_val: unused validation labels (signature compat).
        """
        self._clf.fit(X, y)

    def predict(self, X):
        """Predict hard integer class indices.

        Args:
            X: feature matrix.

        Returns:
            Array of predicted class indices.
        """
        return self._clf.predict(X)

    def save(self, path):
        """Persist the pipeline with joblib, creating parent dirs.

        Args:
            path: destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)


def create_model(cfg: Any) -> FastLinearSVMModel:
    """Build a fast linear SVM from merged config.

    Args:
        cfg: merged config node; reads ``model.alpha`` (or ``C``) and ``max_iter``.

    Returns:
        FastLinearSVMModel: configured non-incremental classifier.
    """
    # Hyperparameters are optional; model.yaml need not exist.
    model_cfg = _cfg_get(cfg, "model", None)

    # Read 'alpha' directly, or convert 'C' (inverse regularization) if passed.
    alpha = _cfg_get(model_cfg, "alpha", None) or _cfg_get(cfg, "alpha", None)
    if alpha is None:
        c_val = _cfg_get(model_cfg, "C", None) or _cfg_get(cfg, "C", 1.0)
        alpha = 1.0 / float(c_val)

    max_iter = _cfg_get(model_cfg, "max_iter", None) or _cfg_get(cfg, "max_iter", 1000)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)

    return FastLinearSVMModel(
        alpha=float(alpha), max_iter=int(max_iter), random_state=int(seed)
    )
