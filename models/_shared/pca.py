"""Shared PCA wrapper — single deterministic linear control.

Honors the standard fit contract as a one-shot transform with no epochs.
Thin wrappers compose this interchangeably with compression.
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


class PCACompressor:
    """Deterministic PCA transform with scaler."""

    def __init__(self, n_components: int = 32, random_state: int = 42):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=self.n_components, random_state=self.random_state)
        self._fitted = False

    def fit(self, X) -> PCACompressor:
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.fit_transform(X)
        self._pca.fit(Xs)
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("PCACompressor not fitted")
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float32))
        return np.asarray(self._pca.transform(Xs), dtype=np.float32)


class PCAModel:
    """Runner-contract wrapper: PCA latent plus classifier."""

    supports_incremental = False

    def __init__(self, n_components: int = 32, random_state: int = 42):
        from sklearn.linear_model import LogisticRegression

        self.compressor = PCACompressor(
            n_components=n_components, random_state=random_state
        )
        self._clf = LogisticRegression(max_iter=500)
        self.n_fits_ = 0

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_fits_ == 0:
            self.compressor.fit(X)
        Z = self.compressor.transform(X)
        self._clf.fit(Z, y)
        self.n_fits_ += 1

    def predict(self, X):
        Z = self.compressor.transform(np.asarray(X))
        return np.asarray(self._clf.predict(Z), dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.compressor._scaler,
                "pca": self.compressor._pca,
                "clf": self._clf,
                "n_components": self.compressor.n_components,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_pca_model(cfg: Any) -> PCAModel:
    model_cfg = _cfg_get(cfg, "model", None)
    pca_cfg = _cfg_get(model_cfg, "pca", None)
    n_comp = _cfg_get(pca_cfg, "n_components", None)
    if n_comp is None:
        gate = _cfg_get(model_cfg, "denoising_gate", None)
        n_comp = _cfg_get(gate, "latent_dim", 32)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return PCAModel(n_components=int(n_comp), random_state=int(seed))
