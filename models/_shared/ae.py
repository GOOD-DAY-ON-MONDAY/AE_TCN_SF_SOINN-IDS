"""Shared AE module — single reusable compression behavior.

Tracer-bullet sklearn implementation without torch so smoke runs stay green
in the verified env. AE trains over epochs via MLPRegressor reconstruction;
PCA wrapper stays the deterministic control. Thin wrappers compose this.
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


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


class AECompressor:
    """Denoising-style compressor: scaler plus MLP reconstruction."""

    def __init__(
        self, latent_dim: int = 32, random_state: int = 42, max_iter: int = 200
    ):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self.latent_dim = int(latent_dim)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self._scaler = StandardScaler()
        self._mlp = MLPRegressor(
            hidden_layer_sizes=(64, self.latent_dim, 64),
            activation="relu",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._fitted = False

    def fit(self, X) -> AECompressor:
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.fit_transform(X)
        self._mlp.fit(Xs, Xs)
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("AECompressor not fitted")
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.transform(X)
        w0, w1 = self._mlp.coefs_[0], self._mlp.coefs_[1]
        b0, b1 = self._mlp.intercepts_[0], self._mlp.intercepts_[1]
        h0 = _relu(Xs @ w0 + b0)
        latent = _relu(h0 @ w1 + b1)
        return np.asarray(latent, dtype=np.float32)


class AEModel:
    """Runner-contract wrapper: AE latent plus linear classifier."""

    supports_incremental = False

    def __init__(
        self, latent_dim: int = 32, random_state: int = 42, max_iter: int = 200
    ):
        from sklearn.linear_model import LogisticRegression

        self.compressor = AECompressor(
            latent_dim=latent_dim, random_state=random_state, max_iter=max_iter
        )
        self._clf = LogisticRegression(max_iter=500)
        self.n_fits_ = 0

    def fit(self, X, y, X_val=None, y_val=None):
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
                "mlp": self.compressor._mlp,
                "clf": self._clf,
                "latent_dim": self.compressor.latent_dim,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_ae_model(cfg: Any) -> AEModel:
    model_cfg = _cfg_get(cfg, "model", None)
    gate = _cfg_get(model_cfg, "denoising_gate", None)
    latent = _cfg_get(gate, "latent_dim", None)
    if latent is None:
        latent = _cfg_get(model_cfg, "latent_dim", 32)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return AEModel(latent_dim=int(latent), random_state=int(seed))
