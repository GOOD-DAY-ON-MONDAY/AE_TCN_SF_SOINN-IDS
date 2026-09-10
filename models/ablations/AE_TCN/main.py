"""AE plus TCN ablation — compression with sequence modeling, no clustering."""

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


class AETCNModel:
    """Shared AE latents refined by shared TCN features plus classifier."""

    supports_incremental = False

    def __init__(
        self,
        latent_dim: int = 32,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        random_state: int = 42,
        max_iter: int = 200,
    ):
        from sklearn.linear_model import LogisticRegression

        from models._shared.ae import AECompressor
        from models._shared.tcn import TCNExtractor

        self.compressor = AECompressor(
            latent_dim=int(latent_dim),
            random_state=int(random_state),
            max_iter=int(max_iter),
        )
        self.extractor = TCNExtractor(
            in_dim=int(latent_dim),
            channels=list(channels),
            kernel_size=int(kernel_size),
            dilations=list(dilations),
            random_state=int(random_state),
        )
        self._clf = LogisticRegression(max_iter=500)
        self.n_fits_ = 0

    def _features(self, X) -> np.ndarray:
        Z = self.compressor.transform(np.asarray(X))
        return self.extractor.transform(Z)

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        X = np.asarray(X)
        y = np.asarray(y)
        if self.n_fits_ == 0:
            self.compressor.fit(X)
        Z = self._features(X)
        self._clf.fit(Z, y)
        self.n_fits_ += 1

    def predict(self, X):
        return np.asarray(self._clf.predict(self._features(X)), dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.compressor._scaler,
                "mlp": self.compressor._mlp,
                "latent_dim": self.compressor.latent_dim,
                "weights": self.extractor.weights,
                "biases": self.extractor.biases,
                "channels": self.extractor.channels,
                "clf": self._clf,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_model(cfg: Any) -> AETCNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    gate = _cfg_get(model_cfg, "denoising_gate", None)
    latent = _cfg_get(gate, "latent_dim", None)
    if latent is None:
        latent = _cfg_get(model_cfg, "latent_dim", 32)
    eng = _cfg_get(model_cfg, "temporal_engine", None)
    channels = _cfg_get(eng, "channels", [64, 64, 32])
    kernel = _cfg_get(eng, "kernel_size", 3)
    dilations = _cfg_get(eng, "dilations", [1, 2, 4, 8])
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return AETCNModel(
        latent_dim=int(latent),
        channels=list(channels),
        kernel_size=int(kernel),
        dilations=list(dilations),
        random_state=int(seed),
    )
