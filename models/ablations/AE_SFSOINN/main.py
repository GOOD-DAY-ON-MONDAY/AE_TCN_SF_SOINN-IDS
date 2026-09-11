"""AE plus SF-SOINN ablation — compression with clustering only, no TCN."""

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


class AESFSOINNModel:
    """Shared AE latents clustered by shared SF-SOINN prototypes."""

    supports_incremental = True

    def __init__(
        self,
        latent_dim: int = 32,
        similarity_threshold: float = 0.5,
        random_state: int = 42,
        max_iter: int = 200,
    ):
        from models._shared.ae import AECompressor
        from models._shared.sfsoinn import SFSOINNCluster

        self.compressor = AECompressor(
            latent_dim=int(latent_dim),
            random_state=int(random_state),
            max_iter=int(max_iter),
        )
        self.cluster = SFSOINNCluster(
            similarity_threshold=float(similarity_threshold)
        )
        self._fitted = False
        self.n_fits_ = 0

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._fitted:
            self.compressor.fit(X)
            self.cluster.fit(self.compressor.transform(X), y)
            self._fitted = True
        else:
            self.cluster.partial_fit(self.compressor.transform(X), y)
        self.n_fits_ += 1

    def partial_fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._fitted:
            self.compressor.fit(X)
            self.cluster.fit(self.compressor.transform(X), y)
            self._fitted = True
        else:
            self.cluster.partial_fit(self.compressor.transform(X), y)

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("AESFSOINNModel not fitted")
        return self.cluster.predict(self.compressor.transform(np.asarray(X)))

    def predict_and_adapt(self, X):
        if not self._fitted:
            raise RuntimeError("AESFSOINNModel not fitted")
        Z = self.compressor.transform(np.asarray(X))
        return self.cluster.predict_and_adapt(Z)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "scaler": self.compressor._scaler,
                "mlp": self.compressor._mlp,
                "latent_dim": self.compressor.latent_dim,
                "prototypes": self.cluster.prototypes,
                "labels": self.cluster.labels,
                "counts": self.cluster.counts,
                "edges": [tuple(sorted(e)) for e in self.cluster.edges],
                "edge_ages": list(self.cluster.edges.values()),
                "threshold": self.cluster.threshold,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_model(cfg: Any) -> AESFSOINNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    gate = _cfg_get(model_cfg, "denoising_gate", None)
    latent = _cfg_get(gate, "latent_dim", None)
    if latent is None:
        latent = _cfg_get(model_cfg, "latent_dim", 32)
    hunter = _cfg_get(model_cfg, "zeroday_hunter", None)
    thr = _cfg_get(hunter, "similarity_threshold", 0.5)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return AESFSOINNModel(
        latent_dim=int(latent),
        similarity_threshold=float(thr),
        random_state=int(seed),
    )
