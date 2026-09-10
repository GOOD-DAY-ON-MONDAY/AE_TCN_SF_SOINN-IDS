"""Full proposed AE plus TCN plus SF-SOINN — headline pipeline."""

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


class AETCNSFSOINNModel:
    """Shared AE latents refined by TCN then clustered by SF-SOINN."""

    supports_incremental = True

    def __init__(
        self,
        latent_dim: int = 32,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        similarity_threshold: float = 0.5,
        random_state: int = 42,
        max_iter: int = 200,
    ):
        from models._shared.ae import AECompressor
        from models._shared.sfsoinn import SFSOINNCluster
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
        self.cluster = SFSOINNCluster(
            similarity_threshold=float(similarity_threshold)
        )
        self._fitted = False
        self.n_fits_ = 0

    def _features(self, X) -> np.ndarray:
        Z = self.compressor.transform(np.asarray(X))
        return self.extractor.transform(Z)

    def fit(self, X, y, X_val=None, y_val=None):
        del X_val, y_val
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._fitted:
            self.compressor.fit(X)
            self.cluster.fit(self._features(X), y)
            self._fitted = True
        else:
            self.cluster.partial_fit(self._features(X), y)
        self.n_fits_ += 1

    def partial_fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if not self._fitted:
            self.compressor.fit(X)
            self.cluster.fit(self._features(X), y)
            self._fitted = True
        else:
            self.cluster.partial_fit(self._features(X), y)

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("AETCNSFSOINNModel not fitted")
        return self.cluster.predict(self._features(np.asarray(X)))

    def predict_and_adapt(self, X):
        if not self._fitted:
            raise RuntimeError("AETCNSFSOINNModel not fitted")
        X = np.asarray(X, dtype=np.float32)
        Z = self._features(X)
        out: list[int] = []
        next_label = max(self.cluster.labels, default=-1) + 1
        for zi in Z:
            idx, dist = self.cluster._nearest(zi)
            if idx < 0 or dist > self.cluster.threshold:
                self.cluster.prototypes.append(zi.astype(np.float32))
                self.cluster.labels.append(int(next_label))
                self.cluster.counts.append(1)
                out.append(int(next_label))
                next_label += 1
            else:
                out.append(self.cluster.labels[idx])
        return np.asarray(out, dtype=int)

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
                "prototypes": self.cluster.prototypes,
                "labels": self.cluster.labels,
                "counts": self.cluster.counts,
                "threshold": self.cluster.threshold,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_model(cfg: Any) -> AETCNSFSOINNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    gate = _cfg_get(model_cfg, "denoising_gate", None)
    latent = _cfg_get(gate, "latent_dim", None)
    if latent is None:
        latent = _cfg_get(model_cfg, "latent_dim", 32)
    eng = _cfg_get(model_cfg, "temporal_engine", None)
    channels = _cfg_get(eng, "channels", [64, 64, 32])
    kernel = _cfg_get(eng, "kernel_size", 3)
    dilations = _cfg_get(eng, "dilations", [1, 2, 4, 8])
    hunter = _cfg_get(model_cfg, "zeroday_hunter", None)
    thr = _cfg_get(hunter, "similarity_threshold", 0.5)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return AETCNSFSOINNModel(
        latent_dim=int(latent),
        channels=list(channels),
        kernel_size=int(kernel),
        dilations=list(dilations),
        similarity_threshold=float(thr),
        random_state=int(seed),
    )
