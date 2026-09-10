"""PCA plus SF-SOINN ablation — linear control with clustering only, no TCN."""

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


class PCASFSOINNModel:
    """Shared PCA latents clustered by shared SF-SOINN prototypes."""

    supports_incremental = True

    def __init__(
        self,
        n_components: int = 32,
        similarity_threshold: float = 0.5,
        random_state: int = 42,
    ):
        from models._shared.pca import PCACompressor
        from models._shared.sfsoinn import SFSOINNCluster

        self.compressor = PCACompressor(
            n_components=int(n_components),
            random_state=int(random_state),
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
            raise RuntimeError("PCASFSOINNModel not fitted")
        return self.cluster.predict(self.compressor.transform(np.asarray(X)))

    def predict_and_adapt(self, X):
        if not self._fitted:
            raise RuntimeError("PCASFSOINNModel not fitted")
        X = np.asarray(X, dtype=np.float32)
        Z = self.compressor.transform(X)
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
                "pca": self.compressor._pca,
                "n_components": self.compressor.n_components,
                "prototypes": self.cluster.prototypes,
                "labels": self.cluster.labels,
                "counts": self.cluster.counts,
                "threshold": self.cluster.threshold,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_model(cfg: Any) -> PCASFSOINNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    pca_cfg = _cfg_get(model_cfg, "pca", None)
    n_comp = _cfg_get(pca_cfg, "n_components", None)
    if n_comp is None:
        gate = _cfg_get(model_cfg, "denoising_gate", None)
        n_comp = _cfg_get(gate, "latent_dim", 32)
    hunter = _cfg_get(model_cfg, "zeroday_hunter", None)
    thr = _cfg_get(hunter, "similarity_threshold", 0.5)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return PCASFSOINNModel(
        n_components=int(n_comp),
        similarity_threshold=float(thr),
        random_state=int(seed),
    )
