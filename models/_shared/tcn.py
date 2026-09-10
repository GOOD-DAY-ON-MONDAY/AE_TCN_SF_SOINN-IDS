"""Shared TCN module — reusable sequence modeling on torch-ready harness.

Tracer-bullet numpy implementation without torch so smoke stays green.
Fixed random dilated causal filters honor channels, kernel, and dilations;
classifier trains on pooled features. Torch migration keeps this interface.
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


def _causal_conv1d(x: np.ndarray, weight: np.ndarray, dilation: int) -> np.ndarray:
    n, _, t = x.shape
    _, _, k = weight.shape
    pad = (k - 1) * dilation
    xp = np.pad(x, ((0, 0), (0, 0), (pad, 0)))
    out = np.zeros((n, weight.shape[0], t), dtype=np.float32)
    for ki in range(k):
        start = pad - ki * dilation
        shifted = xp[:, :, start : start + t]
        out += np.einsum("nct,oc->not", shifted.astype(np.float32), weight[:, :, ki])
    return out


class TCNExtractor:
    """Fixed random dilated causal feature extractor."""

    def __init__(
        self,
        in_dim: int = 121,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        random_state: int = 42,
    ):
        self.in_dim = int(in_dim)
        self.channels = [int(c) for c in channels]
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        self.random_state = int(random_state)
        rng = np.random.default_rng(self.random_state)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        prev = self.in_dim
        for out_ch in self.channels:
            scale = np.sqrt(2.0 / (prev * self.kernel_size))
            w = (
                rng.normal(size=(out_ch, prev, self.kernel_size)).astype(np.float32)
                * scale
            )
            b = np.zeros((out_ch,), dtype=np.float32)
            self.weights.append(w)
            self.biases.append(b)
            prev = out_ch

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        seq = X[:, None, :]
        h = seq
        for i, w in enumerate(self.weights):
            d = self.dilations[i % len(self.dilations)]
            h = _causal_conv1d(h, w, d) + self.biases[i][None, :, None]
            h = np.maximum(0, h)
        pooled = h.mean(axis=2)
        assert pooled.shape == (n, self.channels[-1])
        return pooled.astype(np.float32)


class TCNModel:
    """Runner-contract wrapper: TCN features plus classifier."""

    supports_incremental = False

    def __init__(
        self,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        random_state: int = 42,
    ):
        from sklearn.linear_model import LogisticRegression

        self.channels = [int(c) for c in channels]
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        self.random_state = int(random_state)
        self.extractor: TCNExtractor | None = None
        self._clf = LogisticRegression(max_iter=500)
        self.n_fits_ = 0

    def _ensure_extractor(self, in_dim: int):
        if self.extractor is None:
            self.extractor = TCNExtractor(
                in_dim=in_dim,
                channels=self.channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                random_state=self.random_state,
            )

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self._ensure_extractor(X.shape[1])
        assert self.extractor is not None
        Z = self.extractor.transform(X)
        self._clf.fit(Z, y)
        self.n_fits_ += 1

    def predict(self, X):
        assert self.extractor is not None, "TCNModel not fitted"
        Z = self.extractor.transform(np.asarray(X))
        return np.asarray(self._clf.predict(Z), dtype=int)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "weights": self.extractor.weights if self.extractor else None,
                "biases": self.extractor.biases if self.extractor else None,
                "clf": self._clf,
                "channels": self.channels,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_tcn_model(cfg: Any) -> TCNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    eng = _cfg_get(model_cfg, "temporal_engine", None)
    channels = _cfg_get(eng, "channels", [64, 64, 32])
    kernel = _cfg_get(eng, "kernel_size", 3)
    dilations = _cfg_get(eng, "dilations", [1, 2, 4, 8])
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    return TCNModel(
        channels=list(channels),
        kernel_size=int(kernel),
        dilations=list(dilations),
        random_state=int(seed),
    )
