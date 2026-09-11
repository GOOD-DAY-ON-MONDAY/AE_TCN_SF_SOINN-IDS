"""Shared AE module — single reusable compression behavior.

Real torch encoder->bottleneck->decoder per ADR 0008: Linear stacks
(128->64->latent / mirror), MSE reconstruction, Adam(1e-3), Gaussian input
noise sigma=0.01 (denoising), trained on ``get_device()``. PCA wrapper stays the
deterministic control (ADR 0003). Thin wrappers compose this.
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


class _AENet:
    """Encoder→bottleneck→decoder torch module (built lazily to keep import light)."""

    def __init__(self, in_dim: int, latent_dim: int):
        from torch import nn

        self.net = nn.Sequential(
            # encoder
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
            # decoder (mirror)
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, in_dim),
        )

    def encode(self, x):
        for layer in self.net[:5]:
            x = layer(x)
        return x

    def forward(self, x):
        return self.net(x)


class AECompressor:
    """Denoising-style compressor: scaler plus torch AE reconstruction."""

    def __init__(
        self, latent_dim: int = 32, random_state: int = 42, max_iter: int = 20
    ):
        from sklearn.preprocessing import StandardScaler

        from models._shared.baseline_helpers import get_device, print_device_check

        self.latent_dim = int(latent_dim)
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self._scaler = StandardScaler()
        self._fitted = False
        self._device = get_device()
        print_device_check("AE", self._device)
        self._holder = None
        self.net = None  # built on fit once in_dim is known

    def _build(self, in_dim: int):
        import torch

        from models._shared.baseline_helpers import seed_torch

        seed_torch(self.random_state)
        self._holder = _AENet(in_dim=in_dim, latent_dim=self.latent_dim)
        self.net = self._holder.net.to(self._device)
        self._opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)

    def fit(self, X) -> AECompressor:
        import torch

        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.fit_transform(X)
        self._build(Xs.shape[1])
        self.net.train()
        dataset = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
        batch_size = 256
        n = dataset.shape[0]
        for _ in range(self.max_iter):
            perm = torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                batch = dataset[idx]
                noisy = batch + 0.01 * torch.randn_like(batch)
                recon = self.net(noisy)
                loss = torch.nn.functional.mse_loss(recon, batch)
                self._opt.zero_grad()
                loss.backward()
                self._opt.step()
        self._fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        import torch

        if not self._fitted:
            raise RuntimeError("AECompressor not fitted")
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.transform(X)
        with torch.no_grad():
            latents = self._holder.encode(
                torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
            )
        return latents.cpu().numpy().astype(np.float32)

    def reconstruction_loss(self, X) -> float:
        """Mean MSE reconstruction loss on clean inputs (test seam)."""
        import torch

        if not self._fitted:
            raise RuntimeError("AECompressor not fitted")
        Xs = self._scaler.transform(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            inp = torch.as_tensor(Xs, dtype=torch.float32, device=self._device)
            recon = self.net(inp)
            return float(torch.nn.functional.mse_loss(recon, inp).item())

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "state_dict": self.net.state_dict(),
                "scaler": self._scaler,
                "latent_dim": self.latent_dim,
                "in_dim": self.net[0].in_features,
            },
            path,
        )

    def load(self, path) -> AECompressor:
        payload = joblib.load(Path(path))
        self._scaler = payload["scaler"]
        self._build(int(payload["in_dim"]))
        self.net.load_state_dict(payload["state_dict"])
        self._fitted = True
        return self


class AEModel:
    """Runner-contract wrapper: AE latent plus linear classifier."""

    supports_incremental = False

    def __init__(
        self, latent_dim: int = 32, random_state: int = 42, max_iter: int = 20
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
                "ae_state": self.compressor.net.state_dict(),
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
