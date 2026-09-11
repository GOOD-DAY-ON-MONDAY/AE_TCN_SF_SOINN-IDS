"""Shared TCN module — real gradient-trained causal dilated Conv1d network.

Real torch implementation per ADR 0006: WaveNet-style residual blocks
(``nn.Conv1d``, left-padding for causality, BatchNorm + GELU + dropout),
trained end-to-end with a linear classifier head via Adam on ``get_device()``.
Per-sample temporal context comes from sliding windows over consecutive rows
(window ``T`` ending at each sample); pooled features keep the
``(N, channels[-1])`` numpy contract so all four consumers compose this
unchanged. Seeding happens per-instance via ``seed_torch`` — no import-time
weight caching (Runner seed-fix bug class, see ADR 0006).
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


def _windows(X: np.ndarray, window_size: int) -> np.ndarray:
    """Build per-sample temporal windows ending at each row.

    Row ``i`` gets the window ``X[i-w+1 : i+1]`` (left zero-padded), so every
    original sample keeps its index and gains temporal context from the
    preceding ``w-1`` rows. Output shape ``(n, w, d)``.
    """
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    w = int(window_size)
    if n < w:
        pad = np.zeros((w - n, d), dtype=np.float32)
        X = np.concatenate([pad, X], axis=0)
        n = w
    pad = np.zeros((w - 1, d), dtype=np.float32)
    Xp = np.concatenate([pad, X], axis=0)
    out = np.empty((n, w, d), dtype=np.float32)
    for i in range(n):
        out[i] = Xp[i : i + w]
    return out


class _TCNNet:
    """Lazily-constructed torch network (torch imported only when needed)."""

    def __init__(
        self,
        in_dim: int,
        channels: list[int],
        kernel_size: int,
        dilations: list[int],
        dropout: float,
        n_classes: int,
    ):
        import torch.nn as nn

        class TCNNet(nn.Module):
            def __init__(self):
                super().__init__()
                blocks = []
                prev = in_dim
                for out_ch, d in zip(channels, dilations, strict=False):
                    pad = (kernel_size - 1) * d
                    blocks.append(
                        nn.Sequential(
                            nn.ConstantPad1d((pad, 0), 0.0),
                            nn.Conv1d(prev, out_ch, kernel_size, dilation=d),
                            nn.BatchNorm1d(out_ch),
                            nn.GELU(),
                            nn.Dropout(dropout),
                        )
                    )
                    prev = out_ch
                self.blocks = nn.ModuleList(blocks)
                # Residual projections: 1x1 conv when channel counts differ.
                res = []
                c_in = in_dim
                for c_out in channels:
                    res.append(
                        nn.Conv1d(c_in, c_out, 1)
                        if c_in != c_out
                        else nn.Identity()
                    )
                    c_in = c_out
                self.residuals = nn.ModuleList(res)
                self.head = nn.Linear(channels[-1], n_classes)

            def features(self, x):  # x: (N, T, C_in)
                h = x.transpose(1, 2)  # (N, C, T)
                for block, res in zip(self.blocks, self.residuals, strict=True):
                    identity = res(h)
                    h = block(h)
                    h = h + identity
                return h.mean(dim=2)  # (N, C_last) mean-pool over time

            def forward(self, x):
                return self.head(self.features(x))

        self.net = TCNNet()


class TCNExtractor:
    """Gradient-trained causal dilated TCN feature extractor (ADR 0006)."""

    def __init__(
        self,
        in_dim: int = 121,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        random_state: int = 42,
        window_size: int = 8,
        dropout: float = 0.1,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
    ):
        from models._shared.baseline_helpers import seed_torch

        self.in_dim = int(in_dim)
        self.channels = [int(c) for c in channels]
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        self.random_state = int(random_state)
        self.window_size = int(window_size)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        seed_torch(self.random_state)
        self._holder: _TCNNet | None = None
        self._n_classes = 2
        self._fitted = False

    # -- torch plumbing ----------------------------------------------------
    def _ensure_net(self, n_classes: int):
        if self._holder is None or self._n_classes != n_classes:
            self._n_classes = int(n_classes)
            self._holder = _TCNNet(
                in_dim=self.in_dim,
                channels=self.channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                dropout=self.dropout,
                n_classes=self._n_classes,
            )
            import torch

            from models._shared.baseline_helpers import get_device, print_device_check

            self._device = get_device()
            self._holder.net.to(self._device)
            print_device_check("TCN", self._device)

    @property
    def net(self):
        if self._holder is None:
            self._ensure_net(self._n_classes)
        assert self._holder is not None
        return self._holder.net

    # -- training ----------------------------------------------------------
    def fit(self, X, y) -> "TCNExtractor":
        """Train conv blocks + linear head end-to-end (cross-entropy)."""
        import torch
        import torch.nn as nn

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        classes = np.unique(y)
        label_map = {c: i for i, c in enumerate(classes)}
        self.classes_ = classes
        self._ensure_net(len(classes))
        net = self.net
        dev = self._device
        W = _windows(X, self.window_size)
        Xt = torch.from_numpy(W)
        yt = torch.from_numpy(
            np.asarray([label_map[v] for v in y], dtype=np.int64)
        )
        opt = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()
        net.train()
        n = len(Xt)
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, self.batch_size):
                idx = perm[s : s + self.batch_size]
                xb, yb = Xt[idx].to(dev), yt[idx].to(dev)
                opt.zero_grad()
                loss = loss_fn(net(xb), yb)
                loss.backward()
                opt.step()
        self._fitted = True
        return self

    # -- inference ---------------------------------------------------------
    def transform(self, X) -> np.ndarray:
        """Pooled TCN features, numpy in → numpy out (consumer contract)."""
        import torch

        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        net = self.net
        was_training = net.training
        net.eval()
        outs = []
        bs = max(self.batch_size, 256)
        with torch.no_grad():
            for s in range(0, n, bs):
                W = _windows(X[s : s + bs], self.window_size)
                outs.append(
                    net.features(torch.from_numpy(W).to(self._device))
                    .cpu()
                    .numpy()
                )
        if was_training:
            net.train()
        pooled = np.concatenate(outs, axis=0)
        assert pooled.shape == (n, self.channels[-1])
        return pooled.astype(np.float32)

    def decision_logits(self, X) -> np.ndarray:
        """Head logits (used by TCNModel.predict)."""
        import torch

        X = np.asarray(X, dtype=np.float32)
        net = self.net
        net.eval()
        W = _windows(X, self.window_size)
        with torch.no_grad():
            logits = net(torch.from_numpy(W).to(self._device)).cpu().numpy()
        return logits


class TCNModel:
    """Runner-contract wrapper: end-to-end trained TCN with linear head."""

    supports_incremental = False

    def __init__(
        self,
        channels: list[int] | tuple[int, ...] = (64, 64, 32),
        kernel_size: int = 3,
        dilations: list[int] | tuple[int, ...] = (1, 2, 4, 8),
        random_state: int = 42,
        window_size: int = 8,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
    ):
        self.channels = [int(c) for c in channels]
        self.kernel_size = int(kernel_size)
        self.dilations = [int(d) for d in dilations]
        self.random_state = int(random_state)
        self.window_size = int(window_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.extractor: TCNExtractor | None = None
        self.n_fits_ = 0

    def _ensure_extractor(self, in_dim: int):
        if self.extractor is None:
            self.extractor = TCNExtractor(
                in_dim=in_dim,
                channels=self.channels,
                kernel_size=self.kernel_size,
                dilations=self.dilations,
                random_state=self.random_state,
                window_size=self.window_size,
                epochs=self.epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

    def fit(self, X, y, X_val=None, y_val=None):
        X = np.asarray(X)
        y = np.asarray(y)
        self._ensure_extractor(X.shape[1])
        assert self.extractor is not None
        self.extractor.fit(X, y)
        self.n_fits_ += 1

    def predict(self, X):
        assert self.extractor is not None, "TCNModel not fitted"
        logits = self.extractor.decision_logits(np.asarray(X))
        return np.asarray(
            [self.extractor.classes_[i] for i in logits.argmax(axis=1)], dtype=int
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self.extractor is not None
        joblib.dump(
            {
                "state_dict": self.extractor.net.state_dict(),
                "classes": self.extractor.classes_,
                "channels": self.channels,
                "kernel_size": self.kernel_size,
                "dilations": self.dilations,
                "window_size": self.window_size,
                "in_dim": self.extractor.in_dim,
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
    dropout = _cfg_get(eng, "dropout", 0.1)
    window = _cfg_get(eng, "window_size", 8)
    training = _cfg_get(cfg, "training", None)
    seed = _cfg_get(training, "random_seed", 42)
    epochs = _cfg_get(training, "epochs", 20)
    batch = _cfg_get(training, "batch_size", 256)
    lr = _cfg_get(training, "learning_rate", 1e-3)
    return TCNModel(
        channels=list(channels),
        kernel_size=int(kernel),
        dilations=list(dilations),
        random_state=int(seed),
        window_size=int(window),
        epochs=int(epochs),
        batch_size=int(batch),
        learning_rate=float(lr),
    )
