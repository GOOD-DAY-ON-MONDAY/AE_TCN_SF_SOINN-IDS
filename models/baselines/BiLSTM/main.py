"""BiLSTM baseline — real torch bidirectional LSTM (GPU-aware), ticket 03.

Per-flow prediction: each flow is represented by the BiLSTM encoding of its
temporal context window (the ``window_size`` flows ending at it, stride 1),
so ``predict(X)`` stays per-flow (1-D hard integer indices, pure read).

``supports_incremental = True``: ``partial_fit`` takes continued SGD steps on
new data. Catastrophic forgetting under ``partial_fit`` is a *feature* of this
baseline — no replay or regularization is added to mask it.

Torch seeding and device selection use the shared helpers in
``models/_shared/baseline_helpers.py`` — the Runner itself never imports torch
(ADR 0002, grep-guarded).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import torch
from torch import nn


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


class _BiLSTMNet(nn.Module):
    """Bidirectional LSTM encoder + linear classification head."""

    def __init__(self, n_features: int, hidden: int, n_classes: int):
        """Build the BiLSTM + head.

        Args:
            n_features (int): per-timestep feature dimension.
            hidden (int): LSTM hidden size per direction.
            n_classes (int): number of output classes.
        """
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=int(hidden),
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Linear(2 * int(hidden), n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-flow logits from windowed sequences.

        Args:
            x (torch.Tensor): (batch, window, n_features) float tensor.

        Returns:
            torch.Tensor: (batch, n_classes) logits (last-timestep encoding).
        """
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class BiLSTMModel:
    """Incremental torch BiLSTM exposing fit / partial_fit / predict / save."""

    supports_incremental = True

    def __init__(
        self,
        window_size: int = 8,
        hidden: int = 32,
        random_state: int = 42,
        epochs: int = 10,
        lr: float = 1e-3,
        batch_size: int = 128,
        partial_fit_steps: int = 3,
    ):
        """Build the model on ``get_device()``'s selection and print the check.

        Args:
            window_size (int): temporal context window (flows) per prediction.
            hidden (int): LSTM hidden size per direction.
            random_state (int): seed for torch init (injected per seed by the
                Runner via cfg.training.random_seed).
            epochs (int): training epochs per fit call.
            lr (float): Adam learning rate.
            batch_size (int): minibatch size.
            partial_fit_steps (int): SGD steps per partial_fit call.
        """
        from models._shared.baseline_helpers import get_device, print_device_check

        self.window_size = int(window_size)
        self.hidden = int(hidden)
        self.random_state = int(random_state)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.partial_fit_steps = int(partial_fit_steps)
        self.device = get_device()
        print_device_check("BiLSTM", self.device)
        self._net: _BiLSTMNet | None = None
        self._scaler_mean: np.ndarray | None = None
        self._scaler_scale: np.ndarray | None = None
        self._n_classes: int | None = None
        self._n_features: int | None = None
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Any) -> "BiLSTMModel":
        """Build a BiLSTMModel from the merged config.

        Args:
            cfg (Any): merged config node; reads ``model.bilstm.*`` and
                ``training.random_seed``.

        Returns:
            BiLSTMModel: configured incremental model on the selected device.
        """
        model_cfg = _cfg_get(cfg, "model", None)
        b_cfg = _cfg_get(model_cfg, "bilstm", None)
        training = _cfg_get(cfg, "training", None)
        seed = _cfg_get(training, "random_seed", 42)
        return cls(
            window_size=int(_cfg_get(b_cfg, "window_size", 8)),
            hidden=int(_cfg_get(b_cfg, "hidden", 32)),
            random_state=int(seed),
            epochs=int(_cfg_get(b_cfg, "epochs", 10)),
            lr=float(_cfg_get(b_cfg, "lr", 1e-3)),
            batch_size=int(_cfg_get(b_cfg, "batch_size", 128)),
        )

    # -- internals ---------------------------------------------------------

    def _scale(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted scaler (fit-on-first-use semantics).

        Args:
            X (np.ndarray): raw feature matrix.

        Returns:
            np.ndarray: scaled feature matrix.
        """
        X = np.asarray(X, dtype=np.float32)
        if self._scaler_mean is None:
            self._scaler_mean = X.mean(axis=0)
            self._scaler_scale = X.std(axis=0)
            self._scaler_scale[self._scaler_scale == 0] = 1.0
        return (X - self._scaler_mean) / self._scaler_scale

    def _windows(self, Xs: np.ndarray) -> torch.Tensor:
        """Build stride-1 context windows ending at each flow (start-padded).

        Args:
            Xs (np.ndarray): scaled feature matrix (n, d).

        Returns:
            torch.Tensor: (n, window_size, d) float tensor on the device.
        """
        n, d = Xs.shape
        w = self.window_size
        pad = np.zeros((w - 1, d), dtype=np.float32)
        padded = np.concatenate([pad, Xs], axis=0)
        idx = np.arange(n)[:, None] + np.arange(w)[None, :]
        windows = padded[idx]  # (n, w, d)
        return torch.from_numpy(np.ascontiguousarray(windows)).to(self.device)

    def _build(self, n_features: int, n_classes: int) -> None:
        """(Re)build the network for the given shapes, seeded and on-device.

        Args:
            n_features (int): per-timestep feature dimension.
            n_classes (int): number of classes seen so far.
        """
        from models._shared.baseline_helpers import seed_torch

        seed_torch(self.random_state)
        self._n_features = n_features
        self._n_classes = n_classes
        self._net = _BiLSTMNet(n_features, self.hidden, n_classes).to(self.device)

    def _train_steps(self, Xt: torch.Tensor, yt: torch.Tensor, steps: int) -> None:
        """Run ``steps`` epochs of CE-loss SGD over the given tensors.

        Args:
            Xt (torch.Tensor): windowed inputs (n, w, d).
            yt (torch.Tensor): integer labels (n,).
            steps (int): number of epochs to run.
        """
        assert self._net is not None
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        n = len(Xt)
        self._net.train()
        for _ in range(steps):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self._net.eval()

    # -- contract ----------------------------------------------------------

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the BiLSTM on context windows of the training flows.

        Args:
            X: training feature matrix.
            y: training integer labels.
            X_val: unused validation features (signature compat).
            y_val: unused validation labels (signature compat).
        """
        del X_val, y_val
        Xs = self._scale(np.asarray(X, dtype=np.float32))
        y_np = np.asarray(y, dtype=np.int64).reshape(-1)
        n_classes = int(y_np.max()) + 1
        if (
            self._net is None
            or self._n_features != Xs.shape[1]
            or self._n_classes != n_classes
        ):
            self._build(Xs.shape[1], n_classes)
        Xt = self._windows(Xs)
        yt = torch.from_numpy(y_np).to(self.device)
        self._train_steps(Xt, yt, self.epochs)
        self._fitted = True

    def partial_fit(self, X, y):
        """Take continued SGD steps on new data (forgetting is expected).

        Args:
            X: new feature matrix.
            y: new integer labels.
        """
        if not self._fitted or self._net is None:
            # First contact behaves like fit.
            self.fit(X, y)
            return
        Xs = self._scale(np.asarray(X, dtype=np.float32))
        y_np = np.asarray(y, dtype=np.int64).reshape(-1)
        Xt = self._windows(Xs)
        yt = torch.from_numpy(y_np).to(self.device)
        self._train_steps(Xt, yt, self.partial_fit_steps)

    def predict(self, X):
        """Predict hard integer class indices per flow (pure read).

        Args:
            X: feature matrix.

        Returns:
            np.ndarray: 1-D integer class indices, one per flow.
        """
        if not self._fitted or self._net is None:
            raise RuntimeError("BiLSTMModel not fitted")
        Xs = self._scale(np.asarray(X, dtype=np.float32))
        with torch.no_grad():
            logits = self._net(self._windows(Xs))
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def predict_and_adapt(self, X):
        """Contract-only in v1: pure read, never called by the Runner loop.

        Args:
            X: feature matrix.

        Returns:
            np.ndarray: 1-D integer class indices.
        """
        return self.predict(X)

    def save(self, path):
        """Persist the state_dict + scaler/metadata, creating parent dirs.

        Args:
            path: destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict() if self._net else None,
                "window_size": self.window_size,
                "hidden": self.hidden,
                "scaler_mean": self._scaler_mean,
                "scaler_scale": self._scaler_scale,
                "n_classes": self._n_classes,
                "n_features": self._n_features,
                "random_state": self.random_state,
                "fitted": self._fitted,
            },
            path,
        )


def create_model(cfg: Any) -> BiLSTMModel:
    """Build a torch BiLSTM from merged config.

    Args:
        cfg: merged config node; reads ``model.bilstm.*`` and
            ``training.random_seed``.

    Returns:
        BiLSTMModel: configured incremental BiLSTM on the selected device.
    """
    return BiLSTMModel.from_config(cfg)
