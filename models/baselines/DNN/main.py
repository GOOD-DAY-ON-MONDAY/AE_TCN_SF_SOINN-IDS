"""DNN baseline — real torch feedforward network (GPU-aware), per ADR 0005.

Implements the v1 contract: ``create_model(cfg)`` factory, ``fit(X, y,
X_val=None, y_val=None)`` (re-callable), ``predict(X)`` returning hard integer
class indices (pure read), and ``save(path)``. Not incremental.

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


class _MLP(nn.Module):
    """Feedforward network: input -> hidden layers -> n_classes logits."""

    def __init__(self, n_features: int, hidden: tuple[int, ...], n_classes: int):
        """Build the MLP stack.

        Args:
            n_features (int): input feature dimension.
            hidden (tuple[int, ...]): hidden layer widths.
            n_classes (int): number of output classes.
        """
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, int(h)), nn.ReLU()]
            prev = int(h)
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for ``x``.

        Args:
            x (torch.Tensor): (batch, n_features) float tensor.

        Returns:
            torch.Tensor: (batch, n_classes) logits.
        """
        return self.net(x)


class DNNModel:
    """Non-incremental torch MLP exposing fit / predict / save."""

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        random_state: int = 42,
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
    ):
        """Build the model on ``get_device()``'s selection and print the check.

        Args:
            hidden (tuple[int, ...]): hidden layer widths.
            random_state (int): seed for torch init (injected per seed by the
                Runner via cfg.training.random_seed).
            epochs (int): training epochs per fit call.
            lr (float): Adam learning rate.
            batch_size (int): minibatch size.
        """
        from models._shared.baseline_helpers import get_device, print_device_check

        self.hidden = tuple(int(h) for h in hidden)
        self.random_state = int(random_state)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = get_device()
        print_device_check("DNN", self.device)
        self._net: _MLP | None = None
        self._n_classes: int | None = None
        self._n_features: int | None = None
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Any) -> "DNNModel":
        """Build a DNNModel from the merged config.

        Args:
            cfg (Any): merged config node; reads ``model.dnn.hidden`` and
                ``training.random_seed``.

        Returns:
            DNNModel: configured model on the selected device.
        """
        model_cfg = _cfg_get(cfg, "model", None)
        dnn_cfg = _cfg_get(model_cfg, "dnn", None)
        hidden = _cfg_get(dnn_cfg, "hidden", (128, 64))
        training = _cfg_get(cfg, "training", None)
        seed = _cfg_get(training, "random_seed", 42)
        epochs = _cfg_get(dnn_cfg, "epochs", 30)
        lr = _cfg_get(dnn_cfg, "lr", 1e-3)
        batch_size = _cfg_get(dnn_cfg, "batch_size", 256)
        return cls(
            hidden=tuple(hidden),
            random_state=int(seed),
            epochs=int(epochs),
            lr=float(lr),
            batch_size=int(batch_size),
        )

    def _build(self, n_features: int, n_classes: int) -> None:
        """(Re)build the network for the given shapes, seeded and on-device.

        Args:
            n_features (int): input feature dimension.
            n_classes (int): number of classes seen in training.
        """
        from models._shared.baseline_helpers import seed_torch

        seed_torch(self.random_state)
        self._n_features = n_features
        self._n_classes = n_classes
        self._net = _MLP(n_features, self.hidden, n_classes).to(self.device)

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the MLP with Adam + CrossEntropyLoss on the selected device.

        Args:
            X: training feature matrix.
            y: training integer labels.
            X_val: unused validation features (signature compat).
            y_val: unused validation labels (signature compat).
        """
        del X_val, y_val
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64).reshape(-1)
        n_classes = int(y_np.max()) + 1
        if (
            self._net is None
            or self._n_features != X_np.shape[1]
            or self._n_classes != n_classes
        ):
            self._build(X_np.shape[1], n_classes)
        assert self._net is not None
        Xt = torch.from_numpy(X_np).to(self.device)
        yt = torch.from_numpy(y_np).to(self.device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self._net.train()
        n = len(Xt)
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
        self._net.eval()
        self._fitted = True

    def predict(self, X):
        """Predict hard integer class indices (pure read, eval mode, no_grad).

        Args:
            X: feature matrix.

        Returns:
            np.ndarray: 1-D integer class indices.
        """
        if not self._fitted or self._net is None:
            raise RuntimeError("DNNModel not fitted")
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._net(Xt)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def save(self, path):
        """Persist the state_dict + metadata, creating parent dirs.

        Args:
            path: destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._net.state_dict() if self._net else None,
                "hidden": self.hidden,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "random_state": self.random_state,
                "fitted": self._fitted,
            },
            path,
        )


def create_model(cfg: Any) -> DNNModel:
    """Build a torch DNN from merged config.

    Args:
        cfg: merged config node; reads ``model.dnn.*`` and
            ``training.random_seed``.

    Returns:
        DNNModel: configured non-incremental torch MLP on the selected device.
    """
    return DNNModel.from_config(cfg)
