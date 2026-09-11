"""Distillation baseline — true knowledge distillation (GPU-aware), ticket 04.

Per ADR 0005:
- **Teacher** = the DNN baseline architecture (torch MLP, hidden (128, 64)),
  trained normally on hard labels.
- **Student** = a strictly smaller torch MLP (hidden (64, 32)).
- The student is trained on a combined loss: ``alpha * T^2 * KL(teacher_T ||
  student_T) + (1 - alpha) * CE(student, y)`` with temperature ``T`` — real
  soft-target training, not the retired argmax-collapsed approximation.

``predict`` uses the student only. Not incremental. Torch seeding and device
selection use the shared helpers in ``models/_shared/baseline_helpers.py`` —
the Runner itself never imports torch (ADR 0002, grep-guarded).
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import torch
from torch import nn

from models.baselines.DNN.main import _MLP, _cfg_get


class DistillationModel:
    """Non-incremental teacher-student model exposing fit / predict / save."""

    supports_incremental = False

    def __init__(
        self,
        temperature: float = 2.0,
        alpha: float = 0.5,
        random_state: int = 42,
        teacher_hidden: tuple[int, ...] = (128, 64),
        student_hidden: tuple[int, ...] = (64, 32),
        epochs: int = 30,
        lr: float = 1e-3,
        batch_size: int = 256,
    ):
        """Build the model on ``get_device()``'s selection and print the check.

        Args:
            temperature (float): KD softening temperature T.
            alpha (float): weight of the KD (soft) term vs the CE (hard) term.
            random_state (int): seed for torch init (injected per seed by the
                Runner via cfg.training.random_seed).
            teacher_hidden (tuple[int, ...]): teacher hidden widths (DNN arch).
            student_hidden (tuple[int, ...]): student hidden widths (smaller).
            epochs (int): training epochs per fit call (teacher and student).
            lr (float): Adam learning rate.
            batch_size (int): minibatch size.
        """
        from models._shared.baseline_helpers import get_device, print_device_check

        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.teacher_hidden = tuple(int(h) for h in teacher_hidden)
        self.student_hidden = tuple(int(h) for h in student_hidden)
        self.epochs = int(epochs)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.device = get_device()
        print_device_check("Distillation", self.device)
        self._teacher: _MLP | None = None
        self._student: _MLP | None = None
        self._n_classes: int | None = None
        self._n_features: int | None = None
        self._fitted = False

    @classmethod
    def from_config(cls, cfg: Any) -> "DistillationModel":
        """Build a DistillationModel from the merged config.

        Args:
            cfg (Any): merged config node; reads ``model.distillation.*`` and
                ``training.random_seed``.

        Returns:
            DistillationModel: configured model on the selected device.
        """
        model_cfg = _cfg_get(cfg, "model", None)
        d_cfg = _cfg_get(model_cfg, "distillation", None)
        training = _cfg_get(cfg, "training", None)
        seed = _cfg_get(training, "random_seed", 42)
        return cls(
            temperature=float(_cfg_get(d_cfg, "temperature", 2.0)),
            alpha=float(_cfg_get(d_cfg, "alpha", 0.5)),
            random_state=int(seed),
            epochs=int(_cfg_get(d_cfg, "epochs", 30)),
            lr=float(_cfg_get(d_cfg, "lr", 1e-3)),
            batch_size=int(_cfg_get(d_cfg, "batch_size", 256)),
        )

    def _build(self, n_features: int, n_classes: int) -> None:
        """Build teacher + student for the given shapes, seeded and on-device.

        Args:
            n_features (int): input feature dimension.
            n_classes (int): number of classes seen in training.
        """
        from models._shared.baseline_helpers import seed_torch

        seed_torch(self.random_state)
        self._n_features = n_features
        self._n_classes = n_classes
        self._teacher = _MLP(n_features, self.teacher_hidden, n_classes).to(
            self.device
        )
        self._student = _MLP(n_features, self.student_hidden, n_classes).to(
            self.device
        )

    def _minibatches(self, n: int):
        """Yield random index batches over ``n`` samples.

        Args:
            n (int): dataset size.

        Yields:
            torch.Tensor: 1-D index tensor for one minibatch.
        """
        perm = torch.randperm(n, device=self.device)
        for start in range(0, n, self.batch_size):
            yield perm[start : start + self.batch_size]

    def fit(self, X, y, X_val=None, y_val=None):
        """Train the teacher on hard labels, then the student on KD loss.

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
            self._teacher is None
            or self._n_features != X_np.shape[1]
            or self._n_classes != n_classes
        ):
            self._build(X_np.shape[1], n_classes)
        assert self._teacher is not None and self._student is not None
        Xt = torch.from_numpy(X_np).to(self.device)
        yt = torch.from_numpy(y_np).to(self.device)
        n = len(Xt)
        T = self.temperature

        # ---- Phase 1: teacher on hard labels ----------------------------
        opt_t = torch.optim.Adam(self._teacher.parameters(), lr=self.lr)
        ce = nn.CrossEntropyLoss()
        self._teacher.train()
        for _ in range(self.epochs):
            for idx in self._minibatches(n):
                opt_t.zero_grad()
                loss = ce(self._teacher(Xt[idx]), yt[idx])
                loss.backward()
                opt_t.step()
        self._teacher.eval()

        # ---- Phase 2: student on alpha * T^2 * KL + (1-alpha) * CE -------
        opt_s = torch.optim.Adam(self._student.parameters(), lr=self.lr)
        kl = nn.KLDivLoss(reduction="batchmean")
        self._student.train()
        with torch.no_grad():
            teacher_probs = torch.softmax(self._teacher(Xt) / T, dim=1)
        for _ in range(self.epochs):
            for idx in self._minibatches(n):
                opt_s.zero_grad()
                student_logp = torch.log_softmax(self._student(Xt[idx]) / T, dim=1)
                soft = kl(student_logp, teacher_probs[idx]) * (T * T)
                hard = ce(self._student(Xt[idx]), yt[idx])
                loss = self.alpha * soft + (1.0 - self.alpha) * hard
                loss.backward()
                opt_s.step()
        self._student.eval()
        self._fitted = True

    def predict(self, X):
        """Predict hard integer class indices with the student (pure read).

        Args:
            X: feature matrix.

        Returns:
            np.ndarray: 1-D integer class indices.
        """
        if not self._fitted or self._student is None:
            raise RuntimeError("DistillationModel not fitted")
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        with torch.no_grad():
            logits = self._student(Xt)
        return logits.argmax(dim=1).cpu().numpy().astype(np.int64)

    def save(self, path):
        """Persist teacher + student state_dicts and KD metadata.

        Args:
            path: destination file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "teacher_state_dict": (
                    self._teacher.state_dict() if self._teacher else None
                ),
                "student_state_dict": (
                    self._student.state_dict() if self._student else None
                ),
                "teacher_hidden": self.teacher_hidden,
                "student_hidden": self.student_hidden,
                "temperature": self.temperature,
                "alpha": self.alpha,
                "n_features": self._n_features,
                "n_classes": self._n_classes,
                "random_state": self.random_state,
                "fitted": self._fitted,
            },
            path,
        )


def create_model(cfg: Any) -> DistillationModel:
    """Build a true-KD Distillation model from merged config.

    Args:
        cfg: merged config node; reads ``model.distillation.*`` and
            ``training.random_seed``.

    Returns:
        DistillationModel: configured non-incremental KD model on the device.
    """
    return DistillationModel.from_config(cfg)
