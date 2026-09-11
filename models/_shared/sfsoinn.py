"""Shared SF-SOINN module — real incremental self-organizing network.

Literature-faithful SF-SOINN per ADR 0007 (Successive Forgetting SOINN,
SOINN family): two-distance similarity-thresholded node insertion, edge
insertion with age increment, edge-age pruning, win-count-based node deletion
(forgetting), and class-label propagation over edges.

Documented simplifications (ADR 0007): a single global
``similarity_threshold`` (second-neighbor threshold = 2×) replaces adaptive
per-node thresholds; no topological learning-rate decay (win-count deletion
covers the forgetting claim).

Incremental contract unchanged: ``predict`` is pure-read; adaptation happens
only in ``partial_fit`` / ``predict_and_adapt``. Clustering is CPU-bound
numpy by design (small prototype counts).
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


class SFSOINNCluster:
    """SF-SOINN prototype network with edges, aging, and forgetting."""

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        max_edge_age: int = 50,
        delete_period: int = 200,
        noise_threshold: int = 5,
    ):
        self.threshold = float(similarity_threshold)
        self.threshold2 = 2.0 * float(similarity_threshold)
        self.max_edge_age = int(max_edge_age)
        self.delete_period = int(delete_period)
        self.noise_threshold = int(noise_threshold)
        self.prototypes: list[np.ndarray] = []
        self.labels: list[int] = []  # -1 = unlabeled (propagated at predict)
        self.counts: list[int] = []  # win counts
        self.edges: dict[frozenset, int] = {}  # {node-pair: age}
        self._samples_seen = 0
        self._next_new_label = 0

    # -- topology helpers ---------------------------------------------------
    def _two_nearest(self, x: np.ndarray) -> tuple[int, float, int, float]:
        """Return (n1, d1, n2, d2): first and second nearest nodes."""
        if not self.prototypes:
            return -1, float("inf"), -1, float("inf")
        P = np.stack(self.prototypes)
        dists = np.linalg.norm(P - x, axis=1)
        order = np.argsort(dists)
        n1 = int(order[0])
        d1 = float(dists[n1])
        if len(order) < 2:
            return n1, d1, -1, float("inf")
        n2 = int(order[1])
        return n1, d1, n2, float(dists[n2])

    def _add_edge(self, a: int, b: int) -> None:
        if a == b or a < 0 or b < 0:
            return
        self.edges[frozenset((a, b))] = 0

    def _age_and_prune(self, winner: int) -> None:
        """Increment ages of the winner's other edges; prune expired ones."""
        to_delete = []
        for edge, age in self.edges.items():
            if winner in edge:
                new_age = age + 1
                if new_age > self.max_edge_age:
                    to_delete.append(edge)
                else:
                    self.edges[edge] = new_age
        for edge in to_delete:
            del self.edges[edge]

    def _delete_noisy_nodes(self) -> None:
        """Successive forgetting: remove low-win-count nodes and their edges."""
        if len(self.prototypes) <= 2:
            return
        keep = [i for i, c in enumerate(self.counts) if c >= self.noise_threshold]
        if len(keep) == len(self.prototypes):
            return
        remap = {old: new for new, old in enumerate(keep)}
        self.prototypes = [self.prototypes[i] for i in keep]
        self.labels = [self.labels[i] for i in keep]
        self.counts = [self.counts[i] for i in keep]
        self.edges = {
            frozenset(remap[n] for n in edge): age
            for edge, age in self.edges.items()
            if all(n in remap for n in edge)
        }

    def _propagate_labels(self) -> None:
        """Unlabeled nodes inherit the majority label of their component."""
        n = len(self.prototypes)
        adj: dict[int, set[int]] = {i: set() for i in range(n)}
        for edge in self.edges:
            a, b = tuple(edge)
            adj[a].add(b)
            adj[b].add(a)
        seen = set()
        for start in range(n):
            if start in seen:
                continue
            comp, stack = [], [start]
            while stack:
                u = stack.pop()
                if u in seen:
                    continue
                seen.add(u)
                comp.append(u)
                stack.extend(adj[u] - seen)
            labeled = [self.labels[i] for i in comp if self.labels[i] >= 0]
            if labeled:
                vals, counts = np.unique(labeled, return_counts=True)
                majority = int(vals[np.argmax(counts)])
                for i in comp:
                    if self.labels[i] < 0:
                        self.labels[i] = majority

    # -- core sample processing ---------------------------------------------
    def _process(self, x: np.ndarray, y: int | None) -> None:
        n1, d1, n2, d2 = self._two_nearest(x)
        self._samples_seen += 1
        if n1 < 0 or n2 < 0 or d1 > self.threshold or d2 > self.threshold2:
            # Node insertion: sample is beyond the similarity thresholds.
            if y is not None:
                label = int(y)
                self._next_new_label = max(self._next_new_label, label + 1)
            else:
                label = -1
            self.prototypes.append(x.astype(np.float32))
            self.labels.append(label)
            self.counts.append(0)
            return
        # Edge insertion between first and second nearest; age the rest.
        self._add_edge(n1, n2)
        self._age_and_prune(n1)
        # Winner update with SOINN learning rate 1/(wins+1).
        c = self.counts[n1]
        self.prototypes[n1] = (
            self.prototypes[n1] + (x - self.prototypes[n1]) / (c + 1)
        ).astype(np.float32)
        self.counts[n1] = c + 1
        if y is not None:
            self.labels[n1] = int(y)
            self._next_new_label = max(self._next_new_label, int(y) + 1)
        if self._samples_seen % self.delete_period == 0:
            self._delete_noisy_nodes()
            self._propagate_labels()

    # -- public API ----------------------------------------------------------
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y, strict=True):
            self._process(xi, int(yi))
        self._delete_noisy_nodes()
        self._propagate_labels()

    def partial_fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=int)
        for xi, yi in zip(X, y, strict=True):
            self._process(xi, int(yi))
        self._delete_noisy_nodes()
        self._propagate_labels()

    def predict(self, X) -> np.ndarray:
        """Pure read: nearest node's label (propagated if unlabeled)."""
        X = np.asarray(X, dtype=np.float32)
        self._propagate_labels()
        out = []
        for xi in X:
            n1, d1, _, _ = self._two_nearest(xi)
            if n1 < 0:
                out.append(0)
            else:
                out.append(self.labels[n1] if self.labels[n1] >= 0 else 0)
        return np.asarray(out, dtype=int)

    def predict_and_adapt(self, X) -> np.ndarray:
        """Predict and grow the network for novel inputs (zero-day path)."""
        X = np.asarray(X, dtype=np.float32)
        out = []
        for xi in X:
            n1, d1, _, _ = self._two_nearest(xi)
            if n1 < 0 or d1 > self.threshold:
                label = self._next_new_label
                self._next_new_label += 1
                self.prototypes.append(xi.astype(np.float32))
                self.labels.append(int(label))
                self.counts.append(1)
                out.append(int(label))
            else:
                out.append(self.labels[n1] if self.labels[n1] >= 0 else 0)
                self._process(xi, None)
        return np.asarray(out, dtype=int)


class SFSOINNModel:
    """Runner-contract incremental wrapper."""

    supports_incremental = True

    def __init__(self, similarity_threshold: float = 0.5):
        self.cluster = SFSOINNCluster(similarity_threshold=similarity_threshold)
        self.n_fits_ = 0
        self._fitted = False

    def fit(self, X, y, X_val=None, y_val=None):
        if not self._fitted:
            self.cluster.fit(X, y)
            self._fitted = True
        else:
            self.cluster.partial_fit(X, y)
        self.n_fits_ += 1

    def partial_fit(self, X, y):
        if not self._fitted:
            self.cluster.fit(X, y)
            self._fitted = True
        else:
            self.cluster.partial_fit(X, y)

    def predict(self, X):
        return self.cluster.predict(X)

    def predict_and_adapt(self, X):
        return self.cluster.predict_and_adapt(X)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "prototypes": self.cluster.prototypes,
                "labels": self.cluster.labels,
                "counts": self.cluster.counts,
                "edges": [tuple(sorted(e)) for e in self.cluster.edges],
                "edge_ages": list(self.cluster.edges.values()),
                "threshold": self.cluster.threshold,
                "max_edge_age": self.cluster.max_edge_age,
                "delete_period": self.cluster.delete_period,
                "noise_threshold": self.cluster.noise_threshold,
                "n_fits_": self.n_fits_,
            },
            path,
        )


def create_sfsoinn_model(cfg: Any) -> SFSOINNModel:
    model_cfg = _cfg_get(cfg, "model", None)
    hunter = _cfg_get(model_cfg, "zeroday_hunter", None)
    thr = _cfg_get(hunter, "similarity_threshold", 0.5)
    return SFSOINNModel(similarity_threshold=float(thr))
