"""Shared SF-SOINN module contract tests (real SF-SOINN, ADR 0007)."""

from __future__ import annotations

import numpy as np


def _cfg(threshold: float = 5.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        model=SimpleNamespace(
            zeroday_hunter=SimpleNamespace(similarity_threshold=threshold)
        ),
        training=SimpleNamespace(random_seed=42),
    )


def test_sfsoinn_incremental_flag_and_methods():
    from models._shared.sfsoinn import create_sfsoinn_model

    model = create_sfsoinn_model(_cfg())
    assert getattr(model, "supports_incremental", False) is True
    assert hasattr(model, "partial_fit")
    assert hasattr(model, "predict_and_adapt")


def test_sfsoinn_learns_new_class_without_breaking_old(tmp_path):
    from models._shared.sfsoinn import create_sfsoinn_model
    from src.runner import check_predict_contract

    rng = np.random.default_rng(0)
    X0 = rng.normal(loc=0.0, scale=0.5, size=(20, 6)).astype(np.float32)
    y0 = np.zeros(20, dtype=int)
    X1 = rng.normal(loc=8.0, scale=0.5, size=(10, 6)).astype(np.float32)
    y1 = np.ones(10, dtype=int)
    model = create_sfsoinn_model(_cfg(threshold=5.0))
    model.fit(X0, y0)
    check_predict_contract(model, X0)
    before = model.predict(X0)
    assert (before == 0).mean() > 0.8
    model.partial_fit(X1, y1)
    after_old = model.predict(X0)
    assert (after_old == 0).mean() > 0.7
    model.save(tmp_path / "sfsoinn_artifact")
    assert (tmp_path / "sfsoinn_artifact").is_file()


def test_sfsoinn_creates_edges_and_ages_them():
    """Real SF-SOINN: edges form between nearby nodes and age on updates."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(1)
    X = rng.normal(loc=0.0, scale=0.3, size=(60, 4)).astype(np.float32)
    c = SFSOINNCluster(similarity_threshold=2.0, max_edge_age=5)
    c.fit(X, np.zeros(60, dtype=int))
    assert len(c.edges) > 0, "no edges created — edge insertion missing"
    assert any(a > 0 for a in c.edges.values()), "edge ages never incremented"


def test_sfsoinn_edge_age_pruning():
    """Edges older than max_edge_age are pruned."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(2)
    X = rng.normal(loc=0.0, scale=0.3, size=(200, 4)).astype(np.float32)
    c = SFSOINNCluster(similarity_threshold=2.0, max_edge_age=3, delete_period=10**9)
    c.fit(X, np.zeros(200, dtype=int))
    assert all(a <= c.max_edge_age for a in c.edges.values())


def test_sfsoinn_forgets_noisy_nodes():
    """Win-count-based node deletion (successive forgetting) removes nodes."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(3)
    X_main = rng.normal(loc=0.0, scale=0.3, size=(100, 4)).astype(np.float32)
    X_noise = rng.normal(loc=50.0, scale=0.1, size=(3, 4)).astype(np.float32)
    X = np.concatenate([X_main, X_noise], axis=0)
    y = np.zeros(103, dtype=int)
    c = SFSOINNCluster(
        similarity_threshold=2.0, delete_period=50, noise_threshold=5
    )
    c.fit(X, y)
    # The 3 far-away singleton nodes have win counts < 5 → forgotten.
    assert len(c.prototypes) < 103
    far_dists = [
        float(np.linalg.norm(np.asarray(p) - X_main[0])) for p in c.prototypes
    ]
    assert max(far_dists) < 20, "noise nodes near loc=50 were not forgotten"


def test_sfsoinn_label_propagation():
    """Unlabeled nodes inherit majority label of their connected component."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(4)
    X = rng.normal(loc=0.0, scale=0.3, size=(40, 4)).astype(np.float32)
    c = SFSOINNCluster(similarity_threshold=2.0)
    c.fit(X, np.ones(40, dtype=int))
    # Simulate an unlabeled node inside the same dense region.
    c.labels[0] = -1
    preds = c.predict(X[:5])
    assert (preds == 1).all(), "label propagation failed for unlabeled node"


def test_sfsoinn_predict_is_pure_read():
    """predict must not mutate the network (incremental contract)."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(5)
    X = rng.normal(loc=0.0, scale=0.3, size=(30, 4)).astype(np.float32)
    c = SFSOINNCluster(similarity_threshold=2.0)
    c.fit(X, np.zeros(30, dtype=int))
    n_proto, n_edges = len(c.prototypes), len(c.edges)
    c.predict(X)
    c.predict(X[:3])
    assert len(c.prototypes) == n_proto and len(c.edges) == n_edges


def test_sfsoinn_predict_and_adapt_grows_for_novel():
    """Novel far-away inputs get fresh labels via predict_and_adapt."""
    from models._shared.sfsoinn import SFSOINNCluster

    rng = np.random.default_rng(6)
    X = rng.normal(loc=0.0, scale=0.3, size=(30, 4)).astype(np.float32)
    c = SFSOINNCluster(similarity_threshold=2.0)
    c.fit(X, np.zeros(30, dtype=int))
    n_before = len(c.prototypes)
    novel = rng.normal(loc=30.0, scale=0.3, size=(2, 4)).astype(np.float32)
    out = c.predict_and_adapt(novel)
    # First novel sample creates a new cluster (label 1); the second joins it
    # (within threshold of the new prototype) — correct SF-SOINN behavior.
    assert out[0] == 1 and out[1] == 1, f"novel labels wrong: {out}"
    # Two-distance rule: the second novel sample is close to the new node but
    # its second-nearest (main cluster) is beyond threshold2 → SOINN inserts
    # an additional node there too (literature-faithful).
    assert len(c.prototypes) == n_before + 2
    # A second, distant novel point creates yet another cluster.
    novel2 = rng.normal(loc=-30.0, scale=0.3, size=(1, 4)).astype(np.float32)
    out2 = c.predict_and_adapt(novel2)
    assert out2[0] == 2, f"second novel cluster label wrong: {out2}"
