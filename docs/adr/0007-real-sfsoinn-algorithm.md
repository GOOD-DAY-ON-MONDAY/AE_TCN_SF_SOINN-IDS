# ADR 0007: Literature-faithful SF-SOINN with documented simplifications

## Status

Accepted

## Context

`models/_shared/sfsoinn.py` is a tracer bullet: one mean prototype per class,
distance-gated mean-update-or-append in `partial_fit`, and a heuristic
`predict_and_adapt`. There is no edge structure, no edge-age pruning, no node
deletion, and no forgetting — none of the mechanisms that make SF-SOINN
(Successive Forgetting SOINN, from the SOINN family of incremental
self-organizing networks) a real algorithm. This is the component the project
is named after and the one most likely to be quietly under-built. Four
consumers depend on it: `proposed/AE_TCN_SFSOINN`, `ablations/AE_SFSOINN`,
`ablations/PCA_SFSOINN`, `ablations/PCA_TCN_SFSOINN`.

## Question

Does the paper's incremental/forgetting claim require the full literature
algorithm (node/edge insertion, similarity thresholds, edge-age pruning, node
deletion, forgetting), or is a named, documented simplification acceptable?

## Options

1. **Full literature SF-SOINN** — two-stage thresholding (first/second
   nearest node), edge insertion between nearest nodes with age increment,
   edge-age pruning, node deletion of low-magnitude nodes, class-label
   propagation over edges, periodic forgetting.
2. **Named simplification** — keep prototype nodes and labels, add the
   mechanisms that the paper's claims actually rest on, drop the rest with
   explicit documentation.
3. Keep the tracer bullet.

## Decision

Option 2, with the simplification named and documented here and in the module
docstring. Rationale: the paper's claims are (a) incremental learning without
retraining, (b) novel-class (zero-day) emergence via prototype insertion, and
(c) forgetting of obsolete prototypes. The mechanisms required are:

- **Similarity-thresholded node insertion** — two-distance threshold
  (distances to first and second nearest nodes; insert when the sample is
  farther than both), replacing the single-threshold heuristic.
- **Edge insertion with age** — connect the first and second nearest nodes
  with an edge; increment edge ages of the winner's other edges; remove edges
  whose age exceeds `max_edge_age`.
- **Node deletion (forgetting)** — periodically delete nodes whose win count
  is below `noise_threshold` (the "successive forgetting" in SF-SOINN).
- **Class-label propagation over edges** — predict via the winner's label,
  propagated through connected components so unlabeled/new nodes inherit
  neighbor labels.

**Documented simplifications** (explicitly not implemented, and why):

- *Adaptive per-node similarity thresholds* (Furao & Hasegawa's
  distance-ratio-based thresholds): replaced by the two-distance rule with a
  global `similarity_threshold` from `cfg.model.zerohunter` — keeps the
  existing config surface stable and is a standard simplification.
- *Periodic topological learning-rate decay*: win-count-based deletion covers
  the forgetting claim; decay adds tuning surface without changing the claim.

Clustering is legitimately CPU-bound (small prototype counts, numpy distance
computations); no torch device is introduced for SF-SOINN.

## Consequences

- The incremental contract is unchanged: `supports_incremental=True`,
  `partial_fit`, `predict_and_adapt` keep their signatures and semantics
  (`predict` stays pure-read; adaptation happens only in
  `predict_and_adapt`).
- `SFSOINNCluster` gains `edges`, `ages`, `win_counts` state; `save()` and
  the four consumers' `save()` payloads must serialize the new state.
- Consumers' inline `predict_and_adapt` loops (duplicated against
  `cluster._nearest`) are replaced by delegation to the shared cluster's
  `predict_and_adapt` so the real algorithm lives in one place.
- Smoke-verified only (`--limit`); full-dataset runs on Kaggle.