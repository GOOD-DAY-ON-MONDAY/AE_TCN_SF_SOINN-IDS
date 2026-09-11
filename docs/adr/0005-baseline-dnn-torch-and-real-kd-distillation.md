# ADR 0005: DNN becomes a real torch network; Distillation uses true KL-divergence KD

Date: 2026-09-11
Status: Accepted
Decides: naming/scope questions raised in the Phase 1 baseline audit.

## Context

The Phase 1 audit found:

- **DNN** ([models/baselines/DNN/main.py](../../models/baselines/DNN/main.py)) is a
  genuine but *shallow* sklearn `MLPClassifier` — CPU-only, no GPU path, no device
  check — while the synopsis's comparison table presents it as a "DNN" baseline.
- **Distillation** ([models/baselines/Distillation/main.py](../../models/baselines/Distillation/main.py))
  trains a sklearn MLP teacher and a `LogisticRegression` student on
  **argmax-collapsed** soft targets ([models/_shared/baseline_helpers.py](../../models/_shared/baseline_helpers.py),
  `DistillationHelper.distill`) — temperature/alpha are largely cosmetic because
  soft probability distributions are never actually fitted against.

Both misrepresent what the baselines claim in a results table.

## Decision

1. **DNN → real torch feedforward network.** Rebuild
   `models/baselines/DNN` as a torch MLP: moved to `get_device()`'s selection,
   with a printed device check at construction, framework seeding via a shared
   helper in `models/_shared/` (the Runner never imports torch — grep-guarded,
   ADR 0002). The sklearn MLP implementation is replaced, not renamed away:
   the directory keeps the name `DNN` because that is what the comparison
   table promises.

2. **Distillation → real knowledge distillation.** Rebuild with a torch
   teacher and torch student trained on **temperature-scaled soft targets with
   true KL-divergence loss** against teacher probabilities (plus the standard
   hard-label CE term weighted by alpha). The argmax-collapsed sklearn
   approximation is retired. Teacher/student pair for this project is defined
   as: teacher = the DNN baseline architecture (torch MLP, same hidden sizes),
   student = a strictly smaller torch MLP; both GPU-aware via `get_device()`.

## Consequences

- DNN, BiLSTM, and Distillation are all torch-based, GPU-aware baselines;
  SVM / Fast_SVM / RF remain CPU-only sklearn (legitimate, documented).
- All torch models seed their framework in `create_model` via the shared
  helper; the Runner seeds numpy/stdlib random only (Phase 0 fix).
- Results-table claims: "DNN" and "Distillation" now mean what the synopsis
  says; no methodology-note caveat is needed for either.
- Smoke verification stays `--limit`-only in this environment; full runs
  happen on Kaggle.
