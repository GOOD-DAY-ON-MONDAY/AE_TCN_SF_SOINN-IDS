# Runner Architecture — Consolidated Decisions

Single source of truth for the model-comparison Runner. Feed this
whole file to any agent (aider, etc.) before asking it to implement
or extend the Runner — do not rely on chat memory carrying these
decisions forward, since edit-format/mode switches or session
restarts can silently drop prior context.

Status: all items below are CONFIRMED unless marked otherwise.

---

## Round 1

**Q1 — Model contract (base):** Every model exposes `fit(X, y)` and
`predict(X)`. The Runner owns data loading, seed control, evaluation,
metrics, and artifact saving. Model internals (sklearn vs torch,
epochs, optimizer, device) stay invisible to the Runner. `fit()` must
be callable again later on the same object without resetting it
(required by the zero-day protocol, Q5/Q6).

**Q2 — Unit of `--model`:** A model directory (e.g.
`models/proposed/AE_TCN_SFSOINN/`), not a single file. The directory
is the unit of identity — its path becomes the model's display name
in results. A fixed entry file (`main.py`) inside it is what the
Runner loads. Single-file models simply live alone in their own
directory.

**Q3 — Dataset choices:** `--dataset {netml2020,cicids2017}`, enforced
by argparse choices. No implicit third option.

**Q4 — Metrics recorded per run:** Classification metrics (accuracy,
precision, recall, F1, confusion matrix) AND operational metrics
(inference latency ms, peak memory GB, train time) — measured by the
Runner itself, not by model code. `--seeds 1 2 3` on one invocation
produces mean ± std automatically, no extra teammate code needed.
(Superseded in Round 2, Q4 below — weighted AND macro variants both
required.)

**Q5 — Continual-learning scope (final form, after two revisions):**
Every model that supports incremental learning exposes two additional
methods beyond `fit`/`predict`:
- `partial_fit(X, y)` — controlled, reproducible learning of new data.
  Used by the Runner's scientific evaluation loop. Sequence: `predict()`
  to check a withheld class is flagged unknown → `partial_fit()` to
  teach it → `predict()` again on known classes to measure retention.
- `predict_and_adapt(X)` — true live behavior (SOINN's natural mode:
  no train/test split, spawns a cluster immediately on an unknown
  sample). Used ONLY for a live demo of the deployed system, NEVER in
  the scientific evaluation loop (would break reproducibility across
  seeds).
`predict()` itself must stay a pure read — same input always returns
the same output, no matter how many times it's called.
Models without incremental support (RF, SVM, plain classifiers)
implement neither extra method. Runner checks `supports_incremental`
before calling either.

**Q6 — Output/registration layout:** Each run writes to
`models/<family>/<model>/<dataset>/seed_<n>/` (metrics.json, confusion
matrix plot, model artifact). Every run also appends one row to
`reports/comparison_results/all_runs.csv`.

**Q7 — Hyperparameter config (final form, after two revisions):**
Three-layer merge, in memory only, for the duration of a single run —
`base_config.yaml` on disk is NEVER modified:
1. `configs/base_config.yaml` (shared defaults)
2. `<model_dir>/model.yaml` (optional, model-specific)
3. `--config <file>` (optional, ad-hoc CLI override)
Precedence: later layers override matching keys; unspecified keys
inherit from the layer below. Example invocation:
`python -m src.train --model models/baselines/SVM --dataset netml2020 --config config_SVM.yaml --seed 1`
**Implementation gap (still open):** `src/utils/config.py` currently
only loads a single file — the merge logic and `--config` flag do not
exist yet and must be built.

**Q8 — Data format to models:** Plain numpy arrays —
`X_train, y_train, X_val, y_val, X_test, y_test` — already loaded and
label-encoded by the existing data loader (using `class_map` from
`base_config.yaml`) before any model sees them. Framework-specific
conversion (torch tensors, one-hot) happens inside each model
implementation. The Runner never imports torch.

**Q9 — Entrypoint (final, after verification):** `python -m src.train`.
CONFIRMED: the package is flat `src/` — there is NO `netml_cl`
subpackage; this was verified directly (`find . -maxdepth 2 -type d`),
not assumed. Any earlier code/docs referencing `src.netml_cl.*` were
wrong and should be corrected to `src.*`. Recorded as final: the
package name is `src`, flat — not to be nested or renamed again
without updating this document.

**Q10 — Evaluate-only mode:** Deferred to v2. v1 requires a full
`fit()` call before every `predict()`. No standalone `--eval` that
reloads a saved artifact and re-scores without retraining. Low risk to
defer since Q6's artifact-saving already puts the pieces in place.

---

## Round 2

**Q1 — Factory convention:** `main.py` exposes `create_model(cfg)`,
returning an object with the agreed methods. `cfg` is the fully merged
config namespace (base → model.yaml → `--config`), so hyperparameters
reach the model without the Runner needing to know them.

**Q2 — Artifact saving:** Add `save(path)` as a required third
contract method (alongside `fit`, `predict`) — model serializes
itself; the Runner never pickles it blindly (fragile across
sklearn/torch versions). Optional `load(path)` deferred to v2's
`--eval`.

**Q3 — Validation data signature:** `fit(X, y, X_val=None, y_val=None)`
— models that don't use validation data simply ignore it. NOTE: the
actual *source* of val/test data is still blocked on the split files
and the `labels_available: false` situation from earlier — this
signature does not resolve that; it stays an open dependency.

**Q4 — Predict return & metric set (REVISED):** `predict(X)` returns
hard integer class indices — no probabilities, no `predict_proba` in
v1 (mAP deferred to v2). Metric set reports BOTH weighted AND
macro-averaged precision/recall/F1, alongside accuracy, latency
(ms/flow), and peak memory (GB, process RSS) — eight metrics total logged
per run. Reason macro was added: weighted averaging is dominated by
the majority class (benign, which dominates both datasets), which
would mask exactly the minority-class / zero-day performance this
project is meant to measure.

**Q5 — Seeds flag shape:** `--seeds` (plural, `nargs="+"`), one Run
per seed per invocation, aggregated into mean ± std. Default =
`[training.random_seed]` (42) when omitted.

**Q6 — Zero-day gating in v1:** The Runner's zero-day loop SHIPS in
v1, but only actually executes when the model declares
`supports_incremental = True` AND
`splitting.zero_day_classes.<dataset>` is non-empty in config.
Otherwise it prints a yellow "skipped (not an incremental model / no
zero-day classes configured)" line. `predict_and_adapt` remains
contract-only in v1 — no CLI path exposed yet, exists purely so the
contract doesn't need to change later.

**Assumptions (confirmed, no changes):**
- Rerunning the same seed overwrites its `seed_<n>/` directory (runs
  are deterministic; a yellow notice prints on overwrite).
- Runner code lives in `src/train.py` (CLI + subcommands) and
  `src/runner.py` (run logic, contract validation, nearest-match
  errors).
- `all_runs.csv` columns: `model, dataset, seed, accuracy, precision,
  recall, f1, macro_precision, macro_recall, macro_f1, latency_ms,
  peak_mem_gb, train_time_s, run_dir, timestamp`.

---

## Open items / not yet resolved

- Config merge logic (Q7) — not implemented yet, only decided.
- Val/test data source wiring (Round 2 Q3) — blocked on split files
  and the `labels_available` question.
- `zero_day_classes.<dataset>` in `base_config.yaml` — still empty;
  must be filled from real per-class counts via
  `scripts/validate_data.py`, not guessed.
- No firm date attached yet to when the zero-day experiment (Q5/Q6)
  actually gets exercised end-to-end — flagged previously as needing
  one, since it's the project's headline result.

---

## CLI Design Language

(`train`/`list-models`/`list-datasets`/`show-config`), the pre-run
plan header, color-with-meaning rules, suggest-the-fix error messages,
the consistent end-of-run summary table, per-phase progress bars
during training AND testing, and the visual-polish notes on panels
and consistent iconography. That file's content is unchanged by this
round of decisions and still applies as-is.