# AE_TCN_SFSOINN — Continual-Learning IDS Runner

A continual-learning network intrusion detection system — Autoencoder → 1D-TCN
for known-threat classification, with SF-SOINN handling zero-day classes
incrementally — compared against classical and deep baselines (SVM, Random
Forest, BiLSTM) on the NetML 2020 and CICIDS2017 flow datasets. The repo is
built around a model-comparison **Runner** ([`src/train.py`](src/train.py)):
any model that satisfies a small directory contract can be trained,
evaluated, and aggregated by the same harness, so baseline and proposed
results are directly comparable.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Important:** on this machine plain `python` / `pip` are not on PATH (and a
stale `python3.14` is what created the broken `.venv/bin/pip` shebang).
Always invoke the venv interpreter explicitly as `.venv/bin/python`, and use
`.venv/bin/python -m pip` rather than `.venv/bin/pip`.

## For teammates: how to add and run your own model

### 1. The model directory contract

Create a directory anywhere under `models/` (two levels deep, e.g.
`models/baselines/my_model/`) containing a single `main.py` that exposes:

- `create_model(cfg)` — factory taking the fully merged config namespace;
  returns your model object.
- `fit(X, y, X_val=None, y_val=None)` — train on numpy arrays. Must be
  re-callable on the same object without resetting it. `X_val`/`y_val` may be
  `None` (test labels are currently unavailable; see below).
- `predict(X)` — return 1-D hard integer class indices, as a **pure read**
  (no side effects; the Runner checks repeated calls give identical output).
- `save(path)` — serialize the fitted model.

If your model supports incremental/continual learning, additionally set
`supports_incremental = True` and implement:

- `partial_fit(X, y)` — used by the Runner's zero-day protocol
  (predict → teach withheld class → re-predict).
- `predict_and_adapt(X)` — contract-only in v1; the Runner never calls it.

A complete working example is [`models/baselines/SVM/main.py`](models/baselines/SVM/main.py) —
a non-incremental baseline that implements the full contract in ~80 lines.
For an incremental model, add `supports_incremental = True` plus the two
methods (the contract checks in [`src/runner.py`](src/runner.py) will name
exactly what is missing if you get it wrong).

### 2. Run it

```bash
.venv/bin/python -m src.train train --model models/baselines/my_model --dataset netml2020 --seed 1
```

`--dataset` is `netml2020` or `cicids2017`; `--seed` accepts multiple values
(`--seeds 1 2 3`) and results are aggregated into mean ± std.

### 3. The `--limit N` flag — for smoke-testing ONLY

`--limit 5000` makes the harness read a small slice of the data for fast
end-to-end debugging (it re-reads with a growing cap until at least 2 classes
appear, because the raw file is class-grouped).

**⚠️ Never report accuracy numbers from a `--limit` run — they are not
meaningful.**

### 4. Other subcommands

```bash
.venv/bin/python -m src.train list-models     # runnable model directories
.venv/bin/python -m src.train list-datasets   # valid dataset names
.venv/bin/python -m src.train show-config --model models/baselines/SVM --dataset netml2020
```

`show-config` prints the effective three-layer config merge
(`configs/base_config.yaml` → optional `<model_dir>/model.yaml` → optional
`--config` file), annotating which layer supplied each key. Nothing on disk
is modified.

### 5. Where results land

- Per-seed run directory: `models/<family>/<model>/<dataset>/seed_<n>/`
  containing `metrics.json`, `confusion_matrix.png`, and the saved
  `model_artifact`.
- Aggregate CSV across all runs: `reports/comparison_results/all_runs.csv`.

### Evaluation data caveat

Both datasets currently have `labels_available: false` in
[`configs/base_config.yaml`](configs/base_config.yaml): the competition
test-set annotations are withheld, so the Runner evaluates on a stratified
validation split held out of the training data, and the data seam returns
`None` for the test arrays. Keep this in mind when comparing results.

## Downloading data

```bash
.venv/bin/python -m src.data.download --dataset netml2020   # or cicids2017 / all
.venv/bin/python -m src.data.download --dataset all --force # re-download
```
