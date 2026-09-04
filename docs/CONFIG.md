# `configs/base_config.yaml` — field-by-field guide

This file explains every field in `configs/base_config.yaml` for someone
who has never seen it. The config is loaded by `src/utils/config.py`:

```python
from src.utils.config import load_config

cfg = load_config()
cfg.data.netml2020.feature_dim   # 121
cfg.model.temporal_engine.channels  # [64, 64, 32]
```

`load_config()` validates every required key and raises `ConfigError`
with a message naming the exact missing (or invalid) key — it never
fails silently. You can smoke-test the config from the repo root with:

```bash
python -m src.utils.config
```

---

## Dataset context (read this first)

Two datasets live under `data/raw/`, and each is **already pre-split by
the original challenge into four folders** — we never re-split them
ourselves:

| Folder | Contents |
|---|---|
| `2_training_set/` | Flow features as a `.json.gz` file (e.g. `2_training_set.json.gz`) |
| `2_training_annotations/` | Ground-truth labels — **only exists for the training set** |
| `1_test-std_set/` | Standard test split (see label warning below) |
| `0_test-challenge_set/` | Challenge test split (see label warning below) |

Both datasets share the **same schema: 121 raw features per flow**.

- **netml2020** — 387,268 training rows, 21 classes
- **cicids2017** — 441,116 training rows, 8 classes

---

## `data`

### `data.<dataset>.raw_dir`
Root folder of the raw dataset on disk (e.g. `data/raw/netml2020`).

### `data.<dataset>.training_set`
Path to the training **features** file (`.json.gz`). Each record is one
network flow with 121 feature values.

### `data.<dataset>.training_annotations`
Path to the training **labels**. These are the only ground-truth labels
shipped with the datasets, and they cover the training set only.

### `data.<dataset>.test_std_dir` and `data.<dataset>.test_challenge_dir`
Paths to the two held-out test folders. **See the label-availability
section below before using them for anything.**

### `data.<dataset>.labels_available`
`true` or `false`: whether ground-truth annotations exist for the
held-out test splits (`test_std` / `test_challenge`). **Both are
currently `false`** — see below for why and what that means.

### `data.<dataset>.feature_dim`
Number of raw features per flow. Must be `121` for both datasets; the
loader rejects anything else.

### `data.<dataset>.num_classes` and `data.<dataset>.class_map`
`class_map` maps each class name to an integer index used consistently
across training and evaluation. `num_classes` is its size; the loader
checks the two agree. The order (name → index) is fixed here so that
checkpointed models stay compatible.

### `data.processed_dir`
Where cleaned/encoded intermediate artifacts (encoded features, split
indices, cached arrays) are written.

---

## ⚠️ Label availability in the test sets — what we found

**Finding:** ground-truth labels were **not found** for
`1_test-std_set` or `0_test-challenge_set`. To be fully transparent:
the record-level verification could not be executed when the config was
written (raw data not readable in that session; the manifests in
`data/splits/` are empty). The conclusion rests on the NetML 2020
challenge design itself, in which the test-std and test-challenge splits
were held-back leaderboard sets whose annotations were not distributed
to participants. The config therefore defaults both datasets to
`labels_available: false`.

**What this means practically:**

- The test sets can only be used for **unsupervised inference / demo
  purposes** (e.g. showing the deployed model routing unseen flows), **not
  for evaluation**. All quantitative evaluation must use the stratified
  validation split drawn from the labeled training set.
- **Before relying on this:** once the raw data is present, run
  `scripts/validate_data.py`. If it discovers annotation files next to
  either test set, flip that dataset's `labels_available` to `true` and
  update this document. Until that check passes, treat both test sets as
  unlabeled.

---

## `splitting`

**No train/test split is configured — by design.** The raw data is
already pre-divided by file into training / test-std / test-challenge
folders by the original challenge. Only two splits are configured here:

### `splitting.val_split` (0.15)
A stratified 15% held out **from the training set only**, used for
early stopping and model selection. "Stratified" means each class keeps
its original proportion in the split, which matters here because these
datasets are heavily imbalanced (e.g. `benign` dominates).

### `splitting.random_seed` (42)
Controls the shuffling used by the validation split, so everyone gets
the identical split.

### ⚠️ `splitting.zero_day_classes` — currently empty (TODO)

```yaml
zero_day_classes:
  netml2020: []     # TODO
  cicids2017: []    # TODO
```

These lists name the classes **excluded from initial TCN training** and
held out for Stage 3 (SF-SOINN) zero-day evaluation — the core
continual-learning experiment.

**Why this must be filled in from real data, not guessed:**

1. The choice has to be **3–5 lower-frequency, non-benign classes per
   dataset**. Only `scripts/validate_data.py` (with its per-class count
   report, `docs/data_report.png`) can tell us which classes actually
   have enough samples to be a meaningful holdout — a class with a
   handful of flows would make the zero-day evaluation statistically
   meaningless.
2. Picking classes "from memory" risks holding out classes that are too
   rare (no signal), too common (cripples the trained model), or benign
   (breaks the threat model — benign traffic is never a zero-day attack
   class).
3. These lists gate the headline continual-learning numbers (retention
   and plasticity). If the holdout is wrong, every downstream claim in
   the results tables is wrong with it.

**Action:** run `scripts/validate_data.py`, review
`docs/data_report.png`, then fill in both lists.

---

## `model`

### `model.denoising_gate.latent_dim` (32)
The denoising autoencoder compresses each 121-dim raw flow into a
32-dim latent vector. This is the bottleneck that forces the model to
learn a compact representation before the temporal stage.

### `model.temporal_engine`
The temporal convolutional network (TCN) that processes sequences of
latent embeddings:

- `channels: [64, 64, 32]` — output width of each conv block (narrowing
  toward the head).
- `kernel_size: 3` — width of each convolution window.
- `dilations: [1, 2, 4, 8]` — dilation per layer; the exponentially
  growing receptive field lets the TCN see long flow sequences cheaply.
- `dropout: 0.2` — regularization between blocks.

### `model.zeroday_hunter.similarity_threshold` (0.5)
The SF-SOINN component creates a new cluster when an incoming latent
vector's similarity to all existing nodes falls below this threshold.
**Starting value only** — it must be empirically tuned once real latent
vectors exist; do not treat 0.5 as final.

### `model.routing.confidence_threshold` (0.5)
The known-vs-unknown gate: flows whose classifier confidence is at or
above this value are routed to the closed-set classifier; the rest go to
the zero-day hunter.

---

## `training`

- `batch_size: 256` — flows per optimization step.
- `learning_rate: 0.001` — optimizer step size.
- `random_seed: 42` — global seed for reproducibility of weight
  initialization, shuffling, and dropout masks.

## `device`

- `preference: auto` — resolved at runtime by `utils/device.py`
  (`cuda` → `mps` → `cpu`). Override with `cuda`/`mps`/`cpu` to pin.

---

## What `src/utils/config.py` validates

- Every required top-level section and key exists — errors name the
  exact missing key (e.g. `Missing required config key:
  'data.cicids2017.class_map'`).
- `feature_dim` equals 121 for both datasets.
- `class_map` is non-empty and its length equals `num_classes`.
- `labels_available` is a real boolean.
- `zero_day_classes.<dataset>` exists as a list (it may be empty while
  the TODO is pending, but the key must be present).
- All model/training/device keys are present.
