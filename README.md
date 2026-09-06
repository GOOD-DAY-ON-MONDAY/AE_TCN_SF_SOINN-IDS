# AE_TCN_SFSOINN — Continual-Learning NIDS

A research framework for **Continual-Learning Network Intrusion Detection Systems (NIDS)**.

The proposed model uses a three-stage pipeline:

1. **AE Gate** — Denoising autoencoder compressing 121-feature raw flows into latent vectors.
2. **1D-TCN Engine** — Dilated causal convolutions for classifying known attack/benign traffic.
3. **SF-SOINN Zero-Day Hunter** — Routes low-confidence predictions (`< confidence_threshold`) to an incremental learner to discover unseen classes without retraining.

All models — including baselines (SVM, Random Forest, BiLSTM) — are trained and evaluated through a unified harness ([`src/train.py`](src/train.py)) across two datasets:

| Dataset | Flows | Classes |
|---|---|---|
| NetML 2020 | 387k | 21 |
| CICIDS2017 | 441k | 8 |

---

## Layout

```text
├── configs/
│   ├── base_config.yaml         # Defaults, class maps, hyperparams
│   └── feature_meta.json        # 121-feature flow schema
├── models/
│   ├── baselines/               # Standard baselines (SVM, RF, BiLSTM)
│   ├── proposed/                # AE, TCN, AE_TCN_SFSOINN
│   └── <family>/<model>/<dataset>/seed_<n>/  # Per-run outputs & artifacts
├── reports/
│   └── comparison_results/all_runs.csv       # Appended run metrics
├── src/
│   ├── train.py                 # CLI & REPL entrypoint
│   ├── runner.py                # Harness logic & contract checks
│   ├── builder/
│   │   └── train_form.py        # Interactive `train` form for the builder shell
│   ├── data/                    # Pipeline & zero-day validator
│   └── utils/                   # Config merge & Rich UI
└── tests/                       # Pytest suite
```

---

## Setup

```bash
# 1. Create environment
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# 2. Download datasets
.venv/bin/python -m src.data.download --dataset all
```

> **Note:** Always run via `.venv/bin/python` to ensure binary shebang compliance.

---

## Training

```bash
# Single seed
.venv/bin/python -m src.train train --model models/baselines/SVM --dataset netml2020 --seed 1

# Multiple seeds
.venv/bin/python -m src.train train --model models/baselines/SVM --dataset netml2020 --seeds 1 2 3
```

---

## CLI Inspection & REPL

```bash
.venv/bin/python -m src.train list-models     # List runnable model directories
.venv/bin/python -m src.train list-datasets   # List supported datasets
.venv/bin/python -m src.train show-config     # Inspect the 3-layer merged config
.venv/bin/python -m src.train builder         # Interactive Claude-style REPL shell
```

### Interactive `train` form (inside the builder shell)

Typing `train` with **no flags** inside the builder REPL launches an interactive form (`src/builder/train_form.py`) built with `InquirerPy`, instead of failing on missing arguments. It prompts for:

| Prompt | Widget | Details |
|---|---|---|
| Model | `inquirer.select` | Scrollable menu scanning `models/` for directories containing a `main.py` |
| Dataset | `inquirer.select` | Choice of `netml2020` or `cicids2017` |
| Seeds | `inquirer.text` | Defaults to `42` |
| Row limit | `inquirer.text` | Optional; leave blank for the full dataset |

The form's answers are assembled into a token list equivalent to a fully-flagged command, e.g.:

```text
train --model models/baselines/SVM --dataset netml2020 --seed 42
```

This is routed through the **same argparse parser** used by the one-shot CLI — the interactive path and the scripted path never drift apart.

---

## Model Contract

To add a new model, create a directory with a single `main.py` anywhere under `models/<family>/<model_name>/`.

### Required interface

| Member | Type | Purpose |
|---|---|---|
| `create_model(cfg)` | Function | Factory returning the initialized model instance from the merged config. |
| `fit(X, y, X_val=None, y_val=None)` | Method | Fits on NumPy arrays. Must be re-callable without resetting internal state. |
| `predict(X)` | Method | Pure, read-only. Returns a 1-D array of class indices. Side-effect free. |
| `save(path)` | Method | Serializes the model artifact (`joblib`, `torch.save`, etc.). |
| `supports_incremental` | Flag | Set `True` if the model supports continual learning via `partial_fit`. |
| `partial_fit(X, y)` | Method | *Optional.* Required if `supports_incremental = True`, used in zero-day loops. |

### Minimal example — `models/baselines/SVM/main.py`

```python
from pathlib import Path
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

class SVMModel:
    def __init__(self, C: float = 1.0):
        self._clf = make_pipeline(StandardScaler(), LinearSVC(C=C, dual="auto", max_iter=10000))

    def fit(self, X, y, X_val=None, y_val=None):
        self._clf.fit(X, y)

    def predict(self, X):
        return self._clf.predict(X)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._clf, path)

def create_model(cfg):
    return SVMModel(C=float(getattr(cfg.model, "C", 1.0)))
```

---

## Configuration

Configs are merged in-memory, in increasing order of precedence:

1. `configs/base_config.yaml` — global defaults
2. `<model_dir>/model.yaml` — optional per-model overrides
3. `--config <file>` — CLI runtime flag

---

## Debugging & Zero-Day Protocols

**Smoke testing**
Add `--limit 5000` to slice a small dataset for rapid debugging.

> Do **not** report benchmark results from `--limit` runs.

**Zero-day evaluation**
Activated automatically when `supports_incremental = True` and target classes are listed in `splitting.zero_day_classes`. Sequence:

1. Evaluate performance on base classes.
2. Teach the new class via `partial_fit`.
3. Verify retention on the base classes.

**Outputs**
Metrics are appended to `reports/comparison_results/all_runs.csv`, capturing:
- Accuracy
- Macro / weighted precision, recall, F1
- Inference latency (ms/flow)
- Memory (GB RSS)
- Runtime

**Evaluation caveat**
Test-set ground truths are withheld (`labels_available: false`); evaluation instead runs on a 15% stratified validation split.

---

## Development

```bash
.venv/bin/python -m pytest   # Run unit & contract test suite
```

---

## Further Reading

- [`docs/CONFIG.md`](docs/CONFIG.md) — full configuration specification