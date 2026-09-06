## To-Do List

- [ ] Confirm shared 121-feature schema between NetML2020 and CICIDS2017 (already verified)
- [ ] Define zero-day holdout classes for NetML2020 (proportional to 21-class taxonomy)
- [ ] Define zero-day holdout classes for CICIDS2017 (proportional to 8-class taxonomy — larger % per class)
- [ ] Pull per-class sample counts for both datasets (pending from earlier)
- [ ] Build train/val/test splits **separately** for each dataset, excluding zero-day holdout classes from training
- [ ] Set up PyArrow chunked loading pipeline (shared across both datasets)
- [ ] Train baseline models (SVM, RF, BiLSTM) on both datasets
- [ ] Implement BiLSTM incremental fine-tuning protocol (sequential zero-day class introduction + retention measurement)
- [ ] Train ablation models (PCA-32, AE+TCN w/o SF-SOINN, TCN on raw 121 w/o AE)
- [ ] Train full NetML-CL (AE+TCN+SF-SOINN) on both datasets
- [ ] Run all trainable models across multiple seeds, log mean ± std for each metric
- [ ] Compile per-dataset results table (classification + continual learning + operational metrics)
- [ ] Compile ablation results table (AE vs PCA, with/without SF-SOINN, with/without AE)
- [ ] Write up comparison discussion tying results back to Objectives I–V

---

## Model Training Matrix

| Model | NetML2020 | CICIDS2017 | Seeds | Notes |
|---|:---:|:---:|:---:|---|
| SVM | ✓ | ✓ | 1–2 | Deterministic given fixed hyperparams |
| Random Forest | ✓ | ✓ | 3 | Variance from bootstrap/feature sampling |
| BiLSTM (+ incremental fine-tune) | ✓ | ✓ | 3–5 | Needs sequential zero-day fine-tune loop for forgetting curve |
| PCA-32 + TCN + SF-SOINN (ablation) | ✓ | ✓ | 2–3 | Tests whether AE's non-linearity earns its cost vs. cheap linear PCA |
| AE-32 + TCN, no SF-SOINN (ablation) | ✓ | ✓ | 2–3 | Softmax-only classifier; isolates SF-SOINN's contribution |
| Raw 121 + TCN + SF-SOINN (ablation) | ✓ | ✓ | 2–3 | Tests whether AE's compression is necessary at all |
| **Full NetML-CL** (AE+TCN+SF-SOINN) | ✓ | ✓ | 3–5 | Main proposed model — heaviest seed budget |

---

## Results to Pull, Per Model/Dataset Combination

**Classification metrics**
- Accuracy, Precision, Recall, F1-Score (on known/test classes)

**Continual learning metrics** (full model + BiLSTM only — others are static, expected to fail here by design)
- Knowledge Retention Rate (accuracy on old classes after learning new zero-day class)
- Plasticity Score (speed/accuracy of clustering new zero-day class)

**Operational metrics**
- Inference Latency (ms)
- Peak Memory Utilization (GB)
- Training time (relative units, per Figure 8 style comparison)

**Ablation-specific**
- AE-32 vs PCA-32: downstream TCN accuracy + SF-SOINN retention rate (isolates non-linearity's value)
- With vs without SF-SOINN: retention rate on zero-day classes (isolates SF-SOINN's value)
- With vs without AE: latency + memory + accuracy (isolates AE's value)

**Final comparison table** (cross-model, cross-dataset)
- One summary table: rows = 5 main models (SVM, RF, BiLSTM, ablations optional, NetML-CL), columns = {NetML2020, CICIDS2017} × {Accuracy, F1, Retention Rate, Latency}  mirrors your existing Table 1 structure but with actual experimental numbers instead of projected/theoretical ones.


Use the /to-spec skill now.

Do not run a grilling/interview session — this has already been done.
Read CONTEXT.md in full and treat every item in it marked
CONFIRMED as an already-agreed decision, not something to re-question.
Synthesize it directly into a spec for the model-comparison Runner.

The spec must cover, at minimum: the model contract (fit/predict/save,
plus optional partial_fit/predict_and_adapt for incremental models),
the CLI shape and subcommands, the config merge order (base ->
model.yaml -> --config override, in-memory only), the metrics set
(weighted + macro precision/recall/F1, accuracy, latency, peak
memory), the seed/reproducibility behavior, the output/artifact
layout, and the zero-day gating logic (Round 2 Q6).

Also read CONTEXT.md and fold its subcommands, progress-bar,
and output-formatting requirements into the spec as the CLI/UX
section — don't drop these in favor of only the backend contract.

Explicitly note in the spec, as open/unresolved items, anything from
CONTEXT.md's "Open items / not yet resolved" section
(config merge logic not yet implemented, val/test data source still
blocked, zero_day_classes still empty, no firm date on the zero-day
experiment) — do not silently resolve these or invent values for them
while writing the spec.

## Goal

Produce a single, complete spec document synthesized from the context
below, using the to-spec skill. Do not run an interview/grilling
session — every decision needed already exists in the context section.

Do not report this task as complete until ALL of the following are
true, verified by you actually reading the file back after writing it
— not by assuming the write succeeded:

1. A spec file exists on disk at a path you state explicitly.
2. You have re-opened that file and confirm it is non-empty.
3. The file contains a distinct, identifiable section for EACH of the
   following (check off each one explicitly in your final report):
   - Model contract (fit/predict/save, plus optional
     partial_fit/predict_and_adapt for incremental models)
   - CLI shape and subcommands (train / list-models / list-datasets /
     show-config), pulled from CONTEXT.md (## CLI Design Language line 166 ) , not just the backend
   - Config merge order (base_config.yaml -> model.yaml -> --config
     override, in-memory only, base file never modified on disk)
   - Full metrics set (accuracy, weighted + macro precision/recall/F1,
     latency ms, peak memory GB) — eight metrics, not a subset
   - Seed/reproducibility behavior (--seeds plural, mean ± std,
     default from config)
   - Output/artifact layout (per-run directory structure +
     all_runs.csv columns, listed exactly)
   - Zero-day gating logic (ships in v1, only executes when
     supports_incremental=True AND zero_day_classes non-empty,
     otherwise prints a skip notice)
4. The file contains an explicit "Open / Unresolved" section listing,
   verbatim in substance, these four items — NOT silently resolved or
   filled in with invented values:
   - Config merge logic is decided but not yet implemented
   - Val/test data source is still blocked on split files and the
     labels_available question
   - zero_day_classes per dataset is still empty and must come from
     real per-class counts, not a guess
   - No firm date yet exists for when the zero-day experiment runs
     end-to-end
5. Nothing in the spec contradicts anything stated as CONFIRMED in the
   context below. If you believe something below should change, STOP
   and ask — do not silently overwrite a confirmed decision.

If any of the five checks above fail, fix it and re-verify before
declaring the task done. State explicitly, item by item, that each of
the five checks passed — a general "done" statement without this
checklist is not acceptable.

## Context

The following is the full, confirmed decision record for this
project. Treat every item marked CONFIRMED as settled — do not
re-question it.

 @CONTEXT.md

---
(Runner Architecture — Consolidated Decisions, as provided)
---

Do the following as three sequential phases. Do not start a phase
until the previous phase's verification step has actually completed
and been shown to me — do not silently skip ahead or batch commits
together.

## PHASE 1 — Docstring and comment cleanup

Go through every .py file in the repo (src/, models/) and clean up
comments and docstrings, with these distinctions:

REMOVE:
- comments that just restate what the next line obviously does
  (e.g. "# increment counter", "# loop over rows")
- redundant or unnecessary information inside docstrings — anything
  that repeats the function name in prose, restates an obvious type
  from the signature, or pads with filler rather than adding real
  understanding

KEEP or WRITE PROPERLY: every function/class gets a docstring stating
what it does, its arguments (name, type, meaning), and what it
returns — a complete argument mapping, but concise: real information
only, no padding.

DO NOT REMOVE inline comments that explain WHY non-obvious logic
exists, even though they aren't docstrings — specifically:
- the reasoning behind the config merge precedence order
- the single-class re-read logic in --limit (the class-grouping fix)
- the OOM-driven memory choices in the data loader (feature_dim vs
  the old 2048 buffer, float32, no pandas round-trip)
- the labels_available: false fallback-to-val-split behavior
These are hard-won decisions with real failure history behind them —
stripping them because they're not docstrings would delete knowledge,
not noise. If unsure whether something is "restates the obvious" vs
"explains a non-obvious why," keep it.

Verification (must pass before committing):
  .venv/bin/python -m src.train train --model models/baselines/SVM --dataset netml2020 --seed 1 --limit 5000
Paste the real output. Only if it still passes:
  git add -A && git commit -m "Clean up comments and docstrings; ensure complete, non-redundant argument documentation"
Paste the actual `git log -1` output confirming the commit landed.
STOP HERE and show me all of the above before starting Phase 2.

## PHASE 2 — requirements.txt (verified, not guessed)

Do not write requirements.txt from memory or assumption. Run:
  .venv/bin/pip freeze > /tmp/installed.txt
Scan every .py file's actual import statements across src/ and
models/, derive the real list of third-party packages used (exclude
stdlib: os, sys, json, argparse, etc.), and cross-reference against
/tmp/installed.txt for accurate version pins.

Paste the final requirements.txt content here. Then verify it
installs cleanly in a fresh throwaway venv:
  python3 -m venv /tmp/verify_env && /tmp/verify_env/bin/pip install -r requirements.txt
Paste that real output, confirming no errors. Only then:
  git add requirements.txt && git commit -m "Add requirements.txt with verified dependencies and pinned versions"
Paste the actual `git log -1` output confirming the commit landed.
STOP HERE and show me all of the above before starting Phase 3.

## PHASE 3 — README.md + template main.py

Re-verify every claim against the actual current repo state before
writing anything — do not write from memory of past conversation,
since prior docs in this project drifted from reality more than once
(docs/CONFIG.md was wrong about the package path; CONTEXT.md was
accidentally a saved prompt instead of real content). Run
`find . -maxdepth 2 -type d` and `cat` real entrypoint files first.

Structure:
1. One-paragraph project description (AE_TCN_SFSOINN continual-
   learning IDS vs baseline comparison Runner).
2. Setup: venv creation, requirements.txt install, and the fact that
   `python` isn't on PATH here — `.venv/bin/python` must be used
   explicitly (this caused real confusion earlier, don't omit it).
3. "For teammates: how to add and run your own model":
   - the model directory contract (fit/predict/save, optional
     partial_fit/predict_and_adapt for incremental models)
   - exact run command: python -m src.train train --model <dir>
     --dataset <name> --seed <n>
   - the --limit N flag, explicitly labeled: "for smoke-testing only
     — never report accuracy numbers from a --limit run, they are
     not meaningful"
   - list-models / list-datasets / show-config subcommands
   - where results land (metrics.json path, all_runs.csv)

Also create models/_template/main.py — a copy-pasteable skeleton
model directory implementing the full contract with
NotImplementedError bodies and complete, non-redundant docstrings
(same standard as Phase 1), so a teammate can copy this folder as
their starting point.

Paste the real README.md and models/_template/main.py content here
before committing. Only after I can see and confirm both:
  git add README.md models/_template/main.py && git commit -m "Add README with team usage guide and model template"
Paste the actual `git log -1` output confirming the commit landed.