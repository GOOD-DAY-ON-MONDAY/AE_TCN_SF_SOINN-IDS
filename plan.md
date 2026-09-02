## To-Do List

- [ ] Confirm shared 121-feature schema between NetML2020 and CICIDS2017 (already verified)
- [ ] Define zero-day holdout classes for NetML2020 (proportional to 21-class taxonomy)
- [ ] Define zero-day holdout classes for CICIDS2017 (proportional to 8-class taxonomy — larger % per class)
- [ ] Pull per-class sample counts for both datasets (pending from earlier)
- [ ] Build train/val/test splits **separately** for each dataset, excluding zero-day holdout classes from training
- [ ] Set up PyArrow chunked loading pipeline (shared across both datasets)
- [ ] Implement device-agnostic router (CUDA/MPS/CPU)
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