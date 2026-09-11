# ADR 0006: Real gradient-trained TCN via nn.Conv1d, trained end-to-end

## Status

Accepted

## Context

`models/_shared/tcn.py` is a tracer bullet (ADR 0002): `TCNExtractor` draws
fixed random dilated causal filters once at init and never updates them; only
a sklearn `LogisticRegression` head learns. Four consumers depend on this
interface: `proposed/AE_TCN_SFSOINN`, `ablations/AE_TCN`, `ablations/PCA_TCN`,
`ablations/PCA_TCN_SFSOINN`. The TCN is a core component of the paper's
proposed model, so a frozen random projection is not publishable.

## Question

Should the TCN become a real `nn.Conv1d` causal dilated network trained
end-to-end with the classifier on `get_device()`, and what exact block spec
keeps the four consumers stable?

## Options

1. **Real torch TCN, end-to-end training** — WaveNet-style residual blocks,
   `nn.Conv1d` with left-padding for causality, trained jointly with the
   classifier head via Adam on `get_device()`.
2. Keep frozen random filters, upgrade only the head to torch.
3. Train the TCN with a separate reconstruction objective, freeze, then fit
   the classifier (two-stage).

## Decision

Option 1. The TCN's value in the proposed model is *learned* temporal feature
extraction; a frozen projection is indistinguishable from random features and
would undermine the paper's ablation story (PCA_TCN vs AE_TCN vs full model).

### Exact block spec (interface contract for all four consumers)

- Input: `(N, T, C_in)` where `T` is a small sliding window (default 8,
  stride 4, via `SequenceBatcher` from `baseline_helpers`) over the
  compressor latents; per-sample features are mean-pooled over time.
- Block: for each dilation `d` in `cfg.model.temporal_engine.dilations`
  (default `[1, 2, 4, 8]`), one residual block:
  - `Conv1d(c_in, c_out, kernel_size, dilation=d, padding=(k-1)*d)` (left pad
    only, causal), `BatchNorm1d`, `GELU`, dropout `p=0.1`.
  - Residual: identity projection `Conv1d(c_in, c_out, 1)` when channel
    counts differ; skip connection added before the activation.
- Channels schedule from `cfg.model.temporal_engine.channels`
  (default `[64, 64, 32]`); kernel from `kernel_size` (default 3).
- Output: pooled `(N, channels[-1])` features feeding the classifier head
  (`nn.Linear(channels[-1], n_classes)`) — trained jointly, cross-entropy.
- Device: `get_device()` from `models/_shared/baseline_helpers.py`; device
  verified via `print_device_check("TCN", device)`.
- Seeding: `seed_torch(seed)` called in `create_model` per Runner seed
  iteration (no import-time caching of weights).

## Consequences

- `TCNExtractor.transform` keeps its signature (numpy in → pooled numpy out)
  so consumers' `_features` pipelines stay unchanged; it now runs the trained
  torch encoder under the hood.
- `TCNModel.fit` trains conv + head end-to-end; `save` stores
  `state_dict()` instead of numpy weights. Consumers' `save()` payloads that
  touch `extractor.weights/biases` must be updated to the new artifact
  (ticket-scoped, all four consumers).
- Smoke runs are `--limit`-scoped only; full-dataset runs stay on Kaggle.