# ADR 0008: Real torch encoder/decoder AE for the shared compressor

## Status

Accepted

## Context

`models/_shared/ae.py` genuinely learns (sklearn `MLPRegressor` trained on
reconstruction, encoder half extracted manually in `transform`), but it is
CPU-only sklearn standing in for a real encoder/decoder torch AE. Three
consumers depend on it: `proposed/AE_TCN_SFSOINN`, `ablations/AE_TCN`,
`ablations/AE_SFSOINN`. ADR 0005 set the precedent of promoting baseline DNN
to torch on `get_device()`.

## Question

Promote to a real torch encoder→bottleneck→decoder on `get_device()`, or is
sklearn `MLPRegressor` an acceptable documented simplification for the shared
compressor specifically?

## Options

1. **Real torch AE** — encoder/decoder `nn.Linear` stacks, trained by
   reconstruction (MSE) with Adam on `get_device()`, seeded via
   `seed_torch`.
2. Keep sklearn `MLPRegressor` and document it as a simplification.
3. Variational AE (VAE) with KL regularization.

## Decision

Option 1. Reasoning:

- The AE is the first stage of the proposed pipeline; its latents feed the
  TCN and SF-SOINN. A CPU sklearn bottleneck both limits scale (Kaggle
  full-dataset runs) and breaks the "all deep components share one training
  substrate" coherence of the paper.
- ADR 0005 already established the torch-on-`get_device()` precedent with
  `seed_torch`/`print_device_check`; reusing it costs nothing extra.
- A VAE (option 3) is rejected: it changes the model class the ablations
  compare against and adds tuning surface; a plain denoising-style AE matches
  the config's `denoising_gate` naming (Gaussian input noise during training,
  clean reconstruction target).

### Spec

- Encoder: `Linear(in_dim, 128) → ReLU → Linear(128, 64) → ReLU →
  Linear(64, latent_dim)`; Decoder mirrors it back to `in_dim`.
- Training: MSE on standardized inputs, Adam(lr=1e-3), epochs from
  `cfg.training.epochs` (default 20), batch size 256; Gaussian noise
  σ=0.01 on inputs during training (denoising).
- `transform` returns encoder outputs (latents) as numpy — signature
  unchanged for consumers.
- Device: `get_device()`; verified via `print_device_check("AE", device)`.
- `save` stores `state_dict()` + scaler; consumers' `save()` payloads that
  touch `compressor._mlp` are updated ticket-scoped.

## Consequences

- `AECompressor.fit/transform` signatures unchanged; internals swap from
  sklearn to torch.
- `PCACompressor` (ADR 0003) is untouched — it remains the deterministic
  control by design.
- Smoke-verified only (`--limit`); full-dataset runs on Kaggle.