# PCA Honors Fit Contract as One-Shot Deterministic Transform

PCA has no epochs while AE trains over epochs, yet the Runner only knows one model contract, so the PCA wrapper implements the same fit, predict, and save methods where fit does a single deterministic computation and ignores validation and epoch settings, because forcing fake epochs would mislead seed aggregation and retention measurement.

## Considered Options

- Separate PCA-only contract outside the Runner: rejected because the Runner would need branching for every ablation and comparison rows would lose uniformity.
- Fake epoch loop around PCA to mirror AE: rejected because it would fabricate loss curves and training time with no scientific meaning.

## Consequences

- PCA fit stays reproducible across seeds given fixed hyperparameters, save persists the fitted transform, and downstream TCN and SF-SOINN code consumes PCA and AE outputs interchangeably.
