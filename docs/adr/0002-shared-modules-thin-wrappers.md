# Shared Modules with Thin Wrappers

AE, PCA, TCN, and SF-SOINN each back multiple final models, so we implement each once as a shared module and keep every model directory as a thin composer that only wires shared pieces via its factory, because eleven duplicated copies would diverge and make ablations incomparable.

## Considered Options

- Duplicate AE, TCN, and SF-SOINN code per model directory: rejected because fixes would need eleven edits and ablation deltas would be untrustworthy.
- Shared code under src components: rejected because model-family code should stay beside models for discoverability by the Runner scanner.

## Consequences

- Shared modules own all training logic and hyperparameters; model directories own only identity and composition, and contract tests target the shared modules first.
