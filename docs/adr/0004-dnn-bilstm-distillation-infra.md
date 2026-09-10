# DNN, BiLSTM, and Distillation Reuse Torch Harness with Targeted Extensions

DNN, BiLSTM, and Distillation do not need new Runner plumbing beyond what TCN and AE already require, so DNN reuses the shared torch classifier harness with different layers, BiLSTM adds sequence handling plus incremental support for retention measurement, and Distillation adds a teacher-student helper with temperature and weighting, because assuming full reuse would hide sequence and distillation specifics while full isolation would triple harness code.

## Considered Options

- Fully standalone implementations with no shared torch helpers: rejected because training loops, device handling, and metric hooks would be copied three times.
- Forcing all three through the identical TCN harness with no extensions: rejected because BiLSTM needs sequential fine-tuning and Distillation needs soft-target loss the base harness lacks.

## Consequences

- No Runner changes are needed; new shared helpers cover sequence batching, incremental fine-tuning, and teacher-student training, and each baseline stays comparable on latency and memory.
