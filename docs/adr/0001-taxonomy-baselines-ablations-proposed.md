# 5-5-1 Taxonomy: Baselines vs Ablations vs Proposed

Eleven requested models mix independent reference points with architecture-family variants, so we split them as 5 true baselines (SVM, Random Forest, DNN, BiLSTM, Distillation), 5 ablations (standalone AE plus SF-SOINN, PCA plus SF-SOINN, AE plus TCN, PCA plus TCN, PCA plus TCN plus SF-SOINN), and 1 full proposed model (AE plus TCN plus SF-SOINN), because flattening them into one bucket would hide which gains come from the proposal versus generic classifier strength.

## Considered Options

- Flat layout under baselines and proposed only: rejected because ablations would be misread as independent baselines.
- 7-model synopsis subset only: rejected because the four extra PCA and TCN ablations isolate non-linearity and clustering contributions the synopsis cannot.

## Consequences

- Directory identity follows taxonomy: baselines, ablations, and proposed families each own their model directories, and results group by family.
