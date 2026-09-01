# IDS_CL_SF-SOINN

Continual-learning NIDS: Autoencoder → 1D-TCN (known-threat classification) → SF-SOINN (zero-day clustering, no backprop). Benchmarked against Random Forest, SVM, and BiLSTM on NetML 2020 and CICIDS2017.

**Status:** folder structure only — no code written yet. This README describes the intended workflow so contributions land in the right place.

## Structure (planned)

```
IDS_CL_SF-SOINN/
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   │   ├── netml2020/
│   │   │   ├── training.csv
│   │   │   ├── annotation.csv
│   │   │   └── testing.csv
│   │   └── cicids2017/
│   │       ├── training.csv
│   │       ├── annotation.csv
│   │       └── testing.csv
│   ├── processed/
│   │   ├── netml2020/
│   │   └── cicids2017/
│   └── splits/
│       ├── netml_split.json
│       └── cicids2017_split.json
├── src/
│   ├── utils/
│   │   ├── cleaner.py
│   │   └── annotator.py
│   ├── models/
│   ├── dataloader.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   ├── proposed/
│   │   ├── AE/
│   │   ├── TCN/
│   │   └── AE_TCN_SFSOINN/
│   │       ├── seed_42/
│   │       ├── seed_123/
│   │       ├── seed_7/
│   │       ├── seed_2024/
│   │       └── seed_999/
│   └── baseline/
│       ├── random_forest/
│       ├── svm/
│       └── bilstm/
├── reports/
│   ├── proposed/
│   └── comparison_results/
├── logs/
├── configs/
├── .gitignore
├── requirements.txt
├── references.bib
└── README.md
```

## Seed plan

| Model | Seeds |
|---|---|
| Random Forest | 1 |
| SVM | 1 |
| BiLSTM | 3 (`42, 123, 7`) |
| Proposed (AE+TCN+SF-SOINN) | 5 (`42, 123, 7, 2024, 999`) |


## script usage

To download the data use :
    python src/utils/dwnld_data.py --dataset {netml2020 || cicids2017 || all (for both)}