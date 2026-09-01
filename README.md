# IDS_CL_SF-SOINN

Continual-learning NIDS: Autoencoder → 1D-TCN (known-threat classification) → SF-SOINN (zero-day clustering, no backprop). Benchmarked against Random Forest, SVM, and BiLSTM on NetML 2020 and CICIDS2017.

**Status:** folder structure only — no code written yet. This README describes the intended workflow so contributions land in the right place.

## Structure (planned)

IDS_CL_SF-SOINN/
├── .github/
│   └── workflows/
├── data/
│   ├── raw/
│   │   ├── netML2020/
│   │   │   ├── training.csv
│   │   │   ├── annotation.csv
│   │   │   └── testing.csv
│   │   └── cicids2017/
│   │       ├── training.csv
│   │       ├── annotation.csv
│   │       └── testing.csv
│   ├── processed/
│   │   ├── netML/
│   │   └── cicids2027/
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
│   │   └── AE_TCN_SF-SOINN/
│   │       ├── seed_42/
│   │       ├── seed_123/
│   │       ├── .
│   │       ├── .
│   │       └── .
│   └── baseline/
│       ├── Random Forest/
│       ├── SVM/
│       └── BiLSTM/
├── reports/
│   ├── proposed/
│   └── comparsion results/
├── logs/
├── configs/
├── .gitignore
├── requirements.txt
├── references.bib
└── README.md
## Seed plan

| Model | Seeds |
|---|---|
| Random Forest | 1 |
| SVM | 1 |
| BiLSTM | 3 (`42, 123, 7`) |
| Proposed (AE+TCN+SF-SOINN) | 5 (`42, 123, 7, 2024, 999`) |

## Next steps

1. Write `download_data.py` and pull the raw data.
2. Generate and commit the split files in `data/splits/`.
3. Write `dataloader.py` against the committed split.
4. Write `logger.py` (`log_result`) and `config_loader.py`.
5. Build one reference model end-to-end (e.g. Random Forest) as the template for everyone else.

Once step 5 lands, this README will be updated with the actual usage commands.
