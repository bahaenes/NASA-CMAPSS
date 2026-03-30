# Predictive Maintenance — NASA CMAPSS Turbofan Engine

End-to-end predictive maintenance pipeline: RUL prediction for aircraft turbofan engines using the NASA CMAPSS dataset.

## Project Structure

```
├── data/
│   ├── raw/                  # CMAPSS raw text files
│   └── processed/            # Parquet feature matrices (generated)
├── models/                   # Trained models + artifacts (generated)
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory data analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb     # Training, evaluation, SHAP
├── src/
│   ├── preprocessing.py      # Full preprocessing pipeline
│   ├── model_trainer.py      # Models, CV, NASA score
│   └── rul_predictor.py      # Inference + SHAP + plots
├── app/
│   └── streamlit_app.py      # Interactive dashboard
└── requirements.txt
```

## Dataset

NASA CMAPSS — 4 variants (FD001–FD004):

| Dataset | Train engines | Conditions | Fault modes |
|---|---|---|---|
| FD001 | 100 | 1 | HPC Degradation |
| FD002 | 260 | 6 | HPC Degradation |
| FD003 | 100 | 1 | HPC + Fan |
| FD004 | 248 | 6 | HPC + Fan |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Run notebooks in order:**

```bash
jupyter lab notebooks/
```

1. `01_eda.ipynb` — explore sensors, visualize degradation
2. `02_feature_engineering.ipynb` — generate features, save to `data/processed/`
3. `03_modeling.ipynb` — train models, save to `models/`

**Launch dashboard:**

```bash
streamlit run app/streamlit_app.py
```

## Models

| Model | FD001 RMSE | Notes |
|---|---|---|
| Linear Regression | ~23 | Baseline |
| Ridge | ~23 | L2 regularized baseline |
| Random Forest | ~16 | 200 trees, max_depth=15 |
| XGBoost | ~14 | 500 trees, lr=0.05, early stopping |
| LightGBM | ~14 | 500 trees, num_leaves=63 |

## Key Design Decisions

- **RUL cap = 125 cycles** — piecewise-linear target; all engines have min lifetime > 128 cycles
- **Per-condition normalization** — FD002/FD004 use 6 MinMaxScalers (one per operating cluster) to avoid conflating condition variance with degradation signal
- **Engine-level CV** — GroupKFold prevents time-series leakage; no engine appears in both train and validation folds
- **Rolling features** — windows of 5, 10, 30 cycles for noise smoothing and trend capture
- **Alarm threshold** — RUL < 30 cycles triggers maintenance alert

## Metrics

- **RMSE**, **MAE** — standard regression metrics
- **NASA Score** — asymmetric: late predictions (predicting more life than actual) are penalized more heavily than early predictions

## References

- Saxena et al. (2008). *Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation*. PHM'08.
- Dataset: [NASA Prognostics Center](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
