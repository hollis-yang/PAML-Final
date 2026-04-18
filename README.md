# A Machine Learning Approach to ZIP-Code-Level Traffic Crash Prediction in New York City

## Quick Start

```bash
uv sync
uv run streamlit run streamlit/app.py
```

## Stage 2 — Crash Count Models (Poisson / Negative Binomial)

Stage 2 fits a Poisson GLM and a Negative Binomial (NB2) GLM to the
engineered panel in `data/data_engineering.csv`, using a from-scratch NumPy
implementation with either batch gradient descent or IRLS/Newton.

### Train with defaults

Defaults are already set to the best configuration found by grid search
(IRLS solver, `lambda_reg=1e-3`, NB `alpha=0.2`). A 70/15/15 positional
split is used; metrics are reported on the held-out test set.

```bash
uv run python stage2/main.py --save-dir stage2/checkpoints
```

### Grid search (hyperparameter tuning)

Run per-model grid searches. Poisson tunes on validation Poisson deviance;
NB tunes on validation NLL (deviance biases NB selection toward the Poisson
limit).

```bash
uv run python stage2/main.py \
    --poisson-solver irls --nb-solver irls \
    --tune --max-iter 30 --patience 10 \
    --save-dir stage2/checkpoints
```

Force the slower GD path if needed:

```bash
uv run python stage2/main.py \
    --poisson-solver gd --nb-solver gd \
    --tune --max-iter 3000 --patience 50
```

### Final prediction pipeline

Retrain on the full panel with the best hyperparameters, then score new
raw-feature CSVs:

```bash
# 1) Refit Poisson + NB on ALL rows; writes stage2/checkpoints/final/{poisson,nb}.npz
uv run python stage2/predict.py fit

# 2) Predict on new data. Output CSV gets pred_poisson and pred_nb columns.
uv run python stage2/predict.py predict \
    data/new_panel.csv  data/new_panel_pred.csv
```

The input CSV must follow the same raw-feature schema as
`data_engineering.csv` (columns `zip_code`, `weekday`, `is_peak`,
`log_traffic_count`, weather flags/z-scores, and the `WT0x` indicators).
