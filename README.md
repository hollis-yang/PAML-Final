# A Machine Learning Approach to ZIP-Code-Level Traffic Crash Prediction in New York City

## Quick Start

```bash
uv sync
uv run streamlit run streamlit/app.py
```

## Stage 1 — Traffic Volume Models (Ridge / Bayesian Linear)

Stage 1 fits a Ridge Regression and a Bayesian Linear Regression to the
engineered panel in `data/data_engineering.csv`, using a from-scratch NumPy
implementation with closed-form analytical solutions.

### Offline experiment and hyperparameter tuning

Run the offline experiment pipeline. A strict 70/15/15 chronological
split is used, combined with out-of-fold (OOF) target encoding to prevent
in-sample leakage. Grid searches are performed exclusively on the validation
set, and unbiased metrics are reported on the held-out test set.

```bash
uv run python -m stage1.main --csv data/data_engineering.csv
```

### Final prediction pipeline

**1) Refit on the full panel.** Retrain Ridge + Bayesian on ALL rows with
the best hyperparameters discovered offline. Computes global target encoding
mappings and writes `stage1/checkpoints/final/{ridge,bayesian}.npz`.

```bash
uv run python -m stage1.predict fit
```

**2) Single-record scoring (Streamlit).** `TrafficPredictor.predict_one(record)`
dynamically reconstructs target encodings and exact One-Hot schemas. It
returns a dict with `traffic_ridge` and `traffic_bayesian` — used by the
Streamlit app.

**3) Batch CSV scoring.** Score a new raw-feature CSV; the output CSV
gets `pred_traffic_ridge` and `pred_traffic_bayesian` columns.

```bash
uv run python -m stage1.predict predict \
    data/new_panel.csv  data/new_panel_pred.csv
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

**1) Refit on the full panel.** Retrain Poisson + NB on ALL rows with
the best hyperparameters; writes
`stage2/checkpoints/final/{poisson,nb}.npz`.

```bash
uv run python stage2/predict.py fit
```

**2) Single-record scoring (Streamlit).** `CrashPredictor.predict_one(record)`
returns a dict with `mu_poisson`, `mu_nb`, `nb_sd`, and the 95% NB
prediction interval — used by the Streamlit app.

**3) Batch CSV scoring.** Score a new raw-feature CSV; the output CSV
gets `pred_poisson` and `pred_nb` columns.

```bash
uv run python stage2/predict.py predict \
    data/new_panel.csv  data/new_panel_pred.csv
```

The input CSV must follow the same raw-feature schema as
`data_engineering.csv` (columns `zip_code`, `weekday`, `is_peak`,
`log_traffic_count`, weather flags/z-scores, and the `WT0x` indicators).
