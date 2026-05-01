"""
Stage 2: retrain on the full engineered panel with the best hyperparameters
found by grid search, then predict crash counts on new data.

Two CLI subcommands and a Python API.

CLI
---
  fit      -- Refit Poisson + NB on ALL rows of the engineered panel and save
              final models to stage2/checkpoints/final/. The grid-search step
              has already happened (see main.py); no held-out evaluation runs
              here — downstream use is pure prediction.

  predict  -- Load the saved final models and score a new CSV that follows the
              same raw-feature schema as data_engineering.csv. Outputs a CSV
              with `pred_poisson` and `pred_nb` columns appended.

Python API (for Streamlit / notebooks / single-row inference)
-------------------------------------------------------------
    from stage2.predict import CrashPredictor
    predictor = CrashPredictor()              # loads models once
    result = predictor.predict_one({          # single-row dict
        "zip_code": 11207, "weekday": 1, "is_peak": 1,
        "log_traffic_count": 8.5,
        "is_rain": 0, "log_prcp": 0.0,
        "is_snow": 0, "log_snow": 0.0,
        "has_snow_depth": 0, "log_snwd": 0.0,
        "tavg_z": 0.3, "awnd_z": -0.2,
        "WT01": 0, "WT02": 0, "WT03": 0, "WT04": 0, "WT06": 0, "WT08": 0,
    })
    # -> {"mu_poisson": 1.87, "mu_nb": 1.89, "nb_sd": 1.63,
    #     "nb_ci95": (0, 5), "alpha": 0.2}

CLI examples
------------
    uv run python stage2/predict.py fit
    uv run python stage2/predict.py predict \\
        data/new_panel.csv  data/new_panel_pred.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from stage2.model import (
        _BINARY_FEATURES,
        _CONT_FEATURES,
        load_model,
        load_stage2_data,
        train_model,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import (
        _BINARY_FEATURES,
        _CONT_FEATURES,
        load_model,
        load_stage2_data,
        train_model,
    )


# Best config from grid search on the engineered panel
# (see main.py for the grids that produced these):
#   Poisson: val PoissonDev minimized at lambda_reg=1e-3 under IRLS.
#   NB     : val NLL minimized at alpha=0.2, lambda_reg=1e-3 under IRLS.
BEST_POISSON = dict(solver="irls", lambda_reg=1e-3)
BEST_NB = dict(solver="irls", lambda_reg=1e-3, alpha=0.15)

# IRLS converges in ~9 iters via step-norm tolerance; these caps are generous.
FIT_MAX_ITER = 50
FIT_PATIENCE = 10

DEFAULT_CSV = "data/data_engineering.csv"
DEFAULT_MODEL_DIR = "stage2/checkpoints/final"


# ---------------------------------------------------------------------------
# Feature alignment for unseen CSVs
# ---------------------------------------------------------------------------

def _featurize_to_schema(df: pd.DataFrame,
                         feature_names: list[str]) -> np.ndarray:
    """
    Convert a raw-panel DataFrame into an (n, d) matrix aligned to the exact
    column schema the model was trained on.

    Parameters
    ----------
    df : DataFrame with the raw columns produced by the Stage 2 feature
        engineering pipeline (at minimum: zip_code, weekday, and every column
        in _BINARY_FEATURES / _CONT_FEATURES).
    feature_names : list[str]
        The `feature_names_` stored on the trained model. Starts with
        'intercept', then one-hot columns (prefixed 'zip_' and 'dow_'), then
        binary, then continuous — but this function does not rely on that
        ordering, only on the names themselves.

    Behavior
    --------
    * One-hot columns missing from the new data (e.g. a zip that only appears
      in training) are filled with 0 — equivalent to "this row is not that
      zip", which is correct.
    * One-hot columns present in the new data but not in training (new zip
      codes the model has never seen) are dropped with a warning; they get
      folded into the dropped baseline category.
    * Required raw columns that are missing raise a clear error.
    """
    required = set(_BINARY_FEATURES + _CONT_FEATURES + ["zip_code", "weekday"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input data is missing required columns: {sorted(missing)}")

    cats = pd.get_dummies(
        df[["zip_code", "weekday"]].astype("category"),
        prefix=["zip", "dow"],
        drop_first=False,          # keep every level; reindex drops extras
        dtype=np.float64,
    )
    raw = pd.concat(
        [cats,
         df[_BINARY_FEATURES].astype(np.float64),
         df[_CONT_FEATURES].astype(np.float64)],
        axis=1,
    )

    # A dummy column in the new data that isn't in feature_names is either
    # (a) the baseline level the training pipeline dropped — expected, no-op,
    # or (b) a genuinely new category the model has never seen. We can't tell
    # (a) from (b) without the training categorical index, so we warn only
    # when MORE than one baseline-looking column per prefix is missing.
    want = [c for c in feature_names if c != "intercept"]
    want_set = set(want)
    for prefix in ("zip_", "dow_"):
        extras = [c for c in raw.columns
                  if c.startswith(prefix) and c not in want_set]
        # One missing column per prefix is expected (drop_first=True); more
        # means unseen levels were present in the input.
        if len(extras) > 1:
            print(f"[predict] warning: {len(extras) - 1} unseen '{prefix}' "
                  f"level(s) in input; those rows will fall back to baseline.")

    aligned = raw.reindex(columns=want, fill_value=0.0)
    X = aligned.to_numpy(dtype=np.float64)
    if "intercept" in feature_names:
        X = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
    return X


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def _fit_with_best(model_type: str, X_fit: np.ndarray, y_fit: np.ndarray,
                   *, verbose: bool) -> object:
    """Train a single model on (X_fit, y_fit) with best hyperparameters.

    No validation set is passed — IRLS converges by step-norm tolerance, and
    feeding any subset of X_fit as 'val' would bias early stopping.
    """
    params = BEST_POISSON if model_type == "poisson" else BEST_NB
    return train_model(
        model_type, X_fit, y_fit,
        X_val=None, y_val=None,
        max_iter=FIT_MAX_ITER, patience=FIT_PATIENCE,
        verbose=verbose,
        **params,
    )


def cmd_fit(args: argparse.Namespace) -> None:
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path.resolve()}")

    print("=" * 72)
    print("Stage 2 final fit (best hyperparameters, IRLS)")
    print("=" * 72)

    t0 = time.time()
    X, y, feats = load_stage2_data(str(csv_path))
    print(f"[data] X={X.shape}  y={y.shape}  n_features={len(feats)}  "
          f"({time.time() - t0:.1f}s)")
    print(f"[fit] using ALL data: n={len(y)}  (no held-out eval)")

    save_dir = Path(args.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Poisson] {BEST_POISSON}")
    t0 = time.time()
    poisson = _fit_with_best("poisson", X, y, verbose=not args.quiet)
    print(f"[Poisson] n_iter={poisson.n_iter_}  "
          f"fit time={time.time() - t0:.1f}s")

    print(f"\n[NB] {BEST_NB}")
    t0 = time.time()
    nb = _fit_with_best("nb", X, y, verbose=not args.quiet)
    print(f"[NB] n_iter={nb.n_iter_}  fit time={time.time() - t0:.1f}s")

    extra = {"train_size": int(X.shape[0]),
             "note": "fit on full engineered panel, no held-out eval"}
    p_path = poisson.save(save_dir / "poisson.npz",
                          feature_names=feats, extra=dict(extra))
    nb_path = nb.save(save_dir / "nb.npz",
                      feature_names=feats, extra=dict(extra))
    print(f"\n[save] Poisson -> {p_path}")
    print(f"[save] NB      -> {nb_path}")


# ---------------------------------------------------------------------------
# predict — Python API
# ---------------------------------------------------------------------------

class CrashPredictor:
    """Single entry point for Stage 2 inference.

    Loads both the final Poisson and NB models once, caches their shared
    feature schema, and exposes two methods:
      * `predict_one(record)`  — single observation dict (Streamlit / app use)
      * `predict_frame(df)`    — batch DataFrame (CSV pipeline use)

    Both models are loaded regardless of which is queried, because loading is
    cheap (~1.5 MB npz each) and the typical caller wants both numbers.
    """

    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Model dir not found: {model_dir.resolve()}. "
                f"Run `python stage2/predict.py fit` first.")
        self.model_dir = model_dir
        self.poisson = load_model(model_dir / "poisson.npz")
        self.nb = load_model(model_dir / "nb.npz")
        # feature_names_ is attached dynamically by _load_model_npz; go
        # through getattr so static analyzers don't complain.
        fnames = getattr(self.poisson, "feature_names_", None)
        if fnames is None:
            raise RuntimeError(
                "Loaded model is missing feature_names_ metadata; "
                "regenerate it via `python stage2/predict.py fit`.")
        self.feature_names: list[str] = list(fnames)
        self.alpha = float(getattr(self.nb, "alpha", 0.2))

    # ----- single-row -------------------------------------------------

    def predict_one(self, record: dict) -> dict:
        """Predict crash count for one (zip, weekday, is_peak, weather) row.

        Parameters
        ----------
        record : dict
            One row of raw engineered features. Must contain the same keys as
            a row of data_engineering.csv except for the target.

        Returns
        -------
        dict with
            mu_poisson : float   — Poisson conditional mean
            mu_nb      : float   — NB conditional mean
            nb_sd      : float   — sqrt(mu + alpha * mu^2)
            nb_ci95    : (int,int) — 95% prediction interval for observed count
            alpha      : float   — NB dispersion parameter actually used
        """
        batch = self.predict_frame(pd.DataFrame([record]))
        return {
            "mu_poisson": float(batch["mu_poisson"][0]),
            "mu_nb": float(batch["mu_nb"][0]),
            "nb_sd": float(batch["nb_sd"][0]),
            "nb_ci95": (int(batch["nb_lo95"][0]), int(batch["nb_hi95"][0])),
            "alpha": self.alpha,
        }

    # ----- batch ------------------------------------------------------

    def predict_frame(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
        """Predict for every row of a DataFrame. Returns per-row arrays."""
        X = _featurize_to_schema(df, self.feature_names)
        mu_p = self.poisson.predict(X)
        mu_nb = self.nb.predict(X)
        sd = np.sqrt(mu_nb + self.alpha * mu_nb ** 2)
        lo, hi = _nb_interval(mu_nb, self.alpha, level=0.95)
        return {
            "mu_poisson": mu_p,
            "mu_nb": mu_nb,
            "nb_sd": sd,
            "nb_lo95": lo,
            "nb_hi95": hi,
        }


def _nb_interval(mu: np.ndarray, alpha: float,
                 *, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """NB2 prediction interval at integer quantiles.

    NB2 with mean mu and Var = mu + alpha*mu^2 maps to scipy's (n, p)
    parameterization as:
        n = 1 / alpha               (same for all rows; must be > 0)
        p = 1 / (1 + alpha * mu)    (per row)
    """
    from scipy.stats import nbinom
    if alpha <= 0:
        # Degenerate: NB collapses to Poisson; use Poisson quantiles.
        from scipy.stats import poisson
        lo_q = (1 - level) / 2
        hi_q = 1 - lo_q
        return (poisson.ppf(lo_q, mu).astype(int),
                poisson.ppf(hi_q, mu).astype(int))
    n = 1.0 / alpha
    p = 1.0 / (1.0 + alpha * np.asarray(mu, dtype=np.float64))
    lo_q = (1 - level) / 2
    hi_q = 1 - lo_q
    return (nbinom.ppf(lo_q, n, p).astype(int),
            nbinom.ppf(hi_q, n, p).astype(int))


# ---------------------------------------------------------------------------
# predict — CLI wrapper
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found: {in_path.resolve()}")

    print(f"[predict] loading models from {args.model_dir}")
    predictor = CrashPredictor(args.model_dir)

    print(f"[predict] reading {in_path}")
    df = pd.read_csv(in_path)
    preds = predictor.predict_frame(df)
    print(f"[predict] scored n={len(df)} rows; "
          f"mu_nb mean={preds['mu_nb'].mean():.3f}  "
          f"max={preds['mu_nb'].max():.3f}")

    out = df.copy()
    out["pred_poisson"] = preds["mu_poisson"]
    out["pred_nb"] = preds["mu_nb"]
    out["nb_sd"] = preds["nb_sd"]
    out["nb_lo95"] = preds["nb_lo95"]
    out["nb_hi95"] = preds["nb_hi95"]

    if "log_crash_count" in df.columns:
        y_true = np.rint(np.expm1(df["log_crash_count"].to_numpy())).clip(0)
        for name, col in [("poisson", "pred_poisson"), ("nb", "pred_nb")]:
            mu = out[col].to_numpy()
            mae = float(np.mean(np.abs(y_true - mu)))
            rmse = float(np.sqrt(np.mean((y_true - mu) ** 2)))
            print(f"[predict] {name} on input (decoded target): "
                  f"MAE={mae:.4f}  RMSE={rmse:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[predict] wrote {len(out)} rows -> {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pf = sub.add_parser("fit", help="Refit on the full panel with best hp.")
    pf.add_argument("--csv", default=DEFAULT_CSV,
                    help="Engineered panel CSV.")
    pf.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Directory to save final models into.")
    pf.add_argument("--quiet", action="store_true")
    pf.set_defaults(func=cmd_fit)

    pp = sub.add_parser("predict", help="Score a new raw-panel CSV.")
    pp.add_argument("input", help="Input CSV with raw Stage 2 features.")
    pp.add_argument("output", help="Output CSV with pred_* columns appended.")
    pp.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                    help="Directory holding poisson.npz and nb.npz.")
    pp.set_defaults(func=cmd_predict)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
