"""
Stage 1: Production Pipeline and Frontend Inference API.

CLI Examples:
    python -m stage1.predict fit
    python -m stage1.predict predict input.csv output.csv

Python API (for Streamlit):
    from stage1.predict import TrafficPredictor
    model = TrafficPredictor()
    result = model.predict_one({"zip_code": 10001, "is_peak": 1, ...})
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from stage1.model import (
    build_te_mappings, apply_te_mappings, build_ohe_matrices, 
    _featurize_to_schema, _load_model_npz,
    RidgeRegressionScratch, BayesianRegressionScratch
)

# Best hyperparameters discovered in Stage 1 offline experiment
BEST_L2 = 0.0
BEST_ALPHA = 0.001
DEFAULT_MODEL_DIR = "stage1/checkpoints/final"

# ---------------------------------------------------------------------------
# CLI: fit (Train on ALL data for production)
# ---------------------------------------------------------------------------
def cmd_fit(args):
    print("=" * 72)
    print("Stage 1 Final Fit (Full Panel, 100% Data, saving production models)")
    print("=" * 72)
    
    df = pd.read_csv(args.csv)
    target = 'log_traffic_count'
    
    # Generate Target Encoding Mappings on full dataset
    global_mean, te_mappings = build_te_mappings(df, target)
    df_encoded = apply_te_mappings(df, global_mean, te_mappings)
    
    y = df_encoded[target].values
    X, _, _, feature_names = build_ohe_matrices(df_encoded, target_col=target)
    
    save_dir = Path(args.model_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Store TE mappings into 'extra' so the predictor can use them at inference
    extra_meta = {
        "te_global_mean": global_mean,
        "te_mappings": te_mappings
    }

    # Train & Save Ridge
    ridge = RidgeRegressionScratch(l2_penalty=BEST_L2).fit(X, y)
    r_path = ridge.save(save_dir / "ridge.npz", feature_names, extra_meta)
    
    # Train & Save Bayesian
    bayes = BayesianRegressionScratch(alpha=BEST_ALPHA).fit(X, y)
    b_path = bayes.save(save_dir / "bayesian.npz", feature_names, extra_meta)
    
    print(f"[Fit] Processed {len(df)} rows, {len(feature_names)} features.")
    print(f"[Save] Ridge -> {r_path}\n[Save] Bayesian -> {b_path}")

# ---------------------------------------------------------------------------
# Python API: TrafficPredictor (Frontend Integration)
# ---------------------------------------------------------------------------
class TrafficPredictor:
    """Single entry point for Stage 1 inference (Streamlit ready)."""
    
    def __init__(self, model_dir=DEFAULT_MODEL_DIR):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Run `python -m stage1.predict fit` first.")
            
        self.ridge = _load_model_npz(model_dir / "ridge.npz", "ridge")
        self.bayes = _load_model_npz(model_dir / "bayesian.npz", "bayesian")
        
        # Restore encoding state
        self.feature_names = self.ridge.feature_names_
        extra = self.ridge.meta_["extra"]
        self.global_mean = extra["te_global_mean"]
        self.te_mappings = extra["te_mappings"]

    def predict_one(self, record: dict) -> dict:
        """
        Predict traffic count for a single incoming dictionary from Streamlit.
        Example: {"zip_code": "10001", "weekday": "2", "is_peak": 1, "tavg_z": 0.5, ...}
        """
        batch_results = self.predict_frame(pd.DataFrame([record]))
        return {
            "traffic_ridge": float(batch_results["ridge"][0]),
            "traffic_bayesian": float(batch_results["bayesian"][0])
        }

    def predict_frame(self, df: pd.DataFrame) -> dict:
        """Batch prediction preserving logic schemas."""
        # 1. Apply saved target encodings
        df_encoded = apply_te_mappings(df, self.global_mean, self.te_mappings)
        # 2. Align features to exact training schema
        X = _featurize_to_schema(df_encoded, self.feature_names)
        
        # 3. Predict and invert log transformation
        return {
            "ridge": np.expm1(self.ridge.predict(X)),
            "bayesian": np.expm1(self.bayes.predict(X))
        }

def cmd_predict(args):
    predictor = TrafficPredictor(args.model_dir)
    df = pd.read_csv(args.input)
    preds = predictor.predict_frame(df)
    df["pred_traffic_ridge"] = preds["ridge"]
    df["pred_traffic_bayesian"] = preds["bayesian"]
    df.to_csv(args.output, index=False)
    print(f"[Predict] Scored {len(df)} rows. Saved to {args.output}")

# ---------------------------------------------------------------------------
# CLI Setup
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    
    pf = sub.add_parser("fit")
    pf.add_argument("--csv", default="data_engineering.csv")
    pf.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pf.set_defaults(func=cmd_fit)
    
    pp = sub.add_parser("predict")
    pp.add_argument("input")
    pp.add_argument("output")
    pp.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    pp.set_defaults(func=cmd_predict)
    
    args = p.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()