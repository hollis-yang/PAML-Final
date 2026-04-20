"""
Stage 1 driver: Load data, engineer features (OOF TE), tune models, report test set.
Run:
    python -m stage1.main --csv data_engineering.csv
"""

import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

from stage1.model import (
    calculate_metrics,
    chronological_split_3way,
    add_target_encodings_oof,
    build_ohe_matrices,
    RidgeRegressionScratch,
    BayesianRegressionScratch
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data_engineering.csv")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def tune_hyperparameters(X_train, y_train, X_val, y_val_real):
    # Ridge
    print("--- Tuning Ridge on VALIDATION Set ---")
    penalties = [50.0, 10.0, 1.0, 0.1, 0.01, 0.0]
    best_ridge_rmse, best_l2 = float('inf'), None
    for p in penalties:
        m = RidgeRegressionScratch(l2_penalty=p).fit(X_train, y_train)
        rmse = np.sqrt(np.mean((y_val_real - np.expm1(m.predict(X_val))) ** 2))
        print(f"L2={p:<6} | Val RMSE={rmse:.2f}")
        if rmse < best_ridge_rmse:
            best_ridge_rmse, best_l2 = rmse, p

    # Bayesian
    print("\n--- Tuning Bayesian on VALIDATION Set ---")
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0]
    best_bayes_rmse, best_alpha = float('inf'), None
    for a in alphas:
        m = BayesianRegressionScratch(alpha=a).fit(X_train, y_train)
        rmse = np.sqrt(np.mean((y_val_real - np.expm1(m.predict(X_val))) ** 2))
        print(f"Alpha={a:<6} | Val RMSE={rmse:.2f}")
        if rmse < best_bayes_rmse:
            best_bayes_rmse, best_alpha = rmse, a

    print(f"\n>> BEST: Ridge L2={best_l2}, Bayesian Alpha={best_alpha}\n")
    return best_l2, best_alpha

def main():
    args = parse_args()
    print("=" * 72)
    print("Stage 1: Traffic Volume Prediction Pipeline (Offline Experiment)")
    print("=" * 72)

    df = pd.read_csv(args.csv)
    target = 'log_traffic_count'

    tr_df, vl_df, te_df = chronological_split_3way(df, val_ratio=0.15, test_ratio=0.15)
    print(f"[split] Train: {len(tr_df)}  Val: {len(vl_df)}  Test: {len(te_df)}")

    # Target Encoding
    tr_df, vl_df, te_df, _, _ = add_target_encodings_oof(tr_df, vl_df, te_df)
    
    y_train = tr_df[target].values
    y_val = vl_df[target].values
    y_test_real = np.expm1(te_df[target].values)

    # Build OHE
    X_tr, X_vl, X_te, feats = build_ohe_matrices(tr_df, vl_df, te_df)
    print(f"[features] Dimension: {len(feats)}")

    best_l2, best_alpha = tune_hyperparameters(X_tr, y_train, X_vl, np.expm1(y_val))

    print("--- Final Training (Train + Val) -> TEST ---")
    X_full = np.vstack([X_tr, X_vl])
    y_full = np.concatenate([y_train, y_val])

    ridge = RidgeRegressionScratch(l2_penalty=best_l2).fit(X_full, y_full)
    r_preds = np.expm1(ridge.predict(X_te))
    
    bayes = BayesianRegressionScratch(alpha=best_alpha).fit(X_full, y_full)
    b_preds = np.expm1(bayes.predict(X_te))

    rm = calculate_metrics(y_test_real, r_preds)
    bm = calculate_metrics(y_test_real, b_preds)

    print(f"\n{'Model':<15} {'RMSE':>8} {'MAE':>8} {'R2':>8} {'WMAPE':>9}")
    print("-" * 55)
    print(f"{'Ridge':<15} {rm['rmse']:>8.1f} {rm['mae']:>8.1f} {rm['r2']:>8.4f} {rm['wmape']:>8.2f}%")
    print(f"{'Bayesian':<15} {bm['rmse']:>8.1f} {bm['mae']:>8.1f} {bm['r2']:>8.4f} {bm['wmape']:>8.2f}%")

if __name__ == "__main__":
    main()