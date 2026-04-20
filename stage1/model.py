"""
Stage 1: Traffic Count Prediction Models & Feature Engineering
From scratch with NumPy/Pandas.
"""

from __future__ import annotations
import math
import json
import os
import numpy as np
import pandas as pd

# ==========================================
# 0. Shared Utility Functions
# ==========================================
def calculate_metrics(y_true, y_pred):
    mae    = np.mean(np.abs(y_true - y_pred))
    rmse   = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2     = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    wmape  = np.sum(np.abs(y_true - y_pred)) / (np.sum(y_true) + 1e-8)
    return {"mae": mae, "rmse": rmse, "r2": r2, "wmape": wmape * 100}

def chronological_split_3way(df, val_ratio=0.2, test_ratio=0.2):
    n          = len(df)
    val_start  = int(n * (1 - val_ratio - test_ratio))
    test_start = int(n * (1 - test_ratio))
    return (
        df.iloc[:val_start].copy(),
        df.iloc[val_start:test_start].copy(),
        df.iloc[test_start:].copy()
    )

def get_kfold_indices(n, n_splits):
    """Pure NumPy replacement for sklearn.model_selection.KFold"""
    indices = np.arange(n)
    folds = np.array_split(indices, n_splits)
    for i in range(n_splits):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_splits) if j != i])
        yield train_idx, val_idx

# ==========================================
# 1. Target Encoding Pipeline
# ==========================================
TARGET_ENC_GROUPS = [
    ['zip_code'],
    ['weekday'],
    ['is_peak'],
    ['zip_code', 'is_peak'],
    ['zip_code', 'weekday'],
    ['weekday', 'is_peak'],
    ['zip_code', 'weekday', 'is_peak'],
]

def build_te_mappings(df, target='log_traffic_count'):
    """Builds a dictionary mapping for out-of-sample Target Encoding application."""
    global_mean = float(df[target].mean())
    mappings = {}
    for keys in TARGET_ENC_GROUPS:
        group_name = '_'.join(keys)
        if len(keys) == 1:
            s = df.groupby(keys[0])[target].mean()
            mappings[group_name] = {str(k): float(v) for k, v in s.items()}
        else:
            s = df.groupby(keys)[target].mean()
            mappings[group_name] = {'|'.join(str(x) for x in k): float(v) for k, v in s.items()}
    return global_mean, mappings

def apply_te_mappings(df, global_mean, mappings):
    """Applies stored Target Encoding mappings to new/test data."""
    df_out = df.copy()
    for keys in TARGET_ENC_GROUPS:
        group_name = '_'.join(keys)
        col_name = 'te_' + group_name
        if len(keys) == 1:
            lookup_keys = df_out[keys[0]].astype(str)
        else:
            lookup_keys = df_out[keys].astype(str).agg('|'.join, axis=1)
        
        df_out[col_name] = lookup_keys.map(mappings[group_name]).fillna(global_mean)
    return df_out

def add_target_encodings_oof(train_df, val_df, test_df, target='log_traffic_count', n_splits=5):
    """K-Fold Out-of-Fold encoding for train to prevent leakage, plus standard mapping for val/test."""
    global_mean = train_df[target].mean()
    oof_encodings = {'te_' + '_'.join(keys): np.full(len(train_df), global_mean) for keys in TARGET_ENC_GROUPS}
    
    for fit_pos, oof_pos in get_kfold_indices(len(train_df), n_splits):
        fit_df = train_df.iloc[fit_pos]
        oof_df = train_df.iloc[oof_pos]
        _, fold_mappings = build_te_mappings(fit_df, target)
        oof_mapped = apply_te_mappings(oof_df, global_mean, fold_mappings)
        for keys in TARGET_ENC_GROUPS:
            col = 'te_' + '_'.join(keys)
            oof_encodings[col][oof_pos] = oof_mapped[col].values

    train_out = train_df.copy()
    for col, values in oof_encodings.items():
        train_out[col] = values

    # Apply full-train mappings to val and test
    global_mean, full_mappings = build_te_mappings(train_df, target)
    val_out = apply_te_mappings(val_df, global_mean, full_mappings)
    test_out = apply_te_mappings(test_df, global_mean, full_mappings)

    return train_out, val_out, test_out, global_mean, full_mappings

# ==========================================
# 2. Model Definitions
# ==========================================
class RidgeRegressionScratch:
    def __init__(self, l2_penalty=10.0, **kwargs):
        self.l2_penalty = float(l2_penalty)
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n, f = X.shape
        Xb = np.c_[np.ones(n), X]
        I = np.eye(f + 1)
        I[0, 0] = 0
        A = np.dot(Xb.T, Xb) + self.l2_penalty * I
        w = np.dot(np.linalg.pinv(A), np.dot(Xb.T, y))
        self.bias = w[0]
        self.weights = w[1:]
        return self

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
    def save(self, path, feature_names=None, extra=None):
        return _save_model_npz(self, "ridge", path, feature_names, extra)
    
    @classmethod
    def load(cls, path):
        return _load_model_npz(path, expected_type="ridge")


class BayesianRegressionScratch:
    def __init__(self, alpha=1.0, beta=None, **kwargs):
        self.alpha = float(alpha)
        self.beta = beta
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y):
        n, f = X.shape
        if self.beta is None:
            self.beta = 1.0 / (np.var(y) + 1e-8)
        Xb = np.c_[np.ones(n), X]
        I = np.eye(f + 1)
        I[0, 0] = 0
        S_N_inv = self.alpha * I + self.beta * np.dot(Xb.T, Xb)
        self.w_cov = np.linalg.pinv(S_N_inv)
        self.w_mean = self.beta * np.dot(self.w_cov, np.dot(Xb.T, y))
        return self

    def predict(self, X):
        Xb = np.c_[np.ones(X.shape[0]), X]
        return np.dot(Xb, self.w_mean)
    
    def save(self, path, feature_names=None, extra=None):
        return _save_model_npz(self, "bayesian", path, feature_names, extra)
    
    @classmethod
    def load(cls, path):
        return _load_model_npz(path, expected_type="bayesian")

# ==========================================
# 3. Persistence & Schemas
# ==========================================
def _save_model_npz(model, model_type, path, feature_names=None, extra=None):
    if hasattr(model, 'weights') and model.weights is None:
        raise RuntimeError("Cannot save an unfitted model.")
    path = str(path)
    if not path.endswith(".npz"): path += ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    hp = {"l2_penalty": getattr(model, "l2_penalty", None),
          "alpha": getattr(model, "alpha", None),
          "beta": getattr(model, "beta", None)}
    
    weights = getattr(model, "weights", getattr(model, "w_mean", None))
    bias = getattr(model, "bias", 0.0) # w_mean has bias at index 0 for Bayesian

    meta = {
        "model_type": model_type,
        "hyperparameters": hp,
        "feature_names": list(feature_names) if feature_names else None,
        "extra": extra or {},
    }

    np.savez(path, weights=np.asarray(weights, dtype=np.float64), 
             bias=np.asarray(bias, dtype=np.float64), meta=np.array(json.dumps(meta)))
    return path

def _load_model_npz(path, expected_type=None):
    with np.load(path, allow_pickle=False) as arch:
        weights = np.asarray(arch["weights"], dtype=np.float64)
        bias = np.asarray(arch["bias"], dtype=np.float64)
        meta = json.loads(str(arch["meta"]))

    mt = meta["model_type"]
    if expected_type and mt != expected_type:
        raise ValueError(f"Expected {expected_type}, got {mt}")

    hp = meta["hyperparameters"]
    if mt == "ridge":
        m = RidgeRegressionScratch(l2_penalty=hp["l2_penalty"])
        m.weights, m.bias = weights, bias
    else:
        m = BayesianRegressionScratch(alpha=hp["alpha"], beta=hp["beta"])
        m.w_mean = weights # Bayesian w_mean encompasses bias at index 0 in predict

    m.feature_names_ = meta.get("feature_names")
    m.meta_ = meta
    return m

def build_ohe_matrices(train_df, val_df=None, test_df=None, target_col='log_traffic_count'):
    drop_cols = [c for c in [target_col, 'log_crash_count', 'crash_count'] if c in train_df.columns]
    def encode(df):
        if df is None: return None
        d = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()
        if 'zip_code' in d.columns: d['zip_code'] = d['zip_code'].astype(str)
        if 'weekday' in d.columns: d['weekday']  = d['weekday'].astype(str)
        return pd.get_dummies(d, drop_first=True)

    X_tr_df = encode(train_df)
    features = list(X_tr_df.columns)
    
    X_tr = X_tr_df.values.astype(float)
    X_vl = None
    X_te = None

    if val_df is not None:
        X_vl_df = encode(val_df)
        X_tr_df, X_vl_df = X_tr_df.align(X_vl_df, join='left', axis=1, fill_value=0)
        X_vl = X_vl_df.values.astype(float)
    if test_df is not None:
        X_te_df = encode(test_df)
        X_tr_df, X_te_df = X_tr_df.align(X_te_df, join='left', axis=1, fill_value=0)
        X_te = X_te_df.values.astype(float)

    return X_tr, X_vl, X_te, features

def _featurize_to_schema(df, feature_names):
    """Aligns inference data to the exact schema the model was trained on."""
    cats = pd.get_dummies(
        df[["zip_code", "weekday"]].astype("category"),
        prefix=["zip_code", "weekday"],
        drop_first=False, dtype=np.float64
    )
    raw = pd.concat([cats, df.drop(columns=["zip_code", "weekday"]).astype(np.float64)], axis=1)
    aligned = raw.reindex(columns=feature_names, fill_value=0.0)
    return aligned.to_numpy(dtype=np.float64)