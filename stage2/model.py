"""
Stage 2: Crash Count Prediction Models (from scratch with NumPy).

Implements:
  - Poisson Regression (log-link, L2-regularized MLE via batch GD)
  - Negative Binomial Regression (NB2, fixed dispersion alpha)
  - Evaluation metrics: MAE, RMSE, Poisson Deviance
  - Likelihood Ratio Test (Poisson vs NB)

No sklearn / statsmodels / torch — only NumPy (+ scipy.stats.chi2 for p-value).
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

# Continuous features that are already standardized / log-scaled upstream.
_CONT_FEATURES = [
    "log_traffic_count", "log_prcp", "log_snow", "log_snwd",
    "tavg_z", "awnd_z",
]
# Binary flags already 0/1. WT0x are NOAA weather-type indicators:
#   WT01 fog, WT02 heavy fog, WT03 thunder, WT04 ice pellets/sleet,
#   WT06 glaze/rime, WT08 smoke/haze. WT05 (hail) is dropped because it is
# all-zero in this dataset and would contribute a dead column.
_BINARY_FEATURES = [
    "is_peak", "is_rain", "is_snow", "has_snow_depth",
    "WT01", "WT02", "WT03", "WT04", "WT06", "WT08",
]


def load_stage2_data(
    csv_path: str,
    *,
    target_col: str = "log_crash_count",
    decode_target: bool = True,
    drop_first: bool = True,
    add_intercept: bool = True,
):
    """
    Load the engineered Stage 2 panel and produce (X, y, feature_names).

    Parameters
    ----------
    csv_path : str
        Path to data_engineering.csv.
    target_col : str
        Target column name in the CSV (default 'log_crash_count').
    decode_target : bool
        If True, invert log1p to recover integer crash counts:
        y = round(expm1(log_crash_count)). Required for Poisson/NB.
    drop_first : bool
        Drop the first level of each categorical when one-hot encoding,
        to avoid perfect collinearity with the intercept.
    add_intercept : bool
        Prepend a constant-1 column to X.

    Returns
    -------
    X : np.ndarray  (n, d)  float64
    y : np.ndarray  (n,)    float64 (non-negative integers if decode_target)
    feature_names : list[str]
    """
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        raise ValueError(f"target_col={target_col!r} not found in {csv_path}")

    # Target
    y_raw = df[target_col].to_numpy(dtype=np.float64)
    if decode_target:
        y = np.rint(np.expm1(y_raw)).clip(min=0.0)
    else:
        y = y_raw

    # One-hot encode categoricals
    cats = pd.get_dummies(
        df[["zip_code", "weekday"]].astype("category"),
        prefix=["zip", "dow"],
        drop_first=drop_first,
        dtype=np.float64,
    )

    # Assemble feature matrix in a deterministic column order
    parts = [cats,
             df[_BINARY_FEATURES].astype(np.float64),
             df[_CONT_FEATURES].astype(np.float64)]
    X_df = pd.concat(parts, axis=1)

    feature_names = list(X_df.columns)
    X = X_df.to_numpy(dtype=np.float64)

    if add_intercept:
        X = np.hstack([np.ones((X.shape[0], 1), dtype=np.float64), X])
        feature_names = ["intercept"] + feature_names

    return X, y, feature_names


def temporal_split(
    X: np.ndarray, y: np.ndarray,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
):
    """
    Positional 70/15/15 split. The engineered CSV has no date column, so this
    assumes rows are ordered chronologically within each (zip, peak) group.
    """
    if abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    n = len(y)
    n_tr = int(n * ratios[0])
    n_va = int(n * (ratios[0] + ratios[1]))
    return (X[:n_tr], y[:n_tr],
            X[n_tr:n_va], y[n_tr:n_va],
            X[n_va:], y[n_va:])


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

# Safe range for the linear predictor x^T beta so that exp(.) neither
# overflows nor underflows to zero in float64.
_ETA_CLIP = 20.0
_EPS = 1e-12


def _safe_exp(eta: np.ndarray) -> np.ndarray:
    """Compute exp(eta) after clipping eta to [-_ETA_CLIP, _ETA_CLIP]."""
    return np.exp(np.clip(eta, -_ETA_CLIP, _ETA_CLIP))


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def poisson_deviance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Poisson deviance:  2 * sum( y*log(y/mu) - (y - mu) ).
    Handles y = 0 via the convention 0*log(0) = 0.
    """
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.clip(np.asarray(y_pred, dtype=np.float64), _EPS, None)
    # y * log(y/mu) with y=0 contributing 0
    term = np.zeros_like(y)
    mask = y > 0
    term[mask] = y[mask] * np.log(y[mask] / mu[mask])
    dev = 2.0 * np.sum(term - (y - mu))
    return float(dev)


# ---------------------------------------------------------------------------
# Poisson Regression
# ---------------------------------------------------------------------------

def _fit_irls_glm(model, X: np.ndarray, y: np.ndarray,
                  X_val: np.ndarray | None, y_val: np.ndarray | None,
                  *, family: str, alpha: float = 0.0,
                  patience: int = 10, verbose: bool = False):
    """
    Iteratively Reweighted Least Squares (IRLS) / Newton for a log-link GLM.

    Supports Poisson (family='poisson') and NB2 with fixed alpha
    (family='nb'). Each iteration solves an L2-regularized weighted least
    squares problem:

        (X^T W X + 2 * lambda_reg * I) beta_new = X^T W z

    with weights and working response:
        Poisson: W = mu,                  z = eta + (y - mu) / mu
        NB2:     W = mu / (1 + alpha*mu), z = eta + (y - mu) / mu

    `model` is an instance of PoissonRegression or NegBinomialRegression;
    its `lambda_reg`, `max_iter`, and `tol` are used as hyperparameters.

    Fills model.beta_, model.train_loss_, model.val_deviance_, model.n_iter_.
    Returns the fitted `model`.
    """
    n, d = X.shape
    beta = np.zeros(d, dtype=np.float64)

    prev_loss = model._loss(X, y, beta)
    model.train_loss_ = [prev_loss]
    model.val_deviance_ = []

    best_val = math.inf
    best_beta = beta.copy()
    best_iter = 0
    stale = 0

    use_val = X_val is not None and y_val is not None
    reg_eye = 2.0 * model.lambda_reg * np.eye(d, dtype=np.float64)

    tag = "Poisson-IRLS" if family == "poisson" else "NB-IRLS"
    if verbose:
        extra = f"  alpha={alpha}" if family == "nb" else ""
        print(f"[{tag}] fit  n={n}  d={d}  lambda_reg={model.lambda_reg}"
              f"{extra}  max_iter={model.max_iter}")
        print(f"[{tag}] iter   0  train_loss={prev_loss:.6f}")

    stop_reason = "max_iter"
    for it in range(1, model.max_iter + 1):
        eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        mu_safe = np.maximum(mu, _EPS)

        # Working response (same for Poisson and NB with log link)
        z = eta + (y - mu) / mu_safe

        # Weights
        if family == "poisson":
            W = mu_safe
        else:  # NB2
            W = mu_safe / (1.0 + alpha * mu_safe)

        # Normal equations: A = X^T W X + 2*lambda*I,  b = X^T W z
        Xw = X * W[:, None]
        A = X.T @ Xw + reg_eye
        b = X.T @ (W * z)

        try:
            beta_new = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Singular Hessian (can happen with perfectly collinear features
            # and lambda_reg=0); fall back to least-squares solve.
            beta_new, *_ = np.linalg.lstsq(A, b, rcond=None)

        step_norm = float(np.linalg.norm(beta_new - beta))
        beta = beta_new

        cur_loss = model._loss(X, y, beta)
        model.train_loss_.append(cur_loss)

        if use_val:
            mu_val = _safe_exp(X_val @ beta)
            if family == "poisson":
                dev = poisson_deviance(y_val, mu_val)
            else:
                dev = model.neg_log_likelihood(y_val, mu_val)
            model.val_deviance_.append(dev)
            if dev < best_val - model.tol:
                best_val = dev
                best_beta = beta.copy()
                best_iter = it
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    stop_reason = f"early_stop (patience={patience})"
                    model.n_iter_ = it
                    break

        # IRLS converges very fast; stop when parameter step is tiny.
        denom_beta = max(float(np.linalg.norm(beta)), 1.0)
        if step_norm / denom_beta < model.tol:
            stop_reason = f"converged (||dbeta||/||beta||<{model.tol:g})"
            model.n_iter_ = it
            break

        if verbose:
            msg = (f"[{tag}] iter {it:3d}  "
                   f"train_loss={cur_loss:.6f}  "
                   f"||dbeta||={step_norm:.3e}  "
                   f"||beta||={float(np.linalg.norm(beta)):.3e}")
            if use_val:
                msg += f"  val_dev={model.val_deviance_[-1]:.2f}"
            print(msg)
    else:
        model.n_iter_ = model.max_iter

    model.beta_ = best_beta if use_val else beta

    if verbose:
        final_loss = model._loss(X, y, model.beta_)
        summary = (f"[{tag}] done  stop={stop_reason}  "
                   f"n_iter={model.n_iter_}  "
                   f"train_loss={final_loss:.6f}  "
                   f"||beta||={float(np.linalg.norm(model.beta_)):.3e}")
        if use_val:
            summary += f"  best_val_dev={best_val:.2f} @ iter {best_iter}"
        print(summary)
    return model


class PoissonRegression:
    """
    L2-regularized Poisson regression with log link.

    Two solvers are available:
      - 'gd':   batch gradient descent on per-sample mean loss (needs many iters).
      - 'irls': Iteratively Reweighted Least Squares / Newton; converges in
                ~5-20 iters but each iter costs O(n*d^2 + d^3).

        lambda_i = exp(x_i^T beta)
        L(beta)  = sum_i [ exp(x_i^T beta) - y_i * x_i^T beta ] + lambda_reg * ||beta||^2
    """

    def __init__(self, lr: float = 0.01, lambda_reg: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-6,
                 solver: str = "gd"):
        """
        Parameters
        ----------
        lr : float
            Learning rate (used only when solver='gd').
        lambda_reg : float
            L2 regularization strength.
        max_iter : int
            Maximum number of iterations (GD needs many more than IRLS).
        tol : float
            Convergence tolerance. For GD: relative change of training loss.
            For IRLS: relative L2 change of beta.
        solver : {'gd', 'irls'}
            Which optimizer to use.
        """
        if solver not in ("gd", "irls"):
            raise ValueError("solver must be 'gd' or 'irls'")
        self.lr = float(lr)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.solver = solver

        self.beta_: np.ndarray | None = None
        self.train_loss_: list[float] = []
        self.val_deviance_: list[float] = []
        self.n_iter_: int = 0

    # -- core math -----------------------------------------------------------

    def _neg_log_likelihood(self, X: np.ndarray, y: np.ndarray,
                            beta: np.ndarray) -> float:
        """Non-regularized Poisson NLL (up to the y! constant), summed."""
        eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def _loss(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """Per-sample mean L2-regularized objective (used by the optimizer)."""
        n = X.shape[0]
        return (self._neg_log_likelihood(X, y, beta) / n
                + self.lambda_reg * float(np.dot(beta, beta)))

    def _grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Gradient of the per-sample mean objective."""
        n = X.shape[0]
        mu = _safe_exp(X @ beta)
        return (X.T @ (mu - y)) / n + 2.0 * self.lambda_reg * beta

    # -- public API ----------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            patience: int = 10, verbose: bool = False) -> "PoissonRegression":
        """
        Fit using the configured solver ('gd' or 'irls').

        Parameters
        ----------
        X_train, y_train : ndarray
            Training design matrix and non-negative integer targets.
        X_val, y_val : ndarray, optional
            Validation set; enables early stopping on validation Poisson
            deviance.
        patience : int
            Early-stopping patience.
        verbose : bool
            If True, print per-iteration progress.
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)

        if self.solver == "irls":
            Xv = np.asarray(X_val, dtype=np.float64) if X_val is not None else None
            yv = np.asarray(y_val, dtype=np.float64) if y_val is not None else None
            return _fit_irls_glm(self, X, y, Xv, yv,
                                 family="poisson",
                                 patience=patience, verbose=verbose)

        n, d = X.shape

        beta = np.zeros(d, dtype=np.float64)
        prev_loss = self._loss(X, y, beta)
        self.train_loss_ = [prev_loss]
        self.val_deviance_ = []

        best_val = math.inf
        best_beta = beta.copy()
        best_iter = 0
        stale = 0

        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv = np.asarray(X_val, dtype=np.float64)
            yv = np.asarray(y_val, dtype=np.float64)

        if verbose:
            print(f"[Poisson] fit  n={n}  d={d}  lr={self.lr}  "
                  f"lambda_reg={self.lambda_reg}  max_iter={self.max_iter}")
            print(f"[Poisson] iter     0  train_loss={prev_loss:.6f}")

        stop_reason = "max_iter"
        for it in range(1, self.max_iter + 1):
            grad = self._grad(X, y, beta)
            g_norm = float(np.linalg.norm(grad))
            beta -= self.lr * grad

            cur_loss = self._loss(X, y, beta)
            self.train_loss_.append(cur_loss)

            if use_val:
                dev = poisson_deviance(yv, _safe_exp(Xv @ beta))
                self.val_deviance_.append(dev)
                if dev < best_val - self.tol:
                    best_val = dev
                    best_beta = beta.copy()
                    best_iter = it
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stop_reason = f"early_stop (patience={patience})"
                        self.n_iter_ = it
                        break

            denom = max(abs(prev_loss), 1.0)
            rel_change = abs(prev_loss - cur_loss) / denom
            if rel_change < self.tol:
                stop_reason = f"converged (rel_change<{self.tol:g})"
                self.n_iter_ = it
                break
            prev_loss = cur_loss

            if verbose and (it <= 5 or it % 20 == 0):
                msg = (f"[Poisson] iter {it:5d}  "
                       f"train_loss={cur_loss:.6f}  "
                       f"||grad||={g_norm:.3e}  "
                       f"||beta||={float(np.linalg.norm(beta)):.3e}")
                if use_val:
                    msg += f"  val_dev={self.val_deviance_[-1]:.2f}"
                print(msg)
        else:
            self.n_iter_ = self.max_iter

        # Restore best-on-val weights if we tracked them
        self.beta_ = best_beta if use_val else beta

        if verbose:
            final_loss = self._loss(X, y, self.beta_)
            summary = (f"[Poisson] done  stop={stop_reason}  n_iter={self.n_iter_}  "
                       f"train_loss={final_loss:.6f}  "
                       f"||beta||={float(np.linalg.norm(self.beta_)):.3e}")
            if use_val:
                summary += (f"  best_val_dev={best_val:.2f} @ iter {best_iter}")
            print(summary)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted mean lambda_hat = exp(X @ beta)."""
        if self.beta_ is None:
            raise RuntimeError("Model has not been fit yet.")
        return _safe_exp(np.asarray(X, dtype=np.float64) @ self.beta_)

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Full Poisson log-likelihood (including the -log(y!) constant),
        used for likelihood ratio tests.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        eta = np.clip(X @ self.beta_, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        # log y! via lgamma(y+1)
        from math import lgamma
        log_y_fact = np.array([lgamma(int(v) + 1) for v in y])
        return float(np.sum(y * eta - mu - log_y_fact))

    # -- persistence ---------------------------------------------------------

    def save(self, path, feature_names: list[str] | None = None,
             extra: dict | None = None) -> str:
        """
        Serialize model to a single `.npz` file.

        Stores: model_type, beta, hyperparameters, training history,
        optional feature_names and any extra metadata dict.
        """
        return _save_model_npz(self, "poisson", path,
                               feature_names=feature_names, extra=extra)

    @classmethod
    def load(cls, path) -> "PoissonRegression":
        """Load a PoissonRegression previously saved with `.save()`."""
        return _load_model_npz(path, expected_type="poisson")


# ---------------------------------------------------------------------------
# Negative Binomial Regression (NB2)
# ---------------------------------------------------------------------------

class NegBinomialRegression:
    """
    NB2 regression with fixed dispersion alpha, log link, trained by batch GD
    on beta only.

        mu_i   = exp(x_i^T beta)
        Var(y) = mu + alpha * mu^2

    NLL (L2-reg on beta):
        L(beta; alpha) = -sum_i [ lgamma(y + 1/alpha) - lgamma(1/alpha)
                                  - lgamma(y + 1)
                                  + y * log(alpha*mu / (1 + alpha*mu))
                                  - (1/alpha) * log(1 + alpha*mu) ]
                        + lambda_reg * ||beta||^2

    Gradient wrt beta (alpha fixed):
        dL/dbeta = -X^T [ y - mu*(y + 1/alpha) / (mu + 1/alpha) ]
                   + 2 * lambda_reg * beta
    """

    def __init__(self, lr: float = 0.01, lambda_reg: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-6, alpha: float = 1.0,
                 solver: str = "gd"):
        """
        Parameters
        ----------
        alpha : float
            Fixed dispersion parameter (>= 0). alpha -> 0 recovers Poisson.
        solver : {'gd', 'irls'}
            Optimizer: batch GD (many cheap iters) or IRLS/Newton
            (few expensive iters, usually much faster overall).
        """
        if solver not in ("gd", "irls"):
            raise ValueError("solver must be 'gd' or 'irls'")
        self.lr = float(lr)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha = float(alpha)
        self.solver = solver

        self.beta_: np.ndarray | None = None
        self.train_loss_: list[float] = []
        self.val_deviance_: list[float] = []
        self.n_iter_: int = 0

    # -- core math -----------------------------------------------------------

    def neg_log_likelihood(self, y: np.ndarray, mu: np.ndarray) -> float:
        """
        NB2 negative log-likelihood for given y and fitted means mu.
        Uses scipy-free lgamma via math.lgamma (vectorized via np.vectorize
        on the gamma terms that involve integer-ish values).
        """
        y = np.asarray(y, dtype=np.float64)
        mu = np.clip(np.asarray(mu, dtype=np.float64), _EPS, None)
        a = self.alpha

        if a <= _EPS:
            # Poisson limit
            return float(np.sum(mu - y * np.log(mu))) + \
                float(np.sum(_lgamma_vec(y + 1.0)))

        inv_a = 1.0 / a
        # lgamma terms
        lg_y_inv = _lgamma_vec(y + inv_a)
        lg_inv = math.lgamma(inv_a)
        lg_yp1 = _lgamma_vec(y + 1.0)

        am = a * mu
        # Use log1p for numerical stability: log(1 + a*mu)
        log1p_am = np.log1p(am)
        # log(a*mu / (1 + a*mu)) = log(a*mu) - log1p(a*mu)
        log_ratio = np.log(am) - log1p_am

        ll = (lg_y_inv - lg_inv - lg_yp1
              + y * log_ratio - inv_a * log1p_am)
        return float(-np.sum(ll))

    def _loss(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """Per-sample mean L2-regularized NB2 objective (optimizer-facing)."""
        n = X.shape[0]
        mu = _safe_exp(X @ beta)
        return (self.neg_log_likelihood(y, mu) / n
                + self.lambda_reg * float(np.dot(beta, beta)))

    def _grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Gradient wrt beta of the per-sample mean objective."""
        n = X.shape[0]
        mu = _safe_exp(X @ beta)
        inv_a = 1.0 / max(self.alpha, _EPS)
        # residual = y - mu*(y + 1/a) / (mu + 1/a)
        resid = y - mu * (y + inv_a) / (mu + inv_a)
        return -(X.T @ resid) / n + 2.0 * self.lambda_reg * beta

    # -- public API ----------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            patience: int = 10, verbose: bool = False) -> "NegBinomialRegression":
        """
        Fit using the configured solver ('gd' or 'irls'), with optional
        early stopping on validation Poisson deviance.
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)

        if self.solver == "irls":
            Xv = np.asarray(X_val, dtype=np.float64) if X_val is not None else None
            yv = np.asarray(y_val, dtype=np.float64) if y_val is not None else None
            return _fit_irls_glm(self, X, y, Xv, yv,
                                 family="nb", alpha=self.alpha,
                                 patience=patience, verbose=verbose)

        n, d = X.shape

        beta = np.zeros(d, dtype=np.float64)
        prev_loss = self._loss(X, y, beta)
        self.train_loss_ = [prev_loss]
        self.val_deviance_ = []

        best_val = math.inf
        best_beta = beta.copy()
        best_iter = 0
        stale = 0

        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv = np.asarray(X_val, dtype=np.float64)
            yv = np.asarray(y_val, dtype=np.float64)

        if verbose:
            print(f"[NB2] fit  n={n}  d={d}  lr={self.lr}  "
                  f"lambda_reg={self.lambda_reg}  alpha={self.alpha}  "
                  f"max_iter={self.max_iter}")
            print(f"[NB2] iter     0  train_loss={prev_loss:.6f}")

        stop_reason = "max_iter"
        for it in range(1, self.max_iter + 1):
            grad = self._grad(X, y, beta)
            g_norm = float(np.linalg.norm(grad))
            beta -= self.lr * grad

            cur_loss = self._loss(X, y, beta)
            self.train_loss_.append(cur_loss)

            if use_val:
                dev = poisson_deviance(yv, _safe_exp(Xv @ beta))
                self.val_deviance_.append(dev)
                if dev < best_val - self.tol:
                    best_val = dev
                    best_beta = beta.copy()
                    best_iter = it
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        stop_reason = f"early_stop (patience={patience})"
                        self.n_iter_ = it
                        break

            denom = max(abs(prev_loss), 1.0)
            rel_change = abs(prev_loss - cur_loss) / denom
            if rel_change < self.tol:
                stop_reason = f"converged (rel_change<{self.tol:g})"
                self.n_iter_ = it
                break
            prev_loss = cur_loss

            if verbose and (it <= 5 or it % 20 == 0):
                msg = (f"[NB2] iter {it:5d}  "
                       f"train_loss={cur_loss:.6f}  "
                       f"||grad||={g_norm:.3e}  "
                       f"||beta||={float(np.linalg.norm(beta)):.3e}")
                if use_val:
                    msg += f"  val_dev={self.val_deviance_[-1]:.2f}"
                print(msg)
        else:
            self.n_iter_ = self.max_iter

        self.beta_ = best_beta if use_val else beta

        if verbose:
            final_loss = self._loss(X, y, self.beta_)
            summary = (f"[NB2] done  stop={stop_reason}  n_iter={self.n_iter_}  "
                       f"train_loss={final_loss:.6f}  "
                       f"||beta||={float(np.linalg.norm(self.beta_)):.3e}")
            if use_val:
                summary += f"  best_val_dev={best_val:.2f} @ iter {best_iter}"
            print(summary)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted mean mu_hat = exp(X @ beta)."""
        if self.beta_ is None:
            raise RuntimeError("Model has not been fit yet.")
        return _safe_exp(np.asarray(X, dtype=np.float64) @ self.beta_)

    def log_likelihood(self, X: np.ndarray, y: np.ndarray) -> float:
        """Full NB2 log-likelihood on (X, y), for likelihood-ratio tests."""
        mu = self.predict(X)
        return -self.neg_log_likelihood(y, mu)

    # -- persistence ---------------------------------------------------------

    def save(self, path, feature_names: list[str] | None = None,
             extra: dict | None = None) -> str:
        """Serialize model to a single `.npz` file. See PoissonRegression.save."""
        return _save_model_npz(self, "nb", path,
                               feature_names=feature_names, extra=extra)

    @classmethod
    def load(cls, path) -> "NegBinomialRegression":
        """Load a NegBinomialRegression previously saved with `.save()`."""
        return _load_model_npz(path, expected_type="nb")


# ---------------------------------------------------------------------------
# Persistence (shared save/load for both model classes)
# ---------------------------------------------------------------------------

import json
import os


def _save_model_npz(model, model_type: str, path,
                    feature_names: list[str] | None = None,
                    extra: dict | None = None) -> str:
    """
    Save a fitted model to a single `.npz` file.

    Layout inside the archive:
      - 'beta'        : float64 (d,) coefficient vector
      - 'train_loss'  : float64 history (possibly empty)
      - 'val_dev'     : float64 history (possibly empty)
      - 'meta'        : 0-d array holding a JSON string with
                        model_type, hyperparameters, feature_names,
                        n_iter_, and caller-supplied `extra`.
    """
    if model.beta_ is None:
        raise RuntimeError("Cannot save a model that has not been fit.")

    path = str(path)
    if not path.endswith(".npz"):
        path = path + ".npz"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    hp: dict = {
        "lr": model.lr,
        "lambda_reg": model.lambda_reg,
        "max_iter": model.max_iter,
        "tol": model.tol,
        "solver": getattr(model, "solver", "gd"),
    }
    if model_type == "nb":
        hp["alpha"] = model.alpha

    meta = {
        "model_type": model_type,
        "hyperparameters": hp,
        "n_iter_": int(model.n_iter_),
        "n_features": int(model.beta_.shape[0]),
        "feature_names": list(feature_names) if feature_names else None,
        "extra": extra or {},
    }

    np.savez(
        path,
        beta=np.asarray(model.beta_, dtype=np.float64),
        train_loss=np.asarray(model.train_loss_, dtype=np.float64),
        val_dev=np.asarray(model.val_deviance_, dtype=np.float64),
        meta=np.array(json.dumps(meta)),
    )
    return path


def _load_model_npz(path, expected_type: str | None = None):
    """
    Reconstruct a PoissonRegression or NegBinomialRegression from an npz
    produced by `_save_model_npz`. If `expected_type` is given, verifies the
    archive matches that model_type.
    """
    path = str(path)
    with np.load(path, allow_pickle=False) as arch:
        beta = np.asarray(arch["beta"], dtype=np.float64)
        train_loss = arch["train_loss"].tolist() if "train_loss" in arch.files else []
        val_dev = arch["val_dev"].tolist() if "val_dev" in arch.files else []
        meta = json.loads(str(arch["meta"]))

    model_type = meta["model_type"]
    if expected_type is not None and model_type != expected_type:
        raise ValueError(
            f"Archive contains model_type={model_type!r}, "
            f"but expected {expected_type!r}."
        )

    hp = meta["hyperparameters"]
    solver = hp.get("solver", "gd")
    if model_type == "poisson":
        m = PoissonRegression(lr=hp["lr"], lambda_reg=hp["lambda_reg"],
                              max_iter=hp["max_iter"], tol=hp["tol"],
                              solver=solver)
    elif model_type == "nb":
        m = NegBinomialRegression(lr=hp["lr"], lambda_reg=hp["lambda_reg"],
                                  max_iter=hp["max_iter"], tol=hp["tol"],
                                  alpha=hp["alpha"], solver=solver)
    else:
        raise ValueError(f"Unknown model_type in archive: {model_type!r}")

    m.beta_ = beta
    m.train_loss_ = list(train_loss)
    m.val_deviance_ = list(val_dev)
    m.n_iter_ = int(meta.get("n_iter_", 0))
    m.feature_names_ = meta.get("feature_names")
    m.meta_ = meta
    return m


def load_model(path):
    """
    Load a saved Stage-2 model without needing to know its type in advance.

    Returns a fitted PoissonRegression or NegBinomialRegression.
    """
    return _load_model_npz(path, expected_type=None)


# ---------------------------------------------------------------------------
# Likelihood Ratio Test
# ---------------------------------------------------------------------------

def _lgamma_vec(arr: np.ndarray) -> np.ndarray:
    """Vectorized math.lgamma over a NumPy array."""
    flat = np.asarray(arr, dtype=np.float64).ravel()
    out = np.empty_like(flat)
    for i, v in enumerate(flat):
        out[i] = math.lgamma(v)
    return out.reshape(np.asarray(arr).shape)


def likelihood_ratio_test(ll_poisson: float, ll_nb: float,
                          df: int = 1) -> tuple[float, float]:
    """
    Likelihood ratio test: H0 = Poisson (restricted), H1 = NB (unrestricted).

        LR = 2 * (ll_nb - ll_poisson)    ~   chi2(df)   under H0

    Since alpha = 0 lies on the boundary of its parameter space, the strictly
    correct null distribution is 0.5*chi2(0) + 0.5*chi2(1); the classical
    chi2(1) p-value is then roughly 2x conservative. We return the classical
    version here (and also halve it as `p_boundary`-style adjustment via
    scipy if available; otherwise fall back to math.erfc-based upper tail).

    Parameters
    ----------
    ll_poisson, ll_nb : float
        Log-likelihoods of the two fitted models on the SAME data.
    df : int
        Degrees of freedom (default 1: one extra parameter, alpha).

    Returns
    -------
    (lr_stat, p_value) : tuple of floats
    """
    lr_stat = 2.0 * (ll_nb - ll_poisson)
    if lr_stat < 0:
        lr_stat = 0.0

    try:
        from scipy.stats import chi2
        p = float(chi2.sf(lr_stat, df))
    except Exception:
        # Fallback: chi2 with df=1 upper tail = erfc(sqrt(x/2))
        if df == 1:
            p = float(math.erfc(math.sqrt(lr_stat / 2.0)))
        else:
            # Crude series-free fallback: use the regularized upper
            # incomplete gamma via math.gamma approximation.
            # Q(k/2, x/2); implement via simple numerical integration.
            p = _chi2_sf_fallback(lr_stat, df)
    return float(lr_stat), p


def _chi2_sf_fallback(x: float, df: int) -> float:
    """Crude chi2 survival function for df != 1 without scipy."""
    # Simpson integration of the chi2 pdf from x out to a large cutoff.
    if x <= 0:
        return 1.0
    k = df
    ln_norm = -(k / 2.0) * math.log(2.0) - math.lgamma(k / 2.0)
    upper = max(x + 50.0, 10.0 * x)
    n = 2000
    h = (upper - x) / n
    total = 0.0
    for i in range(n + 1):
        t = x + i * h
        pdf = math.exp(ln_norm + (k / 2.0 - 1.0) * math.log(t) - t / 2.0)
        w = 1 if i == 0 or i == n else (4 if i % 2 else 2)
        total += w * pdf
    return (h / 3.0) * total


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "poisson": PoissonRegression,
    "nb": NegBinomialRegression,
    "negbinomial": NegBinomialRegression,
}


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate a fitted model on (X, y).

    Returns
    -------
    dict with keys: mae, rmse, poisson_deviance, log_likelihood.
    """
    y_pred = model.predict(X)
    out = {
        "mae": mae(y, y_pred),
        "rmse": rmse(y, y_pred),
        "poisson_deviance": poisson_deviance(y, y_pred),
    }
    if hasattr(model, "log_likelihood"):
        out["log_likelihood"] = model.log_likelihood(X, y)
    return out


def train_model(model_type: str,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val: np.ndarray | None = None,
                y_val: np.ndarray | None = None,
                *,
                lr: float = 1e-3,
                lambda_reg: float = 1.0,
                max_iter: int = 1000,
                tol: float = 1e-6,
                alpha: float = 1.0,
                solver: str = "gd",
                patience: int = 10,
                verbose: bool = False):
    """
    Unified training entry point for Stage 2.

    Parameters
    ----------
    model_type : {'poisson', 'nb'}
        Which count model to fit.
    X_train, y_train : ndarray
        Training features and non-negative integer targets.
    X_val, y_val : ndarray, optional
        Validation set; enables early stopping on validation Poisson deviance.
    lr, lambda_reg, max_iter, tol : float/int
        Optimization hyperparameters (see PoissonRegression / NegBinomialRegression).
    alpha : float
        Dispersion parameter (only used when model_type='nb').
    patience : int
        Early-stopping patience (only used when a validation set is provided).
    verbose : bool
        Print progress during fitting.

    Returns
    -------
    Fitted model instance.
    """
    key = model_type.lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type={model_type!r}. "
            f"Expected one of {sorted(_MODEL_REGISTRY)}."
        )

    cls = _MODEL_REGISTRY[key]
    if cls is NegBinomialRegression:
        model = cls(lr=lr, lambda_reg=lambda_reg,
                    max_iter=max_iter, tol=tol, alpha=alpha, solver=solver)
    else:
        model = cls(lr=lr, lambda_reg=lambda_reg,
                    max_iter=max_iter, tol=tol, solver=solver)

    model.fit(X_train, y_train, X_val, y_val,
              patience=patience, verbose=verbose)
    return model


def tune_hyperparameters(model_type: str,
                         X_train: np.ndarray, y_train: np.ndarray,
                         X_val: np.ndarray, y_val: np.ndarray,
                         param_grid: dict,
                         *,
                         metric: str = "poisson_deviance",
                         max_iter: int = 1000,
                         tol: float = 1e-6,
                         patience: int = 10,
                         solver: str = "gd",
                         verbose: bool = False) -> dict:
    """
    Grid-search hyperparameter tuning against a validation metric.

    Parameters
    ----------
    model_type : {'poisson', 'nb'}
    X_train, y_train, X_val, y_val : ndarray
        Train/val splits. The validation set is used both for early stopping
        and for scoring each configuration.
    param_grid : dict[str, list]
        Mapping from hyperparameter name to list of candidate values.
        Recognized keys: 'lr', 'lambda_reg', 'alpha' (NB only).
        Unspecified keys fall back to the defaults in train_model.
    metric : {'mae', 'rmse', 'poisson_deviance', 'neg_log_likelihood'}
        Lower-is-better metric to minimize on the validation set.
    max_iter, tol, patience, verbose : passed through to train_model.

    Returns
    -------
    dict with keys:
        'best_params'   : winning hyperparameter combination
        'best_score'    : corresponding validation metric
        'best_model'    : fitted model trained with best_params
        'results'       : list of (params, metrics_dict) for every combo tried

    Example
    -------
    >>> grid = {'lr': [1e-3, 1e-4], 'lambda_reg': [0.1, 1.0, 10.0]}
    >>> res = tune_hyperparameters('poisson', Xtr, ytr, Xva, yva, grid)
    >>> res['best_params']
    """
    allowed = {"lr", "lambda_reg", "alpha"}
    unknown = set(param_grid) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in param_grid: {sorted(unknown)}. "
                         f"Allowed: {sorted(allowed)}.")
    if model_type.lower() == "poisson" and "alpha" in param_grid:
        raise ValueError("'alpha' is not a hyperparameter of Poisson regression.")

    allowed_metrics = {"mae", "rmse", "poisson_deviance", "neg_log_likelihood"}
    if metric not in allowed_metrics:
        raise ValueError(f"metric must be one of {sorted(allowed_metrics)}.")

    import itertools
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    results = []
    best_score = math.inf
    best_params = None
    best_model = None

    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        model = train_model(
            model_type, X_train, y_train, X_val, y_val,
            max_iter=max_iter, tol=tol, patience=patience,
            solver=solver,
            verbose=False,
            **params,
        )
        scores = evaluate(model, X_val, y_val)
        if metric == "neg_log_likelihood":
            score = -scores.get("log_likelihood", math.nan)
        else:
            score = scores[metric]
        results.append({"params": params, "val_metrics": scores})

        if verbose:
            print(f"  params={params}  val_{metric}={score:.4f}")

        if score < best_score:
            best_score = score
            best_params = params
            best_model = model

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_model": best_model,
        "results": results,
    }
