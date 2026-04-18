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

class PoissonRegression:
    """
    L2-regularized Poisson regression with log link, trained by batch GD.

        lambda_i = exp(x_i^T beta)
        L(beta)  = sum_i [ exp(x_i^T beta) - y_i * x_i^T beta ] + lambda_reg * ||beta||^2
        grad     = X^T (mu - y) + 2 * lambda_reg * beta
    """

    def __init__(self, lr: float = 0.01, lambda_reg: float = 1.0,
                 max_iter: int = 1000, tol: float = 1e-6):
        """
        Parameters
        ----------
        lr : float
            Learning rate for batch gradient descent.
        lambda_reg : float
            L2 regularization strength.
        max_iter : int
            Maximum number of GD iterations.
        tol : float
            Convergence tolerance on relative change of training loss.
        """
        self.lr = float(lr)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.beta_: np.ndarray | None = None
        self.train_loss_: list[float] = []
        self.val_deviance_: list[float] = []
        self.n_iter_: int = 0

    # -- core math -----------------------------------------------------------

    def _neg_log_likelihood(self, X: np.ndarray, y: np.ndarray,
                            beta: np.ndarray) -> float:
        """Non-regularized Poisson NLL (up to the y! constant)."""
        eta = np.clip(X @ beta, -_ETA_CLIP, _ETA_CLIP)
        mu = np.exp(eta)
        return float(np.sum(mu - y * eta))

    def _loss(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> float:
        """Full L2-regularized objective."""
        return self._neg_log_likelihood(X, y, beta) + \
            self.lambda_reg * float(np.dot(beta, beta))

    def _grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Gradient of the L2-regularized objective."""
        mu = _safe_exp(X @ beta)
        return X.T @ (mu - y) + 2.0 * self.lambda_reg * beta

    # -- public API ----------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            patience: int = 10, verbose: bool = False) -> "PoissonRegression":
        """
        Fit via batch gradient descent with optional early stopping on
        validation Poisson deviance.

        Parameters
        ----------
        X_train, y_train : ndarray
            Training design matrix and non-negative integer targets.
        X_val, y_val : ndarray, optional
            Validation set; if given, training stops when the val deviance
            has not improved for `patience` consecutive iterations.
        patience : int
            Early-stopping patience (only used when val set is provided).
        verbose : bool
            If True, print progress every 50 iterations.
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        d = X.shape[1]

        beta = np.zeros(d, dtype=np.float64)
        prev_loss = self._loss(X, y, beta)
        self.train_loss_ = [prev_loss]
        self.val_deviance_ = []

        best_val = math.inf
        best_beta = beta.copy()
        stale = 0

        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv = np.asarray(X_val, dtype=np.float64)
            yv = np.asarray(y_val, dtype=np.float64)

        for it in range(1, self.max_iter + 1):
            grad = self._grad(X, y, beta)
            beta -= self.lr * grad

            cur_loss = self._loss(X, y, beta)
            self.train_loss_.append(cur_loss)

            if use_val:
                dev = poisson_deviance(yv, _safe_exp(Xv @ beta))
                self.val_deviance_.append(dev)
                if dev < best_val - self.tol:
                    best_val = dev
                    best_beta = beta.copy()
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        if verbose:
                            print(f"[Poisson] early stop at iter {it}, "
                                  f"best val dev = {best_val:.4f}")
                        beta = best_beta
                        self.n_iter_ = it
                        self.beta_ = beta
                        return self

            # Relative-change convergence check on training loss
            denom = max(abs(prev_loss), 1.0)
            if abs(prev_loss - cur_loss) / denom < self.tol:
                if verbose:
                    print(f"[Poisson] converged at iter {it}")
                self.n_iter_ = it
                break
            prev_loss = cur_loss

            if verbose and it % 50 == 0:
                msg = f"[Poisson] iter {it:5d}  train_loss={cur_loss:.4f}"
                if use_val:
                    msg += f"  val_dev={self.val_deviance_[-1]:.4f}"
                print(msg)
        else:
            self.n_iter_ = self.max_iter

        # Use best-on-validation weights if we tracked them
        self.beta_ = best_beta if use_val else beta
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
                 max_iter: int = 1000, tol: float = 1e-6, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float
            Fixed dispersion parameter (>= 0). alpha -> 0 recovers Poisson.
        """
        self.lr = float(lr)
        self.lambda_reg = float(lambda_reg)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.alpha = float(alpha)

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
        """L2-regularized NB2 objective."""
        mu = _safe_exp(X @ beta)
        return self.neg_log_likelihood(y, mu) + \
            self.lambda_reg * float(np.dot(beta, beta))

    def _grad(self, X: np.ndarray, y: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Gradient wrt beta (alpha fixed)."""
        mu = _safe_exp(X @ beta)
        inv_a = 1.0 / max(self.alpha, _EPS)
        # residual = y - mu*(y + 1/a) / (mu + 1/a)
        resid = y - mu * (y + inv_a) / (mu + inv_a)
        return -(X.T @ resid) + 2.0 * self.lambda_reg * beta

    # -- public API ----------------------------------------------------------

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray | None = None,
            y_val: np.ndarray | None = None,
            patience: int = 10, verbose: bool = False) -> "NegBinomialRegression":
        """
        Fit via batch GD with optional early stopping on validation
        Poisson deviance (a robust, scale-free proxy for count fit quality).
        """
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.float64)
        d = X.shape[1]

        beta = np.zeros(d, dtype=np.float64)
        prev_loss = self._loss(X, y, beta)
        self.train_loss_ = [prev_loss]
        self.val_deviance_ = []

        best_val = math.inf
        best_beta = beta.copy()
        stale = 0

        use_val = X_val is not None and y_val is not None
        if use_val:
            Xv = np.asarray(X_val, dtype=np.float64)
            yv = np.asarray(y_val, dtype=np.float64)

        for it in range(1, self.max_iter + 1):
            grad = self._grad(X, y, beta)
            beta -= self.lr * grad

            cur_loss = self._loss(X, y, beta)
            self.train_loss_.append(cur_loss)

            if use_val:
                dev = poisson_deviance(yv, _safe_exp(Xv @ beta))
                self.val_deviance_.append(dev)
                if dev < best_val - self.tol:
                    best_val = dev
                    best_beta = beta.copy()
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        if verbose:
                            print(f"[NB2] early stop at iter {it}, "
                                  f"best val dev = {best_val:.4f}")
                        beta = best_beta
                        self.n_iter_ = it
                        self.beta_ = beta
                        return self

            denom = max(abs(prev_loss), 1.0)
            if abs(prev_loss - cur_loss) / denom < self.tol:
                if verbose:
                    print(f"[NB2] converged at iter {it}")
                self.n_iter_ = it
                break
            prev_loss = cur_loss

            if verbose and it % 50 == 0:
                msg = f"[NB2] iter {it:5d}  train_loss={cur_loss:.4f}"
                if use_val:
                    msg += f"  val_dev={self.val_deviance_[-1]:.4f}"
                print(msg)
        else:
            self.n_iter_ = self.max_iter

        self.beta_ = best_beta if use_val else beta
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
                    max_iter=max_iter, tol=tol, alpha=alpha)
    else:
        model = cls(lr=lr, lambda_reg=lambda_reg,
                    max_iter=max_iter, tol=tol)

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
