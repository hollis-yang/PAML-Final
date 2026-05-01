"""
Stage 2 driver: load engineered panel, train Poisson and NB models,
tune hyperparameters, and report test-set metrics plus a Poisson-vs-NB
likelihood ratio test.

Run from the project root:

    uv run python -m stage2.main
    uv run python -m stage2.main --csv data/data_engineering.csv --tune
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Support both `python -m stage2.main` (package form) and
# `python stage2/main.py` (direct form) by falling back to a local import.
try:
    from stage2.model import (
        evaluate,
        likelihood_ratio_test,
        load_stage2_data,
        mae,
        poisson_deviance,
        rmse,
        temporal_split,
        train_model,
        tune_hyperparameters,
    )
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from model import (
        evaluate,
        likelihood_ratio_test,
        load_stage2_data,
        mae,
        poisson_deviance,
        rmse,
        temporal_split,
        train_model,
        tune_hyperparameters,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2 training driver.")
    p.add_argument("--csv", default="data/data_engineering.csv",
                   help="Path to engineered panel CSV.")

    # Shared optimizer knobs. Defaults reflect the best configuration found
    # by grid search on the engineered panel: IRLS solver converges in ~9
    # iters, so max_iter=30 with patience=10 is ample. Override via flags
    # to reproduce the old GD path.
    p.add_argument("--max-iter", type=int, default=30,
                   help="Max iterations (IRLS converges in ~10; GD needs many).")
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience on val deviance.")

    # Poisson-specific. Best val PoissonDev at lambda_reg=1e-3 under IRLS.
    p.add_argument("--poisson-solver", choices=("gd", "irls"), default="irls",
                   help="Optimizer for Poisson.")
    p.add_argument("--poisson-lr", type=float, default=0.02,
                   help="Poisson GD learning rate (unused with IRLS).")
    p.add_argument("--poisson-lambda", type=float, default=1e-3,
                   help="Poisson L2 regularization strength.")

    # NB-specific. Best val NLL at alpha=0.2, lambda_reg=1e-3 under IRLS
    # (LR stat=825 vs Poisson on test set, p~1e-181).
    p.add_argument("--nb-solver", choices=("gd", "irls"), default="irls",
                   help="Optimizer for NB.")
    p.add_argument("--nb-lr", type=float, default=0.5,
                   help="NB GD learning rate (unused with IRLS).")
    p.add_argument("--nb-lambda", type=float, default=1e-3,
                   help="NB L2 regularization strength.")
    p.add_argument("--nb-alpha", type=float, default=0.2,
                   help="NB dispersion parameter (fixed).")

    p.add_argument("--tune", action="store_true",
                   help="Run per-model grid searches (separate grids for "
                        "Poisson and NB).")
    p.add_argument("--save-dir", default=None,
                   help="If set, save fitted models to this directory "
                        "(poisson.npz, nb.npz).")
    p.add_argument("--quiet", action="store_true", help="Suppress fit logs.")
    return p.parse_args()


# Per-model tuning grids. `lr` only matters under solver='gd' (IRLS ignores it
# and we strip it below to avoid redundant combos).
POISSON_GRID = {
    "lr": [0.005, 0.01, 0.02, 0.05],
    "lambda_reg": [1e-5, 1e-4, 1e-3],
}
# NB alpha grid. Prior tune found alpha=0.2 near the lower end of [0.1, 1.0],
# so the grid is extended downward to [0.05, 0.08] to check if smaller values
# improve val NLL. Lambda is held to two values since a prior sweep showed it
# moves val NLL by <0.1 (n=546k >> d=200, regularization barely bites).
NB_GRID = {
    "lr": [0.1, 0.3, 0.5, 1.0],
    "lambda_reg": [1e-4, 1e-3],
    "alpha": [0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5],
}


def _grid_for_solver(grid: dict, solver: str) -> dict:
    """IRLS ignores `lr`, so drop it to avoid fitting duplicate models."""
    if solver == "irls":
        return {k: v for k, v in grid.items() if k != "lr"}
    return dict(grid)


def _print_metrics(name: str, metrics: dict) -> None:
    print(f"[{name}] "
          f"MAE={metrics['mae']:.4f}  "
          f"RMSE={metrics['rmse']:.4f}  "
          f"PoissonDev={metrics['poisson_deviance']:.2f}  "
          f"logLik={metrics.get('log_likelihood', float('nan')):.2f}")


def _baseline_metrics(y_tr: np.ndarray, y_te: np.ndarray) -> dict:
    """Predict-the-mean baseline on the test set, for reference."""
    y_pred = np.full_like(y_te, fill_value=float(np.mean(y_tr)),
                          dtype=np.float64)
    return {
        "mae": mae(y_te, y_pred),
        "rmse": rmse(y_te, y_pred),
        "poisson_deviance": poisson_deviance(y_te, y_pred),
    }


def _describe_y(name: str, y: np.ndarray) -> None:
    quant = np.quantile(y, [0.5, 0.9, 0.99, 1.0])
    zero_frac = float(np.mean(y == 0))
    print(f"  {name}: n={len(y)}  mean={y.mean():.3f}  var={y.var():.3f}  "
          f"zero_frac={zero_frac:.3f}  "
          f"p50={quant[0]:.1f}  p90={quant[1]:.1f}  "
          f"p99={quant[2]:.1f}  max={quant[3]:.0f}")


def main() -> None:
    args = parse_args()
    verbose = not args.quiet
    t_start = time.time()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path.resolve()}")

    print("=" * 72)
    print("Stage 2: crash count prediction (Poisson / Negative Binomial)")
    print("=" * 72)

    print(f"\n[data] loading {csv_path}")
    t0 = time.time()
    X, y, feats = load_stage2_data(str(csv_path))
    print(f"[data] loaded in {time.time() - t0:.1f}s  "
          f"X={X.shape}  y={y.shape}  n_features={len(feats)}")
    _describe_y("y (all)", y)
    # Quick overdispersion diagnostic
    ratio = float(y.var() / max(y.mean(), 1e-12))
    print(f"  var(y)/mean(y) = {ratio:.3f}  "
          f"({'~Poisson (ratio ~1)' if abs(ratio-1) < 0.3 else 'overdispersed'})")

    X_tr, y_tr, X_va, y_va, X_te, y_te = temporal_split(X, y)
    print(f"[split] train={X_tr.shape[0]}  val={X_va.shape[0]}  "
          f"test={X_te.shape[0]} (ratios 70/15/15, positional)")
    _describe_y("y_train", y_tr)
    _describe_y("y_val  ", y_va)
    _describe_y("y_test ", y_te)

    baseline = _baseline_metrics(y_tr, y_te)
    print(f"[baseline] predict-the-mean  "
          f"MAE={baseline['mae']:.4f}  "
          f"RMSE={baseline['rmse']:.4f}  "
          f"PoissonDev={baseline['poisson_deviance']:.2f}")

    # ---------------- Poisson ----------------
    print("\n" + "-" * 72)
    print("Poisson Regression")
    print("-" * 72)
    t0 = time.time()
    if args.tune:
        p_grid = _grid_for_solver(POISSON_GRID, args.poisson_solver)
        print(f"[Poisson/tune] solver={args.poisson_solver}  grid: {p_grid}")
        res_p = tune_hyperparameters(
            "poisson", X_tr, y_tr, X_va, y_va,
            param_grid=p_grid,
            metric="poisson_deviance",
            solver=args.poisson_solver,
            max_iter=args.max_iter, patience=args.patience, verbose=verbose,
        )
        print(f"[Poisson/tune] best params: {res_p['best_params']}  "
              f"val PoissonDev={res_p['best_score']:.2f}  "
              f"({len(res_p['results'])} combos)")
        poisson = res_p["best_model"]
    else:
        poisson = train_model(
            "poisson", X_tr, y_tr, X_va, y_va,
            lr=args.poisson_lr, lambda_reg=args.poisson_lambda,
            solver=args.poisson_solver,
            max_iter=args.max_iter, patience=args.patience, verbose=verbose,
        )
    print(f"[Poisson] fit time: {time.time() - t0:.1f}s  "
          f"n_iter={poisson.n_iter_}")
    _print_metrics("Poisson/val ", evaluate(poisson, X_va, y_va))
    _print_metrics("Poisson/test", evaluate(poisson, X_te, y_te))

    # ---------------- Negative Binomial ------
    print("\n" + "-" * 72)
    print(f"Negative Binomial Regression (alpha={args.nb_alpha})")
    print("-" * 72)
    t0 = time.time()
    if args.tune:
        nb_grid = _grid_for_solver(NB_GRID, args.nb_solver)
        # Use NLL for NB tuning: poisson_deviance only scores mean fit and
        # biases selection toward alpha->0 (Poisson limit), hiding NB's gain.
        print(f"[NB/tune] solver={args.nb_solver}  metric=neg_log_likelihood"
              f"  grid: {nb_grid}")
        res_nb = tune_hyperparameters(
            "nb", X_tr, y_tr, X_va, y_va,
            param_grid=nb_grid,
            metric="neg_log_likelihood",
            solver=args.nb_solver,
            max_iter=args.max_iter, patience=args.patience, verbose=verbose,
        )
        print(f"[NB/tune] best params: {res_nb['best_params']}  "
              f"val NLL={res_nb['best_score']:.2f}  "
              f"({len(res_nb['results'])} combos)")
        nb = res_nb["best_model"]
    else:
        nb = train_model(
            "nb", X_tr, y_tr, X_va, y_va,
            lr=args.nb_lr, lambda_reg=args.nb_lambda, alpha=args.nb_alpha,
            solver=args.nb_solver,
            max_iter=args.max_iter, patience=args.patience, verbose=verbose,
        )
    print(f"[NB] fit time: {time.time() - t0:.1f}s  n_iter={nb.n_iter_}")
    _print_metrics("NB/val ", evaluate(nb, X_va, y_va))
    _print_metrics("NB/test", evaluate(nb, X_te, y_te))

    # ---------------- Summary table ----------
    print("\n" + "-" * 72)
    print("Test-set comparison")
    print("-" * 72)
    mp = evaluate(poisson, X_te, y_te)
    mn = evaluate(nb, X_te, y_te)
    header = f"{'metric':<18}{'baseline':>14}{'Poisson':>14}{'NB':>14}"
    print(header)
    for key, label in [("mae", "MAE"), ("rmse", "RMSE"),
                       ("poisson_deviance", "PoissonDev")]:
        print(f"{label:<18}{baseline[key]:>14.4f}"
              f"{mp[key]:>14.4f}{mn[key]:>14.4f}")

    # ---------------- Likelihood Ratio Test --
    ll_p = poisson.log_likelihood(X_te, y_te)
    ll_nb = nb.log_likelihood(X_te, y_te)
    lr_stat, p_value = likelihood_ratio_test(ll_p, ll_nb, df=1)
    print("\n" + "-" * 72)
    print("Likelihood Ratio Test (Poisson vs NB, test set)")
    print("-" * 72)
    print(f"  ll_Poisson = {ll_p:.2f}")
    print(f"  ll_NB      = {ll_nb:.2f}")
    print(f"  LR stat    = {lr_stat:.4f}  (df=1)")
    print(f"  p-value    = {p_value:.4g}")
    if p_value < 0.05:
        print("  -> reject H0: NB fits significantly better than Poisson.")
    else:
        print("  -> fail to reject H0: Poisson is adequate.")

    # ---------------- Save models ------------
    if args.save_dir:
        print("\n" + "-" * 72)
        print(f"Saving models to {args.save_dir}")
        print("-" * 72)
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        poisson_path = poisson.save(
            save_dir / "poisson.npz",
            feature_names=feats,
            extra={
                "test_metrics": mp,
                "val_metrics": evaluate(poisson, X_va, y_va),
                "train_size": int(X_tr.shape[0]),
            },
        )
        nb_path = nb.save(
            save_dir / "nb.npz",
            feature_names=feats,
            extra={
                "test_metrics": mn,
                "val_metrics": evaluate(nb, X_va, y_va),
                "train_size": int(X_tr.shape[0]),
            },
        )
        print(f"[save] Poisson -> {poisson_path}")
        print(f"[save] NB      -> {nb_path}")

    print(f"\n[done] total wall time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
