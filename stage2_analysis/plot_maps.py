"""
Stage 2 spatial diagnostic maps.

Joins the Stage 2 engineered panel (and final model predictions) to the
NYC MODZCTA shapefile and renders a set of choropleths that summarize
model behavior at the ZIP level:

  1. Observed  mean crash count per ZIP per 12-hour window.
  2. Predicted mean crash count per ZIP (NB2, averaged over all rows).
  3. Residual  (observed - predicted) per ZIP -- where the model misses.
  4. Predicted peak vs off-peak difference per ZIP -- the temporal signal.
  5. NB conditional standard deviation per ZIP -- sqrt(mu + alpha * mu^2).

A combined 2x2 summary figure is also written.

Outputs go to stage2_analysis/figures/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from stage2.model import load_model, load_stage2_data  # noqa: E402
from stage2.predict import _featurize_to_schema  # noqa: E402


DATA_CSV = REPO_ROOT / "data" / "data_engineering.csv"
SHP_PATH = REPO_ROOT / "data" / "geo_data" / "nyc_zip" / "nyc_zip.shp"
MODEL_DIR = REPO_ROOT / "stage2" / "checkpoints" / "final"
FIG_DIR = REPO_ROOT / "stage2_analysis" / "figures"


def _choropleth(ax, gdf, column, *, title, cmap, vmin=None, vmax=None,
                cbar_label=""):
    gdf.plot(column=column, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
             linewidth=0.2, edgecolor="white", legend=True,
             legend_kwds={"label": cbar_label, "shrink": 0.6})
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load everything --------------------------------------------------
    print(f"[load] shapefile {SHP_PATH.name}")
    gdf = gpd.read_file(SHP_PATH)[["modzcta", "geometry"]].copy()
    gdf["zip_code"] = gdf["modzcta"].astype(int)

    print(f"[load] models from {MODEL_DIR}")
    if not (MODEL_DIR / "nb.npz").exists():
        raise SystemExit(
            f"Final NB model not found at {MODEL_DIR}. "
            f"Run `python stage2/predict.py fit` first.")
    poisson = load_model(MODEL_DIR / "poisson.npz")
    nb = load_model(MODEL_DIR / "nb.npz")

    print(f"[load] engineered panel {DATA_CSV.name}")
    X, y, feats = load_stage2_data(str(DATA_CSV))
    df = pd.read_csv(DATA_CSV)
    assert len(df) == len(y)

    # Predict with both models on the full panel.
    mu_p = poisson.predict(X)
    mu_nb = nb.predict(X)
    alpha = float(getattr(nb, "alpha", 0.2))

    df = df.assign(y=y, mu_p=mu_p, mu_nb=mu_nb)

    # --- Per-ZIP aggregations --------------------------------------------
    agg = df.groupby("zip_code").agg(
        obs_mean=("y", "mean"),
        pred_mean_p=("mu_p", "mean"),
        pred_mean_nb=("mu_nb", "mean"),
    )
    agg["residual"] = agg["obs_mean"] - agg["pred_mean_nb"]

    # Off-peak vs peak predicted means (uses the is_peak column). Off-peak
    # (is_peak=0) turns out to dominate peak everywhere — off-peak windows
    # span more hours of the day, so absolute counts are higher there.
    peak_pivot = (df.groupby(["zip_code", "is_peak"])["mu_nb"].mean()
                    .unstack("is_peak"))
    peak_pivot.columns = ["pred_offpeak", "pred_peak"]
    agg = agg.join(peak_pivot)
    agg["pred_off_minus_peak"] = agg["pred_offpeak"] - agg["pred_peak"]

    # NB conditional sd, averaged over rows: sd_i = sqrt(mu_i + alpha*mu_i^2)
    df["sd_nb"] = np.sqrt(df["mu_nb"] + alpha * df["mu_nb"] ** 2)
    agg["sd_nb_mean"] = df.groupby("zip_code")["sd_nb"].mean()

    gdf = gdf.merge(agg, on="zip_code", how="left")

    # --- Print a small leaderboard ---------------------------------------
    top_obs = agg["obs_mean"].nlargest(10)
    print("\n[top-10 ZIPs by observed mean crash count per 12h]")
    print(top_obs.round(3).to_string())
    print("\n[top-10 ZIPs where model UNDER-predicts (largest +residual)]")
    print(agg["residual"].nlargest(10).round(3).to_string())
    print("\n[top-10 ZIPs where model OVER-predicts (largest -residual)]")
    print(agg["residual"].nsmallest(10).round(3).to_string())

    # --- Individual maps --------------------------------------------------
    vmax_obs = float(max(agg["obs_mean"].max(), agg["pred_mean_nb"].max()))
    # Use same color scale for observed and predicted so they're comparable
    vmax_shared = np.ceil(vmax_obs * 10) / 10

    panels = [
        dict(col="obs_mean",
             title="Observed mean crashes per ZIP (per 12h window)",
             cmap="YlOrRd", vmin=0, vmax=vmax_shared,
             cbar_label="crashes", fname="01_observed.png"),
        dict(col="pred_mean_nb",
             title="NB predicted mean crashes per ZIP",
             cmap="YlOrRd", vmin=0, vmax=vmax_shared,
             cbar_label="crashes", fname="02_predicted_nb.png"),
        dict(col="residual",
             title="Residual (observed − NB predicted)",
             cmap="RdBu_r",
             vmin=-float(agg["residual"].abs().max()),
             vmax=float(agg["residual"].abs().max()),
             cbar_label="crashes", fname="03_residual.png"),
        dict(col="pred_off_minus_peak",
             title="NB predicted: off-peak − peak",
             cmap="PuBuGn", vmin=0,
             vmax=float(agg["pred_off_minus_peak"].max()),
             cbar_label="Δ crashes", fname="04_off_minus_peak.png"),
        dict(col="sd_nb_mean",
             title=f"NB conditional SD per ZIP (α={alpha})",
             cmap="magma_r", vmin=0,
             vmax=float(agg["sd_nb_mean"].max()),
             cbar_label="SD(y|x)", fname="05_nb_sd.png"),
    ]

    for p in panels:
        fig, ax = plt.subplots(figsize=(8, 9))
        _choropleth(ax, gdf, p["col"], title=p["title"], cmap=p["cmap"],
                    vmin=p["vmin"], vmax=p["vmax"], cbar_label=p["cbar_label"])
        fig.tight_layout()
        out = FIG_DIR / p["fname"]
        fig.savefig(out, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] {out.relative_to(REPO_ROOT)}")

    # --- Combined 2x2 summary --------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    for ax, p in zip(axes.flat, panels[:4]):
        _choropleth(ax, gdf, p["col"], title=p["title"], cmap=p["cmap"],
                    vmin=p["vmin"], vmax=p["vmax"], cbar_label=p["cbar_label"])
    fig.suptitle("Stage 2 — NB model diagnostic maps (NYC, ZIP-level)",
                 fontsize=14, y=0.995)
    fig.tight_layout()
    out = FIG_DIR / "summary_2x2.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
