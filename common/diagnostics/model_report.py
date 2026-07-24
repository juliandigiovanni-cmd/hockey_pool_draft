"""Model fit diagnostics: metrics table, predicted-vs-actual, residual plots, baseline comparison.

All functions accept the `models` and `validations` dicts produced by pool_ranking.run_pool_ranking().
Outputs written to:
  {season_dir}/results/diagnostics/model_metrics.csv
  {season_dir}/plots/diagnostics/pva_{key}.png
  {season_dir}/plots/diagnostics/residuals_{key}.png
  {season_dir}/plots/diagnostics/rolling_oos_{key}.png  (if rolling eval data available)

Designed to run automatically at the end of every --stage predict run.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import SeasonConfig
from common.models.training import TrainedModel

logger = logging.getLogger(__name__)

_FRIENDLY = {
    "forward_points":    "Fwd Points/G",
    "defense_points":    "Def Points/G",
    "defense_plus_minus":"Def +/−/G",
    "goalie_wins":       "Goalie Wins/G",
    "goalie_shutouts":   "Goalie SO/G",
    "goalie_gaa":        "Goalie GAA",
    "goalie_save_pct":   "Goalie Sv%",
}


# --------------------------------------------------------------------------- metrics table

def metrics_table(models: dict[str, TrainedModel | None],
                  validations: dict, cfg: SeasonConfig) -> pd.DataFrame:
    """Write model_metrics.csv with one row per model. Returns the DataFrame."""
    rows = []
    for key, m in models.items():
        if m is None:
            rows.append({"model": key, "label": _FRIENDLY.get(key, key)})
            continue
        met = m.metrics
        row = {
            "model": key,
            "label": _FRIENDLY.get(key, key),
            "algorithm": m.model_type,
            "n_train": met.get("n_train"),
            "n_test": met.get("n_test"),
            "test_r2": round(met.get("r2", float("nan")), 4),
            "train_r2": round(met.get("train_r2", float("nan")), 4),
            "baseline_r2": round(met.get("baseline_r2", float("nan")), 4),
            "beats_baseline": (met.get("r2", float("-inf")) > met.get("baseline_r2", float("-inf"))),
            "overfitting_ratio": round(met.get("overfitting_ratio", float("nan")), 4),
            "mae": round(met.get("mae", float("nan")), 4),
            "rmse": round(met.get("rmse", float("nan")), 4),
            "cv_mean": round(met.get("cv_mean", float("nan")), 4),
        }
        # In-sample holdout ranking metrics
        for mk in ("spearman_r", "concordance", "directional_acc", "bias"):
            row[mk] = round(met.get(mk, float("nan")), 4)

        val = validations.get(key)
        if val and val.get("oos"):
            oos = val["oos"]
            row["oos_r2"] = round(oos.get("r2", float("nan")), 4)
            row["oos_mae"] = round(oos.get("mae", float("nan")), 4)
            row["oos_n"] = oos.get("n")
            row["oos_chosen_model"] = val.get("chosen", "")
            for mk in ("spearman_r", "concordance", "directional_acc", "bias"):
                row[f"oos_{mk}"] = round(oos.get(mk, float("nan")), 4)
            row["oos_top1_match_max"] = oos.get("top1_match_max")
            row["oos_top1_match_min"] = oos.get("top1_match_min")
        else:
            for col in ("oos_r2", "oos_mae", "oos_n", "oos_chosen_model",
                        "oos_spearman_r", "oos_concordance", "oos_directional_acc",
                        "oos_bias", "oos_top1_match_max", "oos_top1_match_min"):
                row[col] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir = cfg.season_dir / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_metrics.csv"
    df.to_csv(out_path, index=False)
    logger.info("Wrote model metrics table: %s", out_path)

    # Print a compact summary to stdout
    print("\n=== Model fit summary ===")
    cols = ["label", "algorithm", "test_r2", "baseline_r2", "beats_baseline",
            "spearman_r", "concordance", "directional_acc",
            "oos_r2", "oos_spearman_r", "oos_concordance"]
    print_cols = [c for c in cols if c in df.columns]
    print(df[print_cols].to_string(index=False))
    return df


# --------------------------------------------------------------------------- predicted vs actual

def predicted_vs_actual_plots(models: dict[str, TrainedModel | None], cfg: SeasonConfig) -> None:
    plots_dir = cfg.season_dir / "plots" / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for key, m in models.items():
        if m is None or m.y_test is None or m.y_pred_test is None:
            logger.info("PvA plot: no test data for %s (needs retrain); skipping", key)
            continue
        y, yp = m.y_test, m.y_pred_test
        r2 = m.metrics.get("r2", float("nan"))
        baseline_r2 = m.metrics.get("baseline_r2", float("nan"))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # left: predicted vs actual
        ax = axes[0]
        lims = [min(y.min(), yp.min()), max(y.max(), yp.max())]
        ax.scatter(y, yp, alpha=0.4, s=15, color="#2196F3")
        ax.plot(lims, lims, "r--", lw=1, label="y=x")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{_FRIENDLY.get(key, key)}\ntest R²={r2:.3f}  baseline R²={baseline_r2:.3f}")
        ax.legend(fontsize=8)

        # right: residuals vs predicted
        ax2 = axes[1]
        resid = yp - y
        ax2.scatter(yp, resid, alpha=0.4, s=15, color="#FF5722")
        ax2.axhline(0, color="k", lw=1)
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Residual (pred − actual)")
        ax2.set_title(f"{_FRIENDLY.get(key, key)} — residuals")

        fig.tight_layout()
        out_path = plots_dir / f"pva_{key}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("Wrote PvA+residual plot: %s", out_path)


# --------------------------------------------------------------------------- rolling OOS timeline

def rolling_oos_plot(rolling_results: dict[str, pd.DataFrame], cfg: SeasonConfig) -> None:
    """Plot OOS R² by holdout year for each model that has rolling eval data."""
    if not rolling_results:
        return
    plots_dir = cfg.season_dir / "plots" / "diagnostics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    keys_with_data = [k for k, df in rolling_results.items() if df is not None and not df.empty]
    if not keys_with_data:
        return

    n = len(keys_with_data)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, key in enumerate(keys_with_data):
        df = rolling_results[key]
        ax = axes_flat[i]
        ax.plot(df["holdout_year"], df["r2"], marker="o", color="#4CAF50")
        ax.axhline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("Holdout year")
        ax.set_ylabel("OOS R²")
        ax.set_title(_FRIENDLY.get(key, key))
        ax.set_ylim(-1, 1)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("Rolling OOS R² by holdout year (train on prior seasons → predict holdout)", fontsize=10)
    fig.tight_layout()
    out_path = plots_dir / "rolling_oos_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Wrote rolling OOS plot: %s", out_path)


# --------------------------------------------------------------------------- entrypoint

def generate_diagnostics(models: dict[str, TrainedModel | None], validations: dict,
                          cfg: SeasonConfig, rolling_results: dict | None = None) -> None:
    """Write all diagnostic outputs. Called from pool_ranking.run_pool_ranking()."""
    logger.info("Generating model diagnostics...")
    metrics_table(models, validations, cfg)
    predicted_vs_actual_plots(models, cfg)
    if rolling_results:
        rolling_oos_plot(rolling_results, cfg)
    logger.info("Model diagnostics complete. See results/diagnostics/ and plots/diagnostics/")
