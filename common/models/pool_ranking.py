"""Pool-ranking integration — the corrected successor to legacy finalpool.

What changed vs. finalpool v1.7: that script threw away the tuned family models and re-derived a
single RandomForest(n_estimators=100, max_depth=10) per target from scratch. This module instead
consumes the persisted, 5-algorithm-selected family models (common/models/{forwards,defense,
goalies}.py) — training + persisting them if they aren't on disk yet — so the model-selection
work actually reaches the rankings. It also replaces the hardcoded Marner/Ehlers/Dobson trades
with common.scrape.rosters.apply_current_team_overrides (live roster snapshot + config fallback).

Flow: load processed skater/goalie data -> train (5-algo select) + OOS-validate each target ->
build next-season prediction rows by appending placeholder rows and reusing the exact feature
pipeline -> apply current-team overrides -> project games (forwards/defense cfg.games_per_season,
goalies 3-yr avg capped at cfg.games_per_season)
-> score pool_points per cfg.scoring -> write rankings + report to results_dir and plots to plots_dir.
"""

from __future__ import annotations

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.config import SeasonConfig
from common.diagnostics.model_report import generate_diagnostics
from common.features import engineering as fe
from common.models import defense as dfe
from common.models import forwards as fwd
from common.models import goalies as goa
from common.models import training as tr
from common.scrape.rosters import apply_current_team_overrides

logger = logging.getLogger(__name__)

_ID_COLS = ["player_id", "player_name", "position", "team_abbrev"]


# --------------------------------------------------------------------------- data + models

def _load_processed(cfg: SeasonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    skater = pd.read_csv(cfg.processed_dir / "skater_team_data.csv")
    goalie = pd.read_csv(cfg.processed_dir / "goalie_team_data.csv")
    return skater, goalie


def _load_or_train(cfg: SeasonConfig, skater: pd.DataFrame, goalie: pd.DataFrame,
                   retrain: bool) -> tuple[dict, dict]:
    """Return ({model_key: TrainedModel}, {model_key: validation-result-or-None}).

    Trains (and persists) when retrain=True or a model is missing on disk; otherwise reuses the
    persisted joblib models. Validation results are only available on the training path.
    """
    specs = {"forward_points": (fwd, skater), "defense_points": (dfe, skater),
             "defense_plus_minus": (dfe, skater),
             **{f"goalie_{t}": (goa, goalie) for t in goa.TARGETS}}

    models, validations = {}, {}
    trained_cache: dict[int, dict] = {}
    for key in specs:
        loaded = None if retrain else tr.load_model(cfg, key)
        if loaded is not None:
            models[key] = loaded
            validations[key] = None
            continue
        module, df = specs[key]
        cache_id = id(module)
        if cache_id not in trained_cache:
            logger.info("Training %s family models...", module.__name__)
            trained_cache[cache_id] = module.train(df, cfg, persist=True)
        target = key.split("_", 1)[1] if module is not goa else key.replace("goalie_", "")
        res = trained_cache[cache_id].get(target)
        models[key] = res.get("model") if res else None
        validations[key] = res
    return models, validations


# --------------------------------------------------------------------------- prediction rows

def _augment_with_prediction_rows(raw_df: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    """Append one placeholder row per recently-active player for the season being predicted.

    The placeholder carries only identity/team columns; all stats are NaN. Running the normal
    feature pipeline then fills its lag features from real history, so no separate synthetic-lag
    code is needed. Contemporaneous (current-season) columns stay NaN — a real limitation for the
    defense plus/minus model, whose team-context features are unknown for a season not yet played.
    """
    df = fe.add_year(raw_df)
    latest = int(df["year"].max())
    recent = df[df["year"] == latest]
    keep = [c for c in _ID_COLS if c in recent.columns]
    ph = recent[keep].drop_duplicates("player_id").copy()
    ph["season"] = int(cfg.season_id)
    return pd.concat([raw_df, ph], ignore_index=True)


def _predict_rows(engineer_fn, aug_raw: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    target_year = int(cfg.season_id[:4])
    eng = engineer_fn(aug_raw, cfg)
    rows = eng[eng["year"] == target_year].copy()
    return apply_current_team_overrides(rows, cfg)


def _projected_games(raw_df: pd.DataFrame, games_per_season: int, default: int) -> dict:
    """3-year rolling average of games_played_player per player, capped at games_per_season."""
    df = fe.add_year(raw_df)
    out = {}
    for pid, g in df.sort_values("year").groupby("player_id"):
        gp = pd.to_numeric(g["games_played_player"], errors="coerce").dropna().tail(3)
        gp = gp[gp > 0]
        out[pid] = min(gp.mean(), games_per_season) if len(gp) else default
    return out


# --------------------------------------------------------------------------- scoring

def compute_pool_points(cfg: SeasonConfig, fwd_df: pd.DataFrame, def_df: pd.DataFrame,
                        goal_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    sc = cfg.scoring
    pm_w = sc.get("defense", {}).get("plus_minus", 1)
    g = sc.get("goalies", {})
    win_pts, sut_pts = g.get("win", 1), g.get("shutout_win", 3)
    min_gp = g.get("min_gp_for_bonus", 40)
    gaa_bonus, sv_bonus = g.get("best_gaa_bonus", 10), g.get("best_save_pct_bonus", 10)

    fwd_df = fwd_df.copy()
    fwd_df["pool_points"] = fwd_df["pred_points_per_game"] * fwd_df["projected_games"]

    def_df = def_df.copy()
    def_df["predicted_points"] = def_df["pred_points_per_game"] * def_df["projected_games"]
    def_df["predicted_plus_minus"] = def_df["pred_plus_minus_per_game"] * def_df["projected_games"]
    def_df["pool_points"] = def_df["predicted_points"] + pm_w * def_df["predicted_plus_minus"]

    goal_df = goal_df.copy()
    wins = (goal_df["pred_wins_per_game"] * goal_df["projected_games"]).clip(lower=0)
    shutouts = (goal_df["pred_shutouts_per_game"] * goal_df["projected_games"]).clip(lower=0)
    shutouts = np.minimum(shutouts, wins)
    goal_df["predicted_wins"] = wins
    goal_df["predicted_shutouts"] = shutouts
    goal_df["pool_points"] = win_pts * (wins - shutouts) + sut_pts * shutouts
    goal_df["qualified"] = goal_df["projected_games"] >= min_gp
    goal_df["gaa_bonus"] = False
    goal_df["save_pct_bonus"] = False
    qual = goal_df[goal_df["qualified"]]
    if len(qual):
        if "pred_gaa" in goal_df.columns and qual["pred_gaa"].notna().any():
            idx = qual["pred_gaa"].idxmin()
            goal_df.loc[idx, ["pool_points", "gaa_bonus"]] = [goal_df.loc[idx, "pool_points"] + gaa_bonus, True]
        if "pred_save_pct" in goal_df.columns and qual["pred_save_pct"].notna().any():
            idx = qual["pred_save_pct"].idxmax()
            goal_df.loc[idx, ["pool_points", "save_pct_bonus"]] = [goal_df.loc[idx, "pool_points"] + sv_bonus, True]

    return {"forward": fwd_df, "defense": def_df, "goalie": goal_df}


# --------------------------------------------------------------------------- prediction blending

def _blend(df: pd.DataFrame, pred_col: str, ref_col: str, alpha: float) -> pd.Series:
    """Blend model prediction with a historical reference to reduce compression at extremes.

    alpha = weight on model; (1-alpha) = weight on reference.
    Calibrated from OOS concordance: alpha = 2*(concordance - 0.5).
    Falls back to model prediction where the reference is NaN (new players).
    """
    if ref_col not in df.columns:
        return df[pred_col]
    blended = alpha * df[pred_col] + (1 - alpha) * df[ref_col]
    return blended.fillna(df[pred_col])


# --------------------------------------------------------------------------- outputs

def _rank(df: pd.DataFrame, position: str) -> pd.DataFrame:
    cols = [c for c in ["player_id", "player_name", "team_abbrev",
                        "projected_games", "pool_points", "pool_points_full_season"]
            if c in df.columns]
    out = df[cols].sort_values("pool_points", ascending=False).reset_index(drop=True)
    out.insert(0, "position", position)
    out.insert(0, "rank", out.index + 1)
    return out


def _write_outputs(cfg: SeasonConfig, ranked: dict[str, pd.DataFrame],
                   validations: dict, models: dict) -> None:
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"finalpool_{cfg.season}"

    overall = pd.concat(ranked.values(), ignore_index=True).sort_values(
        "pool_points", ascending=False).reset_index(drop=True)
    overall["overall_rank"] = overall.index + 1

    for pos, df in ranked.items():
        df.to_csv(cfg.results_dir / f"{prefix}_{pos}_rankings.csv", index=False)
    overall.to_csv(cfg.results_dir / f"{prefix}_overall_rankings.csv", index=False)

    try:
        with pd.ExcelWriter(cfg.results_dir / f"{prefix}_rankings.xlsx") as xl:
            overall.to_excel(xl, sheet_name="overall", index=False)
            for pos, df in ranked.items():
                df.to_excel(xl, sheet_name=pos, index=False)
    except Exception as e:  # pragma: no cover
        logger.warning("Could not write xlsx: %s", e)

    lines = [f"NHL POOL RANKINGS — {cfg.season}", "=" * 60, "", "Model selection & validation:"]
    for key, val in validations.items():
        m = models.get(key)
        if m is None:
            lines.append(f"  {key}: (no model)")
            continue
        line = f"  {key}: {m.model_type}  test_r2={m.metrics['r2']:.3f}"
        if val and val.get("oos"):
            line += f"  OOS_r2={val['oos']['r2']:.3f} (chose '{val['chosen']}')"
        lines.append(line)
    lines += ["", "Top 20 overall:", "-" * 60]
    for _, r in overall.head(20).iterrows():
        lines.append(f"  {r['overall_rank']:>3}. {r['player_name']:<26} {r['position']:<8} "
                     f"{r['pool_points']:>7.1f}")
    (cfg.results_dir / f"{prefix}_report.txt").write_text("\n".join(lines))
    logger.info("Wrote rankings + report to %s", cfg.results_dir)


def _write_plots(cfg: SeasonConfig, models: dict) -> None:
    cfg.plots_dir.mkdir(parents=True, exist_ok=True)
    for key, m in models.items():
        if m is None:
            continue
        try:
            imp = tr.feature_importance(m).head(15)
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.barh(imp["feature"][::-1], imp["importance"][::-1])
            ax.set_title(f"{key} — top features ({m.model_type})")
            fig.tight_layout()
            fig.savefig(cfg.plots_dir / f"feature_importance_{key}.png", dpi=150)
            plt.close(fig)
        except Exception as e:  # pragma: no cover
            logger.warning("Plot failed for %s: %s", key, e)


# --------------------------------------------------------------------------- entrypoint

def run_pool_ranking(cfg: SeasonConfig, retrain: bool = True) -> dict[str, pd.DataFrame]:
    """Full integration. retrain=False reuses persisted joblib models when all are present."""
    skater, goalie = _load_processed(cfg)
    models, validations = _load_or_train(cfg, skater, goalie, retrain)

    skater_aug = _augment_with_prediction_rows(skater, cfg)
    goalie_aug = _augment_with_prediction_rows(goalie, cfg)

    # Per-player projected games: 3-year rolling avg of actual GP, capped at games_per_season
    skater_gp_map = _projected_games(skater, cfg.games_per_season, default=70)
    goalie_gp_map = _projected_games(goalie, cfg.games_per_season, default=20)

    # Forwards
    fwd_rows = _predict_rows(fwd.engineer, skater_aug, cfg)
    fwd_rows["pred_points_per_game"] = models["forward_points"].predict(fwd_rows)
    fwd_rows["projected_games"] = fwd_rows["player_id"].map(skater_gp_map).fillna(70)

    # Defense
    def_rows = _predict_rows(dfe.engineer, skater_aug, cfg)
    def_rows["pred_points_per_game"] = models["defense_points"].predict(def_rows)
    def_rows["pred_plus_minus_per_game"] = models["defense_plus_minus"].predict(def_rows)
    def_rows["projected_games"] = def_rows["player_id"].map(skater_gp_map).fillna(70)

    # Blend model predictions with 2-year historical averages to correct compression at the top.
    # Alpha weights derived from OOS concordance: alpha = 2*(concordance - 0.5).
    #   defense_points OOS concordance ~0.72  → alpha=0.45 (model trusted, hist corrects compression)
    #   defense_plus_minus OOS concordance ~0.54 → alpha=0.10 (model barely beats chance; hist dominates)
    def_rows["pred_points_per_game"] = _blend(
        def_rows, "pred_points_per_game", "points_per_game_hist_avg", alpha=0.45)
    def_rows["pred_plus_minus_per_game"] = _blend(
        def_rows, "pred_plus_minus_per_game", "plus_minus_per_game_hist_avg", alpha=0.10)

    # Goalies
    goal_rows = _predict_rows(goa.engineer, goalie_aug, cfg)
    goal_rows["projected_games"] = goal_rows["player_id"].map(goalie_gp_map).fillna(20)
    for target, col in [("wins", "pred_wins_per_game"), ("shutouts", "pred_shutouts_per_game"),
                        ("gaa", "pred_gaa"), ("save_pct", "pred_save_pct")]:
        m = models.get(f"goalie_{target}")
        goal_rows[col] = m.predict(goal_rows) if m is not None else np.nan

    scored = compute_pool_points(cfg, fwd_rows, def_rows, goal_rows)

    # Also score assuming everyone plays a full season, for comparison
    fwd_full = fwd_rows.copy(); fwd_full["projected_games"] = cfg.games_per_season
    def_full = def_rows.copy(); def_full["projected_games"] = cfg.games_per_season
    goal_full = goal_rows.copy(); goal_full["projected_games"] = cfg.games_per_season
    scored_full = compute_pool_points(cfg, fwd_full, def_full, goal_full)
    for pos in scored:
        scored[pos]["pool_points_full_season"] = scored_full[pos]["pool_points"].values

    ranked = {pos: _rank(df, pos) for pos, df in scored.items()}

    _write_outputs(cfg, ranked, validations, models)
    _write_plots(cfg, models)
    generate_diagnostics(models, validations, cfg)
    return ranked
