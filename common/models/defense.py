"""Defense model: dual target (points/game and plus-minus/game) with independent pipelines.

Both models use strictly lagged features only. The previous version allowed contemporaneous
team context for plus/minus, but those features are NaN→0 at prediction time (future season
not played yet), causing a systematic train/predict mismatch. Career averages for plus_minus
are added instead (more stable signal than any single-year lag given autocorr r≈0.279).

The plus_minus target is modeled as a residual after removing a defenseman's own points
contribution (plus_minus_per_game - points_per_game), rather than raw plus_minus_per_game
directly: a defenseman's own goal/assist is both a personal point and an on-ice goal-for event,
so points and plus/minus are not independent signals. This is an approximation, not an exact
accounting identity — NHL plus/minus excludes power-play goals, which points does not — but it
isolates the defensive/team-context component the raw target otherwise conflates with a player's
own offense. pool_ranking.py reconstructs the full plus_minus prediction by adding this
residual's prediction back to the points model's prediction before blending/scoring.

Pool scoring for defense is goals + assists + net plus/minus.
"""

from __future__ import annotations

import pandas as pd

from common.config import SeasonConfig
from common.features import engineering as fe
from common.models import training as tr

TARGETS = {"points": "points_per_game", "plus_minus_residual": "plus_minus_residual_per_game"}
_TARGET_COLS = ["points_per_game", "plus_minus_per_game", "plus_minus_residual_per_game",
                "points_player", "total_points", "plus_minus", "total_plus_minus"]

_INDIVIDUAL = ["goals", "assists", "shots", "games_played_player", "shooting_pct",
               "time_on_ice_per_game", "points_player", "points_per_game", "plus_minus",
               "plus_minus_per_game", "hits", "blocked_shots", "penalty_minutes"] + fe.DERIVED_TEAM_COLS


def is_defense(df: pd.DataFrame) -> pd.Series:
    return df["position"].astype(str).str.upper().str.contains("D", na=False)


def _add_teammate_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged team-strength proxies for the quality of a defenseman's supporting cast."""
    df = df.copy()
    proxies = {
        "teammate_offensive_strength": "team_goals_for_per_game_lag1",
        "teammate_defensive_strength": "team_goals_against_per_game_lag1",
        "teammate_consistency": "team_goal_diff_per_game_lag1",
        "teammate_elite": "elite_team_lag1",
    }
    for out, src in proxies.items():
        if src in df.columns:
            df[out] = df[src].abs() if out == "teammate_consistency" else df[src]
    return df


def engineer(df: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    df = fe.add_year(df)
    df = df[is_defense(df)].copy()
    df = fe.per_game_target(df, "points_player", "points_per_game")
    df["total_points"] = pd.to_numeric(df["points_player"], errors="coerce").fillna(0)
    df["total_plus_minus"] = pd.to_numeric(df["plus_minus"], errors="coerce").fillna(0)
    df = fe.per_game_target(df, "total_plus_minus", "plus_minus_per_game")
    df["plus_minus_residual_per_game"] = df["plus_minus_per_game"] - df["points_per_game"]
    df = fe.add_covid_indicators(df)
    df = fe.add_years_played(df, prime_range=(3, 12))  # D peak later than forwards
    df = fe.add_team_goal_differential(df)
    df = fe.clean_common_columns(df, toi_default=20.0)
    df = fe.join_moneypuck(df, cfg, "skaters")
    lags = fe.lag_feature_list(df, _INDIVIDUAL)
    df = fe.create_lag_features(df, lags, cfg.lag_years, fe.resolve_min_training_year(df, cfg))
    df = fe.add_career_averages(df, ["plus_minus", "plus_minus_per_game"])
    df = fe.engineer_features(df, cfg.lag_years,
                              interaction_metrics=("plus_minus_per_game", "points_per_game", "time_on_ice_per_game"),
                              covid_metrics=("plus_minus_per_game", "points_per_game"))
    return _add_teammate_quality(df)


def build_xy_for(df: pd.DataFrame, target: str):
    allow_team = False  # contemporaneous team stats are NaN at prediction time; lagged proxies remain
    cols = fe.select_feature_columns(df, _TARGET_COLS, allow_contemporaneous_team=allow_team)
    X, y = fe.build_xy(df, cols, TARGETS[target])
    return X, y, df.loc[X.index, "year"]


def train(df_raw: pd.DataFrame, cfg: SeasonConfig, persist: bool = True) -> dict:
    eng = engineer(df_raw, cfg)
    out = {}
    for target in TARGETS:
        res = tr.train_all_vs_exclude_latest(lambda f, t=target: build_xy_for(f, t), eng, target, cfg)
        if persist:
            tr.save_model(res["model"], cfg, f"defense_{target}")
        out[target] = res
    return out
