"""Forward model: single target, points per game, strict lagged-only features.

Thin wrapper over the shared engineering + training harness. Pool scoring for forwards is
goals + assists (= points), so the model predicts points-per-game and the pool step multiplies
by projected games. No leakage exceptions here — forwards are the strict case.
"""

from __future__ import annotations

import pandas as pd

from common.config import SeasonConfig
from common.features import engineering as fe
from common.models import training as tr

TARGETS = {"points": "points_per_game"}
_TARGET_COLS = ["points_per_game", "points_player", "total_points"]

_INDIVIDUAL = ["goals", "assists", "shots", "games_played_player", "shooting_pct",
               "time_on_ice_per_game", "points_player", "points_per_game",
               "gw_goals", "hits", "blocked_shots", "penalty_minutes"]


def is_forward(df: pd.DataFrame) -> pd.Series:
    return ~df["position"].astype(str).str.upper().str.contains("D", na=False)


def engineer(df: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    df = fe.add_year(df)
    df = df[is_forward(df)].copy()
    df = fe.per_game_target(df, "points_player", "points_per_game")
    df["total_points"] = pd.to_numeric(df["points_player"], errors="coerce").fillna(0)
    df = fe.add_covid_indicators(df)
    df = fe.add_years_played(df, prime_range=(3, 10))
    df = fe.clean_common_columns(df, toi_default=15.0)
    df = fe.join_moneypuck(df, cfg, "skaters")
    lags = fe.lag_feature_list(df, _INDIVIDUAL)
    df = fe.create_lag_features(df, lags, cfg.lag_years, fe.resolve_min_training_year(df, cfg))
    return fe.engineer_features(df, cfg.lag_years)


def build_xy_for(df: pd.DataFrame, target: str = "points"):
    cols = fe.select_feature_columns(df, _TARGET_COLS, allow_contemporaneous_team=False)
    X, y = fe.build_xy(df, cols, TARGETS[target])
    return X, y, df.loc[X.index, "year"]


def train(df_raw: pd.DataFrame, cfg: SeasonConfig, persist: bool = True) -> dict:
    """Returns {'points': <train_all_vs_exclude_latest result>}; persists the chosen model."""
    eng = engineer(df_raw, cfg)
    res = tr.train_all_vs_exclude_latest(lambda f: build_xy_for(f, "points"), eng, "points", cfg)
    if persist:
        tr.save_model(res["model"], cfg, "forward_points")
    return {"points": res}
