"""Goalie model: quadruple target with the 40+ GP qualification rule.

Targets: wins/game and shutouts/game for all goalies; GAA and save% for qualified goalies only
(>= min_gp_for_bonus games, per config, matching the pool's bonus-eligibility rule — a goalie
who barely played shouldn't win the best-GAA/best-save% bonus on a tiny sample). All four are
strict lagged-only (goalie legacy v2.0 used strict exclusions throughout).

Pool scoring: 1 pt/win, 3 pts total for a shutout win, +10 best GAA and +10 best save% among
qualified goalies (applied in pool_ranking, not here).
"""

from __future__ import annotations

import pandas as pd

from common.config import SeasonConfig
from common.features import engineering as fe
from common.models import training as tr

TARGETS = {"wins": "wins_per_game", "shutouts": "shutouts_per_game",
           "gaa": "goals_against_average", "save_pct": "save_percentage"}
QUALIFIED_ONLY = ("gaa", "save_pct")
_TARGET_COLS = list(TARGETS.values()) + ["wins_player", "shutouts", "goals_against_avg",
                                         "save_pct", "total_wins", "total_shutouts"]

_INDIVIDUAL = ["wins_player", "losses_player", "ot_losses_player", "shutouts",
               "goals_against_player", "goals_against_avg", "save_pct", "saves", "shots_against",
               "games_played_player", "games_started", "time_on_ice",
               "wins_per_game", "shutouts_per_game", "goals_against_average", "save_percentage",
               "total_wins", "total_shutouts"] + fe.DERIVED_TEAM_COLS


def _min_gp(cfg: SeasonConfig) -> int:
    return int(cfg.scoring.get("goalies", {}).get("min_gp_for_bonus", 40))


def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_wins"] = pd.to_numeric(df["wins_player"], errors="coerce").fillna(0).clip(lower=0)
    df["total_shutouts"] = pd.to_numeric(df["shutouts"], errors="coerce").fillna(0).clip(lower=0)
    df = fe.per_game_target(df, "total_wins", "wins_per_game")
    df = fe.per_game_target(df, "total_shutouts", "shutouts_per_game")
    df["goals_against_average"] = pd.to_numeric(df["goals_against_avg"], errors="coerce").clip(0, 10).fillna(3.0)
    sv = pd.to_numeric(df["save_pct"], errors="coerce")
    sv = sv / 100 if sv.max(skipna=True) and sv.max(skipna=True) > 1 else sv
    df["save_percentage"] = sv.clip(0.5, 1.0).fillna(0.900)
    return df


def engineer(df: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    df = fe.add_year(df)
    df = _create_targets(df)
    df = fe.add_covid_indicators(df)
    df = fe.add_years_played(df, prime_range=(3, 12))
    df = fe.add_team_goal_differential(df)
    df = fe.clean_common_columns(df, toi_default=0.0)
    df = fe.join_moneypuck(df, cfg, "goalies")
    lags = fe.lag_feature_list(df, _INDIVIDUAL)
    df = fe.create_lag_features(df, lags, cfg.lag_years, fe.resolve_min_training_year(df, cfg))
    df = fe.add_career_averages(df, ["save_pct", "goals_against_avg", "shutouts",
                                     "wins_per_game", "save_percentage", "goals_against_average",
                                     "shutouts_per_game"])
    return fe.engineer_features(df, cfg.lag_years,
                                interaction_metrics=("wins_per_game", "save_pct", "goals_against_avg", "shutouts"),
                                covid_metrics=("wins_per_game", "save_pct"))


def build_xy_for(df: pd.DataFrame, target: str, cfg: SeasonConfig):
    frame = df
    cols = fe.select_feature_columns(frame, _TARGET_COLS, allow_contemporaneous_team=False)
    X, y = fe.build_xy(frame, cols, TARGETS[target],
                       core_lag_prefixes=("wins_player_lag", "games_played_player_lag"))
    return X, y, frame.loc[X.index, "year"]


_LINEAR_ONLY = ("ridge", "lasso", "elastic_net")


def train(df_raw: pd.DataFrame, cfg: SeasonConfig, persist: bool = True) -> dict:
    eng = engineer(df_raw, cfg)
    out = {}
    for target in TARGETS:
        # Restrict GAA and save_pct to linear models: near-zero autocorr means tree models
        # overfit badly on this data, while regularized linear models regress toward the mean.
        kwargs = {"model_types": _LINEAR_ONLY} if target in QUALIFIED_ONLY else {}
        try:
            res = tr.train_all_vs_exclude_latest(
                lambda f, t=target: build_xy_for(f, t, cfg), eng, target, cfg, **kwargs)
        except (ValueError, RuntimeError) as e:
            out[target] = {"model": None, "error": str(e)}
            continue
        if persist:
            tr.save_model(res["model"], cfg, f"goalie_{target}")
        out[target] = res
    return out
