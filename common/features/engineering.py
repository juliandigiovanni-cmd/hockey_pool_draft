"""Shared feature engineering for all three position families.

This is the genuinely shared ~80% that the legacy forward/defense/goalie scripts each
re-implemented near-verbatim: lag features, experience/COVID controls, team-context
identification and lagging, the MoneyPuck xG join, and the leakage-safe feature selection.

Leakage model (important): instead of the legacy denylist that tried to enumerate every
contemporaneous column to drop, feature selection here is an *allowlist* — a column is only
eligible as a predictor if it is lagged, a known pre-season control (COVID indicators), or an
engineered feature derived from lagged data. Nothing contemporaneous can slip through by being
forgotten. The one deliberate exception is `allow_contemporaneous_team=True`, used by the
defense plus/minus model (see common/models/defense.py) which is documented as intentionally
allowed to see current-season *team* context. Individual current-season stats are never allowed.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

COVID_YEARS = (2019, 2020)  # 2019-20 (suspended) and 2020-21 (shortened) seasons

# Team-context columns as emitted by common/clean/merge.py. Skater and goalie schemas name a
# few of these differently (wins vs wins_team, face_off_win_pct_team vs face_off_win_pct), so
# this is a union matched by presence rather than positional slicing.
TEAM_FEATURE_NAMES = [
    "wins", "losses", "ot_losses", "wins_team", "losses_team", "ot_losses_team",
    "points_team", "goals_for", "goals_against", "goals_against_team", "goal_diff",
    "points_pct", "pp_pct", "pk_pct", "shots_for_per_game", "shots_against_per_game",
    "face_off_win_pct_team", "face_off_win_pct",
]

# Derived team-strength columns created by add_team_goal_differential(); these count as team
# context for the contemporaneous-team allowance.
DERIVED_TEAM_COLS = [
    "team_goal_differential", "team_goals_for_per_game", "team_goals_against_per_game",
    "team_goal_diff_per_game", "strong_offensive_team", "strong_defensive_team", "elite_team",
]

# Substrings marking a column as an engineered-from-lag feature (always allowlisted).
_ENGINEERED_MARKERS = ("_x_", "_hist_avg", "_hist_std", "_trend", "_per_game_lag",
                       "rel_team", "teammate", "normalized_team", "_performance")

_EXPERIENCE_LAG_BASE = ["years_played", "years_played_squared", "years_played_cubed"]


# --------------------------------------------------------------------------- base frame

def add_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["season_str"] = df["season"].astype(str)
    df["year"] = df["season_str"].str[:4].astype(int)
    return df


def add_covid_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["covid_2019_2020"] = (df["year"] == 2019).astype(int)
    df["covid_2020_2021"] = (df["year"] == 2020).astype(int)
    df["covid_season"] = df["year"].isin(COVID_YEARS).astype(int)
    return df


def add_years_played(df: pd.DataFrame, prime_range: tuple[int, int] = (3, 10)) -> pd.DataFrame:
    """Career length + polynomial/stage terms. prime_range differs by position (D peaks later)."""
    df = df.copy()
    first = df.groupby("player_id")["year"].transform("min")
    df["first_season_year"] = first
    df["years_played"] = (df["year"] - first + 1).clip(lower=1, upper=25)
    df["years_played_squared"] = df["years_played"] ** 2
    df["years_played_cubed"] = df["years_played"] ** 3
    lo, hi = prime_range
    df["rookie_year"] = (df["years_played"] == 1).astype(int)
    df["prime_years"] = df["years_played"].between(lo, hi).astype(int)
    df["veteran_years"] = (df["years_played"] > hi).astype(int)
    return df


def add_team_goal_differential(df: pd.DataFrame) -> pd.DataFrame:
    """Team goals-for/against context (used by defense plus/minus and goalie models)."""
    df = df.copy()
    ga_col = "goals_against" if "goals_against" in df.columns else "goals_against_team"
    if "goals_for" not in df.columns or ga_col not in df.columns:
        logger.warning("Team goal columns not found; skipping team goal differential features")
        return df
    gp = df.get("games_played_team")
    gp = (gp if gp is not None else pd.Series(82, index=df.index)).replace(0, 82).fillna(82)
    gf, ga = df["goals_for"].fillna(0), df[ga_col].fillna(0)
    df["team_goal_differential"] = gf - ga
    df["team_goals_for_per_game"] = gf / gp
    df["team_goals_against_per_game"] = ga / gp
    df["team_goal_diff_per_game"] = df["team_goal_differential"] / gp
    df["strong_offensive_team"] = (df["team_goals_for_per_game"] > df["team_goals_for_per_game"].quantile(0.75)).astype(int)
    df["strong_defensive_team"] = (df["team_goals_against_per_game"] < df["team_goals_against_per_game"].quantile(0.25)).astype(int)
    df["elite_team"] = ((df["team_goal_diff_per_game"] > df["team_goal_diff_per_game"].quantile(0.8))
                        & (df["team_goals_for_per_game"] > df["team_goals_for_per_game"].median())).astype(int)
    return df


def clean_common_columns(df: pd.DataFrame, toi_default: float = 15.0) -> pd.DataFrame:
    """Normalize shooting_pct/time_on_ice and drop rows with no usable games-played record."""
    df = df.copy()
    if "shooting_pct" in df.columns and df["shooting_pct"].max(skipna=True) is not np.nan:
        if df["shooting_pct"].max(skipna=True) and df["shooting_pct"].max(skipna=True) > 1:
            df["shooting_pct"] = df["shooting_pct"] / 100
    if "time_on_ice_per_game" not in df.columns:
        df["time_on_ice_per_game"] = toi_default
    df["time_on_ice_per_game"] = df["time_on_ice_per_game"].fillna(toi_default)
    return df


def per_game_target(df: pd.DataFrame, numerator_col: str, out_col: str,
                    games_col: str = "games_played_player") -> pd.DataFrame:
    """Create a per-game target (0 where games_played == 0). Used by every position's targets."""
    df = df.copy()
    games = pd.to_numeric(df[games_col], errors="coerce").fillna(0)
    num = pd.to_numeric(df[numerator_col], errors="coerce").fillna(0)
    df[out_col] = np.where(games > 0, num / games.replace(0, np.nan), 0.0)
    df[out_col] = df[out_col].fillna(0.0)
    return df


def resolve_min_training_year(df: pd.DataFrame, cfg: SeasonConfig) -> int:
    if cfg.min_training_year is not None:
        return cfg.min_training_year
    return int(df["year"].min()) + cfg.lag_years


# --------------------------------------------------------------------------- lags

def identify_team_features(df: pd.DataFrame) -> list[str]:
    out = []
    for col in TEAM_FEATURE_NAMES:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            out.append(col)
    return out


def lag_feature_list(df: pd.DataFrame, individual_features: list[str]) -> list[str]:
    """Everything that should be lagged: caller's individual stats + power-play + MoneyPuck
    + team-context + experience columns that are actually present."""
    pp = [c for c in df.columns if c.startswith("pp") and "lag" not in c]
    mp = [c for c in df.columns if c.startswith("mp_") and "lag" not in c]
    feats = individual_features + pp + mp + identify_team_features(df) + _EXPERIENCE_LAG_BASE
    return [f for f in dict.fromkeys(feats) if f in df.columns]


def create_lag_features(df: pd.DataFrame, lag_features: list[str], lag_years: int,
                        min_training_year: int) -> pd.DataFrame:
    """Vectorized per-player lags (groupby/shift), then filter to years with full lag history.

    A cleaner, much faster equivalent of the legacy per-player Python loop: players with
    insufficient history get NaN lags automatically and are dropped later in build_xy.
    """
    df = df.sort_values(["player_id", "year"]).copy()
    g = df.groupby("player_id", sort=False)
    new_cols = {}
    for feature in lag_features:
        for lag in range(1, lag_years + 1):
            new_cols[f"{feature}_lag{lag}"] = g[feature].shift(lag)
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df[df["year"] >= min_training_year].copy()


def engineer_features(df: pd.DataFrame, lag_years: int,
                      interaction_metrics: tuple[str, ...] = ("goals", "assists", "points_per_game", "shots"),
                      covid_metrics: tuple[str, ...] = ("points_per_game", "goals", "assists")) -> pd.DataFrame:
    """Derive hist-avg/std/trend, per-game rates, and experience/COVID interaction terms from lags."""
    df = df.copy()
    lag_cols = [c for c in df.columns if re.search(r"_lag\d+$", c)]
    bases = sorted({re.sub(r"_lag\d+$", "", c) for c in lag_cols})

    derived = {}
    for base in bases:
        cols = [f"{base}_lag{l}" for l in range(1, lag_years + 1) if f"{base}_lag{l}" in df.columns]
        if not cols:
            continue
        derived[f"{base}_hist_avg"] = df[cols].mean(axis=1, skipna=True)
        if len(cols) >= 2:
            derived[f"{base}_hist_std"] = df[cols].std(axis=1, skipna=True).fillna(0)
            derived[f"{base}_trend"] = (df[f"{base}_lag1"] - df[f"{base}_lag2"]).fillna(0)

    # per-game rates from lagged counting stats
    for lag in range(1, lag_years + 1):
        gp = f"games_played_player_lag{lag}"
        gp = gp if gp in df.columns else f"games_played_lag{lag}"
        if gp not in df.columns:
            continue
        for stat in ("goals", "assists"):
            sc = f"{stat}_lag{lag}"
            if sc in df.columns:
                derived[f"{stat}_per_game_lag{lag}"] = (df[sc] / df[gp].replace(0, np.nan)).fillna(0)

    # experience interactions
    if "years_played" in df.columns:
        perf_lags = [c for c in lag_cols if any(m in c for m in [f"{k}_lag" for k in interaction_metrics])]
        for col in perf_lags[:15]:
            derived[f"{col}_x_experience"] = df[col] * df["years_played"]

    # COVID interactions
    if "covid_season" in df.columns:
        cov_lags = [c for c in lag_cols if any(m in c for m in [f"{k}_lag" for k in covid_metrics])]
        for col in cov_lags[:10]:
            derived[f"{col}_x_covid"] = df[col] * df["covid_season"]

    df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
    return df.replace([np.inf, -np.inf], np.nan)


# --------------------------------------------------------------------------- MoneyPuck join

def _normalize_name(s: pd.Series) -> pd.Series:
    def norm(x):
        x = unicodedata.normalize("NFKD", str(x)).encode("ascii", "ignore").decode()
        return re.sub(r"[^a-z0-9]", "", x.lower())
    return s.map(norm)

# Best-effort MoneyPuck column picks (their exact names couldn't be network-verified this
# session). Only columns actually present are joined; anything missing is silently skipped.
_MP_SKATER_COLS = ["I_F_xGoals", "I_F_xOnGoal", "I_F_shotAttempts", "I_F_highDangerShots",
                   "I_F_xRebounds", "onIce_xGoalsPercentage", "gameScore"]
_MP_GOALIE_COLS = ["xGoals", "xOnGoal", "highDangerShots", "mediumDangerShots", "xFreeze", "penalties"]


def join_moneypuck(df: pd.DataFrame, cfg: SeasonConfig, entity: str) -> pd.DataFrame:
    """Left-join MoneyPuck xG/shot-quality columns (prefixed mp_) onto the NHL API frame.

    Primary join: player_id integer (same value in both sources as 'playerId'/'player_id') +
    season start year — exact, no name ambiguity. Fallback: normalized name + season for any
    rows the ID join misses (covers edge cases where playerId is absent in older MP data).
    Missing file or column-name drift degrades to a no-op. Joined columns are current-season,
    so callers must lag them (lag_feature_list picks up mp_* automatically) to avoid leakage.
    """
    path = cfg.raw_dir / "moneypuck" / f"moneypuck_{entity}.csv"
    if not path.exists():
        logger.info("MoneyPuck %s file not found (%s); skipping xG features", entity, path)
        return df
    try:
        mp = pd.read_csv(path)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Could not read MoneyPuck %s: %s; skipping", entity, e)
        return df

    season_col = "season" if "season" in mp.columns else "season_start_year"
    if season_col not in mp.columns:
        logger.warning("MoneyPuck %s missing season column; skipping", entity)
        return df
    if "situation" in mp.columns:
        mp = mp[mp["situation"] == "all"]

    wanted = _MP_SKATER_COLS if entity == "skaters" else _MP_GOALIE_COLS
    present = [c for c in wanted if c in mp.columns]
    if not present:
        logger.warning("No expected MoneyPuck %s stat columns present; skipping", entity)
        return df

    mp_prefixed = [f"mp_{c}" for c in present]
    mp = mp.rename(columns={c: f"mp_{c}" for c in present})
    mp["_mp_year"] = pd.to_numeric(mp[season_col], errors="coerce")
    mp["_mp_pid"] = pd.to_numeric(mp.get("playerId"), errors="coerce") if "playerId" in mp.columns else np.nan
    name_col = next((c for c in ("name", "player", "playerName") if c in mp.columns), None)
    mp["_mp_name"] = _normalize_name(mp[name_col]) if name_col else ""

    df = df.copy()
    df["_df_pid"] = pd.to_numeric(df["player_id"], errors="coerce")
    df["_df_year"] = df["year"]
    df["_df_name"] = _normalize_name(df["player_name"])

    # --- Primary join: player_id + season year ---
    mp_id = (mp[["_mp_pid", "_mp_year"] + mp_prefixed]
             .dropna(subset=["_mp_pid"])
             .drop_duplicates(["_mp_pid", "_mp_year"]))
    out = df.merge(
        mp_id.rename(columns={"_mp_pid": "_df_pid", "_mp_year": "_df_year"}),
        on=["_df_pid", "_df_year"], how="left",
    )
    id_matched = out[mp_prefixed[0]].notna().sum()

    # --- Fallback join: normalized name + season for rows the ID join missed ---
    unmatched_mask = out[mp_prefixed[0]].isna()
    name_matched = 0
    if unmatched_mask.any() and name_col is not None:
        mp_name = (mp[["_mp_name", "_mp_year"] + mp_prefixed]
                   .drop_duplicates(["_mp_name", "_mp_year"]))
        unmatched_idx = out.index[unmatched_mask]
        # reset_index so the merge preserves original positions; restore after
        fallback = (out.loc[unmatched_idx, ["_df_name", "_df_year"]]
                    .reset_index()
                    .merge(mp_name.rename(columns={"_mp_name": "_df_name", "_mp_year": "_df_year"}),
                           on=["_df_name", "_df_year"], how="left")
                    .set_index("index"))
        for col in mp_prefixed:
            out.loc[unmatched_idx, col] = fallback.loc[unmatched_idx, col]
        name_matched = out.loc[unmatched_idx, mp_prefixed[0]].notna().sum()

    total = out[mp_prefixed[0]].notna().sum()
    logger.info(
        "MoneyPuck %s: %d/%d rows matched (id=%d, name-fallback=%d, unmatched=%d)",
        entity, total, len(out), id_matched, name_matched, len(out) - total,
    )
    return out.drop(columns=["_df_pid", "_df_year", "_df_name"])


# --------------------------------------------------------------------------- feature selection

def select_feature_columns(df: pd.DataFrame, target_cols: list[str],
                           allow_contemporaneous_team: bool = False,
                           max_missing_frac: float = 0.70) -> list[str]:
    """Allowlist of leakage-safe predictor columns (see module docstring for the rule)."""
    team_allow = set(identify_team_features(df) + DERIVED_TEAM_COLS)

    def allowed(col: str) -> bool:
        if col in target_cols:
            return False
        low = col.lower()
        if "lag" in low or "covid" in low:
            return True
        if any(m in col for m in _ENGINEERED_MARKERS):
            return True
        if allow_contemporaneous_team and col in team_allow:
            return True
        return False

    cols = []
    for col in df.columns:
        if not allowed(col):
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().mean() < 0.5:
                continue
        if df[col].isna().mean() > max_missing_frac:
            continue
        cols.append(col)
    return cols


def build_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str,
             core_lag_prefixes: tuple[str, ...] = ("goals_lag", "assists_lag", "games_played")
             ) -> tuple[pd.DataFrame, pd.Series]:
    """Materialize a numeric feature matrix + target, dropping rows lacking core lag history."""
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(df[target_col], errors="coerce")
    core = [c for c in X.columns if any(c.startswith(p) for p in core_lag_prefixes)]
    if core:
        drop = X[core].isna().all(axis=1) | y.isna()
    else:
        drop = X.isna().all(axis=1) | y.isna()
    X, y = X[~drop].fillna(0), y[~drop]
    return X, y
