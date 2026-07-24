"""Cross-source data reconciliation: NHL API vs MoneyPuck (regular season only).

Joins both sources on player_id + season start year, then compares the counting stats
that both sources report independently. Produces a summary CSV (one row per stat pair)
and an outlier CSV (player-seasons where the two sources disagree by more than a threshold).

Usage (from repo root):
    python3 -c "
    from common.config import load_season_config
    from common.diagnostics.reconcile import run_reconciliation
    run_reconciliation(load_season_config('2026-27'))
    "

Or add --stage reconcile to run_season.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------- skater stat pairs
# (nhl_col, mp_col, mp_derived_fn, display_name, outlier_threshold)
# mp_derived_fn: lambda(mp_row) → value, or None if mp_col is direct
_SKATER_PAIRS = [
    ("games_played_player", "games_played",         None,                                         "games_played",  3),
    ("goals",               "I_F_goals",             None,                                         "goals",         5),
    ("assists",             None,                    lambda mp: mp["mp_I_F_primaryAssists"] + mp["mp_I_F_secondaryAssists"], "assists", 5),
    ("points_player",       "I_F_points",            None,                                         "points",        5),
    ("shots",               "I_F_shotsOnGoal",       None,                                         "shots_on_goal", 8),
]

# ----------------------------------------------------------------------- goalie stat pairs
_GOALIE_PAIRS = [
    ("games_played_player", "games_played",      None,  "games_played",   3),
    ("goals_against_player","goals",             None,  "goals_against",  3),
    ("shots_against",       "ongoal",            None,  "shots_against",  8),
]


def _load_and_filter_mp(path: Path, entity: str) -> pd.DataFrame:
    mp = pd.read_csv(path)
    if "situation" in mp.columns:
        pre = len(mp)
        mp = mp[mp["situation"] == "all"].copy()
        logger.info("MP %s: filtered %d → %d rows (situation=='all')", entity, pre, len(mp))
    else:
        logger.warning("MP %s: no 'situation' column — using all rows", entity)
    mp["_mp_year"] = pd.to_numeric(mp["season"], errors="coerce")
    mp["_mp_pid"] = pd.to_numeric(mp["playerId"], errors="coerce")
    return mp


def _load_nhl(path: Path) -> pd.DataFrame:
    nhl = pd.read_csv(path)
    nhl["_nhl_year"] = pd.to_numeric(nhl["season"], errors="coerce") // 10000
    nhl["_nhl_pid"] = pd.to_numeric(nhl["player_id"], errors="coerce")
    # Verify regular season only: NHL API season format 20XXYYYY — no separate playoff indicator
    # in the processed file; we trust the scraper fetches regular-season only.
    return nhl


def _compare_pair(joined: pd.DataFrame, nhl_col: str, mp_col, mp_fn, display: str,
                  threshold: int) -> tuple[dict, pd.DataFrame]:
    """Compute comparison stats for one column pair. Returns (summary_row, outlier_df)."""
    nhl_vals = pd.to_numeric(joined[f"nhl_{nhl_col}"], errors="coerce")
    if mp_fn is not None:
        # derived column — apply lambda to the joined frame which has mp_ prefixed cols
        try:
            mp_vals = mp_fn(joined)
        except KeyError as e:
            logger.warning("reconcile: derived column missing: %s; skipping %s", e, display)
            return {}, pd.DataFrame()
    else:
        col = f"mp_{mp_col}"
        if col not in joined.columns:
            logger.warning("reconcile: %s not in joined frame; skipping", col)
            return {}, pd.DataFrame()
        mp_vals = pd.to_numeric(joined[col], errors="coerce")

    mask = nhl_vals.notna() & mp_vals.notna()
    if mask.sum() < 10:
        logger.warning("reconcile: too few matched rows for %s (%d); skipping", display, mask.sum())
        return {}, pd.DataFrame()

    a, b = nhl_vals[mask], mp_vals[mask]
    diff = (a - b).abs()
    r, _ = scipy_stats.pearsonr(a, b)
    summary = {
        "stat": display,
        "nhl_col": nhl_col,
        "mp_col": mp_col if mp_col else "(derived)",
        "n_matched": int(mask.sum()),
        "pearson_r": round(r, 4),
        "mean_abs_diff": round(diff.mean(), 3),
        "median_abs_diff": round(diff.median(), 3),
        "pct_within_1": round((diff <= 1).mean() * 100, 1),
        "pct_within_5": round((diff <= 5).mean() * 100, 1),
        "max_abs_diff": round(diff.max(), 1),
    }

    outliers_idx = mask & (diff > threshold)
    outlier_df = pd.DataFrame({
        "entity": "",  # filled by caller
        "stat": display,
        "player_name": joined.loc[outliers_idx, "player_name"].values,
        "season_year": joined.loc[outliers_idx, "season_year"].values,
        "nhl_team": joined.loc[outliers_idx, "nhl_team"].values,
        f"nhl_{display}": a[outliers_idx].values,
        f"mp_{display}": b[outliers_idx].values,
        "abs_diff": diff[outliers_idx].values,
    }).sort_values("abs_diff", ascending=False).reset_index(drop=True)
    return summary, outlier_df


def _reconcile_entity(nhl_path: Path, mp_path: Path, entity: str,
                      pairs: list) -> tuple[list[dict], pd.DataFrame]:
    if not nhl_path.exists():
        logger.warning("reconcile: NHL %s file not found: %s", entity, nhl_path)
        return [], pd.DataFrame()
    if not mp_path.exists():
        logger.warning("reconcile: MP %s file not found: %s", entity, mp_path)
        return [], pd.DataFrame()

    nhl = _load_nhl(nhl_path)
    mp = _load_and_filter_mp(mp_path, entity)

    # Build joinable MP frame: id + year + all stat columns used in pairs
    mp_stat_cols = []
    for _, mp_col, mp_fn, _, _ in pairs:
        if mp_col and mp_col in mp.columns:
            mp_stat_cols.append(mp_col)
        elif mp_fn is not None:
            # derived from primaryAssists / secondaryAssists
            for extra in ("I_F_primaryAssists", "I_F_secondaryAssists"):
                if extra in mp.columns and extra not in mp_stat_cols:
                    mp_stat_cols.append(extra)
    mp_sub = mp[["_mp_pid", "_mp_year"] + mp_stat_cols].drop_duplicates(["_mp_pid", "_mp_year"])
    mp_sub = mp_sub.rename(columns={c: f"mp_{c}" for c in mp_stat_cols})

    nhl_needed = list({nhl_col for nhl_col, *_ in pairs} | {"player_name", "team_abbrev"})
    nhl_needed = [c for c in nhl_needed if c in nhl.columns]
    nhl_sub = nhl[["_nhl_pid", "_nhl_year"] + nhl_needed].copy()
    nhl_sub = nhl_sub.rename(columns={c: f"nhl_{c}" for c in nhl_needed})

    joined = nhl_sub.merge(
        mp_sub.rename(columns={"_mp_pid": "_nhl_pid", "_mp_year": "_nhl_year"}),
        on=["_nhl_pid", "_nhl_year"], how="inner",
    )
    joined["player_name"] = joined["nhl_player_name"] if "nhl_player_name" in joined.columns else ""
    joined["season_year"] = joined["_nhl_year"]
    joined["nhl_team"] = joined["nhl_team_abbrev"] if "nhl_team_abbrev" in joined.columns else ""

    logger.info("reconcile: %s — %d rows after inner join (id+year)", entity, len(joined))

    summaries, all_outliers = [], []
    for nhl_col, mp_col, mp_fn, display, threshold in pairs:
        summary, outlier_df = _compare_pair(joined, nhl_col, mp_col, mp_fn, display, threshold)
        if summary:
            summaries.append(summary)
        if not outlier_df.empty:
            all_outliers.append(outlier_df)

    return summaries, pd.concat(all_outliers, ignore_index=True) if all_outliers else pd.DataFrame()


def run_reconciliation(cfg: SeasonConfig) -> None:
    out_dir = cfg.season_dir / "results" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries: list[dict] = []
    all_outliers: list[pd.DataFrame] = []

    logger.info("=== Source reconciliation: NHL API vs MoneyPuck ===")
    for entity, nhl_file, pairs in [
        ("skaters", "skater_team_data.csv", _SKATER_PAIRS),
        ("goalies", "goalie_team_data.csv", _GOALIE_PAIRS),
    ]:
        nhl_path = cfg.processed_dir / nhl_file
        mp_path = cfg.raw_dir / "moneypuck" / f"moneypuck_{entity}.csv"
        summaries, outliers = _reconcile_entity(nhl_path, mp_path, entity, pairs)
        for s in summaries:
            s["entity"] = entity
        all_summaries.extend(summaries)
        if not outliers.empty:
            outliers["entity"] = entity
            all_outliers.append(outliers)

    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)[
            ["entity", "stat", "nhl_col", "mp_col", "n_matched",
             "pearson_r", "mean_abs_diff", "median_abs_diff",
             "pct_within_1", "pct_within_5", "max_abs_diff"]
        ]
        out_csv = out_dir / "source_reconciliation.csv"
        summary_df.to_csv(out_csv, index=False)
        logger.info("Wrote reconciliation summary: %s", out_csv)
        print("\n=== Source reconciliation summary ===")
        print(summary_df.to_string(index=False))

    if all_outliers:
        outlier_df = pd.concat(all_outliers, ignore_index=True)
        out_csv = out_dir / "source_reconciliation_outliers.csv"
        outlier_df.to_csv(out_csv, index=False)
        logger.info("Wrote %d outlier rows: %s", len(outlier_df), out_csv)
        print(f"\n=== {len(outlier_df)} outlier player-seasons (diff > threshold) ===")
        print(outlier_df.head(20).to_string(index=False))
    else:
        print("\nNo outliers above threshold — sources agree well.")
