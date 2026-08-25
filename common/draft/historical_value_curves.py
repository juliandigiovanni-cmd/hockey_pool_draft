"""Aggregates 18 seasons of scored historical player-seasons into one value curve per position.

For each season, players are ranked within position by pool_points; the curve is the
equal-weighted average pool_points at each rank across all seasons that had a player at that
rank. Averaging across 18 years smooths single-season noise (a scoring-model change, a weak
goalie class, etc.) out of the scarcity signal, the same "trust a multi-year pattern over one
year" idea the pipeline already applies at a smaller scale (2-3 year lag averaging in
common/features/engineering.py, common/models/pool_ranking.py's 3-year GP window).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from common.config import SeasonConfig

MAX_RANK = 150


def _season_rank_series(df: pd.DataFrame) -> pd.Series:
    ranked = df.sort_values("pool_points", ascending=False).reset_index(drop=True)
    ranked.index = ranked.index + 1
    return ranked["pool_points"]


def build_value_curves(scored: dict[str, pd.DataFrame], max_rank: int = MAX_RANK) -> dict[str, pd.Series]:
    """Return {position: Series indexed by rank (1..max_rank) of avg pool_points across seasons}."""
    curves = {}
    for pos, df in scored.items():
        per_season = [_season_rank_series(g) for _, g in df.groupby("year")]
        wide = pd.concat(per_season, axis=1)
        wide = wide[wide.index <= max_rank]
        curves[pos] = wide.mean(axis=1).sort_index()
    return curves


@dataclass
class ValueCurves:
    curves: dict[str, pd.Series]

    def value(self, position: str, rank: int) -> float:
        """pool_points of the rank-th best player at `position` (1-indexed); floors at the
        curve's last known value once `rank` exceeds available historical depth."""
        s = self.curves[position]
        if rank in s.index:
            v = s.loc[rank]
            if pd.notna(v):
                return float(v)
        valid = s.dropna()
        return float(valid.iloc[-1]) if len(valid) else 0.0


def cross_check_vs_current_season(cfg: SeasonConfig, curves: dict[str, pd.Series]) -> dict[str, float]:
    """Pearson correlation between the historical-average curve's SHAPE (value normalized to a
    fraction of its own #1) and this season's projected ranking's shape, per position, as a
    plausibility check (not blended into the curves). Comparing normalized shape rather than raw
    rank order matters here: both curves are sorted descending by construction, so a plain
    rank-vs-rank (Spearman) correlation would trivially be 1.0 regardless of whether the actual
    decay shapes agree."""
    out = {}
    prefix = f"finalpool_{cfg.season}"
    for pos in curves:
        path = cfg.results_dir / f"{prefix}_{pos}_rankings.csv"
        if not path.exists():
            continue
        current = pd.read_csv(path).sort_values("rank")["pool_points"]
        current.index = range(1, len(current) + 1)
        n = min(len(current), len(curves[pos].dropna()))
        if n < 10 or current.iloc[0] == 0 or curves[pos].iloc[0] == 0:
            continue
        hist_norm = curves[pos].iloc[:n] / curves[pos].iloc[0]
        current_norm = current.iloc[:n] / current.iloc[0]
        out[pos] = float(hist_norm.corr(current_norm, method="pearson"))
    return out
