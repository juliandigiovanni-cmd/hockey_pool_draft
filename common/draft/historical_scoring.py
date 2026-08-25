"""Scores 18 seasons (2008-09..2025-26) of real historical stats under the pool's scoring rules.

Reuses common.models.pool_ranking.compute_pool_points unchanged — it operates on
pred_points_per_game / projected_games / pred_wins_per_game / etc. columns, so each historical
season's actual stats are mapped onto those same names (actual rate standing in for "predicted"
rate) rather than reimplementing the scoring formula.

Season-length normalization: three seasons weren't 82 games (2012-13 lockout: 48; 2019-20 COVID
suspension: games varied by team, ~70 on average; 2020-21 COVID: 56). Per-game rates already
neutralize this for the *rate* half of the formula; the remaining piece is that
`projected_games` is set to NORMALIZED_SEASON_GAMES * (games_played / that season's actual
length) instead of raw games_played. This simultaneously (a) rescales counting production to an
82-game pace and (b) scales the goalie bonus-qualification threshold proportionally — a goalie
who played all 48 games of 2012-13 is treated as having played a "full" season, not benched for
falling short of the fixed 40-game bar meant for an 82-game season.
"""

from __future__ import annotations

import pandas as pd

from common.config import SeasonConfig
from common.features import engineering as fe
from common.models.pool_ranking import compute_pool_points

NORMALIZED_SEASON_GAMES = 82

# Actual NHL regular-season length by year (season-start year, matching fe.add_year's `year`
# column), for seasons that weren't a standard 82-game slate. 2019-20 was suspended mid-schedule
# at varying team-by-team game counts (68-71) and never resumed the regular season; 70 is the
# league-wide average used as a single representative length.
SEASON_LENGTH_OVERRIDES = {2012: 48, 2019: 70, 2020: 56}

_FORWARD_POSITIONS = {"C", "L", "R"}


def _season_length(year: int) -> int:
    return SEASON_LENGTH_OVERRIDES.get(year, NORMALIZED_SEASON_GAMES)


def _rescaled_games(df: pd.DataFrame) -> pd.Series:
    length = df["year"].map(_season_length)
    return NORMALIZED_SEASON_GAMES * (df["games_played"] / length)


def _rate(numerator: pd.Series, games: pd.Series) -> pd.Series:
    return (numerator / games.replace(0, pd.NA)).fillna(0.0)


def _score_season(cfg: SeasonConfig, skater_season: pd.DataFrame,
                  goalie_season: pd.DataFrame) -> dict[str, pd.DataFrame]:
    fwd = skater_season[skater_season["position"].isin(_FORWARD_POSITIONS)].copy()
    fwd["pred_points_per_game"] = _rate(fwd["goals"] + fwd["assists"], fwd["games_played"])
    fwd["projected_games"] = _rescaled_games(fwd)

    dfn = skater_season[skater_season["position"] == "D"].copy()
    dfn["pred_points_per_game"] = _rate(dfn["goals"] + dfn["assists"], dfn["games_played"])
    dfn["pred_plus_minus_per_game"] = _rate(dfn["plus_minus"], dfn["games_played"])
    dfn["projected_games"] = _rescaled_games(dfn)

    goa = goalie_season.copy()
    goa["pred_wins_per_game"] = _rate(goa["wins"], goa["games_played"])
    goa["pred_shutouts_per_game"] = _rate(goa["shutouts"], goa["games_played"])
    goa["pred_gaa"] = goa["goals_against_avg"]
    goa["pred_save_pct"] = goa["save_pct"]
    goa["projected_games"] = _rescaled_games(goa)

    return compute_pool_points(cfg, fwd, dfn, goa)


def load_historical_pool_points(cfg: SeasonConfig) -> dict[str, pd.DataFrame]:
    """Return {'forward'/'defense'/'goalie': DataFrame} of every player-season 2008-09..2025-26,
    scored under the pool's rules and 82-game-pace normalized. Columns include year, season,
    player_id, player_name, pool_points."""
    skater = pd.read_csv(cfg.raw_dir / "nhl_skater_stats.csv")
    goalie = pd.read_csv(cfg.raw_dir / "nhl_goalie_stats.csv")
    skater = fe.add_year(skater[skater["games_played"] > 0])
    goalie = fe.add_year(goalie[goalie["games_played"] > 0])

    scored = {"forward": [], "defense": [], "goalie": []}
    for year, skater_season in skater.groupby("year"):
        goalie_season = goalie[goalie["year"] == year]
        if goalie_season.empty:
            continue
        season_scored = _score_season(cfg, skater_season, goalie_season)
        for pos, df in season_scored.items():
            df = df.copy()
            df["year"] = year
            scored[pos].append(df)

    return {pos: pd.concat(frames, ignore_index=True) for pos, frames in scored.items()}
