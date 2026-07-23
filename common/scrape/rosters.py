"""Current-team lookup — replaces last year's hardcoded per-season trade overrides.

The NHL API doesn't expose a "team roster as of an arbitrary past date" endpoint, so this
pulls each team's live current roster. Run the scrape stage close to draft day (per
config.roster_as_of_date) so the snapshot reflects the latest offseason trades, free-agent
signings, and waiver claims before generating predictions.
"""

from __future__ import annotations

import logging

import pandas as pd

from common.config import SeasonConfig
from common.scrape.nhl_api import NHLAPIClient, RAW_FILES, WEB_API_BASE

logger = logging.getLogger(__name__)

CURRENT_ROSTERS_FILE = "nhl_current_rosters.csv"

_ROSTER_GROUPS = {"forwards": None, "defensemen": "D", "goalies": "G"}


def get_current_roster(client: NHLAPIClient, team_abbrev: str) -> pd.DataFrame:
    data = client._get(f"{WEB_API_BASE}/v1/roster/{team_abbrev}/current")
    if not data:
        return pd.DataFrame()
    rows = []
    for group, fallback_position in _ROSTER_GROUPS.items():
        for p in data.get(group, []):
            first = p.get("firstName", {}).get("default", "")
            last = p.get("lastName", {}).get("default", "")
            rows.append(
                {
                    "player_id": p.get("id"),
                    "player_name": f"{first} {last}".strip(),
                    "team_abbrev": team_abbrev,
                    "position": fallback_position or p.get("positionCode"),
                }
            )
    return pd.DataFrame(rows)


def _active_team_abbrevs(cfg: SeasonConfig) -> list[str]:
    """Teams that played in the most recently completed season = teams that exist now.

    nhl_teams.csv is the full all-time franchise list (every code the NHL API has ever
    returned, including defunct/relocated ones like QUE/ATL/HFD) — fetching a "current
    roster" for those always 404s, so restrict to teams present in the latest season of
    nhl_team_stats.csv instead.
    """
    teams_path = cfg.raw_dir / RAW_FILES["teams"]
    team_stats_path = cfg.raw_dir / RAW_FILES["team_stats"]
    if not teams_path.exists() or not team_stats_path.exists():
        raise FileNotFoundError(f"Run the scrape stage first — missing {teams_path} or {team_stats_path}")

    teams_df = pd.read_csv(teams_path)
    team_stats = pd.read_csv(team_stats_path, dtype={"season": str})
    latest_season = team_stats["season"].max()
    active_ids = team_stats.loc[team_stats["season"] == latest_season, "team_id"].unique()
    return teams_df.loc[teams_df["team_id"].isin(active_ids), "team_abbrev"].dropna().unique().tolist()


def update_current_rosters(cfg: SeasonConfig, client: NHLAPIClient | None = None) -> pd.DataFrame:
    """Fetch and save each currently-active team's current roster as this season's
    team-assignment source."""
    client = client or NHLAPIClient()
    rosters = [
        get_current_roster(client, abbrev)
        for abbrev in _active_team_abbrevs(cfg)
    ]
    rosters_df = pd.concat([r for r in rosters if not r.empty], ignore_index=True)

    out_path = cfg.raw_dir / CURRENT_ROSTERS_FILE
    rosters_df.to_csv(out_path, index=False)
    logger.info(
        f"Saved current rosters for {rosters_df['player_id'].nunique()} players to {out_path}"
    )
    return rosters_df


def apply_current_team_overrides(
    df: pd.DataFrame,
    cfg: SeasonConfig,
    player_id_col: str = "player_id",
    player_name_col: str = "player_name",
    team_col: str = "team_abbrev",
) -> pd.DataFrame:
    """Overrides `team_col` with each player's current team, for prediction-time rows.

    Priority: current roster snapshot (by player_id) > config.trade_overrides (by name,
    manual fallback only) > whatever team the player's most recent stats row already has.
    """
    df = df.copy()
    rosters_path = cfg.raw_dir / CURRENT_ROSTERS_FILE
    if rosters_path.exists():
        rosters_df = pd.read_csv(rosters_path)
        team_by_player_id = rosters_df.set_index("player_id")["team_abbrev"]
        mapped = df[player_id_col].map(team_by_player_id)
        df[team_col] = mapped.combine_first(df[team_col])
    else:
        logger.warning(
            "No current roster snapshot found — run the roster update step first; "
            "falling back to trade_overrides / last known team in the meantime"
        )

    for override in cfg.trade_overrides:
        mask = df[player_name_col] == override["player"]
        df.loc[mask, team_col] = override["team"]

    return df
