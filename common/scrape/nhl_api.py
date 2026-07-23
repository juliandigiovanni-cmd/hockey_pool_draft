"""NHL API client for team/skater/goalie stats and standings.

Migrated from 2025-26/hockey_scrape_v3.2.py: drops the dead legacy statsapi.web.nhl.com
fallback (unused, decommissioned by the NHL), drops game/schedule fetching (never
consumed downstream), and replaces the old "download everything every run, in two
manually-split eras" __main__ script with `update_raw_data`, which only fetches seasons
missing from the existing raw CSVs.
"""

from __future__ import annotations

import logging
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

STATS_API_BASE = "https://api.nhle.com/stats/rest"
WEB_API_BASE = "https://api-web.nhle.com"

RAW_FILES = {
    "teams": "nhl_teams.csv",
    "team_stats": "nhl_team_stats.csv",
    "skater_stats": "nhl_skater_stats.csv",
    "goalie_stats": "nhl_goalie_stats.csv",
    "standings": "nhl_standings.csv",
}


def season_id(start_year: int) -> str:
    """e.g. season_id(2008) -> '20082009'."""
    return f"{start_year}{start_year + 1}"


def season_ids_through(start_year: int, end_year: int) -> list[str]:
    return [season_id(y) for y in range(start_year, end_year + 1)]


class NHLAPIClient:
    def __init__(self, base_delay: float = 0.5):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "hockeyanalytics-pool-model/1.0", "Accept": "application/json"}
        )
        self.base_delay = base_delay

    def _get(self, url: str, params: dict | None = None) -> dict | None:
        time.sleep(self.base_delay)
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            logger.error(f"Request failed for {url}: {e}")
            return None

    def get_teams(self) -> pd.DataFrame:
        data = self._get(f"{STATS_API_BASE}/en/team")
        if not data or not data.get("data"):
            return pd.DataFrame()
        rows = [
            {
                "team_id": t.get("id"),
                "team_abbrev": t.get("triCode"),
                "team_name": t.get("fullName"),
                "franchise_id": t.get("franchiseId"),
            }
            for t in data["data"]
        ]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["franchise_id"] = df["franchise_id"].astype("Int64")
        logger.info(f"Retrieved {len(df)} teams")
        return df

    def get_team_stats(self, season: str) -> pd.DataFrame:
        params = {"cayenneExp": f"seasonId={season} and gameTypeId=2"}
        data = self._get(f"{STATS_API_BASE}/en/team/summary", params)
        if not data or not data.get("data"):
            logger.warning(f"No team stats returned for season {season}")
            return pd.DataFrame()
        rows = [
            {
                "season": season,
                "team_id": t.get("teamId"),
                "team_name": t.get("teamName"),
                "games_played": t.get("gamesPlayed"),
                "wins": t.get("wins"),
                "losses": t.get("losses"),
                "ot_losses": t.get("otLosses"),
                "points": t.get("points"),
                "goals_for": t.get("goalsFor"),
                "goals_against": t.get("goalsAgainst"),
                "goal_diff": t.get("goalDifferential"),
                "points_pct": t.get("pointPct"),
                "pp_pct": t.get("ppPct"),
                "pk_pct": t.get("pkPct"),
                "shots_for_per_game": t.get("shotsForPerGame"),
                "shots_against_per_game": t.get("shotsAgainstPerGame"),
                "face_off_win_pct": t.get("faceoffWinPct"),
            }
            for t in data["data"]
        ]
        return pd.DataFrame(rows)

    def get_player_stats(self, season: str, player_type: str = "skaters") -> pd.DataFrame:
        endpoint = "skater" if player_type == "skaters" else "goalie"
        url = f"{STATS_API_BASE}/en/{endpoint}/summary"
        limit = 100
        start = 0
        rows: list[dict] = []
        while True:
            params = {
                "limit": limit,
                "start": start,
                "cayenneExp": f"seasonId={season} and gameTypeId=2",
            }
            data = self._get(url, params)
            batch = (data or {}).get("data") or []
            if not batch:
                break
            for p in batch:
                if player_type == "skaters":
                    rows.append(
                        {
                            "season": season,
                            "player_id": p.get("playerId"),
                            "player_name": p.get("skaterFullName"),
                            "team_abbrev": p.get("teamAbbrevs"),
                            "position": p.get("positionCode"),
                            "games_played": p.get("gamesPlayed"),
                            "goals": p.get("goals"),
                            "assists": p.get("assists"),
                            "points": p.get("points"),
                            "plus_minus": p.get("plusMinus"),
                            "penalty_minutes": p.get("penaltyMinutes"),
                            "shots": p.get("shots"),
                            "shooting_pct": p.get("shootingPct"),
                            "time_on_ice_per_game": p.get("timeOnIcePerGame"),
                            "face_off_win_pct": p.get("faceoffWinPct"),
                            "pp_goals": p.get("ppGoals"),
                            "pp_points": p.get("ppPoints"),
                            "sh_goals": p.get("shGoals"),
                            "sh_points": p.get("shPoints"),
                            "gw_goals": p.get("gameWinningGoals"),
                            "hits": p.get("hits"),
                            "blocked_shots": p.get("blockedShots"),
                        }
                    )
                else:
                    rows.append(
                        {
                            "season": season,
                            "player_id": p.get("playerId"),
                            "player_name": p.get("goalieFullName"),
                            "team_abbrev": p.get("teamAbbrevs"),
                            "games_played": p.get("gamesPlayed"),
                            "games_started": p.get("gamesStarted"),
                            "wins": p.get("wins"),
                            "losses": p.get("losses"),
                            "ot_losses": p.get("otLosses"),
                            "saves": p.get("saves"),
                            "shots_against": p.get("shotsAgainst"),
                            "save_pct": p.get("savePct"),
                            "goals_against": p.get("goalsAgainst"),
                            "goals_against_avg": p.get("goalsAgainstAverage"),
                            "time_on_ice": p.get("timeOnIce"),
                            "shutouts": p.get("shutouts"),
                            "goals": p.get("goals"),
                            "assists": p.get("assists"),
                            "points": p.get("points"),
                            "penalty_minutes": p.get("penaltyMinutes"),
                        }
                    )
            if len(batch) < limit:
                break
            start += limit
        df = pd.DataFrame(rows)
        logger.info(f"Retrieved {len(df)} {player_type} for season {season}")
        return df

    def _get_standings_final_date(self, season: str) -> str | None:
        data = self._get(f"{WEB_API_BASE}/v1/standings-season")
        if not data or not data.get("seasons"):
            return None
        for entry in data["seasons"]:
            if str(entry.get("id")) == str(season):
                return entry.get("standingsStart")
        return None

    def get_standings(self, season: str) -> pd.DataFrame:
        standings_date = self._get_standings_final_date(season)
        if not standings_date:
            logger.warning(f"Could not resolve a standings date for season {season}")
            return pd.DataFrame()
        data = self._get(f"{WEB_API_BASE}/v1/standings/{standings_date}")
        if not data or not data.get("standings"):
            return pd.DataFrame()
        rows = [
            {
                "season": season,
                "team_id": s.get("teamId"),
                "team_name": s.get("teamName"),
                "team_abbrev": s.get("teamAbbrev"),
                "conference_abbrev": s.get("conferenceAbbrev"),
                "division_abbrev": s.get("divisionAbbrev"),
                "games_played": s.get("gamesPlayed"),
                "wins": s.get("wins"),
                "losses": s.get("losses"),
                "ot_losses": s.get("otLosses"),
                "points": s.get("points"),
                "point_pct": s.get("pointPct"),
                "goals_for": s.get("goalFor"),
                "goals_against": s.get("goalAgainst"),
                "goal_diff": s.get("goalDifferential"),
            }
            for s in data["standings"]
        ]
        return pd.DataFrame(rows)

    def get_player_game_log(self, player_id: int, season: str, game_type: int = 2) -> pd.DataFrame:
        """Per-game log for one player/season — used for rolling-form features."""
        data = self._get(f"{WEB_API_BASE}/v1/player/{player_id}/game-log/{season}/{game_type}")
        game_log = (data or {}).get("gameLog") or []
        if not game_log:
            return pd.DataFrame()
        df = pd.DataFrame(game_log)
        df.insert(0, "player_id", player_id)
        df.insert(1, "season", season)
        return df

    def fetch_seasons(self, seasons: list[str]) -> dict[str, pd.DataFrame]:
        """Fetch team/skater/goalie stats + standings for exactly the given seasons."""
        collected: dict[str, list[pd.DataFrame]] = {
            "team_stats": [],
            "skater_stats": [],
            "goalie_stats": [],
            "standings": [],
        }
        for season in seasons:
            logger.info(f"Fetching season {season}")
            collected["team_stats"].append(self.get_team_stats(season))
            collected["skater_stats"].append(self.get_player_stats(season, "skaters"))
            collected["goalie_stats"].append(self.get_player_stats(season, "goalies"))
            collected["standings"].append(self.get_standings(season))
        return {
            key: pd.concat([df for df in dfs if not df.empty], ignore_index=True)
            if any(not df.empty for df in dfs)
            else pd.DataFrame()
            for key, dfs in collected.items()
        }


def _existing_seasons(path: Path) -> set[str]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, usecols=["season"], dtype={"season": str})
    return set(df["season"].unique())


def _prior_season_name(season: str) -> str | None:
    """'2026-27' -> '2025-26'."""
    m = re.match(r"^(\d{4})-(\d{2})$", season)
    if not m:
        return None
    start = int(m.group(1)) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def seed_from_prior_season(cfg: SeasonConfig) -> None:
    """If this season's raw_dir has no data yet, copy the previous season's raw CSVs forward,
    so update_raw_data only has to fetch the newly-completed season instead of the entire
    history again — without this, every new season folder starts empty and re-downloads
    2008-present from scratch on its first run, every single year.
    """
    if (cfg.raw_dir / RAW_FILES["team_stats"]).exists():
        return
    prior = _prior_season_name(cfg.season)
    if not prior:
        return
    src = cfg.season_dir.parent / prior / "data" / "raw"
    if not (src / RAW_FILES["team_stats"]).exists():
        return
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    for filename in RAW_FILES.values():
        s = src / filename
        if s.exists():
            shutil.copy(s, cfg.raw_dir / filename)
    logger.info(f"Seeded raw data for {cfg.season} from {src}")


def update_raw_data(cfg: SeasonConfig, client: NHLAPIClient | None = None) -> None:
    """Fetch only the seasons missing from cfg.raw_dir, then merge and re-save.

    Teams are re-fetched every call since it's a single cheap request.
    """
    client = client or NHLAPIClient()
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    seed_from_prior_season(cfg)

    current_year = int(cfg.season_id[:4])
    wanted_seasons = season_ids_through(cfg.history_start_year, current_year - 1)

    have_seasons = _existing_seasons(cfg.raw_dir / RAW_FILES["team_stats"])
    missing_seasons = [s for s in wanted_seasons if s not in have_seasons]

    teams_df = client.get_teams()
    if not teams_df.empty:
        teams_df.to_csv(cfg.raw_dir / RAW_FILES["teams"], index=False)

    if not missing_seasons:
        logger.info("Raw data already covers all requested seasons — nothing to fetch")
        return

    logger.info(f"Fetching {len(missing_seasons)} missing season(s): {missing_seasons}")
    new_data = client.fetch_seasons(missing_seasons)

    for key in ("team_stats", "skater_stats", "goalie_stats", "standings"):
        out_path = cfg.raw_dir / RAW_FILES[key]
        new_df = new_data[key]
        if out_path.exists():
            existing_df = pd.read_csv(out_path, dtype={"season": str})
            combined = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined = new_df
        if not combined.empty:
            combined.to_csv(out_path, index=False)
            logger.info(f"Saved {len(combined)} total rows to {out_path}")


def update_game_logs(
    cfg: SeasonConfig, client: NHLAPIClient | None = None, seasons_back: int = 1
) -> None:
    """Per-game logs for rolling-form features, for only the most recent `seasons_back`
    completed season(s) — not a full historical backfill, which would mean one API call
    per player per season across 2008-2025 for marginal benefit over the existing
    season-level lag features. Requires update_raw_data to have already run.
    """
    client = client or NHLAPIClient()
    current_year = int(cfg.season_id[:4])
    seasons = season_ids_through(current_year - seasons_back, current_year - 1)

    skaters = pd.read_csv(cfg.raw_dir / RAW_FILES["skater_stats"], dtype={"season": str})
    goalies = pd.read_csv(cfg.raw_dir / RAW_FILES["goalie_stats"], dtype={"season": str})
    players = pd.concat(
        [skaters[["player_id", "season"]], goalies[["player_id", "season"]]], ignore_index=True
    )
    players = players[players["season"].isin(seasons)].drop_duplicates()

    out_path = cfg.raw_dir / "nhl_game_logs.csv"
    have_pairs: set[tuple[int, str]] = set()
    if out_path.exists():
        existing = pd.read_csv(out_path, usecols=["player_id", "season"], dtype={"season": str})
        have_pairs = set(zip(existing["player_id"], existing["season"]))

    to_fetch = [
        (row.player_id, row.season)
        for row in players.itertuples()
        if (row.player_id, row.season) not in have_pairs
    ]
    if not to_fetch:
        logger.info("Game logs already up to date for the requested seasons")
        return

    logger.info(f"Fetching game logs for {len(to_fetch)} player-seasons")
    new_logs = [client.get_player_game_log(pid, season) for pid, season in to_fetch]
    new_df = pd.concat([df for df in new_logs if not df.empty], ignore_index=True)
    if new_df.empty:
        return

    combined = pd.concat([pd.read_csv(out_path), new_df], ignore_index=True) if out_path.exists() else new_df
    combined.to_csv(out_path, index=False)
    logger.info(f"Saved {len(combined):,} game-log rows to {out_path}")
