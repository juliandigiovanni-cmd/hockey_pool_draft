"""Merges scraped skater/goalie stats with team stats into model-ready CSVs.

Migrated from 2025-26/data_merge.py: paths now come from SeasonConfig instead of being
resolved relative to the script's own file location, and the never-called
`standardize_team_ids_after_merge` + hardcoded Utah/Arizona-specific debug prints from
the original are dropped in favor of a generic merge-rate check that works for any
franchise relocation.
"""

from __future__ import annotations

import logging

import pandas as pd

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

REQUIRED_FILES = {
    "skater": "nhl_skater_stats.csv",
    "goalie": "nhl_goalie_stats.csv",
    "team": "nhl_team_stats.csv",
    "tcode": "nhl_teams.csv",
}


def load_raw_data(cfg: SeasonConfig) -> dict[str, pd.DataFrame]:
    data = {}
    for key, filename in REQUIRED_FILES.items():
        path = cfg.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required raw file: {path} — run the scrape stage first")
        data[key] = pd.read_csv(path)
        logger.info(f"Loaded {filename}: {len(data[key]):,} rows")
    return data


def prepare_team_data(df_team: pd.DataFrame, df_tcode: pd.DataFrame) -> pd.DataFrame:
    """Attach team_abbrev/franchise_id to team stats and dedupe team-seasons."""
    df = df_team.merge(
        df_tcode[["team_id", "team_abbrev", "franchise_id"]], on="team_id", how="left"
    )
    before = len(df)
    df = df.drop_duplicates(subset=["team_abbrev", "season"])
    if len(df) != before:
        logger.info(f"Removed {before - len(df)} duplicate team-season rows")
    logger.info(f"Team data prepared: {len(df):,} unique team-seasons")
    return df


def extract_first_team(df_player: pd.DataFrame) -> pd.DataFrame:
    """For players traded mid-season (team_abbrev like 'BOS,NYR'), keep only the first team."""
    df = df_player.copy()
    multi_team_count = df["team_abbrev"].str.contains(",", na=False).sum()
    if multi_team_count:
        logger.info(f"{multi_team_count:,} player-seasons had multiple teams — keeping first team only")
    df["team_abbrev"] = df["team_abbrev"].str.split(",").str[0].str.strip()
    return df


def merge_player_team_data(
    df_player: pd.DataFrame, df_team: pd.DataFrame, player_type: str
) -> pd.DataFrame:
    df_merged = df_player.merge(
        df_team, on=["team_abbrev", "season"], how="left", suffixes=("_player", "_team")
    )
    merge_rate = df_merged["team_id"].notna().mean() * 100
    logger.info(f"{player_type} merge success rate: {merge_rate:.1f}%")
    if merge_rate < 90:
        failed = df_merged.loc[df_merged["team_id"].isna(), ["team_abbrev", "season"]].drop_duplicates()
        logger.warning(f"Low merge rate for {player_type}; unmatched team/season pairs:\n{failed.head(10)}")
    return df_merged


def create_synthetic_team_ids(df_merged: pd.DataFrame) -> pd.DataFrame:
    """Franchises that relocated (e.g. Arizona -> Utah) get multiple team_ids across
    seasons; collapse each franchise to a single synthetic team_id for consistency."""
    if "franchise_id" not in df_merged.columns:
        return df_merged

    franchise_team_counts = df_merged.groupby("franchise_id")["team_id"].nunique()
    multi_team_franchises = franchise_team_counts[franchise_team_counts > 1].index.tolist()
    if not multi_team_franchises:
        return df_merged

    df = df_merged.copy()
    df["team_id_original"] = df["team_id"]
    for i, franchise_id in enumerate(multi_team_franchises, start=1000):
        if pd.isna(franchise_id):
            continue
        mask = df["franchise_id"] == franchise_id
        df.loc[mask, "team_id"] = i
    logger.info(f"Collapsed {len(multi_team_franchises)} relocated franchise(s) to synthetic team_ids")
    return df


def clean_and_merge(cfg: SeasonConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs the full clean/merge pipeline and writes skater_team_data.csv / goalie_team_data.csv."""
    raw = load_raw_data(cfg)
    team_data = prepare_team_data(raw["team"], raw["tcode"])

    skater = create_synthetic_team_ids(
        merge_player_team_data(extract_first_team(raw["skater"]), team_data, "skater")
    )
    goalie = create_synthetic_team_ids(
        merge_player_team_data(extract_first_team(raw["goalie"]), team_data, "goalie")
    )

    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    skater.to_csv(cfg.processed_dir / "skater_team_data.csv", index=False)
    goalie.to_csv(cfg.processed_dir / "goalie_team_data.csv", index=False)
    logger.info(
        f"Saved skater_team_data.csv ({len(skater):,} rows) and "
        f"goalie_team_data.csv ({len(goalie):,} rows) to {cfg.processed_dir}"
    )
    return skater, goalie
