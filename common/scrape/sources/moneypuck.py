"""MoneyPuck.com free CSV data client.

Adds expected-goals (xG), shot-quality, and goalie GSAx (goals saved above expected)
features that the raw NHL API stats don't include — no auth, no rate limits, direct
CSV downloads (moneypuck.com/data.htm). Season-summary files are published per
season-start-year at:

    https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/{entity}.csv

where entity is 'skaters', 'goalies', or 'teams'. This URL pattern is MoneyPuck's
long-standing public convention (referenced across third-party tools/datasets); this
sandbox's network couldn't reach moneypuck.com directly to do a live check (connection
reset, likely Cloudflare bot protection on this environment's IP) — verify with a real
run of the scrape stage on a normal network before relying on it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
ENTITIES = ("skaters", "goalies", "teams")


def fetch_season_summary(start_year: int, entity: str, game_type: str = "regular") -> pd.DataFrame:
    url = f"{BASE_URL}/{start_year}/{game_type}/{entity}.csv"
    try:
        df = pd.read_csv(url)
    except Exception as e:
        logger.error(f"Failed to fetch MoneyPuck {entity} for {start_year}: {e}")
        return pd.DataFrame()
    df["season_start_year"] = start_year
    return df


def _existing_years(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["season_start_year"])["season_start_year"].unique())


def update_moneypuck_data(cfg: SeasonConfig) -> None:
    """Fetch skaters/goalies/teams season-summary CSVs, incrementally by season-start-year."""
    out_dir = cfg.raw_dir / "moneypuck"
    out_dir.mkdir(parents=True, exist_ok=True)

    current_year = int(cfg.season_id[:4])
    wanted_years = list(range(cfg.history_start_year, current_year))

    for entity in ENTITIES:
        out_path = out_dir / f"moneypuck_{entity}.csv"
        have_years = _existing_years(out_path)
        missing_years = [y for y in wanted_years if y not in have_years]
        if not missing_years:
            logger.info(f"MoneyPuck {entity}: already have all requested seasons")
            continue

        logger.info(f"Fetching MoneyPuck {entity} for years {missing_years}")
        new_frames = [fetch_season_summary(y, entity) for y in missing_years]
        new_df = pd.concat([f for f in new_frames if not f.empty], ignore_index=True)
        if new_df.empty:
            continue

        if out_path.exists():
            combined = pd.concat([pd.read_csv(out_path), new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined):,} MoneyPuck {entity} rows to {out_path}")
