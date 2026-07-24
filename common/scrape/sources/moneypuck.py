"""MoneyPuck.com free CSV data client.

Adds expected-goals (xG), shot-quality, and goalie GSAx (goals saved above expected)
features that the raw NHL API stats don't include — no auth, no rate limits, direct
CSV downloads (moneypuck.com/data.htm). Season-summary files are published per
season-start-year at:

    https://moneypuck.com/moneypuck/playerData/seasonSummary/{year}/regular/{entity}.csv

where entity is 'skaters', 'goalies', or 'teams'. Bare `pd.read_csv(url)` uses urllib with
Python's default User-Agent, which Cloudflare (fronting moneypuck.com) resets outright —
confirmed on a real run, not just the earlier dev sandbox. Fetching through `requests` with
a browser-like User-Agent (same fix already used by common/scrape/nhl_api.py against a
different host) avoids that. If moneypuck.com is unreachable for some other reason, this
degrades to an empty result rather than failing the whole pipeline (see update_moneypuck_data).
"""

from __future__ import annotations

import io
import logging
import shutil
import time
from pathlib import Path

import pandas as pd
import requests

from common.config import SeasonConfig
from common.scrape.nhl_api import _prior_season_name

logger = logging.getLogger(__name__)

BASE_URL = "https://moneypuck.com/moneypuck/playerData/seasonSummary"
ENTITIES = ("skaters", "goalies", "teams")

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/csv,*/*",
}


def fetch_season_summary(start_year: int, entity: str, game_type: str = "regular",
                         _retry_delay: float = 15.0) -> pd.DataFrame:
    url = f"{BASE_URL}/{start_year}/{game_type}/{entity}.csv"
    for attempt in range(2):
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            if resp.status_code == 429:
                if attempt == 0:
                    logger.warning(f"MoneyPuck rate-limited for {entity}/{start_year}; retrying after {_retry_delay}s")
                    time.sleep(_retry_delay)
                    continue
                logger.error(f"MoneyPuck rate-limited for {entity}/{start_year} after retry; skipping")
                return pd.DataFrame()
            resp.raise_for_status()
            df = pd.read_csv(io.StringIO(resp.text))
        except Exception as e:
            logger.error(f"Failed to fetch MoneyPuck {entity} for {start_year}: {e}")
            return pd.DataFrame()
        df = df.copy()
        df["season_start_year"] = start_year
        return df
    return pd.DataFrame()


def _existing_years(path: Path) -> set[int]:
    if not path.exists():
        return set()
    return set(pd.read_csv(path, usecols=["season_start_year"])["season_start_year"].unique())


def _seed_from_prior_season(cfg: SeasonConfig, out_dir: Path) -> None:
    """Same idea as nhl_api.seed_from_prior_season: copy last year's MoneyPuck CSVs forward
    so this year only has to fetch the newly-completed season, not the full history again."""
    prior = _prior_season_name(cfg.season)
    if not prior:
        return
    src = cfg.season_dir.parent / prior / "data" / "raw" / "moneypuck"
    if not src.exists():
        return
    for entity in ENTITIES:
        dst_file = out_dir / f"moneypuck_{entity}.csv"
        src_file = src / f"moneypuck_{entity}.csv"
        if src_file.exists() and not dst_file.exists():
            shutil.copy(src_file, dst_file)
            logger.info(f"Seeded MoneyPuck {entity} for {cfg.season} from {src_file}")


def update_moneypuck_data(cfg: SeasonConfig) -> None:
    """Fetch skaters/goalies/teams season-summary CSVs, incrementally by season-start-year."""
    out_dir = cfg.raw_dir / "moneypuck"
    out_dir.mkdir(parents=True, exist_ok=True)
    _seed_from_prior_season(cfg, out_dir)

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
        new_frames = []
        for i, y in enumerate(missing_years):
            if i > 0:
                time.sleep(1.0)
            f = fetch_season_summary(y, entity)
            if not f.empty:
                new_frames.append(f)
        if not new_frames:
            logger.warning(f"MoneyPuck {entity}: no years fetched successfully; skipping this entity")
            continue
        new_df = pd.concat(new_frames, ignore_index=True)

        if out_path.exists():
            combined = pd.concat([pd.read_csv(out_path), new_df], ignore_index=True)
        else:
            combined = new_df
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined):,} MoneyPuck {entity} rows to {out_path}")
        time.sleep(3.0)  # brief pause between entities to avoid rate limits
