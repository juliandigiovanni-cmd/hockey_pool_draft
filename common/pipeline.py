"""Stage functions run_season.py wires into a CLI: scrape -> clean -> train -> predict."""

from __future__ import annotations

import logging

from common.clean.merge import clean_and_merge
from common.config import SeasonConfig
from common.models.pool_ranking import run_pool_ranking
from common.scrape.nhl_api import NHLAPIClient, update_game_logs, update_raw_data
from common.scrape.rosters import update_current_rosters
from common.scrape.sources.moneypuck import update_moneypuck_data

logger = logging.getLogger(__name__)


def _best_effort(name: str, fn, *args) -> None:
    """Enhancement data sources (rosters/game-logs/MoneyPuck) shouldn't take down the whole
    annual run if one of them is unreachable or errors — clean/train/predict don't require them,
    and each already degrades gracefully downstream when its file is missing."""
    try:
        fn(*args)
    except Exception as e:
        logger.warning(f"{name} failed, continuing without it: {e}")


def scrape(cfg: SeasonConfig) -> None:
    client = NHLAPIClient()
    update_raw_data(cfg, client)  # required by clean_and_merge; let failures here propagate
    _best_effort("Current-roster update", update_current_rosters, cfg, client)
    _best_effort("Game-log update", update_game_logs, cfg, client)
    _best_effort("MoneyPuck update", update_moneypuck_data, cfg)


def clean(cfg: SeasonConfig) -> None:
    clean_and_merge(cfg)


def train(cfg: SeasonConfig) -> None:
    run_pool_ranking(cfg, retrain=True)


def predict(cfg: SeasonConfig) -> None:
    run_pool_ranking(cfg, retrain=False)


STAGES = {
    "scrape": scrape,
    "clean": clean,
    "train": train,
    "predict": predict,
}


def run_stage(cfg: SeasonConfig, stage: str) -> None:
    if stage == "all":
        for name in ("scrape", "clean", "train", "predict"):
            logger.info(f"=== Stage: {name} ===")
            STAGES[name](cfg)
        return
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}'. Choose from: all, {', '.join(STAGES)}")
    STAGES[stage](cfg)
