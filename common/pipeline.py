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


def scrape(cfg: SeasonConfig) -> None:
    client = NHLAPIClient()
    update_raw_data(cfg, client)
    update_current_rosters(cfg, client)
    update_game_logs(cfg, client)
    update_moneypuck_data(cfg)


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
