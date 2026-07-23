"""Loads a season's config.yaml and resolves its paths against the repo root."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class SeasonConfig:
    season: str
    season_id: str
    history_start_year: int
    games_per_season: int
    season_dir: Path
    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    results_dir: Path
    plots_dir: Path
    roster_as_of_date: str
    trade_overrides: list[dict]
    lag_years: int
    min_training_year: int | None
    validation: dict
    scoring: dict
    raw: dict = field(repr=False)


def load_season_config(season: str) -> SeasonConfig:
    """season is the folder name under the repo root, e.g. '2026-27'."""
    season_dir = REPO_ROOT / season
    config_path = season_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.yaml found for season '{season}' at {config_path}"
        )

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    paths = raw["paths"]
    return SeasonConfig(
        season=raw["season"],
        season_id=raw["season_id"],
        history_start_year=raw["history_start_year"],
        games_per_season=raw.get("games_per_season", 82),
        season_dir=season_dir,
        raw_dir=season_dir / paths["raw_dir"],
        processed_dir=season_dir / paths["processed_dir"],
        models_dir=season_dir / paths["models_dir"],
        results_dir=season_dir / paths["results_dir"],
        plots_dir=season_dir / paths["plots_dir"],
        roster_as_of_date=raw["roster_as_of_date"],
        trade_overrides=raw.get("trade_overrides") or [],
        lag_years=raw["lag_years"],
        min_training_year=raw.get("min_training_year"),
        validation=raw["validation"],
        scoring=raw["scoring"],
        raw=raw,
    )
