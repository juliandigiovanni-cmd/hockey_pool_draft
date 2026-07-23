#!/usr/bin/env python3
"""CLI entry point for the annual hockey pool pipeline.

Usage:
    python run_season.py --season 2026-27 --stage all
    python run_season.py --season 2026-27 --stage scrape
    python run_season.py --season 2026-27 --stage predict   # reuse persisted models

Next year: copy 2026-27/config.yaml into 2027-28/config.yaml, update season/season_id/
roster_as_of_date, and run with --season 2027-28.
"""

from __future__ import annotations

import argparse
import logging

from common.config import load_season_config
from common.pipeline import run_stage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="Season folder name, e.g. 2026-27")
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "scrape", "clean", "train", "predict"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_season_config(args.season)
    run_stage(cfg, args.stage)


if __name__ == "__main__":
    main()
