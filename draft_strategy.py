#!/usr/bin/env python3
"""Position-order draft strategy tool (not a live player-by-player draft assistant — see
PROJECT.md's deferred "P6 - Interactive draft-day tool" milestone for that).

Builds F/D/G value curves from 18 seasons (2008-09..2025-26) of real historical stats scored
under the pool's rules, then Monte Carlo-simulates the 9-team snake draft to compare candidate
position-ordering strategies and recommend the best one per draft slot.

Usage:
    python draft_strategy.py --season 2026-27
    python draft_strategy.py --season 2026-27 --my-slot 4
"""

from __future__ import annotations

import argparse
import logging

from common.config import load_season_config
from common.draft.historical_scoring import load_historical_pool_points
from common.draft.historical_value_curves import (
    ValueCurves,
    build_value_curves,
    cross_check_vs_current_season,
)
from common.draft.pool_structure import load_draft_config
from common.draft.report import generate_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help="Season folder name, e.g. 2026-27")
    parser.add_argument("--my-slot", type=int, default=None,
                        help="Restrict console output to one draft slot (1-9); "
                             "the full report always covers every slot")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_season_config(args.season)
    dcfg = load_draft_config(cfg)

    logging.info("Scoring 18 seasons of historical stats under the pool's rules...")
    scored = load_historical_pool_points(cfg)

    logging.info("Building value curves...")
    curves = ValueCurves(build_value_curves(scored))
    cross_check = cross_check_vs_current_season(cfg, curves.curves)

    logging.info("Simulating draft strategies for all %d slots (%d sims/policy/slot)...",
                dcfg.num_teams, dcfg.mc_sims)
    generate_report(cfg, curves, dcfg, cross_check)

    report_path = cfg.results_dir / "draft" / "draft_strategy_report.txt"
    print(report_path.read_text())
    if args.my_slot:
        table_path = cfg.results_dir / "draft" / f"round_priority_slot_{args.my_slot}.csv"
        print(f"\nRound-by-round table for slot {args.my_slot}: {table_path}")


if __name__ == "__main__":
    main()
