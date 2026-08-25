#!/usr/bin/env python3
"""Live, player-level draft-day assistant — tracks real picks as they happen during an actual
draft (yours and every opponent's, entered by hand — this pool's draft is manual/verbal, there's
no platform to pull picks from) and recommends who to take on your turn.

Distinct from draft_strategy.py, which only recommends a position ORDER (built from historical
simulation, before the draft). This tool recommends actual PLAYERS, live, using the current
season's real predicted rankings and reusing draft_strategy's winning position-order policy for
your slot (see common/draft/strategy_sim.py's POLICIES, reused unchanged here against live data
instead of a Monte Carlo simulation).

Usage:
    python live_draft.py --season 2026-27 --my-slot 4
    python live_draft.py --season 2026-27 --my-slot 4 --top-n 10
    python live_draft.py --season 2026-27 --my-slot 4 --reset   # discard any in-progress draft

State is saved after every pick to results/draft/live_draft_state.json — re-running the same
command (without --reset) resumes an in-progress draft exactly where it left off.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from common.config import load_season_config
from common.draft import live_repl
from common.draft.pool_structure import load_draft_config

logger = logging.getLogger(__name__)


def _load_rankings(cfg) -> pd.DataFrame:
    path = cfg.results_dir / f"finalpool_{cfg.season}_overall_rankings.csv"
    if not path.exists():
        sys.exit(
            f"No rankings found at {path}.\n"
            f"Run the prediction pipeline first: "
            f"python run_season.py --season {cfg.season} --stage predict"
        )
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", required=True, help="Season folder name, e.g. 2026-27")
    parser.add_argument("--my-slot", type=int, required=True,
                        help="Your draft slot (1-num_teams) — determines whose turn "
                             "recommendations are shown for")
    parser.add_argument("--top-n", type=int, default=8,
                        help="How many top available players to show per position (default 8)")
    parser.add_argument("--reset", action="store_true",
                        help="Ignore any in-progress draft state and start fresh")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_season_config(args.season)
    dcfg = load_draft_config(cfg)

    if not (1 <= args.my_slot <= dcfg.num_teams):
        sys.exit(f"--my-slot must be between 1 and {dcfg.num_teams} for this pool.")

    rankings = _load_rankings(cfg)
    state_path = live_repl.default_state_path(cfg)

    if args.reset and state_path.exists():
        state_path.unlink()

    if state_path.exists():
        session = live_repl.LiveDraftSession.resume(cfg, dcfg, rankings, state_path, args.top_n)
        if session.my_slot != args.my_slot:
            print(f"WARNING: saved draft state is for slot {session.my_slot}, but you passed "
                 f"--my-slot {args.my_slot}.")
            confirm = input("Continue using the SAVED slot? [y/N] ").strip().lower()
            if confirm != "y":
                sys.exit("Aborted. Re-run with --reset to start a new draft for the slot you "
                         "specified, or without --my-slot mismatch to resume the saved one.")
        else:
            print(f"Resuming in-progress draft ({len(session.log)} picks already made).")
    else:
        policy_name, policy_reason = live_repl.load_policy_for_slot(cfg, args.my_slot)
        session = live_repl.LiveDraftSession.new(cfg, dcfg, rankings, args.my_slot,
                                                 policy_name, policy_reason, args.top_n,
                                                 state_path)

    live_repl.run(session)


if __name__ == "__main__":
    main()
