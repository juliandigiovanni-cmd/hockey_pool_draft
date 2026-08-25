"""Writes the draft-strategy outputs: value/scarcity curves, a policy comparison, and a
round-by-round recommended position-priority table for every draft slot (1..num_teams)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from common.config import SeasonConfig
from common.draft.historical_value_curves import ValueCurves
from common.draft.pool_structure import POSITIONS, DraftConfig
from common.draft.strategy_sim import PolicyResult, best_policy, simulate_slot

logger = logging.getLogger(__name__)


def _draft_dir(cfg: SeasonConfig) -> Path:
    d = cfg.results_dir / "draft"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_value_curves(cfg: SeasonConfig, curves: ValueCurves, dcfg: DraftConfig) -> None:
    max_rank = max(dcfg.league_totals.values()) + 10
    rows = []
    for rank in range(1, max_rank + 1):
        row = {"rank": rank}
        for pos in POSITIONS:
            if rank <= dcfg.league_totals[pos] + 10:
                row[pos] = curves.value(pos, rank)
        rows.append(row)
    pd.DataFrame(rows).to_csv(_draft_dir(cfg) / "value_curves.csv", index=False)


def _round_table(dcfg: DraftConfig, my_slot: int, result: PolicyResult) -> pd.DataFrame:
    from common.draft.pool_structure import picks_for_slot
    picks = picks_for_slot(my_slot, dcfg.num_teams, dcfg.rounds)
    rows = []
    for round_no, (pick_no, dist) in enumerate(zip(picks, result.round_distribution), start=1):
        top_pos = max(dist, key=dist.get) if dist else None
        rows.append({
            "round": round_no,
            "overall_pick": pick_no,
            "recommended_position": top_pos,
            "confidence_pct": round(dist.get(top_pos, 0) * 100, 1) if top_pos else None,
            "forward_pct": round(dist.get("forward", 0) * 100, 1),
            "defense_pct": round(dist.get("defense", 0) * 100, 1),
            "goalie_pct": round(dist.get("goalie", 0) * 100, 1),
        })
    return pd.DataFrame(rows)


def _scarcity_takeaways(curves: ValueCurves, dcfg: DraftConfig) -> list[str]:
    """Two distinct scarcity signals per position: (1) how much value is lost across the whole
    draftable range (top pick vs the very last one anyone in the league drafts), and (2) the
    "cliff" just past that cutoff (last drafted vs 15 spots later) — the more decision-relevant
    signal for urgency, since it captures how much worse things get if you're one round late,
    not just the overall range."""
    lines = []
    for pos in POSITIONS:
        n = dcfg.league_totals[pos]
        top = curves.value(pos, 1)
        last = curves.value(pos, n)
        beyond = curves.value(pos, n + 15)
        range_drop_pct = (top - last) / top * 100 if top else 0
        cliff_drop_pct = (last - beyond) / last * 100 if last else 0
        urgency = "high — being late costs a lot" if cliff_drop_pct > 25 else "low — a fallback option remains nearby"
        lines.append(f"{pos.capitalize()}: value drops {range_drop_pct:.0f}% across the whole "
                     f"draftable range (#1 to #{n}); missing-out risk if you wait past #{n} is "
                     f"{urgency} ({cliff_drop_pct:.0f}% further drop by #{n + 15}).")
    return lines


def generate_report(cfg: SeasonConfig, curves: ValueCurves, dcfg: DraftConfig,
                    cross_check: dict[str, float]) -> None:
    _write_value_curves(cfg, curves, dcfg)

    policy_rows = []
    winners = {}
    for slot in range(1, dcfg.num_teams + 1):
        results = simulate_slot(curves, dcfg, slot)
        for r in results:
            policy_rows.append({"my_slot": slot, "policy": r.policy,
                               "avg_total_value": round(r.avg_total_value, 1)})
        winner = best_policy(results)
        winners[slot] = winner
        _round_table(dcfg, slot, winner).to_csv(
            _draft_dir(cfg) / f"round_priority_slot_{slot}.csv", index=False)

    pd.DataFrame(policy_rows).to_csv(_draft_dir(cfg) / "policy_comparison.csv", index=False)

    lines = [f"DRAFT POSITION-ORDER STRATEGY — {cfg.season}", "=" * 60, "",
            "Historical value-curve cross-check vs current-season projections (normalized-shape Pearson r):"]
    for pos, r in cross_check.items():
        lines.append(f"  {pos}: {r:.2f}")
    lines += ["", "Scarcity summary (based on 18-season aggregated value curves):"]
    lines += [f"  {l}" for l in _scarcity_takeaways(curves, dcfg)]
    lines += ["", "Best-performing policy by draft slot:"]
    for slot, winner in winners.items():
        lines.append(f"  Slot {slot}: {winner.policy}  (avg team value {winner.avg_total_value:.1f})")
    lines += ["", "Round-by-round tables written per slot: round_priority_slot_<1-9>.csv",
             "Full value/scarcity curves: value_curves.csv",
             "Policy comparison (all slots x policies): policy_comparison.csv"]

    (_draft_dir(cfg) / "draft_strategy_report.txt").write_text("\n".join(lines))
    logger.info("Wrote draft strategy report to %s", _draft_dir(cfg))
