"""Pool roster rules and snake-draft pick math for the 9-team draft.

Bench/IR slots are substitutable into the active lineup twice a week (unlimited for injuries),
generous enough that every drafted player is treated as contributing roughly full value — so
"cap" below is starters + bench combined, not starters alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from common.config import SeasonConfig

POSITIONS = ("forward", "defense", "goalie")


@dataclass(frozen=True)
class DraftConfig:
    num_teams: int
    rounds: int
    caps: dict[str, int]          # per-team draft cap by position
    league_totals: dict[str, int]  # caps * num_teams
    opponent_temperature: float
    mc_sims: int


def load_draft_config(cfg: SeasonConfig) -> DraftConfig:
    raw = cfg.raw.get("draft", {})
    num_teams = raw.get("num_teams", 9)
    roster = raw.get("roster", {
        "forward": {"starters": 7, "bench": 3},
        "defense": {"starters": 3, "bench": 2},
        "goalie": {"starters": 1, "bench": 1},
    })
    caps = {pos: roster[pos]["starters"] + roster[pos]["bench"] for pos in POSITIONS}
    rounds = sum(caps.values())
    return DraftConfig(
        num_teams=num_teams,
        rounds=rounds,
        caps=caps,
        league_totals={pos: caps[pos] * num_teams for pos in POSITIONS},
        opponent_temperature=raw.get("opponent_temperature", 1.5),
        mc_sims=raw.get("mc_sims", 1500),
    )


def team_at_pick(pick: int, num_teams: int) -> int:
    """1-indexed overall pick -> 1-indexed team slot, snaking each round."""
    round_no = (pick - 1) // num_teams + 1
    pos_in_round = (pick - 1) % num_teams + 1
    return pos_in_round if round_no % 2 == 1 else num_teams + 1 - pos_in_round


def picks_for_slot(slot: int, num_teams: int, rounds: int) -> list[int]:
    """All overall pick numbers belonging to `slot` (1..num_teams) across every round."""
    return [p for p in range(1, num_teams * rounds + 1) if team_at_pick(p, num_teams) == slot]
