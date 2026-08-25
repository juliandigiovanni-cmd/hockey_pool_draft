"""Live draft state: tracks real picks as they happen during an actual draft, and reuses
`strategy_sim`'s position-choice policies (`POLICIES`) unchanged by duck-typing the same
interface `strategy_sim._State` exposes to them — `next_value`, `value_at_offset`, `my_counts`,
`caps`, `remaining_need`, `intervening_picks`. The only semantic swap from the simulation: instead
of looking up a historical value *curve* by rank, `next_value`/`value_at_offset` look up the
actual best-still-available real player's predicted `pool_points` from the current season's
rankings, since in live use we know exactly which real players are already gone.

`eligible()` enforces the starters-before-bench draft mechanic (see `pool_structure.py`): all 11
starter slots (7F/3D/1G) must be filled before any bench slot opens up, for every team.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from common.draft.pool_structure import POSITIONS, DraftConfig, team_at_pick


class LivePool:
    """Wraps a season's overall-rankings DataFrame plus which players have been taken so far."""

    def __init__(self, rankings: pd.DataFrame):
        required = {"player_id", "player_name", "position", "pool_points"}
        missing = required - set(rankings.columns)
        if missing:
            raise ValueError(f"rankings DataFrame missing required columns: {missing}")
        self.rankings = rankings.reset_index(drop=True)
        self.taken: dict[int, int] = {}  # player_id -> overall_pick number taken at

    def is_taken(self, player_id: int) -> bool:
        return player_id in self.taken

    def mark_taken(self, player_id: int, pick_no: int) -> None:
        self.taken[player_id] = pick_no

    def unmark_taken(self, player_id: int) -> None:
        self.taken.pop(player_id, None)

    def available(self, position: str | None = None) -> pd.DataFrame:
        df = self.rankings[~self.rankings["player_id"].isin(self.taken)]
        if position is not None:
            df = df[df["position"] == position]
        return df.sort_values("pool_points", ascending=False)

    def row(self, player_id: int) -> pd.Series:
        match = self.rankings[self.rankings["player_id"] == player_id]
        if match.empty:
            raise KeyError(f"No player with player_id={player_id} in rankings")
        return match.iloc[0]


@dataclass
class _LiveState:
    """Duck-typed analog of `strategy_sim._State` — implements exactly the members
    `strategy_sim.POLICIES` functions call, backed by live `LivePool` data instead of a
    historical value curve."""

    pool: LivePool
    dcfg: DraftConfig
    my_slot: int
    team_counts: dict[int, dict[str, int]] = field(default_factory=dict)
    _pick: int = 1  # next pick number to be made, 1-indexed (matches team_at_pick)

    def __post_init__(self):
        if not self.team_counts:
            self.team_counts = {t: {p: 0 for p in POSITIONS}
                                for t in range(1, self.dcfg.num_teams + 1)}

    @property
    def my_counts(self) -> dict:
        return self.team_counts[self.my_slot]

    @property
    def caps(self) -> dict:
        return self.dcfg.caps

    def next_value(self, position: str) -> float:
        return self.value_at_offset(position, 0)

    def value_at_offset(self, position: str, offset: int) -> float:
        avail = self.pool.available(position)
        if len(avail) == 0:
            return 0.0
        idx = min(max(offset, 0), len(avail) - 1)
        return float(avail.iloc[idx]["pool_points"])

    def remaining_need(self, position: str) -> int:
        return sum(self.caps[position] - c[position] for c in self.team_counts.values())

    def intervening_picks(self) -> int:
        """Opponent picks between the pick about to happen and the user's next turn after it."""
        total_picks = self.dcfg.num_teams * self.dcfg.rounds
        p = self._pick + 1
        while p <= total_picks and team_at_pick(p, self.dcfg.num_teams) != self.my_slot:
            p += 1
        return max(p - self._pick - 1, 0)

    def eligible(self, team: int) -> list[str]:
        """Positions this team may draft next. All 11 starter slots (7F/3D/1G) must be filled,
        in any order/mix among still-open starter categories, before any bench slot opens up —
        e.g. a 2nd goalie is never eligible until the starter F/D/G slots are all full, even
        though total goalie capacity (starter + bench) would otherwise allow it."""
        counts = self.team_counts[team]
        if sum(counts.values()) < sum(self.dcfg.starter_caps.values()):
            return [p for p in POSITIONS if counts[p] < self.dcfg.starter_caps[p]]
        return [p for p in POSITIONS if counts[p] < self.caps[p]]

    def current_team(self) -> int:
        return team_at_pick(self._pick, self.dcfg.num_teams)

    def is_complete(self) -> bool:
        return self._pick > self.dcfg.num_teams * self.dcfg.rounds

    def apply(self, team: int, player_id: int, position: str) -> None:
        if position not in self.eligible(team):
            raise ValueError(f"Team {team} has no open {position} slot (cap already filled)")
        self.team_counts[team][position] += 1
        self.pool.mark_taken(player_id, self._pick)
        self._pick += 1

    def undo_last(self, team: int, player_id: int, position: str) -> None:
        self._pick -= 1
        self.pool.unmark_taken(player_id)
        self.team_counts[team][position] -= 1


@dataclass
class PickEvent:
    overall_pick: int
    team: int
    player_id: int
    player_name: str
    position: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {"overall_pick": self.overall_pick, "team": self.team,
                "player_id": self.player_id, "player_name": self.player_name,
                "position": self.position, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PickEvent":
        return cls(overall_pick=d["overall_pick"], team=d["team"], player_id=d["player_id"],
                  player_name=d["player_name"], position=d["position"],
                  timestamp=d["timestamp"])


def replay(events: list[PickEvent], pool: LivePool, dcfg: DraftConfig, my_slot: int) -> _LiveState:
    """Rebuild a `_LiveState` by replaying a persisted pick log against a fresh `LivePool`.

    This is the only path (besides live `apply()`) that produces state, so persisted and
    in-memory state can never diverge — a resume is just a replay of the same events a live
    session would have applied one at a time.
    """
    state = _LiveState(pool=pool, dcfg=dcfg, my_slot=my_slot)
    for ev in events:
        if ev.overall_pick != state._pick:
            raise ValueError(f"Pick log out of order: expected pick {state._pick}, "
                             f"got {ev.overall_pick}")
        state.apply(ev.team, ev.player_id, ev.position)
    return state
