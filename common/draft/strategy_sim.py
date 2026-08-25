"""Monte Carlo simulation over POSITION choices (never specific players) to find which draft
strategy — which position to take at each of the user's own picks — yields the highest total
team value for a 9-team snake draft, given the historical value curves.

Opponents (per the user's confirmed simplification) are modeled as picking the best player
available among positions they still need: softmax over their eligible positions' current
top-of-curve value, so the same simulation run produces varied but plausible outcomes rather
than one brittle deterministic mock draft. The user's own picks are driven by one of a few
candidate policies below; comparing their simulated outcomes is how "which position order is
best" gets answered empirically rather than solved as an exact (and much harder) optimal-control
problem.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from common.draft.historical_value_curves import ValueCurves
from common.draft.pool_structure import POSITIONS, DraftConfig, team_at_pick


def _softmax_choice(rng: np.random.Generator, eligible: list[str], values: list[float],
                    temperature: float) -> str:
    if len(eligible) == 1:
        return eligible[0]
    v = np.array(values, dtype=float)
    v = (v - v.max()) / max(temperature, 1e-6)
    w = np.exp(v)
    w /= w.sum()
    return rng.choice(eligible, p=w)


def _value_greedy(eligible: list[str], state: "_State") -> str:
    return max(eligible, key=lambda p: state.next_value(p))


def _balanced_need(eligible: list[str], state: "_State") -> str:
    """Prioritize whichever eligible position the user's own roster is furthest below its
    final target fraction for, relative to the others."""
    def fill_frac(p: str) -> float:
        return state.my_counts[p] / state.caps[p]
    return min(eligible, key=fill_frac)


def _urgency_greedy(eligible: list[str], state: "_State") -> str:
    """VONA-style: pick the position with the largest expected value erosion by the user's
    next turn. Expected picks-at-position-p before the next turn is estimated as the
    intervening opponent picks scaled by p's share of total remaining league-wide need."""
    remaining_need = {p: state.remaining_need(p) for p in POSITIONS}
    total_need = sum(remaining_need.values()) or 1
    k = state.intervening_picks()

    def erosion(p: str) -> float:
        expected_taken = round(k * remaining_need[p] / total_need)
        now = state.next_value(p)
        later = state.value_at_offset(p, expected_taken)
        return now - later

    return max(eligible, key=erosion)


POLICIES = {
    "value_greedy": _value_greedy,
    "balanced_need": _balanced_need,
    "urgency_greedy": _urgency_greedy,
}


@dataclass
class _State:
    curves: ValueCurves
    dcfg: DraftConfig
    my_slot: int
    taken: dict = field(default_factory=lambda: {p: 0 for p in POSITIONS})
    team_counts: dict = field(default_factory=dict)
    _pick: int = 1

    def __post_init__(self):
        if not self.team_counts:
            self.team_counts = {t: {p: 0 for p in POSITIONS} for t in range(1, self.dcfg.num_teams + 1)}

    @property
    def my_counts(self) -> dict:
        return self.team_counts[self.my_slot]

    @property
    def caps(self) -> dict:
        return self.dcfg.caps

    def next_value(self, position: str) -> float:
        return self.curves.value(position, self.taken[position] + 1)

    def value_at_offset(self, position: str, offset: int) -> float:
        return self.curves.value(position, self.taken[position] + 1 + max(offset, 0))

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
        return [p for p in POSITIONS if self.team_counts[team][p] < self.caps[p]]

    def apply(self, team: int, position: str) -> float:
        value = self.next_value(position)
        self.team_counts[team][position] += 1
        self.taken[position] += 1
        self._pick += 1
        return value


def _run_one_draft(rng: np.random.Generator, curves: ValueCurves, dcfg: DraftConfig,
                   my_slot: int, my_policy_fn) -> tuple[float, list[str]]:
    state = _State(curves=curves, dcfg=dcfg, my_slot=my_slot)
    total_picks = dcfg.num_teams * dcfg.rounds
    my_total_value = 0.0
    my_sequence: list[str] = []

    for pick in range(1, total_picks + 1):
        team = team_at_pick(pick, dcfg.num_teams)
        eligible = state.eligible(team)
        if not eligible:
            continue
        if team == my_slot:
            position = my_policy_fn(eligible, state)
        else:
            values = [state.next_value(p) for p in eligible]
            position = _softmax_choice(rng, eligible, values, dcfg.opponent_temperature)
        value = state.apply(team, position)
        if team == my_slot:
            my_total_value += value
            my_sequence.append(position)

    return my_total_value, my_sequence


@dataclass
class PolicyResult:
    policy: str
    my_slot: int
    avg_total_value: float
    round_distribution: list[dict[str, float]]  # per round: {position: fraction of sims}


def simulate_slot(curves: ValueCurves, dcfg: DraftConfig, my_slot: int,
                  seed: int = 42) -> list[PolicyResult]:
    results = []
    for name, fn in POLICIES.items():
        rng = np.random.default_rng(seed)
        totals = []
        round_counters = [Counter() for _ in range(dcfg.rounds)]
        for _ in range(dcfg.mc_sims):
            total_value, sequence = _run_one_draft(rng, curves, dcfg, my_slot, fn)
            totals.append(total_value)
            for r, pos in enumerate(sequence):
                round_counters[r][pos] += 1
        dist = [{pos: cnt / sum(c.values()) for pos, cnt in c.items()} if c else {}
               for c in round_counters]
        results.append(PolicyResult(policy=name, my_slot=my_slot,
                                    avg_total_value=float(np.mean(totals)),
                                    round_distribution=dist))
    return results


def best_policy(results: list[PolicyResult]) -> PolicyResult:
    return max(results, key=lambda r: r.avg_total_value)
