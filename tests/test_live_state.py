"""Tests for common/draft/live_state.py: pick application, undo, and log-replay (the mechanism
resume relies on) produce identical state to an unbroken run, plus the cap-rejection guard and
the duck-typed interface strategy_sim.POLICIES actually depends on.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.draft.live_state import LivePool, PickEvent, _LiveState, replay
from common.draft.pool_structure import DraftConfig
from common.draft.strategy_sim import POLICIES


@pytest.fixture
def dcfg() -> DraftConfig:
    # Small 2-team, 1-slot-per-position draft (6 total picks) for fast, easy-to-reason-about tests.
    caps = {"forward": 1, "defense": 1, "goalie": 1}
    return DraftConfig(num_teams=2, rounds=3, caps=caps,
                      league_totals={p: c * 2 for p, c in caps.items()},
                      opponent_temperature=1.5, mc_sims=10)


@pytest.fixture
def rankings() -> pd.DataFrame:
    rows = []
    for i, (name, pos, pts) in enumerate([
        ("F1", "forward", 100.0), ("F2", "forward", 90.0), ("F3", "forward", 80.0),
        ("D1", "defense", 70.0), ("D2", "defense", 60.0), ("D3", "defense", 50.0),
        ("G1", "goalie", 40.0), ("G2", "goalie", 30.0), ("G3", "goalie", 20.0),
    ], start=1):
        rows.append({"player_id": i, "player_name": name, "position": pos,
                    "team_abbrev": "XXX", "pool_points": pts})
    return pd.DataFrame(rows)


def _state(dcfg, rankings, my_slot=1) -> _LiveState:
    return _LiveState(pool=LivePool(rankings), dcfg=dcfg, my_slot=my_slot)


def test_next_value_and_offset(dcfg, rankings):
    s = _state(dcfg, rankings)
    assert s.next_value("forward") == 100.0
    assert s.value_at_offset("forward", 0) == 100.0
    assert s.value_at_offset("forward", 1) == 90.0
    assert s.value_at_offset("forward", 99) == 80.0  # clamps to the last available


def test_apply_updates_counts_and_pick(dcfg, rankings):
    s = _state(dcfg, rankings)
    assert s.current_team() == 1
    s.apply(team=1, player_id=1, position="forward")  # F1
    assert s.team_counts[1]["forward"] == 1
    assert s._pick == 2
    assert s.pool.is_taken(1)
    assert s.next_value("forward") == 90.0  # F1 no longer available


def test_eligible_excludes_filled_positions(dcfg, rankings):
    s = _state(dcfg, rankings)
    s.apply(team=1, player_id=1, position="forward")
    assert "forward" not in s.eligible(1)
    assert "forward" in s.eligible(2)  # team 2 untouched


def test_apply_rejects_when_cap_full(dcfg, rankings):
    s = _state(dcfg, rankings)
    s.apply(team=1, player_id=1, position="forward")  # fills team 1's 1 forward slot
    with pytest.raises(ValueError):
        s.apply(team=1, player_id=2, position="forward")
    # rejected pick must not have mutated state
    assert s.team_counts[1]["forward"] == 1
    assert not s.pool.is_taken(2)


def test_undo_reverts_exactly(dcfg, rankings):
    s = _state(dcfg, rankings)
    s.apply(team=1, player_id=1, position="forward")
    s.undo_last(team=1, player_id=1, position="forward")
    assert s.team_counts[1]["forward"] == 0
    assert s._pick == 1
    assert not s.pool.is_taken(1)
    assert s.next_value("forward") == 100.0


def test_is_complete(dcfg, rankings):
    s = _state(dcfg, rankings)
    picks = [  # snake order for 2 teams x 3 rounds: 1,2,2,1,1,2
        (1, 1, "forward"), (2, 4, "defense"), (2, 7, "goalie"),
        (1, 5, "defense"), (1, 8, "goalie"), (2, 2, "forward"),
    ]
    for team, pid, pos in picks:
        assert not s.is_complete()
        s.apply(team, pid, pos)
    assert s.is_complete()


def test_replay_matches_incremental_apply(dcfg, rankings):
    picks = [
        (1, 1, "forward", "F1"), (2, 4, "defense", "D1"), (2, 7, "goalie", "G1"),
        (1, 5, "defense", "D2"), (1, 8, "goalie", "G2"), (2, 2, "forward", "F2"),
    ]
    incremental = _state(dcfg, rankings)
    for team, pid, pos, _name in picks:
        incremental.apply(team, pid, pos)

    events = [PickEvent(overall_pick=i + 1, team=t, player_id=pid, player_name=name,
                       position=pos, timestamp="2026-01-01T00:00:00")
             for i, (t, pid, pos, name) in enumerate(picks)]
    replayed = replay(events, LivePool(rankings), dcfg, my_slot=1)

    assert replayed.team_counts == incremental.team_counts
    assert replayed._pick == incremental._pick
    assert replayed.pool.taken == incremental.pool.taken


def test_pick_event_json_round_trip():
    ev = PickEvent(overall_pick=3, team=2, player_id=42, player_name="Test Player",
                  position="forward", timestamp="2026-01-01T00:00:00")
    assert PickEvent.from_dict(ev.to_dict()) == ev


def test_replay_rejects_out_of_order_log(dcfg, rankings):
    events = [PickEvent(overall_pick=2, team=1, player_id=1, player_name="F1",
                       position="forward", timestamp="2026-01-01T00:00:00")]
    with pytest.raises(ValueError):
        replay(events, LivePool(rankings), dcfg, my_slot=1)


@pytest.mark.parametrize("policy_name", list(POLICIES.keys()))
def test_policies_run_against_live_state(dcfg, rankings, policy_name):
    """The core reuse claim: strategy_sim.POLICIES functions, written against the Monte Carlo
    simulation's _State, work unchanged when duck-typed against a live _LiveState."""
    s = _state(dcfg, rankings, my_slot=1)
    eligible = s.eligible(1)
    position = POLICIES[policy_name](eligible, s)
    assert position in eligible
