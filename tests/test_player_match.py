"""Tests for common/draft/player_match.py's fuzzy resolver, against realistic ambiguous/typo/
already-taken cases (modeled on real cases found in the actual 2026-27 rankings: multiple
"Elias Pettersson"s, a "Makar" surname shared by an elite defenseman and an unrelated depth
forward) using a small synthetic DataFrame so the tests stay deterministic and fast rather than
depending on live, regenerable rankings data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from common.draft import player_match


@pytest.fixture
def universe() -> pd.DataFrame:
    return pd.DataFrame([
        {"player_id": 1, "player_name": "Nathan MacKinnon", "position": "forward", "pool_points": 107.9},
        {"player_id": 2, "player_name": "Cale Makar", "position": "defense", "pool_points": 98.1},
        {"player_id": 3, "player_name": "Taylor Makar", "position": "forward", "pool_points": 2.2},
        {"player_id": 4, "player_name": "Elias Pettersson", "position": "forward", "pool_points": 45.8},
        {"player_id": 5, "player_name": "Elias Pettersson", "position": "defense", "pool_points": -1.4},
        {"player_id": 6, "player_name": "Connor McDavid", "position": "forward", "pool_points": 106.3},
    ])


def test_exact_match(universe):
    result = player_match.resolve("Cale Makar", universe, taken={})
    assert isinstance(result, player_match.Unique)
    assert result.row["player_id"] == 2


def test_exact_match_case_insensitive(universe):
    result = player_match.resolve("cale makar", universe, taken={})
    assert isinstance(result, player_match.Unique)
    assert result.row["player_id"] == 2


def test_token_containment_partial_name(universe):
    result = player_match.resolve("mcdavid", universe, taken={})
    assert isinstance(result, player_match.Unique)
    assert result.row["player_id"] == 6


def test_ambiguous_shared_surname(universe):
    result = player_match.resolve("makar", universe, taken={})
    assert isinstance(result, player_match.Ambiguous)
    ids = set(result.candidates["player_id"])
    assert ids == {2, 3}


def test_ambiguous_duplicate_full_name(universe):
    result = player_match.resolve("Elias Pettersson", universe, taken={})
    assert isinstance(result, player_match.Ambiguous)
    ids = set(result.candidates["player_id"])
    assert ids == {4, 5}


def test_typo_tolerant_fallback(universe):
    result = player_match.resolve("Nathn MacKinon", universe, taken={})
    assert isinstance(result, player_match.Unique)
    assert result.row["player_id"] == 1


def test_already_taken(universe):
    result = player_match.resolve("Cale Makar", universe, taken={2: 5})
    assert isinstance(result, player_match.AlreadyTaken)
    assert result.row["player_id"] == 2
    assert result.taken_pick == 5


def test_ambiguous_collapses_to_unique_once_one_is_taken(universe):
    # "makar" matches both Cale and Taylor; once Taylor is taken, only Cale remains available.
    result = player_match.resolve("makar", universe, taken={3: 7})
    assert isinstance(result, player_match.Unique)
    assert result.row["player_id"] == 2


def test_not_found(universe):
    result = player_match.resolve("Zzznotarealplayer", universe, taken={})
    assert isinstance(result, player_match.NotFound)


def test_empty_query(universe):
    result = player_match.resolve("   ", universe, taken={})
    assert isinstance(result, player_match.NotFound)
