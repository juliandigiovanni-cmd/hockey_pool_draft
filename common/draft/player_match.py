"""Fuzzy player-name resolution for live draft entry. Typing an exact full name correctly under
time pressure during a real draft is slow and error-prone, so partial/typo-tolerant matching is
the primary UX, with disambiguation only when a query is genuinely ambiguous. Stdlib-only
(`difflib`) per this codebase's no-new-heavyweight-dependencies convention.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from typing import Union

import pandas as pd


@dataclass
class Unique:
    row: pd.Series


@dataclass
class Ambiguous:
    candidates: pd.DataFrame


@dataclass
class AlreadyTaken:
    row: pd.Series
    taken_pick: int


@dataclass
class NotFound:
    query: str


MatchResult = Union[Unique, Ambiguous, AlreadyTaken, NotFound]


def _token_containment_mask(query: str, names: pd.Series) -> pd.Series:
    tokens = query.lower().split()
    lowered = names.str.lower()
    mask = pd.Series(True, index=names.index)
    for t in tokens:
        mask &= lowered.str.contains(t, regex=False)
    return mask


def resolve(query: str, universe: pd.DataFrame, taken: dict[int, int]) -> MatchResult:
    """Resolve a typed query to a player.

    `universe`: the FULL rankings DataFrame (player_id, player_name, position, pool_points, ...),
    including already-taken players — needed to distinguish "already taken" from "no such
    player" (a much clearer message live than a bare not-found).
    `taken`: {player_id: overall_pick_number}.
    """
    query = query.strip()
    if not query:
        return NotFound(query=query)

    names = universe["player_name"]

    # 1. exact case-insensitive full-name match (fast path for the common case)
    candidates = universe[names.str.lower() == query.lower()]

    # 2. token-containment: every whitespace-split token of the query is a substring of the name
    if candidates.empty:
        candidates = universe[_token_containment_mask(query, names)]

    # 3. typo-tolerant fallback
    if candidates.empty:
        close = difflib.get_close_matches(query, names.tolist(), n=6, cutoff=0.6)
        candidates = universe[names.isin(close)]

    if candidates.empty:
        return NotFound(query=query)

    not_taken = candidates[~candidates["player_id"].isin(taken)]
    taken_candidates = candidates[candidates["player_id"].isin(taken)]

    if len(not_taken) == 1:
        return Unique(row=not_taken.iloc[0])
    if len(not_taken) > 1:
        return Ambiguous(candidates=not_taken)
    # every match found is already off the board
    row = taken_candidates.iloc[0]
    return AlreadyTaken(row=row, taken_pick=taken[int(row["player_id"])])
