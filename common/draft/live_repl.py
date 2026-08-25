"""Interactive command loop for the live draft-day assistant. Opened once at the start of a real
draft and left running for its duration (~1-2 hours, up to `num_teams * rounds` picks) — picks are
streamed in as they're announced, rather than re-invoking the script per pick.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from common.config import SeasonConfig
from common.draft import player_match
from common.draft.live_state import LivePool, PickEvent, _LiveState, replay
from common.draft.pool_structure import POSITIONS, DraftConfig
from common.draft.strategy_sim import POLICIES

STATE_FILENAME = "live_draft_state.json"
_ABBREV = {"forward": "F", "defense": "D", "goalie": "G"}

HELP_TEXT = """\
Commands:
  pick <name>          Mark the current team's pick (team implied by turn order)
  pick @<team> <name>  Mark a pick for an explicit team number (out-of-order correction)
  show                 Show current turn + your roster fill + recommendation (if your turn)
  top <position> [n]   Show the top n available players at a position, any time
  sleepers [pos] [n]   Show dark-horse (breakout_score) candidates, any time
  undo                 Undo the most recent pick (asks to confirm)
  help                 Show this message
  quit / exit          Save and exit (resume later by re-running without --reset)
"""
_DARK_HORSE_MARKER = " \U0001F525"  # fire emoji, flags dark_horse=True players in listings


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _pos_abbrev(position: str) -> str:
    return _ABBREV.get(position, position)


def default_state_path(cfg: SeasonConfig) -> Path:
    return cfg.results_dir / "draft" / STATE_FILENAME


def load_policy_for_slot(cfg: SeasonConfig, my_slot: int) -> tuple[str, str]:
    """Read policy_comparison.csv (written by draft_strategy.py) to find the winning policy for
    this draft slot. Falls back to balanced_need (the near-universal winner per the pipeline's
    own historical simulation) with a warning if that file doesn't exist yet."""
    path = cfg.results_dir / "draft" / "policy_comparison.csv"
    if not path.exists():
        return "balanced_need", (" (default — policy_comparison.csv not found; run "
                                 "draft_strategy.py to compute a slot-specific winner)")
    df = pd.read_csv(path)
    slot_rows = df[df["my_slot"] == my_slot]
    if slot_rows.empty:
        return "balanced_need", f" (default — no policy_comparison.csv row for slot {my_slot})"
    winner = slot_rows.loc[slot_rows["avg_total_value"].idxmax()]
    return str(winner["policy"]), (f" (won avg {winner['avg_total_value']:.1f} for slot "
                                   f"{my_slot} per policy_comparison.csv)")


class LiveDraftSession:
    def __init__(self, cfg: SeasonConfig, dcfg: DraftConfig, pool: LivePool, state: _LiveState,
                my_slot: int, policy_name: str, policy_reason: str, top_n: int,
                state_path: Path, log: list[PickEvent], created_at: str | None = None):
        self.cfg = cfg
        self.dcfg = dcfg
        self.pool = pool
        self.state = state
        self.my_slot = my_slot
        self.policy_name = policy_name
        self._policy_reason = policy_reason
        self.top_n = top_n
        self.state_path = state_path
        self.log = log
        self._created_at = created_at or _now_iso()
        self._pending_ambiguous: tuple[int, pd.DataFrame] | None = None

    # -------------------------------------------------------------- construction

    @classmethod
    def new(cls, cfg: SeasonConfig, dcfg: DraftConfig, rankings: pd.DataFrame, my_slot: int,
           policy_name: str, policy_reason: str, top_n: int, state_path: Path) -> "LiveDraftSession":
        pool = LivePool(rankings)
        state = _LiveState(pool=pool, dcfg=dcfg, my_slot=my_slot)
        session = cls(cfg, dcfg, pool, state, my_slot, policy_name, policy_reason, top_n,
                     state_path, log=[])
        session._save()
        return session

    @classmethod
    def resume(cls, cfg: SeasonConfig, dcfg: DraftConfig, rankings: pd.DataFrame,
              state_path: Path, top_n: int) -> "LiveDraftSession":
        data = json.loads(state_path.read_text())
        my_slot = int(data["my_slot"])
        policy_name = data.get("policy", "balanced_need")
        events = [PickEvent.from_dict(d) for d in data["picks"]]
        pool = LivePool(rankings)
        state = replay(events, pool, dcfg, my_slot)
        return cls(cfg, dcfg, pool, state, my_slot, policy_name, "", top_n, state_path,
                  log=events, created_at=data.get("created_at"))

    # -------------------------------------------------------------- persistence

    def _save(self) -> None:
        data = {
            "season": self.cfg.season,
            "my_slot": self.my_slot,
            "policy": self.policy_name,
            "created_at": self._created_at,
            "updated_at": _now_iso(),
            "picks": [ev.to_dict() for ev in self.log],
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(self.state_path)  # atomic on POSIX

    # -------------------------------------------------------------- pending disambiguation

    def has_pending(self) -> bool:
        return self._pending_ambiguous is not None

    def clear_pending(self) -> None:
        self._pending_ambiguous = None

    def resolve_pending(self, choice: int) -> None:
        team, candidates = self._pending_ambiguous
        self._pending_ambiguous = None
        if not (1 <= choice <= len(candidates)):
            print(f"Cancelled — {choice} is out of range (1-{len(candidates)}).")
            return
        self._commit_pick(team, candidates.iloc[choice - 1])

    # -------------------------------------------------------------- commands

    def cmd_pick(self, arg: str) -> None:
        arg = arg.strip()
        team = self.state.current_team()
        if arg.startswith("@"):
            team_str, _, rest = arg[1:].partition(" ")
            try:
                team = int(team_str)
            except ValueError:
                print(f"Invalid team number: {team_str!r}")
                return
            if not (1 <= team <= self.dcfg.num_teams):
                print(f"Team must be 1-{self.dcfg.num_teams}")
                return
            query = rest.strip()
        else:
            query = arg

        if not query:
            print("Usage: pick <player name>   (or  pick @<team> <player name>  to override "
                 "whose turn it is)")
            return

        result = player_match.resolve(query, self.pool.rankings, self.pool.taken)
        if isinstance(result, player_match.NotFound):
            print(f"No player found matching {result.query!r}.")
        elif isinstance(result, player_match.AlreadyTaken):
            print(f"{result.row['player_name']} was already picked (pick #{result.taken_pick}).")
        elif isinstance(result, player_match.Ambiguous):
            self._pending_ambiguous = (team, result.candidates)
            print(f"Multiple matches for {query!r}:")
            for i, (_, r) in enumerate(result.candidates.head(9).iterrows(), start=1):
                print(f"  {i}. {r['player_name']:<26} {_pos_abbrev(r['position']):<3} "
                     f"pool_points={r['pool_points']:.1f}")
            print("Enter a number to confirm, or any other command to cancel.")
        else:
            self._commit_pick(team, result.row)

    def _commit_pick(self, team: int, row: pd.Series) -> None:
        position = str(row["position"])
        if position not in self.state.eligible(team):
            counts = self.state.team_counts[team]
            if sum(counts.values()) < sum(self.dcfg.starter_caps.values()):
                print(f"Team {team}'s starter {position} slot(s) are already full "
                     f"(cap {self.dcfg.starter_caps[position]}) — bench slots don't open until "
                     f"all 11 starter slots are filled. Pick rejected.")
            else:
                print(f"Team {team} already has a full {position} roster "
                     f"(cap {self.dcfg.caps[position]}) — pick rejected.")
            return
        pick_no = self.state._pick
        player_id = int(row["player_id"])
        self.state.apply(team, player_id, position)
        event = PickEvent(overall_pick=pick_no, team=team, player_id=player_id,
                         player_name=str(row["player_name"]), position=position,
                         timestamp=_now_iso())
        self.log.append(event)
        self._save()
        print(f"-> Team {team} picks {row['player_name']} ({_pos_abbrev(position)}).")
        if self.state.is_complete():
            self.print_final_summary()
        else:
            self.print_turn_status()

    def cmd_undo(self) -> None:
        if not self.log:
            print("Nothing to undo.")
            return
        last = self.log[-1]
        confirm = input(f"Undo pick #{last.overall_pick}: Team {last.team} - "
                       f"{last.player_name}? [y/N] ").strip().lower()
        if confirm != "y":
            print("Cancelled.")
            return
        self.state.undo_last(last.team, last.player_id, last.position)
        self.log.pop()
        self._save()
        print(f"Undone: pick #{last.overall_pick} ({last.player_name}).")
        self.print_turn_status()

    def cmd_top(self, arg: str) -> None:
        parts = arg.split()
        if not parts:
            print(f"Usage: top <{'|'.join(POSITIONS)}> [n]")
            return
        position = parts[0].lower()
        if position not in POSITIONS:
            print(f"Unknown position {position!r}. Choose from: {', '.join(POSITIONS)}")
            return
        n = self.top_n
        if len(parts) > 1:
            try:
                n = int(parts[1])
            except ValueError:
                print(f"Invalid count: {parts[1]!r}")
                return
        self._print_top(position, n, header=f"Top {n} available {position}:")

    def cmd_sleepers(self, arg: str) -> None:
        if "dark_horse" not in self.pool.rankings.columns:
            print("No breakout_score/dark_horse data in this season's rankings — re-run "
                 "run_season.py --stage predict to generate it.")
            return
        parts = arg.split()
        position = None
        if parts and parts[0].lower() in POSITIONS:
            position = parts[0].lower()
            parts = parts[1:]
        n = self.top_n
        if parts:
            try:
                n = int(parts[0])
            except ValueError:
                print(f"Invalid count: {parts[0]!r}")
                return

        found_any = False
        for pos in [position] if position else list(POSITIONS):
            avail = self.pool.available(pos)
            sleepers = (avail[avail["dark_horse"]]
                       .sort_values("breakout_score", ascending=False).head(n))
            if sleepers.empty:
                continue
            found_any = True
            print(f"Sleepers — {pos}:")
            for _, r in sleepers.iterrows():
                print(f"    {r['player_name']:<26} {r['team_abbrev']:<5} "
                     f"pool_points={r['pool_points']:.1f}  breakout={r['breakout_score']:.2f}")
        if not found_any:
            where = f" at {position}" if position else ""
            print(f"No dark-horse candidates currently available{where}.")

    # -------------------------------------------------------------- display

    def banner(self) -> str:
        total = self.dcfg.num_teams * self.dcfg.rounds
        return (f"Live draft — {self.cfg.season}, you are slot {self.my_slot}/{self.dcfg.num_teams}, "
               f"{self.dcfg.rounds} rounds ({total} total picks). Policy: {self.policy_name}"
               f"{self._policy_reason}. Type 'help' for commands.")

    def prompt(self) -> str:
        team = self.state.current_team()
        marker = "*" if team == self.my_slot else " "
        return f"[pick {self.state._pick}{marker}] > "

    def print_turn_status(self) -> None:
        team = self.state.current_team()
        pick_no = self.state._pick
        if team == self.my_slot:
            print(f"\n=== Pick #{pick_no} — YOUR TURN (team {team}) ===")
            self.print_recommendation()
        else:
            print(f"Pick #{pick_no}: Team {team}'s turn.")

    def print_recommendation(self) -> None:
        eligible = self.state.eligible(self.my_slot)
        if not eligible:
            print("All your roster slots are full.")
            return
        rec_position = POLICIES[self.policy_name](eligible, self.state)
        print(f"Policy: {self.policy_name}{self._policy_reason}")
        fill = self.state.my_counts
        print("Your roster: " + ", ".join(f"{p}={fill[p]}/{self.dcfg.caps[p]}" for p in POSITIONS))
        print(f"Recommended position: {rec_position.upper()}")
        self._print_top(rec_position, self.top_n, header="  Top available:")
        for p in eligible:
            if p != rec_position:
                self._print_top(p, min(5, self.top_n), header=f"  Also eligible — {p}:")

    def print_status_panel(self) -> None:
        team = self.state.current_team()
        print(f"Pick #{self.state._pick} — Team {team}'s turn"
             f"{' (you)' if team == self.my_slot else ''}.")
        for t in range(1, self.dcfg.num_teams + 1):
            fill = self.state.team_counts[t]
            you = " (you)" if t == self.my_slot else ""
            print(f"  Team {t}{you}: " +
                 ", ".join(f"{p}={fill[p]}/{self.dcfg.caps[p]}" for p in POSITIONS))
        if team == self.my_slot:
            self.print_recommendation()

    def print_final_summary(self) -> None:
        print("\n=== DRAFT COMPLETE ===")
        my_picks = [ev for ev in self.log if ev.team == self.my_slot]
        print(f"Your team (slot {self.my_slot}), {len(my_picks)} picks:")
        for ev in my_picks:
            print(f"  Pick #{ev.overall_pick:>3}  {ev.player_name:<26} ({_pos_abbrev(ev.position)})")

    def _print_top(self, position: str, n: int, header: str) -> None:
        print(header)
        avail = self.pool.available(position).head(n)
        if avail.empty:
            print("    (none available)")
            return
        has_dark_horse = "dark_horse" in avail.columns
        for _, r in avail.iterrows():
            marker = _DARK_HORSE_MARKER if has_dark_horse and bool(r["dark_horse"]) else ""
            print(f"    {r['player_name']:<26} {r['team_abbrev']:<5} "
                 f"pool_points={r['pool_points']:.1f}{marker}")


def run(session: LiveDraftSession) -> None:
    print(session.banner())
    if session.state.is_complete():
        session.print_final_summary()
        return
    session.print_turn_status()

    while not session.state.is_complete():
        try:
            line = input(session.prompt()).strip()
        except EOFError:
            print("\nEnd of input — state saved.")
            break

        if not line:
            continue

        if session.has_pending() and line.lstrip("-").isdigit():
            session.resolve_pending(int(line))
            continue
        if session.has_pending():
            session.clear_pending()
            print("Selection cancelled.")

        cmd, _, arg = line.partition(" ")
        cmd = cmd.lower()

        if cmd in ("quit", "exit"):
            print("Exiting. State saved — resume anytime with the same command (no --reset).")
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "show":
            session.print_status_panel()
        elif cmd == "top":
            session.cmd_top(arg)
        elif cmd == "sleepers":
            session.cmd_sleepers(arg)
        elif cmd == "undo":
            session.cmd_undo()
        elif cmd == "pick":
            session.cmd_pick(arg)
        else:
            print(f"Unknown command: {cmd!r}. Type 'help' for commands.")
