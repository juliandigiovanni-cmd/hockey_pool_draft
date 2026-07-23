Continue the hockey pool draft-prep project. Status and next steps below — full history is in PROJECT.md at the repo root if more context is needed.

## Status
- Full pipeline rebuilt into a shared `common/` library + `run_season.py` CLI (config-driven, one command per season), committed and pushed to github.com/juliandigiovanni-cmd/hockey_pool_draft (private).
- Verified end-to-end on real 2026-27 data. Current rankings already exist at `2026-27/results/finalpool_2026-27_overall_rankings.csv` (and `.xlsx`).
- 2026-07-23: NHL season length increased 82 -> 84 games starting 2026-27. Added `games_per_season`
  to `SeasonConfig`/`config.yaml` (see PROJECT.md changelog) so this is config-driven going forward;
  rankings already regenerated with 84-game projections.

## Next steps (priority order)

1. **Closer to draft day**: bump `roster_as_of_date` in `2026-27/config.yaml`, then re-run
   `python run_season.py --season 2026-27 --stage scrape` followed by `--stage predict` to
   refresh rankings with the latest trades/signings/roster moves.
2. **Optional — MoneyPuck data source**: `common/scrape/sources/moneypuck.py` was network-blocked
   (connection reset) as of 2026-07-23, both from a dev sandbox and this machine, even after
   trying proper browser headers — looks like a network/firewall-level block, not a bot check.
   Try again from a different network sometime. If it starts working, xG/shot-quality features
   flow into the models automatically with no further code changes. If it stays blocked, consider
   Natural Stat Trick as a fallback (needs a manually-approved access key, more setup).
3. **Optional — better modeling**: several targets currently have weak R² (goalie
   GAA/save%/shutouts, defense plus-minus) — worth investigating once MoneyPuck data is flowing,
   or independently (feature engineering, alternate algorithms, or just leaning on the bonus
   argmax/argmin logic for the weak targets rather than the raw regression).
4. **Deferred**: interactive tool for analyzing/selecting players live during the actual draft.
   Explicitly last-priority from the original ask; nothing built yet.
5. **Next year (2027-28)**: copy `2026-27/config.yaml` to a new `2027-28/config.yaml`, update the
   season/trade-override/roster-date fields, and run — that's the whole annual re-run process.
