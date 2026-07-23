# Hockey Pool Model — Project Tracker

Recurring annual project: scrape NHL data, clean/merge it, train ML models, and produce
draft-day player rankings for a custom fantasy hockey pool. The draft happens once a year;
this doc tracks the architecture, decisions, and a changelog across seasons so each year's
re-run starts from "copy last season's config, update it, run" rather than a from-scratch rebuild.

## Architecture

```
common/            shared package, reused every season
  config.py          loads a season's config.yaml
  scrape/
    nhl_api.py         NHL API client (teams, skater/goalie stats, standings, per-game logs)
    rosters.py         current-team-as-of-date lookup (handles offseason trades/FA/waivers)
    sources/
      moneypuck.py       free CSV client for xG/shot-quality/GSAx (moneypuck.com/data.htm)
  clean/
    merge.py           merges skater/goalie stats with team data into model-ready CSVs
  features/
    engineering.py     shared lag-feature / season-utility functions
  models/
    training.py        shared train/tune/evaluate harness (one implementation, not per-position copies)
    forwards.py / defense.py / goalies.py   per-position target + feature configs
    pool_ranking.py    unified scoring step, consumes the persisted tuned models
  pipeline.py          stage functions: scrape / clean / train / predict

run_season.py       CLI entry point: python run_season.py --season 2026-27 --stage all|scrape|clean|train|predict

<season>/            one folder per draft year, e.g. 2026-27/
  config.yaml           season id, trade overrides, scoring rules, roster-snapshot date
  data/{raw,processed}/ scraped + cleaned data (gitignored)
  models/                persisted joblib model artifacts (gitignored)
  results/               rankings/reports csv/xlsx/txt (gitignored)
  plots/                 diagnostics (gitignored)
```

`2025-26/` is last year's original ad-hoc build (24 near-duplicate model scripts, no shared
library, nothing persisted, no git tracking). It's kept as historical reference only — not
imported by anything in `common/`.

## Pool scoring rules

Source of truth: `2025-26/instructions_for_claude_6Sept25.docx`, encoded per-season in each
season's `config.yaml` under `scoring:`.

- Forwards: goals + assists
- Defense: goals + assists + plus/minus
- Goalies (min 40 GP to qualify for bonuses): 1 pt/win, 3 pts total for a shutout win
  (not additive on top of the win point), +10 for best GAA among qualified goalies,
  +10 for best save % among qualified goalies

## Known modeling tradeoffs (carried over deliberately, not bugs)

- Defense plus/minus model uses contemporaneous (current-season) team features, while the
  points model stays strictly lagged — plus/minus is hard to predict from lagged data alone,
  so this is a documented leakage tradeoff specific to that one target.
- Validation methodology: train on all data vs. train excluding the most recent season and
  check against it out-of-sample; the better-performing approach per position/target is used
  for the final prediction.

## Data sources

Research (2026-07-23) compared MoneyPuck, Natural Stat Trick, Evolving-Hockey, Elite
Prospects, and unused NHL API endpoints, constrained to **free sources only**. Decision:

- **NHL API** (`api.nhle.com`, `api-web.nhle.com`) — primary source, free, no key required.
  Historical stats from 2008-09 onward (the NHL API doesn't support earlier seasons).
  Expanded this rebuild to also pull per-player per-game logs (`nhl_game_logs.csv`, most
  recent season only — a full historical backfill would mean one API call per player per
  season across 2008-2025 for marginal benefit over the existing season-level lag features)
  for rolling-form features, and current-team rosters (see `rosters.py` above).
- **MoneyPuck** (`common/scrape/sources/moneypuck.py`) — added, but **unreachable in practice
  so far** (2026-07-23): every request gets an immediate `Connection reset by peer`, both from
  the dev sandbox and from the user's real machine. Tried switching from bare `pd.read_csv(url)`
  to `requests` with a browser-like User-Agent (a classic Cloudflare bot-check fix) — made no
  difference, so this looks like a network-level block (firewall/ASN/IP), not a bot-UA check.
  The scrape stage degrades gracefully when this happens (logs a warning, skips MoneyPuck,
  continues to clean/train/predict) rather than failing the run. Revisit if/when reachable from
  a different network, or drop it in favor of Natural Stat Trick if it stays blocked.
- **Natural Stat Trick** — skipped for now. Free, but requires requesting a manually-approved
  access key before automated pulls work, and its stats mostly overlap MoneyPuck's. Revisit
  only if MoneyPuck's feature set proves insufficient.
- **Evolving-Hockey** — skipped. The valuable parts (RAPM/GAR/xGAR, projections, CSV
  downloads) are paywalled; the free tier offers nothing MoneyPuck/NST don't already cover.
- **Elite Prospects** — skipped. Bio/prospect-history data, not stats-focused; scraping ToS
  is unclear and the API is commercial. Revisit only if a rookie/prospect-projection feature
  is added later.

## Season changelog

### 2026-27
- Rebuilt from `2025-26`'s logic into a shared `common/` library + config-driven per-season
  folders, so future seasons are a config copy + run rather than a code copy.
- Replaced hardcoded trade overrides with `common/scrape/rosters.py`, which snapshots each
  player's current team as of a configurable date — summer trades/FA/waiver moves are picked
  up automatically instead of requiring a manual script edit every offseason.
- Persisting trained models (joblib) for the first time; the "finalpool" ranking step now
  consumes those artifacts directly instead of re-deriving a simplified version of the same
  targets from scratch (as the old `finalpool_points_model_*` scripts did).
- Data-source research restricted to free sources only, given budget constraints.
- Working in July, before the 2026-27 season starts: no new season stats exist yet to scrape;
  this run refreshes historical data through 2025-26 and current rosters only.
- Modeling consolidation (`common/features/engineering.py`, `common/models/*.py`) done with
  Opus. Leakage prevention switched from the legacy denylist (enumerate contemporaneous
  columns to exclude) to an allowlist (only lag/COVID/engineered/documented-exception columns
  are eligible) — safer by construction. Verified end-to-end against real 2025-26 data via
  `run_season.py --stage clean` then `--stage train`: full pipeline ~49s, `--stage predict`
  (reusing persisted models) ~2s. Output was hockey-plausible: MacKinnon/Kucherov/McDavid top
  forwards, Makar top defenseman, Hellebuyck/Vasilevskiy top goalies. Goalie GAA/save%/defense
  plus-minus have low or negative R² (same as last year — honestly surfaced in the report
  rather than hidden; the pool-points bonus logic only needs the argmax/argmin among qualified
  goalies, not a well-fit regression).
- Test artifacts from that verification (copied-in raw CSVs, trained models, results, plots)
  were removed afterward — `2026-27/` is clean and ready for a real `--stage all` run.
- **First real `--stage all` run (2026-07-23) found and fixed three bugs**:
  1. **Crash**: `update_moneypuck_data()` raised `ValueError: No objects to concatenate` when
     every year's fetch failed for an entity (which happens every run right now — see MoneyPuck
     note above) — took down `clean`/`train`/`predict` with it. Fixed: guard against zero
     successful fetches, log a warning, skip that entity. Also wrapped the roster/game-log/
     MoneyPuck scrape steps in `common/pipeline.py` with a best-effort try/except, since none of
     them are required by `clean_and_merge` — one enhancement source being down should never
     block the required NHL stats -> clean -> train -> predict path.
  2. **Redundant re-download**: `update_raw_data()`/`update_moneypuck_data()` only skipped
     seasons/years already present in *that season's own* `raw_dir` — since every new season
     folder starts empty, each year's first scrape re-fetched the entire 2008-present history
     from scratch (~150-200+ NHL API requests) before reaching the one actually-new season.
     Fixed: `nhl_api.seed_from_prior_season()` / `moneypuck._seed_from_prior_season()` now copy
     the previous season's raw CSVs forward on first run, so only the newly-completed season
     needs fetching. Chains automatically from 2027-28 onward.
  3. **Roster fetch noise**: `update_current_rosters()` looped over all 62 all-time franchise
     codes from `nhl_teams.csv` (including 29 defunct/relocated ones like QUE/ATL/HFD), each
     good for a guaranteed 404. Fixed: derive the active-team list from the most recent season
     in `nhl_team_stats.csv` instead — the roster data itself was always fine (800/800 players
     across the real 32 teams), this only cut the noise/wasted requests.
  - Re-ran end-to-end after fixes: scrape (skips cached history correctly), clean (16,273 skater
    rows, includes the just-completed 2025-26 season), train (~1 min), predict (~2s, reused
    models). Real 2026-27 rankings: McDavid #1, Draisaitl #2, MacKinnon #3, Kucherov #4,
    Celebrini #5 — hockey-plausible, and reflects real current data.
- **NHL regular season expanded from 82 to 84 games, effective 2026-27**: added a
  `games_per_season` field to `SeasonConfig`/`config.yaml` (default 82 for back-compat with
  seasons that don't set it) instead of hardcoding the game count. `2026-27/config.yaml` sets
  it to 84. `common/models/pool_ranking.py` now uses `cfg.games_per_season` for forward/defense
  projected games and as the cap on the goalie 3-year rolling-average games projection, so
  future game-count changes are a one-line config edit, not a code change. Left the `82` fallback
  in `common/features/engineering.py` (`add_team_goal_differential`) alone — it's a missing-data
  default for historical (genuinely 82-game) seasons' team-level games-played, and doesn't affect
  the 2026-27 prediction row since that row's goal columns are NaN -> 0 regardless of denominator.
  Re-ran `--stage predict` (reusing persisted models, no retrain needed) to refresh
  `2026-27/results/` with 84-game projections.

## Future milestones (not in scope for the 2026-27 rebuild)

- Interactive tool for analyzing and selecting players dynamically live during the draft.
