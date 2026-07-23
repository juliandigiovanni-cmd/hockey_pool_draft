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
- **MoneyPuck** (`common/scrape/sources/moneypuck.py`) — added. Free CSV downloads, no
  signup/approval needed, adds xGoals/shot-quality/goalie GSAx features the NHL API doesn't
  have. *Not yet verified with a live fetch* — this session's sandbox network couldn't reach
  moneypuck.com (connection reset, likely Cloudflare bot protection on the sandbox IP);
  confirm the CSV URLs still resolve on a real run of the scrape stage.
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

## Future milestones (not in scope for the 2026-27 rebuild)

- Interactive tool for analyzing and selecting players dynamically live during the draft.
