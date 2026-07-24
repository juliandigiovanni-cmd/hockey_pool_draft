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
  diagnostics/
    reconcile.py       cross-source stat comparison (NHL API vs MoneyPuck counting stats)
    model_report.py    model fit diagnostics: metrics table, PvA plots, baseline R², rolling OOS
  pipeline.py          stage functions: scrape / clean / train / predict

run_season.py       CLI entry point: python run_season.py --season 2026-27 --stage all|scrape|clean|train|predict

<season>/            one folder per draft year, e.g. 2026-27/
  config.yaml           season id, trade overrides, scoring rules, roster-snapshot date
  data/{raw,processed}/ scraped + cleaned data (gitignored)
  models/                persisted joblib model artifacts (gitignored)
  results/               rankings/reports csv/xlsx/txt (gitignored)
  plots/                 diagnostics (gitignored)
  results/diagnostics/   source_reconciliation.csv, model_metrics.csv
  plots/diagnostics/     pva_*.png (predicted-vs-actual + residuals per model)
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

## Known modeling limitations (as of 2026-07-24)

- **Goalie GAA and save%**: OOS R² < 0, meaning the model predicts *worse* than predicting
  the training-set mean. These targets are fundamentally hard to predict from prior seasons;
  season-to-season goalie stats have high variance and low autocorrelation. In the pool,
  the bonus scoring only needs the argmax/argmin among 40+ GP qualifiers — the bonus is
  awarded to whoever scores highest/lowest, so even a weak model's ranking at the top is
  informative. But this should be revisited: consider predicting the percentile rank rather
  than the raw stat, or switching to a simpler "stability-weighted career average" baseline.
- **Defense plus/minus OOS R²=0.031**: essentially random out-of-sample. The contemporaneous
  team-context features (allowed for this target) don't help for future-season prediction
  because team composition changes. The model is useful only insofar as it captures some
  signal from lagged team quality. Worth exploring a joint points+plus/minus model (see Future).
- **Traded players — team context**: players who changed teams during a historical season are
  assigned the *first* team listed in the NHL API's comma-joined `team_abbrev`. Their season
  totals (goals, assists, games_played) are preserved correctly; only the team-context features
  (used mainly by the defense plus/minus model) may be inaccurate. Per-team game logs are only
  available for the most recent season, so a majority-team fix across all 18 training seasons
  would require expanding game-log history (one API call per player per historical season).
- **Pool scoring rules are constant**: no year-to-year variation in the scoring system, so
  no time-varying config is needed.

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
- **MoneyPuck** (`common/scrape/sources/moneypuck.py`) — **now working (2026-07-24)**.
  Previous block was network/ASN-level (not a bot-UA check); resolved by switching networks.
  All 18 seasons (2008–2025) fetched successfully: 81,355 skater rows, 8,510 goalie rows.
  Provides xG, shot-quality (high/medium/low danger), gameScore, and goalie GSAx features
  unavailable from the NHL API. Features are joined at the feature-engineering step and lagged
  before use (no leakage). Join key: `playerId` integer (same integer in both sources) + season
  start year — match rate ~94.5%. Remaining ~5.5% unmatched are fringe players with very few
  games, genuinely absent from MoneyPuck's data. Rate-limit handling added: 1s between year
  fetches, 3s between entities, 15s retry on 429.
  Data confirmed regular-season only: scraper uses `/regular/` in the URL path; MoneyPuck's
  `situation=="all"` aggregate row correctly sums across all regular-season game situations.
  Cross-source reconciliation (see `source_reconciliation.csv`) confirms near-perfect agreement
  with NHL API counting stats: Pearson r ≥ 0.9999 for goals, assists, points, games_played.
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

- **2026-07-24 — MoneyPuck integration complete**:
  - Network block resolved (different network). All 18 seasons fetched for skaters/goalies/teams.
  - Rate-limit handling added to scraper (1s inter-year, 3s inter-entity, 15s retry on 429).
  - MoneyPuck join switched from name-matching to player_id (exact integer, same in both sources):
    match rate 93% → 94.5%; remaining 5.5% genuinely absent from MoneyPuck's records.
  - Cross-source reconciliation script added (`common/diagnostics/reconcile.py`): NHL API and
    MoneyPuck agree at Pearson r ≥ 0.9999 on goals, assists, points, games_played. The few
    games_played outliers (e.g. Seth vs Caleb Jones in the same season) trace to historical
    surname collisions in MoneyPuck, not real data errors — the player_id join prevents these
    from affecting the model.
  - Data confirmed regular-season only: `/regular/` URL path + `situation=="all"` filter.
  - Re-ran `--stage predict` with xG features flowing into models for the first time.

- **2026-07-24 — Model diagnostics module added** (`common/diagnostics/model_report.py`):
  - `TrainedModel` now stores `y_test`/`y_pred_test` (persisted in joblib), so predicted-vs-actual
    plots are available on `--stage predict` without retraining.
  - `baseline_r2` added to every model's metrics: predict using training-set mean; any model
    with R² < baseline_r2 is flagged as `beats_baseline=False`.
  - Metrics table written to `results/diagnostics/model_metrics.csv` on every predict run.
  - Predicted-vs-actual + residual plots written to `plots/diagnostics/pva_*.png`.
  - `rolling_oos_eval()` added to `training.py` for gold-standard time-series OOS (train on
    years prior to holdout, predict holdout year); call with `--rolling-eval` when needed.
  - Key findings from first full diagnostic run: forward_points OOS R²=0.665 (healthy);
    defense_plus_minus OOS R²=0.031; goalie GAA OOS R²=-0.203, goalie Sv% OOS R²=-0.839
    (both worse than predicting the mean — flagged in the metrics table).

## Future milestones (not in scope for the 2026-27 rebuild)

- **Joint D-men model**: defenseman points directly contribute to team goals-for, which
  mechanically increases plus/minus. Worth exploring multi-output regression or predicting
  plus/minus residual after accounting for point contributions.
- **Goalie model improvement**: season-to-season save% and GAA have low predictability.
  Candidates: percentile-rank prediction instead of raw stat, stability-weighted career
  average baseline, Bayesian shrinkage toward league mean, or new features (save% by
  shot type from MoneyPuck high/medium/low danger breakdowns).
- **Majority-team for historical traded players**: game logs currently cover only the most
  recent season. Expanding to full history (one API call per player per historical season)
  would allow assigning each player to the team they played most games for in each historical
  season, rather than defaulting to the first listed team.
- **Interactive draft-day tool**: analyze and select players live during the draft. Lowest
  priority per original spec; nothing built yet.
- **2027-28 season**: copy `2026-27/config.yaml` → `2027-28/config.yaml`, update
  season/trade-override/roster-date fields, run `--stage all`. That's the full annual process.
