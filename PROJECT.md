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

  draft/
    historical_scoring.py     scores 18 seasons of actuals under pool rules
    historical_value_curves.py  aggregates seasons into F/D/G value curves
    pool_structure.py         snake-draft pick math, roster caps
    strategy_sim.py           Monte Carlo position-order policy comparison (POLICIES, reused live)
    report.py                 writes draft-strategy CSVs/report
    live_state.py             live draft state (duck-types strategy_sim's POLICIES interface)
    player_match.py           fuzzy/typo-tolerant player-name resolution for live entry
    live_repl.py               interactive command loop for live_draft.py

run_season.py       CLI entry point: python3 run_season.py --season 2026-27 --stage all|scrape|clean|train|predict

<season>/            one folder per draft year, e.g. 2026-27/
  config.yaml           season id, trade overrides, scoring rules, roster-snapshot date
  data/{raw,processed}/ scraped + cleaned data (gitignored)
  models/                persisted joblib model artifacts (gitignored)
  results/               rankings/reports csv/xlsx/txt (gitignored)
  plots/                 diagnostics (gitignored)
  results/diagnostics/   source_reconciliation.csv, model_metrics.csv
  plots/diagnostics/     pva_*.png (predicted-vs-actual + residuals per model)
  results/draft/         draft-strategy curves, policy comparison, round tables, live draft state (gitignored)
```

`draft_strategy.py` (repo root) is a separate CLI entry point: `python3 draft_strategy.py --season
2026-27 [--my-slot 1-9]`, backed by `common/draft/` (historical_scoring.py, historical_value_curves.py,
pool_structure.py, strategy_sim.py, report.py). It answers a different question from the rankings
pipeline above — not "which players," but "in what order should positions be drafted" — and reads
the rankings pipeline's outputs without modifying it.

`live_draft.py` (repo root) is the live, player-level draft-day assistant: `python3 live_draft.py
--season 2026-27 --my-slot <N>`, backed by `common/draft/live_state.py`, `player_match.py`, and
`live_repl.py`. Answers "which player should I take right now" during the actual draft, reusing
`strategy_sim.py`'s position-order policies unchanged against live pick data instead of the
historical Monte Carlo simulation. See "Running the tools" below for full usage.

`2025-26/` is last year's original ad-hoc build (24 near-duplicate model scripts, no shared
library, nothing persisted, no git tracking). It's kept as historical reference only — not
imported by anything in `common/`.

## Running the tools

All commands run from the repo root (`cd` there first):
```
cd /Users/rcejxd36/Library/CloudStorage/Dropbox/hockeyanalytics
```

**macOS note**: use `python3`, not `python` — on this machine (and modern macOS generally),
`python` isn't on `PATH` at all (`which python` finds nothing); only `python3` is. Every command
below uses `python3` for that reason.

**One-time setup** (install dependencies — only needed once, or after `requirements.txt` changes):
```
python3 -m pip install -r requirements.txt
```

**The three CLI tools**, in the order you'd typically use them:

1. **`run_season.py`** — the main pipeline (scrape data, clean it, train models, produce
   rankings). Run this first each season, before either draft tool:
   ```
   python3 run_season.py --season 2026-27 --stage all       # full pipeline
   python3 run_season.py --season 2026-27 --stage predict   # reuse trained models, just re-score
   ```
2. **`draft_strategy.py`** — position-order strategy (what order to draft *positions*), built
   from historical simulation, run once per season before draft day:
   ```
   python3 draft_strategy.py --season 2026-27
   python3 draft_strategy.py --season 2026-27 --my-slot 4    # + your slot's round-by-round table
   ```
3. **`live_draft.py`** — the live, player-level assistant, run *during* the actual draft (leave
   it running in a terminal window for the whole draft — it's interactive):
   ```
   python3 live_draft.py --season 2026-27 --my-slot 4
   ```
   Then type `pick <player name>` for every pick as it's announced (yours and every opponent's —
   this pool's draft is manual/verbal, so there's nothing to pull picks from automatically).
   Type `help` inside the tool for the full command list (`show`, `top`, `undo`, `quit`). State
   saves after every pick, so if you close the terminal or the laptop sleeps, just re-run the
   same command to pick up exactly where you left off.

## Pool scoring rules

Source of truth: `2025-26/instructions_for_claude_6Sept25.docx`, encoded per-season in each
season's `config.yaml` under `scoring:`.

- Forwards: goals + assists
- Defense: goals + assists + plus/minus
- Goalies (min 40 GP to qualify for bonuses): 1 pt/win, +3 bonus for a shutout win
  (additive on top of the win point — 4 total), +10 for best GAA among qualified goalies,
  +10 for best save % among qualified goalies

## Known modeling limitations (as of 2026-07-24)

- **Goalie GAA and save%**: OOS R² < 0, meaning the model predicts *worse* than predicting
  the training-set mean. These targets are fundamentally hard to predict from prior seasons;
  season-to-season goalie stats have high variance and low autocorrelation. In the pool,
  the bonus scoring only needs the argmax/argmin among 40+ GP qualifiers — the bonus is
  awarded to whoever scores highest/lowest, so even a weak model's ranking at the top is
  informative. But this should be revisited: consider predicting the percentile rank rather
  than the raw stat, or switching to a simpler "stability-weighted career average" baseline.
- **Defense plus/minus**: now modeled as a residual after removing the points contribution
  (2026-08-25, see changelog) — OOS R² improved 0.031→0.145, concordance ~0.54→~0.64 on the
  residual target itself. Still comparatively weak: contemporaneous team features were removed
  (NaN at prediction time, training/inference mismatch), and the residual formulation is an
  approximation, not an exact accounting identity (NHL plus/minus excludes power-play goals,
  points doesn't). Prediction blending (alpha=0.15 toward 2-year historical averages, recalibrated
  2026-08-25 from the *reconstructed* prediction's own OOS concordance ~0.57 — lower than the
  residual target's 0.64 alone, since errors from the independently-fit points and residual
  models compound when summed) corrects ranking at the extremes.
- **Traded players — team context**: players who changed teams during a historical season are
  assigned the *first* team listed in the NHL API's comma-joined `team_abbrev`. Their season
  totals (goals, assists, games_played) are preserved correctly; only the team-context features
  (used mainly by the defense plus/minus model) may be inaccurate. Per-team game logs are only
  available for the most recent season, so a majority-team fix across all 18 training seasons
  would require expanding game-log history (one API call per player per historical season).
- **Pool scoring rules are constant**: no year-to-year variation in the scoring system, so
  no time-varying config is needed.

## Known modeling tradeoffs (carried over deliberately, not bugs)

- Defense predictions are blended with 2-year historical averages to correct model compression
  at the extremes; all features are strictly lagged (no contemporaneous leakage). Blend weights
  (alpha) are calibrated from OOS concordance: `alpha = 2*(concordance − 0.5)` — alpha=0.45 for
  points (concordance ~0.72), alpha=0.15 for plus_minus (concordance ~0.57, computed on the
  reconstructed residual+points prediction, not the residual target's own ~0.64 — see the
  2026-08-25 changelog entry). See `pool_ranking._blend()`.
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

- **2026-07-24 — Goalie and defense model failure mode fixes** (commit b5873e1):
  - Removed QUALIFIED_ONLY training filter from goalies: restored the full training dataset
    (~1,100 rows vs. ~340 previously). QUALIFIED_ONLY was applied during training but not at
    prediction time, shrinking the training set for no benefit.
  - Added `_LINEAR_ONLY = ("ridge","lasso","elastic_net")` constant in `goalies.py`. GAA and
    save_pct targets now use only regularized linear models: near-zero season-to-season autocorrelation
    (r≈0.09) means tree models overfit badly, while linear models correctly regress toward the mean.
  - Added `add_career_averages()` for all goalie rate stats (save_pct, GAA, shutouts/game, wins/game)
    and for defense plus_minus and plus_minus_per_game. Career averages are leakage-safe (expanding
    mean with a shift-1 guard) and provide a stable multi-season signal for high-variance rate stats.
  - Removed contemporaneous team features from defense `build_xy_for()` (`allow_team=False`
    always): at prediction time, the current season's team stats are NaN → 0, creating a
    training/inference mismatch for any model that learned on real contemporaneous values.

- **2026-07-24 — Empirical Bayes shrinkage, GSAx features, rank-based OOS metrics** (commit 16637a6):
  - `add_eb_save_pct()` added to `common/features/engineering.py`. Computes a Beta-Binomial
    posterior mean save%: `eb_save_pct = (α + saves) / (α + β + shots)`. Prior α and β are fitted
    by method of moments on seasons with ≥100 shots; fallback prior α=85.0, β=8.5 (centered at 0.909).
    Shrinks backup goalies aggressively toward the league mean; trusts high-volume starters.
    Community standard approach per Thomas (2006). `eb_save_pct` is added to `_ENGINEERED_MARKERS`
    so `select_feature_columns` allowlists it, and is lagged + career-averaged.
  - GSAx added in `goalies.py` `engineer()`: `gsax = mp_xGoals − goals_against_player` from
    MoneyPuck columns. Positive = goalie outperformed expected goals allowed given shot difficulty.
    More persistent than raw save% (literature: r≈0.15–0.30 vs 0.05–0.15).
  - `hd_shot_pct = mp_highDangerShots / shots_against`: controls for the shot-mix the defense
    allowed, separating goalie skill from team-quality effects.
  - `rank_metrics()` added to `common/models/training.py`: Spearman ρ, Kendall τ →
    concordance = (τ+1)/2, directional accuracy, and bias. Applied universally to all 7 model
    targets (forwards, defense, goalies). Motivation: R²≈0 doesn't mean the model has no ranking
    value — pool draft is fundamentally a ranking problem, not a regression problem.
  - `top1_match_max` and `top1_match_min` added to OOS evaluation: does the model's argmax/argmin
    match the actual bonus winner? Expected to be low in a single holdout year; more meaningful
    in rolling OOS across many years.
  - All new metrics appear in `model_metrics.csv` and the console summary.

- **2026-07-24 — Per-player projected games for forwards and defense** (commit d1a50f6):
  - `_projected_games()` in `pool_ranking.py` generalized from goalies-only to any position.
  - Forwards and defense now use a 3-year rolling average of actual games_played_player as their
    projected GP, capped at `cfg.games_per_season` (was flat 84 for all skaters).
  - Fallback: 70 games for players with no recent history (vs 20 for goalies).
  - `pool_points` now reflects injury risk and lineup uncertainty rather than assuming every
    player plays a full season.

- **2026-07-24 — Full-season comparison column** (commit e0e978c):
  - `pool_points_full_season` added to all position DataFrames: pool score assuming all players
    play the full 84-game season (flat projected_games = `cfg.games_per_season`).
  - Both `pool_points` (per-player projected GP) and `pool_points_full_season` appear in all
    output CSVs, so the user can compare injury-risk-adjusted vs pure-value rankings side by side.

- **2026-07-24 — Fix downward GP bias for late-season rookie call-ups; add GP diagnostic** (commit f7fe145):
  - Root cause: `_projected_games()` included all seasons in the 3-year rolling average.
    A player like Lane Hutson with a 15-game debut call-up followed by full seasons was projected
    at ~60 games instead of ~82 — the partial debut season dragged the average down.
  - Fix: historical seasons with GP < 25 are now excluded from the rolling window; the most-recent
    season is always included regardless of GP so genuine injury risk still reduces the projection.
    Lane Hutson confirmed fixed: proj=82 (ratio=1.0 vs last season). No external data sources needed.
  - Added `_last_season_gp()` helper and `_write_gp_diagnostic()` to `pool_ranking.py`.
    On every `--stage predict` run: logs any player where `projected_games < 70% of last_season_gp`
    and writes `results/diagnostics/gp_projection_check.csv` (all 3,397 players sorted by ratio).
  - Result: 13 players remain flagged — all genuine irregular-career cases (suspensions, injury
    history, retirements) rather than residual call-up artifacts. Useful pre-draft review list.

- **2026-07-24 — Defense prediction blending** (commit 7525bd2):
  - Diagnosed two root causes of elite D-men ranking too low: (1) gradient boosting compresses
    top-end predictions — e.g. Makar's lag1 points/game = 1.053, but the model predicted 0.762;
    (2) elastic net predicted near-constant plus_minus for everyone, erasing meaningful signal.
  - Fix: `_blend(df, pred_col, ref_col, alpha)` helper in `pool_ranking.py` blends model
    predictions with 2-year historical per-game averages: `alpha * model_pred + (1−alpha) * hist_avg`.
  - Alpha calibrated from OOS concordance using `alpha = 2*(concordance − 0.5)`:
    - `defense_points`: alpha=0.45 (OOS concordance ~0.72; model trusted, hist corrects compression)
    - `defense_plus_minus`: alpha=0.10 (OOS concordance ~0.54; model barely beats chance; hist dominates)
  - Reference columns: `points_per_game_hist_avg` and `plus_minus_per_game_hist_avg` (2-year lags
    already in the engineered data). Falls back to model prediction for new players with no history.
  - Result: Makar moved from #44 → #4 overall. Top-10 as of 2026-07-24:
    MacKinnon (F), Kucherov (F), McDavid (F), Makar (D), Draisaitl (F),
    Pastrnak (F), Bouchard (D), Suzuki (F), Necas (F), Celebrini (F).

- **2026-08-24 — Draft position-order strategy tool** (`common/draft/`, `draft_strategy.py`):
  - Answers a question distinct from player rankings: given the pool's 9-team snake draft and
    roster rules (7F/3D/1G starters + 3F/2D/1G bench=IR per team — bench/IR counted at full value
    since it's substitutable into the lineup twice a week, unlimited for injuries), in what order
    should *positions* (not specific players) be drafted? Deliberately player-identity-free; the
    live, player-tracking draft-day assistant (P6 below) remains separately deferred.
  - Value curves per position (F/D/G) are built from **18 real NHL seasons (2008-09..2025-26)**,
    scored under the pool's exact rules by reusing `pool_ranking.compute_pool_points()` unchanged
    (only the input columns are adapted from historical actuals). Three non-82-game seasons
    (2012-13 lockout: 48 GP; 2019-20 COVID: ~70 GP; 2020-21 COVID: 56 GP) are rescaled to an
    82-game-equivalent pace, which also proportionally scales the goalie bonus-qualification
    threshold so a goalie who played every game of a shortened season isn't penalized against the
    fixed 40-GP bar. Historical curve shape cross-checked against the current 2026-27 projections:
    normalized-shape Pearson r = 1.00 (forward), 1.00 (defense), 0.95 (goalie).
  - Monte Carlo simulation (1,500 drafts/policy/slot) compares three candidate drafting policies
    — naive best-value, need-balanced pacing, and scarcity/urgency-aware — against opponents
    modeled as best-player-available-among-remaining-needs.
  - **Result: the need-balanced pacing policy (spread F/D/G picks proportional to final roster
    targets throughout the draft) won for all 9 draft slots**, beating both naive best-value and
    pure scarcity-chasing. Scarcity findings: goalies have only a 51% value drop across the whole
    draftable range but a steep 37% further cliff in the 15 picks just past the cutoff (secure a
    goalie early); defense drops steepest overall (61%) but has more cushion past its cutoff
    (17%); forwards are safely deferrable (7% cushion past cutoff).
  - Full writeup with tables: `docs/hockey_pool_pipeline.tex` §12 "Draft Position-Order Strategy".

- **2026-08-25 — Goalie shutout scoring fix, observation weighting, joint models** (model-fit
  pass preceding the deferred P3 interactive draft-day tool; see Future milestones below):
  - **Scoring bug fix**: the goalie shutout bonus was implemented as "3 points total for a
    shutout win" (not additive on top of the win point) — this was the documented rule, but the
    user confirmed the real pool rule is additive: 1 (win) + 3 (shutout bonus) = **4 total**.
    Fixed in `compute_pool_points()` (`pool_ranking.py`): `win_pts * wins + sut_pts * shutouts`
    (was `win_pts * (wins - shutouts) + sut_pts * shutouts`). Verified: a goalie with 10 wins (3
    shutouts) now scores 19 pts (7×1 + 3×4), not the old 16. `historical_scoring.py` reuses this
    function unchanged, so the draft-strategy value curves picked up the fix automatically —
    re-ran `draft_strategy.py` and updated `docs/hockey_pool_pipeline.tex` §12 accordingly (see
    below).
  - **P1 (observation weighting)**: threaded an optional `sample_weight` through `_fit_one()`,
    `train_and_select()`, and `train_all_vs_exclude_latest()` in `training.py` (goalies only;
    `defense.py`/`forwards.py` unaffected). Goalie `build_xy_for()` computes
    `weight = min(GP, 40) / 40`. Weight affects only the estimator `.fit()` call (via
    `search.fit(X_tr, y_tr, sample_weight=...)`); `SelectKBest` feature selection and reported
    test/OOS metrics stay unweighted for comparability. Result: goalie GAA OOS R² −0.203 → −0.062,
    concordance now 0.53 (real ranking signal). Save% OOS R² −0.839 → −0.406, but remained a
    degenerate constant predictor (rank metrics undefined) — didn't suffice alone, triggering P2.
  - **P2 (joint GAA/save% model)**: added `MultiTaskElasticNetCV` as an additional per-target
    candidate in `goalies.py` (`_fit_joint_gaa_save_pct` / `_joint_train_all_vs_exclude`), fit
    on the full (unweighted — MultiTaskElasticNet doesn't support `sample_weight`) feature set
    with shared sparsity across GAA and save% (r=−0.84 algebraic link). Each task's output column
    is wrapped in a `_JointTaskEstimator` proxy so it slots into an ordinary `TrainedModel`
    (persistence/plotting/`pool_ranking.py` prediction call sites unchanged). Chosen per-target
    only if it beats the single-task model's `selection_score`. Result: won for save% — test R²
    −0.221 → −0.046 (now beats baseline), OOS R² −0.406 → −0.185, concordance 0.52 (was
    undefined). GAA kept its single-task (P1-weighted) model, which already won there.
  - **P3 (joint D-men model, residual-after-points)**: `defense.py`'s `plus_minus` target
    replaced with `plus_minus_residual_per_game = plus_minus_per_game - points_per_game`,
    trained via the existing single-target harness (no harness changes needed — lagged/historical
    points features were already legitimate plus_minus-model inputs). `pool_ranking.py`
    reconstructs the full prediction (`pred_plus_minus_residual + pred_points`) before the
    existing hist-avg blend. Approximation, not an exact identity (NHL plus/minus excludes
    power-play goals, points doesn't) — noted in `defense.py`'s docstring. Result: OOS R² 0.031 →
    0.145, concordance ~0.54 → 0.64. Top D-men rankings still hockey-plausible post-fix (Makar #1,
    Bouchard #2, Hutson #3) — see `[[defense_dmen_undervalued]]` memory, this was a recurring
    concern and didn't regress.
  - **Blend alpha recalibration**: the plus_minus blend alpha was still hardcoded at 0.10
    (calibrated pre-P3, from the raw target's ~0.54 concordance). Recomputed properly post-P3:
    the *reconstructed* prediction (`pred_plus_minus_residual + pred_points`, using genuine OOS
    exclude-latest-season predictions from both underlying models) has its own OOS concordance
    of only ~0.57 against actual plus_minus_per_game — lower than the residual target's own 0.64,
    because prediction errors from the two independently-fit models (points, residual) compound
    when summed. `alpha = 2*(0.57-0.5) ≈ 0.15` (was 0.10). Updated in `pool_ranking.py`.
  - Persisted model key renamed `defense_plus_minus` → `defense_plus_minus_residual`
    (`pool_ranking.py`'s `_load_or_train` specs, `model_report.py`'s `_FRIENDLY` display map); the
    stale orphaned `defense_plus_minus.joblib` artifact was deleted.
  - Re-ran `--stage train` (full retrain) and `draft_strategy.py` end-to-end; no pipeline errors.
    The shutout fix raised goalie value enough to flip the winning draft policy at 2 of 9 slots
    (slots 1 and 4 now favor `urgency_greedy` over `balanced_need`, by <0.5 pts out of ~1070-1080
    — a practical statistical tie, not a robust preference); the other 7 slots are unaffected.
    `docs/hockey_pool_pipeline.tex` §10.3, §11 (crosscheck/scarcity tables), and §12 (policy
    table, result statement, and the round-by-round example — switched from slot 4 to slot 5
    since slot 4 is no longer a clean `balanced_need` win) updated to match; PDF rebuilt.
  - **Out of scope this pass** (per priority-ordered discussion): P4 (GP diagnostic Layer 2) —
    no outstanding edge cases to fix; P5 (majority-team for traded players) — data-completeness
    work, not model-fit; P6 (interactive draft-day tool) — deferred pending further discussion.

- **2026-08-25 — Interactive draft-day tool** (`live_draft.py`, `common/draft/live_state.py`,
  `common/draft/player_match.py`, `common/draft/live_repl.py`):
  - The last item on the Future Work list. Answers a question distinct from both the player-
    ranking pipeline ("which players") and the position-order strategy tool ("what order to
    draft positions"): "which specific player should I take right now," live, during the actual
    draft. The league's draft is manual/verbal with no external platform, so every pick — the
    user's own and all 8 opponents' — is entered by hand as it happens; this is a personal,
    single-user CLI tool, not a shared/multi-user app.
  - **Key reuse finding**: `common/draft/strategy_sim.py`'s position-choice policy functions
    (`_value_greedy`/`_balanced_need`/`_urgency_greedy`, exposed via `POLICIES`) are already
    duck-typed against a `state` object exposing exactly six members
    (`next_value`/`value_at_offset`/`my_counts`/`caps`/`remaining_need`/`intervening_picks`) —
    none of them touch the Monte Carlo simulation's internals directly. A new `_LiveState` class
    (`live_state.py`) implements the same six members backed by live data instead of a historical
    value curve, so `POLICIES` is reused **completely unchanged** — zero edits to
    `strategy_sim.py`. `common/draft/pool_structure.py` (`DraftConfig`, `team_at_pick`,
    `picks_for_slot`) is likewise reused unchanged.
  - `player_match.py`: stdlib-only (`difflib`) fuzzy name resolution for live entry under time
    pressure — exact match, then partial-token containment, then typo-tolerant fallback:
    `Unique`/`Ambiguous`/`AlreadyTaken`/`NotFound`. Handles real ambiguous cases in the actual
    2026-27 data (e.g. two players named "Elias Pettersson"; "Makar" matching both Cale Makar
    and an unrelated depth forward) with a numbered disambiguation prompt.
  - `live_repl.py`: a REPL (not per-pick script re-invocation, to avoid reloading/replaying state
    from disk on every keystroke over a ~1-2hr, 153-pick draft) with `pick <name>` (team implied
    by turn order — the primary, lowest-friction path), `pick @<team> <name>` (out-of-order
    correction), `show`, `top <position> [n]`, `undo` (with confirm), `help`, `quit`. On the
    user's own turn, auto-prints a recommendation panel: the winning policy for their slot (read
    from `draft_strategy.py`'s already-computed `results/draft/policy_comparison.csv`, default
    `balanced_need` with a warning if that file doesn't exist), current roster fill vs. cap, the
    policy's recommended position, and the top-N real available players there (by actual
    `pool_points`, plus a shorter list per other eligible position).
  - State persists to `results/draft/live_draft_state.json` (already covered by the existing
    `20*-*/results/*` gitignore rule) as an **event log**, not aggregated state — `undo` is just
    pop-and-replay, and resume replays the whole log through the same `apply()` path a live pick
    would use, so persisted and in-memory state can never diverge. Written atomically
    (temp file + `Path.replace()`) so a crash mid-write can't corrupt it.
  - Added a first-ever pytest suite (`tests/`, `pytest>=8.0` added to `requirements.txt`) for the
    resolver and state/replay logic — a deliberate, scoped departure from this repo's no-test-
    suite convention, justified by how costly a subtle bug in name-matching or undo/resume would
    be live during an actual draft. 22 tests, including a parametrized check that all three
    `POLICIES` functions run correctly against `_LiveState` — the core reuse claim above.
  - Verified end-to-end without a real draft: a scripted mock draft (round-robin eligible
    positions, top-available player each pick, replaying `team_at_pick` order) ran all 153 picks
    cleanly via stdin, produced the correct final roster (10F/5D/2G, exactly matching caps); a
    40-pick-in/quit/resume split produced an identical final roster to the unbroken run; cap-
    rejection and ambiguous-name disambiguation (including the real "Elias Pettersson" and
    "Makar" cases above) verified interactively.
  - A design/algorithm writeup (matching e.g. §9's "Prediction Blending for Defense" level of
    detail) is deliberately deferred until after first real-draft use, per the existing Known
    Limitations note that logging an actual draft would enable future empirical calibration.
  - **2026-08-25 (later same day) — runbook added**: a new "Running the tools" section added to
    both `PROJECT.md` (below) and `docs/hockey_pool_pipeline.tex` §2.3 "Environment and running
    the tools" — working directory, a macOS-specific note to use `python3` (`python` isn't on
    `PATH` at all on this machine — confirmed via `which python`), one-time `pip install`, and
    example invocations for all three CLI tools (`run_season.py`, `draft_strategy.py`,
    `live_draft.py`). The tex/PROJECT.md folder-layout listings were also updated to include
    `live_draft.py` and the three new `common/draft/live_*.py` modules, which were missing.

## Future milestones (not in scope for the 2026-27 rebuild)

Listed in priority order. (P1-P3 from the prior list — observation weighting, joint GAA/save%,
and the joint D-men model — are done; see the 2026-08-25 changelog entries above. The interactive
draft-day tool, formerly P3, is also done — see below.)

- **P1 — GP diagnostic Layer 2 (game-log debut-date confirmation)**: for any player whose debut
  season has GP < 50, fetch the game log via the existing `get_player_game_log()` endpoint and
  check `min(gameDate)`. A first game after February 1 is definitively a late call-up rather than
  injury. Deferred because the GP < 25 threshold filter already handles all known cases; implement
  if edge cases emerge from `gp_projection_check.csv`.
- **P2 — Majority-team for historical traded players**: game logs currently cover only the most
  recent season. Expanding to full history (one API call per player per historical season) would
  assign each player to the team they played most games for, making team-context lag features more
  accurate for the ~5–10% of rows involving mid-season trades. High API cost for marginal gain.
- **2027-28 season**: copy `2026-27/config.yaml` → `2027-28/config.yaml`, update
  season/trade-override/roster-date fields, run `--stage all`. That's the full annual process.
