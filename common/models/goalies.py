"""Goalie model: quadruple target with the 40+ GP qualification rule.

Targets: wins/game and shutouts/game for all goalies; GAA and save% for qualified goalies only
(>= min_gp_for_bonus games, per config, matching the pool's bonus-eligibility rule — a goalie
who barely played shouldn't win the best-GAA/best-save% bonus on a tiny sample). All four are
strict lagged-only (goalie legacy v2.0 used strict exclusions throughout).

Pool scoring: 1 pt/win, +3 bonus for a shutout win (additive, 4 total), +10 best GAA and
+10 best save% among qualified goalies (applied in pool_ranking, not here).

GAA/save_pct also get two extra treatments beyond the shared single-target harness (see
`build_xy_for` and `_joint_train_all_vs_exclude` below): GP-based observation weighting, and
(only if that alone isn't enough) a joint MultiTaskElasticNetCV candidate.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import MultiTaskElasticNetCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

from common.config import SeasonConfig
from common.features import engineering as fe
from common.models import training as tr
from common.models.training import TrainedModel, rank_metrics, selection_score

logger = logging.getLogger(__name__)

TARGETS = {"wins": "wins_per_game", "shutouts": "shutouts_per_game",
           "gaa": "goals_against_average", "save_pct": "save_percentage"}
QUALIFIED_ONLY = ("gaa", "save_pct")
_TARGET_COLS = list(TARGETS.values()) + ["wins_player", "shutouts", "goals_against_avg",
                                         "save_pct", "total_wins", "total_shutouts"]

_INDIVIDUAL = ["wins_player", "losses_player", "ot_losses_player", "shutouts",
               "goals_against_player", "goals_against_avg", "save_pct", "saves", "shots_against",
               "games_played_player", "games_started", "time_on_ice",
               "wins_per_game", "shutouts_per_game", "goals_against_average", "save_percentage",
               "total_wins", "total_shutouts",
               "eb_save_pct", "gsax", "hd_shot_pct"] + fe.DERIVED_TEAM_COLS


def _min_gp(cfg: SeasonConfig) -> int:
    return int(cfg.scoring.get("goalies", {}).get("min_gp_for_bonus", 40))


def _create_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["total_wins"] = pd.to_numeric(df["wins_player"], errors="coerce").fillna(0).clip(lower=0)
    df["total_shutouts"] = pd.to_numeric(df["shutouts"], errors="coerce").fillna(0).clip(lower=0)
    df = fe.per_game_target(df, "total_wins", "wins_per_game")
    df = fe.per_game_target(df, "total_shutouts", "shutouts_per_game")
    df["goals_against_average"] = pd.to_numeric(df["goals_against_avg"], errors="coerce").clip(0, 10).fillna(3.0)
    sv = pd.to_numeric(df["save_pct"], errors="coerce")
    sv = sv / 100 if sv.max(skipna=True) and sv.max(skipna=True) > 1 else sv
    df["save_percentage"] = sv.clip(0.5, 1.0).fillna(0.900)
    return df


def engineer(df: pd.DataFrame, cfg: SeasonConfig) -> pd.DataFrame:
    df = fe.add_year(df)
    df = _create_targets(df)
    df = fe.add_covid_indicators(df)
    df = fe.add_years_played(df, prime_range=(3, 12))
    df = fe.add_team_goal_differential(df)
    df = fe.clean_common_columns(df, toi_default=0.0)
    df = fe.add_eb_save_pct(df)
    df = fe.join_moneypuck(df, cfg, "goalies")
    # GSAx: expected goals against − actual; positive = goalie outperformed shot difficulty
    if "mp_xGoals" in df.columns and "goals_against_player" in df.columns:
        df["gsax"] = (pd.to_numeric(df["mp_xGoals"], errors="coerce")
                      - pd.to_numeric(df["goals_against_player"], errors="coerce"))
    # High-danger shot fraction: controls for shot mix the team's defense allowed
    if "mp_highDangerShots" in df.columns and "shots_against" in df.columns:
        df["hd_shot_pct"] = (pd.to_numeric(df["mp_highDangerShots"], errors="coerce")
                             / pd.to_numeric(df["shots_against"], errors="coerce").replace(0, np.nan))
    lags = fe.lag_feature_list(df, _INDIVIDUAL)
    df = fe.create_lag_features(df, lags, cfg.lag_years, fe.resolve_min_training_year(df, cfg))
    df = fe.add_career_averages(df, ["save_pct", "goals_against_avg", "shutouts",
                                     "wins_per_game", "save_percentage", "goals_against_average",
                                     "shutouts_per_game", "eb_save_pct", "gsax"])
    return fe.engineer_features(df, cfg.lag_years,
                                interaction_metrics=("wins_per_game", "save_pct", "goals_against_avg", "shutouts"),
                                covid_metrics=("wins_per_game", "save_pct"))


def build_xy_for(df: pd.DataFrame, target: str, cfg: SeasonConfig):
    frame = df
    cols = fe.select_feature_columns(frame, _TARGET_COLS, allow_contemporaneous_team=False)
    X, y = fe.build_xy(frame, cols, TARGETS[target],
                       core_lag_prefixes=("wins_player_lag", "games_played_player_lag"))
    # Down-weight low-GP goalie-seasons (1-5 start backups are mostly noise) so they influence
    # the fit less than a full-season starter, without dropping them from training entirely.
    gp = pd.to_numeric(frame.loc[X.index, "games_played_player"], errors="coerce").fillna(0)
    weight = gp.clip(upper=40) / 40
    return X, y, frame.loc[X.index, "year"], weight


_LINEAR_ONLY = ("ridge", "lasso", "elastic_net")
_JOINT_TASKS = ("gaa", "save_pct")


class _JointTaskEstimator:
    """Proxy exposing one task's predict()/coef_ from a shared MultiTaskElasticNetCV fit, so it
    can be wrapped in an ordinary TrainedModel — persistence, plotting, and prediction all work
    unchanged, they just see a normal single-output estimator."""

    def __init__(self, joint_estimator: MultiTaskElasticNetCV, task_index: int):
        self._joint = joint_estimator
        self._task_index = task_index
        self.coef_ = joint_estimator.coef_[task_index]

    def predict(self, X):
        return self._joint.predict(X)[:, self._task_index]


def _fit_joint_gaa_save_pct(frame: pd.DataFrame, cfg: SeasonConfig) -> dict[str, TrainedModel]:
    """Fit GAA and save_pct jointly via MultiTaskElasticNetCV, enforcing shared feature sparsity
    across the algebraically-linked pair (r=-0.84): a feature is selected for both targets or
    neither. Uses the full engineered feature set (no SelectKBest pre-filter) since the model's
    own L1 penalty already does feature selection, and pre-filtering by each target's marginal
    correlation could drop a feature only useful jointly.

    Note: MultiTaskElasticNet doesn't support sample_weight, so unlike the single-task models
    this path trains unweighted — a real limitation, but the point of trying it is to test
    whether shared sparsity transfers GAA's (now GP-weighting-improved) signal to save_pct, not
    to re-derive the weighting benefit itself.
    """
    X_gaa, y_gaa, order, _ = build_xy_for(frame, "gaa", cfg)
    X_sv, y_sv, _, _ = build_xy_for(frame, "save_pct", cfg)
    assert X_gaa.index.equals(X_sv.index), "gaa/save_pct row sets diverged unexpectedly"
    X = X_gaa
    Y = pd.DataFrame({"gaa": y_gaa, "save_pct": y_sv}, index=X.index)

    scaler = RobustScaler()
    Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    idx = np.argsort(np.asarray(order), kind="stable")
    Xs, Y = Xs.iloc[idx], Y.iloc[idx]

    test_size = min(0.25, max(0.15, 40 / len(Xs)))
    split = int(len(Xs) * (1 - test_size))
    X_tr, X_te = Xs.iloc[:split], Xs.iloc[split:]
    Y_tr, Y_te = Y.iloc[:split], Y.iloc[split:]

    folds = min(5, max(3, len(X_tr) // 30))
    model = MultiTaskElasticNetCV(cv=folds, random_state=42, max_iter=5000, n_jobs=-1)
    model.fit(X_tr, Y_tr)
    pred_tr, pred_te = model.predict(X_tr), model.predict(X_te)

    out = {}
    for i, task in enumerate(_JOINT_TASKS):
        yt_tr, yp_tr = Y_tr[task].to_numpy(), pred_tr[:, i]
        yt_te, yp_te = Y_te[task].to_numpy(), pred_te[:, i]
        train_r2 = r2_score(yt_tr, yp_tr)
        test_r2 = r2_score(yt_te, yp_te)
        baseline_pred = np.full(len(yt_te), float(yt_tr.mean()))
        metrics = {
            "r2": test_r2, "train_r2": train_r2,
            "overfitting_ratio": (train_r2 - test_r2) / train_r2 if train_r2 > 0 else 0.0,
            "mae": mean_absolute_error(yt_te, yp_te),
            "rmse": float(np.sqrt(mean_squared_error(yt_te, yp_te))),
            "cv_mean": float("nan"), "n_train": len(X_tr), "n_test": len(X_te),
            "baseline_r2": r2_score(yt_te, baseline_pred),
            **rank_metrics(yt_te, yp_te),
        }
        out[task] = TrainedModel(
            target=task, model_type="multitask_elastic_net",
            estimator=_JointTaskEstimator(model, i), selected_features=list(X.columns),
            scaler=scaler, metrics=metrics,
            best_params={"alpha": float(model.alpha_), "l1_ratio": float(model.l1_ratio_)},
            y_test=yt_te, y_pred_test=yp_te)
    return out


def _joint_train_all_vs_exclude(eng: pd.DataFrame, cfg: SeasonConfig) -> dict[str, dict]:
    """Mirrors `training.train_all_vs_exclude_latest` for the 2-task joint candidate: fit on all
    data and on data excluding the latest season, score the latter OOS on the held-out season,
    and return per-task results shaped like the single-task path so `train()` can compare
    `selection_score()` head-to-head and keep whichever wins, per target.
    """
    latest = int(eng["year"].max())
    joint_all = _fit_joint_gaa_save_pct(eng, cfg)
    out = {t: {"all": joint_all[t], "exclude_latest": None, "oos": None} for t in _JOINT_TASKS}

    excl = eng[eng["year"] < latest]
    if excl["year"].nunique() >= 2:
        joint_ex = _fit_joint_gaa_save_pct(excl, cfg)
        oos_frame = eng[eng["year"] == latest]
        for t in _JOINT_TASKS:
            out[t]["exclude_latest"] = joint_ex[t]
            X_oos, y_oos, _, _ = build_xy_for(oos_frame, t, cfg)
            if len(X_oos) > 0:
                pred = joint_ex[t].predict(X_oos)
                y_arr = np.asarray(y_oos)
                oos_met = {"r2": r2_score(y_arr, pred) if len(y_arr) > 1 else float("nan"),
                          "mae": mean_absolute_error(y_arr, pred), "n": len(y_arr),
                          **rank_metrics(y_arr, pred)}
                if len(y_arr) >= 3:
                    oos_met["top1_match_max"] = int(pred.argmax() == y_arr.argmax())
                    oos_met["top1_match_min"] = int(pred.argmin() == y_arr.argmin())
                out[t]["oos"] = oos_met

    for t in _JOINT_TASKS:
        ex = out[t]["exclude_latest"]
        if ex is not None and selection_score(ex.metrics) > selection_score(joint_all[t].metrics):
            out[t]["chosen"], out[t]["model"] = "exclude_latest", ex
        else:
            out[t]["chosen"], out[t]["model"] = "all", joint_all[t]
    return out


def train(df_raw: pd.DataFrame, cfg: SeasonConfig, persist: bool = True) -> dict:
    eng = engineer(df_raw, cfg)
    out = {}
    for target in TARGETS:
        # Restrict GAA and save_pct to linear models: near-zero autocorr means tree models
        # overfit badly on this data, while regularized linear models regress toward the mean.
        kwargs = {"model_types": _LINEAR_ONLY} if target in QUALIFIED_ONLY else {}
        try:
            res = tr.train_all_vs_exclude_latest(
                lambda f, t=target: build_xy_for(f, t, cfg), eng, target, cfg, **kwargs)
        except (ValueError, RuntimeError) as e:
            out[target] = {"model": None, "error": str(e)}
            continue
        out[target] = res

    # Try a joint MultiTaskElasticNetCV over (gaa, save_pct): only worth it because GP-weighting
    # alone (build_xy_for) still leaves save_pct a degenerate constant predictor. Keep whichever
    # candidate wins by selection_score, independently per target.
    if all(out.get(t, {}).get("model") is not None for t in _JOINT_TASKS):
        try:
            joint = _joint_train_all_vs_exclude(eng, cfg)
            for t in _JOINT_TASKS:
                s_joint, s_single = selection_score(joint[t]["model"].metrics), selection_score(out[t]["model"].metrics)
                if s_joint > s_single:
                    logger.info("Joint multitask model wins for %s (score=%.3f > single-task %.3f)",
                               t, s_joint, s_single)
                    out[t] = joint[t]
        except Exception as e:
            logger.warning("Joint gaa/save_pct model failed, keeping single-task models: %s", e)

    if persist:
        for target, res in out.items():
            if res.get("model") is not None:
                tr.save_model(res["model"], cfg, f"goalie_{target}")
    return out
