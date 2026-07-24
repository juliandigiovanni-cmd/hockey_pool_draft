"""Shared train / tune / select / persist harness for every position family.

Replaces the training code that was copy-pasted across the 24 legacy scripts. One place now
owns: the 5-algorithm zoo (RF / GBM / Ridge / Lasso / ElasticNet), hyperparameter search with
time-aware CV, SelectKBest feature selection, the model-selection score, and — new this year —
joblib persistence so tuned models are saved once and reused by the pool-ranking step instead
of being retrained (and worse, replaced by a simpler RF) at integration time.

Model-selection score (kept from legacy, lightly documented): `r2 - 0.1 * overfitting_ratio`,
where overfitting_ratio = (train_r2 - test_r2) / train_r2. This rewards test-set fit while
penalizing models whose train score runs far ahead of their test score.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     TimeSeriesSplit, train_test_split)
from sklearn.preprocessing import RobustScaler

from common.config import SeasonConfig

logger = logging.getLogger(__name__)

LINEAR_MODELS = ("ridge", "lasso", "elastic_net")
DEFAULT_MODEL_TYPES = ("random_forest", "gradient_boosting", "ridge", "lasso", "elastic_net")


def model_zoo() -> dict[str, dict]:
    """The 5 candidate algorithms and their search spaces (consolidated from the family scripts)."""
    return {
        "random_forest": {
            "model": RandomForestRegressor(random_state=42, n_jobs=-1),
            "params": {"n_estimators": [150, 250], "max_depth": [12, 18],
                       "min_samples_split": [15, 25], "min_samples_leaf": [8, 12],
                       "max_features": ["sqrt", 0.4]},
            "search": "randomized", "n_iter": 8},
        "gradient_boosting": {
            "model": GradientBoostingRegressor(random_state=42, validation_fraction=0.2,
                                               n_iter_no_change=10),
            "params": {"n_estimators": [150, 250], "max_depth": [3, 5],
                       "learning_rate": [0.05, 0.1], "subsample": [0.8, 0.9],
                       "min_samples_split": [15, 25], "max_features": ["sqrt", 0.4]},
            "search": "randomized", "n_iter": 10},
        "ridge": {"model": Ridge(random_state=42),
                  "params": {"alpha": [10.0, 50.0, 100.0, 500.0]}, "search": "grid"},
        "lasso": {"model": Lasso(random_state=42, max_iter=3000),
                  "params": {"alpha": [0.1, 0.5, 1.0, 5.0]}, "search": "grid"},
        "elastic_net": {"model": ElasticNet(random_state=42, max_iter=3000),
                        "params": {"alpha": [0.1, 1.0, 5.0], "l1_ratio": [0.3, 0.5, 0.7]},
                        "search": "grid"},
    }


@dataclass
class TrainedModel:
    """A fitted model plus everything needed to reproduce its prediction path. Persisted via joblib."""
    target: str
    model_type: str
    estimator: Any
    selected_features: list[str]
    scaler: Any | None
    metrics: dict
    best_params: dict = field(default_factory=dict)
    y_test: np.ndarray | None = None       # held-out actuals from training split (for diagnostics)
    y_pred_test: np.ndarray | None = None  # corresponding predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        Xs = X.reindex(columns=self.selected_features).apply(pd.to_numeric, errors="coerce").fillna(0)
        if self.scaler is not None:
            Xt = pd.DataFrame(self.scaler.transform(Xs), columns=self.selected_features, index=Xs.index)
            return self.estimator.predict(Xt)
        return self.estimator.predict(Xs)


def selection_score(metrics: dict) -> float:
    return metrics["r2"] - 0.1 * metrics["overfitting_ratio"]


def _make_cv(n: int, time_order: np.ndarray | None):
    folds = min(5, max(3, n // 30))
    if time_order is not None:
        return TimeSeriesSplit(n_splits=folds)
    return KFold(n_splits=folds, shuffle=True, random_state=42)


def _fit_one(X: pd.DataFrame, y: pd.Series, model_type: str, time_order: np.ndarray | None) -> dict:
    cfg = model_zoo()[model_type]
    scaler = None
    Xf = X
    if model_type in LINEAR_MODELS:
        scaler = RobustScaler()
        Xf = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    order = None
    if time_order is not None:
        order = np.argsort(time_order, kind="stable")
        Xf, y = Xf.iloc[order], y.iloc[order]

    test_size = min(0.25, max(0.15, 40 / len(Xf)))
    # For time-ordered data, hold out the most recent block; otherwise random split.
    if time_order is not None:
        split = int(len(Xf) * (1 - test_size))
        X_tr, X_te, y_tr, y_te = Xf.iloc[:split], Xf.iloc[split:], y.iloc[:split], y.iloc[split:]
        cv_order = np.sort(time_order)[:split]
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(Xf, y, test_size=test_size, random_state=42)
        cv_order = None

    cv = _make_cv(len(X_tr), cv_order)
    common = dict(cv=cv, scoring="r2", n_jobs=-1)
    if cfg["search"] == "randomized":
        search = RandomizedSearchCV(cfg["model"], cfg["params"], n_iter=cfg["n_iter"],
                                    random_state=42, **common)
    else:
        search = GridSearchCV(cfg["model"], cfg["params"], **common)
    search.fit(X_tr, y_tr)

    est = search.best_estimator_
    y_pred = est.predict(X_te)
    train_r2 = est.score(X_tr, y_tr)
    test_r2 = r2_score(y_te, y_pred)
    baseline_pred = np.full(len(y_te), float(y_tr.mean()))
    metrics = {
        "r2": test_r2, "train_r2": train_r2,
        "overfitting_ratio": (train_r2 - test_r2) / train_r2 if train_r2 > 0 else 0.0,
        "mae": mean_absolute_error(y_te, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_te, y_pred))),
        "cv_mean": search.best_score_, "n_train": len(X_tr), "n_test": len(X_te),
        "baseline_r2": r2_score(y_te, baseline_pred),
    }
    return {"estimator": est, "scaler": scaler, "best_params": search.best_params_,
            "metrics": metrics, "y_test": np.asarray(y_te), "y_pred_test": y_pred}


def train_and_select(X: pd.DataFrame, y: pd.Series, target: str,
                     time_order: pd.Series | np.ndarray | None = None,
                     model_types: tuple[str, ...] = DEFAULT_MODEL_TYPES,
                     max_features: int = 50) -> TrainedModel:
    """Feature-select, tune every algorithm, and return the winner by `selection_score`.

    time_order (e.g. the season year per row) enables TimeSeriesSplit CV and a most-recent
    hold-out; pass None to fall back to shuffled KFold.
    """
    if len(X) < 20:
        raise ValueError(f"Insufficient data to train {target}: {len(X)} rows")

    features = list(X.columns)
    if len(features) > 20:
        k = min(max_features, max(15, len(features) // 3))
        selector = SelectKBest(f_regression, k=k).fit(X, y)
        features = X.columns[selector.get_support()].tolist()
    X = X[features]

    order = np.asarray(time_order) if time_order is not None else None
    best = None
    for mt in model_types:
        try:
            res = _fit_one(X, y, mt, order)
        except Exception as e:
            logger.warning("Model %s failed for target %s: %s", mt, target, e)
            continue
        score = selection_score(res["metrics"])
        logger.info("  %s %s: r2=%.3f score=%.3f", target, mt, res["metrics"]["r2"], score)
        if best is None or score > best[0]:
            best = (score, mt, res)

    if best is None:
        raise RuntimeError(f"All models failed for target {target}")
    _, mt, res = best
    logger.info("Selected %s for target %s (score=%.3f)", mt, target, best[0])
    return TrainedModel(target=target, model_type=mt, estimator=res["estimator"],
                        selected_features=features, scaler=res["scaler"],
                        metrics=res["metrics"], best_params=res["best_params"],
                        y_test=res.get("y_test"), y_pred_test=res.get("y_pred_test"))


def feature_importance(tm: TrainedModel) -> pd.DataFrame:
    est = tm.estimator
    imp = est.feature_importances_ if hasattr(est, "feature_importances_") else np.abs(est.coef_)
    return (pd.DataFrame({"feature": tm.selected_features, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


# --------------------------------------------------------------------------- persistence

def save_model(tm: TrainedModel, cfg: SeasonConfig, name: str) -> None:
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    path = cfg.models_dir / f"{name}.joblib"
    joblib.dump(tm, path)
    logger.info("Saved model %s -> %s", name, path)


def load_model(cfg: SeasonConfig, name: str) -> TrainedModel | None:
    path = cfg.models_dir / f"{name}.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# --------------------------------------------------------------------------- OOS validation

def train_all_vs_exclude_latest(build_xy_fn, df: pd.DataFrame, target: str, cfg: SeasonConfig,
                                 **train_kwargs) -> dict:
    """Docx-mandated methodology, implemented once for all positions.

    Trains model (a) on all seasons and model (b) excluding the latest season, then scores (b)
    out-of-sample on that held-out season. `build_xy_fn(frame) -> (X, y, time_order)` lets each
    position supply its own leakage rules. Returns both models, the OOS metrics for (b), and a
    `chosen` key naming the model to use for production prediction (the higher selection_score).
    """
    latest = int(df["year"].max())
    X_all, y_all, order_all = build_xy_fn(df)
    model_all = train_and_select(X_all, y_all, target, time_order=order_all, **train_kwargs)

    out = {"latest_year": latest, "all": model_all, "exclude_latest": None, "oos": None}

    df_excl = df[df["year"] < latest]
    if df_excl["year"].nunique() >= 2:
        X_ex, y_ex, order_ex = build_xy_fn(df_excl)
        model_ex = train_and_select(X_ex, y_ex, target, time_order=order_ex, **train_kwargs)
        out["exclude_latest"] = model_ex

        X_oos, y_oos, _ = build_xy_fn(df[df["year"] == latest])
        if len(X_oos) > 0:
            pred = model_ex.predict(X_oos)
            out["oos"] = {"r2": r2_score(y_oos, pred) if len(y_oos) > 1 else float("nan"),
                          "mae": mean_absolute_error(y_oos, pred), "n": len(y_oos)}

    # Prefer all-data model unless the held-out model scored better on its own selection metric.
    ex = out["exclude_latest"]
    if ex is not None and selection_score(ex.metrics) > selection_score(model_all.metrics):
        out["chosen"] = "exclude_latest"
        out["model"] = ex
    else:
        out["chosen"] = "all"
        out["model"] = model_all
    return out


def rolling_oos_eval(build_xy_fn, df: pd.DataFrame, target: str, cfg: SeasonConfig,
                     min_train_years: int = 5, **train_kwargs) -> pd.DataFrame:
    """Gold-standard time-series OOS check: train on years < holdout, predict holdout year.

    Returns a DataFrame with columns: holdout_year, r2, mae, n, algorithm.
    Expensive (one full train per holdout year) — use sparingly or with --rolling-eval flag.
    Call with the same build_xy_fn used by the position's train() function.
    """
    df = df.copy()
    years = sorted(df["year"].unique())
    if len(years) < min_train_years + 1:
        logger.warning("rolling_oos_eval: not enough years for %s (%d); skipping", target, len(years))
        return pd.DataFrame(columns=["holdout_year", "r2", "mae", "n", "algorithm"])

    rows = []
    holdout_years = years[min_train_years:]  # need min_train_years worth of training data before each holdout
    logger.info("rolling_oos_eval: %s — %d holdout years %s…%s",
                target, len(holdout_years), holdout_years[0], holdout_years[-1])
    for i, holdout_year in enumerate(holdout_years):
        train_df = df[df["year"] < holdout_year]
        oos_df = df[df["year"] == holdout_year]
        try:
            X_tr, y_tr, order_tr = build_xy_fn(train_df)
            model = train_and_select(X_tr, y_tr, target, time_order=order_tr, **train_kwargs)
            X_oos, y_oos, _ = build_xy_fn(oos_df)
            if len(X_oos) < 2:
                continue
            pred = model.predict(X_oos)
            rows.append({
                "holdout_year": holdout_year,
                "r2": r2_score(y_oos, pred),
                "mae": mean_absolute_error(y_oos, pred),
                "n": len(y_oos),
                "algorithm": model.model_type,
            })
            logger.info("  holdout=%d  r2=%.3f  mae=%.3f  n=%d  algo=%s",
                        holdout_year, rows[-1]["r2"], rows[-1]["mae"], rows[-1]["n"], model.model_type)
        except Exception as e:
            logger.warning("rolling_oos_eval: holdout=%d failed: %s", holdout_year, e)
    return pd.DataFrame(rows)
