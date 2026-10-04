"""The five classifiers from the synopsis + hyper-parameter search spaces."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import config
from src.preprocessing import build_pipeline

try:  # XGBoost is optional - we fall back to scikit-learn's Gradient Boosting
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def get_model_zoo(y_train: pd.Series) -> dict:
    """Return {name: (pipeline, param_distributions)}.

    Class imbalance (about 71% disease) is handled with ``class_weight='balanced'``
    (or ``scale_pos_weight`` for XGBoost) instead of duplicating rows, so there is no
    risk of oversampled copies leaking across CV folds.
    """
    rs = config.RANDOM_STATE
    pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    zoo = {
        "Logistic Regression": (
            build_pipeline(LogisticRegression(max_iter=5000, class_weight="balanced",
                                              random_state=rs)),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "KNN": (
            build_pipeline(KNeighborsClassifier()),
            {"model__n_neighbors": randint(3, 40),
             "model__weights": ["uniform", "distance"],
             "model__p": [1, 2]},
        ),
        "SVM": (
            build_pipeline(SVC(probability=True, class_weight="balanced", random_state=rs)),
            {"model__C": loguniform(1e-2, 1e2),
             "model__gamma": loguniform(1e-3, 1e0),
             "model__kernel": ["rbf", "linear"]},
        ),
        "Random Forest": (
            build_pipeline(RandomForestClassifier(class_weight="balanced_subsample",
                                                  random_state=rs, n_jobs=-1), scale=False),
            {"model__n_estimators": randint(150, 600),
             "model__max_depth": [None, 4, 6, 8, 12],
             "model__min_samples_leaf": randint(1, 12),
             "model__max_features": ["sqrt", "log2", 0.5]},
        ),
    }

    if HAS_XGB:
        zoo["XGBoost"] = (
            build_pipeline(XGBClassifier(eval_metric="logloss", scale_pos_weight=pos_weight,
                                         random_state=rs, n_jobs=-1, verbosity=0), scale=False),
            {"model__n_estimators": randint(100, 500),
             "model__max_depth": randint(2, 7),
             "model__learning_rate": loguniform(0.01, 0.3),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.6, 0.4),
             "model__min_child_weight": randint(1, 8),
             "model__reg_lambda": loguniform(0.1, 10)},
        )
    else:
        zoo["Gradient Boosting"] = (
            build_pipeline(GradientBoostingClassifier(random_state=rs), scale=False),
            {"model__n_estimators": randint(100, 400),
             "model__max_depth": randint(2, 5),
             "model__learning_rate": loguniform(0.01, 0.3),
             "model__subsample": uniform(0.6, 0.4),
             "model__min_samples_leaf": randint(1, 12)},
        )
    return zoo


def train_all(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int | None = None):
    """Baseline CV -> randomized search -> tuned CV for every model.

    Returns
    -------
    fitted : dict[str, Pipeline]       best (refit) estimator per model
    table  : pd.DataFrame              cross-validation summary (training data only!)
    """
    n_iter = n_iter or config.N_ITER_SEARCH
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    zoo = get_model_zoo(y_train)
    fitted, rows = {}, []

    for name, (pipe, space) in zoo.items():
        t0 = time.time()
        base = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)

        search = RandomizedSearchCV(
            pipe, space, n_iter=n_iter, scoring="roc_auc", cv=cv,
            random_state=config.RANDOM_STATE, n_jobs=-1, refit=True, error_score="raise",
        )
        search.fit(X_train, y_train)
        fitted[name] = search.best_estimator_

        # Mean/std over the folds of the BEST configuration
        best_idx = search.best_index_
        tuned_mean = search.cv_results_["mean_test_score"][best_idx]
        tuned_std = search.cv_results_["std_test_score"][best_idx]
        rows.append({
            "Model": name,
            "CV AUC (default)": base.mean(),
            "CV AUC (tuned)": tuned_mean,
            "CV AUC std": tuned_std,
            "Best params": {k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                            for k, v in search.best_params_.items()},
            "Seconds": round(time.time() - t0, 1),
        })
        print(f"[train] {name:<20} default AUC={base.mean():.3f}  "
              f"tuned AUC={tuned_mean:.3f} +/- {tuned_std:.3f}  ({time.time() - t0:.0f}s)")

    table = pd.DataFrame(rows).sort_values("CV AUC (tuned)", ascending=False).reset_index(drop=True)
    return fitted, table
