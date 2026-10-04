"""Model interpretability - the 'research gap' about black-box models in the synopsis.

Two complementary views:
  * Permutation importance  - model-agnostic, measured on RAW clinical inputs, on the
    test set: "how much does ROC-AUC drop if this lab value is shuffled?"
  * Model-internal importance - tree importances / logistic-regression coefficients on the
    engineered features (only for models that expose them).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

import config
from src.preprocessing import engineered_feature_names


def _save(fig, name: str) -> None:
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explain] saved {path.name}")


def permutation_report(pipeline, X_test, y_test, model_name: str) -> pd.DataFrame:
    res = permutation_importance(pipeline, X_test, y_test, scoring="roc_auc",
                                 n_repeats=30, random_state=config.RANDOM_STATE, n_jobs=-1)
    df = pd.DataFrame({"feature": X_test.columns,
                       "importance": res.importances_mean,
                       "std": res.importances_std}).sort_values("importance")
    df.sort_values("importance", ascending=False).to_csv(
        config.OUTPUT_DIR / "permutation_importance.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"], xerr=df["std"], color="#2a9d8f")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Permutation importance - {model_name}")
    fig.tight_layout()
    _save(fig, "14_permutation_importance.png")
    return df.sort_values("importance", ascending=False)


def internal_importance(pipeline, model_name: str) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    names = engineered_feature_names(pipeline)
    if hasattr(model, "feature_importances_"):
        vals, label = model.feature_importances_, "Feature importance (impurity)"
    elif hasattr(model, "coef_") and np.ndim(model.coef_) == 2:
        vals, label = model.coef_[0], "Coefficient (standardised features)"
    else:
        return None
    df = pd.DataFrame({"feature": names, "value": vals})
    df = df.reindex(df["value"].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#e76f51" if v > 0 else "#264653" for v in df["value"]]
    ax.barh(df["feature"], df["value"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title(f"{model_name}: {label}")
    fig.tight_layout()
    _save(fig, "15_internal_importance.png")
    return df.sort_values("value", ascending=False)
