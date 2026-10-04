"""Model evaluation: metrics, confidence intervals, threshold tuning and plots."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, fbeta_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import config

COLORS = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#6d597a"]


def _save(fig, name: str) -> None:
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eval] saved {path.name}")


def bootstrap_auc_ci(y_true, y_prob, n: int | None = None, seed: int = 0):
    """95% bootstrap confidence interval for ROC-AUC.

    With only ~115 test patients a single AUC number is noisy - the interval shows
    how much to trust it.
    """
    n = n or config.BOOTSTRAP_SAMPLES
    rng = np.random.default_rng(seed)
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    return np.percentile(scores, [2.5, 97.5])


def metrics_at(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall (Sensitivity)": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
    }


def evaluate_on_test(fitted: dict, X_test, y_test) -> tuple[pd.DataFrame, dict]:
    """Score every tuned model on the untouched test set."""
    rows, probs = [], {}
    for name, model in fitted.items():
        p = model.predict_proba(X_test)[:, 1]
        probs[name] = p
        m = metrics_at(y_test, p, 0.5)
        lo, hi = bootstrap_auc_ci(y_test, p)
        m.update({"Model": name, "AUC 95% CI": f"{lo:.3f} - {hi:.3f}"})
        rows.append(m)
    cols = ["Model", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity",
            "F1", "ROC-AUC", "AUC 95% CI", "PR-AUC", "TN", "FP", "FN", "TP"]
    table = pd.DataFrame(rows)[cols].sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return table, probs


def choose_screening_threshold(pipeline, X_train, y_train, beta: float = 2.0) -> float:
    """Pick the probability cut-off that maximises F-beta on out-of-fold predictions.

    For a SCREENING tool a missed patient (false negative) is worse than a false alarm,
    so we weight recall higher (beta=2). The threshold is chosen on the TRAINING data
    via cross-validation - never on the test set.
    """
    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    scores = [fbeta_score(y_train, (oof >= t).astype(int), beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------
def plot_roc_pr(probs: dict, y_test) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    for (name, p), c in zip(probs.items(), COLORS):
        fpr, tpr, _ = roc_curve(y_test, p)
        ax[0].plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
        pr, rc, _ = precision_recall_curve(y_test, p)
        ax[1].plot(rc, pr, color=c, lw=2, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)")
    ax[1].axhline(np.mean(y_test), color="k", ls="--", lw=1, label="Chance (prevalence)")
    ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall curves (test set)")
    for a in ax:
        a.legend(loc="lower right" if a is ax[0] else "lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, "10_roc_pr_curves.png")


def plot_confusion_matrices(probs: dict, y_test, threshold: dict | float = 0.5) -> None:
    n = len(probs)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 4.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, p) in zip(axes, probs.items()):
        t = threshold[name] if isinstance(threshold, dict) else threshold
        cm = confusion_matrix(y_test, (p >= t).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xticks([0, 1], ["No disease", "Disease"])
        ax.set_yticks([0, 1], ["No disease", "Disease"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} (threshold {t:.2f})")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    _save(fig, "11_confusion_matrices.png")


def plot_metric_comparison(table: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall (Sensitivity)", "F1", "ROC-AUC"]
    x = np.arange(len(table))
    w = 0.16
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, (m, c) in enumerate(zip(metrics, COLORS)):
        ax.bar(x + (i - 2) * w, table[m], w, label=m, color=c)
    ax.set_xticks(x, table["Model"], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the held-out test set (threshold 0.5)")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    _save(fig, "12_model_comparison.png")


def plot_cv_summary(cv_table: pd.DataFrame) -> None:
    t = cv_table.sort_values("CV AUC (tuned)")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(y + 0.2, t["CV AUC (default)"], 0.4, label="Default parameters", color="#adb5bd")
    ax.barh(y - 0.2, t["CV AUC (tuned)"], 0.4, xerr=t["CV AUC std"], label="Tuned", color="#2a9d8f")
    ax.set_yticks(y, t["Model"])
    ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("5-fold CV ROC-AUC (training data)")
    ax.set_title("Effect of hyper-parameter tuning")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, "09_tuning_effect.png")


def plot_best_model_diagnostics(name: str, y_test, p, threshold: float) -> None:
    """Probability histogram, threshold trade-off and calibration of the chosen model."""
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    y_test = np.asarray(y_test)
    ax[0].hist(p[y_test == 0], bins=20, alpha=0.65, color="#2a9d8f", label="No disease")
    ax[0].hist(p[y_test == 1], bins=20, alpha=0.65, color="#e76f51", label="Disease")
    ax[0].axvline(threshold, color="k", ls="--", label=f"screening threshold {threshold:.2f}")
    ax[0].set(title=f"{name}: predicted risk", xlabel="Predicted probability of disease")
    ax[0].legend()

    grid = np.linspace(0.05, 0.95, 91)
    ax[1].plot(grid, [recall_score(y_test, p >= t) for t in grid], label="Recall", color="#e76f51")
    ax[1].plot(grid, [precision_score(y_test, p >= t, zero_division=0) for t in grid],
               label="Precision", color="#264653")
    ax[1].plot(grid, [f1_score(y_test, p >= t, zero_division=0) for t in grid],
               label="F1", color="#e9c46a")
    ax[1].axvline(threshold, color="k", ls="--")
    ax[1].set(title="Threshold trade-off", xlabel="Decision threshold")
    ax[1].legend()

    frac, mean_pred = calibration_curve(y_test, p, n_bins=6, strategy="quantile")
    ax[2].plot(mean_pred, frac, "o-", color="#264653", label="Model")
    ax[2].plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax[2].set(title="Calibration", xlabel="Mean predicted probability", ylabel="Observed fraction")
    ax[2].legend()
    fig.tight_layout()
    _save(fig, "13_best_model_diagnostics.png")
