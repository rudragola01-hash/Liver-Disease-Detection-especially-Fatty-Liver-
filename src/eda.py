"""Exploratory Data Analysis (EDA): understand the data BEFORE modelling."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # works without a display (servers, CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import config

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Liver disease"}


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[eda] saved {path.name}")


def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Create all EDA figures and return a summary table (also saved as CSV)."""
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Summary table ---------------------------------------------------
    summary = df.describe(include="all").T
    summary["missing"] = df.isna().sum()
    summary.to_csv(config.OUTPUT_DIR / "data_summary.csv")
    print(f"[eda] class balance: {df[config.TARGET].value_counts().to_dict()}  "
          f"missing values: {int(df.isna().sum().sum())}")

    # 2) Class balance + gender/age -------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    counts = df[config.TARGET].value_counts().sort_index()
    ax[0].bar([LABELS[i] for i in counts.index], counts.values,
              color=[PALETTE[i] for i in counts.index])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v + 5, f"{v}\n({v / len(df):.0%})", ha="center")
    ax[0].set_title("Class balance")
    ax[0].set_ylim(0, counts.max() * 1.2)

    rate = df.groupby("Gender")[config.TARGET].mean() * 100
    ax[1].bar(rate.index, rate.values, color="#264653")
    for i, v in enumerate(rate.values):
        ax[1].text(i, v + 1, f"{v:.1f}%", ha="center")
    ax[1].set_title("Disease rate by gender (%)")
    ax[1].set_ylim(0, 100)

    for cls, g in df.groupby(config.TARGET):
        sns.kdeplot(g["Age"], ax=ax[2], fill=True, alpha=0.35,
                    color=PALETTE[cls], label=LABELS[cls])
    ax[2].set_title("Age distribution by class")
    ax[2].legend()
    fig.tight_layout()
    _save(fig, "01_class_balance_gender_age.png")

    # 3) Missing values --------------------------------------------------
    miss = df.isna().sum()
    miss = miss[miss > 0]
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 3))
        miss.plot.barh(ax=ax, color="#e9c46a")
        ax.set_title("Missing values per column")
        _save(fig, "02_missing_values.png")

    # 4) Distributions of lab values (log axis because of heavy skew) ----
    feats = config.NUMERIC_FEATURES[1:]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), feats):
        for cls, g in df.groupby(config.TARGET):
            vals = g[col].dropna()
            if col in config.SKEWED_FEATURES:
                vals = np.log1p(vals)
            sns.kdeplot(vals, ax=ax, fill=True, alpha=0.35,
                        color=PALETTE[cls], label=LABELS[cls])
        suffix = " (log1p)" if col in config.SKEWED_FEATURES else ""
        ax.set_title(col + suffix)
        ax.set_xlabel("")
    axes.ravel()[0].legend()
    fig.suptitle("Lab-value distributions by class", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "03_feature_distributions.png")

    # 5) Boxplots --------------------------------------------------------
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, col in zip(axes.ravel(), feats):
        sns.boxplot(data=df, x=config.TARGET, y=col, hue=config.TARGET,
                    palette=PALETTE, legend=False, ax=ax)
        if col in config.SKEWED_FEATURES:
            ax.set_yscale("log")
        ax.set_xticks([0, 1], ["No disease", "Disease"])
        ax.set_xlabel("")
        ax.set_title(col)
    fig.suptitle("Boxplots by class (log scale for skewed features)", y=1.02, fontsize=14)
    fig.tight_layout()
    _save(fig, "04_boxplots.png")

    # 6) Correlation heatmap --------------------------------------------
    corr_df = df[config.NUMERIC_FEATURES + [config.TARGET]].copy()
    corr_df["Gender_Male"] = (df["Gender"] == "Male").astype(int)
    corr = corr_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Spearman correlation (robust to skew / outliers)")
    _save(fig, "05_correlation_heatmap.png")

    # 7) Statistical tests: which features differ between classes? -------
    from scipy.stats import mannwhitneyu

    rows = []
    for col in feats + ["Age"]:
        a = df.loc[df[config.TARGET] == 1, col].dropna()
        b = df.loc[df[config.TARGET] == 0, col].dropna()
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        # rank-biserial effect size: 0 = no difference, +-1 = complete separation
        effect = 2 * stat / (len(a) * len(b)) - 1
        rows.append({"feature": col, "median_disease": a.median(),
                     "median_healthy": b.median(), "p_value": p, "effect_size": effect})
    tests = pd.DataFrame(rows).sort_values("p_value")
    tests.to_csv(config.OUTPUT_DIR / "statistical_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    t = tests.sort_values("effect_size")
    ax.barh(t["feature"], t["effect_size"],
            color=np.where(t["p_value"] < 0.05, "#e76f51", "#adb5bd"))
    ax.axvline(0, color="k", lw=0.8)
    ax.set_title("Effect size (Mann-Whitney) - orange = significant (p<0.05)")
    ax.set_xlabel("Rank-biserial effect size")
    _save(fig, "06_feature_effect_sizes.png")
    return tests
