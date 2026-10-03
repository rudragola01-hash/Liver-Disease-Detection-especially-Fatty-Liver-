"""End-to-end pipeline:  data -> EDA -> train/tune -> evaluate -> explain -> save model.

Usage
-----
    python main.py                 # real Kaggle data (data/indian_liver_patient.csv)
    python main.py --quick         # fewer search iterations (fast sanity run)
    python main.py --demo          # SYNTHETIC data, only to test that the code runs
    python main.py --data path.csv --skip-eda
"""
from __future__ import annotations

import argparse
import json
import warnings

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

import config
from src import data_loader, eda, evaluate, explain

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    ap = argparse.ArgumentParser(description="Liver disease detection pipeline")
    ap.add_argument("--data", default=None, help="path to indian_liver_patient.csv")
    ap.add_argument("--demo", action="store_true", help="use synthetic data (NOT real results)")
    ap.add_argument("--quick", action="store_true", help="fewer tuning iterations")
    ap.add_argument("--skip-eda", action="store_true")
    ap.add_argument("--n-iter", type=int, default=None, help="random-search iterations")
    return ap.parse_args()


def main():
    args = parse_args()
    for d in (config.OUTPUT_DIR, config.FIG_DIR, config.MODEL_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. DATA ---------------------------------------------------------------
    df = data_loader.load_clean(args.data, demo=args.demo)
    if not args.skip_eda:
        eda.run_eda(df)

    X = df[config.RAW_FEATURES]
    y = df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)
    print(f"[main] train={len(X_train)}  test={len(X_test)}  "
          f"disease rate train={y_train.mean():.1%} test={y_test.mean():.1%}")

    # 2. TRAIN + TUNE (imports here so --help is instant) ---------------------
    from src import models
    n_iter = 8 if args.quick else args.n_iter
    fitted, cv_table = models.train_all(X_train, y_train, n_iter=n_iter)
    evaluate.plot_cv_summary(cv_table)
    cv_table.to_csv(config.OUTPUT_DIR / "cv_results.csv", index=False)

    # 3. SELECT best model using CROSS-VALIDATION only (test set stays untouched)
    best_name = cv_table.loc[0, "Model"]
    best = fitted[best_name]
    print(f"\n[main] Best model by CV ROC-AUC: {best_name}")

    # 4. SCREENING THRESHOLD from out-of-fold training predictions ---------------
    threshold = evaluate.choose_screening_threshold(best, X_train, y_train)
    print(f"[main] Screening threshold (F2-optimal on CV): {threshold:.2f}")

    # 5. FINAL EVALUATION on the held-out test set --------------------------------
    test_table, probs = evaluate.evaluate_on_test(fitted, X_test, y_test)
    test_table.to_csv(config.OUTPUT_DIR / "test_results.csv", index=False)
    evaluate.plot_roc_pr(probs, y_test)
    evaluate.plot_confusion_matrices(probs, y_test, 0.5)
    evaluate.plot_metric_comparison(test_table)
    evaluate.plot_best_model_diagnostics(best_name, y_test, probs[best_name], threshold)
    screen = evaluate.metrics_at(y_test, probs[best_name], threshold)

    # 6. EXPLAINABILITY ---------------------------------------------------------
    perm = explain.permutation_report(best, X_test, y_test, best_name)
    explain.internal_importance(best, best_name)

    # 7. SAVE ---------------------------------------------------------------------
    # Refit the winning pipeline on ALL data for deployment (standard practice once
    # model choice and threshold are final). Performance numbers above come from the
    # train/test split and stay honest.
    from sklearn.base import clone
    final = clone(best).fit(X, y)
    bundle = {"pipeline": final, "name": best_name, "threshold": threshold,
              "features": config.RAW_FEATURES, "demo_data": bool(args.demo),
              "feature_ranges": X.describe().loc[["min", "50%", "max"]].to_dict()}
    joblib.dump(bundle, config.MODEL_FILE)

    best_row = test_table[test_table["Model"] == best_name].iloc[0].to_dict()
    best_row.pop("Model")
    summary = {
        "best_model": best_name, "screening_threshold": threshold,
        "test_metrics_at_0.5": {k: (v if isinstance(v, str) else float(v))
                                for k, v in best_row.items()},
        "test_metrics_at_screening_threshold": {k: float(v) for k, v in screen.items()},
        "top_features": perm.head(5)["feature"].tolist(),
        "demo_data": bool(args.demo),
    }
    (config.OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\n================ TEST-SET RESULTS (threshold 0.5) ================")
    print(test_table.drop(columns=["TN", "FP", "FN", "TP"]).round(3).to_string(index=False))
    print(f"\n{best_name} at screening threshold {threshold:.2f}: "
          f"recall={screen['Recall (Sensitivity)']:.2f}  precision={screen['Precision']:.2f}  "
          f"specificity={screen['Specificity']:.2f}")
    print("Top features:", ", ".join(summary["top_features"]))
    print(f"\nModel saved -> {config.MODEL_FILE}")
    print("Figures     ->", config.FIG_DIR)
    if args.demo:
        print("\n!!! DEMO MODE: all numbers above come from SYNTHETIC data. !!!")


if __name__ == "__main__":
    main()
