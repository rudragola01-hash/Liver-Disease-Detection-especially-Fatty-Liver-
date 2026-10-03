"""Streamlit prediction interface.   Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

import config
from src.predict import load_bundle, predict_patient

st.set_page_config(page_title="Liver Disease Screening", page_icon="🩺", layout="wide")

st.title("🩺 Liver Disease Risk Screening")
st.caption("Machine-learning decision-support demo - trained on the Kaggle Indian Liver Patient dataset. "
           "**Not a medical diagnosis.**")

try:
    bundle = load_bundle()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

if bundle.get("demo_data"):
    st.warning("This model was trained on SYNTHETIC demo data - predictions are meaningless. "
               "Run `python main.py` with the real Kaggle CSV.")

rng = bundle["feature_ranges"]
tab_predict, tab_results, tab_about = st.tabs(["Predict", "Model results", "About"])

# ---------------------------------------------------------------- predict
with tab_predict:
    st.subheader("Enter patient information")
    c1, c2, c3 = st.columns(3)

    def num(col, container, step, default, lo=None, hi=None):
        label, unit = config.FEATURE_INFO[col]
        label = f"{label} ({unit})" if unit else label
        lo = float(lo if lo is not None else rng[col]["min"])
        hi = float(hi if hi is not None else rng[col]["max"] * 1.5)
        return container.number_input(label, min_value=lo, max_value=hi,
                                      value=float(default), step=step)

    age = num("Age", c1, 1.0, 45, 1, 100)
    gender = c1.selectbox("Gender", ["Male", "Female"])
    tb = num("Total_Bilirubin", c1, 0.1, 1.0, 0.1, 100)
    db = num("Direct_Bilirubin", c1, 0.1, 0.3, 0.0, 50)
    alp = num("ALP", c2, 5.0, 200, 10, 5000)
    alt = num("ALT", c2, 1.0, 30, 1, 5000)
    ast = num("AST", c2, 1.0, 30, 1, 5000)
    tp = num("Total_Proteins", c3, 0.1, 6.8, 2, 12)
    alb = num("Albumin", c3, 0.1, 3.8, 0.5, 7)
    ag = num("AG_Ratio", c3, 0.05, 1.1, 0.1, 4)

    if st.button("Predict risk", type="primary"):
        if db > tb:
            st.error("Direct bilirubin cannot be higher than total bilirubin.")
        else:
            res = predict_patient({
                "Age": age, "Gender": gender, "Total_Bilirubin": tb, "Direct_Bilirubin": db,
                "ALP": alp, "ALT": alt, "AST": ast, "Total_Proteins": tp,
                "Albumin": alb, "AG_Ratio": ag}, bundle)
            band = res["risk_band"]
            icon = {"Low": "🟢", "Moderate": "🟠", "High": "🔴"}[band]
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated probability of liver disease", f"{res['probability']:.0%}")
            m2.metric("Risk band", f"{icon} {band}")
            m3.metric("Screening result", "Refer for check-up" if res["screen_positive"]
                      else "No flag")
            st.progress(min(max(res["probability"], 0.0), 1.0))
            st.info(f"Model: **{res['model_name']}** - screening threshold "
                    f"{res['threshold']:.2f} (tuned to catch most true patients). "
                    "Always confirm with a qualified clinician.")

# ---------------------------------------------------------------- results
with tab_results:
    st.subheader("Model comparison (held-out test set)")
    f = config.OUTPUT_DIR / "test_results.csv"
    if f.exists():
        st.dataframe(pd.read_csv(f).round(3), use_container_width=True)
    for name, caption in [("10_roc_pr_curves.png", "ROC and Precision-Recall curves"),
                          ("11_confusion_matrices.png", "Confusion matrices"),
                          ("14_permutation_importance.png", "Which inputs matter most"),
                          ("13_best_model_diagnostics.png", "Best-model diagnostics")]:
        p = config.FIG_DIR / name
        if p.exists():
            st.image(str(p), caption=caption)

# ---------------------------------------------------------------- about
with tab_about:
    st.markdown("""
**Pipeline:** clinical feature engineering -> median imputation -> scaling -> classifier
(Logistic Regression, SVM, Random Forest, Gradient Boosting / XGBoost, KNN), tuned with
randomized search + stratified 5-fold cross-validation, evaluated on an untouched test set.

**Ethics & privacy:** no data entered here is stored. The tool supports - never replaces -
clinical judgement.
""")
