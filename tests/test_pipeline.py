"""Quick tests. Run:  pytest -q"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import config
from src.data_loader import clean
from src.preprocessing import ClinicalFeatureEngineer, build_pipeline
from src.synthetic_data import make_synthetic_ilpd


def _data():
    return clean(make_synthetic_ilpd(300, seed=1))


def test_clean_maps_target_and_names():
    df = _data()
    assert set(df[config.TARGET].unique()) <= {0, 1}
    assert "ALT" in df.columns and "Dataset" not in df.columns


def test_feature_engineer_handles_missing_and_zero_division():
    df = _data()[config.RAW_FEATURES].copy()
    df.loc[0, "ALT"] = 0
    df.loc[1, "AG_Ratio"] = np.nan
    out = ClinicalFeatureEngineer().fit(df).transform(df)
    assert not np.isinf(out.to_numpy(dtype=float)).any()
    assert "AST_ALT_Ratio" in out.columns


def test_pipeline_fits_and_predicts_probabilities():
    df = _data()
    X, y = df[config.RAW_FEATURES], df[config.TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, stratify=y, random_state=0)
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    assert p.shape == (len(Xte),) and ((p >= 0) & (p <= 1)).all()


def test_pipeline_accepts_single_patient_row():
    df = _data()
    pipe = build_pipeline(LogisticRegression(max_iter=1000)).fit(
        df[config.RAW_FEATURES], df[config.TARGET])
    row = pd.DataFrame([{"Age": 50, "Gender": "Male", "Total_Bilirubin": 1.0,
                         "Direct_Bilirubin": 0.3, "ALP": 200, "ALT": 30, "AST": 30,
                         "Total_Proteins": 6.8, "Albumin": 3.8, "AG_Ratio": 1.1}])
    assert 0 <= pipe.predict_proba(row)[0, 1] <= 1
