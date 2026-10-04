"""Clinical feature engineering + the scikit-learn pipeline that wraps every model.

Why a Pipeline?
    Imputation, scaling and feature engineering are *learned from the training data
    only*. Putting them inside the Pipeline means they are re-fitted inside every
    cross-validation fold, so no information from validation/test rows leaks into
    training. This is the most common mistake in student ML projects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config

EPS = 1e-6


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns the raw patient table into a numeric model-ready table.

    Steps
    -----
    1. Gender -> binary ``Gender_Male``.
    2. ``log1p`` on right-skewed lab values (bilirubin, ALP, ALT, AST).
    3. Adds clinically meaningful ratios:
         * ``AST_ALT_Ratio``   - the "De Ritis ratio"; helps separate liver-injury patterns.
         * ``Direct_Total_Bili_Ratio`` - share of conjugated bilirubin (obstructive vs. other causes).
         * ``Bilirubin_Indirect_log``  - unconjugated bilirubin.
         * ``Enzyme_Burden`` - log(ALT+AST+ALP): a single "overall enzyme level" signal.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_out_ = list(self._transform(X.head(5)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self._transform(X)

    # ------------------------------------------------------------------
    @staticmethod
    def _transform(X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        out = pd.DataFrame(index=X.index)

        out["Age"] = pd.to_numeric(X["Age"], errors="coerce")
        out["Gender_Male"] = (
            X["Gender"].astype(str).str.strip().str.lower().eq("male").astype(float)
        )

        for col in config.SKEWED_FEATURES:
            out[f"{col}_log"] = np.log1p(pd.to_numeric(X[col], errors="coerce"))

        for col in ["Total_Proteins", "Albumin", "AG_Ratio"]:
            out[col] = pd.to_numeric(X[col], errors="coerce")

        tb = pd.to_numeric(X["Total_Bilirubin"], errors="coerce")
        db = pd.to_numeric(X["Direct_Bilirubin"], errors="coerce")
        alt = pd.to_numeric(X["ALT"], errors="coerce")
        ast = pd.to_numeric(X["AST"], errors="coerce")
        alp = pd.to_numeric(X["ALP"], errors="coerce")

        out["AST_ALT_Ratio"] = ast / (alt + EPS)
        out["Direct_Total_Bili_Ratio"] = db / (tb + EPS)
        out["Bilirubin_Indirect_log"] = np.log1p((tb - db).clip(lower=0))
        out["Enzyme_Burden"] = np.log1p(alt + ast + alp)

        # Ratios can explode for tiny denominators -> cap at a sane value
        out["AST_ALT_Ratio"] = out["AST_ALT_Ratio"].clip(upper=10)
        return out.replace([np.inf, -np.inf], np.nan)


def build_pipeline(estimator, scale: bool = True) -> Pipeline:
    """feature engineering -> median imputation -> scaling -> model."""
    steps = [
        ("features", ClinicalFeatureEngineer()),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def engineered_feature_names(pipeline: Pipeline) -> list[str]:
    return list(pipeline.named_steps["features"].feature_names_out_)
