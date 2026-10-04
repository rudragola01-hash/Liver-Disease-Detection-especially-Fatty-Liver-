"""Synthetic stand-in for the ILPD dataset.

!!! FOR SMOKE-TESTING THE CODE ONLY !!!
Numbers produced from this data are NOT real results. It exists so the whole
pipeline can be executed end to end without internet / the Kaggle download
(e.g. in unit tests or on a machine where the CSV is not available yet).
The schema and rough distributions mimic the real Kaggle file.
"""
import numpy as np
import pandas as pd


def make_synthetic_ilpd(n: int = 583, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    disease = rng.random(n) < 0.714          # ~71% patients, like the real data
    male = rng.random(n) < np.where(disease, 0.78, 0.62)

    age = np.clip(rng.normal(44, 16, n) + disease * 2, 4, 90).round()

    def lognorm(healthy_median, disease_median, sigma_h, sigma_d):
        med = np.where(disease, disease_median, healthy_median)
        sig = np.where(disease, sigma_d, sigma_h)
        return rng.lognormal(np.log(med), sig)

    tb = lognorm(0.8, 1.6, 0.35, 1.0)
    db = np.minimum(tb * rng.uniform(0.15, 0.6, n), tb)
    alp = lognorm(185, 240, 0.30, 0.55)
    alt = lognorm(28, 55, 0.45, 0.95)
    ast = lognorm(30, 75, 0.45, 1.00)
    albumin = np.clip(rng.normal(np.where(disease, 3.0, 3.5), 0.7, n), 0.9, 5.5)
    tp = np.clip(albumin + rng.normal(3.3, 0.6, n), 2.7, 9.6)
    ag = np.clip(albumin / np.maximum(tp - albumin, 0.3), 0.3, 2.8)

    df = pd.DataFrame({
        "Age": age,
        "Gender": np.where(male, "Male", "Female"),
        "Total_Bilirubin": tb.round(1),
        "Direct_Bilirubin": db.round(1),
        "Alkaline_Phosphotase": alp.round().astype(int),
        "Alamine_Aminotransferase": alt.round().astype(int),
        "Aspartate_Aminotransferase": ast.round().astype(int),
        "Total_Protiens": tp.round(1),
        "Albumin": albumin.round(1),
        "Albumin_and_Globulin_Ratio": ag.round(2),
        "Dataset": np.where(disease, 1, 2),     # Kaggle coding: 1 = disease, 2 = no disease
    })
    # The real file has 4 missing A/G ratios
    df.loc[rng.choice(n, 4, replace=False), "Albumin_and_Globulin_Ratio"] = np.nan
    return df
