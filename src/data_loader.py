"""Loading and basic cleaning of the Kaggle Indian Liver Patient dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from src.synthetic_data import make_synthetic_ilpd

DOWNLOAD_HELP = f"""
Dataset not found at: {config.DATA_FILE}

Get it in ONE of these ways:
  1) Manual: open https://www.kaggle.com/datasets/{config.KAGGLE_SLUG}
     -> Download -> unzip -> put 'indian_liver_patient.csv' inside the 'data/' folder.
  2) Automatic: pip install kagglehub   (needs a free Kaggle account/API token)
     then simply run the project again; it will try to download it for you.
  3) Just testing the code? Run with  --demo  (uses SYNTHETIC data, not real results).
"""


def _try_kagglehub() -> Path | None:
    """Try to download the dataset with kagglehub. Returns CSV path or None."""
    try:
        import kagglehub  # type: ignore
    except ImportError:
        return None
    try:
        folder = Path(kagglehub.dataset_download(config.KAGGLE_SLUG))
        csvs = list(folder.glob("*.csv"))
        return csvs[0] if csvs else None
    except Exception as exc:  # network / auth problems
        print(f"[data] kagglehub download failed: {exc}")
        return None


def load_raw(path: str | Path | None = None, demo: bool = False) -> pd.DataFrame:
    """Return the raw dataframe exactly as stored in the CSV."""
    if demo:
        print("[data] !!! DEMO MODE: using SYNTHETIC data. Results are NOT real. !!!")
        return make_synthetic_ilpd()
    csv = Path(path) if path else config.DATA_FILE
    if not csv.exists():
        downloaded = _try_kagglehub()
        if downloaded is None:
            raise FileNotFoundError(DOWNLOAD_HELP)
        csv = downloaded
    print(f"[data] Loading {csv}")
    return pd.read_csv(csv)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise names / target coding and remove exact duplicate patients.

    * Fixes spelling in column names (Protiens, Phosphotase, ...).
    * Kaggle codes the target as 1 = liver patient, 2 = not a patient.
      We convert to the usual 1 = disease, 0 = no disease.
    * Drops exact duplicate rows (ILPD contains a handful). Doing this BEFORE the
      train/test split prevents the same patient appearing in both sets (leakage).
    """
    df = df.rename(columns=config.RAW_TO_CLEAN).copy()
    missing = set(config.RAW_TO_CLEAN.values()) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")

    df[config.TARGET] = (df["Dataset"] == 1).astype(int)
    df = df.drop(columns="Dataset")
    df["Gender"] = df["Gender"].astype(str).str.strip().str.capitalize()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[data] Rows: {before} -> {len(df)} after removing {before - len(df)} duplicates")
    return df


def load_clean(path: str | Path | None = None, demo: bool = False) -> pd.DataFrame:
    return clean(load_raw(path, demo))
