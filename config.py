"""Central configuration for the Liver Disease Detection project.

Everything that is a "setting" lives here so that the rest of the code stays clean
and experiments are easy to reproduce.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_FILE = DATA_DIR / "indian_liver_patient.csv"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = ROOT / "models"
MODEL_FILE = MODEL_DIR / "best_model.joblib"

# Kaggle dataset: "Indian Liver Patient Records" (UCI ILPD, 583 patients)
KAGGLE_SLUG = "uciml/indian-liver-patient-records"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_ITER_SEARCH = 25          # random-search iterations per model
BOOTSTRAP_SAMPLES = 1000    # for the ROC-AUC confidence interval

TARGET = "Disease"          # 1 = liver disease, 0 = healthy / no liver disease

# Clean, readable column names (the raw Kaggle file has spelling errors)
RAW_TO_CLEAN = {
    "Age": "Age",
    "Gender": "Gender",
    "Total_Bilirubin": "Total_Bilirubin",
    "Direct_Bilirubin": "Direct_Bilirubin",
    "Alkaline_Phosphotase": "ALP",
    "Alamine_Aminotransferase": "ALT",
    "Aspartate_Aminotransferase": "AST",
    "Total_Protiens": "Total_Proteins",
    "Albumin": "Albumin",
    "Albumin_and_Globulin_Ratio": "AG_Ratio",
    "Dataset": "Dataset",
}

NUMERIC_FEATURES = [
    "Age", "Total_Bilirubin", "Direct_Bilirubin", "ALP", "ALT", "AST",
    "Total_Proteins", "Albumin", "AG_Ratio",
]
RAW_FEATURES = ["Age", "Gender"] + NUMERIC_FEATURES[1:]

# Strongly right-skewed lab values -> log1p transform helps linear / distance models
SKEWED_FEATURES = ["Total_Bilirubin", "Direct_Bilirubin", "ALP", "ALT", "AST"]

# Friendly names + units used in the app and in plots
FEATURE_INFO = {
    "Age": ("Age", "years"),
    "Gender": ("Gender", ""),
    "Total_Bilirubin": ("Total Bilirubin", "mg/dL"),
    "Direct_Bilirubin": ("Direct Bilirubin", "mg/dL"),
    "ALP": ("Alkaline Phosphatase (ALP)", "IU/L"),
    "ALT": ("Alanine Aminotransferase (ALT / SGPT)", "IU/L"),
    "AST": ("Aspartate Aminotransferase (AST / SGOT)", "IU/L"),
    "Total_Proteins": ("Total Proteins", "g/dL"),
    "Albumin": ("Albumin", "g/dL"),
    "AG_Ratio": ("Albumin / Globulin Ratio", ""),
}
