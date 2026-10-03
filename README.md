# 🩺 Liver Disease Detection using Machine Learning

B.Tech Final Year Project (CSE) - ITS Engineering College, Greater Noida
Team: Reema Kumari, Rahul Attri, Rudra Gola, Shobha Kumari - Supervisor: Mr. Tarun Kumar Chugh

An end-to-end, reproducible Python project that predicts liver disease from routine blood-test
and demographic data. It follows the synopsis objectives exactly: preprocess data, analyse
features, train **Logistic Regression, SVM, Random Forest, Gradient Boosting/XGBoost and KNN**,
tune them, compare with **Accuracy, Precision, Recall, F1, ROC-AUC**, and provide a **simple
prediction interface**.

## 1. Quick start

```bash
pip install -r requirements.txt

# 1) put the Kaggle CSV in data/  (see data/README.md)  - or let kagglehub download it
python main.py            # EDA + training + tuning + evaluation + explainability + save model
streamlit run app.py      # prediction interface in the browser
python -m pytest -q       # unit tests
```

`python main.py --quick` is a faster run; `python main.py --demo` uses *synthetic* data only to
check that the code runs on a machine without the dataset (never quote those numbers).

## 2. Dataset (Kaggle)

**Indian Liver Patient Records** - https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
583 patients, 10 clinical features, label = liver patient (1) or not (2). Features: Age, Gender,
Total/Direct Bilirubin, ALP, ALT (SGPT), AST (SGOT), Total Proteins, Albumin, A/G ratio.

> ⚠️ **Honest note for viva:** this dataset labels *liver disease in general*, not
> *fatty liver (NAFLD)* specifically, and it has no BMI / diet / alcohol columns. The synopsis
> mentions the same dataset (ILPD, 79.83% accuracy in the literature table). In your report call
> the project "Liver Disease Detection" and describe fatty liver as the motivating use-case, or
> switch to a fatty-liver dataset later - the pipeline only needs `config.py` column names changed.

## 3. Project structure

```
liver_disease_detection/
├── main.py                # runs the whole pipeline
├── app.py                 # Streamlit prediction interface
├── config.py              # paths, seeds, column names, settings
├── requirements.txt
├── data/                  # put indian_liver_patient.csv here
├── src/
│   ├── data_loader.py     # load, rename columns, fix target, drop duplicates
│   ├── preprocessing.py   # feature engineering + leakage-safe Pipeline
│   ├── eda.py             # plots + Mann-Whitney statistical tests
│   ├── models.py          # 5 models + hyper-parameter search spaces + training
│   ├── evaluate.py        # metrics, bootstrap CI, threshold tuning, plots
│   ├── explain.py         # permutation importance, coefficients / tree importances
│   ├── predict.py         # score one patient (used by the app)
│   └── synthetic_data.py  # demo data for smoke tests only
├── tests/test_pipeline.py
├── models/                # best_model.joblib is saved here
└── outputs/               # figures/, CSV result tables, summary.json
```

## 4. How it works (explain this in your viva)

| Step | What we do | Why |
|---|---|---|
| Cleaning | Rename columns, recode target to 1 = disease, drop exact duplicates *before* splitting | Duplicates across train/test would inflate scores |
| EDA | Class balance, distributions, boxplots, Spearman correlation, Mann-Whitney tests | Know the data and which features carry signal |
| Feature engineering | `log1p` of skewed enzymes/bilirubin, **AST/ALT (De Ritis) ratio**, direct/total bilirubin ratio, indirect bilirubin, enzyme burden | Domain-driven features help linear/distance models |
| Pipeline | engineer → median impute → scale → model, all inside one `Pipeline` | Everything is learned on training folds only → **no data leakage** |
| Imbalance (~71% disease) | `class_weight='balanced'` / `scale_pos_weight` | No oversampling, so no duplicated rows leaking across folds |
| Tuning | `RandomizedSearchCV`, stratified 5-fold, scoring = ROC-AUC | Compares default vs tuned (objective 4 of the synopsis) |
| Model choice | Best **cross-validated** AUC on the training set | The test set is never used to choose anything |
| Screening threshold | F2-optimal cut-off from out-of-fold predictions | Missing a patient is worse than a false alarm |
| Final evaluation | Held-out 20% test set: accuracy, precision, recall, specificity, F1, ROC-AUC with **bootstrap 95% CI**, PR-AUC, confusion matrices | One small test set is noisy - the CI shows how much |
| Explainability | Permutation importance on raw lab values + coefficients / tree importances | Closes the "black-box" research gap in the synopsis |
| Deployment | Winning pipeline refit on all data, saved with `joblib`, served by Streamlit | Simple clinician-style interface |

## 5. Outputs you get after `python main.py`

* `outputs/figures/01…06` EDA, `09` tuning effect, `10` ROC/PR curves, `11` confusion matrices,
  `12` model comparison, `13` threshold/calibration diagnostics, `14–15` feature importance
* `outputs/cv_results.csv`, `outputs/test_results.csv`, `outputs/summary.json`,
  `outputs/statistical_tests.csv`, `outputs/permutation_importance.csv`
* `models/best_model.joblib`

## 6. Realistic expectations

On the real ILPD data, published and typical results are about **70-80% accuracy and ROC-AUC of
roughly 0.75-0.85**: the dataset is small, noisy and overlapping. The synopsis states a "90%+" target; that is
unlikely on this dataset, and an honest 75-80% with proper validation is a *stronger* result than
a leaked 95%. Report whatever `main.py` gives you, together with the confidence intervals.

## 7. Limitations & future scope

* Small single-source dataset (583 rows, mostly one region) → needs external validation.
* Not a fatty-liver-specific dataset (no BMI, diet, alcohol).
* Decision support only - never a diagnosis.
* Future: SHAP explanations, XGBoost/LightGBM, calibration, multi-centre data, ultrasound-image CNN.

## 8. Ethics

Voluntary participation, informed consent, anonymised data, and clinician oversight - as in the
synopsis. The app does not store any entered values.
