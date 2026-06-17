"""
FAPE — COMPAS Baseline Models
Phase 4 — Baseline Modeling
Criminal Justice Domain

Baseline models for COMPAS recidivism dataset.
Three classifiers: Logistic Regression, Random Forest, Gradient Boosting.
Standard metrics + fairness metrics by race and sex.
Fairness metrics: demographic parity, equalized odds, disparate impact ratio.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_compas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, classification_report)

SENSITIVE_ATTRS = ["race", "sex"]
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def preprocess(df):
    """Encode categoricals, return X, y, sensitive groups."""
    df = df.copy()
    le = LabelEncoder()
    for col in ["c_charge_degree", "race", "sex", "score_text"]:
        df[col] = le.fit_transform(df[col].astype(str))
    feature_cols = ["age", "c_charge_degree", "sex", "priors_count",
                   "days_b_screening_arrest", "decile_score"]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, df


def fairness_metrics(y_true, y_pred, group_series, group_name):
    """Compute demographic parity, equalized odds, disparate impact."""
    results = {}
    groups = group_series.unique()
    base_rate = y_pred.mean()
    for group in sorted(groups):
        mask = group_series == group
        if mask.sum() < 10:
            continue
        group_pred = y_pred[mask]
        group_true = y_true[mask]
        pos_rate = group_pred.mean()
        # Demographic parity difference vs overall
        dp_diff = pos_rate - base_rate
        # Equalized odds — TPR and FPR
        tp_mask = group_true == 1
        fp_mask = group_true == 0
        tpr = group_pred[tp_mask].mean() if tp_mask.sum() > 0 else 0
        fpr = group_pred[fp_mask].mean() if fp_mask.sum() > 0 else 0
        # Disparate impact ratio vs best group
        results[group] = {
            "n": mask.sum(),
            "pos_rate": pos_rate,
            "dp_diff": dp_diff,
            "tpr": tpr,
            "fpr": fpr
        }
    return results


def run_baselines():
    print("FAPE Phase 4 — COMPAS Baseline Models")
    print("=" * 50)

    df_raw = load_compas()
    X, y, df_enc = preprocess(df_raw)

    # Stratified split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train recidivism rate: {y_train.mean():.1%}")
    print(f"Test recidivism rate: {y_test.mean():.1%}")

    print(f"\n--- Standard Metrics ---")
    results = {}
    for name, model in MODELS.items():
        if name == "LogisticRegression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            y_prob = model.predict_proba(X_test_sc)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        results[name] = {"y_pred": y_pred, "y_prob": y_prob,
                         "acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}
        print(f"  {name:<25} acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1:.3f} auc={auc:.3f}")

    print(f"\n--- Fairness Metrics by Race ---")
    race_test = df_raw["race"].iloc[idx_test].reset_index(drop=True)
    for name, res in results.items():
        print(f"  {name}:")
        fm = fairness_metrics(y_test, res["y_pred"], race_test, "race")
        for group, metrics in fm.items():
            print(f"    {str(group):<20} n={metrics['n']:,} pos={metrics['pos_rate']:.1%} dp_diff={metrics['dp_diff']:+.3f} tpr={metrics['tpr']:.3f} fpr={metrics['fpr']:.3f}")

    print(f"\n--- Fairness Metrics by Sex ---")
    sex_test = df_raw["sex"].iloc[idx_test].reset_index(drop=True)
    for name, res in results.items():
        print(f"  {name}:")
        fm = fairness_metrics(y_test, res["y_pred"], sex_test, "sex")
        for group, metrics in fm.items():
            print(f"    {str(group):<10} n={metrics['n']:,} pos={metrics['pos_rate']:.1%} dp_diff={metrics['dp_diff']:+.3f} tpr={metrics['tpr']:.3f} fpr={metrics['fpr']:.3f}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for name, model in MODELS.items():
        X_cv = X_train_sc if name == "LogisticRegression" else X_train
        cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring="roc_auc")
        print(f"  {name:<25} AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


    print(f"\n--- Disparate Impact Ratio by Race ---")
    races_dir = ["African-American","Hispanic","Other"]
    cauc_rates = {}
    for name, res in results.items():
        cauc_rate = res["y_pred"][race_test=="Caucasian"].mean()
        cauc_rates[name] = cauc_rate
        print(f"  {name} (Caucasian ref={cauc_rate:.1%}):")
        for race in races_dir:
            mask = race_test == race
            if mask.sum() < 10: continue
            rate = res["y_pred"][mask].mean()
            dir_val = rate / cauc_rate
            flag = " ⚠️ >1.25 threshold" if dir_val > 1.25 else ""
            print(f"    {race:<20} rate={rate:.1%} DIR={dir_val:.2f}{flag}")
    print(f"  Note: DIR>1.25 is legally actionable under EEOC disparate impact doctrine")

    print(f"\n--- Key Findings ---")
    best = max(results.items(), key=lambda x: x[1]["auc"])
    print(f"  Best model: {best[0]} (AUC={best[1]['auc']:.3f})")
    print(f"  Note: Fairness Stage 2 will apply ThresholdOptimizer to equalize FPR by race")
    print(f"  Note: African-American FPR expected ~2x Caucasian FPR — see EDA findings")

    print(f"\n--- COMPAS Baseline complete ---")
    return results


if __name__ == "__main__":
    results = run_baselines()
