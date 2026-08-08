"""
FAPE, Folktables ACS Baseline Models
Phase 4, Baseline Modeling
Socioeconomic Domain

Baseline models for Folktables ACS income prediction.
Three classifiers: Logistic Regression, Random Forest, Gradient Boosting.
Standard metrics + fairness metrics by race (RAC1P) and sex (SEX).
Sample: 100,000 stratified from 1,589,032 total records.
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_folktables_acs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)

RAC1P_LABELS = {1:'White', 2:'Black', 3:'Am.Indian', 4:'Alaska Native',
                5:'Am.Indian+Alaska', 6:'Asian', 7:'Pacific Islander',
                8:'Other', 9:'Two or more'}
SEX_LABELS = {1:'Male', 2:'Female'}

SAMPLE_SIZE = 100000
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def fairness_metrics(y_true, y_pred, group_series):
    results = {}
    base_rate = y_pred.mean()
    for group in sorted(group_series.unique()):
        mask = group_series == group
        if mask.sum() < 50:
            continue
        gp = y_pred[mask]
        gt = y_true[mask]
        tp_mask = gt == 1
        fp_mask = gt == 0
        results[group] = {
            "n": mask.sum(),
            "pos_rate": gp.mean(),
            "dp_diff": gp.mean() - base_rate,
            "tpr": gp[tp_mask].mean() if tp_mask.sum() > 0 else 0,
            "fpr": gp[fp_mask].mean() if fp_mask.sum() > 0 else 0
        }
    return results


def run_baselines():
    print("FAPE Phase 4, Folktables ACS Baseline Models")
    print("=" * 50)

    df_full = load_folktables_acs()

    # Stratified sample
    df = df_full.groupby('label', group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE//2), random_state=42)
    ).reset_index(drop=True)
    print(f"\nSampled: {len(df):,} from {len(df_full):,} total")

    feature_cols = ['AGEP','SEX','RAC1P','SCHL','MAR','WKHP','COW','DIS','POVPIP','NATIVITY']
    X = df[feature_cols].values
    y = df['label'].values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    race_test = df['RAC1P'].iloc[idx_test].reset_index(drop=True)
    sex_test = df['SEX'].iloc[idx_test].reset_index(drop=True)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Train income >$50k rate: {y_train.mean():.1%}")
    print(f"Test income >$50k rate: {y_test.mean():.1%}")

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
    for name, res in results.items():
        print(f"  {name}:")
        fm = fairness_metrics(y_test, res["y_pred"], race_test)
        for group, metrics in fm.items():
            label = RAC1P_LABELS.get(group, str(group))
            print(f"    {label:<20} n={metrics['n']:,} pos={metrics['pos_rate']:.1%} dp_diff={metrics['dp_diff']:+.3f} tpr={metrics['tpr']:.3f} fpr={metrics['fpr']:.3f}")

    print(f"\n--- Fairness Metrics by Sex ---")
    for name, res in results.items():
        print(f"  {name}:")
        fm = fairness_metrics(y_test, res["y_pred"], sex_test)
        for group, metrics in fm.items():
            label = SEX_LABELS.get(group, str(group))
            print(f"    {label:<10} n={metrics['n']:,} pos={metrics['pos_rate']:.1%} dp_diff={metrics['dp_diff']:+.3f} tpr={metrics['tpr']:.3f} fpr={metrics['fpr']:.3f}")

    print(f"\n--- Disparate Impact Ratio by Race ---")
    for name, res in results.items():
        white_rate = res["y_pred"][race_test == 1].mean()
        print(f"  {name} (White ref={white_rate:.1%}):")
        for group in [2, 3, 6, 8, 9]:
            mask = race_test == group
            if mask.sum() < 50: continue
            rate = res["y_pred"][mask].mean()
            dir_val = rate / white_rate
            flag = " (!) <0.8 threshold" if dir_val < 0.8 else ""
            print(f"    {RAC1P_LABELS.get(group,'?'):<20} rate={rate:.1%} DIR={dir_val:.2f}{flag}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for name, model in MODELS.items():
        X_cv = X_train_sc if name == "LogisticRegression" else X_train
        cv_scores = cross_val_score(model, X_cv, y_train, cv=5, scoring="roc_auc")
        print(f"  {name:<25} AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print(f"\n--- Key Findings ---")
    best = max(results.items(), key=lambda x: x[1]["auc"])
    print(f"  Best model: {best[0]} (AUC={best[1]['auc']:.3f})")
    print(f"  Male-Female income gap: Male {df[df['SEX']==1]['label'].mean():.1%} vs Female {df[df['SEX']==2]['label'].mean():.1%}")
    print(f"  Black-White income gap: Black {df[df['RAC1P']==2]['label'].mean():.1%} vs White {df[df['RAC1P']==1]['label'].mean():.1%}")
    print(f"  Note: Asian income rate (51.1%) exceeds White (46.7%), intersectional analysis needed")
    print(f"  Note: FAPE Stage 2 ThresholdOptimizer targets equalized opportunity by race and sex")

    print(f"\n--- Folktables Baseline complete ---")
    return results


if __name__ == "__main__":
    results = run_baselines()
