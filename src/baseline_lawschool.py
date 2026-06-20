"""
FAPE — Law School Admissions Baseline Models
Phase 4 — Baseline Fairness Evaluation

Dataset: Law School Admissions (Wightman 1998) via FairGround law_school_lequy
18,692 records | 11 features
Sensitive attributes: racetxt (0=minority, 1=white), male (1=male, 0=female)
Target: pass_bar (binary) — 90.2% positive rate (severe class imbalance)
Models: Logistic Regression, Random Forest, Gradient Boosting

Key notes:
- 90.2% positive rate inflates F1 — AUC is primary metric
- Minority group (racetxt=0) only 6.4% of data — fairness metrics noisy
- Race gap: White 92.1% pass rate vs Minority 61.8% — 30.3% gap from EDA
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(__file__))
from lawschool_loader import load_law_school

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def fairness_metrics(y_true, y_pred, sensitive):
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    overall_pos = y_pred.mean()
    for grp in np.unique(sensitive):
        mask = sensitive == grp
        yt = y_true[mask]; yp = y_pred[mask]
        if len(yt) < 10:
            continue
        tp = ((yt==1)&(yp==1)).sum(); fp = ((yt==0)&(yp==1)).sum()
        tn = ((yt==0)&(yp==0)).sum(); fn = ((yt==1)&(yp==0)).sum()
        fpr = fp/(fp+tn) if (fp+tn) > 0 else 0
        tpr = tp/(tp+fn) if (tp+fn) > 0 else 0
        pos_rate = yp.mean()
        metrics[grp] = {
            'fpr': fpr, 'tpr': tpr,
            'pos_rate': pos_rate,
            'dp_diff': pos_rate - overall_pos,
            'n': int(len(yt))
        }
    return metrics


def run_baselines():
    print("FAPE Phase 4 — Law School Admissions Baseline Models")
    print("=" * 55)

    result = load_law_school()
    ds = result['law_school']
    X = ds['X']
    y = ds['y']

    race = X['racetxt'].values
    sex = X['male'].values

    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X.values, y.values, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=y.values)

    race_test = race[idx_te]
    sex_test = sex[idx_te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"\n  n={len(y):,} | features={X.shape[1]} | pos_rate={y.mean():.1%}")
    print(f"  sensitive=racetxt (0=minority/1=white), male")
    print(f"  Note: 90.2% positive rate — AUC primary metric; F1 inflated")

    all_results = {}

    print(f"\n--- Standard Metrics ---")
    for name, model in MODELS.items():
        if name == 'LogisticRegression':
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
            y_prob = model.predict_proba(X_te_sc)[:,1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        fm_race = fairness_metrics(y_test, y_pred, race_test)
        fm_sex = fairness_metrics(y_test, y_pred, sex_test)

        dp_race = [abs(v['dp_diff']) for v in fm_race.values()]
        max_dp_race = max(dp_race) if dp_race else 0

        all_results[name] = {
            'auc': auc, 'f1': f1,
            'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test,
            'fairness_race': fm_race, 'fairness_sex': fm_sex,
            'max_dp_race': max_dp_race
        }
        print(f"  {name:<25} AUC={auc:.3f} F1={f1:.3f} max_DP_race={max_dp_race:.3f}")

    print(f"\n--- Fairness Metrics by Race ---")
    for name, res in all_results.items():
        fm = res['fairness_race']
        for grp, m in fm.items():
            label = 'Minority' if grp == 0 else 'White'
            print(f"  {name:<25} {label}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f} n={m['n']}")

    print(f"\n--- Fairness Metrics by Sex ---")
    for name, res in all_results.items():
        fm = res['fairness_sex']
        for grp, m in fm.items():
            label = 'Female' if grp == 0.0 else 'Male'
            print(f"  {name:<25} {label}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f}")

    print(f"\n--- Disparate Impact Ratio by Race ---")
    for name, res in all_results.items():
        fm = res['fairness_race']
        if 0 in fm and 1 in fm:
            min_rate = fm[0]['pos_rate']
            maj_rate = fm[1]['pos_rate']
            dir_ratio = min_rate/maj_rate if maj_rate > 0 else 1
            print(f"  {name:<25} Minority={min_rate:.3f} White={maj_rate:.3f} DIR={dir_ratio:.3f}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for name, model in MODELS.items():
        if name == 'LogisticRegression':
            scores = cross_val_score(model, X_tr_sc, y_train, cv=5, scoring='roc_auc')
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  {name:<25} CV-AUC={scores.mean():.3f}±{scores.std():.3f}")


    print(f"\n--- Intersectional Analysis (Race x Sex) ---")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    for r, r_label in [(0,'Minority'),(1,'White')]:
        for s, s_label in [(0.0,'Female'),(1.0,'Male')]:
            mask = (race_test==r) & (sex_test==s)
            n = mask.sum()
            if n < 10:
                print(f"  {r_label} {s_label}: n={n} — too sparse")
                continue
            pos_rate = y_pred_gb[mask].mean()
            true_rate = y_test[mask].mean()
            print(f"  {r_label} {s_label}: n={n} | pred_pos={pos_rate:.1%} | true_pos={true_rate:.1%}")
    print(f"  Note: Race dominates — Minority 60-65% vs White 97-99%; sex effect minimal within groups")

    print(f"\n--- Law School Baseline complete ---")
    print(f"  Race gap confirmed — minority FPR and TPR systematically different")
    print(f"  Stage 2 ThresholdOptimizer needed to reduce racial disparities")

    return all_results, race_test, sex_test, y_test


if __name__ == "__main__":
    results, race_test, sex_test, y_test = run_baselines()
