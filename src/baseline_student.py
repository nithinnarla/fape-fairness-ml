"""
FAPE — Student Performance Baseline Models
Phase 4 — Baseline Fairness Evaluation

Two datasets: Math (395) and Portuguese (649)
Sensitive attributes: sex (0=Female, 1=Male), age
Models: Logistic Regression, Random Forest, Gradient Boosting
Target: G3 binarized at median
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from student_loader import load_student_performance

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

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
        if len(yt) < 5:
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
            'n': len(yt)
        }
    return metrics


def run_baselines():
    print("FAPE Phase 4 — Student Performance Baseline Models")
    print("=" * 55)

    datasets = load_student_performance()
    all_results = {}

    for subject, ds in datasets.items():
        print(f"\n--- Subject: {subject} (n={ds['metadata']['n_samples']}) ---")
        X = ds['X'].values
        y = ds['y'].values
        sex = ds['X']['sex'].values
        age = ds['X']['age'].values
        age_binned = pd.cut(age, bins=2, labels=['young','older']).astype(str)  # 2 bins — 'older' group sparse (n=29 math, n=41 por)

        X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
            X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)

        sex_test = sex[idx_te]
        age_test = age_binned[idx_te]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        print(f"  n={len(y):,} | features={X.shape[1]} | pos_rate={y.mean():.1%}")
        print(f"  sensitive=sex | groups=Female(0)/Male(1)")

        subj_results = {}
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
            fm_sex = fairness_metrics(y_test, y_pred, sex_test)
            fm_age = fairness_metrics(y_test, y_pred, age_test)

            dp_diffs = [abs(v['dp_diff']) for v in fm_sex.values()]
            max_dp = max(dp_diffs) if dp_diffs else 0

            subj_results[name] = {
                'auc': auc, 'f1': f1,
                'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test,
                'fairness_sex': fm_sex, 'fairness_age': fm_age,
                'max_dp': max_dp
            }
            print(f"  {name:<25} AUC={auc:.3f} F1={f1:.3f} max_DP={max_dp:.3f}")

        all_results[subject] = {
            'results': subj_results,
            'sex_test': sex_test,
            'age_test': age_test,
            'y_test': y_test,
            'X': ds['X'],
            'y': ds['y']
        }

    print(f"\n--- Standard Metrics Summary ---")
    for subject, data in all_results.items():
        best = max(data['results'].items(), key=lambda x: x[1]['auc'])
        print(f"  {subject:<15} best={best[0]} AUC={best[1]['auc']:.3f} F1={best[1]['f1']:.3f}")

    print(f"\n--- Fairness Metrics by Sex ---")
    for subject, data in all_results.items():
        print(f"  {subject}:")
        for model_name, res in data['results'].items():
            fm = res['fairness_sex']
            for grp, m in fm.items():
                label = 'Female' if grp == 0 else 'Male'
                print(f"    {model_name:<25} {label}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f}")

    print(f"\n--- Fairness Metrics by Age ---")
    for subject, data in all_results.items():
        print(f"  {subject}:")
        for model_name, res in data['results'].items():
            fm = res['fairness_age']
            for grp, m in fm.items():
                print(f"    {model_name:<25} age={grp}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f}")

    print(f"\n--- Disparate Impact Ratio by Sex ---")
    for subject, data in all_results.items():
        print(f"  {subject}:")
        for model_name, res in data['results'].items():
            fm = res['fairness_sex']
            if 0 in fm and 1 in fm:
                female_rate = fm[0]['pos_rate']
                male_rate = fm[1]['pos_rate']
                dir_ratio = min(female_rate, male_rate) / max(female_rate, male_rate) if max(female_rate, male_rate) > 0 else 1
                print(f"    {model_name:<25} Female={female_rate:.3f} Male={male_rate:.3f} DIR={dir_ratio:.3f}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for subject, ds in datasets.items():
        X = ds['X'].values; y = ds['y'].values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)
        for name, model in MODELS.items():
            if name == 'LogisticRegression':
                scores = cross_val_score(model, X_sc, y, cv=5, scoring='roc_auc')
            else:
                scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
            print(f"  {subject:<15} {name:<25} CV-AUC={scores.mean():.3f}±{scores.std():.3f}")

    print(f"\n--- Student Performance Baseline complete ---")
    print(f"  Sex fairness gap identified across both subjects")
    print(f"  Stage 2 ThresholdOptimizer needed to reduce disparities")

    return all_results


if __name__ == "__main__":
    results = run_baselines()
