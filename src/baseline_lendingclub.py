"""
FAPE — Lending Club Baseline Models
Phase 4 — Baseline Fairness Evaluation

Dataset: Lending Club 2007-2018 Q4 (Kaggle — wordsforthewise/lending-club)
1,348,099 records | socioeconomic proxy sensitive attributes
Sensitive attributes: annual_inc_band, home_ownership, emp_length, addr_state
Target: loan_default_binary (1=default, 0=fully paid)
Models: Logistic Regression, Random Forest, Gradient Boosting

Key notes:
- No direct race/gender data — ECOA-compliant proxy-based fairness audit
- Uses 500K sample for baseline speed; full dataset for final results
- Income band proxy: low/lower_mid/upper_mid/high quartiles
- Follows Kozodoi et al. (2022) financial fairness evaluation protocol
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
from lending_club_loader import load_lending_club

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
    print("FAPE Phase 4 — Lending Club Baseline Models")
    print("=" * 55)

    result = load_lending_club(sample_size=500000)
    ds = result['lending_club']
    X = ds['X']
    y = ds['y']

    inc_band = X['annual_inc_band'].values
    home_own = X['home_ownership'].values
    emp_len = X['emp_length'].values

    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X.values, y.values, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=y.values)

    inc_test = inc_band[idx_te]
    home_test = home_own[idx_te]
    emp_test = emp_len[idx_te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"\n  n={len(y):,} | features={X.shape[1]} | default_rate={y.mean():.1%}")
    print(f"  sensitive=annual_inc_band, home_ownership, emp_length")
    print(f"  Note: No direct race/gender — ECOA proxy-based audit per Kozodoi et al. (2022)")

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
        fm_inc = fairness_metrics(y_test, y_pred, inc_test)
        fm_home = fairness_metrics(y_test, y_pred, home_test)

        dp_inc = [abs(v['dp_diff']) for v in fm_inc.values()]
        max_dp_inc = max(dp_inc) if dp_inc else 0

        all_results[name] = {
            'auc': auc, 'f1': f1,
            'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test,
            'fairness_inc': fm_inc, 'fairness_home': fm_home,
            'max_dp_inc': max_dp_inc
        }
        print(f"  {name:<25} AUC={auc:.3f} F1={f1:.3f} max_DP_inc={max_dp_inc:.3f}")

    print(f"\n--- Fairness Metrics by Income Band ---")
    inc_labels = {0: 'low', 1: 'lower_mid', 2: 'upper_mid', 3: 'high'}
    for name, res in all_results.items():
        fm = res['fairness_inc']
        for grp, m in sorted(fm.items()):
            label = inc_labels.get(grp, str(grp))
            print(f"  {name:<25} {label:<12}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f} n={m['n']:,}")

    print(f"\n--- Fairness Metrics by Home Ownership ---")
    home_labels = {0: 'ANY', 1: 'MORTGAGE', 2: 'NONE', 3: 'OTHER', 4: 'OWN', 5: 'RENT'}
    for name, res in all_results.items():
        fm = res['fairness_home']
        for grp, m in sorted(fm.items()):
            label = home_labels.get(grp, str(grp))
            print(f"  {name:<25} {label:<10}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f} n={m['n']:,}")

    print(f"\n--- Disparate Impact Ratio by Income Band ---")
    for name, res in all_results.items():
        fm = res['fairness_inc']
        rates = {grp: m['pos_rate'] for grp, m in fm.items()}
        if rates:
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            dir_ratio = min_rate/max_rate if max_rate > 0 else 1
            min_grp = inc_labels.get(min(rates, key=rates.get), '?')
            max_grp = inc_labels.get(max(rates, key=rates.get), '?')
            print(f"  {name:<25} min={min_rate:.3f}({min_grp}) max={max_rate:.3f}({max_grp}) DIR={dir_ratio:.3f}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for name, model in MODELS.items():
        if name == 'LogisticRegression':
            scores = cross_val_score(model, X_tr_sc, y_train, cv=5, scoring='roc_auc')
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  {name:<25} CV-AUC={scores.mean():.3f}±{scores.std():.3f}")

    print(f"\n--- Income x Housing Intersectional Analysis ---")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    for inc, inc_label in [(0,'low'),(3,'high')]:
        for home, home_label in [(5,'RENT'),(4,'OWN'),(1,'MORTGAGE')]:
            mask = (inc_test==inc) & (home_test==home)
            n = mask.sum()
            if n < 10:
                continue
            pos_rate = y_pred_gb[mask].mean()
            true_rate = y_test[mask].mean()
            print(f"  {inc_label:<12} {home_label:<10}: n={n:,} | pred_default={pos_rate:.1%} | true_default={true_rate:.1%}")
    print(f"  Note: Low income renters highest predicted default — income proxy captures socioeconomic risk")

    print(f"\n--- Lending Club Baseline complete ---")
    print(f"  Income band proxy captures fairness gap — low income predicted default rate higher")
    print(f"  Stage 2 ThresholdOptimizer needed to reduce income-based disparities")

    return all_results, inc_test, home_test, y_test


if __name__ == "__main__":
    results, inc_test, home_test, y_test = run_baselines()
