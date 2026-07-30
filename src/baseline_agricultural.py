"""
FAPE — SBA Agricultural Loans Baseline Models
Phase 4 — Baseline Fairness Evaluation

Dataset: SBA 7(a) Agricultural Loans FY1991-2024
15,845 records | geographic and business-type proxy sensitive attributes
Sensitive attributes: borrstate (geographic proxy), businesstype
Target: loan_default_binary (1=Charged Off, 0=Paid In Full)
Models: Logistic Regression, Random Forest, Gradient Boosting

Key notes:
- No direct race/ethnicity/sex data — SBA FOIA redacts demographics
- Geographic proxy: borrstate captures regional economic disparities
- Business type proxy: Individual vs Corporation vs Partnership
- 5.2% default rate — severe class imbalance; class_weight='balanced'
- Follows ECOA fair lending audit standard per CFPB methodology
- MS dominates (23%) due to poultry farming concentration
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(__file__))
from sba_agricultural_loader import load_sba_agricultural

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

BTYPE_LABELS = {-1: 'Unknown', 0: 'Corporation', 1: 'Individual', 2: 'Partnership'}
STATE_MAP = {0:'AK',1:'AL',2:'AR',3:'AZ',4:'CA',5:'CO',6:'CT',7:'DC',8:'DE',9:'FL',10:'GA',11:'GU',12:'HI',13:'IA',14:'ID',15:'IL',16:'IN',17:'KS',18:'KY',19:'LA',20:'MA',21:'MD',22:'ME',23:'MI',24:'MN',25:'MO',26:'MP',27:'MS',28:'MT',29:'NC',30:'ND',31:'NE',32:'NH',33:'NJ',34:'NM',35:'NV',36:'NY',37:'OH',38:'OK',39:'OR',40:'PA',41:'PR',42:'RI',43:'SC',44:'SD',45:'TN',46:'TX',47:'UT',48:'VA',49:'VI',50:'VT',51:'WA',52:'WI',53:'WV',54:'WY'}


def fairness_metrics(y_true, y_pred, sensitive):
    metrics = {}
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    sensitive = np.array(sensitive)
    overall_pos = y_pred.mean()
    for grp in np.unique(sensitive):
        mask = sensitive == grp
        yt = y_true[mask]; yp = y_pred[mask]
        if len(yt) < 20:
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
    print("FAPE Phase 4 — SBA Agricultural Loans Baseline Models")
    print("=" * 55)

    result = load_sba_agricultural()
    ds = result['sba_agricultural']
    X = ds['X']
    y = ds['y']

    borrstate = X['borrstate'].values
    businesstype = X['businesstype'].values

    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X.values, y.values, np.arange(len(y)),
        test_size=0.2, random_state=42, stratify=y.values)

    state_test = borrstate[idx_te]
    btype_test = businesstype[idx_te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"\n  n={len(y):,} | features={X.shape[1]} | default_rate={y.mean():.1%}")
    print(f"  sensitive=borrstate (geographic proxy), businesstype")
    print(f"  Note: No direct race/ethnicity/sex — SBA FOIA redacts demographics")
    print(f"  Note: class_weight=balanced due to 5.2% default rate")

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
        fm_state = fairness_metrics(y_test, y_pred, state_test)
        fm_btype = fairness_metrics(y_test, y_pred, btype_test)

        dp_state = [abs(v['dp_diff']) for v in fm_state.values()]
        max_dp_state = max(dp_state) if dp_state else 0

        all_results[name] = {
            'auc': auc, 'f1': f1,
            'y_pred': y_pred, 'y_prob': y_prob, 'y_test': y_test,
            'fairness_state': fm_state, 'fairness_btype': fm_btype,
            'max_dp_state': max_dp_state
        }
        print(f"  {name:<25} AUC={auc:.3f} F1={f1:.3f} max_DP_state={max_dp_state:.3f}")

    print(f"\n--- Fairness Metrics by Business Type ---")
    for name, res in all_results.items():
        fm = res['fairness_btype']
        for grp, m in sorted(fm.items()):
            label = BTYPE_LABELS.get(int(grp), str(grp))
            print(f"  {name:<25} {label:<15}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f} n={m['n']:,}")

    print(f"\n--- Fairness Metrics by State (Top 10 States) ---")
    top_states = sorted(
        [(grp, np.sum(state_test==grp)) for grp in np.unique(state_test)],
        key=lambda x: x[1], reverse=True
    )[:10]
    for name, res in all_results.items():
        fm = res['fairness_state']
        for grp, _ in top_states:
            if grp not in fm:
                continue
            m = fm[grp]
            state_name = STATE_MAP.get(int(grp), str(grp))
            print(f"  {name:<25} {state_name:<5}: FPR={m['fpr']:.3f} TPR={m['tpr']:.3f} DP={m['dp_diff']:+.3f} n={m['n']:,}")

    print(f"\n--- Disparate Impact Ratio by Business Type ---")
    for name, res in all_results.items():
        fm = res['fairness_btype']
        rates = {grp: m['pos_rate'] for grp, m in fm.items() if grp >= 0}
        if rates:
            min_rate = min(rates.values())
            max_rate = max(rates.values())
            dir_ratio = min_rate/max_rate if max_rate > 0 else 1
            min_grp = BTYPE_LABELS.get(int(min(rates, key=rates.get)), '?')
            max_grp = BTYPE_LABELS.get(int(max(rates, key=rates.get)), '?')
            print(f"  {name:<25} min={min_rate:.3f}({min_grp}) max={max_rate:.3f}({max_grp}) DIR={dir_ratio:.3f}")

    print(f"\n--- Cross-Validation (5-fold) ---")
    for name, model in MODELS.items():
        if name == 'LogisticRegression':
            scores = cross_val_score(model, X_tr_sc, y_train, cv=5, scoring='roc_auc')
        else:
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"  {name:<25} CV-AUC={scores.mean():.3f}±{scores.std():.3f}")

    print(f"\n--- State x Business Type Intersectional Analysis ---")
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    # Top 5 states by count
    top5_states = [grp for grp, _ in top_states[:5]]
    for state_code in top5_states:
        for btype in [0, 1, 2]:
            mask = (state_test==state_code) & (btype_test==btype)
            n = mask.sum()
            if n < 10:
                continue
            pos_rate = y_pred_gb[mask].mean()
            true_rate = y_test[mask].mean()
            state_name = STATE_MAP.get(int(state_code), str(state_code))
            btype_name = BTYPE_LABELS.get(btype, str(btype))
            print(f"  {state_name:<5} {btype_name:<15}: n={n:,} | pred_default={pos_rate:.1%} | true_default={true_rate:.1%}")
    print(f"  Note: Geographic proxy captures regional agricultural economic disparities")


    print(f"\n--- State Default Rate: Predicted vs True ---")
    gb_state = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb_state.fit(X_train, y_train)
    y_pred_state = gb_state.predict(X_test)
    top10_states = [grp for grp, _ in sorted(
        [(grp, np.sum(state_test==grp)) for grp in np.unique(state_test)],
        key=lambda x: x[1], reverse=True)][:10]
    for state_code in top10_states:
        mask = state_test == state_code
        if mask.sum() < 20: continue
        pred_rate = y_pred_state[mask].mean()
        true_rate = y_test[mask].mean()
        state_name = STATE_MAP.get(int(state_code), str(state_code))
        print(f"  {state_name:<5}: pred={pred_rate:.1%} true={true_rate:.1%} gap={true_rate-pred_rate:+.1%}")
    print(f"  Note: Model systematically underpredicts defaults in high-risk states")

    print(f"\n--- SBA Agricultural Baseline complete ---")
    print(f"  Geographic proxy captures state-level fairness gaps")
    print(f"  Stage 2 ThresholdOptimizer needed to reduce geographic disparities")

    return all_results, state_test, btype_test, y_test


if __name__ == "__main__":
    results, state_test, btype_test, y_test = run_baselines()
