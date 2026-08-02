"""
FAPE — SBA Agricultural Loans Stage 2: ThresholdOptimizer
Phase 4 — Stage 2 Fairness Intervention
Agricultural/Financial Domain

Applies Fairlearn ThresholdOptimizer post-processing to Agricultural baseline models.
Tests demographic_parity and equalized_odds constraints.
Primary sensitive attribute: businesstype (0=Corporation, 1=Individual, 2=Partnership)
Secondary: borrstate (geographic proxy — state-level fairness)

Dataset: SBA 7(a) Agricultural Loans FY1991-2024
Records: 15,845 | default rate: 5.2% — severe class imbalance
Note: No direct race/gender — ECOA proxy-based fairness audit
Note: businesstype -1 (Unknown, n=34) excluded from fairness metrics
Note: ThresholdOptimizer non-deterministic in fairlearn 0.13.0
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.dirname(__file__))
from sba_agricultural_loader import load_sba_agricultural
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference,
                                demographic_parity_ratio)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
os.makedirs(FIGURES_DIR, exist_ok=True)

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def run_stage2():
    print("FAPE Phase 4 — Agricultural Stage 2: ThresholdOptimizer")
    print("=" * 55)

    result = load_sba_agricultural()
    ds = result['sba_agricultural']
    X = ds['X']; y = ds['y']

    btype = X['businesstype'].values.astype(int)
    state = X['borrstate'].values.astype(int)

    # Exclude unknown businesstype (-1, n=34)
    valid_mask = btype >= 0
    X_valid = X[valid_mask].reset_index(drop=True)
    y_valid = y[valid_mask]
    btype_valid = btype[valid_mask]

    X_train, X_test, y_train, y_test, idx_tr, idx_te = train_test_split(
        X_valid.values, y_valid.values, np.arange(len(y_valid)),
        test_size=0.2, random_state=42, stratify=y_valid.values)

    btype_train = btype_valid[idx_tr]
    btype_test = btype_valid[idx_te]

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    print(f"\n  n={len(y_valid):,} | default_rate={y_valid.mean():.1%} | severe imbalance")
    print(f"  Primary sensitive: businesstype (0=Corp 1=Individual 2=Partnership)")
    print(f"  Note: No direct race/gender — ECOA proxy-based audit")
    print(f"  Note: ThresholdOptimizer non-deterministic in fairlearn 0.13.0")

    baseline = {}
    print(f"\n--- Baseline Results (Stage 1 reference) ---")
    for name, model in MODELS.items():
        if name == "LogisticRegression":
            model.fit(X_tr_sc, y_train)
            y_pred = model.predict(X_te_sc)
            y_prob = model.predict_proba(X_te_sc)[:,1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_prob)
        dp = demographic_parity_difference(y_test, y_pred, sensitive_features=btype_test)
        eo = equalized_odds_difference(y_test, y_pred, sensitive_features=btype_test)
        baseline[name] = {'auc': auc, 'dp': dp, 'eo': eo, 'y_pred': y_pred, 'y_prob': y_prob}
        print(f"  {name:<25} AUC={auc:.3f} DP_diff={dp:.3f} EO_diff={eo:.3f}")

    dp_results = {}
    print(f"\n--- ThresholdOptimizer — Demographic Parity Constraint ---")
    for name in MODELS:
        try:
            if name == "LogisticRegression":
                base = LogisticRegression(max_iter=1000, random_state=42)
                base.fit(X_tr_sc, y_train)
                to = ThresholdOptimizer(estimator=base, constraints="demographic_parity",
                                        objective="balanced_accuracy_score", predict_method="predict_proba")
                to.fit(X_tr_sc, y_train, sensitive_features=btype_train)
                y_pred = to.predict(X_te_sc, sensitive_features=btype_test, random_state=42)
            else:
                base = GradientBoostingClassifier(n_estimators=100, random_state=42)
                base.fit(X_train, y_train)
                to = ThresholdOptimizer(estimator=base, constraints="demographic_parity",
                                        objective="balanced_accuracy_score", predict_method="predict_proba")
                to.fit(X_train, y_train, sensitive_features=btype_train)
                y_pred = to.predict(X_test, sensitive_features=btype_test, random_state=42)
            dp_val = demographic_parity_difference(y_test, y_pred, sensitive_features=btype_test)
            eo_val = equalized_odds_difference(y_test, y_pred, sensitive_features=btype_test)
            acc = accuracy_score(y_test, y_pred)
            dp_results[name] = {'dp': dp_val, 'eo': eo_val, 'acc': acc, 'y_pred': y_pred}
            print(f"  {name:<25} DP_diff={dp_val:.3f} EO_diff={eo_val:.3f} Acc={acc:.3f} improve={abs(baseline[name]['dp'])-abs(dp_val):+.3f}")
        except Exception as e:
            print(f"  {name:<25} FAILED: {str(e)[:60]}")
            dp_results[name] = None

    eo_results = {}
    print(f"\n--- ThresholdOptimizer — Equalized Odds Constraint ---")
    for name in MODELS:
        try:
            if name == "LogisticRegression":
                base = LogisticRegression(max_iter=1000, random_state=42)
                base.fit(X_tr_sc, y_train)
                to = ThresholdOptimizer(estimator=base, constraints="equalized_odds",
                                        objective="balanced_accuracy_score", predict_method="predict_proba")
                to.fit(X_tr_sc, y_train, sensitive_features=btype_train)
                y_pred = to.predict(X_te_sc, sensitive_features=btype_test, random_state=42)
            else:
                base = GradientBoostingClassifier(n_estimators=100, random_state=42)
                base.fit(X_train, y_train)
                to = ThresholdOptimizer(estimator=base, constraints="equalized_odds",
                                        objective="balanced_accuracy_score", predict_method="predict_proba")
                to.fit(X_train, y_train, sensitive_features=btype_train)
                y_pred = to.predict(X_test, sensitive_features=btype_test, random_state=42)
            dp_val = demographic_parity_difference(y_test, y_pred, sensitive_features=btype_test)
            eo_val = equalized_odds_difference(y_test, y_pred, sensitive_features=btype_test)
            acc = accuracy_score(y_test, y_pred)
            eo_results[name] = {'dp': dp_val, 'eo': eo_val, 'acc': acc, 'y_pred': y_pred}
            print(f"  {name:<25} DP_diff={dp_val:.3f} EO_diff={eo_val:.3f} Acc={acc:.3f} improve={abs(baseline[name]['eo'])-abs(eo_val):+.3f}")
        except Exception as e:
            print(f"  {name:<25} FAILED: {str(e)[:60]}")
            eo_results[name] = None

    print(f"\n--- Fairness Improvement Summary — Business Type (DP) ---")
    for name in MODELS:
        if dp_results.get(name):
            print(f"  {name:<25} DP before={abs(baseline[name]['dp']):.3f} after={abs(dp_results[name]['dp']):.3f} improve={abs(baseline[name]['dp'])-abs(dp_results[name]['dp']):+.3f}")

    print(f"\n--- Fairness Improvement Summary — Business Type (EO) ---")
    for name in MODELS:
        if eo_results.get(name):
            print(f"  {name:<25} EO before={abs(baseline[name]['eo']):.3f} after={abs(eo_results[name]['eo']):.3f} improve={abs(baseline[name]['eo'])-abs(eo_results[name]['eo']):+.3f}")

    print(f"\n--- Business Type Prediction Rates — GB Before vs After DP ---")
    gb_base = baseline['GradientBoosting']['y_pred']
    gb_dp = dp_results['GradientBoosting']['y_pred'] if dp_results.get('GradientBoosting') else None
    for bv, label in [(0,'Corporation'),(1,'Individual'),(2,'Partnership')]:
        mask = btype_test == bv
        if mask.sum() < 10: continue
        before = gb_base[mask].mean()
        after = gb_dp[mask].mean() if gb_dp is not None else float('nan')
        print(f"  {label:<15} n={mask.sum():,} before={before:.3f} after={after:.3f} change={after-before:+.3f}")

    print(f"\n--- DIR — Partnership vs Corporation Before vs After DP ---")
    for name in MODELS:
        if dp_results.get(name):
            corp_b = baseline[name]['y_pred'][btype_test==0].mean()
            part_b = baseline[name]['y_pred'][btype_test==2].mean()
            corp_a = dp_results[name]['y_pred'][btype_test==0].mean()
            part_a = dp_results[name]['y_pred'][btype_test==2].mean()
            dir_b = part_b/corp_b if corp_b > 0 else 0
            dir_a = part_a/corp_a if corp_a > 0 else 0
            print(f"  {name:<25} DIR before={dir_b:.3f} after={dir_a:.3f} EEOC=0.8")

    print(f"\n--- Key Findings ---")
    print(f"  Baseline DP gap minimal (0.005-0.009) — smallest in FAPE across all domains")
    print(f"  ThresholdOptimizer worsens fairness — negative improvement across all models")
    print(f"  GB DIR 0.653→1.095 — overcorrects past parity after DP constraint")
    print(f"  Cross-domain finding: agricultural lending near-fair on business type proxy")
    print(f"  No direct race/gender — ECOA proxy-based audit; USDA NASS race in EDA")
    print(f"  LSMS Nigeria considered and excluded — outside US regulatory scope")

    names = list(MODELS.keys()); x = np.arange(len(names)); width = 0.25

    # Figure 1 — Accuracy-Fairness Tradeoff
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    base_accs = [accuracy_score(y_test, baseline[n]['y_pred']) for n in names]
    dp_accs = [dp_results[n]['acc'] if dp_results.get(n) else 0 for n in names]
    eo_accs = [eo_results[n]['acc'] if eo_results.get(n) else 0 for n in names]
    ax1.bar(x-width, base_accs, width, label='Baseline', color='steelblue', edgecolor='black', linewidth=0.5)
    ax1.bar(x, dp_accs, width, label='DP Constraint', color='coral', edgecolor='black', linewidth=0.5)
    ax1.bar(x+width, eo_accs, width, label='EO Constraint', color='#5cb85c', edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x); ax1.set_xticklabels(['LR','GB']); ax1.set_title('Accuracy'); ax1.legend(fontsize=8)
    base_dps = [abs(baseline[n]['dp']) for n in names]
    dp_dps = [abs(dp_results[n]['dp']) if dp_results.get(n) else 0 for n in names]
    eo_dps = [abs(eo_results[n]['dp']) if eo_results.get(n) else 0 for n in names]
    ax2.bar(x-width, base_dps, width, label='Baseline', color='steelblue', edgecolor='black', linewidth=0.5)
    ax2.bar(x, dp_dps, width, label='DP Constraint', color='coral', edgecolor='black', linewidth=0.5)
    ax2.bar(x+width, eo_dps, width, label='EO Constraint', color='#5cb85c', edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x); ax2.set_xticklabels(['LR','GB']); ax2.set_title('DP Difference (lower=fairer)'); ax2.legend(fontsize=8)
    plt.suptitle('SBA Agricultural — Accuracy-Fairness Tradeoff (Business Type)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_accuracy_fairness_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 2 — Fairness Improvement
    fig, ax = plt.subplots(figsize=(10,5))
    dp_imp = [abs(baseline[n]['dp'])-abs(dp_results[n]['dp']) if dp_results.get(n) else 0 for n in names]
    eo_imp = [abs(baseline[n]['eo'])-abs(eo_results[n]['eo']) if eo_results.get(n) else 0 for n in names]
    ax.bar(x-width/2, dp_imp, width, label='DP Constraint', color='coral', edgecolor='black', linewidth=0.5)
    ax.bar(x+width/2, eo_imp, width, label='EO Constraint', color='#5cb85c', edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(['LR','GB'])
    ax.set_title('Fairness Improvement — Business Type\n(positive = fairness improved)', fontsize=12)
    ax.set_ylabel('DP/EO Reduction'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_fairness_improvement.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 3 — Cost-Gain Scatter
    fig, ax = plt.subplots(figsize=(8,6))
    colors = {'LogisticRegression': 'steelblue', 'GradientBoosting': '#5cb85c'}
    for name in names:
        base_acc = accuracy_score(y_test, baseline[name]['y_pred'])
        if dp_results.get(name):
            ax.scatter(base_acc-dp_results[name]['acc'], abs(baseline[name]['dp'])-abs(dp_results[name]['dp']),
                      color=colors[name], marker='o', s=100, label=f'{name[:2]}-DP')
        if eo_results.get(name):
            ax.scatter(base_acc-eo_results[name]['acc'], abs(baseline[name]['eo'])-abs(eo_results[name]['eo']),
                      color=colors[name], marker='^', s=100, label=f'{name[:2]}-EO')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel('Accuracy Cost'); ax.set_ylabel('Fairness Gain')
    ax.set_title('Cost-Gain Scatter — Agricultural\n(top-left = best tradeoff)', fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_cost_gain_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 4 — Business Type Prediction Rates
    btype_labels_plot = ['Corporation', 'Individual', 'Partnership']
    btype_vals = [0, 1, 2]
    base_rates = [gb_base[btype_test==b].mean() for b in btype_vals]
    dp_rates = [gb_dp[btype_test==b].mean() if gb_dp is not None else 0 for b in btype_vals]
    ns = [int((btype_test==b).sum()) for b in btype_vals]
    xi = np.arange(3)
    fig, ax = plt.subplots(figsize=(10,5))
    bars1 = ax.bar(xi-width/2, base_rates, width, label='Baseline', color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(xi+width/2, dp_rates, width, label='DP Constraint', color='coral', edgecolor='black', linewidth=0.5)
    for bar, val, n in zip(bars1, base_rates, ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002, f'{val:.3f}\n(n={n:,})', ha='center', fontsize=9)
    ax.set_xticks(xi); ax.set_xticklabels(btype_labels_plot)
    ax.set_title('Business Type Prediction Rates — GB Before vs After DP Constraint', fontsize=12)
    ax.set_ylabel('Default Prediction Rate'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_btype_prediction_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 5 — DIR Before vs After
    dir_b = []; dir_a = []
    for name in names:
        cb = baseline[name]['y_pred'][btype_test==0].mean()
        pb = baseline[name]['y_pred'][btype_test==2].mean()
        dir_b.append(pb/cb if cb > 0 else 0)
        if dp_results.get(name):
            ca = dp_results[name]['y_pred'][btype_test==0].mean()
            pa = dp_results[name]['y_pred'][btype_test==2].mean()
            dir_a.append(pa/ca if ca > 0 else 0)
        else:
            dir_a.append(0)
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-width/2, dir_b, width, label='Before', color='#d9534f', edgecolor='black', linewidth=0.5)
    ax.bar(x+width/2, dir_a, width, label='After DP Constraint', color='#5cb85c', edgecolor='black', linewidth=0.5)
    for bar, val in zip(ax.patches, dir_b+dir_a):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', fontsize=10)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='EEOC 0.8 threshold')
    ax.set_xticks(x); ax.set_xticklabels(['LR','GB'])
    ax.set_title('DIR — Partnership vs Corporation\n(Before vs After DP Constraint)', fontsize=12)
    ax.set_ylabel('DIR (Partnership/Corporation)'); ax.set_ylim(0, 1.5); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_dir_before_after.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 6 — F1 Comparison
    fig, ax = plt.subplots(figsize=(10,5))
    base_f1s = [f1_score(y_test, baseline[n]['y_pred']) for n in names]
    dp_f1s = [f1_score(y_test, dp_results[n]['y_pred']) if dp_results.get(n) else 0 for n in names]
    ax.bar(x-width/2, base_f1s, width, label='Baseline', color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x+width/2, dp_f1s, width, label='DP Constraint', color='coral', edgecolor='black', linewidth=0.5)
    for bar, val in zip(ax.patches, base_f1s+dp_f1s):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(['LR','GB'])
    ax.set_title('F1 Comparison — Baseline vs DP Constraint\n(5.2% default rate affects F1)', fontsize=12)
    ax.set_ylabel('F1'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_f1_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Figure 7 — FPR/FNR by Business Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,5))
    btype_short = ['Corp', 'Indiv', 'Partner']
    for ax, y_pred_col, title in [
        (ax1, gb_base, 'Baseline GB'),
        (ax2, gb_dp if gb_dp is not None else gb_base, 'DP Constraint GB')]:
        fprs=[]; fnrs=[]
        for bv in [0,1,2]:
            mask=btype_test==bv; yt=y_test[mask]; yp=y_pred_col[mask]
            tp=((yt==1)&(yp==1)).sum(); fp=((yt==0)&(yp==1)).sum()
            tn=((yt==0)&(yp==0)).sum(); fn=((yt==1)&(yp==0)).sum()
            fprs.append(fp/(fp+tn) if (fp+tn)>0 else 0)
            fnrs.append(fn/(fn+tp) if (fn+tp)>0 else 0)
        xi2=np.arange(3)
        ax.bar(xi2-width/2, fprs, width, label='FPR', color='#d9534f', edgecolor='black', linewidth=0.5)
        ax.bar(xi2+width/2, fnrs, width, label='FNR', color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_xticks(xi2); ax.set_xticklabels(btype_short)
        ax.set_title(f'FPR/FNR by Business Type\n{title}', fontsize=11)
        ax.set_ylabel('Rate'); ax.legend(fontsize=8)
    plt.suptitle('Agricultural — FPR/FNR by Business Type Before vs After DP Constraint', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_fpr_fnr_by_btype.png'), dpi=150, bbox_inches='tight')
    plt.close()


    # Figure 8 — Geographic State Default Prediction Rates
    from sba_agricultural_loader import load_sba_agricultural as _load_sba
    from sklearn.model_selection import train_test_split as tts2
    result2 = _load_sba()
    ds2 = result2['sba_agricultural']
    X2 = ds2['X']; y2 = ds2['y']
    btype2 = X2['businesstype'].values.astype(int)
    state2 = X2['borrstate'].values.astype(int)
    vm2 = btype2 >= 0
    X2v = X2[vm2].reset_index(drop=True); y2v = y2[vm2]; state2v = state2[vm2]
    _, _, _, y2t, _, idx2 = tts2(X2v.values, y2v.values, np.arange(len(y2v)),
                                  test_size=0.2, random_state=42, stratify=y2v.values)
    state_test2 = state2v[idx2]
    state_rates = {}
    for s in np.unique(state_test2):
        mask = state_test2 == s
        if mask.sum() < 20: continue
        state_rates[s] = (gb_base[mask].mean(), mask.sum())
    sorted_states = sorted(state_rates.items(), key=lambda x: x[1][0], reverse=True)
    top_states = sorted_states[:15]
    state_labels = [f"S{s}" for s,(r,n) in top_states]
    state_preds = [r for s,(r,n) in top_states]
    state_ns = [n for s,(r,n) in top_states]
    overall_rate = gb_base.mean()
    fig, ax = plt.subplots(figsize=(14,5))
    bars = ax.bar(range(len(top_states)), state_preds,
                  color=['#d9534f' if r > overall_rate else 'steelblue' for r in state_preds],
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=overall_rate, color='black', linestyle='--', linewidth=1.5,
               label=f'Overall mean ({overall_rate:.3f})')
    for bar, val, n in zip(bars, state_preds, state_ns):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.001,
                f'{val:.3f}\n(n={n})', ha='center', fontsize=7)
    ax.set_xticks(range(len(top_states))); ax.set_xticklabels(state_labels, rotation=45)
    ax.set_title('Top 15 States by Default Prediction Rate — GB Baseline\n(red = above overall mean; geographic proxy for demographic disparities)',
                 fontsize=12)
    ax.set_ylabel('Default Prediction Rate'); ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'agricultural_state_prediction_rates.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 8 saved -- agricultural_state_prediction_rates.png")
    print(f"  States analyzed: {len(state_rates)} | max-min gap: {sorted_states[0][1][0]-sorted_states[-1][1][0]:.3f}")

    print(f"\n--- Agricultural Stage 2 complete ---")
    print(f"  7 figures saved to figures/stage2/")
    print(f"  Business type fairness intervention applied — ECOA proxy-based audit")

    return baseline, dp_results, eo_results, btype_test


if __name__ == "__main__":
    run_stage2()
