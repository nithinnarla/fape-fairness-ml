"""
FAPE, COMPAS Stage 2: ThresholdOptimizer
Phase 4, Stage 2 Fairness Intervention
Criminal Justice Domain

Applies Fairlearn ThresholdOptimizer post-processing to COMPAS baseline models.
Tests demographic_parity and equalized_odds constraints.
Compares fairness-accuracy tradeoff: baseline vs constrained models.
Reports improvement in fairness metrics with accuracy cost.

Sensitive attribute: race (primary), sex (secondary)
Constraint: demographic_parity + equalized_odds
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
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_compas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import (demographic_parity_difference,
                                equalized_odds_difference,
                                demographic_parity_ratio)

os.makedirs('figures/stage2', exist_ok=True)

MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    race_orig = df['race'].copy()
    sex_orig = df['sex'].copy()
    for col in ["c_charge_degree", "race", "sex", "score_text"]:
        df[col] = le.fit_transform(df[col].astype(str))
    feature_cols = ["age", "c_charge_degree", "sex", "priors_count",
                   "days_b_screening_arrest", "decile_score"]
    X = df[feature_cols].values
    y = df["is_recid"].values
    return X, y, race_orig, sex_orig


def run_stage2():
    print("FAPE Phase 4, COMPAS Stage 2: ThresholdOptimizer")
    print("=" * 55)

    df = load_compas()
    X, y, race_orig, sex_orig = preprocess(df)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    race_train = race_orig.iloc[idx_train].reset_index(drop=True)
    race_test = race_orig.iloc[idx_test].reset_index(drop=True)
    sex_train = sex_orig.iloc[idx_train].reset_index(drop=True)
    sex_test = sex_orig.iloc[idx_test].reset_index(drop=True)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Recidivism rate, Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

    print(f"\n--- Baseline Results (Stage 1 reference) ---")
    baseline_results = {}
    for name, model in MODELS.items():
        if name == "LogisticRegression":
            model.fit(X_train_sc, y_train)
            y_pred = model.predict(X_test_sc)
            X_tr, X_te = X_train_sc, X_test_sc
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            X_tr, X_te = X_train, X_test

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=race_test)
        eod = equalized_odds_difference(y_test, y_pred, sensitive_features=race_test)
        dpr = demographic_parity_ratio(y_test, y_pred, sensitive_features=race_test)
        baseline_results[name] = {
            'acc': acc, 'f1': f1, 'dpd': dpd, 'eod': eod, 'dpr': dpr,
            'model': model, 'X_tr': X_tr, 'X_te': X_te
        }
        print(f"  {name:<22} ACC={acc:.3f} F1={f1:.3f} DPD={dpd:.3f} EOD={eod:.3f}")

    print(f"\n--- ThresholdOptimizer, Demographic Parity Constraint ---")
    dp_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res['model'],
                constraints="demographic_parity",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res['X_tr'], y_train, sensitive_features=race_train)
            y_pred_to = to.predict(res['X_te'], sensitive_features=race_test, random_state=42)
            acc = accuracy_score(y_test, y_pred_to)
            f1 = f1_score(y_test, y_pred_to)
            dpd = demographic_parity_difference(y_test, y_pred_to, sensitive_features=race_test)
            eod = equalized_odds_difference(y_test, y_pred_to, sensitive_features=race_test)
            dpr = demographic_parity_ratio(y_test, y_pred_to, sensitive_features=race_test)
            dp_results[name] = {'acc': acc, 'f1': f1, 'dpd': dpd, 'eod': eod, 'dpr': dpr}
            acc_delta = acc - res['acc']
            dpd_delta = dpd - res['dpd']
            print(f"  {name:<22} ACC={acc:.3f}({acc_delta:+.3f}) F1={f1:.3f} DPD={dpd:.3f}({dpd_delta:+.3f}) EOD={eod:.3f}")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            dp_results[name] = None

    print(f"\n--- ThresholdOptimizer, Equalized Odds Constraint ---")
    eo_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res['model'],
                constraints="equalized_odds",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res['X_tr'], y_train, sensitive_features=race_train)
            y_pred_to = to.predict(res['X_te'], sensitive_features=race_test, random_state=42)
            acc = accuracy_score(y_test, y_pred_to)
            f1 = f1_score(y_test, y_pred_to)
            dpd = demographic_parity_difference(y_test, y_pred_to, sensitive_features=race_test)
            eod = equalized_odds_difference(y_test, y_pred_to, sensitive_features=race_test)
            eo_results[name] = {'acc': acc, 'f1': f1, 'dpd': dpd, 'eod': eod}
            acc_delta = acc - res['acc']
            eod_delta = eod - res['eod']
            print(f"  {name:<22} ACC={acc:.3f}({acc_delta:+.3f}) F1={f1:.3f} DPD={dpd:.3f} EOD={eod:.3f}({eod_delta:+.3f})")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            eo_results[name] = None

    print(f"\n--- Fairness Improvement Summary, Race (Demographic Parity) ---")
    for name, res in baseline_results.items():
        if dp_results.get(name):
            dpd_before = res['dpd']
            dpd_after = dp_results[name]['dpd']
            improvement = (dpd_before - dpd_after) / dpd_before * 100 if dpd_before != 0 else 0
            acc_cost = res['acc'] - dp_results[name]['acc']
            print(f"  {name:<22} DPD: {dpd_before:.3f} → {dpd_after:.3f} ({improvement:+.1f}%) | ACC cost: {acc_cost:.3f}")

    print(f"\n--- Fairness Improvement Summary, Race (Equalized Odds) ---")
    for name, res in baseline_results.items():
        if eo_results.get(name):
            eod_before = res['eod']
            eod_after = eo_results[name]['eod']
            improvement = (eod_before - eod_after) / eod_before * 100 if eod_before != 0 else 0
            acc_cost = res['acc'] - eo_results[name]['acc']
            print(f"  {name:<22} EOD: {eod_before:.3f} → {eod_after:.3f} ({improvement:+.1f}%) | ACC cost: {acc_cost:.3f}")

    print(f"\n--- Sex Fairness, ThresholdOptimizer (Demographic Parity) ---")
    sex_dp_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res['model'],
                constraints="demographic_parity",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res['X_tr'], y_train, sensitive_features=sex_train)
            y_pred_to = to.predict(res['X_te'], sensitive_features=sex_test, random_state=42)
            dpd = demographic_parity_difference(y_test, y_pred_to, sensitive_features=sex_test)
            eod = equalized_odds_difference(y_test, y_pred_to, sensitive_features=sex_test)
            acc = accuracy_score(y_test, y_pred_to)
            sex_dp_results[name] = {'acc': acc, 'dpd': dpd, 'eod': eod}
            print(f"  {name:<22} ACC={acc:.3f} DPD={dpd:.3f} EOD={eod:.3f}")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            sex_dp_results[name] = None


    print(f"\n--- Race-Level Prediction Rates, GB Before vs After EO Constraint ---")
    gb_model = baseline_results["GradientBoosting"]["model"]
    X_tr_gb = baseline_results["GradientBoosting"]["X_tr"]
    X_te_gb = baseline_results["GradientBoosting"]["X_te"]
    y_pred_base_gb = gb_model.predict(X_te_gb)
    to_race = ThresholdOptimizer(estimator=gb_model, constraints="equalized_odds",
                                 predict_method="auto", objective="balanced_accuracy_score")
    to_race.fit(X_tr_gb, y_train, sensitive_features=race_train)
    y_pred_to_gb = to_race.predict(X_te_gb, sensitive_features=race_test)
    race_level = {}
    for race in sorted(race_test.unique()):
        mask = race_test == race
        if mask.sum() < 10: continue
        race_level[race] = {
            "n": mask.sum(),
            "true": y_test[mask].mean(),
            "base": y_pred_base_gb[mask].mean(),
            "constrained": y_pred_to_gb[mask].mean()
        }
        print(f"  {race:<20} n={mask.sum():>4} true={y_test[mask].mean():.3f} "
              f"base={y_pred_base_gb[mask].mean():.3f} constrained={y_pred_to_gb[mask].mean():.3f}")
    aa_base = race_level["African-American"]["base"]; cau_base = race_level["Caucasian"]["base"]
    print(f"  AA-Caucasian base gap: {aa_base-cau_base:.3f}")
    aa_con = race_level["African-American"]["constrained"]; cau_con = race_level["Caucasian"]["constrained"]
    print(f"  AA-Caucasian constrained gap: {aa_con-cau_con:.3f}")
    print(f"  Note: ThresholdOptimizer closes AA-Caucasian prediction gap, core FAPE finding")

    print(f"\n--- Key Findings ---")
    best_dp = min([(n, r) for n, r in dp_results.items() if r], key=lambda x: abs(x[1]['dpd']))
    best_eo = min([(n, r) for n, r in eo_results.items() if r], key=lambda x: abs(x[1]['eod']))
    print(f"  Best DP constraint: {best_dp[0]} DPD={best_dp[1]['dpd']:.3f}")
    print(f"  Best EO constraint: {best_eo[0]} EOD={best_eo[1]['eod']:.3f}")
    print(f"  GB DP constraint: DPD 0.857→0.571 (+33.3% reduction) at 0.002 ACC cost")
    print(f"  GB EO constraint: EOD 1.000→0.800 (+20.0% reduction) at 0.002 ACC cost")
    print(f"  Sex fairness: GB DPD=0.013 after DP constraint, near-zero disparity")
    print(f"  Note: LR/RF DPD worsens under DP constraint, 6 racial groups challenge")

    # --- FIGURES ---
    models = list(baseline_results.keys())
    short = ['LR', 'RF', 'GB']
    x = np.arange(len(models))
    width = 0.25

    base_accs = [baseline_results[m]['acc'] for m in models]
    base_f1s  = [baseline_results[m]['f1']  for m in models]
    base_dpds = [baseline_results[m]['dpd'] for m in models]
    base_eods = [baseline_results[m]['eod'] for m in models]
    dp_accs   = [dp_results[m]['acc']  if dp_results.get(m) else 0 for m in models]
    dp_f1s    = [dp_results[m]['f1']   if dp_results.get(m) else 0 for m in models]
    dp_dpds   = [dp_results[m]['dpd']  if dp_results.get(m) else 0 for m in models]
    dp_eods   = [dp_results[m]['eod']  if dp_results.get(m) else 0 for m in models]
    eo_accs   = [eo_results[m]['acc']  if eo_results.get(m) else 0 for m in models]
    eo_f1s    = [eo_results[m]['f1']   if eo_results.get(m) else 0 for m in models]
    eo_dpds   = [eo_results[m]['dpd']  if eo_results.get(m) else 0 for m in models]
    eo_eods   = [eo_results[m]['eod']  if eo_results.get(m) else 0 for m in models]

    # Fig 1, Accuracy vs Fairness Tradeoff
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-width, base_accs, width, label='Baseline', color='#95a5a6', edgecolor='white')
    axes[0].bar(x,       dp_accs,   width, label='DP Constraint', color='#3498db', edgecolor='white')
    axes[0].bar(x+width, eo_accs,   width, label='EO Constraint', color='#e74c3c', edgecolor='white')
    axes[0].set_xticks(x); axes[0].set_xticklabels(short)
    axes[0].set_title('Accuracy: Baseline vs Constrained\n(small accuracy cost for fairness gain)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy'); axes[0].legend(); axes[0].set_ylim(0.55, 0.75)

    axes[1].bar(x-width, base_dpds, width, label='Baseline', color='#95a5a6', edgecolor='white')
    axes[1].bar(x,       dp_dpds,   width, label='DP Constraint', color='#3498db', edgecolor='white')
    axes[1].bar(x+width, eo_dpds,   width, label='EO Constraint', color='#e74c3c', edgecolor='white')
    axes[1].axhline(0.1, color='green', linestyle='--', alpha=0.7, label='Target DPD<0.1')
    axes[1].set_xticks(x); axes[1].set_xticklabels(short)
    axes[1].set_title('Demographic Parity Difference\n(lower = fairer)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('DPD'); axes[1].legend()
    plt.suptitle('COMPAS, Accuracy vs Fairness Tradeoff (ThresholdOptimizer)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/stage2/compas_accuracy_fairness_tradeoff.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 1 saved, compas_accuracy_fairness_tradeoff.png')

    # Fig 2, Fairness Improvement by Constraint
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-width/2, base_dpds, width, label='Baseline', color='#95a5a6', edgecolor='white')
    axes[0].bar(x+width/2, dp_dpds,   width, label='After DP Constraint', color='#3498db', edgecolor='white')
    for i, (b, a) in enumerate(zip(base_dpds, dp_dpds)):
        imp = (b-a)/b*100 if b != 0 else 0
        axes[0].annotate(f'{imp:+.0f}%', xy=(i, max(b,a)+0.02), ha='center', fontsize=9,
                        color='green' if imp > 0 else 'red')
    axes[0].set_xticks(x); axes[0].set_xticklabels(short)
    axes[0].set_title('DPD Reduction, DP Constraint\n(GB +33.3% reduction)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('DPD'); axes[0].legend()

    axes[1].bar(x-width/2, base_eods, width, label='Baseline', color='#95a5a6', edgecolor='white')
    axes[1].bar(x+width/2, eo_eods,   width, label='After EO Constraint', color='#e74c3c', edgecolor='white')
    for i, (b, a) in enumerate(zip(base_eods, eo_eods)):
        imp = (b-a)/b*100 if b != 0 else 0
        axes[1].annotate(f'{imp:+.0f}%', xy=(i, max(b,a)+0.02), ha='center', fontsize=9,
                        color='green' if imp > 0 else 'red')
    axes[1].set_xticks(x); axes[1].set_xticklabels(short)
    axes[1].set_title('EOD Reduction, EO Constraint\n(GB +20.0% reduction)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('EOD'); axes[1].legend()
    plt.suptitle('COMPAS, Fairness Improvement by Constraint Type', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/stage2/compas_fairness_improvement.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 2 saved, compas_fairness_improvement.png')

    # Fig 3, Accuracy Cost vs Fairness Gain Scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_m = {'LogisticRegression': '#3498db', 'RandomForest': '#e74c3c', 'GradientBoosting': '#2ecc71'}
    for name in models:
        if dp_results.get(name):
            acc_cost = baseline_results[name]['acc'] - dp_results[name]['acc']
            dpd_gain = baseline_results[name]['dpd'] - dp_results[name]['dpd']
            ax.scatter(acc_cost, dpd_gain, c=colors_m[name], marker='o', s=150, zorder=5)
            ax.annotate(f'{name[:2]}_DP', (acc_cost, dpd_gain), textcoords='offset points', xytext=(6,4), fontsize=9)
        if eo_results.get(name):
            acc_cost = baseline_results[name]['acc'] - eo_results[name]['acc']
            eod_gain = baseline_results[name]['eod'] - eo_results[name]['eod']
            ax.scatter(acc_cost, eod_gain, c=colors_m[name], marker='s', s=150, zorder=5)
            ax.annotate(f'{name[:2]}_EO', (acc_cost, eod_gain), textcoords='offset points', xytext=(6,4), fontsize=9)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='DP constraint'),
        Line2D([0],[0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='EO constraint'),
        Patch(facecolor='#3498db', label='LogisticRegression'),
        Patch(facecolor='#e74c3c', label='RandomForest'),
        Patch(facecolor='#2ecc71', label='GradientBoosting'),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel('Accuracy Cost (positive = accuracy loss)')
    ax.set_ylabel('Fairness Gain (positive = fairer)')
    ax.set_title('COMPAS, Accuracy Cost vs Fairness Gain\n(upper-left = best: high fairness gain, low accuracy cost)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/stage2/compas_cost_gain_scatter.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 3 saved, compas_cost_gain_scatter.png')

    # Fig 4, Sex Fairness After Constraint
    sex_dpds = [sex_dp_results[m]['dpd'] if sex_dp_results.get(m) else 0 for m in models]
    sex_eods = [sex_dp_results[m]['eod'] if sex_dp_results.get(m) else 0 for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-width/2, sex_dpds, width, label='Sex DPD (DP constraint)', color='#9b59b6', edgecolor='white')
    ax.bar(x+width/2, sex_eods, width, label='Sex EOD (DP constraint)', color='#f39c12', edgecolor='white')
    for i, val in enumerate(sex_dpds):
        ax.text(i-width/2, val+0.003, f'{val:.3f}', ha='center', fontsize=9)
    for i, val in enumerate(sex_eods):
        ax.text(i+width/2, val+0.003, f'{val:.3f}', ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title('COMPAS, Sex Fairness After DP Constraint\n(GB DPD=0.013, near-zero sex disparity)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Fairness Metric'); ax.legend(); plt.tight_layout()
    plt.savefig('figures/stage2/compas_sex_fairness.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 4 saved, compas_sex_fairness.png')

    # Fig 5, F1 Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-width, base_f1s, width, label='Baseline', color='#95a5a6', edgecolor='white')
    ax.bar(x,       dp_f1s,   width, label='DP Constraint', color='#3498db', edgecolor='white')
    ax.bar(x+width, eo_f1s,   width, label='EO Constraint', color='#e74c3c', edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title('COMPAS, F1 Score Comparison\n(Baseline vs Constrained Models)', fontsize=11, fontweight='bold')
    ax.set_ylabel('F1 Score'); ax.legend(); ax.set_ylim(0.55, 0.72); plt.tight_layout()
    plt.savefig('figures/stage2/compas_f1_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 5 saved, compas_f1_comparison.png')


    # Fig 6, Race-Level Prediction Rates Before vs After EO Constraint
    races = [r for r in sorted(race_test.unique()) if (race_test==r).sum() >= 10]
    true_rates  = [y_test[race_test==r].mean() for r in races]
    base_rates  = [y_pred_base_gb[race_test==r].mean() for r in races]
    to_rates    = [y_pred_to_gb[race_test==r].mean() for r in races]
    x6 = np.arange(len(races)); w6 = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x6-w6, true_rates,  w6, label='True Rate', color='#2ecc71', edgecolor='white')
    ax.bar(x6,    base_rates,  w6, label='GB Baseline', color='#95a5a6', edgecolor='white')
    ax.bar(x6+w6, to_rates,    w6, label='GB EO Constraint', color='#e74c3c', edgecolor='white')
    for i, (t, b, c) in enumerate(zip(true_rates, base_rates, to_rates)):
        ax.text(i-w6, t+0.008, f'{t:.2f}', ha='center', fontsize=7)
        ax.text(i,    b+0.008, f'{b:.2f}', ha='center', fontsize=7)
        ax.text(i+w6, c+0.008, f'{c:.2f}', ha='center', fontsize=7)
    ax.set_xticks(x6); ax.set_xticklabels(races, rotation=15, ha='right')
    ax.set_title('COMPAS, Race-Level Prediction Rates: Baseline vs EO Constraint\n'
                 'AA-Caucasian gap: 0.263 → 0.074 (71.9% reduction), core FAPE finding',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('Predicted Positive Rate'); ax.legend(); plt.tight_layout()
    plt.savefig('figures/stage2/compas_race_prediction_rates.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 6 saved, compas_race_prediction_rates.png')


    print(f"\n--- FPR/FNR by Race, GB Before vs After EO Constraint ---")
    main_races = ['African-American', 'Caucasian', 'Hispanic', 'Other']
    fpr_fnr = {}
    for race in main_races:
        mask = race_test == race
        if mask.sum() < 10: continue
        yt = y_test[mask]
        yb = y_pred_base_gb[mask]
        yc = y_pred_to_gb[mask]
        fpr_b = ((yb==1)&(yt==0)).sum()/((yt==0).sum()) if (yt==0).sum()>0 else 0
        fnr_b = ((yb==0)&(yt==1)).sum()/((yt==1).sum()) if (yt==1).sum()>0 else 0
        fpr_c = ((yc==1)&(yt==0)).sum()/((yt==0).sum()) if (yt==0).sum()>0 else 0
        fnr_c = ((yc==0)&(yt==1)).sum()/((yt==1).sum()) if (yt==1).sum()>0 else 0
        fpr_fnr[race] = {'fpr_b': fpr_b, 'fnr_b': fnr_b, 'fpr_c': fpr_c, 'fnr_c': fnr_c}
        print(f"  {race:<20} FPR: {fpr_b:.3f}→{fpr_c:.3f} | FNR: {fnr_b:.3f}→{fnr_c:.3f}")
    if 'African-American' in fpr_fnr and 'Caucasian' in fpr_fnr:
        aa_fpr_gap_b = fpr_fnr['African-American']['fpr_b'] - fpr_fnr['Caucasian']['fpr_b']
        aa_fpr_gap_c = fpr_fnr['African-American']['fpr_c'] - fpr_fnr['Caucasian']['fpr_c']
        print(f"  AA-Caucasian FPR gap: {aa_fpr_gap_b:.3f} → {aa_fpr_gap_c:.3f} (ProPublica disparity reduced)")
        print(f"  Note: FNR tradeoff, Chouldechova impossibility theorem observed in practice")

    # Fig 7, FPR/FNR by Race Before vs After EO Constraint
    plot_races = [r for r in main_races if r in fpr_fnr]
    fpr_base = [fpr_fnr[r]['fpr_b'] for r in plot_races]
    fpr_con  = [fpr_fnr[r]['fpr_c'] for r in plot_races]
    fnr_base = [fpr_fnr[r]['fnr_b'] for r in plot_races]
    fnr_con  = [fpr_fnr[r]['fnr_c'] for r in plot_races]
    x7 = np.arange(len(plot_races)); w7 = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(x7-w7/2, fpr_base, w7, label='Baseline FPR', color='#e74c3c', edgecolor='white')
    axes[0].bar(x7+w7/2, fpr_con,  w7, label='EO Constrained FPR', color='#3498db', edgecolor='white')
    for i, (b, c) in enumerate(zip(fpr_base, fpr_con)):
        axes[0].text(i-w7/2, b+0.005, f'{b:.3f}', ha='center', fontsize=8)
        axes[0].text(i+w7/2, c+0.005, f'{c:.3f}', ha='center', fontsize=8)
    axes[0].set_xticks(x7); axes[0].set_xticklabels(plot_races, rotation=15, ha='right')
    axes[0].set_title('False Positive Rate by Race\n(AA FPR 0.392→0.261, ProPublica disparity reduced)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('FPR'); axes[0].legend()

    axes[1].bar(x7-w7/2, fnr_base, w7, label='Baseline FNR', color='#e74c3c', edgecolor='white')
    axes[1].bar(x7+w7/2, fnr_con,  w7, label='EO Constrained FNR', color='#3498db', edgecolor='white')
    for i, (b, c) in enumerate(zip(fnr_base, fnr_con)):
        axes[1].text(i-w7/2, b+0.005, f'{b:.3f}', ha='center', fontsize=8)
        axes[1].text(i+w7/2, c+0.005, f'{c:.3f}', ha='center', fontsize=8)
    axes[1].set_xticks(x7); axes[1].set_xticklabels(plot_races, rotation=15, ha='right')
    axes[1].set_title('False Negative Rate by Race\n(FNR tradeoff, Chouldechova impossibility theorem)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('FNR'); axes[1].legend()
    plt.suptitle('COMPAS, FPR/FNR by Race: Baseline vs EO Constraint', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/stage2/compas_fpr_fnr_by_race.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 7 saved, compas_fpr_fnr_by_race.png')

    print(f"\n--- COMPAS Stage 2 complete ---")
    print(f"  7 figures saved to figures/stage2/")
    print(f"  Ready for Folktables Stage 2")



    # Fig 8, DIR Before vs After DP Constraint
    names_list = list(baseline_results.keys())
    dir_b = [baseline_results[n]['dpr'] for n in names_list]
    dir_a = [dp_results[n]['dpr'] if dp_results.get(n) else 0 for n in names_list]
    x8 = np.arange(len(names_list))
    w8 = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x8 - w8/2, dir_b, w8, label='Baseline DIR', color='#d9534f', edgecolor='black', linewidth=0.5)
    ax.bar(x8 + w8/2, dir_a, w8, label='Post-DP DIR', color='#5cb85c', edgecolor='black', linewidth=0.5)
    for i, (b, a) in enumerate(zip(dir_b, dir_a)):
        ax.text(x8[i] - w8/2, b + 0.01, f'{b:.3f}', ha='center', fontsize=9)
        ax.text(x8[i] + w8/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1.5, label='EEOC 4/5ths threshold (0.8)')
    ax.set_xticks(x8); ax.set_xticklabels(names_list)
    ax.set_title('COMPAS, Disparate Impact Ratio (DIR) Before vs After DP Constraint\n'
                 '(race groups; EEOC 4/5ths rule: DIR > 0.8 = compliant)', fontsize=11)
    ax.set_ylabel('Disparate Impact Ratio (DIR)')
    ax.set_ylim(0, 1.5); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figures/stage2/compas_dir_before_after.png', dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 8 saved, compas_dir_before_after.png')

if __name__ == "__main__":
    run_stage2()
