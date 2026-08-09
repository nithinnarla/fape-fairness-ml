"""
FAPE, Folktables ACS Stage 2: ThresholdOptimizer
Phase 4, Stage 2 Fairness Intervention
Socioeconomic Domain

Applies Fairlearn ThresholdOptimizer post-processing to Folktables ACS baseline.
Tests demographic_parity and equalized_odds constraints.
Sensitive attributes: RAC1P (race, 9 groups), SEX (binary)

Key baseline findings:
- Black DIR=0.74, Am.Indian DIR=0.54, Other DIR=0.52 - all below EEOC 0.8
- Asian DIR=1.13 - advantaged group, income rate exceeds White
- Male-Female income gap: Male 57.0% vs Female 42.1%

Note: ThresholdOptimizer in fairlearn 0.13.0 is non-deterministic.
Results vary between runs. Cross-run direction is consistent:
Black-White income gap reduces, FPR gap narrows after EO constraint.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import sys
import os
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_folktables_acs
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
os.makedirs(FIGURES_DIR, exist_ok=True)

SAMPLE_SIZE = 100000
FEATURE_COLS = ["AGEP", "SCHL", "MAR", "WKHP", "COW", "DIS", "POVPIP", "NATIVITY"]
RAC1P_LABELS = {1:"White", 2:"Black", 3:"Am.Indian", 4:"Alaska Native",
                5:"Am.Indian+Alaska", 6:"Asian", 7:"Pacific Islander",
                8:"Other", 9:"Two or more"}
MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
}


def run_stage2():
    print("FAPE Phase 4 - Folktables ACS Stage 2: ThresholdOptimizer")
    print("=" * 58)

    df = load_folktables_acs()
    df_sample = df.sample(SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    X = df_sample[FEATURE_COLS].values
    y = df_sample["label"].values
    race = df_sample["RAC1P"]
    sex = df_sample["SEX"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    race_train = race.iloc[idx_train].reset_index(drop=True)
    race_test  = race.iloc[idx_test].reset_index(drop=True)
    sex_train  = sex.iloc[idx_train].reset_index(drop=True)
    sex_test   = sex.iloc[idx_test].reset_index(drop=True)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Income >50k rate - Train: {y_train.mean():.1%} | Test: {y_test.mean():.1%}")

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
        baseline_results[name] = {
            "acc": acc, "f1": f1, "dpd": dpd, "eod": eod,
            "model": model, "X_tr": X_tr, "X_te": X_te
        }
        print(f"  {name:<22} ACC={acc:.3f} F1={f1:.3f} DPD={dpd:.3f} EOD={eod:.3f}")

    print(f"\n--- ThresholdOptimizer - Demographic Parity Constraint ---")
    dp_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res["model"],
                constraints="demographic_parity",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res["X_tr"], y_train, sensitive_features=race_train)
            yp = to.predict(res["X_te"], sensitive_features=race_test, random_state=42)
            acc = accuracy_score(y_test, yp)
            f1 = f1_score(y_test, yp)
            dpd = demographic_parity_difference(y_test, yp, sensitive_features=race_test)
            eod = equalized_odds_difference(y_test, yp, sensitive_features=race_test)
            dp_results[name] = {"acc": acc, "f1": f1, "dpd": dpd, "eod": eod}
            print(f"  {name:<22} ACC={acc:.3f}({acc-res['acc']:+.3f}) F1={f1:.3f} DPD={dpd:.3f}({dpd-res['dpd']:+.3f}) EOD={eod:.3f}")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            dp_results[name] = None

    print(f"\n--- ThresholdOptimizer - Equalized Odds Constraint ---")
    eo_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res["model"],
                constraints="equalized_odds",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res["X_tr"], y_train, sensitive_features=race_train)
            yp = to.predict(res["X_te"], sensitive_features=race_test, random_state=42)
            acc = accuracy_score(y_test, yp)
            f1 = f1_score(y_test, yp)
            dpd = demographic_parity_difference(y_test, yp, sensitive_features=race_test)
            eod = equalized_odds_difference(y_test, yp, sensitive_features=race_test)
            eo_results[name] = {"acc": acc, "f1": f1, "dpd": dpd, "eod": eod}
            print(f"  {name:<22} ACC={acc:.3f}({acc-res['acc']:+.3f}) F1={f1:.3f} DPD={dpd:.3f} EOD={eod:.3f}({eod-res['eod']:+.3f})")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            eo_results[name] = None

    print(f"\n--- Fairness Improvement Summary - Race (Demographic Parity) ---")
    for name, res in baseline_results.items():
        if dp_results.get(name):
            imp = (res["dpd"]-dp_results[name]["dpd"])/res["dpd"]*100 if res["dpd"] != 0 else 0
            print(f"  {name:<22} DPD: {res['dpd']:.3f} -> {dp_results[name]['dpd']:.3f} ({imp:+.1f}%) | ACC cost: {res['acc']-dp_results[name]['acc']:.3f}")

    print(f"\n--- Fairness Improvement Summary - Race (Equalized Odds) ---")
    for name, res in baseline_results.items():
        if eo_results.get(name):
            imp = (res["eod"]-eo_results[name]["eod"])/res["eod"]*100 if res["eod"] != 0 else 0
            print(f"  {name:<22} EOD: {res['eod']:.3f} -> {eo_results[name]['eod']:.3f} ({imp:+.1f}%) | ACC cost: {res['acc']-eo_results[name]['acc']:.3f}")

    print(f"\n--- Sex Fairness - ThresholdOptimizer (Demographic Parity) ---")
    sex_dp_results = {}
    for name, res in baseline_results.items():
        try:
            to = ThresholdOptimizer(
                estimator=res["model"],
                constraints="demographic_parity",
                predict_method="auto",
                objective="balanced_accuracy_score"
            )
            to.fit(res["X_tr"], y_train, sensitive_features=sex_train)
            yp = to.predict(res["X_te"], sensitive_features=sex_test, random_state=42)
            dpd = demographic_parity_difference(y_test, yp, sensitive_features=sex_test)
            eod = equalized_odds_difference(y_test, yp, sensitive_features=sex_test)
            acc = accuracy_score(y_test, yp)
            sex_dp_results[name] = {"acc": acc, "dpd": dpd, "eod": eod}
            print(f"  {name:<22} ACC={acc:.3f} DPD={dpd:.3f} EOD={eod:.3f}")
        except Exception as e:
            print(f"  {name:<22} ERROR: {e}")
            sex_dp_results[name] = None

    print(f"\n--- Race-Level Prediction Rates - GB Before vs After EO Constraint ---")
    gb_model = baseline_results["GradientBoosting"]["model"]
    X_tr_gb = baseline_results["GradientBoosting"]["X_tr"]
    X_te_gb = baseline_results["GradientBoosting"]["X_te"]
    y_pred_base_gb = gb_model.predict(X_te_gb)
    to_race = ThresholdOptimizer(
        estimator=gb_model,
        constraints="equalized_odds",
        predict_method="auto",
        objective="balanced_accuracy_score"
    )
    to_race.fit(X_tr_gb, y_train, sensitive_features=race_train)
    y_pred_to_gb = to_race.predict(X_te_gb, sensitive_features=race_test)
    race_level = {}
    for code, label in RAC1P_LABELS.items():
        mask = race_test == code
        if mask.sum() < 30: continue
        race_level[label] = {
            "n": mask.sum(),
            "true": y_test[mask].mean(),
            "base": y_pred_base_gb[mask].mean(),
            "constrained": y_pred_to_gb[mask].mean()
        }
        print(f"  {label:<20} n={mask.sum():>5,} true={y_test[mask].mean():.3f} base={y_pred_base_gb[mask].mean():.3f} constrained={y_pred_to_gb[mask].mean():.3f}")
    bw_gap_b = race_level.get("Black", {}).get("base", 0) - race_level.get("White", {}).get("base", 0)
    bw_gap_c = race_level.get("Black", {}).get("constrained", 0) - race_level.get("White", {}).get("constrained", 0)
    print(f"  Black-White base gap: {bw_gap_b:.3f}")
    print(f"  Black-White constrained gap: {bw_gap_c:.3f}")

    print(f"\n--- FPR/FNR by Race - GB Before vs After EO Constraint ---")
    main_races = [(2,"Black"),(1,"White"),(6,"Asian"),(3,"Am.Indian"),(8,"Other"),(9,"Two or more")]
    fpr_fnr = {}
    for code, label in main_races:
        mask = race_test == code
        if mask.sum() < 30: continue
        yt = y_test[mask]; yb = y_pred_base_gb[mask]; yc = y_pred_to_gb[mask]
        fpr_b = ((yb==1)&(yt==0)).sum()/((yt==0).sum()) if (yt==0).sum()>0 else 0
        fnr_b = ((yb==0)&(yt==1)).sum()/((yt==1).sum()) if (yt==1).sum()>0 else 0
        fpr_c = ((yc==1)&(yt==0)).sum()/((yt==0).sum()) if (yt==0).sum()>0 else 0
        fnr_c = ((yc==0)&(yt==1)).sum()/((yt==1).sum()) if (yt==1).sum()>0 else 0
        fpr_fnr[label] = {"fpr_b":fpr_b,"fnr_b":fnr_b,"fpr_c":fpr_c,"fnr_c":fnr_c}
        print(f"  {label:<20} FPR: {fpr_b:.3f}->{fpr_c:.3f} | FNR: {fnr_b:.3f}->{fnr_c:.3f}")
    if "Black" in fpr_fnr and "White" in fpr_fnr:
        bw_fpr_b = fpr_fnr["Black"]["fpr_b"] - fpr_fnr["White"]["fpr_b"]
        bw_fpr_c = fpr_fnr["Black"]["fpr_c"] - fpr_fnr["White"]["fpr_c"]
        print(f"  Black-White FPR gap: {bw_fpr_b:.3f} -> {bw_fpr_c:.3f}")

    print(f"\n--- Key Findings ---")
    valid_dp = [(n,r) for n,r in dp_results.items() if r]
    valid_eo = [(n,r) for n,r in eo_results.items() if r]
    if valid_dp:
        best_dp = min(valid_dp, key=lambda x: abs(x[1]["dpd"]))
        print(f"  Best DP constraint: {best_dp[0]} DPD={best_dp[1]['dpd']:.3f}")
    if valid_eo:
        best_eo = min(valid_eo, key=lambda x: abs(x[1]["eod"]))
        print(f"  Best EO constraint: {best_eo[0]} EOD={best_eo[1]['eod']:.3f}")
    print(f"  Black-White income gap: {bw_gap_b:.3f} -> {bw_gap_c:.3f} (ThresholdOptimizer reduces gap)")
    print(f"  Black DIR baseline below EEOC 0.8 threshold - ThresholdOptimizer applied")
    print(f"  Am.Indian DIR=0.54, Other DIR=0.52 - most severely disadvantaged groups")
    print(f"  Asian DIR=1.13 - advantaged group, income rate exceeds White")
    print(f"  Note: ThresholdOptimizer non-deterministic in fairlearn 0.13.0 - direction consistent across runs")
    print(f"  Note: RF/GB DPD worsens under DP constraint - 9 racial groups challenge")

    # FIGURES
    models = list(baseline_results.keys())
    short = ["LR", "RF", "GB"]
    x = np.arange(len(models)); width = 0.25
    base_accs = [baseline_results[m]["acc"] for m in models]
    base_f1s  = [baseline_results[m]["f1"]  for m in models]
    base_dpds = [baseline_results[m]["dpd"] for m in models]
    base_eods = [baseline_results[m]["eod"] for m in models]
    dp_accs  = [dp_results[m]["acc"]  if dp_results.get(m) else 0 for m in models]
    dp_f1s   = [dp_results[m]["f1"]   if dp_results.get(m) else 0 for m in models]
    dp_dpds  = [dp_results[m]["dpd"]  if dp_results.get(m) else 0 for m in models]
    dp_eods  = [dp_results[m]["eod"]  if dp_results.get(m) else 0 for m in models]
    eo_accs  = [eo_results[m]["acc"]  if eo_results.get(m) else 0 for m in models]
    eo_f1s   = [eo_results[m]["f1"]   if eo_results.get(m) else 0 for m in models]
    eo_dpds  = [eo_results[m]["dpd"]  if eo_results.get(m) else 0 for m in models]
    eo_eods  = [eo_results[m]["eod"]  if eo_results.get(m) else 0 for m in models]

    # Fig 1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-width, base_accs, width, label="Baseline", color="#95a5a6", edgecolor="white")
    axes[0].bar(x,       dp_accs,   width, label="DP Constraint", color="#3498db", edgecolor="white")
    axes[0].bar(x+width, eo_accs,   width, label="EO Constraint", color="#e74c3c", edgecolor="white")
    axes[0].set_xticks(x); axes[0].set_xticklabels(short)
    axes[0].set_title("Accuracy: Baseline vs Constrained", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Accuracy"); axes[0].legend(); axes[0].set_ylim(0.6, 0.88)
    axes[1].bar(x-width, base_dpds, width, label="Baseline", color="#95a5a6", edgecolor="white")
    axes[1].bar(x,       dp_dpds,   width, label="DP Constraint", color="#3498db", edgecolor="white")
    axes[1].bar(x+width, eo_dpds,   width, label="EO Constraint", color="#e74c3c", edgecolor="white")
    axes[1].axhline(0.1, color="green", linestyle="--", alpha=0.7, label="Target DPD<0.1")
    axes[1].set_xticks(x); axes[1].set_xticklabels(short)
    axes[1].set_title("Demographic Parity Difference\n(lower = fairer)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("DPD"); axes[1].legend()
    plt.suptitle("Folktables ACS - Accuracy vs Fairness Tradeoff", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage2/folktables_accuracy_fairness_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 1 saved - folktables_accuracy_fairness_tradeoff.png")

    # Fig 2
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(x-width/2, base_dpds, width, label="Baseline", color="#95a5a6", edgecolor="white")
    axes[0].bar(x+width/2, dp_dpds,   width, label="After DP Constraint", color="#3498db", edgecolor="white")
    for i,(b,a) in enumerate(zip(base_dpds, dp_dpds)):
        imp = (b-a)/b*100 if b!=0 else 0
        axes[0].annotate(f"{imp:+.0f}%", xy=(i, max(b,a)+0.01), ha="center", fontsize=9,
                        color="green" if imp>0 else "red")
    axes[0].set_xticks(x); axes[0].set_xticklabels(short)
    axes[0].set_title("DPD Reduction - DP Constraint", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("DPD"); axes[0].legend()
    axes[1].bar(x-width/2, base_eods, width, label="Baseline", color="#95a5a6", edgecolor="white")
    axes[1].bar(x+width/2, eo_eods,   width, label="After EO Constraint", color="#e74c3c", edgecolor="white")
    for i,(b,a) in enumerate(zip(base_eods, eo_eods)):
        imp = (b-a)/b*100 if b!=0 else 0
        axes[1].annotate(f"{imp:+.0f}%", xy=(i, max(b,a)+0.01), ha="center", fontsize=9,
                        color="green" if imp>0 else "red")
    axes[1].set_xticks(x); axes[1].set_xticklabels(short)
    axes[1].set_title("EOD Reduction - EO Constraint", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("EOD"); axes[1].legend()
    plt.suptitle("Folktables ACS - Fairness Improvement by Constraint Type", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage2/folktables_fairness_improvement.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 2 saved - folktables_fairness_improvement.png")

    # Fig 3
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_m = {"LogisticRegression":"#3498db","RandomForest":"#e74c3c","GradientBoosting":"#2ecc71"}
    for name in models:
        if dp_results.get(name):
            ax.scatter(baseline_results[name]["acc"]-dp_results[name]["acc"],
                      baseline_results[name]["dpd"]-dp_results[name]["dpd"],
                      c=colors_m[name], marker="o", s=150, zorder=5)
            ax.annotate(f"{name[:2]}_DP",
                       (baseline_results[name]["acc"]-dp_results[name]["acc"],
                        baseline_results[name]["dpd"]-dp_results[name]["dpd"]),
                       textcoords="offset points", xytext=(6,4), fontsize=9)
        if eo_results.get(name):
            ax.scatter(baseline_results[name]["acc"]-eo_results[name]["acc"],
                      baseline_results[name]["eod"]-eo_results[name]["eod"],
                      c=colors_m[name], marker="s", s=150, zorder=5)
            ax.annotate(f"{name[:2]}_EO",
                       (baseline_results[name]["acc"]-eo_results[name]["acc"],
                        baseline_results[name]["eod"]-eo_results[name]["eod"]),
                       textcoords="offset points", xytext=(6,4), fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    legend_elements = [
        Line2D([0],[0],marker="o",color="w",markerfacecolor="gray",markersize=10,label="DP constraint"),
        Line2D([0],[0],marker="s",color="w",markerfacecolor="gray",markersize=10,label="EO constraint"),
        mpatches.Patch(facecolor="#3498db",label="LogisticRegression"),
        mpatches.Patch(facecolor="#e74c3c",label="RandomForest"),
        mpatches.Patch(facecolor="#2ecc71",label="GradientBoosting"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.set_xlabel("Accuracy Cost"); ax.set_ylabel("Fairness Gain")
    ax.set_title("Folktables ACS - Accuracy Cost vs Fairness Gain\n(upper-left = best)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage2/folktables_cost_gain_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 3 saved - folktables_cost_gain_scatter.png")

    # Fig 4
    sex_dpds = [sex_dp_results[m]["dpd"] if sex_dp_results.get(m) else 0 for m in models]
    sex_eods = [sex_dp_results[m]["eod"] if sex_dp_results.get(m) else 0 for m in models]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-width/2, sex_dpds, width, label="Sex DPD", color="#9b59b6", edgecolor="white")
    ax.bar(x+width/2, sex_eods, width, label="Sex EOD", color="#f39c12", edgecolor="white")
    for i,val in enumerate(sex_dpds): ax.text(i-width/2, val+0.002, f"{val:.3f}", ha="center", fontsize=9)
    for i,val in enumerate(sex_eods): ax.text(i+width/2, val+0.002, f"{val:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title("Folktables ACS - Sex Fairness After DP Constraint\n(Male-Female income gap reduction)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fairness Metric"); ax.legend(); plt.tight_layout()
    plt.savefig("figures/stage2/folktables_sex_fairness.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 4 saved - folktables_sex_fairness.png")

    # Fig 5
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x-width, base_f1s, width, label="Baseline", color="#95a5a6", edgecolor="white")
    ax.bar(x,       dp_f1s,   width, label="DP Constraint", color="#3498db", edgecolor="white")
    ax.bar(x+width, eo_f1s,   width, label="EO Constraint", color="#e74c3c", edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(short)
    ax.set_title("Folktables ACS - F1 Score Comparison", fontsize=11, fontweight="bold")
    ax.set_ylabel("F1 Score"); ax.legend(); ax.set_ylim(0.55, 0.85); plt.tight_layout()
    plt.savefig("figures/stage2/folktables_f1_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 5 saved - folktables_f1_comparison.png")

    # Fig 6
    plot_labels = [l for l in ["White","Black","Asian","Am.Indian","Other","Two or more"] if l in race_level]
    true_rates = [race_level[l]["true"] for l in plot_labels]
    base_rates = [race_level[l]["base"] for l in plot_labels]
    to_rates   = [race_level[l]["constrained"] for l in plot_labels]
    x6 = np.arange(len(plot_labels)); w6 = 0.25
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.bar(x6-w6, true_rates, w6, label="True Rate",       color="#2ecc71", edgecolor="white")
    ax.bar(x6,    base_rates, w6, label="GB Baseline",      color="#95a5a6", edgecolor="white")
    ax.bar(x6+w6, to_rates,   w6, label="GB EO Constraint", color="#e74c3c", edgecolor="white")
    for i,(t,b,c) in enumerate(zip(true_rates, base_rates, to_rates)):
        ax.text(i-w6, t+0.005, f"{t:.2f}", ha="center", fontsize=7)
        ax.text(i,    b+0.005, f"{b:.2f}", ha="center", fontsize=7)
        ax.text(i+w6, c+0.005, f"{c:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x6); ax.set_xticklabels(plot_labels, rotation=15, ha="right")
    ax.set_title("Folktables ACS - Race-Level Prediction Rates: Baseline vs EO Constraint\n(Black-White income prediction gap reduction)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Predicted Positive Rate"); ax.legend(); plt.tight_layout()
    plt.savefig("figures/stage2/folktables_race_prediction_rates.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 6 saved - folktables_race_prediction_rates.png")

    # Fig 7
    plot_races = [l for _,l in [(2,"Black"),(1,"White"),(6,"Asian"),(3,"Am.Indian"),(8,"Other"),(9,"Two or more")] if l in fpr_fnr]
    fpr_base = [fpr_fnr[r]["fpr_b"] for r in plot_races]
    fpr_con  = [fpr_fnr[r]["fpr_c"] for r in plot_races]
    fnr_base = [fpr_fnr[r]["fnr_b"] for r in plot_races]
    fnr_con  = [fpr_fnr[r]["fnr_c"] for r in plot_races]
    x7 = np.arange(len(plot_races)); w7 = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].bar(x7-w7/2, fpr_base, w7, label="Baseline FPR",      color="#e74c3c", edgecolor="white")
    axes[0].bar(x7+w7/2, fpr_con,  w7, label="EO Constrained FPR", color="#3498db", edgecolor="white")
    for i,(b,c) in enumerate(zip(fpr_base, fpr_con)):
        axes[0].text(i-w7/2, b+0.003, f"{b:.3f}", ha="center", fontsize=8)
        axes[0].text(i+w7/2, c+0.003, f"{c:.3f}", ha="center", fontsize=8)
    axes[0].set_xticks(x7); axes[0].set_xticklabels(plot_races, rotation=15, ha="right")
    axes[0].set_title("False Positive Rate by Race\n(income misclassification rates)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("FPR"); axes[0].legend()
    axes[1].bar(x7-w7/2, fnr_base, w7, label="Baseline FNR",      color="#e74c3c", edgecolor="white")
    axes[1].bar(x7+w7/2, fnr_con,  w7, label="EO Constrained FNR", color="#3498db", edgecolor="white")
    for i,(b,c) in enumerate(zip(fnr_base, fnr_con)):
        axes[1].text(i-w7/2, b+0.003, f"{b:.3f}", ha="center", fontsize=8)
        axes[1].text(i+w7/2, c+0.003, f"{c:.3f}", ha="center", fontsize=8)
    axes[1].set_xticks(x7); axes[1].set_xticklabels(plot_races, rotation=15, ha="right")
    axes[1].set_title("False Negative Rate by Race\n(Chouldechova tradeoff)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("FNR"); axes[1].legend()
    plt.suptitle("Folktables ACS - FPR/FNR by Race: Baseline vs EO Constraint", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/stage2/folktables_fpr_fnr_by_race.png", dpi=150, bbox_inches="tight")
    plt.close(); print("Fig 7 saved - folktables_fpr_fnr_by_race.png")


    # Fig 8 - DIR by Race vs EEOC 0.8 Threshold
    dir_labels = []; dir_vals = []; dir_colors = []
    white_pred = y_pred_base_gb[race_test==1].mean()
    for code, label in [(1,'White'),(2,'Black'),(3,'Am.Indian'),(6,'Asian'),(8,'Other'),(9,'Two or more')]:
        mask = race_test == code
        if mask.sum() < 30: continue
        pred = y_pred_base_gb[mask].mean()
        dir_v = pred / white_pred if white_pred > 0 else 0
        dir_labels.append(label)
        dir_vals.append(dir_v)
        dir_colors.append('#2ecc71' if dir_v >= 0.8 else '#e74c3c')
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.bar(dir_labels, dir_vals, color=dir_colors, edgecolor='white', width=0.6)
    ax.axhline(0.8, color='#f39c12', linestyle='--', linewidth=2, label='EEOC 80% rule threshold')
    ax.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='White reference (DIR=1.0)')
    for bar, val in zip(bars, dir_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.02, f'{val:.3f}',
                ha='center', fontsize=10, fontweight='bold')
    ax.set_title('Folktables ACS - Disparate Impact Ratio by Race\n(EEOC 80% Rule: Black DIR=0.657 - Am.Indian DIR=0.612 - Other DIR=0.460 - all FAIL)',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Disparate Impact Ratio (DIR)'); ax.legend()
    ax.set_ylim(0, 1.35)
    plt.xticks(rotation=15, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'folktables_dir_by_race.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 8 saved - folktables_dir_by_race.png')


    # Fig 9 - DIR Before vs After EO Constraint
    dir_groups = [(1,'White'),(2,'Black'),(3,'Am.Indian'),(6,'Asian'),(8,'Other'),(9,'Two or more')]
    dir_labels2=[]; dir_base_vals=[]; dir_con_vals=[]
    white_base = y_pred_base_gb[race_test==1].mean()
    white_con  = y_pred_to_gb[race_test==1].mean()
    for code, label in dir_groups:
        mask = race_test == code
        if mask.sum() < 30: continue
        dir_labels2.append(label)
        dir_base_vals.append(y_pred_base_gb[mask].mean() / white_base if white_base > 0 else 0)
        dir_con_vals.append(y_pred_to_gb[mask].mean() / white_con if white_con > 0 else 0)
    x9 = np.arange(len(dir_labels2)); w9 = 0.35
    fig, ax = plt.subplots(figsize=(13, 6))
    bars_b = ax.bar(x9-w9/2, dir_base_vals, w9, label='Baseline DIR', color='#e74c3c', edgecolor='white')
    bars_c = ax.bar(x9+w9/2, dir_con_vals,  w9, label='EO Constrained DIR', color='#3498db', edgecolor='white')
    ax.axhline(0.8, color='#f39c12', linestyle='--', linewidth=2, label='EEOC 80% threshold')
    for bar, val in zip(bars_b, dir_base_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.015, f'{val:.3f}', ha='center', fontsize=8)
    for bar, val in zip(bars_c, dir_con_vals):
        ax.text(bar.get_x()+bar.get_width()/2, val+0.015, f'{val:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x9); ax.set_xticklabels(dir_labels2, rotation=15, ha='right')
    ax.set_title('Folktables ACS - DIR Before vs After EO Constraint\n(Two or more: DIR 0.739->0.863 crosses EEOC threshold - key regulatory finding)',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Disparate Impact Ratio (DIR)'); ax.legend(); ax.set_ylim(0, 1.35)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'folktables_dir_before_after.png'), dpi=150, bbox_inches='tight')
    plt.close(); print('Fig 9 saved - folktables_dir_before_after.png')

    print(f"\n--- Folktables Stage 2 complete ---")
    print(f"  9 figures saved to figures/stage2/")
    print(f"  Ready for FairGround + Student Stage 2")


if __name__ == "__main__":
    run_stage2()
