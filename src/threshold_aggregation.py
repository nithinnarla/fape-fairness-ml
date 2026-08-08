"""
FAPE, ThresholdOptimizer Results Aggregation
Stage 2: All Models × All Domains Summary

Aggregates ThresholdOptimizer results for all 3 models (LR, RF, GB)
across all 7 FAPE domains into a unified comparison table and figures.

Complements cross_domain_comparison.py (GB only) by showing
model-level variation within each domain.

Key question: Does GradientBoosting consistently outperform LR and RF
under ThresholdOptimizer constraints across domains?

Output: 6 figures saved to figures/stage2/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── All model results from Stage 2 scripts ─────────────────────────────────
# Format: domain → model → {baseline, post_dp, post_eo}
RESULTS = {
    'COMPAS': {
        'LR':  {'baseline_acc': 0.686, 'baseline_dpd': 0.545, 'baseline_eod': 0.701,
                'dp_acc': 0.657, 'dp_dpd': 0.714, 'eo_acc': 0.659, 'eo_eod': 0.654},
        'RF':  {'baseline_acc': 0.637, 'baseline_dpd': 0.568, 'baseline_eod': 0.686,
                'dp_acc': 0.602, 'dp_dpd': 0.714, 'eo_acc': 0.607, 'eo_eod': 0.731},
        'GB':  {'baseline_acc': 0.674, 'baseline_dpd': 0.857, 'baseline_eod': 1.000,
                'dp_acc': 0.675, 'dp_dpd': 0.571, 'eo_acc': 0.677, 'eo_eod': 0.659},
    },
    'Folktables': {
        'LR':  {'baseline_acc': 0.819, 'baseline_dpd': 0.352, 'baseline_eod': 0.571,
                'dp_acc': 0.799, 'dp_dpd': 0.340, 'eo_acc': 0.807, 'eo_eod': 0.333},
        'RF':  {'baseline_acc': 0.829, 'baseline_dpd': 0.319, 'baseline_eod': 0.823,
                'dp_acc': 0.794, 'dp_dpd': 0.394, 'eo_acc': 0.823, 'eo_eod': 0.861},
        'GB':  {'baseline_acc': 0.845, 'baseline_dpd': 0.320, 'baseline_eod': 0.333,
                'dp_acc': 0.825, 'dp_dpd': 0.339, 'eo_acc': 0.828, 'eo_eod': 0.334},
    },
    'Law School': {
        'LR':  {'baseline_acc': None, 'baseline_auc': 0.872, 'baseline_dpd': 0.329, 'baseline_eod': 0.543,
                'dp_acc': 0.765, 'dp_dpd': 0.011, 'eo_acc': 0.684, 'eo_eod': 0.060},
        'RF':  None,
        'GB':  {'baseline_acc': None, 'baseline_auc': 0.878, 'baseline_dpd': 0.351, 'baseline_eod': 0.528,
                'dp_acc': 0.754, 'dp_dpd': 0.030, 'eo_acc': 0.757, 'eo_eod': 0.007},
    },
    'Lending Club': {
        'LR':  {'baseline_acc': None, 'baseline_auc': 0.706, 'baseline_dpd': 0.018, 'baseline_eod': 0.038,
                'dp_acc': 0.616, 'dp_dpd': 0.019, 'eo_acc': 0.569, 'eo_eod': 0.047},
        'RF':  None,
        'GB':  {'baseline_acc': None, 'baseline_auc': 0.712, 'baseline_dpd': 0.024, 'baseline_eod': 0.053,
                'dp_acc': 0.651, 'dp_dpd': 0.018, 'eo_acc': 0.661, 'eo_eod': 0.049},
    },
    'Agricultural': {
        'LR':  {'baseline_acc': None, 'baseline_auc': 0.727, 'baseline_dpd': 0.005, 'baseline_eod': 0.005,
                'dp_acc': 0.665, 'dp_dpd': 0.016, 'eo_acc': 0.636, 'eo_eod': 0.047},
        'RF':  None,
        'GB':  {'baseline_acc': None, 'baseline_auc': 0.938, 'baseline_dpd': 0.009, 'baseline_eod': 0.073,
                'dp_acc': 0.862, 'dp_dpd': 0.031, 'eo_acc': 0.834, 'eo_eod': 0.177},
    },
    'FairGround (Education/law_school_lequy)': {
        'LR':  {'baseline_acc': 0.913, 'baseline_dpd': 0.329, 'baseline_eod': 0.543,
                'dp_acc': 0.765, 'dp_dpd': 0.010, 'eo_acc': 0.683, 'eo_eod': 0.061},
        'RF':  {'baseline_acc': 0.907, 'baseline_dpd': 0.336, 'baseline_eod': 0.524,
                'dp_acc': 0.897, 'dp_dpd': 0.012, 'eo_acc': 0.906, 'eo_eod': 0.472},
        'GB':  {'baseline_acc': 0.910, 'baseline_dpd': 0.342, 'baseline_eod': 0.518,
                'dp_acc': 0.751, 'dp_dpd': 0.014, 'eo_acc': 0.762, 'eo_eod': 0.016},
    },
    'Student (math)': {
        'LR':  {'baseline_acc': 0.646, 'baseline_dpd': 0.212, 'baseline_eod': 0.204,
                'dp_acc': 0.620, 'dp_dpd': 0.010, 'eo_acc': 0.646, 'eo_eod': 0.188},
        'RF':  {'baseline_acc': 0.633, 'baseline_dpd': 0.235, 'baseline_eod': 0.263,
                'dp_acc': 0.557, 'dp_dpd': 0.363, 'eo_acc': 0.633, 'eo_eod': 0.180},
        'GB':  {'baseline_acc': 0.658, 'baseline_dpd': 0.237, 'baseline_eod': 0.314,
                'dp_acc': 0.582, 'dp_dpd': 0.215, 'eo_acc': 0.658, 'eo_eod': 0.114},
    },
}

DOMAINS = list(RESULTS.keys())
MODELS = ['LR', 'RF', 'GB']

def safe_get(domain, model, key):
    entry = RESULTS.get(domain, {}).get(model)
    if entry is None:
        return None
    return entry.get(key)

def fmt(val):
    if val is None:
        return "N/A"
    return f"{val:.3f}"

def has_accuracy(domain):
    """True if this domain's baseline_acc is a real number, not None (i.e. reports accuracy not AUC)."""
    for m in MODELS:
        v = safe_get(domain, m, 'baseline_acc')
        if v is not None:
            return True
    return False
COLORS = {'LR': '#3498db', 'RF': '#e67e22', 'GB': '#2ecc71'}


def run_threshold_aggregation():
    print("FAPE, ThresholdOptimizer Results Aggregation")
    print("=" * 50)

    print("\n--- Baseline Accuracy by Model and Domain ---")
    for domain in DOMAINS:
        accs = {m: safe_get(domain, m, 'baseline_acc') for m in MODELS}
        print(f"  {domain:<15} LR={fmt(accs['LR'])} RF={fmt(accs['RF'])} GB={fmt(accs['GB'])}")

    print("\n--- Post-DP DPD by Model and Domain ---")
    for domain in DOMAINS:
        dpds = {m: safe_get(domain, m, 'dp_dpd') for m in MODELS}
        print(f"  {domain:<15} LR={fmt(dpds['LR'])} RF={fmt(dpds['RF'])} GB={fmt(dpds['GB'])}")

    print("\n--- Key Findings (computed from RESULTS, not hardcoded -- see Decision 12) ---")
    acc_doms = [d for d in DOMAINS if has_accuracy(d)]
    auc_doms = [d for d in DOMAINS if not has_accuracy(d)]
    gb_best_acc = sum(1 for d in acc_doms if safe_get(d, 'GB', 'baseline_acc') == max(
        safe_get(d, m, 'baseline_acc') or 0 for m in MODELS))
    print(f"  GB highest baseline accuracy in {gb_best_acc}/{len(acc_doms)} true-accuracy domains: {acc_doms}")
    gb_best_auc = sum(1 for d in auc_doms if safe_get(d, 'GB', 'baseline_auc') == max(
        safe_get(d, m, 'baseline_auc') or 0 for m in MODELS if safe_get(d, m, 'baseline_auc') is not None))
    print(f"  GB highest baseline AUC in {gb_best_auc}/{len(auc_doms)} AUC-only domains: {auc_doms}")

    best_dp_improve, best_dp_domain = -999, None
    for d in DOMAINS:
        base = safe_get(d, 'GB', 'baseline_dpd')
        after = safe_get(d, 'GB', 'dp_dpd')
        if base is not None and after is not None and base > 0:
            improve = (base - after) / base
            if improve > best_dp_improve:
                best_dp_improve, best_dp_domain = improve, d
    print(f"  GB best DP improvement: {best_dp_domain} ({best_dp_improve*100:.1f}% reduction)")

    worst_dp_improve, worst_dp_domain = 999, None
    for d in DOMAINS:
        base = safe_get(d, 'GB', 'baseline_dpd')
        after = safe_get(d, 'GB', 'dp_dpd')
        if base is not None and after is not None and base > 0:
            improve = (base - after) / base
            if improve < worst_dp_improve:
                worst_dp_improve, worst_dp_domain = improve, d
    direction = "counterproductive" if worst_dp_improve < 0 else "smallest improvement"
    print(f"  GB {direction}: {worst_dp_domain} ({worst_dp_improve*100:+.1f}% change)")

    w = 0.25

    # Figure 1 -- Baseline performance by model across domains, split by metric
    # (4 domains report true accuracy; 3 report AUC -- these cannot share one axis,
    # see Decision 13 in methodology_decisions.md)
    acc_domains = [d for d in DOMAINS if has_accuracy(d)]
    auc_domains = [d for d in DOMAINS if not has_accuracy(d)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x1 = np.arange(len(acc_domains))
    for i, model in enumerate(MODELS):
        vals = [safe_get(d, model, 'baseline_acc') for d in acc_domains]
        vals = [v if v is not None else 0 for v in vals]
        ax1.bar(x1 + (i-1)*w, vals, w, label=model, color=COLORS[model],
                edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(acc_domains, rotation=15, ha='right', fontsize=9)
    ax1.set_title('Baseline Accuracy (true classification accuracy)', fontsize=11)
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.5, 1.0)

    x2 = np.arange(len(auc_domains))
    for i, model in enumerate(MODELS):
        vals = [safe_get(d, model, 'baseline_auc') for d in auc_domains]
        if all(v is None for v in vals):
            continue
        vals = [v if v is not None else 0 for v in vals]
        ax2.bar(x2 + (i-1)*w, vals, w, label=model, color=COLORS[model],
                edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(auc_domains, rotation=15, ha='right', fontsize=9)
    ax2.set_title('Baseline AUC (RandomForest not implemented -- see Decision 16)', fontsize=11)
    ax2.set_ylabel('AUC')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.5, 1.0)

    plt.suptitle('Baseline Performance Across All 7 FAPE Domains -- Split by Metric Type', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_baseline_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved -- aggregation_baseline_accuracy.png (split accuracy/AUC panels)")

    # Figure 2 -- Post-DP DPD by model across domains
    x = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(MODELS):
        dpds = [safe_get(d, model, 'dp_dpd') for d in DOMAINS]
        dpds = [v if v is not None else 0 for v in dpds]
        ax.bar(x + (i-1)*w, dpds, w, label=f'{model} post-DP', color=COLORS[model],
               edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='EEOC threshold (0.1)')
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=15, ha='right', fontsize=9)
    ax.set_title('Post-ThresholdOptimizer DPD, LR vs RF vs GB\n'
                 'Across All 7 FAPE Domains (DP Constraint)', fontsize=12)
    ax.set_ylabel('Demographic Parity Disparity (DPD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_post_dp_dpd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved -- aggregation_post_dp_dpd.png")

    # Figure 3 -- Post-EO EOD by model across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(MODELS):
        eods = [safe_get(d, model, 'eo_eod') for d in DOMAINS]
        eods = [v if v is not None else 0 for v in eods]
        ax.bar(x + (i-1)*w, eods, w, label=f'{model} post-EO', color=COLORS[model],
               edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='EEOC threshold (0.1)')
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=15, ha='right', fontsize=9)
    ax.set_title('Post-ThresholdOptimizer EOD, LR vs RF vs GB\n'
                 'Across All 7 FAPE Domains (EO Constraint)', fontsize=12)
    ax.set_ylabel('Equalized Odds Disparity (EOD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_post_eo_eod.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved -- aggregation_post_eo_eod.png")

    # Figure 4, Heatmap: best post-DP DPD per model per domain
    dpd_matrix = np.array([
        [safe_get(d, m, 'dp_dpd') if safe_get(d, m, 'dp_dpd') is not None else np.nan for d in DOMAINS] for m in MODELS
    ])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(dpd_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=DOMAINS, yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'DPD (lower = fairer)'})
    ax.set_title('Post-DP ThresholdOptimizer DPD Heatmap, All Models × All Domains\n'
                 '(lower = fairer; green = better)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_dpd_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved -- aggregation_dpd_heatmap.png")


    # Figure 5, Heatmap: post-EO EOD per model per domain
    eod_matrix = np.array([
        [safe_get(d, m, 'eo_eod') if safe_get(d, m, 'eo_eod') is not None else np.nan for d in DOMAINS] for m in MODELS
    ])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(eod_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=DOMAINS, yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'EOD (lower = fairer)'})
    ax.set_title('Post-EO ThresholdOptimizer EOD Heatmap, All Models × All Domains\n'
                 '(lower = fairer; green = better)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_eod_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved -- aggregation_eod_heatmap.png")


    # Figure 6, Heatmap: accuracy cost under DP constraint
    def acc_cost(d, m):
        base = safe_get(d, m, 'baseline_acc')
        dp = safe_get(d, m, 'dp_acc')
        if base is None or dp is None:
            return np.nan
        return base - dp

    acc_cost_matrix = np.array([
        [acc_cost(d, m) for d in DOMAINS] for m in MODELS
    ])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(acc_cost_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=DOMAINS, yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'Accuracy Cost (higher = worse)'})
    ax.set_title('Accuracy Cost Under DP Constraint, All Models × All Domains\n'
                 '(higher = more accuracy lost; red = high cost)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_acc_cost_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved -- aggregation_acc_cost_heatmap.png")

    print(f"\n--- ThresholdOptimizer Aggregation complete ---")
    print(f"  6 figures saved to figures/stage2/")
    print(f"  See Key Findings above for computed baseline performance and DP improvement leaders")
    print(f"  Note: 3 domains (Law School, Lending Club, Agricultural) report AUC not accuracy,")
    print(f"        and lack RandomForest results entirely -- see Decisions 13 and 16")

    return RESULTS


if __name__ == "__main__":
    run_threshold_aggregation()
