"""
FAPE, ThresholdOptimizer Results Aggregation
Fairness Intervention: All Models × All Domains Summary

Aggregates ThresholdOptimizer results for all 3 models (LR, RF, GB)
across the seven domain scripts into a unified comparison table and figures.
MEPS, the eighth evaluation, is held in MEPS_RESULTS and reported in Table 2.
Models a domain did not evaluate are left blank in the figures, not drawn as zero.

Complements cross_domain_comparison.py (GB only) by showing
model-level variation within each domain.

Key question: Does GradientBoosting consistently outperform LR and RF
under ThresholdOptimizer constraints across domains?

Output: 8 figures saved to figures/stage2/
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

# ── All model results from the intervention scripts ─────────────────────────────────
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
        'LR':  {'baseline_acc': 0.740, 'baseline_dpd': 0.301, 'baseline_eod': 0.728,
                'dp_acc': 0.717, 'dp_dpd': 0.348, 'eo_acc': 0.693, 'eo_eod': 0.314},
        'RF':  {'baseline_acc': 0.726, 'baseline_dpd': 0.290, 'baseline_eod': 0.732,
                'dp_acc': 0.706, 'dp_dpd': 0.193, 'eo_acc': 0.712, 'eo_eod': 0.836},
        'GB':  {'baseline_acc': 0.756, 'baseline_dpd': 0.302, 'baseline_eod': 0.767,
                'dp_acc': 0.740, 'dp_dpd': 0.373, 'eo_acc': 0.721, 'eo_eod': 0.417},
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

# MEPS Panel 19 FY2015 (Healthcare), evaluated via the FairGround pipeline rather than a
# standalone intervention script, so it is held separately from RESULTS above. Values verified
# against a pinned-environment run of stage2_fairground_threshold.py, which uses FairGround's
# documented 41 MEPS features (the raw file's visit counts define the label). Single source
# of truth for make_results_table.py and fairness_drift_monitor.py.
MEPS_RESULTS = {
    'LR': {'baseline_acc': 0.855, 'baseline_dpd': 0.069, 'baseline_eod': 0.034,
           'dp_acc': 0.784, 'dp_dpd': 0.013, 'eo_acc': 0.795, 'eo_eod': 0.030},
    'RF': {'baseline_acc': 0.857, 'baseline_dpd': 0.089, 'baseline_eod': 0.053,
           'dp_acc': 0.724, 'dp_dpd': 0.253, 'eo_acc': 0.826, 'eo_eod': 0.071},
    'GB': {'baseline_acc': 0.859, 'baseline_dpd': 0.092, 'baseline_eod': 0.056,
           'dp_acc': 0.784, 'dp_dpd': 0.013, 'eo_acc': 0.799, 'eo_eod': 0.039},
}

# The two sensitivity checks, as printed by notebooks/group_size_check.ipynb and
# notebooks/threshold_holdout_check.ipynb. Each entry is DPD (baseline, after the DP
# constraint): as reported and over race groups with at least 30 test records for COMPAS
# and Folktables; with thresholds chosen on the training split and on a held-out quarter
# of it for Student (Math) and MEPS. make_results_table.py builds Tables 6 and 7 from these,
# and check_consistency.py traces every value back to the notebook output.
GROUP_SIZE_CHECK = {
    ('COMPAS', 'LR'): {'all_groups': (0.545, 0.714), 'at_least_30': (0.385, 0.187)},
    ('COMPAS', 'RF'): {'all_groups': (0.568, 0.714), 'at_least_30': (0.202, 0.163)},
    ('COMPAS', 'GB'): {'all_groups': (0.857, 0.571), 'at_least_30': (0.361, 0.200)},
    ('Folktables', 'LR'): {'all_groups': (0.301, 0.348), 'at_least_30': (0.301, 0.114)},
    ('Folktables', 'RF'): {'all_groups': (0.290, 0.193), 'at_least_30': (0.276, 0.177)},
    ('Folktables', 'GB'): {'all_groups': (0.302, 0.373), 'at_least_30': (0.302, 0.078)},
}
HOLDOUT_CHECK = {
    ('Student (math)', 'LR'): {'training_split': (0.212, 0.010), 'held_out': (0.238, 0.186)},
    ('Student (math)', 'RF'): {'training_split': (0.235, 0.363), 'held_out': (0.235, 0.007)},
    ('Student (math)', 'GB'): {'training_split': (0.237, 0.215), 'held_out': (0.161, 0.159)},
    ('MEPS', 'LR'): {'training_split': (0.069, 0.013), 'held_out': (0.068, 0.031)},
    ('MEPS', 'RF'): {'training_split': (0.089, 0.253), 'held_out': (0.086, 0.037)},
    ('MEPS', 'GB'): {'training_split': (0.092, 0.013), 'held_out': (0.092, 0.054)},
}
RF_TRAINING_ACCURACY = {'Student (math)': 1.000, 'MEPS': 0.996}

DOMAINS = list(RESULTS.keys())
MODELS = ['LR', 'RF', 'GB']
# Short names for axes, so the tick labels do not run into each other.
LABELS = {'FairGround (Education/law_school_lequy)': 'FairGround', 'Student (math)': 'Student'}
def short(name):
    return LABELS.get(name, name)

def safe_get(domain, model, key):
    entry = RESULTS.get(domain, {}).get(model)
    if entry is None and domain == 'MEPS':
        entry = MEPS_RESULTS.get(model)
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

    print("\n--- Key Findings (computed from RESULTS, not hardcoded; see Decision 12) ---")
    # MEPS lives in its own dict because it was added after the original seven.
    # Fold it in for the accuracy comparison; the domain loops above stay on RESULTS.
    acc_doms = [d for d in DOMAINS + ['MEPS'] if has_accuracy(d)]
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

    # Figure 1, Baseline performance by model across domains, split by metric
    # (4 domains report true accuracy; 3 report AUC, so they cannot share one axis,
    # see Decision 13 in methodology_decisions.md)
    acc_domains = [d for d in DOMAINS if has_accuracy(d)]
    auc_domains = [d for d in DOMAINS if not has_accuracy(d)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    x1 = np.arange(len(acc_domains))
    for i, model in enumerate(MODELS):
        vals = [safe_get(d, model, 'baseline_acc') for d in acc_domains]
        vals = [v if v is not None else np.nan for v in vals]
        ax1.bar(x1 + (i-1)*w, vals, w, label=model, color=COLORS[model],
                edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_xticks(x1)
    ax1.set_xticklabels([short(d) for d in acc_domains], rotation=15, ha='right', fontsize=9)
    ax1.set_title('Baseline Accuracy (true classification accuracy)', fontsize=11)
    ax1.set_ylabel('Accuracy')
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.5, 1.0)

    x2 = np.arange(len(auc_domains))
    for i, model in enumerate(MODELS):
        vals = [safe_get(d, model, 'baseline_auc') for d in auc_domains]
        if all(v is None for v in vals):
            continue
        vals = [v if v is not None else np.nan for v in vals]
        ax2.bar(x2 + (i-1)*w, vals, w, label=model, color=COLORS[model],
                edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels([short(d) for d in auc_domains], rotation=15, ha='right', fontsize=9)
    ax2.set_title('Baseline AUC (RandomForest not implemented, see Decision 16)', fontsize=11)
    ax2.set_ylabel('AUC')
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.5, 1.0)

    plt.suptitle('Baseline Performance Across the Seven Domain Scripts, Split by Metric Type', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_baseline_accuracy.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved, aggregation_baseline_accuracy.png (split accuracy/AUC panels)")

    # Figure 2, Post-DP DPD by model across domains
    x = np.arange(len(DOMAINS))
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(MODELS):
        dpds = [safe_get(d, model, 'dp_dpd') for d in DOMAINS]
        dpds = [v if v is not None else np.nan for v in dpds]
        ax.bar(x + (i-1)*w, dpds, w, label=f'{model} post-DP', color=COLORS[model],
               edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='DPD 0.1 convention')
    ax.set_xticks(x)
    ax.set_xticklabels([short(d) for d in DOMAINS], rotation=15, ha='right', fontsize=9)
    ax.set_title('Post-ThresholdOptimizer DPD, LR vs RF vs GB\n'
                 'Seven Domain Scripts, DP Constraint (RF not evaluated in 3 domains)', fontsize=12)
    ax.set_ylabel('Demographic Parity Disparity (DPD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_post_dp_dpd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved, aggregation_post_dp_dpd.png")

    # Figure 3, Post-EO EOD by model across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, model in enumerate(MODELS):
        eods = [safe_get(d, model, 'eo_eod') for d in DOMAINS]
        eods = [v if v is not None else np.nan for v in eods]
        ax.bar(x + (i-1)*w, eods, w, label=f'{model} post-EO', color=COLORS[model],
               edgecolor='black', linewidth=0.5, alpha=0.85)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='0.1 reference')
    ax.set_xticks(x)
    ax.set_xticklabels([short(d) for d in DOMAINS], rotation=15, ha='right', fontsize=9)
    ax.set_title('Post-ThresholdOptimizer EOD, LR vs RF vs GB\n'
                 'Seven Domain Scripts, EO Constraint (RF not evaluated in 3 domains)', fontsize=12)
    ax.set_ylabel('Equalized Odds Disparity (EOD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_post_eo_eod.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved, aggregation_post_eo_eod.png")

    # Figure 4, Heatmap: best post-DP DPD per model per domain
    dpd_matrix = np.array([
        [safe_get(d, m, 'dp_dpd') if safe_get(d, m, 'dp_dpd') is not None else np.nan for d in DOMAINS] for m in MODELS
    ])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(dpd_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=[short(d) for d in DOMAINS], yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'DPD (lower = fairer)'})
    ax.set_title('Post-DP ThresholdOptimizer DPD Heatmap, All Models × All Domains\n'
                 '(lower = fairer; green = better)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_dpd_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved, aggregation_dpd_heatmap.png")


    # Figure 5, Heatmap: post-EO EOD per model per domain
    eod_matrix = np.array([
        [safe_get(d, m, 'eo_eod') if safe_get(d, m, 'eo_eod') is not None else np.nan for d in DOMAINS] for m in MODELS
    ])
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(eod_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=[short(d) for d in DOMAINS], yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'EOD (lower = fairer)'})
    ax.set_title('Post-EO ThresholdOptimizer EOD Heatmap, All Models × All Domains\n'
                 '(lower = fairer; green = better)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_eod_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved, aggregation_eod_heatmap.png")


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
                ax=ax, xticklabels=[short(d) for d in DOMAINS], yticklabels=MODELS,
                linewidths=0.5, cbar_kws={'label': 'Accuracy Cost (higher = worse)'})
    ax.set_title('Accuracy Cost Under DP Constraint, All Models × All Domains\n'
                 '(higher = more accuracy lost; red = high cost)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_acc_cost_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved, aggregation_acc_cost_heatmap.png")

    # Figure 7, the effectiveness pattern: baseline DPD against DPD after the DP constraint
    # for every evaluated model-domain pair. Points below the diagonal improved. Arrows move
    # the pairs that worsened to where the two sensitivity checks place them.
    names = DOMAINS + ['MEPS']
    pairs = [(d, m, safe_get(d, m, 'baseline_dpd'), safe_get(d, m, 'dp_dpd'))
             for d in names for m in MODELS if safe_get(d, m, 'dp_dpd') is not None]
    high = [p for p in pairs if p[2] > 0.2]
    low = [p for p in pairs if p[2] < 0.05]
    improved_high = sum(after < base for _, _, base, after in high)
    worsened_low = sum(after > base for _, _, base, after in low)
    moves = [(key, check['all_groups'], check['at_least_30']) for key, check in GROUP_SIZE_CHECK.items()]
    moves += [(key, check['training_split'], check['held_out']) for key, check in HOLDOUT_CHECK.items()]
    moves = [(key, start, end) for key, start, end in moves if start[1] > start[0] and end[1] < end[0]]
    print(f"\n--- Effectiveness pattern ---")
    print(f"  DPD improved in {improved_high} of {len(high)} pairs with baseline DPD above 0.2, "
          f"worsened in {worsened_low} of {len(low)} pairs below 0.05")
    print(f"  Worsened pairs that improve under a check: "
          + ", ".join(f"{d} {m} ({start[1]:.3f} to {end[1]:.3f})" for (d, m), start, end in moves))

    lo, hi = 0.003, 1.3
    palette = dict(zip(names, sns.color_palette('tab10', len(names))))
    markers = {'LR': 'o', 'RF': 's', 'GB': '^'}
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axvspan(lo, 0.05, color='#d9d9d9', alpha=0.5, lw=0)
    ax.axvspan(0.2, hi, color='#fde0c5', alpha=0.5, lw=0)
    ax.plot([lo, hi], [lo, hi], color='black', linestyle='--', linewidth=1)
    for (d, m), start, end in moves:
        ax.annotate('', xy=end, xytext=start, zorder=2,
                    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.3, shrinkA=7, shrinkB=5))
    for d, m, base, after in pairs:
        ax.scatter(base, after, marker=markers[m], s=95, color=palette[d], edgecolor='black',
                   linewidth=0.6, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_xlabel('Baseline DPD (log scale)', fontsize=11)
    ax.set_ylabel('DPD after the DP constraint (log scale)', fontsize=11)
    ax.text(0.0034, 1.15, 'near-fair\n(below 0.05)', fontsize=9, color='#555555', va='top')
    ax.text(0.21, 1.15, 'high disparity\n(above 0.2)', fontsize=9, color='#8a4b12', va='top')
    domain_handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=8, color=palette[d],
                                 markeredgecolor='black', label=short(d)) for d in names]
    model_handles = [plt.Line2D([], [], marker=markers[m], linestyle='', markersize=8, color='white',
                                markeredgecolor='black', label=m) for m in MODELS]
    line_handles = [plt.Line2D([], [], color='black', linestyle='--', lw=1, label='no change'),
                    plt.Line2D([], [], color='#555555', lw=1.3, label='where a check moves\na worsened pair')]
    first = ax.legend(handles=domain_handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=9,
                      title='Evaluation', title_fontsize=9)
    ax.add_artist(first)
    ax.legend(handles=model_handles + line_handles, loc='upper left', bbox_to_anchor=(1.02, 0.55), fontsize=9,
              title='Model', title_fontsize=9)
    ax.set_title(f'Constraint effect against baseline disparity, all {len(pairs)} model-domain pairs\n'
                 f'Improved in {improved_high} of {len(high)} pairs above 0.2, worsened in {worsened_low} '
                 f'of {len(low)} below 0.05; points below the diagonal improved', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_effectiveness_pattern.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 7 saved, aggregation_effectiveness_pattern.png")

    # Figure 8, the two sensitivity checks side by side. Each pair is drawn twice: once as the
    # intervention scripts run it, once under the check. An arrow points from the disparity before
    # the constraint to the disparity after it, so a pair that reverses direction between the two
    # designs is the whole argument of Sections 6.2 and 6.4 in one view.
    panels = [
        ('Groups below 30 test records dropped', GROUP_SIZE_CHECK, 'all_groups', 'at_least_30',
         'every group', 'groups of 30 or more'),
        ('Thresholds chosen on held-out records', HOLDOUT_CHECK, 'training_split', 'held_out',
         'thresholds on the training split', 'thresholds held out'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (title, table, run_key, check_key, run_label, check_label) in zip(axes, panels):
        entries = list(table.items())
        for index, ((domain, model), values) in enumerate(entries):
            for offset, key, marker in ((-0.16, run_key, 'o'), (0.16, check_key, 's')):
                before, after = values[key]
                colour = '#d9534f' if after > before else '#5cb85c'
                ax.annotate('', xy=(index + offset, after), xytext=(index + offset, before),
                            arrowprops=dict(arrowstyle='->', color=colour, lw=2, shrinkA=3, shrinkB=1))
                ax.scatter(index + offset, before, marker=marker, s=45, color='white',
                           edgecolor='black', linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(entries)))
        ax.set_xticklabels([f'{short(d)}\n{m}' for (d, m), _ in entries], fontsize=9)
        ax.set_xlim(-0.6, len(entries) - 0.4)
        ax.set_ylim(0, max(v for _, values in entries for pair in values.values() for v in pair) * 1.18)
        ax.set_ylabel('Disparity (DPD for the DP constraint)', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        handles = [plt.Line2D([], [], marker='o', linestyle='', markersize=8, color='white',
                              markeredgecolor='black', label=run_label),
                   plt.Line2D([], [], marker='s', linestyle='', markersize=8, color='white',
                              markeredgecolor='black', label=check_label),
                   plt.Line2D([], [], color='#5cb85c', lw=2, label='disparity fell'),
                   plt.Line2D([], [], color='#d9534f', lw=2, label='disparity rose')]
        ax.legend(handles=handles, loc='upper right', fontsize=8, framealpha=0.95)
    reversed_pairs = sum(1 for _, table, run_key, check_key, _, _ in panels
                         for values in table.values()
                         if values[run_key][1] > values[run_key][0] and values[check_key][1] < values[check_key][0])
    checked = sum(len(table) for _, table, _, _, _, _ in panels)
    fig.suptitle(f'What the two sensitivity checks change, {checked} model-domain pairs\n'
                 f'The constraint reverses direction in {reversed_pairs} of them; the marker shows '
                 f'the disparity before the constraint, the arrow head after it', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'aggregation_check_designs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Fig 8 saved, aggregation_check_designs.png")
    print(f"  The constraint reverses direction under a check in {reversed_pairs} of {checked} pairs")

    print(f"\n--- ThresholdOptimizer Aggregation complete ---")
    print(f"  8 figures saved to figures/stage2/")
    print(f"  See Key Findings above for computed baseline performance and DP improvement leaders")
    print(f"  Note: 3 domains (Law School, Lending Club, Agricultural) report AUC not accuracy,")
    print(f"        and lack RandomForest results entirely, see Decisions 13 and 16")

    return RESULTS


if __name__ == "__main__":
    run_threshold_aggregation()
