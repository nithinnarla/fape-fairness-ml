"""
FAPE, Cross-Domain Fairness Comparison
Fairness Intervention: ThresholdOptimizer Results Across the Eight Evaluations

Compares gradient boosting's fairness intervention results across the eight
evaluations:
- COMPAS (criminal justice, race)
- Folktables (income prediction, race)
- Law School (bar passage, race)
- Lending Club (loan default, income band)
- Agricultural (SBA loan default, business type)
- FairGround (law_school_lequy sub-dataset, race)
- MEPS (healthcare utilization, race; runs through the FairGround pipeline)
- Student (Math, sex)

Every DPD, EOD and accuracy value is read from threshold_aggregation.RESULTS
and MEPS_RESULTS, so these figures cannot disagree with Table 2. Accuracy cost is left empty for
Law School, Lending Club and Agricultural, whose intervention scripts report AUC
rather than accuracy. Disparate impact ratio is shown only for the three
domains that compute a fixed-pair ratio before and after the constraint; both
lending domains predict default, so there a ratio above 1.0 is the adverse
direction.

Output: 6 figures saved to figures/stage2/
"""

import os
import sys
import warnings
import logging

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
logging.disable(logging.CRITICAL)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from threshold_aggregation import RESULTS, MEPS_RESULTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(REPO_ROOT, 'figures', 'stage2')
os.makedirs(FIGURES_DIR, exist_ok=True)

DPD_CONVENTION = 0.1
DIR_CONVENTION = 0.8
MODELS = ['LR', 'RF', 'GB']

# Figure labels mapped to threshold_aggregation.RESULTS keys
RESULTS_KEY = {
    'COMPAS': 'COMPAS',
    'Folktables': 'Folktables',
    'Law School': 'Law School',
    'Lending Club': 'Lending Club',
    'Agricultural': 'Agricultural',
    'FairGround': 'FairGround (Education/law_school_lequy)',
    'MEPS': None,   # held in MEPS_RESULTS
    'Student': 'Student (math)',
}

# Fixed-pair disparate impact ratio under gradient boosting, before and after the
# DP constraint, as printed by each intervention script (Section 3.5 of the paper):
# minority over majority (bar passage), partnership over corporation (predicted
# default), lowest income quartile over highest (predicted default).
FIXED_PAIR_DIR = {
    'Law School': (0.643, 0.957),
    'Lending Club': (2.778, 0.973),
    'Agricultural': (0.653, 1.042),
}


def _improvement(before, after):
    return round(100 * (before - after) / before, 1)


def _domain_row(label):
    gb = MEPS_RESULTS['GB'] if label == 'MEPS' else RESULTS[RESULTS_KEY[label]]['GB']
    has_accuracy = gb.get('baseline_acc') is not None
    return {
        'baseline_dpd': gb['baseline_dpd'], 'post_dp_dpd': gb['dp_dpd'],
        'dp_improve': _improvement(gb['baseline_dpd'], gb['dp_dpd']),
        'baseline_eod': gb['baseline_eod'], 'post_eo_eod': gb['eo_eod'],
        'eo_improve': _improvement(gb['baseline_eod'], gb['eo_eod']),
        'dp_acc_cost': round(gb['baseline_acc'] - gb['dp_acc'], 3) if has_accuracy else None,
        'eo_acc_cost': round(gb['baseline_acc'] - gb['eo_acc'], 3) if has_accuracy else None,
        'dir': FIXED_PAIR_DIR.get(label),
    }


DOMAINS = {label: _domain_row(label) for label in RESULTS_KEY}
domain_names = list(DOMAINS.keys())


def pair_counts():
    """Model-domain pairs by baseline DPD band, over all eight evaluations (Section 5.5)."""
    pairs = []
    for source in list(RESULTS.values()) + [MEPS_RESULTS]:
        for model in MODELS:
            entry = source.get(model)
            if entry is not None:
                pairs.append((entry['baseline_dpd'], entry['dp_dpd']))
    high = [p for p in pairs if p[0] > 0.2]
    low = [p for p in pairs if p[0] < 0.05]
    return (sum(1 for b, a in high if a < b), len(high),
            sum(1 for b, a in low if a > b), len(low))


def run_cross_domain_comparison():
    print("FAPE, Cross-Domain Fairness Comparison (gradient boosting)")
    print("=" * 50)

    print("\n--- Domain Summary ---")
    for domain, m in DOMAINS.items():
        cost = 'n/a (AUC)' if m['dp_acc_cost'] is None else f"{m['dp_acc_cost']:.3f}"
        print(f"  {domain:<13} DP improve: {m['dp_improve']:+.1f}% | "
              f"EO improve: {m['eo_improve']:+.1f}% | ACC cost (DP): {cost}")

    improved_high, n_high, worsened_low, n_low = pair_counts()
    best_dp = max(domain_names, key=lambda d: DOMAINS[d]['dp_improve'])
    worst_dp = min(domain_names, key=lambda d: DOMAINS[d]['dp_improve'])
    costed = [d for d in domain_names if DOMAINS[d]['dp_acc_cost'] is not None]
    costliest = max(costed, key=lambda d: DOMAINS[d]['dp_acc_cost'])
    print("\n--- Key Cross-Domain Findings (computed) ---")
    print(f"  Largest GB DPD improvement: {best_dp} ({DOMAINS[best_dp]['dp_improve']:+.1f}%)")
    print(f"  Largest GB DPD worsening: {worst_dp} ({DOMAINS[worst_dp]['dp_improve']:+.1f}%)")
    print(f"  Highest GB accuracy cost under DP: {costliest} ({DOMAINS[costliest]['dp_acc_cost']:.3f})")
    print(f"  Pairs with baseline DPD > 0.2 that improved: {improved_high} of {n_high}")
    print(f"  Pairs with baseline DPD < 0.05 that worsened: {worsened_low} of {n_low}")
    print("  This is a pattern with exceptions, not a rule (Section 6.1)")

    x = np.arange(len(domain_names))
    w = 0.35
    subtitle = 'Eight Evaluations, Gradient Boosting'

    # Figure 1, DPD before/after across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, [DOMAINS[d]['baseline_dpd'] for d in domain_names], w, label='Baseline DPD',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x + w/2, [DOMAINS[d]['post_dp_dpd'] for d in domain_names], w,
           label='Post-ThresholdOptimizer DPD', color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--', linewidth=1, label='DPD 0.1 convention')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, rotation=15, ha='right', fontsize=9)
    ax.set_title(f'Demographic Parity Difference, Before vs After ThresholdOptimizer\n{subtitle}', fontsize=12)
    ax.set_ylabel('Demographic Parity Difference (DPD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_dpd_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved, cross_domain_dpd_comparison.png")

    # Figure 2, EOD before/after across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, [DOMAINS[d]['baseline_eod'] for d in domain_names], w, label='Baseline EOD',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x + w/2, [DOMAINS[d]['post_eo_eod'] for d in domain_names], w,
           label='Post-ThresholdOptimizer EOD', color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--', linewidth=1, label='0.1 reference')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, rotation=15, ha='right', fontsize=9)
    ax.set_title(f'Equalized Odds Difference, Before vs After ThresholdOptimizer\n{subtitle}', fontsize=12)
    ax.set_ylabel('Equalized Odds Difference (EOD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_eod_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved, cross_domain_eod_comparison.png")

    # Figure 3, Accuracy cost vs fairness improvement, domains with a true accuracy metric only
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for ax, improve_key, cost_key, title, ylabel in (
            (ax1, 'dp_improve', 'dp_acc_cost', 'DP Constraint', 'DPD Improvement % (positive = better)'),
            (ax2, 'eo_improve', 'eo_acc_cost', 'EO Constraint', 'EOD Improvement % (positive = better)')):
        for domain in costed:
            improve, cost = DOMAINS[domain][improve_key], DOMAINS[domain][cost_key]
            ax.scatter(cost, improve, s=120, color='#2ecc71' if improve > 0 else '#e74c3c',
                       edgecolor='black', linewidth=0.5, zorder=3)
            ax.annotate(domain, (cost, improve), textcoords='offset points', xytext=(5, 5), fontsize=8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.set_title(f'{title} - Accuracy Cost vs Fairness Improvement', fontsize=11)
        ax.set_xlabel('Accuracy Cost (positive = worse)')
        ax.set_ylabel(ylabel)
    uncosted = [d for d in domain_names if d not in costed]
    plt.suptitle('Accuracy-Fairness Tradeoff, Gradient Boosting\n'
                 f'Accuracy cost not computable for {", ".join(uncosted)} (AUC only); '
                 'costs include the change to a balanced-accuracy objective', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_acc_fairness_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved, cross_domain_acc_fairness_scatter.png")

    # Figure 4, Fixed-pair DIR before/after (paper Figure 2)
    dir_names = [d for d in domain_names if DOMAINS[d]['dir'] is not None]
    baseline_dirs = [DOMAINS[d]['dir'][0] for d in dir_names]
    post_dirs = [DOMAINS[d]['dir'][1] for d in dir_names]
    x_dir = np.arange(len(dir_names))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x_dir - w/2, baseline_dirs, w, label='Baseline DIR',
           color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x_dir + w/2, post_dirs, w, label='Post-ThresholdOptimizer DIR',
           color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=DIR_CONVENTION, color='black', linestyle='--', linewidth=1.5, label='0.8 four-fifths convention')
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Parity (DIR=1.0)')
    ax.set_xticks(x_dir)
    outcome = {'Law School': 'bar passage', 'Lending Club': 'predicted default', 'Agricultural': 'predicted default'}
    ax.set_xticklabels([f'{d}\n({outcome[d]})' for d in dir_names], fontsize=10)
    ax.set_title('Fixed-Pair Disparate Impact Ratio, Before vs After DP Constraint (Gradient Boosting)\n'
                 'For predicted default, a ratio above 1.0 is the adverse direction', fontsize=11)
    ax.set_ylabel('Disparate Impact Ratio (DIR)')
    ax.legend(fontsize=9)
    for i, (b, p) in enumerate(zip(baseline_dirs, post_dirs)):
        # white boxes keep the labels readable where they sit on the parity line
        label_box = dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.9)
        ax.text(x_dir[i] - w/2, b + 0.02, f'{b:.3f}', ha='center', fontsize=9, bbox=label_box)
        ax.text(x_dir[i] + w/2, p + 0.02, f'{p:.3f}', ha='center', fontsize=9, bbox=label_box)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_dir_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved, cross_domain_dir_comparison.png")

    # Figure 5, Metrics summary heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    rows = [('Baseline DPD', 'baseline_dpd'), ('Post-DP DPD', 'post_dp_dpd'),
            ('Baseline EOD', 'baseline_eod'), ('Post-EO EOD', 'post_eo_eod'), ('DP Acc Cost', 'dp_acc_cost')]
    matrix = np.array([[np.nan if DOMAINS[d][key] is None else DOMAINS[d][key] for d in domain_names]
                       for _, key in rows])
    sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', mask=np.isnan(matrix),
                ax=ax, xticklabels=domain_names, yticklabels=[r[0] for r in rows],
                linewidths=0.5, cbar_kws={'label': 'Value'})
    for j, d in enumerate(domain_names):
        if DOMAINS[d]['dp_acc_cost'] is None:
            ax.text(j + 0.5, len(rows) - 0.5, 'AUC', ha='center', va='center', fontsize=9, color='gray')
    ax.set_title(f'Cross-Domain Fairness Metrics Summary\n{subtitle}', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_metrics_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved, cross_domain_metrics_heatmap.png")

    # Figure 6, DP and EO improvement % ranking (paper Figure 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    for ax, key, title, xlabel in ((ax1, 'dp_improve', 'DP Constraint', 'DPD Improvement % (positive = better)'),
                                   (ax2, 'eo_improve', 'EO Constraint', 'EOD Improvement % (positive = better)')):
        ordered = sorted(domain_names, key=lambda d: DOMAINS[d][key])
        vals = [DOMAINS[d][key] for d in ordered]
        bars = ax.barh(range(len(ordered)), vals, color=['#2ecc71' if v > 0 else '#e74c3c' for v in vals],
                       edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(ordered)))
        ax.set_yticklabels(ordered, fontsize=9)
        ax.axvline(x=0, color='black', linewidth=1)
        for bar, val in zip(bars, vals):
            ax.text(max(val, 0) + 3, bar.get_y() + bar.get_height()/2, f'{val:+.1f}%', va='center', ha='left', fontsize=8)
        ax.set_xlim(min(min(vals), 0) * 1.05 - 5, max(vals) + 30)
        ax.set_title(f'{title} - Gradient Boosting Improvement %\n(green = improvement, red = degradation)',
                     fontsize=11)
        ax.set_xlabel(xlabel)
    plt.suptitle('ThresholdOptimizer Effectiveness Ranking, Eight Evaluations\n'
                 f'Across all model-domain pairs: improved in {improved_high} of {n_high} with baseline DPD > 0.2; '
                 f'worsened in {worsened_low} of {n_low} with baseline DPD < 0.05', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_improvement_ranking.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved, cross_domain_improvement_ranking.png")

    print("\n--- Cross-Domain Comparison complete ---")
    print("  6 figures saved to figures/stage2/")

    return DOMAINS


if __name__ == "__main__":
    run_cross_domain_comparison()
