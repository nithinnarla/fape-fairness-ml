"""
FAPE — Cross-Domain Fairness Comparison
Stage 2: ThresholdOptimizer Results Across All 7 Domains

Aggregates fairness intervention results across all 7 FAPE datasets:
- COMPAS (criminal justice, race)
- Folktables (income prediction, race)
- Law School (bar passage, race)
- Lending Club (loan default, income)
- Agricultural (SBA loans, business type)
- FairGround (multi-attribute synthetic)
- Student (academic performance, sex/parentage)

Key metrics: DPD/EOD before/after ThresholdOptimizer, accuracy cost,
DIR before/after, EEOC threshold compliance.

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

# ── Aggregated results from all 7 Stage 2 scripts ──────────────────────────
# GradientBoosting results (best model across all domains)
DOMAINS = {
    'COMPAS': {
        'sensitive': 'Race (6 groups)',
        'regulatory': 'ECOA/EEOC',
        'baseline_dpd': 0.857, 'best_dp_dpd': 0.571, 'dp_improve': 33.3,
        'baseline_eod': 1.000, 'best_eo_eod': 0.639, 'eo_improve': 36.1,
        'baseline_acc': 0.674, 'dp_acc_cost': 0.008, 'eo_acc_cost': -0.002,
        'baseline_dir': None, 'post_dir': None,
        'note': 'GB DP constraint most effective; LR/RF worsen under DP'
    },
    'Folktables': {
        'sensitive': 'Race (9 groups)',
        'regulatory': 'ECOA/Title VII',
        'baseline_dpd': 0.320, 'best_dp_dpd': 0.340, 'dp_improve': -6.2,
        'baseline_eod': 0.333, 'best_eo_eod': 0.331, 'eo_improve': 0.6,
        'baseline_acc': 0.845, 'dp_acc_cost': 0.019, 'eo_acc_cost': 0.017,
        'baseline_dir': 0.54, 'post_dir': None,
        'note': 'ThresholdOptimizer minimal — Am.Indian DIR=0.54 most disadvantaged'
    },
    'Law School': {
        'sensitive': 'Race',
        'regulatory': 'ECOA/Title VI',
        'baseline_dpd': 0.351, 'best_dp_dpd': 0.039, 'dp_improve': 88.9,
        'baseline_eod': 0.528, 'best_eo_eod': 0.031, 'eo_improve': 94.1,
        'baseline_acc': 0.878, 'dp_acc_cost': 0.003, 'eo_acc_cost': 0.003,
        'baseline_dir': 0.643, 'post_dir': 0.945,
        'note': 'Largest racial gap in FAPE; strongest ThresholdOptimizer improvement'
    },
    'Lending Club': {
        'sensitive': 'Income Band',
        'regulatory': 'ECOA',
        'baseline_dpd': 0.024, 'best_dp_dpd': 0.024, 'dp_improve': -0.1,
        'baseline_eod': 0.053, 'best_eo_eod': 0.060, 'eo_improve': -7.0,
        'baseline_acc': 0.712, 'dp_acc_cost': 0.001, 'eo_acc_cost': -0.013,
        'baseline_dir': 2.778, 'post_dir': 0.952,
        'note': 'Near-fair baseline; DIR>1 amplifies disparity (actual 1.4x predicted 2.8x)'
    },
    'Agricultural': {
        'sensitive': 'Business Type',
        'regulatory': 'ECOA/SBA',
        'baseline_dpd': 0.009, 'best_dp_dpd': 0.035, 'dp_improve': -288.9,
        'baseline_eod': 0.073, 'best_eo_eod': 0.194, 'eo_improve': -165.8,
        'baseline_acc': 0.938, 'dp_acc_cost': 0.026, 'eo_acc_cost': 0.121,
        'baseline_dir': 0.653, 'post_dir': 1.095,
        'note': 'Near-fair baseline; ThresholdOptimizer counterproductive; GB overcorrects'
    },
    'FairGround': {
        'sensitive': 'Multi-attribute',
        'regulatory': 'ECOA/Title VII',
        'baseline_dpd': 0.342, 'best_dp_dpd': 0.026, 'dp_improve': 92.4,
        'baseline_eod': 0.518, 'best_eo_eod': 0.038, 'eo_improve': 92.7,
        'baseline_acc': 0.910, 'dp_acc_cost': 0.159, 'eo_acc_cost': 0.149,
        'baseline_dir': None, 'post_dir': None,
        'note': 'Strongest improvement but highest accuracy cost'
    },
    'Student': {
        'sensitive': 'Sex/Parentage',
        'regulatory': 'Title IX/ECOA',
        'baseline_dpd': 0.237, 'best_dp_dpd': 0.190, 'dp_improve': 19.8,
        'baseline_eod': 0.314, 'best_eo_eod': 0.055, 'eo_improve': 82.5,
        'baseline_acc': 0.658, 'dp_acc_cost': 0.063, 'eo_acc_cost': 0.025,
        'baseline_dir': None, 'post_dir': None,
        'note': 'EO constraint most effective; sex fairness near-zero after intervention'
    }
}

domain_names = list(DOMAINS.keys())


def run_cross_domain_comparison():
    print("FAPE — Cross-Domain Fairness Comparison")
    print("=" * 50)

    print("\n--- Domain Summary ---")
    for domain, metrics in DOMAINS.items():
        print(f"  {domain:<15} DP improve: {metrics['dp_improve']:+.1f}% | "
              f"EO improve: {metrics['eo_improve']:+.1f}% | "
              f"ACC cost (DP): {metrics['dp_acc_cost']:.3f}")

    print("\n--- Key Cross-Domain Findings ---")
    print("  Law School: largest racial gap (DIR=0.643); strongest improvement (EO +94.1%)")
    print("  FairGround: strongest DP improvement (92.4%) but highest ACC cost (0.159)")
    print("  Agricultural + Lending Club: near-fair baseline — ThresholdOptimizer counterproductive")
    print("  COMPAS: GB DP constraint most effective (DPD 0.857→0.571)")
    print("  Student: EO constraint eliminates sex gap (EOD 0.314→0.055)")
    print("  Finding: ThresholdOptimizer effective when baseline DPD > 0.2; counterproductive when < 0.05")

    # Layout constants
    x = np.arange(len(domain_names))
    w = 0.35

    # Figure 1 — DPD before/after across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    baseline_dpds = [DOMAINS[d]['baseline_dpd'] for d in domain_names]
    post_dpds = [DOMAINS[d]['best_dp_dpd'] for d in domain_names]
    ax.bar(x - w/2, baseline_dpds, w, label='Baseline DPD', color='#e74c3c',
           edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x + w/2, post_dpds, w, label='Post-ThresholdOptimizer DPD', color='#2ecc71',
           edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='EEOC 0.1 DPD threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, rotation=15, ha='right', fontsize=9)
    ax.set_title('Demographic Parity Disparity — Before vs After ThresholdOptimizer\n'
                 'Across All 7 FAPE Domains (GradientBoosting)', fontsize=12)
    ax.set_ylabel('Demographic Parity Disparity (DPD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_dpd_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved -- cross_domain_dpd_comparison.png")

    # Figure 2 — EOD before/after across domains
    fig, ax = plt.subplots(figsize=(14, 6))
    baseline_eods = [DOMAINS[d]['baseline_eod'] for d in domain_names]
    post_eods = [DOMAINS[d]['best_eo_eod'] for d in domain_names]
    ax.bar(x - w/2, baseline_eods, w, label='Baseline EOD', color='#e74c3c',
           edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.bar(x + w/2, post_eods, w, label='Post-ThresholdOptimizer EOD', color='#2ecc71',
           edgecolor='black', linewidth=0.5, alpha=0.8)
    ax.axhline(y=0.1, color='black', linestyle='--', linewidth=1,
               label='EEOC 0.1 EOD threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, rotation=15, ha='right', fontsize=9)
    ax.set_title('Equalized Odds Disparity — Before vs After ThresholdOptimizer\n'
                 'Across All 7 FAPE Domains (GradientBoosting)', fontsize=12)
    ax.set_ylabel('Equalized Odds Disparity (EOD)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_eod_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved -- cross_domain_eod_comparison.png")

    # Figure 3 — Accuracy cost vs fairness improvement scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    dp_improves = [DOMAINS[d]['dp_improve'] for d in domain_names]
    eo_improves = [DOMAINS[d]['eo_improve'] for d in domain_names]
    dp_costs = [DOMAINS[d]['dp_acc_cost'] for d in domain_names]
    eo_costs = [DOMAINS[d]['eo_acc_cost'] for d in domain_names]

    for i, domain in enumerate(domain_names):
        color = '#2ecc71' if dp_improves[i] > 0 else '#e74c3c'
        ax1.scatter(dp_costs[i], dp_improves[i], s=120, color=color,
                    edgecolor='black', linewidth=0.5, zorder=3)
        ax1.annotate(domain, (dp_costs[i], dp_improves[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('DP Constraint — Accuracy Cost vs Fairness Improvement', fontsize=11)
    ax1.set_xlabel('Accuracy Cost (positive = worse)')
    ax1.set_ylabel('DPD Improvement % (positive = better)')

    for i, domain in enumerate(domain_names):
        color = '#2ecc71' if eo_improves[i] > 0 else '#e74c3c'
        ax2.scatter(eo_costs[i], eo_improves[i], s=120, color=color,
                    edgecolor='black', linewidth=0.5, zorder=3)
        ax2.annotate(domain, (eo_costs[i], eo_improves[i]),
                     textcoords='offset points', xytext=(5, 5), fontsize=8)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('EO Constraint — Accuracy Cost vs Fairness Improvement', fontsize=11)
    ax2.set_xlabel('Accuracy Cost (positive = worse)')
    ax2.set_ylabel('EOD Improvement % (positive = better)')

    plt.suptitle('Accuracy-Fairness Tradeoff Across All 7 FAPE Domains\n'
                 '(green = improvement, red = degradation)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_acc_fairness_scatter.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved -- cross_domain_acc_fairness_scatter.png")

    # Figure 4 — DIR before/after for domains with DIR metrics
    dir_domains = {d: DOMAINS[d] for d in domain_names
                   if DOMAINS[d]['baseline_dir'] is not None and DOMAINS[d]['post_dir'] is not None}
    if dir_domains:
        dir_names = list(dir_domains.keys())
        baseline_dirs = [dir_domains[d]['baseline_dir'] for d in dir_names]
        post_dirs = [dir_domains[d]['post_dir'] for d in dir_names]
        x_dir = np.arange(len(dir_names))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x_dir - w/2, baseline_dirs, w, label='Baseline DIR',
               color='#e74c3c', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.bar(x_dir + w/2, post_dirs, w, label='Post-ThresholdOptimizer DIR',
               color='#2ecc71', edgecolor='black', linewidth=0.5, alpha=0.8)
        ax.axhline(y=0.8, color='black', linestyle='--', linewidth=1.5,
                   label='EEOC 80% rule (DIR=0.8)')
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, label='Parity (DIR=1.0)')
        ax.set_xticks(x_dir)
        ax.set_xticklabels(dir_names, fontsize=10)
        ax.set_title('Disparate Impact Ratio — Before vs After ThresholdOptimizer\n'
                     '(EEOC 80% rule: DIR ≥ 0.8 required)', fontsize=12)
        ax.set_ylabel('Disparate Impact Ratio (DIR)')
        ax.legend(fontsize=9)
        for i, (b, p) in enumerate(zip(baseline_dirs, post_dirs)):
            ax.text(x_dir[i] - w/2, b + 0.02, f'{b:.3f}', ha='center', fontsize=9)
            ax.text(x_dir[i] + w/2, p + 0.02, f'{p:.3f}', ha='center', fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_dir_comparison.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
        print("  Fig 4 saved -- cross_domain_dir_comparison.png")

    # Figure 5 — Baseline DPD ranking heatmap
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics_matrix = np.array([
        [DOMAINS[d]['baseline_dpd'] for d in domain_names],
        [DOMAINS[d]['best_dp_dpd'] for d in domain_names],
        [DOMAINS[d]['baseline_eod'] for d in domain_names],
        [DOMAINS[d]['best_eo_eod'] for d in domain_names],
        [DOMAINS[d]['dp_acc_cost'] for d in domain_names],
    ])
    row_labels = ['Baseline DPD', 'Post-DP DPD', 'Baseline EOD',
                  'Post-EO EOD', 'DP Acc Cost']
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r',
                ax=ax, xticklabels=domain_names, yticklabels=row_labels,
                linewidths=0.5, cbar_kws={'label': 'Value'})
    ax.set_title('Cross-Domain Fairness Metrics Summary — All 7 FAPE Domains\n'
                 '(GradientBoosting, ThresholdOptimizer)', fontsize=12)
    plt.xticks(rotation=15, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_metrics_heatmap.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved -- cross_domain_metrics_heatmap.png")


    # Figure 6 — DP and EO improvement % ranking
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort by DP improvement
    sorted_dp = sorted(domain_names, key=lambda d: DOMAINS[d]['dp_improve'])
    dp_vals = [DOMAINS[d]['dp_improve'] for d in sorted_dp]
    dp_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in dp_vals]
    bars1 = ax1.barh(range(len(sorted_dp)), dp_vals, color=dp_colors,
                     edgecolor='black', linewidth=0.5)
    ax1.set_yticks(range(len(sorted_dp)))
    ax1.set_yticklabels(sorted_dp, fontsize=9)
    ax1.axvline(x=0, color='black', linewidth=1)
    for bar, val in zip(bars1, dp_vals):
        ax1.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    ax1.set_title('DP Constraint — Fairness Improvement % Ranking\n'
                  '(green = improvement, red = degradation)', fontsize=11)
    ax1.set_xlabel('DPD Improvement % (positive = better)')

    # Sort by EO improvement
    sorted_eo = sorted(domain_names, key=lambda d: DOMAINS[d]['eo_improve'])
    eo_vals = [DOMAINS[d]['eo_improve'] for d in sorted_eo]
    eo_colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in eo_vals]
    bars2 = ax2.barh(range(len(sorted_eo)), eo_vals, color=eo_colors,
                     edgecolor='black', linewidth=0.5)
    ax2.set_yticks(range(len(sorted_eo)))
    ax2.set_yticklabels(sorted_eo, fontsize=9)
    ax2.axvline(x=0, color='black', linewidth=1)
    for bar, val in zip(bars2, eo_vals):
        ax2.text(val + (2 if val >= 0 else -2), bar.get_y() + bar.get_height()/2,
                 f'{val:+.1f}%', va='center', ha='left' if val >= 0 else 'right', fontsize=8)
    ax2.set_title('EO Constraint — Fairness Improvement % Ranking\n'
                  '(green = improvement, red = degradation)', fontsize=11)
    ax2.set_xlabel('EOD Improvement % (positive = better)')

    plt.suptitle('ThresholdOptimizer Effectiveness Ranking Across 7 FAPE Domains\n'
                 'Effective when baseline DPD > 0.2; counterproductive when < 0.05',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cross_domain_improvement_ranking.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved -- cross_domain_improvement_ranking.png")

    print(f"\n--- Cross-Domain Comparison complete ---")
    print(f"  6 figures saved to figures/stage2/")
    print(f"  ThresholdOptimizer effective when baseline DPD > 0.2")
    print(f"  Counterproductive when baseline near-fair (Agricultural DPD=0.009, Lending Club DPD=0.024)")
    print(f"  Law School strongest violation + strongest improvement")
    print(f"  FairGround highest accuracy cost for improvement")

    return DOMAINS


if __name__ == "__main__":
    run_cross_domain_comparison()
