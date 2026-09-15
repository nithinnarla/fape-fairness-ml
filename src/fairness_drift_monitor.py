"""
FAPE, Fairness Drift Detection and Deployment Monitoring
Phase 4, Stage 4: CUSUM-Based Fairness Drift Detection

Simulates post-deployment fairness monitoring:
1. Three model versions per domain and model: v1 baseline, v2 the
   constrained model that gets deployed, v3 a synthetic distribution shift
2. A one-sided CUSUM monitor that starts when v2 is deployed
3. The 0.1 demographic parity difference research convention as the level
   the monitor watches (a convention, not the EEOC four-fifths rule, which
   is a ratio at 0.8)

How the shift is built, and what that means for the results:
- v3 moves each model linearly from its post-constraint DPD toward a point
  60% of the way back to its baseline DPD. The size of each regression is
  therefore fixed by construction at a share of that model's correction.
- Which models the monitor flags during the shift follows directly from
  their baseline and post-constraint values. The drift stage demonstrates
  that the monitor catches such a regression; it does not show that large
  corrections are empirically more fragile. No production drift was
  observed (Decision 7).

Monitoring starts at deployment. The baseline model is never deployed, so
v1 observations are shown in the figures but never monitored. An alert in
the v2 window means the constrained model was already above 0.1 when it
went live, which is a different finding from drift in v3.

All values come from threshold_aggregation.RESULTS and MEPS_RESULTS, so
this stage cannot disagree with Table 2. Models a domain did not evaluate
under the intervention (random forest for Law School, Lending Club and
Agricultural) are excluded rather than filled with placeholders.

Domains: COMPAS, Folktables, Law School, Lending Club, Agricultural,
FairGround, MEPS, Student
Output: 9 figures saved to figures/stage2/
"""

import os
import sys
import zlib
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

DPD_CONVENTION = 0.1   # level the monitor watches, see Section 3.5 of the paper
CUSUM_SLACK = 0.01
N_POINTS = 30
DEPLOY_START = 10      # v2, the constrained model, goes live here
SHIFT_START = 20       # v3, the synthetic shift, begins here
SHIFT_SHARE = 0.6      # v3 moves this share of the way back toward baseline
SEED = 42

DOMAINS = ['COMPAS', 'Folktables', 'Law School', 'Lending Club', 'Agricultural', 'FairGround', 'MEPS', 'Student']
MODELS = ['LR', 'RF', 'GB']

# Short names used in figures, mapped to the keys used in threshold_aggregation.RESULTS
RESULTS_KEY = {
    'FairGround': 'FairGround (Education/law_school_lequy)',
    'Student': 'Student (math)',
}

COLORS = {'LR': '#2ecc71', 'RF': '#3498db', 'GB': '#e74c3c'}
DOMAIN_COLORS = {
    'COMPAS': '#e74c3c', 'Folktables': '#3498db', 'Law School': '#2ecc71', 'MEPS': '#8e44ad',
    'Lending Club': '#9b59b6', 'Agricultural': '#f39c12', 'FairGround': '#1abc9c',
    'Student': '#e67e22'
}


def _entry(domain, model):
    source = MEPS_RESULTS if domain == 'MEPS' else RESULTS[RESULTS_KEY.get(domain, domain)]
    return source.get(model)


def _pair(domain, model, before, after):
    """(v1, v2) for one metric, or None when the model was not evaluated."""
    e = _entry(domain, model)
    if e is None:
        return None
    v1 = e.get(before)
    if v1 is None and before == 'baseline_acc':
        v1 = e.get('baseline_auc')   # AUC domains, see Decision 13
    v2 = e.get(after)
    if v1 is None or v2 is None:
        return None
    return {'v1': v1, 'v2': v2}


def _table(before, after):
    return {d: {m: _pair(d, m, before, after) for m in MODELS} for d in DOMAINS}


# v1 = baseline, v2 = post-constraint, straight from the results dict
STAGE2_RESULTS = _table('baseline_dpd', 'dp_dpd')
EOD_RESULTS = _table('baseline_eod', 'eo_eod')
ACC_RESULTS = _table('baseline_acc', 'dp_acc')   # AUC for Law School, Lending Club, Agricultural


def series_rng(*parts):
    """Deterministic generator per series, so excluding a model never reshuffles another's noise."""
    return np.random.default_rng(SEED + zlib.crc32('|'.join(parts).encode()))


def simulate_drift_timeseries(v1_value, v2_value, rng, n_points=N_POINTS):
    """
    Simulate one metric across three model versions.

    v1: baseline, stable around v1_value
    v2: deployed constrained model, stable around v2_value
    v3: synthetic shift, moving linearly from v2_value toward a point
        SHIFT_SHARE of the way back to v1_value
    """
    n_v1 = n_points // 3
    n_v2 = n_points // 3
    n_v3 = n_points - n_v1 - n_v2

    v1 = np.clip(rng.normal(v1_value, max(v1_value * 0.05, 0.003), n_v1), 0, 1)
    v2 = np.clip(rng.normal(v2_value, max(v2_value * 0.05, 0.003), n_v2), 0, 1)

    shift_target = v1_value * SHIFT_SHARE + v2_value * (1 - SHIFT_SHARE)
    v3 = np.linspace(v2_value, shift_target, n_v3)
    v3 += rng.normal(0, max(v2_value * 0.03, 0.002), n_v3)
    v3 = np.clip(v3, 0, 1)

    return np.concatenate([v1, v2, v3])


def cusum_detect(timeseries, level=DPD_CONVENTION, slack=CUSUM_SLACK, alert_at=DPD_CONVENTION, start=DEPLOY_START):
    """
    One-sided CUSUM on the deployed model.

    From `start` onward the score accumulates how far the metric sits above
    level + slack, floored at zero; an alert fires whenever the score passes
    alert_at. Observations before deployment are never monitored.

    Returns: (cusum_scores, alert_indices)
    """
    cusum = np.zeros(len(timeseries))
    alerts = []
    for i in range(start, len(timeseries)):
        previous = cusum[i - 1] if i > start else 0.0
        cusum[i] = max(0.0, previous + timeseries[i] - level - slack)
        if cusum[i] > alert_at:
            alerts.append(i)
    return cusum, alerts


def alert_window(first_alert):
    if first_alert is None:
        return 'none'
    return 'deployment' if first_alert < SHIFT_START else 'shift'


def _mark_not_evaluated(ax, row, col):
    ax.text(col, row, 'n/e', ha='center', va='center', fontsize=10, color='gray')


def run_fairness_drift_monitor():
    print("FAPE Phase 4, Stage 4: Fairness Drift Detection")
    print("=" * 55)

    print("\n--- Simulating Distribution Shift ---")
    all_timeseries = {d: {} for d in DOMAINS}
    all_cusum = {d: {} for d in DOMAINS}
    all_alerts = {d: {} for d in DOMAINS}

    for domain in DOMAINS:
        for model in MODELS:
            pair = STAGE2_RESULTS[domain][model]
            if pair is None:
                all_timeseries[domain][model] = all_cusum[domain][model] = all_alerts[domain][model] = None
                continue
            ts = simulate_drift_timeseries(pair['v1'], pair['v2'], series_rng(domain, model, 'dpd'))
            cusum, alerts = cusum_detect(ts)
            all_timeseries[domain][model] = ts
            all_cusum[domain][model] = cusum
            all_alerts[domain][model] = alerts

    for domain in DOMAINS:
        parts = []
        for model in MODELS:
            alerts = all_alerts[domain][model]
            if alerts is None:
                parts.append(f"{model}=n/e")
            else:
                first = alerts[0] if alerts else None
                parts.append(f"{model}={first if first is not None else '-'}({alert_window(first)})")
        print(f"  {domain:<13} first alert: " + "  ".join(parts))

    # Figure 1, DPD timeseries across model versions (GB)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        ts = all_timeseries[domain]['GB']
        ax.plot(range(len(ts)), ts, color=DOMAIN_COLORS[domain], linewidth=2)
        ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--',
                   linewidth=1, label='DPD 0.1 convention')
        ax.axvline(x=DEPLOY_START, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(x=SHIFT_START, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ymax = max(ax.get_ylim()[1], ts.max() * 1.1)
        ax.text(5, ymax * 0.95, 'v1', ha='center', fontsize=8, color='gray')
        ax.text(15, ymax * 0.95, 'v2', ha='center', fontsize=8, color='gray')
        ax.text(25, ymax * 0.95, 'v3', ha='center', fontsize=8, color='gray')
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('DPD')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Simulated DPD Across 3 Model Versions, GB Model\n'
                 'v1=baseline (not deployed), v2=post-DP constraint (deployed), v3=synthetic shift',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_dpd_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 1 saved, drift_dpd_timeseries.png")

    # Figure 2, CUSUM scores (GB)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        cusum = all_cusum[domain]['GB']
        alerts = all_alerts[domain]['GB']
        ax.plot(range(len(cusum)), cusum, color=DOMAIN_COLORS[domain], linewidth=2)
        ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--',
                   linewidth=1, label='Alert threshold')
        ax.axvline(x=DEPLOY_START, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        if alerts:
            ax.axvline(x=alerts[0], color='red', linestyle='-',
                       linewidth=1.5, alpha=0.7, label=f'First alert t={alerts[0]}')
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('CUSUM Score')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('CUSUM Scores After Deployment, GB Model\n'
                 'Monitoring starts at t=10; score above threshold = alert',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_cusum_scores.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 2 saved, drift_cusum_scores.png")

    # Figure 3, Alert count heatmap
    alert_matrix = np.array([
        [np.nan if all_alerts[d][m] is None else len(all_alerts[d][m]) for m in MODELS]
        for d in DOMAINS
    ])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.ma.masked_invalid(alert_matrix), cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(MODELS)))
    ax.set_yticks(range(len(DOMAINS)))
    ax.set_xticklabels(MODELS)
    ax.set_yticklabels(DOMAINS)
    for i in range(len(DOMAINS)):
        for j in range(len(MODELS)):
            if np.isnan(alert_matrix[i, j]):
                _mark_not_evaluated(ax, i, j)
            else:
                count = int(alert_matrix[i, j])
                ax.text(j, i, str(count), ha='center', va='center',
                        fontsize=11, color='black' if count < 10 else 'white')
    plt.colorbar(im, ax=ax, label='CUSUM alerts after deployment')
    ax.set_title('CUSUM Alerts After Deployment, All Models × All Domains\n'
                 '(counts include models already above DPD 0.1 when deployed; n/e = not evaluated)',
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_alert_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 3 saved, drift_alert_heatmap.png")

    # Figure 4, Model versioning DPD v1/v2/v3
    x = np.arange(len(DOMAINS))
    w = 0.25
    fig, ax = plt.subplots(figsize=(16, 6))
    for i, model in enumerate(MODELS):
        idx = [k for k, d in enumerate(DOMAINS) if STAGE2_RESULTS[d][model] is not None]
        v1_vals = [STAGE2_RESULTS[DOMAINS[k]][model]['v1'] for k in idx]
        v2_vals = [STAGE2_RESULTS[DOMAINS[k]][model]['v2'] for k in idx]
        v3_vals = [np.mean(all_timeseries[DOMAINS[k]][model][SHIFT_START:]) for k in idx]
        pos = x[idx] + (i - 1) * w
        ax.bar(pos - 0.08, v1_vals, w * 0.8, label=f'{model} v1',
               color=COLORS[model], alpha=0.4, edgecolor='black', linewidth=0.5)
        ax.bar(pos, v2_vals, w * 0.8, label=f'{model} v2',
               color=COLORS[model], alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.bar(pos + 0.08, v3_vals, w * 0.8, label=f'{model} v3',
               color=COLORS[model], alpha=1.0, edgecolor='black', linewidth=0.5)
    ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--',
               linewidth=1, label='DPD 0.1 convention')
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=15, ha='right', fontsize=9)
    ax.set_title('Model Versioning, DPD Across v1/v2/v3\n'
                 'v1=baseline, v2=post-constraint, v3=mean under synthetic shift; '
                 'RF absent where not evaluated',
                 fontsize=12)
    ax.set_ylabel('Demographic Parity Difference (DPD)')
    ax.legend(fontsize=7, ncol=4, bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_model_versioning.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 4 saved, drift_model_versioning.png")

    # Figure 5, First alert after deployment (paper Figure 4)
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.axhspan(DEPLOY_START, SHIFT_START, color='#bdbdbd', alpha=0.35, zorder=0)
    ax.axhspan(SHIFT_START, N_POINTS, color='#fdae6b', alpha=0.35, zorder=0)
    ax.text(len(DOMAINS) - 0.45, (DEPLOY_START + SHIFT_START) / 2,
            'v2: constrained\nmodel deployed', ha='right', va='center', fontsize=9, color='#424242')
    ax.text(len(DOMAINS) - 0.45, (SHIFT_START + N_POINTS) / 2,
            'v3: synthetic\nshift', ha='right', va='center', fontsize=9, color='#7f2704')
    for i, model in enumerate(MODELS):
        for k, domain in enumerate(DOMAINS):
            pos = k + (i - 1) * w
            alerts = all_alerts[domain][model]
            if alerts is None:
                ax.text(pos, DEPLOY_START + 0.9, 'n/e', ha='center', va='bottom',
                        fontsize=8, color='gray', rotation=90)
            elif not alerts:
                ax.text(pos, DEPLOY_START + 0.9, 'none', ha='center', va='bottom',
                        fontsize=8, color=COLORS[model], rotation=90)
            else:
                ax.vlines(pos, DEPLOY_START, alerts[0], color=COLORS[model], linewidth=2, zorder=2)
                ax.scatter(pos, alerts[0], s=110, color=COLORS[model], edgecolor='black',
                           linewidth=0.7, zorder=3)
    handles = [plt.Line2D([0], [0], marker='o', linestyle='', markersize=9,
                          markerfacecolor=COLORS[m], markeredgecolor='black') for m in MODELS]
    ax.legend(handles, MODELS, fontsize=9, loc='upper left')
    ax.set_xticks(x)
    ax.set_xticklabels(DOMAINS, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(DEPLOY_START - 1, N_POINTS)
    ax.set_xlim(-0.6, len(DOMAINS) - 0.4)
    ax.set_ylabel('Time step of first alert')
    ax.set_title('First CUSUM Alert After Deployment, All Models × All Domains\n'
                 'Markers at the start of the grey band: already above DPD 0.1 when deployed. '
                 'Orange band: regression under synthetic shift.',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_first_alert_time.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 5 saved, drift_first_alert_time.png")

    # Figure 6, Accuracy-fairness trajectory v1 -> v2 -> v3 (GB)
    fig, ax = plt.subplots(figsize=(14, 8))
    for domain in DOMAINS:
        acc = ACC_RESULTS[domain]['GB']
        dpd = STAGE2_RESULTS[domain]['GB']
        v3_dpd = np.mean(all_timeseries[domain]['GB'][SHIFT_START:])
        v3_acc = acc['v2']   # accuracy is not simulated; held at the deployed model's value
        ax.annotate('', xy=(v3_dpd, v3_acc), xytext=(dpd['v2'], acc['v2']),
                    arrowprops=dict(arrowstyle='->', color=DOMAIN_COLORS[domain], lw=1.5))
        ax.scatter([dpd['v1'], dpd['v2'], v3_dpd], [acc['v1'], acc['v2'], v3_acc],
                   color=DOMAIN_COLORS[domain], s=60, zorder=5)
        ax.text(dpd['v1'] + 0.005, acc['v1'], f'{domain}\nv1', fontsize=7,
                color=DOMAIN_COLORS[domain])
    ax.axvline(x=DPD_CONVENTION, color='black', linestyle='--',
               linewidth=1, label='DPD 0.1 convention')
    ax.set_xlabel('Demographic Parity Difference (DPD)')
    ax.set_ylabel('Accuracy (AUC for Law School, Lending Club, Agricultural)')
    ax.set_title('Accuracy-Fairness Trajectory, GB Model\n'
                 'v1=baseline, v2=post-constraint, v3=synthetic shift '
                 '(v3 accuracy not simulated, held at v2)',
                 fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_acc_fairness_trajectory.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 6 saved, drift_acc_fairness_trajectory.png")

    # EOD series share the construction above, seeded separately from DPD
    eod_timeseries = {}
    for domain in DOMAINS:
        eod_timeseries[domain] = {}
        for model in MODELS:
            pair = EOD_RESULTS[domain][model]
            eod_timeseries[domain][model] = None if pair is None else simulate_drift_timeseries(
                pair['v1'], pair['v2'], series_rng(domain, model, 'eod'))

    # Figure 7, EOD timeseries (GB)
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()
    for i, domain in enumerate(DOMAINS):
        ax = axes[i]
        ts = eod_timeseries[domain]['GB']
        ax.plot(range(len(ts)), ts, color=DOMAIN_COLORS[domain], linewidth=2)
        ax.axhline(y=DPD_CONVENTION, color='black', linestyle='--',
                   linewidth=1, label='0.1 reference')
        ax.axvline(x=DEPLOY_START, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax.axvline(x=SHIFT_START, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ymax = max(ax.get_ylim()[1], ts.max() * 1.1)
        ax.text(5, ymax * 0.95, 'v1', ha='center', fontsize=8, color='gray')
        ax.text(15, ymax * 0.95, 'v2', ha='center', fontsize=8, color='gray')
        ax.text(25, ymax * 0.95, 'v3', ha='center', fontsize=8, color='gray')
        ax.set_title(domain, fontsize=10)
        ax.set_xlabel('Time step')
        ax.set_ylabel('EOD')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Simulated EOD Across 3 Model Versions, GB Model\n'
                 'v1=baseline, v2=post-EO constraint, v3=synthetic shift',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'drift_eod_timeseries.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("  Fig 7 saved, drift_eod_timeseries.png")

    def magnitude_heatmap(series, pairs, metric, filename, fig_number):
        matrix = np.array([
            [np.nan if pairs[d][m] is None
             else max(0.0, np.mean(series[d][m][SHIFT_START:]) - pairs[d][m]['v2'])
             for m in MODELS] for d in DOMAINS
        ])
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='YlOrRd', mask=np.isnan(matrix),
                    ax=ax, xticklabels=MODELS, yticklabels=DOMAINS,
                    linewidths=0.5, cbar_kws={'label': f'{metric} Drift Magnitude (v3 mean minus v2)'})
        for i in range(len(DOMAINS)):
            for j in range(len(MODELS)):
                if np.isnan(matrix[i, j]):
                    ax.text(j + 0.5, i + 0.5, 'n/e', ha='center', va='center', fontsize=10, color='gray')
        ax.set_title(f'{metric} Drift Magnitude Under Synthetic Shift\n'
                     '(about 30% of each correction by construction; 0 where the constraint worsened the metric; n/e = not evaluated)',
                     fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Fig {fig_number} saved - {filename}")

    # Figure 8, DPD drift magnitude heatmap
    magnitude_heatmap(all_timeseries, STAGE2_RESULTS, 'DPD', 'drift_magnitude_heatmap.png', 8)

    # Figure 9, EOD drift magnitude heatmap
    magnitude_heatmap(eod_timeseries, EOD_RESULTS, 'EOD', 'drift_eod_magnitude_heatmap.png', 9)

    at_deployment, in_shift, no_alert = [], [], []
    for domain in DOMAINS:
        for model in MODELS:
            alerts = all_alerts[domain][model]
            if alerts is None:
                continue
            label = f"{domain} {model}"
            window = alert_window(alerts[0] if alerts else None)
            {'deployment': at_deployment, 'shift': in_shift, 'none': no_alert}[window].append(label)

    print(f"\n--- Stage 4 Drift Detection complete ---")
    print(f"  9 figures saved to figures/stage2/")
    print(f"  Above DPD 0.1 at deployment ({len(at_deployment)}): {', '.join(at_deployment)}")
    print(f"  Regression flagged under shift ({len(in_shift)}): {', '.join(in_shift)}")
    print(f"  Never flagged ({len(no_alert)}): {', '.join(no_alert)}")
    print(f"  Shift is synthetic and sized as a share of each correction (Decision 7)")


if __name__ == "__main__":
    run_fairness_drift_monitor()
