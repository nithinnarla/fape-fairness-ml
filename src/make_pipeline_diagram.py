"""
FAPE, Stage 0: pipeline diagram for the paper (Figure 1).

Replaces the hand-made version so the figure regenerates from code like every
other figure in the repo. Text is kept in sync with docs/paper_draft.md:
eight evaluations, and metric coverage stated per metric rather than claiming
all four are computed everywhere (Sections 3.5, 5.4 and 5.6).

Output: figures/paper/fape_pipeline_diagram.png
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrow

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'figures', 'paper')
os.makedirs(OUT_DIR, exist_ok=True)

STAGES = [
    ("Stage 1: Data preprocessing",
     "8 evaluations, sensitive attributes, stratified sampling", "#1e9e6a"),
    ("Stage 2: Baseline classification",
     "LR, RF and GB with default hyperparameters", "#2b7bba"),
    ("Stage 3: Fairness intervention",
     "ThresholdOptimizer, DP and EO constraints", "#7d6ecf"),
    ("Stage 4: Drift monitoring",
     "CUSUM from deployment, DPD 0.1 convention", "#d4941e"),
]

# Coverage matches Sections 3.5, 5.4, 5.6: DPD and EOD everywhere; DIR and
# accuracy cost only where the domain pipeline computes them.
METRICS = [
    ("DPD", "Parity difference", "all 8"),
    ("EOD", "Odds difference", "all 8"),
    ("DIR", "Impact ratio", "3 fixed pairs + Folktables"),
    ("Accuracy cost", "Baseline minus constrained", "5 evaluations"),
]

def build():
    fig, ax = plt.subplots(figsize=(11, 10))
    ax.set_xlim(0, 10); ax.set_ylim(-0.45, 10.1); ax.axis('off')

    box_x, box_w, box_h, gap = 1.3, 7.4, 1.35, 0.52
    top = 9.5

    for i, (title, sub, colour) in enumerate(STAGES):
        y = top - i * (box_h + gap) - box_h
        ax.add_patch(FancyBboxPatch(
            (box_x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=colour, edgecolor='none'))
        ax.text(box_x + box_w / 2, y + box_h * 0.62, title,
                ha='center', va='center', fontsize=14, fontweight='bold', color='white')
        ax.text(box_x + box_w / 2, y + box_h * 0.26, sub,
                ha='center', va='center', fontsize=10.5, color='white')
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrow(
                box_x + box_w / 2, y - 0.08, 0, -gap + 0.22,
                width=0.035, head_width=0.20, head_length=0.17,
                length_includes_head=True, color='#444444'))

    # side annotations, as in the original
    ax.text(9.05, top - box_h / 2 - (box_h + gap), "Regulatory\ncontext\nper domain",
            ha='left', va='center', fontsize=10, color='#333333')
    ax.text(0.95, top - box_h / 2 - 2 * (box_h + gap), "Post-processing,\nno retraining",
            ha='right', va='center', fontsize=10, style='italic', color='#333333')

    # metric strip
    strip_y = top - 4 * (box_h + gap) - 0.95
    ax.text(5.0, strip_y + 0.62, "Metrics reported per evaluation",
            ha='center', va='center', fontsize=12.5, color='#222222')

    mw, mgap = 1.98, 0.14
    total = len(METRICS) * mw + (len(METRICS) - 1) * mgap
    mx = 5.0 - total / 2
    for name, desc, cover in METRICS:
        ax.add_patch(FancyBboxPatch(
            (mx, strip_y - 1.16), mw, 1.06,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor='#eeece7', edgecolor='#b9b5ac', linewidth=0.9))
        ax.text(mx + mw / 2, strip_y - 0.34, name,
                ha='center', va='center', fontsize=11, fontweight='bold', color='#222222')
        ax.text(mx + mw / 2, strip_y - 0.66, desc,
                ha='center', va='center', fontsize=8.0, color='#555555',
                linespacing=1.25)
        ax.text(mx + mw / 2, strip_y - 0.96, cover,
                ha='center', va='center', fontsize=8.0, style='italic', color='#777777')
        mx += mw + mgap

    out = os.path.join(OUT_DIR, 'fape_pipeline_diagram.png')
    plt.savefig(out, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"saved {out}")

if __name__ == '__main__':
    build()
