"""
FAPE: generate the Section 5 results table directly from threshold_aggregation.RESULTS.

Exists so the manuscript table cannot drift from the numbers the scripts produce.
docs/cross_domain_results_table.md was hand-annotated and its prose contains errors
(see git history); this regenerates from the source dict instead.

MEPS values come from the FairGround pipeline's meps_panel_19_fy2015 sub-dataset,
verified against a pinned-environment run.

Output: docs/results_table.md  (paste-ready markdown)
"""

import os, importlib.util

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location(
    "ta", os.path.join(REPO_ROOT, "src", "threshold_aggregation.py"))
ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)
R = dict(ta.RESULTS)

MEPS = ta.MEPS_RESULTS

ORDER = [
    ("COMPAS", R['COMPAS']),
    ("Folktables", R['Folktables']),
    ("Law School", R['Law School']),
    ("Lending Club", R['Lending Club']),
    ("Agricultural", R['Agricultural']),
    ("FairGround", R['FairGround (Education/law_school_lequy)']),
    ("MEPS", MEPS),
    ("Student", R['Student (math)']),
]

def cell(entry, before, after):
    if entry is None:
        return "n/e"
    b, a = entry.get(before), entry.get(after)
    if b is None or a is None:
        return "n/e"
    arrow = "→"
    return f"{b:.3f}{arrow}{a:.3f}"

def build():
    lines = []
    lines.append("| Evaluation | DPD LR | DPD RF | DPD GB | EOD LR | EOD RF | EOD GB |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, mods in ORDER:
        row = [name]
        for b, a in [('baseline_dpd', 'dp_dpd'), ('baseline_eod', 'eo_eod')]:
            for mdl in ['LR', 'RF', 'GB']:
                row.append(cell(mods.get(mdl), b, a))
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)

    caption = ("\n\n**Table 2.** Demographic parity difference and equalized odds difference, "
               "baseline to post-constraint, for every model-domain pair. "
               "n/e marks a model not evaluated under the intervention in that domain "
               "(Section 3.3). Values are produced by src/make_results_table.py from the "
               "same source the figures use.\n")

    out = os.path.join(REPO_ROOT, "docs", "results_table.md")
    with open(out, "w") as fh:
        fh.write(table + caption)
    print(table)
    print(f"\nwritten to {out}")
    print(f"table words: {len((table+caption).split())}")

if __name__ == "__main__":
    build()
