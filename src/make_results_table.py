"""
FAPE: generate the results tables directly from threshold_aggregation.RESULTS.

Exists so the tables cannot drift from the numbers the scripts produce. Two files:
- docs/results_table.md, Table 2 of the paper (paste-ready markdown)
- docs/cross_domain_results_table.md, the longer tables behind Sections 5 and 6:
  baseline performance, accuracy cost, disparate impact ratio and the two checks

MEPS values come from the FairGround pipeline's meps_panel_19_fy2015 sub-dataset,
verified against a pinned-environment run.

The disparate impact ratios and the two check tables are not in RESULTS. They are
copied below from the printed output of the intervention notebooks (notebooks/stage2_*),
notebooks/group_size_check.ipynb and notebooks/threshold_holdout_check.ipynb, and
src/check_consistency.py confirms each number still appears in that output. The
sentences under each table are written by hand, so the claims they make are
asserted against the numbers first: if a rerun changes a result, this script stops
instead of writing a sentence that is no longer true.

Output: docs/results_table.md, docs/cross_domain_results_table.md
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
    ("Student (Math)", R["Student (math)"]),
]
MODELS = ['LR', 'RF', 'GB']
AUC_ONLY = {"Law School", "Lending Club", "Agricultural"}   # their intervention scripts report AUC, not accuracy

# Copied from printed output; check_consistency.py traces every number back to it.
# Gradient boosting's fixed-pair disparate impact ratio (intervention notebooks).
DIR_ROWS = [
    ("Law School", "Minority / majority, predicted bar passage", "Favorable", "0.643", "0.957",
     "Moves above the 0.8 convention"),
    ("Lending Club", "Lowest / highest income quartile, predicted default", "Adverse", "2.778", "0.973",
     "Reaches parity because predicted default rises to about 40% in every quartile"),
    ("Agricultural", "Partnership / corporation, predicted default", "Adverse", "0.653", "1.042",
     "Ends just past parity as every business type's predicted default rate rises to 16 to 19%"),
    ("Folktables", "Each race group / White, predicted income above $50K, EO constraint", "Favorable",
     "See note", "See note",
     "Black 0.72→0.83, multiracial 0.70→0.92, American Indian 0.61→1.00, "
     "Pacific Islander 0.72→1.03, Other 0.42→just under 0.8"),
]
# The two checks live next to RESULTS in threshold_aggregation.py, which also draws them.
# Group size: DPD under the DP constraint over all race groups and over groups with at
# least 30 test records. Threshold fitting: thresholds chosen on the training split and on
# a held-out quarter of it, plus each random forest's training accuracy.
DISPLAY = {"Student (math)": "Student (Math)"}
GROUP_SIZE_ROWS = [(DISPLAY.get(d, d), m, v["all_groups"], v["at_least_30"])
                   for (d, m), v in ta.GROUP_SIZE_CHECK.items()]
HOLDOUT_ROWS = [(DISPLAY.get(d, d), m, v["training_split"], v["held_out"])
                for (d, m), v in ta.HOLDOUT_CHECK.items()]
RF_TRAINING_ACCURACY = {DISPLAY.get(d, d): f"{v:.3f}" for d, v in ta.RF_TRAINING_ACCURACY.items()}


def cell(entry, before, after):
    if entry is None:
        return "n/e"
    b, a = entry.get(before), entry.get(after)
    if b is None or a is None:
        return "n/e"
    arrow = "→"
    return f"{b:.3f}{arrow}{a:.3f}"


def results_table_text():
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
    return table + caption


def pairs(rows):
    """Every evaluated (evaluation, model, entry) triple."""
    return [(n, m, e[m]) for n, e in rows for m in MODELS if e[m] is not None]


def reduction(r, before, after):
    return (r[before] - r[after]) / r[before] * 100


def arrow(before_after):
    return "→".join(f"{v:.3f}" for v in before_after)


def cross_table_text():
    rows = [("FairGround (law_school_lequy)" if n == "FairGround" else n, e) for n, e in ORDER]
    everything = pairs(rows)
    high = [(n, m, r) for n, m, r in everything if r["baseline_dpd"] > 0.2]
    low = [(n, m, r) for n, m, r in everything if r["baseline_dpd"] < 0.05]
    middle = sorted({n for n, m, r in everything if 0.05 <= r["baseline_dpd"] <= 0.2})
    improved_high = [(n, m) for n, m, r in high if r["dp_dpd"] < r["baseline_dpd"]]
    worsened_low = [(n, m) for n, m, r in low if r["dp_dpd"] > r["baseline_dpd"]]
    failed_high = [(n, m) for n, m, r in high if r["dp_dpd"] >= r["baseline_dpd"]]
    dp_improves = {(n, m): r["dp_dpd"] < r["baseline_dpd"] for n, m, r in everything}
    eo_change = {(n, m): reduction(r, "baseline_eod", "eo_eod") for n, m, r in everything}

    def once(dropped):
        kept = [(n, m, r) for n, m, r in high if n != dropped]
        return sum(r["dp_dpd"] < r["baseline_dpd"] for _, _, r in kept), len(kept)

    # The hand-written sentences below make these claims.
    acc_rows = [(n, e) for n, e in rows if n not in AUC_ONLY]
    best_acc = {n: max(MODELS, key=lambda m: e[m]["baseline_acc"]) for n, e in acc_rows}
    assert best_acc == {"COMPAS": "LR", "Folktables": "GB", "FairGround (law_school_lequy)": "LR",
                        "MEPS": "GB", "Student (Math)": "GB"}, best_acc
    assert all(e["GB"]["baseline_auc"] > e["LR"]["baseline_auc"] for n, e in rows if n in AUC_ONLY)
    assert worsened_low == [("Lending Club", "LR"), ("Agricultural", "LR"), ("Agricultural", "GB")], worsened_low
    assert sum(n in ("Law School", "FairGround (law_school_lequy)") for n, _, _ in high) == 5
    assert middle == ["MEPS"], middle
    assert dp_improves[("MEPS", "LR")] and dp_improves[("MEPS", "GB")] and not dp_improves[("MEPS", "RF")]
    assert failed_high == [("COMPAS", "LR"), ("COMPAS", "RF"), ("Folktables", "LR"), ("Folktables", "GB"),
                           ("Student (Math)", "RF")], failed_high
    assert max(eo_change, key=eo_change.get) == ("Law School", "GB")
    assert all(eo_change[("Student (Math)", m)] > 0 for m in MODELS)
    for n in ("COMPAS", "Folktables", "MEPS"):
        assert eo_change[(n, "LR")] > 0 and eo_change[(n, "GB")] > 0 and eo_change[(n, "RF")] < 0, n

    out = []
    w = out.append
    w("# FAPE Cross-Domain Results Tables")
    w("")
    w("> **Record note, September 15 2026.** Rebuilt from `threshold_aggregation.RESULTS` and `MEPS_RESULTS` after the MEPS and Folktables feature fixes (Decisions 24 and 25 in methodology_decisions.md). The July to August version, which predated those fixes and had several key findings that no longer matched its own tables, is in the git history. Every value below reproduced in a fresh environment built from requirements.txt. docs/results_table.md is the generated Table 2 of the paper; this file adds baseline performance, accuracy cost, disparate impact ratio and the two sensitivity checks.")
    w("")
    w("n/e marks a model not evaluated under the intervention in that domain. Law School, Lending Club and Agricultural carry logistic regression and gradient boosting through the intervention and report AUC rather than accuracy (Decisions 13 and 16).")
    w("")
    w("---")
    w("")
    w("## Table 1, Baseline Performance")
    w("")
    w("| Evaluation | LR | RF | GB | Metric |")
    w("|---|---|---|---|---|")
    for name, e in rows:
        if name in AUC_ONLY:
            w(f"| {name} | {e['LR']['baseline_auc']:.3f} | n/e | {e['GB']['baseline_auc']:.3f} | AUC |")
        else:
            w(f"| {name} | {e['LR']['baseline_acc']:.3f} | {e['RF']['baseline_acc']:.3f} | {e['GB']['baseline_acc']:.3f} | Accuracy |")
    w("")
    w("Gradient boosting has the highest accuracy in three of the five accuracy evaluations (Folktables, MEPS, Student) and logistic regression in the other two (COMPAS, FairGround). Gradient boosting has the highest AUC in all three AUC evaluations. Random forest is not evaluated under the intervention for the AUC domains; the separate baseline scripts, which use their own preprocessing and samples, give it an AUC of 0.854 for Law School, 0.920 for Agricultural and 0.699 for Lending Club, below gradient boosting in the same scripts. Accuracy and AUC are not compared with each other.")
    w("")
    w("---")
    w("")
    w("## Table 2, Demographic Parity Difference, Baseline → After the DP Constraint")
    w("")
    w("| Evaluation | LR | RF | GB |")
    w("|---|---|---|---|")
    for name, e in rows:
        w(f"| {name} | " + " | ".join(cell(e[m], "baseline_dpd", "dp_dpd") for m in MODELS) + " |")
    w("")
    w("## Table 3, Equalized Odds Difference, Baseline → After the EO Constraint")
    w("")
    w("| Evaluation | LR | RF | GB |")
    w("|---|---|---|---|")
    for name, e in rows:
        w(f"| {name} | " + " | ".join(cell(e[m], "baseline_eod", "eo_eod") for m in MODELS) + " |")
    w("")
    w("**What Tables 2 and 3 show**")
    w(f"- The DP constraint improved DPD in {len(improved_high)} of the {len(high)} model-domain pairs with baseline DPD above 0.2. The {len(high) - len(improved_high)} exceptions are {', '.join(f'{n} {m}' for n, m in failed_high)}.")
    w(f"- It worsened DPD in {len(worsened_low)} of the {len(low)} pairs with baseline DPD below 0.05 (both Agricultural models and Lending Club's logistic regression).")
    w("- Five of the high-disparity pairs use law_school_lequy twice, once through Law School and once through FairGround. Counting that data once gives {} of {} or {} of {}.".format(*once("FairGround (law_school_lequy)"), *once("Law School")))
    w("- MEPS is the only evaluation between 0.05 and 0.2. Logistic regression and gradient boosting improve under DP, and random forest worsens.")
    w("- The COMPAS and Folktables values are set by race groups with fewer than 30 test records, and the random forest worsening in Student and MEPS comes from thresholds fit on the training split. Tables 6 and 7 show both checks.")
    w(f"- Under EO, Law School's gradient boosting improves most ({eo_change[('Law School', 'GB')]:.1f}%), all three Student models improve, and COMPAS, Folktables and MEPS each improve under logistic regression and gradient boosting and worsen under random forest.")
    w("")
    w("---")
    w("")
    w("## Table 4, Accuracy Cost of the DP Constraint (baseline accuracy minus constrained accuracy)")
    w("")
    w("| Evaluation | LR | RF | GB |")
    w("|---|---|---|---|")
    costs = {}
    for name, e in acc_rows:
        for m in MODELS:
            costs[(name, m)] = round(e[m]["baseline_acc"] - e[m]["dp_acc"], 3)
        w(f"| {name} | " + " | ".join(f"{costs[(name, m)]:+.3f}" for m in MODELS) + " |")
    fg = "FairGround (law_school_lequy)"
    assert sorted(costs, key=costs.get)[-2:] == [(fg, "LR"), (fg, "GB")], sorted(costs, key=costs.get)[-2:]
    w("")
    w(f"Negative means accuracy rose. Values are computed from the three-decimal numbers in RESULTS, so a script's own printed cost can differ by 0.001. Law School, Lending Club and Agricultural are omitted because their intervention scripts report AUC. FairGround's logistic regression and gradient boosting pay the most ({costs[(fg, 'LR')]:.3f} and {costs[(fg, 'GB')]:.3f}) while its random forest pays {costs[(fg, 'RF')]:.3f} for a similar DPD reduction. These costs also include the change from the default threshold to a balanced-accuracy objective (Decision 23).")
    w("")
    w("---")
    w("")
    w("## Table 5, Disparate Impact Ratio, Gradient Boosting")
    w("")
    w("| Evaluation | Ratio | Outcome | Baseline | After | Reading |")
    w("|---|---|---|---|---|---|")
    for row in DIR_ROWS:
        w("| " + " | ".join(row) + " |")
    w("")
    w("COMPAS, FairGround and MEPS compute no fixed-pair ratio. Student computes a female-to-male ratio for the baseline only (0.487 in Math, 2.084 in Portuguese). The 0.8 line is a research convention here, not a legal test for any of these domains, and for an adverse outcome a ratio above 1.0 is the harmful direction (Decision 21).")
    w("")
    w("---")
    w("")
    w("## Table 6, Group-Size Check (src/group_size_check.py)")
    w("")
    w("DPD after the DP constraint, measured over all race groups and over groups with at least 30 test records.")
    w("")
    w("| Evaluation | Model | All groups | Groups n≥30 |")
    w("|---|---|---|---|")
    for n, m, all_groups, large in GROUP_SIZE_ROWS:
        assert all_groups == (round(dict(rows)[n][m]["baseline_dpd"], 3), round(dict(rows)[n][m]["dp_dpd"], 3)), (n, m)
        w(f"| {n} | {m} | {arrow(all_groups)} | {arrow(large)} |")
    large_improves = {(n, m): large[1] < large[0] and large[0] > 0.2 for n, m, _, large in GROUP_SIZE_ROWS}
    assert all(large_improves.values())
    recovered = sum(large_improves.get(p, False) for p in failed_high)
    w("")
    w(f"COMPAS's test split has 7 Asian defendants and 1 Native American defendant; Folktables' has 5 Alaska Native respondents and 25 in the combined American Indian and Alaska Native category. Measured on the larger groups, all six models improve, and the high-disparity count becomes {len(improved_high) + recovered} of {len(high)}.")
    w("")
    w("## Table 7, Threshold-Fitting Check (src/threshold_holdout_check.py)")
    w("")
    w("DPD under the DP constraint when thresholds are chosen on the training split (the design the intervention scripts use) and on a held-out quarter of it.")
    w("")
    w("| Evaluation | Model | Training-split thresholds | Held-out thresholds |")
    w("|---|---|---|---|")
    for n, m, stage2, held in HOLDOUT_ROWS:
        assert stage2 == (round(dict(rows)[n][m]["baseline_dpd"], 3), round(dict(rows)[n][m]["dp_dpd"], 3)), (n, m)
        if m == "RF":
            assert stage2[1] > stage2[0] and held[1] < held[0], (n, m)
        else:
            assert 0 < held[0] - held[1] < stage2[0] - stage2[1], (n, m)
        w(f"| {n} | {m} | {arrow(stage2)} | {arrow(held)} |")
    w("")
    w(f"Random forest fits its training records almost perfectly (training accuracy {RF_TRAINING_ACCURACY['Student (Math)']} in Student and {RF_TRAINING_ACCURACY['MEPS']} in MEPS), so thresholds chosen on them transfer poorly. With held-out thresholds its worsening reverses, while logistic regression and gradient boosting improve less in all four cases. The baselines in the held-out column differ slightly because those models train on 75% of the training split.")
    w("")
    return "\n".join(out)


def build():
    outputs = {
        os.path.join(REPO_ROOT, "docs", "results_table.md"): results_table_text(),
        os.path.join(REPO_ROOT, "docs", "cross_domain_results_table.md"): cross_table_text(),
    }
    for path, text in outputs.items():
        with open(path, "w") as fh:
            fh.write(text)
        print(f"written to {path}")
    table = outputs[os.path.join(REPO_ROOT, "docs", "results_table.md")]
    print(table)
    print(f"table words: {len(table.split())}")


if __name__ == "__main__":
    build()
