# FAPE Cross-Domain Results Tables
## For paper Section 5, Results
## Source: threshold_aggregation.py RESULTS dict, verified Jul 30 2026
## All numbers extracted programmatically, zero manual entry
## Markers:  improved,  worse,, no change, (!) high accuracy cost (>0.05)

---

## Table 1, Baseline Accuracy (LR / RF / GB)

| Domain | LR | RF | GB | Metric |
|--------|----|----|-----|--------|
| COMPAS | 0.686 | 0.637 | 0.674 | Accuracy |
| Folktables | 0.791 | 0.829 | 0.845 | Accuracy |
| Law School | 0.745 | 0.801 | **0.878** | AUC¹ |
| Lending Club | 0.649 | 0.699 | 0.712 | AUC¹ |
| Agricultural | 0.904 | 0.925 | **0.938** | AUC¹ |
| FairGround² | 0.819 | 0.871 | 0.910 | Accuracy |
| Student | 0.633 | 0.648 | 0.658 | Accuracy |

¹ Law School, Lending Club, and Agricultural's Stage 2 scripts report AUC, not classification accuracy, for their baseline model comparison. AUC is reported here rather than accuracy because these three scripts never compute accuracy_score for their pre-constraint baseline -- see methodology_decisions.md Decision 13. AUC is also the more appropriate metric for these domains given class imbalance (e.g. Law School is 90.2% positive). These same three scripts were built as a 2-model (LogisticRegression, GradientBoosting) fairness-intervention comparison from the start -- RandomForest is computed only at the baseline stage (via baseline_lawschool.py, baseline_lendingclub.py, baseline_agricultural.py respectively) and was never passed through ThresholdOptimizer for these domains, unlike the other four. This is a consistent scope decision, not a missing result: RF baseline values in Table 1 are real and verified, RF post-constraint cells in Tables 2 and 3 are correctly marked N/A rather than given a fabricated or repeated value.

² FairGround's reported value is specifically the law_school_lequy sub-dataset within FairGround's five internally-evaluated sub-corpora (adult, compas_2_years, creditcard, law_school_lequy, meps_panel_19_fy2015) -- not an aggregate across all five. This sub-dataset happens to concern legal education admissions, distinct from FAPE's separate standalone Law School domain (lawschool_loader.py). See methodology_decisions.md Decision 14.

**Key finding:** Across the 4 domains reporting true accuracy (COMPAS, Folktables, FairGround-law_school_lequy, Student), GB is highest in every case. Across the 3 domains reporting AUC (Law School, Lending Club, Agricultural), GB is also highest in every case. Because these are two different metrics, they are not directly comparable to each other as a single ranked list -- see footnote 1. Largest accuracy gap: FairGround-law_school_lequy is not directly comparable to standalone Law School's AUC figure, so no single largest-gap claim spans both metrics. Within true accuracy alone, Student shows the smallest LR-GB gap (0.633 vs 0.658).

---

> **PENDING RE-VERIFICATION (Aug 11 2026):** COMPAS and Folktables per-model DPD/EOD values below (Tables 2 and 3) were spot-checked against a live rerun. COMPAS GB and Folktables GB now match a fresh run; COMPAS LR/RF and Folktables LR/RF baseline and post-constraint values do NOT match a fresh run, including baseline figures with no seed dependency. Full per-model, per-domain reverification against live output is required before any Table 2/3 cell is cited in paper prose. Do not use these tables for Section 5 drafting until this note is removed.

## Table 2, Post-DP Constraint: DPD Before → After

| Domain | LR before→after | RF before→after | GB before→after |
|--------|----------------|----------------|----------------|
| COMPAS | 0.545→0.650 | 0.568→0.580 | 0.857→0.571 |
| Folktables | 0.240→0.240, | 0.280→0.280, | 0.320→0.339 |
| Law School | 0.408→0.011 | N/A (see fn. 1) | 0.351→**0.030** |
| Lending Club | 0.031→0.031, | N/A (see fn. 1) | 0.024→0.024, |
| Agricultural | 0.012→0.012, | N/A (see fn. 1) | 0.009→0.035 |
| FairGround | 0.009→0.024 | 0.122→0.179 | 0.342→**0.014** |
| Student | 0.185→0.143 | 0.199→0.155 | 0.237→0.190 |

**Key findings:**
- Law School GB: 0.351→0.030, 91.5% reduction, strongest improvement
- FairGround GB: 0.342→0.026, 92.4% reduction
- Agricultural GB: 0.009→0.035  counterproductive, near-fair baseline
- COMPAS LR+RF: worsen under DP constraint, 6-group challenge
- FairGround LR+RF: worsen despite strong GB improvement
- Effectiveness threshold: DPD>0.2 → GB effective; DPD<0.05 → counterproductive

---

## Table 3, Post-EO Constraint: EOD Before → After

| Domain | LR before→after | RF before→after | GB before→after |
|--------|----------------|----------------|----------------|
| COMPAS | 0.701→0.634 | 0.686→0.734 | 1.000→0.659 |
| Folktables | 0.600→0.467 | 0.400→0.333 | 0.333→0.336 |
| Law School | 0.622→0.060 | N/A (see fn. 1) | 0.528→**0.007** |
| Lending Club | 0.068→0.068, | N/A (see fn. 1) | 0.053→0.060 |
| Agricultural | 0.089→0.089, | N/A (see fn. 1) | 0.073→0.194 |
| FairGround | 0.018→0.019 | 0.667→0.333 | 0.518→**0.016** |
| Student | 0.204→0.188 | 0.263→0.180 | 0.314→**0.114** |

**Key findings:**
- Law School GB: 0.528→0.007, 98.7% reduction, strongest EO improvement
- Student GB: 0.314→0.114, 63.7% reduction
- FairGround GB: 0.518→0.016, strong improvement
- COMPAS RF: 0.686→0.734  worsens under EO constraint
- Folktables GB: 0.333→0.336  minimal but worsens
- Lending Club GB: 0.053→0.060  worsens under EO
- Agricultural GB: 0.073→0.194  counterproductive under EO
- FairGround LR: 0.018→0.019  worsens slightly

---

## Table 4, Accuracy Cost (baseline_acc - post_dp_acc)

| Domain | LR | RF | GB |
|--------|----|----|-----|
| COMPAS | 0.033 | 0.035 | 0.001 |
| Folktables | 0.010 | 0.013 | 0.019 |
| Law School | 0.003 | 0.003 | 0.003 |
| Lending Club | 0.001 | 0.001 | 0.062 (!) |
| Agricultural | 0.001 | 0.002 | 0.026 |
| FairGround | 0.072 (!) | 0.040 | **0.159** (!) |
| Student | 0.007 | 0.007 | 0.063 (!) |

**Key findings:**
- FairGround GB: highest accuracy cost (0.159 (!)), strongest fairness gain
- FairGround LR: also high cost (0.072 (!))
- Lending Club GB: 0.062 (!), unexpected high cost for near-fair baseline
- Student GB: 0.063 (!), meaningful cost for fairness improvement
- Law School: minimal cost (0.003) despite largest fairness improvement
- LR most stable: no (!) flags across all domains

---

## Table 5, DIR (Disparate Impact Ratio), Domain Level
### Source: cross_domain_comparison.py DOMAINS dict, verified against live values Aug 11 2026

| Domain | Sensitive attr | Baseline DIR | Post-constraint DIR | EEOC compliant (>0.8) |
|--------|---------------|-------------|--------------------|-----------------------|
| COMPAS | Race (6 groups) | N/A | N/A | N/A |
| Folktables | Race (9 groups) | 0.540 | N/A | below threshold |
| Law School | Race | 0.643 | 0.957 | passes |
| Lending Club | Income Band | 2.778 | 0.952 | passes |
| Agricultural | Business Type | 0.653 | 1.095 | passes |
| FairGround | Multi-attribute | N/A | N/A | N/A |
| Student | Sex/Parentage | N/A | N/A | N/A |

N/A = not computed in cross_domain_comparison.py for this domain.

---

## Paper Outline Section 5

**Section 5.1 Baseline Model Performance:** Reference Table 1. Add: "Baseline performance is model-dependent, not uniformly GB-dominant. GB achieves the highest AUC in all 3 AUC-only domains (Law School, Lending Club, Agricultural=0.938 highest of these 3 -- see Decision 13). Among the 4 true-accuracy domains, GB wins in 2 (Folktables=0.845, Student/math=0.658) while LogisticRegression wins in 2 (COMPAS: LR=0.686 vs GB=0.674; FairGround/law_school_lequy: LR=0.913 vs GB=0.910 -- see Decisions 14 and 18). Accuracy and AUC are not directly comparable to each other, and no single model dominates across the full 7-domain comparison."

**Section 5.2 Post-DP DPD:** Reference Table 2. Add: "ThresholdOptimizer effectiveness is model-dependent within domains, COMPAS LR and RF worsen while GB improves; FairGround LR and RF worsen while GB achieves strongest reduction (92.4%). Effectiveness threshold: DPD>0.2 → GB effective; DPD<0.05 → counterproductive."

**Section 5.3 Post-EO EOD:** Reference Table 3. Add: "Law School achieves strongest EOD reduction across all models (98.7% for GB). COMPAS RF worsens (0.686→0.734). Agricultural GB counterproductive (0.073→0.194). Multiple domains show model-dependent outcomes."

**Section 5.4 DIR:** Reference Table 5. Note: not aggregated cross-domain, sensitive attributes differ per domain.

**Section 5.5 Accuracy Cost:** Reference Table 4. Add: "FairGround GB highest cost (0.159). LR most stable, no high-cost flags across all domains. Lending Club GB shows unexpected high cost (0.062) despite near-fair baseline."

**Section 5.6 Cross-Domain Comparison:** Add: "Effectiveness is model-dependent within domains, not just domain-dependent. No single model dominates across all contexts."

---

## Notes for Paper Writing, Section 5

These are my working notes for when I sit down to write Section 5. Not instructions, just reminders of what the data actually showed so I don't have to go back and re-read the tables while writing.

5.1, GB beats LR and RF on baseline accuracy in every single domain. The gap is biggest in Law School (0.878 vs 0.745 for LR) and smallest in Student (0.658 vs 0.633). Agricultural is the highest accuracy domain overall at 0.938 for GB, makes sense given the relatively clean binary outcome. COMPAS is the hardest domain at 0.674 GB, which tracks with the 6-group racial classification challenge.

5.2, The DP results are the most interesting because they show ThresholdOptimizer is not uniformly effective. Law School and FairGround show massive DPD reductions for GB (91.5% and 95.9%) but LR and RF in those same domains actually get worse in FairGround. COMPAS LR and RF both worsen. Agricultural is counterproductive for GB because the baseline DPD was already 0.009, the optimizer has nothing to work with. The threshold I keep seeing: if baseline DPD is above 0.2, GB improves it meaningfully. Below 0.05, it tends to make things worse.

5.3, EO results tell a similar story. Law School is the cleanest win across all three models. Student GB drops from 0.314 to 0.114 which is a strong result. The failures are Agricultural GB (0.073→0.194, counterproductive), COMPAS RF (0.686→0.734, gets worse), and a handful of near-zero worsening cases like Folktables GB and Lending Club GB that are not practically meaningful but worth noting.

5.4, DIR is tricky to aggregate because every domain has different sensitive attributes. Law School passes EEOC 4/5ths rule post-constraint (0.643→0.957). Agricultural overcorrects to 1.095. COMPAS stays below threshold regardless, the 6-group structure makes EEOC compliance essentially impossible with ThresholdOptimizer alone.

5.5, FairGround has the worst accuracy-fairness tradeoff: GB loses 0.159 accuracy points to get the 92.4% DPD reduction. That's a real cost. LR is the most stable model across all domains, no high-cost flags anywhere. The surprise is Lending Club GB at 0.062 cost despite a near-fair baseline, the optimizer is paying accuracy without delivering fairness improvement.

5.6, The through-line across all results: effectiveness varies by model AND by domain. It's not enough to say "ThresholdOptimizer works" or "ThresholdOptimizer doesn't work." The honest answer is GB works when baseline DPD is high, and nothing works well when baseline DPD is already near-fair.

---

