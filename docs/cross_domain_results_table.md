# FAPE Cross-Domain Results Tables

> **Record note, September 15 2026.** Rebuilt from `threshold_aggregation.RESULTS` and `MEPS_RESULTS` after the MEPS and Folktables feature fixes (Decisions 24 and 25 in methodology_decisions.md). The July to August version, which predated those fixes and had several key findings that no longer matched its own tables, is in the git history. Every value below reproduced in a fresh environment built from requirements.txt. docs/results_table.md is the generated Table 2 of the paper; this file adds baseline performance, accuracy cost, disparate impact ratio and the two sensitivity checks.

n/e marks a model not evaluated under the intervention in that domain. Law School, Lending Club and Agricultural carry logistic regression and gradient boosting through the intervention and report AUC rather than accuracy (Decisions 13 and 16).

---

## Table 1, Baseline Performance

| Evaluation | LR | RF | GB | Metric |
|---|---|---|---|---|
| COMPAS | 0.686 | 0.637 | 0.674 | Accuracy |
| Folktables | 0.740 | 0.726 | 0.756 | Accuracy |
| Law School | 0.872 | n/e | 0.878 | AUC |
| Lending Club | 0.706 | n/e | 0.712 | AUC |
| Agricultural | 0.727 | n/e | 0.938 | AUC |
| FairGround (law_school_lequy) | 0.913 | 0.907 | 0.910 | Accuracy |
| MEPS | 0.855 | 0.857 | 0.859 | Accuracy |
| Student (Math) | 0.646 | 0.633 | 0.658 | Accuracy |

Gradient boosting has the highest accuracy in three of the five accuracy evaluations (Folktables, MEPS, Student) and logistic regression in the other two (COMPAS, FairGround). Gradient boosting has the highest AUC in all three AUC evaluations. Random forest is not evaluated under the intervention for the AUC domains; the separate baseline scripts, which use their own preprocessing and samples, give it an AUC of 0.854 for Law School, 0.920 for Agricultural and 0.699 for Lending Club, below gradient boosting in the same scripts. Accuracy and AUC are not compared with each other.

---

## Table 2, Demographic Parity Difference, Baseline → After the DP Constraint

| Evaluation | LR | RF | GB |
|---|---|---|---|
| COMPAS | 0.545→0.714 | 0.568→0.714 | 0.857→0.571 |
| Folktables | 0.301→0.348 | 0.290→0.193 | 0.302→0.373 |
| Law School | 0.329→0.011 | n/e | 0.351→0.030 |
| Lending Club | 0.018→0.019 | n/e | 0.024→0.018 |
| Agricultural | 0.005→0.016 | n/e | 0.009→0.031 |
| FairGround (law_school_lequy) | 0.329→0.010 | 0.336→0.012 | 0.342→0.014 |
| MEPS | 0.069→0.013 | 0.089→0.253 | 0.092→0.013 |
| Student (Math) | 0.212→0.010 | 0.235→0.363 | 0.237→0.215 |

## Table 3, Equalized Odds Difference, Baseline → After the EO Constraint

| Evaluation | LR | RF | GB |
|---|---|---|---|
| COMPAS | 0.701→0.654 | 0.686→0.731 | 1.000→0.659 |
| Folktables | 0.728→0.314 | 0.732→0.836 | 0.767→0.417 |
| Law School | 0.543→0.060 | n/e | 0.528→0.007 |
| Lending Club | 0.038→0.047 | n/e | 0.053→0.049 |
| Agricultural | 0.005→0.047 | n/e | 0.073→0.177 |
| FairGround (law_school_lequy) | 0.543→0.061 | 0.524→0.472 | 0.518→0.016 |
| MEPS | 0.034→0.030 | 0.053→0.071 | 0.056→0.039 |
| Student (Math) | 0.204→0.188 | 0.263→0.180 | 0.314→0.114 |

**What Tables 2 and 3 show**
- The DP constraint improved DPD in 9 of the 14 model-domain pairs with baseline DPD above 0.2. The 5 exceptions are COMPAS LR, COMPAS RF, Folktables LR, Folktables GB, Student (Math) RF.
- It worsened DPD in 3 of the 4 pairs with baseline DPD below 0.05 (both Agricultural models and Lending Club's logistic regression).
- Five of the high-disparity pairs use law_school_lequy twice, once through Law School and once through FairGround. Counting that data once gives 6 of 11 or 7 of 12.
- MEPS is the only evaluation between 0.05 and 0.2. Logistic regression and gradient boosting improve under DP, and random forest worsens.
- The COMPAS and Folktables values are set by race groups with fewer than 30 test records, and the random forest worsening in Student and MEPS comes from thresholds fit on the training split. Tables 6 and 7 show both checks.
- Under EO, Law School's gradient boosting improves most (98.7%), all three Student models improve, and COMPAS, Folktables and MEPS each improve under logistic regression and gradient boosting and worsen under random forest.

---

## Table 4, Accuracy Cost of the DP Constraint (baseline accuracy minus constrained accuracy)

| Evaluation | LR | RF | GB |
|---|---|---|---|
| COMPAS | +0.029 | +0.035 | -0.001 |
| Folktables | +0.023 | +0.020 | +0.016 |
| FairGround (law_school_lequy) | +0.148 | +0.010 | +0.159 |
| MEPS | +0.071 | +0.133 | +0.075 |
| Student (Math) | +0.026 | +0.076 | +0.076 |

Negative means accuracy rose. Values are computed from the three-decimal numbers in RESULTS, so a script's own printed cost can differ by 0.001. Law School, Lending Club and Agricultural are omitted because their intervention scripts report AUC. FairGround's logistic regression and gradient boosting pay the most (0.148 and 0.159) while its random forest pays 0.010 for a similar DPD reduction. These costs also include the change from the default threshold to a balanced-accuracy objective (Decision 23).

---

## Table 5, Disparate Impact Ratio, Gradient Boosting

| Evaluation | Ratio | Outcome | Baseline | After | Reading |
|---|---|---|---|---|---|
| Law School | Minority / majority, predicted bar passage | Favorable | 0.643 | 0.957 | Moves above the 0.8 convention |
| Lending Club | Lowest / highest income quartile, predicted default | Adverse | 2.778 | 0.973 | Reaches parity because predicted default rises to about 40% in every quartile |
| Agricultural | Partnership / corporation, predicted default | Adverse | 0.653 | 1.042 | Ends just past parity as every business type's predicted default rate rises to 16 to 19% |
| Folktables | Each race group / White, predicted income above $50K, EO constraint | Favorable | See note | See note | Black 0.72→0.83, multiracial 0.70→0.92, American Indian 0.61→1.00, Pacific Islander 0.72→1.03, Other 0.42→just under 0.8 |

COMPAS, FairGround and MEPS compute no fixed-pair ratio. Student computes a female-to-male ratio for the baseline only (0.487 in Math, 2.084 in Portuguese). The 0.8 line is a research convention here, not a legal test for any of these domains, and for an adverse outcome a ratio above 1.0 is the harmful direction (Decision 21).

---

## Table 6, Group-Size Check (src/group_size_check.py)

DPD after the DP constraint, measured over all race groups and over groups with at least 30 test records.

| Evaluation | Model | All groups | Groups n≥30 |
|---|---|---|---|
| COMPAS | LR | 0.545→0.714 | 0.385→0.187 |
| COMPAS | RF | 0.568→0.714 | 0.202→0.163 |
| COMPAS | GB | 0.857→0.571 | 0.361→0.200 |
| Folktables | LR | 0.301→0.348 | 0.301→0.114 |
| Folktables | RF | 0.290→0.193 | 0.276→0.177 |
| Folktables | GB | 0.302→0.373 | 0.302→0.078 |

COMPAS's test split has 7 Asian defendants and 1 Native American defendant; Folktables' has 5 Alaska Native respondents and 25 in the combined American Indian and Alaska Native category. Measured on the larger groups, all six models improve, and the high-disparity count becomes 13 of 14.

## Table 7, Threshold-Fitting Check (src/threshold_holdout_check.py)

DPD under the DP constraint when thresholds are chosen on the training split (the design the intervention scripts use) and on a held-out quarter of it.

| Evaluation | Model | Training-split thresholds | Held-out thresholds |
|---|---|---|---|
| Student (Math) | LR | 0.212→0.010 | 0.238→0.186 |
| Student (Math) | RF | 0.235→0.363 | 0.235→0.007 |
| Student (Math) | GB | 0.237→0.215 | 0.161→0.159 |
| MEPS | LR | 0.069→0.013 | 0.068→0.031 |
| MEPS | RF | 0.089→0.253 | 0.086→0.037 |
| MEPS | GB | 0.092→0.013 | 0.092→0.054 |

Random forest fits its training records almost perfectly (training accuracy 1.000 in Student and 0.996 in MEPS), so thresholds chosen on them transfer poorly. With held-out thresholds its worsening reverses, while logistic regression and gradient boosting improve less in all four cases. The baselines in the held-out column differ slightly because those models train on 75% of the training split.
