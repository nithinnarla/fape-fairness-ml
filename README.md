# FAPE, Fairness Auditing for Production Environments

## The Problem I Kept Running Into

Eight years building ML systems in financial services, healthcare, and workforce analytics, and the same failure mode repeated across every deployment. A model ships. Aggregate metrics look clean. Stakeholders sign off. Six months later someone notices the error rate for one demographic group is twice what it is for another. Not because anyone was careless. Because nobody was measuring the right thing, and the tools available weren't built to catch it in production.

What surprised me when I started pulling on this thread in late 2025 wasn't that the problem existed, it's documented everywhere. It's that the open-source fairness tools most research builds on are designed for research environments. AIF360, Fairlearn and the What-If Tool analyze a model at one point in time and stop there. None of them address what happens six months after deployment when the demographic composition of users shifts, the model gets retrained, or a third-party vendor swaps the underlying algorithm. The fairness guarantee you validated at launch doesn't automatically hold in production. Cloud platforms such as Amazon SageMaker Clarify now check bias on live data, but the open research tooling has nothing like it.

FAPE is my attempt to build that infrastructure.

---

## Research Question

**Broad motivation:** How can fairness interventions reduce demographic bias in production ML systems while maintaining predictive accuracy?

**This paper specifically asks:** Do post-processing fairness constraints generalize across heterogeneous high-stakes deployment domains, criminal justice, healthcare, education, financial services, at acceptable accuracy cost?

This paper evaluates the constraint against each of the three classifiers separately, never against a combined ensemble. Whether a post-processing constraint behaves differently on an ensemble than on its components is left to future work.

This is a **Causal** research question. We are testing whether applying post-processing constraints *causes* measurable bias reduction across heterogeneous deployment domains at once, not just whether bias reduction is possible in a single controlled setting. Each before-and-after comparison also changes the decision objective, from scikit-learn's default threshold to the balanced accuracy ThresholdOptimizer maximizes, so the paper reads the constraint's effect together with that change rather than in isolation (Section 6.4).

---

## What Existing Systems Get Wrong

Three failure modes that the fairness literature treats as solved but aren't in production:

**Failure 1, Single-domain evaluation.** Most fairness papers validate on COMPAS or Adult Income. Comparative studies such as Chen et al. (2023) test mitigation methods across several of the standard benchmark datasets, but those cover a narrow set of domains, and I found none that runs one intervention across criminal justice, healthcare, lending, legal admissions and education together. A recent AAAI 2026 paper found that equalizing base rates in hiring data appeared to reach parity on traditional measures but left about 10% disparity when measured with audit-study data.

**Failure 2, Static auditing.** AIF360 and Fairlearn produce point-in-time fairness snapshots. Sculley et al. (2015) documented that production ML systems degrade silently over time. A January 2026 paper worked out, in theory, how to audit group fairness when a model owner keeps updating the model. The open-source toolkits have no continuous monitoring, and cloud monitors such as SageMaker Clarify judge each window of live data on its own. FAPE's Stage 4 is an open proof of concept for sequential monitoring, tested on a simulated shift.

**Failure 3, Single-metric optimization.** Chouldechova (2017) proved mathematically that satisfying equalized odds and calibration simultaneously is impossible when base rates differ across groups. Papers that optimize for one metric and report it as evidence of fairness are measuring the wrong thing. FAPE reports several fairness metrics side by side, making the tradeoffs visible.

---

## Pipeline Architecture

**Stage 1, Data Preprocessing:**
- Demographic attribute extraction and validation
- Outcome base rates compared across demographic groups during EDA
- Per-domain feature preprocessing
- Data quality audit across every domain before any model training

**Stage 2, Baseline Classification:**
- Three classifiers trained separately per domain: logistic regression, random forest, gradient boosting (scikit-learn)
- Default hyperparameters across all domains, isolates fairness intervention as the variable
- Unconstrained model establishes the accuracy-fairness tradeoff baseline
- Per-domain models with identical architecture, cross-domain comparison stays clean

**Stage 3, Fairness Auditing:**
- Demographic parity difference and equalized odds difference everywhere; disparate impact ratio and accuracy cost where the domain pipeline computes them
- Fairlearn ThresholdOptimizer applies post-processing constraints without retraining
- Cross-domain fairness metric comparison, the central empirical contribution
- Regulatory context noted per domain: ECOA for the two lending domains, the 0.8 disparate impact ratio as a research convention elsewhere

**Stage 4, Deployment Monitoring:**
- CUSUM-based drift detection applied to demographic parity difference
- Alerts when fairness constraints drift beyond threshold post-deployment
- Model versioning, tracks fairness across model updates, not just at launch
- The stage the open-source fairness toolkits do not provide

---

## What Makes This Different

Most fairness papers ask: "Can we reduce bias in this dataset?" FAPE asks: "Does the same bias-reduction intervention generalize across fundamentally different deployment contexts, and does it hold after the model ships?"

The distinction matters because every major institution deploying ML for consequential decisions operates across multiple domains simultaneously. A bank uses ML for credit scoring, fraud detection, and hiring. A hospital uses ML for triage, diagnosis, and resource allocation. The research community has given them domain-specific tools and point-in-time audits, and left the comparison across domains to them.

The field is actively contested on two questions FAPE addresses. First: whether post-processing constraints generalize across domains or whether each deployment requires bespoke solutions, I found no study that tests this across this many distinct domains. Second: whether continuous monitoring can detect fairness drift before harm accumulates, where recent work on auditing under model updates is still theoretical. FAPE gives cross-domain evidence on the first and a simulated proof of concept for the second.

---

## Datasets

| Dataset | Year | Records | Domain | Access |
|---------|------|---------|--------|--------|
| COMPAS Recidivism | 2013-2014 | 6,172 | Criminal Justice | Public |
| Folktables ACS | 2021 | 1,589,032 | Socioeconomic | Public |
| FairGround Corpus | 2025 | 1,955,063 | Multi-domain | Public |
| Student Performance | 2008 | 1,044 | Education | UCI ML |
| Law School Admissions | 1991-2000 | 18,692 | Education/Legal | Public (via FairGround) |
| Lending Club | 2007-2018 | 1,348,099 | Financial | Kaggle |
| USDA NASS Census | 2022 | 7,334 | Agriculture (baseline) | Public |
| SBA 7(a) NAICS-11 | FY1991-2026 | 15,845 | Agriculture | Public |
| LSMS-ISA Nigeria | 2018-2019 | 30,312 | Agriculture | World Bank |
| MEPS Panel 19 FY2015 | 2015 | 15,830 | Healthcare | Public (via FairGround) |
| MIMIC-III Clinical | 2001-2012 | not obtained | Healthcare (dropped) | PhysioNet |

**Verified: 4,952,901 distinct records.** Law School Admissions (law_school_lequy) and MEPS Panel 19 are both sub-datasets of the FairGround corpus, so their 18,692 and 15,830 records are already inside FairGround's 1,955,063 and are counted once. Two FairGround datasets, law_school_tensorflow and stop_question_and_frisk_data, fail to load in the pinned environment because fairml-datasets 0.2.5 calls np.bool, which numpy 1.26 does not have; stop_question_and_frisk_data adds 8,947 records (4,961,848 in total) when a cached copy from an earlier download is present. MIMIC-III was never obtained and is not part of the study. USDA NASS is used descriptively in EDA only and LSMS-ISA Nigeria was excluded from modeling (Decision 11); both are counted here as collected.

Dataset notes:
- COMPAS: 6,172 records verified, ProPublica Broward County Florida 2013-2014
- Folktables ACS: 1,589,032 records verified, replaces Adult Income per Ding et al. (2021)
- FairGround Corpus: 1,955,063 records in the 36 datasets that load with fairml-datasets 0.2.5, out of the 38 it lists without the large-dataset option (Simson et al. 2025)
- Student Performance: 1,044 records verified, math and Portuguese variants combined
- Law School Admissions: 18,692 records verified, race and sex, bar passage outcome
- Lending Club: 1,348,099 records verified, socioeconomic proxy fairness at production scale
- USDA NASS Census 2022: 7,334 aggregate rows, racial disparity baseline, not individual-level training data. CIPSEA (7 U.S.C. §2204) prohibits public release of individual farm records.
- SBA 7(a) NAICS-11: 15,845 individual agricultural business loans FY1991-2026, binary default outcome, geographic proxy attributes
- LSMS-ISA Nigeria Wave 4: 30,312 individual farm households, sex and education as sensitive attributes, food security outcome. Only large-scale publicly downloadable individual-level agricultural dataset with demographic attributes. Evaluated during design and excluded from the ML pipeline (Decision 11).
- MEPS Panel 19 FY2015: 15,830 records, race as the sensitive attribute, drawn from the FairGround corpus. This is the healthcare evaluation the study ran.
- MIMIC-III required PhysioNet credentialed registration, which did not come through. Healthcare is covered by MEPS instead, so no result in this repository depends on MIMIC-III.

---

## Evaluation Metrics

- **Performance:** accuracy, or AUC for Law School, Lending Club and Agricultural, and F1 for every model and domain; the COMPAS and Folktables baselines also report precision and recall
- **Fairness:** Demographic parity difference and equalized odds difference for every evaluation; disparate impact ratio and accuracy cost where computable
- **Regulatory context:** ECOA for the two lending domains; elsewhere the 0.8 disparate impact ratio from the EEOC four-fifths rule is a research convention, not a compliance test. Both lending domains predict default, so a ratio above 1.0 is the adverse direction there
- **Drift detection:** CUSUM statistics for fairness metric drift post-deployment
- **Cross-domain comparison:** Per-model, per-domain before-and-after values compared directly; no inferential test is applied, since each domain yields one measurement per model

---

## Tech Stack

Python 3.11.9, scikit-learn 1.9.1, Fairlearn 0.13.0, fairml-datasets 0.2.5, folktables 0.0.12, pandas 2.2.2, numpy 1.26.4, matplotlib, seaborn, jupyter.

scikit-learn is pinned at 1.9.1 because fairml-datasets 0.2.5 requires 1.5.2 or newer, and every reported value was produced and verified under 1.9.1. aif360 and scipy arrive transitively through fairml-datasets, so neither is pinned here; no code in this repository imports either one.

Full dependency list: `requirements.txt`

---

## Checking the Numbers

Every script that produces a reported value has a notebook in `notebooks/` holding its printed output, and the documents are checked against those outputs:

```bash
python src/make_results_table.py    # rebuilds docs/results_table.md and docs/cross_domain_results_table.md
python src/check_consistency.py     # checks the paper, README, outline and tables against the notebooks
```

The check trains nothing and finishes in a few seconds. It fails if a results table no longer matches what the scripts printed, if a number in the paper cannot be traced to the evaluation it describes, if a citation and the reference list disagree, or if the paper goes over the JASIST word limits. Rerun a script's notebook after changing the script; the check also flags a notebook older than its script.

---

## Research Timeline

- November 2025: Research conception, observed systematic demographic disparities in production ML deployments across financial services and healthcare engagements
- December 2025: Literature review, gap identification, criminal justice, socioeconomic, healthcare, education, and financial domains scoped
- January 2026: Methodology design, 4-stage framework developed, fairness metric selection
- February 2026: Research question formalized, design rationale documented
- March 2026: GitHub repository created, active development begins
- April 2026: Data pipeline implementation, COMPAS and Folktables ACS loaders committed, Phase 1 literature documentation in progress
- May 2026: Full dataset pipeline complete across 9 verified datasets, criminal justice, socioeconomic, education, financial, and agricultural domains. Phase 1 literature documentation complete
- June 2026: Stage 1 complete, EDA across all 7 domains (COMPAS, Folktables, Law School, Lending Club, Agricultural, FairGround, Student); baseline models (LR, RF, GB) trained and evaluated; 83 EDA figures committed
- July 2026: Stage 2 complete, ThresholdOptimizer DP/EO constraints across all 7 domains; cross-domain comparison complete; Law School strongest improvement (EO +98.7%); Agricultural counterproductive when near-fair (DPD -244.4%); core finding: effectiveness tracks baseline DPD, with documented exceptions; 68 figures committed (56 domain + 6 aggregation + 6 cross-domain)
- July 2026: Stage 3 complete, DIR metric added where each domain's pipeline supports it (before-and-after for Law School, Lending Club, Agricultural; baseline only for Folktables); paper outline committed
- July 2026: Stage 4 complete, CUSUM-based fairness drift detection; 9 drift figures; Law School + FairGround + Student earliest alerts
- August 2026: Paper writing begins
- September 2026: Full draft complete; healthcare evaluation (MEPS Panel 19) added, bringing the study to eight evaluations; all metric values re-verified against a clean install of the pinned environment; scikit-learn pin corrected from 1.4.2, which conflicted with fairml-datasets and could not be installed; Stage 4 monitor corrected to start at deployment, after the original version was found to alert on pre-deployment baseline values; COMPAS and Folktables metrics found to be set by test groups of fewer than 30 records; MEPS rerun on the 41 features FairGround documents, after the raw file's visit counts turned out to define its utilization label, and Folktables rerun without POVPIP, which is built from family income; random forest's worsening under the constraint traced to thresholds fit on its training data; targeting Sep 29 submission to JASIST

---

## Status

Stages 1-4 complete. Paper drafted and internally verified; targeting JASIST Sep 29 2026. The study now covers eight domain evaluations across seven independent data sources, after a healthcare evaluation (MEPS Panel 19) was added in September; entries dated before then describe the seven-domain study as it stood at the time, including drift results the September entry corrects.

**Stage 1 (complete):** EDA + baseline models across the original seven domains, 83 EDA figures. MEPS was added later and evaluated through the FairGround pipeline, so it has no separate EDA set.
**Stage 2 (complete):** ThresholdOptimizer fairness interventions across all eight evaluations, cross-domain comparison done, 68 figures committed from the original seven domains (56 domain + 6 aggregation + 6 cross-domain).
**Stage 3 (complete):** DIR where the domain pipeline supports it: a fixed-pair ratio before and after for Law School, Lending Club and Agricultural, and per-race ratios under the equalized odds constraint for Folktables; paper outline committed.
**Stage 4 (complete):** CUSUM monitoring from deployment under a synthetic shift; flags 9 constrained models already above DPD 0.1 when deployed and 5 that regress under the shift, all 5 from Law School and FairGround, which share the same underlying data; 9 figures committed.

Target venue: JASIST, submission Sep 29 2026

---

## Paper

"When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations", full draft complete, targeting JASIST Sep 29 2026

---

## References

- Angwin et al. (2016), Machine Bias, ProPublica
- Chouldechova (2017), Fair Prediction with Disparate Impact, Big Data
- Hardt, Price & Srebro (2016), Equality of Opportunity in Supervised Learning, NeurIPS
- Obermeyer et al. (2019), Dissecting Racial Bias in an Algorithm Used to Manage Health, Science
- Sculley et al. (2015), Hidden Technical Debt in Machine Learning Systems, NeurIPS
- Mitchell et al. (2019), Model Cards for Model Reporting, FAccT
- Ding et al. (2021), Retiring Adult: New Datasets for Fair Machine Learning, NeurIPS
- Simson et al. (2025), Bias Begins with Data: The FairGround Corpus, arXiv
- Johnson et al. (2016), MIMIC-III Clinical Database, Scientific Data
- Wightman (1998), LSAC National Longitudinal Bar Passage Study
- Cortez & Silva (2008), Student Performance Dataset, UCI ML Repository
- Weerts et al. (2023), Fairlearn: Assessing and Improving Fairness of AI Systems
- Bellamy et al. (2019), AI Fairness 360: An Extensible Toolkit, IBM Journal
- Sariola et al. (2026), The Illusion of Fairness: Auditing Fairness Interventions in Algorithmic Hiring, AAAI
- Ajarra & Basu (2026), Auditing Fairness under Model Updates, arXiv 2601.05909
- Friedler et al. (2019), A Comparative Study of Fairness-Enhancing Interventions in Machine Learning, FAT*
- Chen et al. (2023), A Comprehensive Empirical Study of Bias Mitigation Methods for Machine Learning Classifiers, ACM TOSEM
