# FAPE — Fairness-Aware Predictive Ensemble

## The Problem I Kept Running Into

Eight years building ML systems in financial services, healthcare, and workforce analytics — and the same failure mode repeated across every deployment. A model ships. Aggregate metrics look clean. Stakeholders sign off. Six months later someone notices the error rate for one demographic group is twice what it is for another. Not because anyone was careless. Because nobody was measuring the right thing, and the tools available weren't built to catch it in production.

What surprised me when I started pulling on this thread in late 2025 wasn't that the problem existed — it's documented everywhere. It's that every existing fairness tool is designed for research environments. AIF360, Fairlearn, What-If Tool — all of them evaluate on one or two datasets, produce a static report, and stop there. None of them address what happens six months after deployment when the demographic composition of users shifts, the model gets retrained, or a third-party vendor swaps the underlying algorithm. The fairness guarantee you validated at launch doesn't automatically hold in production. Nobody has built the infrastructure to check.

FAPE is my attempt to build that infrastructure.

---

## Research Question

**Broad motivation:** How can ensemble ML models reduce demographic bias while maintaining predictive accuracy?

**This paper specifically asks:** Do post-processing fairness constraints generalize across heterogeneous high-stakes deployment domains — criminal justice, healthcare, education, financial services — at acceptable accuracy cost?

This is a **Causal** research question. We are testing whether applying post-processing constraints *causes* measurable, statistically significant bias reduction across heterogeneous deployment domains simultaneously — not just whether bias reduction is possible in a single controlled setting.

---

## What Existing Systems Get Wrong

Three failure modes that the fairness literature treats as solved but aren't in production:

**Failure 1 — Single-domain evaluation.** Every major fairness paper validates on COMPAS or Adult Income. Nobody has tested whether interventions that work in criminal justice also work in healthcare, financial services, and education simultaneously. A recent AAAI 2026 paper showed that equalizing base rates appears to achieve fairness parity using traditional measures but produces ~10% disparity when measured correctly — exactly the kind of finding that disappears when you only look at aggregate metrics in one domain.

**Failure 2 — Static auditing.** AIF360 and Fairlearn produce point-in-time fairness snapshots. Sculley et al. (2015) documented that production ML systems degrade silently over time. A January 2026 paper on fairness auditing under model updates confirmed that real-world model changes can fundamentally alter fairness properties. No existing framework monitors this continuously. FAPE's Stage 4 does.

**Failure 3 — Single-metric optimization.** Chouldechova (2017) proved mathematically that satisfying equalized odds and calibration simultaneously is impossible when base rates differ across groups. Papers that optimize for one metric and report it as evidence of fairness are measuring the wrong thing. FAPE reports all four major fairness metrics simultaneously — making the tradeoffs visible rather than hiding them.

---

## Pipeline Architecture

**Stage 1 — Data Preprocessing:**
- Demographic attribute extraction and validation
- Label bias detection — identifying when outcome labels encode historical discrimination
- Feature engineering with fairness-aware feature selection
- Data quality audit across all 7 domains before any model training

**Stage 2 — Ensemble Classification:**
- XGBoost baseline — industry standard for tabular classification
- Default hyperparameters across all domains — isolates fairness intervention as the variable
- Unconstrained model establishes the accuracy-fairness tradeoff baseline
- Per-domain models with identical architecture — cross-domain comparison stays clean

**Stage 3 — Fairness Auditing:**
- Demographic parity difference, equalized odds difference, disparate impact ratio, individual fairness score — all four simultaneously
- Fairlearn ThresholdOptimizer applies post-processing constraints without retraining
- Cross-domain fairness metric comparison — the central empirical contribution
- Regulatory mapping: each metric tied to its domain-specific legal standard

**Stage 4 — Deployment Monitoring:**
- CUSUM-based drift detection calibrated specifically for fairness metrics
- Alerts when fairness constraints drift beyond threshold post-deployment
- Model versioning — tracks fairness across model updates, not just at launch
- The stage that existing frameworks skip entirely

---

## What Makes This Different

Most fairness papers ask: "Can we reduce bias in this dataset?" FAPE asks: "Does the same bias-reduction intervention generalize across fundamentally different deployment contexts — and does it hold after the model ships?"

The distinction matters because every major institution deploying ML for consequential decisions operates across multiple domains simultaneously. A bank uses ML for credit scoring, fraud detection, and hiring. A hospital uses ML for triage, diagnosis, and resource allocation. The research community has given them domain-specific tools. Nobody has given them a cross-domain production auditing framework.

The field is actively contested on two questions FAPE addresses directly. First: whether post-processing constraints generalize across domains or whether each deployment requires bespoke solutions — no paper has empirically tested this at scale across 5 distinct domains. Second: whether continuous monitoring can detect fairness drift before harm accumulates — recent theoretical work on auditing under model updates confirms this is open. FAPE provides the empirical answer to both.

---

## Datasets

| Dataset | Year | Records | Domain | Access |
|---------|------|---------|--------|--------|
| COMPAS Recidivism | 2013-2014 | 6,172 | Criminal Justice | Public |
| Folktables ACS | 2021 | 1,589,032 | Socioeconomic | Public |
| FairGround Corpus | 2025 | 1,964,010 | Multi-domain | Public |
| MIMIC-III Clinical | 2001-2012 | — | Healthcare | PhysioNet |
| Student Performance | 2008 | 1,044 | Education | UCI ML |
| Law School Admissions | 1991-2000 | 18,692 | Education/Legal | Public |
| Lending Club | 2007-2018 | 1,348,099 | Financial | Kaggle |
| USDA NASS Census | 2022 | 7,334 | Agriculture (baseline) | Public |
| SBA 7(a) NAICS-11 | FY1991-2024 | 15,845 | Agriculture | Public |
| LSMS-ISA Nigeria | 2018-2019 | 30,312 | Agriculture | World Bank |
| MIMIC-III Clinical | 2001-2012 | — | Healthcare | PhysioNet |

**Verified: 4,980,540 records (all confirmed except MIMIC-III) | MIMIC-III pending PhysioNet approval**

Dataset notes:
- COMPAS: 6,172 records verified — ProPublica Broward County Florida 2013-2014
- Folktables ACS: 1,589,032 records verified — replaces Adult Income per Ding et al. (2021)
- FairGround Corpus: 1,964,010 records verified — 44 fairness-annotated datasets (Fabris et al. 2025)
- Student Performance: 1,044 records verified — math and Portuguese variants combined
- Law School Admissions: 18,692 records verified — race and sex, bar passage outcome
- Lending Club: 1,348,099 records verified — socioeconomic proxy fairness at production scale
- USDA NASS Census 2022: 7,334 aggregate rows — racial disparity baseline, not individual-level training data. CIPSEA (7 U.S.C. §2204) prohibits public release of individual farm records.
- SBA 7(a) NAICS-11: 15,845 individual agricultural business loans FY1991-2024 — binary default outcome, geographic proxy attributes
- LSMS-ISA Nigeria Wave 4: 30,312 individual farm households — sex and education as sensitive attributes, food security outcome. Only large-scale publicly downloadable individual-level agricultural dataset with demographic attributes.
- MIMIC-III requires PhysioNet credentialed registration — access pending

---

## Evaluation Metrics

- **Accuracy:** Precision, Recall, F1 — per domain and per demographic group
- **Fairness:** Demographic parity difference, equalized odds difference, disparate impact ratio, individual fairness score — all four reported simultaneously
- **Regulatory mapping:** Each metric mapped to its domain-specific legal standard (EEOC 80% rule, ProPublica equalized odds standard, ECOA disparate impact)
- **Drift detection:** CUSUM statistics for fairness metric drift post-deployment
- **Statistical significance:** Bootstrap confidence intervals (n=1000) across all fairness metrics
- **Cross-domain comparison:** Paired t-tests for between-domain fairness improvement comparisons

---

## Tech Stack

Python 3.10+, XGBoost, scikit-learn, Fairlearn, AIF360, folktables, scipy, statsmodels, pandas, numpy, matplotlib, seaborn, jupyter

Full dependency list: `requirements.txt`

---

## Research Timeline

- November 2025: Research conception — observed systematic demographic disparities in production ML deployments across financial services and healthcare engagements
- December 2025: Literature review, gap identification, 7 datasets identified across 5 domains
- January 2026: Methodology design, 4-stage framework developed, fairness metric selection
- February 2026: Research question formalized, design rationale documented
- March 2026: GitHub repository created, Phase 1 literature analysis completed
- April 2026: Data pipeline implementation — COMPAS and Folktables ACS loaders committed
- May 2026: Full dataset pipeline complete, EDA notebooks, baseline model
- June 2026: Target submission — JASIST + arXiv simultaneously

---

## Status

🔬 Research in progress
Target venue: JASIST 2026 (arXiv preprint uploaded on submission day)

---

## Paper

"FAPE: Fairness-Aware Predictive Ensemble for Cross-Domain Bias Auditing in Production ML Systems" — Under development

---

## References

- Angwin et al. (2016) — Machine Bias, ProPublica
- Chouldechova (2017) — Fair Prediction with Disparate Impact, Big Data
- Hardt, Price & Srebro (2016) — Equality of Opportunity in Supervised Learning, NeurIPS
- Obermeyer et al. (2019) — Dissecting Racial Bias in an Algorithm Used to Manage Health, Science
- Sculley et al. (2015) — Hidden Technical Debt in Machine Learning Systems, NeurIPS
- Mitchell et al. (2019) — Model Cards for Model Reporting, FAccT
- Ding et al. (2021) — Retiring Adult: New Datasets for Fair Machine Learning, NeurIPS
- Fabris et al. (2025) — FairGround Corpus: Bias Begins with Data, arXiv
- Johnson et al. (2016) — MIMIC-III Clinical Database, Scientific Data
- Wightman (1998) — LSAC National Longitudinal Bar Passage Study
- Cortez & Silva (2008) — Student Performance Dataset, UCI ML Repository
- Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System, KDD
- Weerts et al. (2023) — Fairlearn: Assessing and Improving Fairness of AI Systems
- Bellamy et al. (2019) — AI Fairness 360: An Extensible Toolkit, IBM Journal
- Sariola et al. (2026) — The Illusion of Fairness: Auditing Fairness Interventions in Algorithmic Hiring, AAAI
- Ajarra et al. (2026) — Auditing Fairness under Model Updates, arXiv 2601.05909
