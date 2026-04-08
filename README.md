# FAPE — Fairness-Aware Predictive Ensemble

## Overview
FAPE investigates algorithmic bias and fairness in ML classification systems used in high-stakes decision-making contexts. We evaluate fairness across 7 domains and 5M+ records to ensure comprehensive generalizability of our fairness framework across criminal justice, healthcare, education, financial, and socioeconomic systems.

## Research Question
**Broad motivation:** How can ensemble ML models reduce demographic bias while maintaining predictive accuracy?

**This paper specifically asks:** Do post-processing fairness constraints generalize across heterogeneous high-stakes deployment domains — criminal justice, healthcare, education, financial services — at acceptable accuracy cost?

## Pipeline Architecture
- Stage 1 — Data Preprocessing: Feature engineering and demographic attribute extraction
- Stage 2 — Ensemble Classification: XGBoost with fairness constraints
- Stage 3 — Fairness Auditing: Multi-metric bias evaluation across demographic subgroups
- Stage 4 — Reporting: Cross-domain generalizability analysis and recommendations

## Datasets

| Dataset | Year | Records | Domain | Access |
|---------|------|---------|--------|--------|
| COMPAS Recidivism | 2013-2014 | — | Criminal Justice | Public |
| Folktables ACS | 2021-2023 | — | Socioeconomic | Public |
| FairGround Corpus | 2025 | — | Multi-domain | Public |
| MIMIC-III Clinical | 2001-2012 | — | Healthcare | PhysioNet |
| Student Performance | 2014-2022 | — | Education | UCI ML |
| Law School Admissions | 1991-2023 | — | Education/Legal | Public |
| Lending Club | 2007-2020 | — | Financial | Kaggle |

**Dataset pipeline initialized — downloading and verifying datasets sequentially**

Dataset notes:
- COMPAS is the field's primary validation benchmark — ProPublica's documented racial disparities provide known ground truth for framework validation
- Folktables ACS replaces Adult Income — Ding et al. (2021) demonstrated Adult Income has serious methodological flaws
- FairGround Corpus (Fabris et al. 2025) addresses the reproducibility crisis in fairness benchmarking
- MIMIC-III requires PhysioNet credentialed registration
- Student Performance (649 records) is deliberately included — FAPE must work at small scale as well as large scale
- Lending Club at 2.2M+ records tests framework performance at production scale

## Evaluation Metrics
- Accuracy, Precision, Recall, F1
- Demographic parity difference
- Equalized odds difference
- Disparate impact ratio
- Individual fairness score
- Statistical significance testing across all demographic subgroups

## Methodology
- XGBoost ensemble classifier baseline
- Fairness metrics: demographic parity, equalized odds, disparate impact ratio, individual fairness
- Ablation experiments across demographic subgroups (race, gender, age)
- Cross-domain evaluation for generalizability
- Comparison with existing bias scores per domain
- Statistical significance testing across all experiments

## Tech Stack
Python, XGBoost, scikit-learn, pandas, numpy, matplotlib, seaborn, Fairlearn, AIF360, folktables

## Research Timeline
- December 2025: Research conception, literature review, 
  dataset identification
- January 2026: Methodology design, fairness framework 
  development
- February 2026: Paper outlining, research question 
  refinement
- March 2026: GitHub repository created, active code 
  development begins
- April 2026: Pipeline implementation, fairness metrics 
  testing
- May 2026: Cross-domain evaluation, results analysis, 
  paper writing
- June 2026: Target submission to JASIST

## Status
🔬 Research in progress
Target venue: JASIST 2026

## Paper
"FAPE: Fairness-Aware Predictive Ensemble for Bias 
Detection in High-Stakes ML Systems" — Under development

## References
- Angwin et al. (2016) — Machine Bias, ProPublica
- Ding et al. (2021) — Retiring Adult / Folktables:
  New Datasets for Fair Machine Learning
- Johnson et al. (2016) — MIMIC-III Clinical Database, 
  PhysioNet
- Fabris et al. (2025) — FairGround Corpus: Bias 
  Begins with Data
- Wightman (1998) — LSAC National Longitudinal Bar 
  Passage Study
- Cortez & Silva (2008) — Student Performance 
  Dataset, UCI ML Repository
- Lending Club (2007-2020) — Loan Data, Kaggle  
- Chen & Guestrin (2016) — XGBoost
- Weerts et al. (2023) — Fairlearn Toolkit
- Bellamy et al. (2019) — AI Fairness 360 (AIF360)
