# FAPE — Fairness-Aware Predictive Ensemble

## Overview
FAPE investigates algorithmic bias and fairness in ML classification systems used in high-stakes decision-making contexts. We evaluate fairness across 7 domains and 5M+ records to ensure comprehensive generalizability of our fairness framework across criminal justice, healthcare, education, financial, and socioeconomic systems.

## Research Question
How can ensemble ML models be designed to reduce demographic 
bias while maintaining predictive accuracy across multiple 
high-stakes decision-making domains?

## Pipeline Architecture
- Stage 1 — Data Preprocessing:
  Feature engineering and 
  demographic attribute extraction
- Stage 2 — Ensemble Classification:
  XGBoost with fairness constraints
- Stage 3 — Fairness Auditing:
  Multi-metric bias evaluation
  across demographic subgroups
- Stage 4 — Reporting:
  Cross-domain generalizability 
  analysis and recommendations

## Datasets

| Dataset | Year | Records | Domain | Access |
|---------|------|---------|--------|--------|
| COMPAS Recidivism | 2013-2014 | 7,000+ | Criminal Justice | Public |
| Folktables ACS | 2021-2023 | 3,000,000+ | Socioeconomic | Public |
| FairGround Corpus | 2025 | 50,000+ | Multi-domain | Public |
| MIMIC-III Clinical | 2001-2012 | 46,000+ | Healthcare | PhysioNet |
| Student Performance | 2014-2022 | 649 | Education | UCI ML |
| Law School Admissions | 1991-2023 | 20,000+ | Education/Legal | Public |
| Lending Club | 2007-2020 | 2,200,000+ | Financial | Kaggle |

**Total: 5.3M+ records across 7 high-stakes domains**

Dataset notes:
- Folktables ACS specifically designed to replace legacy 
  Adult Income dataset (Ding et al., 2021)
- FairGround Corpus (2025) addresses known limitations of 
  legacy fairness benchmarks
- MIMIC-III access via PhysioNet credentialed registration
- Cross-domain evaluation ensures FAPE generalizes beyond 
  single-domain studies

## Evaluation Metrics
- Accuracy, Precision, Recall, F1
- Demographic parity difference
- Equalized odds difference
- Disparate impact ratio
- Individual fairness score
- Statistical significance testing
  across all demographic subgroups

## Methodology
- XGBoost ensemble classifier baseline
- Fairness metrics: demographic parity, equalized odds, 
  disparate impact ratio, individual fairness
- Ablation experiments across demographic subgroups 
  (race, gender, age)
- Cross-domain evaluation for generalizability
- Comparison with existing bias scores per domain
- Statistical significance testing across all experiments

## Tech Stack
Python, XGBoost, scikit-learn, pandas, numpy, matplotlib, 
seaborn, Fairlearn, AIF360, folktables

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
- Ding et al. (2021) — RRetiring Adult / Folktables:
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
