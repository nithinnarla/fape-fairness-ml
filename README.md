# FAPE — Fairness-Aware Predictive Ensemble

## Overview
FAPE investigates algorithmic bias and fairness in ML classification systems used in high-stakes decision-making contexts.

## Research Question
How can ensemble ML models be designed to reduce demographic bias while maintaining predictive accuracy in criminal justice risk assessment?

## Dataset
COMPAS Recidivism Dataset
- ProPublica, Broward County Florida
- 7,000+ criminal defendant records
- Features: age, race, sex, priors, charge degree, recidivism outcome

## Methodology
- XGBoost classifier baseline
- Fairness metrics: demographic parity, equalized odds, disparate impact ratio
- Ablation experiments across demographic subgroups
- Comparison with COMPAS scores

## Tech Stack
Python, XGBoost, scikit-learn, pandas, numpy, matplotlib, seaborn, Fairlearn

## Research Timeline
- December 2025: Research conception, literature review, COMPAS dataset identification
- January 2026: Methodology design, fairness framework development
- February 2026: Paper outlining, research question refinement
- March 2026: GitHub repository created, active code development begins
- April 2026: Pipeline implementation, fairness metrics testing
- May 2026: Results analysis, paper writing
- June 2026: Target submission to JASIST

## Status
🔬 Research in progress
Target venue: JASIST 2026

## Paper
"FAPE: Fairness-Aware Predictive Ensemble for Bias Detection in Recidivism Risk Assessment" — Under development
