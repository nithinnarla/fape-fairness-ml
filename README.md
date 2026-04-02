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

## Status
🔬 Research in progress
Target venue: JASIST 2026

## Paper
"FAPE: Fairness-Aware Predictive Ensemble for Bias Detection in Recidivism Risk Assessment" — Under development
