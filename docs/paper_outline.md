# FAPE: Fairness Auditing for Production Environments

> **Record note, September 15 2026.** Revised to match docs/paper_draft.md after the September verification work (Decisions 20 to 27 in methodology_decisions.md). The July planning outline, which used the project's earlier name, seven domains, and results from before the MEPS and Folktables fixes, is in the git history. The draft is the authoritative text; this outline tracks its structure and main numbers.

## Paper Outline, JASIST Submission Target September 29 2026

Title: When Post-Processing Fairness Constraints Help and When They Harm: Evidence from Eight Cross-Domain Evaluations

---

## Abstract (200 words maximum)

- One-time, single-domain fairness audits miss both drift after deployment and variation across domains
- FAPE: a four-stage framework evaluating one post-processing intervention, Fairlearn's ThresholdOptimizer, across eight evaluations (criminal justice, income prediction, legal admissions, credit lending, agricultural lending, a benchmark corpus, healthcare, education)
- DPD and EOD everywhere, DIR and accuracy cost where computable
- Effectiveness tracks baseline disparity: improved in 9 of 14 high-disparity pairs (13 of 14 on groups of at least 30 test records; 6 of 11 to 7 of 12 counting the duplicated dataset once), worsened in 3 of 4 near-fair pairs
- A CUSUM monitor started at deployment, tested on a simulated shift, separates models that never met the 0.1 DPD convention from models that met it and later regressed

---

## 1. Introduction

### 1.1 The Production Fairness Gap
- Consequential ML systems are usually audited once, at launch
- Sculley et al. (2015): production ML degrades silently; Ajarra and Basu (2026): fairness auditing under model updates as its own problem
- AIF360 and Fairlearn produce point-in-time reports and are not designed for post-deployment checks
- Illustration: a credit model that passes at launch, is retrained, and drifts past the parity threshold while accuracy holds

### 1.2 The Cross-Domain Generalization Problem
- Most studies test one dataset; comparative studies reuse a small pool of benchmarks (Section 2.3)
- Open question: does an intervention that works in criminal justice behave the same way in agricultural lending or education?

### 1.3 Contributions
1. One post-processing intervention evaluated across eight evaluations from seven data sources
2. Several metrics per domain, so tradeoffs between criteria stay visible
3. CUSUM sequential monitoring after deployment, tested on a simulated shift
4. An effectiveness pattern with counts (9 of 14, 3 of 4), and each of the five high-disparity exceptions reverses under a group-size check or held-out thresholds

### 1.4 Paper Organization

---

## 2. Related Work

### 2.1 Fairness Interventions
- Pre-processing (Kamiran and Calders 2012), in-processing (Zhang et al. 2018), post-processing (Hardt et al. 2016)
- Post-processing chosen because it works on a model the auditor does not own

### 2.2 Fairness Metrics and Impossibility
- Individual fairness (Dwork et al. 2012); demographic parity and equalized odds (Hardt et al. 2016)
- Disparate impact ratio from the EEOC four-fifths rule, used here as a research convention
- Chouldechova (2017): calibration and equal error rates cannot all hold when base rates differ
- Sariola et al. (2026): equalizing base rates looked like parity on traditional measures but left about 10% disparity measured with audit-study data

### 2.3 Cross-Domain Fairness Evaluation
- Friedler et al. (2019): interventions compared across benchmark datasets, sensitive to the train-test split
- Chen et al. (2023): seventeen mitigation methods, equalized odds post-processing among them, on the five AIF360 datasets
- Simson et al. (2025): FairGround corpus
- FAPE: COMPAS and MEPS from that pool, Folktables in place of Adult, plus Law School, Lending Club, SBA agricultural loans and Student Performance

### 2.4 Production Fairness Monitoring
- Breck et al. (2017): a pre-release inclusion test, but no fairness test among the seven monitoring tests
- Open-source toolkits audit once; Amazon SageMaker Clarify monitors bias per window on live data
- FAPE Stage 4: an open, sequential alternative that accumulates small excesses across windows

---

## 3. Methodology

### 3.1 Framework Overview (Figure 1)
- Four stages: data preprocessing, baseline classification, fairness intervention, drift monitoring
- ECOA provisions for the two lending domains; the 0.8 ratio as a research convention elsewhere; Title VI, Title IX and ACA Section 1557 may apply to deployments, a question the study does not settle

### 3.2 Datasets and Domains (Table 1)

| Domain | Dataset | Sensitive attribute | Records | Regulatory context |
|---|---|---|---|---|
| Criminal justice | COMPAS | Race (6 groups) | 6,172 | 0.8 convention |
| Socioeconomic | Folktables ACS | Race | 1,589,032 (100,000 sample) | 0.8 convention |
| Legal admissions | Law School | Race | 18,692 | 0.8 convention |
| Credit lending | Lending Club | Income band | 1,348,099 filtered (100,000 sample) | ECOA, individual applicants |
| Agricultural lending | SBA 7(a) | Business type | 15,845 | ECOA, business credit |
| Benchmark corpus | FairGround law_school_lequy | Race | 18,692 | 0.8 convention |
| Healthcare | MEPS Panel 19 FY2015 | Race | 15,830 | 0.8 convention |
| Education | Student Performance | Sex | 395 Math, 649 Portuguese | 0.8 convention |

- Feature choices that keep label information out: MEPS uses FairGround's 41 documented features; Folktables drops POVPIP; Student drops the interim grades

### 3.3 Baseline Models
- Logistic regression, random forest, gradient boosting, scikit-learn defaults, no tuning
- Random forest carried through the intervention in five of eight evaluations; Law School, Lending Club and Agricultural use two models and report AUC

### 3.4 Fairness Intervention
- ThresholdOptimizer under demographic parity and equalized odds, fit on each training split, balanced-accuracy objective
- random_state passed to all 22 predict calls, so runs are identical

### 3.5 Evaluation Metrics
- DPD and EOD for all eight; DIR (fixed pairs) and accuracy cost where computable
- DIR direction depends on the outcome: bar passage is favorable, predicted default is adverse

### 3.6 Drift Detection
- 30 simulated observations per model: baseline, deployed constrained model, synthetic shift 60% back toward baseline
- CUSUM from deployment, reference 0.1 plus 0.01 slack, alert when accumulated excess passes 0.1; proof of concept only

---

## 4. Experimental Setup

### 4.1 Implementation
- Python 3.11.9, scikit-learn 1.9.1, Fairlearn 0.13.0, seed 42
- Law School and FairGround both load law_school_lequy (Section 6.4)

### 4.2 Reproducibility
- All code, figures and loaders public; 83 EDA, 68 baseline and 77 Stage 2 and 4 figures
- All seven domain scripts and both checks reproduced exactly in a fresh environment built from requirements.txt

---

## 5. Results

### 5.1 Baseline Performance
- Accuracy: gradient boosting highest for Folktables (0.756), Student (0.658) and MEPS (0.859); logistic regression for COMPAS (0.686) and FairGround (0.913)
- AUC: gradient boosting highest for Law School (0.878), Lending Club (0.712) and Agricultural (0.938)

### 5.2 DPD Under the DP Constraint (Table 2)
- FairGround 96 to 97% reductions; Law School 96.7% and 91.5%
- Agricultural worsens under both models; COMPAS improves only under gradient boosting and Folktables only under random forest, both set by small groups
- MEPS: logistic regression and gradient boosting improve by 81% and 86%, random forest worsens to 0.253
- Lending Club and Student split by model

### 5.3 EOD Under the EO Constraint
- Law School strongest (98.7% for gradient boosting); all three Student models improve
- FairGround law_school_lequy improves (89%, 10%, 97%); creditcard worsens on all three
- COMPAS, Folktables and MEPS improve under logistic regression and gradient boosting and worsen under random forest; Agricultural worsens; Lending Club barely moves

### 5.4 Disparate Impact Ratio (Figure 2)
- Law School 0.643 to 0.957
- Lending Club 2.778 to 0.973 and Agricultural 0.653 to 1.042, reached by raising every group's predicted default rate
- Folktables per race against White under EO: four of five groups below 0.8 cross it
- COMPAS, FairGround and MEPS have no fixed-pair ratio; Student has a baseline-only ratio

### 5.5 Cross-Domain Comparison (Figure 3)
- 9 of 14 high-disparity pairs improve; 3 of 4 near-fair pairs worsen
- Four exceptions (COMPAS, Folktables) disappear on groups of at least 30 records; the fifth (Student random forest) with held-out thresholds
- MEPS is the only middle-range evaluation; two of three pairs improve

### 5.6 Accuracy-Fairness Tradeoff
- FairGround: 0.148 and 0.159 for logistic regression and gradient boosting, 0.010 for random forest
- MEPS 0.071 to 0.133; Student gradient boosting and random forest 0.076; COMPAS and Folktables 0.035 or less

### 5.7 Drift Detection (Figure 4)
- Nine models flagged within two steps of deployment, five flagged seven to eight steps into the shift, seven never flagged
- Regression size is fixed by construction, so these groups describe the monitor, not real fragility

---

## 6. Discussion

### 6.1 The Effectiveness Pattern
- A strong but imperfect guide; audit baseline DPD before applying a constraint
- Folktables' domain-level exception falls to 0.078 on groups of at least 30 records; Lending Club's near-fair exception remains

### 6.2 Multi-Metric Tradeoffs
- Student gradient boosting: EOD down 64%, DPD down 9%; MEPS logistic regression the reverse
- Lending Club and Agricultural DIRs look like successes but come from predicting default for everyone more often
- Small groups: group-size check results (Table 6 of cross_domain_results_table.md)

### 6.3 Production Monitoring Implications
- Recommend CUSUM or comparable monitoring alongside any post-processing constraint; validate on real deployments

### 6.4 Limitations
- Agricultural concerns business entities under ECOA's business-credit provisions
- Before-and-after comparisons also change the decision objective; no minimum group size in the reported metrics
- Thresholds fit on the training split; held-out check for random forest
- Healthcare only through MEPS, a survey dataset; MIMIC-III access not obtained
- Each classifier constrained separately, not a combined ensemble
- Law School and FairGround share law_school_lequy: seven independent sources

---

## 7. Conclusion
- Effectiveness tracks baseline disparity, and the high-disparity exceptions trace to small groups or to where thresholds were fit
- Audit baseline disparity, report several metrics on adequately sized groups, and monitor continuously
- Next steps: validate monitoring on real deployment data; test constraints on combined ensembles

---

## References
The paper's reference list (docs/references.md) is the single source: Ajarra and Basu (2026); Amazon Web Services (n.d.); Angwin et al. (2016); Breck et al. (2017); Chen et al. (2023); Chouldechova (2017); Ding et al. (2021); Dwork et al. (2012); Friedler et al. (2019); Hardt et al. (2016); Kamiran and Calders (2012); Obermeyer et al. (2019); Sariola et al. (2026); Sculley et al. (2015); Simson et al. (2025); Zhang et al. (2018).
