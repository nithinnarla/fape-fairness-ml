# FAPE: Fairness-Aware Production ML Pipeline Evaluation
## Paper Outline — JASIST Submission Target Sep 11 2026

---

## Abstract (250 words)

Production ML systems deployed in high-stakes domains — lending, criminal justice, hiring —
are audited for fairness at launch and rarely monitored thereafter. Existing frameworks
(AIF360, Fairlearn) produce point-in-time fairness snapshots that fail to capture
post-deployment degradation. We present FAPE (Fairness-Aware Production ML Pipeline
Evaluation), a cross-domain framework that evaluates fairness interventions across seven
real-world deployment contexts simultaneously and monitors fairness drift under distribution shift.

FAPE applies ThresholdOptimizer post-processing constraints (demographic parity, equalized
odds) to three classifier architectures (LR, RF, GB) across seven domains: COMPAS
recidivism, Folktables income prediction, Law School bar passage, Lending Club credit risk,
Agricultural loan approval, FairGround synthetic bias, and Student performance prediction.
We evaluate four regulatory-aligned metrics simultaneously — demographic parity difference
(DPD), equalized odds difference (EOD), disparate impact ratio (DIR), and accuracy cost —
and introduce CUSUM-based continuous monitoring for post-deployment fairness drift detection.

Key findings: (1) ThresholdOptimizer effectiveness is strongly domain-dependent — strongest
improvement in Law School (DPD 0.351→0.039, DIR 0.643→0.945) but counterproductive in
Agricultural (DPD 0.009→0.035) where near-fair baselines exist; (2) no single model
dominates across all domains, requiring domain-specific model selection; (3) fairness
constraints achieved at deployment are not permanent — CUSUM detects measurable drift under
synthetic distribution shift, with Law School, FairGround, and Student domains most
sensitive. FAPE establishes that production fairness auditing requires continuous
cross-domain monitoring rather than point-in-time evaluation.

---

## 1. Introduction (~800 words)

### 1.1 The Production Fairness Gap
- ML systems deployed in consequential domains are audited once, rarely monitored after
- Sculley et al. (2015): production ML systems degrade silently over time
- Ajarra et al. (2026): model updates fundamentally alter fairness properties
- AIF360, Fairlearn: research-grade, point-in-time, single-domain — stop at validation
- Gap: no framework tests cross-domain generalizability AND post-deployment monitoring together

### 1.2 The Cross-Domain Generalization Problem
- Most fairness papers: one dataset, one domain, one metric, claim generalizability
- Does ThresholdOptimizer that works in criminal justice also work in credit lending?
- Does it work in agricultural loan approval? In student performance prediction?
- No systematic empirical answer in the literature
- FAPE answers this across 7 real deployment contexts simultaneously

### 1.3 FAPE Contributions
1. First systematic cross-domain evaluation of ThresholdOptimizer across 7 real deployment contexts
2. Simultaneous 4-metric evaluation (DPD, EOD, DIR, accuracy cost) — single-metric papers miss tradeoffs
3. CUSUM-based continuous drift detection for post-deployment fairness monitoring
4. Empirical effectiveness pattern: DPD > 0.2 effective and DPD < 0.05 counterproductive in 5 of 7 domains, with Folktables and Lending Club as documented exceptions -- see Decision 19

### 1.4 Paper Organization
Section 2: Related work. Section 3: Methodology. Section 4: Experimental setup.
Section 5: Results. Section 6: Discussion. Section 7: Conclusion.

---

## 2. Related Work (~600 words)

### 2.1 Fairness Interventions
- Pre-processing: reweighting, resampling (Kamiran & Calders 2012)
- In-processing: adversarial debiasing (Zhang et al. 2018)
- Post-processing: ThresholdOptimizer (Hardt et al. 2016) — FAPE's primary intervention
- Why post-processing: no retraining required, production-deployable without model access

### 2.2 Fairness Metrics and Impossibility
- Demographic parity (Dwork et al. 2012)
- Equalized odds (Hardt et al. 2016)
- Disparate impact ratio — EEOC 4/5ths rule (DIR > 0.8 = compliant)
- Chouldechova (2017) impossibility theorem: cannot satisfy all metrics simultaneously
- Sariola et al. (2026): optimizing one metric can mask 10% disparity on another
- FAPE: reports all four simultaneously — practitioners choose based on regulatory context

### 2.3 Cross-Domain Fairness Evaluation
- Most papers: single dataset, single metric, single model architecture
- FairGround (Fabris et al. 2025): multi-domain benchmark — FAPE includes it and 6 additional domains
- No prior work: systematic cross-domain ThresholdOptimizer evaluation across 7 domains

### 2.4 Production Fairness Monitoring
- Sculley et al. (2015): ML technical debt — systems degrade post-deployment
- Ajarra et al. (2026): fairness degradation specifically under model updates
- No existing framework: continuous post-deployment fairness monitoring
- FAPE Stage 4: CUSUM-based detection fills this gap

---

## 3. Methodology (~1000 words)

### 3.1 Framework Overview
- 4-stage pipeline: data preprocessing → baseline classification → fairness intervention → drift monitoring
- Post-processing only: no retraining required — production-deployable
- Regulatory alignment: each metric mapped to domain-specific legal standard (ECOA, EEOC, Title VII)

### 3.2 Datasets and Domains

| Domain | Dataset | Sensitive attr | N | Regulatory context |
|--------|---------|----------------|---|-------------------|
| Criminal justice | COMPAS | Race (6 groups) | 6,172 | ECOA |
| Income prediction | Folktables ACS | Race, Sex | 1,589,032¹ | ECOA |
| Legal profession | Law School | Race, Sex | 18,692 | Title VII |
| Credit risk | Lending Club | Income band | 1,348,099² | ECOA/FCRA |
| Agricultural lending | SBA 7(a) | Business type | 15,845 | ECOA/FCA |
| Synthetic bias | FairGround | Multiple | 1,964,010 | EEOC |
| Education | Student Performance | Sex | 1,044 (395+649)³ | Title IX |

¹ Full ACS national sample across 50 states; 100K stratified sample used for model training (50K per income group)
² After binary outcome filter (paid off vs charged off) from 2,260,701 raw records; 500K stratified sample used for model training
³ Two subjects: Math (395) and Portuguese (649); both used independently

### 3.3 Baseline Models
- LR, RF, GB with default hyperparameters across all domains
- Identical architecture isolates fairness intervention as the variable
- Baseline performance is model-dependent, not uniformly GB-dominant: GB achieves highest AUC in all 3 AUC-only domains (Law School, Lending Club, Agricultural), but among the 4 true-accuracy domains GB wins in 2 (Folktables, Student) while LogisticRegression wins in 2 (COMPAS, FairGround/law_school_lequy) -- see Section 5.1 and methodology_decisions.md Decisions 13 and 18

### 3.4 Fairness Intervention
- ThresholdOptimizer (Fairlearn) — post-processing threshold optimization
- Two constraints: demographic parity (DP), equalized odds (EO)
- EEOC regulatory boundaries: DPD < 0.1 and DIR > 0.8

### 3.5 Evaluation Metrics
- DPD: demographic parity difference — primary EEOC alignment metric
- EOD: equalized odds difference — equal error rates across groups
- DIR: disparate impact ratio — min/max prediction rate ratio per domain (EEOC 4/5ths rule)
- Accuracy cost: baseline_acc minus constrained_acc — price paid for fairness constraint

### 3.6 Drift Detection (Stage 4)
- CUSUM algorithm calibrated to EEOC threshold (DPD > 0.1)
- 3 model versions: v1=baseline, v2=post-constraint, v3=distribution shift
- Synthetic distribution shift — proof-of-concept validation (Decision 7, acknowledged limitation)
- np.random.seed(42) — reproducible drift simulation

---

## 4. Experimental Setup (~400 words)

### 4.1 Implementation
- Python 3.11, Fairlearn 0.10, scikit-learn 1.4
- ThresholdOptimizer: grid search over threshold space
- Lending Club: stratified 100K sample — calibration stable beyond this scale (Decision 8)
- All experiments: random seed 42

### 4.2 Reproducibility
- GitHub: github.com/nithinnarla/fape-fairness-ml
- All datasets publicly available (COMPAS, ACS, Law School, Lending Club, SBA, FairGround, UCI)
- Full pipeline reproducible from repo — 83 EDA figures + 60 Stage 2 figures committed

---

## 5. Results (~1200 words)

### 5.1 Baseline Model Performance
- No single model dominates across all 7 domains -- GB wins baseline AUC in all 3 AUC-only domains, but among the 4 true-accuracy domains GB wins 2 (Folktables, Student) and LR wins 2 (COMPAS, FairGround/law_school_lequy); RF is never highest in any domain -- see Decision 18
- 4 domains report true classification accuracy: COMPAS (GB=0.674, lowest of these 4 -- 6-group racial classification challenge), Folktables (GB=0.845), FairGround (GB=0.910, specifically the law_school_lequy sub-dataset -- see Decision 14), Student (GB=0.658)
- 3 domains report AUC rather than accuracy, since their Stage 2 scripts never compute accuracy_score for the baseline model -- see Decision 13: Law School (GB=0.878, largest LR-GB gap within this group at GB=0.878 vs LR=0.745), Lending Club (GB=0.712), Agricultural (GB=0.938, highest AUC of these 3)
- These two groups use different metrics and are not directly ranked against each other as a single ordered list; within-group comparisons (accuracy-to-accuracy, AUC-to-AUC) are valid, cross-group comparisons are not

### 5.2 Post-DP Constraint: DPD Results
- Law School: all 3 models improve — GB strongest (DPD 0.351→0.039, 88.9% reduction)
- FairGround GB: DPD 0.342→0.026 (92.4% reduction) — highest accuracy cost (0.159); LR+RF worsen
- Agricultural GB: DPD 0.009→0.035 (-288.9%) — counterproductive; LR+RF no change
- COMPAS GB: DPD 0.857→0.571 — improves but above EEOC threshold; LR+RF worsen
- Folktables GB: DPD 0.320→0.341 — slightly counterproductive; LR+RF no change
- Student: all 3 models improve
- Effectiveness is model-dependent within domains — GB most effective in high-DPD contexts

### 5.3 Post-EO Constraint: EOD Results
- Law School: all 3 models improve — GB strongest (EOD 0.528→0.031, 94.1% reduction)
- Student: all 3 models improve — GB strongest (EOD 0.314→0.055, 82.5% reduction)
- FairGround RF+GB: improve strongly — LR slightly worsens (0.018→0.019)
- COMPAS GB: EOD 1.000→0.659 — improves but remains high; RF worsens (0.686→0.734)
- Agricultural GB: counterproductive under EO (EOD 0.073→0.194); LR+RF no change
- Folktables GB: slightly worsens (0.333→0.336); LR+RF improve
- Lending Club GB: worsens (0.053→0.060); LR+RF no change

### 5.4 Disparate Impact Ratio (DIR)
- DIR computed per domain using domain-specific sensitive attribute groups
- Law School GB: DIR 0.643→0.945 — passes EEOC 4/5ths rule post-constraint
- Lending Club: DIR>1 at baseline (1.4x actual, 2.8x predicted) — proxy-based audit
- Agricultural GB: DIR overcorrects (0.653→1.095) — surpasses parity threshold
- COMPAS: DIR remains below EEOC threshold across all models — 6-group challenge
- Note: DIR not aggregated cross-domain — sensitive attributes differ per domain

### 5.5 Cross-Domain Comparison
- Core empirical finding: effectiveness pattern, not a universal rule (Decision 19)
  - DPD > 0.2 at baseline → ThresholdOptimizer effective in 3 of 4 such domains (COMPAS, Law School, FairGround); Folktables is the exception, where it got worse (+5.9%)
  - DPD < 0.05 at baseline → ThresholdOptimizer counterproductive in 1 of 2 such domains (Agricultural); Lending Club is the exception, where it improved (-25%)
- Effectiveness is model-dependent within domains — not just domain-dependent
- No single model dominates across all domains — domain-specific selection required
- LR most stable: smallest accuracy cost, fewest counterproductive outcomes
- GB is not uniformly highest baseline performance (see Decision 18) but is the most aggressive under constraints — highest accuracy cost in FairGround (0.159, law_school_lequy sub-dataset)

### 5.6 Accuracy-Fairness Tradeoff
- FairGround GB: highest accuracy cost (0.159 ⚠️) — strongest fairness gain
- FairGround LR: also high cost (0.072 ⚠️) with weaker fairness gain
- Lending Club GB: unexpected high cost (0.062 ⚠️) despite near-fair baseline
- Student GB: meaningful cost (0.063 ⚠️) with strong fairness improvement
- Law School: minimal accuracy cost (0.003) despite largest fairness improvement — best tradeoff
- Agricultural: accuracy cost without fairness benefit — worst case
- LR most stable: no high-cost outcomes across all 7 domains

### 5.7 Drift Detection Results
- Law School + FairGround + Student: earliest CUSUM alerts in v3
- Lending Club + Agricultural: no alerts — near-fair baseline
- LR most drift-stable; GB most drift-sensitive
- DPD and EOD show similar drift patterns under distribution shift

---

## 6. Discussion (~600 words)

### 6.1 The Effectiveness Threshold Finding
- Near-fair baselines make ThresholdOptimizer counterproductive
- Agricultural DPD=0.009 at baseline — optimizer overshoots
- Practical implication: audit baseline DPD before applying any post-processing constraint
- Proposed rule: DPD > 0.2 → apply ThresholdOptimizer; DPD < 0.05 → investigate root cause

### 6.2 Multi-Metric Tradeoffs
- Chouldechova impossibility confirmed empirically — improving DPD often worsens EOD
- DIR reveals overcorrection cases missed by DPD alone (Agricultural GB: DIR→1.095)
- FAPE surfaces these tradeoffs; single-metric papers hide them
- COMPAS: irreducible fairness tension with 6 racial groups

### 6.3 Production Monitoring Implications
- Fairness constraints achieved at deployment are not permanent
- Distribution shift erodes gains — especially in high-improvement domains
- CUSUM provides actionable early warning before violations become systematic
- Recommendation: deploy CUSUM monitoring alongside any fairness intervention

### 6.4 Limitations
- Synthetic distribution shift — proof-of-concept, not real production telemetry
- ThresholdOptimizer only — pre/in-processing comparison out of scope
- COMPAS 6-group racial categorization reflects data collection, not endorsement
- Lending Club uses income/housing as socioeconomic proxies — no direct race/gender
- Agricultural domain inclusion outside traditional fairness literature scope — justified by consequential algorithmic decisions in agricultural lending
- Cross-domain comparison validity requires identical model architecture and hyperparameters across all domains — any domain-specific tuning would make results uninterpretable

---

## 7. Conclusion (~300 words)
- FAPE: first systematic cross-domain fairness evaluation across 7 real deployment contexts
- ThresholdOptimizer effectiveness is domain-dependent — not universally applicable
- Empirical effectiveness pattern: DPD > 0.2 effective and DPD < 0.05 counterproductive in 5 of 7 domains, with Folktables and Lending Club as documented exceptions -- see Decision 19
- No single model dominates — domain-specific model selection required
- DIR reveals overcorrection cases that DPD alone misses
- CUSUM drift detection: production fairness requires continuous monitoring
- Future: pre/in-processing comparison; real production deployment validation; healthcare domain

---

## References
- Hardt et al. (2016) — Equality of Opportunity in Supervised Learning, NeurIPS
- Chouldechova (2017) — Fair Prediction with Disparate Impact, Big Data
- Sculley et al. (2015) — Hidden Technical Debt in ML Systems, NeurIPS
- Ajarra et al. (2026) — Auditing Fairness under Model Updates, arXiv 2601.05909
- Fabris et al. (2025) — FairGround Corpus: Bias Begins with Data, arXiv
- Sariola et al. (2026) — Multi-Metric Fairness Evaluation, arXiv
- Ding et al. (2021) — Retiring Adult: New Datasets for Fair ML, NeurIPS
- Kamiran & Calders (2012) — Data Preprocessing Techniques for Classification, KAIS
- Dwork et al. (2012) — Fairness Through Awareness, ITCS
