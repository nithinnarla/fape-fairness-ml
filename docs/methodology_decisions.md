# FAPE — Methodology Decisions Log
## Key Decisions Before Phase 4 Implementation

**Period:** February 2026 — May 2026
**Researcher:** Nithin Narla
**Status:** Complete — decisions locked before Phase 4 experiments begin

---

## Why Document Decisions Before Running Experiments

The temptation in empirical ML research is to run experiments first and then decide what the methodology was based on what worked. That's how you get papers that report results without acknowledging the choices that produced them. I'm documenting these decisions now, before Phase 4, so the paper can't retroactively reframe methodology around favorable results.

There are eleven decisions here. Some were obvious. Some took real thought. A few I'm still not fully comfortable with — I've noted those explicitly.

---

## Decision 1 — Post-Processing Over In-Processing or Pre-Processing

**Decision:** Use Fairlearn ThresholdOptimizer (post-processing) as the fairness intervention.

**Why:** In-processing requires modifying the training pipeline — which assumes you own the training pipeline. In eight years of production ML I have never inherited a system where I could modify training. Post-processing works on any model regardless of how it was built. That's the only approach deployable in the environments FAPE is designed for.

**What I gave up:** In-processing can achieve better accuracy-fairness tradeoffs because the fairness constraint is built into training. Post-processing applies the constraint after the fact and can only adjust decision thresholds — it can't change what the model learned. The paper will acknowledge this tradeoff explicitly.

---

## Decision 2 — Default Hyperparameters for LR, RF, GB Across All Domains

**Decision:** Use identical default hyperparameters for LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier (sklearn) across all seven domains with no domain-specific tuning.

**Why:** The research question is causal — does the fairness intervention cause bias reduction? If I tune separately per domain, accuracy differences across domains could reflect tuning rather than genuine domain variation. Default hyperparameters keep the experimental design clean and isolate the fairness intervention as the variable of interest.

**What I gave up:** Accuracy numbers will be lower than they could be with tuning. A reviewer might push back on this. The response is that optimized accuracy is not the variable of interest — the fairness intervention is. Domain-specific accuracy comparison is a different paper.

**Note:** XGBoost was considered but excluded — sklearn GradientBoostingClassifier provides sufficient performance with simpler reproducibility and no additional dependency.

---

## Decision 3 — Chouldechova Constraint Acknowledged Throughout

**Decision:** Every result section explicitly acknowledges the impossibility theorem constraint — cannot simultaneously satisfy equalized odds and calibration when base rates differ.

**Why:** The COMPAS data shows different recidivism base rates by race. The impossibility theorem applies. Any experiment that reports improved equalized odds is implicitly reporting degraded calibration. Hiding this would mean claiming fairness achievement in a situation where the math proves it's impossible to achieve all fairness properties simultaneously.

**Comfort level:** High. This is the right call and it makes the paper more honest, not less. Reviewers who know the literature will expect this acknowledgment.

---

## Decision 4 — Four Metrics Reported Simultaneously, No Primary Metric

**Decision:** Report demographic parity difference (DPD), equalized odds difference (EOD), disparate impact ratio (DIR), and accuracy cost for every experiment. No single metric designated as primary. Individual fairness score excluded — see Decision 9.

**Why:** Sariola et al. (2026) showed optimizing for one metric can mask 10% disparity on another. Designating a primary metric would invite the paper to be read as optimizing for that metric specifically — which would make the results misleading. All four reported, practitioners decide which matters in their regulatory context.

**What I gave up:** Clean headline results. "FAPE improves demographic parity by X%" is a cleaner claim than four tradeoff curves. The paper will be harder to summarize in an abstract. That's the right tradeoff.

---

## Decision 5 — Cross-Domain Comparison as Central Contribution

**Decision:** The primary empirical contribution is cross-domain comparison of fairness metric behavior, not within-domain improvement over a baseline.

**Why:** Within-domain improvement over a baseline is what every other fairness paper does. The gap FAPE fills is the cross-domain question — do interventions that work in criminal justice also work in education, financial services, and agricultural contexts? That's the question nobody has answered empirically.

**Risk:** Cross-domain comparison requires the experimental design to hold constant everything except the domain — which is why Decisions 2 and 1 above are locked in. If I loosen those constraints, the cross-domain comparison becomes uninterpretable.

---

## Decision 6 — Agricultural Domain Included Despite Literature Absence

**Decision:** Include three agricultural datasets as a distinct domain in the cross-domain evaluation.

**Why:** I searched for fairness papers on agricultural lending and farm household outcomes. Nothing exists. The populations affected — small farmers, agricultural loan applicants, LSMS-ISA farm households in Nigeria — are invisible in a literature that claims to address fairness in high-stakes algorithmic decisions. Including this domain is both a methodological contribution and a statement about whose fairness the field has been ignoring.

**Risk:** Reviewers may push back on agricultural domain inclusion as outside FAPE's stated scope. The response is that the scope is defined by where consequential algorithmic decisions are being made, not by where previous fairness papers have looked.

---

## Decision 7 — Synthetic Distribution Shift for Stage 4 Validation

**Decision:** Use synthetic distribution shift to validate Stage 4 CUSUM drift detection rather than real production data.

**Why:** I don't have access to a live production system. This is a real limitation — Stage 4 validation is controlled rather than real-world. The drift detection results are proof-of-concept.

**Comfort level:** Low. This is the weakest methodological choice in FAPE. The paper will acknowledge it directly. The alternative was to not include Stage 4 at all — which would mean not addressing the production monitoring gap that motivated Stage 4's existence. Proof-of-concept is better than absence.

---

## Decision 8 — Folktables ACS Over Adult Income

**Decision:** Use Folktables ACS as the socioeconomic benchmark, not Adult Income.

**Why:** Ding et al. (2021) documented Adult Income's methodological flaws. Using it in 2026 after that finding is indefensible. Folktables ACS uses US Census data, covers all 50 states, provides multiple prediction tasks, and is 30x larger.

**What I gave up:** Comparability with prior work. Results on Folktables ACS cannot be directly compared to results on Adult Income — different data, different task framing, different population. The paper will acknowledge this rather than treating the benchmark switch as costless.

---

## Decision 9 — Individual Fairness Score Excluded; Replaced with Accuracy Cost

**Decision:** Exclude individual fairness score (IFS) from FAPE's evaluation metrics. Replace with accuracy cost (baseline_acc minus constrained_acc) as the fourth metric alongside DPD, EOD, and DIR.

**Why:** IFS requires a validated similarity metric between individuals — a domain-specific requirement that cannot be generalized across FAPE's seven domains without introducing domain-specific assumptions that undermine the cross-domain comparison. Computing IFS on COMPAS requires a different similarity function than on Lending Club or Student Performance. Including IFS would force domain-specific methodology that contradicts FAPE's central contribution.

**What I gave up:** IFS is foundational (Dwork et al. 2012) and excluding it may invite reviewer questions. The response is that accuracy cost is a more practically meaningful fourth metric for production ML auditing — it directly quantifies the fairness-accuracy tradeoff that practitioners face.

**Comfort level:** High. Accuracy cost is computed directly from committed results, fully reproducible, and maps to a real deployment concern. IFS is deferred to future work with domain-specific similarity metrics.

---

## Decision 10 — MIMIC-III Included as Pending

**Decision:** Include MIMIC-III healthcare domain in the framework design with a pending access note rather than excluding healthcare entirely.

**Why:** Obermeyer et al. (2019) documented the most consequential fairness failure mechanism in the healthcare domain — cost as proxy for health need. Excluding healthcare from FAPE entirely because of access delays would mean the framework doesn't address the domain where the evidence for its value is strongest.

**Implementation:** MIMIC-III loader is built and tested. PhysioNet access is pending. If access comes through before writing time, healthcare results are included. If not, the paper includes the loader, describes the methodology, and notes the access gap explicitly. Either way the framework design includes healthcare.

---

## What Changes After These Decisions Are Locked

Phase 4 begins with these eleven decisions fixed. The experiments cannot change the methodology — they can only produce results within it. If the results are unfavorable under these constraints, the paper reports them honestly rather than retroactively adjusting the methodology to produce better numbers.

That's the standard I'm holding FAPE to.

## Decision 11 — Agricultural Dataset Scope: USDA NASS and LSMS-ISA Nigeria Evaluated and Excluded from ML Pipeline

**Decision:** Include only SBA 7(a) Agricultural Loans in FAPE's ML fairness pipeline. USDA NASS 2022 Census included descriptively in EDA only. LSMS-ISA Nigeria excluded entirely from ML pipeline.

**USDA NASS 2022 Agricultural Census:**
USDA NASS provides aggregate census counts of US agricultural producers by race group — 6 race groups including American Indian/Alaska Native (1,533,317 producers), Asian (601,476), Black/African American (1,049,156), Hispanic (2,114,132), White (78,199,536), and Native Hawaiian/Pacific Islander (71,197 operations). This is the only US dataset with direct race data for agricultural producers. However, USDA NASS is aggregate summary data — not individual records. Cannot run logistic regression, gradient boosting, or ThresholdOptimizer on 6 rows of race group counts. Included descriptively in EDA notebook (Figure 12) to characterize racial composition of US agricultural producers and motivate why agricultural lending fairness matters. Loader committed at src/usda_nass_loader.py.

**Why not ML:** ThresholdOptimizer requires one row per individual with a binary outcome. USDA NASS has one row per race group with aggregate counts. Mathematically incompatible with any ML fairness intervention.

**LSMS-ISA Nigeria GHS-Panel Wave 4 (World Bank):**
Individual-level agricultural household survey — 30,312 individuals, sex and education as sensitive attributes, food security as binary target (48.9% positive rate). Data loads correctly via src/lsms_loader.py. ML pipeline is technically feasible.

**Why excluded:** FAPE's scope is US production ML pipeline fairness auditing under ECOA/EEOC regulatory frameworks. LSMS Nigeria targets food security outcomes in Nigeria — a different regulatory context (no ECOA, no EEOC), different country, different outcome variable. Including it would require reframing FAPE's contribution away from US regulatory compliance auditing. A reviewer would correctly ask: "Why is a Nigeria household survey in a paper about US ML fairness?" No defensible answer exists within FAPE's current framing.

**Future use:** Loader committed at src/lsms_loader.py for potential future international extension of FAPE's methodology to non-US agricultural contexts.

**What USDA NASS descriptive figures show:** White producers hold 78.2M of total producers and 4.1B acres operated — structural dominance that contextualizes why agricultural lending fairness matters for minority farming communities despite the SBA loan proxy-based audit showing near-fair business type predictions.
