# FAPE — Methodology Decisions Log
## Key Decisions Before Phase 4 Implementation

**Period:** February 2026 — May 2026
**Researcher:** Nithin Narla
**Status:** Complete — decisions locked before Phase 4 experiments begin

---

## Why Document Decisions Before Running Experiments

The temptation in empirical ML research is to run experiments first and then decide what the methodology was based on what worked. That's how you get papers that report results without acknowledging the choices that produced them. I'm documenting these decisions now, before Phase 4, so the paper can't retroactively reframe methodology around favorable results.

There are seventeen decisions here. Some were obvious. Some took real thought. A few I'm still not fully comfortable with — I've noted those explicitly.

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

Phase 4 begins with these seventeen decisions fixed. The experiments cannot change the methodology — they can only produce results within it. If the results are unfavorable under these constraints, the paper reports them honestly rather than retroactively adjusting the methodology to produce better numbers.

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

## Decision 12 — Key Findings Summary Lines Are Hardcoded, Not Recomputed

**Decision:** Accept that threshold_aggregation.py's printed "Key Findings" summary (e.g. "GB best DP improvement: Law School", "LR most stable under constraints") are static strings written after one-time manual inspection of the results, not values computed dynamically from the results DataFrame.

**Why this matters:** Live re-execution on Aug 1 2026 confirmed these five summary lines (lines 112-116, 236-237 of threshold_aggregation.py) currently match the underlying numbers exactly. But because they are hardcoded print statements rather than `df.idxmax()`/`df.idxmin()`-style dynamic assertions, they carry no structural guarantee of staying correct if the underlying data, seed, or sklearn version ever changes and the script is re-run. A future re-run with different results would still print the same conclusion text regardless of whether it remained true.

**Why not fixed now:** Making these lines dynamic requires editing an already-verified, currently-correct script, which introduces its own re-verification burden for a benefit (protection against *future* drift) that doesn't change tonight's finding (the *current* claims are accurate). Deferred rather than risked mid-audit.

**Action for future work:** Before final JASIST submission (Aug 19 pre-submission audit), either convert these five print statements to compute their claims dynamically from the results DataFrame, or explicitly note in the paper that these are researcher-verified single-run observations rather than programmatically asserted invariants.

## Decision 13 — Aggregation Table Sources Metrics From stage2_*_threshold.py, Not baseline_*.py (Initially Misdiagnosed)

**Original finding (Aug 2 2026, morning):** COMPAS's aggregation dict baseline_acc (0.674) did not match live output from baseline_compas.py (acc=0.703, auc=0.751), and was initially logged as a likely transcription error.

**Correction (Aug 2 2026, afternoon):** Systematic re-verification across all 7 domains' stage2_*_threshold.py scripts found that threshold_aggregation.py's values match each domain's Stage 2 script's own internally-retrained baseline model -- NOT the standalone baseline_*.py script. COMPAS: stage2_compas_threshold.py prints baseline GradientBoosting ACC=0.674, an exact match. Folktables: stage2_folktables_threshold.py prints ACC=0.845, exact match. Student: stage2_student_threshold.py prints ACC=0.658, exact match.

**Why this happens:** baseline_*.py and stage2_*_threshold.py are two independently-written scripts per domain that each fit their own baseline model (same random_state=42, but not necessarily identical train/test split construction or feature preprocessing). They can and do produce different accuracy figures for what is conceptually "the same" baseline. The aggregation table was built from Stage 2's internally-computed baseline, which is appropriate since Stage 2's before/after comparison needs its own consistent baseline -- but this means baseline_*.py's separately-reported numbers (used elsewhere, e.g. in EDA or standalone baseline discussion) are not always identical to the aggregation table's figures, even though both claim to describe "the baseline model" for that domain.

**Remaining open issue (see Decision 14):** three domains' Stage 2 scripts (Law School, Lending Club, Agricultural) print AUC, not accuracy, for their baseline block -- meaning threshold_aggregation.py's "baseline_acc" column is genuinely accuracy for 4 domains (COMPAS, Folktables, FairGround, Student) and genuinely AUC for 3 domains (Law School, Lending Club, Agricultural), despite being presented under one column header and compared directly against each other in paper_outline.md Section 5.1's cross-domain ranking claim.

**Status:** The metric-mislabeling problem is real and confirmed (see Decision 14 for the FairGround sub-dataset issue found during this same investigation). The original transcription-error hypothesis was incorrect and is retracted here for the record.

**Audit closure (Aug 2 2026, later same day):** All three affected domains individually re-verified live -- Law School AUC=0.878, Lending Club AUC=0.712, Agricultural AUC=0.938 -- each an exact match to threshold_aggregation.py's stored baseline_acc. Confirmed the DPD/EOD metrics carry no equivalent divergence risk: baseline_compas.py (and, by structural pattern, the other standalone baseline_*.py scripts) never compute DPD/EOD at all -- only the Stage2 scripts do, so there is no competing "other" DPD/EOD source to diverge from the way baseline accuracy had two independent sources. This closes the investigation: exactly 3 of 7 domains (Law School, Lending Club, Agricultural) need their baseline_acc column relabeled as AUC or their scripts extended to compute true accuracy before Aug 19 pre-submission audit; the other 4 (COMPAS, Folktables, FairGround, Student) are genuine accuracy and need no change on this front.

## Decision 14 — FairGround's Reported Baseline Is Silently One Sub-Dataset Out of Five (law_school_lequy), Not an Aggregate

**Investigation (Aug 2 2026):** stage2_fairground_threshold.py iterates over five internal sub-datasets defined in SELECTED_DATASETS: adult (Income), compas_2_years (Criminal Justice), creditcard (Credit), law_school_lequy (Education), and meps_panel_19_fy2015 (Healthcare) -- each gets its own independently-trained baseline model and its own printed ACC/DPD/EOD block.

**Finding:** threshold_aggregation.py's single "FairGround" row (baseline_acc=0.910, baseline_dpd=0.342, baseline_eod=0.518) is an exact match to only the law_school_lequy sub-dataset's GradientBoosting results. It is not an average, weighted combination, or representative summary across the five sub-datasets -- it is specifically and only the Education/law-school sub-corpus's numbers, silently presented under the "FairGround" domain label.

**Why this matters:** The other four sub-datasets produce meaningfully different baseline accuracy (adult=0.871, compas_2_years=0.995, creditcard=0.819, meps_panel_19_fy2015=0.992) -- so which sub-dataset gets reported as "FairGround" materially changes the number that appears in the cross-domain comparison table and any paper claims built on it. Additionally, law_school_lequy as FairGround's representative sub-dataset creates a naming collision with FAPE's separate, standalone Law School domain (a different dataset, loaded via lawschool_loader.py, not FairGround's internal law_school_lequy) -- two different datasets both touching on legal education admissions data, one reported as "Law School" and one silently embedded inside "FairGround," is a source of real confusion in interpreting the paper's 7-domain claim.

**Status:** Open, needs a decision before paper submission on one of two paths: (1) clarify in the paper that "FairGround" specifically refers to the law_school_lequy sub-dataset and rename the domain label accordingly to avoid the naming collision with the standalone Law School domain, or (2) recompute "FairGround" as a genuine aggregate/average across all five sub-datasets, which would change every baseline_acc, baseline_dpd, and baseline_eod value currently attributed to FairGround throughout cross_domain_results_table.md and paper_outline.md.

**Not yet checked:** Whether this same silent single-sub-dataset-selection issue exists elsewhere, or whether FairGround is the only domain in FAPE built from a multi-sub-dataset corpus (fairground_loader.py itself contains many more than 5 datasets per earlier EDA work -- SELECTED_DATASETS is a curated subset of 5 chosen for Stage 2 specifically).

## Decision 15 — ThresholdOptimizer Post-Constraint Values Are Non-Deterministic Despite random_state=42; RESOLVED by Adding random_state to .predict()

**Investigation (Aug 2 2026):** While verifying Decision 13's Law School AUC/DIR figures, live re-execution of stage2_lawschool_threshold.py was run twice in immediate succession. Baseline values (pre-constraint) were identical both times: GB AUC=0.878, DP_diff=0.351, EO_diff=0.528 -- confirming the underlying LR/RF/GB models train deterministically with random_state=42, as expected.

**Finding:** Every post-constraint value differed between the two runs:
- Post-DP DP_diff: 0.026 (run 1) vs 0.034 (run 2) vs 0.039 (documented in paper_outline.md/cross_domain_results_table.md)
- Post-DP DP_improve: +0.325 vs +0.316 vs 88.9% reduction (documented)
- Post-EO EO_diff: 0.024 vs 0.020 vs 0.031 (documented)
- Post-EO EO_improve: +0.504 vs +0.508 vs 94.1% (documented)
- Post-constraint DIR: 0.951 (first run) vs 0.964 (second run) vs 0.945 (documented in Abstract and Table 5)

Several scripts already contain a comment acknowledging this ("Note: ThresholdOptimizer non-deterministic in fairlearn 0.13.0 -- results vary slightly between runs" -- present in stage2_lawschool_threshold.py, stage2_folktables_threshold.py, and stage2_lendingclub_threshold.py's live output) but this acknowledgment had not been connected to its consequence: every single-run post-constraint number currently published in paper_outline.md, cross_domain_results_table.md, and the Abstract itself is one sample from a distribution, not a fixed, exactly-reproducible value. random_state=42 controls model training (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier) but does not control whatever internal randomization Fairlearn's ThresholdOptimizer.fit() performs during its own optimization step in this version (fairlearn 0.13.0).

**Why this matters:** This is not a labeling or attribution problem like Decisions 12-14 -- it is a genuine reproducibility gap. A reviewer or reader re-running this exact code with the exact same seed will not get the exact published numbers back. The qualitative conclusions currently drawn (e.g. "Law School passes EEOC 4/5ths rule post-constraint," "88.9% DPD reduction") are directionally robust across the observed variation -- all three DIR runs (0.945, 0.951, 0.964) clear the EEOC>=0.8 threshold, and all three DPD reductions are in the 84-93% range -- but the specific point-estimate numbers currently written into the paper are not individually reproducible.

**Scope:** Confirmed for Law School specifically tonight. Given the shared root cause (ThresholdOptimizer's internal behavior in fairlearn 0.13.0, not a per-domain script difference), this almost certainly affects all 7 domains' post-constraint DPD, EOD, and DIR values, and by extension every accuracy-cost figure computed from post-constraint accuracy. Not yet individually re-verified per domain -- flagged as a required check.

**Required fix before Aug 19 pre-submission audit:** Replace every single-run post-constraint point estimate throughout paper_outline.md, cross_domain_results_table.md, and the Abstract with a mean +/- standard deviation computed across N repeated ThresholdOptimizer runs (e.g. N=10 or N=20) per domain per model per constraint type. This is standard practice for reporting results from any non-deterministic optimization procedure and is the only academically defensible way to present these numbers. A single run's numbers should not be published as if they were exact and reproducible when they are not.

**Not optional, not a stylistic choice:** unlike Decisions 12 and 13 (where AUC-vs-accuracy was a reasonable, defensible choice either way) and Decision 14 (where renaming vs. recomputing FairGround were both legitimate paths), this finding has exactly one correct fix. Presenting known-nondeterministic single-run numbers as fixed point estimates in a submitted paper is a genuine methodological error that a careful reviewer could flag.

**RESOLUTION (Aug 2 2026, same day):** Root cause found -- ThresholdOptimizer.predict() accepts an optional random_state parameter (confirmed via inspect.signature) that was not being passed in any of the 20 .predict() call sites across all 7 stage2_*_threshold.py scripts. Added random_state=42 to every call. Verified via 2 consecutive full-script runs per domain (all 7 domains checked) that every single reported number (baseline and post-constraint DPD, EOD, DIR, accuracy) is now exactly identical across repeated runs. This is a genuine fix, not a statistical workaround -- results are now truly deterministic and reproducible, matching the standard already achieved for model training via random_state=42 on the estimators themselves. The mean+/-std approach originally proposed above is no longer needed. All previously-documented single-run numbers throughout paper_outline.md and cross_domain_results_table.md need to be re-verified against these newly-deterministic values, since several point estimates (e.g. Law School DIR, previously observed as 0.945/0.951/0.964 across different pre-fix runs) will now differ from what was documented before this fix -- confirmed new deterministic Law School DIR after=0.957.

---

## Decision 16 — Law School RandomForest Baseline Value (0.801) Does Not Match Any Current Script Output; Source Unresolved

**Investigation (Aug 2 2026):** While rebuilding the full 7-domain results table after fixing Decision 15 (ThresholdOptimizer determinism), discovered that stage2_lawschool_threshold.py's MODELS dict contains only LogisticRegression and GradientBoosting -- RandomForest is entirely absent from this script. Yet threshold_aggregation.py's Law School row reports RF=0.801, and paper_outline.md/cross_domain_results_table.md both cite this figure.

**Ruled out:** Checked baseline_lawschool.py (the standalone baseline script) as a possible source -- its RandomForest AUC is 0.854, which does not match 0.801 either. So the documented RF=0.801 does not correspond to Stage 2 output (RF doesn't exist there) and does not correspond to the standalone baseline script's RF output (0.854, not 0.801).

**Status: Open, unresolved.** The actual source of 0.801 is not currently known. Possible explanations not yet checked: (1) an earlier version of stage2_lawschool_threshold.py may have included RandomForest and was later removed, with the aggregation dict never updated to match; (2) 0.801 may originate from a notebook run, an ad-hoc script, or a manual calculation not currently in the committed codebase; (3) simple transcription error with no traceable source.

**Required before Aug 19 pre-submission audit:** Either (a) add RandomForest to stage2_lawschool_threshold.py's MODELS dict, re-run, and use the genuinely-produced value, replacing 0.801 with whatever RF actually produces once added, or (b) if RF is intentionally excluded from Law School's Stage 2 analysis for a specific reason, remove the RF column/value for Law School throughout all tables rather than reporting an unverifiable number. Do not carry 0.801 forward into the final paper without resolving this.

**Given the volume of findings tonight (Decisions 12-16) and to avoid runaway investigation on writing day, this is being logged rather than chased further right now.** Recommend a dedicated, focused session before Aug 19 to resolve this specific item, ideally starting from git blame/log history on stage2_lawschool_threshold.py to check whether RandomForest was ever present and removed.

## Decision 17 — Table 1 (Baseline Model Performance) Required a Full Rebuild From Verified Live Output, Not Incremental Patching

**Investigation (Aug 2 2026):** After resolving Decision 15 (ThresholdOptimizer determinism), attempted to systematically re-verify every domain's Table 1 entry against live script output before propagating corrected numbers. Found discrepancies far more widespread than Decisions 13/14/16 had already documented:

- Folktables LR: documented 0.791, live Stage 2 shows 0.819
- Law School LR: documented 0.745, live Stage 2 shows AUC=0.872 (not comparable -- different metric, see Decision 13)
- Law School RF: documented 0.801, script has no RandomForest at all (see Decision 16)
- Lending Club LR: documented 0.649, live Stage 2 shows AUC=0.706
- Agricultural LR: documented 0.904, live Stage 2 shows AUC=0.727; standalone baseline_agricultural.py shows AUC=0.759 -- neither matches
- Agricultural RF: documented 0.921, script has no RandomForest at all (same pattern as Law School)
- FairGround: SELECTED_DATASETS contains 5 sub-corpora (adult, compas_2_years, creditcard, law_school_lequy, meps_panel_19_fy2015), each with distinct baseline results per model. Matching all 5 blocks against the documented Table 1 row (LR=0.819, RF=0.871, GB=0.910) found no single sub-dataset matches all three models -- GB=0.910 matches law_school_lequy exactly (as Decision 14 found), but RF=0.871 actually matches adult's RF value, and LR=0.819 matches creditcard's LR value. The documented FairGround row is a composite of three different sub-datasets' individual model columns, not one coherent sub-dataset row as Decision 14 originally concluded, and not an aggregate across all five either.
- Student: script evaluates two sub-subjects (math, portuguese) with different results; documented Table 1 values match "math" specifically, but this was not previously stated anywhere

**Conclusion:** GradientBoosting values were reliably accurate across every domain checked (COMPAS, Folktables, Law School, Lending Club, Agricultural all matched exactly). LogisticRegression values were wrong or mismatched in the majority of non-COMPAS domains. RandomForest is entirely absent from 2 of 7 domains' Stage 2 scripts despite having documented values. The likely explanation is that Table 1 was assembled at different points across the project's history from a mix of sources (different script versions, standalone baseline scripts, possibly manual entry) rather than generated in one consistent pass -- consistent with Decision 12's finding that the aggregation script itself is a hardcoded dictionary, not a live computation.

**Decision:** Rather than continue tracing each individual number's uncertain provenance, Table 1 is being fully rebuilt from scratch using only today's (Aug 2 2026) verified, deterministic live script output -- the same data captured while resolving Decision 15. Where a script does not compute a given model (RandomForest missing for Law School and Agricultural), that cell will be marked as not available rather than populated with an unverifiable historical number. Where a domain has multiple sub-datasets or sub-subjects (FairGround, Student), the specific one used will be explicitly named in the table rather than left implicit.

**This is the correct and final data source going forward.** Any future changes to the Stage 2 scripts should trigger a full Table 1 re-verification, not a patch to individual cells, given how much implicit assumption-drift accumulated the first time.
