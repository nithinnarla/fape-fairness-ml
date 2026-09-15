# FAPE, Methodology Decisions Log

> **Record note, September 15 2026.** This is a running log, so each entry keeps the date and the state of the project when it was written, and later entries correct earlier ones. Decisions 1 to 10 were set before Phase 4, when the study had seven domains; it now has eight evaluations across seven independent data sources. Where an entry was later overtaken, it carries a dated update or correction, and docs/paper_draft.md is the current statement of the study.

## Key Decisions Before Phase 4 Implementation

**Period:** February 2026 to September 2026
**Researcher:** Nithin Narla
**Status:** Decisions 1 to 10 locked before Phase 4 experiments began; Decisions 11 to 27 added during implementation and verification

---

## Why Document Decisions Before Running Experiments

The temptation in empirical ML research is to run experiments first and then decide what the methodology was based on what worked, which produces papers that report results without acknowledging the choices behind them. I'm documenting these decisions now, before Phase 4, so the paper can't retroactively reframe methodology around favorable results.

The first ten decisions below were fixed before Phase 4. Decisions 11 onward record what implementation and verification turned up, including errors in earlier entries. Some were obvious and some took real thought; a few I'm still not fully comfortable with, and I've noted those explicitly.

---

## Decision 1, Post-Processing Over In-Processing or Pre-Processing

**Decision:** Use Fairlearn ThresholdOptimizer (post-processing) as the fairness intervention.

**Why:** In-processing requires modifying the training pipeline, which assumes you own the training pipeline. In eight years of production ML I have never inherited a system where I could modify training. Post-processing works on any model regardless of how it was built, which makes it the only approach deployable in the environments FAPE is designed for.

**What I gave up:** In-processing can achieve better accuracy-fairness tradeoffs because the fairness constraint is built into training. Post-processing applies the constraint after the fact and can only adjust decision thresholds, it can't change what the model learned. The paper will acknowledge this tradeoff explicitly.

---

## Decision 2, Default Hyperparameters for LR, RF, GB Across All Domains

**Decision:** Use identical default hyperparameters for LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier (sklearn) across all seven domains with no domain-specific tuning.

**Why:** The research question is causal, does the fairness intervention cause bias reduction? If I tune separately per domain, accuracy differences across domains could reflect tuning rather than genuine domain variation. Default hyperparameters keep the experimental design clean and isolate the fairness intervention as the variable of interest.

**What I gave up:** Accuracy numbers will be lower than they could be with tuning. A reviewer might push back on this. The response is that optimized accuracy is not the variable of interest, the fairness intervention is. Domain-specific accuracy comparison is a different paper.

**Note:** XGBoost was considered but excluded, sklearn GradientBoostingClassifier provides sufficient performance with simpler reproducibility and no additional dependency.

---

## Decision 3, Chouldechova Constraint Acknowledged Throughout

**Decision:** Acknowledge the impossibility result throughout: calibration and equal error rates cannot all hold when base rates differ.

**Why:** The COMPAS data shows different recidivism base rates by race. The impossibility result applies: a model whose error rates are equalized across groups cannot also stay calibrated within each group. FAPE does not measure calibration, so the paper names this tradeoff rather than quantifying it. Hiding it would mean claiming fairness in a situation where the math shows not every fairness property can hold at once.

**Comfort level:** High. Reviewers who know the literature will expect this acknowledgment.

**Update (September 15 2026):** The paper states the result in Sections 2.2 and 6.2 and ties the multi-metric design to it in Section 3.5.

---

## Decision 4, Four Metrics Reported Simultaneously, No Primary Metric

**Decision:** Report demographic parity difference (DPD), equalized odds difference (EOD), disparate impact ratio (DIR), and accuracy cost for every experiment. No single metric designated as primary. Individual fairness score excluded, see Decision 9.

**Why:** Sariola et al. (2026) found that equalizing base rates in hiring data looked like parity on traditional measures but left about 10% disparity when measured with audit-study data, so a single number measured one way can mislead. Designating a primary metric would invite the paper to be read as optimizing for that metric specifically, which would make the results misleading. All four reported, practitioners decide which matters in their regulatory context.

**What I gave up:** Clean headline results. "FAPE improves demographic parity by X%" is a cleaner claim than four tradeoff curves. The paper will be harder to summarize in an abstract, and I think that is the right tradeoff.

**Update (September 15 2026):** DPD and EOD are reported for all eight evaluations. DIR and accuracy cost are reported only where a domain's pipeline computes them, and paper Section 3.5 says so.

---

## Decision 5, Cross-Domain Comparison as Central Contribution

**Decision:** The primary empirical contribution is cross-domain comparison of fairness metric behavior, not within-domain improvement over a baseline.

**Why:** Within-domain improvement over a baseline is what every other fairness paper does. The gap FAPE fills is the cross-domain question, do interventions that work in criminal justice also work in education, financial services, and agricultural contexts? I found no study that answers it for one intervention across this range of domains; comparative studies such as Chen et al. (2023) compare many methods on a few standard benchmark datasets (Decision 27).

**Risk:** Cross-domain comparison requires the experimental design to hold constant everything except the domain, which is why Decisions 2 and 1 above are locked in. If I loosen those constraints, the cross-domain comparison becomes uninterpretable.

---

## Decision 6, Agricultural Domain Included Despite Literature Absence

**Decision:** Include three agricultural datasets as a distinct domain in the cross-domain evaluation.

**Why:** I searched for fairness papers on agricultural lending and farm household outcomes and found none. The populations affected, small farmers, agricultural loan applicants, LSMS-ISA farm households in Nigeria, are invisible in a literature that claims to address fairness in high-stakes algorithmic decisions. Including this domain is both a methodological contribution and a statement about whose fairness the field has been ignoring.

**Risk:** Reviewers may push back on agricultural domain inclusion as outside FAPE's stated scope. The response is that the scope is defined by where consequential algorithmic decisions are being made, not by where previous fairness papers have looked.

**Update:** Decision 11 narrowed the modeling to SBA 7(a) loans; USDA NASS stays in EDA and LSMS-ISA Nigeria was excluded.

---

## Decision 7, Synthetic Distribution Shift for Stage 4 Validation

**Decision:** Use synthetic distribution shift to validate Stage 4 CUSUM drift detection rather than real production data.

**Why:** I don't have access to a live production system. This is a real limitation, Stage 4 validation is controlled rather than real-world. The drift detection results are proof-of-concept.

**Comfort level:** Low. This is the weakest methodological choice in FAPE. The paper will acknowledge it directly. The alternative was to not include Stage 4 at all, which would mean not addressing the production monitoring gap that motivated Stage 4's existence. Proof-of-concept is better than absence.

**Update (September 15 2026):** The monitor built on this choice scored pre-deployment values and has been corrected; see Decision 20.

---

## Decision 8, Folktables ACS Over Adult Income

**Decision:** Use Folktables ACS as the socioeconomic benchmark, not Adult Income.

**Why:** Ding et al. (2021) documented Adult Income's methodological flaws. Using it in 2026 after that finding is indefensible. Folktables ACS uses US Census data, covers all 50 states, provides multiple prediction tasks, and is 30x larger.

**What I gave up:** Comparability with prior work. Results on Folktables ACS cannot be directly compared to results on Adult Income, different data, different task framing, different population. The paper will acknowledge this rather than treating the benchmark switch as costless.

---

## Decision 9, Individual Fairness Score Excluded; Replaced with Accuracy Cost

**Decision:** Exclude individual fairness score (IFS) from FAPE's evaluation metrics. Replace with accuracy cost (baseline_acc minus constrained_acc) as the fourth metric alongside DPD, EOD, and DIR.

**Why:** IFS requires a validated similarity metric between individuals, a domain-specific requirement that cannot be generalized across FAPE's seven domains without introducing domain-specific assumptions that undermine the cross-domain comparison. Computing IFS on COMPAS requires a different similarity function than on Lending Club or Student Performance. Including IFS would force domain-specific methodology that contradicts FAPE's central contribution.

**What I gave up:** IFS is foundational (Dwork et al. 2012) and excluding it may invite reviewer questions. The response is that accuracy cost is a more practically meaningful fourth metric for production ML auditing, it directly quantifies the fairness-accuracy tradeoff that practitioners face.

**Comfort level:** High. Accuracy cost is computed directly from committed results, fully reproducible, and maps to a real deployment concern. IFS is deferred to future work with domain-specific similarity metrics.

---

## Decision 10, MIMIC-III Included as Pending

**Decision:** Include MIMIC-III healthcare domain in the framework design with a pending access note rather than excluding healthcare entirely.

**Why:** Obermeyer et al. (2019) documented the most consequential fairness failure mechanism in the healthcare domain, cost as proxy for health need. Excluding healthcare from FAPE entirely because of access delays would mean the framework doesn't address the domain where the evidence for its value is strongest.

**Implementation:** MIMIC-III loader is built and tested. PhysioNet access is pending. If access comes through before writing time, healthcare results are included. If not, the paper includes the loader, describes the methodology, and notes the access gap explicitly. Either way the framework design includes healthcare.

**Correction (September 15 2026):** No MIMIC-III loader exists in this repository, and none appears anywhere in its git history. The implementation line above is not accurate, and the paper no longer claims a loader was built. Access did not come through, and healthcare is covered by MEPS Panel 19 through the FairGround corpus instead.

---

## What Changes After These Decisions Are Locked

Phase 4 begins with these ten decisions fixed. The experiments cannot change the methodology; they can only produce results within it. If the results are unfavorable under these constraints, the paper reports them as they are, and the methodology is not adjusted afterward to produce better numbers. That is the standard I'm holding FAPE to.

## Decision 11, Agricultural Dataset Scope: USDA NASS and LSMS-ISA Nigeria Evaluated and Excluded from ML Pipeline

**Decision:** Include only SBA 7(a) Agricultural Loans in FAPE's ML fairness pipeline. USDA NASS 2022 Census included descriptively in EDA only. LSMS-ISA Nigeria excluded entirely from ML pipeline.

**USDA NASS 2022 Agricultural Census:**
USDA NASS provides aggregate census counts of US agricultural producers by race group, 6 race groups including American Indian/Alaska Native (56,203 producers), Asian (22,788), Black/African American (41,807), Hispanic (112,379), White (3,219,263), and Native Hawaiian/Pacific Islander (3,419), corrected 2026-08-17 after the original figures were found to be 20-36x too large for every category, sourced against a single unified USDA NASS 2022 producer table (via Farm Credit Administration report citing 2017/2022 Ag Census directly). It is the only US source I found with race data for agricultural producers. However, USDA NASS is aggregate summary data, not individual records. Cannot run logistic regression, gradient boosting, or ThresholdOptimizer on 6 rows of race group counts. Included descriptively in EDA notebook (Figure 12) to characterize racial composition of US agricultural producers and motivate why agricultural lending fairness matters. Loader committed at src/usda_nass_loader.py.

**Why not ML:** ThresholdOptimizer requires one row per individual with a binary outcome. USDA NASS has one row per race group with aggregate counts. Mathematically incompatible with any ML fairness intervention.

**LSMS-ISA Nigeria GHS-Panel Wave 4 (World Bank):**
Individual-level agricultural household survey, 30,312 individuals, sex and education as sensitive attributes, food security as binary target (48.9% positive rate). Data loads correctly via src/lsms_loader.py. ML pipeline is technically feasible.

**Why excluded:** FAPE's scope is fairness auditing of US production ML pipelines, where US law such as ECOA sets the regulatory context. LSMS Nigeria targets food security outcomes in Nigeria, a different regulatory context, different country, different outcome variable. Including it would require reframing FAPE's contribution away from the US setting. A reviewer would correctly ask: "Why is a Nigeria household survey in a paper about US ML fairness?" No defensible answer exists within FAPE's current framing.

**Future use:** Loader committed at src/lsms_loader.py for potential future international extension of FAPE's methodology to non-US agricultural contexts.

**What USDA NASS descriptive figures show:** In the corrected NASS table, White producers number 3,219,263 and operate 823.9 million acres, far more than every other group combined. That concentration is the context for why agricultural lending fairness matters for minority farming communities, even though the SBA 7(a) audit, which has no race field and uses business type, shows near-fair predictions.

**Correction (September 15 2026):** This paragraph previously gave 78.2M producers and 4.1B acres, the inflated figures from before the August 17 correction above.

## Decision 12, Key Findings Summary Lines Are Hardcoded, Not Recomputed

**Decision:** Accept that threshold_aggregation.py's printed "Key Findings" summary (e.g. "GB best DP improvement: Law School", "LR most stable under constraints") are static strings written after one-time manual inspection of the results, not values computed dynamically from the results DataFrame.

**Why this matters:** Live re-execution on Aug 1 2026 confirmed these five summary lines (lines 112-116, 236-237 of threshold_aggregation.py) currently match the underlying numbers exactly. But because they are hardcoded print statements rather than `df.idxmax()`/`df.idxmin()`-style dynamic assertions, they carry no structural guarantee of staying correct if the underlying data, seed, or sklearn version ever changes and the script is re-run. A future re-run with different results would still print the same conclusion text regardless of whether it remained true.

**Why not fixed now:** Making these lines dynamic requires editing an already-verified, currently-correct script, which introduces its own re-verification burden for a benefit (protection against *future* drift) that doesn't change tonight's finding (the *current* claims are accurate). Deferred rather than risked mid-audit.

**Action for future work:** Before final JASIST submission (Aug 19 pre-submission audit), either convert these five print statements to compute their claims dynamically from the results DataFrame, or explicitly note in the paper that these are researcher-verified single-run observations rather than programmatically asserted invariants.

**Resolved:** threshold_aggregation.py computes its findings since the August 3 rebuild (Decision 18). On September 15 2026 the printed findings in the Stage 2 domain scripts, several of which no longer matched their own output, were also made computed.

## Decision 13, Aggregation Table Sources Metrics From stage2_*_threshold.py, Not baseline_*.py (Initially Misdiagnosed)

**Original finding (Aug 2 2026, morning):** COMPAS's aggregation dict baseline_acc (0.674) did not match live output from baseline_compas.py (acc=0.703, auc=0.751), and was initially logged as a likely transcription error.

**Correction (Aug 2 2026, afternoon):** Systematic re-verification across all 7 domains' stage2_*_threshold.py scripts found that threshold_aggregation.py's values match each domain's Stage 2 script's own internally-retrained baseline model, not the standalone baseline_*.py script. COMPAS: stage2_compas_threshold.py prints baseline GradientBoosting ACC=0.674, an exact match. Folktables: stage2_folktables_threshold.py prints ACC=0.845, exact match. Student: stage2_student_threshold.py prints ACC=0.658, exact match.

**Why this happens:** baseline_*.py and stage2_*_threshold.py are two independently-written scripts per domain that each fit their own baseline model (same random_state=42, but not necessarily identical train/test split construction or feature preprocessing). They can and do produce different accuracy figures for what is conceptually "the same" baseline. The aggregation table was built from Stage 2's internally-computed baseline, which is appropriate since Stage 2's before/after comparison needs its own consistent baseline, but this means baseline_*.py's separately-reported numbers (used elsewhere, e.g. in EDA or standalone baseline discussion) are not always identical to the aggregation table's figures, even though both claim to describe "the baseline model" for that domain.

**Remaining open issue (see Decision 14):** three domains' Stage 2 scripts (Law School, Lending Club, Agricultural) print AUC, not accuracy, for their baseline block, meaning threshold_aggregation.py's "baseline_acc" column is true accuracy for 4 domains (COMPAS, Folktables, FairGround, Student) and AUC for 3 domains (Law School, Lending Club, Agricultural), despite being presented under one column header and compared directly against each other in paper_outline.md Section 5.1's cross-domain ranking claim.

**Status:** The metric-mislabeling problem is real and confirmed (see Decision 14 for the FairGround sub-dataset issue found during this same investigation). The original transcription-error hypothesis was incorrect and is retracted here for the record.

**Audit closure (Aug 2 2026, later same day):** All three affected domains individually re-verified live: Law School AUC=0.878, Lending Club AUC=0.712, Agricultural AUC=0.938, each an exact match to threshold_aggregation.py's stored baseline_acc. Confirmed the DPD/EOD metrics carry no equivalent divergence risk: baseline_compas.py (and, by structural pattern, the other standalone baseline_*.py scripts) never compute DPD/EOD at all; only the Stage2 scripts do, so there is no competing "other" DPD/EOD source to diverge from the way baseline accuracy had two independent sources. This closes the investigation: exactly 3 of 7 domains (Law School, Lending Club, Agricultural) need their baseline_acc column relabeled as AUC or their scripts extended to compute true accuracy before Aug 19 pre-submission audit; the other 4 (COMPAS, Folktables, FairGround, Student) are genuine accuracy and need no change on this front.

## Decision 14, FairGround's Reported Baseline Is Silently One Sub-Dataset Out of Five (law_school_lequy), Not an Aggregate

**Investigation (Aug 2 2026):** stage2_fairground_threshold.py iterates over five internal sub-datasets defined in SELECTED_DATASETS: adult (Income), compas_2_years (Criminal Justice), creditcard (Credit), law_school_lequy (Education), and meps_panel_19_fy2015 (Healthcare), and each gets its own independently-trained baseline model and its own printed ACC/DPD/EOD block.

**Finding:** threshold_aggregation.py's single "FairGround" row (baseline_acc=0.910, baseline_dpd=0.342, baseline_eod=0.518) is an exact match to only the law_school_lequy sub-dataset's GradientBoosting results. It is not an average, weighted combination, or representative summary across the five sub-datasets; it is specifically and only the Education/law-school sub-corpus's numbers, silently presented under the "FairGround" domain label.

**Why this matters:** The other four sub-datasets produce meaningfully different baseline accuracy (adult=0.871, compas_2_years=0.995, creditcard=0.819, meps_panel_19_fy2015=0.992), so which sub-dataset gets reported as "FairGround" materially changes the number that appears in the cross-domain comparison table and any paper claims built on it. Also, law_school_lequy as FairGround's representative sub-dataset creates a naming collision with FAPE's separate, standalone Law School domain (a different dataset, loaded via lawschool_loader.py, not FairGround's internal law_school_lequy). Two different datasets both touching on legal education admissions data, one reported as "Law School" and one silently embedded inside "FairGround," is a source of real confusion in interpreting the paper's 7-domain claim.

**Status:** Open, needs a decision before paper submission on one of two paths: (1) clarify in the paper that "FairGround" specifically refers to the law_school_lequy sub-dataset and rename the domain label accordingly to avoid the naming collision with the standalone Law School domain, or (2) recompute "FairGround" as a genuine aggregate/average across all five sub-datasets, which would change every baseline_acc, baseline_dpd, and baseline_eod value currently attributed to FairGround throughout cross_domain_results_table.md and paper_outline.md.

**Not yet checked:** Whether this same silent single-sub-dataset-selection issue exists elsewhere, or whether FairGround is the only domain in FAPE built from a multi-sub-dataset corpus (fairground_loader.py itself contains many more than 5 datasets per earlier EDA work; SELECTED_DATASETS is a curated subset of 5 chosen for Stage 2 specifically).

**Resolved (September 14 2026):** Closed via path (1), plus an answer to the open question. The domain is relabelled "FairGround (Education/law_school_lequy)" in threshold_aggregation.RESULTS and the manuscript names the sub-dataset wherever the row is cited, so nothing reports an unlabelled single sub-dataset as an aggregate. On "not yet checked": the issue does exist elsewhere, and it is worse than a labelling problem. lawschool_loader.py line 51 selects law_school_lequy as well, so the Law School domain and the FairGround row are built from the same source data. Section 6.4 discloses this and reframes the pair as an unplanned preprocessing-sensitivity measurement (GB baseline DPD 0.351 vs 0.342, post-constraint 0.030 vs 0.014), and Sections 1.3, 5.5 and 6.1 carry a duplicate-adjusted bracket on the headline heuristic. One further consequence: meps_panel_19_fy2015, listed above as one of the five sub-datasets at 0.992, became the eighth evaluation added in September. That 0.992 turned out to come from label leakage, and after Decision 24 its GB baseline accuracy is 0.859.

## Decision 15, ThresholdOptimizer Post-Constraint Values Are Non-Deterministic Despite random_state=42; RESOLVED by Adding random_state to .predict()

**Investigation (Aug 2 2026):** While verifying Decision 13's Law School AUC/DIR figures, live re-execution of stage2_lawschool_threshold.py was run twice in immediate succession. Baseline values (pre-constraint) were identical both times: GB AUC=0.878, DP_diff=0.351, EO_diff=0.528, confirming the underlying LR/RF/GB models train deterministically with random_state=42, as expected.

**Finding:** Every post-constraint value differed between the two runs:
- Post-DP DP_diff: 0.026 (run 1) vs 0.034 (run 2) vs 0.039 (documented in paper_outline.md/cross_domain_results_table.md)
- Post-DP DP_improve: +0.325 vs +0.316 vs 88.9% reduction (documented)
- Post-EO EO_diff: 0.024 vs 0.020 vs 0.031 (documented)
- Post-EO EO_improve: +0.504 vs +0.508 vs 94.1% (documented)
- Post-constraint DIR: 0.951 (first run) vs 0.964 (second run) vs 0.945 (documented in Abstract and Table 5)

Several scripts already contain a comment acknowledging this ("Note: ThresholdOptimizer non-deterministic in fairlearn 0.13.0, results vary slightly between runs", present in stage2_lawschool_threshold.py, stage2_folktables_threshold.py, and stage2_lendingclub_threshold.py's live output) but this acknowledgment had not been connected to its consequence: every single-run post-constraint number currently published in paper_outline.md, cross_domain_results_table.md, and the Abstract itself is one sample from a distribution, not a fixed, exactly-reproducible value. random_state=42 controls model training (LogisticRegression, RandomForestClassifier, GradientBoostingClassifier) but does not control whatever internal randomization Fairlearn's ThresholdOptimizer.fit() performs during its own optimization step in this version (fairlearn 0.13.0).

**Why this matters:** This is not a labeling or attribution problem like Decisions 12-14; it is a genuine reproducibility gap. A reviewer or reader re-running this exact code with the exact same seed will not get the exact published numbers back. The qualitative conclusions currently drawn (e.g. "Law School passes EEOC 4/5ths rule post-constraint," "88.9% DPD reduction") held directionally across the observed variation: all three DIR runs (0.945, 0.951, 0.964) clear the EEOC>=0.8 threshold, and all three DPD reductions are in the 84-93% range, but the specific point-estimate numbers currently written into the paper are not individually reproducible.

**Scope:** Confirmed for Law School specifically tonight. Given the shared root cause (ThresholdOptimizer's internal behavior in fairlearn 0.13.0, not a per-domain script difference), this almost certainly affects all 7 domains' post-constraint DPD, EOD, and DIR values, and by extension every accuracy-cost figure computed from post-constraint accuracy. Not yet individually re-verified per domain, flagged as a required check.

**Required fix before Aug 19 pre-submission audit:** Replace every single-run post-constraint point estimate throughout paper_outline.md, cross_domain_results_table.md, and the Abstract with a mean +/- standard deviation computed across N repeated ThresholdOptimizer runs (e.g. N=10 or N=20) per domain per model per constraint type. This is standard practice for reporting results from any non-deterministic optimization procedure and is the only academically defensible way to present these numbers. A single run's numbers should not be published as if they were exact and reproducible when they are not.

**Not optional, not a stylistic choice:** unlike Decisions 12 and 13 (where AUC-vs-accuracy was a reasonable, defensible choice either way) and Decision 14 (where renaming vs. recomputing FairGround were both legitimate paths), this finding has exactly one correct fix. Presenting known-nondeterministic single-run numbers as fixed point estimates in a submitted paper is a genuine methodological error that a careful reviewer could flag.

**RESOLUTION (Aug 2 2026, same day):** Root cause found: ThresholdOptimizer.predict() accepts an optional random_state parameter (confirmed via inspect.signature) that was not being passed in any of the 20 .predict() call sites across all 7 stage2_*_threshold.py scripts. Added random_state=42 to every call. Verified via 2 consecutive full-script runs per domain (all 7 domains checked) that every single reported number (baseline and post-constraint DPD, EOD, DIR, accuracy) is now exactly identical across repeated runs. (A September 15 2026 count finds 22 seeded predict calls across the seven scripts, the number the paper gives; the stale non-determinism notes in the scripts were removed the same day.) This is a genuine fix, not a statistical workaround: results are now truly deterministic and reproducible, matching the standard already achieved for model training via random_state=42 on the estimators themselves. The mean+/-std approach originally proposed above is no longer needed. All previously-documented single-run numbers throughout paper_outline.md and cross_domain_results_table.md need to be re-verified against these newly-deterministic values, since several point estimates (e.g. Law School DIR, previously observed as 0.945/0.951/0.964 across different pre-fix runs) will now differ from what was documented before this fix; confirmed new deterministic Law School DIR after=0.957.

---

## Decision 16, Law School RandomForest Baseline Value (0.801) Does Not Match Any Current Script Output; Source Unresolved

**Investigation (Aug 2 2026):** While rebuilding the full 7-domain results table after fixing Decision 15 (ThresholdOptimizer determinism), discovered that stage2_lawschool_threshold.py's MODELS dict contains only LogisticRegression and GradientBoosting; RandomForest is entirely absent from this script. Yet threshold_aggregation.py's Law School row reports RF=0.801, and paper_outline.md/cross_domain_results_table.md both cite this figure.

**Ruled out:** Checked baseline_lawschool.py (the standalone baseline script) as a possible source; its RandomForest AUC is 0.854, which does not match 0.801 either. So the documented RF=0.801 does not correspond to Stage 2 output (RF doesn't exist there) and does not correspond to the standalone baseline script's RF output (0.854, not 0.801).

**Status: Open, unresolved.** The actual source of 0.801 is not currently known. Possible explanations not yet checked: (1) an earlier version of stage2_lawschool_threshold.py may have included RandomForest and was later removed, with the aggregation dict never updated to match; (2) 0.801 may originate from a notebook run, an ad-hoc script, or a manual calculation not currently in the committed codebase; (3) simple transcription error with no traceable source.

**Required before Aug 19 pre-submission audit:** Either (a) add RandomForest to stage2_lawschool_threshold.py's MODELS dict, re-run, and use the value it produces, replacing 0.801 with whatever RF actually produces once added, or (b) if RF is intentionally excluded from Law School's Stage 2 analysis for a specific reason, remove the RF column/value for Law School throughout all tables rather than reporting an unverifiable number. Do not carry 0.801 forward into the final paper without resolving this.

**Given the volume of findings tonight (Decisions 12-16) and to avoid runaway investigation on writing day, this is being logged rather than chased further right now.** Recommend a dedicated, focused session before Aug 19 to resolve this specific item, ideally starting from git blame/log history on stage2_lawschool_threshold.py to check whether RandomForest was ever present and removed.

**Resolved (Aug 16 2026):** During Section 5 drafting and full source-verification pass, this was closed via option (b). RandomForest is confirmed intentionally absent from stage2_lawschool_threshold.py (and, by the same pattern, from stage2_lendingclub_threshold.py and stage2_agricultural_threshold.py; all three AUC-reporting domains share this design). The RF column for these three domains has been removed from paper_outline.md and marked N/A with an explanatory footnote in cross_domain_results_table.md, rather than carrying forward an unverifiable number. Law School's actual RF baseline, confirmed live via baseline_lawschool.py, is 0.854, and this value is used where RF baseline (not post-constraint) figures are cited, since that script does compute it even though Stage 2 does not.

## Decision 17, Table 1 (Baseline Model Performance) Required a Full Rebuild From Verified Live Output, Not Incremental Patching

**Investigation (Aug 2 2026):** After resolving Decision 15 (ThresholdOptimizer determinism), attempted to systematically re-verify every domain's Table 1 entry against live script output before propagating corrected numbers. Found discrepancies far more widespread than Decisions 13/14/16 had already documented:

- Folktables LR: documented 0.791, live Stage 2 shows 0.819
- Law School LR: documented 0.745, live Stage 2 shows AUC=0.872 (not comparable: different metric, see Decision 13)
- Law School RF: documented 0.801, script has no RandomForest at all (see Decision 16)
- Lending Club LR: documented 0.649, live Stage 2 shows AUC=0.706
- Agricultural LR: documented 0.904, live Stage 2 shows AUC=0.727; standalone baseline_agricultural.py shows AUC=0.759; neither matches
- Agricultural RF: documented 0.921, script has no RandomForest at all (same pattern as Law School)
- FairGround: SELECTED_DATASETS contains 5 sub-corpora (adult, compas_2_years, creditcard, law_school_lequy, meps_panel_19_fy2015), each with distinct baseline results per model. Matching all 5 blocks against the documented Table 1 row (LR=0.819, RF=0.871, GB=0.910) found no single sub-dataset matches all three models: GB=0.910 matches law_school_lequy exactly (as Decision 14 found), but RF=0.871 matches adult's RF value, and LR=0.819 matches creditcard's LR value. The documented FairGround row is a composite of three different sub-datasets' individual model columns, not one coherent sub-dataset row as Decision 14 originally concluded, and not an aggregate across all five either.
- Student: script evaluates two sub-subjects (math, portuguese) with different results; documented Table 1 values match "math" specifically, but this was not previously stated anywhere

**Conclusion:** GradientBoosting values were reliably accurate across every domain checked (COMPAS, Folktables, Law School, Lending Club, Agricultural all matched exactly). LogisticRegression values were wrong or mismatched in the majority of non-COMPAS domains. RandomForest is entirely absent from 2 of 7 domains' Stage 2 scripts despite having documented values. The likely explanation is that Table 1 was assembled at different points across the project's history from a mix of sources (different script versions, standalone baseline scripts, possibly manual entry) rather than generated in one consistent pass, consistent with Decision 12's finding that the aggregation script itself is a hardcoded dictionary, not a live computation.

**Decision:** Rather than continue tracing each individual number's uncertain provenance, Table 1 is being fully rebuilt from scratch using only today's (Aug 2 2026) verified, deterministic live script output, the same data captured while resolving Decision 15. Where a script does not compute a given model (RandomForest missing for Law School and Agricultural), that cell will be marked as not available rather than populated with an unverifiable historical number. Where a domain has multiple sub-datasets or sub-subjects (FairGround, Student), the specific one used will be explicitly named in the table rather than left implicit.

**This is the correct and final data source going forward.** Any future changes to the Stage 2 scripts should trigger a full Table 1 re-verification, not a patch to individual cells, given how much implicit assumption-drift accumulated the first time.

**Scope expansion confirmed (Aug 2 2026, later same session):** Checking threshold_aggregation.py's full dictionary structure (not just baseline_acc) confirms the entire dict (dp_acc, dp_dpd, eo_acc, eo_eod for every domain) predates the Decision 15 ThresholdOptimizer determinism fix. Spot-check against tonight's live COMPAS run: dict says dp_acc=0.673, eo_acc=0.675; live post-fix run shows dp_acc=0.675, eo_acc=0.677. Small but real gaps, confirming the entire dictionary needs regeneration from live, deterministic output, not just the baseline_acc column originally flagged. This is a larger task than initially scoped: rerun all 7 stage2_*_threshold.py scripts fresh, capture complete LR/RF/GB output for baseline + both constraints, rebuild threshold_aggregation.py's dictionary entirely from that output. Correctly deferred to a dedicated pre-Aug-19 session rather than attempted at the end of tonight's already-extensive audit; doing this rushed risks exactly the pattern seen earlier tonight where quick fixes required deeper correction (Decisions 13, 14->17).

**Scope correction (Aug 3 2026, rebuild session):** While capturing complete output for the Table 1 rebuild, confirmed that RandomForest is missing from the Lending Club and Agricultural Stage 2 scripts as well as Law School's (as originally documented above), all three of the same domains already flagged in Decision 13 for reporting AUC instead of accuracy. MODELS dict check confirms: Law School {LR, GB}, Lending Club {LR, GB}, Agricultural {LR, GB}, all missing RandomForest. COMPAS, Folktables, FairGround, and Student all have the full {LR, RF, GB} set. The same 3 scripts that skip accuracy computation also skip RandomForest entirely, which suggests these three were built to a simpler 2-model template than the other 4. The rebuilt Table 1 will mark RandomForest as N/A for Law School, Lending Club, and Agricultural; the earlier scope covered only Law School.

**Rebuild attempt and final scope finding (Aug 3 2026):** Captured complete, verified, deterministic LR/RF/GB baseline+DP+EO output for all 7 domains directly from the now-fixed Stage 2 scripts (post Decision 15 fix). This data is confirmed correct and ready to use; see git history for the full captured dataset. Attempted to write it into threshold_aggregation.py's RESULTS dict, marking RandomForest as None for Law School/Lending Club/Agricultural and renaming FairGround and Student to their specific sub-dataset/sub-subject. Two structural problems surfaced during this attempt, beyond just updating the dict: (1) every downstream loop in threshold_aggregation.py assumes all 3 models exist for every domain, fixed via a safe_get()/fmt() helper pair for the print-statement logic. (2) Figure 1 (baseline accuracy bar chart) cannot be patched the same way: it plots 'accuracy' as a single unified y-axis metric across all 7 domains, but 3 of those domains store AUC, not accuracy (Decision 13), so plotting them on the same axis would be scientifically misleading even after fixing the None-handling crash. This requires an actual figure redesign (e.g. splitting into two panels, one for the 4 true-accuracy domains and one for the 3 AUC domains), not a data fix.

**Decision:** Reverted threshold_aggregation.py to its original (pre-fix) state rather than ship a partially-patched script with a scientifically confusing figure. The verified 7-domain dataset captured tonight is preserved in this document's git history and is ready to be dropped into a properly redesigned script. Full script + figure rebuild remains correctly scoped to a dedicated pre-Aug-19 session, now with the complete data already in hand and the exact structural blockers identified precisely; the next session should not need to re-derive any of tonight's numbers, only implement the None-safe logic throughout the script and redesign Figure 1 to correctly separate the two metric types.

**Resolved (Aug 3 2026):** The rebuild was finished in the next session (see Decision 18), with RandomForest marked not evaluated for the three AUC domains and the baseline figure split by metric.

## Decision 18, "GB Achieves Highest Baseline Accuracy Across All 7 Domains" Is False; Never Actually Verified Until Tonight

**Investigation (Aug 3 2026):** While finishing the threshold_aggregation.py rebuild (adding computed Key Findings logic to replace the hardcoded strings flagged in Decision 12), the new computed check revealed that GB does not win baseline accuracy/AUC in all 7 domains; it wins in 2 of 4 true-accuracy domains (Folktables, Student) and all 3 AUC-only domains (Law School, Lending Club, Agricultural), but LOSES to LogisticRegression in COMPAS (LR=0.686 vs GB=0.674) and FairGround/law_school_lequy (LR=0.913 vs GB=0.910).

**Why this was never caught:** Every prior check tonight (Decisions 13, 16, 17) verified whether GB's specific documented number matched live script output; it never checked whether GB was the best-performing model in each domain. The claim "GB achieves highest baseline accuracy across all 7 domains" (present in paper_outline.md Section 3.3, Section 5.1, and cross_domain_results_table.md's Key Finding) is a comparative claim across models, not a single-number verification, and nothing in tonight's audit, or apparently at any point before tonight, computed this comparison. GB's individual numbers were correct; the comparative claim built on top of them was not.

**Status:** Confirmed false as currently written. paper_outline.md and cross_domain_results_table.md both need this claim corrected to reflect the real pattern: GB wins 5 of 7 domains (or however the comparison should be properly framed given the accuracy/AUC split from Decision 13), LR wins 2 of 7 (COMPAS, FairGround/law_school_lequy). This is not a rebuild-scope item like Decision 17; it is a direct, false claim in the paper's current text that must be corrected before Aug 19, independent of whether the full Table 1 rebuild is finished.

**Broader implication:** This raises real doubt about whether other comparative claims in paper_outline.md (e.g. "LR most stable under constraints," "RF intermediate performance") were ever verified computationally, or were similarly asserted without checking. All comparative claims in Section 5 should be treated as unverified until each is checked the same way this one just was.

**Resolved (September 14 2026):** Closed in both places. paper_draft.md Section 5.1 now states the pattern over the five true-accuracy evaluations, GB highest in three (Folktables 0.845, Student 0.658, MEPS 0.992; after Decisions 25 and 24 Folktables and MEPS read 0.756 and 0.859, with GB still highest) and LR highest in COMPAS (0.686) and FairGround-law_school_lequy (0.913). cross_domain_results_table.md's Table 1 Key Finding, which still read "GB is highest in every case" until today, was corrected to the same pattern. threshold_aggregation.py computes this count live rather than storing it, and was fixed the same day to include MEPS, so the printed figure now reads 3/5 instead of 2/4. The broader implication above was acted on during the Section 5 source-verification pass: every comparative claim carried into the manuscript is computed from the RESULTS dict rather than asserted.

## Decision 19, "DPD>0.2 Effective, DPD<0.05 Counterproductive" Effectiveness Threshold Is False as a Clean Rule; Has Real Exceptions in 2 of 7 Domains

**Investigation (Aug 4 2026):** While verifying claims for the paper's Introduction section, checked the outline's stated "core empirical finding" (ThresholdOptimizer is effective when baseline DPD > 0.2 and counterproductive when baseline DPD < 0.05) against GB's genuine baseline and post-DP DPD values across all 7 domains, extracted directly from the rebuilt RESULTS dictionary (see Decision 17).

**Finding:** The rule holds for 5 of 7 domains but has two direct exceptions:
- Folktables: baseline DPD=0.320 (>0.2, predicted effective) -> post-DP DPD=0.339, a +5.9% increase. The intervention made fairness worse despite baseline DPD exceeding the "effective" threshold.
- Lending Club: baseline DPD=0.024 (<0.05, predicted counterproductive) -> post-DP DPD=0.018, a -25% improvement. The intervention helped despite baseline DPD falling below the "counterproductive" threshold.

Domains where the rule does hold: COMPAS (0.857->0.571, improved), Law School (0.351->0.030, improved), FairGround (0.342->0.014, improved), Agricultural (0.009->0.031, worsened as predicted). Student (0.237->0.215) is a weak, borderline improvement (-9.3%) that technically fits but doesn't strongly confirm the rule either.

**Why this matters:** Like Decision 18, this is a claim about a pattern across the 7-domain comparison that was asserted as a clean empirical finding without the actual baseline-DPD-vs-post-DP-DPD comparison being run domain-by-domain against the real, deterministic (Decision 15-fixed) data. 2 of 7 exceptions is a real failure rate for a rule being presented as a general threshold in the paper's Introduction and Results sections.

**Status:** The threshold is a real, directionally useful heuristic for roughly 5 of 7 domains, not a universal rule. paper_outline.md (Introduction Section 1.3, Results Section 5.2) and the drafted Introduction text need to state this with the correct caveat (effectiveness depends on domain-specific factors beyond baseline DPD alone, with Folktables and Lending Club as documented exceptions) rather than presenting a clean two-threshold rule as if it held universally across all 7 domains.

**Resolved (September 14 2026):** Acted on and rescoped. The manuscript does not carry the domain-level "5 of 7" framing at all. Section 5.5 counts every model-domain pair instead of one representative model per domain: the constraint improved DPD in 9 of the 14 pairs with baseline DPD above 0.2 and worsened it in 3 of the 4 pairs below 0.05. Because three of those 14 high-baseline pairs come from law_school_lequy, which Section 6.4 shows is evaluated twice, Sections 1.3, 5.5 and 6.1 also state the duplicate-adjusted bracket of 6 of 11 to 7 of 12. The paper presents this as a heuristic for where to look, explicitly not a universal threshold, which is what this decision asked for. The counts held after the MEPS and Folktables reruns (Decisions 24 and 25), and on groups of at least 30 test records the high-disparity count is 13 of 14 (Decision 22).

---

## Decision 20, Stage 4 CUSUM Scored Pre-Deployment Baseline Values; Every Alert Fired at t=1

**Investigation (September 15 2026):** Reading Section 6 of the paper against fairness_drift_monitor.py showed the CUSUM running over the whole simulated series, including the ten baseline observations before the constrained model is deployed. Any model with a baseline DPD above 0.11 accumulated past the alert level on the first step. Every high-baseline domain, COMPAS and Folktables included, first alerted at t=1 with identical counts, so "Law School, FairGround and Student trigger the earliest alerts" was never true. Two more problems sat alongside it: random forest placeholder series for Law School, Lending Club and Agricultural, copies of the logistic regression values, were included in the alert counts, and the simulated shift moves each model 60% of the way back toward its baseline, so the size of every regression is set by the size of its correction.

**Fix:** Monitoring starts at deployment (t=10). Values are read from threshold_aggregation.RESULTS rather than literals, not-evaluated models are excluded, each series has its own deterministic seed, figure labels no longer call 0.1 DPD an EEOC threshold, and paper Figure 4 separates models already above 0.1 at deployment from regressions under the shift.

**Result:** Eight constrained models are flagged at deployment (all three COMPAS and Folktables models, Student's random forest and gradient boosting). Five meet 0.1 after the constraint and are flagged seven to eight steps into the shift (Law School's two models, FairGround's three, all law_school_lequy data). Eight are never flagged. The abstract and Sections 1.3, 3.6, 5.7, 6.3 and 7 now report this and state that the shift's construction decides which models regress past 0.1.

**Update (September 15 2026, later):** After the MEPS rerun (Decision 24), MEPS's random forest ends above 0.1 after the constraint, so nine models are flagged at or one step after deployment (Folktables' random forest at the second step), five under the shift, and seven never.

---

## Decision 21, DIR in the Lending Domains Measures Predicted Default, and One DIR Value Was Stale

**Finding (September 15 2026):** Lending Club and Agricultural both label Charged Off as 1, so the selection rate inside their DIR is the rate of being flagged for default, an adverse outcome, and the four-fifths reading does not transfer directly. Their ratios reached parity because the DP constraint raised gradient boosting's predicted default rate in every group: from 1 to 3% to about 40% in every Lending Club income quartile, against a 20% default rate, and from about 2% to between 16 and 19% for every Agricultural business type, against 5%. Agricultural's 0.653 to 1.042 is a move toward parity that ends just past it, not a move "past parity rather than toward it". Separately, Folktables' baseline DIR of 0.54 existed only as a hardcoded print string. Computed from the run, gradient boosting's per-race ratios against White under the equalized odds constraint start at Other 0.46 (0.68 after) and multiracial 0.74 (0.86 after), with Pacific Islanders, a group of 33, moving from 0.69 to 1.02.

**Fix:** Sections 3.5, 5.4 and 6.2 and the Figure 2 caption read each domain's ratio in the direction its outcome requires and report the rate changes behind the lending ratios.

**Update (September 15 2026, later the same day):** The Lending Club script now computes the observed default ratio instead of printing it: 1.44 for the lowest income quartile over the highest in the test split, with gradient boosting's baseline predicted default rate between 1.0% and 3.4% across quartiles, so the paper says "under 4%". The Folktables per-race ratios above were superseded when Folktables was rerun without POVPIP (Decision 25).

---

## Decision 22, COMPAS Metrics Are Set by Test Groups of Seven and One Records

**Finding (September 15 2026):** COMPAS's DPD and EOD are computed over all six race groups with no minimum group size, and the test split holds 7 Asian defendants and 1 Native American defendant. Every COMPAS DPD value in Table 2, and every baseline EOD value, has one of those two groups at an extreme. Gradient boosting's baseline DPD of 0.857 is Native American 1.000 minus Asian 0.143, and the identical 0.714 that logistic regression and random forest reach after the constraint is the Asian group's 5 of 7. On the four groups holding 1,227 of the 1,235 test records the constraint narrows the gap for all three models (logistic regression 0.385 to 0.187, random forest 0.202 to 0.163, gradient boosting 0.361 to 0.200), which would make the high-disparity tally 11 of 14 rather than 9. The paper's earlier explanation, that the impossibility result manifests once there are more than two groups, was wrong.

**Fix:** Table 2 keeps the values the pipeline computes. Section 6.2 reports the four-group measurement, Section 6.4 lists the missing minimum group size as a limitation, and src/compas_group_size_check.py reproduced every figure above (since replaced by src/group_size_check.py, see the update below).

**Open:** Recomputing COMPAS with a minimum group size would change Table 2 and the headline counts. That is a methodology decision to settle before submission.

**Update (September 15 2026, later the same day):** Folktables has the same problem. Its test split holds 5 Alaska Native respondents and 25 in the combined American Indian and Alaska Native category, and every Folktables value except the logistic regression and gradient boosting DPD baselines has one of those groups at an extreme. On the groups with at least 30 test records (19,970 of 20,000), the DP constraint narrows the gap for all three models after the Decision 25 rerun: 0.301 to 0.114, 0.276 to 0.177 and 0.302 to 0.078. With COMPAS and Folktables both measured this way, the high-disparity tally is 13 of 14, or 10 of 11 and 11 of 12 counting law_school_lequy once. src/group_size_check.py covers both domains and replaces src/compas_group_size_check.py. The open question stands for both domains; the paper keeps the pipeline's values in Table 2 and reports the group-size measurement in Section 6.2.

---

## Decision 23, Before-and-After Comparisons Also Change the Decision Objective

**Finding (September 15 2026):** Every ThresholdOptimizer call uses objective="balanced_accuracy_score", while the baseline models use scikit-learn's default decision threshold. Each before-and-after comparison therefore changes the decision objective as well as adding the fairness constraint, and accuracy costs and rate shifts such as Lending Club's include both effects.

**Fix:** Section 3.4 states the objective and Section 6.4 discloses the confound. Separating the two effects would need an unconstrained balanced-accuracy baseline, which this study did not run.

---

## Decision 24, MEPS Features Included the Columns That Define Its Label

**Finding (September 15 2026):** The FairGround file for meps_panel_19_fy2015 has 1,831 columns. Its label, UTILIZATION, is 1 exactly when OBTOTV15 + OPTOTV15 + ERTOT15 + IPNGTD15 + HHTOTD15 is 10 or more, which holds for all 15,830 rows, and stage2_fairground_threshold.py trained on every non-target column, those five counts included. That is why gradient boosting reached 0.992 accuracy. FairGround documents a 41-feature set for this dataset (Dataset.get_feature_columns, the standard AIF360 MEPS features) that leaves them out. baseline_fairground.py had the same problem through its top-50-variance selection, and compas_2_years in the FairGround pipeline carried recidivism outcome and COMPAS score columns (0.995 accuracy with them, 0.682 without); compas_2_years is not reported in the paper.

**Fix:** fairground_loader.documented_feature_columns returns FairGround's documented columns minus the target and the survey weight PERWT15F, and both FairGround scripts use it. law_school_lequy, creditcard and adult are unaffected, since every column in those files is documented, and their results reproduce exactly. MEPS moves from DPD 0.115/0.116/0.110 and accuracy 0.971/0.976/0.992 to DPD 0.069/0.089/0.092 and accuracy 0.855/0.857/0.859 for LR/RF/GB. Under the DP constraint LR and GB improve by 81% and 86% while RF worsens to 0.253 (Decision 26). Table 2, Sections 5 and 6, the drift results and the aggregation, cross-domain and drift figures were regenerated.

---

## Decision 25, Folktables Used POVPIP, Which Is Built From the Income Being Predicted

**Finding (September 15 2026):** The Folktables label is personal income above $50,000, and the Stage 2 and baseline feature sets included POVPIP, the family income-to-poverty ratio. Family income contains the person's own income, so POVPIP carries much of the label: its correlation with the label is 0.58, and gradient boosting's accuracy falls from 0.845 to 0.756 without it. No decision record gave a reason for including it.

**Fix:** POVPIP is dropped from the model features in stage2_folktables_threshold.py and baseline_folktables.py. The loader still reads it, so the population (1,589,032 records) and the sample are unchanged, and EDA still describes it. New Stage 2 values for LR/RF/GB: baseline DPD 0.301/0.290/0.302, post-DP 0.348/0.193/0.373; baseline EOD 0.728/0.732/0.767, post-EO 0.314/0.836/0.417. The high-disparity tally stays at 9 of 14, but the Folktables pair that improves is now random forest rather than logistic regression. Under the EO constraint gradient boosting's DIR against White crosses 0.8 for Black (0.72 to 0.83), multiracial (0.70 to 0.92), American Indian (0.61 to 1.00) and Pacific Islander respondents (0.72 to 1.03), and Other rises from 0.42 to just under 0.8.

---

## Decision 26, Thresholds Are Fit on the Training Split, Which Misleads for Random Forest

**Finding (September 15 2026):** Every Stage 2 script passes the training split to ThresholdOptimizer, which refits the model and chooses thresholds on the records it was trained on, as Fairlearn's documented examples do. Random forest fits those records almost perfectly (training accuracy 1.000 in Student, 0.996 in MEPS), so its thresholds transfer poorly. Chosen on a held-out quarter of the training split instead, random forest's DPD under the DP constraint goes from 0.235 to 0.007 in Student and 0.086 to 0.037 in MEPS, where the Stage 2 runs worsen it to 0.363 and 0.253. For logistic regression and gradient boosting the held-out design gives smaller improvements in all four cases.

**Decision:** Keep the Stage 2 design, which matches Fairlearn's documented usage, and report the held-out check as a limitation (paper Section 6.4) rather than switching designs two weeks before submission. With Decision 22's group-size measurement, each of the five high-disparity exceptions reverses under one of the two checks. src/threshold_holdout_check.py reproduces the figures.

---

## Decision 27, Four Literature Claims Corrected Against the Sources

**Finding (September 15 2026):** Reading the cited sources again turned up four claims the paper and these records overstated.
- Breck et al. (2017) do include a fairness item, Model 7, "the model has been tested for considerations of inclusion". What their rubric lacks is a fairness test among its seven monitoring tests. "Zero fairness tests" in literature_analysis.md and research_design_rationale.md is wrong.
- Ajarra and Basu (2026) is a theoretical paper on auditing statistical parity when a model owner updates the model: sample complexity and which updates preserve the property. It does not show that fairness changes while accuracy stays stable, as the paper, outline and literature analysis said.
- Sariola et al. (2026) found that equalizing base rates looked like parity on traditional measures but left about 10% absolute disparity when measured with audit-study data. It is a measurement result, not "one metric masking another".
- "No existing framework offers continuous fairness monitoring" is false: Amazon SageMaker Clarify monitors bias metrics on live data, window by window, with confidence intervals and alerts. The open-source toolkits AIF360 and Fairlearn have no monitoring. On cross-domain evaluation, Friedler et al. (2019) and Chen et al. (2023) compare interventions across several benchmark datasets, and Chen et al. include equalized odds post-processing on the five AIF360 datasets, so "no prior work evaluated one post-processing intervention across many contexts" also overstated.

**Fix:** Paper Sections 1.1, 1.2, 1.3, 2.2, 2.3 and 2.4 now state what each source shows, cite Friedler et al. (2019), Chen et al. (2023) and the SageMaker Clarify documentation, and describe Stage 4 as an open, sequential alternative to per-window monitoring. The README follows. literature_review.md, literature_analysis.md and research_design_rationale.md now correct these statements in place, each with a dated note at the top.
