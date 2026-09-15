# FAPE, Research Design Rationale

> **Record note, September 15 2026.** Written February to May 2026 and corrected in place on September 15 2026 where it misdescribed a source, a legal standard or what was built: the Breck et al. (2017), Ajarra and Basu (2026) and Sariola et al. (2026) summaries, the claim that no monitoring tool exists, the mapping of the 0.8 ratio to ECOA, the calibration tradeoff FAPE never measured, the unbuilt proxy detection and MIMIC-III loader, and the Lending Club attributes. The design reasoning itself is kept as it was written. Decisions 24 to 27 in methodology_decisions.md record the later fixes, and docs/paper_draft.md is the current statement of the study.

## Why We Built It This Way

**Period:** February 2026, May 2026
**Researcher:** Nithin Narla
**Status:** Complete, major design decisions documented before Phase 4 implementation

---

## Why I'm Writing This Down

Design decisions made in February look obvious in May and completely mysterious in September. I've watched research projects lose track of why they made specific choices, then spend weeks in paper writing trying to reconstruct reasoning that should have been captured when it was fresh. This document is insurance against that.

The other reason: I need to be able to defend every decision in a review or across a table from a professor. "We used GradientBoostingClassifier because everyone uses it" is not a defense. (Note: XGBoost was considered but sklearn GradientBoostingClassifier was chosen for reproducibility, see Decision 2.) Every choice in FAPE's design has a specific reason and I want to be able to articulate it without hesitation.

---

## The Research Question, Why Causal and Not Something Easier

I spent more time on the research question framing than on any other single design decision. It determines what counts as evidence, what the paper can legitimately claim, and what a reviewer can legitimately push back on.

The easy version of this question is Descriptive: document what fairness looks like across multiple domains. That would be a contribution, but a limited one: it tells practitioners what the problem looks like without saying what to do about it. The Comparative version, compare FAPE against existing frameworks, is defensible but it puts the paper in a direct competition with AIF360 and Fairlearn that I'd rather avoid framing it as.

The Causal framing is harder to defend and more valuable if it holds: does applying post-processing fairness constraints cause measurable bias reduction across heterogeneous high-stakes deployment domains simultaneously, at acceptable accuracy cost? This is the question practitioners need answered. They're not asking whether bias exists, Angwin et al. (2016) answered that. They're asking whether an intervention works, and whether it works consistently across the contexts they actually deploy in.

The causal framing has a real constraint: the baseline model has to be identical across all domains. Same LR/RF/GB architecture (sklearn), same hyperparameters, same training procedure. If I let domain-specific tuning creep in, I can't attribute fairness differences to the intervention, because they could be model configuration artifacts. I accept that constraint because it's the only way to keep the cross-domain comparison clean.

---

## The 4-Stage Pipeline, The Decision Behind Each Stage

**Stage 1, Why proxy detection before model training**

The Obermeyer et al. (2019) finding is what drove Stage 1 into the design. Healthcare cost used as a proxy for health need, the racial bias in the algorithm's outputs wasn't visible in any explicit feature, only in the relationship between cost and actual health need. Running a fairness audit on model outputs while ignoring proxy relationships in the input features would mean auditing the symptom and missing the cause.

**Correction, 2026-08-17:** the Cramér's V feature-vs-sensitive-attribute proxy detection described below was planned here but was never implemented in any of the 7 domain scripts (confirmed by direct code search, zero matches). What Stage 1 computes is feature-vs-label correlation (e.g. eda_compas.py's priors_count-vs-recidivism correlation), which measures predictive power, not proxy relationship to a sensitive attribute. This document's original design intent is preserved below for historical accuracy, but readers should treat the Cramér's V proxy-detection step as unbuilt, not as a description of the current pipeline. The paper draft itself (paper_draft.md) does not claim this analysis was performed, so this gap has not propagated into the submitted work, but it should be resolved before Stage 1 is described in any future methodology writing: either implement the original Cramér's V step, or formally drop it from the design rationale as a considered-but-abandoned idea.

**Resolved (September 15 2026):** Dropped. The README describes Stage 1 as EDA base-rate comparisons and per-domain preprocessing, and the paper claims no proxy analysis.

**Stage 2, Why LR/RF/GB and why default hyperparameters**

LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier (sklearn) across all seven domains (eight evaluations once MEPS was added). Three architectures capture the spectrum from linear to ensemble, LR as interpretable baseline, RF as bagging ensemble, GB as boosting ensemble. XGBoost was considered but excluded for simpler reproducibility with no additional dependency.

Default hyperparameters is the more interesting constraint. Domain-specific tuning would improve accuracy numbers, probably meaningfully. But tuned models across domains would mean I can't isolate the fairness intervention as the variable being tested. Any fairness difference across domains could be a tuning artifact rather than a genuine domain difference. Default hyperparameters keeps the experimental design clean at the cost of headline accuracy numbers.

**Stage 3, Why post-processing and not in-processing**

In eight years of production ML I have never worked in an environment where I owned the model. You inherit it from a vendor, from a previous team, from a partner organization. In-processing fairness constraints require retraining, which means you need to own the training pipeline. Post-processing works on any model regardless of how it was built.

Fairlearn's ThresholdOptimizer is the specific implementation because it's the cleanest operationalization of Hardt et al.'s (2016) equalized odds approach. It applies the constraint post-training without modifying the underlying model. The limitation is that the thresholds are fitted to one model and have to be refitted whenever the model changes; Ajarra and Basu (2026) study the related problem of auditing fairness when model owners keep updating their models. Stage 4 exists partly because of this limitation.

Four metrics simultaneously, demographic parity, equalized odds, disparate impact ratio, and accuracy cost, because after Sariola et al. (2026) I can't in good conscience report one metric and call it done. They found that equalizing base rates looked like parity on traditional measures but left about 10% disparity when measured with audit-study data. The paper reports demographic parity and equalized odds everywhere, and the other two where a domain's pipeline computes them. Individual fairness score excluded, requires domain-specific similarity metric incompatible with cross-domain comparison. See Decision 9.

**Stage 4, Why deployment monitoring exists at all**

Sculley et al. (2015) said production ML systems degrade silently. Breck et al. (2017) built a 28-test production readiness rubric in which fairness appears once, as a pre-release inclusion test, and never among the monitoring tests. Ajarra and Basu (2026) worked out how to audit fairness when models are updated. Commercial platforms such as Amazon SageMaker Clarify now monitor bias on live data, but the open-source toolkits most research uses have nothing like it. Stage 4 is my attempt at an open, sequential version.

The honest limitation: I don't have access to a live production system. Stage 4 uses synthetic distribution shift to simulate deployment conditions. The drift detection is proof-of-concept, not production-validated. The paper will say this explicitly.

---

## Dataset Selection, The Reasoning Behind Each Choice

**COMPAS (6,172 records verified), Criminal justice**

COMPAS is required as the field's primary validation benchmark. Any fairness paper that doesn't engage with it will be questioned in review, and ProPublica's documented racial disparities provide known ground truth I can validate against before running a single experiment. I know what the bias looks like. If FAPE doesn't find it, something is wrong with the framework.

The limitation I'll state explicitly: one algorithm, one county in Florida, 2013-2014. This is not a representative sample of criminal justice AI. Using it for field comparability, not for generalization.

**Folktables ACS (1,589,032 records verified), Socioeconomic**

Ding et al. (2021) showed Adult Income is methodologically flawed. Using Adult Income in 2026 after that finding would be defending a benchmark I know has problems. Folktables ACS uses US Census American Community Survey data, income, employment, and mobility prediction tasks across all 50 states. 1.58 million records compared to Adult Income's 48K. The scale difference alone gives FAPE more statistical power for the fairness comparisons.

**FairGround Corpus (1,964,010 records verified), Multi-domain**

44 fairness-annotated datasets, of which 38 load without the large-dataset option and 37 of those prepare under the pinned numpy, the 1,964,010 records above. What Simson et al. (2025) built is a pre-processed collection of datasets with sensitive attributes and fairness metadata already identified, work that would have taken months to do from scratch across this many domains. I haven't found another paper that uses FairGround as part of a multi-domain evaluation framework.

**Student Performance (1,044 records verified), Education**

395 records in the math variant, 649 in Portuguese, 1,044 combined. The small scale is deliberate. A fairness auditing framework that only works at production scale, millions of records, isn't useful for most real-world audits. Small dataset fairness is a specific challenge because demographic subgroups can be too small for reliable metric estimation. If FAPE's metrics degrade at this scale, that's a finding worth reporting.

**Law School Admissions (18,692 records verified), Education/Legal**

Race as the constrained attribute and sex as a secondary check, bar passage as outcome. This is a domain where fairness and meritocracy claims collide in practice, law school admissions processes explicitly use predictive models and the stakes are high. Having two education datasets with different characteristics (1,044 vs 18,692 records, different sensitive attribute distributions) strengthens the within-domain variation analysis.

**Lending Club (1,348,099 records verified), Financial**

1.35 million records, production scale. The design challenge: Lending Club doesn't collect race or gender, which is typical in financial services. FAPE uses socioeconomic attributes instead, income band as the audited attribute with home ownership as a secondary check, consistent with what the fairness literature uses when protected attributes aren't available. This is the real-world scenario where proxy detection in Stage 1 would matter most, which is why its absence (see the correction above) is a limitation.

MIMIC-III is planned for healthcare but requires PhysioNet credentialed registration. Access pending. Healthcare is the domain where Obermeyer et al. documented the most consequential bias mechanism and I want it in the evaluation. [Access never came through and no loader was built. Healthcare is covered by MEPS Panel 19 through the FairGround corpus.]

Agricultural domain confirmed, USDA NASS Census 2022 (7,334 aggregate rows), SBA 7(a) NAICS-11 loans (15,845 individual records), and LSMS-ISA Nigeria Wave 4 (30,312 farm households) all verified and loaded. [Decision 11 later kept only SBA 7(a) for modeling.]

Searched for fairness papers on agricultural lending and farm household outcomes before committing to this domain and found none. Small farmers, agricultural loan applicants, farm households in developing economies, populations making consequential decisions increasingly mediated by algorithmic systems, and I found no evaluation of the fairness implications.

Three datasets covering different terrain. USDA NASS provides aggregate racial baseline on US farm ownership, not individual training data, but ground truth for what racial disparity in agricultural access looks like. SBA 7(a) provides individual-level agricultural loan records with binary default outcomes. LSMS-ISA Nigeria Wave 4 provides 30,312 farm household records from a context where the fairness literature has almost no presence, sex and education as sensitive attributes, food security as outcome. The only large-scale publicly downloadable individual-level agricultural dataset with demographic attributes I could find.

---

## Fairness Metric Selection, Why These Four

**Demographic parity difference**, positive outcome rates equal across groups. Simplest metric, the difference form of the comparison the four-fifths rule makes as a ratio. Limitation: can be gamed by lowering outcomes for the advantaged group. Report it because practitioners expect it, not because it's the most informative.

**Equalized odds difference**, true positive and false positive rates equal across groups. What Hardt et al. (2016) formalized. More demanding than demographic parity. The impossibility result applies here: calibration and equal error rates cannot generally hold together when base rates differ, except under narrow conditions. FAPE does not measure calibration, so it names this tradeoff rather than quantifying it.

**Disparate impact ratio**, positive outcome rate for disadvantaged group divided by advantaged group. The 0.8 line comes from the EEOC's four-fifths rule for employment. ECOA does not adopt it, so FAPE treats 0.8 as a research convention everywhere, including Lending Club and SBA 7(a), where the outcome is default and a ratio above 1.0 is the adverse direction.

**Accuracy cost**, baseline_acc minus constrained_acc. The practical price paid for applying a fairness constraint. This is the metric practitioners face in deployment decisions. Individual fairness score (IFS) was originally planned here but excluded, Dwork et al. (2012) identified that defining "similar" requires a task-specific similarity metric that cannot be generalized across FAPE's domains. See Decision 9.

---

## What This Design Cannot Do

Three things FAPE cannot claim that I want to be explicit about before writing starts.

Cannot solve the fairness problem. Chouldechova's impossibility theorem is mathematics, not a limitation of the current implementation. Calibration and equal error rates cannot generally hold together when base rates differ. FAPE makes the tradeoff visible, it doesn't eliminate it.

Cannot generalize from these domains to all high-stakes ML contexts. Seven domains is a meaningful sample, not an exhaustive one. [Now eight evaluations from seven data sources, with healthcare covered by one survey dataset, MEPS.] The paper will be explicit about coverage and resist overclaiming generalizability.

Cannot replace domain expertise. Which metric to prioritize given a specific regulatory context, whether a bias pattern constitutes actionable harm, how to weigh accuracy against fairness given the stakes, these require human judgment. FAPE surfaces information; it doesn't make decisions.
