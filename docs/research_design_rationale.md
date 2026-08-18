# FAPE, Research Design Rationale
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

The easy version of this question is Descriptive: document what fairness looks like across multiple domains. That's a contribution but a limited one, it tells practitioners what the problem looks like, not what to do about it. The Comparative version, compare FAPE against existing frameworks, is defensible but it puts the paper in a direct competition with AIF360 and Fairlearn that I'd rather avoid framing it as.

The Causal framing is harder to defend and more valuable if it holds: does applying post-processing fairness constraints cause measurable bias reduction across heterogeneous high-stakes deployment domains simultaneously, at acceptable accuracy cost? This is the question practitioners actually need answered. They're not asking whether bias exists, Angwin et al. (2016) answered that. They're asking whether an intervention works, and whether it works consistently across the contexts they actually deploy in.

The causal framing has a real constraint: the baseline model has to be identical across all domains. Same LR/RF/GB architecture (sklearn), same hyperparameters, same training procedure. If I let domain-specific tuning creep in, I can't attribute fairness differences to the intervention, they could be model configuration artifacts. That's a constraint I'm accepting because it's the only way to make the cross-domain comparison clean.

---

## The 4-Stage Pipeline, The Decision Behind Each Stage

**Stage 1, Why proxy detection before model training**

The Obermeyer et al. (2019) finding is what drove Stage 1 into the design. Healthcare cost used as a proxy for health need, the racial bias in the algorithm's outputs wasn't visible in any explicit feature, only in the relationship between cost and actual health need. Running a fairness audit on model outputs while ignoring proxy relationships in the input features would mean auditing the symptom and missing the cause.

**Correction, 2026-08-17:** the Cramér's V feature-vs-sensitive-attribute proxy detection described below was planned here but was never actually implemented in any of the 7 domain scripts (confirmed by direct code search, zero matches). What Stage 1 actually computes is feature-vs-label correlation (e.g. eda_compas.py's priors_count-vs-recidivism correlation), which measures predictive power, not proxy relationship to a sensitive attribute. This document's original design intent is preserved below for historical accuracy, but readers should treat the Cramér's V proxy-detection step as unbuilt, not as a description of the current pipeline. The paper draft itself (paper_draft.md) does not claim this analysis was performed, so this gap has not propagated into the submitted work, but it should be resolved before Stage 1 is described in any future methodology writing: either implement the original Cramér's V step, or formally drop it from the design rationale as a considered-but-abandoned idea.

**Stage 2, Why LR/RF/GB and why default hyperparameters**

LogisticRegression, RandomForestClassifier, and GradientBoostingClassifier (sklearn) across all seven domains. Three architectures capture the spectrum from linear to ensemble, LR as interpretable baseline, RF as bagging ensemble, GB as boosting ensemble. XGBoost was considered but excluded for simpler reproducibility with no additional dependency.

Default hyperparameters is the more interesting constraint. Domain-specific tuning would improve accuracy numbers, probably meaningfully. But tuned models across domains would mean I can't isolate the fairness intervention as the variable being tested. Any fairness difference across domains could be a tuning artifact rather than a genuine domain difference. Default hyperparameters keeps the experimental design clean at the cost of headline accuracy numbers.

**Stage 3, Why post-processing and not in-processing**

In eight years of production ML I have never worked in an environment where I owned the model. You inherit it from a vendor, from a previous team, from a partner organization. In-processing fairness constraints require retraining, which means you need to own the training pipeline. Post-processing works on any model regardless of how it was built.

Fairlearn's ThresholdOptimizer is the specific implementation because it's the cleanest operationalization of Hardt et al.'s (2016) equalized odds approach. It applies the constraint post-training without modifying the underlying model. The limitation, which Ajarra et al. (2026) made explicit, is that the constraint has to be reapplied every time the model updates. Stage 4 exists partly because of this limitation.

Four metrics simultaneously, demographic parity, equalized odds, disparate impact ratio, and accuracy cost, because after Sariola et al. (2026) I can't in good conscience report one metric and call it done. Optimizing for one can mask 10% disparity on another. The paper shows all four and shows the tradeoffs. Individual fairness score excluded, requires domain-specific similarity metric incompatible with cross-domain comparison. See Decision 9.

**Stage 4, Why deployment monitoring exists at all**

Sculley et al. (2015) said production ML systems degrade silently. Breck et al. (2017) built a 28-test production readiness rubric with zero fairness tests. Ajarra et al. (2026) confirmed fairness specifically degrades under model updates. Eleven years of documented knowledge and no fairness framework has built the monitoring infrastructure. Stage 4 is my attempt to close that gap.

The honest limitation: I don't have access to a live production system. Stage 4 uses synthetic distribution shift to simulate deployment conditions. The drift detection is proof-of-concept, not production-validated. The paper will say this explicitly.

---

## Dataset Selection, The Reasoning Behind Each Choice

**COMPAS (6,172 records verified), Criminal justice**

Non-negotiable. The field's primary validation benchmark. Any fairness paper that doesn't engage with COMPAS will be questioned in review, ProPublica's documented racial disparities provide known ground truth I can validate against before running a single experiment. I know what the bias looks like. If FAPE doesn't find it, something is wrong with the framework.

The limitation I'll state explicitly: one algorithm, one county in Florida, 2013-2014. This is not a representative sample of criminal justice AI. Using it for field comparability, not for generalization.

**Folktables ACS (1,589,032 records verified), Socioeconomic**

Ding et al. (2021) showed Adult Income is methodologically flawed. Using Adult Income in 2026 after that finding would be defending a benchmark I know has problems. Folktables ACS uses US Census American Community Survey data, income, employment, and mobility prediction tasks across all 50 states. 1.58 million records compared to Adult Income's 48K. The scale difference alone gives FAPE more statistical power for the fairness comparisons.

**FairGround Corpus (1,964,010 records verified), Multi-domain**

44 fairness-annotated datasets. What Fabris et al. (2025) built is essentially a pre-processed collection of datasets with sensitive attributes and fairness metadata already identified, work that would have taken months to do from scratch across this many domains. FAPE is the first paper to use FairGround as part of a multi-domain evaluation framework. That's a contribution worth noting in the paper.

**Student Performance (1,044 records verified), Education**

395 records in the math variant, 649 in Portuguese, 1,044 combined. The small scale is deliberate. A fairness auditing framework that only works at production scale, millions of records, isn't useful for most real-world audits. Small dataset fairness is a specific challenge because demographic subgroups can be too small for reliable metric estimation. If FAPE's metrics degrade at this scale, that's a finding worth reporting.

**Law School Admissions (18,692 records verified), Education/Legal**

Race and sex as sensitive attributes, bar passage as outcome. This is a domain where fairness and meritocracy claims collide in practice, law school admissions processes explicitly use predictive models and the stakes are high. Having two education datasets with different characteristics (1,044 vs 18,692 records, different sensitive attribute distributions) strengthens the within-domain variation analysis.

**Lending Club (1,348,099 records verified), Financial**

1.35 million records, production scale. The design challenge: Lending Club doesn't collect race or gender, which is typical in financial services. FAPE uses socioeconomic proxies, geographic indicators, income, employment stability, housing status, consistent with what the fairness literature uses when protected attributes aren't available. This is the real-world scenario where proxy detection in Stage 1 matters most.

MIMIC-III is planned for healthcare but requires PhysioNet credentialed registration. Access pending. Healthcare is the domain where Obermeyer et al. documented the most consequential bias mechanism and I want it in the evaluation. If it doesn't come through before writing time, the paper will include the loader and note the access gap.

Agricultural domain confirmed, USDA NASS Census 2022 (7,334 aggregate rows), SBA 7(a) NAICS-11 loans (15,845 individual records), and LSMS-ISA Nigeria Wave 4 (30,312 farm households) all verified and loaded.

Searched for fairness papers on agricultural lending and farm household outcomes before committing to this domain. Nothing exists. The domain is completely absent from the fairness literature. Small farmers, agricultural loan applicants, farm households in developing economies, populations making consequential decisions increasingly mediated by algorithmic systems, and nobody has evaluated the fairness implications.

Three datasets covering different terrain. USDA NASS provides aggregate racial baseline on US farm ownership, not individual training data, but ground truth for what racial disparity in agricultural access actually looks like. SBA 7(a) provides individual-level agricultural loan records with binary default outcomes. LSMS-ISA Nigeria Wave 4 provides 30,312 farm household records from a context where the fairness literature has essentially no presence, sex and education as sensitive attributes, food security as outcome. The only large-scale publicly downloadable individual-level agricultural dataset with demographic attributes I could find.

---

## Fairness Metric Selection, Why These Four

**Demographic parity difference**, positive outcome rates equal across groups. Simplest metric, maps directly to EEOC 80% rule. Limitation: can be gamed by lowering outcomes for the advantaged group. Report it because practitioners expect it, not because it's the most informative.

**Equalized odds difference**, true positive and false positive rates equal across groups. What Hardt et al. (2016) formalized. More demanding than demographic parity. The Chouldechova constraint applies here, cannot simultaneously satisfy equalized odds and calibration when base rates differ. FAPE will report this tradeoff explicitly.

**Disparate impact ratio**, positive outcome rate for disadvantaged group divided by advantaged group. Below 0.8 triggers ECOA adverse impact standard in financial services. Including this gives FAPE direct regulatory mapping for Lending Club and SBA 7(a) results.

**Accuracy cost**, baseline_acc minus constrained_acc. The practical price paid for applying a fairness constraint. This is the metric practitioners actually face in deployment decisions. Individual fairness score (IFS) was originally planned here but excluded, Dwork et al. (2012) identified that defining "similar" requires a task-specific similarity metric that cannot be generalized across FAPE's seven domains. See Decision 9.

---

## What This Design Cannot Do

Three things FAPE cannot claim that I want to be explicit about before writing starts.

Cannot solve the fairness problem. Chouldechova's impossibility theorem is mathematics, not a limitation of the current implementation. Satisfying equalized odds and calibration simultaneously is impossible when base rates differ. FAPE makes the tradeoff visible, it doesn't eliminate it.

Cannot generalize from these domains to all high-stakes ML contexts. Seven domains is a meaningful sample, not an exhaustive one. Healthcare is planned but pending. The paper will be explicit about coverage and resist overclaiming generalizability.

Cannot replace domain expertise. Which metric to prioritize given a specific regulatory context, whether a bias pattern constitutes actionable harm, how to weigh accuracy against fairness given the stakes, these require human judgment. FAPE surfaces information; it doesn't make decisions.
