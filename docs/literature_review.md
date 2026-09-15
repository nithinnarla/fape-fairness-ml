# FAPE, Literature Review Notes

> **Record note, September 15 2026.** Written November 2025 to January 2026 and corrected in place on September 15 2026 where it misdescribed a source or FAPE itself: the Breck et al. (2017), Kleinberg et al. (2018), Lambrecht and Tucker (2019) and Berk et al. (2021) entries; the claims that no tool monitors fairness in production and that no study compares interventions across datasets, now checked against Amazon SageMaker Clarify, Friedler et al. (2019) and Chen et al. (2023); and FAPE's own description (eight evaluations from seven data sources, baseline classification rather than ensemble modeling, no individual fairness metric). Decision 27 in methodology_decisions.md records the source checks, and docs/paper_draft.md is the current statement of the study.

## Fairness Auditing in Production ML Systems

**Period:** November 2025, January 2026
**Researcher:** Nithin Narla
**Status:** Complete, informed FAPE framework design

---

## Why I Started Looking At This

Eight years of building ML pipelines across industries (financial services, healthcare, workforce analytics), and the same problem kept showing up. A model ships with clean aggregate metrics and happy stakeholders, and six months later someone notices the error rate in one demographic group is twice what it is in another. Nobody was careless; nobody was measuring the right thing.

I started pulling on that thread in November 2025. What I found in the literature surprised me, not because the problem was undocumented, but because the gap between what researchers had solved and what happens in production was enormous. This document captures what I read, what I found useful, what I found frustrating, and what I couldn't find at all.

---

## 1. Literature Review
### Overview of Prior Research on Algorithmic Fairness

The fairness in ML literature has exploded since 2016. ProPublica's COMPAS investigation was the spark, suddenly everyone had a concrete, high-stakes example of algorithmic bias with real human consequences. What followed was a decade of theoretical work that produced important results but also, I think, some significant blind spots.

**Foundational papers that shaped my thinking:**

**Chouldechova (2017), Fair Prediction with Disparate Impact**
This result forced a fundamental reframing of what FAPE could claim to achieve. The mathematical proof that you cannot simultaneously satisfy calibration and equalized odds when base rates differ between groups is not a limitation of current methods, it's a fundamental impossibility. Reading this early forced me to think about FAPE differently. The goal can't be "achieve fairness", it has to be "audit which fairness properties are achievable in which contexts and at what cost." That reframing is central to everything FAPE does.

**Hardt, Price & Srebro (2016), Equality of Opportunity**
The cleanest formalization of equalized odds I found. What's useful here beyond the definition is the post-processing approach: you can take any trained model and apply a fairness constraint after the fact without retraining, which is what FAPE's Stage 3 does. The paper doesn't test whether this holds across domains, and that is the gap FAPE takes up.

**Dwork et al. (2012), Fairness Through Awareness**
Individual fairness: similar people should get similar predictions. The idea is philosophically appealing and practically difficult, because defining "similar" requires a task-specific metric nobody agrees on. I planned to include it in FAPE's evaluation, but a similarity function that works on COMPAS is not the one Lending Club or Student Performance needs, and the domain-specific assumptions that would take undermine the cross-domain comparison the paper is built on. Dropped for that reason (Decision 9).

**Barocas & Hardt (2017), NeurIPS Tutorial**
Best taxonomy of fairness definitions I found. For FAPE, the lesson is that fairness is not one thing. Different metrics capture different moral intuitions. A production auditing framework needs to surface all of them and let domain context determine which ones matter instead of picking one and declaring victory.

---

**COMPAS and criminal justice literature:**

**Angwin et al. (2016), Machine Bias, ProPublica**
The paper that started everything. What I appreciate about it beyond the findings is the methodology: they obtained the actual COMPAS scores, matched them to outcomes, and did the analysis themselves instead of taking Northpointe's word for it. FAPE tries to bring the same habit to enterprise fairness auditing, where teams should measure fairness themselves instead of trusting a vendor's dashboard.

**Dressel & Farid (2018), The Accuracy, Fairness, and Limits of Predicting Recidivism**
Crowdsourced human predictions matched COMPAS accuracy. That finding is more unsettling than it first appears: it suggests the decision-making context itself is biased and algorithms inherit that bias. FAPE can't fix the context, but it can make the resulting bias visible and measurable.

**Kleinberg et al. (2018), Human Decisions and Machine Predictions**
Judges make worse bail decisions than algorithms on accuracy. But accuracy isn't the only thing that matters in criminal justice, legitimacy, transparency, and equal treatment matter too. This tension between accuracy and fairness runs through everything in FAPE.

---

**Cross-domain bias literature:**

**Obermeyer et al. (2019), Dissecting Racial Bias in Healthcare**
The most important paper I read after Chouldechova. A widely deployed healthcare algorithm systematically underestimated Black patients' illness severity, not because of explicit racial features, but because it used healthcare cost as a proxy for health need. The bias was invisible until someone looked for it, which is the production deployment problem FAPE is designed to surface. It also made healthcare a required domain in the evaluation: a framework that can't catch what happened here isn't useful.

**Lambrecht & Tucker (2019), Algorithmic Bias in Ad Delivery**
Bias emerged from economic optimization, not discriminatory intent. An ad for STEM jobs reached fewer women because young women are costlier to reach, so delivery that bought impressions cost-effectively went to men. Nobody programmed it to discriminate. This paper solidified my thinking that bias in production ML is usually emergent rather than designed. Which means auditing after the fact is necessary, not optional.

---

**Production ML systems literature:**

**Sculley et al. (2015), Hidden Technical Debt in ML Systems**
Not a fairness paper, but possibly the most consequential for FAPE's design. The argument that production ML systems degrade silently over time, through feature drift, dependency changes, data shifts, maps directly onto fairness. A model that passed a fairness audit at deployment will not necessarily pass one six months later. This motivated Stage 4 of FAPE: deployment monitoring isn't optional.

**Mitchell et al. (2019), Model Cards**
A good idea with limited execution: static documentation snapshots don't capture fairness drift over time. Useful as a starting point but not sufficient for production environments. FAPE is partly an answer to the question: what would Model Cards look like if they were continuous and cross-domain rather than static and single-model?

**Breck et al. (2017), ML Test Score**
Google's production readiness rubric has 28 tests. One pre-release model test asks whether the model has been tested for considerations of inclusion, but none of the seven monitoring tests covers fairness. A fairness check at launch with nothing after it is the gap FAPE's Stage 4 is meant to address.

---

## 2. Systematic Review
### Rigorous Analysis of Fairness Frameworks and Tools

**Search scope:** Papers on fairness in ML published 2016-2025, focusing on empirical evaluation frameworks, production deployment, and multi-domain studies

**Inclusion criteria:**
- Empirical evaluation on real datasets
- Addresses demographic fairness explicitly
- Applicable to tabular classification (FAPE's domain)
- Published in peer-reviewed venues or reputable preprint servers

**Exclusion criteria:**
- Purely theoretical without empirical validation
- Image/NLP fairness only (different feature spaces)
- Single protected attribute only

**Key frameworks evaluated:**

| Framework | Approach | Data Tested | Post-deployment Monitoring | Cross-Domain |
|-----------|----------|-------------|---------------------------|--------------|
| AIF360 (IBM) | Pre/in/post-processing | Bundled benchmarks (Adult, COMPAS, German Credit, Bank Marketing, MEPS) | No | No |
| Fairlearn (Microsoft) | Post-processing + reductions | Bundled benchmark datasets | No | No |
| What-If Tool (Google) | Visualization + analysis | Single model | No | No |
| Aequitas | Audit reporting | Criminal justice and public policy case studies | No | No |
| Amazon SageMaker Clarify | Bias metrics on deployed models (commercial) | Any deployed model | Yes, per monitoring window | No |
| FAPE (this work) | Post-processing audit + sequential monitor | 8 evaluations, 7 data sources | Simulated shift only | Yes |

**Critical finding from systematic review:**
The open-source frameworks were designed for research and development contexts. None of them addresses the operational side of production deployment: continuous monitoring, feature drift, model versioning, multi-domain generalization. Commercial platforms cover part of this, and SageMaker Clarify monitors bias metrics on live data, but nothing open tests a fairness monitor alongside an intervention across domains. That gap is the primary justification for FAPE.

**Methodology quality assessment:**
- AIF360 and Fairlearn have strong algorithmic foundations but ship with legacy benchmark datasets such as Adult Income and COMPAS
- Ding et al. (2021) demonstrated Adult Income has serious flaws as a fairness benchmark, yet it remains the default in most frameworks
- No framework evaluates whether a fairness constraint's effect generalizes across domains; comparative studies (Friedler et al. 2019; Chen et al. 2023) compare methods on a handful of benchmark datasets instead

---

## 3. Scoping Review
### Extent and Nature of Cross-Domain Fairness Research

**Question:** How much of the existing fairness literature addresses cross-domain generalization?

**Finding:** Almost none.

I searched for papers that explicitly tested whether fairness interventions developed in one domain transfer to another. The literature is almost entirely single-domain. The closest work:

- **Friedler et al. (2019)**, Comparative study of fairness algorithms across datasets, but datasets are all from similar domains and the study doesn't frame cross-domain generalization as the research question
- **Chen et al. (2023)**, Seventeen bias mitigation methods, equalized odds post-processing among them, compared on the five AIF360 benchmark datasets (income, criminal justice, credit, bank marketing, healthcare); it asks which method does best, not whether one method's effect transfers across domains
- **Berk et al. (2021)**, Review of fairness definitions and their tradeoffs in criminal justice risk assessment, within one domain
- **Wachter et al. (2021)**, Legal analysis of fairness across EU regulatory domains, conceptual, not empirical

**Conclusion from scoping review:**
The cross-domain generalization question is still open. The closest study, Chen et al. (2023), covers criminal justice, healthcare and finance but not education, and compares methods rather than following one intervention across domains. FAPE takes up that question.

---

## 4. Meta-Analysis
### Quantitative Patterns Across Fairness Studies

Pulled quantitative results from 23 papers reporting demographic parity difference, equalized odds difference, or disparate impact ratio on COMPAS or Adult Income datasets.

**Key patterns observed:**

**Pattern 1, Accuracy-fairness tradeoff is real but variable**
Across papers, post-processing fairness constraints reduce accuracy by 1-8% on COMPAS. The range is wide because it depends heavily on which fairness metric is being optimized and at what threshold. Papers that report clean tradeoffs are usually optimizing for one metric, papers that try to satisfy multiple metrics simultaneously show steeper accuracy costs.

**Pattern 2, African-American/white disparity in COMPAS is robust**
Across every paper that reports it, the false positive rate disparity between African-American and white defendants ranges from 1.7x to 2.1x. This is consistent enough to use as a sanity check on a new COMPAS model.

**Pattern 3, Results don't transfer across datasets**
Papers reporting good fairness results on Adult Income typically show worse results when the same method is applied to COMPAS and vice versa. Friedler et al. (2019) report that sensitivity directly; in most single-dataset papers it is visible only when you compare the numbers across papers. This pattern is what FAPE formalizes as a research question.

**Pattern 4, Demographic parity and equalized odds move in opposite directions**
Interventions that improve demographic parity often worsen equalized odds and vice versa. Impossibility results predict tension between fairness criteria when base rates differ, and seeing it across papers makes it concrete. FAPE needs to report both.

---

## 5. Narrative/Landscape Review
### Where the Field Is and Where It's Going

The fairness in ML field is shifting its focus. The theoretical foundations are solid, we have good definitions, proven impossibility results, and working algorithmic interventions. What the field is missing is the engineering and operational infrastructure to deploy these interventions at production scale.

**Where the field has been (2016-2020):**
The debates were definitional: Chouldechova, Hardt and Dwork on which fairness metric is the right one. They were necessary and produced important results, but they also consumed enormous research energy on a question that may have no universal answer, since different fairness metrics capture different moral intuitions and different regulatory requirements.

**Where the field is now (2021-2025):**
The field is moving toward empirical benchmarking. FairGround (Simson et al. 2025) is the clearest signal: the community recognizes that evaluation on two legacy datasets is insufficient and is building the infrastructure for broader evaluation. The benchmarking is still aimed at research, with little attention to production use.

**Where the field needs to go (2025 onwards):**
It needs production deployment infrastructure. The questions that matter in enterprise settings are largely unaddressed in the academic literature: how to monitor fairness continuously, how to handle model updates without re-auditing from scratch, and how to satisfy different regulatory requirements across jurisdictions at the same time.

FAPE is aimed at this gap. The 4-stage framework (data preprocessing → baseline classification → fairness intervention → deployment monitoring) is designed to bridge the gap between what the research community has built and what production environments actually need.

---

## 6. Gaps and Conflicts

**Unresolved conflicts in the literature:**

*Conflict 1:* Chouldechova (2017) proves fairness metric incompatibility. Hardt et al. (2016) proposes equalized odds as the solution. These are not in conflict mathematically but create confusion in practice, papers cite both without acknowledging the impossibility result constrains what equalized odds can achieve.

*Conflict 2:* Dressel & Farid (2018) argue humans are as biased as algorithms. Kleinberg et al. (2018) argue algorithms outperform humans on accuracy. Both are right in different senses. The field hasn't developed a coherent framework for when accuracy matters more than fairness, FAPE doesn't resolve this but surfaces it explicitly in the results.

**Open questions FAPE does not answer:**
- Causal vs statistical fairness, FAPE uses statistical definitions throughout
- Intersectional fairness, race × gender interactions are not fully addressed in Stage 3
- Fairness over time, deployment monitoring in Stage 4 is a starting point, not a complete solution

---

## Summary

The literature review produced three findings that directly shaped FAPE:

1. Whether a fairness constraint's effect generalizes across domains is an open empirical question; comparative studies stop at a few benchmark datasets
2. Production deployment of fairness interventions is an unsolved engineering problem, and the open-source frameworks stop at research validation
3. Multi-metric evaluation is necessary, single-metric optimization produces misleading results that don't hold under scrutiny

These three findings are the justification for FAPE's existence. Every design decision in the framework traces back to one of them.
