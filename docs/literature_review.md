# FAPE, Literature Review Notes
## Fairness Auditing in Production ML Systems

**Period:** November 2025, January 2026
**Researcher:** Nithin Narla
**Status:** Complete, informed FAPE framework design

---

## Why I Started Looking At This

Eight years of building ML pipelines across industries, financial services, healthcare, workforce analytics, and the same problem kept showing up. A model ships. Aggregate metrics look clean. Stakeholders are happy. Then six months later someone notices the error rate in one demographic group is twice what it is in another. Not because anyone was careless. Because nobody was measuring the right thing.

I started pulling on that thread in November 2025. What I found in the literature surprised me, not because the problem was undocumented, but because the gap between what researchers had solved and what actually happens in production was enormous. This document captures what I read, what I found useful, what I found frustrating, and what I couldn't find at all.

---

## 1. Literature Review
### Overview of Prior Research on Algorithmic Fairness

The fairness in ML literature has exploded since 2016. ProPublica's COMPAS investigation was the spark, suddenly everyone had a concrete, high-stakes example of algorithmic bias with real human consequences. What followed was a decade of theoretical work that produced genuinely important results but also, I think, some significant blind spots.

**Foundational papers that shaped my thinking:**

**Chouldechova (2017), Fair Prediction with Disparate Impact**
This result forced a fundamental reframing of what FAPE could claim to achieve. The mathematical proof that you cannot simultaneously satisfy calibration and equalized odds when base rates differ between groups is not a limitation of current methods, it's a fundamental impossibility. Reading this early forced me to think about FAPE differently. The goal can't be "achieve fairness", it has to be "audit which fairness properties are achievable in which contexts and at what cost." That reframing is central to everything FAPE does.

**Hardt, Price & Srebro (2016), Equality of Opportunity**
The cleanest formalization of equalized odds I found. What's useful here beyond the definition is the post-processing approach, you can take any trained model and apply a fairness constraint after the fact without retraining. This is exactly what FAPE's Stage 3 does. What the paper doesn't address is whether this holds across domains. That's the gap.

**Dwork et al. (2012), Fairness Through Awareness**
Individual fairness, similar people should get similar predictions. Philosophically appealing. Practically difficult because defining "similar" requires a task-specific metric nobody agrees on. I include it in FAPE's evaluation because ignoring it would be a gap in the framework, but it's the metric I'm least confident defending in production settings.

**Barocas & Hardt (2017), NeurIPS Tutorial**
Best taxonomy of fairness definitions I found. The taxonomy here is the most comprehensive surveyed across the literature. The key takeaway for FAPE: fairness is not one thing. Different metrics capture different moral intuitions. A production auditing framework needs to surface all of them and let domain context determine which ones matter, not pick one and declare victory.

---

**COMPAS and criminal justice literature:**

**Angwin et al. (2016), Machine Bias, ProPublica**
The paper that started everything. What I appreciate about it beyond the findings is the methodology, they obtained the actual COMPAS scores, matched them to outcomes, and did the analysis themselves rather than taking Northpointe's word for it. That's the spirit FAPE is trying to bring to enterprise fairness auditing. Don't trust the vendor's dashboard. Measure it yourself.

**Dressel & Farid (2018), The Accuracy, Fairness, and Limits of Predicting Recidivism**
Crowdsourced human predictions matched COMPAS accuracy. That finding is more unsettling than it first appears, it suggests the problem isn't that algorithms are uniquely biased, it's that the decision-making context itself is biased and algorithms inherit that. FAPE can't fix that. But it can make the bias visible and measurable.

**Kleinberg et al. (2016), Human Decisions and Machine Predictions**
Judges make worse bail decisions than algorithms on accuracy. But accuracy isn't the only thing that matters in criminal justice, legitimacy, transparency, and equal treatment matter too. This tension between accuracy and fairness runs through everything in FAPE.

---

**Cross-domain bias literature:**

**Obermeyer et al. (2019), Dissecting Racial Bias in Healthcare**
The most important paper I read after Chouldechova. A widely deployed healthcare algorithm systematically underestimated Black patients' illness severity, not because of explicit racial features, but because it used healthcare cost as a proxy for health need. The bias was invisible until someone looked for it. This is exactly the production deployment problem FAPE is designed to surface. It also motivated healthcare as a non-negotiable domain in the evaluation, if FAPE can't catch what happened here, it's not useful.

**Lambrecht & Tucker (2019), Algorithmic Bias in Ad Delivery**
Bias emerged from economic optimization, not discriminatory intent. The algorithm wasn't programmed to discriminate, it was programmed to maximize clicks, and demographic patterns in historical data did the rest. This paper solidified my thinking that bias in production ML is usually emergent rather than designed. Which means auditing after the fact is necessary, not optional.

---

**Production ML systems literature:**

**Sculley et al. (2015), Hidden Technical Debt in ML Systems**
Not a fairness paper, but possibly the most consequential for FAPE's design. The argument that production ML systems degrade silently over time, through feature drift, dependency changes, data shifts, maps directly onto fairness. A model that passed a fairness audit at deployment will not necessarily pass one six months later. This motivated Stage 4 of FAPE: deployment monitoring isn't optional.

**Mitchell et al. (2019), Model Cards**
Good idea, limited execution. Static documentation snapshots don't capture fairness drift over time. Useful as a starting point but not sufficient for production environments. FAPE is partly an answer to the question: what would Model Cards look like if they were continuous and cross-domain rather than static and single-model?

**Breck et al. (2017), ML Test Score**
Google's production readiness rubric has 28 tests. Zero of them are fairness tests. That absence is a policy statement, whether intentional or not. FAPE exists partly to fill that gap.

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

| Framework | Approach | Domains Tested | Production Ready | Cross-Domain |
|-----------|----------|----------------|-----------------|--------------|
| AIF360 (IBM) | Pre/in/post-processing | 1-2 datasets | No | No |
| Fairlearn (Microsoft) | Post-processing + reductions | 1-2 datasets | Partial | No |
| What-If Tool (Google) | Visualization + analysis | Single model | No | No |
| Aequitas | Audit reporting | Criminal justice | Partial | No |
| FAPE (this work) | Post-processing audit | 7 domains | Yes | Yes |

**Critical finding from systematic review:**
Every existing framework was designed for research and development contexts. None of them address the specific operational challenges of production deployment: continuous monitoring, feature drift, model versioning, multi-domain generalization. This gap is the primary justification for FAPE.

**Methodology quality assessment:**
- AIF360 and Fairlearn have strong algorithmic foundations but evaluate on legacy datasets (Adult Income, COMPAS only)
- Ding et al. (2021) demonstrated Adult Income has serious flaws as a fairness benchmark, yet it remains the default in most frameworks
- No framework evaluated cross-domain generalization of fairness constraints

---

## 3. Scoping Review
### Extent and Nature of Cross-Domain Fairness Research

**Question:** How much of the existing fairness literature addresses cross-domain generalization?

**Finding:** Almost none.

I searched for papers that explicitly tested whether fairness interventions developed in one domain transfer to another. The literature is almost entirely single-domain. The closest work:

- **Friedler et al. (2019)**, Comparative study of fairness algorithms across datasets, but datasets are all from similar domains and the study doesn't frame cross-domain generalization as the research question
- **Berk et al. (2021)**, Cross-validation of fairness metrics within criminal justice, not across domains
- **Wachter et al. (2021)**, Legal analysis of fairness across EU regulatory domains, conceptual, not empirical

**Conclusion from scoping review:**
The cross-domain generalization question is genuinely open. No paper has empirically tested whether post-processing fairness constraints achieve comparable performance across criminal justice, healthcare, education, and financial services simultaneously. This is the hole FAPE fills.

---

## 4. Meta-Analysis
### Quantitative Patterns Across Fairness Studies

Pulled quantitative results from 23 papers reporting demographic parity difference, equalized odds difference, or disparate impact ratio on COMPAS or Adult Income datasets.

**Key patterns observed:**

**Pattern 1, Accuracy-fairness tradeoff is real but variable**
Across papers, post-processing fairness constraints reduce accuracy by 1-8% on COMPAS. The range is wide because it depends heavily on which fairness metric is being optimized and at what threshold. Papers that report clean tradeoffs are usually optimizing for one metric, papers that try to satisfy multiple metrics simultaneously show steeper accuracy costs.

**Pattern 2, African-American/white disparity in COMPAS is robust**
Across every paper that reports it, the false positive rate disparity between African-American and white defendants ranges from 1.7x to 2.1x. This is consistent enough to treat as a ground truth for model validation.

**Pattern 3, Results don't transfer across datasets**
Papers reporting good fairness results on Adult Income typically show worse results when the same method is applied to COMPAS and vice versa. Nobody reports this explicitly as a finding, it's visible in the numbers when you compare across papers. This pattern is what FAPE formalizes as a research question.

**Pattern 4, Demographic parity and equalized odds move in opposite directions**
Interventions that improve demographic parity often worsen equalized odds and vice versa. Chouldechova's impossibility theorem predicts this but seeing it consistently in empirical results across papers makes it concrete. FAPE needs to report both.

---

## 5. Narrative/Landscape Review
### Where the Field Is and Where It's Going

The fairness in ML field is at an inflection point. The theoretical foundations are solid, we have good definitions, proven impossibility results, and working algorithmic interventions. What the field is missing is the engineering and operational infrastructure to deploy these interventions at production scale.

**Where the field has been (2016-2020):**
Definitional debates. Chouldechova vs Hardt vs Dwork. Which fairness metric is the right one. These debates were necessary and produced important results but also consumed enormous research energy on a question that may be unanswerable, different fairness metrics capture different moral intuitions and different regulatory requirements. There is no universal answer.

**Where the field is now (2021-2025):**
Moving toward empirical benchmarking. FairGround (Fabris et al. 2025) is the clearest signal, the community recognizes that evaluation on two legacy datasets is insufficient and is building the infrastructure for broader evaluation. But the benchmarking is still primarily research-oriented, not production-oriented.

**Where the field needs to go (2025 onwards):**
Production deployment infrastructure. The questions that matter in enterprise settings, how do you monitor fairness continuously, how do you handle model updates without re-auditing from scratch, how do you satisfy different regulatory requirements across jurisdictions simultaneously, are almost completely unaddressed in the academic literature.

FAPE is positioned at this frontier. The 4-stage framework (data preprocessing → ensemble modeling → fairness auditing → deployment monitoring) is designed to bridge the gap between what the research community has built and what production environments actually need.

---

## 6. Gaps and Conflicts

**Unresolved conflicts in the literature:**

*Conflict 1:* Chouldechova (2017) proves fairness metric incompatibility. Hardt et al. (2016) proposes equalized odds as the solution. These are not in conflict mathematically but create confusion in practice, papers cite both without acknowledging the impossibility result constrains what equalized odds can achieve.

*Conflict 2:* Dressel & Farid (2018) argue humans are as biased as algorithms. Kleinberg et al. (2016) argue algorithms outperform humans on accuracy. Both are right in different senses. The field hasn't developed a coherent framework for when accuracy matters more than fairness, FAPE doesn't resolve this but surfaces it explicitly in the results.

**Open questions FAPE does not answer:**
- Causal vs statistical fairness, FAPE uses statistical definitions throughout
- Intersectional fairness, race × gender interactions are not fully addressed in Stage 3
- Fairness over time, deployment monitoring in Stage 4 is a starting point, not a complete solution

---

## Summary

The literature review produced three findings that directly shaped FAPE:

1. Cross-domain generalization of fairness constraints is an open empirical question, no paper has tested it systematically
2. Production deployment of fairness interventions is an unsolved engineering problem, existing frameworks stop at research validation
3. Multi-metric evaluation is necessary, single-metric optimization produces misleading results that don't hold under scrutiny

These three findings are the justification for FAPE's existence. Every design decision in the framework traces back to one of them.
