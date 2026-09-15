# FAPE, Literature Analysis

> **Record note, September 15 2026.** Written November 2025 to April 2026 and corrected in place on September 15 2026 where it misdescribed a source or overstated a gap: Breck et al. (2017) do include a pre-release inclusion test; Ajarra and Basu (2026) is a theoretical auditing paper; Sariola et al. (2026) measured their 10% with audit-study data; Kleinberg et al. is a 2018 paper on New York bail data, not COMPAS; commercial fairness monitoring exists (Amazon SageMaker Clarify); and Friedler et al. (2019) and Chen et al. (2023) compare interventions across benchmark datasets. FAPE does not measure calibration, so the lines that promised a calibration tradeoff now say so. Dated snapshots such as the May knowledge map keep their date, with current status in brackets. Decision 27 in methodology_decisions.md records the source checks, and docs/paper_draft.md is the current statement of the study.

## Fairness Auditing in Production ML Systems, Research Protocols

**Period:** November 2025, April 2026
**Researcher:** Nithin Narla
**Status:** Protocols 1-6 complete after COMPAS and Folktables verified. Protocols 7-9 rough notes, still working through implications.

---

## Why These Protocols Now

Loaded COMPAS last week. First thing I did after verification was look at the recidivism rate split by race. The base rate difference is real and it's in the numbers, not a theoretical construct anymore, a constraint I have to work around in every Stage 3 experiment I'll run. Chouldechova's impossibility theorem became a design problem the moment I saw that number. That's why I'm running these protocols now rather than waiting until the full pipeline is built. Real data changes how you read the papers.

---

## Protocol 1, Intake: Paper Table + Clusters + Conflicts

### Paper Table

| Paper | Year | Venue | Core Claim | Cluster | Conflict |
|-------|------|-------|------------|---------|----------|
| Angwin et al., Machine Bias | 2016 | ProPublica | COMPAS produces racially disparate false positive rates | Criminal justice | Northpointe disputes methodology, but ProPublica published the data and anyone can check |
| Chouldechova, Fair Prediction | 2017 | Big Data | Calibration and equal error rates cannot all hold when base rates differ | Impossibility | Conflicts with every paper that claims to achieve fairness without acknowledging this constraint |
| Hardt, Price & Srebro, Equality of Opportunity | 2016 | NeurIPS | Equalized odds is achievable via post-processing | Post-processing | Predates Chouldechova's proof, so it doesn't address the tradeoff with calibration |
| Dressel & Farid, Limits of Predicting Recidivism | 2018 | Science Advances | Crowdsourced humans match COMPAS accuracy | Human vs algorithm | Pulls against Kleinberg, matters for FAPE because if humans are equally biased the problem is the context not the algorithm |
| Kleinberg et al., Human Decisions and Machine Predictions | 2018 | Quarterly Journal of Economics | Judges make worse bail decisions than an algorithm on New York City data | Human vs algorithm | Pulls against Dressel in a different setting, both rigorous, both overclaim generalization |
| Obermeyer et al., Racial Bias in Healthcare | 2019 | Science | Healthcare algorithm underestimates Black patients via cost proxy | Cross-domain bias | Bias through proxies not explicit features, most important paper after Chouldechova for understanding what FAPE Stage 1 needs to catch |
| Sculley et al., Hidden Technical Debt | 2015 | NeurIPS | Production ML systems degrade silently | Production deployment | Not a fairness paper but possibly the most consequential for FAPE Stage 4 design |
| Mitchell et al., Model Cards | 2019 | FAccT | Static documentation snapshots capture model behavior | Documentation | Single point in time, which Ajarra and Basu (2026) show falls short once model owners can update the model |
| Dwork et al., Fairness Through Awareness | 2012 | ITCS | Similar people should get similar predictions | Individual fairness | Philosophically appealing, practically difficult, defining similar requires a task-specific metric nobody agrees on |
| Barocas & Hardt, NeurIPS Tutorial | 2017 | NeurIPS | Taxonomy of fairness definitions across legal and technical contexts | Taxonomy | No universal correct definition, the taxonomy is the contribution not a resolution |
| Lambrecht & Tucker, Algorithmic Bias in Ad Delivery | 2019 | Management Science | Bias emerges from economic optimization not discriminatory intent | Emergent bias | Structural not designed, important for understanding why fairness needs checking after the fact |
| Breck et al., ML Test Score | 2017 | IEEE BigData | 28-point production readiness rubric | Production deployment | One pre-release test for considerations of inclusion, and no fairness test among the seven monitoring tests |
| Ding et al., Retiring Adult | 2021 | NeurIPS | Adult Income dataset has serious methodological flaws | Dataset quality | The field kept using it anyway, which is why FAPE uses Folktables ACS instead |
| Ajarra and Basu, Auditing Under Model Updates | 2026 | arXiv | Theory of auditing group fairness when model owners update their models: sample complexity and which updates preserve the audited property | Production fairness | A release-time model card or a one-time post-processing result says nothing about updates the audit did not anticipate |
| Sariola et al., Illusion of Fairness | 2026 | AAAI | Equalizing base rates in hiring data looks like parity on traditional measures but leaves ~10% disparity measured with audit-study data | Measurement | How disparity is measured can change the conclusion, a reason not to rely on one number |
| Weerts et al., Fairlearn | 2023 | JMLR | Practical post-processing fairness constraints | Tools | Stops at validation, no post-deployment monitoring |
| Simson et al., FairGround Corpus | 2025 | arXiv | 44 fairness-annotated datasets for reproducible cross-domain evaluation | Benchmark | First serious attempt to fix the benchmark monoculture, 1,964,010 records verified |

### Clusters

Five distinct conversations in this literature. They barely cite each other, which is itself a finding.

**The impossibility conversation:** Chouldechova, Barocas & Hardt, Sariola. Everything here is working out the implications of one mathematical result from 2017. Chouldechova is the anchor, the impossibility theorem is not a limitation of current methods, it's mathematics. Every paper in this cluster that doesn't acknowledge it is either optimizing for one metric at the expense of another or using a fairness definition narrow enough to avoid the constraint.

**The production deployment conversation:** Sculley, Breck, Mitchell, Ajarra. None of these papers are primarily about fairness definitions. They're asking what happens to ML systems after they ship. The finding across all four, systems degrade and documentation goes stale, maps directly onto fairness even though most of these papers don't use the word. The most telling data point is Breck's 28-test rubric: fairness appears once, as a pre-release inclusion test, and never among the monitoring tests.

**The cross-domain evidence conversation:** Angwin (criminal justice), Obermeyer (healthcare), Lambrecht & Tucker (advertising). Each paper studies one domain, finds a specific bias mechanism, and stops. They don't cite each other. The criminal justice bias comes from historical disparities in the outcome variable. The healthcare bias comes from using cost as a proxy for health need. The advertising bias comes from economic optimization. Three different mechanisms, three different domains, zero cross-domain synthesis. That's the gap FAPE is designed to fill.

**The human vs algorithm conversation:** Dressel & Farid vs Kleinberg. The two reach different conclusions on different data, COMPAS in Broward County and bail decisions in New York City. I spent time on this because it matters for how FAPE frames the problem. If Dressel is right, humans are as biased as algorithms and the problem isn't the algorithm, it's the decision-making context. FAPE can't fix that. But it can make the bias visible and measurable regardless of source.

**The dataset quality conversation:** Ding et al. Adult Income is load-bearing for a decade of fairness research and it's methodologically flawed. Ding et al. said so in 2021. The field nodded and kept using it. That's why FAPE uses Folktables ACS instead, not because it's newer but because the benchmark problem is documented and I'm not going to validate methods on data I know is flawed.

### Conflicts I Can't Resolve

The Dressel-Kleinberg tension is real and I'm not papering over it. Both studies are rigorous. Both overclaim. FAPE uses COMPAS because it's the field standard, not because Broward County 2013-2014 represents all criminal justice contexts. The paper needs to say this.

The impossibility-post-processing conflict is more fundamental. Hardt proposes equalized odds. Chouldechova proves it trades off against calibration when base rates differ. The COMPAS data has that base rate difference, I can see it now. When FAPE applies ThresholdOptimizer in Stage 3, equalizing error rates means giving up calibration, which FAPE does not measure. The paper has to name that tradeoff and report several metrics rather than one metric called fairness.

---

## Protocol 2, Contradiction Finder

Three real tensions, where the claims pull in different directions:

**Humans vs algorithms in criminal justice:** Dressel & Farid (2018), MTurk workers with minimal case information match COMPAS accuracy. Kleinberg et al. (2018), judges in New York City make systematically worse bail decisions than an algorithm on accuracy. Both empirically correct in their own setups. Both present findings as general claims. Both overclaim. The contradiction matters for FAPE because framing algorithmic bias as the problem implies humans would do better, which Kleinberg challenges directly.

**Post-processing as solution vs impossibility theorem:** Hardt et al. (2016), ThresholdOptimizer is the practical solution to demographic bias. Chouldechova (2017), calibration and equal error rates cannot all hold when base rates differ. Both correct, both cited widely, and papers often cite both without reconciling them. Every FAPE Stage 3 experiment lives inside this tension.

**Static documentation vs dynamic systems:** Mitchell et al. (2019), model cards as the accountability mechanism. Ajarra and Basu (2026), a theory of auditing fairness when the model keeps being updated, which release-time documentation cannot cover. Commercial platforms now monitor bias on live data (Amazon SageMaker Clarify), but the open-source toolkits researchers use have no equivalent. FAPE Stage 4 is an open attempt.

---

## Protocol 3, Citation Chain: Three Concepts Tracked

**Equalized odds:**
Hardt et al. (2016) define and propose it → Chouldechova (2017) proves incompatibility with calibration → Dressel & Farid (2018) apply to COMPAS → Barocas & Hardt (2017) taxonomize relative to other definitions → Sariola et al. (2026) show that parity measured the usual way can hide about 10% disparity that audit-study data reveals. The chain ends at a warning: how disparity is measured can decide whether an intervention looks like it worked.

**Production ML degradation:**
Sculley et al. (2015) identify silent degradation as fundamental to production ML → Breck et al. (2017) operationalize readiness with 28 tests, one of them a pre-release inclusion test and none a fairness monitoring test → Mitchell et al. (2019) propose static documentation → Ajarra and Basu (2026) work out how to audit fairness under model updates. Commercial monitors exist, but the open-source research toolkits still stop at a single audit. FAPE Stage 4 is an open attempt at the missing piece.

**Cross-domain fairness:**
Angwin et al. (2016) document criminal justice bias → Obermeyer et al. (2019) document healthcare bias via proxy variables → Lambrecht & Tucker (2019) document advertising bias via economic optimization → nobody connects them. The chain ends at a gap, which is exactly where FAPE starts.

---

## Protocol 4, Gap Scanner: Six Gaps Ranked

**Gap 1, Cross-domain generalization is largely untested.**
Partial gap. Chen et al. (2023) test equalized odds post-processing, with sixteen other methods, on the five AIF360 benchmark datasets covering income, criminal justice, credit, bank marketing and healthcare. I found no study that adds education, legal admissions or agricultural lending, or that asks whether one intervention's effect transfers across domains. FAPE builds the infrastructure and runs that test.

**Gap 2, Open-source fairness monitoring does not exist.**
Sculley (2015) identified the degradation problem, and Ajarra and Basu (2026) formalize auditing fairness under model updates. AIF360, Fairlearn and the What-If Tool have no continuous monitoring. Commercial platforms such as Amazon SageMaker Clarify do monitor bias on live data, one window at a time. FAPE Stage 4 is an open, sequential alternative, tested across domains on a simulated shift.

**Gap 3, Multi-metric reporting is treated as optional.**
Sariola et al. (2026) showed that a parity number measured one way can hide real disparity. FAPE reports demographic parity and equalized odds everywhere, and disparate impact ratio and accuracy cost where computable, so tradeoffs between criteria stay visible. Individual fairness score excluded, see methodology_decisions.md Decision 9.

**Gap 4, Agricultural domain has never appeared in fairness research.**
I went looking for fairness papers on agricultural lending or farm household outcomes and found none. Criminal justice, healthcare, education, financial services, all represented. The populations most affected by algorithmic decisions in agricultural contexts are invisible in the fairness literature. FAPE adds this domain with three datasets verified: USDA NASS Census 2022, SBA 7(a) NAICS-11 loans, LSMS-ISA Nigeria Wave 4. [Decision 11 later kept only SBA 7(a) for modeling; USDA NASS stays in EDA and LSMS-ISA was excluded.]

**Gap 5, Intersectional fairness is theoretically acknowledged and empirically ignored.**
Race and gender evaluated independently in virtually every paper. FAPE's scripts print intersectional breakdowns where sample sizes permit and flag where they don't; the paper does not analyze them.

**Gap 6, Legacy benchmark reliance persists despite documented flaws.**
Ding et al. (2021) documented Adult Income's problems. The field kept using it. FAPE uses Folktables ACS and FairGround because validating on data I know is flawed undermines everything the paper claims.

---

## Protocol 5, Methodology Audit

**Angwin et al. (2016):** They got the actual COMPAS scores from Broward County, matched to outcomes, calculated false positive rates by race. What I respect about this is the methodology, they didn't take Northpointe's word for it, they ran the analysis themselves. That's the spirit FAPE is trying to bring to enterprise fairness auditing. Limitation: one county, one algorithm, one two-year window. The finding is solid. The generalization claim is not.

**Chouldechova (2017):** Mathematical proof. I've worked through it. Its key condition, different base rates between groups, holds in the COMPAS data I've now loaded. The result is correct and it's not going away. What the proof doesn't tell you is which metric to deprioritize when you can't satisfy both. That's a values question not a math question, and FAPE doesn't answer it, it surfaces the tradeoff and lets practitioners decide.

**Hardt et al. (2016):** The cleanest formalization of equalized odds I found. ThresholdOptimizer is sound for static data, you can take any trained model and apply a fairness constraint without retraining. This is exactly what FAPE Stage 3 does. What the paper doesn't address, and what Ajarra and Basu take up ten years later, is what happens when the model gets updated. Stage 3 results have a shelf life.

**Sculley et al. (2015):** Not a fairness paper, but possibly the most consequential for FAPE Stage 4 design. The hidden technical debt framework is informed practitioner opinion from Google, not an empirical result. But the finding that systems degrade silently through feature drift and data shifts maps directly onto fairness drift. FAPE Stage 4 tests, on a simulated shift, whether a monitor can catch that kind of degradation in a fairness metric; it does not measure real degradation.

**Obermeyer et al. (2019):** The proxy variable identification is the methodological contribution, healthcare cost used as proxy for health need, creating racial disparities that aren't visible in the model's explicit features. FAPE's Stage 1 EDA looks for relationships of this kind, though the pipeline has no formal proxy test (see the correction in research_design_rationale.md). Limitation: one algorithm, one healthcare system.

---

## Protocol 6, Master Synthesis

I've read a lot of fairness papers now. Here's what I think after working through them with real data loaded.

The theoretical work is done. Chouldechova proved the impossibility in 2017. Barocas and Hardt catalogued the definitions. The field knows what fairness is and what it provably cannot be. That's settled.

What isn't settled, and what I didn't fully appreciate until I started loading actual data, is that the empirical literature has been validating increasingly sophisticated methods on increasingly narrow data for eight years. COMPAS and Adult Income are load-bearing for an entire research program. Ding et al. showed Adult Income is flawed in 2021. The field kept going because changing benchmarks disrupts comparability. That's a collective action problem and it means the empirical literature is more fragile than it looks.

The production deployment gap is where I keep landing. Sculley identified silent degradation in 2015. Breck's 2017 readiness rubric has 28 tests, and fairness shows up once, as a pre-release inclusion test, never as something to monitor. Mitchell proposed model cards in 2019. Ajarra and Basu worked out in 2026 how to audit fairness when models are updated. Commercial monitoring has arrived in cloud platforms, but the open research tooling still audits once. That gap is what FAPE Stage 4 is about.

Having COMPAS (6,172 records) and Folktables ACS (1,589,032 records) loaded changes the texture of reading these papers. Chouldechova's impossibility isn't abstract when the base rate difference is in the data in front of me. When FAPE runs ThresholdOptimizer in Stage 3, equalizing error rates will cost calibration. FAPE doesn't measure calibration, so the paper has to name that tradeoff and report the metrics it does measure side by side.

The full dataset pipeline is now complete. Student Performance (1,044 records), Law School Admissions (18,692), Lending Club (1,348,099), USDA NASS Census (7,334 aggregate rows), SBA 7(a) agricultural loans (15,845), and LSMS-ISA Nigeria Wave 4 (30,312 farm households) all verified and loaded. The agricultural domain is the one I'm most interested in; I found no fairness paper on this population. Small farmers, agricultural loan applicants, farm household outcomes, invisible in the fairness literature. I haven't found another cross-domain fairness evaluation that includes it.

FairGround (Simson et al. 2025) is now verified, 1,964,010 records across 44 fairness-annotated datasets. This changes the benchmark picture meaningfully. I haven't found another paper that uses FairGround inside a multi-domain evaluation framework rather than as a standalone benchmark.

What FAPE can legitimately claim: one intervention evaluated across a wider range of domains than the comparative studies I found, an open sequential fairness monitor that the open-source toolkits lack, and multi-metric reporting that keeps tradeoffs between criteria visible. What it cannot claim: solving the impossibility, generalizing from eight evaluations to all contexts, or removing the need for human judgment about which metric matters in which regulatory setting.

---

## Protocol 7, Assumption Killer

**Assumption 1, COMPAS findings generalize to criminal justice AI broadly:**
They don't. One algorithm, one county, one two-year window. Using it for field comparability, not because Broward County 2013-2014 represents all criminal justice contexts. The paper needs to say this.

**Assumption 2, Post-processing constraints hold after model updates:**
Hardt et al. demonstrated post-processing on static data. Ajarra and Basu (2026) show that an audit has to account for how a model may be updated. Stage 3 findings have a shelf life the paper should acknowledge.

**Assumption 3, A fairness audit at deployment time is sufficient:**
Sculley (2015) and Ajarra and Basu (2026) both argue against it. FAPE Stage 4 challenges the assumption on a simulated shift.

**Assumption 4, Switching from Adult Income to Folktables ACS is costless:**
It creates a comparability problem with prior work. The paper needs to acknowledge this rather than treating the benchmark switch as a free improvement.

**Assumption 5, Group-level fairness metrics capture fair individual treatment:**
They don't necessarily. Dwork et al. (2012) makes the individual-fairness case, and Sariola et al. (2026) show a group parity number can mislead. FAPE reports group metrics because they're the field standard but the paper should not imply group parity equals fair individual treatment.

---

## Protocol 8, Knowledge Map

```
FAPE Knowledge Map, May 2026
Verified: 4,980,540 records across 9 datasets [Law School's 18,692 are also inside FairGround, so 4,961,848 distinct]

CORE PROBLEM
└── Fairness evaluation in production ML is broken
    ├── One-time audits miss post-deployment drift
    ├── Single-metric reporting hides tradeoffs
    └── Domain-specific fixes don't generalize

FAIRNESS THEORY CLUSTER
├── Chouldechova 2017, impossibility theorem
│   └── constrains all FAPE claims, non-negotiable
├── Hardt et al. 2016, equalized odds + ThresholdOptimizer
│   └── FAPE Stage 3 intervention backbone
└── Dwork et al. 2012, individual fairness
    └── IFS excluded from FAPE, see Decision 9; DIR (disparate impact ratio) is FAPE Stage 3 metric

PRODUCTION FAILURE CLUSTER
├── Sculley et al. 2015, ML technical debt
│   └── production systems degrade silently
├── Ajarra and Basu 2026, auditing fairness under model updates
│   └── carries Sculley's concern to fairness, in theory
└── Obermeyer et al. 2019, proxy variable bias
    └── bias propagates through cost proxies, healthcare

BENCHMARK CLUSTER
├── Ding et al. 2021, Folktables ACS
│   └── 1,589,032 records verified, benchmark monoculture problem
├── Simson et al. 2025, FairGround
│   └── 1,964,010 records verified, 44 datasets across domains
└── Angwin et al. 2016, COMPAS
    └── 6,172 records verified, criminal justice baseline

DATASET LANDSCAPE
├── Criminal justice: COMPAS 6,172
├── Socioeconomic: Folktables ACS 1,589,032
├── Benchmark: FairGround 1,964,010
├── Education: Student Performance 1,044
├── Legal: Law School 18,692
├── Financial: Lending Club 1,348,099
├── Agricultural: USDA NASS 7,334 + SBA 7(a) 15,845 + LSMS-ISA 30,312
└── Healthcare: MIMIC-III pending PhysioNet [never obtained; MEPS Panel 19 used instead]

FAPE CORE CONTRIBUTION
└── Cross-domain fairness auditing pipeline
    ├── Gap 1: one intervention not yet followed across many domains
    ├── Gap 2: no open-source fairness drift monitoring
    ├── Gap 3: impossibility results not built into pipelines
    ├── Gap 4: agricultural lending missing from the fairness papers I found
    └── Gap 5: enterprise simulation missing from academic fairness work
```


**What I'm confident about:**
- Chouldechova's impossibility theorem constrains what FAPE can claim. Non-negotiable.
- COMPAS (6,172 records verified) and Folktables ACS (1,589,032 records verified) are solid starting points.
- ThresholdOptimizer works on static data, Hardt et al. established this.
- Production systems degrade silently, Sculley established this, and Ajarra and Basu formalized auditing fairness under updates.
- Bias propagates through proxy variables, Obermeyer established this for healthcare.

**What I think is true but haven't confirmed yet:**
- Cross-domain constraints will show different accuracy-fairness tradeoff profiles by domain, now testable across criminal justice, socioeconomic, education, financial, and agricultural domains simultaneously.
- FairGround corpus (1,964,010 records verified across 44 datasets) extends multi-domain evaluation meaningfully, the benchmark monoculture problem Ding et al. identified has a practical response now.
- Stage 4 CUSUM detection will catch drift a one-time audit misses, but Stage 4 hasn't run yet. [Since run on a simulated shift; see paper Section 5.7.]

**What I'm genuinely uncertain about:**
- Whether seven domains is enough for cross-domain generalization claims [now eight evaluations from seven data sources], the agricultural domain addition (SBA 7(a)) adds a population that has never appeared in fairness literature. USDA NASS and LSMS-ISA Nigeria excluded from ML pipeline, see Decision 11.
- Individual fairness score excluded from evaluation, replaced with accuracy cost. See Decision 9. This uncertainty is resolved.
- How to handle MIMIC-III if PhysioNet access takes longer than expected. [Access never came through; healthcare uses MEPS Panel 19.]

---

## Protocol 9, So What Test

Three things I'd say to someone who doesn't work in ML:

Most fairness studies look at one type of consequential decision, criminal courts, or hospitals, or school admissions, and the comparisons that do span several stick to a few standard datasets. Whether a fix that works in one area holds across many at once is still mostly unchecked. That's what FAPE does. A bank or hospital uses ML across many contexts at once, and few studies tell them whether a fairness fix in lending will hold in hiring or healthcare at the same time.

When organizations check ML systems for fairness before launch, they usually check once and consider it done. But these systems change, retrained on new data, updated by vendors, used by populations that shift. FAPE builds the infrastructure to keep checking after launch. The alternative is finding a problem six months in, after real decisions have already been made about real people.

Most fairness papers pick one way to measure bias, report that number, and call it evidence of fairness. But improving one fairness metric can make another worse, this is mathematically proven. FAPE measures up to four metrics at once and shows how they trade off, so practitioners can choose which metric matters most for their specific legal context rather than having a researcher make that choice for them implicitly.
