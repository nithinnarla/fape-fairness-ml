# FAPE — Literature Analysis
## Fairness Auditing in Production ML Systems — Research Protocols

**Period:** November 2025 — April 2026
**Researcher:** Nithin Narla
**Status:** Protocols 1-6 complete after COMPAS and Folktables verified. Protocols 7-9 rough notes — still working through implications.

---

## Why These Protocols Now

Loaded COMPAS last week. First thing I did after verification was look at the recidivism rate split by race. The base rate difference is real and it's in the numbers — not a theoretical construct anymore, a constraint I have to work around in every Stage 3 experiment I'll run. Chouldechova's impossibility theorem became a design problem the moment I saw that number. That's why I'm running these protocols now rather than waiting until the full pipeline is built. Real data changes how you read the papers.

---

## Protocol 1 — Intake: Paper Table + Clusters + Conflicts

### Paper Table

| Paper | Year | Venue | Core Claim | Cluster | Conflict |
|-------|------|-------|------------|---------|----------|
| Angwin et al. — Machine Bias | 2016 | ProPublica | COMPAS produces racially disparate false positive rates | Criminal justice | Northpointe disputes methodology — but ProPublica published the data and anyone can check |
| Chouldechova — Fair Prediction | 2017 | Big Data | Calibration and equalized odds cannot coexist when base rates differ | Impossibility | Conflicts with every paper that claims to achieve fairness without acknowledging this constraint |
| Hardt, Price & Srebro — Equality of Opportunity | 2016 | NeurIPS | Equalized odds is achievable via post-processing | Post-processing | Proposes equalized odds as the solution without engaging with Chouldechova's proof that it trades off against calibration |
| Dressel & Farid — Limits of Predicting Recidivism | 2018 | Science Advances | Crowdsourced humans match COMPAS accuracy | Human vs algorithm | Directly contradicts Kleinberg — matters for FAPE because if humans are equally biased the problem is the context not the algorithm |
| Kleinberg et al. — Human Decisions and Machine Predictions | 2016 | NBER | Judges make worse bail decisions than algorithms | Human vs algorithm | Directly contradicts Dressel — both rigorous, both overclaim generalization |
| Obermeyer et al. — Racial Bias in Healthcare | 2019 | Science | Healthcare algorithm underestimates Black patients via cost proxy | Cross-domain bias | Bias through proxies not explicit features — most important paper after Chouldechova for understanding what FAPE Stage 1 needs to catch |
| Sculley et al. — Hidden Technical Debt | 2015 | NeurIPS | Production ML systems degrade silently | Production deployment | Not a fairness paper but possibly the most consequential for FAPE Stage 4 design |
| Mitchell et al. — Model Cards | 2019 | FAccT | Static documentation snapshots capture model behavior | Documentation | Single point in time — Ajarra (2026) shows this is inadequate when model updates alter fairness properties |
| Dwork et al. — Fairness Through Awareness | 2012 | ITCS | Similar people should get similar predictions | Individual fairness | Philosophically appealing, practically difficult — defining similar requires a task-specific metric nobody agrees on |
| Barocas & Hardt — NeurIPS Tutorial | 2017 | NeurIPS | Taxonomy of fairness definitions across legal and technical contexts | Taxonomy | No universal correct definition — the taxonomy is the contribution not a resolution |
| Lambrecht & Tucker — Algorithmic Bias in Ad Delivery | 2019 | Management Science | Bias emerges from economic optimization not discriminatory intent | Emergent bias | Structural not designed — important for understanding why FAPE needs Stage 1 proxy detection |
| Breck et al. — ML Test Score | 2017 | IEEE BigData | 28-point production readiness rubric | Production deployment | Zero fairness tests — not an oversight, a policy statement by a major ML practitioner |
| Ding et al. — Retiring Adult | 2021 | NeurIPS | Adult Income dataset has serious methodological flaws | Dataset quality | The field kept using it anyway — which is why FAPE uses Folktables ACS instead |
| Ajarra et al. — Auditing Under Model Updates | 2026 | arXiv | Model updates fundamentally alter fairness properties | Production fairness | Makes Mitchell et al. model cards inadequate and Hardt et al. post-processing results time-limited |
| Sariola et al. — Illusion of Fairness | 2026 | AAAI | Equalizing base rates masks ~10% disparity | Measurement | Standard metrics actively mislead — makes multi-metric reporting non-optional for FAPE |
| Weerts et al. — Fairlearn | 2023 | arXiv | Practical post-processing fairness constraints | Tools | Research-grade only — stops at validation, no production monitoring |
| Fabris et al. — FairGround Corpus | 2025 | arXiv | 44 fairness-annotated datasets for reproducible cross-domain evaluation | Benchmark | First serious attempt to fix the benchmark monoculture — 1,964,010 records verified |

### Clusters

Five distinct conversations in this literature. They barely cite each other, which is itself a finding.

**The impossibility conversation:** Chouldechova, Barocas & Hardt, Sariola. Everything here is working out the implications of one mathematical result from 2017. Chouldechova is the anchor — the impossibility theorem is not a limitation of current methods, it's mathematics. Every paper in this cluster that doesn't acknowledge it is either optimizing for one metric at the expense of another or using a fairness definition narrow enough to avoid the constraint.

**The production deployment conversation:** Sculley, Breck, Mitchell, Ajarra. None of these papers are primarily about fairness definitions. They're asking what happens to ML systems after they ship. The finding across all four — systems degrade, documentation goes stale, nobody is monitoring — maps directly onto fairness even though most of these papers don't use the word. Breck's 28-test rubric with zero fairness tests is the most telling data point in the entire cluster.

**The cross-domain evidence conversation:** Angwin (criminal justice), Obermeyer (healthcare), Lambrecht & Tucker (advertising). Each paper studies one domain, finds a specific bias mechanism, and stops. They don't cite each other. The criminal justice bias comes from historical disparities in the outcome variable. The healthcare bias comes from using cost as a proxy for health need. The advertising bias comes from economic optimization. Three different mechanisms, three different domains, zero cross-domain synthesis. That's the gap FAPE is designed to fill.

**The human vs algorithm conversation:** Dressel & Farid vs Kleinberg. Direct empirical conflict using the same COMPAS dataset. I spent time on this because it matters for how FAPE frames the problem. If Dressel is right, humans are as biased as algorithms and the problem isn't the algorithm — it's the decision-making context. FAPE can't fix that. But it can make the bias visible and measurable regardless of source.

**The dataset quality conversation:** Ding et al. Adult Income is load-bearing for a decade of fairness research and it's methodologically flawed. Ding et al. said so in 2021. The field nodded and kept using it. That's why FAPE uses Folktables ACS instead — not because it's newer but because the benchmark problem is documented and I'm not going to validate methods on data I know is flawed.

### Conflicts I Can't Resolve

The Dressel-Kleinberg conflict is genuine and I'm not papering over it. Both studies are rigorous. Both overclaim. FAPE uses COMPAS because it's the field standard — not because Broward County 2013-2014 represents all criminal justice contexts. The paper needs to say this.

The impossibility-post-processing conflict is more fundamental. Hardt proposes equalized odds. Chouldechova proves it trades off against calibration when base rates differ. The COMPAS data has that base rate difference — I can see it now. When FAPE applies ThresholdOptimizer in Stage 3, it improves equalized odds at a measurable cost to calibration. The paper has to report both and explain the tradeoff rather than reporting one metric and calling it fairness.

---

## Protocol 2 — Contradiction Finder

Three genuine contradictions — not differences in emphasis but incompatible empirical claims or logical conflicts:

**Humans vs algorithms in criminal justice:** Dressel & Farid (2018) — MTurk workers with minimal case information match COMPAS accuracy. Kleinberg et al. (2016) — judges make systematically worse bail decisions than algorithms on accuracy. Both empirically correct in their own setups. Both present findings as general claims. Both overclaim. The contradiction matters for FAPE because framing algorithmic bias as the problem implies humans would do better — which Kleinberg challenges directly.

**Post-processing as solution vs impossibility theorem:** Hardt et al. (2016) — ThresholdOptimizer is the practical solution to demographic bias. Chouldechova (2017) — satisfying equalized odds and calibration simultaneously is impossible when base rates differ. Both correct. Both cited widely. Neither acknowledges the tension between them. Every FAPE Stage 3 experiment lives inside this contradiction.

**Static documentation vs dynamic systems:** Mitchell et al. (2019) — model cards as the accountability mechanism. Ajarra et al. (2026) — model updates fundamentally alter fairness properties, making release-time documentation misleading. Seven years after model cards were proposed, nobody has built the dynamic monitoring alternative. FAPE Stage 4 is the attempt.

---

## Protocol 3 — Citation Chain: Three Concepts Tracked

**Equalized odds:**
Hardt et al. (2016) define and propose it → Chouldechova (2017) proves incompatibility with calibration → Dressel & Farid (2018) apply to COMPAS → Barocas & Hardt (2017) taxonomize relative to other definitions → Sariola et al. (2026) show standard measurement masks 10% disparity. The chain ends at a sobering place — the metric the field settled on is being shown to actively mislead in certain configurations.

**Production ML degradation:**
Sculley et al. (2015) identify silent degradation as fundamental to production ML → Breck et al. (2017) operationalize readiness with 28 tests and zero fairness tests → Mitchell et al. (2019) propose static documentation → Ajarra et al. (2026) confirm fairness degrades specifically under model updates. Ten years between Sculley and Ajarra. Nothing built in between. FAPE Stage 4 is the attempt to close that gap.

**Cross-domain fairness:**
Angwin et al. (2016) document criminal justice bias → Obermeyer et al. (2019) document healthcare bias via proxy variables → Lambrecht & Tucker (2019) document advertising bias via economic optimization → nobody connects them. The chain ends at a gap — which is exactly where FAPE starts.

---

## Protocol 4 — Gap Scanner: Five Gaps Ranked

**Gap 1 — Cross-domain generalization is empirically untested.**
Complete gap. No paper has tested whether equalized odds post-processing that works in criminal justice also works in healthcare, education, and financial services simultaneously. FAPE builds the infrastructure and runs the test.

**Gap 2 — Production fairness monitoring infrastructure does not exist.**
Sculley (2015) identified the problem. Ajarra (2026) confirmed it applies specifically to fairness. AIF360, Fairlearn, What-If Tool — none address continuous monitoring. FAPE Stage 4 is the first attempt in a multi-domain context.

**Gap 3 — Multi-metric reporting is treated as optional.**
Sariola et al. (2026) showed single-metric optimization creates illusions. FAPE reports demographic parity, equalized odds, disparate impact ratio, and individual fairness simultaneously — making the Chouldechova tradeoffs visible rather than hidden.

**Gap 4 — Agricultural domain has never appeared in fairness research.**
I went looking for fairness papers on agricultural lending or farm household outcomes. Nothing. Criminal justice, healthcare, education, financial services — all represented. The populations most affected by algorithmic decisions in agricultural contexts are invisible in the fairness literature. FAPE adds this domain with three datasets now verified: USDA NASS Census 2022, SBA 7(a) NAICS-11 loans, LSMS-ISA Nigeria Wave 4.

**Gap 5 — Intersectional fairness is theoretically acknowledged and empirically ignored.**
Race and gender evaluated independently in virtually every paper. FAPE reports intersectional breakdowns where sample sizes permit and flags where they don't.

**Gap 6 — Legacy benchmark reliance persists despite documented flaws.**
Ding et al. (2021) documented Adult Income's problems. The field kept using it. FAPE uses Folktables ACS and FairGround because validating on data I know is flawed undermines everything the paper claims.

---

## Protocol 5 — Methodology Audit

**Angwin et al. (2016):** They got the actual COMPAS scores from Broward County, matched to outcomes, calculated false positive rates by race. What I respect about this is the methodology — they didn't take Northpointe's word for it, they ran the analysis themselves. That's the spirit FAPE is trying to bring to enterprise fairness auditing. Limitation: one county, one algorithm, one two-year window. The finding is solid. The generalization claim is not.

**Chouldechova (2017):** Mathematical proof. I've worked through it. The assumptions — calibration holds and base rates differ between groups — both hold in the COMPAS data I've now loaded. The result is correct and it's not going away. What the proof doesn't tell you is which metric to deprioritize when you can't satisfy both. That's a values question not a math question, and FAPE doesn't answer it — it surfaces the tradeoff and lets practitioners decide.

**Hardt et al. (2016):** The cleanest formalization of equalized odds I found. ThresholdOptimizer is sound for static data — you can take any trained model and apply a fairness constraint without retraining. This is exactly what FAPE Stage 3 does. What the paper doesn't address — and what Ajarra closes seven years later — is whether this holds when the model gets updated. Stage 3 results have a shelf life.

**Sculley et al. (2015):** Not a fairness paper — but possibly the most consequential for FAPE Stage 4 design. The hidden technical debt framework is informed practitioner opinion from Google, not an empirical result. But the finding that systems degrade silently through feature drift and data shifts maps directly onto fairness drift. FAPE Stage 4 is partly an empirical test of whether the degradation Sculley describes actually manifests in fairness metrics specifically.

**Obermeyer et al. (2019):** The proxy variable identification is the methodological contribution — healthcare cost used as proxy for health need, creating racial disparities that aren't visible in the model's explicit features. This is the mechanism FAPE Stage 1 is designed to catch across all domains. Limitation: one algorithm, one healthcare system.

---

## Protocol 6 — Master Synthesis

I've read a lot of fairness papers now. Here's what I actually think after working through them with real data loaded.

The theoretical work is done. Chouldechova proved the impossibility in 2017. Barocas and Hardt catalogued the definitions. The field knows what fairness is and what it provably cannot be. That's settled.

What isn't settled — and what I didn't fully appreciate until I started loading actual data — is that the empirical literature has been validating increasingly sophisticated methods on increasingly narrow data for eight years. COMPAS and Adult Income are load-bearing for an entire research program. Ding et al. showed Adult Income is flawed in 2021. The field kept going because changing benchmarks disrupts comparability. That's a collective action problem and it means the empirical literature is more fragile than it looks.

The production deployment gap is where I keep landing. Sculley identified silent degradation in 2015. Breck built a 28-test readiness rubric in 2017 with zero fairness tests — I read that as a practitioner saying fairness is not a production concern, full stop. Mitchell proposed model cards in 2019. Ajarra confirmed in 2026 that model updates fundamentally alter fairness properties. Eleven years of documented awareness and nobody built the monitoring infrastructure. That gap is what FAPE Stage 4 is about.

Having COMPAS (6,172 records) and Folktables ACS (1,589,032 records) loaded changes the texture of reading these papers. Chouldechova's impossibility isn't abstract when the base rate difference is in the data in front of me. When FAPE runs ThresholdOptimizer in Stage 3, I'm going to be trading calibration for equalized odds in a measurable way. The paper has to report both numbers and explain the tradeoff.

The full dataset pipeline is now complete. Student Performance (1,044 records), Law School Admissions (18,692), Lending Club (1,348,099), USDA NASS Census (7,334 aggregate rows), SBA 7(a) agricultural loans (15,845), and LSMS-ISA Nigeria Wave 4 (30,312 farm households) all verified and loaded. The agricultural domain is the one I'm most interested in — no fairness paper has looked at this population. Small farmers, agricultural loan applicants, farm household outcomes — invisible in the fairness literature. FAPE is the first framework to include this domain in a cross-domain fairness evaluation.

FairGround (Fabris et al. 2025) is now verified — 1,964,010 records across 44 fairness-annotated datasets. This changes the benchmark picture meaningfully. FAPE is the first paper to use FairGround as part of a multi-domain evaluation framework rather than as a standalone benchmark.

What FAPE can legitimately claim: cross-domain evaluation at a scale nobody has run, continuous monitoring infrastructure that doesn't exist elsewhere, multi-metric reporting that makes the Chouldechova constraints visible. What it cannot claim: solving the impossibility, generalizing from five domains to all contexts, or removing the need for human judgment about which metric matters in which regulatory setting.

---

## Protocol 7 — Assumption Killer

**Assumption 1 — COMPAS findings generalize to criminal justice AI broadly:**
They don't. One algorithm, one county, one two-year window. Using it for field comparability, not because Broward County 2013-2014 represents all criminal justice contexts. The paper needs to say this.

**Assumption 2 — Post-processing constraints hold after model updates:**
Hardt et al. demonstrated ThresholdOptimizer on static data. Ajarra (2026) shows this fails under model updates. Stage 3 findings have a shelf life the paper should acknowledge.

**Assumption 3 — A fairness audit at deployment time is sufficient:**
Sculley (2015) and Ajarra (2026) both contradict this. FAPE Stage 4 is explicitly designed to challenge this assumption empirically.

**Assumption 4 — Switching from Adult Income to Folktables ACS is costless:**
It creates a comparability problem with prior work. The paper needs to acknowledge this rather than treating the benchmark switch as a free improvement.

**Assumption 5 — Group-level fairness metrics capture fair individual treatment:**
They don't necessarily. Dwork et al. (2012) and Sariola et al. (2026) both show this. FAPE reports group metrics because they're the field standard but the paper should not imply group parity equals fair individual treatment.

---

## Protocol 8 — Knowledge Map

**What I'm confident about:**
- Chouldechova's impossibility theorem constrains what FAPE can claim. Non-negotiable.
- COMPAS (6,172 records verified) and Folktables ACS (1,589,032 records verified) are solid starting points.
- ThresholdOptimizer works on static data — Hardt et al. established this.
- Production systems degrade silently — Sculley established this, Ajarra confirmed it for fairness.
- Bias propagates through proxy variables — Obermeyer established this for healthcare.

**What I think is true but haven't confirmed yet:**
- Cross-domain constraints will show different accuracy-fairness tradeoff profiles by domain — now testable across criminal justice, socioeconomic, education, financial, and agricultural domains simultaneously.
- FairGround corpus (1,964,010 records verified across 44 datasets) extends multi-domain evaluation meaningfully — the benchmark monoculture problem Ding et al. identified has a practical response now.
- Stage 4 CUSUM detection will catch drift a one-time audit misses — but Stage 4 hasn't run yet.

**What I'm genuinely uncertain about:**
- Whether nine datasets across five domains is enough for cross-domain generalization claims — the agricultural domain addition (USDA NASS, SBA 7(a), LSMS-ISA Nigeria) adds a population that has never appeared in fairness literature.
- Whether individual fairness in Stage 3 will be defensible without a domain-specific similarity metric.
- How to handle MIMIC-III if PhysioNet access takes longer than expected.

---

## Protocol 9 — So What Test

Three things I'd say to someone who doesn't work in ML:

Every major fairness study looks at one type of consequential decision — criminal courts, or hospitals, or school admissions. Nobody has checked whether fixes that work in one area also hold across all areas simultaneously. That's what FAPE does. A bank or hospital uses ML across many contexts at once, and nobody has told them whether a fairness fix in lending will hold in hiring or healthcare at the same time.

When organizations check ML systems for fairness before launch, they usually check once and consider it done. But these systems change — retrained on new data, updated by vendors, used by populations that shift. FAPE builds the infrastructure to keep checking after launch. The alternative is finding a problem six months in, after real decisions have already been made about real people.

Most fairness papers pick one way to measure bias, report that number, and call it evidence of fairness. But improving one fairness metric can make another worse — this is mathematically proven. FAPE measures all four standard metrics at once and shows how they trade off, so practitioners can choose which metric matters most for their specific legal context rather than having a researcher make that choice for them implicitly.
