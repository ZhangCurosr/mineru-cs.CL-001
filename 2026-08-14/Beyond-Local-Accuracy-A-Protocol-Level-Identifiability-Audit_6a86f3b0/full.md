# Beyond Local Accuracy: A Protocol-Level Identifiability Audit for Controlled LLM Reasoning Evaluation

Junhao Luo Ning Huang\* Ziqi Sha\* Wenxuan Tang\* Wei Deng† School of Statistics and Data Science, Southwestern University of Finance and Economics Chengdu, China

## Abstract

LLM benchmark scores can be precise even when the observation protocol does not identify the behavioral property they are intended to measure. In a controlled, solver-grounded setting, we formalize a protocol-level identifiability audit over a finite behavioral policy class: given policies H, observation support O, and estimand τ, we test whether O separates every pair with different τ. The audit requires zero model calls and resolves our diagnostic case: base-only observation collapses seven frozen deterministic policies into one equivalence class; full support yields seven classes and no cross-estimand collisions; every leave-oneout support retains a constructive collision witness. Empirically, both constrained-generation variants have pair-validity 1.0, yet base accuracy and selective-response fidelity diverge— 0.620 versus 0.324 across six balanced oracletransition directions (cluster-bootstrap 95% CI [0.600, 0.642] vs. [0.304, 0.345])—and the gap recurs on a second deterministic source (0.646 vs. 0.331). The audit also synthesizes a minimum identifying support O\* for the frozen policy class: two cells instead of the full 36- cell tensor. This case shows how evaluationdesign validity can be checked structurally before model inference and why base correctness does not determine intervention-response fidelity.

## 1 Introduction

LLM evaluation increasingly relies on accuracy over static benchmarks as a proxy for reasoning quality, despite longstanding concerns that broad capability claims can outrun what benchmark measurements support (Raji et al., 2021; Salaudeen et al., 2025). A model that answers base queries correctly is often described as “faithful," “sound," or “capable," and its failure modes are read off from where accuracy drops. This practice conflates two different quantities. Static correctness asks whether the model's answer matches an oracle on fixed inputs. Intervention-response fidelity asks whether the model's answer changes in the direction prescribed by a controlled intervention on the input. These are different estimands, and nothing guarantees that a protocol that measures one also measures the other.

This conflation is not merely theoretical. The fragility of reported accuracy under controlled reinstantiation is well documented for standard LLM reasoning benchmarks: chain-of-thought prompting unlocks high accuracy on GSM8K (Wei et al., 2022), yet symbolic re-instantiation with surfacelevel edits drops accuracy by up to 65% even though the added clauses do not change the answer (Mirzadeh et al., 2025). Counterfactual task variants likewise reveal systematic degradation when default task assumptions are altered, distinguishing transferable abstract reasoning from narrower taskspecific procedures (Wu et al., 2024). That work diagnoses counterfactual fragility empirically; we instead ask whether a protocol's observation support point-identifies its claimed estimand on a declared policy class. Recent audits of hundreds of LLM benchmarks find that many operationalize their target constructs through tasks and metrics whose validity is left unjustified (Bean et al., 2025), and that the distributional assumptions underlying standard aggregation can change model rankings (Siska et al., 2024). Work on chain-of-thought faithfulness has shown that generated explanations need not reflect the reasoning actually used (Turpin et al., 2023; Lanham et al., 2023), while counterfactual prompting studies emphasize that an intervention's effect cannot be attributed to the targeted factor without a semantic-preserving baseline (Yang et al. 2026). What these lines share is a growing recognition that measurement validity depends on what an evaluation protocol is capable of distinguishing.

Yet the question is rarely asked in identification terms: given the observations a protocol collects, is the target estimand point-identified at all?

We study this distinction in a controlled setting: solver-grounded three-valued reasoning over open-world signed-Horn theories. An independent solver fixes, before any model run, the oracle label for each intervention world. Models answer under a representation contract (TRUE/FALSE/UNKNOWN labels), a readout channel (generated choice or candidate scoring), and a label mapping (one of the six $S _ { 3 }$ permutations of TRUE/FALSE/UNKNOWN to A/B/C). We ask: when does the observation support of an evaluation protocol suffice to identify a target behavioral estimand? This setting serves as a controlled testbed for protocol diagnosis rather than evidence of unrestricted transport to natural-language evaluation.

Our contribution is a protocol-level identifiability audit: before interpreting a reported score, test whether the protocol collects observations that separate the target estimand from behaviorally distinct alternatives. Identification is defined relative to the declared estimand, policy class, and observation support.

• Finite-class protocol identifiability criterion. We define behavioral policy classes, observation equivalence, and point identification, and prove that a target estimand is point-identified exactly when policies with different estimand values cannot collide on the observed support—relative to the declared policy class and support (Section 2).

• Collision-based synthesis of minimum identifying support. The same collision structure that detects under-identification synthesizes a minimal identifying support $O ^ { * } { \mathrm { : } }$ an exact minimum hitting set over cross-estimand distinguishing cells. In the frozen seven-policy case this synthesis yields 26 minimum supports of size two (none mandatory), versus 36 for the full tensor; the numerical minimum is case-specific, while the synthesis procedure is the methodological contribution (Section 4).

• Controlled empirical demonstration. On frozen responses from two instruction-tuned models, we show that local accuracy and intervention-response fidelity diverge under pair-valid constrained generation, across six balanced oracle-transition directions and a deterministic second source (Sections 3–5).

The central claim is structural and protocolrelative: identification holds on the frozen policy class and support, and its measurement consequences hold across readout contracts, transition directions, and two deterministic sources.

## 2 Formal Setup: Protocol-Level Identification

We cast evaluation design as an identification problem: before asking how accurately a protocol estimates a target behavioral quantity, we ask whether the protocol's observation support is in principle sufficient to identify that quantity. This separates design validity (can the support distinguish the estimand's values?) from estimation accuracy (how noisy is the estimate on a finite sample?), and it lets us diagnose under-identification structurally— without running any model.

## 2.1 Behavioral policies, observation support, and two estimands

Let H be a finite class of deterministic behavioral policies. A policy $h \in \mathcal H$ maps an input under a representation contract to a response. An observation support O is the set of responses a protocol collects. The theoretical target is the fullsupport contract-stable selective-response property $\tau _ { \mathrm { f u l l } } : \mathcal { H }  \{ 0 , 1 \}$ : it equals one only when a policy is base-correct, follows the target oracle, remains stable on matched sham, and does so across both readouts and all six mappings. This binary property is the estimand used by the theorem and synthetic recovery.

The empirical pilot also reports a distinct local paired selective-response rate, $\theta _ { \mathrm { l o c a l } }$ , over units (model, cluster, π, r). A unit succeeds when its base and target responses are valid and oraclecorrect and its sham response is valid and semantically equals its paired base response. Thus $\theta _ { \mathrm { l o c a l } }$ averages contract-local successes, whereas the fullsupport empirical criterion requires every mapping– readout contract to succeed within a model–cluster. Neither denominator nor estimand level is interchangeable.

Two policies $h , h ^ { \prime }$ are observationally equivalent on $O ,$ written $h \equiv _ { O } h ^ { \prime }$ , if they produce identical observable responses on O. Any estimator based only on $O$ must assign the same value to equivalent policies.

![](images/e9f0d2894ba7b6c825fbd7b8f0b7ffe3e64cbee4f9d9097957edee0b3fca24e5.jpg)  
Figure 1: Protocol design overview. (1) Audited policies: seven frozen deterministic behaviors H1–H7 (ideal semantic updater, target inertia, arbitrary edit reactor, fixed-label responder, mapping-conditional shortcut, generationselection failure, candidate-scoring success) with their base/target/sham response rules. (2) Observation geometry: the world×mapping×readout tensor with $3 \times 6 \times 2 = 3 6$ cells; orange marks the selected support O. (3) Identifability audit: base-only support yields 1 class with 6 cross-estimand collisions (every τ=0 alternative is observationally equivalent to the ideal updater on base alone); full support yields 7 classes with 0 collisions (IDENTIFIED). (4) Minimum identifying support O\*: the smallest cell subset that identifies the claim—just 2 cells on this policy class (budget $= | O ^ { * } | )$ . (5) Empirical measurement consequence: base accuracy (62%) and selective-response fidelity (32%) are distinct quantities $( \Delta \approx 3 0 \mathrm { p p } )$

## 2.2 Finite-class identification criterion

Theorem 1 (Point identification on finite behavior classes). τ is point-identified on O if and only if for all h, $h ^ { \prime } \in \mathcal { H } , \tau ( h ) \neq \tau ( h ^ { \prime } )$ implies h ≠o h′.

Proof. (⇒) If some h, h′ with $\tau ( h ) \neq \tau ( h ^ { \prime } )$ satisfy $h \equiv _ { \cal O } h ^ { \prime } .$ then any estimator î measurable with respect to O assigns $\hat { \tau } ( h ) = \hat { \tau } ( h ^ { \prime } )$ , so at least one is wrong. (←) If all pairs with different τ are inequivalent, then τ is constant on each equivalence class, so the estimator $\hat { \tau } ( h ) = \tau ( h ^ { * } )$ for any $h ^ { * } \equiv _ { O }$ h is well-defined and correct. □

The theorem is deliberately finite and nonstatistical: it states a structural condition on the support and the policy class, independent of sample size. Statistical uncertainty (Section 5) sits on top of identification, not in place of it—a protocol that is not point-identified cannot be rescued by more data, only by a larger support.

## 2.3 Interventional response tensor

We organize observations as an Interventional $R e \mathrm { - }$ sponse Tensor $R _ { w , \pi , r }$ . Its indices are world w, label permutation π, and readout r. Worlds are base, target, and sham; $\pi \in S _ { 3 } ;$ readouts are generated choice and candidate scoring. Each cell stores the raw response, validity, decoded semantic response, and solver-grounded oracle. The tensor has three indices, while the protocol requires target and matched-sham observations, paired readouts, and all six mappings. These are four diagnostically distinct support components, not four orthogonal tensor axes. Full support has $3 \times 6 \times 2 = 3 6$ cells per model–cluster.

## 2.4 Corollaries: axis insufficiency

Theorem 1 yields four corollaries, each obtained by exhibiting a pair of policies with different τ that are equivalent when a support component is removed. We state them informally; Section 4 constructs the witnesses.

Base insufficiency. $O = \{ { \bf { b a s e } } \}$ cannot separate a selective updater from target inertia (both answer base correctly).

Sham insufficiency. Dropping matched sham equates a selective updater with an any-edit reactor (both update on target; only sham distinguishes them).

Readout insufficiency. Dropping paired readout equates a semantic failure with a contractspecific reportability failure (a model that fails on one readout is indistinguishable from one that fails semantically on both).

Mapping insufficiency. Dropping full $S _ { 3 }$ equates a semantic updater with an unseen-mapping label shortcut (a model correct on the observed mapping is indistinguishable from one that hard-codes labels).

Each corollary is a constructive collision, not a heuristic: the synthetic witnesses of Section 4 realize each pair explicitly.

## 3 Protocol Support and Component Essentiality

## 3.1 Solver-grounded three-valued semantics

The pilot uses open-world signed-Horn theories under three-valued (TRUE/FALSE/UNKNOWN) semantics. An independent solver fixes, before any model run, the oracle label for each world by checking whether the query is entailed, refuted, or undetermined by the theory. This solver grounding is essential: it removes the experimenter's judgment from the oracle, so the target oracle is a property of the theory plus the intervention, not of a model annotation.

## 3.2 World construction

Three worlds are constructed per cluster. The base world is the original input. The target world applies an edit that changes the solver oracle, while the sham world applies an edit that preserves it. Within the frozen policy class, observing both roles separates oracle-directed response from an anyedit reactor. This is a protocol-discrimination role, not clean causal isolation: target uses support replacement whereas sham adds a query-disjoint fact, so edit type perfectly distinguishes the roles and length, edit-distance, and lexical cues also differ. Formal oracle validation has zero mismatches over 144 worlds, and the automatic surface audit flags material surface confounding. For the human semantic audit, two reviewers independently annotated all target and sham items from 48 balanced clusters (96 items total). Each reviewer received an independently randomized order with role, expected label, and model output hidden, and assigned TRUE, FALSE, UNKNOWN, or UNSURE. They agreed on 96/96 items (Cohen's $\kappa = 1 . 0 )$ , sO no adjudication was required. This audit checks semantic labels, not causal comparability. Consequently, target/sham contrasts identify behavior under these protocol conditions, not a pure semanticedit effect.

## 3.3 Four support components

The frozen audit varies four diagnostically distinct support components:

• target separates selective updating from retaining a base response after the oracle changes;

• matched sham separates oracle-directed updating from responding to any input edit;

• paired readout reveals whether validity and diagnosis depend on the response channel;

• full $S _ { 3 }$ mapping support reveals dependence on a subset of surface-label contracts.

These are protocol requirements over three tensor indices: target and sham are distinct world roles within w, while readout and mapping correspond to r and π. The labels describe behavioral distinctions in observed responses and do not assert internal causes.

## 3.4 Axis-essentiality proposition

Proposition 1. If policies h, h′ exist with $\tau ( h ) \neq$ $\tau ( h ^ { \prime } )$ that are separable under the full support but collide when a support component a is removed, then component a is essential for the policy class and estimand.

The complete subset lattice (Section 4) supplies constructive evidence for all four components: each leave-one-out support retains a collision between the ideal updater $( \tau _ { \mathrm { f u l l } } = 1 )$ and at least one $\tau _ { \mathrm { f u l l } } =$ 0 alternative. Removing any component therefore destroys point identification on this policy class.

## 4 Synthetic Policy Recovery

This section performs zero model calls: it validates that the measurement design and the analyzer can recover behavioral differences that are known by construction, separating design validity from any property of a particular LLM. Concretely, we ask whether the observation support O suffices to pointidentify the target estimand $\tau$ (selective-response fidelity) on a finite, frozen policy class H, and we answer by exhaustive enumeration rather than by estimation.

<table><tr><td></td><td>Support Axis added</td><td>|0|</td><td>Classes</td><td>Coll.</td><td>Identified</td></tr><tr><td> $O _ { 0 }$ </td><td>base</td><td>1</td><td>1</td><td>6</td><td>no</td></tr><tr><td> $O _ { 1 }$ </td><td>+ target</td><td>2</td><td>2</td><td>3</td><td>no</td></tr><tr><td> $O _ { 2 }$ </td><td>+ sham</td><td>3</td><td>3</td><td>2</td><td>no</td></tr><tr><td> $O _ { 3 }$ </td><td>+ readout</td><td>6</td><td>5</td><td>1</td><td>no</td></tr><tr><td> $O _ { 4 }$ </td><td> $+ \operatorname { f u l l } S _ { 3 }$ </td><td>36</td><td></td><td>70</td><td>yes</td></tr></table>

Table 1: One fixed addition path through the 16- subset lattice. [O| is projection size, “Classes" counts observation-equivalence classes among seven deterministic policies, and $\mathrm { \hat { \Omega } C o l l . \vec { \Omega } ^ { \mathrm { \prime } } }$ counts cross-estimand collision pairs. Only full support $O _ { 4 }$ point-identifies $\tau _ { \mathrm { f u l l } }$ on the frozen policy class.

## 4.1 Frozen policy class

We implement seven deterministic policies plus one stochastic mixture. The reference ideal semantic updater has $\tau _ { \mathrm { f u l l } } { = } 1$ ; the other six deterministic policies have $\tau _ { \mathrm { f u l l } } { = } 0$ . They cover inertia and indiscriminate reaction (target inertia, any-edit reactor); label-based shortcuts (fixed-label responder, mapping-conditional shortcut); and readoutspecific failures (generated-only failure, scoringonly success). The separate stochastic mixture mixes ideal and inertia behavior at a declared rate and is analyzed only at its stated granularity.

## 4.2 Complete subset lattice and one addition path

We enumerate the complete $2 ^ { 4 } = 1 6$ subset lattice over target, sham, paired readout, and full $S _ { 3 }$ mapping support. For readability, Table 1 and Figure 2 show one fixed cumulative addition path, $O _ { 0 }$ through $O _ { 4 } ;$ the path is illustrative rather than a unique ordering. The leave-one-out results in Table 2 establish component necessity without relying on that ordering, and Appendix D reports all 16 subsets.

## 4.3 Recovery results

Under $O _ { 0 }$ (base only), all seven deterministic policies project to a single observation-equivalence class, and the estimand is not point-identified: six collision pairs link the ideal updater to every $\tau { = } 0$ alternative. Each added axis splits classes: $O _ { 1 }$ yields 2 classes (collapses 3 collisions), $O _ { 2 }$ yields 3 (2 remain), $O _ { 3 }$ yields 5 (1 remains, between ideal\_semantic\_updater and mapping\_conditional\_shortcut), and $O _ { 4 }$ yields 7 singleton classes with no estimand collisions (point\_identified=1). The trajectory is monotone: removing any axis from $O _ { 4 }$ reintroduces at least one collision.

<table><tr><td>Removed</td><td>|0| Coll.</td><td>One witness against ideal</td></tr><tr><td>Target</td><td>24</td><td>3 target inertia</td></tr><tr><td>Matched sham</td><td>24</td><td>1 any-edit reactor</td></tr><tr><td>Paired readout</td><td>18</td><td>1 generated-only failure</td></tr><tr><td>Full  $S _ { 3 }$ </td><td>6</td><td>1 mapping shortcut</td></tr></table>

Table 2: Leave-one-out audit from the canonical 16- subset lattice. Every reduced support has at least one cross-estimand collision; full support has projection size 36 and zero collisions.

## 4.4 Component essentiality via leave-one-out witnesses

For each component a, we remove only a from full support. Every leave-one-out support retains a cross-estimand collision (Table 2), while full support has none. Target removal yields three collisions involving the ideal policy; each other removal yields one exclusive witness. Thus all four components are necessary for point identification of $\tau _ { \mathrm { f u l l } }$ on this policy class.

## 4.5 Synthesis of minimum identifying support 0\*

The same collision structure that certifies underidentification also enables protocol synthesis: instead of auditing a fixed support, find the smallest support that still separates every cross-estimand pair. For each pair $h , h ^ { \prime }$ with $\tau ( h ) \neq \tau ( h ^ { \prime } )$ , let $D _ { h , h ^ { \prime } } = \{ o \in O _ { \mathrm { f u l l } } : h ( o ) \neq h ^ { \prime } ( o ) \}$ be its distinguishing cells. A support $S \subseteq O _ { \mathrm { f u l l } }$ point-identifies τ iff S is a hitting set of $\{ D _ { h , h ^ { \prime } } \} , \mathrm { i . e . } S \cap D _ { h , h ^ { \prime } } \neq \emptyset$ for every cross-estimand pair. We enumerate all minimum-size hitting sets exactly.

The synthesis procedure is the methodological contribution; its numerical outcome is case-specific. On the frozen seven-policy class it yields a striking result: $| O ^ { * } | = 2$ cells—not 36. There are 26 distinct 2-cell supports (no cell is mandatory): every minimum pairs one target-generated-choice cell with one sham-candidate-scoring cell (target mapping in {1, 3, 5} paired with any sham mapping, or {2, 4} paired with sham mapping $\geq 2 ;$ Appendix N). Removing either cell from $O ^ { * }$ reintroduces a collision. Under equal cell costs the minimal cost is 2; if candidate scoring costs three times a generated choice, every 2-cell support costs 4 (one scoring cell at 3 plus one generated cell at 1). Thus the audit does not demand the full tensor: identification has a precise price, and the framework both detects under-identification and synthesizes the cheapest support that removes it.

![](images/c4f8da10a66b1443aa84f30e7b6d4d7f9c9ef1a4fd56c5466c260351b0a59686.jpg)  
Figure 2: Support-component essentiality: the full power-set lattice over target (T), sham (S), paired readout (R), and full- $S _ { 3 }$ mapping (M). Each node reports its observation-equivalence class count and cross-estimand collision count over the seven frozen policies. Only full support TSRM is identified (7 classes, 0 collisions; UNIQUE). The four rank-3 nodes are the leave-one-out witnesses: removing any component reintroduces at least one collision (e.g., removing target leaves SRM with 3 collisions between target inertia and the ideal updater); removing more components only adds collisions and merges classes.

The cost saving is not purely formal: scoring the same real responses under the 2-cell support $O ^ { * }$ still reproduces the accuracy-selective dissociation (gaps of ≈0.71 and ≈0.52 across the two models), while the strict 36-cell criterion is satisfied by 0/48 model-clusters, matching the full-support audit.

## 4.6 Granularity bound on mixture recovery

The stochastic mixture is recovered only at its declared granularity: the analyzer identifies the equivalence class containing the mixture (separating it from every deterministic alternative under $O _ { 4 } )$ but does not identify the exact mixture parameters from behavioral observation alone. This is a genuine identifiability bound, not an analyzer limitation: two mixtures with the same support-projected behavior are observationally equivalent by construction, so no estimator based on O can separate them.

## 5 Empirical Consequences on Frozen Model Responses

## 5.1 Diagnostic case: original 24-cluster pilot

The diagnostic case spans 24 clusters, six mappings, two readouts, three worlds, and two models: Qwen/Qwen2.5-7B-Instruct (Qwen Team, 2024) and meta-1lama/Llama-3.1-8B-Instruct (Llama Team, AI @ Meta, 2024). This gives 1,728 cells and 576 paired base/target/sham units. Target flips the oracle to UNKNOWN; matched sham preserves it. Generated choice parses free-form

A/B/C answers, whereas candidate scoring uses conditional sequence log-probability. All intervals use 10,000 whole-cluster bootstrap replicates; contract cells are repeated measurements, not independent units.

## 5.2 Two empirical estimands

We distinguish semantic response fidelity from endto-end contract fidelity. The former evaluates decoded semantic behavior conditional on a valid response; the latter additionally treats failures to produce a valid response under the required readout contract as failures. The paired-readout audit is therefore primarily an audit of end-to-end protocol dependence. The frozen local estimand $\theta _ { \mathrm { l o c a l } }$ is the paired selective-response rate over all base-target pairs: the proportion of 576 paired units where base and target are valid and oracle-correct and sham is valid and semantically equals its paired base response. Whole-cluster bootstrap resamples the 24 clusters and preserves all repeated contracts within each cluster. Separately, the empirical full-support criterion is evaluated on 48 model–clusters and requires the selective-response contract to hold across all 36 world-mapping-readout cells. This second criterion is the empirical counterpart of $\tau _ { \mathrm { f u l l } } ;$ it is not an ancillary sensitivity analysis of $\theta _ { \mathrm { l o c a l } }$

## 5.3 Conditional fidelity is the headline

Because selective response requires base correctness, target following, and sham stability, the headline is conditional: only 0.138 of base-correct units satisfy it (32/232; cluster-bootstrap 95% CI [0.068, 0.218]). Target following is 0.168, versus 0.918 sham stability, locating the shortfall mainly in target response. Separately, 0/48 model–clusters satisfy the full-support criterion. These rates do not become the binary synthetic property; they show why selective response must be measured rather than inferred from local correctness.

<table><tr><td>Metric</td><td>Estimate [95% CI]</td><td>N</td></tr><tr><td>Base accuracy</td><td>0.403</td><td>576</td></tr><tr><td>Local response | base-correct</td><td>0.138 [0.068, 0.218] 232</td><td></td></tr><tr><td>Target-follow | base-correct</td><td>0.168—</td><td>232</td></tr><tr><td>Sham-stay | base-correct</td><td>0.918</td><td>232</td></tr><tr><td>Local paired response</td><td>0.056 [0.026, 0.090] 576</td><td></td></tr><tr><td>Full-support response</td><td>0.000—</td><td>48</td></tr></table>

Table 3: Estimand-separated results. The bold conditional headline is selective response among base-correct units (cluster bootstrap); target-follow and sham-stay decompose it. Local paired response and the full-support model-cluster criterion are distinct estimands.

## 5.4 Behavioral composition

The five-class rule yields 317 inertia diagnoses and 183 invalid units, but does not require base correctness. A mutually exclusive audit instead finds 183 reportability failures, 179 base-knowledge failures, and 164 target-inertia cases; target inertia is therefore the largest base-correct semantic nonresponse, not the dominant category overall. Sham stability holds for 349/576 units.

## 5.5 Contract and transition robustness

We report three bounded robustness checks. Figure 3 shows the balanced-transition analysis; Appendix F reports equal-validity contracts and bounded source transport. Equal-validity contracts give both constrained-generation variants pair-validity 1.0; pooled candidate scoring has pairvalidity 0.9826. Every model-readout stratum nevertheless retains a positive accuracy-response gap. Balanced transitions cover six directions across 48 clusters: equal-weighted selective response is 0.324 (95% CI [0.304, 0.345]), versus base accuracy 0.620 ([0.600, 0.642]), with $P ( \Delta \ > \ 0 )$ 二 1.0 over synchronized cluster bootstraps. Conditional response remains 0.244–0.259 across the two mapped readouts. Source transport is limited to one deterministic second source (24 clusters), with rates 0.331 ([0.312, 0.350]) and 0.646 ([0.618, 0.674]).

A second solver-generated deterministic source uses an independent synthetic vocabulary and covers all six non-identity TRUE/FALSE/UNKNOWN transition directions, with four clusters per direction (24 total). We generated the full set, applied a deterministic shuffle with seed 20260806, and performed no result-based filtering. Intervals use 10,000 synchronized whole-cluster bootstrap replicates (seed 20260805); balanced pooling weights directions equally and resamples clusters within direction. Contract cells remain repeated measurements. These checks do not establish population transport or a pure target-edit effect.

## 6 Behavioral Diagnoses Across Protocol Components

We localize observed contract failures without inferring internal causes; all decompositions are exploratory and post-result. Table 4 stratifies by model and readout, and Appendix Figures 4–5 audit mapping dependence and readout validity.

Target and sham. Target follow is 0.0677; the exclusive audit finds 164 target-inertia and 179 base-knowledge failures. Sham stability is 0.6059 (349/576); of the remaining 227 units, 183 are invalid and 44 valid but non-stable. Thus target and sham answer different diagnostic questions.

Mapping and readout. Semantic equivariance is 0.0625 (18/288 groups), rising to 0.1023 after validity filtering; fixed-label attachment is 0.0590. Generated-choice validity is 0.4560, versus 0.9942 for scoring. When both readouts are valid, semantic answers always agree (base 132/132, target 125/125, sham 137/137). Thus the 115/288 fiveclass agreement in Figure 5 reflects validity and diagnosis availability, not demonstrated semantic disagreement: paired readout matters mainly to end-to-end contract fidelity.

Contract and transition. Rates vary by model and readout, but base accuracy and local response induce the same four-contract ordering; we claim no ranking reversal. FALSE→UNKNOWN and TRUE→UNKNOWN rates are 0.0224 and 0.0947, an exploratory asymmetry.

## 7 Related Work

We focus on one component of evaluation validity: identifiability relative to a declared behavioral class and support, which our framework makes mechanically auditable. Measurement work separates constructs from operationalizations and questions benchmark validity (Jacobs and Wallach, 2021;

![](images/064ecce26f4a01c0660f4e5c562b7cee98c312c519a309d11634d3f377a2831d.jpg)  
Figure 3: Accuracy-response gaps across six balanced oracle-transition directions with synchronized clusterbootstrap intervals. Squares show base accuracy; circles show selective response. These are measurement consequences, not pure edit effects or population transport.

Wallach et al., 2025; Bowman and Dahl, 2021; Bean et al., 2025; Siska et al., 2024). Behavioral tests and contrast sets show that accuracy can hide intervention failures (Ribeiro et al., 2020; Gardner et al., 2020); counterfactual prompting motivates semantic-preserving baselines (Yang et al., 2026). Chain-of-thought work studies whether explanations reflect process (Turpin et al., 2023; Lanham et al., 2023; Yee et al., 2024; Paul et al., 2024; Tutek et al., 2025; Zaman and Srivastava, 2025). We instead use constructive collisions to audit point identification of black-box response fidelity, without inferring mechanisms.

## 8 Discussion

Identification depends jointly on estimand, policy class, and support. On the frozen class, base-only observation collapses all policies; full support separates them; and each leave-one-out support retains a cross-estimand collision. The 5.56% local rate and 0/48 full-support result answer different questions, preventing local success from implying contract stability.

For protocol design, declare the estimand, enumerate plausible collisions, and add separating measurements. For $\tau _ { \mathrm { f u l l } }$ on this class, all four support components have leave-one-out witnesses. Collision synthesis then converts measurement budget into a finite optimization problem: under the stated cell-cost model, a minimum separating support has two cells. This pre-inference audit complements broader construct-validity arguments.

## 9 Conclusion

For the frozen class, the audit exposes base-only collisions, verifies full-support identification, and synthesizes minimum separating support. Empirically, accuracy-response gaps persist under pairvalid generation, six balanced transitions, and one deterministic second source. Evaluation should state its behavioral estimand and test whether support identifies it before interpreting a score. The present study establishes this workflow in a controlled solver-grounded reasoning setting.

## Limitations

The structural results are relative to a frozen sevenpolicy class, not an exhaustive model of naturallanguage behavior; additional policies may require more support. Solver-grounded signed-Horn data provide exact oracles and enable exhaustive audits; external validity beyond this controlled setting remains to be established. Evidence covers two instruction-tuned models, 24 diagnostic clusters, 48 balanced clusters, and one bounded synthetic source change, with cluster-level uncertainty; it does not establish transport to natural tasks or model populations. Target and sham differ in edit type and surface form, so their contrast supports protocol discrimination, not a pure edit effect. Local and full-support criteria use different units and cannot be pooled. Post-result behavioral diagnoses do not identify mechanisms.

## Ethical Considerations

Explicit estimands and identification checks reduce over-interpretation of behavioral scores as evidence of ability or mechanism. Qwen2.5 and Llama 3.1 are used under their public licenses; inference is local and no weights are redistributed.

## Data and Code Availability

Code, data, and frozen evaluation artifacts will be released upon acceptance.

## References

Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia Hafner, Harry Mayne, Jan Batzner, Negar Foroutan Eghlidi, Chris Schmitz, Karolina Korgul, Hunar Batra, Oishi Deb, Emma Beharry, Cornelius Emde, Thomas Foster, Anna Gausen, María Grandury, Sophia Han, Valentin Hofmann, Lujain Ibrahim, and 23 others. 2025. Measuring what matters: Construct validity in large language model benchmarks. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Samuel R. Bowman and George Dahl. 2021. What wil it take to fix benchmarking in natural language understanding? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4843–4855, Online. Association for Computational Linguistics.

Matt Gardner, Yoav Artzi, Victoria Basmov, Jonathan Berant, Ben Bogin, Sihao Chen, Pradeep Dasigi, Dheeru Dua, Yanai Elazar, Ananth Gottumukkala, Nitish Gupta, Hannaneh Hajishirzi, Gabriel Ilharco, Daniel Khashabi, Kevin Lin, Jiangming Liu, Nelson F. Liu, Phoebe Mulcaire, Qiang Ning, and 7 others. 2020. Evaluating models' local decision boundaries via contrast sets. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1307–1323, Online. Association for Computational Linguistics.

Abigail Z. Jacobs and Hanna Wallach. 2021. Measurement and fairness. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 375–385, New York, NY, USA. Association for Computing Machinery.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamilė Lukošiūtė, Karina Nguyen, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, and 11 others. 2023. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702.

Llama Team, AI @ Meta. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. 2025. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. In International Conference on Learning Representations (ICLR).

Debjit Paul, Robert West, Antoine Bosselut, and Boi Faltings. 2024. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 15012– 15032, Miami, Florida, USA. Association for Computational Linguistics.

Qwen Team. 2024. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115.

Inioluwa Deborah Raji, Emily Denton, Emily M. Bender, Alex Hanna, and Amandalynne Paullada. 2021. AI and the everything in the whole wide world benchmark. In Advances in Neural Information Processing Systems: Datasets and Benchmarks Track, volume 34. Curran Associates, Inc.

Marco Tulio Ribeiro, Tongshuang Wu, Carlos Guestrin, and Sameer Singh. 2020. Beyond accuracy: Behavioral testing of NLP models with CheckList. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4902– 4912, Online. Association for Computational Linguistics.

Olawale Salaudeen, Anka Reuel, Ahmed Ahmed, Suhana Bedi, Zachary Robertson, Sudharsan Sundar, Ben Domingue, Angelina Wang, and Sanmi Koyejo. 2025. Measurement to meaning: A validitycentered framework for AI evaluation. arXiv preprint arXiv:2505.10573.

Charlotte Siska, Katerina Marazopoulou, Melissa Ailem, and James Bono. 2024. Examining the robustness of LLM evaluation to the distributional assumptions of benchmarks. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10406–10421, Bangkok, Thailand. Association for Computational Linguistics.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. 2023. Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023).

Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, and Yonatan Belinkov. 2025. Measuring chain of thought faithfulness by unlearning reasoning steps. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9935–9960, Suzhou, China. Association for Computational Linguistics.

Hanna Wallach, Meera Desai, A. Feder Cooper, Angelina Wang, Chad Atalla, Solon Barocas, Su Lin Blodgett, Alexandra Chouldechova, Emily Corvi, P. Alex Dow, Jean Garcia-Gathright, Alexandra Olteanu, Nicholas J. Pangakis, Stefanie Reed, Emily Sheng, Dan Vann, Jennifer Wortman Vaughan, Matthew Vogel, Hannah Washington, and Abigail Z. Jacobs. 2025. Position: Evaluating generative AI systems is a social science measurement challenge. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 82232–82251. PMLR.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35 (NeurIPS).

Zhaofeng Wu, Linlu Qiu, Alexis Ross, Ekin Akyürek, Boyuan Chen, Bailin Wang, Najoung Kim, Jacob Andreas, and Yoon Kim. 2024. Reasoning or reciting? exploring the capabilities and limitations of language models through counterfactual tasks. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1819–1862, Mexico City, Mexico. Association for Computational Linguistics.

Zihao Yang, Mosh Levy, Yoav Goldberg, and Byron C. Wallace. 2026. Compared to what? baselines and metrics for counterfactual prompting. arXiv preprint arXiv:2605.01048. Accepted at COLM 2026.

Evelyn Yee, Alice Li, Chenyu Tang, Yeon Ho Jung, Ramamohan Paturi, and Leon Bergen. 2024. Dissociation of faithful and unfaithful reasoning in LLMs. arXiv preprint arXiv:2405.15092.

Kerem Zaman and Shashank Srivastava. 2025. A causal lens for evaluating faithfulness metrics. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 29425–29449, Suzhou, China. Association for Computational Linguistics.

## A Model-Readout Behavioral Composition

<table><tr><td>Contract</td><td>Sel. Nonsel. Perv. Inert.</td><td></td><td></td><td></td><td>Inval.</td><td>Total</td></tr><tr><td>Qwen·scoring</td><td>17</td><td>2</td><td>18</td><td>107</td><td>0</td><td>144</td></tr><tr><td>Qwen·choice</td><td>9</td><td>2</td><td>12</td><td>83</td><td>38</td><td>144</td></tr><tr><td>Llama·scoring</td><td>6</td><td>3</td><td>7</td><td>123</td><td>5</td><td>144</td></tr><tr><td>Llama·choice</td><td>0</td><td>0</td><td>0</td><td>4</td><td>140</td><td>144</td></tr><tr><td>Total</td><td>32</td><td>7</td><td>37</td><td>317</td><td>183</td><td>576</td></tr></table>

Table 4: Five-class behavioral composition by model– readout contract. Labels are behavioral diagnoses; inertia does not imply base correctness.

![](images/9343f87e14eb367591266102cb327f15aafa29135ac97b61bf9f996ccb279ff9.jpg)  
Figure 4: Mapping audit: rates before (n = 288) and after validity filtering (n = 176). Numerators remain fixed for semantic equivariance, fixed-label attachment, and equivariance plus correctness, so each rate rises by 2.4–4.0 percentage points; denominator choice changes rates without changing counts.

## B Symbols and Policy Definitions

Table 5 summarizes the notation used in the main text. A policy h is a deterministic function from a world-mapping-readout triple $( w , \pi , r )$ to a raw response. The semantic response is the raw response decoded under the contract defined by π; the oracle semantic response is fixed by the independent solver before any model run.

<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td>H</td><td>Finite behavioral policy class</td></tr><tr><td>0</td><td>Observation support (set of projected responses)</td></tr><tr><td> $\tau _ { \mathrm { f u l l } } ( h ) \in \{ 0 , 1 \}$ </td><td>Full-support contract-stable selective-response property</td></tr><tr><td>θlocal</td><td>Local paired selective-response rate over model-cluster-mapping-</td></tr><tr><td>h ≡0 h′</td><td>readout units Observational equivalence on support O</td></tr><tr><td>w ∈ {base, target, sham} π ∈ S3</td><td>Intervention world Label permutation (bijection TRUE/FALSE/UNKNOWN</td></tr><tr><td>r ∈ {gc, cs}</td><td>→ A/B/C) Readout: generated choice or</td></tr><tr><td> $R _ { w , \pi , r }$ </td><td>candidate scoring Interventional response</td></tr><tr><td>Ok</td><td>tensor cell Support after adding k observation axes</td></tr></table>

Table 5: Notation summary.

## C Exclusive Witnesses for Axis Essentiality

For each support component a we inspect policies with different $\tau _ { \mathrm { f u l l } }$ that separate under full support but collide under the corresponding leave-one-out support. The witnesses are realized by the seven deterministic policies of Section 4:

![](images/5f313502b5ee370ce7d7557b80c1f8dad3cb82c3e69d3fbcdcb524ba0e100022.jpg)  
Figure 5: Paired readout transition from generated choice to candidate scoring $( n ~ = ~ 2 8 8 )$ as proportional flows: 173 invalid generated choices are recovered to valid by scoring and valid never regresses (0 valid→invalid), alongside 110 valid–-valid and 5 invalid– invalid pairs. Among both-valid responses, semantics agree completely in base, target, and sham worlds.

• Target update: ideal\_semantic\_updater vs. target\_inertia. Both answer base correctly; only the target world separates them.

• Matched sham: ideal\_semantic\_updater vs. any\_edit\_reactor. Both answer base and target in the same direction; only the sham world (where the reactor changes but the updater does not) separates them.

• Paired readout: ideal\_semantic\_updater vs. generated\_only\_failure. The two readouts agree on base and target but disagree on whether the failure is readout-specific.

• Full S3: ideal\_semantic\_updater vs. mapping\_conditional\_shortcut.The shortcut updates only on a subset of label permutations; full $S _ { 3 }$ exposes the dependence.

These witnesses make Proposition 1 constructive: omitting any axis reintroduces at least one collision.

## D Complete Synthetic Recovery Decomposition

Table 6 reports all 16 subsets in the canonical lattice. Let T, S, R, and M denote target, sham, paired

readout, and full $S _ { 3 }$ mapping support. Only TSRM point-identifies Tfull.
<table><tr><td>Support</td><td>|0|</td><td>Classes</td><td>Coll.</td><td>Support</td><td>t |0|</td><td>Classes Coll.</td></tr><tr><td>∅</td><td>1</td><td>1</td><td>6</td><td>TS</td><td>3</td><td>3 2</td></tr><tr><td>T</td><td>2</td><td>2</td><td>3</td><td>TR</td><td>4</td><td>4 2</td></tr><tr><td>S</td><td>2</td><td>2</td><td>5</td><td>TM</td><td>12</td><td>4 2</td></tr><tr><td>R</td><td>2</td><td>2</td><td>5</td><td>SR</td><td>4</td><td>3 4</td></tr><tr><td>M</td><td>6</td><td>2</td><td>5</td><td>SM</td><td>12</td><td>3 4</td></tr><tr><td>RM</td><td>12</td><td>3</td><td>4</td><td>TSR</td><td>6</td><td>5 1</td></tr><tr><td>TSM</td><td>18</td><td>5</td><td>1</td><td>TRM</td><td>24</td><td>6 1</td></tr><tr><td>SRM</td><td>24</td><td>4</td><td>3</td><td>TSRM</td><td>36</td><td>7 0</td></tr></table>

Table 6: Complete 16-subset support lattice. “Coll." counts cross-estimand policy pairs. The four threecomponent rows are the leave-one-out supports.

## E Oracle-Transition Asymmetry

Table 7 reports the frozen primary estimand stratified by the base-to-target oracle transition. The data are exploratory and post-result. Selective response is more than four times more frequent after TRUE base oracles than after FALSE base oracles suggesting that models update more readily when a previously entailed proposition becomes undetermined than when a previously refuted proposition does.

<table><tr><td>Transition</td><td colspan="3">Units Sel. rate Sel. if base-correct</td></tr><tr><td>FALSE→UNKNOWN</td><td>312</td><td>0.022</td><td>0.065</td></tr><tr><td>TRUE→UNKNOWN</td><td>264</td><td>0.095</td><td>0.202</td></tr></table>

Table 7: Oracle-transition asymmetry. Intervals are omitted because this decomposition is exploratory and not part of the confirmatory estimand.

## F Contract and Source Robustness

G Target-Sham Isolation Audit

## H Exclusive Behavioral Diagnoses

Figure 8 shows the mutually exclusive behavioral diagnoses over the 576 paired units. Reportability failure (183) is the largest category, followed by base-knowledge failure (179) and target inertia (164); target inertia is the largest base-correct semantic non-response category but not the dominant diagnosis overall.

## I Implementation Details

Solver grounding. For each cluster, an independent solver computes the oracle label (TRUE, FALSE, or UN-KNOWN) for the query with respect to the open-world signed-Horn theory. The target world is generated by adding a literal that flips an entailed or refuted query to UNKNOWN; the matched sham world applies a surface edit that preserves the oracle. All oracle labels are fixed before any model call.

![](images/2604e3dca65e211134a11e6487df93db9cfbd6db1b1f92f703350aeafcec9fd1.jpg)

(b) External transport (deterministic second source)  
![](images/37e6bd1685d82f2004d97e805c7fd08bb8000acdcc13ee580380ccbe4060fa1a.jpg)  
Figure 6: Robustness readout contracts (equal validity) and bounded source transport: the accuracy-selective gap persists in every model-readout and source-model-readout stratum, with positive ∆ annotations.

Readouts. The generated-choice readout presents the query in free-form text and parses the model's output into one of A/B/C using a contract-specific prompt. The candidate-scoring readout requests log-probabilities for the three candidates under the same contract and selects the highest. Both readouts are decoded into the semantic TRUE/FALSE/UNKNOWN space via the inverse of the active S3 permutation.

Inference. Both checkpoints are frozen at the versions reported in Section 5. Generated choice uses greedy decoding (do\_sample=false, max\_new\_tokens=8); candidate scoring selects the candidate with the largest conditional sequence log-probability sum. The real-model pilot required no hyperparameter search or fine-tuning. The synthetic policy-recovery study requires no model inference and enumerates the frozen policies exhaustively.

Attention-mask reproducibility check. After correcting the Llama generated-choice attention-mask handling, a completed rerun produced 864 responses (432 regenerated choice responses plus 432 reused scoring responses) exactly equal to the prior recorded outputs, including all raw and parsed responses. This rules out the missing-mask warning as an implementation-level alternative explanation for the reported results.

Bootstrap. Inference is performed at the cluster level via whole-cluster bootstrap (10,000 replicates). Cells within a cluster are repeated measurements under different contracts, not independent replicates, so the cluster is the resampling unit.

## J Algorithms for Equivalence-Class and Collision Analysis

The synthetic recovery study is a pure enumeration over the frozen policy class. Algorithm 1 constructs observation profiles, partitions policies into observation-equivalence classes, and records every pair of policies that share a profile while differing on the target estimand. Algorithm 2 then checks each support component for essentiality by removing that component from the full support and testing whether any collision remains.

![](images/a83a009e8f0a5e5c348dce4727fd9939cb90895630ec564d909a68d12c94fce6.jpg)  
(c) response length

![](images/ae2a63025eff43f8f5b2d80e44cd194b211cfe2984c6e03b2f139036e036aa79.jpg)

![](images/0342e2fa2006398e0e32f1e7986c6fa038a6f5432bf45c960efb005266a32c6d.jpg)

(d) edit type (perfect separator)  
![](images/709247613f76e0cb5f53e04a8f7d89b87a1bc1b5567935b10e07b591113195a0.jpg)  
Figure 7: Target-sham isolation audit: the formal oracle passes on 144 worlds; the surface audit flags material confounding, and the blinded human review passed with perfect inter-annotator agreement (96/96, Cohen's $\kappa = 1 . 0 )$ The right panels show the target/sham surface imbalance that motivates the matched negative-control construction.

![](images/5c29374eecd9fee5cc7229a8cdfe05b6adaa5ea7e62d60e0673e83144061d7c1.jpg)  
Figure 8: Six mutually exclusive diagnoses over 576 paired units (counts in bars, cumulative share in red). Three archetypes—reportability failure, baseknowledge failure, and target inertia—account for 91.3% of units. These post-result labels do not imply internal causes.

```latex
Algorithm 1 Equivalence classes and collisions
Require: Policy class ${ \mathcal { H } } ,$ target estimand $\tau : \mathscr { H }  \{ 0 , 1 \}$
observation support O
Ensure: Equivalence classes C and collision set Coll
1: for $h \in { \dot { \mathcal { H } } }$ do
2: $p [ h ]  \{ ( o , h ( o ) ) : o \in O \}$ observation profile
3: end for
4: C ← partition of H by equality of $p [ h ]$
5: $\mathsf { C o l l } \overset { \cdot } { \gets } \{ ( h , h ^ { \prime } ) \in \dot { \mathcal { H } } ^ { 2 } : \overset { \cdot } { \tau } ( h ) \overset { \cdot } { \neq } \tau ( \dot { h } ^ { \prime } ) \wedge p [ h ] = p [ h ^ { \prime } ] \}$
6: return $( \dot { \boldsymbol { { \mathcal { C } } } } , \mathsf { C o l l } )$
```

Algorithm 2 Check axis essentiality   
Require: $\mathcal { H } , \ \tau ,$ full support $O _ { 4 } ,$ components $A \quad =$   
$\{ T , S , R , M \}$   
Require: T: target; S: sham; $R { : }$ paired readout; M: full- $S _ { 3 }$   
mapping   
Ensure: Set E of essential support components   
$1 \colon { \cal O } _ { - T }  { \cal O } _ { S R M } ; { \cal O } _ { - S } \stackrel {  } {  } { \cal O } _ { T R M } ; { \cal O } _ { - R }  { \cal O } _ { T S M } ;$   
$O _ { - M }  O _ { T S R }$   
2: $E  \emptyset$   
3: for each support component $a \in A$ do   
4: remove support component a by selecting $O _ { - a }$   
5: $( { \mathcal { C } } , { \mathsf { C o l l } } ) \mathrel { \mathop { \longleftarrow } } { \mathrm { C o M P U T E C L A S S E S } } ( { \mathcal { H } } , \tau , \breve { O } _ { - a } )$   
6: if Coll $\neq \emptyset$ then   
7: mark a essential; $E  E \cup \{ a \}$   
8: end if   
9: end for   
10: return $E$

Both algorithms are deterministic and run without any model call; their cost is $O ( | \mathcal { H } | ^ { 2 } \cdot | O _ { 4 } | )$ for the full enumeration reported in Table 1.

## K Frozen Policy Definitions

Table 8 summarizes seven deterministic policies and the separately analyzed stochastic mixture. The deterministic policy class is used for all componentessentiality checks.

<table><tr><td>Policy</td><td>T</td><td>Behavior on full  $O _ { 4 }$ </td></tr><tr><td>ideal_semantic_ updater</td><td>1</td><td>Correct on base; follows the target oracle; keeps sham identical to base; behaves consistently</td></tr><tr><td>target_inertia</td><td>0</td><td>across both readouts and all six  $S _ { 3 }$  mappings. Correct on base; repeats the base answer on target and sham regardless of</td></tr><tr><td>any_edit_reactor</td><td>0</td><td>oracle change. Changes answer whenever the input is edited, both on target and on matched</td></tr><tr><td>fixed_label_ responder</td><td>0</td><td>sham. Always returns the same surface label; separable only once all six  $S _ { 3 }$  mappings are observed.</td></tr><tr><td>mapping_ conditional_ shortcut</td><td>0 of</td><td>Updates only on a subset  $S _ { 3 }$  mappings; appears semantic on the observed mapping.</td></tr><tr><td>generated_only_ failure</td><td>0</td><td>Fails semantically on generated choice while answering correctly under candidate scoring.</td></tr><tr><td>scoring_only_ success</td><td>0</td><td>Answers correctly only under candidate scoring; the symmetric counterpart to the previous policy.</td></tr><tr><td>stochastic_ mixture</td><td>0-1</td><td>Mixes ideal updater behavior with target inertia at a declared rate; recovered only at its mixture granularity.</td></tr></table>

Table 8: Frozen behavioral policies in the synthetic recovery study. Policies are deterministic except for the stochastic mixture, which is a weighted combination of two deterministic policies.

## L Interventional Response Tensor Schema

The full observation support $O _ { 4 }$ is the projection of the interventional response tensor $R _ { w , \pi , r }$ onto all three indices. Table 9 lists the index ranges and their semantics; the product space contains $3 \times 6 \times 2 = 3 6$ cells per (model, cluster).

## M Data Provenance

All reported numbers and figures are generated from frozen analysis outputs.

<table><tr><td>Index</td><td>Values</td><td>Meaning</td></tr><tr><td>World  $w$ </td><td>base, target, sham</td><td>Intervention applied to the query. Target flips oracle to UNKNOWN;</td></tr><tr><td></td><td>Mapping π 6 permutations of  $S _ { 3 }$ </td><td>sham preserves it. Bijection from semantic labels TRUE, FALSE, UNKNOWN to surface</td></tr><tr><td>Readout r gc, cs</td><td></td><td>labels A, B, C. gc: generated choice (free-form output parsed to A/B/C); cs: candidate scoring (log-probability</td></tr></table>

Table 9: Tensor indices for the interventional response tensor $R _ { w , \pi , r }$

## N Minimum-Support Synthesis Illustration

(a) Full support  
![](images/d4469c609d72ae2463d40b559b12c1e7799d70f443de9ea4952c80612d766f2e.jpg)

![](images/2c49efc457bed1a4d59f4c772f735fb5568866656f12308a57e5065465571c98.jpg)

(b) Minimal O\*  
![](images/dc72caf983ac7b766ac9f338a089b74a6d3c662d2cef7c0f4701dd9041fc1d5b.jpg)  
shade = # of the 26 minima that use the cell  
★ marks one example 2-cell minimum {target|m1|gen, sham|m0|score}

(c) Reduced, non-identifying  
![](images/d8c13aa09f5f50be60db04fc0d06d0c9c5d9a2659e998cb618e3689f107db7c9.jpg)

![](images/a6f3d294cadc5b1a3da6c344ea18f58be2f71adcaf4ef5cae8b5057d884af990.jpg)  
Figure 9: Protocol synthesis on the 36-cell tensor. (a) Full support: each cell is shaded by its distinguishing load—how many of the 6 cross-estimand pairs it separates (0–4). Target-world cells carry the most load; base-world generated-choice cells separate no pair. (b) Minimal identifying support $O ^ { * } \colon$ all 26 two-cell minima are {one target generated-choice cell, one sham candidate-scoring cell}. Shade = how many of the 26 minima use the cell; no cell is mandatory and 25 of the 36 cells appear in no minimum. The starred pair is one example minimum. (c) A reduced 2-cell support {targetlm1lgen, baselmOlgen} is not identifying: the baselmOlgen cell separates no pair, so 2 of the 6 cross-estimand pairs (any-edit, gen-only-fail) remain inseparable (red); the bottom strip shows per-pair separation