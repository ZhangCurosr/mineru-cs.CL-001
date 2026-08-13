# CAUSAL STRUCTURE IS INDUCIBLE BUT FUNCTION-ALLY DECOUPLED: THE ROUTING/READOUT BOUND-ARY OF A TYPED MECHANISM LIBRARY

Xining Xun

Tsingjiao Information Science (Beijing) Co., Ltd.

## ABSTRACT

When a language model answers an interventional question, the computation it must perform depends on the type of evidence the query requires — different types are different computations. We report a decoupling in how a transformer organizes causal knowledge: slot×type structure induced by type-level supervision organizes routing, yet remains functionally decoupled from answer readout. We establish this with a typed mechanism library — discrete mechanism slots, partitioned by evidence type, addressable and auditable at the state level — used as a measurement instrument on a causal-world benchmark with exact interventional ground truth, under a frozen evaluation protocol, at two scales (22.6M and 125M). Four preregistered findings. (i) Origin. Slot×type organization is induced by type-level supervision: it does not emerge in architecturally identical unsupervised controls, it cannot be bought by content-free gating labels, and where it appears it is statistically attributable to the supervision signal, replicating at 125M under a powered, preregistered attribution protocol (three fresh seeds, two fresh unsupervised controls, 1,280 fresh queries; gate passed on all nine criterion cells). (ii) Boundary. The induced structure is a typed routing index with a sharp routing/readout boundary: structural slot codes scaffold routing but do not drive answer readout $( | \Delta \hat { y } | \leq 3 . 4 \times 1 0 ^ { - 6 }$ with exactly zero collateral, three seeds, stable across a 5.6× scale window) — we therefore make no behavioral-editability claim. (iii) Cost. The structure is free: LM quality matches a parameter-matched monolith within 0.0082 nats (preregistered tolerance, three seeds; behavioral battery 50/50). (iv) Trust. The library state is exactly local under edit and bit-exactly revertible — 250 single-edit and 1,000 stacked reverts per seed, three seeds, zero failures — an auditable state-management substrate for causal knowledge, with properties that hold by construction and are verified under preregistered criteria. We further find that the unsupervised null itself moves with scale — a moving null — which scopes how organization claims may be inherited across scales (§4.1): architecture comparisons that reuse a null calibrated at one scale may be confounded at another. Every claim above is tied to a preregistered, machine-checkable criterion archived before the data it governs; the full audit trail — including one criterion we failed and how the frozen protocol handled it — is released as an appendix.

## 1 INTRODUCTION

A large part of causal-reasoning and interpretability research rests on an implicit assumption we call the circuit-board intuition: if we can expose a discrete internal structure — a slot, a neuron, a circuit — and modify it, the modification should show up in the model's output. Locate-and-edit methods write into localized weights; probing and circuit analyses read structure out and treat it as the computation. The intuition is load-bearing: without it, exposing structure buys understanding but no control. Is it true? We built a controlled platform to test the assumption end to end, and the answer is no — in a precise, measured, and replicated sense.

The platform is a causal-world testbed with exact interventional ground truth, plus a typed mechanism library (MM) on a standard transformer backbone (Vaswani et al., 2017): N discrete slots. statically partitioned over evidence types (identity, relation, sign, ...), with a gating head trained by a small auxiliary loss encouraging type-consistent routing. Real text corpora confound evidence type with topic, lexical cues, and frequency; here every world is a small structural causal model, interventional ground truth is computable exactly, and every query carries a known evidence type — at two scales (22.6M and 125M parameters). One architectural choice is what makes the coming negative result informative rather than trivial: the library is the model's only writable causal state, and the backbone consumes library-mediated representations — the architecture gives the explicit structure every condition to sit on the readout pathway. What training does with that opportunity is the empirical question.

Three questions organize the paper. Where does typed organization come from — does it emerge for free, can it be bought by arbitrary supervision, or is it induced by supervision that is about something? What is the structure for — does it sit on the pathway that produces answers, or is it an epiphenomenal index? And what does it cost and what can be trusted — does the structure degrade the base model, and can its state be edited and audited exactly? Our aim is not to improve causal reasoning per se — every capability result in this paper is a parity result — but to measure, under preregistered criteria, what explicit causal state can and cannot deliver.

Contributions. The findings come first; the instrument that produced them is the third contribution;   
the protocol is how we make all of it believable.

1. Scientific finding: the routing/readout boundary (H-α). Induced slot×type structure scaffolds routing but does not drive answer readout: 150/150 structural edits move the readout ${ \mathsf { b y } } \leq 2 { \times } 1 0 ^ { - 4 }$ with exactly zero collateral $( | \Delta \hat { y } | \leq 3 . 4 \times 1 0 ^ { - 6 }$ on the criterion protocol, three seeds), stable across a 5.6× scale window. The boundary falsifies the circuit-board intuition for explicit-state architectures — and it is not an architectural triviality: training decoupled a structure that was built to participate.

2. Origin, attributed: induced, not emergent, not buyable. Slot×type organization is induced by type-level supervision (permutation-tested MI): content-free gating labels are unlearnable (two protocols, four seeds, 22.6M); unsupervised controls show no organization at 22.6M; and the supervised effect replicates across three seeds $( \mathrm { z } = 4 . 4 7 / 5 . 9 9 / 7 . 0 2$ , Stouffer $z = 1 0 . 1 ;$ excess 0.072– 0.124 nats, moderate effect grade per frozen tiering). At 125M the attributable increment replicates against two fresh unsupervised controls under a preregistered paired-attribution gate: three fresh type seeds clear the amplitude bar $( \mathrm { z } = 1 5 . 1 5 / 1 3 . 2 0 / 1 3 . 7 7 ;$ excess 0.1275/0.1025/0.1202 nats against the frozen ≥ 0.10), and all six paired contrasts clear the frozen $z _ { \Delta } \geq 3$ bar (5.02–7.87; archive A20.21).

3. Methodological caution: the moving null. The unsupervised null is not scale-free: a structuralprior control sits on the null at 22.6M but carries weak, stable organization at 125M (z=2.38; §4.1). Organization claims calibrated at one scale and evaluated at another may inherit a null that no longer applies. We release the paired-permutation attribution protocol built to survive this — frozen intersection gate, frozen edge clause prescribing exactly one remedy (a powered replication frozen before any replication data) — as a reusable template.

4. Instrument: a zero-cost, auditable state-management substrate. The typed library (600 slots, 6 evidence types at 125M) costs nothing measurable in LM quality $( \mathrm { { g a p } \leq 0 . 0 0 8 2 }$ nats at a preregistered tolerance; battery 50/50 both arms); edits have exactly zero off-path collateral (three seeds); and the revert audit passes in full — single-edit and depth-20 stacked revert, backbone zero-touch, zero-contamination contract for failed edits, three seeds, zero failures.

## 2 ARCHITECTURE

## 2.1 MECHANISM LIBRARY

The library is a tensor of N typed slots (N=600 at the 125M configuration, N=200 at 22.6M). Slots are statically partitioned over evidence types (6 types at 125M: identity / child / relation / sign / confidence / block-reserved; 3 types at 22.6M), with a per-type floor $( \beta _ { \mathrm { { f l o o r } } } = 0 . 3 )$ preventing collapse. Each slot carries a mechanism payload (edge list, sign bits, scalar parameters) and audit buffers (usage counters) that are part of the model's st ate\_di ct and therefore of every snapshot.

![](images/6e20b29d5dcd64e05c78c640f8427a4affee9eed6511126190fe4ece7f179ac3.jpg)  
Figure 1: Architecture overview (mm125: 126.21M total; N=600 slots, 6 evidence types). Green path: edit interface with audited bit-exact rollback (A22, §4.5). Dashed orange path: the measured H-α boundary (§4.2).

## 2.2 TYPED ROUTING AND GATING

Evidence tokens are routed to slots by a gating head. Three gate modes define our arm structure: type (supervised by the type-classification auxiliary loss $\lambda _ { g } = 0 . 1 )$ , emergent $( \lambda _ { g } = 0 ;$ , free routing), and blocks $( \lambda _ { g } = 0 ;$ block-structured routing prior without type labels). The two unsupervised arms are the null model for the organization claim: any slot×type alignment they exhibit is organization that supervision cannot be credited for. Gumbel noise (Jang et al., 2017; Maddison et al., 2017) (τ annealed over the first 1,500 steps) shapes early exploration; an entropy floor and load-balancing term $( \lambda _ { l b } = 0 . 0 1 )$ prevent degenerate routing.

## 2.3 EDIT INTERFACE

Edits are applied through an Edi t Sessi on: snapshot library state → apply operator → (optionally) verify → commit or undo. Four structural operators (flip\_sign, add\_edge, remove\_edge, swap\_edge) plus a bounded parameter operator (param\_edit, ≤50 steps, row-masked lr $1 0 ^ { - 3 } )$ cover the library's writable dimensions. Composition is native: compose\_legal validates operator sequences against library invariants before application. All operators are human-readable state transformations — but what they change behaviorally is bounded by H-α (§4.2), and we scope our claims accordingly.

## 2.4 BACKBONE AND TRAINING

A 116.66M-parameter causal-LM backbone (total 126.21M with library and heads at mm125) is trained on online-sampled causal worlds with the standard LM loss plus answer head, gating, and load-balancing terms. All arms share the frozen recipe: 40k steps, lr $6 \times 1 0 ^ { - 4 }$ cosine to $6 \times 1 0 ^ { - 5 }$ with 1.5k warmup, $\beta _ { \mathrm { f l o o r } } = 0 . 3$ , Gumbel 1.0, $\lambda _ { l b } = 0 . 0 1$ ; arms differ only in gate\_mode and $\lambda _ { g } . \mathrm { A }$ parameter-matched monolithic transformer (MONO, 125M) serves as the LM-quality baseline.

## 3 EXPERIMENTAL PROTOCOL

Criteria first. Every headline claim is paired with a preregistered, machine-checkable criterion archived (with md5-chained scripts) before the data it governs. Permutation tests (2,000 permutations, archived seeds; Pesarin & Salmaso, 2010) drive all mutual-information claims; editing claims are pass/fail with zero statistical degrees of freedom. A pipeline-validation gate accompanies every MI-based criterion: unsupervised control arms must sit within the null band $( | z | < 2 )$ for the criterion to be interpretable — a lesson burned into the protocol by an earlier incident in which a leaky pipeline fabricated significance (Appendix B, Case A12).

When a criterion fails. The protocol specifies in advance what happens next: suspension of the affected claim, a frozen diagnostic battery, and — where warranted — a revised estimand whose thresholds are frozen before the data it governs. This path was executed once in this project (the

Table 1: Preregistered criteria and current verdicts (full archive in Appendix A).
<table><tr><td>Criterion</td><td>Claim under test</td><td>Bar (frozen)</td><td>Verdict</td></tr><tr><td>S1 (22.6M)</td><td>typed organization</td><td> $\operatorname { p e r - s e e d } z \geq 3 ;$  mean-excess tiers frozen</td><td>PASS — moderate grade (z=4.47/5.99/7.02; excess</td></tr><tr><td>S1 (125M raw)</td><td>typed organization</td><td>(strong: ≥ 0.10) same, 3 seeds</td><td>0.072–0.124, mean 0.0998) raw PASS (z 5.62–7.75) attribution resolved by A20-R</td></tr><tr><td>A20-R (125M attribution</td><td>attributable increment over</td><td>per-seed  $z \ge 3 \land$  excess  $\geq 0 . 1 0 ;$ </td><td>PASS (z 13.20–15.15; excess 0.1025–0.1275; z∆ 5.02–7.87;</td></tr><tr><td>replication)</td><td>unsupervised</td><td>6/6 paired  $z _ { \Delta } \geq 3$ </td><td>archive A20.21)</td></tr><tr><td>Pipeline gate</td><td>controls MI pipeline</td><td>unsupervised arms</td><td>blocks  $z { = } 2 . 3 8 \to$  attribution</td></tr><tr><td>S2</td><td>integrity pathway boundary</td><td> $| z | < 2$   $| \Delta \hat { y } | \le$ </td><td>protocol (§4.1) PASS  $( \leq 3 . 4 \times 1 0 ^ { - 6 } ; 0 . 0 \mathrm { e x a c t l y } )$ </td></tr><tr><td>S3</td><td>/ locality zero LM cost</td><td> $1 0 ^ { - 3 } /$  \collateral &lt; 0.01  $\mathrm { g a p } \leq$ </td><td>PASS (≤ 0.0082; 50/50)</td></tr><tr><td>A22</td><td>bit-exact rollback</td><td>0.02 ∧ battery  $\geq 9 5 \%$  R1–R5 all pass, ×3 seeds PASS</td><td></td></tr><tr><td>A21</td><td>persistence under continued training</td><td>C1–C3 (frozen)</td><td>[PENDING: unfrozen 2026-08-06 (A20.12 condition met); execution to</td></tr></table>

Table 2: Arms and frozen recipe (125M unless noted).
<table><tr><td>Arm</td><td>gate_mode</td><td> $\lambda _ { g }$ </td><td>role</td><td>seeds</td></tr><tr><td>type</td><td>type</td><td>0.1</td><td>treatment</td><td>s0–s2 (archived) + s3–s5 (A20-R confirmatory)</td></tr><tr><td>emergent</td><td>emergent</td><td>0</td><td>unsupervised control</td><td>s0 (archived) + s1 (A20-R) s0 (archived) + s1 (A20-R control)</td></tr><tr><td>blocks</td><td>blocks</td><td>0</td><td>structural-prior control</td><td>+ s2 (A20-R robustness, non-criterion)</td></tr><tr><td>MONO</td><td></td><td></td><td>LM-quality baseline (125M monolith)</td><td>s0</td></tr></table>

125M pipeline-gate failure behind §4.1's branching); the full decision tree as executed — including the miss we did not re-judge — is shown in Appendix B (Figure 8).

All arms: 40k steps, lr $6 \times 1 0 ^ { - 4 }$ cosine $\to 6 \times 1 0 ^ { - 5 }$ (warmup 1.5k), $\beta _ { \mathrm { f l o o r } } = 0 . 3 .$ , Gumbel 1.0, Twarmup = 1500, $\lambda _ { l b } = 0 . 0 1 ;$ mm125 = 116.66M backbone + library (126.21M total), N=600, 6 types; mm $1 2 5 = 2 2$ .6M, N=200, 3 types.

Arms and seeds. At 125M: type ×6 seeds (s0–s2 archived; s3–s5 A20-R confirmatory), emergent ×2 (s0 archived; s1 A20-R control), blocks ×3 (s0 archived; s1 A20-R control; s2 A20-R robustnessdisclosure member — not a criterion cell), MONO ×1 (LM baseline). At 22.6M: the same arm family at N=200 / 3 types.

Anti-rescue discipline. No re-judging archived values, no threshold edits after data, no seed top-ups, no second replications. Every protocol revision is archived before the data it affects. The single revision of this project (binary pipeline check → paired attribution gate) followed this rule, and the event is reported in full (§4.1, Appendix B).

## 4 RESULTS

4.1 THE MOVING NULL, AND THE ORIGIN OF TYPED ORGANIZATION (CRITERION ${ \textrm { S 1 + } }$ ATTRIBUTION GATE)

22.6M (archived, final). Type-supervised routing produces slot×type organization that replicates across three seeds: per-seed $\mathbf { \bar { z } } = 4 . 4 7 / 5 . 9 9 / 7 . 0 \bar { 2 }$ (each $p \leq 5 \times 1 0 ^ { - 4 }$ ; Stouffer $z = 1 0 . 1$ ; Stouffer et al., 1949), debiased MI excess 0.072–0.124 nats (mean 0.0998), while both unsupervised arms sit on the null (emergent $\mathbf { z } { = } { + } 0 . 2 1$ , blocks $\mathrm { z } { = } { + } 0 . 1 4 ; \mathrm { T } 3$ archive pull). Per the frozen tiering, the verdict is PASS at the moderate grade: the per-seed z gate was met 3/3; the strong-grade mean-excess bar (0.10) was missed by 0.0002 and is recorded as such (§7, Limitations). A control condition with semantically arbitrary labels was not learnable at all (archived control F1, two protocols, four seeds): the gating head exploits genuine type structure, not label noise.

Table 3: Paired attribution contrasts (archived, 125M). $\Delta _ { \mathrm { e x c e s s } } = \mathrm { t y }$ ype-arm MI excess — control-arm MI excess.
<table><tr><td>Contrast</td><td> $\Delta _ { \mathrm { { e x c e s s } } } ( \mathrm { { n a t s } ) }$ </td><td> $z _ { \Delta }$ </td><td>p</td></tr><tr><td>s0 vs blocks</td><td>+0.091</td><td> $+ 3 . 1 5$ </td><td> ${ < } 1 \mathrm { e } { \cdot } 4$ </td></tr><tr><td>s0 vs emergent</td><td>+0.122</td><td>+4.47</td><td>&lt;1e-4</td></tr><tr><td>s1 vs blocks</td><td>+0.114</td><td>+3.89</td><td>&lt;1e-4</td></tr><tr><td>s1 vs emergent</td><td>+0.145</td><td>+5.21</td><td>&lt;1e-4</td></tr><tr><td>s2 vs blocks</td><td>+0.085</td><td>+2.68</td><td>0.0035</td></tr><tr><td>s2 vs emergent</td><td>+0.116</td><td>+3.92</td><td> ${ < } 1 \mathrm { e } { \cdot } 4$ </td></tr></table>

125M: the null moved. All three type seeds pass the raw amplitude gate: $\mathrm { z } = 6 . 0 9 / 7 . 7 5 / 5 . 6 2$ excess = +0.141 / +0.164 / +0.134 nats (seed-aligned; 320 queries, 2,000 permutations). The pipelinevalidation gate, however, did not pass cleanly: the blocks arm measured z=+2.38 (excess +0.050, p=0.012), above the $| z | < 2$ null band, while emergent sat at $\mathbf { z } { = } { + } 1 . 0 9$ . Per the preregistered decision tree, S1 at 125M was suspended pending diagnosis. The diagnostic finding is itself a result of this paper: the unsupervised null is not scale-free. At 22.6M both unsupervised arms sit exactly on the null (emergent $\mathbf { z } = + 0 . 2 1$ , blocks $\scriptstyle \mathbf { z } = + 0 . 1 4 )$ ; at 125M the blocks arm carries a small, stable slot×type alignment (+0.050 nats; $_ { z = 2 . 3 8 }$ , cross-permutation-seed sd 0.031). A fresh blocks seed settles the level at which the effect lives (D-d, blocks s1: $\scriptstyle \mathbf { Z } = + 3 . 8 9$ , excess +0.065 nats, criterion protocol): the elevation replicates across seeds of the same arm — it is arm-level, i.e., the blocks routing prior itself carries weak slot×type organization at 125M (s0: +0.050; s1: +0.065), not a seed-level fluctuation. Because scale and type-count changed together (N 200→600, 3→6 types), the baseline shift's cause is confounded in our data and we report it as such — we do not decompose what we cannot decompose. Either way, the methodological lesson stands: an organization claim at one scale cannot be silently inherited by another scale, because the null itself can move. We also record, as a post-hoc observation (archived as such; not a preregistered hypothesis), that all three 125M excess values exceed the 22.6M range — an apparent ≈1.5× amplification of effect size with scale whose cause we do not decompose.

Attribution: revising the estimand, not the bar. Three preregistered checks established the diagnosis: (i) the pipeline is calibrated — a random-init model gives $_ { z = - 0 . 2 2 7 }$ and label-shuffled audits give $z \in [ - 1 . 7 2 , + 1 . 3 6 ]$ , all within null; (ii) the blocks elevation is a stable measurement, not a permutation artifact — across the archived permutation seed plus four fresh ones (five values per arm), blocks $z \in [ 2 . 3 4 , 2 . 4 2 ]$ (sd 0.031); (iii) a paired attribution test shows the type-supervised increment over each control is positive on all six seed×control contrasts $( \Delta _ { \mathrm { e x c e s s } } + 0 . 0 8 5 . . . + 0 . 1 4 5$ nats). The binary pipeline check (unsupervised $| z | < 2 )$ assumes the unsupervised null is zero. When that assumption failed, we did not lower any bar; we revised the estimand. The paired-permutation attribution test applies the same label permutation to both arms synchronously and normalizes the arm difference: $z _ { \Delta } = ( \Delta - \mathrm { n u l l \mathrm { \_ m e a n } } )$ /null\_sd over 2,000 paired permutations. On archived data (320 queries), five of six contrasts clear $z _ { \Delta } \geq 3$ , and the weakest (s2 vs blocks) reaches 2.68 (p=0.0035). Because the protocol froze an intersection rule, the archived verdict is “gate not met"; because the effect sizes are uniformly positive and the miss is marginal, the protocol's own edge clause opened exactly one remedy — a powered replication on fresh seeds with fresh worlds (A20-R: n\_worlds=160 → 1,280 queries, gate unchanged, protocol frozen before any replication data). We release the test, its gates, and its edge clause as a reusable template for organization claims whose unsupervised null may be nonzero.

Replication (A20-R, verdict 2026-08-06; archive A20.21). The powered replication (three fresh type seeds, two fresh unsupervised controls, 1,280 fresh queries, fresh world set; gate: per-seed amplitude $z \geq 3 \wedge$ excess $\geq 0 . 1 0 ,$ and paired $z _ { \Delta } \ge 3$ against each control) executed exactly as frozen, and the gate passed on all nine criterion cells (Table 4) — this section's branch was selected mechanically, not argued.

![](images/84e5ba88b62a0cc83721491fe996f2d8b5bd598484e5136f229fcf8b49225c34.jpg)

(b) Archived 125M — significance  
![](images/a942ab775617e588827ca2a00882f39a862153768bc3565f95e3e73a9d5ac4e5.jpg)

(c) A20-R replication — MI excess  
![](images/5bee72dc1fecf449e3882ecd8d26cc35a5a4f32ec083803122353d2072492041.jpg)

(d) A20-R replication — significance  
![](images/ef0cf4e46dff103bf3246b138e8b50cf04304c047db3663ba838128be1420e69.jpg)  
Figure 2: Organization at 125M, archived (a,b) and replication (c,d). Type seeds clear both bars in both collections; the blocks arm's small but nonzero elevation (archived z=2.38) is what triggered the attribution protocol below, and the replication controls sit higher still — the moving null in raw form. The hatched blocks $\mathbf { s } 2$ arm is the robustness-disclosure member and does not enter the gate.

Table 4: A20-R confirmatory replication (125M; n\_worlds=160 → 1,280 fresh queries; 2,000 paired permutations; fresh world set disjoint from the archived criterion set). All nine criterion cells pass the frozen intersection gate.
<table><tr><td>Criterion cell</td><td> ${ \bf s } 3$ </td><td> $\mathrm { s 4 }$ </td><td> $\mathrm { s 5 }$ </td><td>Frozen bar</td></tr><tr><td>Amplitude z</td><td>15.15</td><td>13.20</td><td>13.77</td><td> $\geq 3$ </td></tr><tr><td>MI excess (nats)</td><td>+0.1275</td><td>+0.1025</td><td>+0.1202</td><td> $\geq 0 . 1 0$ </td></tr><tr><td> $z _ { \Delta }$  vs emergent s1  $\left( \Delta _ { \mathrm { e x c e s s } } \right)$ </td><td></td><td>+7.87 (+0.0806) +5.92 (+0.0554)</td><td>+7.29 (+0.0734)</td><td>≥3</td></tr><tr><td> $z _ { \Delta }$  vs blocks s1  $\left( \Delta _ { \mathrm { e x c e s s } } \right)$ </td><td></td><td>+7.26 (+0.0704) +5.02 (+0.0452)</td><td>+6.64 (+0.0631)</td><td> $\geq 3$ </td></tr></table>

Replication control arms measured in the same collection: emergent s1 $z { = } 7 . 5 8 \mathrm { ~ / ~ }$ excess +0.0471; blocks s1 $_ { z = 1 0 . 8 8 \mathrm { ~ / ~ } }$ excess +0.0573. Robustness disclosure (non-criterion): against a second, stronger blocks control $( \mathrm { s } 2 ; \mathrm { z } { = } 1 2 . 0 5 $ , excess +0.0810 — the unsupervised null strengthens further at this third arm), $z _ { \Delta } = + 4 . 5 3 / + 2 . 2 1 ( \mathrm { p } { = } 0 . 0 1 4 5 ) / + 3 . 8 6 ;$ the s4 cell lands below 3 and is reported as such. Per the frozen anti-rescue clause this robustness member does not enter, substitute for, or alter the gate verdict. We therefore state: at both scales, type supervision produces slot×type organization that is statistically attributable to the supervision signal, above what unsupervised routing priors provide.

## 4.2 THE ROUTING/READOUT BOUNDARY (H-α): WHAT THE LIBRARY IS NOT

Three findings scope every editability word in this paper: (i) applying all 150/150 structural flips changes the answer readout $\mathsf { b y } \leq 2 \times \mathsf { 1 0 ^ { - 4 } }$ with exactly zero collateral — the structural codes are not on the readout pathway; (ii) ROME-style row fine-tuning is plastic at the edited point (0.855) but does not generalize to held-out variants (0.124); (iii) the tgC generalization probe fails its bar at both scales (0.124 at 22.6M; 0.217, three-seed mean, at 125M; frozen bar 0.30). We therefore describe the library as a typed routing index with exact state-level edit semantics, and we explicitly do not claim behavioral editability of the world model.

![](images/7581fd16dad20eaf05b35c9bff6e73871a2e6c773be3d39c097b6846db41fab7.jpg)

![](images/59539fefd0ef0bb4105b2b4c857b4ef665d60229eb10b2ccb50707a442c62ed0.jpg)

Figure 3: The pipeline is calibrated and the offending measurement is stable — the basis on which the protocol concluded “the null moved" rather than “the pipeline broke". (a) D-a calibration: randominit model and five label-shuffled audits all inside the null band. (b) D-b stability: z across five permutation seeds per arm; the blocks elevation is tight (sd 0.03), i.e., a real measurement, not a permutation artifact.  
![](images/106d339130ce1fa469fc4353d4cc936d6eba8684c83515db8fdfd02dbdbe5770.jpg)

![](images/197e11342c94135840f11132a21d5ae45253e0d99b4fd320dfe99334c689f420.jpg)  
Figure 4: Attribution contrasts, archived (a) and replication (b), with the frozen $z _ { \Delta } = 3$ bar. (a) Archived collection (320 queries): five of six contrasts pass, one (weakest seed × strongest baseline) lands at 2.68 and the intersection gate is not met. (b) A20-R replication (1,280 queries): all six criterion contrasts pass; gray diamonds are the robustness-disclosure contrasts against blocks s2 (non-criterion). The intersection rule makes no exception for the 2.68 — and neither do we; the remedy is the powered replication, not a lower bar.

![](images/ffc846dd297200cb3dc352181ce95e7d38b5938c68a6248a519d94a2dcd20d20.jpg)

![](images/1d2473b4bdb2881dc98c8947e7f8330b173c272c2e4ddaa874cd3264728866ec.jpg)  
Figure 5: Negative results, kept prominent: the library is editable as state, not established as behaviorally editable. (a) 150/150 structural edits land on the library while the readout moves $\leq 2 \times 1 0 ^ { - \tilde { 4 } }$ and collateral is exactly zero; (b) parameter-route editing is plastic at the train point (0.855) but does not generalize (0.124), and the tgC probe misses its 0.30 bar (0.217).

This decoupling is not an architectural triviality. The library is the model's only writable causal state and the backbone consumes library-mediated representations: the architecture gave the explicit structure every condition to participate in readout, and training decoupled it anyway. We regard this boundary — measured, replicated, and stable across a 5.6× scale window — as a contribution: it maps precisely where an explicit-state architecture's guarantees end, and it falsifies the “circuit-board" intuition that exposed structure is necessarily on the computation's main path.

## 4.3 LOCALITY AT THE STATE LEVEL (CRITERION S2)

Paired flip edits (150 queries/seed, three seeds) at 125M: answer-readout change $| \Delta \hat { y } | \leq 3 . 4 \times 1 0 ^ { - 6 }$ (criterion $\leq 1 0 ^ { - 3 }$ , met with ≥ 2.5 orders of magnitude margin) and off-path collateral exactly 0.0 on all three seeds. Edits to the library are exactly local at the state level, and the F6 pathway boundary — structural slot codes do not feed the answer readout — replicates at 5.6× scale without exception. The bounded parameter operator remains plastic at the training point (gain 0.884/0.935/0.917) but does not teach the readout to consume structural codes: tgC mean 0.217 < 0.30 (capacity/generalization branch, preregistered tree). This is the project's central negative result and it is scale-invariant; §4.2 states what it scopes.

## 4.4 ZERO LM COST (CRITERION S3)

LM-quality gap vs the parameter-matched monolith: 0.0058 / 0.0082 / 0.0071 nats across the three type seeds (criterion $\leq 0 . 0 2 )$ , and the 50-item Simpson-paradox behavioral battery passes 50/50 on both arms (criterion ≥ 95%). The typed library costs nothing measurable in base capability.

## 4.5 BIT-EXACT REVERSIBILITY (CRITERION A22)

The library state is bit-exactly revertible as state: the preregistered R1-R5 audit passes on all three seeds with zero failures — 250/250 single-edit restores; 1,000/1,000 depth-20 stacked reverts; 1,250/1,250 backbone zero-touch comparisons; zero-contamination on every failed edit (147–150 sampled instances per seed; the count varies because an edge constructed to be absent exists in a few worlds, making the edit legal — full counts in Table C4, audit figure in Appendix C). Two properties matter more than the log: rollback completeness holds by construction (the audit buffers live in the model's st ate\_di ct, so no snapshot can forget revertible state), and the toolchain itself passed a preregistered engineering-validation pass (A22.1; two defects caught and fixed) before any criterion data existed. Per our frozen wording this is a statement about toolchain state, not about behavioral edit semantics (§4.2); the operator set is specified in §2.3 and the full audit table in Appendix C.

![](images/5e5413e15456411dbae38d2bfd6314e410b63e1a78a8b5619bf20aa92a53f207.jpg)

![](images/2041b45bd6b56f74f56228175f20108b43675a0a17d8c442ab3b769673efa192.jpg)

Figure 6: Locality and cost. (a) Readout change is ≥ 2.5 orders of magnitude below the bar; collateral is not “small" but exactly zero. (b) LM gap within preregistered tolerance on all seeds.  
![](images/b69fb2091613a1e6667cedfe65c19de580acb4449cf612da7a5d5fd0a4906c1b.jpg)  
Figure 7: Descriptive pooled view across collections (no confirmatory pooling). Type-arm excess replicates at a level far above every unsupervised control in both collections; the replication controls themselves sit above the archived ones — the moving null of §4.1, visible in raw form.

## 5 ANALYSIS

## 5.1 STABILITY AND SEED VARIANCE

Across permutation seeds, every arm's z is stable (worst sd 0.20, type s1). Raw type-arm excess spans 0.134–0.164 (between-seed sd ≈0.015): real seed-level variance exists, and three seeds is a small sample — which is why the replication doubles the confirmatory base.

## 5.2 A DESCRIPTIVE POOLED VIEW (NO INFERENTIAL WEIGHT)

Figure 7 pools, descriptively, the archived and replication collections: the six type seeds' excess values overlap across collections (archived 0.134–0.164; replication 0.1025–0.1275), while every unsupervised control sits far below (0.019–0.081). Per the preregistered rule there is no confirmatory pooling — the archived and replication verdicts stand on their own frozen criteria, and this plot carries no inferential weight.

## 6 RELATED WORK

Model editing. Locate-and-edit methods modify knowledge inside distributed weights: ROME-style rank-one updates identify a decisive MLP layer and write a new key-value association (Meng et al., 2022), and MEMIT-style successors extend this to thousands of simultaneous edits (Meng et al., 2023); MEND learns a hypernetwork that maps gradients to weight edits (Mitchell et al., 2022). A large follow-up literature measures locality, generalization, and persistence trade-offs on behavioral benchmarks (CounterFact, zsRE, and derivatives: Meng et al., 2022; Levy et al., 2017). Notably, localization itself has been shown not to be causal for editing outcomes (Hase et al., 2024) — a finding consonant with our routing/readout boundary, which offers one testable hypothesis for why such mismatches arise (§7). Our work differs in where the editable object lives: not an approximate subspace inside weights, but an explicit, enumerable state with construction-guaranteed locality (off-path collateral exactly zero) and bit-exact rollback. We deliberately do not frame this as a behavioral shoot-out — those methods report behavioral generalization on their benchmarks, while our own H-α boundary shows our structural edits are not on the readout pathway at all. The correct contrast is the explicitness and auditability of the editable state, and the strength of the guarantees one can state before, rather than estimate after, the edit.

Mechanistic interpretability and probing. A parallel tradition discovers internal structure post-hoc: linear and nonlinear probes (Adi et al., 2017; Hewitt & Liang, 2019), sparse dictionary decompositions of superposed features (Elhage et al., 2022; Bricken et al., 2023), and circuit-level analyses of routing and attention (Olah et al., 2020; Wang et al., 2023). These methods read structure out of models that were never asked to have any; our approach is complementary — structure is built in, and the claim that it materialized is itself subjected to preregistered, permutation-tested measurement against unsupervised nulls, with pipeline-validation gates guarding the measurement instrument. Our moving-null finding (§4.1) is directly relevant to this literature: any organization or specialization claim that calibrates its null at one scale and evaluates at another risks inheriting a null that no longer applies.

World models, memory, and structured state. External-memory architectures (Graves et al., 2014; 2016), object-centric slot representations (Locatello et al., 2020), and algorithmic-state lines of work all endow models with structured, addressable state; causal-benchmark work supplies worlds with known ground-truth structure (Jin et al., 2023; 2024; Kicıman et al., 2023). Our synthetic causal worlds follow the second tradition as a measurement instrument: because ground-truth causal structure is known by construction, criteria of the form “exactly zero collateral" or “bit-exact restore" become checkable in a way that naturalistic benchmarks cannot support. Typed evidence routing as an inductive bias — and the evidence-type supervision that organizes it — connects both to recent work on evidence-type competition (Xun, 2026) and to routing analyses in the mixture-of-experts literature (Shazeer et al., 2017; Fedus et al., 2022; Zhou et al., 2022).

Preregistration and audit culture. Our audit trail is methodologically kin to registered reports and preregistration practice in the experimental sciences (Nosek et al., 2018; Chambers, 2013), adapted to ML measurement (Forde & Paganini, 2019): criteria archived before data, frozen decision trees, hash-chained scripts, and full publication of failed gates. Figure 8 shows the tree as executed; Appendix B reports the two measurement case studies that shaped it.

## 7 DISCUSSION

Scope, set by evidence. Our contribution is not a SOTA improvement; it is the falsification of a widely held implicit assumption, plus the measurement machinery that made the falsification unavoidable. What the evidence supports: an explicit typed library can be trained at zero LM cost; it organizes under type supervision (at both scales, attributable to the supervision signal per §4.1); edits to it are exactly local and bit-exactly reversible at the state level; and its behavioral boundary is sharp, measured, and stable across a 5.6× scale window. What it does not support: behavioral editability (H-α), and any claim that the unsupervised null is scale-free (§4.1).

“If the library does not change behavior, what is it for?" The result redirects rather than deflates model editing: the resistance to behavior change is not in the slot contents — those are exactly editable — but at the routing/readout connection. Editing research aimed at explicit state should target bridging H-α (teaching readouts to consume structural codes) rather than refining the state representation further; editing research aimed at distributed weights should treat our boundary as a baseline expectation, not an anomaly, given how readily the optimizer produced it here.

Why does such decoupling exist? A speculation on residual-stream architecture. We close with a hypothesis, explicitly labeled as such — our data establish the boundary, not its cause. Discrete slot codes are high-precision, low-bandwidth objects, well suited as addresses; coupling them directly to the readout would tie a sparsely-updated, discrete pathway to every output position, plausibly injecting gradient noise into the dense computation. A transformer trained by gradient descent may therefore find it favorable to keep discrete structure on the routing side while distributed weights carry the answer — a separation of addressing from computation that acts as an implicit regularizer, discovered by the optimizer rather than imposed by us. This account is consistent with independent reports that localization need not determine editing outcomes (Hase et al., 2024), and it is falsifiable: interventions that bridge the boundary (e.g., training the readout to consume slot codes) should, under this account, trade off against training stability. Whether the same boundary arises for emergent, non-engineered structure is the open question our instrument cannot answer — and the one we consider most important.

Limitations. Synthetic causal worlds; two scales within one architecture family; three-seed arm budgets (six after replication); the 22.6M organization effect is moderate-grade (three-seed mean excess 0.0998 against the frozen strong-tier bar of 0.10 — a 0.0002 miss recorded per the frozen tiering, not relabeled); the baseline-emergence confound (§4.1) is reported, not resolved; persistence under continued training (A21) remains untested — the criterion is frozen and was unfrozen for execution on 2026-08-06 (archive A20.12/A20.21), but no A21 data exist yet.

Why we report the pipeline event. The binary-check failure at 125M, the suspended criterion, and the frozen-threshold handling are not embarrassments to minimize — they are the audit system working in public. We submit that organization claims should be made this way, and we release the machinery (scripts, hashes, decision trees) so others can hold us — and themselves — to it.

## AI USE STATEMENT

A large-language-model assistant was used to help draft and copy-edit the manuscript, to adversarially proof-check its claims against the underlying experimental data and audit archive, and to develop and debug the training and evaluation code. All scientific claims, experimental design, data, analyses, and conclusions were made and verified by the author, who takes full responsibility for the content of this paper.

## REPRODUCIBILITY STATEMENT

Every quantitative claim in this paper is tied to a preregistered, machine-checkable criterion archived — with md5-chained scripts — before the data it governs; the full audit trail (entries A12–A22.3, including the failed gate and the execution incidents) is released as supplementary material. The causal-world generator, the frozen training recipe (Table 2), all archived seeds (world, permutation, and training), and the evaluation code are released to enable exact reproduction: every world is reconstructible from its seed, and every reported number is traceable to a named archive entry. The A20-R replication protocol, including its power analysis and anti-rescue clauses, is included in full in the archive.

## ETHICS STATEMENT

This work uses only synthetic causal-world data; no human subjects, personal data, or scraped corpora are involved. The research goal — auditable, exactly revertible state management for causal knowledge — is oriented toward safer and more transparent model maintenance. We foresee no application-specific negative impacts beyond those of general-purpose language-model research; the benchmark's synthetic nature means conclusions about naturalistic settings require the further work we flag in Limitations.

## REFERENCES

Yossi Adi, Einat Kermany, Yonatan Belinkov, Ofer Lavi, and Yoav Goldberg. Fine-grained analysis of sentence embeddings using auxiliary prediction tasks. In The Fifth International Conference on Learning Representations (ICLR 2017), 2017.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E. Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning, 2023. Transformer Circuits Thread.

Christopher D. Chambers. Registered reports: A new publishing initiative at Cortex. Cortex, 49(3): 609–610, 2013.

Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. Transformer Circuits Thread.

William Fedus, Barret Zoph, and Noam Shazeer. Switch Transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal of Machine Learning Research, 23(120):1–39, 2022.

Jessica Zosa Forde and Michela Paganini. The scientific method in the science of machine learning, 2019. arXiv:1904.10922.

Alex Graves, Greg Wayne, and Ivo Danihelka. Neural Turing machines, 2014. arXiv:1410.5401.

Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwińska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou, Adrià Puigdomènech Badia, Karl Moritz Hermann, Yori Zwols, Georg Ostrovski, Adam Cain, Helen King, Christopher Summerfield, Phil Blunsom, Koray Kavukcuoglu, and Demis Hassabis. Hybrid computing using a neural network with dynamic external memory. Nature, 538 (7626):471–476, 2016.

Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun. Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in large language models. Transactions on Machine Learning Research (TMLR), 2024.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP 2019), pp. 2733–2743, 2019.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-Softmax. In The Fifth International Conference on Learning Representations (ICLR 2017), 2017.

Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojas Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, and Bernhard Schölkopf. CLadder: Assessing causal reasoning in language models. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), Datasets and Benchmarks Track, 2023.

Zhijing Jin, Jiarui Liu, Zhiheng Lyu, Spencer Poff, Mrinmaya Sachan, Rada Mihalcea, Mona Diab, and Bernhard Schölkopf. Can large language models infer causation from correlation? In The Twelfth International Conference on Learning Representations (ICLR 2024), 2024.

Emre Kicıman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. arXiv preprint arXiv:2305.00050, 2023.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pp. 333–342, 2017.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020), pp. 11525–11538, 2020.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete distribution: A continuous relaxation of discrete random variables. In The Fifth International Conference on Learning Representations (ICLR 2017), 2017.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 17359–17372, 2022.

Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau. Mass-editing memory in a Transformer. In The Eleventh International Conference on Learning Representations (ICLR 2023), 2023.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. Fast model editing at scale. In The Tenth International Conference on Learning Representations (ICLR 2022), 2022.

Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, and David T. Mellor. The preregistration revolution. Proceedings of the National Academy of Sciences, 115(11):2600–2606, 2018.

Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3), 2020.

Fortunato Pesarin and Luigi Salmaso. Permutation Tests for Complex Data: Theory, Applications and Software. Wiley, 2010.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In The Fifth International Conference on Learning Representations (ICLR 2017), 2017.

Samuel A. Stouffer, Edward A. Suchman, Leland C. DeVinney, Shirley A. Star, and Robin M. Williams, Jr. The American Soldier: Adjustment during Army Life, volume 1. Princeton University Press, 1949.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems 30 (NeurIPS 2017), 2017.

Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: A circuit for indirect object identification in GPT-2 small. In The Eleventh International Conference on Learning Representations (ICLR 2023), 2023.

Xining Xun. Evidence-type competition: How typed supervision shapes causal mechanism allocation in language models. arXiv preprint arXiv:2607.29484, 2026.

Yanqi Zhou, Tao Lei, Hanxiao Liu, Nan Du, Yanping Huang, Vincent Zhao, Andrew M. Dai, Zhifeng Chen, Quoc V. Le, and James Laudon. Mixture-of-experts with expert choice routing. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), pp. 7103–7114, 2022.

## A PREREGISTRATION ARCHIVE (SUMMARY)

A-series protocol chain (each entry archived before the data it governs; md5-chained scripts; full text in repository):

• A12: pipeline-integrity incident — a leaky aggregation fabricated significance; remediation: raw-pair permutation MI (mi-perm\_test) + binary pipeline check. Lesson embedded in all later criteria.

• A18/A19: editing probes; H-α boundary established (strong form); contributions re-scoped (“editable" removed; library = typed routing index); tgC branch decision.

• A20 (S1/S2/S3 at 125M) → S2, S3 pass; pipeline gate fails (blocks $\mathrm { z } { = } 2 . 3 8 )  \mathrm { A } 2 0 . 8 { - } \mathrm { A } 2 0 . 1 4$ event chain: diagnostics D-a-D-d; paired-attribution revision (frozen before data); 100-round adversarial review; A20-R powered replication protocol (frozen, locked); execution incidents (interrupted runs, env failure, deadlock) logged with zero data impact.

• A20-R (powered attribution replication) → executed as frozen (TO–T6 unified chain; QC gates passed: 6/6 checkpoint-args assertions, bit-exact redundant collection); gate passed on all nine criterion cells (2026-08-06, A20.21) → §4.1 BRANCH-A selected mechanically; robustness member (blocks s2) disclosed per the anti-rescue clause.

• A21 (persistence): drafted, reviewed 20 rounds, frozen; unfrozen 2026-08-06 upon the A20-R gate pass (A20.12 condition met); execution to be scheduled — no A21 data exist yet.

• A22 (revert toolchain): R1–R5 criteria + engineering validation (2 defects caught and fixed before criterion data); unfrozen early by preregistered orthogonality ruling; passed (§4.5).

## B MEASUREMENT CASE STUDIES

Case A12 (fabricated significance). A leaky aggregation step once fabricated significance on null data; the remediation (raw-pair permutation MI plus a binary pipeline check) now guards every MI claim in this paper.

Case A20 (the moving null). The 125M pipeline-gate failure; D-a calibration; D-b stability; pairedcontrast revision; frozen-threshold discipline; powered replication design (null. $\mathrm { s d } \approx 0 . 0 2 9 \dot { @ } 3 \bar { 2 } 0 \mathrm { q } $ ≈ 0.015@1280q; gate $\iff \Delta _ { \mathrm { e x c e s s } } \geq 0 . 0 4 7 $ observed worst case 0.085, margin $\geq 1 . 8 \times )$ . Figure 8 shows the tree as executed; Table 3 is the full D-c table; both narrative branches were pre-frozen.

## C FULL DATA TABLES

Table C1: S1 raw measurements at 125M (archived, frozen; 320 queries, 2,000 permutations per arm).
<table><tr><td>Arm</td><td>Z</td><td>MI excess (nats)</td><td>p</td></tr><tr><td>type s0</td><td>+6.09</td><td>+0.141</td><td> $< 1 / 2 0 0 0$ </td></tr><tr><td>type s1</td><td>+7.75</td><td>+0.164</td><td> $< 1 \dot { / } 2 0 0 0$ </td></tr><tr><td>type s2</td><td>+5.62</td><td>+0.134</td><td> $< 1 / 2 0 0 0$ </td></tr><tr><td>emergent s0</td><td>+1.09</td><td>+0.019</td><td>0.1465</td></tr><tr><td>blocks s0</td><td>+2.38</td><td>+0.050</td><td>0.012</td></tr></table>

Table C2: D-a pipeline calibration (two independent null families, all inside $| z | < 2 )$
<table><tr><td>Null</td><td>Z</td><td>excess</td><td>p</td></tr><tr><td>random-init checkpoint</td><td>-0.227</td><td>-0.0015</td><td>0.5345</td></tr><tr><td>blocks s0 label-shuffle K=1</td><td>+1.355</td><td>+0.0284</td><td>0.0990</td></tr><tr><td>blocks s0 label-shuffle K=2</td><td>-0.151</td><td>-0.0032</td><td>0.5525</td></tr><tr><td>blocks s0 label-shuffle K=3</td><td>-1.720</td><td>-0.0365</td><td>0.9655</td></tr><tr><td>blocks s0 label-shuffle K=4</td><td>+0.205</td><td>+0.0044</td><td>0.4030</td></tr><tr><td>blocks s0 label-shuffle K=5</td><td>-0.924</td><td>-0.0196</td><td>0.8225</td></tr></table>

![](images/aeed5af4b6363da89e8e7d3c58f3682c6edbec63a9ab8282571b839673b135f3.jpg)  
Figure 8: The preregistered decision tree, as executed (Case A20). Every node is an archived event with archived values; every threshold shown was frozen before the data it governs.

Table C3: D-b permutation-seed stability (archived seed + four fresh seeds per arm).
<table><tr><td>Arm</td><td>archived z</td><td>four fresh-seed z</td><td>sd</td><td>range</td></tr><tr><td>type s0</td><td>6.09</td><td>+6.06 / +6.13 / +6.00 / +6.03</td><td>0.048</td><td>[6.00, 6.13]</td></tr><tr><td>type s1</td><td>7.75</td><td>+7.25 /+7.79 /+7.52 /+7.65</td><td>0.200</td><td>[7.25, 7.79]</td></tr><tr><td>type s2</td><td>5.62</td><td>+5.56 / +5.30 / +5.55 / +5.53</td><td>0.105</td><td>[5.30, 5.62]</td></tr><tr><td>emergent s0</td><td>1.09</td><td>+1.13/+1.13 /+1.05 /+1.11</td><td>0.032</td><td>[1.05, 1.13]</td></tr><tr><td>blocks s0</td><td>2.38</td><td>+2.41 /+2.42 /+2.34 /+2.39</td><td>0.031</td><td>[2.34, 2.42]</td></tr></table>

Table C4: A22 revert-toolchain audit counts (three seeds; all verdicts PASS, zero failures).
<table><tr><td>Check</td><td>s0</td><td>s1</td><td> $\mathbf { s } 2$ </td></tr><tr><td>R1 single-edit bit-exact (50 worlds  $\times 5$  operators)</td><td>250/250</td><td>250/250</td><td>250/250</td></tr><tr><td>R2 depth-20 stacked rollback</td><td>1000/1000</td><td>1000/1000</td><td>1000/1000</td></tr><tr><td>R3 backbone zero-touch comparisons</td><td>1250/1250</td><td>1250/1250</td><td>1250/1250</td></tr><tr><td>R4 function-level deviation  $\bar { \leq } 1 0 ^ { - 6 }$ </td><td>250/250</td><td>250/250</td><td>250/250</td></tr><tr><td>R5a empty-stack no-op contract</td><td>50/50</td><td>50/50</td><td>50/50</td></tr><tr><td>R5b failed-edit zero-contamination</td><td>147/147</td><td>150/150</td><td>150/150</td></tr></table>

R5b's s0 count of 147 (vs 150) is constructive sampling, not a criterion gap: in three worlds the edge constructed to be absent actually exists, making the edit legal and hence not a failure-semantics instance. S2/S3 per-seed values appear in §4.3–§4.4; the A20-R replication table is Table 4 (§4.1).

![](images/0202ff810abf0cb3e0d3db6b853fdd38e689f7179d1beeed4ff3999adb33a6bb.jpg)  
Figure 9: Revert-toolchain audit results. Every preregistered check passes on every seed.

## D CLAIMS MATRIX

Table 5: Claims matrix (every row traceable to a named archive entry).
<table><tr><td>Claim</td><td>Criterion</td><td>Status</td></tr><tr><td>Typed organization, 22.6M</td><td>S1 (per-seed z ≥ 3; frozen mean-excess tiers)</td><td>Supported — moderate grade (z=4.47/5.99/7.02; excess 0.072–0.124, mean 0.0998)</td></tr><tr><td>Typed organization, 125M, attributed</td><td>S1 + paired-attribution gate</td><td>Supported (A20-R gate passed on all nine criterion cells, 2026-08-06; archive A20.21)</td></tr><tr><td>Exact edit locality</td><td>S2 (collateral &lt; 0.01)</td><td>Supported (0.0 exactly, ×3)</td></tr><tr><td>F6 pathway boundary scale-invariant</td><td>S2  $( | \Delta \hat { y } | \leq 1 0 ^ { - 3 } )$ </td><td>Supported  $( \leq 3 . 4 \times 1 0 ^ { - 6 } , \times 3 )$ </td></tr><tr><td>Zero LM cost</td><td>S3 (gap ≤  $0 . 0 2 \land \mathrm { b a t t e r y } \geq 9 5 \% )$ </td><td>Supported  $\left( \leq 0 . 0 0 8 2 ; 5 0 / 5 0 \right)$ </td></tr><tr><td>Toolchain bit-exact rollback</td><td>A22 R1–R5</td><td>Supported (all pass, ×3)</td></tr><tr><td>Unsupervised null moves pipeline gate + D-b/D-d with scale</td><td>diagnostics</td><td>Observed (blocks +0.050 nats at 125M vs ≈ 0 at 22.6M; cause confounded,</td></tr><tr><td>Effect-size growth with scale (≈ 1.5×)</td><td>— (not preregistered)</td><td>reported as such) Post-hoc observation, labeled as such</td></tr><tr><td>Behavioral editability</td><td></td><td></td></tr><tr><td>Persistence under continued training</td><td>A21 C1–C3</td><td>Not claimed (H-α) [PENDING: criterion frozen; unfrozen</td></tr></table>