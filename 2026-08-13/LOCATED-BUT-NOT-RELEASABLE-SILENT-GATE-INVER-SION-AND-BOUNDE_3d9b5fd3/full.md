# LOCATED BUT NOT RELEASABLE: SILENT GATE INVER-SION AND BOUNDED LINEAR RELEASE

Xining Xun

Tsingjiao Information Science (Beijing) Co., Ltd.

## ABSTRACT

A growing body of work reports that language models represent task-relevant latent structure that they fail to use. Whether such structure, once located, can be converted into behavior is a separate question that is rarely tested end to end. We submit the complete pipeline—detect, localize, and release—to a fully preregistered stress test on a 25.7M transformer trained on causal-evidence discrimination, where a known suppression phenomenon (latent causal structure present but behaviorally unused) has previously been documented. Every threshold, every claim template, and every branch of the decision tree was hashed and archived before any corresponding data existed. Three findings emerge. (i) Localization succeeds: interventions at observation-evidence channels of mid layers restore target behavior on otherwise-suppressed worlds (paired release advantage 0.563 and 0.854, 97.5% CIs excluding zero; best-site release rate 0.889). (ii) Gating fails out of distribution: a detector calibrated to trigger on zero out-of-distribution calibration worlds triggers on 6.9–7.3% of held-out in-distribution generations and on zero of the 2,400 heldout generations that actually need it—a complete inversion that silently reduces the gated pipeline to its base model. (iii) Linear release is capped: removing the gate entirely and injecting a per-instance linear direction unconditionally yields a monotone dose–response that plateaus far below the preregistered release margin (intercept 0.382 → 0.311 → 0.264 vs. threshold ≤ 0.08), and per-instance adaptivity adds less than ±0.03 over the fixed direction. The failure is therefore doubly located: the detector is OOD-inverted, and the entire family of linear release directions at this site and resolution is bounded away from sufficiency. The two failures are dissociable, and neither overturns the localization result—representation-level localization and behavior-level release come apart. All artifacts, hashes, and claim templates are released for audit.

Keywords: mechanistic interpretability, activation steering, causal reasoning, linear probes, preregistration, negative results, representation–behavior gap

## 1 INTRODUCTION

Interpretability research has become very good at finding structure: probes recover latent variables from activations (Belinkov, 2022; Hewitt & Liang, 2019; Burns et al., 2023), causal-tracing methods localize the components that mediate a behavior (Vig et al., 2020; Meng et al., 2022), and steering vectors shift model outputs along semantically meaningful directions (Turner et al., 2023; Zou et al., 2023; Li et al., 2023; Subramani et al., 2022). Implicit in much of this literature is a pipeline assumption: once structure is detected and localized, releasing it into behavior is an engineering matter.

Recent evidence complicates this assumption. Steering vectors generalize unreliably across prompts and distribution shifts (Tan & Chanin, 2024); strong detectors of internal states coexist with failed control in concurrent work (Anonymous, 2026); and in recent work, a small transformer trained on causal-evidence discrimination was shown to suppress—rather than lack—causal structure: the structure is linearly decodable from its residual stream, yet the model behaves as if it were absent (Xun, 2026). Together these results suggest a representation–behavior gap that is structural, not incidental. What they do not answer is the question that matters for intervention-based methods: if we locate the suppressed structure precisely, can we release it into behavior, end to end, under preregistered criteria?

This paper answers that question with a complete, preregistered failure decomposition (Figure 1). We take the suppression phenomenon of Xun (2026) as ground truth and build a four-component pipeline— an OOD detector u, a contrast forward pass, a direction generator (calibration-fitted g or per-instance), and a residual-stream injector—that must (a) decide when intervention is needed, (b) construct the intervention, and (c) deliver it without damaging in-distribution (ID) behavior. Every threshold, evaluation set, statistic, claim template, and fallback branch was fixed and hashed before any corresponding measurement (Section 3); the full audit chain is reproduced in Appendix A.

Our findings decompose the failure into two dissociable parts:

1. Localization is real and sufficient. Transplanting activations at observation-evidence channels of mid layers restores target behavior on suppressed worlds, with paired advantages 0.563 and 0.854 (97.5% confidence intervals excluding zero) and a best-site release rate of 0.889 (Section 5). The structure to be released exists and is site-specific.

2. End-to-end release fails twice, for two independent reasons. First, the detector that passed its calibration with zero OOD triggers inverts out of distribution: it fires on 6.9–7.3% of held-out ID generations and on exactly 0 of 2,400 generations from the worlds that need intervention, so the “gated” pipeline is behaviorally identical to the base model (Section 6). Second, even with the gate removed entirely—unconditional injection of a per-instance linear direction at the localized site—behavior moves monotonically with dose but plateaus far below the preregistered release margin; halving the dose loses performance monotonically, and per-instance adaptivity contributes at most ±0.03 over a fixed direction (Section 7). The entirefamily of linear release directions at this site and resolution is thus bounded away from sufficiency.

The decomposition is the contribution. Neither failure alone would license our conclusions: an OODinverted detector leaves open the possibility that a better detector rescues release; a capped linear direction leaves open the possibility that the gate was the only obstacle. Together, and only together with the localization result, they establish that the representation–behavior gap in this system is not a localization gap, not (only) a routing gap, and not closable by any linear release direction at the tested resolution—a conclusion that preregistration protects from post-hoc re-narration. We discuss implications for steering-vector practice and for evaluation standards in Section 8.

Preregistration and reproducibility. All criteria, thresholds, seeds, claim templates, and decision-tree branches were archived with MD5 hashes before the corresponding data existed; every number reported in this paper traces to a hashed artifact, and the release includes the audit log, code manifests, and per-world records (Appendix A). Total measured cost of the terminal experiments reported here is 4.22 GPU-hours on a single consumer-class GPU.

## 2 RELATED WORK

Probing and the limits of representation-level claims. Linear and nonlinear probes recover a wide range of latent variables from activations (Belinkov, 2022), but probe accuracy is not evidence of use: control tasks and information-theoretic bounds show that decodability and functional role dissociate (Hewitt & Liang,

2019). Unsupervised methods can extract latent knowledge without labels (Burns et al., 2023). Our starting point accepts this critique: stage-1 of our audit (Section 4) itself fails a purely probe-based account, which is precisely why the subsequent stages test interventions rather than decodings.

Activation steering and representation engineering. Adding direction vectors to hidden states can steer sentiment, style, truthfulness, and refusal (Subramani et al., 2022; Turner et al., 2023; Zou et al., 2023; Li et al., 2023; Rimsky et al., 2023; Arditi et al., 2024). However, steering vectors generalize unevenly across prompts and are sensitive to distribution shift (Tan & Chanin, 2024), and single-direction mediation results are typically reported at the level of average effects rather than preregistered behavioral criteria. Our B6.6 result quantifies one such boundary precisely: at a verified causal site, the linear-direction family has a measurable, monotone, and insufficient dose–response curve.

Causal mediation and localization. Causal tracing and interchange interventions localize mediators of factual recall, gender bias, and syntactic agreement (Vig et al., 2020; Meng et al., 2022; Geiger et al., 2022). We use a transplant intervention in this tradition for our localization stage (Section 5), but unlike most localization work we carry the located site forward into an end-to-end behavioral pipeline and let it fail publicly under frozen criteria.

Causal reasoning in language models. CLadder (Jin et al., 2023) evaluates formal causal reasoning across the ladder of causation (Pearl, 2009); recent work documented suppression of latent causal structure in a small transformer trained on causal-evidence discrimination (Xun, 2026). The present paper closes the loop: it tests whether the documented-but-suppressed structure can be released into behavior, and decomposes the failure.

Concurrent work. The independent concurrent study The Geometry of Knowing vs. Steering (Anonymous, 2026) reports “perfect detection, failed control” in large language models, conceptually convergent with our detector–release dissociation. Our results differ in method (preregistered decision trees with hashed claim templates, a controlled synthetic causal domain, and a localized causal site) and in conclusion granularity: we separate gate failure (OOD-inverted detector) from release failure (bounded linear direction family) and show each is independently sufficient to sink the pipeline.

Preregistration and negative results. Undisclosed analytic flexibility inflates false positives (Simmons et al., 2011); preregistration constrains it (Nosek et al., 2018). Machine-learning evaluations rarely preregister claim templates—verbatim sentences the authors commit to publishing conditional on outcomes. We do, and we treat both failed terminal experiments as reportable results rather than pilots.

## 3 SETUP, MEASUREMENTS, AND PREREGISTRATION PROTOCOL

## 3.1 MODEL AND TASK

We study C1, a 25.7M-parameter GPT-style decoder-only transformer (Vaswani et al., 2017) (8 layers, naïve attention, vocabulary 41) trained on causal-evidence discrimination as specified in Xun (2026): prompts present observational evidence under a structural causal model and the model answers interventional queries $\operatorname { d o } ( X = x _ { 0 } )  Y = ?$ . The training distribution contains Simpson-trap worlds in which the observational correlation and the causal effect have opposite signs. The suppression phenomenon of interest: on heldout confounded worlds the model’s behavior tracks the observational association even though the causal structure is linearly decodable from its residual stream (Xun, 2026). We use the frozen checkpoint of Xun (2026) throughout; no parameter of C1 is ever updated in this paper.

## 3.2 BEHAVIORAL MEASUREMENTS

All behavioral statistics derive from a two-point response grid. For a world w with answer-relevant span [lo, hi] we measure the unit-span slope of the model’s parsed numeric response,

$$
\Delta \ = \ \frac { \mathrm { p a r s e } ( r ( \mathrm { h i } ) ) - \mathrm { p a r s e } ( r ( \mathrm { l o } ) ) } { \mathrm { h i } - \mathrm { l o } } ,\tag{1}
$$

where $r ( \cdot )$ is greedy decoding and parse extracts the signed numeric answer (unparseable outputs are rejected conservatively). Each world supplies three slope measurements:

$$
\begin{array} { r l } & { \Delta _ { \mathrm { s a m e } } = \Delta \mathrm { ~ u n d e r ~ t h e ~ s a m e ~ e v i d e n c e } ; } \\ & { \Delta _ { \mathrm { z e r o } } = \Delta \mathrm { ~ u n d e r ~ e v i d e n c e ~ z e r o e d ~ a t ~ o b s ~ c h a n n e l s } ; } \\ & { \qquad \Delta _ { p } = \Delta \mathrm { ~ u n d e r ~ i n t e r v e n t i o n } . } \end{array}\tag{2}
$$

The denominator $D = \Delta _ { \mathrm { z e r o } } - \Delta _ { \mathrm { s a m e } }$ quantifies how much behavior the zeroed evidence removes; a world is measurable iff $| D | \geq 0 . 0 5$ (the denominator guard; unmeasurable worlds are excluded from rate statistics but retained in the audit). The release rate of an intervention is

$$
R \ = \ \frac { \Delta _ { p } - \Delta _ { \mathrm { s a m e } } } { \Delta _ { \mathrm { z e r o } } - \Delta _ { \mathrm { s a m e } } } \ \in \ ( - \infty , \infty ) ,\tag{3}
$$

the fraction of the zeroed-evidence behavioral shift restored by the intervention. $R { = } 1$ is perfect release;   
$R { = } 0$ is no effect; $R { < } 0$ overshoots in the wrong direction.

For population-level release we fit the offset-gain regression over a world set W,

$$
\widehat { \Delta _ { w } } \ = \ a \cdot \mathrm { A T E } _ { w } + b , \qquad w \in \mathcal { W } ,\tag{4}
$$

where $\mathrm { A T E } _ { w }$ is the world’s ground-truth causal effect. The ideal responder has $a = 1 , b = 0 ;$ ; a suppressed model exhibits large $b$ (a constant pull toward the observational answer). Our preregistered release criteria read off the intercept b (must drop ${ \mathrm { t o } } \leq + 0 . 0 8 )$ and require the ID slope a to not degrade by more than 0.05.

Two supporting statistics recur. Reversal count: on worlds with $\mathrm { A T E } _ { w } < 0 .$ , the number of worlds on which the sign of the prediction disagrees with the sign of the ATE with margin $| \cdot | > 0 . 0 5$ . McNemar exact test (McNemar, 1947): for paired before/after correctness, one-sided $p = \mathrm { \bar { P r } } [ \mathrm { \bar { B i n } } ( b + c , \frac { 1 } { 2 } ) \leq c ]$ over discordant pairs $( b , c )$ . Rates are reported with Wilson 95% intervals (Wilson, 1927); regression intercepts and paired differences with bootstrap 95% intervals (Efron & Tibshirani, 1993); localization confidence intervals are Bonferroni-corrected across the compared pairs.

## 3.3 REPRESENTATION-LEVEL MEASUREMENTS

For representation comparisons we use centered linear CKA (Kornblith et al., 2019), i.e. the RV coefficient

$$
\mathrm { C K A } ( X , Y ) = { \frac { \mathrm { t r } ( \Sigma _ { X } \Sigma _ { Y } ) } { \sqrt { \mathrm { t r } ( \Sigma _ { X } ^ { 2 } ) \mathrm { t r } ( \Sigma _ { Y } ^ { 2 } ) } } } ,\tag{5}
$$

computed on mean-pooled, layer-normalized features. The site feature map is

$$
\phi _ { \ell } ( x ) = \mathrm { L a y e r N o r m } \big ( \mathrm { M e a n P o o l } _ { \mathrm { o b s } } \big ( h ^ { ( \ell ) } ( x ) \big ) \big ) ,\tag{6}
$$

mean-pooling over the observation-evidence token rows of the residual stream at layer $\ell ,$ concatenated across the site’s layers. The detector score is a probe $u ( x ) \in$ R trained to separate confounded from nonconfounded worlds; the pipeline triggers intervention iff $u ( x ) \geq T$ for a threshold T fixed at calibration.

![](images/170c3a493290ba38948241025e0fad99600ccc00a4bbebc16876d4462b36a94d.jpg)  
Figure 1: The CCG release pipeline and the two failure locations established in this paper. Detection gates the entire intervention path; B6.5 shows the gate inverts out of distribution and never fires on the worlds that need it. B6.6 removes the gate (unconditional injection) and shows the linear release-direction family at the localized site still cannot reach the preregistered margin. The localization stage (Section 5) sits upstream of both and passes.

## 3.4 THE CCG PIPELINE

The contrastive causal gate (CCG) inference pipeline has four components (Figure 1): (i) detect—score $u ( x )$ and compare against $T ; \mathrm { i f } u ( x ) < T$ the prompt follows the ID branch and is answered unmodified; (ii) contrast—run a second forward pass with evidence zeroed at obs channels; (iii) generate—a small learned map g produces a direction δ from the contrast pair; (iv) inject—the residual stream is modified as

$$
h ^ { ( \ell ) } ( x ) \ \gets \ h ^ { ( \ell ) } ( x ) + \lambda \cdot \delta ( x ) , \qquad \ell \in \mathrm { s i t e } ,\tag{7}
$$

followed by a second decoding; conservative refusal rules reject unparseable outputs and degenerate contrasts. Two direction constructions are studied. The calibration-fitted direction (B6.5) uses a two-segment mean-difference vector

$$
\delta _ { \mathrm { Z E R O } } \ : = \ : \overline { { \mathrm { M e a n P o o l } } } \big ( h _ { \mathrm { z e r o } } \big ) \ : - \ : \overline { { \mathrm { M e a n P o o l } } } \big ( h _ { \mathrm { s a m e } } \big ) ,\tag{8}
$$

with raw mean pooling (no LayerNorm), frozen after calibration. The per-instance direction (B6.6) removes all learned parameters and the gate:

$$
\delta ( x ) \ = \ \mathrm { M e a n P o o l } _ { \mathrm { r a w } } \big ( h _ { \mathrm { z e r o } } ( x ) \big ) - \mathrm { M e a n P o o l } _ { \mathrm { r a w } } \big ( h _ { \mathrm { s a m e } } ( x ) \big ) ,\tag{9}
$$

computed per prompt and injected unconditionally $( \lambda \in \{ 0 . 5 , 1 . 0 \} )$ or always-on as a fixed control.

## 3.5 PREREGISTRATION PROTOCOL

Every experiment reported here follows the same discipline, archived in a single append-only audit log (Appendix A):

1. Criteria before data. The full criteria file—thresholds, evaluation sets, seeds, statistics, and verbatim claim templates—is hashed (MD5) and archived before any corresponding measurement is produced.

2. Frozen branches. Each outcome maps to a pre-written claim template; missed criteria follow the prewritten failure branch. Thresholds are never lowered, statistics never re-specified, and seeds never added after seeing data.

3. Protocol amendments precede data. Any amendment (e.g. the B6.6 amendment of Section 7) is archived before any data it governs exists.

4. Pilot transparency. Pilot results that informed design choices are reported alongside confirmatory results, never silently absorbed.

The terminal experiments have fixed evaluation sets: CONF50 (50 held-out confounded worlds at full power), ID40 (40 held-out non-confounded worlds), and a calibration set of 40 worlds (45 drawn minus 5 overlapping CONF50) spanning 8 families with 9 positive worlds scattered across 7 families.

Table 1: Stage-1 audit summary. Each gate was preregistered with its threshold and its failure branch; all three failed, and each failure dictated the design of the next stage. Intervals are 95% (bootstrap or Wilson as marked in the audit).
<table><tr><td>Gate</td><td>Question</td><td>Preregistered criterion</td><td>Result</td><td>Verdict</td></tr><tr><td>BO</td><td>Is readout behavior-coupled?</td><td>≥ 25/50 measurable worlds</td><td>11/50</td><td>FAIL</td></tr><tr><td>B0.5</td><td>Do features align across evidence?</td><td>CKA ≥ 0.7 vs ref. 0.431</td><td>0.061-0.068</td><td>FAIL</td></tr><tr><td>B1</td><td>Is the signal usable by readout?</td><td>OOD/transfer AUC above floor</td><td>worst 0.366; CLadder reversed</td><td>FAIL</td></tr></table>

## 4 STAGE 1: THREE GATES BEFORE ANY RELEASE ATTEMPT

Before attempting release, the audit chain subjected the naive hope—“the structure is decodable, so release should follow”—to three preregistered gates (Table 1). All three failed. These failures are not the subject of this paper, but they define why the pipeline of Figure 1 was necessary at all, and they supply the baselines against which the later results are read.

B0: behavior-coupled readout does not exist. Probes trained to predict the causal answer from residual features achieve measurable behavioral coupling only on 11/50 worlds (< 25, the preregistered sentinel). On the measurable subset, mean release rates by site are 0.785 ([0.637, 0.932], L23<sub>obs</sub>), 0.829 ([0.663, 0.994], $\mathrm { L } 3 4 5 _ { \mathrm { o b s } } )$ , 0.789 ([0.587, 0.962], L2345<sub>all</sub>), 0.326 ([0.112, 0.613], L23<sub>non-obs</sub>), and 0.186 ([−0.104, 0.461], $\mathrm { L } 3 4 5 _ { \mathrm { n o n - o b s } } )$ : coupling concentrates on observation-evidence rows, but 39/50 worlds fall below the denominator guard and are unmeasurable.

B0.5: representations do not align across evidence types. Pairwise CKA (Eq. (5)) between the confounded-world features and their observational counterparts ranges 0.061–0.068 across 9 families, against a within-family reference scale with mean 0.431 [0.144, 0.998] and threshold 0.7. The two evidence types occupy nearly orthogonal feature geometries, foreclosing “readout transfer” and motivating the gated-route design.

B1: the answer is detectable but not usable by the readout. A k=32 MLP probe separates world families with AUC 0.962, yet the behaviorally relevant signal is absent where it matters: ID trigger rate 0.100 (at the boundary), worst leave-one-family-out AUC 0.366 (ood-sign 0.640, ood-scale 0.366, ood-ρ 0.658), and zero-shot CLadder transfer AUC 0.022–0.292 with reversed direction—the probe’s ranking anti-predicts external causal difficulty. A linear head reaches only 0.70–0.74 where the MLP reaches 0.962: the decodable information is nonlinearly encoded and does not survive distribution shift in the direction behavior would need.

Stage 1 therefore establishes the problem’s shape: the causal structure is present (it is decodable), but it is neither behaviorally coupled, nor geometrically aligned across evidence types, nor usable by a generalizing readout. Release, if possible at all, must go through intervention on the residual stream—which first requires knowing where to intervene.

## 5 LOCALIZATION SUCCEEDS: THE SUPPRESSION IS SITE-SPECIFIC

The localization stage (B0-rep) asks whether transplanting activations at a candidate site can restore suppressed behavior, under preregistered criteria C1 (paired release advantage over two control pairs, 97.5%

(a) Pilot: five candidate sites (n=11)  
![](images/1c41f6a4ab04367f5ae0c6cf96234beb7f1e4a0b9163c15701e7c8bb6127fc3b.jpg)

(b) Confirmatory: C1 contrasts (n=48)  
![](images/64dafd6f9b8e3fc1228609abb103d158941a068989f4386b6f4153606ad9c8f2.jpg)  
<sub>pa</sub>i<sub>re</sub>d diff<sub>erence</sub> i<sub>n</sub> R (<sub>97.5% CI</sub>)  
Figure 2: Localization. (a) Pilot screen across five candidate sites (release rate, Eq. (3)); obs-row sites dominate, and $\mathrm { L } 2 3 _ { \mathrm { o b s } }$ is selected as the confirmatory site. (b) Confirmatory paired release advantages (Eq. (10)) with 97.5% Bonferroni-corrected intervals; both intervals exclude zero. Pilot numbers are archival and reported alongside confirmatory numbers per the preregistration protocol.

Table 2: B0-rep localization ledger. C1 requires both paired advantages’ 97.5% CIs to exclude zero; C2 requires best-site release rate $\geq 0 . 5 .$ . Sub-sample bookkeeping: fork $n { = } 1 0$ ineligible worlds and IV $n { = } 3 9$ are reported in the audit. Cost: 3.82 GPU-hours.
<table><tr><td>Criterion</td><td>Statistic</td><td>Value</td><td>Verdict</td></tr><tr><td>Eligibility</td><td>Eligible worlds M</td><td>48/608 (Wilson [0.0601, 0.1031])</td><td>PASS</td></tr><tr><td>C1</td><td>L23 paired advantage</td><td>0.5628 [0.3932, 0.7302]</td><td>PASS</td></tr><tr><td>C1</td><td>L345 paired advantage</td><td>0.8542 [0.6088, 1.0995]</td><td>PASS</td></tr><tr><td>C2</td><td>Best site  $\left( \mathrm { L } 2 3 _ { \mathrm { o b s } } \right)$  mean R</td><td>0.8886</td><td>PASS</td></tr><tr><td>Guard</td><td>±0.01 sensitivity</td><td>both pairs still pass</td><td>PASS</td></tr></table>

Bonferroni-corrected CIs excluding zero) and C2 (best-site release rate $\geq 0 . 5 )$ . Sites are layer sets crossed with token-row classes (obs vs non-obs); the transplant replaces the site’s residual rows of a suppressed world with those of its released counterpart.

Result: full pass. Of 608 screened worlds, $M \ = \ 4 8$ met the eligibility criteria (rate 0.0789, Wilson 95% [0.0601, 0.1031], exceeding the preregistered floor of 40). Both control pairs pass C1 with substantial margins:

$$
\mathrm { L 2 3 ~ p a i r : ~ 0 . 5 6 2 8 ~ [ 0 . 3 9 3 2 , 0 . 7 3 0 2 ] } , \qquad \mathrm { L 3 4 5 ~ p a i r : ~ 0 . 8 5 4 2 ~ [ 0 . 6 0 8 8 , 1 . 0 9 9 5 ] } \qquad ( n = 4 8 ) ,\tag{10}
$$

and the best site, $\mathrm { L } 2 3 _ { \mathrm { o b s } } .$ , attains mean release rate

$$
R _ { \mathrm { { s i t e } } } ~ = ~ 0 . 8 8 8 6 ~ \geq ~ 0 . 5 ~ ( \mathrm { C } 2 , \mathrm { p a s s } ) ,\tag{11}
$$

i.e. the transplant restores, on average, 89% of the behavioral shift that evidence zeroing removes (Eq. (3)). A preregistered ±0.01 guard-band sensitivity check leaves both pairs’ verdicts unchanged. Figure 2 shows the pilot site screen and the confirmatory paired advantages; per protocol, pilot numbers are reported alongside confirmatory ones (Table 2).

What localization does and does not establish. C1/C2 establish the preregistered claim G1 verbatim: “the suppression-by-default under OOD is localized to the observation-evidence row channels, and transplant intervention is sufficient to restore the target behavior.” They do not establish that a deployable pipeline can reproduce the transplant: the transplant uses oracle access to the released counterpart of each suppressed world. Converting localization into release requires a detector (to know when to intervene) and a generator (to construct the direction from the suppressed prompt alone). Those are the components that fail next.

Table 3: B6.5 terminal criteria, all frozen before data. Bold rows failed. The gate triggered on 0/2,400 CONF50 generations, so the “gated” pipeline is numerically identical to the base model on exactly the worlds the intervention exists for.
<table><tr><td>Criterion</td><td>Measurement</td><td>Threshold (frozen)</td><td>Observed</td><td>Verdict</td></tr><tr><td>B65a</td><td>Release intercept b (Eq. (4))</td><td>≤ +0.08</td><td>0.3820 [0.3396, 0.4230]</td><td>FAIL</td></tr><tr><td>B65b</td><td>Slope stability / corr</td><td> $| \Delta a | \leq 0 . 1 0 \ \land \ \mathrm { c o r r } > 0 . 4 0$ </td><td> $0 . 0 ; \mathrm { c o r r 0 . 6 2 9 9 } ( n _ { \mathrm { p o s } } = \dot { 2 } 3 )$ </td><td>PASS</td></tr><tr><td>B65c</td><td>Reversals + McNemar</td><td> $\leq 1 4 / 2 7 \ \wedge \ p < 0 . 0 5$ </td><td> $\mathbf { 2 5 } / 2 7 ; b { = } c { = } 0 , p { = } \mathrm { N a N }$ </td><td>FAIL</td></tr><tr><td>B65d</td><td>ID slope degradation</td><td>≤ 0.05</td><td>0.0038-0.0042</td><td>PASS</td></tr><tr><td>B65e</td><td>ID trigger rate</td><td>≤ 0.02</td><td>0.0688-0.0734</td><td>FAIL</td></tr><tr><td>Gate audit</td><td>CONF50 triggers</td><td>(diagnostic)</td><td>0/2,400</td><td></td></tr></table>

## 6 FAILURE 1: THE GATE INVERTS OUT OF DISTRIBUTION

B6.5 is the preregistered terminal test of the full gated pipeline (Figure 1): detector u with threshold T, contrast pass, calibration-frozen direction δ (Eq. (8)), injection, guarded parse. Five criteria (B65a–e) were frozen before calibration; the verdict was bound to a claim template before any CONF50/ID40 data existed.

Calibration looked clean. The detector is a light linear probe: PCA (k=8) over the site features ϕ $\left( \operatorname { E q . } \left( 6 \right) \right)$ followed by logistic regression (300 epochs, lr $1 0 ^ { - 2 } )$ , labeled y=1 iff the world’s denominator satisfies $D > 0 . 0 5$ , with the per-seed threshold fixed as the intersection across seeds {0, 1, 2}. On the 40-world calibration set (24/8/8 split, 3 seeds), the detector’s threshold settled at $T \in \{ 0 . 6 \dot { 9 } 9 , 0 . 7 \dot { 2 } 8 , 0 . 7 3 7 \}$ with zero validation ID triggers, OOD recall 0.5, parse rate 1.0, and direction norms $\lVert \delta \rVert _ { 2 } = 8 . 3 0$ (two-segment) / 14.35 (three-segment). By every calibration-visible measure, the gate was conservative to the point of never firing.

The terminal test failed, and the failure was silent. On held-out data the pipeline failed three of five criteria (Table 3). The release intercept is $b ~ = ~ 0 . 3 8 2 0$ (boot 95% [0.3396, 0.4230]) against the frozen threshold $b \leq 0 . 0 8$ (B65a, FAIL); ID trigger rates are 0.0688/0.0688/0.0734 (44/44/47 of $n _ { \mathrm { g e n } } = 6 4 0 )$ against the frozen cap 0.02 (B65e, FAIL); and the paired McNemar test is degenerate $( b { = } c { = } 0 , p { = } \mathrm { N a N } )$ with 25/27 negative-ATE worlds sign-reversed—against a cap of 14/27 (B65c, FAIL). Two criteria passed: $| \Delta a | = 0 . 0$ with positive-world correlation 0.6299 (B65b), and ID slope degradations of 0.0038–0.0042 under the 0.05 cap (B65d).

Why it failed: an OOD-inverted detector. The diagnostic row of Table 3 is the finding. Across all 2,400 CONF50 generations—the worlds for which the entire apparatus exists—the gate triggered zero times: GATE ≡ BASE, and the intervention never happened (Figure 3). Simultaneously the same detector fired on 6.9–7.3% of held-out ID generations where intervention is harmful. The detector learned “looks like calibration $\mathrm { O O D } ^ { \cdot \mathrm {, } \cdot }$ rather than “needs intervention,” and calibration, by construction, contained no example of the positive class’s held-out realization. Crucially, no intermediate statistic raised an alarm: calibration metrics were perfect, ID behavior was preserved (B65d), and slope stability passed (B65b). Without the preregistered gate audit, the failure would have read as “the intervention does nothing”—exactly the misreading our own audit log initially recorded and then corrected on inspection of the per-generation records (Appendix A).

![](images/ecd5ac5df5a66fd8ed69656c67be247858689849adf5b3edba81d469990c370a.jpg)  
Figure 3: The gate inversion. Detector trigger rates at calibration (left; 0.0 on validation ID by construction), on held-out ID40 (center; 6.9–7.3% across three seeds, above the frozen 0.02 cap, dashed line), and on heldout CONF50 (right; 0/2,400). The detector is simultaneously too permissive where intervention is harmful and absolutely silent where it is needed.

Attribution is ambiguous—and that is a problem. B6.5’s frozen claim template states the disjunction faithfully: representation-level localization holds (B0-rep G1), while behavior-level release is not achieved at this resolution with this method. But B6.5 alone cannot separate two causes: (i) the gate never fired, so the release direction was never exercised at test time; (ii) the direction itself might be insufficient even if exercised. The B6.6 amendment—archived before any B6.6 data—removes the gate entirely to decide between them.

## 7 FAILURE 2: THE LINEAR RELEASE FAMILY IS BOUNDED

## 7.1 THE B6.6 AMENDMENT

The B6.6 amendment (archived with hashes before any corresponding data) makes one surgical change: remove the gate. The detector score u plays no role; the per-instance direction δ(x) (Eq. (9)) has zero learned parameters; injection (Eq. (7)) is unconditional, at two doses $\lambda \in \{ 0 . 5 \}$ (ARM-S) and λ = 1.0 (ARM-R), with an always-on fixed-direction control re-measured within the experiment (P1 of the amendment). BASE is re-measured in-experiment as well. The amendment froze a three-valued readout with mutually exclusive priority (P2): R1 (a release margin exists → the pipeline design was right and only the gate failed), R2 (no margin in either arm → the linear release family is bounded at this site and resolution), R3 (intermediate), with INDETERMINATE and DESCRIPTIVE-ONLY fallbacks for insufficient ID power or unclassifiable patterns (P3–P5).

Table 4: B6.6 terminal readout (verdict R2: no margin). Dose–response is monotone in λ for both intercept (down) and slope (down); neither arm approaches the frozen release threshold $b \leq 0 . 0 8$ ; ID behavior is not harmed; per-instance adaptivity adds ≈ 0 over the fixed direction. Cost: 0.1476 GPU-hours.
<table><tr><td>Quantity</td><td>BASE (λ=0)</td><td> $\mathsf { A R M - S } \left( \lambda { = } 0 . 5 \right)$ </td><td>ARM-R (λ=1.0)</td></tr><tr><td>Intercept  $b ( \mathrm { E q . } ( 4 ) )$  Slope à</td><td>0.3820</td><td>0.3106 0.6773</td><td>0.2637 0.5721</td></tr><tr><td>ID slope drop Reversals (neg-ATE worlds)</td><td>0.8236  $2 5 / 2 7$ </td><td>-0.0243</td><td>-0.0169 21/27</td></tr><tr><td>McNemar (paired)</td><td></td><td></td><td> $b { = } 1 , c { = } 1 , p { = } 0 . 7 5$ </td></tr><tr><td>Per-world gain vs. always-on Positive-world corr (ARM-R)</td><td>0.328</td><td>+0.0268</td><td>-0.0199</td></tr></table>

## 7.2 RESULT: R2, NO MARGIN

B6.6 (Table 4) answers the attribution question left open by B6.5, and the answer is: both. Unconditional injection moves behavior in the right direction—the intercept falls monotonically with dose, $0 . 3 8 2 0  0 . 3 1 0 6  0 . 2 6 3 7$ (Figure 4)—but the full-strength arm remains more than 0.18 above the frozen release threshold, and pushing dose further trades slope for intercept along the same monotone curve rather than closing the gap. The shrunk arm loses intercept ground monotonically, confirming a smooth dose–response rather than a threshold effect. ID behavior is unharmed in both arms (slope drops −0.017 and −0.024), so the bound is not an artifact of collateral damage. Per-instance adaptivity, the entire point of the $\delta ( x )$ construction, contributes −0.0199 (ARM-R) and +0.0268 (ARM-S) per world over the always-on fixed direction (Figure 5): statistically and practically nothing, against the ≈ 0.18 gap that remains. The preregistered readout is therefore R2\_NO\_MARGIN, with its verbatim claim: per-instance linear directions at this site and resolution likewise cannot be released—the linear releasefamily,fixed and adaptive alike, is bounded as a whole.

Consequences frozen in advance. Under the amendment’s own frozen clauses, R2 prohibits the follow-up escalation experiment (B7) from launching: there is no release margin to route. The decision tree’s terminal state is thus the conjunction we report as our main result—the gate is OOD-inverted and the linear release family at the localized site is bounded below the margin, with localization intact upstream.

## 8 DISCUSSION

The representation–behavior gap is not one gap. Our three positive/negative pairs—decodable but not behaviorally coupled (stage 1), localizable but not gateable (B0-rep vs. B6.5), gate-removable but not linearly releasable (B6.5 vs. B6.6)—show that “the model knows but doesn’t use” decomposes into at least three independent obstacles, each sufficient to sink a naive pipeline. Reporting any one of them alone would support a different (and wrong) remediation story: better probes (stage 1 already falsifies), better calibration (B6.6 falsifies), better sites (B0-rep falsifies). The value of the end-to-end design is that the failure stories are forced to be compatible with one another.

![](images/c225d7262862555cfa32da5493d6ccddcb38bbbd361a49ad60e17f6f1b00fbf1.jpg)

![](images/ad6fb8a9bec1e171722e4bea802c0eda76b24eeea8c2132a714a16c7facb2b79.jpg)

Figure 4: B6.6 dose–response on CONF50. (a) Per-world offset-gain slope ∆ (Eq. (1)) vs. ground-truth ATE for BASE and ARM-R; injection shifts the regression toward the ideal diagonal but the fitted intercept (Eq. (4)) remains far from zero. (b) Fitted intercept b and slope a as functions of dose λ; both are monotone, and the intercept plateaus at more than three times the frozen release threshold (dashed line).  
![](images/01fe05f498e19d2aa93949e46ac6fd4b27819af626ca7af1a50805b039a90340.jpg)  
Figure 5: Per-instance adaptivity contributes approximately nothing. Per-world difference between each adaptive arm and the always-on fixed direction $( n ~ = ~ 5 0$ worlds; black bars mark means −0.0199 and +0.0268). The remaining release gap after full-dose injection is $\approx 0 . 1 8$ , an order of magnitude larger than any adaptive gain.

Silent gating failures are a distinct evaluation hazard. B6.5’s most dangerous property is that it is quiet: every calibration statistic was perfect, ID behavior was preserved, and the intervention simply never ran on the worlds that mattered. Any gated or triggered intervention system—including selective steering, refusal circuits, and tool-routing heads—can fail in this mode, and standard held-out metrics will not catch it unless the trigger itself is audited on the deployment distribution. We recommend that gate-level trigger counts on the target distribution be reported as a first-class metric wherever a gate exists, and note that our own audit initially misread this failure before the per-generation records corrected it (Appendix A)—the misreading is the default; only instrumentation prevents it.

Bounded linearity at a verified site constrains steering practice. Steering-vector methods report average behavioral shifts (Turner et al., 2023; Zou et al., 2023; Li et al., 2023; Subramani et al., 2022), and their unreliability under distribution shift has been documented (Tan & Chanin, 2024). B6.6 adds a sharper datum: at a site whose causal sufficiency was independently verified by transplant, the per-instance linear family exhibits a smooth, monotone dose–response that plateaus at more than three times the required margin, with adaptivity worth less than ±0.03. If linear directions are insufficient even where localization is certain, then steering failures elsewhere cannot be automatically attributed to “wrong layer” or “wrong direction”: the linear parametrization itself is a candidate bottleneck. Nonlinear or multi-site release mechanisms are the obvious next object of study—under the same preregistered discipline, since the space of post-hoc release mechanisms is large and forgiving.

Relation to concurrent work. Anonymous (2026) document “perfect detection, failed control” in large models, convergent with our gate result. Our decomposition suggests their failed control may itself be composite (gate inversion, direction boundedness, or both), and that resolution-level reporting—trigger counts, dose–response curves, and per-world adaptive gains—can separate the causes in their setting too.

Limitations. (i) A single 25.7M model on a synthetic causal domain: the specific numbers are C1’s, and only the dissociation pattern is a candidate for generality. (ii) The release family tested is linear, single-site, and residual-additive; our negative claim is explicitly scoped to that family at that resolution and does not touch nonlinear or distributed release. (iii) The denominator guard excludes 39/50 stage-1 worlds from rate statistics; rates are computed on measurable worlds only, and the audit reports the exclusions. (iv) The eligibility rate for localization (7.9%) means C1/C2 characterize a selected subpopulation of worlds; we report the selection rate and its interval rather than extrapolating. (v) The gate was calibrated without heldout examples of the positive class’s test-time realization; a detector trained with such examples might behave differently—but that is precisely the deployment condition the pipeline was designed for, and it failed there.

Why preregistration mattered here. Every tempting post-hoc move was foreclosed in advance: rethresholding the gate after seeing CONF50, redefining release around the slope rather than the intercept, promoting the adaptive-gain sign flip into a story, or launching the escalation experiment despite R2. The frozen branches forced the paper to be about the failure decomposition rather than about whichever sub-result could be made to look positive. We consider this the methodological contribution most likely to transfer.

## 9 CONCLUSION

We subjected the detect–localize–release pipeline to a fully preregistered end-to-end test on a system with a documented representation–behavior gap. Localization passed decisively: the suppression is site-specific and transplant-sufficient. Release then failed twice, independently: the detector inverted out of distribution and silently disabled the entire intervention path, and the gate-free linear direction family plateaued at more than three times the preregistered threshold with negligible adaptive gain. Representation-level localization and behavior-level release are different problems, and the second is not solved by solving the first. For the growing literature that equates finding structure with controlling it, our results supply a concrete, audited counterexample—and a discipline (hashed criteria, verbatim claim templates, gate-level auditing, mandatory pilot reporting) for making the next counterexample harder to explain away, including by ourselves.

## AI USE STATEMENT

A large-language-model assistant was used to help draft and copy-edit the manuscript, to adversarially proofcheck its claims against the underlying experimental data and audit archive, and to develop and debug the evaluation code. All scientific claims, experimental design, data, analyses, and conclusions were made and verified by the author, who takes full responsibility for the content of this paper.

## ETHICS STATEMENT

This work studies a small synthetic-domain model and introduces no human-subjects, privacy, or dual-use concerns. Its negative results concern the limits of activation interventions and, if anything, counsel caution about intervention-based control claims. All data are generated by code released with the audit artifacts.

## REPRODUCIBILITY STATEMENT

Every number in this paper traces to a hashed artifact in the released audit chain: criteria files were MD5- archived before any corresponding data existed; code manifests, configurations, per-world records, and console logs are included; the model checkpoint is the frozen public artifact of Xun (2026); and all verdicts are reproducible by recomputing the archived statistics from the released records (we did so independently for both terminal experiments; the recomputation matched the pipeline’s own analysis field-by-field). Appendix A summarizes the chain.

## REFERENCES

Anonymous. The geometry of knowing vs. steering: Perfect detection, failed control. arXiv preprint arXiv:2606.24952, 2026.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. Refusal in language models is mediated by a single direction. Advances in Neural Information Processing Systems, 37:136037–136083, 2024.

Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.

Collin Burns, Haotian Ye, Dan Klein, and Jacob Steinhardt. Discovering latent knowledge in language models without supervision. International Conference on Learning Representations, 2023.

Bradley Efron and Robert J. Tibshirani. An Introduction to the Bootstrap. Chapman & Hall, 1993.

Atticus Geiger, Zhengxuan Wu, Hanson Lu, Josh Rozner, Elisa Kreiss, Thomas Icard, Noah Goodman, and Christopher Potts. Inducing causal structure for interpretable neural networks. International Conference on Machine Learning, pp. 7324–7338, 2022.

John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pp. 2733–2743, 2019.

Zhijing Jin, Yuen Chen, Felix Leeb, Luigi Gresele, Ojas Kamal, Zhiheng Lyu, Kevin Blin, Fernando Gonzalez Adauto, Max Kleiman-Weiner, Mrinmaya Sachan, and Bernhard Schölkopf. Cladder: Assessing causal reasoning in language models. Advances in Neural Information Processing Systems, 36:31038– 31065, 2023.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pp. 3519–3529, 2019.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023.

Quinn McNemar. Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2):153–157, 1947.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pp. 17359–17372, 2022.

Brian A. Nosek, Charles R. Ebersole, Alexander C. DeHaven, and David T. Mellor. The preregistration revolution. Proceedings ofthe National Academy ofSciences, 115(11):2600–2606, 2018.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2nd edition, 2009.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. arXiv preprint arXiv:2312.06681, 2023.

Joseph P. Simmons, Leif D. Nelson, and Uri Simonsohn. False-positive psychology: Undisclosed flexibility in data collection and analysis allows presenting anything as significant. Psychological Science, 22(11): 1359–1366, 2011.

Nishant Subramani, Niveditha Suresh, and Matthew E. Peters. Extracting latent steering vectors from pretrained language models. In Findings of the Association for Computational Linguistics: ACL 2022, pp. 566–581, 2022.

Daniel L. Tan and David Chanin. Analysing the generalization and reliability of steering vectors. Advances in Neural Information Processing Systems, 37:139179–139202, 2024.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan Ulisse, Ni Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization. arXiv preprint arXiv:2308.10248, 2023.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Yaron Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. Advances in Neural Information Processing Systems, 33:12388–12401, 2020.

Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927.

Xining Xun. Evidence-type competition: Suppression and release of latent causal structure in a small transformer. arXiv preprint arXiv:2607.29484, 2026.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to AI transparency. arXiv preprint arXiv:2310.01405, 2023.

## A AUDIT CHAIN AND PREREGISTERED CLAIM TEMPLATES

## A.1 AUDIT TIMELINE

Table 5 summarizes the append-only audit log (25 registry entries, B0-REG through B0-REG-19 including sub-entries). Each entry binds a criteria/code artifact by MD5 before any data it governs; verdict entries additionally archive the result artifacts and an independent recomputation cross-check.

Table 5: Audit chain milestones. “Hash-first” means the artifact hash was archived before any governed data existed. Full log released with the artifacts.
<table><tr><td>Entry</td><td>Content</td><td>Discipline</td></tr><tr><td>B0-REG-1... 14</td><td>Stage-1 gates B0/B0.5/B1; site selection; localization criteria C1/C2</td><td>hash-first</td></tr><tr><td>B0-rep ledger</td><td>M=48/608; C1 pairs; C2 0.8886; pilot side-by-side</td><td>frozen claim G1</td></tr><tr><td>B0-REG-15</td><td>B6.5 calibration ledger (3 seeds, hashes of u, δzERo)</td><td>hash-first</td></tr><tr><td>B0-REG-16</td><td>B6.5 verdict FAILED; gate 0/2,400 audit; authorial correction (see below)</td><td>frozen template</td></tr><tr><td>B0-REG-17</td><td>B6.6 amendment registered</td><td>before any B6.6 data</td></tr><tr><td>B0-REG-18</td><td>B6.6 code + config hashes; P1–P5 pins; 31/31 self-tests</td><td>hash-first</td></tr><tr><td>B0-REG-19</td><td>B6.6 verdict R2_NO_MARGIN; recomputation matched; B7 not launched</td><td>frozen template</td></tr></table>

Authorial correction (B0-REG-16 addendum). The first verbal reading of B6.5 recorded the failure as “the injection perturbs behavior without benefit.” Inspection of the per-generation records showed this was wrong: the gate triggered 0/2,400 times, GATE ≡ BASE, and the 25/27 reversals are the base model’s own sign errors on negative-ATE worlds—the suppression phenomenon of Xun (2026) reproducing itself, not an injection artifact. The correction is archived in the same log; we report it in the paper because it is exactly the misreading that un-instrumented gated systems invite (Section 8).

Registered code imperfection. The B6.5 records writer hard-coded "partial": true; the field is cosmetic, was registered rather than silently patched (the code was frozen), and the B6.6 writer was corrected to terminal-state flipping (verified: B6.6 records show partial=false).

## A.2 VERBATIM CLAIM TEMPLATES

The frozen templates governing the terminal verdicts. The canonical archived text is Chinese; the English renderings below are faithful translations prepared for this paper, with the Chinese originals included in the released audit log.

## B6.5 failure template (invoked)

Representation-level localization holds (B0-rep G1), while behavior-level release is not achieved at this resolution with this method.

## B0-rep G1 template (invoked)

The suppression-by-default under OOD is localized to the observation-evidence row channels, and transplant intervention is sufficient to restore the target behavior.

## B6.6 R2 template (invoked)

Per-instance linear directions at this site and resolution likewise cannot be released—the linear release family, fixed and adaptive alike, is bounded as a whole

## A.3 COMPUTE AND ARTIFACTS

Terminal-experiment costs: localization 3.82 GPU·h; B6.5 calibration 0.04; B6.5 terminal 0.2125; B6.6 0.1476; total 4.22 GPU·h. Released artifacts include the audit log, criteria and configuration files with hashes, code manifests (39 entries), per-world record files for both terminal experiments, console logs, and the figure-generation scripts consuming those records.