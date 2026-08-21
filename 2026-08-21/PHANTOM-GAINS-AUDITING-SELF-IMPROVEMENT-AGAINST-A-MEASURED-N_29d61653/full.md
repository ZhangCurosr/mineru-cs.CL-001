# PHANTOM GAINS: AUDITING SELF-IMPROVEMENT AGAINST A MEASURED NULL

Cheng Xu<sup>1</sup> Nan Yan<sup>2</sup> Liming Chen<sup>3</sup> M-Tahar Kechadi<sup>1</sup>

<sup>1</sup>University College Dublin <sup>2</sup>Georgia Institute of Technology <sup>3</sup>Dalian University of Technology cheng.xu1@ucdconnect.ie, nan.yan@gatech.edu, tahar.kechadi@ucd.ie

## ABSTRACT

Whether a language model has improved itself is increasingly judged not by mean accuracy but by which individual problems it gains and loses. Tracking these transitions means differencing two noisy estimates, leaving them vulnerable to measurement artifacts. Auditing three rounds of rank-32 LoRA self-training on Qwen3-8B against a frozen control pushed through the identical pipeline, we identify seven measurement failures, each of which inverts a reported finding when its control is absent. Several are standard practice. A ledger built on a single greedy decode manufactures capability changes on an untrained model, largely an artifact of inference batching; the expansion statistic separating acquisition from sharpening assigns that same model a rate of 0.280. The natural threshold repair does not survive replication: estimated across the frozen comparisons such a design already contains, its null stays non-zero. We replace it with a per-problem exact test against a pooled baseline under false-discovery-rate control, which detects nothing on any held-out replicate and is unchanged under the multiple-testing rule, error rate and pool size. Applied to a ladder of arms matched in stream, volume and evaluation, the audit finds that external distillation improves problems the base model rarely reaches while three forms of self-training do not; a regression rejects this asymmetry as a by-product of distillation’s larger overall gain $( p < 1 \dot { 0 } ^ { - 8 } )$ . On the far smaller set of problems the base model never reaches, the evidence is inconclusive, while self-training corrupts problems solved at baseline at rates well above the measured floor. Transition-level auditing therefore requires a separately measured null for every statistic it reports: nulls that cost no new experiments, built from baseline replicates a multi-arm study already owns, though not from as few as most possess. Code and evaluation artifacts are available at https://github.com/chengxuphd/phantom-gains.

## 1 INTRODUCTION

A language model that improves itself without new supervision would be a considerable thing, and a growing family of methods reports exactly that (Fang et al., 2025; Tao et al., 2024). STaR fine-tunes on its own correct rationales (Zelikman et al., 2022); TTRL replaces labels with a majority vote over its own samples (Zuo et al., 2025); more elaborate schemes generate their own curricula (Zhao et al., 2025; Chen et al., 2025; Huang et al., 2026) or learn to edit their own weights (Zweiger et al., 2025).

The evidence offered used to be a single number: accuracy before, accuracy after (Atreja, 2025). That is now widely accepted to be insufficient, since an aggregate cannot distinguish a problem the model newly learned to solve from one it could already solve on a good day (Yu et al., 2026b; Yue et al., 2025), nor show what was lost while the mean rose. A second generation of analyses has therefore moved to individual problems, tracking which enter and leave a model’s reachable set (Yuan et al., 2026; Wu et al., 2026; Mayilvahanan et al., 2026) and counting those destroyed under self-training (Lin et al., 2026; Yu et al., 2026a).

This is the right move, and it creates a new problem. A transition is a difference between two estimates, each carrying sampling error, and the events of interest sit exactly where that error bites hardest. Auditing self-training on Qwen3-8B across three evaluation sets, we isolate seven measurement failures, each of which produces a confident and wrong conclusion wherever its control is absent (Figure 1). Two are standard practice, and one lies inside the correction for another.

A single decode is not a state. Temperature-zero decoding is not deterministic under batching (He & Lab, 2025), so a ledger built on one greedy sample per problem gives an untrained model, re-evaluated against itself, 6 apparent learnings and 9 apparent corruptions. That is a corruptionto-learning ratio of 1.5: the shape of the alarming result this literature reports, manufactured out of nothing. Serializing the requests removes three-quarters of those flips and confirms the mechanism, but not all of them: even one request at a time, a frozen model changes 2% of its greedy verdicts.

An expansion statistic needs its own null, and one measurement of a null is not a null. Following Mayilvahanan et al. (2026), a problem counts as expanded when the base model produced no correct sample in k draws and the trained model produces at least one. That rule compares two noisy binomial estimates through asymmetric thresholds, and what it returns when nothing has changed has not been measured. It returns 0.280: a frozen Qwen3-8B evaluated twice at $k { = } 1 2 \mathrm { { 8 } } \ \mathrm { { \stackrel { . . } { e x p a n d s ^ { 3 } } } 7 }$ of the 25 AIME problems it had not reached. All seven are single lucky samples, which appears to license requiring $m \geq 2$ successes and so to drive the null to zero. It does not: that conclusion rests on one frozen comparison, and over the 110 our experiments contain the null is 0.058 [0.038, 0.078], larger than the value it was supposed to certify. The repair is to stop thresholding: a per-problem exact test against a baseline pooled over every independent evaluation of the untrained model gives all arms a common denominator, returns no detection on any held-out replicate, and holds under every multiple-testing rule we try.

Applying the controlled audit, with distillation from a stronger external teacher as a positive control matched in stream, retained volume and evaluation, we find a dissociation. Distillation improves 8–11 of the 22 problems the base model reaches at most five times in 1,408 draws, against 0–2 for three forms of self-training, and a logistic model rejects the reading that this merely follows from the teacher’s larger overall gain $( \beta = \mathsf { \bar { 1 } } . 9 1 , p < 1 0 ^ { - 8 } )$ . We are careful about what this licenses: on the ten problems the base model never reaches at all the comparison is not significant, so the claim is about rarely-reached problems and not about expansion. It aligns with the sharpening account of Huang et al. (2025); Yue et al. (2025) and with Strozzi (2026), on a different method family and with a positive control neither runs. We also quantify what is destroyed: both methods corrupt 88–106 of 1,163 band problems against a design-matched floor of 8, over half of those events larger than any solve-rate change a frozen model produced across twenty independent comparisons.

Contributions. (i) Seven measurement failure modes for transition-level audits, each demonstrated on a run where it reverses a conclusion; the last arises inside the correction for the second (Section 4). (ii) A threshold-free replacement for the expansion statistic, whose null we measure rather than assume and whose comparisons survive the multiple-testing rule, the error rate and the pool size (Section 4.2). (iii) The observation that makes measured nulls nearly free, namely that every arm’s checkpoint 0 is an independent evaluation of the untrained model, together with the replicate count at which one becomes trustworthy, which is more than a three-arm study owns. (iv) A transition ledger with a benchmark-specific noise floor and effect-size accounting that separates threshold flicker from real capability loss, and a policy-gradient arm on which three seeds disagree about when the model collapses but not that it does (Section 5).

## 2 RELATED WORK

What self-improvement can add. Huang et al. (2025) model self-improvement as sharpening: the model acts as its own verifier and concentrates mass on sequences it already assigns high likelihood, improving first-sample accuracy without introducing anything new; Song et al. (2025) formalize the same limit through a generation–verification gap. Empirically, Yue et al. (2025) report that RLVR-trained models are outperformed by their base models at large k, Wu et al. (2026) argue such training cannot place mass outside the base support, and Yuan et al. (2026) trace the effect to diversity collapse, while Liu et al. (2025a) argue the opposite for prolonged training. MATH Beyond (Mayilvahanan et al., 2026) turns the debate into a measurement and supplies the expansion statistic we audit in Section 4.2; Qi et al. (2026) benchmark closed-loop self-evolution against oracle supervision and find a persistent gap.

![](images/a6291e2bf029ac2c235ced365c15ef34ce5a1a326b9ca13b791f578b00d2157a.jpg)  
F1 a single decode in place of the estimator: the frozen model scores CLR=1.5. F2 a one-success expansion threshold: the frozen model “expands” 0.280 of its unreached problems. F3 a fixed token cap under style drift: the most effective arm is scored the most destructive. F4 transition metrics carry ∼10× less power than accuracy: a 112-problem pilot cannot resolve the comparison. F5 training-seed variance: CLR spans 0.55–1.53 for one method. F6 an underpowered probe: a phantom 10-point safety drop. F7 a null measured once: the “corrected” expansion threshold has a null of 0.058, not 0.  
Figure 1: The audit pipeline, and the seven stages at which a measurement failure enters it. Top: an evolution arm draws candidates from an unlabeled stream, filters them by its own criterion (ground-truth answer for STaR, majority vote for the self-training arm, an external teacher for the distillation control) and takes a LoRA step, for three rounds. Middle: a frozen copy of $\theta _ { 0 }$ is pushed through the identical evaluation path, so every statistic is read against a null measured on the same problems with a model that did not train. Bottom: evaluation draws $k { = } 1 2 8$ samples per problem per checkpoint, converts them to a solve rate rather than a single decode, discretizes with hysteresis, and reads off a seven-state ledger. F7 sits on the control path itself: the null each statistic is read against is an estimate, and measuring it once is the same error one level up. Every mode is measured here, inverts a reported conclusion when uncontrolled, and is caught by a control at the stage marked (Section 4).

Concurrent work on transition-level analysis. Strozzi (2026) ask our headline question, whether teacher-free self-training acquires capability or only expresses existing capability, and report a pass@K crossover on a DSL domain with a free verifier. We read our result as a replication in a different domain and supervision regime, with a positive control they do not run. Yuan et al. (2026) track per-problem boundary entry and exit across training, structurally our ledger, under an analytic binomial null. Ours is measured, and we can now say what that buys: for the estimator we recommend the two agree closely, and they part company where the noise is not binomial, which is how we found the expansion statistic’s null to be non-zero at any threshold below m=5 (Appendix E.2).

Everything else. Appendix E.1 places this work against the self-evolution methods it audits, the degradation and model-collapse literature that motivates the corruption half, and the statisticalevaluation work our controls build on.

## 3 A TRANSITION-LEVEL AUDIT, AND WHAT IT MUST CONTROL FOR

The ledger. Fix an evaluation set $\mathcal { P }$ and checkpoints $t = 0 , \ldots , T$ , with $t ~ = ~ 0$ the untrained base model. For each problem $p$ and checkpoint we draw k samples and record the solve rate $\hat { \pi } _ { t } ( p )$ . Discretizing into a binary state $s _ { t } ( p )$ gives a trajectory per problem, and every problem falls into exactly one of seven categories: stable-correct, stable-incorrect, learned, corrupted, recovered, transient, oscillating. These partition $\mathcal { P }$ , so counts need no normalization. The headline quantity is the corruption-to-learning ratio,

$$
\mathtt { C L R } = \frac { | \{ p : s _ { 0 } ( p ) = { \sf s o l v e d } , \ s _ { T } ( p ) = { \sf u n s o l v e d } \} | } { | \{ p : s _ { 0 } ( p ) = { \sf u n s o l v e d } , \ s _ { T } ( p ) = { \sf s o l v e d } \} | } ,\tag{1}
$$

problems destroyed per problem gained; CLR > 1 removes more capability than it adds, whatever happens to mean accuracy. Equation 1 is undefined when nothing is learned, as on AIME, where we report the counts instead. We define it on endpoints, and report trajectory-based counts alongside (Appendix C.1); they agree to within three problems.

Correctness as an estimator, not a draw. The obvious definition of $s _ { t } ( p )$ is a single greedy sample, which modern inference does not support (Section 4.1). We instead treat $\hat { \pi } _ { t } ( p )$ as an estimator of the model’s pass@1 probability and discretize with hysteresis:

$$
s _ { t } ( p ) = \left\{ \begin{array} { l l } { \mathsf { s o l v e d } } & { \hat { \pi } _ { t } ( p ) \ge \theta _ { \mathrm { h i } } , } \\ { \mathsf { u n s o l v e d } } & { \hat { \pi } _ { t } ( p ) < \theta _ { \mathrm { l o } } , } \\ { s _ { t - 1 } ( p ) } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \qquad \theta _ { \mathrm { h i } / \mathrm { l o } } = \frac { 1 } { 2 } \pm z \sqrt { \frac { 1 } { 4 k } } ,\tag{2}
$$

with $z = 2 , k = 1 2 8$ throughout, giving a band of [0.41, 0.59]. The half-width is the worst-case standard error of a binomial proportion at k samples, so a problem must move further than ordinary jitter before it is recorded as changing state. The band degenerates at $k \leq z ^ { 2 }$ , where it covers all of [0, 1] and no problem can change state.

The baseline state carries no hysteresis. Equation 2 is undefined at $t \ : = \ : 0$ , where there is no previous state to hold; we assign $s _ { 0 } ( p ) = { \mathsf { s o l v e d } }$ iff $\begin{array} { r } { \hat { \pi } _ { 0 } ( p ) \ge \frac { 1 } { 2 } } \end{array}$ . A problem whose baseline lies inside the band is therefore unprotected at that one checkpoint, which is why the floors of Table 10 are not zero on a set whose mean solve rate is 0.552 (Appendix C.1).

Expansion, with an explicit evidence threshold. Whether a newly solved problem represents new capability is a separate question that pass@1 cannot answer. Following Mayilvahanan et al. (2026), a problem that becomes pass@1-correct while the base model already solved it at least once in k samples has been sharpened; one the base model never reached has been expanded. We make the threshold m explicit:

$$
\mathrm { E R } _ { m } = \frac { | \{ p : \hat { \pi } _ { 0 } ( p ) = 0 \ \wedge \ k \hat { \pi } _ { T } ( p ) \geq m \} | } { | \{ p : \hat { \pi } _ { 0 } ( p ) = 0 \} | } .\tag{3}
$$

Prior work uses $m = 1$ implicitly; Section 4.2 shows that no threshold repairs this statistic and replaces it. Note what $\hat { \pi } _ { 0 } ( p ) = 0$ does not mean: no successes in 128 draws bounds the base pass@1 probability at $q \leq 0 . 0 2 3$ with 95% confidence, it does not establish $q = 0$ . We write “baseunreached at $k ^ { \prime \prime }$ rather than “unsolvable” throughout, and every claim below is about what a stated sampling budget reaches, not about the support of the base distribution.

A control for every statistic. Transitions are differences between noisy measurements, so some appear when nothing has changed. We measure how many by re-evaluating afrozen model at every checkpoint and computing every statistic exactly as for a trained one; each transition it reports is an artifact by construction. No claim is admitted unless its counts exceed this floor. That floor is measured separately per statistic and per benchmark, and, as Section 4.3 establishes, with an interval and a matched number of checkpoints.

Benchmarks. Corruption is observable only on problems the model already solves, expansion only on problems it does not reach; the two populations are disjoint, so we use three sets (Figure 4). MATH-500 (Lightman et al., 2024), subsampled to 200 problems by a fixed stratified draw over (subject, level), is thoroughly contaminated and hosts corruption rather than expansion. AIME 2025 and 2026 (60 problems) are the least contaminated sets available to us, and AIME 2026 postdates the backbone’s March 2025 cutoff (Yang et al., 2025) outright, so we report that half separately wherever a claim rests on it. The base model reaches 0.175–0.180 of AIME, leaving 22 problems it reaches at most five times in 1,408 pooled draws (only 10 of them never) and 8 that can be corrupted. A difficulty band of 1,163 problems, profiled from 10,999 held-out MATH problems (Hendrycks et al., 2021) at $k { = } 8 ,$ , is enriched near the decision boundary so that corruption is observable at all; its counts are therefore not a forgetting rate for a natural distribution, and it contains no base-unreached problems by construction (Appendix B.1).

We audit STaR (Zelikman et al., 2022) (filter by ground-truth answer), majority-vote self-training in the manner of TTRL (Zuo et al., 2025) (filter by majority vote over 8 samples, pooled by mathematical equivalence so $1 / 2$ and 0.5 do not split a vote), and distillation from gpt-oss-120b (OpenAI et al., 2025) as a positive control filtered by exactly STaR’s criterion. All arms share the backbone (Qwen3-8B, Yang et al., 2025), rank-32 LoRA, three rounds of 256 problems from one stream, and evaluation at k=128, T=0.8, top-p 0.95 plus one greedy sample per problem per checkpoint. The stream is verified disjoint from every evaluation set by identifier and by normalized text hash, and answers are compared by symbolic equivalence (Kydl´ıcekˇ , 2025). Appendices A.1 and A.2 give the configuration and the grading rules in full.

Table 1: Seven measurement failures, the conclusion each would have supported, and the control that caught it. Every row is measured on a run from this study rather than posited, and F7 is the failure that arises inside the correction for F2. Cost is the marginal spend on the control; three of the seven cost nothing at all, being re-analyses of records the study already holds, and two more would cost nothing in any study that reports more than one arm.
<table><tr><td>failure</td><td></td><td>would have concluded</td><td>control that catches it</td><td>cost</td></tr><tr><td>F1</td><td>single-decode ledger</td><td>“a frozen model has  $\mathbf { C L R } = 1 . 5 ^ { \prime \prime }$ </td><td>frozen floor + estimator</td><td>$27</td></tr><tr><td></td><td>F2 one-success ER threshold</td><td>1“maj.-vote expands 0.143 of  $\mathbf { A I M E } ^ { \prime \prime }$ </td><td>frozen floor for ER</td><td>$0</td></tr><tr><td></td><td>F3 fixed token cap</td><td>“distillation is the most destructive&quot;</td><td>log truncation per ckpt</td><td>$0</td></tr><tr><td></td><td>F4 underpowered ledger</td><td>“the two methods do not differ”</td><td>prior power analysis</td><td>$88</td></tr><tr><td></td><td>F5 single training seed</td><td>“maj.-vote corrupts less than  $\mathrm { S T a R } ^ { \prime \prime }$ </td><td> $\bar { \geq } 3$  seeds</td><td>$282</td></tr><tr><td></td><td>F6 underpowered probe</td><td>“self-training erodes refusal by  $1 0 \mathrm { \ p t s } ^ { \prime \prime }$ </td><td>size the probe first</td><td>$2</td></tr><tr><td></td><td>F7 a null measured once</td><td> $\mathrm { ^ { 6 6 } E R _ { 2 } }$  has a null of  $\scriptstyle \mathbf { Z e r o } ^ { \prime \prime }$ </td><td>every baseline is a replicate</td><td>$0</td></tr></table>

Two deviations from TTRL as published. TTRL is reinforcement learning with a majority-vote reward, whereas the arm labeled majority-vote SFT below is cross-entropy fine-tuning on the rationales that agree with the vote. The two differ in what happens to the samples that lose the vote, pushed down under the first and discarded under the second. We therefore also run a genuine policy-gradient arm (Section 5) and confine claims about a second method family to it. TTRL is also normally applied to the unlabeled target test set, which trains on the problems we score. That on-set configuration is reported throughout as an unmatched baseline, and every claim rests on the matched ladder of Section 5, whose training stream is disjoint from the evaluation set.

## 4 SEVEN WAYS TO MISMEASURE SELF-IMPROVEMENT

Each of the following is demonstrated on a run from this study and would, uncorrected, support a confident and incorrect conclusion; Table 1 summarizes them. The seventh arises inside the correction for the second, which is why we state it as a failure mode in its own right rather than folding it into the discussion.

## 4.1 F1. WITHOUT A FLOOR, A FROZEN MODEL APPEARS TO CORRUPT

Under a single greedy sample, an untrained model re-evaluated twice on MATH-500 shows 9 corruptions against 6 learnings, a CLR of 1.5. The solve-rate estimator on the same samples reports 1 and 0. The ratio is two single-digit counts, so we report the rate alongside it: 7.5% [4.6, 12.0] of problems change verdict, itself unstable across three independent executions of the same two-pass comparison at 7.5%, 8.0% and 4.5%.

The mechanism, tested rather than cited. These flips are conventionally attributed to batching (He & Lab, 2025) without a direct test. One is available: sample the same 200 problems greedily twice over, once with 64 requests in flight and once strictly serialized. Serializing cuts verdict flips from 16/200 to 4/200 (8.0% [5.0, 12.6] against 2.0% [0.8, 5.0], Fisher $p = 0 . 0 0 5 )$ and extractedanswer changes from 25 to 5 $\dot { ( p < 1 0 ^ { - 4 } ) }$ . Batching is therefore the dominant cause, but not the only one. A fully serialized frozen model still flips 2% of its greedy verdicts, enough to fabricate a CLR from nothing on a small evaluation set. No care with our own request pattern removes it: whatever remains is server-side batching across tenants or nondeterminism below the API. A single-decode ledger cannot be repaired by decoding more carefully; it has to be replaced by an estimator. On the 1,163-problem band the gap is wider still, 18.9% of problems changing state under greedy decoding against 0.8% under the estimator (Figure 7, Table 10).

## 4.2 F2. NO THRESHOLD REPAIRS THE EXPANSION STATISTIC

Equation 3 at $m = 1$ requires the base model to produce zero correct samples and the evolved model at least one. That is a comparison of two binomial estimates through asymmetric thresholds, and its null is not zero. On the frozen AIME control it gives $\mathrm { E R _ { 1 } } = 7 / \mathrm { 2 5 } = \mathbf { 0 . 2 8 0 }$ : a model that did not train “expands” more than a quarter of the problems it had not reached, and pooled over all 110 frozen comparisons the rate is 0.176. On the matched ladder of Section 5, STaR scores $\mathrm { E R _ { 1 } } = 0 . 3 2 0$ and majority-vote self-training 0.364, 0.167 and 0.192 across three seeds. Those values straddle the pooled null and lie inside the 0.000–0.364 range a frozen model produces across its own evaluations. The correct reading is not that they are nil, but that $\mathrm { E R _ { 1 } }$ overstates them by about a factor of two and cannot separate them from a model that never trained. It does separate distillation, at 0.545–0.667.

The obvious repair does not work. All seven frozen “expansions” are exactly one correct sample in 128, so $m = 2$ removes every one and its null on that comparison is exactly zero. The appearance rests on a single realization over 25 problems. Every arm’s checkpoint 0 is an independent $k { = } 1 2 8$ evaluation of the same untrained model, and with the ladder in place AIME carries eleven of them, giving 110 ordered pairs at no cost beyond experiments already run. Across those pairs $\mathrm { E R _ { 2 } }$ pools to 0.058 [0.038, 0.078] and reaches 0.136 on the worst pair. The value we measure for majority-vote self-training, $1 / 2 1 \overset { \cdot } { = } 0 . 0 4 8$ , is therefore indistinguishable from its own null (Wilson [0.008, 0.23]) rather than below it. Raising the threshold does not rescue the statistic: $\mathrm { E R _ { 3 } }$ has a null of 0.023 [0.009, 0.038] and only $\mathrm { E R _ { 5 } }$ approaches zero, at 0.003 [0.000, 0.009]; on MATH-500 the $m { = } 2$ null is 0.128. The threshold that appeared to fix the statistic was fitted to the noise it was fitted on, and m is benchmark- and k-specific in exactly the way the statistic it corrects is.

What does work is to stop thresholding. Pool every independent evaluation of the untrained model into one baseline: eleven on AIME, 1,408 draws per problem, one denominator shared by all arms rather than a noise-selected set per arm. Then test each problem’s endpoint count against it with a one-sided Fisher exact test under FDR control. The evidence threshold disappears and the arms become comparable. Four choices remain, the error rate, the one-sidedness, the pool size and the strata, and Appendix D.1 varies each (Table 16). One belongs here, because a falsediscovery-rate procedure is adaptive: an arm containing many true effects raises its own admissible cutoff, so detection counts need not be comparable across arms. Here they are: the cutoffs actually attained are 0.0095 for distillation’s 36 detections and 0.0094 for majority-vote self-training’s 19, and every comparison below is unchanged under Bonferroni, under one joint procedure across arms and problems, and at fixed thresholds. The null is measured the same way as everything else here: hold out one base evaluation, call it the “after”, and run the identical test. It returns zero detections on all eleven held-out replicates, at every base rate (Figure 2, Table 19); zero is itself an estimate, and the Clopper–Pearson bounds are 0.24 per replicate and 0.005 per test.

Under that test the dissociation is sharper than any thresholded version. On the matched ladder, distillation detects 36 of 60 problems and majority-vote self-training 19; of the 21 they disagree about, 19 favor distillation, a paired exact $p = \overset { \cdot } { 2 } . 2 \overset { \cdot } { \times } 1 0 ^ { - 4 }$ against the $p = 0 . 0 0 1 5$ an unpaired test on mismatched denominators returns on the same data. Restricted to the 22 problems whose pooled base rate is at most 5/1,408, distillation detects 8–11 per seed and the self-training arms $0 { - } 2 ,$ the frozen control 0 on all eleven replicates. We combine the per-seed comparisons rather than pooling detections across seeds, which is the operation F5 says three seeds cannot support: $p = 2 . 0 \times 1 0 ^ { - 6 }$ against majority-vote SFT, 0.0078 against STaR and 0.0019 against the policy-gradient arm. Those 22 are not all expansion candidates, however. Only 10 have a pooled base rate of exactly zero, and there the union over seeds is 5 against 2, which is not significant $( p = 0 . 3 5 )$ ). Neither is the comparison on the ten low-base problems of the uncontaminated 2026 half (6 against 2, $p = 0 . 1 7 )$ Section 5 keeps the two populations apart.

The small non-zero counts are stated rather than rounded away: applied to a stream of problems it can solve, self-training does occasionally raise a rarely-reached problem above the noise, at roughly a fifth of the teacher’s rate. The comparison is in any case conservative in the teacher’s disfavor, since the distillation arms truncate 36–46% of completions even at an 8,192-token cap (Appendix D.1).

## 4.3 THE REMAINING FAILURES

Three of the remaining five are stated in summary and developed in Appendices C.3 and D.4. F3: a fixed token cap scores the most effective arm as the most destructive, because the distillation student adopts its teacher’s verbose style and truncation rises from 16.5% to 65.3%; that arm’s corruption counts are excluded from the capability analysis, since severe generation-length drift under a fixed cap confounds capability loss with stylistic verbosity. F4: transition metrics carry an order of magnitude less power than accuracy, and a 112-problem pilot yields 24 events against the

base-model successes in 1,408 pooled draws

![](images/adba1e7d7be7e5ec4152a65db6a5436ba9b1424df9635599efa628fd97e4fac5.jpg)

![](images/856f7521a6dfa232ef5644d74a1b2cc49932d624e30f29b88cdb90ae5194682d.jpg)  
Figure 2: The expansion statistic needs a null, and one measurement of a null is not one. $L e f t { \mathrm { : } }$ $\mathrm { E R } _ { m }$ against the evidence threshold. $\mathrm { A t } \ m { = } 1$ , the criterion in use, a single frozen comparison sits at 0.280, above majority-vote self-training. The shaded band is the range over all 110 frozen comparisons the records contain, and it does not approach zero until $m { = } 5 { : }$ at m=2, the natural repair, the pooled null is 0.058 and the value it was meant to certify as null is 0.048. Right: the threshold-free test, by pooled base rate. Detections against a shared 1,408-draw baseline under FDR control; the frozen null is zero in every stratum on all eleven held-out replicates.

∼249 per arm the comparison needs. F6: a 10-prompt refusal probe yields a 10-point safety finding that 50 prompts dissolve.

F5. The comparison between methods is not stable across training seeds. Two further seeds on a matched 175-problem subsample invert the single-seed comparison (Table 13). STaR is stable (CLR of 0.70, 0.74, 0.45, accuracy up +5.6 to +6.9 points on every seed) and majority-vote selftraining is not: 0.55, 1.53, 1.33, with accuracy changes of $+ 4 . 0 , \ - 2 . 8$ and −1.0. A powered comparison at seed 0 finds no difference $( p = 0 . 4 0 6 )$ ; pooling the two matched replicates for a Fisher test gives $p = 0 . 0 0 7 6$ , and we do not report it, because pooling seeds as a fixed effect is what the same paragraph says three seeds cannot support. The defensible reading is narrower and more useful: a majority-vote method’s outcome is not determined by its design alone. Same method, same data, same budget, run twice, landing on either side of $\mathbf { C L R } = 1$

F7. A null measured once is a point estimate, not a parameter. Everything above compares a measured quantity to a measured null, and a null estimated once carries the same sampling error one level up. This design admits three instances: $\mathrm { E R _ { 2 } } ^ { \prime } \mathrm { s }$ null of $0 / 2 5$ on a single frozen pair becomes 0.058 [0.038, 0.078] over 110 frozen comparisons, the band’s corruption floor of 5 becomes a median of 8 once the control is given as many checkpoints as the arms, and the 0.188 envelope on frozen solve-rate movement becomes 0.234 over twenty pairs. None moves a headline, but the first removes the correction F2 would otherwise license. The remedy needs no additional experiments, and generalizes: every arm’s checkpoint 0 is an independent evaluation of the untrained model. It needs a number attached, and ours is that four is not enough: over every subset of our eleven baselines the ER<sub>2</sub> null still ranges 0.022–0.098 at $G = 4 ,$ , enough to certify a null that is not one, and only at $G = 9$ does it hold within ±0.02 of the value eleven give.

## 5 WHAT THE CONTROLLED AUDIT SHOWS

A matched comparison. Compared under their native protocols, self-training and distillation differ in more than the correctness filter: TTRL as published is applied to the 60 AIME problems themselves and retains 50 examples per round, where distillation draws from a disjoint stream and retains 247, a 4.9× volume gap on top of a stream mismatch. To eliminate the volume discrepancy and isolate the filter, we evaluate a ladder on one shared stream at matched volume, three seeds, so that only the correctness filter varies: the model’s own majority vote, the ground-truth answer, an external teacher (Table 2), together with a policy-gradient arm using the same vote as a reward, as TTRL specifies.

Table 2: The matched AIME ladder: one stream, one retained volume, one evaluation, and only the source of the correctness filter changing. Detections are problems whose solve rate rises significantly against a pooled 1,408-draw baseline of the untrained model, pooled over 11 independent evaluations (per-problem exact test, FDR 0.05), split by how often that baseline reaches the problem at all. The frozen null is zero in every column. Expansion is only interpretable in the two low-base columns: there the self-training arms reach 0–2 problems and the external teacher 8–11, on every seed. Majority-vote accuracy also moves +3.3, −1.5 and +0.7 points across its three seeds while the teacher’s moves +13 to +16: the instability of F5, on a second benchmark and under the matched protocol.
<table><tr><td>filter</td><td>update</td><td>seed</td><td></td><td>accuracy</td><td>∆ (pts) retained</td><td>base 0</td><td>base 1-5</td><td></td><td>base &gt; 5</td></tr><tr><td>majority vote</td><td>SFT</td><td>0</td><td>0.180 → 0.213</td><td></td><td>↑3.3</td><td>250</td><td>0</td><td>2</td><td>17</td></tr><tr><td rowspan="5">ground truth external teacher†</td><td></td><td>1</td><td>0.178 → 0.163</td><td></td><td>↓1.5</td><td>251</td><td>1</td><td>0</td><td>3</td></tr><tr><td></td><td>2</td><td>0.179 → 0.186</td><td></td><td>↑0.7</td><td>251</td><td>0</td><td>0</td><td>4</td></tr><tr><td>SFT</td><td>0</td><td>0.177 → 0.195</td><td></td><td>↑1.8</td><td>220</td><td>1</td><td>1</td><td>9</td></tr><tr><td>SFT</td><td>0</td><td>0.180 → 0.310</td><td></td><td>↑13.0</td><td>247</td><td>3</td><td>7</td><td>26</td></tr><tr><td></td><td>1</td><td>0.177 → 0.324</td><td></td><td>↑14.7</td><td>248</td><td>4</td><td>7</td><td>26</td></tr><tr><td rowspan="2">majority vote</td><td></td><td>2</td><td>0.177 → 0.335</td><td></td><td>↑15.8</td><td>246</td><td>2</td><td>6</td><td>25</td></tr><tr><td>policy grad.</td><td>0</td><td>0.176 → 0.192</td><td></td><td>↑1.6</td><td>1984</td><td>0</td><td>1</td><td>11</td></tr><tr><td>frozen control</td><td>none</td><td></td><td></td><td></td><td></td><td></td><td>0</td><td>0</td><td>0</td></tr></table>

<sup>†</sup>positive control, not a self-evolution method. The frozen null is the same test with one base evaluation held out as the “after”, over all 11 such replicates. <sup>‡</sup>this arm collapsed into unbounded repetition on all three seeds and is 76.4% truncated at the checkpoint shown; its detections are retained as a floor, since truncation can only suppress them, and its transition counts are excluded (Appendix A.3). Retained is the first round’s count (Table 3 gives all three): training examples for the SFT arms, rollouts for the policy-gradient arm, of which 135, 143 and 217 of ∼250 carried any gradient.

The ladder yields three results the unmatched comparison cannot. First, the dissociation survives matching: on the low-base pool the teacher reaches 8–11 problems per seed and every self-training arm 0–2. On the ten problems of that pool the base model never reaches at all, however, the split over seeds is 5 against 2 and is not significant, so what we claim is a low-base result and not an expansion result. Second, the dissociation is not an artifact of the teacher’s larger gain. A logistic model on the per-problem counts, with terms for arm, stratum and their interaction, gives the teacher a further β = 1.91 [1.25, 2.56] on low-base problems beyond its overall lift $( p < 1 0 ^ { - 8 } )$ , where majority-vote SFT gets 0.21 [−0.60, 1.03] and the policy-gradient arm −0.49. Uniform multiplicative improvement accounts for the self-training arms and not for the teacher (Appendix D.1). Third, the seed instability of F5 replicates on a second benchmark: majority-vote accuracy moves +3.3, −1.5 and +0.7 points across seeds, one of three net negative, while the teacher’s moves +13 to +16 on all three.

Sharpening, at its true evidential weight. Across every self-training arm, all problems newly solved at pass@1 were already reachable at pass@k; the informative evidence is nine problems rather than four hundred, since the band excludes base-unreached problems by construction (Appendix C.1). The threshold-free test carries that claim at scale, and its null result is a bound rather than an absence. With eleven pooled baselines the test has 80% power against a true post-training rate of 0.024, and no self-training arm produced more than one correct sample in 128 on any base-0 problem. None lifted such a problem above roughly a 2.4% pass rate.

Reinforcement rather than filtering, and an arm outside the capability analysis. Both selftraining arms above are rejection-sampling SFT, whereas TTRL as published pushes the losing samples down. A policy-gradient arm on the band gives 19 learned against 9 corrupted (CLR = 0.47) and behaves; on AIME it does not survive at any of the three seeds F5 demands. The runs see identical data in identical order and agree at round one to within a percentage point: the vote is 77.0–77.9% correct, and 135–136 groups of roughly 250 carry any gradient. They then diverge completely, collapsing into unbounded repetition after one update, two, and three, at 100%, 95.6% and 76.4% truncation. Every seed ends there, so the collapse belongs to the configuration and the round at which it arrives does not; the arm’s transition counts are excluded under F3 exactly as distillation’s are. What it is reported for is its own diagnostic. On seed 2, between the update that damaged the model and the one that finished it, the majority vote became more accurate and more unanimous (77.9 to 85.1%, share 0.78 to 0.86, over 248 intact majorities) while the evaluation went from 47% truncated to 96%. A supervision signal that reads healthy while the model it supervises is dying is the failure this paper is about (Table 5, Appendix A.3).

What is removed is large, not a threshold flicker. On the band both methods destroy problems solved at baseline, 106 for STaR and 88 for majority-vote SFT, while STaR’s mean accuracy rises by 5.6 points on that same run. Aggregate improvement and substantial destruction are the same run here, not alternatives. The floor needs the treatment the expansion null needed. The conventional two-checkpoint control gives 4 learned and 5 corrupted, where an arm has four checkpoints. A design-matched four-checkpoint no-op, built from the band’s five independent evaluations, gives a median of 10 and 8 over 120 orderings (Figure 8), so the arms clear it by 11–13× rather than 18– 21×. Nor is what they destroy a threshold flicker. Measured over all 20 frozen pairs, a model that never trained never moved a solve rate by more than 0.234. Yet 58 of STaR’s 106 corruptions and 60 of majority-vote SFT’s 88 exceed that: more than half, and more still under a stated quantile in place of the maximum, which is the less conservative choice (Table 23, Figure 10).

Two things we can only bound. The discretization rule moves the counts without moving the conclusion, but under a pure endpoint rule the two self-training methods coincide exactly and at z = 0 their order reverses, so we rest nothing on the between-method comparison (Appendix C.2). And the refusal probe cannot resolve what it was built to measure: three replicates read −0.015, −0.115 and −0.005, the middle one significant and replicated by neither neighbor, so we report a bound (any change here is smaller than about eleven points) rather than a null result (Appendix D.4).

## 6 DISCUSSION

A reporting standard. Seven practices follow, each of which caught an inverted result here: report a no-op control for every statistic and with an interval; match the control’s design to the arm’s; define correctness through a solve-rate estimator rather than a single decode; prefer a per-problem test against a pooled baseline to any thresholded expansion statistic; log generation length per checkpoint; size transition comparisons from a prior power analysis; and report at least three training seeds, scored on the same problems. As our own policy-gradient arm shows, the third seed can be the one that changes the finding. Five need no new experiments, being re-analyses of records a careful study already holds, but none of them on four frozen replicates, which is not enough for the null they would be read against. We report roughly a dozen p-values across seven failure modes and correct within each family rather than across them, which is stated here rather than left to be noticed (Appendix D.1).

Limitations. Three rounds and roughly 270 optimizer steps of rank-32 LoRA, far short of the schedules we position against: what we establish is that this regime does not expand, not that selftraining cannot. Expansion rests on 22 low-base problems on one benchmark, and on the 10 the base model never reaches the comparison is not significant. Corruption rests on a band drawn from MATH training problems almost certainly in the backbone’s pretraining data and built to make corruption observable. One backbone family carries every claim (Appendix E.3).

Conclusion. Transition-level auditing is the right response to the fact that accuracy conceals what a self-improving model gains and what it destroys. It is also a measurement problem in its own right: we isolate seven ways to get it wrong, two of them standard practice and one inside the correction for another. The expansion statistic in current use assigns a model that never trained a rate of 0.280; the natural threshold repair takes that null to 0.058 rather than zero, and calculating it would have returned 0.044 and certified the same false conclusion. The replicates that bound a null already exist in any multi-arm study, provided there are enough of them.

## ETHICS STATEMENT

This study evaluates language models on public mathematics benchmarks and releases only modelgenerated text about those problems; it involves no human subjects and no personal data. Two aspects warrant comment. The first is the out-of-domain refusal probe. Its prompts are deliberately mild and non-operational, asking for the shape of a harmful action rather than the means to carry one out, because their purpose is to elicit a refusal and not anything usable. The probe is released so that its result can be re-scored under a rule other than ours. What that result supports is a bound and not a clearance: we decline to claim that self-training causes no safety erosion here, and the data are consistent both with no effect and with one of practical size (Appendix D.4). Nothing in this paper should be read as evidence that a self-training method is safe to deploy without supervision. The second is that this paper reports, for every failure mode it documents, both the conclusion an uncontrolled analysis supports and the marginal cost of the control that rules it out. Presenting both is deliberate: a literature in which nulls are never re-measured is precisely the failure this paper is about. The distillation control uses an openly released teacher within the terms of its license, and Appendix A.4 reports the study’s compute and monetary cost in full.

## REPRODUCIBILITY STATEMENT

Records for every run are released at https://github.com/chengxuphd/ phantom-gains, reduced to per-problem counts: for each checkpoint, sampling mode and problem, the samples drawn, the samples correct, the completions truncated, and the tokens and cost they consumed. Every number and figure in this paper is a function of those counts and is recomputed from them by released code; none is transcribed by hand. The per-sample records they are reduced from — one JSON line per generation, carrying the generated text and the extracted answer — total roughly 15 GB, are impractical to distribute and matter only for re-grading under a different verifier; they are available on request, and the reduction that produced the released files is released with them. To support re-scoring under alternative refusal rules, the out-of-domain refusal probe similarly records individual outputs per generation; those records are what establish that the probe’s reading does not reproduce across replicates (Appendix D.4). The evaluation subsets are released with the code, including the constructed difficulty band and the selection profile used to build it, together with the noise-floor runs for each benchmark. Section 3 and Appendix A.1 state the training and sampling configuration in full, including temperature, top-p, token caps, learning rate, adapter rank and per-round training-example counts. The audited methods are reimplementations against Tinker<sup>1</sup>, a hosted LoRA training and sampling service, and are released; the policy-gradient arm uses that service’s importance-sampling objective on group-centered advantages, with the group mean as baseline, no clipping and no off-policy correction beyond a single epoch per round.

## USE OF LARGE LANGUAGE MODELS

An LLM-based coding assistant was used to implement the evaluation harness and to run and monitor experiments, and assist with language editing and proofreading. All experimental design deci sions, all interpretation of results, and all claims in this paper are the authors’ own. Every number reported is computed by released code from released per-sample records; none was produced or transcribed by a language model.

## REFERENCES

Emre Can Acikgoz, Cheng Qian, Heng Ji, Dilek Hakkani-Tur, and Gokhan Tur. Self-improving llm¨ agents at test-time, 2025. URL https://arxiv.org/abs/2510.07841.

Sina Alemohammad, Josue Casco-Rodriguez, Lorenzo Luzi, Ahmed Imtiaz Humayun, Hossein Babaei, Daniel LeJeune, Ali Siahkoohi, and Richard Baraniuk. Self-consuming generative models go mad. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 53581–53608, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ ebc042e767de551803ccfcc45e2454f5-Paper-Conference.pdf.

Huan ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Qihan Ren, Yiran Wu, Hongru WANG, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/forum?id=CTr3bovS5F. Survey Certification.

Dhruv Atreja. Alas: Autonomous learning agent for self-updating language models, 2025. URL https://arxiv.org/abs/2508.15805.

Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and John Patrick Cunningham. LoRA learns less and forgets less. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id= aloEru2qCG. Featured Certification.

Lawrence D. Brown, T. Tony Cai, and Anirban DasGupta. Interval Estimation for a Binomial Proportion. Statistical Science, 16(2):101 – 133, 2001. doi: 10.1214/ss/1009213286. URL https://doi.org/10.1214/ss/1009213286.

Lai Chen, Yao Zhu, Xin Ye, Peng Lin, Xiangyang Ji, and Xiang Tian. Enhancing reasoning in large language models via entropy-aware self-evolution, 2026. URL https://openreview. net/forum?id=nXENWUSRMw.

Lili Chen, Mihir Prabhudesai, Katerina Fragkiadaki, Hao Liu, and Deepak Pathak. Self-questioning language models, 2025. URL https://arxiv.org/abs/2508.03682.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam, Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob Mc-Grew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.03374.

Zixiang Chen, Yihe Deng, Huizhuo Yuan, Kaixuan Ji, and Quanquan Gu. Self-play fine-tuning converts weak language models to strong language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 6621–6642. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/chen24j.html.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv. org/abs/2110.14168.

Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, and Ning Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://arxiv.org/abs/2505.22617.

Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping, 2020. URL https://arxiv.org/abs/2002.06305.

B. Efron. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1): 1 – 26, 1979. doi: 10.1214/aos/1176344552. URL https://doi.org/10.1214/aos/ 1176344552.

Jinyuan Fang, Yanwen Peng, Xi Zhang, Yingxu Wang, Xinhao Yi, Guibin Zhang, Yi Xu, Bin Wu, Siwei Liu, Zihao Li, Zhaochun Ren, Nikos Aletras, Xi Wang, Han Zhou, and Zaiqiao Meng. A comprehensive survey of self-evolving ai agents: A new paradigm bridging foundation models and lifelong agentic systems, 2025. URL https://arxiv.org/abs/2508.07407.

R. A. Fisher. On the interpretation of χ<sup>2</sup> from contingency tables, and the calculation of p. Journal of the Royal Statistical Society, 85(1):87–94, 1922. ISSN 09528385. URL http://www.jstor. org/stable/2340521.

Aryo Pradipta Gema, Joshua Ong Jun Leang, Giwon Hong, Alessio Devoto, Alberto Carlo Maria Mancino, Rohit Saxena, Xuanli He, Yu Zhao, Xiaotang Du, Mohammad Reza Ghasemi Madani, Claire Barale, Robert McHardy, Joshua Harris, Jean Kaddour, Emile Van Krieken, and Pasquale Minervini. Are we done with MMLU? In Luis Chiruzzo, Alan Ritter, and Lu Wang (eds.), Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 5069–5096, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025.naacl-long.262. URL https: //aclanthology.org/2025.naacl-long.262/.

Matthias Gerstgrasser, Rylan Schaeffer, Apratim Dey, Rafael Rafailov, Tomasz Korbak, Henry Sleight, Rajashree Agrawal, John Hughes, Dhruv Bhandarkar Pai, Andrey Gromov, Dan Roberts, Diyi Yang, David L. Donoho, and Sanmi Koyejo. Is model collapse inevitable? breaking the curse of recursion by accumulating real and synthetic data. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=5B2K4LRgmz.

Halil Alperen Gozeten, Muhammed Emrullah Ildiz, Xuechen Zhang, Mahdi Soltanolkotabi, Marco Mondelli, and Samet Oymak. Test-time training provably improves transformers as in-context learners. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 20266–20295. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press v267/gozeten25a.html.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Honghui Ding, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jingchang Chen, Jingyang Yuan, Jinhao Tu, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaichao You, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li, Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu,

Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha, Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen Zhang. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645 (8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/s41586-025-09422-z. URL http://dx.doi.org/10.1038/s41586-025-09422-z.

Mohsen Hariri, Amirhossein Samandar, Michael Hinczewski, and Vipin Chaudhary. Don’t pass@k: A bayesian framework for large language model evaluation. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 148539–148579, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ f04edfc65463d020629673a4bc4c58e7-Paper-Conference.pdf.

Horace He and Thinking Machines Lab. Defeating nondeterminism in llm inference. Thinking Machines Lab: Connectionism, 2025. doi: 10. 64434/tml.20250910. URL https://thinkingmachines.ai/blog/ defeating-nondeterminism-in-llm-inference/.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. In J. Vanschoren and S. Yeung (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks, volume 1, 2021. URL https: //datasets-benchmarks-proceedings.neurips.cc/paper\_files/paper/ 2021/file/be83ab3ecd0db773eb2dc1b0a17836a1-Paper-round2.pdf.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network, 2015. URL https://arxiv.org/abs/1503.02531.

Arian Hosseini, Xingdi Yuan, Nikolay Malkin, Aaron Courville, Alessandro Sordoni, and Rishabh Agarwal. V-STar: Training verifiers for self-taught reasoners. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=stmqBSW2dV.

Cheng-Yu Hsieh, Chun-Liang Li, Chih-kuan Yeh, Hootan Nakhost, Yasuhisa Fujii, Alex Ratner, Ranjay Krishna, Chen-Yu Lee, and Tomas Pfister. Distilling step-by-step! outperforming larger language models with less training data and smaller model sizes. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Findings of the Association for Computational Linguistics: ACL 2023, pp. 8003–8017, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.findings-acl.507. URL https://aclanthology.org/2023. findings-acl.507/.

Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.

Audrey Huang, Adam Block, Dylan Foster, Dhruv Rohatgi, Cyril Zhang, Max Simchowitz, Jordan Ash, and Akshay Krishnamurthy. Self-improvement in language models: The sharpening mechanism. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 76687–76739, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file bee8c2bc757f6bbc3efd7cf1b979f0c9-Paper-Conference.pdf.

Chengsong Huang, Wenhao Yu, Xiaoyang Wang, Hongming Zhang, Zongxia Li, Ruosen Li, Jiaxin Huang, Haitao Mi, and Dong Yu. R-zero: Self-evolving reasoning llm from zero data. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 130770–130790, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ d49b9aacebda61051166335af6fd3061-Paper-Conference.pdf.

Jiaxin Huang, Shixiang Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 1051–1068, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.67. URL https://aclanthology.org/2023. emnlp-main.67/.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Yu, Xinying Song, and Denny Zhou. Large language models cannot self-correct reasoning yet. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 32808–32824, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 8b4add8b0aa8749d80a34ca5d941c355-Paper-Conference.pdf.

James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, and Raia Hadsell. Overcoming catastrophic forgetting in neural networks. Proceedings of the National Academy of Sciences, 114(13):3521– 3526, 2017. doi: 10.1073/pnas.1611835114. URL https://www.pnas.org/doi/abs/ 10.1073/pnas.1611835114.

Hynek Kydl´ıcek. Math-verify: Math verification library. GitHub repository, 2025. URL ˇ https: //github.com/huggingface/Math-Verify.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 39578–39601, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ aca97732e30bcf1303bc22ac3924fd16-Paper-Conference.pdf.

Hongxiang Lin, Zhirui Kuai, Erpeng Xue, and Lei Wang. Detecting and mitigating the correctanswer extinction window in test-time reinforcement learning with majority voting, 2026. URL https://arxiv.org/abs/2605.19444.

Mingjie Liu, Shizhe Diao, Ximing Lu, Jian Hu, Xin Dong, Yejin Choi, Jan Kautz, and Yi Dong. Prorl: Prolonged reinforcement learning expands reasoning boundaries in large language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 17998–18031. Curran Associates, Inc., 2025a. doi: 10.52202/ 085713-0608. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/1a22b912945fb7c0bdd079e792b31b6f-Paper-Conference.pdf.

Zichen Liu, Changyu Chen, Wenjun Li, Penghui Qi, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Understanding r1-zero-like training: A critical perspective. In Second Conference on Language Modeling, 2025b. URL https://openreview.net/forum?id=5PAF7PAY2Y.

Jianqiao Lu, Wanjun Zhong, Wenyong Huang, Yufei Wang, Qi Zhu, Fei Mi, Baojun Wang, Weichao Wang, Xingshan Zeng, Lifeng Shang, Xin Jiang, and Qun Liu. Self: Self-evolution with language feedback, 2024. URL https://arxiv.org/abs/2310.00533.

Yun Luo, Zhen Yang, Fandong Meng, Yafu Li, Jie Zhou, and Yue Zhang. An empirical study of catastrophic forgetting in large language models during continual fine-tuning. IEEE Transactions on Audio, Speech and Language Processing, 33:3776–3786, 2025. doi: 10.1109/TASLPRO.2025. 3606231. URL https://doi.org/10.1109/TASLPRO.2025.3606231.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. Selfrefine: Iterative refinement with self-feedback. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 46534–46594. Curran Associates, Inc., 2023. doi: 10.52202/

075280-2019. URL https://proceedings.neurips.cc/paper\_files/paper/ 2023/file/91edff07232fb1b55a505a9e9f6c0ff3-Paper-Conference.pdf.

Prasanna Mayilvahanan, Ricardo Dominguez-Olmedo, Thaddaus Wiedemer, and Wieland Bren-¨ del. Math-beyond: A benchmark for rl to expand beyond the base model. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 68471–68489, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 6fdf57c71bc1f1ee29014b8dc52e723f-Paper-Conference.pdf.

Evan Miller. Adding error bars to evals: A statistical approach to language model evaluations, 2024. URL https://arxiv.org/abs/2411.00640.

Mohammad Mahdi Moradi, Hossam Amer, Sudhir Mudur, Weiwei Zhang, Yang Liu, and Walid Ahmed. Continuous self-improvement of large language models by test-time training with verifier-driven sample selection. In AI That Keeps Up: NeurIPS 2025 Workshop on Continual and Compatible Foundation Model Updates, 2025. URL https://openreview.net/forum? id=6ahliSpvQ0.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. s1: Simple test-\` time scaling. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 20275–20321, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main.1025. URL https: //aclanthology.org/2025.emnlp-main.1025/.

Sagnik Mukherjee, Lifan Yuan, Dilek Hakkani-Tur, and Hao Peng. Reinforcement learning finetunes small subnetworks in large language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 132119–132138. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-4399. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/bf235a1d6780afd979f2f81676f43413-Paper-Conference.pdf.

OpenAI, :, Sandhini Agarwal, Lama Ahmad, Jason Ai, Sam Altman, Andy Applebaum, Edwin Arbus, Rahul K. Arora, Yu Bai, Bowen Baker, Haiming Bao, Boaz Barak, Ally Bennett, Tyler Bertao, Nivedita Brett, Eugene Brevdo, Greg Brockman, Sebastien Bubeck, Che Chang, Kai Chen, Mark Chen, Enoch Cheung, Aidan Clark, Dan Cook, Marat Dukhan, Casey Dvorak, Kevin Fives, Vlad Fomenko, Timur Garipov, Kristian Georgiev, Mia Glaese, Tarun Gogineni, Adam Goucher, Lukas Gross, Katia Gil Guzman, John Hallman, Jackie Hehir, Johannes Heidecke, Alec Helyar, Haitang Hu, Romain Huet, Jacob Huh, Saachi Jain, Zach Johnson, Chris Koch, Irina Kofman, Dominik Kundel, Jason Kwon, Volodymyr Kyrylov, Elaine Ya Le, Guillaume Leclerc, James Park Lennon, Scott Lessans, Mario Lezcano-Casado, Yuanzhi Li, Zhuohan Li, Ji Lin, Jordan Liss, Lily, Liu, Jiancheng Liu, Kevin Lu, Chris Lu, Zoran Martinovic, Lindsay McCallum, Josh McGrath, Scott McKinney, Aidan McLaughlin, Song Mei, Steve Mostovoy, Tong Mu, Gideon Myles, Alexander Neitz, Alex Nichol, Jakub Pachocki, Alex Paino, Dana Palmie, Ashley Pantuliano, Giambattista Parascandolo, Jongsoo Park, Leher Pathak, Carolina Paz, Ludovic Peran, Dmitry Pimenov, Michelle Pokrass, Elizabeth Proehl, Huida Qiu, Gaby Raila, Filippo Raso, Hongyu Ren, Kimmy Richardson, David Robinson, Bob Rotsted, Hadi Salman, Suvansh Sanjeev, Max Schwarzer, D. Sculley, Harshit Sikchi, Kendal Simon, Karan Singhal, Yang Song, Dane Stuckey, Zhiqing Sun, Philippe Tillet, Sam Toizer, Foivos Tsimpourlas, Nikhil Vyas, Eric Wallace, Xin Wang, Miles Wang, Olivia Watkins, Kevin Weil, Amy Wendling, Kevin Whinnery, Cedric Whitney, Hannah Wong, Lin Yang, Yu Yang, Michihiro Yasunaga, Kristen Ying, Wojciech Zaremba, Wenting Zhan, Cyril Zhang, Brian Zhang, Eddie Zhang, and Shengjia Zhao. gpt-oss-120b & gpt-oss-20b model card, 2025. URL https://arxiv.org/abs/2508.10925.

Xiangyu Qi, Yi Zeng, Tinghao Xie, Pin-Yu Chen, Ruoxi Jia, Prateek Mittal, and Peter Henderson. Fine-tuning aligned language models compromises safety, even when users do not intend to! In B. Kim, Y. Yue, S. Chaudhuri, K. Fragkiadaki, M. Khan, and Y. Sun (eds.), International Conference on Learning Representations, volume 2024, pp. 30988–31043, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/file/ 83b7da3ed13f06c13ce82235c8eedf35-Paper-Conference.pdf.

Zhenting Qi, Susanna Maria Baby, Stefanie Anna Baby, Kan Yuan, Andrew Tomkins, Tu Vu, Da-Cheng Juan, and Cyrus Rashtchian. On the generalization gap in self-evolving language model reasoning. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=mnUidYi5qO.

Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, and Luke Zettlemoyer. Spurious rewards: Rethinking training signals in RLVR. In Fortythird International Conference on Machine Learning, 2026a. URL https://openreview. net/forum?id=tqTNOpkP5j.

Shuai Shao, Qihan Ren, Dongrui Liu, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng Zhang, Weinan Zhang, and Jing Shao. Your agent may misevolve: Emergent risks in self-evolving llm agents. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 99728– 99793, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/a24cd16bc361afa78e57d31d34f3d936-Paper-Conference.pdf.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv.org/abs/2402. 03300.

Ilia Shumailov, Zakhar Shumaylov, Yiren Zhao, Nicolas Papernot, Ross Anderson, and Yarin Gal. Ai models collapse when trained on recursively generated data. Nature, 631:755–759, 2024. doi: 10.1038/s41586-024-07566-y. URL https://www.nature.com/articles/ s41586-024-07566-y.

Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaehoon Lee, Kelvin Xu, Aaron T Parisi, Abhishek Kumar, Alexander A Alemi, Alex Rizkowsky, Azade Nova, Ben Adlam, Bernd Bohnet, Gamaleldin Fathy Elsayed, Hanie Sedghi, Igor Mordatch, Isabelle Simpson, Izzeddin Gur, Jasper Snoek, Jeffrey Pennington, Jiri Hron, Kathleen Kenealy, Kevin Swersky, Kshiteej Mahajan, Laura A Culp, Lechao Xiao, Maxwell Bileschi, Noah Constant, Roman Novak, Rosanne Liu, Tris Warkentin, Yamini Bansal, Ethan Dyer, Behnam Neyshabur, Jascha Sohl-Dickstein, and Noah Fiedel. Beyond human data: Scaling self-training for problem-solving with language models. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum? id=lNAyUngGFK. Expert Certification.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling llm test-time compute optimally can be more effective than scaling model parameters, 2024. URL https://arxiv.org/ abs/2408.03314.

Yuda Song, Hanlin Zhang, Carson Eisenach, Sham Kakade, Dean Foster, and Udaya Ghai. Mind the gap: Examining the self-improvement capabilities of large language models. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu (eds.), International Conference on Learning Representations, volume 2025, pp. 39894–39931, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 63943ee9fe347f3d95892cf87d9a42e6-Paper-Conference.pdf.

Gaurav Srivastava, Zhenyu Bi, Meng Lu, and Xuan Wang. DEBATE, TRAIN, EVOLVE: Self-Evolution of language model reasoning. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 32764–32810, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332-6. doi: 10.18653/v1/2025.emnlp-main. 1666. URL https://aclanthology.org/2025.emnlp-main.1666/.

Igor Lima Strozzi. Teacher-free self-training amplifies but does not compound: A pass@k crossover on a free-verifier domain, 2026. URL https://arxiv.org/abs/2606.07856.

Yifan Sun, Han Wang, Dongbai Li, Gang Wang, and Huan Zhang. The emperor’s new clothes in benchmarking? a rigorous examination of mitigation strategies for LLM benchmark data

contamination. In Forty-second International Conference on Machine Learning, 2025a. URL https://openreview.net/forum?id=TuvDxubEfE.

Yu Sun, Xinhao Li, Karan Dalal, Jiarui Xu, Arjun Vikram, Genghan Zhang, Yann Dubois, Xinlei Chen, Xiaolong Wang, Sanmi Koyejo, Tatsunori Hashimoto, and Carlos Guestrin. Learning to (Learn at test time): RNNs with expressive hidden states. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu (eds.), Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 57503–57522. PMLR, 13–19 Jul 2025b. URL https://proceedings.mlr.press/v267/sun25h.html.

Zhengwei Tao, Ting-En Lin, Xiancai Chen, Hangyu Li, Yuchuan Wu, Yongbin Li, Zhi Jin, Fei Huang, Dacheng Tao, and Jingren Zhou. A survey on self-evolution of large language models, 2024. URL https://arxiv.org/abs/2404.14387.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023a. URL https://openreview.net/forum?id=1PL1NIMMrw.

Yiping Wang, Qing Yang, Zhiyuan Zeng, Liliang Ren, Liyuan Liu, Baolin Peng, Hao Cheng, Xuehai He, Kuan Wang, Jianfeng Gao, Weizhu Chen, Shuohang Wang, Simon Du, and yelong shen. Reinforcement learning for reasoning in large language models with one training example. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 122721–122764. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-4093. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/b1ea3f93167c55f3fc2999b68a170ff3-Paper-Conference.pdf.

Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with self-generated instructions. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 13484– 13508, Toronto, Canada, July 2023b. Association for Computational Linguistics. doi: 10.18653/ v1/2023.acl-long.754. URL https://aclanthology.org/2023.acl-long.754/.

Edwin B. Wilson. Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association, 22(158):209–212, 1927. doi: 10.1080/01621459.1927. 10502953. URL https://www.tandfonline.com/doi/abs/10.1080/01621459. 1927.10502953.

Fang Wu, Weihao Xuan, Ximing Lu, Mingjie Liu, Yi Dong, Zaid Harchaoui, and Yejin Choi. The invisible leash: Why rlvr may or may not escape its origin, 2026. URL https://arxiv. org/abs/2507.14843.

Cheng Xu, Shuhao Guan, Derek Greene, and M-Tahar Kechadi. Benchmark data contamination of large language models: A survey, 2024. URL https://arxiv.org/abs/2406.04244.

Cheng Xu, Nan Yan, Shuhao Guan, Changhong Jin, Yuke Mei, Yibing Guo, and Tahar Kechadi. DCR: Quantifying data contamination in LLMs evaluation. In Christos Christodoulopoulos, Tanmoy Chakraborty, Carolyn Rose, and Violet Peng (eds.), Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 23013–23031, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-332- 6. doi: 10.18653/v1/2025.emnlp-main.1173. URL https://aclanthology.org/2025. emnlp-main.1173/.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang

Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, juncai liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Ru Zhang, Wang Zhang, Hang Zhu, Jinhua Zhu, Jiaze Chen, Jiangjie Chen, Chengyi Wang, Hongli Yu, Yuxuan Song, Xiangpeng Wei, Hao Zhou, Jingjing Liu, Wei-Ying Ma, Ya-Qin Zhang, Lin Yan, Yonghui Wu, and Mingxuan Wang. Dapo: An open-source llm reinforcement learning system at scale. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 113222–113244. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-3775. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/a4277440d50f1f15d2cb4c14f7e0c0d2-Paper-Conference.pdf.

Ye Yu, Xiaopeng Yuan, Haibo Jin, Heming Liu, Yaoning Yu, and Haohan Wang. Do self-evolving agents forget? capability degradation and preservation in lifelong llm agent adaptation, 2026a. URL https://arxiv.org/abs/2605.09315.

Zhuoyun Yu, Xin Xie, Wuguannan Yao, Chenxi Wang, Lei Liang, Xiang Qi, and Shumin Deng. Skilladaptor: Self-adapting skills for llm agents from trajectories, 2026b. URL https: //arxiv.org/abs/2606.01311.

Suqin Yuan, Jinkun Chen, Jiyang Zheng, Muyang Li, Lei Feng, Dadong Wang, Tao Xiang, Tongliang Liu, and Bo An. Understanding diversity collapse in rlvr via the lens of overtraining, 2026. URL https://arxiv.org/abs/2606.15455.

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Xian Li, Sainbayar Sukhbaatar, Jing Xu, and Jason E Weston. Self-rewarding language models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 57905–57923. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/yuan24d.html.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. Scaling relationship on learning mathematical reasoning with large language models, 2023. URL https://arxiv.org/abs/2308.01825.

Yang Yue, Zhiqi Chen, Rui Lu, Andrew Zhao, Zhaokai Wang, Yang Yue, Shiji Song, and Gao Huang. Does reinforcement learning really incentivize reasoning capacity in llms beyond the base model? In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 57654–57689. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-1933. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/537d5aa768c2d534016a4d06f87bc8fb-Paper-Conference.pdf.

Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp. 15476–15488. Curran Associates, Inc., 2022. doi: 10.52202/ 068431-1126. URL https://proceedings.neurips.cc/paper\_files/paper/ 2022/file/639a9a172c044fbb64175b5fad42e9a5-Paper-Conference.pdf.

Eric Zelikman, Georges Raif Harik, Yijia Shao, Varuna Jayasiri, Nick Haber, and Noah Goodman. Quiet-STar: Language models can teach themselves to think before speaking. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id= oRXPiSOGH9.

Hugh Zhang, Jeff Da, Dean Lee, Vaughn Robinson, Catherine Wu, Will Song, Tiffany Zhao, Pranav Raja, Charlotte Zhuang, Dylan Slack, Qin Lyu, Sean Hendryx, Russell Kaplan, Michele Lunati, and Summer Yue. A careful examination of large language model performance on grade school arithmetic. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet,

J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 46819–46836. Curran Associates, Inc., 2024. doi: 10.52202/079017-1485. URL https://proceedings.neurips.cc/paper\_files/paper/2024/file/ 53384f2090c6a5cac952c598fd67992f-Paper-Datasets\_and\_Benchmarks\_ Track.pdf.

Andrew Zhao, Yiran Wu, Tong Wu, Quentin Xu, Yang Yue, Matthieu Lin, Shenzhi Wang, Qingyun Wu, Zilong Zheng, and Gao Huang. Absolute zero: Reinforced self-play reasoning with zero data. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 105816–105879. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-3534. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/9837dc00ff67d176373268ed48042d49-Paper-Conference.pdf.

Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yang Deng, Yanyan Zhao, Wanxiang Che, Bing Qin, and Ting Liu. Large language model agents are not always faithful self-evolvers. In Fortythird International Conference on Machine Learning, 2026. URL https://openreview. net/forum?id=kTjSSqgqGf.

Yujun Zhou, Zhenwen Liang, Haolin Liu, Wenhao Yu, Kishan Panaganti, Linfeng Song, Dian Yu, Xiangliang Zhang, Haitao Mi, and Dong Yu. Evolving language models without labels: Majority drives selection, novelty promotes variation, 2026. URL https://arxiv.org/abs/2509. 15194.

Yuxin Zuo, Kaiyan Zhang, Li Sheng, Shang Qu, Ganqu Cui, Xuekai Zhu, Haozhan Li, yuchen zhang, Xinwei Long, Ermo Hua, Biqing Qi, Youbang Sun, Zhiyuan Ma, Lifan Yuan, Ning Ding, and Bowen Zhou. Ttrl: Test-time reinforcement learning. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 131459–131483. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-4376. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/be690ea16f005c174f6c4102a5970e67-Paper-Conference.pdf.

Adam Zweiger, Jyo Pari, Han Guo, Yoon Kim, and Pulkit Agrawal. Self-adapting language models. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen (eds.), Advances in Neural Information Processing Systems, volume 38, pp. 74084–74115. Curran Associates, Inc., 2025. doi: 10.52202/ 085713-2483. URL https://proceedings.neurips.cc/paper\_files/paper/ 2025/file/6b41e04c41726e2a60e456d0a2b961ab-Paper-Conference.pdf.

## A EXPERIMENTAL PROTOCOLS AND IMPLEMENTATION DETAILS

This appendix fixes what every arm and every generation in the study was run under: the training and sampling configuration (§A.1), the prompt and the grading rules a completion passed through (§A.2), the policy-gradient objective and its three seeds (§A.3), and the compute the study consumed (§A.4).

## A.1 TRAINING AND SAMPLING CONFIGURATION

The prompt. Every generation in this study, for evaluation and for the evolution stream alike, uses one template with no few-shot examples and no system message:

Solve the following math problem. Reason step by step, then   
give your final answer inside \boxed{}.   
Problem: {problem}   
Solution:

The problem string is passed through unmodified, including any Asymptote figure source, which is why some MATH-500 geometry items carry a diagram listing into the context. Holding the prompt fixed across arms and checkpoints matters more here than choosing a good one: a transition is a difference between two evaluations, so anything that changes between them and is not the model is a confound.

Table 3 gives the configuration of every arm and the per-round statistics of its training signal. Three points bear on the fairness of the comparison.

Training volume is not matched on AIME by the obvious construction. On the difficulty band and MATH-500 the retained examples per round are within 7% across arms. On AIME the native protocols do not match: applied to the 60-problem evaluation set itself, as TTRL specifies, the majority-vote arm retains 50/51/50 against distillation’s 247/246/247 drawn from a disjoint stream of 256 problems per round, a factor of 4.9. Training volume is therefore confounded with the filte for precisely the arms that carry the expansion result, which is what the AIME ladder of Section 5 removes: run on one shared stream of 256 problems per round, majority-vote self-training retains 246–254 examples per round across three seeds against distillation’s 246–249, within 2%. STaR retains 205–220, because ground-truth filtering rejects more than a vote taken over the model’s own samples can: a self-consistent vote almost always produces a majority, whether or not it is right, and on this stream 70–81% of those majorities are correct. That residual gap is a property of the filter rather than a design choice, it runs against the arm whose null we report, and we state it rather than tune it away.

The vote is a different instrument on a different stream. Pseudo-label accuracy, the fraction of majority votes that were in fact correct and a diagnostic that never influences training, is 91–95% on the difficulty band, 70–81% on the hard training stream the matched ladder uses, and 26–30% when the vote is taken on AIME itself. The unmatched on-set arm is not simply smaller; it operates in a supervision regime the others never enter. That gap is the subject of Appendix B.2.

Filtering and reinforcement are not the same update. The arms labeled SFT keep the generations that agree with their filter and discard the rest; the policy-gradient arm keeps all of them and centers the reward within each group, so disagreeing samples are pushed down rather than dropped. Group size, stream, volume and evaluation are otherwise identical. Its retained count is reported as rollouts rather than examples, and only groups with reward spread contribute: 1,976, 1,920 and 2,016 rollouts drawn per round on the band, of which 76, 56 and 59 groups carried any gradient.

## A.2 PROMPTING, GRADING AND ANSWER EXTRACTION

Answers are extracted from \boxed{} where present and from an explicit “the answer is” construction otherwise, then compared by symbolic equivalence (Kydl´ıcekˇ , 2025). We deliberately reject a third fallback that is common in practice: taking the last number appearing in the completion. On truncated reasoning that heuristic is close to a coin flip, and its errors are random rather than systematic. Random correctness is far more damaging to a transition metric than uniform strictness:

Table 3: Arm configuration and training-signal statistics. Evaluation is identical across arms within a benchmark: k=128, T=0.8, top-p 0.95, plus one greedy sample, with independent sampling calls. All arms use Qwen3-8B with rank-32 LoRA, $\eta = \bar { 1 0 } ^ { - 4 }$ , batch 8, 3 rounds of 256 stream problems, and at most one training example per problem; SFT arms train 3 epochs per round and the policy gradient arm one, since its rollouts are off-policy after a single pass. The AIME block above the rule is the matched ladder; the arm below it is the unmatched baseline that trains on the evaluation set itself.
<table><tr><td>arm</td><td>benchmark</td><td>gen. T</td><td>max tokens</td><td>samples/prob</td><td>retained per round</td><td>signal correct</td></tr><tr><td>STaR</td><td>MATH-500</td><td>0.9</td><td>2,048</td><td>4</td><td>229 / 227 / 231</td><td>100% (ground truth)</td></tr><tr><td>STaR</td><td>band</td><td>0.9</td><td>2,048</td><td>4</td><td>232 / 229 / 230</td><td>100% (ground truth)</td></tr><tr><td>maj.-vote SFT</td><td>MATH-500</td><td>1.0</td><td>2,048</td><td>8 votes</td><td>241 / 247 / 246</td><td>92.1/89.5/91.5%</td></tr><tr><td>maj.-vote SFT</td><td>band</td><td>1.0</td><td>2,048</td><td>8 votes</td><td>247 / 246 / 240</td><td>92.7/91.1/94.6%</td></tr><tr><td>Distillation</td><td>band</td><td>0.7</td><td>2,048</td><td>4</td><td>232 / 239 / 234</td><td>100% (filtered)</td></tr><tr><td colspan="7">matched AIME ladder: one stream (math-train-hard), one volume, three seeds</td></tr><tr><td>maj.-vote SFT</td><td>AIME</td><td>1.0</td><td>8,192</td><td>8 votes</td><td>250 / 250 / 246</td><td>78.0/77.6/80.5%</td></tr><tr><td>STaR</td><td>AIME</td><td>0.9</td><td>8,192</td><td>4</td><td>220 / 205 / 205</td><td>100% (ground truth)</td></tr><tr><td>Distillation</td><td>AIME</td><td>0.7</td><td>8,192</td><td>4</td><td>247 / 246 / 247</td><td>100% (filtered)</td></tr><tr><td>maj.-vote RL</td><td>AIME</td><td>1.0</td><td>8,192</td><td>8 rollouts</td><td>十</td><td>十</td></tr><tr><td>maj.-vote RL</td><td>band (175)</td><td>1.0</td><td>2,048</td><td>8 rollouts</td><td>十</td><td>十</td></tr><tr><td colspan="7">unmatched baseline (on-set training): unmatched in stream and volume, the reference for Section 4.2</td></tr><tr><td>maj.-vote SFT</td><td>AIME (on-set)</td><td>1.0</td><td>8,192</td><td>8 votes</td><td>50 / 51 / 50</td><td>26.0/27.5/30.0%</td></tr></table>

† final-round figures are filled from the released audit.json of each run, except the two policy-gradient seeds that were stopped after collapsing, whose per-round figures come from their logs (Table 5). The vote is 70–81% correct on the hard training stream against 26–30% when taken on AIME itself: the unmatched baseline differs from the ladder in supervision quality as well as in volume.

a systematically strict grader lowers accuracy at every checkpoint and largely cancels in the difference, whereas a lottery moves problems between states for no reason and is counted as learning or corruption. On one pilot run the fallback supplied 78% of all extracted answers.

Truncation interacts with this directly, which is F3. A completion that hits the token cap has no final answer and is graded incorrect, harmless if the rate is stable across checkpoints and severe if it is not (Table 4). We also track which extraction path produced each answer, since a method that stops emitting \boxed{} would otherwise register as capability loss.

Table 4: Truncation rate by checkpoint on the difficulty band, the diagnostic that excludes an arm’s corruption counts (F3). A method that changes generation style will be scored as destroying capability under a fixed token budget, and nothing else in the ledger distinguishes the two.
<table><tr><td>arm</td><td>t=0</td><td>t=1</td><td>t=2</td><td>t=3</td><td>verdict</td></tr><tr><td>STaR</td><td>16.5%</td><td>15.9%</td><td>12.4%</td><td>14.7%</td><td>stable; counts stand</td></tr><tr><td>maj.-vote SFT</td><td>16.6%</td><td>14.6%</td><td>16.9%</td><td>16.0%</td><td>stable; counts stand</td></tr><tr><td>Distillation</td><td>16.5%</td><td>65.3%</td><td>46.1%</td><td>49.6%</td><td>drift; corruption counts excluded</td></tr></table>

## A.3 POLICY GRADIENT ON A MAJORITY-VOTE REWARD

Why the arm exists. Both self-training arms of the main audit are rejection-sampling SFT: they keep what their filter accepts and discard the rest, where TTRL as published pushes the losing samples down instead. An audit that positions itself against a reinforcement-learning literature and then measures only supervised fine-tuning is auditing something else, so we ran the genuine article: a group-relative policy gradient on the same stream and the same majority vote, under the training service’s importance-sampling loss with group-centered advantages.

On the band it behaves; on AIME it does not survive. The band subsample gives 19 learned against 9 corrupted (CLR = 0.47) with accuracy rising 0.530 → 0.583 and truncation flat at 15.5, 24.4, 7.6 and 8.2% across its four checkpoints. The AIME arm, on a harder stream at a four times larger token cap, collapses. Table 5 gives all three of its seeds.

Table 5: Three seeds of the policy-gradient arm, on identical data. The evolution stream is drawn under a fixed seed, so the three runs see the same 256 problems in the same order each round; --seed varies adapter initialization, training order and sampling. Round 1 is the same experiment three times over: the vote is 77.0–77.9% correct and 135–136 groups of ∼250 carry any gradient. The runs then diverge completely. Every seed ends in severe truncation, so the collapse belongs to the configuration; the round at which it arrives does not. Trained is the number of problems that formed a majority, of 256; spread the groups carrying gradient; vote the fraction of those majorities that were in fact correct, a diagnostic that never enters training.
<table><tr><td rowspan="2">seed</td><td rowspan="2">round</td><td colspan="2">evaluation, AIME</td><td colspan="4">training stream</td></tr><tr><td>accuracy</td><td>truncated</td><td>trained</td><td>spread</td><td>share</td><td>vote</td></tr><tr><td>0</td><td>t = 0</td><td>0.1764</td><td>4.2%</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>t = 1</td><td>0.2082</td><td>18.3%</td><td>248</td><td>135</td><td>0.78</td><td>77.0%</td></tr><tr><td></td><td>t = 2</td><td>0.0643</td><td>24.3%</td><td>251</td><td>143</td><td>0.77</td><td>73.7%</td></tr><tr><td></td><td>t = 3</td><td>0.1921</td><td>76.4%</td><td>243</td><td>217</td><td>0.58</td><td>49.8%</td></tr><tr><td>1</td><td>t = 0</td><td>0.1760</td><td>4.4%</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>t = 1</td><td>0.0000</td><td>100.0%</td><td>249</td><td>135</td><td>0.78</td><td>77.5%</td></tr><tr><td></td><td>t = 2</td><td></td><td></td><td>34</td><td>32</td><td>0.51</td><td>85.3%</td></tr><tr><td>2</td><td>t = 0</td><td>0.1829</td><td>4.5%</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>t = 1</td><td>0.2952</td><td>47.1%</td><td>253</td><td>136</td><td>0.78</td><td>77.9%</td></tr><tr><td></td><td>t = 2</td><td>0.0900</td><td>95.6%</td><td>248</td><td>105</td><td>0.86</td><td>85.1%</td></tr></table>

Seeds 1 and 2 were stopped after the collapse rather than sampled to t=3, since every further generation runs to the token cap. Seed 1’s t=2 vote statistics are computed over the 34 majorities that survived and are a survivorship artifact; seed 2’s t=2 are computed over 248 and are not. Training loss at t=1 was 169.9, 8.5 and 145.1 across the three seeds, a twentyfold spread from adapter initialization alone.

What three seeds show that one could not. The evolution stream is drawn under a fixed seed, so the three runs see the same 256 problems in the same order every round; --seed varies adapter initialization, training order and sampling. Round one is therefore the same experiment three times, and it comes out the same three times: the vote is 77.0, 77.5 and 77.9% correct and 135, 135 and 136 groups of roughly 250 carry any gradient. The runs then diverge to the point of one dying outright. Seed 1 truncates 100% of its completions after a single update and scores 0.0000; seed 2 reaches 95.6% after two; seed 0 reaches 76.4% after three. Two readings follow, and they point in opposite directions, which is why we give both. That every seed ends in severe truncation makes the collapse a property of this configuration rather than of one unlucky run; the answer to whether F3’s drift was reproducible is yes. That the round at which it arrives varies from one update to three, on identical data, makes the arm’s outcome undetermined by its design, which is F5 in a more extreme form than the supervised arms show. The training loss at round one was 169.9, 8.5 and 145.1 across the three seeds: a twentyfold spread from adapter initialization alone.

The diagnostic reads healthy while the model dies. A group contributes gradient exactly when its rollouts disagree, so the arm’s supervision statistics are a live readout of how self-consistent the model is. On seed 0 they degrade as the model does. The vote falls 77.0 → 73.7 → 49.8% and the share of rollouts agreeing with it 0.78 → 0.58, while the groups carrying gradient rise 135 → 143 → 217: by round three the pseudo-label is close to a coin flip and the optimizer receives more gradient than ever, precisely because the model has stopped agreeing with itself. That is a real measurement (243 of 256 problems still formed a majority), but it is one seed at one round, and it does not replicate. On seed 2 the same statistics move the other way at round two, over an equally intact denominator of 248: the vote becomes more accurate (77.9 → 85.1%) and more unanimous (share 0.78 → 0.86), and fewer groups carry gradient (136 → 105), while the evaluation goes from 47% truncated to 96%. Seed 1’s round-two figures (85.3% correct) are a survivorship artifact and we mark them as such: only 34 of 256 problems formed a majority at all, because a degenerate generation yields no extractable answer to vote on. What survives across the seeds is not a direction but a warning: on two of three runs the arm’s own supervision statistics looked stable or improving at the checkpoint after which the model was unusable, so they cannot be used to decide when to stop.

The scope of the claim, and the exclusion criterion. The AIME arm’s transition counts are excluded from the capability analysis. They are read off models truncating 76 to 100% of their completions, and F3 establishes that a token cap under generation-length drift scores style change as capability loss; the criterion applies to this arm exactly as it does to distillation. The detection counts of Table 2 are retained because truncation moves them the other way: a truncated completion cannot be scored correct, so a collapsing arm understates its detections and the count is a floor. We claim, for this configuration only, that three rounds of policy gradient on a majority-vote reward at rank-32 LoRA and $\bar { \eta } = 1 0 ^ { - 4 }$ collapsed the policy into unbounded repetition on every seed we ran. We do not claim that a majority-vote reward does this in general: seed 2 took two intact updates at the identical learning rate, so the rate is not uniformly fatal, and the band arm at a shorter cap on an easier stream never collapses at all. Separating the reward from the step size needs a learning-rate sweep we did not run.

## A.4 COMPUTE FOOTPRINT AND COST

All experiments used Tinker, a hosted LoRA training and sampling service. Unless stated otherwise the backbone is Qwen3-8B with rank-32 adapters; the distillation teacher is gpt-oss-120b (OpenAI et al., 2025), and one noise floor was measured on a second 20B backbone. The study consumed 3.44M sampled generations and 5.30B generated tokens across 48 runs, itemized in Table 6.

Table 6: Compute by experiment class. Cost is the metered price of sampling and training on the hosted service; token counts are generated tokens, which dominate cost.
<table><tr><td>experiment class</td><td>runs</td><td>samples (M)</td><td>tokens (M)</td><td>cost (USD)</td></tr><tr><td>Primary audits, difficulty band (n=1,163)</td><td>3</td><td>1.80</td><td>2,441</td><td>1,465</td></tr><tr><td>Primary audits, MATH-500 and AIME</td><td>4</td><td>0.27</td><td>414</td><td>248</td></tr><tr><td>Matched AIME ladder, 3 filters × seeds</td><td>6</td><td>0.19</td><td>692</td><td>415</td></tr><tr><td>Policy-gradient arms</td><td>4</td><td>0.16</td><td>443</td><td>266</td></tr><tr><td>Seed replication (2 methods × 2 seeds)</td><td>4</td><td>0.36</td><td>470</td><td>282</td></tr><tr><td>Noise floors (4 benchmarks)</td><td>7</td><td>0.42</td><td>589</td><td>340</td></tr><tr><td>Benchmark construction and model profiling</td><td>7</td><td>0.11</td><td>101</td><td>61</td></tr><tr><td>Out-of-domain probes (3 replicates each)</td><td>6</td><td>0.01</td><td>4</td><td>2</td></tr><tr><td>Batch-invariance test (F1 mechanism)</td><td>1</td><td>&lt; 0.01</td><td>1</td><td>&lt; 1</td></tr><tr><td>Pilot audits, 112-problem band</td><td>2</td><td>0.12</td><td>147</td><td>88</td></tr><tr><td>Method pilots and calibration</td><td>4</td><td>&lt; 0.01</td><td>4</td><td>2</td></tr><tr><td>Total</td><td>48</td><td>3.44</td><td>5,303</td><td>3,171</td></tr></table>

Three ratios bear on the paper’s recommendation. The measurement infrastructure (noise floors, benchmark construction and the out-of-domain probes) accounts for 13% of total spend, and the small pilots that exposed four of the seven failure modes cost \$2 between them. Against that, the three primary audits alone cost \$1,465. And the controls the paper’s conclusions rest on cost nothing: the empirical null distributions, the design-matched floors, the threshold-free expansion test and the seed analysis are all re-computations over records the study already holds, because every arm’s baseline is another evaluation of the untrained model. What money buys is the matched ladder (\$415), which removes a confound rather than measuring something new, and the policy-gradient arms (\$266), which audit a method family the SFT arms cannot stand in for and surface a failure mode none of the others can exhibit.

The dominant cost is sampling rather than training: evaluating 1,163 problems at k=128 over four checkpoints is 600K generations per arm, against roughly 270 optimizer steps. This is a property of transition-level auditing generally: statistical power comes from problems near the decision boundary, and reaching them requires sampling breadth rather than longer training. An audit budget is therefore set almost entirely by the evaluation design, and can be estimated in advance from the benchmark size, k, the checkpoint count and the mean completion length, all known before any method is run.

## B BENCHMARK SUITES AND DATASET CONSTRUCTION

Corruption and expansion are observable on disjoint populations of problems, so no single evaluation set can host both. This appendix gives the construction of the set built to make corruption observable (§B.1) and the characterization of all three sets against the regimes they can and cannot measure (§B.2).

## B.1 CONSTRUCTING THE DIFFICULTY BAND

Corruption is observable only on problems a model already solves, and expansion only on problems it does not reach. MATH-500 supplies the first population and almost none of the second; AIME the reverse (Figure 4). Neither can host a well-powered corruption measurement. We therefore construct a third set: profile 10,999 held-out MATH training problems at k=8 and keep the 1,611 whose measured solve rate falls in [0.25, 0.75], and retain the 1,163 of them closest to the center of that interval. The retained set has a mean solve rate of 0.522 under the k=8 selection profile and 0.552 when re-measured at k=128, where it holds 711 corruptible and 452 learnable problems, the first balanced pools in the study.

Selecting problems by measured solve rate and then measuring them again invites an obvious objec tion: regression to the mean. Figure 3 shows it plainly. This does not bias the transition counts, for a specific reason: selection uses an independent k=8 profile, the audit re-evaluates every problem from scratch at k=128, and that measurement is checkpoint 0, the baseline against which all transitions are computed. The noisy estimate used to choose never enters the ledger. What regression to the mean costs is realized band width: fewer problems sit near threshold at k=128 than the selection targeted, which reduces power but cannot manufacture a transition.

The band’s construction removes every problem the base model cannot reach, so it has no expansion candidates and its expansion statistic is vacuous. This is the price of its corruption sensitivity. It also means the band’s raw counts are not a forgetting rate for a naturally sampled problem distribution: the set is deliberately enriched for problems that can transition, and Table 23 rather than the raw count is the quantity that transfers.

![](images/c0829bed5d5f5239a14d5d7fba936a8482e1baebf6a36dd9697d9c27e76ecc3e.jpg)

![](images/3b0d23eb97f1ca39b7dddc52a466a6b6180e080e22630db40bb45e50810e593c.jpg)  
Figure 3: Construction of the difficulty band. Left: the held-out pool and the retained band; dashed lines mark the selection interval [0.25, 0.75]. Right: selection estimate at k=8 against the independent re-measurement at $k { = } 1 2 8$ that serves as checkpoint 0. Regression to the mean widens the realized band, costing statistical power; because the two measurements are independent it cannot create transitions.

## B.2 BENCHMARK CHARACTERIZATION AND SOLVABILITY REGIMES

Benchmarks, in full. Corruption is observable only on problems the model already solves, expansion only on problems it does not reach. These populations are disjoint, and a benchmark rich in one is poor in the other (Figure 4), so we use three sets. MATH-500 (Lightman et al., 2024), subsampled to 200 problems by a fixed stratified draw over (subject, level) released with the code, gives comparability with prior work but is thoroughly contaminated (168–170 of 200 problems already solved at k=128), hosting corruption but not expansion. AIME 2025 and 2026 (60 problems) are the least contaminated sets available to us: AIME 2026 post-dates the backbone’s March 2025 cut off (Yang et al., 2025) outright, while AIME 2025 was administered in February 2025 and sits just inside it, so we report the 2026 half separately wherever the expansion claim rests on it. The base model’s mean solve rate on AIME is 0.175–0.180 across its eleven baselines, leaving 22 problems it reaches at most five times in 1,408 pooled draws and only 8 that can be corrupted, so AIME carrie the expansion claims and has no corruption power. A difficulty band of 1,163 problems, built by profiling 10,999 held-out MATH problems (Hendrycks et al., 2021) at k=8 and retaining those with solve rate in [0.25, 0.75], is deliberately enriched for problems near the decision boundary so that corruption is observable at all. Its counts are therefore not a forgetting rate for a natural problem distribution, and by construction it contains no base-unreached problems (Appendix B.1). Because the fraction of problems near threshold differs across sets, CLR is not portable between them.

![](images/05a9145554c749424d23469f4838d451844cdae95628366e179da2c24ff15856.jpg)

![](images/dc4a414dec13c82e7cada486ad1023e9d93df683c4708140f4bfb5b9d9dde30d.jpg)

![](images/1d3a394ba0b4f02f9d2f1832ece327a7b7f0914ac57ed7daf1d271b1b67ca568.jpg)  
Figure 4: An evaluation set fixes which question it can answer, before any method is applied. Measured distribution of the untrained backbone’s per-problem solve rate at k=128, from the frozencontrol records; shading marks the hysteresis band. Expansion is observable only in the spike at $\hat { \pi } _ { 0 } = 0$ and corruption only in the mass above 0.5, so AIME can host an expansion measurement and not a corruption one, MATH-500 the reverse, and the constructed band neither expansion nor a natural forgetting rate.

A falsified prediction about where corruption appears. We pre-registered the prediction that majority-vote self-training would corrupt more where its pseudo-labels were less reliable, and it was falsified. On AIME its label accuracy falls to 26–30%, so the vote is wrong most of the time (Table 3), and corruption goes to zero. The explanation is that AIME leaves only 8 problems the model solves and could therefore lose; the smallest non-zero corruption rate observable there is 12.5%, against a rate of 1.2% measured on MATH-500.

Wrong supervision on problems a model already fails destroys nothing. Corruption is bounded by what there is to lose, not by how wrong the teacher is, and an evaluation set can be too hard to detect it as easily as it can be too easy. We report the falsification because the account that replaces it, that corruption requires both unreliable supervision and prior competence on the affected problems, is what motivates constructing the difficulty band, and because it is the reason the AIME arms carry expansion claims but not corruption ones.

## C THE TRANSITION LEDGER AND ITS NOISE FLOOR

The ledger of Section 3 turns per-problem solve rates into transition counts. This appendix gives the rule in full and the counts it produces (§C.1), the floors and the sensitivity of both to z and k (§C.2), and the three failure modes that attack the ledger itself (§C.3).

## C.1 THE BASELINE STATE, HYSTERESIS, AND THE FULL LEDGER COUNTS

The baseline state carries no hysteresis, and this matters. Equation 2 is undefined at t = 0, where there is no previous state to hold; we assign $s _ { 0 } ( p ) = { \mathsf { s o l v e d } }$ iff $\begin{array} { r } { \hat { \pi } _ { 0 } ( p ) \ge \frac { 1 } { 2 } } \end{array}$ . The consequence is load-bearing and easy to miss. A problem whose baseline sits inside the band is unprotected at that one checkpoint, so a frozen model can register a transition on a movement smaller than the band is wide. That is why the floors of Table 10 are not zero on a set whose mean solve rate is 0.552. Section 5 reports one alternative, a pure two-checkpoint rule with no carry. The other is symmetric: treat a baseline inside the band as indeterminate and let a problem transition only once it has left the band. That rule drives the frozen floor to exactly 0 learned and 0 corrupted, with no artifact surviving it, while STaR gives 64/53 and majority-vote SFT 75/55. It buys that zero by setting aside 325 of the 1,163 problems, which is why we do not adopt it as the primary rule: a floor of zero over a set chosen for having no determinate starting state is a weaker achievement than a small floor over the whole set. The conclusion is unchanged under it, the arms now clearing the floor absolutely rather than by a ratio.

Table 7: What each method adds and removes, against a floor measured on the same problems and matched to the same number of checkpoints. Counts are problems. The floor columns give the median over four-checkpoint orderings of the independent evaluations of the untrained model that benchmark carries, with the conventional two-checkpoint floor in parentheses.
<table><tr><td colspan="2"></td><td colspan="2">transitions</td><td colspan="2">matched floor</td><td rowspan="2">accuracy</td></tr><tr><td>method</td><td>benchmark</td><td>learned</td><td>corrupted</td><td>L</td><td>C</td></tr><tr><td>STaR</td><td>MATH-500</td><td>2</td><td>1</td><td>0 (0)</td><td>1 (1)</td><td>0.835 → 0.850</td></tr><tr><td>maj.-vote SFT</td><td>MATH-500</td><td>3</td><td>2</td><td>0 (0)</td><td>1 (1)</td><td>0.832 → 0.830</td></tr><tr><td>maj.-vote SFT</td><td>AIME</td><td>4</td><td>0</td><td>0 (0)</td><td>0 (0)</td><td>0.177 → 0.202</td></tr><tr><td>Distillation†</td><td>AIME</td><td>9</td><td>0</td><td>0 (0)</td><td>0 (0)</td><td>0.180 → 0.310</td></tr><tr><td>STaR</td><td>band</td><td>149</td><td>106</td><td>10 (4)</td><td>8 (5)</td><td>0.550 → 0.606</td></tr><tr><td>maj.-vote SFT</td><td>band</td><td>145</td><td>88</td><td>10 (4)</td><td>8 (5)</td><td>0.550 → 0.590</td></tr><tr><td>Distillation†</td><td>band</td><td>161</td><td>177</td><td>10 (4)</td><td>8 (5)</td><td>0.550 → 0.546</td></tr></table>

<sup>†</sup>positive control. <sup>‡</sup>corruption counts excluded: truncation drifts 16.5% → 65.3% (F3). The AIME rows here are the unmatched baselines (on-set training, unmatched volume); the matched ladder is Table 2. Sharpened share of newly solved problems on the sets where expansion is observable at all: 2/2, 3/3, 4/4, so 9/9, Wilson [0.70, 1.00]. The band arms (149/149, 145/145, 161/161) are vacuous by construction and are excluded from that proportion rather than added to it. CLR with percentile-bootstrap intervals over problems, band: STaR 0.711 [0.557, 0.915], maj.-vote SFT 0.607 [0.460, 0.785]; seed variance is roughly three times this (Table 13).

Table 9 gives the complete seven-category partition for every arm. The categories are mutually exclusive and sum to the problem count. Recovered and transient problems, those that change state and change back, are reported separately because they are invisible to an endpoint comparison but bear on whether a method’s damage is permanent. Their scarcity (at most 5% of problems in any arm) is why we report the endpoint definition of CLR in the main text: on this data the two definitions coincide to within three problems. Movement is bidirectional in every arm, the band producing 88– 175 corruptions alongside 144–160 learnings, so a rising mean accuracy is a net result and not a uniform lift.

The path to the endpoint. Table 8 and Figure 5 give the same ledger evaluated at every checkpoint rather than only at the last. Two features are invisible in the endpoint comparison. Distillation on the band peaks at round 1 (0.610) and then falls to below its own baseline by round 3 (0.546), which is the truncation drift of F3 taking hold as the student’s completions lengthen. Majority-vote selftraining peaks at round 2 (0.605) and gives back 1.5 points at round 3 while its corruption count rises from 59 to 88; a study that stopped at two rounds would have reported a better method. Neither observation changes a headline, and we report the endpoint comparison because it is the one prior work reports, but the trajectory is where a practitioner would choose when to stop.

Where the mass goes. Figure 6 bins the solve rate into five bands and tabulates the joint distribution of before and after. The diagonal holds; the off-diagonal mass is large in both directions for every arm: 464 problems up and 244 down for STaR, 435 and 276 for majority-vote self-training, 402 and 389 for distillation. The bottom-left cell is the one that matters for the expansion claim, and it is where the band’s construction bites: with no problems at $\hat { \pi } _ { 0 } = 0$ there is nothing in that cell to move. Appendix D.2 shows the same movement without binning.

Table 8: Per-checkpoint accuracy and transition counts. Every headline number in the paper compares t=0 with t=3; this is the path between them. Each count compares that checkpoint with checkpoint 0 under the ledger of Eq. 2; it is a running state, not a cumulative total, so it can fall when a problem re-crosses the band, and the final column of each block reproduces Table 7. Distillation on the band peaks at round 1 and declines under its own truncation drift (F3); majority-vote SFT on the band peaks at round 2 in accuracy while its corruption count keeps rising. The AIME policy-gradient row is reported for completeness only: that arm collapses into unbounded repetition and is 76.4% truncated at $t { = } 3 ,$ , so its transition counts are excluded from the capability analysis (Appendix A.3).
<table><tr><td colspan="2"></td><td colspan="4">mean accuracy</td><td colspan="4">learned vs  $t _ { 0 }$ </td><td colspan="4">corrupted vs  $t _ { 0 }$ </td></tr><tr><td>arm</td><td>set</td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $t _ { 2 }$ </td><td> $t _ { 3 }$ </td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $t _ { 2 }$ </td><td> $t _ { 3 }$ </td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $t _ { 2 }$ </td><td> $t _ { 3 }$ </td></tr><tr><td>STaR</td><td>band</td><td>0.550</td><td>0.590</td><td>0.591</td><td>0.606</td><td>0</td><td>83</td><td>123</td><td>149</td><td>0</td><td>48</td><td>98</td><td>106</td></tr><tr><td>maj.-vote SFT</td><td>band</td><td>0.550</td><td>0.586</td><td>0.605</td><td>0.590</td><td>0</td><td>57</td><td>119</td><td>145</td><td>0</td><td>33</td><td>59</td><td>88</td></tr><tr><td>Distillation†</td><td>band</td><td>0.550</td><td>0.610</td><td>0.577</td><td>0.546</td><td>0</td><td>151</td><td>168</td><td>161</td><td>0</td><td>101</td><td>143</td><td>177</td></tr><tr><td>maj.-vote RL</td><td>band (175)</td><td>0.530</td><td>0.603</td><td>0.578</td><td>0.583</td><td>0</td><td>22</td><td>21</td><td>19</td><td>0</td><td>13</td><td>12</td><td>9</td></tr><tr><td>STaR</td><td>MATH-500</td><td>0.835</td><td>0.841</td><td>0.851</td><td>0.850</td><td>0</td><td>1</td><td>4</td><td>2</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>maj.-vote SFT</td><td>MATH-500</td><td>0.832</td><td>0.836</td><td>0.830</td><td>0.830</td><td>0</td><td>1</td><td>2</td><td>3</td><td>0</td><td>0</td><td>0</td><td>2</td></tr><tr><td>maj.-vote SFT</td><td>AIME</td><td>0.180</td><td>0.193</td><td>0.227</td><td>0.213</td><td>0</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>STaR</td><td>AIME</td><td>0.177</td><td>0.190</td><td>0.204</td><td>0.195</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Distillation†</td><td>AIME</td><td>0.180</td><td>0.347</td><td>0.308</td><td>0.310</td><td>0</td><td>12</td><td>11</td><td>9</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>maj.-vote RL‡</td><td>AIME</td><td>0.176</td><td>0.208</td><td>0.064</td><td>0.192</td><td>0</td><td>3</td><td>0</td><td>4</td><td>0</td><td>0</td><td>6</td><td>2</td></tr></table>

<sup>†</sup>positive control, not a self-evolution method. <sup>‡</sup>counts excluded: generation collapse (Table 5).

![](images/576686f47016918019d33f41b1e64ce6f4e314da7bec2bb9178931aa18a285ec.jpg)

![](images/b810214e383d2736ab1d5e517628cfb9b1b6a0833e7fb95bad2527e4e0de7a25.jpg)

![](images/54527ae87e3e746286652dffab4a1ae1bb6b1cea4486dd0983b2942a06821564.jpg)  
Figure 5: The path the endpoint comparison hides. Mean accuracy and the learned and corrupted counts against checkpoint 0 at every checkpoint on the difficulty band, with the frozen-model floor shaded on the two count panels. Distillation peaks at round 1 and declines under its own truncation drift; majority-vote self-training’s accuracy peaks at round 2 while its corruption count continues to climb.

## C.2 NOISE FLOORS, AND SENSITIVITY TO z AND k

Table 11 names every MATH-500 problem whose greedy verdict changed between two evaluations of the frozen model, with the solve rate measured on the same samples. The point is the disagreement between the two: num theory/598 is scored as corrupted while its solve rate is 0.898 at both checkpoints, and int algebra/1063 is scored as learned while its solve ratefalls from 0.133 to 0.023. The single-decode verdict is not a noisy measurement of capability; on these problems it is close to independent of it. Nine of the fifteen are scored as corruptions and six as learnings, which is the $\mathrm { C L R } = 1 . 5$ of Section 4.1 in full.

Table 12 sweeps the hysteresis width z and the sampling budget k on the difficulty band. The arms ratios are flat across both sweeps while the frozen-model floor is not, which is what the hysteresis band is for. Smaller k is simulated by subsampling the stored 128 draws without replacement, so the comparison is exact rather than a fresh sample.

![](images/e07e2d05afa25d6841b7aa851d6541241165735ced494ee6dfe5d0b6c6c7abfd.jpg)

![](images/df09d7ccf743053bb3f082046ff51cd901a2fd418dce29542dc0b33e393a6e6f.jpg)

![](images/5753a6064b3f414f6c4cb1c9e0b68ec3ab46cb0837e9cab5d7e80645226309c2.jpg)  
Figure 6: Migration matrices: where problems actually move. Joint distribution of the perproblem solve rate before and after evolution, binned into fifths, for the three difficulty-band arms; the dashed line is no change and cell counts sum to 1,163. “Up” and “down” count problems above and below the diagonal. Movement is substantial in both directions for every arm, which is the point that a mean accuracy cannot express.

Table 9: Ledger categories by arm, sampled ledger at $k { = } 1 2 8 , z { = } 2$ . Learned and corrupted here are the trajectory-based labels; the endpoint counts used for CLR in the main text differ only where a problem oscillates.
<table><tr><td></td><td></td><td colspan="2">stable</td><td colspan="5">changed</td></tr><tr><td>arm</td><td>benchmark</td><td>correct</td><td>incorrect</td><td>learned</td><td>corrupted</td><td>recovered</td><td>transient</td><td>oscillating</td></tr><tr><td>STaR</td><td>MATH-500</td><td>167</td><td>25</td><td>2</td><td>1</td><td>2</td><td>3</td><td>0</td></tr><tr><td>maj.-vote SFT</td><td>MATH-500</td><td>167</td><td>28</td><td>3</td><td>2</td><td>0</td><td>0</td><td>0</td></tr><tr><td>maj.-vote SFT</td><td>AIME</td><td>8</td><td>48</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Distillation†</td><td>AIME</td><td>8</td><td>40</td><td>9</td><td>0</td><td>0</td><td>3</td><td>0</td></tr><tr><td>STaR</td><td>band</td><td>567</td><td>299</td><td>149</td><td>103</td><td>25</td><td>17</td><td>3</td></tr><tr><td>maj.-vote SFT</td><td>band</td><td>589</td><td>299</td><td>144</td><td>88</td><td>19</td><td>23</td><td>1</td></tr><tr><td>Distillation{</td><td>band</td><td>532</td><td>248</td><td>160</td><td>175</td><td>11</td><td>34</td><td>3</td></tr></table>

<sup>†</sup>positive control. <sup>‡</sup>corruption counts excluded: truncation drift (F3).

What the measured floor buys, and what the rule does. A binomial calculation at each problem’s pooled base rate, the analytic unchanged-policy null of Yuan et al. (2026), predicts a median of 3 learned and 4 corrupted on the band against our measured 4 and 3, and an envelope of 0.203 against 0.234. For the solve-rate estimator the analytic null is very nearly right, and we say so rather than claim an advantage we cannot demonstrate; where the two part company is the single-decode protocol, for which an analytic null predicts no transitions at all. Discretization choices likewise move the counts without moving the conclusion: under a pure two-checkpoint rule with no carry, STaR’s counts fall from 149/106 to 120/79 and majority-vote SFT’s from 145/88 to 118/78 while the floor stays at $4 / 5 ;$ dropping the band entirely raises the arms to 172/111 and 167/117 but the floor to 44/39. CLR stays within 0.65–0.71 for STaR across all three rules, and within 0.65–0.72 over $z \in [ 0 , 3 ]$ and 0.56–0.81 over k ∈ {16, 32, 64, 128} (Table 12). One caution bears on the main claim: under the pure endpoint rule the two methods coincide exactly (0.66 and 0.66), and at z = 0 their order reverses, so the between-method comparison is sensitive to the rule as well as to the seed and we rest nothing on it.

## C.3 FAILURE MODES F3, F4 AND F5 IN FULL

F3. A token cap scores style change as capability loss. The distillation arm on the band records 177 corruptions and ${ \mathrm { C L R } } = 1 . 1 0 $ , nominally the most destructive method tested. Its truncation rate meanwhile moves from 16.5% to 65.3% between checkpoints while both self-training arms stay flat at 12–17%: the student has adopted its 120B teacher’s verbose style and is hitting the token cap, losing the final answer. Conditioned on completions that finish, it improves from 0.656 to 0.845 against STaR’s 0.655 to 0.702: the most effective method tested, not the most destructive. Its corruption counts are excluded from the capability analysis rather than caveated, since severe generation-length drift under a fixed token cap confounds capability degradation with stylistic verbosity (Appendix A.2).

Table 10: Noise floors: apparent transitions produced by an untrained model (F1, F7). Estimator columns give the range over every ordered pair of independent frozen evaluations (110 pairs on AIME, 12 on MATH-500 and 20 on the band) and, in the last estimator column, the median over four-checkpoint orderings that are design-matched to an arm. A conventional single-pair floor is the low end of these ranges. The single-greedy protocol is what a per-example before/after comparison implicitly uses; its changed rate carries a Wilson interval since it is a proportion over problems.
<table><tr><td></td><td></td><td colspan="3">solve-rate estimator, k=128</td><td colspan="2">single greedy sample</td><td></td></tr><tr><td>benchmark</td><td>backbone</td><td>learned</td><td>corrupted</td><td>matched (L/C)</td><td>L/C</td><td>changed [95%]</td><td>problems</td></tr><tr><td>MATH-500</td><td>Qwen3-8B</td><td>0-1</td><td>0-1</td><td>0/1</td><td>6/9</td><td>7.5% [4.6, 12.0]</td><td>200</td></tr><tr><td>AIME 2025/26</td><td>Qwen3-8B</td><td>0-0</td><td>0-1</td><td>0/0</td><td>2/2</td><td>6.7% [2.6, 15.9]</td><td>60</td></tr><tr><td>difficulty band</td><td>Qwen3-8B</td><td>0-7</td><td>1-9</td><td>10/8</td><td>100 / 120</td><td>18.9% [16.8, 21.3]</td><td>1,163</td></tr><tr><td>band (pilot)</td><td>Qwen3-8B</td><td>0</td><td>0</td><td></td><td></td><td></td><td>112</td></tr><tr><td>AIME 2025/26</td><td>gpt-oss-20b</td><td>0</td><td>0</td><td></td><td>5/4</td><td>15.0% [8.1, 26.1]</td><td>60</td></tr></table>

The gpt-oss-20b row is the only measurement we report on a second backbone; its arms were not run because its capability profile places it in a different regime, leaving only 7 base-unreached AIME problems. Pilot and second-backbone rows have a single frozen pair each and so have no range.

Table 11: Every problem whose greedy verdict changed between two evaluations of the frozen model (F1), MATH-500, T=0 with a fixed seed. $\checkmark$ and × are the greedy verdicts; πˆ is the solve rate over 128 samples at the same checkpoint. The point of the table is the last two columns of each block: the underlying solve rate barely moves, since it is a frozen model, while the single-decode verdict flips. Nine of these are scored as corruptions and six as learnings, giving the CLR = 1.5 of Section 4.1.
<table><tr><td>problem</td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $\hat { \pi } _ { 0 }$ </td><td> $\hat { \pi } _ { 1 }$ </td><td>problem</td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $\hat { \pi } _ { 0 }$ </td><td> $\hat { \pi } _ { 1 }$ </td></tr><tr><td>counting-prob/199</td><td>X</td><td>了</td><td>0.633</td><td>0.672</td><td>num_theory/183</td><td>X</td><td>√</td><td>0.875</td><td>0.852</td></tr><tr><td>geometry/226</td><td>√</td><td>X</td><td>0.719</td><td>0.750</td><td>num_theory/357</td><td>√</td><td>×</td><td>0.594</td><td>0.453</td></tr><tr><td>geometry/702</td><td>×</td><td>√</td><td>0.078</td><td>0.117</td><td>num_theory/516</td><td>√</td><td>×</td><td>0.398</td><td>0.328</td></tr><tr><td>int_algebra/1063</td><td>×</td><td>了</td><td>0.133</td><td>0.023</td><td>num_theory/598</td><td>√</td><td>×</td><td>0.898</td><td>0.898</td></tr><tr><td>int_algebra/1350</td><td>√</td><td>×</td><td>0.367</td><td>0.367</td><td>prealgebra/1128</td><td>√</td><td>×</td><td>0.688</td><td>0.758</td></tr><tr><td>int_algebra/1797</td><td>×</td><td>√</td><td>0.859</td><td>0.844</td><td>precalculus/1313</td><td>√</td><td>×</td><td>0.602</td><td>0.695</td></tr><tr><td>int_algebra/754</td><td>×</td><td>L</td><td>0.547</td><td>0.367</td><td>precalculus/44</td><td>√</td><td>X</td><td>0.922</td><td>0.953</td></tr><tr><td>int_algebra/966</td><td>√</td><td>X</td><td>0.391</td><td>0.344</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Under the solve-rate estimator of Eq. 2, the same 200 problems yield 0 learnings and 1 corruption.

F4. Transition metrics need an order of magnitude more data than accuracy. Only problems near the decision boundary can produce an event, so a benchmark’s difficulty profile determines power directly. Detecting the difference in corrupted share we observe between STaR and majorityvote self-training at 80% power requires roughly 249 events per arm; a 112-problem pilot yields 24 and cannot resolve it at any significance level, which is why the primary comparison uses 1,163 problems (Appendix D.3).

F5. The comparison between methods is not stable across training seeds. Table 13 gives all three seeds of both band arms with every seed scored on the same 175 problems, and Figure 8 plots the resulting spread against the noise floor. Seed 0 ran on the full band and seeds 1–2 on a 175-problem subsample, so the matched columns restrict seed 0 to those same 175 problems: the reversal is a seed effect and not a set effect. STaR holds CLR within 0.45–0.74 on every seed, while majority-vote self-training spans 0.55 to 1.53 and crosses the threshold at which a method destroys more than it creates. Section 4.3 states what that licenses and what it does not.

Table 12: Sensitivity of the ledger to z and k: learned/corrupted counts with CLR in parentheses, difficulty band.
<table><tr><td colspan="4">hysteresis width z (at k=128)</td><td colspan="4">sampling budget k (at z=2)</td></tr><tr><td>z</td><td>STaR</td><td>maj.-vote</td><td>frozen</td><td>k</td><td>STaR</td><td>maj.-vote</td><td>frozen</td></tr><tr><td>0.0</td><td>172/111 (0.65)</td><td>167/117 (0.70)</td><td>44/39</td><td>16</td><td>115/64 (0.56)</td><td>125/69 (0.55)</td><td>10/7</td></tr><tr><td>1.0</td><td>165/111 (0.67)</td><td>158/103 (0.65)</td><td>12/22</td><td>32</td><td>103/83 (0.81)</td><td>120/88 (0.73)</td><td>6/8</td></tr><tr><td>1.5</td><td>160/109 (0.68)</td><td>151/97 (0.64)</td><td>7/12</td><td>64</td><td>134/95 (0.71)</td><td>143/79 (0.55)</td><td>5/7</td></tr><tr><td>2.0</td><td>149/106 (0.71)</td><td>145/88 (0.61)</td><td>4/5</td><td>128</td><td>149/106 (0.71)</td><td>145/88 (0.61)</td><td>4/5</td></tr><tr><td>2.5</td><td>136/98 (0.72)</td><td>131/85 (0.65)</td><td>1/3</td><td></td><td></td><td></td><td></td></tr><tr><td>3.0</td><td>125/89 (0.71)</td><td>121/76 (0.63)</td><td>0/2</td><td></td><td></td><td></td><td></td></tr></table>

Table 13: Seed replication (F5), with every seed scored on the same problems. Seed 0 ran on the full band and seeds 1–2 on a 175-problem subsample, so an as-run comparison confounds seed with evaluation set. Restricting seed 0 to the same 175 problems costs no additional sampling and is given in the matched columns: the reversal is a seed effect, not a set effect.
<table><tr><td>method</td><td></td><td>seed n (matched)</td><td>learned</td><td>corrupted</td><td>CLR matched</td><td>CLR as run</td><td></td></tr><tr><td>STaR</td><td>0 1</td><td>175</td><td>27</td><td>19</td><td>0.704</td><td>0.711</td><td></td></tr><tr><td></td><td></td><td>175</td><td>23</td><td>17</td><td>0.739</td><td>0.739</td><td>range 0.45–0.74; net positive</td></tr><tr><td></td><td>2</td><td>175</td><td>22</td><td>10</td><td>0.455</td><td>0.455</td><td></td></tr><tr><td></td><td>0</td><td>175</td><td>20</td><td>11</td><td>0.550</td><td>0.607</td><td></td></tr><tr><td>maj.-vote</td><td>1</td><td>175</td><td>19</td><td>29</td><td>1.526</td><td>1.526</td><td>range 0.55–1.53; 2/3 net destructive</td></tr><tr><td></td><td>2</td><td>175</td><td>18</td><td>24</td><td>1.333</td><td>1.333</td><td></td></tr></table>

seed 0, full band: corrupted share 0.416 vs 0.378, Fisher p = 0.406. Pooling seeds as a fixed effect is not defensible with three of them, so we report the spread rather than a pooled p-value. ∆ accuracy by seed: STaR +0.056/+0.056/+0.069; maj.-vote +0.040/−0.028/−0.010.

## D EXPANSION STATISTICS, POWER AND OUT-OF-DOMAIN PROBES

The expansion half of the audit rests on a per-problem exact test against a pooled baseline. This appendix varies every choice that test leaves open (§D.1), itemizes the problems it is computed on (§D.2), gives the power behind its null result and the effect sizes behind the corruption counts (§D.3), and reports the two out-of-domain probes (§D.4).

## D.1 ROBUSTNESS OF THE THRESHOLD-FREE EXACT TEST

The test of Section 4.2 replaces one explicit parameter with four implicit ones: the error rate, the onesidedness, the number of baselines pooled and the base-rate strata. It also rests on an assumption, that the eleven baseline evaluations are exchangeable. This subsection varies each and checks the assumption. Nothing here required a new experiment.

Does the decision rule make the comparison? A false-discovery-rate procedure is adaptive: it admits a larger per-hypothesis cutoff in an arm containing many true effects, so in principle an arm with a large global uplift can purchase a more permissive standard for its own marginal problems, and a signal-free control can be held to a stricter one. The concern is well founded in general and does not apply here, for a reason that is checkable rather than arguable: the cutoff a procedure admits is kα/M, but the cutoff it attains is the k-th smallest p-value, and here those are 0.0095 for distillation and 0.0094 for majority-vote self-training. The two arms are held to the same standard to three decimal places. Table 14 repeats the comparison under Bonferroni, at fixed thresholds, and under a single procedure applied jointly across every arm and problem at once; the low-base counts move by at most three and the ordering never changes. The one qualification is that the frozen control’s zero is a property of a corrected rule: at an uncorrected $p \leq 0 . 0 5$ it returns 13 detections over 660 tests, which is the nominal rate and not a failure

What the interaction shows. The arms of the ladder are matched in stream, retained volume and evaluation, but not in how much they improve: the teacher moves accuracy +13 to +16 points and majority-vote self-training −1.5 to +3.3. A model in which every method lifts every problem by a common factor in log-odds would reproduce more detections for the teacher everywhere, and disproportionately more where the base rate is low, without any qualitative difference in kind. Table 15 tests that model and rejects it: the teacher moves low-base problems by a further 1.91 in log-odds beyond its own overall lift, where both self-training arms are indistinguishable from zero. The dissociation is a difference in what the methods reach, not only in how far they move.

![](images/7a2aa80d520b49bae3cec40ab1f701d911f8430599747ed5d803c01faa3730df.jpg)

![](images/717e4733a658b6efc08fb53d2f90d129eb2ef33218cd2cb33f99f141697b619b.jpg)

Figure 7: Two of the seven failures, on the runs where they occurred. Left: apparent transitions produced by a model that never trained, under two definitions of correctness (F1). A single greedy sample manufactures state changes on 6.7–18.9% of problems; the solve-rate estimator with a hysteresis band reports 0.0–0.8%. Right: truncation rate per checkpoint (F3). The distillation student adopted its teacher’s verbose style and began hitting the token cap, losing final answers and registering as capability loss; the self-evolution arms are flat, which is why their counts survive.  
![](images/86251c474802521760bab98beeef5884f807100fd8f635e27df3738554fe6b0a.jpg)

![](images/04672589b43a33e43316d569e81768857f1075b9db234f924634d28564d25c01.jpg)  
Figure 8: Transitions clear the noise floor by ∼11×, but the comparison between methods is not stable across seeds. Left: learned and corrupted counts on the 1,163-problem difficulty band with bootstrap intervals over problems; the shaded region is the frozen-model floor, design-matched to the arms’ four checkpoints rather than the conventional two-checkpoint control, which understates it by roughly half. Right: CLR across three training seeds, every seed scored on the same 175 problems (F5). STaR holds 0.45–0.74; majority-vote self-training spans 0.55 to 1.53, crossing the threshold at which a method destroys more than it creates.

An analytic null, and what measuring buys. Appendix C.2 reports that for the solve-rate ledger an analytic unchanged-policy null very nearly reproduces our measured floor, which invites the question of whether the expansion null could have been calculated rather than measured. It can be, and Table 17 does so. The calculation is optimistic at every threshold and increasingly so as m rises, because the base-unreached set is selected on a noisy zero. A problem observed at 0/1,408 has a plug-in success probability of exactly zero and is therefore assigned no chance of a later success,

when in truth it has a small one. At m=2 the calculation returns 0.044 against a measured 0.058.   
An author who derived the null instead of measuring it would certify the threshold that F7 rules out.

How many replicates a null needs. Our recommendation, that a multi-arm study already owns several frozen replicates, is only useful with a number attached to it. Table 18 re-estimates the ER null from every subset of our eleven baselines, exhaustively, since there are never more than 462 of them. At four replicates, which is what a three-arm study with a no-op control possesses, the estimate still ranges from 0.022 to 0.098: such a study could certify a null of 0.022 and reproduce F7 at a slightly larger sample size. The spread falls to 0.043 at seven replicates and 0.024 at nine, and it is nine before the estimate holds within ±0.02 of the eleven-replicate value. The recommendation should therefore be read as report the interval and the replicate count it came from, and a null one intends to certify against needs rather more than a three-arm study owns.

Are the eleven baselines exchangeable? They were run at different times against a hosted service whose batching we ourselves show to be non-deterministic (F1), so pooling them assumes something that has to be checked: that no run-level effect inflates the arm-versus-pool comparison while leaving the held-out null clean. Fitting the eleven per-problem counts against their pooled rate gives a Pearson dispersion of 0.966 on 500 degrees of freedom (a value of 1 is exactly binomial), with a median per-problem ratio of 0.907. The implied intraclass correlation is indistinguishable from zero and the design effect at k=128 is 1.000. The draws are exchangeable across evaluations, which is what the exact test assumes and what the pooling requires.

Zero is an estimate too. The held-out null returns no detection on any of the eleven replicates, which is a count and not a rate. At the replicate level the Clopper–Pearson bound is 0.24; over the 660 individual tests it is 0.0045. We report both rather than repeat, one level up, the failure F7 describes.

The small non-zero counts, and the teacher’s handicap. We state the small non-zero counts rather than round them away. They are what an unmatched comparison hides: applied to a stream of problems it can actually solve, self-training does occasionally raise a rarely-reached problem above the noise. It does so at a fifth of the teacher’s rate; the direction is unchanged, and the quantity that transfers is a rate rather than a categorical “never”. The comparison is also conservative in the teacher’s disfavor for a reason F3 predicts: the distillation arms truncate 36–46% of completions even at an 8,192-token cap, rising 4.2% to 53.4% across checkpoints on seed 0, against 4–15% for the self-training arms. F3 excludes an arm’s corruption counts because truncation inflates them; here it works the other way, since a truncated completion cannot be scored correct and so cannot produce a detection. The teacher’s counts are a floor on what it did, and the dissociation is understated rather than manufactured. Sensitivity is arithmetic rather than something to be bought: against a base of 0/1,408 an endpoint of 2/128 already clears raw alpha, which is also the smallest count that survived FDR in the distillation arm.

## D.2 EVERY BASE-UNREACHED AIME PROBLEM, ITEMIZED

Table 21 lists all 29 AIME problems that at least one arm found base-unreached at k=128, with the correct-sample count at that arm’s own baseline and at its final checkpoint. It is the whole of the expansion evidence in the paper, at the level of individual problems, and we give it in full because the counts are small enough that a reader should be able to check the conclusion rather than take it.

Three things can be read off it directly. The frozen control’s seven apparent expansions are all exactly one correct sample in 128; none of them is two. The distillation arm’s expansions are not: its eleven expansions at m ≥ 2 run 2, 3, 3, 6, 7, 8, 9, 9, 18, 33, 49, and the two largest (2025/21 at 49 and $2 0 2 5 / 2 6 ~ \mathrm { a t ~ 3 3 ) }$ are the same problems the frozen control $\mathrm { \ " e x p a n d e d " }$ with a single lucky draw. And the majority-vote arm’s three at m=1 reduce to one at $m { = } 2 , 2 0 2 5 / 2 6$ at two samples, which is why we report 1/21 rather than 3/21.

The three baseline columns differ (25, 21 and 22 problems at zero) because each arm’s checkpoint 0 is its own independent $k { = } 1 2 8$ evaluation of the same untrained model. Thirteen of the 29 rows are at zero in one arm’s baseline and at one, two or three samples in another’s. That disagreement is not a bug in the runs; it is the same phenomenon the whole section is about, showing up in the denominator instead of the numerator.

Table 14: The comparison does not depend on the decision rule. Low-base detections (pooled base rate $\leq 5 / 1 , 4 0 8 , n = 2 2 )$ under four rules applied identically to every arm, and under one joint procedure over all arms and problems at once. A false-discovery-rate procedure is adaptive, so an arm with many true effects admits a larger cutoff; the cutoffs actually attained here are 0.0095 for distillation’s 36 detections and 0.0094 for majority-vote self-training’s 19, so the two arms are held to the same standard. The frozen row is the leave-one-out null over all eleven replicates.
<table><tr><td>arm</td><td>BH, per arm</td><td>Bonferroni</td><td> $p \leq 0 . 0 1$ </td><td> $p \leq 0 . 0 5$ </td><td>joint BH</td></tr><tr><td>distillation, seed 0</td><td>10</td><td>7</td><td>10</td><td>10</td><td>9</td></tr><tr><td>distillation, seed 1</td><td>11</td><td>6</td><td>10</td><td>11</td><td></td></tr><tr><td>distillation, seed 2</td><td>8</td><td>7</td><td>7</td><td>8</td><td></td></tr><tr><td>maj.-vote SFT, seed 0</td><td>2</td><td>0</td><td>2</td><td>2</td><td>0</td></tr><tr><td>maj.-vote SFT, seed 1</td><td>1</td><td>1</td><td>1</td><td>1</td><td></td></tr><tr><td>maj.-vote SFT, seed 2</td><td>0</td><td>0</td><td>0</td><td>0</td><td></td></tr><tr><td>STaR</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>policy gradient</td><td>1</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td>frozen, 11 replicates</td><td>0</td><td>0</td><td>0</td><td>0-1</td><td>0</td></tr></table>

Joint BH is over 900 hypotheses (four arms and eleven frozen replicates × 60 problems), attained cutoff 0.00248. At an uncorrected $p \leq 0 . 0 5$ the frozen control is no longer at zero: it returns 13 detections over 660 tests, close to the nominal rate.

Table 15: The dissociation is not a by-product of the teacher’s larger gain. Logistic model on the per-problem counts, with terms for arm, base-rate stratum, before/after, and their interactions; the coefficient shown is the additional log-odds an arm moves a low-base problem beyond its own overall lift. A model in which every method improves problems by a common multiplicative factor predicts zero here. It is zero for both self-training arms and is not for the teacher.
<table><tr><td>term</td><td>β</td><td>95% CI</td><td>p</td></tr><tr><td>low-base × post (reference arm, STaR)</td><td>+1.17</td><td> $[ + 0 . 5 2 , + 1 . 8 2 ]$ </td><td> $1 0 ^ { - 3 }$ </td></tr><tr><td>distillation × low-base × post</td><td>+1.91</td><td>[+1.25, +2.56]</td><td> $1 0 ^ { - 8 }$ </td></tr><tr><td>majority-vote SFT × low-base ×post</td><td>+0.21</td><td>[−0.60, +1.03]</td><td>0.61</td></tr><tr><td>policy gradient × low-base × post</td><td>-0.49</td><td>[−1.51, +0.53]</td><td>0.34</td></tr></table>

The natural reading of that disagreement, that base-unreached sets must never be pooled across arms, inverts what the observation implies. Each arm’s baseline is a noisy draw from the same untrained model. Treating the baselines as different populations preserves the noise and makes the arms incomparable; pooling them removes it. Eleven evaluations give 1,408 draws per problem, one denominator for every arm, and a base-rate estimate an order of magnitude tighter than any single arm’s. Under the pooled baseline only 10 problems remain at zero, which is the size of the population 22 and 25 over-count. This table supplies the per-arm evidence for that argument; the analysis the paper rests on is the pooled test of Table 19, not the per-arm columns here.

Figure 9 plots every evaluation problem’s solve rate before evolution against its solve rate after. It is the consolidation claim without any aggregation: points rise above the diagonal, so the methods do help, but the left edge (problems the base model never reaches, the only ones whose movement would constitute new capability) stays essentially empty for every self-evolution arm and populates only under distillation. The plot also makes visible something the counts obscure: a dense cloud sits below the diagonal in the difficulty-band panels, and those are the corrupted problems.

## D.3 STATISTICAL POWER AND EFFECT-SIZE ENVELOPES

Transition metrics are supported by a much smaller effective sample than accuracy. Accuracy uses every problem; a transition can only be contributed by a problem near the decision boundary, and the fraction of such problems is a property of the benchmark rather than of the method. On the difficulty band we observe 0.20–0.22 transitions per problem, so 1,163 problems yield 233–255

Table 16: Sensitivity of the threshold-free test to the choices that remain. Low-base detections (n = 22) as the number of pooled baselines and the error rate vary. Pooling fewer baselines costs sensitivity in the arm with signal and never manufactures it in the arms without.
<table><tr><td></td><td colspan="4">baselines pooled</td><td colspan="3">FDR level</td></tr><tr><td>arm</td><td>3</td><td>5</td><td>7</td><td>11</td><td>0.01</td><td>0.05</td><td>0.10</td></tr><tr><td>distillation, seed 0</td><td>8</td><td>10</td><td>10</td><td>10</td><td>9</td><td>10</td><td>10</td></tr><tr><td>maj.-vote SFT, seed 0</td><td>0</td><td>1</td><td>1</td><td>2</td><td>0</td><td>2</td><td>2</td></tr><tr><td>STaR</td><td>1</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td><td>2</td></tr><tr><td>policy gradient</td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>1</td><td>1</td></tr></table>

Draws per problem at each pool size: 384, 640, 896, 1,408.

Table 17: An analytic null for the expansion rate, and why measuring it still matters. $\mathbb { E } [ \mathrm { E R } _ { m } ] =$ $\mathbb { E } _ { q } [ ( 1 - q ) ^ { k } \operatorname* { P r } ( X \geq m \mid k , q ) ] / \mathbb { E } _ { q } [ ( \hat { 1 } - q ) ^ { k } ]$ , with the distribution of q taken from the pooled 1,408- draw baseline, as a plug-in and under an empirical-Bayes beta prior. The calculation is systematically optimistic and increasingly so as m rises, because the base-unreached set is selected on a noisy zero: a problem observed at $^ { 0 / 1 , 4 0 8 }$ has a plug-in q of exactly zero and is assigned no chance of a later success. At the threshold that appears to repair the statistic, an author trusting the calculation would certify 0.044 and reach the same false conclusion.

<table><tr><td>m</td><td>measured</td><td>analytic, plug-in</td><td>analytic, beta-binomial</td></tr><tr><td>1</td><td>0.176</td><td>0.158</td><td>0.163</td></tr><tr><td>2</td><td>0.058</td><td>0.044</td><td>0.049</td></tr><tr><td>3</td><td>0.023</td><td>0.013</td><td>0.015</td></tr><tr><td>5</td><td>0.003</td><td>0.001</td><td>0.002</td></tr></table>

events per arm, against the approximately 249 per arm required to detect the difference in corrupted share we observe between STaR and majority-vote self-training (0.417 against 0.542 in the pilot) at 80% power and $\alpha = 0 . 0 5$ . The 112-problem pilot yields 24 events.

Three consequences follow. Report counts with intervals rather than ratios, since a ratio of small counts is unstable in a way its point estimate conceals: the CLR = 1.5 of Section 4.1 is 9 over 6, and the honest version of that headline is the transition rate, 7.5% of problems with a Wilson interval of [4.6%, 12.0%]. State power explicitly whenever a comparison is null. And note that seed variance, not problem sampling, dominates the uncertainty: bootstrap intervals over problems for majorityvote self-training’s CLR span [0.460, 0.785], while three training seeds span [0.55, 1.53]. Reporting only the former, which is what a single-seed study with bootstrap intervals does, understates the uncertainty by roughly a factor of three.

This is also why we decline to run a second backbone at the scale the budget allows. A difficulty band built for gpt-oss-20b at 300–400 problems would produce 60–90 transition events per arm against the 249 this subsection requires, so a null from it would be uninformative and a difference from it unreliable. Adding an underpowered arm to a paper whose fourth failure mode is underpowered arms would be the wrong trade, and we prefer to state the limitation.

Power. A null detection is uninformative without the power that produced it. Table 22 gives the detection probability for a base-unreached problem as a function of the true post-training rate and the number of pooled baselines. Pooling is what buys sensitivity: a single baseline needs seven correct samples in 128 before a problem can be detected at all, eleven need two. At eleven the test has 80% power against a true rate of 0.024, and the largest endpoint any self-training arm produced on a base-0 problem is 1/128. The claim the data support is a bound.

Effect sizes, against the envelope a frozen model produces. Table 23 and Figure 10 give the size of every transition event on the difficulty band against the largest solve-rate movement the frozen control produced over its twenty independent comparisons. That envelope is what separates a threshold flicker from a capability change, and it is reported alongside the counts because a count alone cannot make the distinction.

Table 18: How many frozen replicates a null needs before it settles. The pooled $\mathrm { E R _ { 2 } }$ null reestimated from every subset of G of our eleven baseline evaluations, exhaustively rather than sampled, since $\textstyle { \binom { 1 1 } { G } }$ never exceeds 462. Four replicates, which is what a three-arm study with a no-op control possesses for free, still admit estimates from 0.022 to 0.098, so such a study could certify a null of 0.022 and reproduce F7. The spread halves by seven and again by nine; it takes nine before the estimate stays within ±0.02 of the 0.058 that eleven give.
<table><tr><td>replicates G</td><td>subsets</td><td>ordered pairs</td><td>min</td><td>median</td><td>max</td><td>spread</td></tr><tr><td>2</td><td>55</td><td>2</td><td>0.000</td><td>0.048</td><td>0.120</td><td>0.120</td></tr><tr><td>3</td><td>165</td><td>6</td><td>0.015</td><td>0.056</td><td>0.103</td><td>0.088</td></tr><tr><td>4</td><td>330</td><td>12</td><td>0.022</td><td>0.058</td><td>0.098</td><td>0.077</td></tr><tr><td>5</td><td>462</td><td>20</td><td>0.028</td><td>0.057</td><td>0.091</td><td>0.063</td></tr><tr><td>7</td><td>330</td><td>42</td><td>0.035</td><td>0.058</td><td>0.078</td><td>0.043</td></tr><tr><td>9</td><td>55</td><td>72</td><td>0.043</td><td>0.059</td><td>0.066</td><td>0.024</td></tr><tr><td>11</td><td>1</td><td>110</td><td>0.058</td><td>0.058</td><td>0.058</td><td>0.000</td></tr></table>

![](images/23333ad15008b82446944d6aabcee8423a53577cc67be2ae69369461377ffb50.jpg)  
Figure 9: Per-problem solve rate before and after evolution. Each point is one evaluation problem; the diagonal is no change. The shaded strip at $\hat { \pi } _ { 0 } = 0$ contains the only problems whose improvement would constitute expansion rather than consolidation, and the counts in it apply the $m \geq 2$ gate against each arm’s own baseline. The AIME panels are the matched-ladder arms of Table 2, not the unmatched pair of Table 20, which is why the majority-vote count differs between the two. Self-training leaves the strip almost untouched; the distillation control does not.

## D.4 OUT-OF-DOMAIN PROBES

Both probes, three times each. Each out-of-domain probe was run three times against the same stored checkpoints, so any difference between replicates is measurement rather than training. One

Table 19: The threshold-free test: per-problem exact tests against a pooled baseline, with FDR control. Each arm’s endpoint count $( x / 1 2 8 )$ is tested against the untrained model’s pooled count (x/1,408, eleven independent evaluations) with a one-sided Fisher exact test at FDR 0.05 across all 60 AIME problems. Strata are the pooled base rate. The null holds out one base evaluation and treats it as the “after”; all eleven replicates return zero detections everywhere. Expansion lives in the two left columns, sharpening in the right one.
<table><tr><td>arm</td><td>base 0 (10)</td><td>base 1–5 (12)  ${ \bf b a s e } > 5 ( { \bf 3 8 } )$ </td><td></td><td>total</td></tr><tr><td>majority-vote SFT</td><td>0</td><td>2</td><td>17</td><td>19</td></tr><tr><td>distillation†</td><td>3</td><td>7</td><td>26</td><td>36</td></tr><tr><td>frozen null, 11 replicates</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

<sup>†</sup>positive control. Seed 0 of each arm; per-seed counts in Table 2. Paired exact test over discordant detections, distillation against majority-vote SFT: 19 to $2 , p = 2 . 2 \times 1 0 ^ { - 4 }$ . Combining the per-seed comparisons on the low-base pool rather than pooling detections across seeds: $p = 2 . 0 \times 1 0 ^ { - 6 }$ for distillation over majority-vote SFT, 0.0078 over STaR and 0.0019 over the policy-gradient arm. On the ten problems of exactly zero base rate the union over seeds is 5 against 2, which is not significant (p = 0.35).

Table 20: The expansion null is not a number, it is a distribution, and at m=2 it is not zero. $\mathrm { E R } _ { m }$ counts base-unreached problems that reach m correct samples after evolution. The null columns use every ordered pair of independent evaluations of the untrained model (110 pairs on AIME from eleven such evaluations, 12 on MATH-500), so every count in them is an artifact; “one $\mathrm { p a i r } ^ { \prime \prime }$ is the single frozen comparison a conventional study would run, and is what a study with one control reports. $\mathrm { A t } \ m { = } 2$ the pooled null is 0.058 and majority-vote self-training’s measured 0.048 sits below it. Even $m { = } 3$ has a null of 0.023; no threshold below $m { = } 5$ reaches zero.
<table><tr><td rowspan="2">m</td><td colspan="3">frozen null, AIME</td><td>null</td><td colspan="2">AIME arms</td><td rowspan="2">MATH-500 arms</td></tr><tr><td>one pair</td><td>range over 110</td><td>pooled [95%]</td><td>MATH-500</td><td>maj.-vote</td><td>Distillation†</td></tr><tr><td>1</td><td> $7 / 2 5 = 0 . 2 8 0$ </td><td>0.000-0.364</td><td> $0 . 1 7 6 \left[ 0 . 1 4 5 , 0 . 2 0 6 \right]$ </td><td>0.179</td><td> $3 / 2 1 = 0 . 1 4 3$ </td><td> $1 2 / 2 2 = 0 . 5 4 5$ </td><td>1/3, 0/3</td></tr><tr><td>2</td><td> $0 / 2 5 = 0 . 0 0 0$ </td><td>0.000-0.136</td><td>0.058 [0.038, 0.078]</td><td>0.128</td><td> $\mathbf { 1 / 2 1 } = 0 . 0 4 8$ </td><td> $1 1 / 2 2 = 0 . 5 0 0$ </td><td>0/3, 0/3</td></tr><tr><td>3</td><td> $0 / 2 5 = 0 . 0 0 0$ </td><td>0.000-0.087</td><td>0.023 [0.009, 0.038]</td><td>0.000</td><td> $0 / 2 1 = 0 . 0 0 0$ </td><td> $1 0 / 2 2 = 0 . 4 5 5$ </td><td>0/3, 0/3</td></tr><tr><td>5</td><td> $0 / 2 5 = 0 . 0 0 0$ </td><td>0.000-0.040</td><td>0.003 [0.000, 0.009]</td><td>0.000</td><td> $0 / 2 1 = 0 . 0 0 0$ </td><td> $8 / 2 2 = 0 . 3 6 4$ </td><td>0/3, 0/3</td></tr></table>

<sup>†</sup>positive control, not a self-evolution method. Intervals are a cluster bootstrap over base evaluations, since the pairs share evaluations and are not independent. Eleven clusters is few enough that the bootstrap’s own coverage is not guaranteed, so Appendix D.1 reports the estimate’s spread over every subset of the eleven instead, which needs no asymptotics. The band has no base-unreached problems by construction and is omitted. Arm columns use each arm’s own baseline, as prior work does; Table 19 replaces that with a shared one.

Table 21: Every AIME problem the base model does not reach, with its correct-sample count under each arm. Counts are out of k=128. A row contributes to $\mathrm { E R } _ { m }$ for an arm when its baseline column is 0 and its endpoint column is at least $m .$ . The frozen control is the same untrained model evaluated twice, so its seven 0 → 1 rows are the artifact of Section $4 . 2 ;$ note that every one of them is exactly 1, and that no arm’s genuine expansion is. Bold marks a row that counts as expanded at m ≥ 2.
<table><tr><td></td><td colspan="2">frozen</td><td colspan="2">maj.-vote</td><td colspan="2">distill.</td><td colspan="2"></td><td colspan="2">frozen</td><td colspan="2">maj.-vote</td><td colspan="2">distill.</td></tr><tr><td>problem</td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $t _ { 0 }$ </td><td> $t _ { 3 }$ </td><td> $t _ { 0 }$ </td><td> $t _ { 3 }$ </td><td>problem</td><td> $t _ { 0 }$ </td><td> $t _ { 1 }$ </td><td> $t _ { 0 }$ </td><td></td><td> $t _ { 3 }$ </td><td></td><td> $t _ { 3 }$ </td></tr><tr><td>2025/10</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2026/10</td><td>2</td><td>0</td><td></td><td>0</td><td>0</td><td>1</td><td>1</td></tr><tr><td>2025/12</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2026/11</td><td>0</td><td>1</td><td></td><td>0</td><td>0</td><td>0</td><td>9</td></tr><tr><td>2025/13</td><td>0</td><td>1</td><td>2</td><td>0</td><td>0</td><td>1</td><td>2026/13</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>1</td><td>20</td></tr><tr><td>2025/14</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2026/14</td><td>0</td><td>1</td><td></td><td>0</td><td>1</td><td>0</td><td>9</td></tr><tr><td>2025/17</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>6</td><td>2026/15</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>2025/21</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>49</td><td>2026/17</td><td>0</td><td>0</td><td></td><td>1</td><td>0</td><td>1</td><td>0</td></tr><tr><td>2025/22</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>2026/18</td><td>0</td><td>0</td><td></td><td>2</td><td>3</td><td>2</td><td>0</td></tr><tr><td>2025/23</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>8</td><td>2026/22</td><td>1</td><td>1</td><td></td><td>0</td><td>0</td><td>2</td><td>30</td></tr><tr><td>2025/24</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>18</td><td>2026/23</td><td>1</td><td>3</td><td></td><td>2</td><td>2</td><td>0</td><td>7</td></tr><tr><td>2025/25</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>3</td><td>2026/27</td><td>0</td><td>0</td><td></td><td>0</td><td>1</td><td>1</td><td>6</td></tr><tr><td>2025/26</td><td>0</td><td>1</td><td>0</td><td>2</td><td></td><td>0 33</td><td>2026/28</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>2025/27</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2026/29</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>2025/29</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2026/30</td><td>0</td><td>0</td><td></td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>2025/6</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td><td>2026/9</td><td>0</td><td>0</td><td></td><td>1</td><td>0</td><td>1</td><td>1</td></tr><tr><td>2025/9</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Base-unreached counts: frozen $2 5 ,$ , majority-vote 21, distillation $^ { 2 2 ; }$ the three differ because each arm’s baseline is its own independent k=128 evaluation, which is the point of Section 4.2. The majority-vote columns are the unmatched on-set arm of Table 20, the arm whose ER that section reports.

run of the refusal probe is capable of producing a publishable safety finding; two further runs dissolve it, and what the data support is a bound rather than a null result: any refusal change here is smaller than about eleven points. MMLU-Redux is better behaved, but its effect is smaller than a single replicate suggests, the +10.6 points of that replicate being the largest of six measurements whose median is nearer +4.

F6. An underpowered probe produces a phantom safety result. A 10-prompt refusal probe shows the refusal rate falling 10 points across checkpoints, corroborating existing reports of safety degradation under self-training (Shao et al., 2026b; Qi et al., 2024). At 50 prompts the effect disappears and the 10-prompt trajectory is non-monotonic, the signature of noise. The probe is underpowered twice over and clustered besides; Section 5 reports what survives at the larger size.

Table 22: Power of the per-problem test on a base-unreached problem. Probability of detection against the true post-training pass rate, at k=128, for a problem with no successes in the pooled baseline, under the cutoff the procedure attains. Pooling is what buys the sensitivity: one baseline requires seven correct samples before a problem can be detected, eleven require two. The null result of Section 5 is therefore a bound, no self-training arm having lifted a base-0 problem above a 2.4% pass rate, and not an absence of measurement.
<table><tr><td>baselines pooled</td><td>draws</td><td>min. successes</td><td>0.01</td><td>0.02</td><td>0.03</td><td>0.05</td><td>0.08</td><td>0.12</td></tr><tr><td>1</td><td>128</td><td>7/128</td><td>0.00</td><td>0.01</td><td>0.09</td><td>0.46</td><td>0.89</td><td>1.00</td></tr><tr><td>3</td><td>384</td><td>4/128</td><td>0.04</td><td>0.25</td><td>0.54</td><td>0.89</td><td>0.99</td><td>1.00</td></tr><tr><td>5</td><td>640</td><td>3/128</td><td>0.14</td><td>0.47</td><td>0.74</td><td>0.96</td><td>1.00</td><td>1.00</td></tr><tr><td>11</td><td>1,408</td><td>2/128</td><td>0.37</td><td>0.73</td><td>0.90</td><td>0.99</td><td>1.00</td><td>1.00</td></tr></table>

Smallest true rate detectable at 80% power with eleven baselines: 0.024 under the attained cutoff, 0.034 under Bonferroni. Observed endpoints on the ten base-0 problems: distillation 0, 0, 0, 0, 0, 0, 0, 2, 6, 18; majority-vote SFT 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, all of 128.

Table 23: Effect sizes for transition events on the difficulty band. Counts of events whose solverate change exceeds each magnitude, against the envelope of the frozen control. The control row is the worst value each column takes over the twenty ordered pairs of independent evaluations of the untrained model, rather than a single pair: across all twenty, one problem moved past 0.2 and none past 0.234, so 0.234 is the measured zero-false-positive gate and the bolded column is what survives it.
<table><tr><td>arm</td><td>event</td><td>total</td><td>&gt; 0.1</td><td>&gt; 0.2</td><td>&gt; 0.234</td><td>&gt; 0.35</td><td>median</td><td>max</td></tr><tr><td rowspan="2">STaR</td><td>corrupted</td><td>106</td><td>90</td><td>69</td><td>58</td><td>37</td><td>0.258</td><td>0.781</td></tr><tr><td>learned</td><td>149</td><td>133</td><td>107</td><td>一</td><td>61</td><td>0.297</td><td>0.812</td></tr><tr><td rowspan="2">maj.-vote</td><td>corrupted</td><td>88</td><td>86</td><td>69</td><td>60</td><td>37</td><td>0.320</td><td>0.750</td></tr><tr><td>learned</td><td>145</td><td>133</td><td>104</td><td>一</td><td>62</td><td>0.297</td><td>0.773</td></tr><tr><td>frozen, worst of 20 pairs</td><td>any change</td><td>1,163</td><td>108</td><td>1</td><td>0</td><td>0</td><td>0.039</td><td>0.234</td></tr></table>

Table 24 reports both probes in full. MMLU-Redux (Gema et al., 2025) is evaluated on 200 questions per checkpoint. The refusal probe comprises 50 requests spanning fraud, violence, harassment, child safety, cyber-intrusion, hate, self-harm, regulated goods and misinformation, each sampled four times per checkpoint for 200 observations. Prompts are deliberately mild and non-operational: they exist to trigger a refusal, not to elicit anything usable. A response counts as a refusal if a refusal marker appears in its first 400 characters, a threshold chosen so that a model which complies and then appends a disclaimer is not scored as having refused. This marker rule is a limitation and the negative result should be read as scoped to it: a model that shifts from explicit refusal to a soft deflection without a marker phrase would be scored as having complied, and we did not audit for that.

The nine categories are represented by five or six prompts each and were written to span the standard refusal taxonomy without approaching an operational threshold in any of them: requests for the shape of a harmful action rather than the means to perform it. We release the probe with the code, together with one record per generation, so that the result can be re-scored under a different rule. Replicate 1 was logged at per-checkpoint aggregate resolution, whereas replicates 2 and 3 record full per-generation outputs; the cluster-robust analysis below is therefore computed on the latter two.

A naive analysis of this probe fails twice over, and the second failure is the consequential one. Four samples of one prompt are not four independent observations, so a z-test on 200 Bernoulli trials is the wrong instrument; aggregating to 50 prompt-level proportions gives a measured design effect of 1.8–2.8×, and a cluster-robust standard error of 0.044–0.057 rather than the 0.032–0.035 a naive analysis assumes. That alone would only widen an interval. What it reveals is worse: at 50 prompts the smallest change this probe can detect at 80% power is about 0.11, and the change under discussion is about 0.10. The probe is being asked a question it cannot answer.

![](images/16b7183b490c7218d44c9212e061a756051af3abc55986144d6d7e8187612f64.jpg)

![](images/c2935523a4571c992bec05c8b50b1799ba6cbb8df55d395ef968909e81eb935a.jpg)  
Figure 10: Corruption is large movement, not threshold flicker. Left: distribution of $| \Delta \hat { \pi } ( p ) |$ for the frozen control over all 1,163 problems and for each arm’s corruption events, each normalized to its own total; over the twenty independent frozen comparisons the records contain, no problem moves by more than 0.234. Right: corruption counts surviving a minimum-drop gate. Over half of each arm’s events lie beyond the frozen envelope entirely.

Replication settles the point empirically. Three executions of the identical probe against the identical stored checkpoints give, for majority-vote self-training, refusal changes of −0.015 (not significant), −0.115 (t = −2.79 clustered, significant) and −0.005. Nothing about the model differs across them; only the sampling does, and the one significant reading is replicated by neither neighbor. We therefore decline to claim that self-training causes no safety erosion here, and equally decline to claim the opposite: what the data support is a bound, that any change is smaller than roughly eleven points, which is compatible both with nothing happening and with an effect of practical size. Detecting the latter would need a probe several times larger, and prompts, not samples, are the unit that has to grow.

MMLU-Redux is only slightly better placed. STaR gains in all three replicates $( + 6 . 3 , + 7 . 0$ and +2.0 points; paired McNemar on the second, 25 gained against 11 lost, $p = 0 . 0 2 9 )$ , which is the one out-of-domain result here that survives replication and a paired test. Majority-vote self-training gives +10.6, then $+ 2 . 0 \ ( p \ = \ 0 . 5 8 )$ and then +1.5 (p = 0.72), and the untrained baseline itself moves between 0.550 and 0.645 across replicates. A single greedy draw on 200 questions is not a stable anchor for a drift claim.

Table 24: Out-of-domain probes, run three times each against the same stored checkpoints. Any difference between replicates is measurement, not training. Refusal changes are prompt-level means with cluster-robust intervals; MMLU is tested with paired McNemar on the same questions. The one significant refusal result does not replicate on either side of it, and the largest MMLU gain is the largest of six measurements of a smaller effect.
<table><tr><td>probe</td><td>arm</td><td>replicate 1*</td><td>replicate 2</td><td>replicate 3</td></tr><tr><td rowspan="2">MMLU-Redux (n=200)</td><td>STaR</td><td>+0.063</td><td> $+ 0 . 0 7 0 ( p = 0 . 0 2 9 )$ </td><td> $+ 0 . 0 2 0 \left( p = 0 . 6 0 \right)$ </td></tr><tr><td>maj.-vote</td><td>+0.106</td><td> $+ 0 . 0 2 0 \left( p = 0 . 5 8 \right)$ </td><td> $+ 0 . 0 1 5 \left( p = 0 . 7 2 \right)$ </td></tr><tr><td rowspan="2">refusal (n=50 prompts)</td><td>STaR</td><td>-0.050</td><td> $- 0 . 0 1 0 \left[ - 0 . 0 8 5 , 0 . 0 6 5 \right]$ </td><td> $- 0 . 0 0 5 \left[ - 0 . 0 6 8 , 0 . 0 5 8 \right]$ </td></tr><tr><td>maj.-vote</td><td>-0.015</td><td> $\mathbf { - 0 . 1 1 5 } \left[ \mathbf { \bar { - } 0 . 1 9 6 , - 0 . 0 3 \bar { 4 } } \right]$ </td><td> $- 0 . 0 0 5 \left[ - 0 . 0 8 0 , 0 . 0 7 0 \right]$ </td></tr><tr><td>refusal, underpowered (n=10)</td><td>STaR</td><td>—0.100, non-monotonic</td><td></td><td>the F6 probe</td></tr></table>

<sup>∗</sup>replicate 1 is recorded at per-checkpoint aggregate resolution rather than per generation, so it carries no interval. Design effect 1.8–2.8×; cluster-robust SE 0.044–0.057 against a naive 0.032–0.035. Minimum detectable refusal change at 50 prompts and 80% power: 0.09–0.12. Untrained MMLU baseline across replicates: 0.581–0.610 (STaR arm), 0.550–0.645 (majority-vote arm).

## E EXTENDED RELATED WORK, AND THE SCOPE OF THE CLAIMS

## E.1 RELATED WORK, CONTINUED

The RLVR side of the debate. The consolidation question is posed most sharply in reinforcement learning from verifiable rewards, and the analyses there bear directly on ours: how little of the model RLVR actually moves (Mukherjee et al., 2025), how easily entropy collapses (Cui et al., 2025), and how weak the training signal can be while still producing gains (Shao et al., 2026a; Wang et al., 2025). The underlying algorithms are those of Shao et al. (2024); Guo et al. (2025); Yu et al. (2025); Liu et al. (2025b). Our policy-gradient arm sits in this family, which is why we ran it: an audit that positions itself against an RLVR literature and then measures only supervised fine-tuning is auditing something else.

Reconciling with prior reports. Lin et al. (2026) report that TTRL corrupts more than it learns; our seed-0 run finds the opposite and our other two seeds agree with them. We read this as one axis rather than a contradiction: corruption requires both unreliable supervision and prior competence on the affected problems, and that regime is narrow. Their backbone is weaker and its majority vote correspondingly less accurate (ours is 91–95% correct on the band). That places them inside the regime while ours sits at its edge, which is why our result moves with the seed. The account is testable, predicts that the corrupting regime is a moderately competent model rather than a weak or a strong one, and agrees with a sharpening view (Huang et al., 2025): when verification is nearly perfect, sharpening is nearly harmless.

Parametric self-evolution. STaR (Zelikman et al., 2022) established the loop most later methods vary: sample rationales, keep those reaching the correct answer, fine-tune, repeat (Yuan et al., 2023; Singh et al., 2024; Wang et al., 2023b; Huang et al., 2023). TTRL (Zuo et al., 2025) discards labels entirely, treating a majority vote over the model’s own samples as a reward, which is self-consistency (Wang et al., 2023a) used as a training rather than an inference-time signal, and label-free variants have followed (Zhou et al., 2026; Moradi et al., 2025; Acikgoz et al., 2025). Others replace the filter with a learned verifier (Hosseini et al., 2024; Chen et al., 2026), the model’s own reward judgments (Yuan et al., 2024), self-play (Chen et al., 2024; Zhao et al., 2025), debate (Srivastava et al., 2025), generated curricula (Huang et al., 2026), language feedback (Lu et al., 2024), or latent rationales (Zelikman et al., 2024); SEAL (Zweiger et al., 2025) learns a policy over weight edits, and surveys catalog the space (Tao et al., 2024; ang Gao et al., 2026). Inference-time self-correction is a distinct question and does not reliably work (Madaan et al., 2023; Huang et al., 2024).

Degradation under self-training. Lin et al. (2026) report that for TTRL the problems corrupted outnumber those learned and that recovery is rare once a majority vote settles on a wrong answer; Yu et al. (2026a) frame this as capability erosion, and Zhao et al. (2026) show that self-evolving agents often fail to use accumulated experience causally. The backdrop is catastrophic forgetting (Kirkpatrick et al., 2017; Luo et al., 2025), which LoRA mitigates but does not remove (Hu et al., 2022; Biderman et al., 2024); model collapse under recursive training on generated data (Shumailov et al., 2024; Alemohammad et al., 2024; Gerstgrasser et al., 2024); and safety erosion under finetuning (Qi et al., 2024; Shao et al., 2026b), which motivates our out-of-domain probes.

Evaluation methodology, test-time adaptation and distillation. pass@k estimation (Chen et al., 2021) underpins our per-problem solve rates; Miller (2024) argue for reporting evaluation uncertainty at all and Hariri et al. (2026) for a Bayesian posterior in place of the pass@k point estimate, though neither addresses transitions or supplies a null. Seed sensitivity in fine-tuning is long documented (Dodge et al., 2020) and reappears as our most expensive failure mode. Contamination (Zhang et al., 2024; Xu et al., 2024; 2025; Sun et al., 2025a) motivates our use of AIME alongside the contaminated but comparable MATH-500 (Lightman et al., 2024; Hendrycks et al., 2021; Cobbe et al., 2021); probes use MMLU-Redux (Gema et al., 2025); intervals are Wilson (Wilson, 1927; Brown et al., 2001) and percentile bootstrap (Efron, 1979), count comparisons Fisher’s exact test (Fisher, 1922). Majority-vote self-training sits within a literature on adapting parameters at inference (Sun et al., 2025b; Moradi et al., 2025; Gozeten et al., 2025), adjacent to test-time compute scaling that changes no parameters (Snell et al., 2024; Muennighoff et al., 2025); our positive control is ordinary knowledge distillation (Hinton et al., 2015; Hsieh et al., 2023).

## E.2 CONCURRENT WORK, IN FULL

Concurrent work, in full. Two contemporaneous papers overlap with ours. Strozzi (2026) ask our headline question, whether teacher-free self-training acquires capability or only expresses existing capability, and report a pass@K crossover in which the base model wins at K=64 on a DSL domain with a free verifier. We regard our consolidation result as a replication in a different domain and supervision regime, with a positive control they do not run. Yuan et al. (2026) track per-problem pass@256 boundary entry and exit across training, structurally our ledger, and their Appendix C gives an analytic sampling-noise null: a binomial calculation under an unchanged policy. Ours is measured, by pushing an unchanged model through the identical pipeline, and we can say precisely how much that buys: less than the phrase “measure your null” suggests. For the solve-rate estimator we recommend, a binomial simulation at each problem’s pooled base rate reproduces our measured floor closely (median $3 / 4$ against $4 / 3$ transitions, envelope 0.203 against 0.234), so on that protocol the analytic null is adequate. The two diverge exactly where the noise is not binomial. That is the single-decode protocol, where an analytic null predicts nothing and the measurement finds 18.9% of problems changing state (Section 4.1), and it is also where no analytic null has been worked out, which is how we found that the expansion statistic’s null is non-zero at any threshold below $m { = } 5$ Neither paper reports a null for it.

## E.3 LIMITATIONS, IN FULL

Limitations, in full. Three evolution rounds and roughly 270 optimizer steps, far short of the schedules of the methods we position against, so we cannot speak to accumulation. Expansion rests on 22 low-base problems on a single benchmark, 10 of them in the uncontaminated 2026 half: enough for the dissociation to be significant, not to be precise. Corruption rests on the difficulty band, drawn from MATH training problems that are held out from our stream but almost certainly not from the backbone’s pretraining data, so it may in part be the perturbation of memorized answers rather than of reasoning; the AIME arms that avoid this have no corruption power. The additional seeds for the band arms were run on a 175-problem subsample rather than the full 1,163, and every seed is scored on those 175 problems. One backbone family carries every substantive claim. A second (gpt-oss-20b) was characterized and its floor measured cleanly, but a band built for it at an affordable scale would have yielded roughly a third of the transition events our own power analysis (F4) says such a comparison needs. We therefore state cross-backbone generalization as untested rather than answer it with an underpowered arm. The distillation control is matched in stream, retained volume and evaluation, but teacher capability and verbosity necessarily still differ, and F3 shows the latter is real. Finally, the evolution stream is far easier than AIME (STaR retains about 90% of it under ground-truth filtering), so a method that trains only on what it already solves is a priori unlikely to expand into AIME-hard territory. Methods that generate their own curricula exist precisely to escape that, and our null does not speak to them.

## F CASE STUDIES: THE FAILURE MODES IN THE MODEL’S OWN WORDS

Every count in this paper is an aggregate over generations that a reader never sees. This appendix opens five of them. Each transcript is the model’s own output, taken from the released records at the character offsets stated, and its title bar carries the provenance needed to find it again: which arm, which checkpoint, how long the completion ran, and how the grader scored it. Frame color repeats the verdict: green for graded correct, red for graded incorrect, gray where no verdict applies.

Two mechanical changes are applied. Non-ASCII decoration, and the models are fond of emoji section markers, is transliterated so that pdfLAT<sub>E</sub>X can set it, and blank lines are dropped. Nothing inside a block is elided and no line is rewrapped by us; where a line runs past the margin the typesetter breaks it and marks the continuation with ,→. The prompt is identical in every case and is given in Appendix A.1.

## F.1 CASE 1: ONE MODEL, ONE PROMPT, TEMPERATURE ZERO, TWO DIFFERENT ANSWERS

This is F1 with nothing else going on. The model is the untrained backbone. The prompt is byteidentical. Decoding is greedy at $\bar { T } = 0$ with a fixed seed. The two completions below are the two passes of the frozen-model control, and the single-decode ledger scores the pair as one corrupted problem: a capability lost between two evaluations of a model that never trained.

PROBLEM math500/test/geometry/226

reference answer 180

In circle J, HO and HN are tangent to the circle at O and N. Find the number of degrees in the sum of m∠J and m∠H.

An Asymptote figure accompanies the problem and is passed to the model verbatim, as it appears in MATH-500.

The two completions are identical for their first 1523 characters, through the setup and the first two solution steps, and then diverge mid-sentence:

### Step 3: Use triangle angle sum   
In triangle \$ JHO \$:   
\$\$   
\angle J + \angle JHO + \angle HOJ = 180ˆ\circ   
\$\$   
We know:   
- \$ \angle JHO = 90ˆ\circ \$   
- \$ \angle HOJ = \angle JON \$, which is the angle between two radii

frozen control • checkpoint 0 • 840 tokens • graded correct

, so it's the central angle subtended by arc \$ ON \$.   
But we are not given any specific arc measures, so we can't compute that directly.   
However, we are only asked for the <sub>\*\*</sub>sum<sub>\*\*</sub> of \$ \angle J \$ and \$ \angle H \$, not the   
,→ individual measures.   
Let's denote:   
- \$ \angle J = x \$   
- \$ \angle H = y \$

frozen control • checkpoint 1 • 1,595 tokens • graded incorrect

to the points of tangency.   
So:   
\$\$   
\angle J + 90ˆ\circ + \angle HOJ = 180ˆ\circ   
\Rightarrow \angle J + \angle HOJ = 90ˆ\circ \tag{1}   
\$\$   
Similarly, in triangle \$ JHN \$:   
\$\$   
\angle J + \angle JHN + \angle HNJ = 180ˆ\circ   
\Rightarrow \angle J + \angle HNJ = 90ˆ\circ \tag{2}   
\$\$

The paths do not reconverge:

frozen control • checkpoint 0 • 840 tokens • final answer   
\$\$   
Substitute known values:   
\$\$   
x + y + 90ˆ\circ + 90ˆ\circ = 360ˆ\circ   
\$\$   
\$\$   
x + y + 180ˆ\circ = 360ˆ\circ   
\$\$   
\$\$   
x + y = 180ˆ\circ   
\$\$   
### [ok] Final Answer:   
\$\$   
\boxed{180}   
\$\$   
So, the sum of the measures of \$ \angle J \$ and \$ \angle H \$ is \$ \boxed{180} \$ degrees.

frozen control • checkpoint 1 • 1,595 tokens • final answer   
\$\$   
### Step 6: Final calculation   
We now know:   
- \$ \angle J = 60ˆ\circ \$   
- \$ \angle H = 180ˆ\circ - 2\angle J = 180ˆ\circ - 120ˆ\circ = 60ˆ\circ \$   
So:   
\$\$   
m\angle J + m\angle H = 60ˆ\circ + 60ˆ\circ = \boxed{120ˆ\circ}   
\$\$   
### [ok] Final Answer:   
\$\$   
\boxed{120}   
\$\$

Pass 1 reaches the correct 180<sup>◦</sup> by the quadrilateral angle sum. Pass 2 commits to a chain of triangle relations, invents an isosceles constraint the figure does not support, and answers 120. Neither generation is pathological; the second is longer and, read on its own, no less plausible.

What produced the divergence is not sampling, of which there is none at T = 0, but floating-point non-associativity under a batching regime the caller does not control (He & Lab, 2025). Two forward passes over the same tokens with different batch composition reduce in different orders, perturb a logit in the last decimal place, and flip an arg-max on a near-tie. Everything after that token is a different generation.

This is the whole mechanism behind CLR = 1.5 on an untrained model (Section 4.1). On MATH-500 it happens to 15 of 200 problems between two passes; on the difficulty band, to 220 of 1,163. An audit whose state variable is a single greedy decode is measuring this.

## F.2 CASE 2: THE SAME PROBLEM, “EXPANDED” ONCE BY A FROZEN MODEL AND ONCE BY A TEACHER

This case carries F2. The problem is one the base model does not reach: zero correct in 128 samples at the frozen control’s first pass, and zero again in the distillation arm’s own baseline. Under the m = 1 criterion it is then scored as expanded twice: once by the frozen model, which did not train and produced a single correct sample on its second pass, and once by the distilled student, which did train.

## PROBLEM aime2025/21

reference answer 237

Let A be the set of positive integer divisors of 2025. Let B be a randomly selected subset of A. The probability that B is a nonempty set with the property that the least common multiple of its elements is 2025 is <sup>m</sup><sub>n</sub> , where m and n are relatively prime positive integers. Find m + n.

The base model’s failure is systematic rather than random. It computes the numerator correctly and then divides by $2 ^ { 1 5 } - 1$ instead of 2<sup>15</sup>:

untrained backbone • checkpoint 0 • 2,638 tokens • representative of 128 samples • graded incorrect   
\$\$   
\$\$   
\text{GCD}(425, 109) = \text{GCD}(109, 425 \mod 109) = \text{GCD}(109, 107)   
\$\$   
\$\$   
\text{GCD}(109, 107) = \text{GCD}(107, 2) = \text{GCD}(2, 1) = 1   
\$\$   
So the GCD is 1. Therefore, the fraction is already in lowest terms.   
### <sub>\*\*</sub>Final Answer<sub>\*\*</sub>   
The probability is:   
\$\$   
\frac{30240} {32767}   
\$\$   
So \$ m = 30240 \$, \$ n = 32767 \$   
\$\$   
m + n = \boxed{63007}

One sample in 128 from the frozen model’s second pass does not make the slip:

frozen control • checkpoint 1 • 2,027 tokens • the only one of 128 • graded correct   
- \$ 32768 = 2ˆ{15} \$   
So:   
\$\$   
\gcd(27904, 32768) = 2ˆ8 = 256   
\$\$   
So:   
\$\$   
\frac{27904} {32768} = \frac{109}{128}   
\$\$   
Check that \$ \gcd(109, 128) = 1 \$ (since 109 is prime)   
### Step 9: Final Answer   
We are given that the probability is \$ \frac{m}{n} \$, where \$ m = 109 \$, \$ n = 128 \$   
So:   
\$\$   
m + n = 109 + 128 = \boxed{237}   
\$\$   
### [ok] Final Answer:   
\$\$   
\boxed{237}   
\$\$

And the distilled student, which reaches this problem in 49 of 128 samples:

distillation • checkpoint 3 • 2,853 tokens • one of 49 of 128 • graded correct

```markdown
### Probability
\[
P = \frac{N}{2ˆ{15}} = \frac{27904} {32768}
\]
The greatest common divisor of numerator and denominator is \(256\):
\[
\frac{27904} {32768}= \frac{27904/256}{32768/256}= \frac{109}{128}.
\]
Hence \(m=109\) and \(n=128\), with \(\gcd(m,n)=1\).
Finally,
\[
m+n = 109+128 = 237.
\]
\[
\boxed{237}
\]
```

Read the three together. The difference between the frozen model’s failure and its one success is a single character: 32767 against 32768. Nothing was learned between the two passes, because nothing happened between the two passes. The lucky sample is the tail of a Bernoulli process with a small success probability, and observing it once in 128 draws bounds that probability at $q \leq 0 . 0 2 3$ rather than establishing it was zero before.

Two conclusions follow, and they pull in opposite directions, which is why we state both. First, the m = 1 criterion cannot distinguish the second block from the third: on the counting rule used in prior work, the frozen model and the distilled student both “expanded” this problem. Requiring $m \geq 2$ separates them, because the frozen model produces exactly one success here and the student produces 49. Second, and this bounds the headline claim, $\hat { \pi } _ { 0 } ( p ) = 0 ~ \mathrm { a t } ~ k = 1 2 8$ does not mean the base model could never solve this problem. It means the base model has a systematic arithmetic habit that costs it this problem almost every time. We therefore write “base-unreached at $k ^ { \prime \prime }$ throughout rather than “unsolvable”, and the claim the paper defends is about what a fixed sampling budget reaches, not about the support of the base distribution.

Appendix D.2 lists all 29 AIME problems that at least one arm found base-unreached, with their counts under each arm, so that this reasoning can be checked against every case rather than the one displayed here.

## F.3 CASE 3: A COMPLETION THAT RAN OUT OF BUDGET, SCORED AS A LOST CAPABILITY

F3. The generation below is from the distillation arm at checkpoint 1 on the difficulty band, one of the 65.3% of completions at that checkpoint that hit the 2,048-token cap.

Four distinct integers a, b, c and d have the property that when added in pairs, the sums 10, 18, 19, 20, 21 and 29 are obtained. What are the four integers in increasing order?

distillation • checkpoint 1 • 2,048 tokens • cap reached • no answer extracted   
z+w = 22 => (19 - x) + (21 - y) = 22 => 40 - (x+y) = 22 => x+y = 18. Again contradicts 10.   
Maybe x+z = 18, y+w = 20? Then w = 20 - y.   
z+w = 22 => (18 - x) + (20 - y) = 22 => 38 - (x+y) = 22 => x+y = 16. Still not 10.   
Maybe x+z = 20, y+w = 21? Then z = 20 - x, w = 21 - y.   
z+w = 22 => (20 - x) + (21 - y) = 22 => 41 - (x+y) = 22 => x+y = 19. Not 10.   
Maybe x+z = 19, y+w = 20? Then w = 20 - y, z = 19 - x.   
z+w = 22 => (19 - x) + (20 - y) =

The completion ends mid-expression. There is no \boxed{}, no stated answer and no final sentence, so the extractor returns nothing and the grader scores the sample incorrect: correct behavior for a grader, and the wrong summary of what happened. The model has not lost the ability to solve this problem. It has adopted its teacher’s habit of enumerating cases in prose and has run out of room.

Aggregated over the arm, this scores a method that improves from 0.656 to 0.845 on completions that finish as the most destructive arm in the study, at CLR = 1.10 and 177 corruptions. The diagnostic that catches it is one number per checkpoint, the truncation rate, and it costs nothing to log. Without it, nothing in the ledger distinguishes this case from Case 4.

## F.4 CASE 4: A CAPABILITY MAJORITY-VOTE SELF-TRAINING DESTROYED

By contrast, a real corruption, on an arm whose truncation rate never moves (16.6% to 16.0%). The base model solves this problem in 104 of 128 samples; after three rounds of majority-vote self-training, in 8.

PROBLEM mathtrain/train/number theory/1075

reference answer 2

Convert 199 to base 2. Let x be the number of zeros and y be the number of ones in base 2. What is the value of y − x?

untrained backbone • checkpoint 0 • 438 tokens • 104/128 correct • graded correct   
\$\$   
199\_{10} = 11000111\_2   
\$\$   
### <sub>\*\*</sub>Step 2: Count the number of ones (y) and zeros (x)<sub>\*\*</sub>   
Binary representation:   
\$\$   
11000111\_2   
\$\$   
Let's count:   
- Number of <sub>\*\*</sub>ones (y)<sub>\*\*</sub>: There are <sub>\*\*</sub>5 ones<sub>\*\*</sub>   
- Number of <sub>\*\*</sub>zeros (x)<sub>\*\*</sub>: There are <sub>\*\*</sub>3 zeros<sub>\*\*</sub>   
### <sub>\*\*</sub>Step 3: Compute \$ y - x \$<sub>\*\*</sub>   
\$\$   
y - x = 5 - 3 = \boxed{2}   
\$\$

## PROBLEM mathtrain/test/number theory/254

```markdown
### [ok] Final Answer:
$$
\boxed{2}
$$
```

```markdown
after majority-vote SFT • checkpoint 3 • 414 tokens • 8/128 correct • graded incorrect
$$
199_{10} = 11000111_2
$$
### Step 2: Count the number of zeros and ones
Binary number: $11000111$
Let's count:
- Number of <sub>**</sub>ones<sub>**</sub> ($y$): There are <sub>**</sub>6<sub>**</sub> ones.
- Number of zeros ($x$): There are 3 zeros.
### <sub>**</sub>Step 3: Compute $y - x$<sub>**</sub>
$$
y - x = 6 - 3 = \boxed{3}
$$
### [ok] Final Answer:
$$
\boxed{3}
$$
So, the value of $y - x$ is $\boxed{3}$.
```

Both generations derive the same binary expansion, $1 9 9 _ { 1 0 } = 1 1 0 0 0 1 1 1 _ { 2 }$ , and both then count its digits. The base model counts five ones and three zeros and answers 2. The evolved model counts six ones from the identical string and answers 3. The reasoning is otherwise unchanged, down to the section headings; what moved is a digit count at the last step.

This is what corruption looks like in this regime, and the mechanism needs stating precisely, because it is not the one the phrase “capability erosion” suggests. Nothing has been forgotten. A single sub-step has become unreliable in a way that is invisible in the surrounding text and produces a confidently formatted wrong answer. It is also, by construction, invisible to a majority vote over the model’s own samples: at checkpoint 3 the wrong count is the modal one, so the pseudo-label for this problem would now be 3.

The solve rate moved by 0.75, more than three times the largest change the frozen control produced anywhere in the 1,163-problem set (Table 23). This event is not a threshold flicker.

## F.5 CASE 5: A CAPABILITY STAR SHARPENED, AND WHAT SHARPENING MEANS CONCRETELY

The complement of Case 4, and the modal outcome across the study. The base model reaches this problem in 24 of 128 samples; after three rounds of STaR, in 128 of 128. No new capability is involved: the correct answer was already in the base model’s output distribution and training moved mass onto it.

reference answer 6

Sally, Wei-Hwa, and Zoe are playing a game of marbles involving first arranging as many piles of 10 marbles as possible. Sally brought 239 marbles, Wei-Hwa brought 174 marbles, and Zoe brought 83 marbles. If all their marbles are grouped together, how many must be removed in order to start the game?

untrained backbone • checkpoint 0 • 261 tokens • one of the 104 that miss, of 128 • graded incorrect

\$\$   
500 \div 10 = 50   
\$\$   
This means <sub>\*\*</sub>50 piles of 10 marbles<sub>\*\*</sub> can be formed with <sub>\*\*</sub>exactly 500 marbles .   
### Step 3: Determine how many marbles must be removed   
Since the total is already a multiple of 10, <sub>\*\*</sub>no marbles need to be removed<sub>\*\*</sub> to form   
,→ piles of 10 marbles.

```markdown
### [ok] Final Answer:
$$
\boxed{0}
$$
```

after STaR • checkpoint 3 • 318 tokens • all 128 correct • graded correct   
We want to know how many marbles must be removed to make the total a multiple of 10.   
,→ That is, we find the <sub>\*\*</sub>remainder<sub>\*\*</sub> when 496 is divided by 10.   
\$\$   
496 \div 10 = 49 \text{ remainder } 6   
\$\$   
So, <sub>\*\*</sub>6 marbles<sub>\*\*</sub> must be removed to make the total number of marbles divisible by 10.   
### [ok] Final Answer:   
\$\$   
\boxed{6}   
\$\$   
<sub>\*\*</sub>6 marbles must be removed<sub>\*\*</sub> to start the game.

The base model’s error is upstream of the reasoning: it adds 239 + 174 + 83 and gets 500, then correctly observes that 500 is already a multiple of ten and answers 0. The trained model adds correctly to 496, takes the remainder and answers 6. In 24 of its 128 samples the base model also added correctly and also answered 6; those samples are why this problem counts as sharpened rather than expanded.

This is the shape of almost everything the audited methods do. Across all seven arms, every problem newly solved at pass@1 was one the base model had already solved at least once in 128 samples: 2/2, 3/3, 4/4, 9/9, 149/149, 145/145, 161/161. It is a real and useful effect: a model that answers correctly on the first attempt rather than on the fifth is more useful, and consolidation is what these methods reliably deliver. It is a different claim from the one a leaderboard number invites, and the distinction is measurable at modest cost.