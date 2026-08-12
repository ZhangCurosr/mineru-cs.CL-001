# Detecting an Effect Is Not Learning to Act on It: A Reward–SNR Floor for LLM Acquisition Agents

Ying Yuan

University of California, San Diego yingyuan238@gmail.com

## Abstract

Many pipelines can pay a per-example cost to acquire an auxiliary, model-derived observation—an LLM’s structured reasoning, a slow oracle, an expensive measurement—and then must decide when the acquired signal is worth using. Our thesis is a distinction that is easy to miss: detecting that such a signal helps on average is not the same as learning to act on it per instance, and a reward–SNR floor governs when the second is even possible. Even when the acquired signal is genuinely faithful and an in-sample oracle that picks the top-b examples by realized reward shows a sizable apparent gain, no deployable policy can learn when to acquire it: across per-impression, K=4−64 cluster, handdefined regime, and uplift-tree granularities, learned routing never beats random, and a matched-moment i.i.d.-noise placebo reproduces ≥ 100% of the oracle’s apparent gain. In other words, the apparent “learnable structure” is order statistics of noise, not exploitable signal. We explain this with a single distinction—between detecting a mean effect and learning a per-instance acquisition policy—and a reward–SNR detectability floor: routing is estimable offline only if the reward effect’s SNR $\rho = \mu / \sigma$ clears $\rho ^ { \star } ( N ) { \approx } 2 . 8 / \sqrt { N }$ (equivalently $N \ge N _ { \mathrm { m i n } } = ( 2 . 8 / \rho ) ^ { 2 } )$ , a necessary condition we report as such, with a positive control confirming it is a true low-SNR limit rather than a broken pipeline. As a concrete instantiation we introduce Structured Hypothesis Embeddings (SHE): a frozen LLM decomposes a user history into K ranked, confidence-scored, evidence-grounded intent hypotheses, embedded and fused as an input-embedding branch of a recommender. On three public datasets (MIND, REES46, Amazon-Beauty): (i) SHE is faithful (grounded faithfulness +0.0705, distinctiveness 2×, calibratable ECE 0.142 → 0.031); (ii) its downstream value is backbone- and regime-conditional—significant over an ordered GRU backbone (+0.0114, 95% CI [+0.0030, +0.0209]) yet with a global redundancy gap indistinguishable from zero (−0.0005, [−0.0164, +0.0150]); and (iii) learned per-example acquisition collapses at every granularity because all three datasets sit below the floor $( \rho = 0 . 0 4 8 / 0 . 1 3 8 / 0 . 0 1 4 )$ The realizable unit is therefore a design-time regime gate, not a learned per-instance policy; we give an actionable recipe for it. We release code, a 58-claim ledger mapping each claim to a script/CSV/figure, and a one-command reproduction.

## 1 Introduction

A growing number of machine-learning systems are built around a decision that is usually left implicit: should we pay to acquire an auxiliary, model-derived observation for this example, and if so, do we trust it? The observation might be a large language model’s structured reasoning about an input, a slow but accurate oracle, an additional sensor reading, or a human annotation. Acquisition has a real cost (latency, money, compute), so a natural aspiration is to learn a policy—an acquisition agent—that spends the budget only where the auxiliary signal helps. This aspiration is widespread in active learning, value-of-information, learning-to-defer, and LLM-as-feature pipelines.

This paper makes a simple but, we argue, under-appreciated point: detecting that the acquired signal helps on average is not the same as learning to act on it per instance—and whether that per-example acquisition policy is even recoverable from offline data is governed by a signal-to-noise law that is easy to state and easy to violate. If the per-example effect of the acquired signal on the downstream reward has signal-to-noise ratio $\rho = \mu / \sigma$ , then reliably detecting a policy that conditions on that reward requires ρ to exceed a floor $\rho ^ { \star } ( N ) \approx 2 . 8 / \sqrt { N }$ . Below the floor, the “obvious” evidence that learning helps—an insample oracle that picks the top-b examples by realized reward—is an artifact of order statistics of noise, not exploitable structure. We prove the floor with a positive control (a synthetic signal injected at controllable SNR is recovered by the same deployable pipeline once ρ crosses the floor) and we are explicit that the floor is a necessary condition, not a claim of impossibility.

We then ground the abstract “costly observation” in a concrete method, Structured Hypothesis Embeddings (SHE, §3): a frozen LLM turns a user’s interaction history into K ranked, confidence-scored, evidence-grounded hypotheses about the user’s latent intents; these are embedded and fused as an inputembedding branch of a recommender. SHE is a faithful, interpretable signal on its own terms (§4), but its downstream value is backbone- and regime-conditional (§5): it is significant over an ordered sequential backbone yet its global redundancy gap is statistically indistinguishable from zero, with a clean regime split (absorption in sparse histories, complementarity in long multi-intent histories). Attempts to learn when to acquire SHE fail at every granularity we test (§6), and §7–8 show why: on all three datasets the reward SNR is below the detectability floor.

An honesty caveat carried throughout. The motivating production observation—that structured intent helps most in cold-start / underdetermined regimes—is observed, not controlled. Our public-data study is designed to test the mechanism (the SNR floor) rather than to re-derive that observation, and we flag every place where an effect is directional-but-not-significant, in-sample-only, or below the detectability floor.

## Contributions.

• A diagnosis: apparent acquisition “learnability” is order statistics of noise. Across per-impression, cluster (K=4−64), regime, and uplift-tree granularities on three datasets, no deployable policy beats random, and a matched-moment noise placebo reproduces ≥100% of the in-sample oracle’s apparent gain—so the oracle gap that looks like exploitable structure is not (§6, §7).

• The explanation: mean-detection ̸= policy-learnability, and a reward–SNR detectability floor $\rho ^ { \star } ( N ) \approx 2 . 8 / \sqrt { N }$ that separates them, with a positive control establishing it is a genuine low-SNR limit rather than a broken/underpowered pipeline (§8).

• A concrete instantiation, Structured Hypothesis Embeddings: a frozen-LLM input-embedding branch of ranked, confidence-scored, evidence-grounded hypotheses with a cited-vs-non-cited faithfulness metric; it is faithful yet its downstream value is backbone- and regime-conditional (significant over an ordered GRU/SASRec backbone, global redundancy gap indistinguishable from zero) (§3–5).

• An actionable prescription: since per-instance routing is unlearnable below the floor, deploy a design-time regime gate instead; we give a four-step recipe and validate it at the pooled-regime level (§10).

• A reproducible artifact: a 58-claim ledger mapping each claim to a script, CSV and figure, and a one-command offline reproduction.

![](images/562ab0d0d5801d434f037ca47d5427f14990bf3a02e03f676986cdeadb716b1b.jpg)  
Figure 1: The reward–SNR detectability floor is the paper’s thesis in one picture. A costly semantic observation (here SHE) can only support a learned per-instance acquisition policy if its downstream reward SNR clears $\rho ^ { \star } ( N ) = 2 . 8 / \sqrt { N }$ (necessary mean-detection condition, §8). Both content-rich datasets where SHE is afaithful signal sit below the floor (MIND 2.7×, Amazon-Beauty 62× short of $N _ { \mathrm { m i n } } )$ , which is why learned acquisition collapses at every granularity. The one dataset above the floor $( \mathrm { R E E S 4 6 } )$ is detectable— and there the LLM signal significantly hurts downstream AUC. The bottleneck is not whether the LLM can reason, but whether the downstream reward carries enough signal to learn when to acquire that reasoning.

Concretely, we study an acquisition agent—a gate that decides, per example and from cheap sideinformation only, whether to spend the costly LLM call—and ask when such an agent can be learned from offline reward data. Figure 10 states our answer in one picture: a faithful LLM signal is not enough; the downstream reward must clear a reward–SNR floor before any per-instance acquisition policy is even detectable, and our three datasets sit below it—so the apparent in-sample oracle gain is order statistics of noise, not learnable structure. (We use “agent” for this one-shot acquisition gate, not a sequential/RL planner.)

## 2 Problem Setup

Let each example i (an impression / a ranking slate) have a base predictor using cheap features and an optional costly observation $o _ { i }$ obtained by paying a fixed cost c. Using $o _ { i }$ changes a downstream reward $R _ { i }$ (here NDCG@10 of the slate) by a per-example amount $\Delta _ { i } = R _ { i } ( \mathrm { w i t h } \ o _ { i } ) - R _ { i } ( \mathrm { w i t h o u t } )$ . Write $\mu = \mathbb { E } [ \Delta _ { i } ]$ and $\sigma = \sqrt { \mathrm { V a r } [ \Delta _ { i } ] }$ , and define the reward SNR $\rho = \mu / \sigma$ . An acquisition policy $\pi : \quad x _ { i } \mapsto \{ 0 , 1 \}$ decides, from cheap side-information $x _ { i }$ only, whether to pay for $o _ { i } ,$ , subject to a budget $b = \mathbb { E } [ \pi ]$ . We evaluate policies by the realized system reward at budget b against a random-acquisition baseline, with a per-example bootstrap 95% CI, always out-of-fold (the policy never sees the outcome it is deciding to buy).

Two things are worth separating. (a) The value of $o _ { i }$ if you always acquire it—an average-treatment question. (b) The learnability of when to acquire it—a heterogeneous-policy question. Our theory (§8) concerns (b); our recommender experiments (§4–5) concern (a). The two interact: a signal can have real average value yet a per-example acquisition policy for it can be undetectable.

Datasets. We use three public datasets spanning domain × content richness (Table 1): MIND [9] (English news, content-rich, genuinely multi-topic histories), Amazon-Beauty (e-commerce, content-rich titles but short histories), and REES46 (e-commerce sessions, content-thin: 87.9% of windows are single-category). The three differ sharply in history length (median $| H | = 2 0 / 5 / \mathrm { s h o r t } )$ and multi-intent fraction, which is precisely what lets us separate regime-dependent value from a global effect. All feature tables are cached; no proprietary data is used.

Table 1: The three public datasets span two axes: domain × content richness. History length, distinctcategory count, and the sparse / multi-intent slice fractions are recomputed from the cached hypotheses; $\rho$ is the per-example reward SNR (§7). The spread in history length and multi-intent fraction is exactly what makes the value regime-conditional.
<table><tr><td>Dataset</td><td>Domain</td><td>Content</td><td>N</td><td>median |H|</td><td>multi-intent %</td><td>sparse  $\%$ </td><td>ρ</td></tr><tr><td>MIND</td><td>news</td><td>rich</td><td>1263</td><td>20</td><td>45.7</td><td>14.0</td><td>0.048</td></tr><tr><td>Amazon-Beauty</td><td>e-commerce</td><td>rich</td><td>650</td><td>5</td><td>64.6</td><td>34.2</td><td>0.014</td></tr><tr><td>REES46</td><td>e-commerce</td><td>thin (87.9% single-cat)</td><td>498</td><td>short</td><td>一</td><td>一</td><td>0.138</td></tr></table>

![](images/a6afc21de36716a0635119df8823d7b5cb08f003899f95c92dab03cdb48cdb45.jpg)  
Figure 2: What one raw input record looks like in each dataset (real, abbreviated), before the frozen LLM produces structured hypotheses. The three span a $2 \times 2$ of domain × content-richness: MIND and Amazon-Beauty carry rich item text supporting genuine multi-intent decomposition, whereas REES46 exposes only category codes and is dominated by single-category sessions. This contrast is why agent-side facet metrics are strong on MIND/Amazon but floored on REES46 (Table 1).

## 3 Method: Structured Hypothesis Embeddings (SHE)

Given a user history $H = ( e _ { 1 } , \ldots , e _ { n } )$ (news reads or product interactions), a frozen LLM is prompted to emit K intent hypotheses. Each hypothesis $h _ { k }$ is a short natural-language statement of a latent interest, and carries (i) a calibratable confidence $\gamma _ { k } \in [ 0 , 1 ]$ and (ii) a set of evidence indices $E _ { k } \subseteq \{ 1 , \dots , n \}$ citing the history events that support it. We use $K = 3$ . This “Scheme $\mathbf { B } ^ { \ast }$ structured output is contrasted with a “Scheme A” single-summary baseline.

Each hypothesis is embedded, $\boldsymbol { e } _ { k } = \phi ( h _ { k } )$ , in a fixed text-embedding space $( \ell _ { \mathrm { 2 } } { \cdot } \mathrm { n o r m a l i z e d } ,$ so similarity is cosine). For a candidate item with embedding c, the SHE branch produces a small feature vector whose primary coordinate is a confidence-weighted best-facet match

![](images/15efb1f454b93f74982fa2b0c35a74f8da7a25444797184f2a58846b45d3907f.jpg)  
Figure 3: Structured Hypothesis Embedding (SHE) pipeline. A frozen LLM decomposes a user history into $K { = } 3$ ranked, confidence-scored $( \gamma _ { k } )$ , evidence-grounded (cited history indices) intent hypotheses. Each is embedded; a candidate c scores against the best-matchingfacet $f _ { B } ^ { \operatorname* { m a x } } = \operatorname* { m a x } _ { k } \gamma _ { k } \cos ( c , e _ { k } )$ , which is fused as an input-embedding branch alongside a cheap base backbone (mean-pool / GRU / SASRec). The confidences $\gamma _ { k }$ and cited evidence make the branch calibratable and faithfulness-testable. Values shown are a real MIND example (impression 2445, Fig. 4).

$$
f _ { B } ^ { \operatorname* { m a x } } ( { \pmb c } ) = \operatorname* { m a x } _ { k \in \{ 1 , . . . , K \} } \ \gamma _ { k } \cos ( { \pmb c } , { \pmb e } _ { k } ) ,\tag{1}
$$

alongside γ-weighted mean and max variants. The max-over-facets aggregation is the crux: it lets a candidate match any of the user’s disjoint interests rather than a blended average, which is precisely what a single summary (Scheme A, $f _ { A } = \cos ( c , e _ { \mathrm { s u m m a r y } } ) )$ cannot express. SHE plugs in as an input-embedding branch: the downstream model receives $\mid f _ { \mathrm { b a s e } } ( \pmb { c } ) , f _ { B } ( \pmb { c } ) \mid$ ], where $f _ { \mathrm { b a s e } }$ is the cheap backbone (mean-pooled history similarity, ID popularity features, or the state of an ordered sequence model). Fusion is late and the branch is frozen; nothing is back-propagated into the LLM.

Instantiation of $\phi .$ The method is agnostic to the choice of text encoder $\phi ;$ we deliberately state all results in terms of the fixed, ℓ<sub>2</sub>-normalized space rather than a particular model. In our experiments $\phi$ is OpenAI text-embedding-ada-002 for MIND and REES46, and a local 256-dimensional TF-IDF + TruncatedSVD (LSA) space over item titles for Amazon-Beauty (an embedding-API ACL prevented using the same hosted encoder there). Candidate items, hypotheses, and the Scheme-A summary are always embedded in the same space within a dataset, so all comparisons are intra-space; we never compare cosine values across the ada-002 and LSA spaces (§10). Full encoder, feature-block, and downstream-head details are in App. B.

Why “structured” and “grounded” matter. The evidence indices $E _ { k }$ give a testable notion of faithfulness (§4): a hypothesis should be more similar to the history events it cites than to those it does not. The confidences $\gamma _ { k }$ let us calibrate and, in principle, gate. Both are properties of the structure, not of any particular embedding model.

## 4 Agent-Side Quality of the Hypotheses

Before asking whether SHE helps a recommender, we ask whether the hypotheses are good on their own terms. All numbers here are computed on MIND (news, genuine multi-intent) and REES46 (e-commerce, thin single-category); details in Appendix C.

<table><tr><td>MIND impression 2445 — a 14-item, 9-category news history (multi-intent slice). Abbreviated titles: [1] free donuts at Shipley; [2] Trump offer to British teen&#x27;s family; [3] Hailey Bieber gym/butt workout; [4] emotional nurse photo goes</td></tr><tr><td>pollution photos; [9] Kevin Spacey not charged; [10] baby elephant falls in watering hole; [11] MetLife Stadium black cat; [12] man killed at Popeyes over chicken sandwich; [13] Demi Moore on Ashton Kutcher &quot;addiction&quot;; [14] Kim Kardashian gained 18 lbs. The frozen LLM returns three disjoint, confidence-ranked, evidence-grounded hypotheses:</td></tr><tr><td>• h1 (γ=0.66): celebrity / body &amp; fitness / relationships cites {3, 9, 13, 14} • h2 (γ=0.58): sensational crime &amp; political controversy cites {2, 6, 9, 12}</td></tr><tr><td>• h3 (γ=0.47): light viral food &amp; animal-interest items cites {1, 5, 10, 11, 12} The facets are soft, not a hard partition (index 9 is cited by both 1  $h _ { 1 }$  and  $h _ { 2 } ;$  index 12 by both  $h _ { 2 }$  and h3), and each carries a distinct confidence. A candidate scores against its best matching facet via maxk  $\gamma _ { k } \cos ( c , e _ { k } )$  , so a single-summary</td></tr></table>

Figure 4: Qualitative SHE example (real, unedited). One multi-topic history yields three grounded intent facets with calibratable confidences — the concrete mechanism behind the +0.0705 grounded faithfulness and 2× distinctiveness of $\ S 4$

Faithfulness (grounded). We measure the paired difference in cosine similarity between a hypothesis and its cited vs. non-cited history events. On MIND this paired difference is +0.0705 (95% CI [+0.068, +0.073]): hypotheses genuinely track the evidence they cite. (On REES46, where 87.9% of windows are singlecategory, the difference is ≈ 0, as expected—there is nothing to decompose.) Figure 5 plots this as a single paired-∆ bar with CI, deliberately not as two absolute-similarity bars, to avoid the common mis-reading that the non-cited similarity is zero.

Distinctiveness. 1 − cos over hypothesis pairs is 0.204 on MIND versus 0.104 on REES46: on genuinely multi-intent histories SHE produces disjoint facets (∼ 2× the separation of the thin-content dataset).

Calibration. Raw top-1 confidence is over-confident (ECE 0.142, Brier 0.166 on REES46). A cross-fit isotonic map reduces ECE to 0.031 (−78%; Figure 6), a standard, honest, deployable post-hoc fix. We report per-bin counts and label the analysis directional at small N.

## 5 Downstream Value Is Backbone- and Regime-Conditional

Does the SHE branch improve a recommender? The answer depends on what it is fused onto. We first establish the gradient (§5.1), then a controlled ordered-vs-unordered test (§5.2).

## 5.1 A redundancy gradient over baseline strength

We fuse SHE onto a ladder of increasingly strong base features and measure the +SHE lift in NDCG@10 (GroupKFold-by-impression, class-balanced pointwise logistic regression, per-impression bootstrap CI). On MIND the lift descends monotonically as the base gets stronger: $L _ { 0 }$ (subcategory popularity) +0.0161<sup>∗</sup>, $L _ { 1 }$ (ID pair) $+ 0 . 0 1 4 6 ^ { * } , L _ { 2 } \ : ( \mathrm { I D + t e x t } ) + 0 . 0 1 0 0 ^ { * } , L _ { 3 }$ (strong mean-pooled text) +0.0094 (ns); the sparse slice starts ∼ 3× higher at the weak end (Figure 7). This is the honest core of the “redundancy boundary”: SHE’s marginal value shrinks against a strong content baseline, but does not vanish, and is largest exactly where behavioral signal is underdetermined. A controlled degradation sweep (Appendix D) corroborates the gradient and is explicitly labeled a diagnostic corruption, not an achieved real-world lift.

![](images/b00b1903ee22d8b012fb3e2c9de97a9ae709a3f2073c42d57ac39f31aeaf30a8.jpg)  
Figure 5: Agent-side quality on MIND. Grounded faithfulness is a paired cited-minus-non-cited cosine difference (+0.0705), not an absolute similarity; distinctiveness is 2× the thin-content dataset.

![](images/4bc3a53b2154573c5752a945d4f0faa54634cc84dc207206a8fa73f9b8ea5487.jpg)  
Figure 6: Confidence is over-confident but cheaply calibratable: cross-fit isotonic regression reduces ECE 0.142 → 0.031 (−78%). Bins are equalfrequency (∼83/bin); small-N, directional.

Table 2: MIND $2 \times 2$ (NDCG@10, N=1263; 95% bootstrap CI). The global redundancy gap (interaction) is statistically indistinguishable from zero, while the SHE branch remains significant over the ordered GRU backbone. Slice analyses show regime-dependent absorption vs. complementarity.
<table><tr><td>Quantity</td><td>Estimate</td><td>95% CI</td><td>p</td></tr><tr><td>Mean-pool base (A)</td><td>0.3701</td><td></td><td></td></tr><tr><td>Ordered GRU base (C)</td><td>0.3992</td><td></td><td></td></tr><tr><td>+SHE over mean-pool</td><td>+0.0109</td><td>[-0.0046, +0.0277]</td><td>0.165</td></tr><tr><td>+SHE over ordered GRU</td><td>+0.0114</td><td>[+0.0030, +0.0209]</td><td>0.005</td></tr><tr><td>Redundancy gap (interaction)</td><td>-0.0005</td><td>[-0.0164,+0.0150]</td><td>0.919</td></tr></table>

## 5.2 A controlled ordered-vs-unordered test

To ask specifically whether ordered sequential access makes SHE redundant, we run a $2 \times 2$ on MIND (the clean testbed: long, multi-topic histories, median $n { = } 1 9 )$ holding split, slate, labels and the late-fusion ranker fixed and varying only two factors: (unordered mean-pool vs. ordered GRU [3]) × (no LLM vs. +SHE). Because MIND histories are long and multi-topic, the ordered GRU (0.3992) is genuinely stronger than mean-pool (0.3701)—so this is a clean test of ordering, unlike Amazon-Beauty (median history 5, where the GRU is weaker than mean-pool and thus cannot isolate ordering; we report Amazon in Appendix E as an additional backbone check, not a clean ordering test).

The reading of Table 2 is deliberately careful. Ordered access does not globally absorb the LLM signal: SHE adds a significant gain over the ordered GRU $( + 0 . 0 1 1 4 , p { = } 0 . 0 0 5 )$ , and the interaction/redundancy gap is statistically indistinguishable from zero (−0.0005, p=0.919). What is real is a regime split (Figure 8): the gap is positive on sparse/short histories $( + 0 . 0 3 3 / + 0 . 0 2 2$ , absorption—the ordered backbone already captures the little intent present) and reverses on long/multi-intent histories (−0.006/−0.005, complementarity—SHE contributes semantic structure the sequence model cannot subsume). Thus SHE’s value is a function of backbone strength × history regime, not of ordering per se.

![](images/012aec9161e7546c7cfb5aff7ac82f0de7a9317d7615bf1f7cbf6162b6871d61.jpg)

![](images/26471f8b4ebd432add9ebaf5e0ff268c767c2e96c07e8ef40d71c797711a4eab.jpg)  
Figure 7: Redundancy gradient on MIND. The +SHE NDCG@10 lift descends monotonically as the base features get stronger $( L _ { 0 }  L _ { 3 } )$ and is largest on sparse (cold-start) histories. The strong-text rung is not significant; the effect is a gradient, not a chasm.

Robustness (B1–B5). Five checks defend the finding (Appendix F). (B1) A 5-seed sweep: every seed has GRU>mean-pool and a significant SHE gain over the GRU (+0.011 to +0.023). (B2) A degradation sweep (full/truncate/shuffle/mean-pool): the SHE gain is stable and reliably significant only over ordered backbones. (B3) Residualization—projecting SHE onto the sequential state and keeping the residual— retains 101% of the gain (raw +0.0114 → resid +0.0115); the sequence state explains SHE features with $R ^ { 2 } { \approx } 0 . 0 0 { - } 0 . 0 1$ , i.e. SHE is largely orthogonal. (B4) A redundancy probe (predict SHE from the sequence state) gives $R ^ { 2 } \leq 0 . 0 1 0$ on all slices. (B5) A structurally different ordered backbone, SASRec [5] (causal self-attention), replicates the pattern: SHE gain over ordered SASRec +0.0179 [+0.0076, +0.0281], same regime split. We treat SASRec as an additional ordered-backbone check, not a uniformly stronger backbone (here SASRec 0.3713 is on par with mean-pool and below the GRU; attention over-fits at N ≈1.4k).

## 6 Learning When to Acquire Fails at Every Granularity

The recommender results concern the average value of always acquiring SHE. We now ask the acquisition question: can we learn a per-example policy that spends a fixed budget on the impressions where SHE helps most? The answer, across three datasets and the full granularity axis, is no.

Per-impression. A learned out-of-fold policy (predict $\Delta _ { i }$ from cheap features, act out-of-sample) does not beat random acquisition at any budget; neither does an uncertainty or sparse heuristic. The only “winner” is an in-sample oracle that ranks impressions by realized $\Delta _ { i }$ —and a matched-moment i.i.d.-noise placebo reproduces $\ge ~ 1 0 0 \%$ of that oracle’s apparent gain (MIND $+ 0 . 0 5 1 8 \ = \ + 0 . 0 5 1 8 )$ . The oracle is order statistics of noise, not exploitable structure.

Cluster / regime / tree. We test representation clusters (K=4, 8, 16, 32, 64 via KMeans on the sequencestate vector), hand-defined cross-product regimes (sparse × multi-intent × determinacy × diversity), and honest uplift trees, each with per-fold EB-shrunk cluster means and label-free test assignment. No tested granularity significantly beats random at any budget on either MIND or Amazon-Beauty (Figure 9); the granularity curve is flat and near zero. Crucially, a positive control (Appendix G) shows the same deployable pipeline does recover a synthetic cluster signal once its cluster-SNR exceeds ≈ 0.20 (MIND) / 0.35 (Amazon); the real data sit at 0.075 / 0.056, an order of magnitude below. So the null is a genuine low-SNR limit, not a broken or underpowered pipeline. A power analysis confirms the observed $| d |$ is below each granularity’s own $\mathrm { M D E _ { 8 0 } }$ on both datasets, so we phrase the result as not detectable at these $N _ { ☉ }$ , not impossible.

![](images/bf40ab33de1eadb283f2f33427a70c72a6cbc79e38ddfd5c992df5fff6c5c826.jpg)  
Figure 8: Backbone-conditional value on MIND. SHE adds value over both an unordered and an ordered backbone; the global redundancy gap $\mathrm { i s } \approx 0$ , but slices show absorption in sparse histories and complementarity in long multi-intent histories.

What is realizable. The regime cell means are sensible and consistent (dense-single $\approx - 0 . 0 0 1$ ; multiintent +0.0112; cold-start +0.0109). Because pooling averages the reward noise down by ${ \sqrt { n } } ,$ , a design-time subsystem split—apply SHE to a cold-start / multi-intent subsystem, skip dense-single—is realizable, even though a learned per-example policy is not. This is the actionable unit.

## 7 The Mechanism: a Three-Dataset Reward-SNR Story

Why does acquisition collapse? Because the per-example reward effect is buried in noise. Across three datasets spanning domain × content-richness, the reward SNR is small and the conclusion is identical (Table 5):

• MIND (content-rich news): tiny positive per-impression lift (+0.0094), $\rho = 0 . 0 4 8$

• REES46 (content-poor e-commerce): ≈ 0/negative per-window lift (−0.0158; ROC-AUC 0.840 → 0.833, LLM hurts), $\rho = 0 . 1 3 8$

• Amazon-Beauty (content-rich e-commerce): near-zero lift (+0.0012), $\rho ~ = ~ 0 . 0 1 4$ (lowest of the three).

In all three, no deployable per-unit policy beats random and the in-sample oracle gap is reproduced by a noise placebo. The acquisition-limits result is therefore a mechanism (low per-unit reward SNR), not a

![](images/5dc54172d58917b24649c8da6790684eaf00ba67448359e3b2f95d32e0b1f162.jpg)

![](images/cd02885ddc5591f7160cb497ca12035fa068297f9d37737e538dd4d419b74e50.jpg)  
Figure 9: Acquisition collapse across granularity (MIND). From per-impression through $K { = } 4 { - } 6 4$ clusters, hand-defined regimes and uplift trees, no deployable policy significantly beats random (all CIs cross zero). The in-sample oracle is largely reproduced by a matched-noise placebo.

MIND-specific quirk. Note we do not claim a global backbone redundancy is significant on MIND; the global gap is indistinguishable from zero (§5), while slices suggest regime-dependent absorption/complementarity.

## 8 The Reward–SNR Detectability Floor

We now state the law that ties the sections together.

Proposition 1 (reward–SNR detectability floor). Consider detecting, at significance α and power $1 - \beta ,$ that the per-example effect $\Delta _ { i }$ of a costly observation on the reward has positive mean, from N examples with per-example SNR $\rho = \mu / \sigma$ . Detection requires

$$
\rho \ge \rho ^ { \star } ( N ) = \frac { z _ { 1 - \alpha / 2 } + z _ { 1 - \beta } } { \sqrt { N } } \approx \frac { 2 . 8 } { \sqrt { N } } , \qquad N \ge N _ { \operatorname* { m i n } } ( \rho ) = \left( \frac { z _ { 1 - \alpha / 2 } + z _ { 1 - \beta } } { \rho } \right) ^ { 2 } = \left( \frac { 2 . 8 } { \rho } \right) ^ { 2 } ,\tag{2}
$$

$$
a t \alpha { = } 0 . 0 5 , 1 - \beta { = } 0 . 8 ( s o z _ { 1 - \alpha / 2 } + z _ { 1 - \beta } \approx 1 . 9 6 + 0 . 8 4 = 2 . 8 ) .
$$

This is a standard one-sample mean-detection bound; its force here is interpretive. A policy that learns when to acquire must at minimum detect that the conditioned-on reward has signal; if even the average effect is below the floor, a heterogeneous policy conditioned on the same noisy reward is a fortiori undetectable. We are explicit that $\operatorname { E q . }$ . 2 is a necessary mean-detectability condition, not a sufficient policy-learning / regret bound, and not an impossibility theorem.

Consistency and dataset placement. Two independent routes agree on the floor to within 1.3×: the closed form gives $\rho ^ { \star } ( 1 2 6 3 ) = 0 . 0 7 9$ , while a semi-synthetic learnability sweep (inject a feature-predictable effect at controllable SNR, measure the captured fraction of the tau-oracle) locates the threshold near 0.10. Placing the datasets against Eq. 2 (Figure 10, Table 3):

MIND sits 1.6× below its SNR floor (needing $\sim 2 . 7 \times$ more data, $N _ { \mathrm { m i n } } \approx 3 4 0 0 )$ ; Amazon-Beauty sits 7.9× below (needing ∼ 60× more data). REES46 is the one powered case $\left( \rho = 0 . 1 3 8 > \rho ^ { \star } = 0 . 1 2 6 \right)$ —and there the effect is significantly negative (the LLM branch hurts). This last point matters for honesty: the acquisition collapse is not merely a power excuse, because the one dataset with enough power shows the acquired signal is net-negative, not net-positive.

Table 3: Datasets against the detectability floor. We report precise, dataset-specific gaps rather than an “order of magnitude” shorthand. “Powered” = 1 means the sample is above the floor.
<table><tr><td>Dataset</td><td>N</td><td>ρ</td><td> $\rho _ { 8 0 } ^ { \star } ( N )$ </td><td> $N _ { \mathrm { m i n } }$ </td><td> ${ \rho ^ { \star } } / { \rho }$ </td><td>Effect</td></tr><tr><td>MIND</td><td>1263</td><td>0.048</td><td>0.0788</td><td>3403</td><td>1.64</td><td>≈ 0 (positive)</td></tr><tr><td>Amazon-Beauty</td><td>650</td><td>0.014</td><td>0.1098</td><td>40000</td><td>7.85</td><td>≈ 0 (negative)</td></tr><tr><td>REES46</td><td>498</td><td>0.138</td><td>0.1255</td><td>412</td><td>0.91</td><td>significantly negative</td></tr></table>

![](images/8527e09456558e6703f0632737ab738e8aa4ff1813d5fa99352c0b8f05ffd1f1.jpg)  
Figure 10: The reward–SNR detectability floor $\rho ^ { \star } ( N ) \approx 2 . 8 / \sqrt { N }$ (80% power). MIND and Amazon-Beauty sit below the floor; REES46 is above it and its effect is significantly negative. The floor is a necessary condition, not an impossibility theorem.

A sufficiency-side check (HTE-SNR, Appendix H) closes the “a cleverer heterogeneous policy still wins” loophole: correlation, out-of-fold $R ^ { 2 }$ , and a heterogeneity-SNR against a permutation null are all inside the null band on both datasets—there is no learnable heterogeneity to exploit.

## 9 Related Work

LLM-as-feature for recommendation. KAR [10] and RLMRec [7] augment recommenders with LLMderived knowledge or representations. We differ by (i) a structured, evidence-grounded, confidence-scored hypothesis representation with a testable faithfulness metric, and (ii) a focus on when the signal is (un)learnable to acquire rather than average lift. Active learning and value of information [8] learn what to query; we give a detectability floor that governs whether such a policy is estimable offline at all. Uplift / heterogeneous treatment effects [6] motivate our per-example $\Delta _ { i }$ estimation and our HTE-SNR null check. Selective prediction / learning-to-defer [1, 4] route examples to an abstain/expert option; our acquisition policy is a deferral to a costly LLM observation, and our contribution is the SNR limit on learning that routing. Calibration [2] underlies our confidence post-hoc fix. Finally, structured LLM hypotheses with confidences were used for unsupervised cluster-geometry scoring in single-cell gene-set annotation by our prior HypoGeneAgent [11]; we reuse ranked hypotheses+confidence but move to a different domain and add a supervised downstream task, learned conditional weighting, grounded faithfulness, and the acquisition/SNR analysis. Net novelty: the reward–SNR detectability floor for costly semantic acquisition.

## 10 Discussion, Deployment, and Limitations

Deployment prescription. Do not learn per-instance acquisition from noisy offline rewards. Use pre specified, design-time regime gates: route SHE to (a) cold-start / underdetermined histories, where a strong backbone has little to absorb (absorption-style value), and (b) long multi-intent histories, where SHE contributes complementary semantic structure (complementarity value). These are exactly the two regimes where §5 finds value, and they are addressable at design time because pooling averages the reward noise down—unlike a per-impression policy. Concretely, a practitioner can follow a four-step recipe: (1) estimate the reward SNR $\rho$ and check it against the floor $N _ { \operatorname* { m i n } } = ( 2 . 8 / \rho ) ^ { 2 } \mathrm { - i f } \ N < N _ { \operatorname* { m i n } }$ , do not attempt a learned per-instance router (it will fit noise order statistics); (2) instead pre-specify a small number of gates from cheap, label-free slice features (history length, distinct-category count, sparsity), not from the reward; (3) enable the costly signal only inside the two value regimes above; (4) validate the gate at the pooled-regime level with an out-of-fold 95% CI, never per instance. This turns an unlearnable routing problem into a one-time subsystem-placement decision that is statistically supported.

Limitations. (1) The motivating production observation is observed, not controlled; our public study tests the mechanism, not that observation. (2) The detectability floor is a necessary mean-detection condition, not a policy-learning/regret bound. (3) Downstream hypothesis embeddings on Amazon-Beauty use a local LSA space (embedding-API ACL), which is internally consistent but not identical to the ada-002 space used elsewhere; we flag this and avoid cross-space comparisons (encoder details in App. B). (4) SASRec is an additional ordered-backbone check at N ≈ 1.4k, not a claim of a uniformly stronger backbone. (5) All significance is at the sample sizes reached; several nulls are power-limited and labeled as such. Fullrecomputation of a few private on-pod metrics is labeled precomputed evidence in the artifact.

Conclusion. Costly semantic observations—LLM structured reasoning being a timely instance—obey a reward–SNR detectability floor. When a dataset sits below it, learning when to acquire is not detectable and its apparent in-sample gains are noise order statistics; the realizable unit is a design-time regime gate. Instantiated as Structured Hypothesis Embeddings, the signal is faithful and its downstream value is backboneand regime-conditional, with a global redundancy gap indistinguishable from zero but a clean absorption / complementarity split. We hope the floor is a useful, honest yardstick for the growing class of pipelines that pay to think.

## Reproducibility Statement

All offline results (R1–R13, appendix) are regenerated by a single command, scripts/reproduce.sh, from committed feature tables (CPU, no LLM calls); only from-scratch hypothesis generation needs an LLM. A 58-claim ledger (results/paper claims.csv) maps every claim to a script, a persisted CSV, and a figure; Appendix I reproduces the reviewer-facing subset.

Author contributions. Ying Yuan (corresponding author, yingyuan238@gmail.com) conceived the research question and the core thesis (that detecting a mean effect is distinct from learning a per-instance acquisition policy), formulated the costly-semantic- observation problem and the reward–SNR detectability floor, designed and implemented the Structured Hypothesis Embeddings method and the full experimental pipeline (data processing, hypothesis generation, backbone and acquisition studies, the positive control, and all robustness analyses), produced all figures and tables, and wrote the manuscript. Any additional authors and their specific contributions will be recorded here upon joining the project.

Acknowledgments. We thank colleagues for discussion and feedback. This paper uses only publicly available datasets (MIND, REES46, Amazon-Beauty); no proprietary data were used.

## References

[1] Y. Geifman and R. El-Yaniv. Selective classification for deep neural networks. Advances in Neural Information Processing Systems (NeurIPS), 2017.

[2] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks. In International Conference on Machine Learning (ICML), 2017.

[3] B. Hidasi, A. Karatzoglou, L. Baltrunas, and D. Tikk. Session-based recommendations with recurrent neural networks. In International Conference on Learning Representations (ICLR), 2016.

[4] H. Jiang, B. Kim, M. Y. Guan, and M. Gupta. To trust or not to trust a classifier. Advances in Neural Information Processing Systems (NeurIPS), 2018.

[5] W.-C. Kang and J. McAuley. Self-attentive sequential recommendation. In IEEE International Conference on Data Mining (ICDM), 2018.

[6] S. R. Kunzel, J. S. Sekhon, P. J. Bickel, and B. Yu. Metalearners for estimating heterogeneous treatment ¨ effects using machine learning. Proceedings of the National Academy of Sciences, 116(10), 2019.

[7] X. Ren, W. Wei, L. Xia, et al. Representation learning with large language models for recommendation. In The Web Conference (WWW), 2024.

[8] B. Settles. Active Learning. Morgan & Claypool, 2012.

[9] F. Wu, Y. Qiao, J.-H. Chen, et al. MIND: A large-scale dataset for news recommendation. In Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2020.

[10] Y. Xi, W. Liu, J. Lin, et al. Towards open-world recommendation with knowledge augmentation from large language models. In ACM Conference on Recommender Systems (RecSys), 2024.

[11] Y. Yuan et al. HypoGeneAgent: A hypothesis language agent for gene-set cluster resolution selection using Perturb-seq datasets. arXiv preprint arXiv:2509.09740, 2025.

## A SHE prompt templates and agent configuration

This appendix documents the exact agent configuration used to produce the Structured Hypothesis Embeddings (SHE). We reuse the structured-hypothesis format of our prior HypoGeneAgent work [11] (a ranked list of natural-language hypotheses, each with a calibrated confidence and explicit supporting evidence), but re-target it from single-cell gene-set annotation to per-user intent over behavioral sequences. Nothing here is tuned on downstream labels: the language model is a frozen zero-shot reasoner and every prompt below is fixed a priori.

Decoding and model configuration. We use two schemes. Scheme A produces a single blended-intent summary; Scheme B produces the ranked, evidence-grounded Top-3 hypotheses that constitute SHE. Table 4 lists the exact settings. Both are called through the same credential-free proxy transport; no fine-tuning, retrieval, or tool use is involved.

<table><tr><td></td><td>Scheme A (summary)</td><td>Scheme B (SHE, Top-3)</td></tr><tr><td>Model</td><td>GPT-5.4</td><td>GPT-5.5</td></tr><tr><td>Reasoning effort</td><td>low</td><td>high</td></tr><tr><td>Output</td><td>{&quot;summary&quot;: ...}</td><td>{&quot;hypotheses&quot;: [...]}</td></tr><tr><td># hypotheses</td><td>1 (free sentence)</td><td>exactly 3, rank-ordered</td></tr><tr><td>Per-item fields Fine-tuned?</td><td>no (zero-shot)</td><td>hypothesis, confidence, evidence_indices no (zero-shot)</td></tr></table>

Table 4: Agent configuration for the two hypothesis-generation schemes. Scheme B is the source of the SHE feature block (App. B).

Input serialization. Each behavioral window is rendered to a numbered plain-text list, one action per line, so that the evidence indices the model returns are directly interpretable and can be checked against the cited steps (the faithfulness measurement in §4). For e-commerce (REES46, Amazon-Beauty) each line is [i] <action>, category: <category\_code>; for news (MIND) each line is the clicked article title with its category, oldest to newest. The step index [i] is the anchor referenced by evidence indices.

Output schema (Scheme B). The model must return a single JSON object whose hypotheses array holds exactly three elements, ordered by descending strength/urgency, each with: (i) hypothesis — a specific intent statement with a short rationale; (ii) confidence — a calibrated probability in [0, 1]; and (iii) evidence indices — the integer input-step indices that support the hypothesis. The three fixed rank slots (with the enforced third-slot “browsing” fallback below) are what make SHE a fixed-width, alignable feature block across users; the four scalar coordinates derived from this block are given in App. B.

Boundary rules. Three a-priori rules keep the confidences honest and the facets distinct, and are identical across all datasets in a domain: (1) Weak-signal fallback — if the window is uninformative (e.g. all views with no add-to-cart, or a very short single-topic history), every confidence must be below 0.5 and the third hypothesis must be the exact fixed string (“Just browsing / no clear purchase goal” for e-commerce, “Casual browsing / no strong topical interest” for news). (2) Strength gate — only a strong behavioral trigger (e.g. an add-to-cart, or, for news, multiple distinct topics) may push the top hypotheses above 0.7, and the multiple hypotheses must cover genuinely different facets rather than restating one topic. (3) Bias elimination — the model must never infer absolute gender, age, or region from category names. These rules are the reason the raw confidences are usable-but-overconfident and cheaply post-hoc calibratable (ECE 0.142 → 0.031, App. B), and the reason Scheme B produces disjoint evidence facets that the distinctiveness metric rewards.

Verbatim prompts. The complete system prompts follow. The serialized window (above) is appended as the user turn.

News — Scheme A (single summary).

You are a news-recommendation and reading-interest expert. Below is a user’s recent [reading   
,→ history] (news articles they clicked, oldest to newest). In ONE free-text sentence of at most   
,→ 30 words, give a single sharp summary of the user’s current core reading interest. Output   
,→ STRICTLY a JSON object: {"summary": "<your summary>"} -- no markdown, no extra explanation.

News — Scheme B (SHE, Top-3 ranked hypotheses).

You are a top expert in news-recommendation and reader psychology. Below is a user’s recent [   
,→ reading history] (clicked news articles, oldest to newest). Based strictly on the reading   
,→ evidence, infer the user’s Top-3 ranked [Reading-Interest Hypotheses] -- the distinct themes/   
,→ topics the user is most likely to want to read next.   
Output STRICTLY a single JSON object: {"hypotheses": [ ... ]} whose array holds EXACTLY 3   
,→ elements, ordered by descending strength. No markdown (no ‘‘‘json), no extra explanation.   
,→ Each element must contain these fixed keys:   
1. "hypothesis": a specific reading-interest statement with reasoning about the theme.   
2. "confidence": a calibrated probability score between 0.0 and 1.0.   
3. "evidence\_indices": an integer array of the input step indices (the [n] at the start of each   
,→ input line) that support this hypothesis.   
[Hard boundary rules]   
- Weak-signal fallback: if the history is very short or all one narrow topic, every confidence   
,→ must be below 0.5, and the 3rd hypothesis must be exactly "Casual browsing / no strong   
,→ topical interest".   
- Multi-interest capture: when the history spans several distinct topics, the three hypotheses   
,→ MUST cover genuinely different interest facets (do not restate one topic).   
Bias elimination: never fabricate the user’s absolute gender, age, or region.

E-commerce — Scheme A (single summary).

You are an e-commerce consumer-behavior expert. Below is a user’s [behavior sequence] within a   
,→ single shopping window. In ONE free-text sentence of at most 30 words, give a single sharp   
,→ summary of the user’s current core shopping intent. Output STRICTLY a JSON object: {"summary   
,→ ": "<your summary>"} -- no markdown, no extra explanation.

## E-commerce — Scheme B (SHE, Top-3 ranked hypotheses).

You are a top expert in e-commerce consumer behavior and user psychology. Below is a user’s [   
,→ behavior sequence] within a single shopping window. Based strictly on the concrete behavioral   
,→ evidence, infer the user’s Top-3 most urgent [Ranked Intent Hypotheses] (immediate purchase   
,→ motivations).   
Output STRICTLY a single JSON object: {"hypotheses": [ ... ]} whose array holds EXACTLY 3   
,→ elements, ordered by descending urgency. No markdown (no ‘‘‘json), no extra explanation. Each   
,→ element must contain these fixed keys:   
1. "hypothesis": a specific intent-hypothesis statement with behavioral-motivation reasoning.   
2. "confidence": a calibrated probability score between 0.0 and 1.0.   
3. "evidence\_indices": an integer array of the input step indices (the [n] at the start of each   
,→ input line) that support this hypothesis.   
[Hard boundary rules]   
- Weak-signal fallback: if the input is ALL views (view) with NO add-to-cart, every confidence   
,→ must be below 0.5, and the 3rd hypothesis must be exactly "Just browsing / no clear purchase   
,→ goal".   
- Strong-signal trigger: only when an add-to-cart action is present may the top two hypotheses   
,→ exceed 0.7 confidence.   
- Bias elimination: never fabricate the user’s absolute gender, age, or region from category   
,→ names.

## B Embedding and feature-construction details

This section makes the encoder $\phi$ and the SHE feature vector fully concrete and reproducible.

Encoders. ϕ is OpenAI text-embedding-ada-002 (d=1536, returned $\ell _ { 2 }$ -normalized so dot product is cosine) for MIND and REES46, and a local 256-d TF-IDF + TruncatedSVD (LSA) space fit on item titles for Amazon-Beauty (bigram TF-IDF, ≤ 20k vocab, min<sub>df</sub> =2, sublinear tf; SVD to 256-d), which we then $\ell _ { \mathrm { 2 } } { \mathrm { - n o r m a l i z e } }$ . Within a dataset every text—candidate items, the K hypotheses, and the Scheme-A summary—passes through the same $\phi ,$ so all cosines are intra-space; we never compare across the ada-002 and LSA spaces.

What text is embedded. An item is embedded from its content string (MIND: news title; Amazon-Beauty: product title). A user history is summarized on the backbone side by a mean-pool of its item embeddings (unordered) or the final state of a GRU/SASRec over the ordered item embeddings; each hypothesis $h _ { k }$ and the Scheme-A summary are embedded directly from their LLM-generated text. Faithfulness uses the embedded evidence-category strings the hypothesis cites.

SHE feature block. For a candidate c the frozen branch emits a small fixed-width feature vector (no learned parameters inside the branch):

$$
\big [ \underbrace { f _ { \mathrm { m a x } } } _ { \substack { \mathrm { m a x } _ { k } \cos ( c , e _ { k } ) } } , \underbrace { f _ { \mathrm { m a x } } ^ { \gamma } } _ { \substack { \mathrm { m a x } _ { k } \gamma _ { k } \cos ( c , e _ { k } ) } } , \underbrace { f _ { \mathrm { m e a n } } } _ { \substack { \mathrm { ~ 1 ~ } K \sum _ { k } \cos ( c , e _ { k } ) } } , \underbrace { \gamma _ { k ^ { \star } } } _ { \substack { \mathrm { c o n f . ~ o f ~ b e s t ~ f a c e t } } } \big ] ,
$$

where $f _ { \mathrm { m a x } } ^ { \gamma }$ is the primary coordinate of Eq. 1. This block is concatenated with the backbone score $f _ { \mathrm { b a s e } } ( \pmb { c } )$ (and, in the Scheme-A ablation, the single summary match $f _ { A } { = } \cos ( c , e _ { \mathrm { s u m m a r y } } ) )$ .

Downstream head and hygiene. The concatenated features feed a late-fusion head only—an ℓ -regularized logistic ranker (C=1.0, class-balanced) for MIND reranking, and a logistic/Ridge probe for the REES46 acquisition study; nothing is back-propagated into ϕ or the LLM. Features are standardized with statisticsfit on training folds only (cross-fit), and on REES46 the 1536-d hypothesis embeddings are PCA-reduced before the low-N acquisition probe to avoid overfitting. All reported cosines and metrics are on the $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ vectors above.

## C Agent-side details

Metric is cosine on ℓ<sub>2</sub>-normalized embeddings. Each impression yields exactly K=3 Scheme-B hypotheses. Faithfulness bootstraps over impressions; distinctiveness averages pairwise 1 − cos; calibration uses 5-fold cross-fit isotonic regression with equal-frequency bins.

## D Controlled degradation sweep

On MIND, injecting $N ( 0 , \sigma ^ { 2 } )$ into the base feature and re-measuring the +SHE lift yields a smooth monotone rise from +0.0094 (ns, σ=0) to +0.0775 (σ=4). This is a diagnostic controlled corruption illustrating the gradient, not an achieved real-world lift.

## E Amazon-Beauty backbone check

On Amazon-Beauty (median history 5) the GRU (0.4104) is weaker than mean-pool $( 0 . 4 7 7 7 ) ;$ ordered access does not strengthen the backbone, so Amazon-Beauty cannot cleanly isolate ordering. It is reported as an additional backbone-strength check, consistent with the richness gradient (SHE significant over the weak ordered GRU, redundant over strong mean-pool).

## F Backbone robustness B1–B5

Full tables for the 5-seed sweep, degradation sweep, residualization $( \mathrm { S H E _ { r e s i d } = \mathrm { S H E - \mathrm { P r o j } _ { s e q } ( S H E ) } } ,$ retains 101%), redundancy probe $( R ^ { 2 } \leq 0 . 0 1 0 )$ , and the SASRec second backbone (+0.0179 over ordered SASRec) are in results/mind/backbone redundancy {2x2,slices,seed sweep,residualization,sa

## G Positive control for acquisition

Holding the real folds/base/features fixed and replacing the lift with a synthetic cluster signal at controllable cluster-SNR, the exact deployable policy recovers signal once cluster-SNR $\geq 0 . 2 0$ (MIND) / 0.35 (Amazon); real data sit at 0.075 / 0.056. Human-regime and random-rotated true-cluster variants shift the threshold but the real data remain far below all of them.

## H HTE-SNR sufficiency check

corr $( \hat { s } _ { i } , \Delta _ { i } )$ , out-of-fold $R ^ { 2 }$ , and a heterogeneity-SNR against a 200-draw permutation null are all inside the null band on both datasets (MIND $\mathrm { c o r r } = - 0 . 0 1 2 , R ^ { 2 } = - 0 . 0 1 0$ ; Amazon corr $= + 0 . 0 0 1 , R ^ { 2 } = - 0 . 0 0 5 )$ no learnable heterogeneity.

## I Reviewer-facing claim table

Table 5 is the master cross-dataset summary. Each main-text claim in results/paper claims.csv lists claim id, dataset, script, evidence CSV, figure, and whether the public artifact regenerates it (all R1–R13 offline results: yes; from-scratch LLM generation: requires gai-proxy).

Table 5: Master cross-dataset summary. $d = { \mathrm { p e r } } { \mathrm { - e x a m p l e } }$ effect (instance / best cluster); $\begin{array} { r } { \mathbf { M D E } _ { 8 0 } = \operatorname* { m i n } . } \end{array}$ imum detectable effect at 80% power; noise-repro% = fraction of the in-sample oracle reproduced by a matched-noise placebo; pos-ctrl thr = cluster-SNR at which the pipeline recovers a synthetic signal; real clust-SNR = measured.
<table><tr><td>Dataset</td><td>N</td><td> $\rho$ </td><td>instance d</td><td> $\mathrm { M D E _ { 8 0 } }$ </td><td>noise-repro%</td><td>pos-ctrl thr</td><td>real clust-SNR</td></tr><tr><td>MIND</td><td>1263</td><td>0.048</td><td>-0.0044</td><td>0.0124</td><td>62%</td><td>0.20</td><td>0.075</td></tr><tr><td>Amazon-Beauty</td><td>650</td><td>0.014</td><td>-0.0003</td><td>0.0075</td><td>242%</td><td>0.35</td><td>0.056</td></tr></table>

## J Datasets, window construction, and preprocessing

All three datasets are public. We convert each into fixed windows — one user history plus a prediction target — and generate hypotheses per window. Table 6 gives the exact construction. A “window” is a point-intime slice: the history is the observed behavior, and the target is a future click (MIND candidate slate), a future purchase (REES46 session), or a held-out next-item slate (Amazon-Beauty). Cold-start (sparse) and multi-intent slices are defined by fixed thresholds, not tuned.

<table><tr><td></td><td>MIND (news)</td><td>Amazon-Beauty</td><td>REES46</td></tr><tr><td>Domain</td><td>news reading</td><td>e-commerce</td><td>e-commerce</td></tr><tr><td>Target Windows generated</td><td>candidate click</td><td>next-item slate</td><td>session purchase</td></tr><tr><td></td><td>1,557</td><td>650</td><td>498</td></tr><tr><td>History length</td><td>median 19, mean 23.6</td><td>median 5</td><td>median ~9</td></tr><tr><td>Slate size (cap)</td><td>40 candidates</td><td>20 candidates</td><td>— (binary)</td></tr><tr><td>sparse rule</td><td> $n _ { \mathrm { h i s t } } \leq 5$ </td><td> $n _ { \mathrm { h i s t } } \leq 5$ </td><td> $n _ { \mathrm { h i s t } } \leq 5$ </td></tr><tr><td>multi rule</td><td> $\geq 8$  distinct cats</td><td> $\geq 2$  distinct subcats</td><td>(mostly single-cat)</td></tr><tr><td>Content richness</td><td>high (multi-topic)</td><td>high (titles)</td><td>low (87.9% single-cat)</td></tr></table>

Table 6: Window construction per dataset. MIND is built with a cohort of 1,600 impressions at stride 5 (keep 1 of every 5) and a 40-candidate slate cap; hypotheses were generated for 1,557 windows (1,549 valid for both schemes). Amazon-Beauty: 650 windows (222 sparse). REES46: 498 session windows. History length is the number of pre-target actions; the sparse/multi slices are the cold-start and multi-intent regimes analyzed throughout.

Per-analysis N. The number of rerankable windows in a given result can be smaller than the generated count (windows with an empty valid slate, or missing a valid hypothesis under a scheme, are dropped). We therefore report the exact N with each result (e.g. MIND N=1263 for the ada-002 late-fusion study, N=1411 for the backbone-conditional LSA study; Amazon N=650; REES46 N=498). No window is dropped on the basis of its outcome.

## K Hyperparameters

Every downstream component is deliberately small and CPU-only; the language model is the sole expensive step (App. L). Table 7 lists all settings; none are tuned against the reported test metrics.

## L Hypothesis-generation cost and latency

Because the language model is called once per window and never fine-tuned, the entire method cost is the generation pass; everything downstream (embedding, ranking, all robustness runs) is CPU-seconds. This asymmetry is exactly what motivates treating a hypothesis as a costly semantic observation: the question is not whether to train, but whether it is worth acquiring the observation at all. Table 8 reports the measured latency.

A design-time gate trades a small call reduction for equal accuracy. Since each acquisition is costly, one might hope to skip it on windows unlikely to benefit. Our floor result (§8) predicts that a learned perinstance router cannot do this reliably; a design-time regime gate (spend only on the sparse/multi-intent regimes) is the prescribed alternative. On MIND (N=1263, over the strong content baseline $[ f _ { \mathrm { h i s t } } ] )$ the gate calls the model on 86.4% of windows and matches spending everywhere: gate NDCG@ $1 0 = 0 . 4 5 5 4$ vs. spend-everywhere 0.4552 (difference +0.0001, 95% CI $[ - 0 . 0 0 3 8 , + 0 . 0 0 4 2 ] \rangle$ ). Relative to the base ranker the gate is $+ 0 . 0 0 9 6 \left[ - 0 . 0 0 0 4 , + 0 . 0 1 9 4 \right] - n o t$ statistically significant, matching spend-everywhere $\left( + 0 . 0 0 9 4 \left[ - 0 . 0 0 1 3 , + 0 . 0 1 9 9 \right] \right)$ . We therefore do not claim an accuracy gain here: on this strong baseline the design-time gate’s only realized benefit is a ∼14% reduction in expensive calls at no accuracy loss, and its non-significance over $f _ { \mathrm { h i s t } }$ is itself consistent with the detectability floor. The paper’s significant value results are the backbone-conditional lift over the ordered GRU (+0.0114, p=0.005, §5) and the agent-quality gains (§4); src/regime gate.py reproduces the numbers above.

<table><tr><td>Component</td><td>Setting</td><td>Value</td></tr><tr><td>Late-fusion ranker</td><td>model</td><td>logistic regression (pointwise)</td></tr><tr><td></td><td>regularization C max iterations</td><td>1.0 2000</td></tr><tr><td></td><td>class weighting</td><td>balanced</td></tr><tr><td></td><td>feature scaling</td><td>StandardScaler, fit on train fold only</td></tr><tr><td>Evaluation</td><td>cross-fit</td><td>GroupKFold(5), grouped by impression</td></tr><tr><td></td><td></td><td></td></tr><tr><td></td><td>prediction</td><td>out-of-fold click probability → rank</td></tr><tr><td></td><td>metrics</td><td>NDCG@10, Recall@10, MRR (per impression)</td></tr><tr><td>Item text space (LSA)</td><td>confidence interval</td><td>bootstrap 95%, 2000 resamples by impression</td></tr><tr><td></td><td>encoder</td><td>TF-IDF + TruncatedSVD</td></tr><tr><td>GRU backbone</td><td>dimensions</td><td>256 (random_state = 0)</td></tr><tr><td></td><td>hidden size</td><td>64</td></tr><tr><td></td><td>optimizer</td><td>Adam, lr  $1 0 ^ { - 2 }$  , weight decay  $1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>epochs</td><td>15, cross-fit (OOF)</td></tr><tr><td>SASRec backbone</td><td>attention</td><td>2 heads, 1 layer, max length 50</td></tr><tr><td></td><td>hidden size</td><td>64</td></tr><tr><td></td><td>optimizer</td><td>Adam, lr  $1 0 ^ { - 3 }$  , weight decay  $1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>epochs</td><td>15, cross-fit (OOF)</td></tr><tr><td>Robustness (B1–B5)</td><td>seeds</td><td>{0, 1, 2, 3, 4}</td></tr></table>

Table 7: Downstream hyperparameters. The ranker, backbones, and LSA space are fixed a priori; the GRU/SASRec inputs are the frozen LSA item vectors, so only the sequence-combination parameters are learned. All CIs use the same impression-level bootstrap.

<table><tr><td></td><td>Scheme A</td><td>Scheme B</td><td>Per window (A+B)</td></tr><tr><td>Model</td><td>GPT-5.4</td><td>GPT-5.5</td><td></td></tr><tr><td>Reasoning effort</td><td>low</td><td>high</td><td></td></tr><tr><td>Latency / call</td><td>~2-3s</td><td>~6-17 s</td><td></td></tr><tr><td>MIND</td><td></td><td></td><td>~19–23 s/win</td></tr><tr><td>Amazon-Beauty</td><td>3.0 s (mean)</td><td>8.6 s (mean)</td><td>26.3 s/win</td></tr></table>

Table 8: Measured generation cost. Scheme B (the SHE source) dominates because of high-effort reasoning; its latency is variable per window. End-to-end: Amazon-Beauty’s 650 windows took 284.8 min on a single worker; MIND’s 1,557 windows were generated across 8 parallel workers ( ∼63 min wall ). All downstream analyses run in CPU-seconds from the cached hypotheses, so results are fully reproducible offline without any further model calls.