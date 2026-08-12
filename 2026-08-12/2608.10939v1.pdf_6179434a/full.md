# A Cost-Efficient Routing Pipeline for Multilingual Short-Text Classification Using Small Language Models

Wajdi BEN SAAD

Carthago Labs

Paris, France

Safa MADIOUNI

Université Paris Dauphine-PSL

Paris, France

Abstract—Multilingual short-text classification supports operational systems such as content moderation, customer support routing, and intent recognition, yet aggregate evaluation often hides large differences between high-resource and low-resource languages. Uniform inference policies are simple to deploy, but they assume that all languages are equally well served. In this work, we evaluate a fixed-list routing strategy that keeps stronger languages on a direct multilingual path and selectively sends weaker languages through translation into English before zero-shot classification. The pipeline is fully self-hosted, uses pretrained compact sentence encoders, and requires no taskspecific fine-tuning.

We test the approach on two benchmarks chosen to differ in scale and label granularity: a 15-language subset of SIB-200 for seven-way topic classification and a 15-locale subset of MASSIVE for intent classification over an official 60-intent inventory. On SIB-200, the best overall configuration is R1, which translates only the low-resource tier: high-tier and mid-tier Macro-F1 remain unchanged, while low-tier Macro-F1 rises from 0.4632 to 0.6828. On the MASSIVE subset, the same low-tier intervention raises low-tier Macro-F1 from 0.2143 to 0.4417, but the best overall result is obtained by full translation, R3, at Macro-F1 0.4647. Across these two benchmarks, selective translation is a reliable intervention for weaker languages, whereas the optimal routing boundary depends on the task. We therefore report routing through tier-level quality gains and tier-level latency rather than a single global efficiency score.

Index Terms—multilingual text classification, routing, translation, sentence embeddings, low-resource languages, intent classification

## I. INTRODUCTION

Multilingual short-text classification is required in content moderation, search assistance, customer support, and multilingual monitoring systems. A common deployment choice is to apply one inference path to every language, either with a single multilingual encoder or by translating every input into a pivot language. That uniform treatment is convenient, but benchmarks such as XTREME [1], SIB-200 [2], and MASSIVE [3] show broad variation across languages, scripts, and task settings even under the same model family.

Translation-based transfer remains a competitive alternative to direct multilingual inference [4], [5], but it adds latency, depends on translation-model coverage, and may introduce artifacts [6]. Our work targets a local, cost-aware deployment regime in which compact self-hosted models remain attractive relative to larger generative systems with higher serving overhead [7], [8]. The practical question is therefore not only which method is strongest overall, but whether translation should be reserved for the languages that benefit from it most.

In this work, we examine that question with a fixedlist routing policy. We compare multilingual-only inference, translation-only inference, and a static router that sends selected language tiers through a translate-then-classify path. Unlike standard translate-and-test pipelines, our goal is not to replace direct multilingual inference everywhere, but to identify where translation should begin and where it should stop. Both paths use the same prototype-matching rule over English labels, and no task-specific fine-tuning is introduced. We conduct the analysis on a 15-language subset of SIB-200 for topic classification and a 15-locale subset of MASSIVE for intent classification. Although MASSIVE is commonly used for supervised multilingual natural language understanding (NLU), here we use only the public test split for zero-shot inference, with no training or weak supervision.

The main finding is localized rather than universal. On SIB-200, translating only the low-resource tier raises lowtier Macro-F1 from 0.4632 to 0.6828 while the high-tier and mid-tier remain unchanged, and R1 is the strongest overall configuration. On the MASSIVE subset, the same low-tier intervention raises low-tier Macro-F1 from 0.2143 to 0.4417, again without changing the stronger tiers under R1, but the strongest overall result is obtained by full translation, R3. Under this setup, the central design question is not whether routing should be made more complex, but where the translation boundary should be placed for the task at hand.

In this paper, we make three contributions:

1) We present a fully self-hosted empirical study of fixedlist routing for multilingual short-text classification without task-specific fine-tuning, evaluated on topic classification and intent classification.

2) We report the results in a deployment-oriented form based on delta Macro-F1 relative to a multilingual-only baseline and latency measured separately by resource tier.

3) We show that the routing boundary is dataset-dependent: SIB-200 is best served by translating only the lowtier, whereas the MASSIVE subset reaches its strongest overall result under full translation even though the lowtier R1 gain still persists.

The remainder of this paper is organised as follows: Section II reviews prior work on multilingual evaluation, crosslingual transfer, and routing-related deployment questions. Section III presents our routing pipeline, inference paths, and reporting strategy. Section IV describes the datasets, language tiers, models, execution environment, and evaluation conditions. Section V reports the main experimental results on SIB-200 and the MASSIVE subset. Section VI interprets the findings, highlights practical implications, and states the main limitations. Section VII concludes the paper and outlines directions for future work.

## II. RELATED WORK

Two strands of prior work are directly relevant here: multilingual evaluation and cross-lingual transfer mechanisms. XTREME [1] and XTREME-R [9] established broad multilingual benchmark suites, SIB-200 [2] widened topicclassification evaluation to a much larger set of languages and dialects, and MASSIVE [3] provided a large multilingual NLU resource with a fine-grained intent inventory. Together, these benchmarks show that multilingual performance is not evenly distributed across languages, scripts, or label spaces. That observation motivates our decision to study routing as a language-dependent design choice rather than to assume one universal path.

Our classification pipeline relies on sentence-level representations. Sentence-BERT [10], LASER [11], and LaBSE [12] showed that sentence embeddings can support strong zero-shot transfer across many languages. Broader multilingual encoder families such as XLM-R [13], InfoXLM [14], and VECO [15] further strengthened the multilingual representation toolkit. This matters here because it lets us compare routing decisions without changing the downstream classifier family. We do not propose a new embedding model; instead, we use compact sentence encoders in a prototype-matching setting so that representation quality remains approximately constant while the routing policy is varied.

Translate-and-test methods provide the second point of comparison. T3L [4] and the re-evaluation by Artetxe et al. [5] show that translation-based transfer remains competitive. At the same time, translation artifacts can distort downstream behavior [6], and full translation pipelines introduce non-trivial operational cost. Larger multilingual generative models such as mT5 [7] and XGLM [8] expand multilingual transfer further, but they target a different operating regime from the compact self-hosted setup studied here.

Work on adaptive inference, such as DeeBERT [16], has shown that selective execution can reduce unnecessary computation when a uniform path is not justified for every input. Our work is related in spirit but differs in granularity: DeeBERT adapts computation depth within a model, whereas we adapt the inference path at the language-tier level under a fixed policy. For local deployment, open translation systems such as OPUS-MT [17] and broader-coverage models such as NLLB [18] make this kind of routing technically feasible.

The gap we address is therefore specific. Prior work has established strong multilingual encoders, strong translate-andtest pipelines, and adaptive inference within models, but it does not directly ask how a simple fixed language list should place the translation boundary in a fully self-hosted zero-shot classifier. Our work addresses that question by holding the classifier constant, sweeping the routing boundary explicitly, and reporting both quality gains and latency at the resource tier where the intervention is applied.

## III. METHOD

## A. Task Formulation

The primary benchmark is seven-way topic classification on SIB-200, and the secondary benchmark is intent classification on a 15-locale subset of MASSIVE. In both cases, let x be an input text and let Y be the set of English labels for the current dataset. For each encoder, we embed the label texts once to obtain a prototype bank $\{ p _ { y } \} _ { y \in \mathcal { V } }$ . Given an input representation $h ( x )$ , prediction is:

$$
\hat { y } = \arg \operatorname* { m a x } _ { y \in \mathcal { V } } \cos ( h ( x ) , p _ { y } ) .\tag{1}
$$

This formulation keeps the decision rule identical across all conditions. The experiment therefore compares inference paths rather than classifier architectures. For SIB-200, the label texts are the seven English topic names. For MASSIVE, the label inventory is extracted programmatically from the official 60-intent metadata, and each intent is represented by a short English description rather than its raw snake-case identifier. This adjustment was adopted after an English-only sanity check showed that the raw identifiers were underdiscriminative for prototype matching.

## B. Inference Paths

We evaluate two paths.

• Multilingual path. The original text is encoded directly with the multilingual checkpoint sentence-transformers/ paraphrase-multilingual-MiniLM-L12-v2.

• Translate-then-classify path. The text is translated locally into English and encoded with the English checkpoint sentence-transformers/paraphrase-MiniLM-L6-v2.

Translation uses Helsinki-NLP OPUS-MT models where a dedicated source-to-English checkpoint exists. If no such checkpoint is available, the system falls back to facebook/ nllb-200-distilled-600M. All translations are cached and reused across routing settings.

![](images/0415c6407a4c8e9ee11b5df83a601d435b0608266251ba07e78ade58465a35d7.jpg)  
Fig. 1. Inference pipeline and routing policy under R0–R3, with each language tier routed to either the multilingual path or the translate-then-classify path.

## C. Routing Policy

The router is a static lookup from language to inference path. Languages are assigned to three analysis tiers: high, mid, and low. The tiers are not separate models. They are reporting groups used to decide which languages are sent through translation under each routing boundary.

We evaluate four routing configurations.

• R0: no translation

• R1: translate only the low-tier

• R2: translate the mid-tier and low-tier

• R3: translate all tiers

The multilingual-only baseline is equivalent to R0, and the translation-only baseline is equivalent to R3. We retain the named baselines because they are useful reference points in the tables.

Figure 1 summarizes the inference pipeline and the routing policy in one view. The classifier itself remains unchanged across all conditions; only the path taken by a language tier changes.

## D. Reporting Strategy

The headline analysis does not rely on a single composite efficiency ratio. Instead, it reports:

• delta Macro-F1 relative to the multilingual-only baseline • mean latency by resource tier

This format makes the effect of routing observable at the point where it occurs. If a tier is left unchanged, the corresponding delta is near zero and its latency remains near the multilingual baseline. If a tier is translated, the quality gain and the translation cost can be read together.

## IV. EXPERIMENTAL SETUP

## A. Benchmarks and Language Selection

We evaluate the routing policy on two benchmarks. The primary benchmark is Davlan/sib200 [2], used here for seven-way topic classification on a 15-language subset with 3,060 test examples. The secondary benchmark is the public AmazonScience/massive resource [3], used here only through its test split for zero-shot intent classification on a 15-locale subset with 44,610 test examples. Each selected MASSIVE locale contributes 2,974 test examples. The official MASSIVE intent catalog contains 60 intents, although the chosen 15-locale subset observes 59 of them in the selected test examples. The label sets remain in English on both inference paths. For SIB-200, the labels are science/technology, travel, politics, sports, health, entertainment, and geography. For MASSIVE, the official 60-intent inventory is extracted programmatically from the dataset metadata and converted into short English intent descriptions for prototype encoding.

Languages are grouped into high, mid, and low analysis tiers. These tiers are not claimed as a universal resource taxonomy. They are an experimental device used to sweep the translation boundary within each benchmark. The hightier groups languages that are comparatively well handled by direct multilingual inference, the low-tier groups languages for which translation is more plausible as an intervention, and the mid-tier captures the remaining intermediate cases. The lowtier is aligned across both benchmarks, while the high-tier and mid-tier differ slightly because the available language sets are not identical.

We fixed these groups after inspecting multilingual-only per-language Macro-F1 within each benchmark, but we did not threshold on Macro-F1 alone. Instead, we used the baseline scores to anchor a shared weak-language set across the two benchmarks and then completed the remaining high-tier and mid-tier split with dataset availability and cross-benchmark comparability in mind. The low-tier multilingual-only scores are clearly weak on MASSIVE (Swahili 0.1309, Amharic 0.1501, Telugu 0.1990, Bengali 0.2066, Afrikaans 0.3197) and are again among the weaker languages on SIB-200 (Telugu 0.2780, Bengali 0.3848, Amharic 0.4348, Swahili 0.4537, Afrikaans 0.6911). We therefore treat the tiers as benchmarkspecific routing groups rather than as a universal languageresource taxonomy.

## B. Models

The multilingual path uses sentence-transformers/ paraphrase-multilingual-MiniLM-L12-v2. The English path uses sentence-transformers/paraphrase-MiniLM-L6-v2. Both are compact sentence-embedding checkpoints in the Sentence-Transformer family [10]. We chose a compact multilingual encoder rather than a larger checkpoint such as XLM-R [13] because the paper targets a self-hosted, cost-aware local deployment setting rather than a modelcapacity comparison. Translation uses OPUS-MT [17] when a language-specific source-to-English checkpoint exists and NLLB [18] otherwise. No task-specific fine-tuning is performed.

TABLE I  
BENCHMARK-SPECIFIC ANALYSIS TIERS USED IN THE ROUTING SWEEP.
<table><tr><td>Benchmark</td><td>Tier</td><td>Languages</td></tr><tr><td rowspan="3">SIB-200</td><td></td><td>High English, French, German, Spanish, Chinese</td></tr><tr><td>Mid</td><td>Arabic, Turkish, Japanese, Polish, Dutch</td></tr><tr><td>Low</td><td>Swahili, Bengali, Telugu, Amharic, Afrikaans</td></tr><tr><td rowspan="3"></td><td>MASSIVE subset High</td><td>English, French, German, Spanish, Portuguese</td></tr><tr><td>Mid</td><td>Arabic, Turkish, Polish, Dutch, Romanian</td></tr><tr><td>Low</td><td>Swahili, Telugu, Bengali, Amharic, Afrikaans</td></tr></table>

## C. Conditions

We evaluate three named conditions:

• Multilingual-only: every language uses the multilingual path

• Translation-only: every language uses translate-thenclassify

• Routing: the fixed-list router applies one of R0 to R3

Within SIB-200, Bengali and Afrikaans use OPUS-MT, whereas Amharic, Swahili, and Telugu use the NLLB fallback. Within the MASSIVE subset, French, German, Spanish, Arabic, Turkish, Polish, Dutch, Bengali, and Afrikaans use OPUS-MT, whereas Portuguese, Romanian, Swahili, Telugu, and Amharic use the NLLB fallback. This mixed backend is part of the evaluated system rather than a post hoc adjustment.

## D. Execution Environment and Metrics

The reported runs were executed on the same local workstation with an Intel Core i7-9750H processor, 16 GB RAM, Python 3.11.14, PyTorch 2.2.2, Transformers 4.57.6, and Datasets 4.8.4. The purpose of this setup is not to maximize throughput, but to keep the measurements tied to a reproducible local deployment scenario. For the MASSIVE subset, NLLB-backed translation locales are executed locally with single-beam decoding so the translation stage remains stable and fully cached.

For each condition, we report overall Macro-F1, accuracy, and mean latency. Our tables use the mean of three cached reruns, with standard deviations retained for audit. The main analysis reports:

• delta Macro-F1 against the multilingual-only baseline

• mean latency for each resource tier

For repeated-use settings with cached translations, effective latency can also be summarized as $t _ { \mathrm { c l a s s i f y } } + t _ { \mathrm { t r a n s l a t e } } / N _ { \mathrm { : } }$ where $t _ { \mathrm { c l a s s i f y } }$ is the observed mean classification latency, $t _ { \mathrm { t r a n s l a t e } }$ is the observed mean translation latency for the relevant tier, and N is the average number of reuses per unique input. We also retain the full threshold sweep from R0 to

TABLE II  
OVERALL ROUTING SWEEP ON THE SIB-200 SUBSET.
<table><tr><td>Condition</td><td>Macro-F1</td><td>Accuracy</td><td>Latency (s)</td></tr><tr><td>Multilingual-only</td><td>0.6687</td><td>0.6614</td><td>0.0036</td></tr><tr><td>R0</td><td>0.6687</td><td>0.6614</td><td>0.0036</td></tr><tr><td>R1</td><td>0.7403</td><td>0.7415</td><td>0.2753</td></tr><tr><td>R2</td><td>0.7195</td><td>0.7193</td><td>0.4095</td></tr><tr><td>R3</td><td>0.6979</td><td>0.6961</td><td>0.5380</td></tr><tr><td>Translation-only</td><td>0.6979</td><td>0.6961</td><td>0.5380</td></tr></table>

TABLE III

SIB-200: R1 VERSUS MULTILINGUAL-ONLY BY RESOURCE TIER.
<table><tr><td>Tier</td><td>Base F1</td><td>R1 F1</td><td>∆F1</td><td>Base Lat.</td><td>R1 Lat.</td></tr><tr><td>High</td><td>0.7700</td><td>0.7700</td><td>0.0000</td><td>0.00345</td><td>0.00350</td></tr><tr><td>Mid</td><td>0.7674</td><td>0.7674</td><td>0.0000</td><td>0.00358</td><td>0.00369</td></tr><tr><td>Low</td><td>0.4632</td><td>0.6828</td><td>+0.2196</td><td>0.00385</td><td>0.81882</td></tr></table>

R3, because the sweep is used to identify where translation should stop. On SIB-200, that sweep reveals a localized lowtier benefit. On the MASSIVE subset, it shows that the same low-tier gain persists but that the strongest overall boundary is broader.

The implementation, configuration files, notebooks, and experiment artifacts will be made available at https://github. com/WajdiBenSaad/multilingual-routing-classifier.

## V. RESULTS

Unless otherwise noted, all reported values are means over three cached reruns.

## A. Primary Benchmark: SIB-200

Table II summarizes the SIB-200 routing sweep as means over three cached reruns. The multilingual-only baseline and R0 are numerically identical, as expected. The strongest overall result is obtained by R1, which translates only the low-tier. Moving the boundary to R2 lowers Macro-F1 from 0.7403 to 0.7195 while increasing mean latency from 0.27534 s to 0.40952 s. Full translation, represented by R3 and the translation-only baseline, remains below R1 on overall quality and is slower still.

The central comparison is R1 against multilingual-only. Table III shows that the high-tier and mid-tier are unchanged to four decimal places, whereas the low-tier improves by +0.2196 Macro-F1. The latency increase is likewise localized: high-tier and mid-tier latency remain close to the multilingual baseline, while low-tier mean latency rises to 0.8188 s because translation is invoked only for that tier.

Across the three cached reruns, Macro-F1 and accuracy were identical at the saved precision for every SIB-200 condition. Standard deviations are therefore informative mainly for latency, where the observed variation remains at the fourth decimal place or smaller. We also tested a reporting sensitivity case in which Arabic and Japanese are moved from the mid-tier to the high-tier. That regrouping leaves the low-tier R1 gain unchanged at +0.2196 Macro-F1 and leaves overall Macro-F1 unchanged at 0.7403. Within the low-tier, the largest SIB-200 gains occur in Telugu (+0.4105), Bengali (+0.2914),

TABLE IV  
OVERALL ROUTING SWEEP ON THE MASSIVE SUBSET.
<table><tr><td>Condition</td><td>Macro-F1</td><td>Accuracy</td><td>Latency (s)</td></tr><tr><td>Multilingual-only</td><td>0.3785</td><td>0.3855</td><td>0.0022</td></tr><tr><td>R0</td><td>0.3785</td><td>0.3855</td><td>0.0022</td></tr><tr><td>R1</td><td>0.4434</td><td>0.4704</td><td>0.0620</td></tr><tr><td>R2</td><td>0.4589</td><td>0.4856</td><td>0.1139</td></tr><tr><td>R3</td><td>0.4647</td><td>0.4943</td><td>0.1515</td></tr><tr><td>Translation-only</td><td>0.4647</td><td>0.4943</td><td>0.1515</td></tr></table>

TABLE V

MASSIVE: R1 VERSUS MULTILINGUAL-ONLY BY RESOURCE TIER.
<table><tr><td>Tier</td><td>Base F1</td><td>R1 F1</td><td>∆F1</td><td>Base Lat.</td><td>R1 Lat.</td></tr><tr><td>High</td><td>0.4645</td><td>0.4645</td><td>0.0000</td><td>0.00185</td><td>0.00198</td></tr><tr><td>Mid</td><td>0.4199</td><td>0.4199</td><td>0.0000</td><td>0.00182</td><td>0.00196</td></tr><tr><td>Low</td><td>0.2143</td><td>0.4417</td><td>+0.2275</td><td>0.00185</td><td>0.18170</td></tr></table>

Amharic (+0.2549), and Swahili (+0.2011), whereas Afrikaans improves only marginally (+0.0135).

## B. Secondary Benchmark: MASSIVE Subset

Table IV summarizes the MASSIVE routing sweep as means over three cached reruns. The overall pattern differs from SIB-200 in one important respect. The localized lowtier intervention under R1 is again strong, but the best overall configuration is no longer selective routing. R1 improves overall Macro-F1 from 0.3785 to 0.4434, R2 reaches 0.4589, and the strongest overall result is obtained by full translation, R3, with Macro-F1 0.4647 and accuracy 0.4943. At the reported precision, R3 and the translation-only baseline are numerically tied.

The localized pattern under R1 nevertheless persists. Table V shows that high-tier and mid-tier Macro-F1 remain unchanged to four decimal places, whereas low-tier Macro-F1 rises from 0.2143 to 0.4417, a gain of +0.2275, with low-tier mean latency rising from 0.0023 s to 0.1817 s.

The per-language gains inside the low-tier are concentrated in Swahili (+0.3116), Telugu (+0.2796), Amharic (+0.2406), Bengali (+0.2297), and Afrikaans (+0.1375). Table VII places these low-tier deltas side by side with the SIB-200 values. What changes relative to SIB-200 is not the existence of the low-tier effect, but the best global boundary: on MASSIVE, extending translation to the mid-tier and then to all tiers continues to improve overall performance.

## C. Cross-Dataset Comparison

The most compact cross-dataset summary is the best boundary selected by each benchmark. SIB-200 reaches its strongest result under R1 with Macro-F1 0.7403 and an R1 low-tier gain of +0.2196. The MASSIVE subset reaches its strongest result under R3 with Macro-F1 0.4647, while its R1 low-tier gain is +0.2275. The repeated low-tier gain therefore generalizes across benchmarks, whereas the best global boundary does not.

The low-tier is served by a mixed local translation backend in both benchmarks. On SIB-200, Bengali and Afrikaans use OPUS-MT, whereas Telugu, Amharic, and Swahili use the NLLB fallback. On the MASSIVE subset, Bengali and Afrikaans again use OPUS-MT, whereas Swahili, Telugu, and Amharic use the NLLB fallback; Portuguese and Romanian also use NLLB outside the low-tier. Gains are therefore not confined to one backend family.

TABLE VI  
BEST OBSERVED CONFIGURATION BY BENCHMARK.
<table><tr><td>Dataset</td><td></td><td>Best cfg. Macro-F1</td><td>Accuracy</td><td>R1 low ∆F1</td></tr><tr><td>SIB-200</td><td>R1</td><td>0.7403</td><td>0.7415</td><td>+0.2196</td></tr><tr><td>MASSIVE</td><td>R3</td><td>0.4647</td><td>0.4943</td><td>+0.2275</td></tr></table>

TABLE VII

LOW-TIER PER-LANGUAGE R1 DELTAS VERSUS MULTILINGUAL-ONLY.
<table><tr><td>Dataset</td><td>Swahili</td><td>Telugu</td><td>Bengali</td><td>Amharic</td><td>Afrikaans</td></tr><tr><td>SIB-200</td><td>+0.2011</td><td>+0.4105</td><td>+0.2914</td><td>+0.2549</td><td>+0.0135</td></tr><tr><td>MASSIVE</td><td>+0.3116</td><td>+0.2796</td><td>+0.2297</td><td>+0.2406</td><td>+0.1375</td></tr></table>

Table VIII isolates the most stable cross-dataset pattern in compact form. In both benchmarks, high-tier and mid-tier deltas remain exactly zero at the reported precision under R1, whereas the low-tier carries the entire quality gain and nearly all added latency.

Under R1, the high-tier and mid-tier examples remain on the multilingual path by construction. Their predictions are therefore identical to the multilingual-only baseline, with zero fixed examples and zero regressed examples in both benchmarks. The paired analyses below focus on the translated low-tier, where predictions actually change.

Table IX shows that the R1 low-tier gains reflect many more corrected predictions than newly introduced errors. The Afrikaans rows explain why its net gain is smaller: on SIB-200, translation fixes 23 Afrikaans examples but regresses 21, leaving only a small net change.

Table X indicates that the main low-tier R1 improvements are not explained by a small number of isolated prediction changes. The repeated cached reruns are retained as a repeatability check, while the paired test uses example-level correctness changes between multilingual-only and R1.

Across both benchmarks, the rerun standard deviations are effectively zero at the stored precision for Macro-F1 and accuracy because the evaluation is deterministic once prototypes and cached translations are fixed. The reruns are therefore most useful as a repeatability audit and as a check that the latency measurements remain stable.

Because translations are cached, the measured per-example latency also admits a repeated-use interpretation. We do not treat effective cached latency as a replacement for the measured end-to-end latency, but it remains relevant for deployment scenarios in which the same or highly similar queries recur.

## VI. DISCUSSION

Our results support a practical but limited claim. Fixedlist routing is not a universal replacement for multilingual inference; it is a boundary-selection problem whose answer depends on the benchmark. On SIB-200, the high-tier and midtier are already well handled by the multilingual path, whereas the low-tier benefits substantially from translation into English. On the MASSIVE subset, that same low-tier rescue effect remains strong, but the best overall setting extends translation further.

TABLE VIII  
CROSS-DATASET R1 SUMMARY BY TIER. DELTA MACRO-F1 IS MEASURED AGAINST MULTILINGUAL-ONLY.
<table><tr><td>Dataset</td><td>Tier</td><td>∆F1</td><td>R1 Latency (s)</td></tr><tr><td rowspan="3">SIB-200</td><td>High</td><td>0.0000</td><td>0.00350</td></tr><tr><td>Mid</td><td>0.0000</td><td>0.00369</td></tr><tr><td>Low</td><td>+0.2196</td><td>0.81882</td></tr><tr><td rowspan="3">MASSIVE</td><td>High</td><td>0.0000</td><td>0.00198</td></tr><tr><td>Mid</td><td>0.0000</td><td>0.00196</td></tr><tr><td>Low</td><td>+0.2275</td><td>0.18170</td></tr></table>

That distinction matters for system design. A simple lookup table was sufficient to expose the main structural effect, but the effect was not identical across tasks. On SIB-200, once translation crosses the low-tier boundary and is extended to the mid-tier, the gains do not persist. This is consistent with the multilingual encoder already providing adequate coverage for mid-tier languages, making translation more likely to add noise than useful transfer for that benchmark. On the MASSIVE subset, the opposite trend appears: the low-tier benefit under R1 is retained, but broader translation continues to improve the overall result. Our current explanation is a hypothesis rather than a demonstrated causal finding. MASSIVE uses a denser 60-intent inventory, and even in the selected subset the evaluation spans 59 observed intents, which may make the direct multilingual prototype path more sensitive to finegrained lexical distinctions than the seven-way topic setup in SIB-200.

Afrikaans is an informative outlier within the low-tier. On SIB-200, its multilingual-only Macro-F1 is already 0.6911 and R1 raises it only to 0.7045, a gain of +0.0135; the paired error analysis shows 23 fixed Afrikaans examples and 21 regressed examples. On MASSIVE, Afrikaans again has the strongest low-tier direct baseline, rising from 0.3197 to 0.4571 under R1, but it remains the smallest low-tier gain (+0.1375), with 658 fixed examples and 244 regressed examples. This pattern is consistent with prior cross-lingual NLP evidence that embedding similarity can predict transfer success [19], but we treat the proximity explanation as a hypothesis rather than a causal finding.

Our study also illustrates a reproducibility point that is often omitted in benchmark reports. A fully local multilingual pipeline may require a hybrid translation backend because bilingual model availability is uneven. Our final system combines OPUS-MT and NLLB for exactly that reason.

For practical deployment, a team can begin with direct multilingual inference for all languages, identify the subset that remains weak under that baseline, and reserve translation only for that subset. Our results suggest that such a policy does not need a complicated learned router to be useful.

The two-benchmark design adds a methodological lesson. SIB-200 and MASSIVE do not simply differ in scale; they probe different classification regimes. Evaluating both lets us separate what appears stable across tasks, namely the repeated low-tier gain, from what remains benchmark-specific, namely the location of the strongest global routing boundary.

Several limitations should be stated plainly. First, two datasets and two task families are still not enough to treat any discovered boundary as universal. Second, the tier assignment is hand-crafted for the experiment; it is not learned from data. Third, the MASSIVE benchmark requires a reported design choice: short English intent descriptions are used as prototype texts because the raw snake-case intent names were too weak in the English-only sanity check. Fourth, the selected MASSIVE subset observes 59 of the 60 official intents, even though the full official inventory is retained in the prototype bank. Fifth, the measurements are local and model-specific, which suits our deployment framing but does not cover every production environment.

Finally, the latency numbers require interpretation rather than compression into a single scalar. Translation cost is substantial in per-example terms, yet it is localized to the translated tier and can be amortized in repeated-use settings through caching. For this reason, we report quality deltas and tier-specific latency side by side.

## VII. CONCLUSION

In this work, we evaluated a fully self-hosted and cost-aware routing pipeline for multilingual short-text classification on two benchmarks. Using pretrained compact sentence encoders and no task-specific fine-tuning, we compared multilingualonly inference, translation-only inference, and a fixed routing sweep over three resource tiers.

The main result is that the best routing boundary is not universal. On SIB-200, translating only the low-resource tier substantially improves low-tier Macro-F1 while leaving the stronger tiers unchanged, and that selective policy is the best overall configuration. On the MASSIVE subset, the same lowtier intervention again yields a large improvement, but the strongest overall result is obtained by full translation, R3. Under the present setup, the routing boundary is therefore the central practical design choice.

The main quantitative outcomes are as follows. On SIB-200, R1 is the best configuration, reaching Macro-F1 0.7403 and improving low-tier Macro-F1 by +0.2196 over multilingualonly. On the MASSIVE subset, R3 is the best overall configuration with Macro-F1 0.4647, while the R1 low-tier intervention still improves low-tier Macro-F1 by +0.2275.

Two aspects of our work are worth emphasizing. First, the evidence supports reporting routing effects at the tier level rather than through a single global efficiency score. That makes it possible to see where translation helps, where it does not, and how much latency is actually incurred by the affected languages. Second, the repeated low-tier gain across both benchmarks is the most stable empirical result we observe.

TABLE IX  
QUALITATIVE PAIRED ERROR ANALYSIS FOR R1 VERSUS MULTILINGUAL-ONLY. “FIXED” COUNTS EXAMPLES THAT ARE WRONG UNDER MULTILINGUAL-ONLY AND CORRECT UNDER R1; “REGRESSED” COUNTS THE REVERSE.
<table><tr><td>Dataset</td><td>Scope</td><td>Base Correct</td><td>R1 Correct</td><td>Fixed by R1</td><td>Regressed by R1</td></tr><tr><td>SIB-200</td><td>Low-tier</td><td>450</td><td>695</td><td>329</td><td>84</td></tr><tr><td>SIB-200</td><td>Afrikaans</td><td>142</td><td>144</td><td>23</td><td>21</td></tr><tr><td>MASSIVE</td><td>Low-tier</td><td>3233</td><td>7019</td><td>4606</td><td>820</td></tr><tr><td>MASSIVE</td><td>Afrikaans</td><td>1016</td><td>1430</td><td>658</td><td>244</td></tr></table>

TABLE X

PAIRED SIGNIFICANCE CHECK FOR THE TRANSLATED LOW-TIER UNDER R1. THE TEST IS MCNEMAR-STYLE OVER SAVED PAIRED EXAMPLE-LEVEL PREDICTIONS, NOT OVER THE DETERMINISTIC CACHED RERUNS.
<table><tr><td>Dataset</td><td>Comparison</td><td>Scope</td><td>∆ Macro-F1</td><td>Fixed</td><td>Regressed</td><td>Paired test</td></tr><tr><td>SIB-200</td><td>R1 vs multilingual-only</td><td>Low-tier</td><td>+0.2196</td><td>329</td><td>84</td><td> $p < 0 . 0 0 1$ </td></tr><tr><td>MASSIVE</td><td>R1 vs multilingual-only</td><td>Low-tier</td><td>+0.2275</td><td>4606</td><td>820</td><td> $p < 0 . 0 0 1$ </td></tr></table>

On SIB-200, low-tier R1 Macro-F1 rises by +0.2196. On the MASSIVE subset, low-tier R1 Macro-F1 rises by +0.2275. What changes across datasets is not whether the low-tier benefits, but how far translation should be extended beyond it.

The contribution is therefore empirical and operational. We show where translation helps, where it does not, and how that decision can be implemented with a simple fixed-list policy using only local resources. More broadly, our results argue for reporting multilingual deployment outcomes at the point where interventions are applied rather than collapsing all languages into a single system-level score. In our experiments, that perspective reveals two distinct outcomes: SIB-200 favors targeted low-tier translation, whereas the larger and harder MASSIVE benchmark favors broader translation. That difference is not a weakness. It is the main cross-dataset finding.

Future work can extend the same analysis to additional datasets, replace the hand-crafted boundary with a learned policy, and test whether task-specific fine-tuning changes the boundary location. It would also be useful to study whether finer-grained language clustering or confidence-aware fallback rules can recover some of the MASSIVE gains without extending translation to every tier. Even without those additions, this work already establishes a useful baseline for deploymentoriented multilingual evaluation: test the routing boundary directly, report the gain and the cost at the tier where the intervention is applied, document any hybrid translation backend honestly, and do not assume that the same boundary will transfer unchanged across tasks.

## REFERENCES

[1] J. Hu, S. Ruder, A. Siddhant, G. Neubig, O. Firat, and M. Johnson, “Xtreme: A massively multilingual multi-task benchmark for evaluating cross-lingual generalisation,” in Proceedings of the 37th International Conference on Machine Learning, ser. Proceedings of Machine Learning Research, vol. 119. PMLR, 2020, pp. 4411–4421. [Online]. Available: https://proceedings.mlr.press/v119/hu20b.html

[2] D. I. Adelani, H. Liu, X. Shen, N. Vassilyev, J. O. Alabi, Y. Mao, H. Gao, and E.-S. A. Lee, “Sib-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects,” in Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics

(Volume 1: Long Papers). St. Julian’s, Malta: Association for Computational Linguistics, 2024, pp. 226–245. [Online]. Available: https://aclanthology.org/2024.eacl-long.14/

[3] J. FitzGerald, C. Hench, C. Peris, S. Mackie, K. Rottmann, A. Sanchez, A. Nash, L. Urbach, V. Kakarala, R. Singh, S. Ranganath, L. Crist, M. Britan, W. Leeuwis, G. Tur, and P. Natarajan, “Massive: A 1m-example multilingual natural language understanding dataset with 51 typologically-diverse languages,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Toronto, Canada: Association for Computational Linguistics, 2023, pp. 4277–4302. [Online]. Available: https://aclanthology.org/2023.acl-long.235/

[4] I. J. Unanue, G. Haffari, and M. Piccardi, “T3l: Translate-and-test transfer learning for cross-lingual text classification,” Transactions of the Association for Computational Linguistics, vol. 11, pp. 1147–1161, 2023. [Online]. Available: https://aclanthology.org/2023.tacl-1.65/

[5] M. Artetxe, V. Goswami, S. Bhosale, A. Fan, and L. Zettlemoyer, “Revisiting machine translation for cross-lingual classification,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. Singapore: Association for Computational Linguistics, 2023, pp. 6489–6499. [Online]. Available: https://aclanthology.org/2023.emnlp-main.399/

[6] M. Artetxe, G. Labaka, and E. Agirre, “Translation artifacts in crosslingual transfer learning,” in Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Online: Association for Computational Linguistics, 2020, pp. 7674–7684. [Online]. Available: https://aclanthology.org/2020.emnlp-main.618/

[7] L. Xue, N. Constant, A. Roberts, M. Kale, R. Al-Rfou, A. Siddhant, A. Barua, and C. Raffel, “mt5: A massively multilingual pre-trained text-to-text transformer,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Online: Association for Computational Linguistics, 2021, pp. 483–498. [Online]. Available: https://aclanthology.org/2021.naacl-main.41/

[8] X. V. Lin, T. Mihaylov, M. Artetxe, T. Wang, S. Chen, D. Simig, M. Ott, N. Goyal, S. Bhosale, J. Du, R. Pasunuru, S. Shleifer, P. S. Koura, V. Chaudhary, B. O’Horo, J. Wang, L. Zettlemoyer, Z. Kozareva, M. Diab, V. Stoyanov, and X. Li, “Few-shot learning with multilingual generative language models,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. Abu Dhabi, United Arab Emirates: Association for Computational Linguistics, 2022, pp. 9019–9052. [Online]. Available: https://aclanthology.org/2022.emnlp-main.616/

[9] S. Ruder, N. Constant, J. Botha, A. Siddhant, O. Firat, J. Fu, P. Liu, J. Hu, D. Garrette, G. Neubig, and M. Johnson, “Xtreme-r: Towards more challenging and nuanced multilingual evaluation,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. Online and Punta Cana, Dominican Republic: Association for Computational Linguistics, 2021, pp. 10 215–10 245. [Online]. Available: https://aclanthology.org/2021.emnlp-main.802/

[10] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings

using siamese bert-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP). Hong Kong, China: Association for Computational Linguistics, 2019, pp. 3982–3992. [Online]. Available: https://aclanthology.org/D19-1410/

[11] M. Artetxe and H. Schwenk, “Massively multilingual sentence embeddings for zero-shot cross-lingual transfer and beyond,” Transactions ofthe Associationfor Computational Linguistics, vol. 7, pp. 597–610, 2019. [Online]. Available: https://aclanthology.org/Q19-1038/

[12] F. Feng, Y. Yang, D. Cer, N. Arivazhagan, and W. Wang, “Language-agnostic bert sentence embedding,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Dublin, Ireland: Association for Computational Linguistics, 2022, pp. 878–891. [Online]. Available: https://aclanthology.org/2022.acl-long.62/

[13] A. Conneau, K. Khandelwal, N. Goyal, V. Chaudhary, G. Wenzek, F. Guzmán, E. Grave, M. Ott, L. Zettlemoyer, and V. Stoyanov, “Unsupervised cross-lingual representation learning at scale,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, 2020, pp. 8440–8451. [Online]. Available: https://aclanthology.org/2020.acl-main.747/

[14] Z. Chi, L. Dong, B. Zheng, S. Huang, X.-L. Mao, H. Huang, and F. Wei, “Infoxlm: An information-theoretic framework for cross-lingual language model pre-training,” in Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. Online: Association for Computational Linguistics, 2021, pp. 3576–3588. [Online]. Available: https://aclanthology.org/2021.naacl-main.280/

[15] F. Luo, W. Wang, J. Liu, Y. Liu, B. Bi, S. Huang, F. Huang, and L. Si, “Veco: Variable and flexible cross-lingual pre-training for language understanding and generation,” in Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Online: Association for Computational Linguistics, 2021, pp. 3980–3994. [Online]. Available: https://aclanthology.org/2021.acl-long.308/

[16] J. Xin, R. Tang, J. Lee, Y. Yu, and J. Lin, “Deebert: Dynamic early exiting for accelerating bert inference,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics. Online: Association for Computational Linguistics, 2020, pp. 2246– 2251. [Online]. Available: https://aclanthology.org/2020.acl-main.204/

[17] J. Tiedemann and S. Thottingal, “Opus-mt – building open translation services for the world,” in Proceedings of the 22nd Annual Conference of the European Association for Machine Translation. Lisboa, Portugal: European Association for Machine Translation, 2020, pp. 479–480. [Online]. Available: https://aclanthology.org/2020.eamt-1.61/

[18] NLLB Team, “Scaling neural machine translation to 200 languages,” Nature, vol. 630, pp. 841–846, 2024. [Online]. Available: https: //www.nature.com/articles/s41586-024-07335-x

[19] T. K. Idris, P. Mitra, and R. Eiselen, “Can embedding similarity predict cross-lingual transfer? a systematic study on african languages,” 2026. [Online]. Available: https://arxiv.org/abs/2601.03168