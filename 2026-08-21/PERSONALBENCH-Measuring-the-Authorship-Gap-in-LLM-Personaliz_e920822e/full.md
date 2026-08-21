# PERSONALBENCH: Measuring the Authorship Gap in LLM Personalization

Yash Ganpat Sawant Independent AI Researcher sawantyash13@gmail.com

## Abstract

Personalized text generation aims to make LLMs write in a specific individual’s style, yet existing benchmarks measure task accuracy or preference alignment rather than whether the model’s output actually resembles the target author’s writing. We introduce PERSONALBENCH, a benchmark that evaluates inferencetime personalization methods through three independent lenses: LUAR (a trained authorship verification model), an LLM-as-judge, and automated stylometrics. Across 50 authors, 1,000 generations, and two model families (Qwen 3, GLM-4), we find that personalization methods do produce author-differentiated output— LUAR discriminates target authors within generated text at AUC=0.918—but this differentiation never crosses the human–LLM boundary. All methods achieve LUAR similarity to real authors in the range 0.484–0.508, below the cross-author human floor of 0.626 (ceiling 0.756). The LLM’s own authorship fingerprint dominates: generated text is more distant from any human author than random humans are from each other. Methods are statistically indistinguishable from each other on LUAR (spread 0.024) despite appearing differentiated on the LLM judge—a discrepancy we trace to circularity between trait extraction and profile extraction. We validate that LUAR reliably measures authorship in our corpus (AUC=0.76 single-post, 0.96 multi-post). We release PERSONALBENCH as a calibrated measuring stick: inference-time personalization modulates the LLM’s style but does not bridge the gap to human authorship.<sup>1</sup>

## 1 Introduction

Large language models are increasingly deployed where personalization matters: writing assistants matching a user’s voice, customer agents maintaining brand tone, and content tools adapting to individual preferences. These applications share a core requirement: the model must write like a specific person. The standard approach is inference-time personalization—conditioning the model via few-shot examples, style profiles, or contrastive prompts at generation time, without modifying model weights.

But does this actually work? Current personalization benchmarks cannot answer this question because they measure the wrong thing. LaMP [Salemi et al., 2024] evaluates task accuracy (headline generation, product rating). PersonalLLM [Zollo et al., 2025] measures preference alignment. PRISM [Kirk et al., 2024] captures stated value preferences. None evaluate whether the generated text sounds like the target author—whether the model’s underlying authorship fingerprint actually shifts toward the target.

We ask a direct question: do inference-time personalization methods change how the model writes, or only what it writes about? To answer this, we need evaluation that goes beyond surfacelevel text similarity. We need metrics grounded in authorship science—the decades-old discipline of identifying authors from their writing style.

We introduce PERSONALBENCH, a benchmark that evaluates personalization through three independent lenses:

1. LUAR [Rivera-Soto et al., 2021]: a neural authorship verification model trained on millions of Reddit posts, providing calibrated similarity scores. Our primary metric.

2. LLM-as-judge: decoupled binary trait matching and same-author assessment, providing interpretable style diagnostics.

3. Automated stylometrics: function word distributions, punctuation patterns, and lexical overlap.

Our central finding is that inference-time personalization produces author-differentiated output that remains trapped in the LLM’s own style space. Within generated text, LUAR discriminates target authors at AUC=0.918—personalization is doing something. But when comparing generated text to real human text, all four methods score between 0.484 and 0.508 on LUAR—below the cross-author human floor of 0.626 (ceiling 0.756). The LLM’s authorship fingerprint is so dominant that personalized output is more distant from the target author than random human authors are from each other. Profile extraction appears to “win” on the LLM judge (TMR=0.542 vs. baseline 0.384), but LUAR shows no corresponding advantage (0.502 vs. 0.484), exposing circularity between the judge’s trait extraction and the method’s profile extraction.

We validate that this is not a measurement failure. LUAR achieves AUC=0.76 on single-post author discrimination in our blog corpus, rising to 0.96 with multiple posts. The authorship gap between human and LLM text is real and measurable—inference-time personalization modulates style within the LLM regime but does not bridge it.

## Our contributions:

1. The PERSONALBENCH benchmark: 50 authors, 1,000 generations, four methods, three independent evaluation metrics including LUAR authorship verification—the first personalization benchmark grounded in authorship science.

2. A multi-metric evaluation framework: combining trained authorship embeddings (LUAR), LLM-as-judge assessment, and classical stylometrics, revealing that cross-metric correlations are uniformly near zero (|r| < 0.07 across primary metrics; up to |r| = 0.17 when including ROUGE-L) and that single-metric evaluation is unreliable.

3. A rigorously validated finding on the human–LLM authorship gap: inference-time personaliza tion produces author-differentiated output within the LLM’s style space (gen↔gen AUC=0.918) but does not bridge the gap to human authorship (gen→real AUC=0.632), supported by calibrated baselines, two model families, and convergent evidence from three metrics.

## 2 Related Work

Personalization benchmarks. LaMP [Salemi et al., 2024] evaluates personalized language model predictions across seven tasks using task-specific accuracy. LongLaMP [Kumar et al., 2024] extends this to long-form generation, introducing content-summary prompts to separate style from content— an approach we adopt. PersonalLLM [Zollo et al., 2025] evaluates preference personalization using synthetic diverse user profiles. PRISM [Kirk et al., 2024] provides real human preference data but targets value alignment. None evaluate authorial style fidelity or use authorship verification model for evaluation.

Authorship verification. Computational authorship analysis spans from function-word frequencies to neural methods [Stamatatos, 2009]. LUAR [Rivera-Soto et al., 2021] learns universal authorship representations via contrastive learning on Reddit data with strong cross-domain transfer. We adopt LUAR as our primary metric for its calibrated authorship similarity scores.

LLM style evaluation and detection. Wang et al. [2025] find that LLMs “still struggle to imitate everyday authors” across 400+ authors—our LUAR analysis corroborates and quantifies this. Panza [Nicolicioiu et al., 2024] and ExPerT [Salemi et al., 2025] evaluate personalized generation but target single systems; we provide a benchmark framework with calibrated baselines. The AIgenerated text detection literature [Mitchell et al., 2023, Kirchenbauer et al., 2023] has established that LLM outputs carry distinctive fingerprints. Our finding of a “generated-text regime” below the human floor on LUAR is consistent: the LLM’s authorship signal resists erasure by personalization.

Table 1: Effect of prompt construction on baseline scores. Naïve extraction leaks author voice, inflating the unpersonalized baseline by 28pp on same-author judgments.
<table><tr><td rowspan="2">Prompt type</td><td colspan="2">Non-personalized</td><td colspan="2">Few-shot vs. baseline</td></tr><tr><td>TMR</td><td>SA%</td><td>∆TMR</td><td>Win%</td></tr><tr><td>Raw first sentence</td><td>0.587</td><td>50%</td><td>-0.080</td><td>23%</td></tr><tr><td>Content summary</td><td>0.384</td><td>22%</td><td>+0.049</td><td>33%</td></tr></table>

LLM-as-judge. Prometheus [Kim et al., 2024] found 60–70% of Likert scores cluster at 3–4; RULERS [Hong et al., 2026] proposed binary evidence-anchored scoring. We build on binary trait matching and reference anchoring [Salemi et al., 2025], adding a decoupled protocol that separates trait scoring from holistic authorship judgment.

## 3 The PERSONALBENCH Benchmark

## 3.1 Data

We use the Blog Authorship Corpus [Schler et al., 2006], containing 681K posts from 19,320 bloggers with demographic metadata. We select 50 authors meeting: (i) at least 200 training posts, (ii) at least 50 test posts, and (iii) average post length ≥ 100 words. Posts are split 80/20 per author into training and test sets, yielding 104K training and 26K test posts. We preprocess by stripping URL placeholders (urlLink) and HTML artifacts. The full corpus of 200 qualifying authors is included in the release for scaling experiments; we report results on 50.

Corpus characteristics. Posts span 1999–2004, covering personal journals, opinion pieces, creative writing, and daily observations. Mean post length is 207 words (σ=189). Authors exhibit substantial style diversity: average sentence length ranges from 8.4 to 24.1 words; type-token ratio from 0.38 to 0.71; function word distributions show distinctive per-author patterns.

## 3.2 Prompt Construction and Contamination Ablation

A critical design decision is how to derive writing prompts from test posts without leaking the author’s voice. Naïve approaches—extracting the first sentence—embed hundreds of characters of the author’s raw text into the prompt, including distinctive vocabulary, punctuation, and tonal markers.

We demonstrate this is not a theoretical concern. When prompts are extracted as raw first sentences, the unpersonalized baseline achieves 50% on same-author judgments. Replacing with LLM-extracted content summaries—neutral descriptions of what the post discusses—drops the baseline to 22%, a 28 percentage point reduction (Table 1). Following Wang et al. [2025], Nicolicioiu et al. [2024], and Kumar et al. [2024], we extract content summaries by prompting: “Summarize this blog post in 1–2 neutral sentences. Describe what it is about, not how it is written. Do not copy the author’s phrasing, tone, or vocabulary.”

## 3.3 Personalization Methods

We evaluate four inference-time methods spanning a progression from no personalization to explicit quantitative style signals. All methods receive the same content-summary prompt; they differ only in how author information is represented.

NON-PERSONALIZED (control). The model receives only the content summary with no author information.

FEW-SHOT (explicit style transfer). Five randomly sampled training posts from the target author are presented alongside an explicit system instruction to match the author’s writing style, sentence structure, tone, vocabulary, and rhetorical patterns.

PROFILE EXTRACTION (explicit abstract style transfer). A two-stage approach: (i) the LLM reads up to 10 training posts and produces a structured style profile covering tone, formality, vocabu lary, sentence structure, and rhetorical devices; (ii) the generation model receives only this abstract profile—no raw samples—and generates. The profile is extracted once per author and reused across prompts.

CONTRASTIVE WITH FEATURES (explicit quantitative style transfer). The model receives the author’s samples alongside contrastive examples from three other authors labeled “avoid these styles,” plus computed stylometric features: average sentence length, vocabulary richness, and top function word frequencies for both target and contrastive authors.

## 3.4 Evaluation Framework

We evaluate generated text through three independent lenses, ordered by the strength of their grounding in authorship science.

## 3.4.1 LUAR Authorship Similarity (Primary)

LUAR (Learning Universal Authorship Representations) [Rivera-Soto et al., 2021] is a transformer trained via contrastive learning on millions of Reddit posts to produce author-discriminative text embeddings. Given two texts, we compute cosine similarity between their LUAR embeddings. Higher similarity indicates the texts are more likely to share an author.

We use LUAR as our primary metric because it integrates authorship signal across syntax, lexical choice, and discourse patterns into a single calibrated score validated on authorship verification benchmarks.

## 3.4.2 LLM-as-Judge: Decoupled Binary Evaluation (Secondary)

Our judge uses three decoupled stages: (1) Trait extraction (cached per author): the judge reads 10 training posts and extracts 5 distinctive style traits as yes/no questions, constrained to writing style only. (2a) Trait scoring: a separate call scores each trait Y/N with evidence, seeing only the generated text. (2b) Same-author judgment: another separate call compares the generated text to the author’s reference post, asking “could these have been written by the same person?” Stages 2a and 2b are decoupled because we empirically found that combining them causes the holistic question to shift trait answers. We use Qwen 3 32B [Qwen Team, 2025] for generation and GLM-4 32B [GLM Team, 2024] for judging to mitigate self-enhancement bias [Zheng et al., 2023].

## Derived metrics.

• Trait Match Rate (TMR) = traits\_present/5: interpretable per-trait diagnostic.

• Same-Author Rate (SA%): percentage receiving a “yes” judgment.

## 3.4.3 Automated Stylometrics (Tertiary)

We compute stylometric similarity between generated text and the author’s training posts:

• Function word cosine (FuncCos): cosine similarity over frequency distributions of 60 common function words—established markers of individual writing style [Argamon et al., 2003].

• Punctuation cosine: similarity over distributions of 10 punctuation marks.

• ROUGE-L [Lin, 2004]: longest common subsequence overlap with the reference post.

Table 2: Method comparison across 50 authors, 1,000 generations. LUAR similarity uses 5-post aggregation (5 generations per author-method pair compared to 5 training posts) and is the primary authorship metric (↑ better). TMR = trait match rate from LLM judge. SA% = same-author rate. FuncCos = function word cosine similarity. Ceiling = real author’s test posts; floor = cross-author random pairs. All methods score below the real-text cross-author floor on LUAR despite appearing differentiated on TMR. CIs use hierarchical bootstrap (resampling authors, then generations within authors) to account for within-author correlation (B=10,000).
<table><tr><td>Method</td><td>LUAR↑</td><td>TMR↑</td><td>SA%↑</td><td>FuncCos ↑</td></tr><tr><td>NON-PERSONALIZED</td><td>0.484±.019</td><td> $0 . 3 8 4 { \scriptstyle \pm . 0 5 8 }$ </td><td>22%±7</td><td>0.741±.011</td></tr><tr><td>FEW-SHOT</td><td>0.508±.020</td><td>0.433±.061</td><td>31%±8</td><td>0.749±.011</td></tr><tr><td>PROFILE EXTRACTION</td><td>0.502±.019</td><td>0.542±.060</td><td>29%±8</td><td>0.761±.010</td></tr><tr><td>CONTRASTIVE</td><td>0.494±.020</td><td>0.447±.059</td><td>36%±8</td><td>0.752±.011</td></tr><tr><td>Real Author (ceiling)</td><td>0.756</td><td>0.427</td><td>30%</td><td></td></tr><tr><td>Cross-Author (floor)</td><td>0.626</td><td>0.390</td><td>7%</td><td></td></tr></table>

## 4 Experiments

## 4.1 Setup

We evaluate 50 authors × 5 prompts × 4 methods = 1,000 generations. The generator is Qwen 3 32B 4-bit, served locally via mlx-lm. The LLM judge is GLM-4 32B 4-bit (different model family). LUAR embeddings are computed using the pretrained model from Rivera-Soto et al. [2021]. Total compute: ∼3,300 LLM calls over ∼24 hours on a single Apple M4 Pro (48GB).

## 4.2 LUAR Validation: The Task Is Measurable

We verify that LUAR can discriminate authorship in our blog corpus despite being trained on Reddit. On 2,500 same-author and 2,500 cross-author pairs, single-post LUAR achieves AUC=0.76 (vs. TF-IDF baseline AUC=0.54); with 5-post aggregation, AUC rises to 0.96. Same-author similarity averages 0.756 (ceiling); cross-author averages 0.626 (floor). We report 5-post aggregated scores throughout.

## 4.3 Main Result: The Human–LLM Authorship Gap

Table 2 presents our central finding. On LUAR, all methods score 0.484–0.508 (spread 0.024), below the cross-author floor of 0.626. Personalization is not doing nothing—gen→target (0.497) exceeds gen→wrong (0.459), AUC=0.632—but output remains in the LLM’s style space. The LLM judge tells a different story: PROFILE EXTRACTION achieves TMR=0.542 vs. baseline 0.384. We analyze this discrepancy in §4.6.1. Figure 1 visualizes the LUAR results.

## 4.4 Calibration Analysis

Table 2 includes calibrated baselines. The floor (cross-author real text) yields LUAR=0.626, TMR=0.390, SA=7%. The ceiling (same-author real text) yields LUAR=0.756, TMR=0.427, SA=30%. Notably, the real author scores lower on TMR (0.427) than PROFILE EXTRACTION (0.542)—evidence of circularity (§4.6.1). All methods score below the real-text floor on LUAR: the best (FEW-SHOT, 0.508) is below 0.626. Generated text is more distant from any human author than random humans are from each other. Comparing generated text to wrong vs. target authors yields a faint signal (gen→target 0.497 vs. gen→wrong 0.459, AUC=0.632).

The generated-text regime. Gen↔gen LUAR similarity averages 0.932 (same target) and 0.858 (different targets)—far above gen↔real (0.522). Within this cluster, LUAR discriminates target authors at AUC=0.918: personalization modulates output within the LLM’s style space without crossing into human territory. Figure 2 confirms all methods produce overlapping distributions; Figure 3 shows calibration across all three metrics.

![](images/201ddd6b7da2f788563b9da0ef99f8f967e849b7537f6797f8a68bc1032ad583.jpg)  
Figure 1: LUAR authorship similarity by method (5-post aggregation). All methods score below the real-text cross-author floor (0.626), far from the real-author ceiling (0.756). The total spread across methods is only 0.024. The LLM’s authorship fingerprint is so dominant that generated text sits in a distinct regime below human-to-human similarity.

Table 3: Pearson correlation between evaluation metrics (n=1,000). All pairwise correlations are near zero, indicating the metrics capture fundamentally different constructs. No single metric is a reliable proxy for another.
<table><tr><td></td><td>LUAR</td><td>TMR</td><td>FuncCos</td></tr><tr><td>LUAR</td><td>1.00</td><td></td><td></td></tr><tr><td>TMR</td><td>0.013</td><td>1.00</td><td></td></tr><tr><td>FuncCos</td><td>0.026</td><td>0.067</td><td>1.00</td></tr></table>

## 4.5 Cross-Metric Agreement

Table 3 reveals that all pairwise correlations among primary metrics are near zero $\left( \left| \boldsymbol { r } \right| < 0 . 0 7 ; \left| \boldsymbol { r } \right| \le \right.$ 0.17 including ROUGE-L). A benchmark using only FuncCos would declare PROFILE EXTRACTION the winner (d=0.23); TMR would amplify this (d=0.58); LUAR would find no differentiation. Single-metric evaluation is unreliable. Figure 4 confirms: high TMR does not correspond to high LUAR (r=0.013), indicating the judge’s signal is orthogonal to the trained authorship model.

## 4.6 Ablations and Diagnostic Analyses

## 4.6.1 TMR Circularity: Why Profile Extraction “Wins” on the Judge

PROFILE EXTRACTION achieves TMR=0.542—far above other methods—but LUAR=0.502, indistinguishable from the pack. This discrepancy has a straightforward explanation: the judge’s trait extraction and the method’s profile extraction perform the same operation. Both ask an LLM to read author samples and extract salient style features. The profile is then used to generate text optimized for exactly the kind of features the judge will check.

![](images/6fa0b9a79c9f216090ff46b6662da3bfe0799577af321413d118fdbe1398bb7e.jpg)  
Figure 2: Distribution of per-generation LUAR similarity scores by method. All four methods produce overlapping distributions centered near 0.44 (1v5 per-generation scores), well below the real-author distribution centered at 0.58.

![](images/250a15cad8f5015e6c1dc0c2b08442948849391f8cdefa20db1728e1e95d531c.jpg)

![](images/5ed48643dd71661ceeae97292b50c800726575e329d69d21ae9d62efe6a2c502.jpg)

![](images/a7b4c02672952a80e966225b160eaa1355e6f74d148d8d324e6f50f6e9f56548.jpg)  
Figure 3: Calibration across three metrics. Each panel shows method scores (bars) with ceiling (real author, green) and floor (cross-author, red) baselines. On LUAR (left), all methods score below the real-text cross-author floor. On TMR (center), profile extraction exceeds the real-author ceiling—evidence of circularity (§4.6.1). On SA% (right), contrastive leads but all methods exceed the chance floor.

Supporting evidence: the real author’s own text scores TMR=0.427—lower than PROFILE EXTRAC-TION’s 0.542. If TMR were measuring genuine authorship fidelity, the real author should set the ceiling. Instead, the method that explicitly optimizes for LLM-extractable traits exceeds the real author, indicating TMR measures instruction-following rather than style transfer.

## 4.6.2 Trait Extraction Stability

We extracted traits 3 times per author for a subset of 10 authors and computed pairwise Jaccard similarity between trait sets. Mean Jaccard similarity is 0.22—the judge produces substantially different traits on each run. This instability further undermines TMR as a reliable authorship metric: the yardstick itself changes between measurements.

![](images/224de3955988d9cd0239a4a02baba5d394b3c19d21ac76530f26a8f796643ec2.jpg)  
Figure 4: LUAR similarity vs. TMR for all 1,000 generations, colored by method. No systematic relationship (r=0.013). Profile extraction’s high TMR does not correspond to high LUAR.

## 4.6.3 Additional Ablations

Classical features. TF-IDF cosine similarity achieves AUC=0.54 on generated text (vs. LUAR’s 0.76 on real text)—classical features cannot find authorship signal when the LLM’s fingerprint dominates.

LUAR aggregation sensitivity. At every aggregation level (1–5 posts), all methods remain below the real-text floor (e.g., 1-post: 0.345–0.368 vs. floor 0.398; 5-post: 0.485–0.508 vs. 0.626). The finding is robust to aggregation choice.

## 4.6.4 Second Generator

We replicated with GLM-4 32B (10 authors, 150 generations). All methods score below the human floor (0.417–0.576 vs. 0.626, AUC=0.671). Method spread is larger (0.16 vs. Qwen’s 0.024), suggesting gap strength varies by model but its existence does not. Cross-model analysis reveals model-specific fingerprints: within-model similarity is high (Qwen↔Qwen 0.918, GLM↔GLM 0.839), cross-model lower (0.753), and gen→real lowest (0.45–0.49).

## 5 Discussion

Surface vs. deep style. Prompting shifts surface features (PROFILE EXTRACTION achieves function word cosine 0.761 vs. baseline 0.741) but not the authorship fingerprint. Gen↔gen analysis shows personalization modulates output within the LLM’s style space (AUC 0.814–0.920) without crossing into human territory (gen↔real AUC=0.664), consistent with Wang et al. [2025]. This parallels AI-generated text detection: DetectGPT [Mitchell et al., 2023] and watermarking [Kirchenbauer et al., 2023] exploit distinctive statistical signatures in LLM outputs. The same fingerprint that enables detection resists erasure by inference-time conditioning—the model’s authorship signal is architecturally embedded, not contextually malleable.

Why metrics disagree. Cross-metric correlations near zero (|r| < 0.07) reveal that “personalization” is not a single construct. TMR measures instruction-following (explaining PROFILE EXTRACTION’s advantage—it optimizes for what TMR measures). FuncCos captures surface lexical alignment. LUAR integrates deep authorship signal across syntax, discourse, and lexical choice. No method moves all three, meaning the methods produce metric-specific artifacts rather than genuine style transfer. Single-metric evaluation of personalization is unreliable.

Instruction strength is not the bottleneck. FEW-SHOT provides real author examples and explicit instructions to match the author’s style—the strongest inference-time signal short of weight modification. Yet it achieves LUAR=0.508, indistinguishable from PROFILE EXTRACTION (0.502) and CONTRASTIVE (0.494). The bottleneck is not instruction quality but the model’s capacity to shift its deep generative distribution at inference time.

Implications. Closing the authorship gap likely requires training-time adaptation: fine-tuning via LoRA, reinforcement learning with style rewards, or continued pretraining on author corpora. PER-SONALBENCH provides the measuring stick—floor (0.626) and ceiling (0.756) define the target range. Beyond personalization, our findings suggest LLM outputs carry an indelible authorship fingerprint that prompt engineering cannot mask, with implications for AI text detection and the fundamental limits of prompt-based style control. The gen↔gen AUC of 0.918 confirms personalization does leave a trace within the model’s output space, but all outputs remain in the LLM’s stylistic regime rather than crossing into human territory.

## 6 Limitations

Two generators, limited diversity. We validate the main finding on two model families (Qwen 3 32B and GLM-4 32B), but both are 32B-scale 4-bit quantized models. More diverse generators— different parameter scales, architectures, and pretraining data—remain to be tested.

Single domain. PERSONALBENCH evaluates blog-style writing from the early 2000s. Generalization to modern social media, email, or academic writing is untested.

LUAR domain gap and distributional confound. LUAR was trained on Reddit, not blogs, and never saw LLM-generated text. Real author text is organic blog writing while generated text responds to content-summary instructions—a format difference that may inflate the gen→real gap. However, the high gen↔gen AUC (0.918)—where all texts share the same format—confirms LUAR detects genuine author signal, and gen→real AUC (0.632) is above chance.

No human validation and other scope. Our LLM judge has not been validated against human authorship judgments; trait stability is low (Jaccard=0.22) and human agreement studies are needed. All data is English; authorship patterns may differ across languages. Both generator and judge use 4-bit quantization, which may affect subtle stylistic capabilities.

## 7 Conclusion

We introduced PERSONALBENCH, the first personalization benchmark grounded in authorship verification. Across 50 authors, 1,000 generations, and four methods, LUAR similarity clusters at 0.484–0.508—below the real-text cross-author floor of 0.626. The LLM’s fingerprint dominates, yet personalization does leave a trace within the generated-text regime (gen↔gen AUC=0.918). We release PERSONALBENCH as a calibrated measuring stick—floor (0.626) and ceiling (0.756) define the authorship gap—designed for extensibility so researchers can evaluate new methods against these baselines. Closing the gap likely requires training-time approaches; PERSONALBENCH provides the target.

## References

Argamon, S., Koppel, M., Fine, J., and Shimoni, A. R. Gender, genre, and writing style in formal written texts. Text & Talk, 23(3):321–346, 2003.

GLM Team. ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793, 2024.

Kirchenbauer, J., Geiping, J., Wen, Y., Katz, J., Miers, I., and Goldstein, T. A watermark for large language models. ICML, 2023.

Kim, S., Shin, J., Cho, Y., et al. Prometheus: Inducing fine-grained evaluation capability in language models. ICLR, 2024.

Kirk, H. R., Vidgen, B., Röttger, P., and Hale, S. A. The PRISM alignment project: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. arXiv preprint arXiv:2404.16019, 2024.

Kumar, I., Viswanathan, S., Yerra, S., Salemi, A., et al. LongLaMP: A benchmark for personalized long-form text generation. arXiv preprint arXiv:2407.11016, 2024.

Zollo, T. P., Siah, A. W. T., Ye, N., Li, A., and Namkoong, H. PersonalLLM: Tailoring LLMs to individual preferences. ICLR, 2025.

Lin, C.-Y. ROUGE: A package for automatic evaluation of summaries. Text Summarization Branches Out, 2004.

Mitchell, E., Lee, Y., Khazatsky, A., Manning, C. D., and Finn, C. DetectGPT: Zero-shot machinegenerated text detection using probability curvature. ICML, 2023.

Nicolicioiu, A., Iofinova, E., Jovanovic, A., Kurtic, E., Nikdan, M., Panferov, A., Markov, I., Shavit, N., and Alistarh, D. Panza: Design and analysis of a fully-local personalized text writing assistant. arXiv preprint arXiv:2407.10994, 2024.

Qwen Team. Qwen3 technical report. arXiv preprint, 2025.

Rivera-Soto, R. A., Miano, O. E., Ordonez, J., Chen, B. Y., Khan, A., Bishop, M., and Andrews, N. Learning universal authorship representations. EMNLP, 2021.

Hong, Y., Yao, H., Shen, B., Xu, W., Wei, H., and Dong, Y. RULERS: Locked rubrics and evidenceanchored scoring for robust LLM evaluation. arXiv preprint arXiv:2601.08654, 2026.

Salemi, A., Mysore, S., Bendersky, M., and Zamani, H. LaMP: When large language models meet personalization. ACL, 2024.

Schler, J., Koppel, M., Argamon, S., and Pennebaker, J. W. Effects of age and gender on blogging. AAAI Spring Symposium on Computational Approaches to Analyzing Weblogs, 2006.

Stamatatos, E. A survey of modern authorship attribution methods. Journal of the American Society for Information Science and Technology, 60(3):538–556, 2009.

Wang, Z., Tripto, N. I., Park, S., Li, Z., and Zhou, J. Catch me if you can? Not yet: LLMs still struggle to imitate the implicit writing styles of everyday authors. EMNLP Findings, 2025.

Salemi, A., Killingback, J., and Zamani, H. ExPerT: Effective and explainable evaluation of personalized long-form text generation. ACL Findings, 2025.

Zheng, L., Chiang, W.-L., Sheng, Y., et al. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. NeurIPS, 2023.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect   
the paper’s contributions and scope?   
Answer: [Yes]

Justification: The abstract and introduction state three specific contributions (benchmark, multi-metric framework, negative finding), all supported by experiments in §4.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors? Answer: [Yes]

Justification: §6 discusses five limitations including two generators with limited diversity, single domain, LUAR domain gap, no human validation, and language/quantization scope.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (correct) proof?

Answer: [N/A]

Justification: This is an empirical benchmark paper with no theoretical results.

## 4. Experiments reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper?

Answer: [Yes]

Justification: §4.1 specifies all models, quantization levels, hardware, and generation counts. All prompts are described in §3. Code and data will be released.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results?

Answer: [Yes]

Justification: We release all code (evaluation pipeline, generation scripts, analysis tools, and figure reproduction), generated outputs, and processed data via a public repository (https://github.com/yashsawant22/personalbench). The Blog Authorship Corpus is publicly available. LUAR is publicly available. The processed 200-author corpus will be hosted on a persistent platform for long-term access.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details necessary to understand the results?

Answer: [Yes]

Justification: §3 describes data selection criteria, train/test splits, prompt construction, and method implementations. §4.1 specifies all model and hardware details.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: Table 2 reports hierarchical bootstrap 95% CIs (B=10,000) for all metrics. We report calibration baselines (floor and ceiling), LUAR AUC validation, and crossmetric correlations with sample sizes.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources needed to reproduce the experiments?

Answer: [Yes]

Justification: §4.1 reports hardware (Apple M4 Pro, 48GB), total LLM calls (∼3,300), and wall-clock time (∼24 hours).

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics?

Answer: [Yes]

Justification: The Blog Authorship Corpus is a publicly available research dataset. We do not generate harmful content. Our work evaluates rather than enables impersonation.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work?

Answer: [Yes]

Justification: Our benchmark could theoretically aid impersonation research, but our central negative finding—that inference-time methods fail at authorship transfer— demonstrates that current techniques cannot produce text indistinguishable from a target author. This finding itself reduces perceived impersonation risk. Positive impacts include better evaluation methodology for personalization research and a calibrated framework for measuring future progress.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible use of data and methods?

Answer: [Yes]

Justification: We use a publicly available, anonymized blog corpus. Generated outputs are evaluated, not deployed. The benchmark measures style fidelity rather than providing tools for impersonation.

## 12. Licenses for existing assets

Question: Are the creators of assets used in the paper properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: The Blog Authorship Corpus [Schler et al., 2006] and LUAR [Rivera-Soto et al., 2021] are properly cited. Both are publicly available for research use.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the responsible use of them discussed?

Answer: [Yes]

Justification: We release the benchmark code, evaluation framework, and generated outputs with documentation and a research-use license.

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots? Answer: [N/A]

Justification: No crowdsourcing or human subject research was conducted.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether compensation was adequate given the risks, whether informed consent was obtained, etc.?

Answer: [N/A]

Justification: No human subjects research was conducted.

## 16. Usage of LLMs

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research?

Answer: [Yes]

Justification: LLMs are central to this work: Qwen 3 32B is used for text generation across all four personalization methods (§3.3), and GLM-4 32B serves as the LLM judge (§3.4.2). Both models, their quantization levels, and serving configurations are fully specified in §4.1. All prompts are reproduced verbatim in Appendix A.

## A Prompt Templates

All prompts used in PERSONALBENCH are reproduced below verbatim. Variables in curly braces (e.g., {prompt}) are filled at runtime.

## A.1 Content Summary Extraction

Used to extract a neutral content summary from each test post, following the approach of Wang et al. [2025] and Kumar et al. [2024]. The summary specifies what the post is about without leaking how the author writes.

System: You are a helpful assistant that summarizes blog post   
topics in plain, neutral language.   
User: Blog post excerpt:   
{snippet}   
Task: In 1-2 plain sentences, what is this blog post about? State   
the topic only. Do not analyze the writing style. Do not copy the   
author’s words.

The summary is then wrapped into a generation prompt: Write a short blog post (2-3 paragraphs) about the following: {summary}.

## A.2 Non-Personalized (Control)

The model receives only the content-summary prompt with no author context:

Write a short blog post (2-3 paragraphs) about the following:   
{summary}

## A.3 Few-Shot

Five randomly sampled training posts are presented with an explicit system instruction to match the author’s style.

System: You are writing in the style of the author shown in the   
examples below. Match their sentence structure, tone, vocabulary,   
rhetorical patterns, punctuation habits, and overall voice as   
closely as possible. A reader who knows this author should find   
it hard to tell your text was not written by them.   
User: Examples from the author:   
[Example 1]   
{sample\_1}   
[Example 2]   
{sample\_2}   
[Example 5]   
{sample\_5}   
Now write a new post in this author’s style on the following topic:   
{prompt}

## A.4 Profile Extraction (Two-Stage)

Stage 1: Profile extraction (one call per author, cached).

You are a forensic linguist analyzing writing samples from one   
author. Produce a detailed style profile that would let someone   
else replicate their writing without seeing the originals.   
Writing samples:

{samples\_text}   
Profile must cover:   
1. TONE: Overall attitude (e.g., confessional, detached,   
enthusiastic, sardonic)   
2. FORMALITY: Register and how it shifts within a piece   
3. VOCABULARY: Complexity, jargon, slang, distinctive word choices   
4. SENTENCE STRUCTURE: Length patterns, fragments, run-ons,   
punctuation habits   
5. RHETORICAL MOVES: How they open posts, transition, close. Use   
of questions, lists, asides   
6. PERSPECTIVE: Point of view, self-reference patterns, how they   
address readers   
7. QUIRKS: Anything distinctive –- ellipses, parentheticals,   
specific phrases, capitalizations   
8. EMOTIONAL TEXTURE: How they handle vulnerability, humor,   
authority   
Be concrete. Quote specific phrases as examples. This profile will   
be used WITHOUT access to the original samples.

## Stage 2: Generation from profile (one call per prompt). The model receives only the abstract profile, no raw samples.

System: You are writing as the person described in the profile   
below. Follow EVERY detail in the profile –- tone, sentence   
structure, quirks, vocabulary level, emotional register. The reader   
should not be able to tell this wasn’t written by the original   
author.   
AUTHOR PROFILE:   
{cached\_profile}   
User: {prompt}

## A.5 Contrastive with Features

The model receives the target author’s samples, contrastive examples from other authors, and computed stylometric features.

System: You must write in the TARGET author’s style, NOT the other   
authors’ styles. Pay close attention to the measurable differences   
between them.   
QUANTITATIVE STYLE COMPARISON:   
Target author –- avg sentence length: {sent\_len} words, vocabulary   
richness: {vocab}, top function words: {top\_func\_str}   
Other authors –- avg sentence length: {contrast\_sent\_len} words,   
vocabulary richness: {contrast\_vocab}   
TARGET AUTHOR writes like this:   
»> {author\_sample\_1}   
»> {author\_sample\_2}   
»> {author\_sample\_3}   
OTHER AUTHORS write like this (AVOID these styles):   
»> {contrast\_sample\_1}   
»> {contrast\_sample\_2}   
»> {contrast\_sample\_3}   
Key differences to maintain: match the target’s sentence length,   
vocabulary level, and function word patterns. If the target uses   
shorter sentences than others, keep yours short. If they use more   
personal pronouns, do the same.   
User: {prompt}

## A.6 Judge Stage 1: Trait Extraction

One call per author, cached. Extracts 5 distinctive style traits as yes/no checkable questions.

Do NOT include traits about:   
- URL formatting, link patterns, or platform markup   
- Generic qualities like "uses complete sentences" or "writes in   
English"   
- Specific topics the author writes about   
- Content preferences –- only writing STYLE   
Return exactly 5 traits. Respond in this exact JSON format:   
{"traits": [{"label": "short trait name", "question": "Does the   
text ...? (yes/no checkable)"}]}

## A.7 Judge Stage 2a: Trait Scoring

One call per generation. Receives only the generated text and the 5 trait questions—no reference post, no holistic question.

Check whether generated text exhibits specific style traits.   
## Generated text   
{generated\_text}   
## Style traits to check   
{traits\_formatted}   
## Instructions   
For EACH trait, answer Y or N. Quote brief evidence from the   
generated text, or explain why the trait is absent.   
Respond in this exact JSON format:   
{"traits": [{"q": 1, "present": true, "evidence": "brief quote   
or explanation"}, {"q": 2, "present": false, "evidence": "brief   
explanation"}]}

## A.8 Judge Stage 2b: Same-Author Judgment

One call per generation. Receives only the reference post and generated text—no trait questions.

You are evaluating whether two texts were written by the same   
person.   
## Text A (reference –- known author)   
{reference\_post}   
## Text B (to evaluate)   
{generated\_text}   
## Instructions   
Based on writing style alone (not topic), could Text B plausibly

have been written by the same person who wrote Text A?   
Consider: sentence structure, vocabulary level, tone, rhetorical   
devices, emotional register, punctuation habits, how they open and   
close, self-reference patterns. Ignore topic differences –- focus   
only on whether the VOICE is the same.   
Answer YES or NO, then give a one-sentence explanation.   
Respond in this exact JSON format:   
{"same\_author": true, "reason": "one sentence explanation"}

## B Example Generations

We present two example generations illustrating the authorship gap. In both cases, methods produce fluent, on-topic text that adopts some surface features of the author’s style, but the underlying voice remains distinctively LLM-generated.

Example 1. Author writes informal, link-heavy retrospective posts with direct reader address.

Content summary: A retrospective look at entries from one year ago, highlighting various topics the author wrote about at that time.

## Reference (real author):

Journey Inside My Mind - One Year Ago. One of the great things about blogging is that, over time, one develops a history to which he or she can return. With that in mind, let’s take a Journey Inside My Mind from one year ago: [links to posts on downloads, compliments, cartoons, Cincinnati Reds, Frank Oz, self-injury, and old friends]

## Profile Extraction:

Cheers!!! I figured I’d take some time today to look back at what I wrote exactly one year ago—I mean, time flies, right? Just flipping through the archives reminded me how much growth and change happen in such a short amount of time. [...]

## Non-Personalized:

Looking Back: A Year of Reflection. A year ago today marked the start of a journey filled with exploration, learning, and personal growth—all captured through thoughtful blog posts. [...]

Profile Extraction captures informal tone (“Cheers!!!”, “I mean”) but produces generic reflective content. The real author’s distinctive list-of-links format and direct “let’s take a Journey” address are absent from all methods. Non-Personalized defaults to a polished editorial voice that no human blogger would use.

Example 2. Author writes devotional posts mixing poetry quotation with liturgical observations.

Content summary: A reflection on a poem about reverence and redemption, paired with personal and date-related notes.

## Reference (real author):

Reverence, by Venerable John Henry Newman, C.O. “I bow at Jesu’s name, for ’tis the Sign / Of awful mercy towards a guilty line. / Of shameful ancestry, in birth defiled [...]” [followed by personal notes and date]

## Profile Extraction:

Today is the First Sunday of Advent—the beginning of a new liturgical year. It strikes me how often this season carries both anticipation and quiet mourning [...] I came across a small, anonymous poem tucked into an old prayer book [...]

## Few-Shot:

Reflections on Reverence and Redemption. In the quiet rhythm of daily life, moments often unfold that whisper truths about the divine—truths captured beautifully in poems that blend reverence with redemption. [...]

Profile Extraction produces liturgically-themed text (“First Sunday of Advent”) that matches the author’s domain but fabricates a poem rather than quoting one. Few-Shot produces generic devotional prose. The real author’s distinctive practice of quoting a specific poem with full attribution is absent from all methods.

## C Additional Materials

Per-author LUAR breakdowns and full generation outputs are available in the code repository.