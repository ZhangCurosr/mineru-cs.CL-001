# From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

Preprint, compiled August 12, 2026

Si’an Xie<sup>1</sup>,<sup>∗</sup>, Jiaxun Liu<sup>2</sup>,<sup>∗</sup>, Biao Yang<sup>3</sup>, Wei Yuan<sup>3</sup>,<sup>†</sup>, Fan Yang<sup>3</sup>, Tingting Gao<sup>3</sup>, Ming Wu<sup>1</sup>,<sup>†</sup>

<sup>1</sup>Beijing University of Posts and Telecommunications <sup>2</sup>Peking University <sup>3</sup>Kuaishou Technology <sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## Abstract

Large language models (LLMs) have made substantial progress on reasoning tasks that require increasingly long and complex inferential chains. This progress primarily reflects reasoning depth. A complementary and comparatively unexamined capability is reasoning breadth: exploring multiple semantic directions in parallel and integrating the resulting clues into one coherent answer. We introduce MPAR-Bench, a bilingual (English–Chinese) benchmark that isolates reasoning breadth through multi-point associative reasoning. Inspired by the cooperative game Just One, each item asks a model to recover a hidden target from several independently generated, semantically diverse clues. We construct 1,000 items using a multi-agent clue-generation pipeline, embedding-based diversity filtering, and human verification. Crucially, only the answer space is drawn from public word lists, whereas every clue set is generated from scratch. Beyond exact-match accuracy, we evaluate models along a coarse-to-fine protocol (accuracy, ANLS, embedding similarity, and reasoning-trace verification) and a perturbation suite that probes four distinct robustness axes: clue masking, order shufling, distractor injection, and multi-step clues. Across evaluated models, perturbations reduce accuracy by 9–18 points in English and 5–12 points in Chinese. Enabling thinking mode improves standard-setting accuracy—especially in English—but does not consistently reduce sensitivity to perturbations, and case-level trace analysis shows that extended reasoning can overturn an initially correct hypothesis. These results indicate that greater reasoning depth does not automatically confer robust reasoning breadth, and that breadth is a measurement axis that current benchmarks largely leave uncovered.

## Introduction

Large language models (LLMs) have rapidly evolved from early neural language models to highly capable Transformerbased systems [1] such as GPT, Gemini, Qwen, and others [2, 3, 4, 5, 6, 7, 8, 9]. Through techniques such as reinforcement learning [10], supervised finetuning [11], Chain-of-Thought (CoT) [12] and Retrieval-Augmented Generation [13], modern LLMs have achieved remarkable success and demonstrated near-human performance in solving practical problems, which depends on their step-by-step linear reasoning. Humans possess another important capability: multi-point associative reasoning, which enables the structured integration of a wide range of concepts. This ability allows humans not only to explore problems in depth, but also to bridge existing concepts and synthesize them into novel ideas. These ability diferences are shown in Fig. 1. An important question remains unresolved: do current LLMs possess the non-linear, cross-domain associative reasoning abilities?

Existing LLM benchmarks [14, 15, 16, 17, 18] primarily emphasize reasoning “depth,” evaluating step-by-step logical deduction and procedural reasoning. In contrast, the evaluation of reasoning “breadth”—the ability to aggregate dispersed semantic signals and perform abstract conceptual convergence—remains largely unexplored. This capability matters whenever the relevant evidence is distributed across diferent semantic perspectives rather than arranged as a single derivation. Multi-document synthesis, cross-domain analogy, hypothesis generation, and reasoning under incomplete or distracting evidence all require a model to hold several partial relations in view and reconcile them into a final prediction [19]. A model may therefore reason deeply along one path while still failing to combine information available across several paths.

![](images/3be6cf869debc050f8dfb84e8acf0360f3575c04f1bc6ab2ea52a72ecdf4d94a.jpg)  
Figure 1: Linear Reasoning and Multi-Point Associative Reasoning Flow Chart.

To fill this gap, we introduce MPAR-Bench, a cognitively inspired benchmark designed to systematically evaluate multipoint associative reasoning in LLMs. To improve evaluation reliability and robustness, we further construct high-quality bilingual test sets with 1,000 questions through a carefully designed multi-agent generation and verification pipeline [20, 21, 22].

![](images/ea349ffb9f52f3a448ae46853cf49c7a0a0c518c56bcb412b555bb99393e4448.jpg)  
Figure 2: A brief introduction of MPAR-Bench.

Beyond exact-match metrics, we propose a fine-grained evaluation framework that analyzes model behavior from reasoning perspectives. This framework enables a more comprehensive investigation of how LLMs “reason broad”, ofering deeper insight into the current capabilities and limitations of associative reasoning in modern language models. A general overview of MPAR-Bench is shown in Fig. 2.

In general, our contributions are summarized as follows:

• A benchmark targeting reasoning breadth. MPAR-Bench operationalizes many-to-one integration from multiple, semantically diverse clues. Unlike RAT-style tests (three fixed compound-word cues) and existing game benchmarks (clue giving, or grouping a fixed word set), MPAR-Bench isolates the guesser-side integration of an open number of free-form clues, and pairs it with a controlled perturbation suite.

• An innovative multi-agent clue-synthesis pipeline. We propose a multi-agent collaborative clue-synthesis framework with embedding-based filtering and human verification, enabling the construction of semantically diverse, high-dificulty evaluation instances while substantially reducing memorization risk.

• A coarse-to-fine evaluation protocol. Beyond exact match, we combine ANLS, embedding similarity, and reasoning-trace verification, and analyze robustness per perturbation type rather than as a single aggregate, exposing a blank space in reasoning breadth.

## Related Work

## LLM in Reasoning Depth

The dominant trajectory of LLM reasoning research extends inferential depth. Chain-of-thought prompting [12], tree- and planstructured search [23, 24], self-verification [25], and process supervision [26] all lengthen or stabilize a single reasoning trajectory, and reinforcement-learned thinking modes push test-time computation further [27]. A parallel line of work documents the failure mode of overthinking, in which additional reasoning steps degrade rather than improve answers [28, 29]. Depth-oriented benchmarks—mathematical [14, 16], knowledge-intensive [15], and broad-coverage suites [17, 18]—are increasingly saturated for frontier models. These tasks answer how far a model can push one chain; they do not answer whether a model can integrate evidence across chains. Reasoning breadth is thus an orthogonal axis, and one on which depth-oriented benchmarks provide little discrimination.

## Associative Reasoning

Associative reasoning has long been studied in cognitive psychology through convergent-thinking instruments, most notably the Remote Associates Test (RAT) [30]. RAT has recently been repurposed to probe LLMs: Schon et al. [31] model associative reasoning processes, Kumar et al. [32] study human–AI convergent and divergent thinking, and generative models have been reported to match or exceed humans on such tests [33, 34]— though such results are hard to interpret, since the test items are publicly available and may have been seen during pre-training [35, 36].

Related open-ended formulations argue for process-based rather than multiple-choice evaluation [37] and for generating explicit associative paths [38]. We inherit the construct of convergent association from this tradition but deliberately depart from its instrument: RAT items are public, largely three-cue, and dominated by fixed phrasal collocations that next token prediction learns readily. MPAR-Bench instead poses an open, variable number of free-form semantic clues that must be integrated, with every clue set synthesized de novo to substantially reduce overlap with public clue–target pairings and lower the risk of memorization. This is what MPAR-Bench adds over prior RAT-on-LLM evaluations: a breadth-oriented benchmark with reduced contamination risk, rather than a re-run of a public convergent-thinking test.

![](images/d7184906214d438e52413528f2a47b5e85f3e20a73c2f25a429bee86a5a2614a.jpg)  
Figure 3: Introduction of Board Game Just One.

## Boardgame-Based Benchmarks

Cooperative and word-association games ofer constrained rules with large state spaces, which helps mitigate contamination and yields human-aligned semantic tasks. Codenames has been used to evaluate one-to-many clue giving and ad-hoc concept forming [39, 40]; the NYT Connections game requires partitioning a fixed set of words into latent groups [41]; and the Word Synchronization Challenge measures two agents converging on a shared word without communication [42]. MPAR-Bench difers along three axes: (i) clue cardinality—models must jointly integrate an open, variable number of clues rather than fixed cues; (ii) association type—clues are free-form semantic descriptions spanning lexical, cultural, phonetic, and world-knowledge relations, not compound-word completions or fixed candidate pools; and (iii) item availability—all clue sets are synthesized from scratch, leaving no public clue–target pairing to memorize.

## Methodology

## Task Definition

Given a clue set $C = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { n } \}$ , the task is to recover a target y such that each clue $c _ { i }$ contributes an independently informative semantic relation to y. Reasoning breadth, in this setting, is the ability to integrate multiple semantically distinct and non-redundant clues into a single coherent answer.

![](images/39ec6ed5d5c37fc4bf7d92578d4ed73ced1defbfd6ce1e29565d7119b5cad879.jpg)  
Figure 4: Word Cloud of MPAR-Bench.

We ensure that each item genuinely requires breadth through two construction-side safeguards. First, clues are generated from diverse semantic angles to maximize the range of associations a model must reconcile. Second, a judge agent and an embeddingbased filter remove synonyms, paraphrases, and near-duplicates, so that each retained clue carries non-overlapping information. At evaluation time, we measure not only whether a model’s prediction matches the target, but also whether that prediction remains stable under perturbation—clue masking, order shufling, distractor injection, and multi-step inference—which probes whether the integration is robust or merely superficial.

## Just One

The design borrows the constraint structure of the cooperative game Just One, in which players give single-word hints to help a guesser infer a hidden target, while direct synonyms, translations, homophones, and duplicate clues are forbidden. These constraints are what make the game a clean instrument for breadth. Rather than rewarding the most obvious lexical association, they force clue writers to approach the target from distinct, indirect angles; the guesser must then integrate fragmented, non-overlapping signals rather than pattern-match a single cue. This yields a constrained multi-point associative reasoning task emphasizing semantic abstraction, conceptual bridging, and integration. An illustrative round is shown in Fig. 3.

## Dataset Construction

Answer space. Target words are drawn from public word lists— RAT-derived vocabulary (collected on the internet) and Just One word cards.

Multi-agent clue generation. Given a target and the clues already accepted, LLM-based agents iteratively propose new clues that remain semantically relevant to the target while minimizing redundancy with existing clues. Each agent is assigned a distinct association angle to encourage coverage across semantic directions. A judge agent then removes clues that are the answer itself, direct synonyms/translations/homophones/morphological variants, exact or near-duplicates of accepted clues, or genuinely low quality [43, 44]. This division of labor mirrors the independent clue-provider and arbiter structure of Just One. Notably, all agents’ prompt are provided in Appendix.

Preprint – From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models
<table><tr><td>Category</td><td>Count</td><td>Rate</td><td>95% Wilson CI</td></tr><tr><td>Unique</td><td>232</td><td></td><td>92.8% [88.6%, 95.4%]</td></tr><tr><td>Ambiguity</td><td>18</td><td>7.2%</td><td>[4.6%, 11.1%]</td></tr></table>

Table 1: Answer uniqueness with 95% Wilson CI

Embedding-based diversity filtering. We use Qwen3- Embedding-8B [45] to score clue–answer and clue–clue similarity, discarding clues that are either trivially close to the answer or too weakly related to be informative, and clue pairs that are near-duplicates [46]. An experimental embedding similarity threshold ranges from 0.3 to 0.8, which serves as a primary filtering step in benchmark construction.

Answer uniqueness and graded acceptability. A central concern is that a clue set may admit more than one reasonable target. We address this in two ways. (i) Construction: the judge stage filters out clue sets that jointly under-determine the target, and problematic items are reconstructed. (ii) Human verification: on a randomly sampled subset of 250 items, two master’s students majoring in NLP independently assess whether each item admits a unique, unambiguous target. Table 1 reports the results with 95% Wilson CI, indicating that 92.8% of the items were judged as having a unique answer.

Bilingual design. MPAR-Bench includes English and Chinese subsets built with the same pipeline (500 validated items each), combining synthesized and native-speaker-authored samples. The English subset emphasizes lexical and abstract associations; the Chinese subset additionally incorporates idioms, characterlevel and pictographic properties, and contemporary cultural memes. Word clouds for each subset are shown in Fig. 4.

Benchmark rationale. MPAR-Bench is intentionally constructed to emphasize long-range, multi-point association rather than lexical overlap or frequent collocations. By synthesizing clues from complementary perspectives while enforcing low redundancy, each item requires integrating sparse and semantically distant evidence into a single target, making successful prediction less dependent on next token co-occurrence patterns and more on associative reasoning.

## Dificulty Settings and Perturbation

We distinguish two complementary facets of reasoning breadth. The Standard setting measures baseline breadth: given complete, well-formed clues under ideal conditions, can the model integrate multiple semantic signals into a correct answer? To further evaluate whether this integration capability remains reliable under more realistic conditions, we introduce an Enhanced setting that systematically perturbs the standard test protocol to simulate information-restricted or noisy environments. These two settings provide a complete picture: the Standard setting establishes what a model can achieve under favorable conditions, while the Enhanced setting reveals whether that capability is resilient enough to matter in practice [47].

Specifically, we implement the following enhanced transformations and perturbations:

• Clue Masking: Randomly masking clues to evaluate model reasoning ability under information deficiency.

![](images/62d57daa09dec6cfe597bbfbb1673be0a1b38273c5e95d8abc539cc1b2bcf4d3.jpg)  
Figure 5: Examples of Enhanced settings in MPAR-Bench.

• Order Shufling: Shufling the clues order to find whether the model’s reasoning process is sensitive to order.

• Distractors: Injecting semantically misleading or irrelevant cue words to test model resistance to noisy contexts and spurious correlations.

• Multi-step Inferring: Increasing the associative semantic distance between clues and the mystery word, forcing models to generate intermediate latent connections rather than relying on direct surface cooccurrence.

Each enhanced setting evenly distributes words from the standard task. We refer to this setting as the Enhanced MPAR-Bench in the remainder of the paper. As shown in Fig. 5, each question has a corresponding enhanced variant, posing a greater challenge to LLMs across multiple dimensions of robustness.

## Evaluation

Associative reasoning cannot be fully captured by a single exactmatch metric, since semantically reasonable predictions may difer lexically from the ground truth, and some correct predictions may come from flawed reasoning processes. To obtain a more comprehensive understanding of LLM behavior, we evaluate models at three progressively finer granularities: accuracy, word-based similarity, and the validity of reasoning trace, i.e., the explanation of how the answer and clues are connecting with each other.

## Accuracy

We first evaluate model performance using exact-match accuracy. A prediction is considered correct if and only if it exactly matches the ground-truth answer. We report accuracy across diferent subsets, including multilingual Standard and Enhanced settings, to measure models’ associative retrieval capability un der varying semantic and linguistic conditions.

Preprint – From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models
<table><tr><td>Model</td><td>English</td><td>Chinese</td></tr><tr><td>GPT-5.2</td><td></td><td>Acc ANLS Emb Trace|Acc ANLS Emb Trace</td></tr><tr><td></td><td></td><td>77.60.7920.8480.933|64.40.7400.823 0.954 Gemini-3.1pro 86.8 0.884 0.915 0.941|72.2 0.802 0.869 0.957</td></tr><tr><td>Sonnet-4.5 Qwen3-max</td><td></td><td>79.0 0.8030.856 0.865|67.4 0.767 0.839 0.927</td></tr><tr><td>Kimi-k2</td><td></td><td>73.2 0.7430.800 0.878|65.00.7330.813 0.928</td></tr><tr><td>Seed-2-pro</td><td>Deepseek-v3.2 69.80.7160.784 0.848|61.20.6860.793 0.893</td><td>71.6 0.729 0.795 0.871|57.4 0.675 0.770 0.888</td></tr></table>

Table 2: Standard MPAR-Bench Results on Thinking Models
<table><tr><td>Model</td><td>English</td><td>Chinese</td></tr><tr><td></td><td>Acc ANLS Emb Trace|Acc ANLS Emb Trace</td><td></td></tr><tr><td>GPT-5.2</td><td>66.60.6810.7640.902|58.80.6690.772 0.932 Gemini-3.1pro 76.9 0.778 0.842 0.906|64.0 0.7220.815 0.926</td><td></td></tr><tr><td>Sonnet-4.5</td><td></td><td>63.4 0.646 0.739 0.836|61.0 0.6890.789 0.873</td></tr><tr><td>Qwen3-max Kimi-k2 Deepseek-v3.2 52.00.5310.639 0.794|52.40.5960.721 0.842</td><td>57.60.5960.6870.82745.80.5470.6830.850</td><td>56.80.5910.679 0.841|54.80.6210.732 0.911</td></tr></table>

Table 3: Enhanced MPAR-Bench Results on Thinking Models

## Word-based Evaluation

To further evaluate semantic proximity between predictions and target concepts, we adopt both Average Normalized Levenshtein Similarity (ANLS) and word embedding similarity to judge at the semantic level.

ANLS [48] is computed using normalized Levenshtein edit distance:

$$
\mathrm { A N L S } ( \hat { y } , y ) = 1 - \frac { d _ { \mathrm { l e v } } ( \hat { y } , y ) } { \operatorname* { m a x } ( | \hat { y } | , | y | ) } ,\tag{1}
$$

where $\hat { y }$ and y denote the model prediction and ground truth respectively, $d _ { \mathrm { l e v } } ( \cdot , \cdot )$ denotes the Levenshtein edit distance, and | · | denotes string length.

Moreover, we compute word embedding similarity using fastText [49] as a word embedding model:

$$
\mathrm { S i m } ( \hat { y } _ { e m b } , y _ { e m b } ) = \frac { \hat { y } _ { e m b } ^ { \top } y _ { e m b } } { \lVert \hat { y } _ { e m b } \rVert _ { 2 } \lVert y _ { e m b } \rVert _ { 2 } } .\tag{2}
$$

where $\hat { y } _ { e m b }$ and $y _ { e m b }$ denote the word embedding of the model prediction and ground truth in fastText respectively, and ∥ · ∥ denotes their Euclidean (ℓ ) norms.

## Reasoning Trace Evaluation

Beyond final-answer accuracy, we assess the validity of intermediate processes via reasoning trace evaluation, decomposed into two dimensions: logical verification and factual verification [50, 25, 26]. Logical verification examines whether reasoning trajectories follow coherent inferential steps from clues to predictions, while factual verification checks whether intermediate claims are factually grounded. This dual assessment distinguishes valid associative reasoning from spurious correlations and hallucinated paths. We manually review a randomly sampled subset of 300 reasoning trace predictions, confirming high consistency (98.7% on factual verification, 94.7% on logical verification) between human and LLM judgement. Evaluation prompts are provided in Appendix.

<table><tr><td rowspan="2">Model</td><td>English</td><td>Chinese</td></tr><tr><td></td><td>Acc ANLS Emb Trace|Acc ANLS Emb Trace</td></tr><tr><td>GPT-5.2</td><td></td><td>59.60.6140.705 0.811|61.80.7180.8040.943</td></tr><tr><td>Sonnet-4.5</td><td>70.40.7160.791 0.838|68.80.7760.843 0.921</td><td>Gemini-3flash 70.0 0.7170.787 0.831|67.0 0.7650.836 0.920</td></tr><tr><td>Qwen3-max</td><td>DeepSeek-v3.2 51.40.5360.644 0.730|60.60.7020.788 0.906</td><td>55.40.5720.6690.792|64.40.7480.820 0.930</td></tr></table>

Table 4: Standard MPAR-Bench Results on Non-thinking Models

<table><tr><td rowspan="2">Model</td><td>English</td><td>Chinese</td></tr><tr><td></td><td>Acc ANLS Emb Trace|Acc ANLS Emb Trace</td></tr><tr><td>GPT-5.2</td><td></td><td>44.20.4600.586 0.754|54.40.6320.749 0.906 Gemini-3flash56.0 0.580 0.675 0.763|61.4 0.6950.791 0.870</td></tr><tr><td>Sonnet-4.5 Qwen3-max</td><td>44.80.4640.5860.751|56.00.6620.7580.874 Deepseek-v3.242.60.4440.566 0.685|54.00.6180.7380.851</td><td>55.80.572 0.679 0.774|62.0 0.7070.797 0.865</td></tr></table>

Table 5: Enhanced MPAR-Bench Results on Non-thinking Models

![](images/90cb9de7588e1a65535303e625345a9fa2dc2dc2dde153c21bc23a80332336af.jpg)  
Figure 6: Challenging Cases in MPAR-Bench. Examples where most models fail to identify the correct answer.

## Experiments and Results

## Implementation Details

We benchmark a diverse set of representative LLM families— including GPT, Gemini, Sonnet, Qwen, Kimi, DeepSeek, and Seed series [2, 3, 4, 5, 6, 7, 8, 9]; the concrete information is shown in Appendix. Evaluations are conducted across both thinking and non-thinking modes, under standard and enhanced settings, on the bilingual subsets of MPAR-Bench.

We also report fine-grained metrics including ANLS, word embedding similarity, and reasoning trace failure analysis to dissect the reasoning behaviors of LLMs from lexical, semantic, and reasoning process perspectives.

Preprint – From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

## Main Results

We analyze the results from three key perspectives: cross-model comparison, the efect of thinking mode, and robustness under perturbation. Challenging cases in MPAR-Bench are shown in Fig. 6, and more detailed experimental results are provided in Appendix.

Model Comparisons. Table 2 and Table 4 report the standard MPAR-Bench results under thinking and non-thinking modes. Under the thinking mode, Gemini-3.1pro leads on both English (86.8%) and Chinese (72.2%), followed by GPT-5.2 and Sonnet-4.5 in English. Under the non-thinking mode, Sonnet-4.5 achieves the highest accuracy in both languages.

Standard vs. Enhanced. Comparing standard (Tables 2, 4) with enhanced settings (Tables 3, 5) reveals consistent degradation under perturbation. For instance, in thinking mode, Deepseek-v3.2 exhibits the largest decline in English, while Kimi-k2 drops the most in Chinese. In non-thinking mode, Seed-2-pro shows the largest decline in English, whereas Qwen3-max drops the sharpest in Chinese. These contrasts indicate that robustness varies substantially across models: some sufer pronounced degradation under perturbation while others remain comparatively stable. Notably, this robustness is strongly model and language-dependent. Detailed information and analyses are shown in Appendix.

Thinking vs. Non-Thinking. Comparing thinking-mode results (Tables 2, 3) against their non-thinking counterparts (Tables 4, 5), we find that thinking mode [26, 27] consistently improves most indicators, but the magnitude of the gain is markedly larger on English than on Chinese: averaged across models, thinking lifts English accuracy by a substantially wider margin and produces clear, stable gains for every model, whereas its efect on Chinese is much smaller and model-dependent (Sonnet-4.5 even shows a slight regression), and the improvement under perturbation is non-monotonic. Moreover, thinking mode is not always reliable: models occasionally overthink and override correct intermediate answers, as discussed in the next section and Appendix. This suggests a crucial distinction: while thinking mode improves reasoning depth, it may not necessarily enhance reasoning breadth.

## Discussion

The benchmark results above establish that current LLMs exhibit measurable but imperfect reasoning breadth, and that this capability degrades under perturbation. A natural follow-up question is: under what conditions does reasoning breadth improve, and what mechanisms cause it to fail? We investigate this question through four complementary lenses. First, we examine overthinking, a failure mode in which extended reasoning actively harms breadth by overriding correct intermediate answers. Second, we characterize information gain curve, asking how breadth scales as more clues become available. Third, we explore scaling laws to determine whether larger models inherently develop broader reasoning. Fourth, we test whether semantic feedback can steer models toward correct answers across multiple refinement rounds. We additionally try to create a structured reasoning skill as an intervention strategy; results are reported in Appendix.

<table><tr><td rowspan="2">Model</td><td colspan="3">English</td><td colspan="3">Chinese</td></tr><tr><td>Wrong Ex.</td><td>Ans. Mention</td><td>Token Len.</td><td>|Wrong Ex.</td><td>Ans. Mention</td><td>Token Len.</td></tr><tr><td>Sonnet-4.5</td><td>105</td><td>0.457</td><td>0.371</td><td>163</td><td>0.706</td><td>0.571</td></tr><tr><td>Qwen3-max</td><td>134</td><td>0.590</td><td>0.515</td><td>174</td><td>0.874</td><td>0.690</td></tr><tr><td>Kimi-k2</td><td>142</td><td>0.585</td><td>0.415</td><td>213</td><td>0.864</td><td>0.366</td></tr><tr><td>Deepseek-v3.2</td><td>151</td><td>0.623</td><td>0.510</td><td>194</td><td>0.830</td><td>0.634</td></tr><tr><td>Seed-2-pro</td><td>143</td><td>0.497</td><td>0.406</td><td>177</td><td>0.751</td><td>0.599</td></tr></table>

Table 6: Overthinking Results on models.

![](images/a118778e2f1116283a299cf75e5adfa85515d4072fef662d86b1811b6e6c0f5f.jpg)  
Figure 7: Information Gain Curve of Seed-2-pro. As number of words increases, accuracy rises but at a decreasing rate.

## Overthinking

A notable failure pattern we observe is overthinking [28, 29]: models initially arrive at the correct answer but subsequently override it during extended reasoning, often drifting toward a semantically related but incorrect concept. This behavior is particularly pronounced in Qwen3-max and Kimi-k2. For instance, given the answer word Philosophy, the model outputs Plato, over-focusing on a representative entity implied by the clues rather than the academic discipline itself.

Table 6 reports a detailed overthinking analysis. “Wrong Ex.” is the total count of incorrect predictions. “Ans. Mention” (ratio) is the proportion of those incorrect cases in which the correct answer appeared in the reasoning trace but was subsequently overridden. “Token Len.” (ratio) is the proportion of incorrect cases whose reasoning length exceeded the model’s average reasoning length on correct samples.

## Information Gain Curve

The information gain curve characterizes how a model’s reasoning accuracy scales with the number of provided clue words; we utilize it to evaluate Seed-2-pro’s capacity to leverage incremental semantic evidence for multi-point associative reasoning. As shown in Fig. 7, accuracy consistently improves as the number of clue words increases, suggesting that the model is able to accumulate and integrate incremental semantic information across multiple clues. This provides preliminary evidence that LLMs possess multi-point associative reasoning capability—the ability to jointly combine several semantically distinct clues into a coherent answer. However, the marginal gain progressively slows down, indicating that while models benefit from richer semantic context, they saturate beyond a certain evidence threshold.

Preprint – From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models

![](images/f9007c7a0d099ff832714ae51493c4ad3f1dd747b770aaa93d84122f0c3cceef.jpg)  
Figure 8: Scaling Law of Qwen3 family in Chinese MPAR-Bench. As model scales up, all metrics improve, except for Qwen3-32B, which has a more severe overthinking issue.

## Scaling Law

We investigate how model scale afects performance on our benchmark. In particular, Fig. 8 reports the scaling behavior of the locally deployed Qwen3 family in thinking mode (0.6B to 32B parameters), evaluated on accuracy, ANLS, and fastText embedding similarity across both English and Chinese datasets under Standard and Enhanced settings. Except for Qwen3-32B, accuracy consistently improves with increasing model size [51, 52, 53]. Case-level inspection finds that Qwen3-32B sufers from overthinking, causing it to reject correct answers during an extended reasoning process.

## Feedback

The proposed feedback method iteratively delivers semantic similarity metrics (ANLS and average word embedding similarity), guiding the model to refine its responses toward the answer across multiple rounds. Fig. 9 visualizes the corresponding trajectories (rounds 3–6) where the mechanism successfully corrected Qwen3-max’s outputs. The results show that LLMs are not fully sensitive to word embedding similarity as a guidance signal; instead, they rely on other internal strategies. While feedback provides opportunities for exploration and sometimes enables recovery of the answer, the revision trajectory remains weakly aligned with semantic indicators, suggesting that models do not naturally exploit surface-level semantic proximity for iterative refinement.

## Conclusion

We introduced MPAR-Bench, a bilingual benchmark that evaluates multi-point associative reasoning—reasoning breadth—in LLMs through 1,000 boardgame-rule-based questions and a coarse-to-fine evaluation protocol spanning accuracy, ANLS, embedding similarity, and reasoning-trace verification, complemented by a four-axis perturbation suite.

Our experiments yield three findings. First, reasoning breadth remains far from solved: the best models reach 86.8%/72.2% accuracy in English/Chinese, with perturbations causing 9–18/5–12 point drops. Second, greater reasoning depth does not automatically confer breadth: thinking mode improves standard-setting accuracy but does not consistently reduce perturbation sensitivity, and case-level analysis reveals that extended reasoning can override correct answers through overthinking. Third, improving breadth appears challenging: scaling model size, adding reasoning strategies, and iterative feedback each bring only partial gains, suggesting that reasoning breadth may be a capability that current training paradigms do not naturally optimize for. We release MPAR-Bench and its pipeline to encourage the community to move beyond depth-oriented evaluation and toward a more complete picture of reasoning—one that values breadth as much as depth.

![](images/d753c4f95411fc475b96aa7bff145aa618af09aff3e27139a7d4d010db90b264.jpg)  
Figure 9: Chinese MPAR-Bench feedback results for Qwen3- max in thinking mode, rounds 3–6. Blue lines show individual cases; green lines show average trends.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[2] OpenAI. Gpt-5.2. https://openai.com/index/ introducing-gpt-5-2/, 2025.

[3] Google. Gemini 3 flash: frontier intelligence built for speed. https://blog.google/ products-and-platforms/products/gemini/ gemini-3-flash/, 2025.

[4] Google. Gemini 3.1 pro: A smarter model for your most complex tasks. https://blog.google/ innovation-and-ai/models-and-research/ gemini-models/gemini-3-1-pro/, 2026.

[5] Anthropic. Introducing claude sonnet 4.5. https://www. anthropic.com/news/claude-sonnet-4-5, 2025.

[6] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[7] Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, et al. Deepseek-v3. 2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025.

[8] Yifan Bai, Yiping Bao, Y Charles, Cheng Chen, Guanduo Chen, Haiting Chen, Huarong Chen, Jiahao Chen, Ningxin Chen, et al. Kimi k2: Open agentic intelligence. arXiv preprint arXiv:2507.20534, 2025.

[9] Bytedance. Seed 2.0. https://seed.bytedance.com/ en/blog/seed-2-0-official-launch, 2026.

[10] Long Ouyang, Jef Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 27730–27744, 2022.

[11] Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zeroshot learners. In International Conference on Learning Representations (ICLR), 2022.

[12] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 24824–24837, 2022.

[13] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-augmented generation

for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), volume 33, pages 9459–9474, 2020.

[14] Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the MATH dataset. In Advances in Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021.

[15] Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. Measuring massive multitask language understanding. In International Conference on Learning Representations (ICLR), 2021.

[16] Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

[17] Aarohi Srivastava et al. Beyond the imitation game: Quantifying and extrapolating the capabilities of language models. Transactions on Machine Learning Research (TMLR), 2023.

[18] Percy Liang, Rishi Bommasani, Tony Lee, et al. Holistic evaluation of language models. Transactions on Machine Learning Research (TMLR), 2023.

[19] Johannes Treutlein, Dami Choi, Jan Betley, Sam Marks, Cem Anil, Roger Grosse, and Owain Evans. Connecting the dots: Llms can infer and verbalize latent structure from disparate training data. Advances in Neural Information Processing Systems, 37:140667–140730, 2024.

[20] Guohao Li, Hasan Abed Al Kader Hammoud, Hani Itani, Dmitrii Khizbullin, and Bernard Ghanem. CAMEL: Communicative agents for “mind” exploration of large language model society. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[21] Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, and Hannaneh Hajishirzi. Self-instruct: Aligning language models with selfgenerated instructions. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 13484–13508, 2023.

[22] Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, and Igor Mordatch. Improving factuality and reasoning in language models through multiagent debate. In Proceedings ofthe 41st International Conference on Machine Learning (ICML), 2024.

[23] Shunyu Yao, Dian Yu, Jefrey Zhao, Izhak Shafran, Thomas Grifiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

[24] Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. Plan-and-solve prompting: Improving zero-shot chain-of-thought reasoning by large language models. In Proceedings ofthe 61st

Annual Meeting of the Association for Computational Linguistics (ACL), pages 2609–2634, 2023.

[25] Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. Large language models are better reasoners with self-verification. In Findings ofthe Associationfor Computational Linguistics: EMNLP, pages 2550–2575, 2023.

[26] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), 2024.

[27] DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. Nature, 2025.

[28] Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. Do NOT think that much for 2+3=? on the overthinking of o1-like LLMs. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics (ACL), 2025.

[29] Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on eficient reasoning for large language models. arXiv preprint arXiv:2503.16419, 2025.

[30] Sarnof Mednick. The associative basis of the creative process. Psychological review, 69(3):220, 1962.

[31] Claudia Schon, Ulrich Furbach, and Marco Ragni. Modeling associative reasoning processes. arXiv preprint arXiv:2201.00716, 2022.

[32] Harsh Kumar, Jonathan Vincentius, Ewan Jordan, and Ashton Anderson. Human creativity in the age of llms: Randomized experiments on divergent and convergent thinking. In Proceedings of the 2025 CHI conference on human factors in computing systems, pages 1–18, 2025.

[33] Astrid Carolus, Martin J Koch, and Shuyan Feng. Timeon-task and instructions help humans to keep up with ai: replication and extension of a comparison of creative performances. Scientific reports, 15(1):20173, 2025.

[34] Vikram Arora, Alex Thabane, Sameer Parpia, Goran Calic, and Mohit Bhandari. Generative artificial intelligence models outperform students on divergent and convergent thinking assessments. Scientific Reports, 15(1):36987, 2025.

[35] Chunyuan Deng, Yilun Zhao, Yuzhao Heng, Yitong Li, Jiannan Cao, Xiangru Tang, and Arman Cohan. Unveiling the spectrum of data contamination in language model: A survey from detection to remediation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16078–16092, 2024.

[36] Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, et al. Benchmarking large language models under data contamination: A survey from static to dynamic evaluation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 10091–10109, 2025.

[37] Zimeng Huang, Jinxin Ke, Xiaoxuan Fan, Yufeng Yang, Yang Liu, Liu Zhonghan, Zedi Wang, Junteng Dai, Haoyi Jiang, Yuyu Zhou, et al. Mm-opera: Benchmarking openended association reasoning for large vision-language models. arXiv preprint arXiv:2510.26937, 2025.

[38] Manya Wadhwa, Tiasa Singha Roy, Harvey Lederman, Junyi Jessy Li, and Greg Durrett. Create: Testing llms for associative creativity. arXiv preprint arXiv:2603.09970, 2026.

[39] Matthew Stephenson, Matthew Sidji, and Benoît Ronval. Codenames as a benchmark for large language models. IEEE Transactions on Games, 2025.

[40] Sherzod Hakimov, Lara Pfennigschmidt, and David Schlangen. Ad-hoc concept forming in the game codenames as a means for evaluating large language models. In Proceedings of the Fourth Workshop on Generation, Evaluation and Metrics (GEM<sup>2</sup>), pages 728–740, 2025.

[41] Tim Merino, Sam Earle, Ryan Sudhakaran, Shyam Sudhakaran, and Julian Togelius. Making new connections: Llms as puzzle generators for the new york times’ connections word game. In Proceedings ofthe AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment, volume 20, pages 87–96, 2024.

[42] Tanguy Cazalets and Joni Dambre. Word synchronization challenge: A benchmark for word association responses for large language models. In International Conference on Human-Computer Interaction, pages 3–19. Springer, 2025.

[43] Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. On llms-driven synthetic data generation, curation, and evaluation: A survey. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 11065–11082, 2024.

[44] Anna Bavaresco, Rafaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, et al. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 2: Short Papers), pages 238–255, 2025.

[45] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.

[46] Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. Semdedup: Data-eficient learning at web-scale through semantic deduplication, 2023. URL https://arxiv. org/abs/2303.09540, 2021.

[47] Guangxiang Zhao, Saier Hu, Xiaoqi Jian, Jinzhu Wu, Yuhan Wu, Lin Sun, and Xiangzheng Zhang. Stress testing generalization: How minor modifications undermine large language model performance. arXiv e-prints, pages arXiv–2502, 2025.

[48] Ali Furkan Biten, Rubèn Tito, Andres Mafla, Lluis Gomez, Marçal Rusiñol, Ernest Valveny, C. V. Jawahar, and Dimosthenis Karatzas. Scene text visual question answering.

In Proceedings ofthe IEEE/CVF International Conference on Computer Vision (ICCV), pages 4291–4301, 2019.

[49] Piotr Bojanowski, Edouard Grave, Armand Joulin, and Tomas Mikolov. Enriching word vectors with subword information. Transactions ofthe Associationfor Computational Linguistics, 5:135–146, 2017. ISSN 2307-387X.

[50] Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, et al. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.

[51] Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jefrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.

[52] Jordan Hofmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pages 30016–30030, 2022.

[53] Rylan Schaefer, Brando Miranda, and Sanmi Koyejo. Are emergent abilities of large language models a mirage? In Advances in Neural Information Processing Systems (NeurIPS), volume 36, 2023.

## Limitations

While this benchmark provides an initial step toward evaluating the multi-point associative reasoning capabilities of LLMs, there remains some room to further broaden its coverage and ecological validity. Future work may extend the benchmark with more diverse and interactive settings to better capture associative reasoning behaviors that arise in real-world environments. In addition, we hope this benchmark can serve as a foundation for systematically studying non-linear associative and creative capabilities in LLMs, as well as for developing evaluation protocols and modeling principles that more closely align with practical applications of multi-point associative reasoning.

## Ethical Considerations

MPAR-Bench is built upon lexical materials sourced from cognitive psychology (the Remote Associates Test) and cooperative gaming (Just One). Throughout dataset construction, we adopted a series of safeguards to preempt potential ethical risks. The benchmark consists solely of word-level clues and semantic associations, carrying no personally identifiable information and requiring no collection of human subject data. All clues were either drawn from publicly available game corpora or synthesized through an LLM-based multi-agent pipeline, after which embedding-based filtering and manual verification were applied to exclude content that could introduce demographic, cultural, or other unintended biases, in accordance with established ethical standards. The bilingual design of our benchmark further ensures equitable treatment of English and Chinese linguistic contexts, with dedicated attention to cultural appropriateness within each subset. Beyond its primary evaluation purpose, MPAR-Bench also serves as a diagnostic instrument for probing model reasoning behaviors and uncovering failure modes, thereby supporting the broader goal of identifying and mitigating ethical vulnerabilities in deployed AI systems.

## Multi-agent Generation Prompts

Prompts are used in English and Chinese in generating, evaluating, and judging models separately.

## Questioner Prompt

You are playing the board game “Just One”.

Secret answer: {word}

Requirements:

1. Give exactly 1 English clue word.

2. The clue must be a common English word or a widely recognized proper noun.

3. The clue must not be the answer itself, a translation, a synonym, a homophone, a made-up word, an obvious morphological variant, or contain the answer as a substring.

4. The clue should be indirect and moderately dificult, but useful when combined with other clues.

5. You MUST approach the answer from this specific association angle: {angle}

## Judger Prompt

You are the judge for the board game “Just One”.

Secret answer: {word}

Already approved clues (LOCKED - do NOT remove or reevaluate these): {locked\_clues}

New candidate clues to evaluate: {clue\_list}.

Your tasks (apply only to the NEW candidates above):

1. Remove any new clue that IS the answer itself, a direct synonym, a translation, a homophone, an obvious morphological variant, or contains the full answer as a substring.

2. Remove EXACT duplicates among the new candidates.

3. Remove any new clue that refers to the SAME specific concept, entity, or phrase as an already-approved clue or another new candidate. Clues that merely belong to the same broad category (e.g., two diferent fictional characters, two diferent countries) are NOT duplicates - keep both.

4. Remove genuinely LOW-QUALITY new clues: completely obscure, made-up, grammatically wrong, or with no logical connection to the answer. Do NOT remove a clue just because it requires one step of reasoning.

5. Do NOT remove a clue simply because it seems “too direct” unless using it would immediately give away the answer with zero reasoning required.

## Player Prompt

You are the one who guess the mystery word for “Just One”.

Based on the clue words given, guess the mystery word and explain the logical connection between each related clue word and the mystery word.

The clue words are “{clue\_word1}”, “{clue\_word2}”, “{clue\_word3}”

Requirements:

1. The mystery word is only one word.

2. Provide the connection between each clue word and the mystery word.

## Evaluation Prompt

You are a rigorous AI logic auditor. Your task is to evaluate each reasoning step produced by an Agent in a word-guessing game.

Break the reasoning into independent steps (atoms) and judge each atom on two dimensions: “Factual Accuracy” and “Logical Soundness”.

For each step:

1. Fact\_Check: Are the objective claims in this step (numbers, ingredients, mechanisms, historical origins, etc.) factually correct? (Pass / Fail)

2. Logic\_Check: Is the reasoning chain from the clue to the predicted answer natural and sound? Are there signs of over-generalization, edge-case pandering, or multi-layer reinterpretation? (Pass / Fail)

Judging Principles (you MUST follow these):

\- Scrutinize specific claims (numbers, ingredients, mechanisms, definitions) — any factual error → Fact Fail

\- If a step cherry-picks a marginal meaning of the clue to fit the answer, ignoring a more obvious association → Logic Fail - If the clue clearly points to a more specific/precise word, but the step generalizes it to a broad concept → Logic Fail

\- If the step requires more than two layers of inference or reinterpretation to connect the clue to the answer → Logic Fail - Only when the reasoning feels natural, direct, and requires no mental gymnastics for an ordinary person should it be judged Logic Pass

## Asset Distribution and Compliance

To ensure reproducibility, both the English and Chinese subsets of MPAR-Bench, along with our evaluation scripts, will be publicly released under the MIT License upon publication. Our evaluation items are generated via frontier LLM APIs (e.g., GPT, Gemini, Qwen), and we have verified that our pipeline adheres to the respective terms of service of these model providers, restricting the usage of our dataset strictly to non-commercial academic benchmarking.

## Experiment Setup

For all evaluated models we keep the oficial default sampling parameters specified in each provider’s release. For reasoning models, we additionally set reasoning\_efort=high where the API supports it. We do not perform any per-model hyperparameter tuning. Detailed model configurations for all experiments are provided in Table 7. And detailed software versions are provided in Table 8.

## Detailed Results

## Token Usage

Tables 9 and 10 report the average token consumption of models operating in thinking mode, encompassing prompt tokens, Chain-of-Thought reasoning tokens, and output tokens.

Token consumption increases consistently from the Standard to the Enhanced setting for nearly all models, reflecting the greater reasoning demand imposed by the four perturbation conditions. The magnitude of this increase, however, varies markedly across models, pointing to fundamentally diferent strategies for allocating reasoning computation.

The most pronounced outlier is Qwen3-max, which exhibits disproportionately high token consumption in English across both settings, averaging roughly 10,000 tokens in Standard MPAR-Bench and 11,700 in Enhanced MPAR-Bench. Qualitative inspection of its generated reasoning trace reveals a consistent tendency toward repetitive self-verification: the model frequently revisits intermediate conclusions, generates multiple redundant candidate answers, and enters circular deliberation loops before converging on a final prediction. This overthinking behavior substantially inflates token usage.

## Enhanced MPAR-Bench Result Analysis

Tables 11 and 12 present the detailed accuracy of model across four perturbation variants in the Enhanced MPAR-Bench setting: Clue Masking, Order Shufling, Distractor Injection, and Multistep Inferring.

The four perturbation types exhibit diferent impacts on model performance. Order Shufling consistently yields the highest accuracy across all models, some of which are even higher than Standard setting. In contrast, Clue Masking causes the most severe degradation in English, with an average drop of 20.0% from Order Shufling. Distractor Injection also substantially reduces performance, particularly for Qwen3-max and Seed-2- pro, both dropping by over 28% from their respective Standard accuracies. This indicates that spurious semantic correlations introduced by irrelevant clue words can efectively derail the model’s reasoning trajectory. Multi-step Inferring occupies an intermediate dificulty level, suggesting that extending the associative chain length moderately taxes the model’s long-range semantic mapping capacity but does not fundamentally break the reasoning process.

Preprint – From Reasoning Depth to Reasoning Breadth: Evaluating Multi-Point Associative Reasoning in Large Language Models 12 12
<table><tr><td>Models</td><td>Model Size</td><td>Access</td><td>Version</td><td>Provider</td></tr><tr><td>GPT-5.2</td><td>undisclosed</td><td>api</td><td>gpt-5.2-2025-12-11</td><td>OpenAI</td></tr><tr><td>Gemini-3.1pro</td><td>undisclosed</td><td>api</td><td>Gemini-3.1pro-preview</td><td>Google</td></tr><tr><td>Gemini-3flash</td><td>undisclosed</td><td>api</td><td>Gemini-3flash-preview</td><td></td></tr><tr><td>Sonnet-4.5</td><td>undisclosed</td><td>api</td><td>claude-sonnet-4-5-20250929</td><td>Anthropic</td></tr><tr><td>Qwen3-max</td><td>undisclosed</td><td>api</td><td>qwen3-max-2026-01-23</td><td rowspan="6">Qwen</td></tr><tr><td>Qwen3-0.6B Qwen3-1.7B</td><td>0.6B 1.7B</td><td>weights</td><td></td></tr><tr><td>Qwen3-4B</td><td>4B</td><td>weights</td><td></td></tr><tr><td>Qwen3-8B</td><td>8B</td><td>weights</td><td></td></tr><tr><td>Qwen3-14B</td><td>14B</td><td>weights</td><td></td></tr><tr><td>Qwen3-32B</td><td>32B</td><td>weights weights</td><td></td></tr><tr><td>Kimi-k2</td><td>1T</td><td>api</td><td>kimi-k2-thinking-251104</td><td>Moonshot AI</td></tr><tr><td>Deepseek-v3.2</td><td>671B</td><td></td><td></td><td>DeepSeek</td></tr><tr><td>Seed-2-pro</td><td>undisclosed</td><td>api api</td><td>Deepseek-v3.2 doubao-seed-2-0-pro-260215</td><td>ByteDance</td></tr></table>

Table 7: Summary of Evaluated Models

<table><tr><td>Component</td><td>Version</td></tr><tr><td>Python</td><td>3.11.14</td></tr><tr><td>PyTorch</td><td>2.6.0</td></tr><tr><td>transformers</td><td>4.57.1</td></tr><tr><td>accelerate</td><td>1.10.1</td></tr><tr><td>vllm</td><td>0.16.0rc2</td></tr><tr><td>openai</td><td>1.96.1</td></tr><tr><td>tiktoken</td><td>0.9.0</td></tr><tr><td>fastText</td><td>0.9.3</td></tr><tr><td>gensim</td><td>4.3.0</td></tr></table>

Table 8: Software versions used in our experiments

<table><tr><td>Model</td><td>Avg Token (English)</td><td>Avg Token (Chinese)</td></tr><tr><td>GPT-5.2</td><td>1741.2</td><td>1338.9</td></tr><tr><td>Gemini-3.1pro</td><td>4940.7</td><td>2409.8</td></tr><tr><td>Sonnet-4.5</td><td>1577.4</td><td>1838.7</td></tr><tr><td>Qwen3-max</td><td>11743.4</td><td>4782.4</td></tr><tr><td>Kimi-k2</td><td>4790.1</td><td>4120.7</td></tr><tr><td>Deepseek-v3.2</td><td>3255.0</td><td>2307.8</td></tr><tr><td>Seed-2-pro</td><td>2140.7</td><td>1858.6</td></tr></table>

Table 10: Token Usage in Thinking Mode LLMs (Enhanced)

<table><tr><td>Model</td><td>Avg Token (English)</td><td>Avg Token (Chinese)</td></tr><tr><td>GPT-5.2</td><td>1355.1</td><td>1263.6</td></tr><tr><td>Gemini-3.1pro</td><td>3551.1</td><td>1883.3</td></tr><tr><td>Sonnet-4.5</td><td>1468.9</td><td>1729.2</td></tr><tr><td>Qwen3-max</td><td>9983.3</td><td>777.9</td></tr><tr><td>Kimi-k2</td><td>4659.5</td><td>3290.7</td></tr><tr><td>Deepseek-v3.2</td><td>3091.0</td><td>2194.4</td></tr><tr><td>Seed-2-pro</td><td>1865.2</td><td>1563.8</td></tr></table>

Table 9: Token Usage in Thinking Mode LLMs (Standard)

<table><tr><td rowspan=2 colspan=4>English Acc(%)ModelMultiMask Shuf. Dis.step</td><td rowspan=1 colspan=2>Chinese Acc(%)</td></tr><tr><td rowspan=1 colspan=2>MultiMask Shuf. Dis.step</td></tr><tr><td rowspan=1 colspan=4>GPT-5.2      56.880.860.868.0</td><td rowspan=2 colspan=2>48.868.0 67.2 51.273.6 66.4 60.8</td></tr><tr><td rowspan=1 colspan=3>Gemini-3.1pro67.286.3 76.7 77.6</td><td rowspan=1 colspan=2>77.6</td><td rowspan=1 colspan=1>55.2</td></tr><tr><td rowspan=1 colspan=2>Sonnet-4.5    53.673.6 56.8</td><td rowspan=2 colspan=3>69.6</td><td rowspan=3 colspan=1>70.4 67.2 56.863.2 58.4 49.635.257.6 48.841.6</td></tr><tr><td rowspan=2 colspan=4>Qwen3-max   56.068.8 43.2 59.2Kimi-k2       49.669.6 49.6 61.6</td></tr><tr><td rowspan=1 colspan=1>48.0 63.</td></tr><tr><td rowspan=1 colspan=4>Deepseek-v3.240.864.8 44.8 57.6</td><td rowspan=1 colspan=2>44.860.057.647.2</td></tr><tr><td rowspan=1 colspan=4>Seed-2-pro    47.267.2 43.2 68.8</td><td rowspan=1 colspan=2>51.268.0 60.8 52.8</td></tr></table>

Table 11: Detailed Results of Enhanced Benchmark on Thinking Models

## Reasoning Trace Error Analysis

Tables 13 through 16 present the reasoning trace evaluation results, decomposing reasoning failures into “fact fail” and “logic fail”.

Across all models, logical error rates substantially exceed factual error rates. In the English thinking standard setting, Deepseekv3.2 exhibits a logical error rate of 43.60%, Kimi-k2 40.56%, and Qwen3-max 38.21%. Even the best-performing model, Gemini-3.1pro, shows a logical error rate of 20.61%. By contrast, factual error rates in the same setting are considerably lower: Deepseek-v3.2 at 13.60%, Kimi-k2 at 15.77%, and Gemini-3.1pro at 11.52%. This indicates that the primary failure in multi-point associative reasoning is not incorrect factual knowledge but rather invalid inferential jumps in constructing the reasoning chain from clues to the answer. Moreover, logical error rates in Chinese reasoning are consistently lower than in English.

Furthermore, in the non-thinking mode, factual error rates increase substantially, which suggests that without an explicit reasoning process, models are more susceptible to factual hallucination.

<table><tr><td rowspan="2">Model</td><td colspan="3">English Acc(%)</td><td rowspan="2">Chinese Acc(%)</td></tr><tr><td>Mask Shuf. Dis.</td><td>Multi step</td><td>Multi Mask Shuf. Dis.</td></tr><tr><td>GPT-5.2</td><td>32.0</td><td>54.5 39.2 51.2</td><td>42.4</td><td>step 67.2 64.8 43.2</td></tr><tr><td>Gemini-3flash Sonnet-4.5</td><td>43.2</td><td>68.8 51.2 60.8 72.8 51.2 53.6</td><td>49.6</td><td>76.0 51.2 68.8 71.2 71.2 52.0</td></tr><tr><td>Qwen3-max</td><td>45.6</td><td>55.2 44.044.8</td><td>53.6 44.0</td><td></td></tr><tr><td></td><td>35.2</td><td></td><td></td><td>65.6 67.2 47.2</td></tr><tr><td>Deepseek-v3.2</td><td>32.0</td><td>56.8 39.2 42.4</td><td></td><td>47.2 61.6 62.444.8</td></tr><tr><td>Seed-2-pro</td><td>31.2</td><td>56.0 33.6 52.8</td><td></td><td>47.2 64.8 58.4 53.6</td></tr></table>

Table 12: Detailed Results of Enhanced Benchmark on Nonthinking Models
<table><tr><td rowspan="3">Model</td><td colspan="3">English</td><td colspan="2">Chinese</td></tr><tr><td colspan="2">Fact(%)</td><td colspan="2">Logic(%)</td><td>Fact(%) Logic(%)</td></tr><tr><td>√ X</td><td></td><td>×</td><td>√ X</td><td>√ X</td></tr><tr><td>GPT-5.2</td><td>1.75 6.61</td><td></td><td>6.3924.64|0.87 2.25|</td><td></td><td>5.5911.69</td></tr><tr><td rowspan="3">Gemini-3.1pro 2.58 11.52 Sonnet-4.5</td><td></td><td></td><td>6.08 20.61</td><td>2.16 2.45</td><td>5.71 7.77</td></tr><tr><td>4.3017.14</td><td></td><td>8.71 31.62</td><td>3.44 5.28</td><td>8.5514.72</td></tr><tr><td>4.3718.96</td><td></td><td>8.0338.21</td><td>2.22 4.94</td><td>9.22 15.30</td></tr><tr><td>Qwen3-max Kimi-k2</td><td>3.91 15.77</td><td></td><td>9.6640.56</td><td>4.327.61</td><td>11.01 24.13</td></tr><tr><td colspan="2">Deepseek-v3.2 6.02 13.60| Seed-2-pro 5.27 20.14</td><td></td><td>7.9034.69|2.48 5.08</td><td>12.8943.60|3.597.73</td><td>10.7824.85 7.0613.33</td></tr></table>

Table 13: Detailed Fact Fail and Logic Fail of Reasoning Trace on Standard MPAR-Bench on Thinking Models

## Structured Reasoning Skill Experiment

We designed a structured three-step reasoning skill to test whether explicit multi-step prompting can enhance reasoning breadth beyond what thinking mode alone achieves. The skill instructs the model to: (1) examine all clues comprehensively and identify their underlying associations; (2) prioritize specific concepts over abstract hypernyms; and (3) perform reverse verification by reasoning backward from the predicted answer to the provided clues. We evaluate this skill on Seed-2-pro (thinking mode, standard setting).

On the English subset, accuracy rises marginally from 71.4% to 72.4% (+1.0pp), ANLS from 0.724 to 0.734, and embedding similarity from 0.802 to 0.807. On the Chinese subset, gains are somewhat larger: accuracy from 64.6% to 67.8% (+3.2pp), ANLS from 0.739 to 0.765, and embedding similarity from 0.820 to 0.838. The modest magnitude of improvement— particularly on English—suggests that thinking-mode models may already perform these associative integration steps implicitly during extended reasoning, and that prompt-level interventions alone ofer limited leverage on the core challenge of multisource evidence integration.

<table><tr><td rowspan="3">Model</td><td colspan="3">English</td><td rowspan="2">Chinese</td></tr><tr><td colspan="2">Fact(%)</td><td>Logic(%) Fact(%) | Logic(%)</td></tr><tr><td>√</td><td>X √</td><td>X √</td><td>X 1 X</td></tr><tr><td>GPT-5.2</td><td>5.3024.06|</td><td>9.93 47.03|</td><td></td><td>1.75 3.87|6.34 12.88</td></tr><tr><td>Gemini-3flash Sonnet-4.5</td><td>4.7435.33 5.40 9.94</td><td>7.8947.73 27.30 45.41</td><td>|4.24 6.79 3.847.18</td><td>8.7815.52 9.2414.10</td></tr><tr><td>Qwen3-max</td><td>4.5531.30</td><td>8.5245.65</td><td>2.61 4.49</td><td>8.32 15.28</td></tr><tr><td>Deepseek-v3.2 8.95 29.71</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>15.10 55.97|2</td><td>2.188.83</td><td>8.84 21.83</td></tr><tr><td>Seed-2-pro</td><td>10.03 37.51</td><td></td><td>15.65 52.143.43 8.27</td><td>8.4720.34</td></tr></table>

Table 14: Detailed Fact Fail and Logic Fail of Reasoning Trace on Standard MPAR-Bench on Non-thinking Models

<table><tr><td rowspan="3">Model</td><td colspan="3">English</td><td colspan="2">Chinese</td></tr><tr><td colspan="2">Fact(%)</td><td colspan="2">Logic(%)</td><td>Fact(%) Logic(%)</td></tr><tr><td>X</td><td>√ X</td><td>√</td><td>X</td><td>X</td></tr><tr><td>GPT-5.2</td><td>2.52 6.22 Gemini-3.1pro 4.07 10.71</td><td colspan="2">|10.38 26.85| 9.85 23.51 6.79 16.49 12.9339.21 6.118.35</td><td>2.272.30 3.113.59</td><td>9.9413.31 10.97 12.56 15.73 22.32</td></tr></table>

Table 15: Detailed Fact Fail and Logic Fail of Reasoning Trace on Enhanced MPAR-Bench on Thinking Models

<table><tr><td rowspan="3">Model</td><td colspan="3">English</td><td colspan="2">Chinese</td></tr><tr><td colspan="2">Fact(%)</td><td>Logic(%)</td><td>Fact(%)</td><td>Logic(%)</td></tr><tr><td></td><td>X √</td><td>×</td><td>√ ×</td><td>√ X</td></tr><tr><td>GPT-5.2</td><td>7.1624.19|13.9747.39|</td><td></td><td></td><td>|3.385.73</td><td>10.11 19.29</td></tr><tr><td>Gemini-3flash</td><td></td><td></td><td>7.2233.00|13.6248.06|</td><td>6.38 11.22</td><td>13.60 24.33</td></tr><tr><td>Sonnet-4.5</td><td></td><td></td><td>8.5226.7014.7946.04</td><td>5.9710.44</td><td>15.73 25.45</td></tr><tr><td>Qwen3-max</td><td></td><td></td><td>6.2525.56|13.21 48.91|</td><td>3.32 9.29</td><td>14.47 25.21</td></tr><tr><td>Deepseek-v3.211.79 29.6320.09 56.30</td><td></td><td></td><td></td><td>[4.58 11.50|</td><td>14.4930.75</td></tr><tr><td>Seed-2-pro</td><td>12.9733.9518.03 50.88</td><td></td><td></td><td></td><td>5.3611.3614.08 24.81</td></tr></table>

Table 16: Detailed Fact Fail and Logic Fail of Reasoning Trace on Enhanced MPAR-Bench on Non-thinking Models