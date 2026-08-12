# EVIL-Detect for NLPCC 2026 Shared Task 6: LLM-Generated Text Detection

Hongrui Bao<sup>1,2</sup>, Hangyu Rong<sup>2</sup>, Zhuoshang Wang<sup>1,2</sup>, Yubing Ren<sup>1,2⋆</sup>, and Yanan Cao<sup>1,2</sup>

<sup>1</sup> Institute of Information Engineering, Chinese Academy of Sciences, Beijing, China {wangzhuoshang,renyubing,caoyanan}@iie.ac.cn

<sup>2</sup> School of Cyber Security, University of Chinese Academy of Sciences, Beijing, China {baohongrui26,ronghangyu23}@mails.ucas.ac.cn

Abstract. The rapid development of large language models (LLMs) has increased the need for reliable detection of LLM-generated text, especially in realistic Chinese scenarios involving human-written text (HWT), LLM-generated text (LGT), and LLM-refined text (HLT). This paper presents EVIL-Detect, a multi-signal ensemble framework with conflictaware fusion for Natural Language Processing and Chinese Computing (NLPCC) 2026 Shared Task 6. The system integrates edit-extent regression, zero-shot likelihood-contrast signals, lexical statistics, and conservative text rules. With calibrated decision boundaries and conflictaware integration, our system improves robustness under strong out-ofdistribution shifts, achieving a macro-F1 score of 0.8888 and ranking first in the oficial evaluation. Our code is available at https://github.com/ bbbbhrrrr/evildetect.

Keywords: Large Language Models · Text Detection · Chinese Text · Three-Class Classification · Ensemble Learning · Conflict Resolution.

## 1 Introduction

Large language models (LLMs) such as GPT-4 [1], Qwen2.5 [2], and Qwen3 [19] can produce fluent, coherent, and instruction-following text. Their wide use also increases risks such as disinformation, academic misconduct, and manipulation of online content, making reliable LLM-generated text detection important for responsible natural language processing (NLP) [3].

Most existing work focuses on English binary detection. Supervised detectors, including early GPT-style detectors [4], RADAR [5], and DeTeCtive [6], finetune classifiers on labeled data and often perform well in matched settings, but their robustness drops under cross-domain or cross-generator shifts. Trainingfree methods, including DetectGPT [7], Fast-DetectGPT [8], DNA-GPT [9], and Binoculars [10], use likelihood-, perturbation-, n-gram-, or perplexity-based signals, but remain sensitive to model choice and thresholds.

Chinese detection is less explored and cannot be handled by direct transfer from English because of diferences in tokenization, lexical granularity, and writing conventions. NLPCC 2025 Shared Task 1 [11] promoted Chinese binary detection, while NLPCC 2026 Shared Task 6 [14] extends the task to three classes: human-written text (HWT), LLM-generated text (LGT), and LLMrefined text (HLT). This setting reflects real artificial intelligence (AI)-assisted writing, where users often use LLMs to rewrite or polish human-written text, and is closely related to CUDRT [12], DetectRL-X [13], and the oficial NLPCC 2026 benchmark [14].

Motivated by this strong out-of-distribution (OOD) setting, we propose EVIL-Detect, short for Edit-aware View-Integrated Learning for Detection. It combines EditLens-inspired edit-extent regression [15], zero-shot likelihood-contrast scoring, lexical statistics, and conservative rule-based corrections. Calibrated decision boundaries and conflict-aware integration are used to fuse complementary signals and resolve boundary cases among HWT, LGT, and HLT.

Our contributions are threefold:

1. We analyze the performance of several representative detection methods in the Chinese three-class setting of HWT, LGT, and HLT.

2. We propose EVIL-Detect, a conflict-aware ensemble framework that integrates edit-extent regression, zero-shot likelihood evidence, lexical statistics, and conservative rule-based corrections.

3. Our system achieved a macro-F1 score of 0.8888 and ranked first in NLPCC 2026 Shared Task 6.

## 2 Related Work

## 2.1 Training-based Supervised Detection

Training-based detectors formulate LLM-generated text detection as supervised classification. Early supervised detectors were developed for GPT-style outputs [4], and later methods such as RADAR [5] and DeTeCtive [6] improved robustness through adversarial learning and contrastive learning. These methods can be strong when training and test data are similar, but they are vulnerable to shifts in domain, generator, prompt style, and text length [3].

## 2.2 Training-free, Statistical, and Representation-based Detection

Training-free methods avoid task-specific training and instead rely on languagemodel statistics. DetectGPT [7], Fast-DetectGPT [8], DNA-GPT [9], Binoculars [10], and EchoPrompt [25] exploit probability curvature, divergent n-grams, cross-model perplexity, or latent prompt restoration for zero-shot detection. For related fine-grained and robust detection settings, AI-editing-extent regression and representation-level features are also useful: EditLens [15] models the degree of AI editing, while RepreGuard [16] detects generated text from hidden representation patterns.

## 2.3 Hybrid Ensemble Detection and Chinese Shared Tasks

Hybrid and ensemble methods are important because diferent detectors fail on diferent samples. EnsemJudge [17], the winning system of NLPCC 2025 Shared Task 1, combines multiple models and voting strategies for Chinese LLMgenerated text detection. CUDRT [12], DetectRL [18], DetectRL-X [13], and NLPCC 2026 Shared Task 6 [14] further emphasize realistic, OOD, and AIedited scenarios. These studies motivate our use of heterogeneous signals and conflict resolution for robust three-class detection.

## 3 Method

In this section, we present EVIL-Detect, our detection system for NLPCC 2026 Shared Task 6. We first describe the observations that motivate the design, then detail the individual modules, and finally present the fusion strategy that combines them into a final prediction.

## 3.1 Observation and Motivation

Most existing methods for LLM-generated text detection are designed for binary classification, distinguishing human-written text from LLM-generated text. However, NLPCC 2026 Shared Task 6 requires a more fine-grained three-way classification among HWT, LGT, and HLT. A direct adaptation to three-way classification is not suficiently robust under the distribution shift between training and evaluation data, as later shown by the representative alternative designs in Sec. 4.3. For example, a direct generative quantized low-rank adaptation (QLoRA) [23] classifier achieves only 0.1690 macro-F1 on testp1.

This motivates us to reconsider the label structure of the task. HWT and LGT can be viewed as two endpoints, whereas HLT is an intermediate state because LLM refinement preserves human content while introducing machinerelated traces. We therefore do not require every component to solve the full three-way task independently: LGT-oriented signals can be reused by grouping HWT and HLT as non-LGT, while HWT/HLT discrimination relies more on editing strength. Since diferent methods show diferent class preferences, as quantified in Sec. 4.3, EVIL-Detect combines a supervised edit-strength module, zero-shot LGT evidence, lexical statistics, and conflict-aware fusion rather than uniform averaging.

## 3.2 Overview of EVIL-Detect

We propose EVIL-Detect, a multi-signal system for Chinese LLM-generated and LLM-refined text detection. As shown in Fig. 1, EVIL-Detect combines four types of evidence: supervised edit-extent signals from EditLens [15] and Soft-EditLens, zero-shot LGT-tendency signals from EchoPrompt, lexical frequency statistics, and conservative text rules. The fusion module integrates these heterogeneous signals with conflict-aware decision logic and outputs the final HWT, LGT, or HLT prediction.

![](images/f2835cd27c67cdead1b24fcc9d8d740b71b96fa19cf92c203fedc8bca9562b49.jpg)  
Fig. 1. Overall architecture of EVIL-Detect.

## 3.3 Supervised Training Module

The supervised training module is inspired by EditLens [15], which formulates AI-edited text detection as continuous edit-extent estimation rather than pure discrete classification. In the original formulation, a similarity metric between a human-written source and its AI-edited version is used as intermediate supervision, and a regression model is trained to predict the amount of AI editing from the text alone.

In our setting, each training group contains a triplet $( h , g , t )$ , where h, g, and t denote the HWT, LGT, and HLT versions respectively. We define a continuous editing-extent target on the HWT–HLT–LGT axis by using HWT and LGT as two anchors:

$$
r ( h ) = 0 , \quad r ( g ) = 1 , \quad r ( t ) = d ( h , t ) .\tag{1}
$$

Here $d ( h , t ) \in [ 0 , 1 ]$ measures how far the LLM-refined text t moves away from the corresponding human-written source h. This formulation treats HLT as an instance-dependent soft editing state rather than an independent hard class. At inference time, the paired HWT text is not required; the trained model predicts $r ( x )$ from the input text x alone. Based on this common formulation, we instantiate the supervised module with two variants.

EditLens The EditLens variant uses a surface-level soft character n-gram distance to construct $\scriptstyle d ( h , t )$ . Let $C _ { n } ( x )$ be the character n-gram count vector of text x. For a set of n-gram orders N and non-negative weights $w _ { n }$ , we compute

$$
d _ { \mathrm { n g } } ( h , t ) = 1 - \frac { \sum _ { n \in \mathcal { N } } w _ { n } \cos ( C _ { n } ( h ) , C _ { n } ( t ) ) } { \sum _ { n \in \mathcal { N } } w _ { n } } .\tag{2}
$$

Diferent n-gram orders and weights produce diferent HLT target distributions. We therefore perform a validation-set sweep over candidate n-gram ranges and weighting schemes, ranking each configuration by how well the resulting HLT scores are separated from the two anchors 0 (HWT) and 1 (LGT). The bestranked configuration has internal sweep id 044, uses orders 1–7 with linearly increasing weights, and is denoted as rank044 in the rest of the paper. The resulting id/text/score instances are used to train a low-rank adaptation (LoRA) [22]- adapted decoder-only LLM with a regression objective. The trained model outputs a continuous editing-extent score, which is later discretized and used as a base signal by the fusion module.

Soft-EditLens Soft-EditLens replaces surface n-gram overlap with semantic phrase-level soft matching. We first segment HWT and HLT texts into short token phrases. For each phrase $q$ in the HLT text, we encode it and find the most similar phrase $p$ in the corresponding HWT text. With phrase count $c _ { t } ( \boldsymbol { q } )$ and embedding function $e ( \cdot )$ , the phrase-level distance is

$$
d _ { \mathrm { p h } } ( h , t ) = 1 - \frac { \sum _ { q \in P ( t ) } c _ { t } ( q ) \operatorname* { m a x } _ { p \in P ( h ) } \cos ( e ( q ) , e ( p ) ) } { \sum _ { q \in P ( t ) } c _ { t } ( q ) } .\tag{3}
$$

Compared with character n-gram overlap, this target is less sensitive to lexical replacement and better captures semantically preserved expressions. Soft-EditLens also changes the learning objective: besides direct regression on the continuous score, we train an ordinal bucket predictor that discretizes the editing-extent axis into ordered intervals. At inference time, the regression model provides a smooth score, while the bucket model provides an interval-level estimate. These two outputs are used as complementary signals in the fusion module, especially for resolving HWT/HLT boundary cases.

## 3.4 Zero-Shot Detection Module

EchoPrompt EchoPrompt [25] is a training-free prompting module that provides auxiliary evidence for LGT detection. The module is based on the intuition that fully LLM-generated text is more compatible with an assistant-style generation distribution, whereas human-written text is less likely to follow such a distribution. For each input text, EchoPrompt computes a normalized likelihoodcontrast score using a prompted instruction-tuned model and its corresponding base model; a higher score indicates stronger LGT tendency. To reduce the dependence on a single prompt $_ { \mathrm { o r } }$ model, we use multiple fixed prompt/model branches, whose outputs are passed to the fusion module as additional discriminative signals rather than standalone predictions.

## 3.5 Lexical Frequency Statistics Module

The lexical frequency statistics module provides a lightweight surface-level view complementary to neural and prompting-based signals. It is based on the observation that some class preferences are reflected in recurring lexical or character

n-gram patterns. We build label-wise frequency lexicons from the training set and retain discriminative n-grams after removing patterns that appear similarly across classes.

For an input text $x ,$ let $G ( x )$ denote its extracted n-grams. We estimate smoothed label-conditional probabilities $p ( g \mid y )$ for each lexical unit $g$ and label $y ,$ and aggregate their preferences through averaged log-odds scores:

$$
s _ { a / b } ( x ) = \frac { 1 } { | G ( x ) | } \sum _ { g \in G ( x ) } \log \frac { p ( g \mid a ) } { p ( g \mid b ) } .\tag{4}
$$

In practice, we use several derived statistics, including LGT-vs-HWT tendency, AI-vs-HWT tendency where LGT and HLT are merged as AI-assisted text, the fraction of machine-oriented lexical units, and the fraction of HWT-oriented lexical units. These statistics are passed to the fusion module as auxiliary lexical evidence rather than used as standalone predictions.

## 3.6 Fusion and Decision Module

The fusion module integrates the outputs of the preceding components into the final HWT/LGT/HLT prediction. The starting point is the EditLens editingextent score. We use two calibrated boundary settings $( \tau _ { 1 } ^ { ( k ) } , \tau _ { 2 } ^ { ( k ) } ,$ ), $\mathrm { ~ \ i ~ { ~ \in ~ \{ ~ 1 , ~ 2 ~ \} ~ } ~ }$ , and convert the score into a base label for each setting:

$$
y ^ { ( k ) } ( x ) = \left\{ \begin{array} { l l } { \mathrm { H W T } , } & { s _ { E } ( x ) \leq \tau _ { 1 } ^ { ( k ) } , } \\ { \mathrm { H L T } , } & { \tau _ { 1 } ^ { ( k ) } < s _ { E } ( x ) < \tau _ { 2 } ^ { ( k ) } , \quad k \in \{ 1 , 2 \} . } \\ { \mathrm { L G T } , } & { s _ { E } ( x ) \geq \tau _ { 2 } ^ { ( k ) } , } \end{array} \right.\tag{5}
$$

Here $s _ { E } ( x )$ denotes the EditLens score, and $y ^ { ( 1 ) }$ and $y ^ { ( 2 ) }$ are the two resulting base labels. These base labels are combined with Soft-EditLens regression/bucket signals and a panel of nine binary LGT-support votes from zero-shot, edit-based, and lexical views (Table 3).

Conflict-aware integration If $y ^ { ( 1 ) }$ and $y ^ { ( 2 ) }$ agree, EVIL-Detect directly keeps the agreed label. If they disagree, the fusion module resolves the conflict according to the label pair involved:

$$
y _ { \mathrm { b a s e } } = \left\{ \begin{array} { l l } { y ^ { ( 1 ) } , } & { y ^ { ( 1 ) } = y ^ { ( 2 ) } , } \\ { R ( y ^ { ( 1 ) } , y ^ { ( 2 ) } , z _ { \mathrm { s o f t } } , V _ { \mathrm { L G T } } ) , } & { y ^ { ( 1 ) } \neq y ^ { ( 2 ) } , } \end{array} \right.\tag{6}
$$

where ${ \mathit { z } } _ { \mathrm { s o f t } }$ represents the Soft-EditLens regression and bucket evidence, $V _ { \mathrm { L G T } }$ denotes the aggregated LGT-support votes, and R(·) $R ( \cdot )$ is the conflict-aware resolver.

We instantiate $R ( \cdot )$ with validation-calibrated thresholds. Let $z _ { r } ( x )$ be the Soft-EditLens regression score, $z _ { b } ( x )$ be its ordinal bucket index, and $v ( x ) =$ $\scriptstyle \sum _ { i = 1 } ^ { 9 } v _ { i } ( x )$ be the number of active LGT-support votes. For HWT/LGT conflicts, R outputs HWT when $z _ { r } ( x ) \leq \gamma _ { H }$ , otherwise outputs LGT when $v ( x ) \geq$ $\gamma _ { H L }$ , and otherwise falls back to HLT. For HWT/HLT conflicts, R outputs HLT if $z _ { b } ( x ) \ge \beta _ { T } ;$ otherwise it outputs HWT. For HLT/LGT conflicts, R outputs LGT if $v ( x ) \geq \gamma _ { T L } ;$ otherwise it keeps HLT. Here $\gamma _ { H }$ is the HWTside Soft-EditLens boundary, $\beta _ { T }$ is the rewriting-strength bucket boundary, and $\gamma _ { H L } , \gamma _ { T L }$ are pair-specific LGT-vote thresholds. This rule design matches the role of each module: EditLens gives the initial continuum label, Soft-EditLens refines HWT/HLT boundaries, and the vote panel provides evidence for fully generated text.

High-precision text rules After conflict-aware integration, we apply a small set of conservative text rules as final corrections. The complete rule set is as follows. First, raw HyperText Markup Language (HTML)-like or Extensible Markup Language (XML)-like structural markup, such as <!DOCTYPE html>, <html>, <div>, <p>, and <a href=...>, is treated as strong evidence for LGT; representative training examples are shown in Appendix A. Second, script or rendering residues, such as document.write $( " < \mathtt { b r } / > " )$ or unfinished tag sequences, also trigger an LGT correction. Third, explicit rewriting or polishing traces, such as answer wrappers indicating that a rewritten or polished version follows, support HLT when the fused label is not already LGT. These rules are applied only after fusion and serve as high-precision safeguards.

## 4 Experiments

In this section, we evaluate EVIL-Detect on NLPCC 2026 Shared Task 6.

## 4.1 Experimental Setup

Dataset We use the oficial data released for NLPCC 2026 Shared Task 6, which requires three-way classification among human-written text (HWT), LLMgenerated text (LGT), and LLM-refined text (HLT). According to the oficial task description [14], the training set is sampled and adapted from the CUDRT dataset [12] and covers four generators, including GPT-4 [1], Qwen-family generators, ChatGLM [20], and Baichuan [21], across two domains, namely news and academic writing. In contrast, the two hidden test phases, testp1 and testp2, are drawn from the Chinese split of DetectRL-X [13] and are constructed as outof-distribution data that may involve unseen domains, unseen generators, and diferent generation schemes, so as to stress-test detector robustness. We remove empty or malformed training samples and clean task-irrelevant artifacts such as role markers, abnormal control characters, redundant whitespace, and obvious generation residues. Formatting cues such as Markdown or HTML fragments are handled conservatively to avoid removing genuine stylistic signals. This preprocessing is applied only to the training data. Detailed data statistics are shown in Table 1.

Table 1. Dataset statistics.
<table><tr><td>Split</td><td>HWT LGT</td><td>HLT</td><td>Total</td></tr><tr><td>Cleaned training set 19,634 18,368 19,587 57,589</td><td></td><td></td><td></td></tr><tr><td>testp1 (official)</td><td>1,200 1,200</td><td>1,200</td><td>3,600</td></tr><tr><td>testp2 (official)</td><td>384 384</td><td>384</td><td>1,152</td></tr></table>

Table 2. Module configuration of EVIL-Detect.
<table><tr><td>Module</td><td>Backbone</td><td>Role</td><td>Type</td></tr><tr><td>EditLens</td><td>Qwen3.5-4B-Base + LoRA editing-extent label</td><td></td><td>supervised</td></tr><tr><td></td><td>Soft-EditLens Qwen3.5-9B-Base + LoRA regression/bucket scores supervised</td><td></td><td></td></tr><tr><td></td><td>EchoPrompt Qwen3.5-4B, Qwen2.5-1.5B LGT vote panel</td><td></td><td>training-free</td></tr><tr><td></td><td>Lexical freq. label-wise n-gram lexicon LGT vote panel</td><td></td><td>training-free</td></tr><tr><td>Fusion</td><td></td><td>evidence integration</td><td>deterministic</td></tr></table>

Metric. The oficial metric is macro-F1 over the three classes. We additionally report accuracy and per-class F1.

Implementation. Table 2 summarises the modules, and Table 3 details the vote panel. EditLens uses Qwen3.5-4B-Base [24] as the backbone with LoRA [22] (rank 16, α 32, dropout 0.05) on all attention and MLP projections, together with a linear regression head for editing-extent prediction. Its target adopts the rank044 soft n-gram configuration selected by the separability sweep. Soft-EditLens uses Qwen3.5-9B-Base [24] with the same LoRA [22] configuration and a dual regression/bucket objective. The zero-shot module is training-free and uses Qwen3.5-4B [24] and Qwen2.5-1.5B [2] instruct/base model pairs. The lexical-frequency module builds label-wise word and character n-gram log-odds lexicons. For fusion, EditLens scores are discretized with two calibrated classboundary settings and combined with Soft-EditLens auxiliary signals and nine binary LGT-support votes through conflict-aware integration, followed by highprecision text rules. All supervised models are trained on NVIDIA V100 graphics processing units (GPUs).

## 4.2 Main Results

Table 4 reports the performance of EVIL-Detect on the two oficial test phases. The system achieves macro-F1 scores of 0.8913 on testp1 and 0.8888 on testp2. The small gap between the two phases indicates that the multi-signal design remains stable across the two evaluation splits.

Among the three classes, LGT obtains the highest F1 scores, reaching 0.9267 on testp1 and 0.9219 on testp2. HLT is consistently the most challenging class, with F1 scores of 0.8391 and 0.8407, which is consistent with its intermediate nature between HWT and LGT. This pattern supports the need for conflictaware fusion, where HWT/HLT and HLT/LGT disagreements are handled with diferent auxiliary signals.

Table 3. Composition of the binary LGT-support vote panel used by the fusion module.
<table><tr><td colspan="2"># Source view LGT-support signal</td></tr><tr><td></td><td>1 EchoPrompt likelihood-contrast vote (Qwen2.5-1.5B, prefix 1)</td></tr><tr><td></td><td>2 EchoPrompt likelihood-contrast vote (Qwen3.5-4B, prefix 1)</td></tr><tr><td></td><td>3 EchoPrompt likelihood-contrast vote (Qwen3.5-4B, prefix 2)</td></tr><tr><td>4 EditLens</td><td>primary editing-extent LGT vote</td></tr><tr><td>5 EditLens</td><td>auxiliary editing-extent LGT vote</td></tr><tr><td>6 Lexical</td><td>low HWT-oriented lexical evidence</td></tr><tr><td>7 Lexical</td><td>AI-vs-HWT lexical log-odds</td></tr><tr><td>8 Lexical</td><td>LGT-vs-HWT lexical log-odds</td></tr><tr><td>9 Lexical</td><td>machine-oriented lexical fraction</td></tr></table>

Table 4. Main results of EVIL-Detect on the two oficial test phases of NLPCC 2026 Task 6.
<table><tr><td></td><td></td><td></td><td>Phase #Samples Macro-F1 Accuracy HWT-F1 LGT-F1 HLT-F1</td><td></td><td></td><td></td></tr><tr><td>testp1</td><td>3,600</td><td>0.8913</td><td>0.8911</td><td>0.9083</td><td>0.9267</td><td>0.8391</td></tr><tr><td>testp2</td><td>1,152</td><td>0.8888</td><td>0.8880</td><td>0.9039</td><td>0.9219</td><td>0.8407</td></tr></table>

## 4.3 Component and Variant Analysis

We evaluate each module in isolation on testp1, together with the main variants explored during development (Table 5). EditLens [15] is the strongest standalone component: the sweep-selected rank044 configuration reaches 0.8494 macro-F1, outperforming the initial soft-n-gram configuration (0.8289). This confirms that the validation-based target sweep improves the separability of the HWT–HLT– LGT continuum.

The other modules are weaker as standalone three-class predictors, but they provide useful auxiliary evidence for fusion. The best zero-shot strategy reaches 0.6325 macro-F1, while the binary HWT/AI strategy reaches 0.4484, indicating that EchoPrompt is better suited for producing LGT-support votes. Lexical statistics show a similar pattern: the best lexical classifier reaches 0.5535 macro-F1, so lexical features are also used as calibrated votes rather than final labels. Soft-EditLens is not listed because its regression and bucket outputs are consumed by the fusion module. With conflict-aware fusion, EVIL-Detect reaches 0.8913 macro-F1, improving over EditLens by 4.19 points and raising HLT and LGT F1 to 0.8391 and 0.9267, respectively.

Table 5. Component analysis on testp1. Standalone scores are diagnostic because some methods are used as auxiliary evidence sources in the final fusion.
<table><tr><td>Method</td><td>Macro-F1</td><td>HWT LGT HLT</td></tr><tr><td>EditLens (rank044)</td><td>0.8494</td><td>0.9067 0.8651 0.7765</td></tr><tr><td>EditLens (initial soft-n-gram)</td><td>0.8289</td><td>0.8325 0.9042 0.7500</td></tr><tr><td>Zero-shot detection (best strategy)</td><td>0.6325</td><td>0.64250.7842 0.4708</td></tr><tr><td>Zero-shot detection (binary HWT/AI)</td><td>0.4484</td><td>0.6294 0.7157 0.0000</td></tr><tr><td>Lexical statistics (best classifier)</td><td>0.5535</td><td>0.6591 0.5040 0.4975</td></tr><tr><td>EVIL-Detect (full fusion)</td><td>0.8913</td><td>0.9083 0.9267 0.8391</td></tr></table>

Table 6. Representative alternative designs evaluated on testp1.
<table><tr><td>Alternative design</td><td>Macro-F1 HWT</td><td></td><td>LGT HLT</td></tr><tr><td>Binoculars</td><td>0.3928</td><td>0.4738 0.5552 0.1493</td><td></td></tr><tr><td>Generative SFT (GLM-4-9B)</td><td>0.1896</td><td>0.5082 0.0148 0.0458</td><td></td></tr><tr><td>Generative SFT (Qwen3.5-9B)</td><td>0.1690</td><td>0.0000 0.0066 0.5004</td><td></td></tr></table>

Representative alternative designs. Table 6 reports representative alternatives explored during development but not included in the final EVIL-Detect system. Direct generative supervised fine-tuning (SFT) classifiers are not robust under the evaluation distribution: the GLM-4-9B [20] and Qwen3.5-9B [24] variants reach only 0.1896 and 0.1690 macro-F1, showing strong class bias. Binoculars [10] also performs poorly as a standalone three-way predictor, with 0.3928 macro-F1. These results support our design choice of using editing-extent modeling as the base signal and treating likelihood-based detectors as auxiliary evidence.

## 4.4 Ablation Study

We conduct an ablation of the fusion module on the released testp2 labels, as shown in Table 7. The EditLens-only baseline [15] obtains 0.8411 macro-F1. Adding conflict-aware integration and high-precision text rules improves the score to 0.8816, and the full system reaches 0.8888 after combining two calibrated EditLens decisions. The improvement mainly comes from boundary cases: compared with EditLens only, the full system reduces LGT→HLT errors from 59 to 30 and HLT→LGT errors from 58 to 28, showing that the fusion module helps correct confusion around HLT.

Table 7. Ablation results of the fusion module on testp2.
<table><tr><td>Configuration</td><td>Macro-F1 HWT</td><td>LGT</td><td>HLT</td></tr><tr><td>Full EVIL-Detect</td><td>0.8888</td><td>0.9039 0.9219 0.8407</td><td></td></tr><tr><td>EVIL-Detect (single-threshold EditLens)</td><td>0.8816</td><td>0.9153 0.9016 0.8280</td><td></td></tr><tr><td>EditLens only</td><td>0.8411</td><td>0.9141 0.8453 0.7640</td><td></td></tr></table>

## 5 Conclusion

This paper presents EVIL-Detect for Chinese LLM-generated and LLM-refined text detection in NLPCC 2026 Shared Task 6. It integrates edit-extent regression, zero-shot LGT evidence, lexical statistics, and conservative text rules through conflict-aware fusion. With calibrated decision boundaries, our system achieved a macro-F1 score of 0.8888 and ranked first in the oficial evaluation. The results suggest that robust three-class detection cannot rely on a single signal; instead, edit-strength modeling, likelihood-based evidence, lexical cues, and high-precision corrections provide complementary information for distinguishing HWT, LGT, and HLT.

Acknowledgments. This work is supported by the Postdoctoral Fellowship Program of China Postdoctoral Science Foundation (CPSF) under Grant Number GZC20251076.

## References

1. Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., et al.: GPT-4 technical report. arXiv preprint arXiv:2303.08774 (2023)

2. Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., et al.: Qwen2.5 technical report. arXiv preprint arXiv:2412.15115 (2024)

3. Wu, J., Yang, S., Zhan, R., Yuan, Y., Chao, L.S., Wong, D.F.: A survey on LLMgenerated text detection: Necessity, methods, and future directions. Computational Linguistics 51(1), 275–338 (2025)

4. Solaiman, I., Brundage, M., Clark, J., Askell, A., Herbert-Voss, A., Wu, J., et al.: Release strategies and the social impacts of language models. arXiv preprint arXiv:1908.09203 (2019)

5. Hu, X., Chen, P.-Y., Ho, T.-Y.: RADAR: Robust AI-text detection via adversarial learning. Advances in Neural Information Processing Systems 36, 15077–15095 (2023)

6. Guo, X., Zhang, S., He, Y., Zhang, T., Feng, W., Huang, H., Ma, C.: DeTeCtive: Detecting AI-generated text via multi-level contrastive learning. Advances in Neural Information Processing Systems 37, 88320–88347 (2024)

7. Mitchell, E., Lee, Y., Khazatsky, A., Manning, C.D., Finn, C.: DetectGPT: Zeroshot machine-generated text detection using probability curvature. In: Proceedings of the 40th International Conference on Machine Learning, pp. 24950–24962. PMLR (2023)

8. Bao, G., Zhao, Y., Teng, Z., Yang, L., Zhang, Y.: Fast-DetectGPT: Eficient zeroshot detection of machine-generated text via conditional probability curvature. In: The Twelfth International Conference on Learning Representations (2024)

9. Yang, X., Cheng, W., Wu, Y., Petzold, L., Wang, W.Y., Chen, H.: DNA-GPT: Divergent n-gram analysis for training-free detection of GPT-generated text. In: The Twelfth International Conference on Learning Representations (2024)

10. Hans, A., Schwarzschild, A., Cherepanova, V., Kazemi, H., Saha, A., Goldblum, M., Geiping, J., Goldstein, T.: Spotting LLMs with binoculars: Zero-shot detection of machine-generated text. In: Proceedings of the 41st International Conference on Machine Learning, pp. 17519–17537. PMLR (2024)

11. Wu, J., Zhan, R., Wang, Q., Yuan, Y., Chao, L.S., Wong, D.F.: Overview of the NLPCC 2025 Shared Task 1: LLM-generated text detection. In: CCF International Conference on Natural Language Processing and Chinese Computing, pp. 263–274. Springer (2025)

12. Tao, Z., Chen, Y., Xi, D., Li, Z., Xu, W.: Toward reliable detection of LLMgenerated texts: A comprehensive evaluation framework with CUDRT. ACM Transactions on Intelligent Systems and Technology 17(2), 1–35 (2026)

13. Wu, J., Liu, Y., Zhu, C., Zhang, H., Wu, Z., Shi, T., Du, Y., Wang, L., Luo, W., Su, J., Wong, D.F.: DetectRL-X: Towards reliable multilingual and real-world LLM-generated text detection. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 38247– 38294. Association for Computational Linguistics (2026)

14. NLP2CT Lab: NLPCC 2026 Shared Task 6: The Second Shared Task on LLM-Generated Text Detection. https://github.com/NLP2CT/ NLPCC-2026-Task6-Detection (2026). Accessed 2 July 2026

15. Thai, K., Emi, B., Masrour, E., Iyyer, M.: EditLens: Quantifying the extent of AI editing in text. arXiv preprint arXiv:2510.03154 (2025)

16. Chen, X., Wu, J., Yang, S., Zhan, R., Wu, Z., Luo, Z., Wang, D., Yang, M., Chao, L.S., Wong, D.F.: RepreGuard: Detecting LLM-generated text by revealing hidden representation patterns. Transactions of the Association for Computational Linguistics 13, 1812–1831 (2025)

17. Wang, Z., Ren, Y., Zhao, G., Zhu, X., Li, H., Cao, Y.: EnsemJudge: Enhancing reliability in Chinese LLM-generated text detection through diverse model ensembles. In: CCF International Conference on Natural Language Processing and Chinese Computing, Singapore, pp. 284–295. Springer (2025)

18. Wu, J., Zhan, R., Wong, D.F., Yang, S., Yang, X., Yuan, Y., Chao, L.S.: DetectRL: Benchmarking LLM-generated text detection in real-world scenarios. Advances in Neural Information Processing Systems 37, 100369–100401 (2024)

19. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

20. Team GLM, Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Zhang, D., et al.: ChatGLM: A family of large language models from GLM-130B to GLM-4 all tools. arXiv preprint arXiv:2406.12793 (2024)

21. Yang, A., Xiao, B., Wang, B., Zhang, B., Bian, C., Yin, C., Lv, C., et al.: Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305 (2023)

22. Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022)

23. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L.: QLoRA: Eficient finetuning of quantized LLMs. Advances in Neural Information Processing Systems 36, 10088–10115 (2023)

24. Qwen Team: Qwen3.5 model cards. https://huggingface.co/Qwen (2026). Accessed 10 July 2026

25. Bao, H., Ren, Y., Cao, Y., You, J., Fang, F., Wang, S.: Once a response, always a response: Detecting LLM-generated text via latent prompt restoration. arXiv preprint arXiv:2608.05741 (2026)

## A HTML-like LGT Examples

We observed LGT training samples with raw markup or rendering residues. Representative short snippets include:

<!DOCTYPE html>

– <html> and <div>

– <a href=...>

– document.write("<br/>")

These patterns appear in LGT training items generated by Baichuan, ChatGLM, and GPT-4. We show only structural snippets because the original samples are long Chinese documents.