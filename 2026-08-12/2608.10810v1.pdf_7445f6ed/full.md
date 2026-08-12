# Surfacing the Unsaid: CUE-Bench for Affective Stance in Chinese Discourse

Zhenyan Zheng\*, Yunyao Zhang\*, Junxi Sheng, Junqing Yu, Zikai Song<sup>†</sup> Huazhong University of Science and Technology {ZhenyanZheng, ikostar, skyesong}@hust.edu.cn

## Abstract

Emotion understanding in discourse requires reasoning beyond surface sentiment, since speakers often convey affect through indirect, implicit, polite, ironic, or deliberately mismatched expressions. Existing emotion benchmarks mainly annotate surface polarity or final emotion categories, while lacking a structured account of how explicit expression, implicit affect, pragmatic intent, and fine-grained emotion interact. This limitation makes current evaluations insensitive to cases where affective meaning is concealed, weakened, inverted, or pragmatically reshaped, thereby obscuring models’ failures in deeper emotion understanding. To address this gap, we introduce CUE-Bench, a Chinese Unsaid Emotion benchmark that centers on Affective Stance and covers diverse communicative scenarios. CUE-Bench constructs nine human-interpretable affective stances from Explicit-Implicit polarity interaction and further provides intent and fine-grained emotion annotations for structured affective inference. Experiments show that incorporating Affective Stance improves fine-grained emotion recognition by 3.5 percentage points and pragmatic intent detection by 7.8 percentage points over strong baselines.

## 1 Introduction

Understanding emotion in language is a central problem for affective computing and humancentered NLP (Picard, 1997; Song et al., 2026). However, real discourse often requires more than recognizing surface sentiment or predicting an isolated emotion category (Zhang et al., 2026b,c). It requires modeling how explicit expression and implicit affective tendency jointly shape the speaker’s communicative orientation, which we define as Affective Stance (Du Bois, 2007). For example, positive wording may imply criticism, neutral wording may conceal commitment, and negative wording may signal affiliation or support. Such explicitimplicit mismatches are common in Chinese discourse, where affect is shaped by politeness, suppression, irony, understatement, and other indirect strategies (Brown and Levinson, 1987; Grice, 1975; Gross, 1998). Without modeling this explicitimplicit relation, existing evaluations may capture either explicit expression or implicit affect in isolation, but miss how their mismatch shapes the speaker’s affective stance.

![](images/b042bd7f4c97a924ae390c813bb4fee7ec2fb895116d5572b9c8da51638ae407.jpg)  
Figure 1: Overview of CUE-Bench. The benchmark links what is said and what is meant through Affective Stance, enabling structured affective inference in Chinese discourse.

Existing emotion benchmarks have advanced affective understanding from three perspectives. Representative resources include implicit emotion recognition benchmarks such as IEST (Klinger et al., 2018), SMP2020-EWECT (Xianwei et al., 2021), and ResEmo (Hu et al., 2024), intent understanding benchmarks such as DailyDialog (Li et al.,

2017), MELD (Poria et al., 2019), and CPED (Chen et al., 2022), and fine-grained emotion benchmarks such as GoEmotions (Demszky et al., 2020) and CMMA (Zhang et al., 2023). However, most of them either focus on explicit affect or implicit affect separately, and their evaluation settings are usually restricted to a single task or communicative domain. Consequently, they provide limited diagnostic power for assessing whether models can infer affective stance, pragmatic intent, and finegrained emotion when expressed affect and unsaid affect are misaligned.

To address this gap, we introduce CUE-Bench, a Chinese Unsaid Emotion benchmark for affective understanding. To operationalize affective stance, we propose the Explicit-Implicit Stance Matrix, a structured framework that explicitly models the interaction between explicit expression and implicit affective tendency. By connecting what is expressed with what remains unsaid, this framework provides a unified perspective for analyzing affective stance, pragmatic intent, and fine-grained emotion. Built on this framework, CUE-Bench contains 51,823 annotated instances with four levels of supervision: explicit and implicit affective layers, nine Affective Stances, eight pragmatic intents, and twenty-five fine-grained emotions. It covers diverse Chinese discourse scenarios beyond dialogue, including open-domain conversation, social media comments, sarcasm-oriented text, customerservice interactions, and question-answering content (Zhang et al., 2025).

CUE-Bench supports Affective Stance Recognition, Pragmatic Intent Understanding, and Finegrained Emotion Classification, spanning explicitimplicit stance interaction, pragmatic motivation, and final emotion interpretation. It is built via a hybrid pipeline of dual-model agreement, human adjudication, and bias-controlled LLM adjudication, using LLMs as a constrained aid rather than a replacement for human validation (Gilardi et al., 2023; Zheng et al., 2023).

In summary, our contributions are:

• We introduce CUE-Bench, a Chinese discourse benchmark that provides a unified setting for unsaid emotion understanding through three connected tasks: Affective Stance Recognition, Pragmatic Intent Understanding, and Fine-grained Emotion Classification.

• We propose the Explicit-Implicit Stance Matrix, a structured framework for modeling affective stance, pragmatic intent, and fine-grained emotion.

## 2 Related Work

## Emotion Benchmarks

Prior emotion-related benchmarks cover three related but largely separate lines of work. (1) Implicit emotion recognition. IEST (Klinger et al., 2018) and Chinese resources such as SMP2020- EWECT (Xianwei et al., 2021) and ResEmo (Hu et al., 2024) study emotions that are not directly expressed through emotion words. Yet they mainly formulate implicitness as emotion prediction, rather than as a structured relation between explicit expression and implicit affective tendency. (2) Pragmatic intent understanding. DailyDialog (Li et al., 2017), Diplomat (Li et al., 2023), and PUB (Sravanthi et al., 2024) provide resources for dialogue acts, situated pragmatic reasoning, or pragmatic capability evaluation. However, they do not explicitly model how pragmatic intent interacts with explicit and implicit affect to reshape the affective meaning of an utterance. (3) Finegrained emotion classification. GoEmotions (Demszky et al., 2020) and CMMA (Zhang et al., 2023) offer rich emotion or multi-affection labels, but still largely treat affective states as independent categories. Overall, existing benchmarks either evaluate explicit or implicit affect in isolation, or focus on a single task or communicative domain. They therefore provide limited diagnostic power for jointly assessing affective stance, pragmatic intent, and fine-grained emotion when what is expressed and what remains unsaid diverge.

## Affective Modeling Methods

Existing methods also address these three aspects separately. (1) Implicit emotion recognition. Early methods (Chronopoulou et al., 2018; Devlin et al., 2019; Cui et al., 2021) rely on neural transfer learning, recurrent encoders, attention mechanisms, and pre-trained language models to infer emotions that are not explicitly expressed (Li et al., 2026c; Chen et al., 2026; Li et al., 2026d); however, they usually predict implicit affect directly without modeling its relation to explicit expression. (2) Pragmatic intent understanding. Intent-oriented methods (Chen et al., 2019; Cai et al., 2022; Ghosal et al., 2020; Zhang et al., 2026e,a) model communicative goals through pre-trained encoders, joint intent-slot learning, dialogue context, or commonsense-enhanced conversational reasoning (Li et al., 2026b,f, 2025), but they often treat intent as an independent target rather than explaining how it reshapes affective meaning. (3) Fine-grained emotion classification. Fine-grained emotion methods (Demszky et al., 2020; Zhang et al., 2023; Devlin et al., 2019) typically map text into rich emotion taxonomies with supervised classifiers, contextual encoders, or multi-label Transformer models (Chen et al., 2025; Li et al., 2026e), while leaving the intermediate links among explicit affect, implicit affect, and pragmatic intent underexplored. In contrast, our Explicit-Implicit Stance Matrix provides a structured intermediate framework that connects explicit expression, implicit affective tendency, pragmatic intent, and fine-grained emotion across the three tasks.

<table><tr><td>Benchmark</td><td>Lang.</td><td>Scale</td><td>Multi- domain</td><td>Exp. affect</td><td>Imp. affect</td><td>Stance</td><td></td><td>Intent Fine-grained Multi-task emotion</td><td></td></tr><tr><td>DailyDialog (Li et al., 2017)</td><td>En.</td><td>13.1k</td><td>X</td><td>V</td><td>X</td><td>X</td><td>√</td><td>x</td><td>√</td></tr><tr><td>IEST (Klinger et al., 2018)</td><td>En.</td><td>191.7k</td><td>x</td><td>X</td><td>√</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>MELD (Poria et al., 2019)</td><td>En.</td><td>13.0k</td><td>x</td><td></td><td>x</td><td>X</td><td>X</td><td>X</td><td>X</td></tr><tr><td>GoEmotions (Demszky et al., 2020)</td><td>En.</td><td>58.0k</td><td>X</td><td>√</td><td>x</td><td>X</td><td>X</td><td>√</td><td>X</td></tr><tr><td>SMP2020-EWECT (Xianwei et al., 2021)</td><td>Zh.</td><td>13.6k</td><td>x</td><td></td><td>x</td><td>X</td><td>X</td><td>x</td><td>X</td></tr><tr><td>CPED (Chen et al., 2022)</td><td>Zh.</td><td>12k</td><td>x</td><td>V</td><td>X</td><td>X</td><td>√</td><td>√</td><td>V</td></tr><tr><td>CMMA (Zhang et al., 2023)</td><td>Zh.</td><td>21.8k</td><td>X</td><td></td><td>X</td><td>X</td><td>√</td><td>X</td><td></td></tr><tr><td>ResEmo (Hu et al., 2024)</td><td>Zh.</td><td>72.5k</td><td>x</td><td>√</td><td>X</td><td>X</td><td>x</td><td>√</td><td></td></tr><tr><td>CUE-Bench</td><td>Zh.</td><td>51.8k</td><td>√</td><td></td><td></td><td>L</td><td></td><td>L</td><td></td></tr></table>

Table 1: Comparison with representative emotion-related benchmarks. Scale is reported at the utterance/comment/instance level when available; for CPED and DailyDialog, we report dialogue-level scale following the original paper. Exp. and Imp. affect denote explicit and implicit affective supervision. Stance denotes whether the benchmark provides Explicit-Implicit affective stance labels. Multi-domain indicates coverage of diverse communicative scenarios beyond a single dialogue or social-media domain.

## 3 CUE-Bench

## 3.1 Design Principles

CUE-Bench is designed to evaluate affective understanding beyond surface emotion classification, following three principles:

1. Context-sensitive affective inference. Each instance includes contextual information, since implicit affect often cannot be inferred from an isolated utterance or text span alone.

2. Explicit-Implicit stance modeling. The benchmark emphasizes how explicit expression and implicit affective tendency jointly shape the speaker’s affective stance, covering realistic Chinese discourse phenomena such as politeness, suppression, irony, understatement, teasing, and indirect refusal.

3. Unified multi-task evaluation. The annotation schema supports three connected tasks: Affective Stance Recognition, Pragmatic Intent Understanding, and Fine-grained Emotion Classification, enabling evaluation from stance interaction to intent reasoning and final emotion interpretation.

## 3.2 Data Collection and Instance Format

We construct CUE-Bench from diverse Chinese discourse sources, including open-domain conversations, social media comments, sarcasm-oriented text, customer-service interactions, and questionanswering content. Candidate instances are retained when contextual information plausibly changes affective interpretation, such as when the target text contains explicit affective markers, follows emotionally charged context, involves politeness or irony, or suggests a mismatch between literal wording and implicit affective tendency.

Each instance is represented as:

$$
x _ { i } = ( C _ { i } , u _ { i } ) ,
$$

where $C _ { i }$ denotes the discourse context, and $u _ { i }$ denotes the target utterance or text span.

The annotation target is defined as:

$$
y _ { i } = ( y _ { i } ^ { \mathrm { e x p } } , y _ { i } ^ { \mathrm { i m p } } , y _ { i } ^ { \mathrm { s t a n c e } } , y _ { i } ^ { \mathrm { i n t e n t } } , y _ { i } ^ { \mathrm { e m o t i o n } } ) ,
$$

where $y _ { i } ^ { \mathrm { { e x p } } }$ and $y _ { i } ^ { \mathrm { i m p } }$ denote explicit and implicit affective layers, $y _ { i } ^ { \mathrm { s t a n c e } }$ denotes the Affective Stance label, $y _ { i } ^ { \mathrm { i n t e n t } }$ denotes pragmatic intent, and y<sup>emotion</sup> denotes fine-grained emotion.

Before annotation, we normalize whitespace, remove duplicated instances, and discard samples that require private or external knowledge for interpretation. Personally identifiable information is masked, and sensitive content is retained only when it is necessary for affective interpretation and can be safely anonymized.

![](images/a807b26c1a5361a1939a3fd60132e4f53a70de5f2c5379b54b68ab3aab1d5f71.jpg)  
Figure 2: Overview of CUE-Bench. The benchmark collects context–target utterance pairs from diverse Chinese dialogue scenarios and models deeper affective understanding through the Explicit–Implicit Stance Matrix. The matrix contrasts the explicit affective signal $e _ { i } , \mathrm { i . e . }$ , what is expressed on the surface, with the implicit affective signal $h _ { i } .$ i.e., what remains unsaid. Guided by Matrix-Guided CoT, the reasoning pipeline progressively infers affective stance, pragmatic intent, and fine-grained emotion, moving from surface expression to intended meaning.

## 3.3 Annotation Protocols

Our annotation strategy combines model-assisted candidate generation, human adjudication, and bias-controlled LLM adjudication. The goal is to obtain scalable annotations while preserving human verification for ambiguous Explicit-Implicit affective relations.

Model-assisted Candidate Annotation. We first use two independent models, $M _ { 1 }$ and $M _ { 2 }$ , to generate candidate annotations for each instance:

$$
\hat { y } _ { i } ^ { ( 1 ) } = M _ { 1 } ( x _ { i } ) , \qquad \hat { y } _ { i } ^ { ( 2 ) } = M _ { 2 } ( x _ { i } ) .
$$

Consistent outputs are retained as high-confidence candidates:

$$
\mathcal { D } _ { a g r } = \{ x _ { i } \mid \hat { y } _ { i } ^ { ( 1 ) } = \hat { y } _ { i } ^ { ( 2 ) } \} ,
$$

while inconsistent outputs are treated as ambiguous cases for further adjudication:

$$
{ \mathcal { D } } _ { d i s } = \{ x _ { i } \mid \hat { y } _ { i } ^ { ( 1 ) } \neq \hat { y } _ { i } ^ { ( 2 ) } \} .
$$

Human Adjudication and Gold Verification. For a subset of $\mathcal { D } _ { d i s }$ , trained annotators verify the candidate annotations by selecting the more appropriate label or revising the annotation when neither candidate is adequate. This produces a human-verified subset that serves as gold supervision and calibration data for later quality control.

Bias-controlled LLM Adjudication. For the remaining ambiguous cases, we use LLMs as constrained adjudicators rather than unconstrained annotators. Each case is adjudicated twice with the candidate order reversed:

$$
b _ { i } ^ { \mathrm { f w d } } = A ( x _ { i } , \hat { y } _ { i } ^ { ( 1 ) } , \hat { y } _ { i } ^ { ( 2 ) } ) , b _ { i } ^ { \mathrm { r e v } } = A ( x _ { i } , \hat { y } _ { i } ^ { ( 2 ) } , \hat { y } _ { i } ^ { ( 1 ) } ) ,
$$

where $b _ { i } ^ { \mathrm { f w d } } , b _ { i } ^ { \mathrm { r e v } } \in \{ 1 , 2 \}$ indicate the selected candidate under each order. We map the reversed decision back to the canonical order as $\bar { b } _ { i } ^ { \mathrm { r e v } } = 3 - b _ { i } ^ { \mathrm { r e v } }$ The retained annotation is:

$$
\tilde { y } _ { i } = \left\{ \begin{array} { l l } { \hat { y } _ { i } ^ { ( b _ { i } ^ { \mathrm { f w d } } ) } , } & { \mathrm { i f } b _ { i } ^ { \mathrm { f w d } } = \bar { b } _ { i } ^ { \mathrm { r e v } } , } \\ { \varnothing , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

Only instances with $\tilde { y } _ { i } \neq \emptyset$ are added to the retained LLM-adjudicated subset. This consistency check reduces positional bias and filters out unstable adjudication cases.

<table><tr><td>Metric</td><td>Affective Stance</td><td>Pragmatic Intent</td><td>Fine-grained Emotion</td></tr><tr><td># Classes</td><td>9</td><td>8</td><td>25</td></tr><tr><td>Krippendorff&#x27;s α</td><td>0.5197</td><td>0.3388</td><td>0.3146</td></tr><tr><td>Majority Agr.</td><td>0.7933</td><td>0.6133</td><td>0.4400</td></tr><tr><td>Avg. κ</td><td>0.5490</td><td>0.3922</td><td>0.3369</td></tr><tr><td>Cond. Avg. κ</td><td></td><td>0.7894</td><td>0.6689</td></tr></table>

Table 2: Inter-annotator agreement on 300 expert reannotated instances. Cond. Avg. κ denotes average Cohen’s kappa computed on instances with consistent Affective Stance. Detailed construction statistics and pairwise agreement results are reported in Appendix B.3.

## 3.4 Statistics and Analysis

Dataset Composition. CUE-Bench is constructed from 60,000 candidate instances through a hybrid pipeline of model agreement, human verification, and bias-controlled LLM adjudication. The final benchmark contains 51,823 retained instances, consisting of high-confidence model-agreement annotations, a human-verified gold subset, and order-consistent LLM-adjudicated annotations. Detailed construction statistics are reported in Appendix B.3.

Adjudication Validation and Noise Estimate. We validate LLM adjudication on the humanverified gold subset, where the adjudicator achieves 89% accuracy when selecting between two modelgenerated candidate annotations. After forward– reverse consistency filtering, we estimate that approximately 1,600 retained instances may remain uncertain, yielding an overall estimated contamination rate of 3.1% in the final dataset. This indicates that constrained LLM adjudication introduces limited residual noise while substantially improving dataset coverage.

Inter-Annotator Agreement. To evaluate annotation reliability, we sample 300 instances from the human-verified gold subset and ask three expert annotators to independently re-annotate them. Table 2 reports agreement at the three annotation layers. Affective Stance obtains the strongest agreement, with a Krippendorff’s α of 0.5197, a majority agreement rate of 79.3%, and an average Cohen’s κ of 0.5490. Pragmatic Intent and Finegrained Emotion show lower raw agreement, with α scores of 0.3388 and 0.3146, respectively, reflecting the higher subjectivity of latent intent and fine-grained affective inference. This pattern is consistent with prior findings that fixed IRR thresholds can be overly rigid for subjective annotation tasks, and that fine-grained emotion annotation often yields moderate or low chance-corrected agreement scores (Wong et al., 2021; Demszky et al., 2020). We therefore report both raw and conditional agreement. When conditioned on consistent Affective Stance, agreement improves substantially: the conditional average κ reaches 0.7894 for Pragmatic Intent and 0.6689 for Fine-grained Emotion. This suggests that Affective Stance provides a useful intermediate structure for localizing annotation ambiguity and reducing downstream uncertainty in intent and emotion labeling. The annotators’ conflicts are not uniform across categories; the detailed confusion matrix is shown in Figure 3.

![](images/a3103ce816452c7b67f79894d640ba1ce9006d95335081e320e4d402f7842651.jpg)  
Figure 3: IAA disagreement matrix among three annotators. Darker cells indicate stronger disagreement between annotator label assignments.

## 4 Explicit-Implicit Stance Matrix

The core of CUE-Bench is the Explicit-Implicit Stance Matrix, a structured framework for modeling how expressed affect and unsaid affect jointly shape the speaker’s affective stance. Rather than treating explicit and implicit affect as two independent labels, the matrix defines Affective Stance as their compositional relation.

## 4.1 Explicit and Implicit Affective Signals

For each instance $x _ { i } = ( C _ { i } , u _ { i } )$ , we distinguish two affective signals. The explicit affective signal describes the affect directly expressed by the target text u<sub>i</sub>, such as affective words, intensifiers, punctuation, praise, complaint, apology, thanks, or direct evaluation. The implicit affective tendency describes the affect inferred from the discourse context $C _ { i }$ together with the target text $u _ { i } ,$ including pragmatic force, speaker intention, and contextual implication.

Let $\pi ( \cdot )$ denote an orientation projection that maps an affective layer onto a three-way affective orientation space. Specifically, we define

$$
e _ { i } = \pi ( y _ { i } ^ { \mathrm { e x p } } ) , \qquad h _ { i } = \pi ( y _ { i } ^ { \mathrm { i m p } } ) ,
$$

where $e _ { i } , h _ { i } \in { \mathcal { O } }$ and $\mathcal { O } = \{ + , 0 , - \}$ . The symbols $+ , 0 ,$ , and − correspond to positive, neutral, and negative affective orientations, respectively. In this formulation, $e _ { i }$ encodes the explicit affective signal anchored in the surface expression of $u _ { i }$ while $h _ { i }$ encodes the latent affective tendency inferred from the utterance together with its context $( C _ { i } , u _ { i } )$

## 4.2 Affective Stance

We define Affective Stance as:

$$
s _ { i } = \phi ( e _ { i } , h _ { i } ) , \qquad s _ { i } = y _ { i } ^ { s t a n c e } ,
$$

where $\phi : \mathcal { O } \times \mathcal { O } \to \mathcal { S }$ maps each Explicit-Implicit pair to one of nine stance categories.

Table 2 shows the resulting matrix, where rows correspond to the explicit signal $e _ { i }$ and columns correspond to the implicit tendency $h _ { i }$ . This formulation makes Affective Stance a structured intermediate representation rather than an independent freeform label: if either $e _ { i }$ or $h _ { i }$ changes, the stance label $s _ { i }$ changes accordingly. The matrix therefore provides a transparent mechanism for capturing alignment, neutralization, concealment, and reversal between what is expressed and what remains unsaid. Detailed definitions and examples of the nine Affective Stances are provided in Appendix D.

## 4.3 Matrix-Guided Chain-of-Thought

The Explicit-Implicit Stance Matrix further provides a structured chain-of-thought for affective inference. Rather than treating the three benchmark tasks as independent predictions, we use the matrix to impose an explicit reasoning order from surface expression to latent meaning and then to downstream affective interpretation. This turns affective understanding into a progressive inference process: identify what is expressed, infer what remains unsaid, resolve their stance relation, interpret the speaker’s pragmatic motivation, and finally determine the fine-grained emotion.

For each instance $x _ { i }$ , the matrix-guided reasoning path is:

$$
\begin{array} { r l } & { ( \hat { e } _ { i } , \hat { h } _ { i } ) = F _ { \mathrm { s i g } } ( x _ { i } ) , } \\ & { ~ \hat { s } _ { i } = \phi ( \hat { e } _ { i } , \hat { h } _ { i } ) , } \\ & { ~ \hat { y } _ { i } ^ { \mathrm { i n t e n t } } = F _ { \mathrm { p r a g } } ( x _ { i } , \hat { e } _ { i } , \hat { h } _ { i } , \hat { s } _ { i } ) , } \\ & { \hat { y } _ { i } ^ { \mathrm { e m o t i o n } } = F _ { \mathrm { e m o } } ( x _ { i } , \hat { e } _ { i } , \hat { h } _ { i } , \hat { s } _ { i } , \hat { y } _ { i } ^ { \mathrm { i n t e n t } } ) . } \end{array}
$$

Here, $\boldsymbol { \hat { e } } _ { i }$ and $\hat { h } _ { i }$ denote the predicted explicit and implicit affective orientations, $\hat { s } _ { i } = \phi ( \hat { e } _ { i } , \hat { h } _ { i } )$ denotes the predicted Affective Stance, $\hat { y } _ { i } ^ { \mathrm { i n t e n t } }$ denotes the predicted pragmatic intent, and $\hat { y } _ { i } ^ { \mathrm { e m o t i o n } }$ denotes the predicted fine-grained emotion.

This chain gives each prediction a structured dependency: stance is inferred from the Explicit-Implicit relation, intent is interpreted under the resulting stance, and fine-grained emotion is decided with both stance and intent as intermediate evidence. In LLM evaluation, we instantiate this process as a normalized prompting protocol, requiring the model to output intermediate fields in the order of explicit signal, implicit tendency, Affective Stance, pragmatic intent, and fine-grained emotion. Compared with direct label prediction, this matrix-guided CoT exposes the model’s reasoning path and enables error analysis at each level of affective inference.

## 5 Experiments

## 5.1 Settings

Models. We evaluate a diverse set of large language models covering both proprietary and open-source families. (1) For proprietary models, we include GPT-4o-mini and DeepSeek-V4- Flash, which represent widely used lightweight instruction-following models with strong general reasoning ability. (2) For open-source models, we evaluate LLaMA-4-Maverick, LLaMA-3.1-8B, and Qwen-series models, including Qwen-3-8B.

LLM prompting baselines. We compare our matrix-guided reasoning method with three standard LLM prompting baselines: (1) Direct prompting, which asks the model to directly output the predicted label from the context and target utterance; (2) Few-shot prompting, which provides annotated demonstrations before prediction; and (3) CoT prompting, which elicits free-form intermediate reasoning before the final prediction.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Affective Stance ↑</td><td colspan="3">Pragmatic Intent ↑</td><td colspan="3">Fine-grained Emotion ↑</td><td rowspan="2">Avg. ↑</td></tr><tr><td>Acc.</td><td>F1</td><td>W-F1</td><td>Acc.</td><td>F1</td><td>W-F1</td><td>Acc.</td><td>F1</td><td>W-F1</td></tr><tr><td rowspan="5">DeepSeek V4-Flash</td><td>Direct</td><td>0.492</td><td>0.408</td><td>0.500</td><td>0.325</td><td>0.277</td><td>0.330</td><td>0.247</td><td>0.170</td><td>0.248</td><td>0.333</td></tr><tr><td>Few-shot</td><td>0.480</td><td>0.412</td><td>0.498</td><td>0.339</td><td>0.288</td><td>0.344</td><td>0.247</td><td>0.174</td><td>0.246</td><td>0.336</td></tr><tr><td>CoT</td><td>0.498</td><td>0.425</td><td>0.516</td><td>0.332</td><td>0.277</td><td>0.343</td><td>0.251</td><td>0.182</td><td>0.253</td><td>0.342</td></tr><tr><td>Ours</td><td>0.510</td><td>0.466</td><td>0.527</td><td>0.459</td><td>0.361</td><td>0.465</td><td>0.313</td><td>0.194</td><td>0.327</td><td>0.402</td></tr><tr><td>∆</td><td>+0.012</td><td>+0.041</td><td>+0.011</td><td>+0.120</td><td>+0.073</td><td>+0.121</td><td>+0.062</td><td>+0.012</td><td>+0.074</td><td>+0.061</td></tr><tr><td rowspan="5">GPT 40-mini</td><td>Direct</td><td>0.317</td><td>0.250</td><td>0.292</td><td>0.281</td><td>0.263</td><td>0.285</td><td>0.115</td><td>0.089</td><td>0.112</td><td>0.223</td></tr><tr><td>Few-shot</td><td>0.341</td><td>0.281</td><td>0.336</td><td>0.302</td><td>0.274</td><td>0.298</td><td>0.226</td><td>0.153</td><td>0.229</td><td>0.271</td></tr><tr><td>CoT</td><td>0.429</td><td>0.343</td><td>0.431</td><td>0.281</td><td>0.262</td><td>0.297</td><td>0.226</td><td>0.150</td><td>0.226</td><td>0.294</td></tr><tr><td>Ours</td><td>0.458</td><td>0.336</td><td>0.444</td><td>0.390</td><td>0.264</td><td>0.383</td><td>0.275</td><td>0.147</td><td>0.266</td><td>0.329</td></tr><tr><td>∆</td><td>+0.029</td><td>-0.007</td><td>+0.013</td><td>+0.088</td><td>-0.010</td><td>+0.085</td><td>+0.049</td><td>-0.006</td><td>+0.037</td><td>+0.035</td></tr><tr><td rowspan="5">8 LLaMA 4 Maverick</td><td>Direct</td><td>0.338</td><td>0.270</td><td>0.326</td><td>0.273</td><td>0.245</td><td>0.273</td><td>0.237</td><td>0.162</td><td>0.232</td><td>0.262</td></tr><tr><td>Few-shot</td><td>0.367</td><td>0.315</td><td>0.385</td><td>0.293</td><td>0.271</td><td>0.307</td><td>0.233</td><td>0.169</td><td>0.233</td><td>0.286</td></tr><tr><td>CoT</td><td>0.454</td><td>0.361</td><td>0.463</td><td>0.248</td><td>0.216</td><td>0.256</td><td>0.253</td><td>0.167</td><td>0.247</td><td>0.296</td></tr><tr><td>Ours</td><td>0.512</td><td>0.410</td><td>0.511</td><td>0.453</td><td>0.342</td><td>0.452</td><td>0.339</td><td>0.183</td><td>0.331</td><td>0.393</td></tr><tr><td>∆</td><td>+0.058</td><td>+0.049</td><td>+0.048</td><td>+0.160</td><td>+0.071</td><td>+0.145</td><td>+0.086</td><td>+0.014</td><td>+0.084</td><td>+0.096</td></tr><tr><td rowspan="5">8 LLaMA 3.1 8B</td><td>Direct</td><td>0.203</td><td>0.194</td><td>0.221</td><td>0.232</td><td>0.208</td><td>0.233</td><td>0.142</td><td>0.086</td><td>0.151</td><td>0.186</td></tr><tr><td>Few-shot</td><td>0.281</td><td>0.256</td><td>0.307</td><td>0.259</td><td>0.231</td><td>0.277</td><td>0.167</td><td>0.112</td><td>0.166</td><td>0.228</td></tr><tr><td>CoT</td><td>0.277</td><td>0.256</td><td>0.311</td><td>0.226</td><td>0.216</td><td>0.248</td><td>0.129</td><td>0.101</td><td>0.135</td><td>0.211</td></tr><tr><td>Ours</td><td>0.362</td><td>0.293</td><td>0.369</td><td>0.285</td><td>0.236</td><td>0.292</td><td>0.190</td><td>0.097</td><td>0.184</td><td>0.256</td></tr><tr><td>∆</td><td>+0.081</td><td>+0.037</td><td>+0.058</td><td>+0.026</td><td>+0.005</td><td>+0.015</td><td>+0.023</td><td>-0.015</td><td>+0.018</td><td>+0.027</td></tr><tr><td rowspan="5">Qwen 3-8B</td><td>Direct</td><td>0.180</td><td>0.154</td><td>0.172</td><td>0.196</td><td>0.210</td><td>0.242</td><td>0.209</td><td>0.128</td><td>0.225</td><td>0.191</td></tr><tr><td>Few-shot</td><td>0.181</td><td>0.183</td><td>0.197</td><td>0.231</td><td>0.214</td><td>0.227</td><td>0.212</td><td>0.121</td><td>0.221</td><td>0.199</td></tr><tr><td>CoT</td><td>0.422</td><td>0.325</td><td>0.430</td><td>0.254</td><td>0.226</td><td>0.257</td><td>0.212</td><td>0.132</td><td>0.206</td><td>0.274</td></tr><tr><td>Ours</td><td>0.424</td><td>0.309</td><td>0.412</td><td>0.364</td><td>0.271</td><td>0.367</td><td>0.260</td><td>0.137</td><td>0.262</td><td>0.312</td></tr><tr><td>∆</td><td>+0.002</td><td>-0.016</td><td>-0.018</td><td>+0.110</td><td>+0.045</td><td>+0.110</td><td>+0.048</td><td>+0.005</td><td>+0.037</td><td>+0.038</td></tr></table>

Table 3: Main results. We report Accuracy, macro-F1, and weighted-F1 on three CUE-Bench tasks: Affective Stance, Pragmatic Intent, and Fine-grained Emotion. ↑ indicates that higher values are better.

Metrics. We report Accuracy (Acc.), macro-F1 (F1), and weighted-F1 (W-F1). Accuracy measures overall correctness, macro-F1 gives equal weight to each class and reflects performance on minority categories, while weighted-F1 accounts for label imbalance by weighting class-wise F1 scores by class frequency.

## 5.2 Main Results

As shown in Table 3, we draw three observations.

Our matrix-guided method achieves the best overall performance. Our method obtains the highest average score across all evaluated models, outperforming the strongest baseline by +0.027 to +0.096. The gains are especially clear on DeepSeek-V4-Flash and LLaMA-4-Maverick, showing the effectiveness of jointly modeling surface expression and hidden affective tendency.

Pragmatic intent shows the most consistent gains. Our method improves Pragmatic Intent accuracy across all models, with gains from +0.026 to +0.160, and also consistently improves weighted-F1. This indicates that communicative intent benefits strongly from the explicit–implicit affective distinction.

<table><tr><td>Model</td><td>Task</td><td>Acc.</td><td>F1</td><td>W-F1</td></tr><tr><td rowspan="3">DeepSeek-V4-Flash</td><td>I</td><td>0.791</td><td>0.703</td><td>0.790</td></tr><tr><td>II</td><td>0.392</td><td>0.243</td><td>0.379</td></tr><tr><td>III</td><td>0.432</td><td>0.219</td><td>0.389</td></tr><tr><td rowspan="3">LLaMA-4-Maverick</td><td>I</td><td>0.739</td><td>0.658</td><td>0.745</td></tr><tr><td>ⅡI</td><td>0.430</td><td>0.251</td><td>0.436</td></tr><tr><td>III</td><td>0.489</td><td>0.254</td><td>0.466</td></tr></table>

Table 4: Ablations on CUE-Bench. Task I: Affective Stance → Pragmatic Intent; Task II: Affective Stance → Fine-grained Emotion; Task III: Affective Stance + Pragmatic Intent → Fine-grained Emotion.

Fine-grained emotion improves but remains harder. Our method improves Fine-grained Emotion accuracy and weighted-F1 across all models, while macro-F1 gains are less stable. This suggests that stance-guided reasoning helps infer emotional tendency, but distinguishing fine-grained emotion categories remains challenging.

![](images/56fe46429c380fb13f8fd0d1d6d3c6de3f9ab8a5a3affbf5592bf9bf6c1cf3fd.jpg)  
Figure 4: Wide light bars show audited support, narrow dark bars show problem-case counts, and the red line shows the problem-case rate.

## 5.3 Ablations

To examine whether the proposed reasoning path benefits downstream affective inference, we conduct oracle-conditioning ablations with DeepSeek-V4-Flash and LLaMA-4-Maverick. As shown in Table 4, we test whether gold Affective Stance and Pragmatic Intent can serve as intermediate evidence for predicting Pragmatic Intent and Fine-grained Emotion. We draw three observations.

Affective Stance provides strong evidence for pragmatic intent. With gold Affective Stance, Pragmatic Intent prediction reaches 0.703 macro-F1 on DeepSeek-V4-Flash and 0.658 on LLaMA-4-Maverick, confirming its value as an intermediate representation.

Pragmatic Intent adds complementary evidence for emotion inference. Adding gold Pragmatic Intent on top of gold Affective Stance improves Fine-grained Emotion accuracy and weighted-F1 for both models, showing that intent contributes information beyond stance alone.

Oracle signals help but do not replace category-level emotion discrimination. Although gold intermediate labels improve Fine-grained Emotion performance, the gains are not uniform across all metrics. This suggests that the reasoning chain provides useful affective evidence, while final emotion prediction still requires direct discrimination among emotion categories.

## 5.4 Analysis

We focus on two research questions that explain the distributional and annotation patterns observed in the CUE-Bench.

RQ1: Why are negative and indirect cases frequent? CUE-Bench contains many negative or negative-leaning cases because we intentionally retain sources rich in implicit affect, such as sarcastic, hostile, conflictual, and emotionally charged online discourse. This increases the density of cases where literal wording is insufficient, making the benchmark more diagnostic for unsaid affect.

Veiled negative cases dominate the benchmark. VEILED NEGATIVE accounts for 22.3% of the data, where neutral surface wording often implies dissatisfaction, reluctance, pressure, or indirect criticism. These cases are challenging because models must recover hidden negative affect from context rather than explicit lexical cues.

Sarcastic negative cases reflect deliberate stress-test design. SARCASTIC NEGATIVE also appears frequently (10.9%) because the corpus includes sarcasm-oriented and hostile-comment data. Thus, the negative skew should be viewed as a benchmark feature rather than a natural base-rate estimate: CUE-Bench is designed to evaluate affective inference under pragmatic mismatch.

RQ2: Where does human–AI annotation friction arise? We analyze a 1,500-instance audit sample from the model-disagreement pool, where GPT adjudication was run in both candidate orders and compared with human review. A case is marked as problematic if the two adjudication orders are inconsistent, or if a consistent GPT decision disagrees with the human-accepted label.

Veiled negative cases are the main source of friction. As shown in Figure 4, problematic cases concentrate heavily in VEILED NEGATIVE: 458 cases, covering 62% of audited VEILED NEGA-TIVE instances. This suggests that neutral-looking utterances with negative implication are especially difficult for AI adjudication.

Friction appears when surface affect and implied affect diverge. SARCASTIC NEGATIVE and UNDERSTATED POSITIVE often require pragmatic reversal or concealment, while FORMULAIC POS-ITIVE requires distinguishing genuine positivity from scripted politeness. These patterns show that human review is most valuable for stance categories with strong explicit–implicit mismatch.

## 6 Conclusion

We introduce CUE-Bench, a Chinese benchmark for unsaid emotion understanding centered on the Explicit-Implicit Stance Matrix. By decomposing affective meaning into explicit signal, implicit tendency, Affective Stance, Pragmatic Intent, and

Fine-grained Emotion, CUE-Bench makes the path from what is said to what is meant directly evaluable. Experiments show that matrix-guided prompting improves overall performance, while oracleconditioning ablations confirm the value of Affective Stance as an intermediate representation. CUE-Bench provides both a benchmark and an analysis framework for studying polite, suppressed, ironic, indirect, and otherwise unsaid affect in Chinese discourse.

## 7 Limitations

CUE-Bench has three main limitations. (1) Residual annotation noise. Although our dual-model annotation pipeline is calibrated and verified with human-labeled data, the final benchmark may still contain unavoidable residual noise. In particular, LLM adjudication is used only as a constrained component with consistency filtering, but some uncertain or contaminated instances may remain. (2) Coarse explicit–implicit orientation space. The three-way explicit/implicit orientation space makes Affective Stance interpretable and easy to operationalize, but it inevitably abstracts away finer affective distinctions that must be recovered at the Fine-grained Emotion stage. (3) Long-tailed and culturally situated labels. CUE-Bench is longtailed: categories such as REPORTIVE NEGATIVE, EMPATHY, and several rare emotions have limited support, so macro-F1 and weighted-F1 should be read together. Moreover, implicit affect and pragmatic intent remain partly subjective and culturally situated, even with detailed guidelines and adjudication. Future extensions can add richer conversational metadata, multimodal signals, and multilingual comparisons while preserving the explicit– implicit stance structure.

## References

Penelope Brown and Stephen C Levinson. 1987. Politeness: Some universals in language usage, volume 4. Cambridge university press.

Fengyu Cai, Wanhao Zhou, Fei Mi, and Boi Faltings. 2022. Slim: Explicit slot-intent mapping with bert for joint multi-intent detection and slot filling. pages 7607–7611.

CCAC 2024 Chinese Sarcasm Calculation Organizers. 2024. Chinese sarcasm calculation evaluation task at CCAC 2024. https://github.com/pjzj220113/ chinese-sarcasm-calculation. Evaluation task dataset and instructions.

Meng Chen, Ruixue Liu, Lei Shen, Shaozu Yuan, Jingyan Zhou, Youzheng Wu, Xiaodong He, and Bowen Zhou. 2020. The JDDC corpus: A large-scale multi-turn Chinese dialogue dataset for E-commerce customer service. In Proceedings ofthe Twelfth Language Resources and Evaluation Conference, pages 459–466, Marseille, France. European Language Resources Association.

Qian Chen, Zhu Zhuo, and Wen Wang. 2019. BERT for joint intent classification and slot filling. arXiv preprint arXiv:1902.10909.

Yirong Chen, Weiquan Fan, Xiaofen Xing, Jianxin Pang, Minlie Huang, Wenjing Han, Qianfeng Tie, and Xiangmin Xu. 2022. Cped: A large-scale chinese personalized and emotional dialogue dataset for conversational ai.

Zhiwei Chen, Yupeng Hu, Zhiheng Fu, Zixu Li, Jiale Huang, Qinlei Huang, and Yinwei Wei. 2026. Intent: Invariance and discrimination-aware noise mitigation for robust composed image retrieval. In AAAI, volume 40, pages 20463–20471.

Zhiwei Chen, Yupeng Hu, Zixu Li, Zhiheng Fu, Xuemeng Song, and Liqiang Nie. 2025. Offset: Segmentation-based focus shift revision for composed image retrieval. In ACM MM, page 6113–6122.

Alexandra Chronopoulou, Aikaterini Margatina, Christos Baziotis, and Alexandros Potamianos. 2018. NTUA-SLP at IEST 2018: Ensemble of neural transfer methods for implicit emotion classification. In Proceedings ofthe 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis, pages 57–64, Brussels, Belgium. Association for Computational Linguistics.

Yiming Cui, Wanxiang Che, Ting Liu, Bing Qin, and Ziqing Yang. 2021. Pre-training with whole word masking for chinese bert. IEEE/ACM transactions on audio, speech, and language processing, 29:3504– 3514.

Dorottya Demszky, Dana Movshovitz-Attias, Jeongwoo Ko, Alan Cowen, Gaurav Nemade, and Sujith Ravi. 2020. GoEmotions: A dataset of fine-grained emotions. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 4040–4054, Online. Association for Computational Linguistics.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Devon018. 2025. CN-SarcasmBench. https: //huggingface.co/datasets/Devon018/

CN-SarcasmBench. Hugging Face dataset; licensed under CC-BY-NC-4.0.

Quanqi Du and Veronique Hoste. 2025. Another approach to agreement measurement and prediction with emotion annotations. In Proceedings ofthe 19th Linguistic Annotation Workshop (LAW-XIX-2025), pages 87–102, Vienna, Austria. Association for Computational Linguistics.

John W Du Bois. 2007. The stance triangle. Stancetaking in discourse: Subjectivity, evaluation, interaction, 164(3):139–182.

Deepanway Ghosal, Navonil Majumder, Alexander Gelbukh, Rada Mihalcea, and Soujanya Poria. 2020. COSMIC: COmmonSense knowledge for eMotion identification in conversations. In Findings ofthe Association for Computational Linguistics: EMNLP 2020, pages 2470–2481, Online. Association for Computational Linguistics.

Fabrizio Gilardi, Meysam Alizadeh, and Maël Kubli. 2023. Chatgpt outperforms crowd workers for text-annotation tasks. Proceedings of the National Academy ofSciences, 120(30).

Herbert P Grice. 1975. Logic and conversation. In Speech acts, pages 41–58. Brill.

James J Gross. 1998. The emerging field of emotion regulation: An integrative review. Review ofgeneral psychology, 2(3):271–299.

Bo Hu, Meng Zhang, Chenfei Xie, Yuanhe Tian, Yan Song, and Zhendong Mao. 2024. RESEMO: A benchmark Chinese dataset for studying responsive emotion from social media content. In Findings of the Association for Computational Linguistics: ACL 2024, pages 16375–16387, Bangkok, Thailand. Association for Computational Linguistics.

Roman Klinger, Orphée De Clercq, Saif Mohammad, and Alexandra Balahur. 2018. IEST: WASSA-2018 implicit emotions shared task. In Proceedings ofthe 9th Workshop on Computational Approaches to Subjectivity, Sentiment and Social Media Analysis, pages 31–42, Brussels, Belgium. Association for Computational Linguistics.

Hengli Li, Song-Chun Zhu, and Zilong Zheng. 2023. Diplomat: A dialogue dataset for situated pragmatic reasoning. Advances in Neural Information Processing Systems, 36:46856–46884.

Wenbing Li, Zikai Song, Hang Zhou, Junqing Yu, Yunyao Zhang, and Wei Yang. 2026a. Lora-mixer: Coordinate modular lora experts through serial attention routing. In International Conference on Learning Representations, volume 2026, pages 14694–14716.

Yanran Li, Hui Su, Xiaoyu Shen, Wenjie Li, Ziqiang Cao, and Shuzi Niu. 2017. DailyDialog: A manually labelled multi-turn dialogue dataset. In Proceedings ofthe Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers),

pages 986–995, Taipei, Taiwan. Asian Federation of Natural Language Processing.

Zixu Li, Zhiwei Chen, Haokun Wen, Zhiheng Fu, Yupeng Hu, and Weili Guan. 2025. Encoder: Entity mining and modification relation binding for composed image retrieval. In AAAI, volume 39, pages 5101–5109.

Zixu Li, Yupeng Hu, Zhiwei Chen, Haokun Wen, Xuemeng Song, and Liqiang Nie. 2026b. Combiner: Composed image retrieval guided by attribute-based neighbor relations. IEEE TIP.

Zixu Li, Yupeng Hu, Zhiwei Chen, Mingyu Zhang, Zhiheng Fu, and Liqiang Nie. 2026c. Conesep: Conebased robust noise-unlearning compositional network for composed image retrieval. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16897–16909.

Zixu Li, Yupeng Hu, Zhiwei Chen, Shiqi Zhang, Qinlei Huang, Zhiheng Fu, and Yinwei Wei. 2026d. Habit: Chrono-synergia robust progressive learning framework for composed image retrieval. In AAAI, volume 40, pages 6762–6770.

Zixu Li, Yupeng Hu, Zhiheng Fu, Zhiwei Chen, Weili Guan, and Liqiang Nie. 2026e. R<sup>3</sup>: Composed video retrieval via reasoning-guided recalling and re-ranking. arXiv preprint arXiv:2606.01113.

Zixu Li, Yupeng Hu, Zhiheng Fu, Zhiwei Chen, Yongqi Li, and Liqiang Nie. 2026f. Tema: Anchor the image, follow the text for multi-modification composed image retrieval. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 24421– 24442.

Hongye Liu, Liang Ding, and Ricardo Henao. 2026. Learning to control summaries with score ranking. arXiv preprint arXiv:2604.17197.

Hongye Liu and Ricardo Henao. 2025. Learning to substitute words with model-based score ranking. In Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 11551–11565.

Rosalind W Picard. 1997. Affective computing.

Robert Plutchik. 2001. The nature of emotions: Human emotions have deep evolutionary roots, a fact that may explain their complexity and provide tools for clinical practice. American scientist, 89(4):344–350.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 527– 536, Florence, Italy. Association for Computational Linguistics.

Zikai Song, Xiajie Li, Yunyao Zhang, Xinglang Zhang, Wei Yang, and Junqing Yu. 2026. Social intelligence modeling: A comprehensive survey from social perception to social simulation.

Settaluri Sravanthi, Meet Doshi, Pavan Tankala, Rudra Murthy, Raj Dabre, and Pushpak Bhattacharyya. 2024. PUB: A pragmatics understanding benchmark for assessing LLMs’ pragmatics capabilities. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 12075–12097, Bangkok, Thailand. Association for Computational Linguistics.

Dali Wang, Yunyao Zhang, Junqing Yu, Yi-Ping Phoebe Chen, Chen Xu, and Zikai Song. 2026. Seeing further and wider: Joint spatio-temporal enlargement for micro-video popularity prediction. Preprint, arXiv:2604.20311.

Yida Wang, Pei Ke, Yinhe Zheng, Kaili Huang, Yong Jiang, Xiaoyan Zhu, and Minlie Huang. 2020. A large-scale chinese short-text conversation dataset. In NLPCC.

Ka Wong, Praveen Paritosh, and Lora Aroyo. 2021. Cross-replication reliability - an empirical approach to interpreting inter-rater reliability. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 7053–7065, Online. Association for Computational Linguistics.

Yafeng Wu, Yunyao Zhang, Liliang Ye, Guiyi Zeng, Junqing Yu, Chen Xu, and Zikai Song. 2026. Hotcomment: A benchmark for evaluating popularity of online comments. arXiv preprint arXiv:2604.25614.

Guo Xianwei, Lai Hua, Xiang Yan, Yu Zhengtao, and Huang Yuxin. 2021. Emotion classification of COVID-19 Chinese microblogs based on the emotion category description. In Proceedings ofthe 20th Chinese National Conference on Computational Linguistics, pages 916–927, Huhhot, China. Chinese Information Processing Society of China.

Xinglang Zhang, Yunyao Zhang, ZeLiang Chen, Junqing Yu, Wei Yang, and Zikai Song. 2026a. Logical phase transitions: Understanding collapse in LLM logical reasoning. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 18836– 18860, San Diego, California, United States. Association for Computational Linguistics.

Yazhou Zhang, Yang Yu, Qing Guo, Benyou Wang, Dongming Zhao, Sagar Uprety, Dawei Song, Qiuchi Li, and Jing Qin. 2023. Cmma: benchmarking multiaffection detection in chinese multi-modal conversations. Advances in Neural Information Processing Systems, 36:18794–18805.

Yunyao Zhang, Yihao Ai, Zuocheng Ying, Qirui Mi, Junqing Yu, Wei Yang, and Zikai Song. 2026b. Coupling macro dynamics and micro states for long-horizon social simulation. arXiv preprint arXiv:2604.05516.

Yunyao Zhang, Zikai Song, Hang Zhou, Wenfeng Ren, Yi-Ping Phoebe Chen, Junqing Yu, and Wei Yang. 2025. ga − s<sup>3</sup>: Comprehensive social network simulation with group agents. In Findings of the Association for Computational Linguistics: ACL 2025, pages 8950–8970, Vienna, Austria. Association for Computational Linguistics.

Yunyao Zhang, Zuocheng Ying, Xinglang Zhang, Junqing Yu, Peng Fang, Xu Chen, Wei Yang, and Zikai Song. 2026c. Intervensim: Intervention-aware social network simulation for opinion dynamics. Preprint, arXiv:2604.06600.

Yunyao Zhang, Xinglang Zhang, Zeliang Chen, Junqing Yu, and Zikai Song. 2026d. Semiotic logical hexagon theory for llm logical reasoning. Preprint, arXiv:2607.21933.

Yunyao Zhang, Xinglang Zhang, Junxi Sheng, Wenbing Li, Junqing Yu, Yi-Ping Phoebe Chen, Wei Yang, and Zikai Song. 2026e. Semantic-aware logical reasoning via a semiotic framework. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 18349–18374, San Diego, California, United States. Association for Computational Linguistics.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, Red Hook, NY, USA. Curran Associates Inc.

## Appendix

This appendix provides implementation details for annotation, examples, prompts, and release documentation.

## The Usage of LLM

In accordance with \*CL policy, LLMs were used as writing and annotation-support tools. For dataset construction, model outputs are treated as candidate labels and rationales. Final labels are humanvalidated.

## A Additional Distributional Analysis

Context-bound stance categories. The rarest Affective Stance is REPORTIVE NEGATIVE (0.6%), where negative surface wording is used with a largely neutral, reportive force. This pattern is more natural in news-style reporting, incident summaries, or analytical long-form writing than in casual online interaction. Because CUE-Bench is dominated by internet dialogue and social-media discourse (Wu et al., 2026), REPORTIVE NEGA-TIVE remains sparse; Zhihu-style long-form text is more likely to contain such cases, but it is not the dominant source. By contrast, FORMULAIC POSI-TIVE (9.2%) is tied to service-oriented interaction, where thanks, apologies, honorifics, and blessing formulas are often routine rather than deeply positive.

Intent and emotion distributions. AUTHENTIC-ITY is the most frequent Pragmatic Intent (34.7%), while socially mediated intents such as POLITE-NESS (16.8%), SUPPRESSION (14.5%), IRONY (12.4%), and RESISTANCE (9.5%) remain substantial. Fine-grained Emotion shows a stronger long tail. The right panel of Figure 5 shows the highfrequency head; low-frequency emotions are omitted from the visualization for readability but remain part of the benchmark and evaluation. The emotion inventory is organized with reference to Plutchik’s emotion wheel (Plutchik, 2001). The prominence of CONTEMPT (20.1%) is consistent with the negative and sarcastic bias of the initial corpus construction, while categories such as SUB-MISSION and HOPE show that the benchmark also captures relational and anticipatory affect.

## B CUE-Bench Details

## B.1 Release Format and Data Split

The released data follow a unified format that supports all three benchmark tasks. Table 5 summarizes the core fields.

We split the dataset by source instance rather than by individual text span to avoid context leakage across training, development, and test sets. The split is stratified by Affective Stance where possible, since the nine-way stance distribution is naturally imbalanced.

## B.2 Source Data and Label Independence

CUE-Bench draws raw Chinese text and dialogue context from LCCC (Wang et al., 2020), JDDC (Chen et al., 2020), the CCAC 2024 Chinese Sarcasm Calculation dataset (CCAC 2024 Chinese Sarcasm Calculation Organizers, 2024), Zhihu QA, and CN-SarcasmBench (Devon018, 2025). These sources are used only as text pools. We do not reuse their original sentiment labels, sarcasm labels, dialogue labels, or any other source-provided annotations as CUE-Bench labels. All released Affective Stance, Pragmatic Intent, and Fine-grained Emotion annotations are produced through our own screening, candidate annotation, human verification, and adjudication pipeline.

## B.3 Annotation Pipeline and Quality Control

We construct CUE-Bench through a hybrid annotation pipeline that combines model-assisted preannotation, human verification, and consistencybased LLM adjudication. Starting from 60,000 candidate instances, GPT-4o-mini and DeepSeek-V4 independently produce candidate annotations. The 20,000 instances on which the two models agree are retained as high-confidence annotations. For the remaining 40,000 disagreement cases, we manually verify 10,000 instances to construct the human-verified gold subset. During this process, annotators select the more appropriate label from the model-generated candidates rather than assuming that all model annotations are incorrect.

To further assess annotation reliability, we sample 300 instances from the human-verified gold subset and ask three expert annotators to independently re-annotate them. As reported in Table 2, Affective Stance obtains the strongest agreement, while Pragmatic Intent and Fine-grained Emotion show lower raw agreement due to their greater subjectivity and dependence on latent affective inference. However, agreement improves substantially when downstream labels are evaluated under consistent Affective Stance. For Pragmatic Intent, the conditional Krippendorff’s α increases to 0.7688. This supports the role of Affective Stance as an intermediate structure for localizing ambiguity and reducing uncertainty in intent and emotion annotation.

![](images/9d92c7c0966b72a611881c4cf0b38ef24a21dcbe85862d912ab94858426d49f3.jpg)

![](images/acfa305d7efe0787e70c9a57058f0d205daa09b1eeb37707790ba6a876e52e73.jpg)

![](images/9c982d088eb12de1494df96e174b2b6d30c9e06b0556bf6d541c890b4fdaba87.jpg)  
Figure 5: Distribution of the three label layers in CUE-Bench. Affective Stance and Pragmatic Intent are shown in full; Fine-grained Emotion shows the ten most frequent categories for readability.

<table><tr><td>Field</td><td>Description</td></tr><tr><td>sample_id</td><td>Anonymized instance identifier.</td></tr><tr><td>context</td><td>Context used for affective interpretation.</td></tr><tr><td>target_text</td><td>Target utterance or text span to be labeled.</td></tr><tr><td>explicit_layer</td><td>Surface affective information.</td></tr><tr><td>implicit_layer</td><td>Context-inferred implicit affective information.</td></tr><tr><td>affective_stance</td><td>Nine-way Affective Stance label.</td></tr><tr><td>pragmatic_intent</td><td>Pragmatic intent label.</td></tr><tr><td>fine_grained_emotion</td><td>Fine-grained emotion label.</td></tr></table>

Table 5: Core fields in the CUE-Bench release format.

For the remaining 30,000 model-disagreement instances, we use GPT-4o-mini as an adjudicator. Validation on the human-verified gold subset shows that the adjudicator achieves 89% accuracy when selecting between two model-generated candidate annotations. To reduce positional bias, each instance is adjudicated twice with the order of candidate labels reversed. Only instances with consistent forward–reverse adjudication decisions are retained, yielding 21,823 additional instances and discarding 8,177 uncertain cases. The final benchmark contains 51,823 instances.

<table><tr><td>Layer</td><td>Expert 1</td><td>Raw κ</td><td>Cond. κ</td></tr><tr><td>Affective Stance</td><td>A</td><td>0.5116</td><td>一</td></tr><tr><td></td><td>B</td><td>0.5864</td><td></td></tr><tr><td>Pragmatic Intent</td><td>A</td><td>0.4057</td><td>0.8746</td></tr><tr><td></td><td>B</td><td>0.3786</td><td>0.7042</td></tr><tr><td>Fine-grained Emotion</td><td>A</td><td>0.3325</td><td>0.6915</td></tr><tr><td></td><td>B</td><td>0.3412</td><td>0.6462</td></tr></table>

Table 6: Pairwise Cohen’s κ scores between the gold labels and expert re-annotations. Conditional scores are computed on instances with consistent Affective Stance.

## B.4 Detailed Pairwise Agreement

Table 6 reports pairwise Cohen’s κ scores between the gold labels and each expert re-annotation. These results complement the aggregate agreement statistics in Table 2 by showing how each expert annotation compares with the gold labels. Conditional scores are computed on instances with consistent Affective Stance.

Interpreting Agreement Scores. We interpret agreement scores in the context of subjective affective annotation rather than relying on a single universal threshold. Prior work has argued that fixed IRR thresholds such as κ or $\alpha > 0 . 6$ can be overly rigid for subjective tasks with genuine ambiguity (Wong et al., 2021). Fine-grained emotion annotation also commonly exhibits moderate or low chance-corrected agreement: GoEmotions reports an average Cohen’s κ of approximately 0.29 across 27 emotion categories (Demszky et al., 2020), and recent work on emotion annotation reports Fleiss κ values of 0.19–0.33 and Krippendorff’s α values of 0.22–0.64 for emotion and valence annotation settings (Du and Hoste, 2025). These findings motivate our use of both raw and conditional agreement scores, where the latter evaluates whether downstream intent and emotion labels become more reliable under a shared Affective Stance interpretation.

## C Annotation Guidelines

## C.1 Explicit Affective Signal

Annotators label the explicit affective signal using only the target utterance. Context may be shown for orientation, but the decision must be justified by surface evidence in the utterance itself.

Positive signal. Use positive when the utterance directly expresses appreciation, happiness, agreement, praise, relief, or encouragement.

Negative signal. Use negative when the utterance directly expresses blame, anger, disappointment, sadness, anxiety, refusal, or complaint.

Neutral signal. Use neutral when the utterance contains no clear surface affective expression, even if the surrounding context is emotional.

## C.2 Implicit Affective Tendency

Annotators label the implicit affective tendency using the dialogue context and the target utterance together. The label should capture the affect implied by the speaker’s stance, conversational goal, or pragmatic force, rather than simply repeating the surface wording.

Positive tendency. Use positive when the utterance implies acceptance, care, support, relief, or friendly intent.

Negative tendency. Use negative when the utterance implies rejection, dissatisfaction, pressure, hostility, disappointment, or sarcasm.

Neutral tendency. Use neutral when context does not support a clear positive or negative affective inference.

## C.3 Affective Stance Assignment

Affective Stance is derived from the ordered pair of explicit affective signal and implicit affective tendency. Annotators therefore do not invent an independent stance label. They first verify the two base signals, map the pair to the Explicit-Implicit Stance Matrix, and then check whether the resulting stance matches the intended interpretation. If the mapped stance feels implausible, annotators must revisit the explicit or implicit signal rather than manually overriding the stance.

Authentic expression. Use AUTHENTIC-ITY when the utterance directly expresses the speaker’s genuine affective position.

Polite mitigation. Use POLITENESS when positive or softened wording primarily serves social etiquette, deference, apology, or service-script politeness.

Affective concealment. Use SUPPRESSION when the speaker hides, weakens, or withholds the implied affect.

Contrastive meaning. Use IRONY when the utterance relies on reversal, sarcasm, exaggerated praise, or contrast between literal and intended meaning.

Oppositional stance. Use RESISTANCE when the utterance implies refusal, opposition, complaint, pressure, or dissatisfaction.

Task-oriented communication. Use FUNC-TIONAL when the utterance mainly reports, informs, requests, or describes without strong interpersonal strategy.

Playful framing. Use HUMOR when the utterance uses playfulness, teasing, or comic framing as the main pragmatic force.

Supportive orientation. Use EMPATHY when the utterance primarily conveys care, comfort, solidarity, or perspective-taking toward another person.

## C.4 Fine-grained Emotion Decision Rules

Annotators label Fine-grained Emotion as the final affective interpretation, using the target utterance, context, Affective Stance, and Pragmatic Intent together. The label should describe the speaker’s inferred affective state, not the emotion that the utterance may cause in the reader.

Specificity preference. Prefer the most specific emotion supported by contextual evidence; use broad labels such as NEUTRAL only when no specific affect is recoverable.

Underlying affect. Distinguish outward negativity from the underlying state: complaint may indicate OUTRAGE, DISAPPOINTMENT, CONTEMPT, or ANXIETY depending on context.

Relational and future-oriented affect. Use relational and anticipatory labels such as SUBMISSION, HOPE, PESSIMISM, or OPTIMISM when the utterance encodes expectation, dependence, resignation, or future orientation.

Ambiguity resolution. When multiple emotions are plausible, choose the label best supported by the target utterance and its immediate conversational context, and flag genuinely ambiguous cases for review.

## D Definitions and Examples of Affective Stances

Table 7 provides detailed definitions and examples for the nine Affective Stances in the Explicit-Implicit Stance Matrix.

## E Formal Setup

## E.1 Notation

Let $D = ( u _ { 1 } , \dotsc . . . , u _ { T } )$ be a dialogue. For a target utterance $u _ { t }$ , the context is $C _ { t } = ( u _ { t - k } , \dots , u _ { t - 1 } )$ for a configurable window size k, or the full previous dialogue history when available. Each benchmark instance is represented as $x _ { i } = ( C _ { i } , u _ { i } )$ and annotated with explicit affective signal, implicit affective tendency, Affective Stance, Pragmatic Intent, and Fine-grained Emotion.

## E.2 Well-Formedness

An annotation is well formed only when the Affective Stance label matches the ordered pair $( y _ { i } ^ { e x p } , y _ { i } ^ { i m p } )$ under $\phi .$ Invalid stance combinations are rejected during export validation. Pragmatic Intent and Fine-grained Emotion are not deterministic functions of the stance, but they must be justified by the same context–target pair and by the intermediate stance interpretation.

## F Dataset Card

## F.1 Intended Use

The dataset is intended for research on Chinese discourse emotion understanding, Affective Stance Recognition, Pragmatic Intent Understanding, Finegrained Emotion Classification, and robust affectaware dialogue systems. It is designed for evaluating how models infer unsaid affect from context rather than for reusing source-dataset sentiment or sarcasm annotations.

## F.2 Out-of-Scope Use

The dataset should not be used to infer private mental states about real individuals, rank (Liu and Henao, 2025; Liu et al., 2026) users by emotional tendency, or make high-stakes decisions about employment, health, credit, or legal status.

## F.3 Source and License Notes

The release records source identifiers so users can trace the text pool from which each instance was drawn. Users should comply with the licenses and terms of the underlying source datasets, especially for sources with non-commercial restrictions. CUE-Bench annotations are newly created and should not be interpreted as inherited labels from the original datasets.

## F.4 Recommended Reporting

Papers using CUE-Bench should report the input setting, context window, prompt or model configuration, split version, Accuracy, macro-F1, weighted-F1, and whether any low-confidence or filtered examples are included. For Task 1, Affective Stance Recognition, reports should include a confusion matrix over the nine stance categories, because this task corresponds to the Explicit-Implicit Stance Matrix. For Task 2 and Task 3, reports should include class-wise F1 or perclass error analysis when space permits, since Pragmatic Intent and Fine-grained Emotion are longtailed and more subjective. When external data augmentation or additional annotation is used, papers should distinguish it clearly from the official CUE-Bench training, development, and test splits.

## G Additional Evaluation Details

## G.1 Input Formatting for LLM Evaluation

Each LLM (Li et al., 2026a; Zhang et al., 2026d; Wang et al., 2026) input contains the dialogue context, the target utterance, task-specific label definitions, and a constrained output schema. For direct prompting settings, the model is asked to predict the target label directly from the given context and target utterance. For chain-based or matrix-guided prompting settings, the prompt specifies an intermediate reasoning order, but the intermediate fields are inferred by the model itself rather than provided as gold labels.

For Affective Stance Recognition, matrix-guided prompting asks the model to first infer the explicit affective signal and implicit affective tendency, and then map the ordered pair to one of the nine stance labels. For Pragmatic Intent and Fine-grained Emotion, matrix-guided prompting similarly requires the model to derive the relevant intermediate fields before predicting the final label. Gold intermediate labels are provided only in the oracle-conditioning ablation settings, where they are inserted into the prompt to test the contribution of each intermediate variable.

<table><tr><td> $e _ { i }$ </td><td> $h _ { i }$ </td><td>Affective Stance</td><td>Typical Phenomenon</td><td>Chinese-English Example</td></tr><tr><td>+ +</td><td></td><td>POSITIVE</td><td>Direct praise, joy, gratitude, approval, support, or affection.</td><td>Zh：太好了！你终于熬出来了，我真的为你骄傲。 En: That&#x27;s wonderful! You finally made it through. I&#x27;m truly proud of you.</td></tr><tr><td></td><td></td><td>+ 0 FORMULAIC POSITIVE</td><td>Customer-service politeness, routine thanks, greetings, scripted apologies, or socially expected positive word- ing.</td><td>Zh：亲，真的非常抱歉给您带来不便，感谢您的理 解，祝您生活愉快。 En: Dear customer, we sincerely apologize for the in- convenience. Thank you for your understanding, and</td></tr><tr><td>十</td><td></td><td>SARCASTIC NEGATIVE</td><td>Sarcasm, irony, backhanded praise, mock compliment, or exaggerated praise used to express criticism.</td><td>have a nice day. Zh：你可真是天才，每次都能精准踩雷。 En: What a genius you are, managing to step on the exact landmine every time.</td></tr><tr><td>0 +</td><td></td><td>UNDERSTATED POSITIVE</td><td>Modesty, humblebragging, restrained pride, indirect approval, or neutral wording that hides satisfaction.</td><td>Zh：也就一般吧，省赛第一而已。 En: It was nothing special, just first place in the provin- cial contest.</td></tr><tr><td></td><td></td><td>0 0 NEUTRAL</td><td>Factual, procedural, descriptive, or informational utterances without clear affective commitment.</td><td>Zh：会议改到下午三点，地点不变。 En: The meeting has been moved to 3 p.m.; the location</td></tr><tr><td>0</td><td>一</td><td>VEILED NEGATIVE</td><td>Neutral wording that implies dissat- isfaction, reluctance, compromise, pressure, rejection, or concealed criti-</td><td>remains unchanged. Zh：行，你继续按这个方案来，结果我就不评价 了。</td></tr><tr><td></td><td>+</td><td>AFFILIATIVE POSITIVE</td><td>cism. Aggressive joking among friends, affectionate blame, self-deprecation for attention, or criticism mixed with</td><td>En: Fine, keep following this plan. I won&#x27;t comment on the result. Zh：你个猪，终于知道好好考一次了。 En: You little pig, you finally learned to take an exam</td></tr><tr><td></td><td>0</td><td>REPORTIVE</td><td>care and expectation. News reporting, objective analysis, factual description, or neutral discus-</td><td>seriously. Zh：事故造成多人受伤，现场交通一度中断。 En: The accident injured several people, and traffic at</td></tr><tr><td></td><td></td><td>NEGATIVE NEGATIVE</td><td>sion of negative events. Direct anger, blame, rejection, disap-</td><td>the scene was temporarily suspended. Zh：蠢货，给我滚出这里。</td></tr><tr><td></td><td></td><td></td><td>pointment, anxiety, sadness, disgust, or dissatisfaction.</td><td>En: You idiot, get out of here.</td></tr></table>

Table 7: Definitions and examples of the nine Affective Stances. Each stance is determined by the composition of explicit affective signal $e _ { i }$ and implicit affective tendency $h _ { i }$

## G.2 Prompting Protocol

Zero-shot prompting includes label definitions and a constrained output format. Few-shot prompting adds demonstrations covering aligned affect, neutral-surface latent affect, and contrastive affect. Free-form CoT prompting asks the model to explain its reasoning before giving the final label, while matrix-guided prompting fixes the reasoning order from explicit signal to implicit tendency, Affective Stance, Pragmatic Intent, and Fine-grained Emotion. All prompts are evaluated with deterministic decoding when the API or model interface supports it.

## G.3 Prediction Normalization

Model outputs are normalized before scoring. We strip formatting artifacts, map aliases to canonical label names, and reject outputs that cannot be matched to the task label space. When a response contains both rationale text and a final answer, only the final structured label field is used for metric computation. This prevents verbose reasoning from being treated as additional labels.

## G.4 Oracle-conditioning Settings

For the ablation study, gold intermediate labels are inserted into the prompt while the target label remains hidden. Task I provides gold Affective Stance and predicts Pragmatic Intent. Task II provides gold Affective Stance and predicts Finegrained Emotion. Task III provides both gold Affective Stance and gold Pragmatic Intent and predicts Fine-grained Emotion.