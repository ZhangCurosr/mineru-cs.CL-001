# Multimodal Rapport Estimation in Real-World HRI

Akihiro Sakuramoto   
Japan Advanced Institute of Science   
and Technology   
Nomi, Ishikawa, Japan   
s2510069@jaist.ac.jp   
Takato Hayashi   
Japan Advanced Institute of Science   
and Technology   
Nomi, Ishikawa, Japan   
hayashi0884@jaist.ac.jp   
Yuki Okafuji   
CyberAgent   
Tokyo, Japan   
The University of Osaka   
Osaka, Japan   
okafuji\_yuki\_xd@cyberagent.co.jp   
Ryo Miyoshi   
CyberAgent   
Tokyo, Japan   
The University of Osaka   
Osaka, Japan   
miyoshi\_ryo@cyberagent.co.jp   
Shogo Okada   
Japan Advanced Institute of Science   
and Technology   
Nomi, Ishikawa, Japan   
okada-s@jaist.ac.jp

![](images/eaec1bbb6fe041ff70c445009c63728518d9a709d19dc1a4f9fae3fdd5961ee8.jpg)

![](images/21ae9af652cecfd6194a16e918e5df952168d610fa639851b82c6e843a0d0f7e.jpg)

![](images/5deb4c78c8428c90e13f2e1ea780a3d0e45522fb1eb95cea4f29aeab07c7fb91.jpg)  
Figure 1: Robot-view examples from the real-world HRI corpus: (a) a single-participant interaction, (b) a two-participant interaction, and (c) a three-participant interaction.

## Abstract

Evaluating interaction quality in real-world HRI is an important challenge. If interaction quality can be estimated reliably, the results can be used to improve dialogue strategies and ultimately enable robots to adapt their behavior autonomously. However, existing automatic evaluation methods have been developed primarily in controlled laboratory settings, and it remains unclear whether they can be directly applied to real-world environments, where users are free to disengage and multi-party participation may arise nat urally. In this study, we investigate the automatic estimation of third-party-rated rapport scores using 62 sessions of multimodal recordings collected in a Japanese drugstore. We compare zeroshot LLMs, pretrained text, audio, and visual models, and their prediction-level fusion. The results show that, in real-world HRI, zero-shot LLMs achieve strong performance, while audio and visual models tend to provide complementary information. In particular, Gemini 2.5 Flash performs strongly as a single model, and a fusion model combining Gemini (text) with HuBERT and V-JEPA performs best overall. Further analyses showed that estimation performance varied across interaction-duration and group-size conditions. These findings suggest that rapport estimation in real-world HRI requires evaluation and model design that account for contextual variability beyond that assumed in laboratory settings.

![](images/c412d6f47366c2a19a79141d596a3b9c134bbb5268b046f358bfe815eb1ed6b7.jpg)

## CCS Concepts

• Human-centered computing → Empirical studies in HCI; Empirical studies in collaborative and social computing; • Computing methodologies → Discourse, dialogue and pragmatics; Neural networks.

## Keywords

human-robot interaction, real-world human-robot interaction, rapport estimation, multimodal late fusion, large language models, third-party annotation

## ACM Reference Format:

Akihiro Sakuramoto, Takato Hayashi, Ryo Miyoshi, Yuki Okafuji, and Shogo Okada. 2026. Multimodal Rapport Estimation in Real-World HRI. In IN-TERNATIONAL CONFERENCE ON MULTIMODAL INTERACTION (ICMI ’26), October 05–09, 2026, Napoli, Italy. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3776574.3831184

## 1 Introduction

Social robots are increasingly being deployed in real-world environments such as commercial facilities and public spaces (e.g., [8, 17]). In such settings, spontaneous interactions arise in which passersby initiate engagement with a robot without prior preparation or instruction [16]. In these interactions, users are free to leave at any time and the timing of conversation onset and termination is unconstrained. Furthermore, multiple participants may naturally join the conversation. This uncontrolled nature creates conditions fundamentally diferent from laboratory settings, where experimenters can regulate participant behavior and conversational flow, and introduces unique challenges for evaluating interaction quality [7].

Evaluating and modeling interaction quality in such real-world settings is therefore an important research challenge. If interaction quality can be reliably estimated, the resulting estimates can inform improvements to dialogue strategies and, ultimately, enable robots to autonomously adapt their behavior. However, much prior work on interaction-quality modeling and evaluation has been developed and studied under controlled laboratory conditions, and it remains unclear whether such findings generalize to real-world deployments where user disengagement is unconstrained [7].

As a first step toward addressing this challenge, this study focuses on rapport as a measure of interaction quality. Rapport is a concept reflecting the relational quality of an interaction, and the Connection-Coordination Rapport (CCR) Scale has recently been proposed as a rapport measure specific to human-robot interaction (HRI) [12]. We construct a dataset of human-robot interactions collected in a real retail environment—specifically, a drugstore in Japan—where a remotely operated (Wizard of Oz) social robot was placed near the store entrance, allowing customers to freely initiate and terminate interactions. Third-party CCR annotations were applied to this dataset, and we establish a multimodal baseline model for automatic rapport estimation. To the best of our knowledge, this is one of the first studies to automatically estimate third-party-rated rapport from multimodal recordings of WoZ-mediated real-world HRI in an uncontrolled retail environment.

The contributions of this study are threefold:

• We construct one of the first real-world HRI datasets with rapport annotation, collected in an uncontrolled retail environment.

• We establish a multimodal baseline for automatic rapport estimation in WoZ-mediated real-world HRI and show that fusing zero-shot large language model (LLM) predictions with embedding-based models yields the best performance on this dataset.

• We present an empirical analysis of rapport in WoZ-mediated real-world HRI, ofering insights into the characteristics of naturally occurring human-robot interactions, such as users’ free entry and exit and multi-party participation, that distinguish them from controlled laboratory settings.

## 2 Related Work

## 2.1 Rapport for Interaction Quality in HRI

Rapport refers to the quality of the relationship that emerges between interaction partners during an interaction. Tickle-Degnen and Rosenthal [23] conceptualized rapport as a dynamic structure consisting of three components: mutual attentiveness, positivity, and coordination. Rather than a stable individual personality trait, rapport is understood as a dyadic property that emerges through interaction.

Rapport is important because it is not merely an impressionbased judgment, but is closely tied to successful relationship building and interaction. In interpersonal interaction research, rapport has been linked to outcomes such as improved learning in educational settings and successful negotiation (e.g., [14, 22]). However, previous rapport scales have primarily relied on first-person evaluation [3, 4, 18].

Lin et al. [12] proposed the Connection-Coordination Rapport (CCR) Scale for HRI, which enables rapport assessment from a third-person perspective. In HRI data, the CCR Scale exhibits a two-factor structure in which items related to mutual attentiveness and coordination cluster under the Coordination factor, whereas items related to interpersonal warmth, including positivity, cluster under the Connection factor. Furthermore, Lin et al. [11] proposed an eight-item reduced-length CCR Scale to reduce response burden and administration time and demonstrated its reliability and validity for third-party video-based assessment. In this study, we use the reduced-length CCR Scale, which is suitable for third-party annotation, to measure the relational quality of interaction in real-world HRI.

## 2.2 Subjective Evaluation of Interaction Quality in HRI

Evaluating interaction quality is a central challenge in HRI. In non-task-oriented human–agent and human–robot interactions, objective metrics such as task success or dialogue length are often insuficient, and subjective or relational criteria such as satisfaction, impression, enjoyment, and rapport have therefore been used to assess interaction quality [19, 21, 24–26].

Wei et al. modeled dialogue-level user satisfaction [26] and further showed that multitask learning can leverage the relationship between exchange-level sentiment and dialogue-level impression [25]. In controlled human–agent interactions, Cerekovic et al. [2] predicted both self-reported and observer-rated rapport from audiovisual social cues and personality measures. In HRI, Pereira et al. [19] examined LLM-based enjoyment detection, while Santana et al. [21] predicted self-reported enjoyment from pretrained audio and text embeddings. In a related multimodal dialogue setting, Wei et al. [24] investigated dialogue-level rapport recognition.

However, most of this work has been conducted in laboratory or online settings where interaction timing, participant roles, and conversational structure are relatively controlled. In real-world HRI, by contrast, interaction onset and termination, duration, and the number of participants are often not fixed, and public-space interactions are shaped by surrounding people and environmental contingencies [1, 7, 16]. Field studies in retail and public environments show that social robots can be deployed in practice [8, 17]. However, automatic estimation of interaction quality in real-world environments remains underexplored. Our study addresses this gap by estimating third-party CCR rapport scores from multimodal data collected in a real-world retail setting.

## 3 Dataset with Rapport Annotation

## 3.1 Real-World HRI Corpus

In this study, we used an interaction dataset recorded at a drugstore (selling cosmetics, daily goods, and medicine) in Japan, where a teleoperated service robot provided store guidance and casual conversation to visiting customers. The robot was operated for a total of 32 hours over six days, resulting in 131 recorded sessions. The data were collected with approval from an institutional ethics committee.

The robot used in the HRI sessions was Sota (produced by Vstone)<sup>1</sup>, a tabletop humanoid robot approximately 0.3 m tall. Sota is equipped with a built-in microphone, and a camera was mounted behind it, enabling the operator to monitor both audio and video in real time while controlling the robot’s speech and gestures. The operator’s voice was captured via a microphone, converted into a robot-like voice, and played through the robot’s speaker, while gestures were controlled using a controller. The robot’s gaze was controlled using an eye-tracking device. Further details regarding data collection are provided by Yano et al. [27].

From the 131 recorded sessions, we applied a series of filters to retain sessions suitable for rapport annotation. Specifically, we excluded sessions involving four or more participants, as large groups frequently engaged in side conversations among themselves independent of the robot, making it dificult to isolate and reliably assess human-robot rapport. We also excluded sessions in which a preschool-aged child (under approximately 6 years of age) was present, as such participants rarely engaged in verbal communication with the robot and exhibited interaction styles markedly diferent from those of other visitors; we consider this population a subject for independent future investigation. Finally, sessions containing fewer than two user utterances across the entire session were excluded, as interactions of such limited verbal content were deemed insuficient for meaningful rapport assessment.

After applying these criteria, the dataset was reduced to 62 sessions comprising 101 participants in total. Excluding the four who produced no usable speech, the remaining 97 participants were included as analysis targets. On average, each analyzed participant produced 7.06 (�� = 5.83) utterances. At the session level, each session contained an average of 11.85 (�� = 9.08) utterances with a mean video duration of 54.23 (�� = 42.42) seconds (see Table 1).

## 3.2 Rapport Annotation

Third-party rapport annotations were assigned to the human-robot interactions described above. Rapport was measured using the eightitem Connection-Coordination Rapport Scale (CCR-8) [11], a shortform rapport scale for HRI whose reliability and validity have been demonstrated for third-party video-based assessment. The scale comprises two four-item factors: Connection and Coordination. Each item was rated on a five-point Likert scale ranging from 1 (strongly disagree) to 5 (strongly agree). The overall rapport score was computed as the mean of all eight items. As the CCR-8 was originally developed in English, the items were translated into Japanese through careful discussion.

Table 1: Dataset statistics after filtering.
<table><tr><td>Statistic</td><td>M ± SD</td></tr><tr><td>Individual level (n = 97)</td><td></td></tr><tr><td># Utterances</td><td>7.06 ± 5.83</td></tr><tr><td>Session level (n = 62)</td><td></td></tr><tr><td>Video duration (seconds)</td><td>54.23 ± 42.42</td></tr><tr><td># Utterances</td><td>11.85 ± 9.08</td></tr><tr><td>Session composition</td><td></td></tr><tr><td># Individual sessions</td><td>28</td></tr><tr><td># Multi-participant sessions</td><td>34</td></tr><tr><td>Rapport scores (individual level, 3 annotators, n = 97)</td><td></td></tr><tr><td>Score</td><td>3.72 ± 0.80</td></tr><tr><td>ICC(2,3)</td><td>.85</td></tr><tr><td>Cronbach&#x27;s α</td><td>.95</td></tr><tr><td>Rapport scores (group level, 3 annotators, n = 34)</td><td></td></tr><tr><td>Score</td><td>3.71 ± 0.81</td></tr><tr><td>ICC(2,3)</td><td>.86</td></tr><tr><td>Cronbach&#x27;s α</td><td>.96</td></tr></table>

Three third-party annotators watched the interaction videos and provided rapport ratings. All three annotators were Japanese speakers and had no conflict of interest with respect to our study; one was a professional service staf member, providing an evaluative perspective grounded in real-world customer interaction experience. Annotators were trained using the CCR-8 item definitions and example videos prior to the main annotation phase. In addition, to supplement the item definition of each of the eight items, example behavioral criteria adapted to the real-world service setting were established in advance through discussion between the researchers and the annotators and shared with all annotators.

Rapport was assessed at the individual level: each annotator rated the rapport that each individual participant appeared to hold toward the robot, regardless ofwhether the session involved a single participant or multiple participants. For the 34 multi-participant sessions, group-level ratings were additionally collected to capture the rapport of the group as a whole toward the robot; however, only the individual-level ratings are used in our experiments.

## 3.3 Annotation Statistics

The final CCR score used in all analyses was the mean of the three annotators’ ratings. Before using the aggregate CCR-8 score, we assessed the internal consistency of its items using Cronbach’s �. At the individual level, using item scores averaged across the three annotators, � was .93, .92, and .95 for the Connection factor, Coordination factor, and overall rapport score, respectively. These high values indicate strong internal consistency. Together with the scoring procedure specified in the original CCR-8 paper [11], they support the use of the aggregate rapport score in the present analyses.

Inter-rater reliability of the rapport score was then estimated using two-way absolute-agreement intraclass correlation coeficients (ICC) [9]. Because the analyses relied on averaged ratings, average-measures ICC(2,3) is reported as the primary reliability statistic. At the individual level, ICC(2,3) for the overall rapport score was .85; at the group level, it was .86. Both values indicate good agreement among the three annotators, justifying the use of their mean rating as a stable and reliable measure of rapport. All subsequent analyses therefore use the mean of the three annotators’ individual-level CCR ratings as the rapport measure.

Descriptive statistics for the annotated rapport scores are summarized in Table 1. The individual-level rapport score yielded $M = 3 . 7 2$ $( S D = 0 . 8 0 )$ , and the group-level score yielded $M = 3 . 7 1 \left( S D = 0 . 8 1 \right)$ Individual-level rapport scores ranged from 1.38 to 4.88. Illustrative low-rapport interactions included cases in which a participant ignored the robot’s prompts or walked away mid-utterance, whereas high-rapport interactions included participants who actively asked questions and sustained the exchange with laughter.

## 4 Rapport Estimation Model

## 4.1 Task Definition

Let $\mathcal { I } = \{ I _ { 1 } , I _ { 2 } , \ldots , I _ { N } \}$ denote a set of � interactions. Each interaction $I _ { i }$ may involve one or more participants. For each participant in an interaction, we create an individual instance consisting of their observed features and corresponding rapport score.

For each instance, we observe a participant’s time series of feature vectors for a given modality � ∈ {text, audio, visual}:

$$
\mathbf { X } ^ { ( m ) } = \bigl ( \mathbf { x } _ { 1 } ^ { ( m ) } , \mathbf { x } _ { 2 } ^ { ( m ) } , \ldots , \mathbf { x } _ { T } ^ { ( m ) } \bigr ) ,\tag{1}
$$

where� denotes the sequence length. Each embedding-based model receives a single-modality input $\mathbf { X } ^ { ( m ) }$ . The estimation target is a scalar rapport score � ∈ R for that participant. The goal of the rapport estimation task is to learn a function

$$
f _ { \theta } : { \bf X } ^ { ( m ) } \longmapsto \hat { y } ,\tag{2}
$$

parameterized by $\theta ,$ such that $\hat { y }$ approximates �.

Because rapport scores are collected from human annotators, absolute score values inevitably contain annotator-specific noise arising from individual diferences in rating style and scale usage. We therefore seek to capture the linear association between predicted and target scores while penalizing discrepancies in their means and variances. Accordingly, we adopt the Concordance Correlation Coeficient (CCC) [10] as the basis for the training objective. CCC is defined as

$$
\rho _ { c } = \frac { 2 \sigma _ { \hat { y } y } } { \sigma _ { \hat { y } } ^ { 2 } + \sigma _ { y } ^ { 2 } + ( \bar { \hat { y } } - \bar { y } ) ^ { 2 } } ,\tag{3}
$$

where $\sigma _ { \hat { y } y }$ is the covariance between predictions and targets, $\sigma _ { \hat { y } } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ are their respective variances, and $\bar { \hat { y } } ,$ , �¯ are their means. CCC jointly captures Pearson correlation (linear association) and mean/variance agreement (absolute calibration), with $\rho _ { c } \in \left[ - 1 , 1 \right]$ and $\rho _ { c } = 1$ indicating perfect agreement. The training loss is defined as

$$
\mathcal { L } = 1 - \rho _ { c } ,\tag{4}
$$

which is minimized when predictions are maximally concordant with the ground-truth rapport scores.

## 4.2 Pretrained Embedding-Based Feature Extraction

4.2.1 Preprocessing. We first prepared aligned multimodal data for each participant through automatic annotation followed by manual correction.

Participant Bounding Boxes. Participant bounding boxes were detected using a DETR-based person detection model from DEIMv2- Wholebody34 [6]. Bounding boxes were extracted for each frame to localize individual participants throughout the interaction.

Transcription and Speaker Identification. Speech transcription was performed using Whisper-large [20], which automatically segmented the audio and generated utterance-level transcripts with timestamps. Each utterance was then manually annotated with a participant ID, allowing us to link transcripts, audio segments, and video clips to individual participants.

While some steps involved manual annotation (e.g., speaker ID assignment), our focus in this work is on rapport estimation rather than fully automatic preprocessing. Developing end-to-end automatic methods for participant tracking and identification remains an important direction for future work.

4.2.2 Feature Extraction. We extract fixed representations for each participant using frozen pretrained models across three modalities: text, audio, and visual. Text and audio representations are extracted at the utterance level, taking as input the transcribed utterance text and the utterance-segmented speech waveform, respectively. Visual representations are extracted at the clip level.

Text. Text representations were computed at the utterance level using Sentence-T5-large [15] (hereafter ST5). Each transcribed utterance text was encoded into a 768-dimensional embedding, which was then L2-normalized.

Audio. Audio representations were computed at the utterance level using HuBERT-large-ll60k [5]. Speech segments were first isolated by cutting the recording at utterance boundaries derived from transcript timestamps. Each resulting waveform segment was encoded into a 1024-dimensional embedding, which was then L2- normalized.

Visual. Visual representations were computed at the clip level using V-JEPA 2.1 ViT-Gigantic [13] (hereafter V-JEPA). Videos were sampled at approximately 8 fps and divided into non-overlapping 64-frame clips, producing a sequence of 1664-dimensional features from each clip. The features were then L2-normalized. When targetuser bounding boxes were available, weighted spatial pooling was applied to focus on the target participant. The resulting temporal feature sequence was aggregated using the additive attention pooling described below.

## 4.3 Embedding-Based Models

Each modality stream shares the same two-stage architecture, based on the model proposed by Santana et al. [21]: an additive attention pooling module that aggregates the variable-length feature sequence into a single interaction embedding, followed by a prediction head that regresses the rapport score from that embedding. A separate model is trained for each modality (text, audio, and visual).

4.3.1 Additive Atention Pooling. Given the feature sequence ${ \bf X } _ { i } ^ { ( m ) } = { \bf \Psi }$ $( \mathbf { x } _ { i , 1 } ^ { ( m ) } , \ldots , \mathbf { x } _ { i , T _ { i } } ^ { ( m ) } )$ ) of length �<sub>�</sub> for interaction $I _ { i } ,$ a learned linear scoring function computes a scalar score for each step:

$$
\begin{array} { r } { s _ { i , t } = \mathbf { w } ^ { \top } \mathbf { x } _ { i , t } ^ { ( m ) } + b , } \end{array}\tag{5}
$$

where w $\epsilon \mathbb { R } ^ { d }$ and $b \in \mathbb { R }$ are modality-specific learnable parameters. Padding positions are masked before the softmax, and the attention weights are computed as

$$
\alpha _ { i , t } = \frac { \exp ( s _ { i , t } ) } { \sum _ { t ^ { \prime } = 1 } ^ { T _ { i } } \exp ( s _ { i , t ^ { \prime } } ) } .\tag{6}
$$

The interaction embedding is then obtained as the weighted sum

$$
\mathbf { v } _ { i } = \sum _ { t = 1 } ^ { T _ { i } } \alpha _ { i , t } \mathbf { x } _ { i , t } ^ { ( m ) } \in \mathbb { R } ^ { d } .\tag{7}
$$

4.3.2 Prediction Head. The interaction embedding v<sub>�</sub> is passed to a two-layer MLP:

$$
\hat { y } _ { i } = \mathbf { W } _ { 2 } \operatorname { D r o p o u t } ( \operatorname { R e L U } ( \mathbf { W } _ { 1 } \mathbf { v } _ { i } + \mathbf { b } _ { 1 } ) ) + b _ { 2 } ,\tag{8}
$$

where Dropout with rate 0.2 is applied after the activation. The output �ˆ<sub>�</sub> is a scalar predicted on the z-score scale, which is inversetransformed to the original rapport score scale before evaluation.

## 4.4 Zero-Shot LLMs

We evaluated three LLMs as zero-shot prompt-based comparators: GPT-5.4, Claude Sonnet 4.6, and Gemini 2.5 Flash. Each model was prompted to act as a third-party annotator and rate the eight CCR-8 items on the same 5-point scale used by human annotators. As with the human annotation procedure, the prompt (written in Japanese) included, for each item, the item definition and example behavioral criteria described in Section 3.2. The model returned a structured JSON object, which was parsed into item scores and averaged to obtain the predicted rapport score for each participant. Depending on the model’s multimodal capabilities, we varied the input modalities across three conditions: text-only (T), text+audio (T+A), and text+audio+visual (T+A+V). In all conditions, the transcript with speaker-ID prefixes was provided as the text input. In the T+A condition, the corresponding wav file was additionally supplied. In the T+A+V condition, the video with per-speaker bounding boxes was further included to provide visual context. Gemini models were queried using the API-default thinking configuration; Claude was queried without extended thinking; GPT-5.4 was queried with reasoning efort none. All LLM predictions were obtained through the respective public APIs in April 2026, with a single run per condition.

## 4.5 Late Fusion

To combine predictions from multiple models, we use unweighted averaging of the prediction scores. We also examined weighted averaging, but its performance did not difer substantially from unweighted averaging; we therefore adopted the equal-weight scheme, which requires no additional parameters.

## 4.6 Evaluation Procedure

4.6.1 Data Splits. The embedding-based models and the Random Baseline were evaluated under the same predefined 30-fold train/val/test split. The splits were constructed at the session level, such that all

participants from the same session are assigned to the same set, ensuring no data leakage across splits. In each fold, the held-out test set was excluded from all stages of training, including model selection and early stopping. A validation set sampled from the remaining non-test data was used for the same purposes. Final metrics were computed globally by pooling the out-of-fold test predictions across all folds, rather than averaging per-fold metrics. For the Random Baseline, test predictions in each fold were sampled with replacement from the target-score distribution of the corresponding training partition; this procedure was repeated with 100 random seeds, and the table reports the mean metrics across seeds.

4.6.2 Metrics. We report three complementary regression metrics. Mean Absolute Error (MAE) measures the average magnitude of prediction errors in the original score scale. Pearson correlation coeficient (PCC) captures the linear association between predicted and ground-truth scores. Concordance Correlation Coeficient (CCC; see Section 4.1) jointly accounts for both correlation and mean/variance agreement, and serves as the primary metric consistent with the training objective. Lower MAE and higher PCC/CCC indicate better performance.

Use of AI Tools. Parts of the experiment, analysis, and figuregeneration code used in this study were developed with the assistance of generative AI coding tools (Claude, Anthropic; Codex, OpenAI).

## 5 Experimental Results

## 5.1 Rapport Estimation Performance

Results are summarized in Table 2.

Table 2: Comparison of CCR rapport estimation performance in real-world HRI. The best value for each metric is shown in bold with underline.

<table><tr><td>Model</td><td>Modality</td><td>MAE↓</td><td>PCC ↑</td><td>CCC↑</td></tr><tr><td>Baseline</td><td></td><td></td><td></td><td></td></tr><tr><td>Random Baseline</td><td></td><td>0.901</td><td>-0.004</td><td>-0.005</td></tr><tr><td>Zero-shot LLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.4</td><td>T</td><td>0.617</td><td>0.644</td><td>0.555</td></tr><tr><td>Claude Sonnet 4.6</td><td>T</td><td>0.741</td><td>0.619</td><td>0.477</td></tr><tr><td>Gemini 2.5 Flash</td><td>T</td><td>0.634</td><td>0.665</td><td>0.580</td></tr><tr><td>Gemini 2.5 Flash</td><td>T+A</td><td>0.592</td><td>0.602</td><td>0.550</td></tr><tr><td>Gemini 2.5 Flash</td><td> $\mathrm { T } { + } \mathrm { A } { + } \mathrm { V }$ </td><td>0.549</td><td>0.625</td><td>0.618</td></tr><tr><td>Pretrained Embeddings</td><td></td><td></td><td></td><td></td></tr><tr><td>ST5</td><td>T</td><td>0.633</td><td>0.327</td><td>0.281</td></tr><tr><td>HuBERT</td><td>A</td><td>0.616</td><td>0.464</td><td>0.460</td></tr><tr><td>V-JEPA</td><td>V</td><td>0.666</td><td>0.331</td><td>0.310</td></tr><tr><td>Late Fusion (unweighted average)</td><td></td><td></td><td></td><td></td></tr><tr><td>ST5+HuBERT</td><td>T+A</td><td>0.565</td><td>0.514</td><td>0.444</td></tr><tr><td>ST5+V-JEPA</td><td>T+V</td><td>0.603</td><td>0.443</td><td>0.341</td></tr><tr><td>HuBERT+V-JEPA</td><td> $_ { \mathrm { A + V } }$ </td><td>0.542</td><td>0.526</td><td>0.465</td></tr><tr><td>ST5+HuBERT+V-JEPA</td><td> $\mathrm { T } { + } \mathrm { A } { + } \mathrm { V }$ </td><td>0.540</td><td>0.567</td><td>0.444</td></tr></table>

Zero-shot LLMs demonstrated strong performance in real-world rapport estimation. In terms of CCC, Gemini 2.5 Flash (T+A+V) achieved the highest score in the table (0.618), while, notably, Gemini 2.5 Flash (T) attained the highest PCC (0.665). Furthermore, the second-best PCC result, GPT-5.4 (T) = 0.644, also relied solely on text input, suggesting that the text modality alone carries substantial useful information for rapport estimation.

For Gemini 2.5 Flash (T), the subscale CCCs were 0.614 for Con nection and 0.473 for Coordination.

Embedding-based models exhibited considerable variation in performance across modalities. The text encoder ST5 (T) yielded the lowest PCC and CCC among all supervised models, highlighting the dificulty of rapport estimation from text embeddings alone. In contrast, the audio encoder HuBERT (A) outperformed all other supervised single-modality models across all three metrics.

When modalities were integrated via late fusion, none of the combinations surpassed the zero-shot LLMs in terms of PCC or CCC. Nevertheless, the fused models tended to achieve lower MAE values, with ST5+HuBERT+V-JEPA (T+A+V) attaining the lowest MAE in the table.

## 5.2 Complementarity Analysis

The results in Section 5.1 show that multimodal LLMs such as Gemini 2.5 Flash (T+A+V) achieved the highest CCC, and Gemini 2.5 Flash (T) delivered competitive performance across all metrics using text input alone. However, multimodal LLMs that directly process audio and video may introduce greater inference latency and monetary cost than text-only LLMs. We therefore focus on Gemini 2.5 Flash (T) as the primary predictor and investigate whether embedding-based audio and visual models can complement it. Results are visualized in Figure 2.

Figure 2 shows a utility–redundancy map in which the vertical axis represents each model’s correlation with the ground-truth rapport score (standalone utility), the horizontal axis represents its correlation with Gemini (T) predictions (redundancy), and marker size encodes the incremental explanatory power $\Delta R ^ { 2 }$ when added to Gemini (T). Models in the upper-left region are therefore both informative and non-redundant with respect to Gemini (T), making them natural candidates for fusion.

Among the embedding-based models, ST5+HuBERT+V-JEPA (T+A+V) and HuBERT+V-JEPA (A+V) occupied the upper-left region ofthe map. Because the fusion analysis replaces the text branch with Gemini (T), the natural complementary candidates are the audio and visual streams that Gemini (T) cannot access; HuBERT+V-JEPA (A+V) combines both non-text modalities while occupying this informative and non-redundant region. We therefore focus on HuBERT+V-JEPA (A+V) as the complementary model. While we focus on A+V, for completeness we also evaluate fusion with each of HuBERT and V-JEPA individually.

To quantify complementarity, we employ two metrics. The partial correlation $r ( y , \hat { y } ^ { A V } \mid \hat { y } ^ { G } )$ measures the association between HuBERT+V-JEPA (A+V) predictions and the ground-truth rapport score after partialling out the variance explained by Gemini (T). The incremental $R ^ { 2 }$ is defined as

$$
\Delta R ^ { 2 } = R ^ { 2 } ( y \sim \hat { y } ^ { G } + \hat { y } ^ { A V } ) - R ^ { 2 } ( y \sim \hat { y } ^ { G } ) ,\tag{9}
$$

where ${ \hat { y } } ^ { G }$ and $\hat { \boldsymbol { y } } ^ { A V }$ denote the predictions ofGemini (T) and HuBERT+V-JEPA (A+V), respectively. For HuBERT+V-JEPA (A+V), we obtained $r ( y , \hat { y } ^ { A V } \mid \hat { y } ^ { G } ) \stackrel { - } { = } 0 . 3 5 9$ and $\Delta R ^ { 2 } = 0 . 0 7 2$ . These results indicate that, although HuBERT+V-JEPA $( \mathsf { A } { + } \mathsf { V } )$ does not match Gemini (T) in standalone performance, it captures rapport-relevant signals that Gemini (T) alone does not account for, and thus has the potential to improve estimation quality when combined via fusion.

![](images/53dca93a04b6c4b5fbfc352713a095b6d6aa6cb6e4e648352955d80ac7e977e3.jpg)  
Figure 2: Utility–redundancy map relative to Gemini 2.5 Flash (T). Point size indicates incremental explanatory value over Gemini (T).

## 5.3 Fusion of LLM with Embedding-Based Models

Table 3 presents the results of fusing Gemini 2.5 Flash (T) with embedding-based models via unweighted prediction-level averaging: each configuration averages the Gemini (T) prediction with one complementary embedding-based predictor. For the T+A+V configuration, the complementary predictor is the equal-weight average of the HuBERT and V-JEPA predictions (A+V) analyzed in Section 5.2, so the efective weights are 1/2 for Gemini (T) and 1/4 each for HuBERT and V-JEPA. All three fusion configurations performed better than the reference baselines across all three metrics.

Table 3: Unweighted prediction-level fusion of Gemini 2.5 Flash (T) with embedding-based models. The best value for each metric is shown in bold with underline.
<table><tr><td>Model</td><td>Modality</td><td>MAE↓</td><td>PCC ↑</td><td>CCC ↑</td></tr><tr><td colspan="5">Reference baselines (from Table 2)</td></tr><tr><td>Gemini 2.5 Flash</td><td>T+A+V</td><td>0.549</td><td>0.625</td><td>0.618</td></tr><tr><td>ST5+HuBERT+V-JEPA</td><td>T+A+V</td><td>0.540</td><td>0.567</td><td>0.444</td></tr><tr><td colspan="5">Unweighted-average fusion with Gemini 2.5 Flash (T)</td></tr><tr><td>+ HuBERT</td><td>T+A</td><td>0.507</td><td>0.660</td><td>0.629</td></tr><tr><td>+ V-JEPA</td><td>T+V</td><td>0.490</td><td>0.712</td><td>0.632</td></tr><tr><td>+ HuBERT+V-JEPA</td><td> $\mathrm { T } { + } \mathrm { A } { + } \mathrm { V }$ </td><td>0.471</td><td>0.717</td><td>0.656</td></tr></table>

The best-performing configuration, Gemini (T) + HuBERT+V-JEPA (T+A+V), achieved an MAE of 0.471, a PCC of 0.717, and a CCC of 0.656.

## 6 Discussion

This study demonstrates that rapport scores in real-world HRI can be estimated to a meaningful degree using both zero-shot LLMs and embedding-based models. In particular, a text-only LLM proved efective as a standalone predictor, and fusing it with audio-visual embedding-based models yielded the best overall performance.

![](images/b1e94ee90975f56940eb8a14634f1c480abd72ed4b68f47ccacafb9af072f05d.jpg)  
Ground-truth rapport score (1-5)  
Figure 3: Predicted versus ground-truth CCR rapport scores for three representative prediction conditions. Dashed lines indicate perfect prediction.

## 6.1 Comparison with Laboratory-Based HRI Studies

We primarily compare our results with Speech-to-Joy [21], a laboratorybased HRI study that served as a methodological reference for our experimental design. However, direct quantitative comparison should be interpreted with caution, as that work targets enjoyment rather than rapport and uses self-reported labels rather than the third-party annotations used in our study.

In our real-world setting, the LLM-based predictor outperforms the embedding-based late fusion model, and text embeddings alone prove weak, ofering little useful signal for estimation. This stands in contrast to the findings of [21], where the late fusion of text and audio embedding-based models (T+A) outperforms LLM-based approaches on correlation metrics (PCC and CCC), and text features prove nearly as informative as acoustic ones. The substantially shorter interaction durations in our deployment—our analyzed interaction units average 53.6 seconds, compared with approximately seven minutes in Speech-to-Joy—constitute one possible explanation for this diference, as shorter interactions may provide less rapport-related information for embedding representations. The complex dynamics of multi-party interactions may further hinder the performance of embedding-based models. In contrast, the LLM-based predictor showed strong performance under these challenging conditions.

## 6.2 Performance across Interaction-Duration Conditions

A defining characteristic ofreal-world HRI deployments, in contrast to laboratory settings, is that the timing and duration of interactions cannot be controlled in advance. Participants approach, engage, and disengage from the robot at their own discretion, resulting in substantial variability in interaction length. Among the 97 individuallevel interaction units analyzed in this study, interaction duration averaged 53.6 seconds (�� = 41.1) with a median of 40.0 seconds, ranging from as brief as 12 seconds to as long as 227 seconds. To examine estimation performance across interaction-duration conditions, we partition the interactions into short (≤40 seconds) and long (>40 seconds) conditions using the median as a threshold, and compare model performance across the two conditions.

Gemini 2.5 Flash (T) achieved a CCC of 0.563 in the short condition and 0.551 in the long condition, with a diference of 0.012 between the conditions. The text-only supervised predictor ST5 (T) increased from 0.168 to 0.411, a diference of 0.243. Among the unimodal audio and visual predictors, HuBERT (A) declined from 0.463 to 0.420, a decrease of 0.043, and V-JEPA (V) declined from 0.330 to 0.295, a decrease of 0.035.

Only the supervised text model ST5 (T) showed a pronounced performance diference between the two conditions, with a marked drop in the short condition. Gemini 2.5 Flash (T) exhibited the smallest variation across conditions, although HuBERT (A) and V-JEPA (V) also varied little.

## 6.3 Performance across Group-Size Conditions

A second factor that distinguishes real-world HRI from laboratory studies is the prevalence of multi-party interactions. Whereas laboratory studies can enforce strictly dyadic interactions, real-world deployments cannot constrain who chooses to join an ongoing interaction with the robot. Our dataset reflects this reality: of the 97 individual-level samples analyzed, 69 involved participants engaged in multi-party interactions. Specifically, 28 samples came from oneperson interactions, 56 from two-person interactions, and 13 from three-person interactions. Group size was determined from each session’s full set of participants, so members who are not among the 97 estimation targets—for example, those who produced little or no usable speech—still count toward the group size of their session. Mean ground-truth rapport was similar in the one- and two-person conditions but lower in the three-person condition (3.774, 3.775, and 3.391, respectively). Given the small three-person subset (� = 13), these descriptive diferences should be interpreted cautiously.

![](images/8681813372295b07e4a0b8272b30234dd2a663f8df8c5c1806f5f4572893db16.jpg)  
Figure 4: CCC trends across one-, two-, and three-participant interactions for each prediction model.

As shown in Figure 4, Gemini 2.5 Flash (T) consistently maintained high CCC across all group-size conditions. Interestingly, it achieved its highest performance in the three-person condition (CCC = 0.721), suggesting that richer multi-party discourse may actually provide the LLM with additional contextual cues that support rapport inference. In contrast, the supervised unimodal predictors— with the exception of ST5 (T)—exhibited a consistent decline as group size increased. In particular, the CCC of V-JEPA (V) decreased from 0.503 to 0.286 and then to 0.043.

Together with the findings in Section 6.2, these results indicate that the supervised embedding-based predictors showed greater performance variation than Gemini 2.5 Flash (T) in our data. The variation was particularly large across group-size conditions. Whether this pattern reflects diferences in the information available to the models or the limited training data available to the supervised models (Section 6.4) remains to be tested with larger datasets and additional LLMs.

## 6.4 Limitations and Future Work

This study has several limitations. First, the data were collected in a single real-world service setting—a drugstore in Japan—and the participants were mainly Japanese speakers within a Japanese cultural context. It therefore remains unclear whether similar patterns would appear in other countries, languages, cultures, settings, or robot embodiments. In addition, because the robot’s speech, gestures, and gaze were controlled by a human operator (WoZ), the rapport ratings may partly reflect the operator’s conversational and social abilities. Accordingly, our findings concern WoZ-mediated social robot interaction, and generalizing them to autonomous HRI requires caution.

Second, the estimation experiments were based on 97 individual cases. Although real-world HRI data are valuable, this sample remains small for robust machine-learning evaluation, and the additional analyses by interaction duration and number of participants relied on even smaller subsets. These findings should therefore be interpreted as exploratory. Moreover, the 97 targets were limited to cases with available audio and transcripts, meaning that encounters with little or no speech or very early disengagement were underrepresented. The results therefore reflect only interactions in which some degree of engagement occurred and the required data were available.

Third, the target variable was not participants’ self-reported rapport, but rapport judged by third parties from observable behavior and interaction content. Our findings therefore demonstrate the potential of third-party rapport estimation rather than direct estimation of participants’ own experienced rapport, enjoyment, or satisfaction. In addition, although the Japanese translation of the CCR-8 showed good inter-rater agreement and high internal consistency, we did not conduct a full psychometric validation of the Japanese version.

Fourth, the zero-shot LLMs and embedding-based predictors were not evaluated under strictly identical conditions because they difered in input format and inference framework. The comparison should therefore be treated as a practical baseline, not evidence that LLMs are inherently superior. Zero-shot LLM performance may also vary with prompts, model versions, API specifications, and input formats. Finally, although Gemini 2.5 Flash performed strongly, the linguistic and interactional cues underlying its rapport estimates remain unclear and should be examined in future work.

## 7 Conclusion

In this study, we examined the automatic estimation of third-partyrated rapport scores in real-world HRI. The zero-shot LLM (Gemini 2.5 Flash (T)) achieved strong predictive performance from text-only input. Fusing Gemini (T) with HuBERT and V-JEPA yielded the best overall performance, suggesting that LLMs and embedding-based predictors are complementary rather than competing components. Future work should develop models robust to variation in interaction duration and multi-party interaction.

## 8 Safe and Responsible Innovation Statement

This study uses multimodal data from real-world HRI to estimate third-party-rated rapport and therefore requires careful attention to privacy protection and responsible use. Such estimation does not directly measure an individual’s internal state and should not be used as the sole basis for evaluating or making decisions about individuals. Any deployment should restrict access to raw data, minimize retention, support de-identification, and provide clear notice and appropriate consent procedures.

## Acknowledgments

We thank Atsuko Harita for her contributions to data annotation. This work was partially supported by JSPS KAKENHI (No. 26K03016) and JST CREST (No. JPMJCR2563).

## References

[1] Atef Ben-Youssef, Chloé Clavel, Slim Essid, Miriam Bilac, Marine Chamoux, and Angelica Lim. 2017. UE-HRI: A New Dataset for the Study of User Engagement in Spontaneous Human-Robot Interactions. In Proceedings of the 19th ACM International Conference on Multimodal Interaction (ICMI ’17). 464–472. https://doi.org/10.1145/3136755.3136814

[2] Aleksandra Cerekovic, Oya Aran, and Daniel Gatica-Perez. 2017. Rapport with Virtual Agents: What Do Human Social Cues and Personality Explain? IEEE Transactions on Afective Computing 8, 3 (2017), 382–395. https://doi.org/10.1109/ TAFFC.2016.2545650

[3] Jonathan Gratch, David DeVault, Gale M. Lucas, and Stacy Marsella. 2015. Negotiation as a Challenge Problem for Virtual Humans. In Intelligent Virtual Agents

(Lecture Notes in Computer Science, Vol. 9238). Springer International Publishing, Cham, 201–215. https://doi.org/10.1007/978-3-319-21996-7\_21

[4] Jonathan Gratch, Ning Wang, Jillian Gerten, Edward Fast, and Robin Dufy. 2007. Creating Rapport with Virtual Agents. In Intelligent Virtual Agents (Lecture Notes in Computer Science, Vol. 4722). Springer Berlin Heidelberg, Berlin, Heidelberg, 125–138. https://doi.org/10.1007/978-3-540-74997-4\_12

[5] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. HuBERT: Self Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM Transactions on Audio, Speech, and Language Processing 29 (2021), 3451–3460. https://doi.org/10.1109/TASLP.2021.3122291

[6] Katsuya Hyodo. 2025. DEIMv2-Wholebody34: Lightweight Human Detection Models Generated on High-Quality Human Data Sets. https://github.com/PINTO0309/ PINTO\_model\_zoo/tree/598e8d5d928e1d5a54511f34f6abede378831c2c/472 DEIMv2-Wholebody34 Git commit 598e8d5d928e1d5a54511f34f6abede378831c2c; accessed 2026-07-13.

[7] Malte Jung and Pamela Hinds. 2018. Robots in the Wild: A Time for More Robust Theories ofHuman-Robot Interaction. ACMTransactions on Human-Robot Interaction 7, 1, Article 2 (May 2018), 5 pages. https://doi.org/10.1145/3208975

[8] Takayuki Kanda, Masahiro Shiomi, Zenta Miyashita, Hiroshi Ishiguro, and Nori hiro Hagita. 2010. A Communication Robot in a Shopping Mall. IEEE Transactions on Robotics 26, 5 (2010), 897–913. https://doi.org/10.1109/TRO.2010.2062550

[9] Terry K. Koo and Mae Y. Li. 2016. A Guideline of Selecting and Reporting Intraclass Correlation Coeficients for Reliability Research. Journal ofChiropractic Medicine 15, 2 (2016), 155–163. https://doi.org/10.1016/j.jcm.2016.02.012

[10] Lawrence I-Kuei Lin. 1989. A Concordance Correlation Coeficient to Evaluate Reproducibility. Biometrics 45, 1 (1989), 255–268. https://doi.org/10.2307/2532051

[11] Ting-Han Lin, Guan Chen, Bilge Mutlu, J. Gregory Trafton, and Sarah Sebo. 2026. The Reduced-Length Connection-Coordination Rapport (CCR) Scale. ACM Transactions on Human-Robot Interaction 15, 3, Article 57 (2026), 29 pages. https: //doi.org/10.1145/3796533

[12] Ting-Han Lin, Hannah Dinner, Tsz Long Leung, Bilge Mutlu, J. Gregory Trafton, and Sarah Sebo. 2025. Connection-Coordination Rapport (CCR) Scale: A Dual-Factor Scale to Measure Human-Robot Rapport. In Proceedings of the 2025 ACM/IEEE International Conference on Human-Robot Interaction (HRI ’25). Mel bourne, Australia, 869–879. https://doi.org/10.1109/HRI61500.2025.10974218

[13] Lorenzo Mur-Labadia, Matthew Muckley, Amir Bar, Mido Assran, Koustuv Sinha, Mike Rabbat, Yann LeCun, Nicolas Ballas, and Adrien Bardes. 2026. V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning. arXiv:2603.14482 [cs.CV] https://arxiv.org/abs/2603.14482

[14] Janice Nadler. 2004. Rapport in Negotiation and Conflict Resolution. Marquette Law Review 87, 4 (2004), 875–882. https://scholarship.law.marquette.edu/mulr vol87/iss4/25

[15] Jianmo Ni, Gustavo Hernandez Abrego, Noah Constant, Ji Ma, Keith Hall, Daniel Cer, and Yinfei Yang. 2022. Sentence-T5: Scalable Sentence Encoders from Pretrained Text-to-Text Models. In Findings ofthe Association for Computational Linguistics: ACL 2022. Association for Computational Linguistics, Dublin, Ireland, 1864–1874. https://doi.org/10.18653/v1/2022.findings-acl.146

[16] Sara Nielsen, Mikael B. Skov, Karl Damkjær Hansen, and Aleksandra Kaszowska. 2023. Using User-Generated YouTube Videos to Understand Unguided Interactions with Robots in Public Places. ACM Transactions on Human-Robot Interaction 12, 1, Article 5 (2023), 40 pages. https://doi.org/10.1145/3550280

[17] Marketta Niemelä, Päivi Heikkilä, Hanna Lammi, and Virpi Oksman. 2019. A Social Robot in a Shopping Mall: Studies on Acceptance and Stakeholder Expecta tions. In Social Robots: Technological, Societal and Ethical Aspects ofHuman-Robot Interaction. Springer, 119–144. https://doi.org/10.1007/978-3-030-17107-0\_7

[18] Tatsuya Nomura and Takayuki Kanda. 2016. Rapport–Expectation with a Robot Scale. International Journal ofSocial Robotics 8, 1 (2016), 21–30. https://doi.org/ 10.1007/s12369-015-0293-z

[19] André Pereira, Lubos Marcinek, Jura Miniota, Sofia Thunberg, Erik Lagerstedt, Joakim Gustafson, Gabriel Skantze, and Bahar Irfan. 2024. Multimodal User Enjoyment Detection in Human-Robot Conversation: The Power of Large Lan guage Models. In Proceedings of the 26th International Conference on Multimodal Interaction (ICMI ’24). 469–478. https://doi.org/10.1145/3678957.3685729

[20] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2023. Robust Speech Recognition via Large-Scale Weak Supervision. In Proceedings of the 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202). PMLR, 28492–28518. https://proceedings.mlr.press/v202/radford23a.html

[21] Ricardo Santana, Bahar Irfan, Erik Lagerstedt, Gabriel Skantze, and André Pereira. 2025. Speech-to-Joy: Self-Supervised Features for Enjoyment Prediction in Human-Robot Conversation. In Proceedings ofthe 27th International Conference on Multimodal Interaction (ICMI ’25). 238–248. https://doi.org/10.1145/3716553. 3750747

[22] Tanmay Sinha and Justine Cassell. 2015. We Click, We Align, We Learn: Impact of Influence and Convergence Processes on Student Learning and Rapport Building. In Proceedings ofthe 1st Workshop on Modeling INTERPERsonal SynchrONy And infLuence (INTERPERSONAL ’15). ACM, 13–20. https://doi.org/10.1145/2823513.

2823516

[23] Linda Tickle-Degnen and Robert Rosenthal. 1990. The Nature of Rapport and Its Nonverbal Correlates. Psychological Inquiry 1, 4 (1990), 285–293. https: //doi.org/10.1207/s15327965pli0104\_1

[24] Wenqing Wei, Sixia Li, Candy Olivia Mawalim, Xiguang Li, Kazunori Komatani, and Shogo Okada. 2025. Influence of Personality Traits and Demographics on Rapport Recognition Using Adversarial Learning. Multimodal Technologies and Interaction 9, 3, Article 18 (2025). https://doi.org/10.3390/mti9030018

[25] Wenqing Wei, Sixia Li, and Shogo Okada. 2022. Investigating the Relationship Between Dialogue and Exchange-Level Impression. In Proceedings ofthe 2022 International Conference on Multimodal Interaction (ICMI ’22). 359–367. https: //doi.org/10.1145/3536221.3556602

[26] Wenqing Wei, Sixia Li, Shogo Okada, and Kazunori Komatani. 2021. Multimodal User Satisfaction Recognition for Non-task Oriented Dialogue Systems. In Proceedings ofthe 2021 International Conference on Multimodal Interaction (ICMI ’21). 586–594. https://doi.org/10.1145/3462244.3479928

[27] Yuga Yano, Yuki Okafuji, Ryo Miyoshi, Sanae Yamashita, and Yoshiki Ohira. 2026. Breaking the 15% Barrier: A Real-World Data-Driven System for Proactive Social Robot Triggered by User Nonverbal Cues. arXiv:2607.11633 [cs.RO] https://arxiv.org/abs/2607.11633 Accepted at the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2026).