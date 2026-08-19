# SpeechSense: A Paralinguistic-Focused Dataset for Fine-Grained Speech Sentiment Analysis

Shicheng Ma   
Department of Computer Science and   
Engineering   
The Chinese University of Hong Kong   
Hong Kong, Hong Kong   
shichengma@cuhk.edu.hk   
Wenqian Cui   
Department of Computer Science and   
Engineering   
The Chinese University of Hong Kong   
Hong Kong, Hong Kong   
wenqian.cui@link.cuhk.edu.hk   
Irwin King   
Department of Computer Science and   
Engineering   
The Chinese University of Hong Kong   
Hong Kong, Hong Kong   
king@cse.cuhk.edu.hk

## Abstract

Recent advances in AI have revolutionized speech processing, yet efective speech understanding requires discerning not just what is said, but how it is said. Speech Sentiment Analysis plays a critical role in decoding these paralinguistic cues for diverse real-world applications such as recruitment and customer service. However, existing Speech Sentiment Analysis research faces two primary limitations. First, dominant approaches rely on text-centric pipelines that cascade Automatic Speech Recognition with text analysis. This process inevitably discards essential acoustic features like prosody and tone, failing to capture attitudinal meanings in acoustically ambiguous utterances. Second, current benchmarks sufer from a mis match in label granularity, prioritizing basic emotions (e.g., happy, sad) over the nuanced interpersonal stances (e.g., confident, impatient) necessary for social sensitivity. To address these limitations, we propose a novel dataset, SpeechSense, for fine-grained speech sentiment analysis. Specifically, we define a specialized 8-class taxonomy of interpersonal stances detectable primarily through prosodic cues beyond lexical content alone. We then construct a curated dataset based on this taxonomy, built from high-fidelity speech synthesis and rigorous human validation. Comprehensive experiments across multi-modal LLMs, text-only LLMs, and speech encoders demonstrate that models with acoustic access consistently outperform text-only baselines. These results empirically validate the primacy of acoustic cues in detecting subtle speaker attitudes, highlighting the necessity of SpeechSense.<sup>1</sup>

## CCS Concepts

• Information systems → Sentiment analysis; • Computing methodologies → Language resources; Neural networks; • Ap plied computing → Sound and music computing.

## Keywords

Speech Sentiment Analysis, Paralinguistics, Multi-modal Large Language Models, Synthetic Data

ACM Reference Format:   
Shicheng Ma, Wenqian Cui, and Irwin King. 2026. SpeechSense: A Paralinguistic-Focused Dataset for Fine-Grained Speech Sentiment Analysis. In Proceedings ofthe 34th ACM International Conference on Multimedia (MM ’26), November 10–14, 2026, Rio de Janeiro, Brazil. ACM, New York, NY, USA, 7 pages. https://doi.org/10.1145/3767308.3838667

## 1 Introduction

Recent advances in Artificial Intelligence (AI) have dramatically improved machines’ ability to process and understand human speech, enabling applications ranging from automated transcription services [30] to end-to-end voice assistants [8, 9, 15, 39]. For efective speech understanding, AI models require not only recognizing what is said (linguistic information) but also discerning how it is said (paralinguistic information) [6, 8, 14], as paralinguistic information carries additional meanings beyond the spoken content. A notable task that involves paralinguistic understanding is Speech Sentiment Analysis (SSA), where acoustic cues are analyzed to decipher fine-grained speaker attitudes and afective states. This capability is especially critical in professional scenarios such as online recruitment, customer service, and healthcare [12, 26, 38]. For instance, distinguishing “confident” from “nervous” delivery during a job interview provides insights that speech content alone misses [10].

However, existing SSA research faces two key limitations. First, dominant approaches rely heavily on text-centric pipelines. In these systems, speech is first converted to text via Automatic Speech Recognition (ASR) before applying Text-based Sentiment Analysis (TSA) models [35]. This cascaded approach inevitably discards paralinguistic cues, including vocal prosody, tone, pauses, stress, and tempo [34]. Yet, these acoustic features are often the primary carriers of attitudinal meaning [13]. For example, the phrase “We can’t wait another decade” can express either passionate advocacy or impatient irritation, depending entirely on the acoustic delivery. This distinction is lost in the text transcription. Additionally, potential transcription errors in the ASR stage can significantly degrade downstream sentiment analysis performance as well [21]. Second, existing research sufers from a mismatch in label granularity. Most current datasets focus on broad sentiment, such as positive or negative, or basic emotions like happy and sad [7, 25]. Dominant datasets in the field, including IEMOCAP [3], RAVDESS [22], and CREMA-D [4], are all predicated on these categorical emotions. While useful for basic classification, these labels fail to capture the nuanced interpersonal stances.

To address these limitations, we propose a novel dataset, Speech-Sense, for fine-grained speech sentiment detection that prioritizes prosodic features over semantic content. Specifically, we introduce a specialized 8-class label set designed to reflect diverse speaker personalities and states, including Confident, Nervous, Warm, Apathetic, Passionate, Impatient, Sarcastic, and Neutral—all detectable primarily through prosodic cues beyond lexical content alone [13, 17, 28, 31, 37]. To overcome the scarcity of human-recorded speech for these attitudes, we construct the dataset utilizing representative Text-to-Speech (TTS) synthesis to generate prosody-rich samples [33]. These samples are subsequently validated through rigorous human annotation. Finally, we validate our dataset through experiments across multi-modal Large Language Models (LLMs) [40], text-only LLMs [29], and speech encoders [1, 16, 30]. Our results demonstrate that models with acoustic access consistently outperform text-only baselines across all architectures, confirming the primacy of prosodic cues for fine-grained sentiment detection.

Our main contributions are summarized as follows:

• We define a fine-grained, paralinguistics-focused sentiment label set that captures nuanced speaker states beyond standard basic emotions.

• We introduce a curated dataset combining high-fidelity synthesized speech with human validation, addressing the data scarcity for attitudinal labels.

• We validate the dataset across multi-modal LLMs, text-only LLMs, and speech encoders, demonstrating that acoustic access is necessary for fine-grained sentiment detection.

## 2 Related Work

## 2.1 Speech Emotion Recognition Datasets

SSA builds directly upon Speech Emotion Recognition (SER), which traditionally focuses on identifying basic emotional states. We first review established SER datasets to identify their limitations in capturing complex social attitudes. The IEMOCAP database [3], containing approximately 12 hours of audiovisual data, remains the most widely used corpus, labeled with categorical emotions such as anger, happiness, sadness, and neutrality. Similarly, RAVDESS [22] and CREMA-D [4] provide acted speech samples across similar basic emotion categories. EMO-DB [2] ofers high-quality German emotional speech but is restricted to seven classes. Despite their contributions, these datasets share a critical limitation: they primarily model basic emotions rather than nuanced attitudes or interpersonal stances. This gap highlights the need for a new dataset tailored to fine-grained speaker states like confidence or apathy.

## 2.2 Methodological Paradigms

Approaches to SSA generally fall into two paradigms: cascade and end-to-end. The cascade paradigm, dominant in industry applications, treats speech analysis as a two-stage process: ASR converts audio to text, followed by TSA models for sentiment classification [35]. While this leverages the semantic reasoning power of LLMs, it sufers from inherent flaws. First, it is vulnerable to error propagation; Li et al. [21] demonstrate that Word Error Rate (WER) in ASR transcripts significantly degrade downstream sentiment detection accuracy. Second, the conversion to text irreversibly strips away paralinguistic information—the “how” of the message [34]. Research indicates that prosodic cues (e.g., pitch contours, rhythm) are often independent of, or even contradictory to, linguistic content (e.g., in sarcasm) [31]. In contrast, end-to-end approaches process raw audio waveforms or continuous speech representations directly, preserving paralinguistic fidelity. Speech encoders such as Whisper [30], HuBERT [16], and Wav2Vec2 [1] learn rich acoustic representations from large-scale audio data and have shown strong performance in speech emotion recognition. More recently, multi-modal LLMs (e.g., Qwen2.5-Omni [40], GPT-4o) have demonstrated the feasibility of understanding speech directly by integrating acoustic and linguistic information in a unified architecture. However, few studies have specifically optimized these architectures for the fine-grained sentiment detection proposed in this work.

## 2.3 Synthetic Data in Speech Processing

The scarcity of labeled data for specific paralinguistic states has driven research into synthetic data generation. While early SER research relies strictly on human-recorded speech, recent advances in expressive TTS have made synthetic data a viable alternative for training and augmentation. Ma et al. [23] empirically validate that SER models trained on high-quality synthesized emotional speech can achieve competitive performance, efectively addressing data scarcity in low-resource emotional classes. Furthermore, very recent works like EmoNet-Voice (2025) [33] have successfully utilized expert-verified synthetic speech to create fine-grained emotion benchmarks. Following this trajectory, we leverage a “Role-Play” prompting strategy grounded in the Stanislavski method of emotion induction [32] to achieve the high-fidelity prosodic realization required for complex speaker attitudes.

## 3 SpeechSense Dataset

SpeechSense captures nuanced interpersonal stances rather than the high-intensity basic emotions targeted by existing corpora.

## 3.1 Fine-Grained Sentiment Label Set

Unlike basic emotions, our objective is to detect speaker attitudes and interpersonal stances that are critical for understanding complex social dynamics. Discrete emotion frameworks such as Ekman’s basic emotions [11] and Plutchik’s wheel [27] were designed to characterize internal afective states rather than communicative stances directed toward an interlocutor. Existing characterizations of interpersonal stance are largely dimensional, and no discrete category set has been standardized for annotating interpersonal stance in speech. We introduce a fine-grained, 8-class label set derived from established acoustic-psychological research. To highlight their distinctive prosodic signatures, these labels are organized into four comparative groups based on their primary acoustic and interactional attributes: Internal Certainty, High-Energy Valence, Social Connection, and Prosodic Deviation. Rather than adopting an existing scheme directly, we adapt the dimensional principles underlying afective models such as PAD [24]—valence, arousal, and dominance—to the interpersonal setting, organizing each group as a contrastive pair that isolates one such distinction. The fourth group additionally captures an orthogonal aspect specific to speech: the alignment between prosody and lexical content. Table 1 details the definitions and acoustic signatures for each attitude.

Table 1: The SpeechSense Label Set. The labels are categorized into four attribute groups based on shared acoustic and interactional characteristics to capture fine-grained speaker stances.
<table><tr><td>Attribute Group</td><td>Label</td><td>Acoustic &amp; Psychological Definition</td></tr><tr><td rowspan="2">Internal Certainty (State of Mind)</td><td>Confident</td><td>Signals dominance and assurance. Acoustically characterized by shorter duration, higher intensity, and decisive falling pitch contours [28].</td></tr><tr><td>Nervous</td><td>Signals anxiety and instability. Manifests through pitch jitter (micro-tremors), higher average pitch, and irregular rhythm [20].</td></tr><tr><td rowspan="2">High-Energy Valence (Positive vs. Negative)</td><td>Passionate</td><td>Reflects intense positive engagement (Enthusiasm). Distinguished by an expanded pitch range and dynamic contours [13].</td></tr><tr><td>Impatient</td><td>Reflects irritation or time-pressure (Urgency). Distinguished from Passionate by clipped diction, abrupt stops, and a rigid, staccato rhythm [13].</td></tr><tr><td rowspan="2">Social Connection (Engagement Level)</td><td>Warm</td><td>Facilitates rapport and empathy. Defined by specific vocal qualities such as a softer timbre, breathy voice quality, and steady pitch stability [37].</td></tr><tr><td>Apathetic</td><td>Reflects detachment and lack of interest. Marked by reduced prosodic variability (monotone), prolonged pauses, and a slower speech rate [17].</td></tr><tr><td rowspan="2">Prosodic Deviation (Complexity)</td><td>Sarcastic</td><td>A complex attitude where prosody contradicts text. Characterized by a lower mean F0, slower tempo, and a nasal quality relative to the baseline [31].</td></tr><tr><td>Neutral</td><td>Serves as the acoustic baseline or control state. Exhibits standard prosody without significant excursions in pitch, energy, or voice quality [22].</td></tr></table>

![](images/fdb6c4cd5d693852dc2078bb5109bce9787118aa588efcc49e531f3ac75e6fa4.jpg)  
Figure 1: The SpeechSense dataset construction pipeline. The framework consists of (1) semantic-prosodic decoupled text design, (2) role-play synthesis using Lovo.ai, and (3) dualstage human validation and filtering.

## 3.2 Dataset Construction Pipeline

We construct the dataset using a rigorous three-stage pipeline designed to minimize artifacts and maximize attitudinal distinctiveness (illustrated in Fig. 1).

1) Stage 1: Semantic-Prosodic Decoupled Text Design. To ensure the model relies on acoustic cues rather than lexical priors, we generate carrier text—semantically neutral sentences that are decoupled from the target labels. We utilize Qwen3-Max to generate 120 carrier sentences per label. The model plays a deliberately limited role here, producing only the carrier text fed to the TTS engine, while the attitude itself is realized at the synthesis stage. What matters for this role is therefore not reasoning ability but constraint adherence and lexical diversity, on which Qwen3-Max outperformed smallerscale alternatives in our preliminary inspection. We adhere to five strict criteria during generation: (1) exclusion of explicit emotional words; (2) realistic conversational content; (3) semantic neutrality to allow reuse across labels; (4) structural diversity (declarative, imperative, conditional) to prevent syntactic overfitting; and (5) moderate duration (3–8 seconds). This produces a corpus where the “sentiment” is conveyed purely by prosody.

2) Stage 2: TTS Engine Selection and Role-Play Synthesis. Authentic fine-grained attitudinal speech is severely limited by data scarcity, privacy, and ethical constraints [33], and recent benchmarks such as EmoNet-Voice [33] and VoxEval [5] likewise rely on commercial TTS engines. Synthesis further enables strict variable control: holding textual content constant across attitudes isolates prosody as the sole diferentiating factor. After evaluating six TTS systems including open-source (CosyVoice, IndexTTS2, Kokoro) and proprietary (GPT-4o, ElevenLabs) platforms, we select Lovo.ai<sup>2</sup> for its humanlike prosody in rendering subtle non-basic sentiment cues, speaker diversity (30+ voice profiles), and scalable API infrastructure.

Crucially, to operationalize these capabilities, we implement a “Role-Play” synthesis methodology grounded in the Stanislavski method [32]. Rather than mechanically adjusting pitch parameters, we map our labels to specific situational acting directives supported by the engine (e.g., configuring the style for Sarcastic as “reading like a victim ofa prank with dry thanks”). This approach leverages the engine’s explicit style controls to yield organic, non-linear prosodic variations that mirror authentic human expression.

3) Stage 3: Human Validation and Filtering. To rigorously filter out synthesis artifacts and expressive misalignments, we establish a robust human-in-the-loop validation pipeline. We deploy the evaluation task via Qualtrics<sup>3</sup> and recruit qualified annotators through the Prolific<sup>4</sup> platform, enforcing strict screening criteria on language proficiency, education, and historical approval rate (see Table 2 for full recruitment and filtering statistics).

Table 2: Human Annotation and Filtering Statistics.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Annotator Recruitment</td><td></td></tr><tr><td>Platform</td><td>Prolific</td></tr><tr><td>Eligible candidate pool</td><td>23,006 (from 150,000+)</td></tr><tr><td>Screening: Language</td><td>Native English</td></tr><tr><td>Screening: Education</td><td>Bachelor&#x27;s+</td></tr><tr><td>Screening: Approval rate</td><td>≥99%</td></tr><tr><td>Annotators per clip</td><td>≥ 3</td></tr><tr><td>Filtering Results</td><td></td></tr><tr><td>Pre-filtering clips Stage 1: Majority Vote retained</td><td>960 623 (93.12% of final set)</td></tr><tr><td>Stage 2: Ref. Alignment retained Final curated clips</td><td>46 (6.88% of final set) 669 (69.7% retention)</td></tr><tr><td>Inter-Annotator Agreement</td><td></td></tr><tr><td>Fleiss&#x27; Kappa (overall)</td><td>0.4437 (Moderate)</td></tr><tr><td>Speaker-Label Balance</td><td></td></tr><tr><td>Speakers covering all 8 labels</td><td>26 / 30</td></tr><tr><td>Speakers covering 7 labels</td><td>4 /30</td></tr><tr><td></td><td></td></tr></table>

During the annotation phase, each audio clip is evaluated by at least three independent annotators. We apply a dual-stage filtering protocol to curate the final dataset. First, under a Majority Vote criterion, clips achieving consensus (3/3) or majority agreement (2/3) are automatically retained. Second, we employ a Reference Alignment strategy for ambiguous cases where annotators lack consensus (i.e., no majority): clips are retained if at least one annotator matches the intended label (the TTS generation target). This step preserves subtle but accurate expressions that may be perceived as ambiguous by some lay annotators, provided they align with the ground truth generation intent. This filtering strategy has been validated in previous crowdsourcing studies to maintain groundtruth quality [36]. Crucially, a substantial 93.12% of the final test samples are retained strictly through majority vote, indicating that the ground truth is predominantly driven by spontaneous human consensus rather than generation bias.

The final curated test dataset consists of 669 high-quality clips, achieving an overall Fleiss’ Kappa of 0.4437. Based on the widely accepted interpretation guidelines established by Landis and Koch [19], this score signifies moderate agreement (defined as the 0.41– 0.60 interval). This score is consistent with established benchmarks (CREMA-D: � = 0.42 [4], IEMOCAP: � = 0.40 [3]) and substantially exceeds recent fine-grained synthetic benchmarks (EmoNet-Voice: Krippendorf’s � = 0.14 [18, 33]).

![](images/045baef974d95a992f72e3c7e3d09ae746115eaeee7f27b24a2451478e0c5f6b.jpg)  
Figure 2: SpeechSense Dataset Label Distribution.

## 3.3 Final Dataset Statistics

The pipeline yields two distinct data subsets. To ensure robust generalization and prevent downstream models from overfitting to specific lexical patterns, we employ distinct text generation sources for the training and evaluation partitions. The SpeechSense Test Set serves as the evaluation “Gold Standard”, comprising 669 highquality audio clips derived from Qwen3-Max carrier text. Following the rigorous human validation described in Section 3.2, this subset maintains a meticulously balanced distribution across the eight classes, as visualized in Fig. 2, with prevalence ranging from 11.2% (Apathetic) to 13.1% (Nervous and Sarcastic). This balance minimizes class bias, ensuring that evaluation metrics reflect genuine prosodic understanding rather than prior probability shifts. Complementing the evaluation set, the SpeechSense Training Set constitutes a larger corpus of synthetic data generated using a diferent LLM family (Gemini 3 Pro). By decoupling the text generation source between training and testing, we ensure that the dataset evaluates the model’s ability to capture acoustic representations rather than specific writing styles or artifacts unique to a single LLM. Although both subsets share the same 30 voice profiles, this overlap does not introduce speaker identity leakage. We ensure a balanced speaker-label matrix: 26 of 30 speakers cover all 8 attitudes, and the remaining 4 cover 7 (see Table 2). Consequently, speaker identity provides no predictive shortcut for the target label, and the model must genuinely disentangle fine-grained prosodic variations rather than memorize speaker timbre. Beyond label coverage, the 30 profiles also exhibit substantial acoustic diversity. Table 3 summarizes their variation measured on a shared control utterance, spanning a broad pitch range and distinct vocal tract characteristics. While this training set uses the same Role-Play TTS methodology to ensure prosodic consistency, it relies on the model’s inherent stability for large-scale weakly-supervised learning. To ensure reproducibility, we publicly release the full dataset (audio, labels, and synthesis directives) under a CC BY 4.0 license, along with supplementary analyses, at https://github.com/Sher13cked/SpeechSense.

Table 3: SpeechSense Dataset and Speaker Statistics.
<table><tr><td>Dataset</td><td>Source</td><td>Val. Method</td><td>Clips</td><td>Spkrs</td></tr><tr><td>Training Set</td><td>Gemini 3 Pro</td><td>Weakly-Sup.</td><td>1,522</td><td>30</td></tr><tr><td>Test Set</td><td>Qwen3-Max</td><td>Human-Val.</td><td>669</td><td>30</td></tr><tr><td colspan="5">Speaker Diversity (shared control utterance, 30 profiles)</td></tr><tr><td colspan="5">Gender balance</td></tr><tr><td colspan="5">Mean F0 range</td></tr><tr><td colspan="2"></td><td colspan="3">190.6–315.8 Hz (std = 41.7)</td></tr><tr><td colspan="2">Formant F1 / F2 (std)</td><td colspan="3">58.3 / 94.2 Hz</td></tr></table>

Table 4: Word Error Rate Analysis.
<table><tr><td>Dataset</td><td>Conf.</td><td>Nerv.</td><td>Warm</td><td>Apat.</td><td>Pass.</td><td>Impat.</td><td>Sarc.</td><td>Neut.</td><td>Avg.</td></tr><tr><td>Training Set (↓)</td><td>5.00%</td><td>2.32%</td><td>0.18%</td><td>9.31%</td><td>1.15%</td><td>6.18%</td><td>8.57%</td><td>9.98%</td><td>5.42%</td></tr><tr><td>Test Set (↓)</td><td>2.90%</td><td>1.43%</td><td>3.88%</td><td>1.38%</td><td>7.20%</td><td>1.27%</td><td>7.98%</td><td>3.26%</td><td>3.70%</td></tr></table>

## 4 Experiments and Results

## 4.1 Speech Synthesis Quality Validation

Given that prosodic variations (e.g., mumbling in Apathetic or rapid speech in Impatient) can inadvertently degrade speech clarity, it is crucial to ensure that the synthesized audio retains high linguistic content preservation.

We employ Whisper-large-v3 [30], an advanced ASR model, to transcribe the entire SpeechSense benchmark. We compute the WER by comparing the ASR transcriptions against the groundtruth carrier text used for generation.

As presented in Table 4, the benchmark achieves an overall WER of3.70% on the curated Test Set and 5.42% on the weakly-supervised Training Set. This verifies that our TTS pipeline yields clear, artifactfree audio suitable for downstream representation learning.

A closer examination reveals performance nuances that align with our prosodic dimensions. The Sarcastic class consistently exhibits higher WER (Test: 7.98%, Train: 8.57%), a pattern that is both expected and desirable, reflecting the acoustic complexity of sarcasm, which naturally challenges ASR models optimized for neutral speech. These results confirm that SpeechSense audio is of suficient quality for the subsequent classification experiments.

## 4.2 Experimental Setup

We design our experimental protocol to answer two core questions: (1) whether acoustic cues are the dominant signal for fine-grained sentiment detection, and (2) whether the learned representations generalize across model architectures. To this end, we benchmark three model families under a unified evaluation framework.

1) Model Architecture. As multi-modal LLMs, we employ Qwen2.5- Omni (3B and 7B), freezing the backbone and appending a linear classification head (Dropout � = 0.1) to map final hidden states to 8 classes. As text-only baselines, we include Qwen2.5-Instruct (3B and 7B), which share the same language backbone but lack an audio encoder. As speech encoders, we evaluate Whisper-large-v3 [30], HuBERT-large [16], and Wav2Vec2-large [1], each paired with attention pooling and a linear classification head. This design spans three paradigms—multi-modal understanding, text-only reasoning, and pure acoustic representation. All models share a unified evaluation protocol (frozen backbone + linear head + cross-entropy loss), ensuring that performance diferences reflect representational capacity rather than training methodology.

Table 5: Main Classification Results.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Modality</td><td colspan="2">Zero-shot</td><td colspan="2">Supervised (Ours)</td></tr><tr><td>Acc</td><td>Macro F1</td><td>Acc</td><td>Macro F1</td></tr><tr><td>Multi-modal LLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-Omni-3B</td><td>Text</td><td>8.22%</td><td>3.79%</td><td>25.26%</td><td>19.71%</td></tr><tr><td>Qwen2.5-Omni-3B</td><td>Audio</td><td>3.74%</td><td>1.31%</td><td>54.86%</td><td>53.38%</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>Text</td><td>11.36%</td><td>6.20%</td><td>26.76%</td><td>22.27%</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>Audio</td><td>11.96%</td><td>2.96%</td><td>56.95%</td><td>56.76%</td></tr><tr><td>Text-only LLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-Instruct-3B</td><td>Text</td><td>15.84%</td><td>6.63%</td><td>14.20%</td><td>4.60%</td></tr><tr><td>Qwen2.5-Instruct-7B</td><td>Text</td><td>11.36%</td><td>4.33%</td><td>25.26%</td><td>15.97%</td></tr><tr><td>Speech Encoders</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper-large-v3</td><td>Audio</td><td>13.30%</td><td>5.03%</td><td>45.44%</td><td>45.06%</td></tr><tr><td>HuBERT-large</td><td>Audio</td><td>10.16%</td><td>3.87%</td><td>44.39%</td><td>43.79%</td></tr><tr><td>Wav2Vec2-large</td><td>Audio</td><td>12.56%</td><td>3.17%</td><td>44.54%</td><td>42.45%</td></tr></table>

2) Training Protocol. All models follow the cross-source evaluation protocol defined in Section 3.3. For multi-modal LLMs and text-only LLMs, we apply Low-Rank Adaptation (LoRA) on the attention modules with the backbone frozen. For speech encoders, we adopt a two-stage strategy: first training only the classification head, then unfreezing the entire network for joint fine-tuning. Hyperparameters are listed in our supplementary repository.

## 4.3 Main Results: The Primacy of Prosody

Table 5 summarizes performance across modalities, model families, and scales, comparing zero-shot and supervised settings.

First, the results underscore the critical necessity of specific prosodic training. Across all model families, zero-shot performance remains near-random, with Macro-F1 scores ranging from 1.31% to 6.63%. This confirms that neither pre-trained multi-modal LLMs nor speech encoders inherently possess fine-grained attitudinal representations. However, fine-tuning on SpeechSense yields substantial performance gains across all architectures: multi-modal LLM audio models improve by up to 50 percentage points in accuracy, and speech encoders achieve 42–45% Macro-F1 from near-random baselines. These improvements demonstrate that the dataset efectively teaches diverse architectures to decode complex prosodic cues that were previously inaccessible to their pre-trained backbones.

Second, the experiments empirically validate the semantic neutrality of our dataset design through converging evidence from two distinct text-only model families. After supervised training, Qwen2.5-Omni text models saturate at 20–22% F1, and Qwen2.5- Instruct text models perform even worse—as low as 4.60% F1 for the 3B variant, which is lower than its own zero-shot baseline (6.63%), indicating that training induces mode collapse in the absence of acoustic signal. This consistent failure across architectures confirms that the textual content is indeed semantically neutral and insufficient for distinguishing attitudes. The significant performance gap between the best audio model (56.95% Acc) and the best text model (26.76% Acc) confirms that the task is solvable almost exclusively through the “how” (paralinguistic information) rather than the “what” (linguistic information).

Third, speech encoders validate that acoustic features alone carry substantial attitudinal signal even without language modeling capacity. Whisper, HuBERT, and Wav2Vec2 achieve 42–45% Macro-F1, demonstrating that fine-grained prosodic cues can be captured by dedicated speech representations. The 10–14 percentage point gap between multi-modal LLM audio models (53–57% F1) and speech encoders (42–45% F1) suggests that the language understanding component of multi-modal LLMs provides additional reasoning capacity on top of prosodic features—likely by integrating acoustic evidence with contextual semantic inference.

## 4.4 Fine-Grained Analysis and Failure Modes

1) Models with Acoustic Access. We analyze the class-wise perfor mance ofmulti-modal LLMs (audio mode) and speech encoders—the two model families that receive acoustic input. Full per-class F1 scores are provided in our supplementary repository.

Across all audio models, Nervous emerges as the most reliably detected attitude (68–80% F1), suggesting that the acoustic markers of anxiety—pitch jitter, elevated mean pitch, and irregular rhythm— are robustly captured by speech representations alone, with speech encoders matching or surpassing multi-modal LLMs. In contrast, Neutral and Confident are consistently the hardest classes (below 45% F1 for most models), reflecting the dificulty of detecting atti tudes defined by the absence of distinctive prosodic features. The two model families also diverge systematically: for high-energy attitudes requiring valence disambiguation, such as Impatient and Passionate, multi-modal LLMs substantially outperform speech encoders, as these attitudes share similar energy profiles but difer in valence—a distinction that benefits from language understanding capacity. Finally, the confusion structure indicates that residual errors are systematic rather than random. Across all audio models, misclassifications concentrate on a small number of prosodically adjacent pairs: Confident and Neutral are mutually confused in both directions, consistent with their shared lack of distinctive prosodic excursions; Confident–Passionate overlap in multi-modal LLMs through their common assertive intensity; and Warm–Sarcastic overlap in speech encoders, reflecting the exaggerated pleasantness characteristic of sarcastic delivery. That errors fall on acoustically interpretable neighbors rather than dispersing arbitrarily indicates genuine prosodic proximity and the dificulty of fine-grained stance discrimination, not ill-defined category boundaries. Full confusion matrices are provided in our supplementary repository.

2) Models without Acoustic Access. We examine the text-only models—multi-modal LLMs (text mode) and text-only LLMs—which receive only transcripts. All four text-only configurations sufer from severe mode collapse, but critically, each collapses toward a diferent dominant class: Omni-3B concentrates nearly 70% of predictions on Impatient, Omni-7B assigns the majority to Sarcastic, Instruct-3B directs 87% to Warm, and Instruct-7B spreads across Confident, Sarcastic, and Impatient. Meanwhile, several classes receive zero predictions across multiple models—for instance, Neutral is never predicted by any of the four text-only configurations. This inconsistency in collapse targets is itselfrevealing: ifthe textual content carried systematic attitudinal signal, diferent models should converge toward similar predictions. Instead, each model resorts to arbitrary priors that vary with architecture and scale, providing strong evidence that the text carries no learnable sentiment information. Detailed per-model prediction distributions are provided in our supplementary repository.

## 4.5 Scaling and Architecture Efects

For multi-modal LLM audio models, scaling from 3B to 7B yields a moderate gain (Macro-F1: 53.38% → 56.76%). This improvement is concentrated on cognitively complex classes such as Sarcastic and Confident, which require disentangling subtle or contradictory prosodic signals, while several other classes show no gain or slight regression. This suggests that the additional reasoning capacity of larger models primarily benefits specific classes rather than uniformly improving detection.

For text-only models, even the 7B variant reaches only 15.97% F1, far below every audio configuration, confirming that scaling cannot compensate for the fundamental absence of acoustic signal.

Among speech encoders, Whisper (45.06%), HuBERT (43.79%), and Wav2Vec2 (42.45%) converge within a narrow 2.6-point band despite distinct pre-training paradigms. This convergence suggests that the attitudinal signal in SpeechSense is robustly accessible to diverse acoustic representations and does not depend on any particular pre-training strategy.

## 5 Limitations

We acknowledge several limitations. First, SpeechSense is built entirely from synthesized speech. While we validate prosodic fidelity through human annotation and WER analysis, a domain gap may exist between synthetic and real speech. We position SpeechSense as a foundational cold-start resource and encourage future work on real-world adaptation. Second, the current taxonomy covers eight interpersonal stances in English only. Extending to additional attitudes and languages would broaden applicability. Third, speaker diversity is limited to 30 voice profiles from a single TTS engine. Although our speaker-label balance analysis confirms no identity leakage, future iterations could incorporate multiple engines and a larger speaker pool to further improve generalization.

## 6 Conclusion

We introduce SpeechSense, a robust dataset that addresses the limitations of existing emotion recognition benchmarks by shifting the focus from basic emotions to fine-grained interpersonal stances, while rigorously isolating paralinguistic attitude from semantic content. We define a taxonomy of eight specific sentiment states, and employ high-fidelity synthesis to build a resource where acoustic generalization is critical. Our experiments across multi-modal LLMs, text-only LLMs, and speech encoders consistently demonstrate that acoustic access is necessary for detecting these nuanced distinctions, as semantic cues alone prove insuficient regardless of model scale. SpeechSense serves as a pivotal step toward more naturalistic human–machine interaction, enabling future models to understand not just the message, but also the social dynamics and nuanced delivery that shape the spoken word.

## Acknowledgments

The research presented in this paper was partially supported by the Research Grants Council of the Hong Kong Special Administrative Region, China (CUHK 2300246, RGC C1043-24G) and (CUHK 14203425, RGC GRF 2151317).

## References

[1] Alexei Baevski, Yuhao Zhou, Abdelrahman Mohamed, and Michael Auli. 2020. wav2vec 2.0: A framework for self-supervised learning of speech representations. Advances in neural information processing systems 33 (2020), 12449–12460.

[2] Felix Burkhardt, A. Paeschke, M. Rolfes, Walter F. Sendlmeier, and Benjamin Weiss. 2005. A database of German emotional speech. In Interspeech 2005. 1517– 1520. doi:10.21437/Interspeech.2005-446

[3] Carlos Busso, Murtaza Bulut, Chi-Chun Lee, Abe Kazemzadeh, Emily Mower, Samuel Kim, Jeannette N. Chang, Sungbok Lee, and Shrikanth S. Narayanan. 2008. IEMOCAP: Interactive Emotional Dyadic Motion Capture Database. Language Resources and Evaluation 42, 4 (Dec. 2008), 335–359. doi:10.1007/s10579-008- 9076-6

[4] Houwei Cao, David G. Cooper, Michael K. Keutmann, Ruben C. Gur, Ani Nenkova, and Ragini Verma. 2014. CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset. IEEE Transactions on Afective Computing 5, 4 (Oct. 2014), 377– 390. doi:10.1109/TAFFC.2014.2336244

[5] Wenqian Cui, Xiaoqi Jiao, Ziqiao Meng, and Irwin King. 2025. Voxeval: Benchmarking the knowledge understanding capabilities of end-to-end spoken language models. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 16735–16753.

[6] Wenqian Cui, Xiao-Hui Li, Daxin Tan, Qiyong Zheng, and Irwin King. 2026. Minimizing modality gap from the input side: Your speech llm can be a prosody aware text llm. arXiv preprint arXiv:2605.05927 (2026).

[7] Wenqian Cui, Pedro Sarmento, and Mathieu Barthet. 2024. MoodLoopGP: Generating emotion-conditioned loop tablature music with multi-granular features. In International Conference on Computational Intelligence in Music, Sound, Art and Design (Part ofEvoStar). Springer, 97–113.

[8] Wenqian Cui, Dianzhi Yu, Xiaoqi Jiao, Ziqiao Meng, Guangyan Zhang, Qichao Wang, Steven Y. Guo, and Irwin King. 2025. Recent Advances in Speech Language Models: A Survey. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (Eds.). Association for Computational Linguistics, Vienna, Austria, 13943–13970. doi:10.18653/v1 2025.acl-long.682

[9] Wenqian Cui, Lei Zhu, Xiaohui Li, Zhihan Guo, Haoli Bai, Lu Hou, and Irwin King. 2025. TurnGuide: Enhancing Meaningful Full Duplex Spoken Interactions via Dynamic Turn-Level Text-Speech Interleaving. arXiv preprint arXiv:2508.07375 (2025).

[10] Timothy DeGroot and Stephan J. Motowidlo. 1999. Why Visual and Vocal Interview Cues Can Afect Interviewers’ Judgments and Predict Job Performance. Journal ofApplied Psychology 84, 6 (1999), 986–993. doi:10.1037/0021-9010.84.6.986

[11] Paul Ekman. 1992. An argument for basic emotions. Cognition & emotion 6, 3-4 (1992), 169–200.

[12] Guy Fagherazzi, Aurélie Fischer, Muhannad Ismael, and Vladimir Despotovic. 2021. Voice for Health: The Use of Vocal Biomarkers from Research to Clinical Practice. Digital Biomarkers 5, 1 (04 2021), 78–88. arXiv:https://karger.com/dib/article-pdf/5/1/78/2576243/000515346.pdf doi:10. 1159/000515346

[13] Robert W Frick. 1985. Communicating emotion: The role of prosodic features. Psychological bulletin 97, 3 (1985), 412.

[14] Zhihan Guo, Wenqian Cui, Guan-Ting Lin, Daxin Tan, Jingyao Li, Qiyong Zheng, Dingdong Wang, Jing Xiong, Han Shi, Jiaya Jia, et al. 2026. A survey of audio reasoning in multimodal foundation models. arXiv preprint arXiv:2605.21008 (2026).

[15] Zhang He, Wenqian Cui, Haoning Xu, Xiao-Hui Li, Lei Zhu, Haoli Bai, Ma Shaohua, and Irwin King. 2026. Mtr-duplexbench: Towards a comprehensive evaluation of multi-round conversations for full-duplex speech language models. In Findings of the Association for Computational Linguistics: ACL 2026. 5334–5351.

[16] Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia, Ruslan Salakhutdinov, and Abdelrahman Mohamed. 2021. Hubert: Self-supervised speech representation learning by masked prediction of hidden units. IEEE/ACM transactions on audio, speech, and language processing 29 (2021), 3451–3460.

[17] Alexandra König, Nicklas Linz, Radia Zeghari, Xenia Klinge, Johannes Tröger, Jan Alexandersson, and Philippe Robert. 2019. Detecting Apathy in Older Adults with Cognitive Disorders Using Automatic Speech Analysis. Journal of Alzheimer’s Disease 69, 4 (June 2019), 1183–1193. doi:10.3233/JAD-181033

[18] Klaus Krippendorf. 2011. Computing Krippendorf’s alpha-reliability. (2011).

[19] J. Richard Landis and Gary G. Koch. 1977. The Measurement of Observer Agreement for Categorical Data. Biometrics 33, 1 (1977), 159–174. http:

//www.jstor.org/stable/2529310

[20] Petri Laukka, Clas Linnman, Fredrik Åhs, Anna Pissiota, Örjan Frans, Vanda Faria, Åsa Michelgård, Lieuwe Appel, Mats Fredrikson, and Tomas Furmark. 2008. In a Nervous Voice: Acoustic Analysis and Perception of Anxiety in Social Phobics’ Speech. Journal ofNonverbal Behavior 32, 4 (Dec. 2008), 195–214. doi:10.1007/s10919-008-0055-9

[21] Yuanchao Li, Peter Bell, and Catherine Lai. 2024. Speech Emotion Recognition With ASR Transcripts: A Comprehensive Study on Word Error Rate and Fusion Techniques. In 2024 IEEE Spoken Language Technology Workshop (SLT). IEEE, Macao, 518–525. doi:10.1109/SLT61566.2024.10832143

[22] Steven R. Livingstone and Frank A. Russo. 2018. The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS): A Dynamic, Multimodal Set of Facial and Vocal Expressions in North American English. PLOS ONE 13, 5 (May 2018), e0196391. doi:10.1371/journal.pone.0196391

[23] Ziyang Ma, Wen Wu, Zhisheng Zheng, Yiwei Guo, Qian Chen, Shiliang Zhang, and Xie Chen. 2024. Leveraging Speech PTM, Text LLM, And Emotional TTS For Speech Emotion Recognition. In ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, Seoul, Korea, Republic of, 11146–11150. doi:10.1109/ICASSP48485.2024.10445906

[24] Albert Mehrabian and James A Russell. 1974. An approach to environmental psychology. MIT Press.

[25] G. H. Mohmad Dar and Radhakrishnan Delhibabu. 2024. Speech Databases, Speech Features, and Classifiers in Speech Emotion Recognition: A Review. IEEE Access 12 (2024), 151122–151152. doi:10.1109/ACCESS.2024.3476960

[26] Iftekhar Naim, Md. Iftekhar Tanveer, Daniel Gildea, and Mohammed Ehsan Hoque. 2018. Automated Analysis and Prediction ofJob Interview Performance. IEEE Transactions on Afective Computing 9, 2 (April 2018), 191–204. doi:10.1109/ TAFFC.2016.2614299

[27] Robert Plutchik. 1980. A general psychoevolutionary theory of emotion. In Theories of emotion. Elsevier, 3–33.

[28] Heather Pon-Barry. 2008. Prosodic Manifestations of Confidence and Uncertainty in Spoken Language. In Interspeech 2008. ISCA, 74–77. doi:10.21437/Interspeech. 2008-16

[29] Qwen Team. 2025. Qwen2.5 Technical Report. arXiv:2412.15115 [cs.CL] https: //arxiv.org/abs/2412.15115

[30] Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine Mcleavey, and Ilya Sutskever. 2023. Robust Speech Recognition via Large-Scale Weak Super vision. In Proceedings ofthe 40th International Conference on Machine Learning (Proceedings ofMachine Learning Research, Vol. 202), Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (Eds.). PMLR, 28492–28518. https://proceedings.mlr.press/v202/radford23a.htm

[31] Patricia Rockwell. 2007. Vocal Features ofConversational Sarcasm: A Comparison of Methods. Journal of Psycholinguistic Research 36, 5 (July 2007), 361–369. doi:10.1007/s10936-006-9049-

[32] Klaus R Scherer. 2003. Vocal communication of emotion: A review of research paradigms. Speech communication 40, 1-2 (2003), 227–256.

[33] Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Felix Friedrich, Maurice Kraus, Kourosh Nadi, Huu Nguyen, Kristian Kersting, and Sören Auer. 2025. EmoNet-Voice: A Fine-Grained, Expert-Verified Benchmark for Speech Emotion Detection. arXiv:2506.09827 [cs] doi:10.48550/arXiv.2506.09827

[34] Björn Schuller, Stefan Steidl, Anton Batliner, Felix Burkhardt, Laurence Devillers, Christian Müller, and Shrikanth Narayanan. 2013. Paralinguistics in Speech and Language—State-of-the-art and the Challenge. Computer Speech & Language 27, 1 (Jan. 2013), 4–39. doi:10.1016/j.csl.2012.02.005

[35] Shariq Shah, Hossein Ghomeshi, Edlira Vakaj, Emmett Cooper, and Shereen Fouad. 2023. A Review of Natural Language Processing in Contact Centre Automation. Pattern Analysis and Applications 26, 3 (Aug. 2023), 823–846. doi:10.1007/s10044- 023-01182-8

[36] Jennifer Smith, Andreas Tsiartas, Valerie Wagner, Elizabeth Shriberg, and Niko letta Bassiou. 2018. Crowdsourcing emotional speech. In 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 5139–5143.

[37] Christina S. Soma, Dillon Knox, Timothy Greer, Keith Gunnerson, Alexander Young, and Shrikanth Narayanan. 2023. It’s Not What You Said, It’s How You Said It: An Analysis of Therapist Vocal Features during Psychotherapy. Counselling and Psychotherapy Research 23, 1 (March 2023), 258–269. doi:10.1002/capr.12489

[38] Lingli Wang, Ni Huang, Yili Hong, Luning Liu, Xunhua Guo, and Guoqing Chen. 2023. Voice-based AI in call center customer service: A natural field experiment. Production and Operations Management 32, 4 (2023), 1002–1018. arXiv:https://doi.org/10.1111/poms.13953 doi:10.1111/poms.13953

[39] Qichao Wang, Ziqiao Meng, Wenqian Cui, Yifei Zhang, Pengcheng Wu, Bingzhe Wu, Irwin King, Liang Chen, and Peilin Zhao. 2025. Ntpp: Generative speech lan guage modeling for dual-channel spoken dialogue via next-token-pair prediction. arXiv preprint arXiv:2506.00975 (2025).

[40] Jin Xu, Zhifang Guo, Jinzheng He, Hangrui Hu, Ting He, Shuai Bai, Keqin Chen, Jialin Wang, Yang Fan, Kai Dang, Bin Zhang, Xiong Wang, Yunfei Chu, and Junyang Lin. 2025. Qwen2.5-Omni Technical Report. arXiv:2503.20215 [cs] doi:10.48550/arXiv.2503.20215