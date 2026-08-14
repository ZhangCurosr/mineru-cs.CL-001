# CASA: CONTENT-ACOUSTIC SPEAKING ASSESSMENT WITH SPEECH ENCODER AND LARGE LANGUAGE MODEL

Nhan Phan<sup>1</sup> Ilona Lahteenm¨ aki¨ <sup>2</sup> Anna von Zansen<sup>2</sup> Olli-Pekka Pauna<sup>1</sup> Yaroslav Getman<sup>1</sup> Tamas Gr´ osz´ <sup>1,3</sup> Mikko Kurimo<sup>1</sup>

<sup>1</sup> Department of Information and Communications Engineering, Aalto University, Espoo, Finland <sup>2</sup> Department of Education, University of Helsinki, Helsinki, Finland <sup>3</sup> Programmable Autonomous Systems Division, Walton Institute, Waterford, Ireland

## ABSTRACT

Research on automatic speaking assessment (ASA) has increasingly adopted multimodal speech large language models to assess learners’ speaking performance. However, existing studies provide limited analysis of how acoustic and content information contribute to predictions and how stable the resulting performance is. We propose CASA, a simpler architecture combining Whisper-medium and Qwen3.5-2B that achieves state-of-the-art performance while providing a more interpretable separation between speech delivery and content.

On the Speak & Improve Corpus 2025, CASA achieves a root mean square error (RMSE) of 0.358, improving on the previous best RMSE while using approximately half the estimated inference parameters. The general-purpose architecture is designed for adaptation to other ASA corpora without structural changes and relies on three handcrafted fluency features. Through ablations and repeated runs, we analyze the individual and complementary contributions of acoustic and content information, examine performance variability, and demonstrate the potential of large language model reasoning for training-free content validation.

Index Terms— Automatic speaking assessment, LLM, ASA, Speak & Improve, Spoken language assessment

## 1. INTRODUCTION

Automatic speaking assessment (ASA) aims to automatically estimate second-language (L2) learners’ oral proficiency from their spoken responses. Compared with assessments conducted exclusively by human raters, ASA can reduce scoring costs, improve scoring consistency, and enable scalable and timely feedback. Nevertheless, holistic speaking assessment remains challenging because speaking proficiency is reflected in multiple interacting sources of evidence, including speech delivery (how something is said) and speech content (what is said).

With recent advances in large language models (LLMs), speech LLMs have increasingly been explored for ASA. Ma et al. [1] used Qwen2-Audio for L2 oral proficiency assessment and achieved state-of-the-art (SOTA) performance on the S&I test set [2], obtaining a Pearson correlation coefficient (PCC) of 0.833; RMSE was not reported. More recently, Lin et al. [3] combined Phi-4-Multimodal with a Whisperlarge encoder [4], achieving an RMSE of 0.360 on the same test set. In contrast, Cai et al. [5] achieved a comparable RMSE of 0.364 without using a speech LLM by combining a Whisper-medium encoder and BERT-based [6] graders with handcrafted acoustic and linguistic features.

Despite their strong performance, speech-LLM-based graders rely on large multimodal backbones, which increase computational requirements during inference and may limit practical deployment. Although Cai et al. [5] avoid using a speech LLM, their system combines multiple graders and handcrafted features. It also relies on a separate ASR model to produce more verbatim transcripts, increasing pipeline complexity and potentially limiting portability. Moreover, previous studies provide limited analysis of the respective contributions of acoustic and textual information. As their implementations have not been publicly released, reproducibility and further investigation are also limited.

In this work, we propose CASA (content-acoustic speaking assessment), a compact, general-purpose two-branch ASA architecture that explicitly separates acoustic information from linguistic information represented in ASR transcripts. The model employs a Whisper-medium encoder for the acoustic branch, and Qwen3.5-2B [7] for the content branch, making it substantially smaller than existing speech-LLM-based systems. Through detailed ablation and component-level analyses, we investigate the individual and complementary contributions of acoustic and textual information. Our model achieves an RMSE of 0.358 on the S&I test set, slightly outperforming the SOTA result of 0.360 while requiring substantially fewer parameters during inference. The architecture is designed as a general-purpose ASA framework that is not tailored to any dataset and uses only three simple fluency features. Our code is publicly available<sup>1</sup>.

## 2. METHOD

## 2.1. Dataset

The spoken language assessment (SLA) task of the S&I Corpus [2] consists of four parts: Parts 1, 3, 4, and 5. Part 1 contains six short questions, with responses typically lasting 10– 20 seconds. Part 3 consists of one-minute open-ended questions in which speakers express their opinions, whereas Part 4 contains one-minute tasks requiring speakers to describe a process illustrated in a graphical prompt. Finally, Part 5 contains five opinion-based questions, each with a maximum response time of 20 seconds. In total, the corpus contains approximately 315 hours of speech and is divided into training, development, and test sets, with the score distributions kept as balanced as possible across the splits. Each part was rated by human assessors and assigned a holistic score based on the CEFR scale [8]. The CEFR levels map to a 2.0–5.5 scale in 0.5 steps $( \mathrm { A } 2 { = } 2 . 0 , \mathrm { A } 2 { + } { = } 2 . 5 , . . . , \mathrm { C } 1 { + } { = } 5 . 5 )$ ; the final score is the arithmetic mean of the four part scores.

## 2.2. Model

Figure 1 presents the proposed model architecture. To support adaptation to other ASA settings, a single shared model scores each part independently. The overall score is then calculated as the arithmetic mean of the per-part predictions.

Our design builds on Phan et al. [9], who showed that a single Whisper-small encoder achieves strong performance but still lags behind SOTA [10]. Their model uses global average pooling, which may discard fine-grained acoustic variation. We instead adopt a Whisper-medium encoder, kept frozen: the encoder–decoder still produces the ASR transcript, while LoRA adapters [11] on the encoder produce the representation used by the acoustic branch.

The acoustic branch, originating from the Whisper encoder, assesses the speaker’s delivery. As the S&I corpus provides no separate delivery score, we supervise it using the holistic per-part CEFR label—the main loss target y—which the branch must predict from acoustics alone.

Because Whisper processes at most 30 s at a time, each answer is split into 30 s chunks up to a 2 min cap, with each chunk yielding 1,500 frame-level vectors. Rather than global average pooling, which discards acoustic detail [9], we concatenate the chunk encodings and average adjacent frame pairs, reducing the temporal resolution from 20 ms to 40 ms and halving the sequence length while preserving most acoustic information.

Parts P1 and P5 comprise several short answers, so before the Transformer we add two zero-initialized, learned embeddings to every frame: a task embedding (which part, P1– P5) and a segment embedding (which answer within the part). This is intended to let the aggregator distinguish frames from different answers.

The tagged frames pass through a two-layer Transformer encoder [12] with rotary position embeddings (RoPE) [13], which help preserve positional information over long concatenated sequences. Although this is less critical for the current S&I data, it supports future adaptation to longer responses. Following the [CLS] pooling of BERT, shown to transfer well to audio representation learning [14], we summarize the sequence with a [CLS] token.

The acoustic summary feeds two heads. An MLP projector maps it to four acoustic soft tokens for the LLM, allowing the acoustic information to be represented across multiple embeddings while remaining compact. A linear auxiliary head maps it to a scalar CEFR estimate that is detached and rendered as text (e.g., acoustic cefr estimate: 4.5) in the LLM prompt, and also drives the auxiliary loss. Both objectives backpropagate into the shared branch, so the main loss (through the soft tokens) and the auxiliary loss (through the auxiliary head) jointly shape the acoustic representation. Since acoustics alone cannot determine CEFR, we keep the auxiliary term from dominating by assigning it a small weight (0.1) and, more importantly, use a tolerance of ±1 score point. Predictions within this margin incur no auxiliary loss, and only deviations beyond the margin are penalized. Among the tested auxiliary-loss weights and tolerance levels, this combination achieved the best performance.

On the content branch, the frozen Whisper encoder– decoder transcribes each answer. The transcripts are generated offline before training for computational efficiency. For each part, the transcript is paired with its task prompt using <TASK part=P1> followed by the corresponding Q1/A1 pairs. Qwen3.5-2B receives the fused input sequence in the following order: the four acoustic soft tokens, the detached acoustic CEFR estimate represented as text, the scoring rubric, the question–answer pairs, and three simple fluency statistics derived from the audio and ASR output: duration, silence ratio, and speech rate (words/s). The base Qwen parameters remain frozen and are adapted with another LoRA adapter. The model performs a single forward pass without text generation, and a linear regression head applied to the final-token representation predicts the part score yˆ.

The training objective combines the main loss from the fused scorer with the auxiliary loss from the acoustic branch:

$$
\begin{array} { r l } & { \qquad \mathscr { L } = \mathrm { M S E } ( \hat { y } , y ) + 0 . 1 \mathrm { M S E } _ { \tau } ( \hat { y } _ { \mathrm { a u x } } , y ) , } \\ & { \qquad \mathrm { M S E } _ { \tau } ( a , b ) = \big ( \operatorname* { m a x } ( | a - b | - \tau , 0 ) \big ) ^ { 2 } , \quad \tau = 1 , } \end{array}
$$

where $\hat { y }$ is the final part prediction, $\hat { y } _ { \mathrm { a u x } }$ is the acoustic-only prediction, and $y$ is the ground-truth per-part CEFR score.

## 3. RESULTS

Training was conducted on a single NVIDIA H100 80 GB GPU with a batch size of 16 and gradient accumulation over two steps. Learning rates were $2 \times 1 0 ^ { - 4 }$ (acoustic LoRA), 1×

![](images/82f154f9688a63ddd5cbaca57b1bc74cc7bc7014a7643be4a7bb1a217988157d.jpg)  
Fig. 1. An overview of the model architecture.

Table 1. Overall performance on the S&I test set. CASA-Crisper uses CrisperWhisper [15].
<table><tr><td>Model</td><td>RMSE</td><td>PCC</td><td>%≤0.5</td><td>%≤1.0</td></tr><tr><td>NTNU [3]</td><td>0.360</td><td>0.827</td><td>85.7</td><td>99.0</td></tr><tr><td>Perezoso [5]</td><td>0.364</td><td>0.826</td><td>83.0</td><td>99.7</td></tr><tr><td>CASA</td><td>0.358</td><td>0.829</td><td>84.7</td><td>98.7</td></tr><tr><td>CASA-Crisper</td><td>0.363</td><td>0.836</td><td>84.0</td><td>99.7</td></tr></table>

$1 0 ^ { - 4 } ( \mathrm { L L M L o R A } ) , 5 \times 1 0 ^ { - 5 }$ (other modules). Training took approximately two hours. Table 1 reports the performance of CASA, with the model configuration and checkpoint selected solely according to overall RMSE on the development set. CASA achieves an RMSE of 0.358, marginally lower than the previous SOTA result of 0.360. As this difference is small, we do not claim a meaningful improvement in accuracy. Instead, CASA provides a better accuracy–model-size tradeoff, achieving comparable SOTA performance with approximately half the estimated inference parameters of the NTNU speech-LLM system, as shown in Table 2. Team Perezoso [5] can likewise be considered SOTA-level despite not using an LLM, although its pipeline combines multiple graders, handcrafted features, and a separate fine-tuned Parakeet-TDT-1.1B ASR model. One Whisper [9] remains the most compact system, using only a Whisper-small encoder, although with a higher RMSE of 0.372.

In terms of architecture, CASA is closer to Cai et al. [5] than to speech-LLM approaches [1, 3]. Cai et al. probe the final six layers of a frozen Whisper-medium encoder, while Lin et al. derive an acoustic proficiency prior from the final representation of a frozen Whisper-large encoder. CASA instead adapts the Whisper-medium encoder across its layers using LoRA. Since acoustic and semantic information is distributed unevenly across Whisper layers [16], this motivates CASA’s task-specific adaptation rather than reliance on a fixed finallayer representation. Cai et al. further combine their Whisper grader with several BERT-based graders and handcrafted features, whereas CASA integrates the acoustic representation with an LLM-based content branch and a tolerance-based auxiliary objective. Apart from three fluency statistics, CASA requires no handcrafted assessment features.

Table 2. Model size vs. performance on the S&I test set. NTNU and Perezoso parameter counts are estimated from their described architectures (no released code).
<table><tr><td>Model</td><td>RMSE</td><td>Total parameter</td></tr><tr><td>CASA</td><td>0.358</td><td>3.13 B</td></tr><tr><td>NTNU [3]</td><td>0.360</td><td>6.24 B</td></tr><tr><td>Perezoso [5]</td><td>0.364</td><td>2.17B</td></tr><tr><td>One Whisper [9]</td><td>0.372</td><td>0.17B</td></tr></table>

## 4. ANALYSIS

The modular acoustic encoder allows us to swap in Crisper-Whisper, a verbatim-transcription Whisper-large that preserves the disfluencies that standard Whisper smooths away. As we use the same hyperparameters as for Whisper-medium, the overall RMSE may be worse than what proper tuning would yield. In Table 3, CASA-Crisper improves on A2, indicating that verbatim transcripts help the system catch the content errors of A2 and B1 speakers. This gain matters because A2 is an under-represented class. B2 and C1, however, get worse: for otherwise fluent speakers, verbatim transcription may introduce inauthentic errors that hide the range marking the top band.

We also explore two self-supervised learning (SSL) models: WavLM [17] and wav2vec2 XLS-R 300M [18], swapping only the acoustic encoder while keeping CASA’s Whisper ASR transcript, so the acoustic representation is the sole difference from CASA. Both substantially underperform CASA, and an English-finetuned variant [19] also provides no improvement. We attribute this gap partly to architectural mismatch: CASA’s aggregator and training recipe were developed for Whisper representations and may not transfer optimally to SSL encoders. SSL encoders are effective for ASA in other architectures [20]; adapting CASA to exploit them (e.g. a dedicated pronunciation head) is left to future work.

Table 3. Overall RMSE by gold CEFR band on S&I; Macro is the unweighted mean RMSE over each CEFR level.
<table><tr><td>Model</td><td>A2</td><td>B1</td><td>B2</td><td>C1</td><td>Macro</td></tr><tr><td>CASA</td><td>0.553</td><td>0.351</td><td>0.290</td><td>0.554</td><td>0.437</td></tr><tr><td>CASA-Crisper</td><td>0.485</td><td>0.335</td><td>0.322</td><td>0.617</td><td>0.440</td></tr></table>

Neither a larger acoustic encoder (CrisperWhisper; Table 1) nor a larger LLM (Qwen3.5-4B, RMSE 0.364) improves RMSE; even pairing Whisper-large-v3 with the 4B LLM yields only 0.362. Performance thus appears not to be limited by model capacity, though larger models may require different hyperparameters.

CASA achieves per-part RMSEs of 0.476, 0.454, 0.490, and 0.444 for P1, P3, P4, and P5, respectively. Performance is weaker on P1 and P4, consistent with Lin et al. [3]. A qualitative analysis of the ASR transcripts, conducted with a co-author in applied linguistics, suggests that task format may partly explain this pattern: short personal questions in P1 and process descriptions in P4 provide less freedom to demonstrate broad linguistic and content proficiency than the opinion-based tasks in P3 and P5. Analyzing the same task formats, Banno et al. [21] similarly found that P1 and P4 place greater emphasis on fluency, whereas P3 and P5 place greater weight on thematic development and other contentrelated dimensions. P1 and P4 may therefore provide fewer discriminative textual cues. In CASA, the task embeddings in the Transformer block proved ineffective: they remained close to their zero initialization, contributing little task information to the acoustic representation. Future work could investigate separate prediction heads or graders for more constrained and open-ended tasks.

To assess the stability of the SOTA-level result, we performed 9 additional runs for each configuration in Table 4: 4 with seed 1011 and 5 with distinct seeds. Here, 4e-4 doubles the acoustic-branch LoRA learning rate, while aux-0 sets the auxiliary-loss weight to 0. The auxiliary head provides a small but consistent improvement across repeated runs, reducing mean RMSE by 0.004 and helping CASA reach SOTA-level performance. In two additional single-run ablations, blocking either the auxiliary-loss gradient or the soft-token-path (main-loss) gradient from reaching the acoustic encoder degraded performance, suggesting the two paths provide complementary supervision to the shared acoustic encoder. Doubling the learning rate to 4e-4 unexpectedly yielded an RMSE of 0.350 on the first run, but subsequent runs did not reproduce it, indicating an unstable configuration rather than a better one.

Table 4. RMSE variability on the S&I test set over 10 runs. The 95% CI is calculated for the mean.
<table><tr><td>Model</td><td>Mean</td><td>Median</td><td>Min-Max</td><td>95% CI</td></tr><tr><td>CASA</td><td>0.363</td><td>0.362</td><td>0.357–0.377</td><td>[0.359,0.367]</td></tr><tr><td>aux-0</td><td>0.367</td><td>0.366</td><td>0.362–0.376</td><td>[0.364,0.370]</td></tr><tr><td>4e-4</td><td>0.378</td><td>0.376</td><td>0.350–0.402</td><td>[0.364,0.392]</td></tr></table>

For CASA, even runs using the same seed produced test RMSEs ranging from 0.357 to 0.365, indicating nondeterminism in the training pipeline. Most distinct seeds produced comparable results, except for the poorer run with seed 6066. The 0.358 result in Table 1 was obtained from the first run of the configuration and was selected on the development set, rather than from the best repeated run. Full run-level results are available in our repository.

Although efficiency remains a primary goal, we argue that LLMs offer substantial value for ASA beyond their parameter scale. In particular, they can adapt to different assessment tasks [1] and provide speech content assessment and feedback without task-specific training [22]. As an illustration, we use CASA’s Qwen3.5-2B for few-shot content validation, prompting it at inference time to judge whether an answer addresses the given question. Replacing the original questions with an unrelated question about nuclear reactors, the LLM flags 99.9% of responses as not addressing the question; pairing responses with real questions from different parts yields 97.3%. This is also practical: judgments took just under 0.1 s per response. Such validation requires reasoning over the semantic relationship between question and answer, which is not available to an acoustic-only grader [9]. We noted that the same judge could also be prompted to flag A2-level content even from ASR transcripts, though inconsistently; we leave this as a direction for future work.

## 5. CONCLUSION

In this work, we introduced CASA, a model architecture that explicitly separates speech delivery from speech content in ASA. CASA achieves SOTA-level performance on the S&I test set while using approximately half the inference parameters of comparable LLM-based systems. Our analyses demonstrate the complementary roles of the acoustic and content branches, identify task-specific modeling and verbatim transcription as promising directions for further improvement, and show that the LLM branch can also support training-free content validation. By releasing our efficient, general-purpose architecture and experimental results, we aim to support reproducible research and further investigation of acoustic and linguistic information in ASA.

## 6. ACKNOWLEDGEMENTS

We would like to thank the following projects and funding agencies: Research Council of Finland through the funding to “DigiTala in action - Automatic speaking assessment supporting integration” (grant no 365232, 365233), and “Automatic assessment of spoken interaction in second language” (grant no 355586, 355587). The computational resources were provided by Aalto ScienceIT.

## 7. REFERENCES

[1] R. Ma, M. Qian, S. Tang, S. Banno, K. M. Knill, and\` M. J. Gales, “Assessment of L2 Oral Proficiency using Speech Large Language Models,” in Interspeech 2025. Aug. 2025, pp. 5078–5082, ISCA.

[2] K. Knill, D. Nicholls, M. Gales, M. Qian, and P. Stroinski, “Speak & Improve Corpus 2025: An L2 English Speech Corpus for Language Assessment and Feedback,” Tech. Rep., Apollo - University of Cambridge Repository, Dec. 2024.

[3] H.-Y. Lin, J.-K. Lin, C.-C. Wang, H.-C. Lu, and B. Chen, “Session-Level Spoken Language Assessment with A Multimodal Foundation Model Via Multi-Target Learning,” in ICASSP 2026, Barcelona, Spain, May 2026, pp. 17327–17331, IEEE.

[4] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. Mcleavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings of ICML. July 2023, vol. 202 of Proceedings of Machine Learning Research, pp. 28492–28518, PMLR.

[5] D. Cai, N. Madnani, and K. Yancey, “Team Perezoso’s ASR and SLA System for Speak & Improve Challenge 2025,” in 10th Workshop on Speech and Language Technology in Education. Aug. 2025, pp. 51–55, ISCA.

[6] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding,” in Proceedings of NAACL-HLT, Minneapolis, Minnesota, June 2019, pp. 4171–4186, Association for Computational Linguistics.

[7] Qwen Team, “Qwen3.5: Towards native multimodal agents,” Feb. 2026.

[8] Council of Europe, Common European Framework of Reference for Languages: Learning, Teaching, Assessment, Cambridge University Press, 2001.

[9] N. Phan, A. Porwal, Y. Getman, E. Voskoboinik, T. Grosz, and M. Kurimo, “One Whisper to Grade Them´ All,” in 10th Workshop on Speech and Language Technology in Education. Aug. 2025, pp. 56–60, ISCA.

[10] M. Qian et al., “Speak & Improve Challenge 2025: Tasks and Baseline Systems,” 2024.

[11] E. J. Hu et al., “LoRA: Low-rank adaptation of large language models,” Proceedings ofICLR, 2022.

[12] A. Vaswani et al., “Attention is all you need,” Advances in neural information processing systems, vol. 30, 2017.

[13] J. Su, M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu, “RoFormer: Enhanced transformer with Rotary Position Embedding,” Neurocomputing, vol. 568, pp. 127063, Feb. 2024.

[14] X. Li and X. Li, “ATST: Audio Representation Learning with Teacher-Student Transformer,” in Interspeech 2022. Sept. 2022, pp. 4172–4176, ISCA.

[15] M. Zusag, L. Wagner, and B. Thallinger, “CrisperWhisper: Accurate Timestamps on Verbatim Speech Transcriptions,” in Interspeech 2024. Sept. 2024, pp. 1265– 1269, ISCA.

[16] N. Glazer et al., “Beyond Transcription: Mechanistic Interpretability in ASR,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 44, pp. 37407–37416, Mar. 2026.

[17] S. Chen et al., “WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing,” IEEE Journal ofSelected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, Oct. 2022.

[18] A. Babu et al., “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in Interspeech 2022. Sept. 2022, pp. 2278–2282, ISCA.

[19] J. Grosman, “Fine-tuned XLSR-53 large model for speech recognition in English,” https: //huggingface.co/jonatasgrosman/ wav2vec2-large-xlsr-53-english, 2021.

[20] S. Banno, K. M. Knill, M. Matassoni, V. Raina, and\` M. Gales, “Assessment of L2 Oral Proficiency Using Self-Supervised Speech Representation Learning,” in 9th Workshop on Speech and Language Technology in Education. Aug. 2023, pp. 126–130, ISCA.

[21] S. Banno, R. Ma, M. Qian, S. Tang, K. Knill, and \` M. Gales, “Natural Language-based Assessment of L2 Oral Proficiency using LLMs,” in 10th Workshop on Speech and Language Technology in Education. Aug. 2025, pp. 189–193, ISCA.

[22] N. Phan et al., “Automated content assessment and feedback for Finnish L2 learners in a picture description speaking task,” in Interspeech 2024, 2024, pp. 317–321.