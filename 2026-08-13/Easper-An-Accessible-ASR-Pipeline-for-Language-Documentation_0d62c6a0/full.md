# Easper: An Accessible ASR Pipeline for Language Documentation

Aso Mahmudi <sup>ID</sup> <sup>1,∗∗</sup>, Ting Dang <sup>ID</sup> <sup>1</sup>, Ekaterina Vylomova <sup>ID</sup> <sup>1</sup>, Nick Thieberger <sup>ID</sup> <sup>2</sup>

<sup>1</sup> School of Computing and Information Systems, The University of Melbourne, Australia <sup>2</sup> School of Languages and Linguistics, The University of Melbourne, Australia amahmudi@student.unimelb.edu.au

## Abstract

Audio transcription is a critical bottleneck in language documentation. While multilingual Automatic Speech Recognition (ASR) models like Whisper offer solutions, field linguists often lack the expertise to utilise them. We present Easper, an open-source, no-code workflow enabling linguists to iteratively fine-tune ASR models via cloud resources directly from ELAN annotations. Deploying ASR also raises a cold start problem: deciding which recordings to transcribe first to bootstrap an accurate model. Using Easper, we evaluate transcription prioritisation strategies on three Vanuatu languages (Bislama, Nafsan, Nguna). We fine-tune models by recording session, comparing Character Error Rate trajectories when prioritising acoustic cleanliness versus linguistic richness. We demonstrate that prioritising lexically rich narratives and increasing acousticphonetic repetition, even in noisy environments, leads to faster improvements in transcription quality.

Index Terms: speech recognition, human-in-the-loop, language documentation, low-resourced languages

## 1. Introduction

Documenting endangered languages is a race against time. Field linguists collect vast amounts of audio recordings to capture linguistic diversity, yet the transcription of this data remains a critical bottleneck [1]. It requires careful listening, speaker segmentation, and consistent orthographic choices across hours of recordings [2]. To tackle this bottleneck, researchers have proposed shifting the documentation paradigm. For example, Bird [3] argued for “sparse transcription,” where annotations link segments of speech to selected meaningful units rather than providing exhaustive word-level text.

While sparse methods are valuable, producing full transcripts remains a primary goal for creating accessible community resources, searchable archives, and robust linguistic analyses. Automatic speech recognition (ASR) systems have the potential to significantly accelerate the transcription of audio recordings in language documentation [4]. Recent efforts have focused on leveraging foundation models like Whisper [5], demonstrating that fine-tuned ASR models can be successfully integrated into language documentation workflows to generate full transcripts [6, 7].

Despite these advances, two critical barriers prevent the widespread adoption of ASR by field linguists. First, finetuning neural models requires technical expertise, specialised corpus formatting, and GPU management that fall outside the typical linguist’s toolkit. Second, integrating ASR into a new documentation project introduces a “cold start” problem regarding data selection. Because manual annotation is highly resource-intensive, it is vital to train initial models as efficiently as possible. However, field linguists currently lack empirical guidelines on what type of recordings constitute the best seed data, whether to prioritise acoustic cleanliness (e.g., low background noise, no overlapping speakers) or linguistic richness (e.g., vocabulary diversity and repetition) when selecting initial data for fine-tuning.

In response to these challenges, we present a comprehensive methodology tailored to the realities of low-resource language documentation. To overcome the technical barrier, we introduce a practical, no-code ASR pipeline that handles the entire workflow: from annotations in ELAN (the standard tool for linguistic annotation [8]), through data preparation and model fine-tuning, to diarised transcription and integration back into ELAN for post-editing. To overcome the strategic barrier, we establish empirically tested data selection guidelines that optimise the human-in-the-loop transcription effort.

Our contributions are two-fold:

• Easper, A Portable Open-Source Workflow: We release an open-source pipeline that integrates ELAN with ASR model fine-tuning, enabling linguists to train and deploy models without expert engineering support.<sup>1</sup>

• Transcription Prioritisation Strategies: Using this pipeline, we simulate a transcription prioritisation scenario across three low-resource languages of Vanuatu. To ensure our evaluation spans the acoustic realities of field linguistics, our dataset contrasts recent, higher-quality recordings (Bislama) with archival fieldwork data (Nafsan and Nguna). We systematically compare five data selection strategies across these corpora, demonstrating that prioritising lexical richness and acoustic-phonetic repetition is more important for achieving higher accuracy than acoustic cleanliness in earlystage model adaptation.

## 2. Related Work

Large multilingual ASR models (e.g., Whisper [5], XLS-R [9], MMS [10], Omnilingual [11]) show impressive zero-shot capabilities but require fine-tuning for under-resourced languages due to non-standardised orthographies and specific transcription preferences [3]. While fine-tuning neural models now outperforms traditional, resource-intensive Kaldi-based systems [12] for language documentation [13, 6], the process remains technically challenging for field linguists without dedicated support for corpus formatting and GPU management.

Efforts to democratize ASR include Elpis [14], which pro vides a graphical interface for training Kaldi and neural models [15] directly from ELAN files. However, Elpis requires local GPU infrastructure and lacks support for fine-tuning modern language models. Furthermore, even as tools become more accessible, a strategic barrier remains: there is little empirical guidance on how linguists should select the initial data for finetuning. Our work addresses both gaps by providing a cloudbased, no-code fine-tuning pipeline paired with an empirical analysis of data selection strategies to guide active elicitation.

## 3. Easper: The Portable ASR Workflow

To enable iterative model improvement during fieldwork, we developed Easper (ELAN-integrated Automatic Speech Recogniser), a lightweight, open-source pipeline designed for linguists with limited technical resources. The workflow consists of three stages: Data Preparation, Fine-Tuning, and On-Device Inference.

## 3.1. Data Preparation

The first stage of the pipeline, implemented in the Easper Dataset Generator (Figure 1a), processes ELAN (.eaf) files and audio recordings to create training data for ASR fine-tuning. Using the Python library pympi-ling [16], the module extracts annotations and performs a series of validation checks designed to support systematic data cleaning.

Each annotation is evaluated against validation rules. Since ASR models work with short audio segments, annotations longer than 30 seconds are flagged for manual review. Such segments are truncated in models and can negatively affect performance. The module also detects overlapping annotations across tiers. All identified issues are summarised in a structured report for correction in ELAN.

In addition, to assess transcription quality, the module computes corpus-level statistics, including character and word frequency distributions. These help identify spelling inconsistencies, typographical errors, and unintended code-switching. Character bigram statistics are also generated to reveal structural patterns and guide future data collection.

After revision, the module exports the final dataset as 16 kHz mono WAV files together with a CSV file linking each audio segment to its transcription. The dataset is then ready for upload to Google Drive and ASR model fine-tuning in the next stage.

## 3.2. ASR Model Fine-Tuning

A major obstacle to the adoption of ASR in language documentation is the computational cost associated with model training and fine-tuning. To address this challenge, our workflow leverages cloud-based platforms such as Google Colab [17]. This separates the heavy computational work of fine-tuning from the limitations of typical laptops.

Among the multilingual ASR models currently available, we recommend using OpenAI’s Whisper-small (244M) and Meta’s XLS-R (300M) in our pipeline. These models were chosen based on three criteria: (1) they can be fine-tuned using freely available cloud computing resources; (2) they have demonstrated strong performance in extremely low-resource language settings; and (3) they are compact enough to run efficiently offline on standard CPU-based computers.

This infrastructure is designed to support an iterative, human-in-the-loop workflow. By prioritising models with compact file sizes (less than 1 GB), we ensure that trained models remain highly portable. Linguists can download fine-tuned models from the cloud and deploy them directly within the offline Easper desktop application. This design enables the development and use of custom ASR systems in remote fieldwork settings, while minimising the need for advanced programming or machine learning expertise.

![](images/cb9af3c310555223b4364ea87929cdb71d369dc0b02eb456551881fc67ea0eb9.jpg)  
(a) Dataset Generator

![](images/e85eab05820800f4919a261b058590a59d924928058b31c593bd44c5e5f4a909.jpg)  
(b) Transcriber  
Figure 1: Screenshot ofEasper Desktop Application

## 3.3. On-Device Transcription

The final stage applies the fine-tuned ASR model to new unannotated audio recordings through a four-step process: (1) speaker diarisation, (2) segmentation, (3) speech recognition, and (4) ELAN export.

For speaker diarisation and segmentation, simple silence detection and amplitude-based segmentation were tested, but these methods were unreliable in noisy recordings or with varied speaking styles. For robust segmentation, we use either the SpeechBrain toolkit [18] or pyannote-audio [19]. These tools detect speaker boundaries and divide the audio into speaker-specific utterances. Since linguists know the number of speakers in advance, providing this information improves diarisation accuracy. The resulting segments are then transcribed using the fine-tuned Whisper model.

To ensure accessibility without technical overhead, the pipeline is packaged as a Python desktop application. This enables local, offline processing of sensitive field data. The graphical interface (Figure 1b) is straightforward: users select a trained model, input an audio file, and adjust a segmentation sensitivity slider to accommodate varying speaking paces and hesitations.

The transcription output is written back to a new ELAN file. Each speaker is assigned a separate tier, and each segment is aligned with its audio interval. This supports semi-automatic transcription, where the ASR provides a first draft, and the linguist performs revision. The modular design allows retraining and retranscription as more data becomes available.

## 4. Methodology

To guide transcription prioritisation during fieldwork, we propose a data selection methodology that evaluates the intrinsic acoustic and linguistic properties of unannotated audio. Unlike standard active learning paradigms that rely on modeluncertainty metrics over isolated speech segments, our approach is strictly tailored to the operational realities of language documentation. In fieldwork, manual transcription is conducted through continuous recording sessions, such as a specific interview, a folktale, or an elicitation task, rather than a randomised assortment of isolated utterances drawn from various days and speakers. To accurately mirror this workflow, we enforce a strict session-level constraint on our pipeline: we treat each recording session as an indivisible, discrete unit of data that must be evaluated and added to the training pool in its entirety.

## 4.1. Session Feature Extraction

For every session within a language’s corpus, we extract a set of independent features representing its overall acoustic quality and linguistic richness:

• Acoustic Features: Signal-to-Noise Ratio (SNR) and the proportion of overlapping speech (OVR).

• Linguistic Features: For session s we measure lexical diversity using the standard Type-Token Ratio,

$$
\mathrm { T y T o } ( s ) = \frac { \# \mathrm { T y p e s } } { \# \mathrm { T o k e n s } } ,
$$

However, TyTo overly rewards words occurring only once, which are difficult for neural networks to learn without repetition, and it is insensitive to session duration. A lengthsensitive alternative, the MATTR measure [20], addresses the duration issue by averaging type-token ratio over a fixed-size sliding window of tokens. However, MATTR measures local lexical density within short windows, which is poorly suited to our highly variable and often short sessions; our objective is instead to capture session-level repetition. We therefore introduce the Normalised Token-to-Type Ratio,

$$
\mathrm { T o T y } ( s ) = { \frac { \# \mathrm { T o k e n s } } { \# \mathrm { T y p e s } \times \mathrm { D u r a t i o n } } } ,
$$

the average number of repetitions per unique word, normalised by session length so that longer sessions do not score higher.

## 4.2. Data Selection Strategies

To simulate a transcription prioritisation workflow, we incrementally add training data session-by-session. We define five distinct data selection strategies that rank the available unannotated sessions based on the extracted features:

• Baseline: Sessions are added in a randomised order;

• SNR Priority: Sessions are added in descending order of Signal-to-Noise Ratio (cleanest background first);

• Minimal Overlap Priority: Sessions are added in ascending order of Speaker Overlap Rate (least overlapping speech first);

• TyTo Priority: Sessions are added in descending order of Type-Token Ratio (highest vocabulary diversity first);

• ToTy Priority: Sessions are added in descending order of Normalised Token-to-Type ratio (highest average repetition of words first).

## 5. Experimental Setup

To evaluate the effectiveness of our proposed methodology, we designed an experimental setup across three distinct fieldwork projects to assess how prioritising these different acoustic or linguistic characteristics impacts the data efficiency of the models.

## 5.1. Dataset

We utilise a corpus comprising ELAN-annotated field recordings from three languages spoken in Vanuatu (two Oceanic and one Creole), accessible with permission on the PARADISEC catalogue.<sup>2</sup>

Table 1: Dataset statisticsfor the target languages.
<table><tr><td>Language</td><td>Total Audio</td><td>Sessions</td><td>Segments</td><td>Tokens</td><td>Types</td></tr><tr><td>Bislama (bis)</td><td>13h45m</td><td>49</td><td>11,575</td><td>123,093</td><td>4,801</td></tr><tr><td>Nafsan (erk)</td><td>14h50m</td><td>32</td><td>9,245</td><td>107,775</td><td>8,526</td></tr><tr><td>Nguna (llp)</td><td>01h01m</td><td>7</td><td>1,098</td><td>7,976</td><td>852</td></tr></table>

• Bislama (ISO 639-3: bis) is the national language of Vanuatu, spoken by approximately 300,000 people. It is a young, largely English-lexified, Creole that developed in the second half of the 19th century [21].

• Nafsan (ISO 639-3: erk) is one of the 138 indigenous languages of Vanuatu and has around 5,000 speakers. The Nafsan writing system was originally created by missionaries in the 1860s and has been largely unchanged since then, except for the more recent addition of tildes to indicate coarticulated labio-velars, p and˜ m [22].˜

• Nguna (ISO 639-3: llp), also known as North Efate, is spoken by approximately 9,500 people across the northern area of Efate and adjacent offshore islands. Its phonology and grammar were extensively documented in earlier descriptive works, making it a valuable comparative anchor in our dataset [23].

This selection offers diverse recording and annotation conditions. We contrast recent, high-quality audio (Bislama) with archival field recordings (Nafsan, Nguna), and highly accurate transcriptions (Bislama, Nguna) with noisier labels (Nafsan). Table 1 details the corpus statistics for these languages prior to fine-tuning, highlighting the varying scales of available data.

## 5.2. Implementation Details

For each of the three languages, we initialise a separate instance of the pre-trained Whisper-Small foundation model (244M parameters). To rigorously evaluate the maximum potential of each data selection strategy, we perform full fine-tuning of the entire model architecture for 3 epochs per step, utilising a batch size of 8 and a learning rate of 1e-5. Prior to training, acoustic features are extracted by estimating the session SNR under the OM-LSA framework [24] and detecting speaker overlaps via pyannote [19].

At each training step, we add the next highest-ranked session to the training pool, based on the chosen strategy. To account for large variance in recording lengths, we implement a minimum duration threshold; if a selected session is shorter than this threshold, we continue adding the subsequent highestranked sessions until the threshold is met for that training step. The model is then fine-tuned on the accumulated data.

## 5.3. Evaluation Metric

We adopt Character Error Rate (CER) as our primary evaluation metric to better capture utility for post-editing by linguists. While Word Error Rate (WER) can be too harsh on nonstandard spellings in low-resource languages, and Phoneme Error Rate (PER) complicates evaluation with language-specific symbol ambiguities [15], CER directly approximates the keystrokes needed for post-editing. Defined as the minimum character edits required to convert the system output into the reference text, CER provides a fairer assessment of practical model performance.

![](images/a3676f3fd3f49b9c502dbdaf696de7264105fddea85c12d8e403cc7d821459ce.jpg)

![](images/7e7e462ff41fc6c15a1ac702f8c1973f3a6be7d0ae9b026857aede8798e780ed.jpg)

![](images/2890aa76734592c00790455925460150bbb3db2906477dc4d487fdf035f88ca4.jpg)  
Higher Normalised Token to Type (ToTy) Higher Signal-to-Noise (SNR) Higher Type to Token (TyTo) Lower Overlapping Speech Random  
Figure 2: Learning curves demonstrating the impact of data selection strategies on Character Error Rate (CER).

## 6. Results and Discussion

We evaluated the five data selection strategies across our languages by incrementally fine-tuning the models session-bysession. To preserve sufficient training data for Nguna, given its extremely limited corpus (61 minutes total), its test set was scaled to 10 minutes, while a fixed 30-minute test set was maintained for Bislama and Nafsan.

## 6.1. Impact of Selection Strategies on Learning Curves

As illustrated in Figure 2, while all strategies generally decrease the CER as more transcribed data is added, their earlystage learning trajectories diverge significantly. The spikes observed in some trajectories likely reflect catastrophic forgetting induced by small, heterogeneous training increments. The Normalised Token-to-Type Ratio (ToTy) priority tends to outperform the other strategies during these early stages, achieving the lowest CER across the target languages.

## 6.2. Robustness of Foundation Models vs. Lexical Sparsity

The results of these simulated transcription prioritisation experiments challenge conventional assumptions about fieldwork data collection. A primary finding of this study is that linguistic distribution outranks acoustic cleanliness when bootstrapping an ASR model for a new language.

The mediocre performance of the SNR and Minimal Overlap strategies suggests that massive pre-trained foundation models like Whisper are already highly robust to the non-stationary background noises and overlaps typical of field recordings. What the foundation model lacks is not acoustic robustness, but rather the specific vocabulary and orthographic rules of the target language. By explicitly feeding the model sessions with high lexical density early in the training process, it rapidly aligns its existing acoustic representations to the novel linguistic space.

## 6.3. Lexical Breadth and Depth

Furthermore, by comparing the Type-Token Ratio (TyTo) with our Normalised Token-to-Type Ratio (ToTy), we observed a critical tension between vocabulary breadth and depth.

While TyTo exposes the model to a rich vocabulary, prioritising ToTy tends to perform better. This confirms that neural networks require sufficient acoustic-phonetic repetitions to reliably learn new grapheme-to-phoneme mappings. A session with a massive but sparse vocabulary (high TyTo) is practically less useful for early fine-tuning than a session that deeply reinforces a core set of vocabulary through repetition (high ToTy).

## 6.4. Practical Recommendations for Linguists

Based on our pipeline development and empirical evaluations, we offer the following guidelines to optimise fieldwork for ASR integration. While capturing clean audio using directional microphones in quiet settings remains vital for archival integrity, ASR fine-tuning benefits most from linguistic richness. Linguists should actively elicit lexically dense, repetitive narratives and prioritise their transcription, even if the recording contains sub-optimal background noise.<sup>3</sup>

## 7. Conclusion

We have presented Easper, a practical, open-source ASR pipeline tailored for language documentation. By integrating ELAN, Whisper, and cloud computing, the workflow enables field linguists to independently handle data preparation, model fine-tuning, and diarised transcription without requiring programming expertise or specialised hardware.

Beyond providing this infrastructure, we addressed the strategic cold start problem of ASR bootstrapping through simulated transcription prioritisation across three Vanuatu languages. Crucially, our findings challenge the assumption that clean audio is paramount for early-stage training. Instead, we demonstrate that prioritising linguistic richness, specifically lexical breadth and acoustic-phonetic repetition, is significantly more effective than acoustic cleanliness for accelerating model adaptation. These results offer field linguists empirical guidelines for optimising their transcription efforts.

Future work will adapt the Easper pipeline for massive legacy archives, such as PARADISEC, which contains over 21,000 hours of audio spanning 1,400 languages, with many lacking transcripts or metadata. Implementing Easper’s iterative workflow is expected to offer a crucial means of accessing and utilising this extensive and previously inaccessible material.

## 8. Acknowledgments

This research was supported by The University of Melbourne’s Research Computing Services and the Petascale Campus Initiative. Funding for Thieberger’s work and for PARADISEC is from ARC DP250102214 and the Language Data Commons of Australia.

## 9. Generative AI Use Disclosure

In accordance with ISCA policy, the authors disclose the use of Generative AI tools to assist in editing and polishing the language of this manuscript. These tools were not used to produce any significant portion of the scientific content, experimental design, or core ideas. The authors take full responsibility and accountability for the final content of this paper.

## 10. References

[1] B. Foley, J. T. Arnold, R. Coto-Solano, G. Durantin, T. M. Ellison, D. Van Esch, S. Heath, F. Kratochvil, Z. Maxwell-Smith, D. Nash et al., “Building speech recognition systems for language documentation: The CoEDL Endangered Language Pipeline and Inference System (ELPIS).” in SLTU, 2018, pp. 205– 209. [Online]. Available: https://www.isca-archive.org/sltu 2018/ foley18 sltu.pdf

[2] F. Seifart, N. Evans, H. Hammarstrom, and S. C. Levinson,¨ “Language documentation twenty-five years on,” Language, vol. 94, no. 4, pp. e324–e345, 2018. [Online]. Available: https://doi.org/10.1353/lan.2018.0070

[3] S. Bird, “Sparse transcription,” Computational Linguistics, vol. 46, no. 4, pp. 713–744, Dec. 2020. [Online]. Available: https://aclanthology.org/2020.cl-4.1/

[4] N. Thieberger, “LD&C possibilities for the next decade,” Language Documentation & Conservation, 2017. [Online]. Available: http://hdl.handle.net/10125/24722

[5] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in Proceedings ofthe 40th International Conference on Machine Learning, ser. ICML’23. JMLR.org, 2023. [Online]. Available: https://dl.acm.org/doi/10.5555/3618408.3619590

[6] S. Guillaume, G. Wisniewski, C. Macaire, G. Jacques, A. Michaud, B. Galliot, M. Coavoux, S. Rossato, M.-C. Nguyen,ˆ and M. Fily, “Fine-tuning pre-trained models for automatic speech recognition, experiments on a fieldwork corpus of japhug (trans-himalayan family),” in Proceedings of the Fifth Workshop on the Use of Computational Methods in the Study of Endangered Languages. Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 170–178. [Online]. Available: https://aclanthology.org/2022.computel-1.21/

[7] B. Billings and B. McDonnell, “Connecting automated speech recognition to transcription practices,” in Proceedings of the Eight Workshop on the Use of Computational Methods in the Study of Endangered Languages. Honolulu, Hawaii, USA: Association for Computational Linguistics, Mar. 2025, pp. 128–132. [Online]. Available: https://aclanthology.org/2025.computel-main.14/

[8] P. Wittenburg, H. Brugman, A. Russel, A. Klassmann, and H. Sloetjes, “ELAN: a professional framework for multimodality research,” in Proceedings of the Fifth International Conference on Language Resources and Evaluation (LREC’06). Genoa, Italy: European Language Resources Association (ELRA), May 2006. [Online]. Available: https://aclanthology.org/L06-1082/

[9] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, and M. Auli, “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” pp. 2278–2282, 2022. [Online]. Available: https://www.isca-archive. org/interspeech 2022/babu22 interspeech.pdf

[10] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, A. Elkahky, Z. Ni, A. Vyas, M. Fazel-Zarandi et al., “Scaling speech technology to 1,000+ languages,” Journal of Machine Learning Research, vol. 25, no. 97, pp. 1–52, 2024. [Online]. Available: https://dl.acm.org/doi/abs/10.5555/3722577.3722674

[11] Omnilingual-ASR-team, “Omnilingual ASR: Open-source multilingual speech recognition for 1600+ languages,” 2025. [Online]. Available: https://arxiv.org/abs/2511.09690

[12] D. Povey, A. Ghoshal, G. Boulianne et al., “The Kaldi speech recognition toolkit,” in IEEE 2011 workshop on automatic speech recognition and understanding, vol. 1. Hawaii, 2011, pp. 5–1.

[13] A. Jones, S. Zhang, J. Hale, M. Renwick, Z. Vrzic, and K. Langston, “Comparing Kaldi-based pipeline elpis and whisper for cakavian transcription,” inˇ Proceedings ofthe Third Workshop on NLP Applications to Field Linguistics. Bangkok, Thailand: Association for Computational Linguistics, Aug. 2024, pp. 61–68. [Online]. Available: https://aclanthology.org/2024.fieldmatters-1. 8/

[14] B. Foley, A. Rakhi, N. Lambourne, N. Buckeridge, and J. Wiles, “Elpis, an accessible speech-to-text tool.” in INTER-SPEECH, 2019, pp. 4624–4625. [Online]. Available: https:// www.isca-archive.org/interspeech 2019/foley19 interspeech.pdf

[15] O. Adams, B. Galliot, G. Wisniewski, N. Lambourne, B. Foley, R. Sanders-Dwyer, J. Wiles, A. Michaud, S. Guillaume, L. Besacier, C. Cox, K. Aplonova, G. Jacques, and N. Hill, “User-friendly automatic transcription of low-resource languages: Plugging ESPnet into elpis,” in Proceedings of the 4th Workshop on the Use ofComputational Methods in the Study ofEndangered Languages Volume 1 (Papers). Online: Association for Computational Linguistics, Mar. 2021, pp. 51–62. [Online]. Available: https://aclanthology.org/2021.computel-1.7

[16] M. Lubbers and F. Torreira, “pympi-ling: a Python module for processing ELANs EAF and Praats TextGrid annotation files.” https://pypi.python.org/pypi/pympi-ling, 2013-2025, version 1.71.

[17] D. S. R. Sukhdeve and S. S. Sukhdeve, “Google colaboratory,” in Google cloud platform for data science: A crash course on big data, machine learning, and data analytics services. Springer, 2023, pp. 11–34. [Online]. Available: https://link.springer.com/ book/10.1007/978-1-4842-9688-2

[18] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong et al., “SpeechBrain: A general-purpose speech toolkit,” arXiv preprint arXiv:2106.04624, 2021. [Online]. Available: https://arxiv.org/abs/2106.04624

[19] H. Bredin, “pyannote.audio 2.1 speaker diarization pipeline: principle, benchmark, and recipe,” in Interspeech 2023, 2023, pp. 1983–1987. [Online]. Available: https://www.isca-archive. org/interspeech 2023/bredin23 interspeech.html

[20] M. A. Covington and J. D. McFall, “Cutting the Gordian Knot: The Moving-Average Type–Token Ratio (MATTR),” Journal of Quantitative Linguistics, vol. 17, no. 2, pp. 94–100, 2010. [Online]. Available: https://doi.org/10.1080/09296171003643098

[21] T. Crowley, Bislama reference grammar. University of Hawaii Press, 2004.

[22] N. Thieberger, A grammar of South Efate: an Oceanic language ofVanuatu. University of Hawaii Press, 2006.

[23] E. E. Facey, Nguna voices: text and culture from Central Vanuatu. University of Calgary Press, 1988.

[24] I. Cohen and B. Berdugo, “Speech enhancement for nonstationary noise environments,” Signal processing, vol. 81, no. 11, pp. 2403–2418, 2001. [Online]. Available: https: //doi.org/10.1109/ICIECS.2009.5363084