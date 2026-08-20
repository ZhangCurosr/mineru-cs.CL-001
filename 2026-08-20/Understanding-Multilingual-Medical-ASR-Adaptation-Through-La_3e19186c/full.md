# Understanding Multilingual Medical ASR Adaptation Through Layer-Wise Analysis

Souranil Kahali<sup>1⋆</sup>, Rituparna Bose<sup>1⋆</sup>, Abner Hernandez<sup>1</sup>, Tomas Arias-Vergara<sup>1</sup>, Andreas Maier<sup>1</sup>, Ning Ma<sup>2</sup>, Paula A. Perez-Toro<sup>1,3</sup>

<sup>1</sup>Pattern Recognition Lab, Friedrich-Alexander-Universitat Erlangen-N ¨ urnberg (FAU), Erlangen, Germany¨ <sup>2</sup>School of Computer Science, University of Sheffield, Sheffield, UK

<sup>3</sup>Chair for AI in Healthcare and Medicine, Technical University of Munich (TUM), Munich, Germany <sup>⋆</sup>Equal contribution

Abstract—Medical automatic speech recognition (MedASR) requires adaptation to specialised terminology, limited annotated clinical data, and multilingual use cases. Although large-scale pretrained ASR models such as Whisper achieve strong generalisation, their behaviour after medical and multilingual adaptation remains insufficiently understood beyond word error rate (WER). This paper investigates how multilingual medical adaptation reshapes the internal representations of Whisper models through layer-wise encoder analysis. We compare zero-shot decoding, English-only fine-tuning, German-only diagnostic fine-tuning, two-stage EN→EN+DE continuation, and direct EN+DE finetuning across Whisper model sizes. Fine-tuning substantially improves MedASR performance, but the best model depends on the adaptation setting: Whisper-Medium gives the lowest English WER (7.72%) and the lowest combined EN+DE WER under direct EN+DE training (26.30%); German-only Whisper-Large-v3 gives the lowest German WER (44.96%), but as a within-corpus diagnostic on 86 single-speaker training utterances rather than robust generalisation. Layer-wise analysis of the twostage Whisper-Small trajectory shows that English medical finetuning produces the dominant encoder shift, whereas multilingual continuation largely preserves the adapted representation space. Domain and language information remain highly recoverable across layers, while linearly recoverable error-predictive cues weaken as WER improves.

Index Terms—Medical ASR, Whisper Fine-tuning, Multilin gual Adaptation, Layer-wise Analysis

## I. INTRODUCTION

Automatic speech recognition (ASR) systems [1]–[3] have made substantial gains in recent years, through Transformerbased architectures and large-scale pretraining. Models such as Whisper [4] and Wav2Vec2 [5] achieve strong performance on general-purpose benchmarks like LibriSpeech [6] and CommonVoice [7]. However, adapting these systems to domainspecific applications-particularly healthcare-presents distinct challenges. Medical speech contains rare and specialised terminology, recorded under varied speaker and acoustic conditions, and high accuracy demands where a single transcription error can carry clinical consequences. Generic ASR models trained on general speech corpora perform poorly in these settings, which motivates targeted domain adaptation.

Prior work has shown that fine-tuning pretrained medical ASR (MedASR) models on domain-specific corpora substantially improves recognition accuracy [8], [9]. Multilingual MedASR has also been explored through initiatives such as MultiMed [10], which introduced benchmarks across clinical languages. In parallel, interpretability research has established that Transformer-based speech encoders organise information hierarchically: lower layers tend to encode acoustic and phonetic information, while deeper layers capture abstract linguistic representations [11], [12]. However, these two research directions, domain adaptation and representation analysis, have largely developed independently. In particular, it remains unclear how medical and multilingual fine-tuning reorganise the internal encoder representations of pretrained ASR models.

Our work aims to close that gap by combining multi-size Whisper evaluation with systematic layer-wise analysis of how medical and multilingual adaptation reshape encoder representations. We define the two-stage EN→EN+DE setting used throughout the paper as English medical fine-tuning followed by continued fine-tuning on combined English–German med ical speech initialised from the English checkpoint. The main contributions are: (i) A systematic comparison of zero-shot, monolingual, continued EN→EN+DE, and direct EN+DE Whisper adaptation across four model sizes; (ii) A layer-wise analysis showing that English medical fine-tuning produces the dominant representational change, whereas multilingual continuation largely preserves the English-adapted representations; and (iii) Evidence that encoder representations become less linearly predictive of transcription errors following adaptation, even as decoder WER improves. Fig. 1 provides a visual overview of the study design. It summarises the data and adaptation pipeline, the main ASR results, and the layer-wise analyses used to interpret the adapted encoder representations.

## II. RELATED WORK

Transformer-based ASR. Large-scale pretrained ASR models are strong starting points for domain adaptation, but their WER performance depends heavily on the target domain. Whisper [4] is an encoder–decoder Transformer trained on large-scale weakly supervised multilingual audio, giving it strong zero-shot and cross-lingual ASR capabilities. Wav2Vec2 [5] uses self-supervised contrastive pretraining and achieves low WER on standard benchmarks after supervised fine-tuning. However, strong general-domain performance does not necessarily transfer to medical speech, where specialised terminology and variable acoustic conditions require targeted adaptation [8], [13].

![](images/f396fe2358cfda0928662811b1465d07e350f1a5fa0be57af3c9f7232239b841.jpg)  
Fig. 1. Overview of the multilingual MedASR adaptation and analysis pipeline. English and German medical speech are used for monolingual fine-tuning two-stage EN→EN+DE continuation, and direct EN+DE fine-tuning. Adapted models are evaluated with WER, and Whisper-Small encoder representations are analysed through drift and probing tasks

Medical ASR. Fine-tuning pretrained ASR models on clinical corpora has consistently improved medical transcription performance [8], [10]. Adedeji et al. [8] showed that domainspecific fine-tuning, combined with large language model postprocessing, improves medical ASR accuracy. The MultiMed project [10] introduced multilingual medical ASR across five languages, demonstrating the feasibility of adapting general ASR models to clinical speech. Its results are not directly comparable to ours because the datasets, languages, clinical scenarios, train/test splits, and decoding setup differ from the English Kaggle and German PoCaP evaluation used here. We therefore restrict direct empirical comparisons to models evaluated on the same test utterances under the same decoding and metric pipeline. Our work builds on this line by studying English–German medical adaptation across multiple Whisper fine-tuning strategies and by providing layer-level evidence of how adaptation changes encoder representations.

Probing and layer-wise analysis. Belinkov and Glass [12] surveyed probing methods for analysing neural language models, establishing the use of lightweight classifiers on frozen representations. Pasad et al. [11] applied layer-wise analysis to self-supervised speech models, showing that different encoder depths encode different phonological and linguistic properties. Prasad and Jyothi [14] probed accent information in endto-end ASR systems. Wiepert et al. [15] examined layer selection for pathological speech feature prediction. To our knowledge, no prior work has jointly examined two-stage multilingual medical ASR adaptation and layer-wise probing of domain, language, and error-predictability signals in a finetuned encoder–decoder ASR model.

Clinical speech analysis. Prior work on pathological speech has often combined acoustic and linguistic embeddings within Transformer-based pipelines [15], [16]. These approaches focus on supervised classification tasks. Our work takes a different angle: we analyse the encoder’s layer-wise behaviour for interpretability, without relying on diagnostic labels.

## III. DATA AND PRE-PROCESSING

This section describes the medical speech datasets and preprocessing pipeline used for the monolingual and multilingual Whisper fine-tuning experiments.

English Medical Corpus. We use the publicly available Kaggle Medical Speech, Transcription, and Intent dataset [17], comprising approximately 8.5 hours of spoken medical phrases with human-written transcriptions. Audio files are resampled to 16 kHz mono before fine-tuning. The dataset is partitioned into speaker-disjoint training, validation, and test splits using speaker identifiers: training (997 samples, 23 speakers), validation (136 samples, 3 speakers), and test (104 samples, 3 speakers). No speaker appears in more than one split, enabling speaker-disjoint evaluation. Mean transcript length is 49.4 characters in training and 52.1 characters in testing.

German Medical Corpus. German data are drawn from the PoCaP Corpus [18], which contains port catheter placement procedures recorded at the Radiology Department of University Hospital Erlangen, Germany. The dataset is not publicly available due to patient privacy regulations. We use audio from the operating physician only. Training uses operation OP-001 (86 samples), while validation (37 samples) and test (38 samples) are file-disjoint halves of OP-002, ensuring no audio overlap. The same resampling and normalisation pipeline is applied as for the English corpus. Because the German training set is small and single-speaker, German results reflect withincorpus adaptation rather than cross-speaker generalisation. We therefore report German-only fine-tuning only as a diagnostic baseline, while the main multilingual comparisons use combined EN+DE training.

Multilingual Dataset. The multilingual dataset is created by merging the English and German splits while preserving text labels and language identifiers. It contains 1,083 training, 173 validation, and 142 test samples, including 104 English and 38 German test utterances. During evaluation, each sample is decoded using language-specific forced decoder identifiers.

Pre-processing and normalisation. Audio is converted to mono and resampled to 16 kHz. We use the corresponding Whisper feature extractor [4], [19] for log-Mel features (80-bin for Base/Small/Medium, 128-bin for Large-v3). Transcripts are kept in their original casing and punctuation; no text normalisation, denoising, voice-activity detection, or augmentation is applied.

General English Contrast Set For the domain probing task, a 100-sample subset of LibriSpeech test-clean [6] is used as a general English contrast. Selecting samples with the same language as the medical set controls for language effects, ensuring that domain probe performance reflects domainspecific encoder content rather than language discrimination.

Reproducibility. We will release the English split manifests, preprocessing scripts, evaluation code, and experiment configurations. The German PoCaP audio cannot be redistributed due to privacy restrictions, but split identifiers and evaluation scripts will be provided.

## IV. FINE-TUNING METHODOLOGY

## A. Training Configuration

All fine-tuning experiments are implemented using Hugging Face Transformers [19] and PyTorch [20] on a single NVIDIA A100 GPU. We evaluate four fine-tuning settings: Englishonly medical fine-tuning, German-only diagnostic fine-tuning, two-stage EN→EN+DE continuation, and direct EN+DE finetuning from pretrained weights.

Training uses Seq2SeqTrainer with cross-entropy loss, greedy decoding for validation WER, and a maximum generation length of 225 tokens. All runs use an effective batch size of 32, mixed precision, gradient checkpointing, and early stopping with patience 5 based on validation WER. Checkpoints are saved every 200 steps, and the best checkpoint is selected by lowest validation WER. We compute word error rate (WER), character error rate (CER), and SemScore [21], where SemScore is the mean cosine similarity between reference and prediction sentence embeddings using paraphrase-multilingual-MiniLM-L12-v2 [22]. For validation and final evaluation, Whisper outputs are generated with greedy decoding and language-specific forced decoder identifiers implemented through the Hugging Face Transformers interface [4], [19]. English-only and German-only checkpoints are decoded on the corresponding monolingual test sets, while EN+DE checkpoints are decoded on language-filtered subsets of the multilingual test set and then pooled for the combined score. Wav2Vec2 baselines use CTC argmax decoding [5]. WER and CER are computed with jiwer on the decoded strings without additional case-folding or punctuation removal. Whisper-Small, Whisper-Medium, and Whisper-Large-v3 are trained under the same configuration; the selected checkpoints are step 400, step 800, and step 400, respectively.

## B. Monolingual Medical Fine-Tuning

For English-only fine-tuning, each Whisper variant is trained on the English medical training set using the shared configuration, with best-checkpoint selection by validation WER. The processor language is fixed to English. The Whisper-Small EN-FT checkpoint used for layer-wise analysis is step 400 (11.98% English test WER); Whisper-Small is chosen because it balances adapted ASR performance with the compute needed for repeated hidden-state extraction across encoder layers.

The German-only diagnostic follows the same configuration on the 86 German training utterances, with best-checkpoint selection by validation WER. Because this setting uses a small single-speaker German training set, only German test performance is reported and interpreted as within-corpus adaptation evidence.

## C. EN+DE Adaptation

For two-stage adaptation, the English fine-tuned checkpoint is used to initialise continued training on the combined English–German dataset. This setting tests whether English medical adaptation can be preserved while introducing German medical speech. As a second multilingual baseline, direct EN+DE fine-tuning starts from pretrained Whisper weights and trains on the same combined English–German data in one stage. In both settings, no validation or test samples are used for gradient updates, batches mix both languages, and samples are decoded with language-specific forced decoder identifiers during evaluation. For the layer-wise analysis, we use the two-stage Whisper-Small multilingual fine-tuned (ML-FT) checkpoint from step 1,400, which achieves 10.91% English and 53.87% German WER.

## V. LAYER-WISE ANALYSIS

We analyse Whisper-Small encoder representations to examine how medical and multilingual fine-tuning affect the internal representation space. The analysis combines hiddenstate extraction, representation-drift measurement, and probing classifiers for domain, language, and error-related information.

## A. Hidden-State Extraction

Whisper-small has 12 encoder Transformer blocks. Including the input embedding output, we extract 13 hidden states per sample (L0–L12). Each hidden state has shape T × 384, where T is the sequence length. We mean-pool each layer over time to obtain one 384-dimensional vector per audio sample and layer. Hidden states are extracted independently from the Pretrained, EN-FT, and multilingual ML-FT checkpoints.

## B. Representation Drift

Representation drift measures how encoder hidden states change after fine-tuning without using task labels. For the same 100 English audio samples, we compute per-layer similarity between checkpoints using two complementary metrics: cosine similarity between mean representations, which captures directional alignment, and Linear Centered Kernel Alignment (CKA) [23], which captures changes in representational geometry. We analyse three pairs: Pretrained → EN-FT, EN-FT → ML-FT, and ML-FT English versus ML-FT German.

## C. Probing Classifiers.

We evaluated three binary probing tasks to analyse what information is encoded across layers.

Domain probe classifies English medical speech (positive) vs. LibriSpeech [6] general English speech. This controls for language but may also reflect corpus-specific differences in accent, speaker population, recording conditions, and background noise.

TABLE I  
ASR WER (%) ON ENGLISH AND GERMAN MEDICAL SPEECH TEST SETS. BASE (B), SMALL (S), MEDIUM (M) AND LARGE (L) WHISPER VERSIONS WHERE ADAPTED.
<table><tr><td>Model</td><td>Adapt</td><td>English↓</td><td>German↓</td><td>Combined.↓</td></tr><tr><td colspan="5">Zero-shot</td></tr><tr><td>WhisperB</td><td>None</td><td>21.30</td><td>70.42</td><td>44.65</td></tr><tr><td>Whispers</td><td>None</td><td>17.57</td><td>62.98</td><td>39.15</td></tr><tr><td>WhisperM</td><td>None</td><td>16.77</td><td>71.99†</td><td>43.02</td></tr><tr><td>WhisperL</td><td>None</td><td>16.50</td><td>56.51</td><td>35.52</td></tr><tr><td colspan="5">Monolingual (Mono) medical fine-tuning</td></tr><tr><td>WhisperB</td><td>Mono</td><td>15.00</td><td>58.86</td><td></td></tr><tr><td>Whispers</td><td>Mono</td><td>11.98</td><td>57.69</td><td></td></tr><tr><td>WhisperM</td><td>Mono</td><td>7.72</td><td>45.94</td><td></td></tr><tr><td>WhisperL</td><td>Mono</td><td>14.37</td><td>44.96</td><td></td></tr><tr><td colspan="5">Two-stage EN→EN+DE (Two-stage) medical fine-tuning</td></tr><tr><td>WhisperB</td><td>Two-stage</td><td>14.91</td><td>58.86</td><td>35.80</td></tr><tr><td></td><td>Two-stage</td><td>10.91</td><td>53.87</td><td>31.33</td></tr><tr><td>WhisperM</td><td>Two-stage</td><td>7.99</td><td>57.49</td><td>31.52</td></tr><tr><td>WhisperL</td><td>Two-stage</td><td>13.22</td><td>53.09</td><td>32.17</td></tr><tr><td colspan="5">Direct multilingual (Multi) medical fine-tuning</td></tr><tr><td>WhisperB</td><td>Multi</td><td>14.64</td><td>61.90</td><td>37.10</td></tr><tr><td>WhisperS</td><td>Multi</td><td>9.94</td><td>52.20</td><td>30.03</td></tr><tr><td>WhisperM</td><td>Multi</td><td>7.81</td><td>46.72</td><td>26.30</td></tr><tr><td>WhisperL</td><td>Multi</td><td>15.35</td><td>47.50</td><td>30.63</td></tr></table>

Mono: EN-only (English) / DE-only diagnostic (German). Multi: direct EN+DE. <sup>†</sup>Wide CI. <sup>‡</sup>Whisper-S two-stage = layer-wise analysis checkpoint.

Language probe classifies English medical vs. German medical speech. Because the two language subsets come from different corpora, the contrast may also reflect speaker and acoustic-condition differences.

These confounds should be considered when interpreting the near-ceiling probe F1 scores reported in Sec. VI.C. WER probe classifies low- vs. high-WER utterances. WER is computed per checkpoint using greedy decoding on 100 English test clips and labelled by rank-based tertile split: bottomthird (low-WER, label=0) vs. top-third (high-WER, label=1), middle third excluded (n=33 per class, applied identically across checkpoints).

For each encoder layer, variable-length hidden-state sequences were converted into fixed-dimensional representations by mean-pooling across the temporal dimension. Each task uses Logistic Regression, Linear SVC, and a one-hidden-layer MLP trained on frozen encoder features with stratified 5-fold cross-validation [24]. Panel-mean macro-F1 is the primary metric. Features are MinMax-scaled and reduced to 50 princi pal components to limit overfitting under small sample sizes. To assess robustness, representation drift and probing analyses are repeated over five bootstrap seeds: 42, 123, 456, 789, and 2024. Layer-wise differences are tested using Kruskal– Wallis tests with Bonferroni and FDR-BH correction [25]. Within each fold, MinMax scaling and PCA reduction to 50 components are fitted only on the training split and then applied to the held-out validation split, preventing validationfold information from entering preprocessing.

## VI. EXPERIMENTS & RESULTS

This section reports ASR performance and representationlevel analyses. We first compare zero-shot and fine-tuned Whisper models with Wav2Vec2 baselines, then analyse Whisper-Small encoder representations to study how medical and multilingual adaptation affect layer-wise information.

## A. ASR Performance: Zero-Shot and Fine-Tuned

Tab. I reports zero-shot and fine-tuned MedASR WER results across all models. Among zero-shot Whisper models, Whisper-Large-v3 performs best on all subsets (English 16.50%, German 56.51%, combined 35.52%). The Wav2Vec2 CTC baselines show poor out-of-domain transfer, with English WER above 92% and German XLSR-53 reaching 76.00%, confirming the need for domain adaptation in this medical setting.

Fine-tuning changes the model ranking. English-only finetuning gives the best English result with Whisper-Medium (7.72% WER), but German transfer without German training is inconsistent: Whisper-Small degrades on German, whereas Whisper-Medium improves relative to its zero-shot baseline. The German-only diagnostic shows useful within-corpus adaptation signal, with Whisper-Large-v3 reaching 44.96% WER and Whisper-Medium 45.94%. Since this setting uses only 86 German training utterances, we do not interpret it as robust German generalisation.

For multilingual training, two-stage EN→EN+DE adaptation improves over zero-shot for all Whisper sizes, with Whisper-Small giving the lowest two-stage combined WER (31.33%). Direct EN+DE fine-tuning gives the best overall combined result, with Whisper-Medium reaching 26.30% combined WER and 46.72% German WER. This may reflect the strong EN–DE data imbalance (997 vs. 86 training utterances): direct EN+DE training learns a shared multilingual representation from the start, whereas continued EN→EN+DE fine-tuning starts from an English-specialised checkpoint. Overall, model scale alone does not determine adapted MedASR performance; the strongest model depends on whether the objective is zero-shot robustness, English adaptation, German within-corpus adaptation, or combined EN+DE performance.

TABLE II  
REPRESENTATIVE TRANSCRIPTION ERRORS. ZS-SMALL / S2-SMALL = WHISPER-SMALL ZERO-SHOT / TWO-STAGE EN→EN+DE; DE-LARGE = GERMAN-ONLY WHISPER-LARGE-V3. ITALICS MARK THE CONTESTED TERM.
<table><tr><td>Type</td><td>Utterances</td></tr><tr><td>EN symptom term EN clinical term</td><td>REF: stomach pain and bloating; ZS: stomach pain and rotting; S2: stomach pain and bloating. REF: could I have a concussion; ZS: could I have a competition; S2: could I have a concussion.</td></tr><tr><td>DE imaging term</td><td>REF: Röntgenthorax, Pneu; ZS: Röntgen Torax, Pneuer; S2: röntgen torax, panor; DE-Large: röntgen thorax, pneu.</td></tr><tr><td>DE anatomy terms</td><td>REF: Klavikola, Vene, Terumo-Nadel; ZS: Gravikula, Wene, Terumo; S2: gravikulare, vene, pteromunadel; DE-Large: klavikula, vene, terumonadel.</td></tr><tr><td>DE hallucination</td><td>REF: die Prostater drüben; ZS/S2: Brotzeit verdrücken/verdrüben; DE-Large: die Prostata drüben.</td></tr></table>

Because the layer-wise analysis below is conducted on the two-stage Whisper-Small trajectory, the representationlevel findings should be interpreted as explaining that specific adaptation path; comparable analyses of direct EN+DE and German-only encoders are left for future work.

Qualitative error audit. To complement aggregate WER, Tab. II shows representative examples from three checkpoints: zero-shot Whisper-Small (ZS-Small), two-stage Whisper-Small EN→EN+DE (S2-Small, the layer-wise model), and German-only Whisper-Large-v3 (DE-Large, 44.96% German WER). English examples compare ZS-Small and S2-Small; DE-Large is included for German rows only, where it achieves the best German WER.

## B. Representation Drift

The drift analysis in Fig. 2 reveals a clear asymmetry between the two adaptation stages. Pretrained→EN-FT produces the largest representational shift, most pronounced in upper layers (L8–L12), indicating that English medical finetuning drives the main encoder reorganisation. In contrast, EN-FT→ML-FT shows substantially smaller drift, suggesting that multilingual continuation largely preserves the Englishadapted representation space.

The ML-FT English vs. German comparison yields high cosine similarity (0.980–0.994) but low Linear CKA (0.040– 0.107), indicating centroid-level cross-lingual alignment while the full representation geometry remains language-sensitive. Tab. III summarises the bootstrap-averaged per-layer ranges.

![](images/e8e30362876a8861b2b7d97fb90ecbf6e9e685a3d973263367d5b77e61188554.jpg)

![](images/31eaaca07e356ed58eda03c79259d0b5ef0ca66494b4e83b6386311639071c15.jpg)  
Fig. 2. Bootstrap-averaged representation drift across encoder layers (L0– L12) over five seeds; shading denotes ±1 std. Cosine similarity is shown on the left and Linear CKA on the right. Pretrained→EN-FT produces the largest upper-layer shift, whereas EN-FT→ML-FT remains close to the adapted encoder. ML-FT EN vs. DE shows high cosine similarity but low Linear CKA, indicating centroid alignment with language-sensitive geometry.

TABLE III  
REPRESENTATION DRIFT SUMMARY OVER FIVE BOOTSTRAP SEEDS. VALUES REPORT THE PER-LAYER RANGE ACROSS L0–L12.
<table><tr><td>Drift pair</td><td>Cosine range</td><td>Lin. CKA range</td></tr><tr><td>Pretrained → EN-FT</td><td>0.906-1.000</td><td>0.946-1.000</td></tr><tr><td> $\mathrm { E N - F T }  \mathrm { M L - F T }$ </td><td>0.989-1.000</td><td>0.988-1.000</td></tr><tr><td>ML-FT: EN vs. DE</td><td>0.980–0.994</td><td>0.040–0.107</td></tr></table>

## C. Probing Results

Domain probe. Fig. 3(a) shows that domain information remains highly separable across all checkpoints and layers, with panel-mean macro-F1 at or near ceiling (≥ 0.984; see Tab. IV). Because many layers achieve nearly identical F1, exact peaklayer claims are not meaningful. The main finding is that medical vs. general speech remains consistently recoverable from frozen encoder representations across adaptation stages. For ML-FT, silhouette scores decrease from 0.33 at L0 to 0.11 at L11, suggesting that deeper-layer domain representations remain classifiable but less geometrically compact.

Language Probe. Fig. 3(b) and Tab. IV show that languageprobe F1 remains near ceiling under bootstrap aggregation. Language identity is almost perfectly separable across checkpoints and layers, with bootstrap mean $\mathrm { F 1 ~  { ~ \geq ~ } ~ 0 . 9 9 0 }$ from L1 onward across all checkpoints. This indicates that Whisper’s multilingual pretraining encodes robust languagediscriminative features that are preserved through both finetuning stages. Since performance is at ceiling across nearly all depths, the result is best interpreted as evidence of preserved language-discriminative structure rather than a meaningful single-layer peak.

WER/Error-Predictability Probe. The WER/errorpredictability probe in Fig. 3(c) and Tab. IV shows an overall decrease in linearly recoverable error information after adaptation, although the layer-wise curves fluctuate across encoder depth. Using rank-based low-/high-WER tertile labels, the best-layer panel-mean macro-F1 drops from $0 . 7 2 1 \pm 0 . 0 2 8$ at L2 for the Pretrained encoder (mean across layers: 0.666), to $0 . 6 1 9 \pm 0 . 0 3 9$ at L6 after EN-FT (mean: 0.569), and to $0 . 5 5 6 \pm 0 . 0 3 3$ at L11 after ML-FT (mean: 0.509, close to binary-chance level). In parallel, decoder WER on the same 100 English clips decreases from 19.87% to 13.61% and 12.48%. Thus, as decoding improves, residual errors become less linearly predictable from frozen encoder activations, suggesting that adaptation reorganises failure modes that were more separable in the pretrained representation space.

![](images/6f53cbd2ed86f3e70213d20cd2cc832e797cf8e48ba86e4fc15f65e5c3dc4f8e.jpg)

![](images/536596dfc6cb0aa69c818724803d4722ff0fa78ffe567ad8a42043581dfa8f69.jpg)

![](images/bd47621b280bad8c7bbfff3b0fa401190ed352844889e4e5f53c31f693a898ae.jpg)  
Fig. 3. Layer-wise probe macro-F1 across encoder layers (L0–L12) for the three probing tasks, bootstrap-aggregated over five seeds (shading: ±1 std). (a) Domain probe (medical English vs. LibriSpeech) and (b) Language probe (English medical vs. German medical) remain near ceiling across all checkpoints and layers, so peak-layer claims are not meaningful here. (c) Error-predictability probe uses rank-based low-/high-WER tertile labels. The pretrained encoder carries the strongest linearly recoverable error signal at the best layer; this signal weakens after English fine-tuning and multilingual continuation while decoder WER improves on the same 100 clips $( 1 9 . 8 7 \% \xrightarrow { } 1 3 . 6 1 \% \xrightarrow { } 1 2 . 4 \hat { 8 } \% )$

Kruskal–Wallis tests pooling 13 layers over five bootstrap seeds (n=65 observations per test) show significant layer-wise variation for all domain and language probes after Bonferroni and FDR-BH correction (all adjusted $q < 1 0 ^ { - 4 } )$ . For the WER probe, only the pretrained checkpoint remains significant after correction $( p = 1 . 4 7 \times 1 0 ^ { - 3 }$ ; Bonferroni $q = 1 . 3 2 \times 1 0 ^ { - 2 }$ FDR-BH $q = 1 . 8 9 \times 1 0 ^ { - 3 } )$ , whereas EN-FT $( p = 0 . 0 5 1 )$ and ML-FT $( p = 0 . 2 1 3 )$ are not significant under either correction. Domain and language information thus remain robustly layerdependent, while error-predictive information becomes weaker and no longer significantly layer-localised after fine-tuning.

TABLE IV  
PROBE SUMMARY OVER FIVE BOOTSTRAP SEEDS. DOMAIN ANDLANGUAGE REPORT THE MINIMUM MACRO-F1 ACROSS LAYERS; WER F1REPORTS THE BEST-LAYER LOW-/HIGH-WER PROBE RESULT UNDER ARANK-BASED TERTILE SPLIT (n=33 PER CLASS).
<table><tr><td>Checkpoint Domain F1 Lang. F1</td><td>min</td><td>min</td><td>WER F1 best layer</td><td>Dec. WER (%)</td></tr><tr><td>Pretrained</td><td> $\geq 0 . 9 8 4$ </td><td> $\geq 0 . 9 9 0$ </td><td> $0 . 7 2 1 \pm 0 . 0 2 8 \ ( \mathrm { L } 2 )$ </td><td>19.87</td></tr><tr><td>EN-FT</td><td> $\geq 0 . 9 8 4$ </td><td> $\geq 0 . 9 9 3$ </td><td> $0 . 6 1 9 \pm 0 . 0 3 9 \ ( \mathrm { L } 6 )$ </td><td>13.61</td></tr><tr><td>ML-FT</td><td>≥ 0.985</td><td> $\geq 0 . 9 9 3$ </td><td> $0 . 5 5 6 \pm 0 . 0 3 3 ( \mathrm { L 1 1 } )$ </td><td>12.48</td></tr></table>

Dec. WER is computed on the same 100 English utterances used to label the WER probe (greedy decoding) and is not layer-wise.

## VII. DISCUSSION & CONCLUSION

This work investigates medical ASR using multi-size Whisper evaluation, English–German adaptation strategies, and layer-wise encoder analysis. Fine-tuning substantially improves MedASR over zero-shot decoding, but model scale alone does not determine the best adapted model under limited clinical data: Whisper-Medium is strongest under Englishonly and direct EN+DE fine-tuning, while Whisper-Large-v3 leads under the German-only diagnostic, which we interpret cautiously as within-corpus adaptation on 86 single-speaker training utterances.

The layer-wise analysis yields three messages. (i) Medical adaptation dominates representational change. English medical fine-tuning produces the largest encoder drift, concentrated in upper layers. (ii) Multilingual continuation preserves learned representations. EN-FT→ML-FT drift is substantially smaller, and the domain and language probe contrasts remain highly separable across layers; their near-ceiling separability partly reflects corpus, speaker, and acoustic-condition confounds (Sec. V-C), so exact peak-layer claims should be treated cautiously. (iii) Improved ASR does not imply stronger linear error signals. The WER probe weakens after each adaptation step even as decoder WER improves, suggesting that fine-tuning reduces simple linearly recoverable failure cues in the encoder.

These findings complement standard WER evaluation and suggest several directions. The persistence of domain and language information across layers motivates parameter-efficient or selective fine-tuning. The weakening of WER-predictive signals suggests a role for difficulty-aware data selection using pretrained encoder representations. The strong separability of the English–German medical contrast motivates further study of shared multilingual encoder architectures for clinical ASR.

Limitations. The medical corpora are small, particularly the single-procedure German subset (86 train / 38 test), and probing uses 100–138 utterances per task; the layer-wise analysis covers only Whisper-Small. Fine-grained layer-specific, cross-speaker, and cross-scale conclusions should therefore be treated cautiously. Future work should validate these findings on larger multilingual clinical corpora and extend the layerwise analysis to larger Whisper variants, direct EN+DE, German-only, and parameter-efficient adaptation strategies.

## VIII. GENERATIVE AI USE DISCLOSURE

Generative artificial intelligence tools were used to assist with language editing, clarity of presentation, and code drafting/debugging. All research ideas, methodology, experiments, analyses, and interpretations were conceived, verified, and carried out by the authors, who take full responsibility for the originality, validity, and integrity of the work.

## REFERENCES

[1] A. Graves, A.-r. Mohamed, and G. Hinton, “Speech recognition with deep recurrent neural networks,” in 2013 IEEE International Conference on Acoustics, Speech and Signal Processing, 2013, pp. 6645–6649.

[2] W. Chan, N. Jaitly, Q. Le, and O. Vinyals, “Listen, attend and spell: A neural network for large vocabulary conversational speech recognition,” in 2016 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE Press, 2016, p. 4960–4964. [Online]. Available: https://doi.org/10.1109/ICASSP.2016.7472621

[3] Q. Zhang, H. Lu, H. Sak, A. Tripathi, E. McDermott, S. Koo, and S. Kumar, “Transformer transducer: A streamable speech recognition model with transformer encoders and rnn-t loss,” in ICASSP 2020- 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2020, pp. 7829–7833.

[4] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[5] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449– 12 460, 2020.

[6] V. Panayotov, G. Chen, D. Povey, and S. Khudanpur, “Librispeech: an asr corpus based on public domain audio books,” in 2015 IEEE international conference on acoustics, speech and signal processing (ICASSP). IEEE, 2015, pp. 5206–5210.

[7] R. Ardila, M. Branson, K. Davis, M. Henretty, M. Kohler, J. Meyer, R. Morais, L. Saunders, F. M. Tyers, and G. Weber, “Common voice: A massively-multilingual speech corpus,” 2020. [Online]. Available: https://arxiv.org/abs/1912.06670

[8] A. Adedeji, S. Joshi, and B. Doohan, “The sound of healthcare: Improving medical transcription asr accuracy with large language models,” arXiv preprint arXiv:2402.07658, 2024.

[9] A. Hernandez, T. Arias-Vergara, A. Maier, and P. A. Perez-Toro,´ “Enhancing ASR accuracy for speakers with parkinson’s disease using instruction-tuned LLMs,” in International Conference on Text, Speech, and Dialogue. Springer, 2025, pp. 153–164.

[10] K. Le-Duc, P. Phan, T.-H. Pham, B. P. Tat, M.-H. Ngo, T. Nguyen-Tang, and T.-S. Hy, “Multimed: Multilingual medical speech recognition via attention encoder decoder,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track), 2025, pp. 1113–1150.

[11] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a self-supervised speech representation model,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021, pp. 914–921.

[12] Y. Belinkov and J. Glass, “Analysis methods in neural language processing: A survey,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 49–72, 2019. [Online]. Available: https://aclanthology.org/Q19-1004/

[13] Y. Liu, X. Yang, and D. Qu, “Exploration of whisper fine-tuning strategies for low-resource asr,” EURASIP Journal on Audio, Speech, and Music Processing, vol. 2024, no. 1, p. 29, 2024.

[14] A. Prasad and P. Jyothi, “How accents confound: Probing for accent information in end-to-end speech recognition systems,” in Proceedings of the 58th annual meeting ofthe associationfor computational linguistics, 2020, pp. 3739–3753.

[15] D. A. Wiepert, R. L. Utianski, J. R. Duffy, J. L. Stricker, L. R. Barnard, D. T. Jones, and H. Botha, “Speech foundation models in healthcare: Effect of layer selection on pathological speech feature prediction,” arXiv preprint arXiv:2402.01796, 2024.

[16] P. A. Perez-Toro, S. P. Bayerl, T. Arias-Vergara, J. C. V´ asquez-Correa,´ P. Klumpp, M. Schuster, E. Noth, J. R. Orozco-Arroyave, and K. Ried-¨ hammer, “Influence of the interviewer on the automatic assessment of alzheimer’s disease in the context of the adresso challenge.” in Interspeech, 2021, pp. 3785–3789.

[17] P. Mooney, “Medical speech, transcription, and intent dataset,” https://www.kaggle.com/datasets/paultimothymooney/ medical-speech-transcription-and-intent, 2020, accessed: June 2026.

[18] K. C. Demir, M. May, A. Schmid, M. Uder, K. Breininger, T. Weise, A. Maier, and S. H. Yang, “Pocap corpus: A multimodal dataset for smart operating room speech assistant using interventional radiology workflow analysis,” in International conference on text, speech, and dialogue. Springer, 2022, pp. 464–475.

[19] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush, “Huggingface’s transformers: State-of-the-art natural language processing,” 2020. [Online]. Available: https://arxiv.org/abs/1910.03771

[20] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Kopf,¨ E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala, “Pytorch: An imperative style, high-performance deep learning library,” 2019. [Online]. Available: https://arxiv.org/abs/1912.01703

[21] A. Aynetdinov and A. Akbik, “Semscore: Automated evaluation of instruction-tuned llms based on semantic textual similarity,” 2024. [Online]. Available: https://arxiv.org/abs/2401.17072

[22] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” 2019. [Online]. Available: https: //arxiv.org/abs/1908.10084

[23] S. Kornblith, M. Norouzi, H. Lee, and G. Hinton, “Similarity of neural network representations revisited,” in International conference on machine learning. PMlR, 2019, pp. 3519–3529.

[24] G. Alain and Y. Bengio, “Understanding intermediate layers using linear classifier probes,” 2018. [Online]. Available: https://arxiv.org/abs/ 1610.01644

[25] Y. Benjamini and Y. Hochberg, “Controlling the false discovery rate: A practical and powerful approach to multiple testing,” Journal of the Royal Statistical Society. Series B (Methodological), vol. 57, no. 1, pp. 289–300, 1995. [Online]. Available: http://www.jstor.org/stable/2346101