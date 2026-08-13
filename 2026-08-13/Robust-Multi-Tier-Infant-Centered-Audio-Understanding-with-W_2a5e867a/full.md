# Robust Multi-Tier Infant-Centered Audio Understanding with Whisper via Structured Speaker Conditioning

Xulin Fan<sup>1</sup>, Jialu Li<sup>2</sup>, Mohammad Nur Hossain Khan<sup>3</sup>, Kexin Hu<sup>1</sup>, Bashima Islam<sup>3</sup>, Mark Hasegawa-Johnson<sup>1</sup>, Nancy L. McElwain<sup>1</sup>

<sup>1</sup> University of Illinois Urbana-Champaign, USA <sup>2</sup> University of Arizona, USA

<sup>3</sup> Worcester Polytechnic Institute, USA

xulinf2@illinois.edu, jialuli@arizona.edu, mkhan@wpi.edu, kexinhu2@illinois.edu, bislam@wpi.edu, jhasegaw@illinois.edu, mcelwn@illinois.edu

## Abstract

Recent advances in model design and self-supervised audio representations have improved speech and audio understanding, yet infant-centered naturalistic recordings remain challenging due to limited labeled data, low signal-to-noise ratio, and cross-family domain shifts. We present a family-conditioned, multi-tier audio tagger that combines a LoRA-finetuned Whisper encoder with a lightweight, target-speaker–aware Transformer for long-context inference and framewise prediction across tiers. To improve temporal coherence, we incorporate a simple sequence-level smoothing loss, and to enhance robustness across households, we introduce a factorized speaker-token design with a shared tier token and a learned family-specific offset, reducing family bias and promoting generalizable representations. Together, these choices enable efficient and effective infant-centered audio tagging of daylong audio recordings in home environments.

Index Terms: Audio Tagging, Speaker Diarization, Vocalization Classification

## 1. Introduction

Infant-centered audio understanding aims to characterize infant–adult interactions across naturalistic settings such as homes and clinics. Depending on the label taxonomy, the task can be formulated as speaker diarization, vocalization classification, or a combination of both. Early studies primarily relied on supervised neural models tailored to specific subtasks, including child–parent speech classification [1] and vocalization categorization [2].

Self-supervised learning (SSL) has substantially advanced speech and audio understanding by enabling models to learn robust representations from large-scale unlabeled corpora before fine-tuning on task-specific data. Foundational SSL encoders such as Wav2Vec2 [3], HuBERT [4], and SSAST [5] consistently outperform purely supervised approaches across diverse downstream tasks. These general-purpose representations have also been successfully applied to infant-centered audio tagging, where the goal is to analyze naturalistic audio recordings collected using wearable devices and identify speaker and vocalization types. Several recent works leveraging SSL frameworks for child–parent interaction analysis [6, 7, 8] demonstrate improved robustness and performance compared with supervised baselines, highlighting the value of transferable audio representations.

An alternative paradigm bypasses additional in-domain pretraining on infant-centered home recordings by directly adapting large-scale foundational speech models [9, 10]. Trained on massive and diverse datasets, these models provide strong and noise-robust acoustic features. In particular, Whisper has been successfully adapted using parameter-efficient techniques such as low-rank adaptation (LoRA) [11] for domain-shifted speech recognition tasks, including multilingual and child speech scenarios [12, 13, 14]. Beyond recognition, foundational encoders have also been explored for tagging tasks. For example, Whisper-AT [15] freezes the Whisper encoder and trains lightweight fusion and classification layers, demonstrating that Whisper representations can effectively support general audio tagging with minimal task-specific overhead. Similarly, recent work investigates adapting foundation models for child–parent diarization with promising results [16], while synthetic data augmentation has been shown to further improve performance in real-world child–parent interaction settings [17].

Despite this progress, infant-centered audio understanding remains challenging. First, unlike standard clip-level datasets such as AudioSet, naturalistic recordings require frame-level labeling at fine temporal resolution, effectively combining diarization with vocalization classification and demanding precise boundary detection. Second, recording conditions vary substantially across households, including differences in the physical environment, background noise, and speaker characteristics, which introduces significant domain shifts between families. Third, wearable microphones produce variable signal quality due to changing distances between the device and speakers, overlapping and background speech at low signal-to-noise ratios are also common, complicating reliable target-speaker analysis.

To address these challenges, we propose a multi-tier, framelevel audio tagging framework that jointly performs speakeraware diarization and vocalization classification in naturalistic infant-centered recordings. Specifically, the model predicts, for each tier (e.g., child, female caregiver, male caregiver, sibling), a frame-aligned sequence of mutually exclusive vocalization labels while allowing simultaneous activity across tiers. Our approach integrates a LoRA-finetuned Whisper encoder with a lightweight sequence model and tier-specific classifiers, and we compare against strong SSL baselines, including Wav2Vecbased systems, to contextualize performance gains. The main contributions of this work are:

1. A sequence-level infant-centered audio tagging framework that combines a LoRA-adapted Whisper encoder, a projector, a target-speaker extractor, and tier-specific classifiers. Unlike prior approaches [7, 6], it effectively models overlapping vocalizations from multiple family members.

2. A factorized speaker-token representation consisting of a shared tier token and a family-specific offset, which is designed to separate tier-level information from familydependent variability and empirically improves cross-family performance on our evaluation split.

3. A sequence-level auxiliary temporal smoothing loss that stabilizes predictions over time and promotes globally consistent framewise labeling.

![](images/76799ca543610d01517cad2438d9bc8508c465b0f09f9e001601719e398f6e3b.jpg)  
Figure 1: Training Stage of Proposed Framework

## 2. Methods

In this section, we will outline our task formulation in section $2 . 1 .$ model architecture in section 2.2, and our design of the loss functions in section 2.3.

## 2.1. Problem Formulation

Let F be a set of families partitioned into training, validation, test families with no overlap. Each family $f \in { \mathcal { F } }$ has $R _ { f }$ audio recordings, indexed by $r \in \{ 1 , \ldots , R _ { f } \}$ , collected as $\chi _ { f } ~ =$ $\left\{ { \bf x } ^ { \left( f , r \right) } \right\} _ { r = 1 } ^ { R _ { f } }$ , where r denotes the r-th clip from family $f .$ Fix a frame duration $\Delta > 0$ (seconds). For a recording $\mathbf { x } ^ { ( f , r ) }$ with duration $T$ seconds, the number of frames is $L ^ { ( f , r ) } = \left\lceil T / \Delta \right\rceil$

We use a fixed set of tiers T = {CHN, FAN, MAN, CXN}, corresponding to the child, female caregiver, male caregiver, and sibling tiers, respectively. Each tier $\tau \in \tau$ has its own label set $\mathcal { V } _ { \tau }$ , where INACTIVE denotes no activity in that tier. For example, the child tier uses

$$
\mathcal { V } _ { \mathrm { C H N } } = \{ \mathrm { I N A C T I V E } , \mathrm { B A B } , \mathrm { C R Y } , \mathrm { F U S } \} ,
$$

where BAB, CRY, and FUS denote babbling, crying, and fussing. Other label abbreviations used in this work include ADS, CDS, SNG, LAU, and CXN for adult-directed speech, childdirected speech, singing/rhythmic speech, laughter, and sibling vocalization. Labels are chosen per frame and are mutually exclusive within a tier, while different tiers may be active simultaneously.

Given a clip and its known family ID $\left( \mathbf { x } ^ { ( f , r ) } , f \right)$ , the task is to output one frame-aligned label sequence per tier $\hat { \mathbf { y } } _ { 1 : L } ^ { ( \tau ) } ( f , r ) \colon$

$$
g \colon ( \mathbf { x } ^ { ( f , r ) } , f ) \longmapsto \left\{ \hat { \mathbf { y } } _ { 1 : L ^ { ( f , r ) } } ^ { ( \tau ) } \right\} _ { \tau \in \mathcal { T } } , \qquad \hat { \mathbf { y } } _ { 1 : L ^ { ( f , r ) } } ^ { ( \tau ) } \in \mathcal { V } _ { \tau } .
$$

## 2.2. Model Architecture

Our model is designed to produce framewise, multi-tier tag sequences from a single audio clip. As illustrated in Fig 1, the architecture consists of four main components: a Whisper encoder backbone, an MLP downsampler, a target speaker extractor, and per-tier framewise classifiers.

Whisper Encoder. The acoustic backbone of our model is a pre-trained Whisper-large-v2 encoder [9]. It processes the raw input waveform into a sequence of D-dimensional hidden representations, capturing rich acoustic information from the audio. The whisper encoder is finetuned with Low rank Adaptation (LoRA) during training.

MLP Projector. Motivated by Slam-LLM [18], we apply a non-overlapping, windowed MLP to the encoder’s output to reduce sequence length for computational efficiency and aggregate local context. Consecutive frames are grouped into windows of size $w ,$ flattened, and then projected by a twolayer MLP. This process yields a shorter feature sequence $Z \in$ $\mathbb { R } ^ { \mathrm { { } } T ^ { \prime } \times D ^ { \prime } }$ , which is shared across all subsequent tiers.

Target Speaker Extractor. This module refines the shared acoustic features Z to focus on a specific target speaker for each tier τ (i.e., CHN, FAN, MAN, CXN). This is achieved by conditioning a lightweight Transformer encoder on a family-aware speaker token.

The speaker token is constructed from two learnable components. First, each tier is associated with a tier token $s _ { \tau } ,$ which represents the average speaker profile for that category $( \mathrm { e . g . }$ the general vocal characteristics of a child). Second, to account for systematic variation across training families, such as recording conditions or speaker characteristics, we introduce a speaker offset $O _ { T , f }$ which is unique to a specific tier in a specific family.

The final family-aware speaker token is

$$
\tilde { s } _ { \tau , f } = s _ { \tau } + o _ { \tau , f } .
$$

This factorized design encourages the tier token $s _ { \tau }$ to capture family-invariant speaker characteristics, while the offset absorbs family-dependent variability. As a result, the shared tier tokens become more robust and generalizable representations across households. This token is prepended to the downsampled sequence $Z ,$ forming the input $\left[ \tilde { s } _ { \tau , f } ; Z \right]$ for the Transformer encoder.

After processing, the output corresponding to the prepended speaker token is discarded. The remaining outputs form a tier-specific, frame-aligned feature sequence $U _ { \tau } \in$ $\mathbf { \mathbb { R } } ^ { T ^ { \prime } \times D ^ { \prime } }$ . This module effectively acts as a speaker extractor, steering the acoustic representation to highlight features relevant to the intended speaker tier in a specific family.

Per-Tier Framewise Classifiers. Finally, for each tier τ , a dedicated two-layer MLP serves as a framewise classifier. It takes the tier-specific sequence $U _ { \tau }$ as input and, at each time step, produces the final classification labels for that tier.

<table><tr><td rowspan="2">Split</td><td rowspan="2">#Fam.</td><td colspan="3">CHN</td><td colspan="4">FAN</td><td colspan="2">MAN</td><td>CXN</td></tr><tr><td>BAB</td><td>CRY</td><td>FUS</td><td>ADS</td><td>CDS</td><td>SNG</td><td>LAU</td><td>ADS</td><td>CDS</td><td>CXN</td></tr><tr><td>Train</td><td>37</td><td>3857.7</td><td>415.6</td><td>2280.4</td><td>2457.1</td><td>2994.7</td><td>743.0</td><td>146.6</td><td>1119.9</td><td>679.2</td><td>1340.2</td></tr><tr><td>Val</td><td>5</td><td>539.9</td><td>83.2</td><td>589.8</td><td>418.1</td><td>412.2</td><td>133.3</td><td>13.0</td><td>281.8</td><td>35.6</td><td>164.2</td></tr><tr><td>Test</td><td>10</td><td>1285.5</td><td>160.9</td><td>781.7</td><td>736.1</td><td>919.1</td><td>235.4</td><td>69.9</td><td>237.8</td><td>123.3</td><td>412.3</td></tr></table>

Table 1: Detailed statistics of the training, validation, and testing splits for annotated data. Each column reports the total annotated duration for each active vocalization labels (in seconds). Columns from left to right are: number of families; child vocalizations (CHN: BAB for babbling, CRYfor crying, FUS for fussing); female caregiver vocalization (FAN: ADS for adult-directed speech, CDS for child-directed speech, SNGfor singing/rhythmic speech, LAUfor laughter); male caregiver vocalization (MAN: ADSfor adult-directed speech, CDSfor child-directed speech); and sibling vocalizations (CXN).

## 2.3. Training Objective

Because training and test families do not overlap, speaker offsets are learned only for training families. At inference time, offsets are set to zero so that predictions rely solely on the shared tier tokens, which represent family-invariant speaker characteristics learned during training.

Let $Z _ { t } ^ { ( \tau ) }$ denote the logits at frame t for tier $\tau ,$ and define the posterior as

$$
\pi _ { t } ^ { ( \tau ) } = \mathrm { s o f t m a x } \big ( Z _ { t } ^ { ( \tau ) } / T \big )
$$

where $T > 0$ is a temperature parameter. In addition to the standard framewise cross-entropy (CE) loss for each tier, we introduce a sequence-level temporal smoothing loss:

$$
\mathcal { L } _ { \mathrm { s m o o t h } } ^ { ( \tau ) } = \frac { 1 } { L - 1 } \sum _ { t = 1 } ^ { L - 1 } \lVert \boldsymbol { \pi } _ { t + 1 } ^ { ( \tau ) } - \boldsymbol { \pi } _ { t } ^ { ( \tau ) } \rVert _ { 2 } ^ { 2 } .
$$

Empirically, we observe that framewise predictions may oscillate between labels within a single vocalization segment when the model is uncertain. The smoothing loss mitigates this behavior by penalizing rapid posterior changes across adjacent frames, thereby encouraging temporally coherent predictions that better reflect the duration structure of vocalizations.

The overall training objective is

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \frac { 1 } { | \mathcal { T } | } \sum _ { \tau \in \mathcal { T } } \Bigl ( \mathcal { L } _ { \mathrm { C E } } ^ { ( \tau ) } + \lambda \mathcal { L } _ { \mathrm { s m o o t h } } ^ { ( \tau ) } \Bigr ) .
$$

## 3. Experiments

## 3.1. Dataset

We conducted experiments on approximately 17 hours of labeled audio recordings from 52 families, collected using the LittleBeats™ [19] wearable device. Infants (55 percent female) were between 4 and 15 months of age (Mean = 8.03 months). The data were partitioned into training (37 families), validation (5 families), and test (10 families) sets with no family overlap. Audio recordings were annotated by trained research assistants using Praat software [20]. All files were double annotated, and Cohen’s kappa was 0.80 or higher for all codes.

For this study, we defined four annotation tiers $\boldsymbol { \mathcal { T } } =$ {CHN, FAN, MAN, CXN}, each with its own set of labels:

$$
\mathcal { V } _ { \mathrm { C H N } } = \{ \mathrm { I N A C T I V E } , \mathrm { B A B } , \mathrm { C R Y } , \mathrm { F U S } \} ,
$$

$$
\begin{array} { r l } & { \mathcal { V } _ { \mathrm { F A N } } = \{ \mathrm { I N A C T I V E , A D S , C D S , S N G , L A U } \} , } \\ & { \qquad \mathcal { V } _ { \mathrm { M A N } } = \{ \mathrm { I N A C T I V E , A D S , C D S } \} , } \\ & { \qquad \mathcal { V } _ { \mathrm { C X N } } = \{ \mathrm { I N A C T I V E , C X N } \} . } \end{array}
$$

The total annotated durations for each active label across the train, validation, and test sets are summarized in Table 1.

## 3.2. Experiment Configuration

Our model takes 30-second audio samples as input. During training, we randomly select 30-second intervals from the recordings to increase coverage of diverse acoustic contexts. For evaluation and testing, the recordings are divided into nonoverlapping 30-second segments.

The Whisper encoder produces frame-level embeddings of dimension $D = 1 2 8 0$ . These embeddings are projected by an MLP that compresses every $w = 5$ consecutive frame into a single embedding of size $D ^ { \prime } = 5 1 2$ , thereby reducing the temporal resolution by a factor of five. The target-speaker extractor consists of a two-layer Transformer encoder with 8 attention heads, a feed-forward dimension of $4 D ^ { \prime } = 2 0 4 8$ , a dropout rate of 0.1, and sinusoidal positional encodings. Tier tokens are randomly initialized, and all speaker offsets are initialized to zero.

For training, we apply low-rank adaptation (LoRA) [11] to the query and value projections of the Whisper encoder, using rank r = 4 and scaling factor $\alpha = 8 .$ All models are trained for 20 epochs with the Adam optimizer at a learning rate of 0.001, with the temporal smoothing loss weighted by $\lambda = 0 . 2$ . The checkpoint achieving the highest kappa score on the validation set is selected for final evaluation. Training requires approximately three hours on a single NVIDIA A100 GPU.

## 3.3. Results

Table 2 reports per-tier Macro-F1 and Cohen’s κ together with across-tier averages (AVG) for our method, external baselines, and ablations. For each tier, both metrics are computed over all classes in that tier, including the INACTIVE class.

Baselines. We compare against two prior approaches. First, we adapt the Time and layer-wise Transformer (TL-TR) with 512 intermediate dimension from Whisper-AT [15] for framewise prediction by modifying its temporal resolution (reducing downsampling from 20× to 5× and removing the second downsampling stage) and attaching the same tier-wise classification heads used in our system.

Second, we include results from W2V-LB [6], which is based on wav2vec 2.0[3] representations pretrained on largescale family audio. Specifically, we use the model variant that aggregates representations from all 12 transformer layers using a learned weighted average and is pretrained on approximately 4300 hours of unlabeled home recordings. Because W2V-LB is designed for single-tier prediction, it cannot model simultaneous active tiers. Therefore, we evaluate it under two protocols. In both cases, training data are preprocessed to remove overlapping active tiers using a fixed priority order (CHN > FAN > $\mathbf { M A N } > \mathbf { C X N } )$ . For evaluation, W2V-LB\* reports performance on test labels processed with the same overlap-removal rule, reflecting its intended operating condition. In contrast, W2V-LB is evaluated on the original multi-tier test labels, providing a direct comparison with our method but necessarily penalizing predictions when multiple tiers are active simultaneously.

<table><tr><td rowspan="2">Methods</td><td colspan="2">AVG</td><td colspan="2">CHN</td><td colspan="2">FAN</td><td colspan="2">MAN</td><td colspan="2">CXN</td></tr><tr><td>Macro-F1</td><td>Kappa</td><td>Macro-F1</td><td>Kappa</td><td>Macro-F1</td><td>Kappa</td><td>Macro-F1</td><td>Kappa</td><td>Macro-F1</td><td>Kappa</td></tr><tr><td> $\mathrm { T L \mathrm { - } T R _ { 5 1 2 } } \ [ 1 5 ]$ </td><td>69.55</td><td>64.04</td><td>58.42</td><td>69.01</td><td>69.76</td><td>67.91</td><td>72.77</td><td>64.61</td><td>77.26</td><td>54.63</td></tr><tr><td>W2V-LB [6]</td><td>67.27</td><td>59.27</td><td>68.72</td><td>71.97</td><td>61.90</td><td>58.79</td><td>61.07</td><td>51.48</td><td>77.39</td><td>54.81</td></tr><tr><td>W2V-LB* [6]</td><td>68.11</td><td>60.68</td><td>68.72</td><td>71.97</td><td>63.24</td><td>60.73</td><td>61.54</td><td>52.11</td><td>78.95</td><td>57.90</td></tr><tr><td>Proposed</td><td>74.88</td><td>68.14</td><td>69.13</td><td>72.25</td><td>71.60</td><td>69.12</td><td>79.92</td><td>73.34</td><td>78.87</td><td>57.85</td></tr><tr><td>w/o LoRA</td><td>70.45</td><td>63.37</td><td>62.64</td><td>70.51</td><td>69.82</td><td>67.04</td><td>71.72</td><td>60.53</td><td>77.66</td><td>55.39</td></tr><tr><td>w/o  $O _ { T , f }$ </td><td>73.32</td><td>67.20</td><td>66.64</td><td>72.37</td><td>71.30</td><td>68.86</td><td>77.34</td><td>71.47</td><td>78.00</td><td>56.11</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { s m o o t h } }$ </td><td>72.66</td><td>66.44</td><td>65.72</td><td>71.94</td><td>71.00</td><td>69.12</td><td>76.61</td><td>69.95</td><td>77.31</td><td>54.76</td></tr><tr><td>Proposed+TTA[21]</td><td>74.94</td><td>68.24</td><td>69.17</td><td>72.33</td><td>71.65</td><td>69.15</td><td>79.89</td><td>73.32</td><td>79.03</td><td>58.16</td></tr></table>

Table 2: Test performance by tier. Macro-F1 and Cohen’s κ (higher is better) for the proposed model and ablations. AVG is the unweighted mean across tiers (CHN, FAN, MAN, CXN). “w/o LoRA” disables LoRA fine-tuning of the Whisper encoder; $" w / o \ o _ { \tau , f } \ "$ removes family-specific offsets; “w/o $\mathcal { L } _ { \mathrm { s m o o t h } }$ ” removes the temporal smoothing loss. Bold indicates the best score in each column and underlined values indicate the second-best. Gray values correspond to results reported under an easier evaluation setting and are therefore excluded when determining the best and second-best scores. Both Macro-F1 and Kappa are reported in percentage (%).

Overall comparison. Our proposed method achieves the best overall performance, reaching 74.88 Macro-F1 and 68.14 κ averaged across tiers. Under the original multi-tier evaluation, it outperforms TL-TR and W2V-LB on the across-tier average and on each tier. The consistent gains indicate that the architecture effectively captures both acoustic structure and speaker-specific characteristics while remaining robust to household variability.

Tier-wise analysis. The largest performance gaps between our method and W2V-LB occur in the adult tiers (FAN and MAN), whereas the difference on the child tier (CHN) is comparatively small under the same multi-tier evaluation setting. One possible explanation is pretraining bias: Whisper is trained on massive speech corpora dominated by adult speech, which may provide stronger representations for adult vocalizations, while its representations for infant and child speech from a close-mic remain relatively out-of-domain. In contrast, W2V-LB is pretrained on large-scale family recordings containing child vocalizations, which may explain its competitive performance on CHN despite its architectural limitations. This observation highlights the importance of both pretraining data distribution and model architecture when addressing domain-specific speech understanding tasks.

Effect of architectural components. The ablation results clarify the role of each component. Removing LoRA (w/o LoRA) leads to consistent degradation across tiers, confirming that parameter-efficient fine-tuning of the Whisper encoder substantially improves in-domain acoustic representations. Eliminating the family offset $( \mathrm { w } / 0 ~ o _ { \tau , f } )$ primarily affects tiers with higher cross-family variability, especially MAN, suggesting that the offset absorbs family-dependent variability during training and allows the shared tier tokens to learn more invariant representations. Removing temporal smoothing (w/o $\scriptstyle { \mathcal { L } } _ { \mathrm { s m o o t h } } )$ reduces temporal stability, with larger drops in CHN and MAN tiers. Overall, these findings indicate that performance gains arise not only from stronger pretrained features but also from architectural inductive biases tailored to multi-speaker naturalistic recordings, where separating shared representations from nuisance variability and enforcing temporal consistency are both critical for robust infant-centered audio understanding.

## 4. Test-Time Adaptation of the offset

Although the proposed framework demonstrates strong performance across families, it does not explicitly adapt the familyspecific offset parameters for unseen test households. In the current design, offsets are learned only for training families, and inference on new families relies solely on the shared tier tokens. While this strategy promotes generalization and avoids additional computation at test time, it may limit the model’s ability to fully capture household-specific acoustic characteristics such as recording conditions, speaker traits, or environmentdependent noise patterns of the test families.

We conducted preliminary experiments exploring unsupervised test-time adaptation (Proposed+TTA in Table 2) using a SUTA-style objective [21] to update offsets for unseen families. At test time, we adapt only the speaker-offset table $\{ O _ { \tau , f } \}$ for unseen families, while freezing all other parameters. Following SUTA [21], we employ a tier-wise weighted combination of entropy minimization (EM) and minimum class confusion (MCC) objectives for calibration:

$$
\mathcal { L } _ { \mathrm { c a l } } = \sum _ { \tau \in \mathcal { T } } \Bigl ( \eta \mathcal { L } _ { \mathrm { E M } } ^ { ( \tau ) } + \left( 1 - \eta \right) \mathcal { L } _ { \mathrm { M C C } } ^ { ( \tau ) } \Bigr ) , \quad \eta \in [ 0 , 1 ] .
$$

As in the original work, EM excludes frames dominated by the CTC blank class in ASR. Analogously in our setting, $\mathcal { L } _ { \mathrm { E M } } ^ { ( \tau ) }$ is computed only over frames whose most probable prediction corresponds to an active vocalization class. We apply the test time adaptation for 5 epochs with a mixing weight $\eta = 0 . 3$ However, the observed improvements were modest. These results suggest that, although adaptation has potential, more effective or stable approaches are needed to make it practically beneficial.

Developing robust unsupervised adaptation mechanisms for unseen families remains an important direction for future work. In particular, methods that can reliably personalize representations without supervision, while preserving stability and efficiency, could further improve performance in realistic deployment scenarios where domain shifts are inevitable.

## 5. Conclusion

We presented a compact framework for infant-centered multitier audio tagging that unifies diarization and vocalization classification. By combining a LoRA-finetuned Whisper encoder with structured speaker conditioning and tier-specific heads, the model supports overlapping speakers and framewise prediction. Experiments on a family-disjoint split show consistent empirical gains over baselines, suggesting the effectiveness of pretrained representations with task-specific inductive biases for naturalistic infant-centered recordings.

## 6. Generative AI Use Disclosure

We used Claude and GPT for language editing and manuscript polishing, including improving clarity of expression, correcting grammatical errors, and formatting L<sup>A</sup>T X tables. All technical content, experimental design, analysis, and scientific contributions are entirely the work of the authors. The authors have carefully reviewed the manuscript and take full responsibility for its content.

## 7. Acknowledgement

This study was supported by funding from the National Institute on Drug Abuse (R34DA050256; R01DA059422). For the experiments presented here, we used the Delta System at the National Center for Supercomputing Applications through AC-CESS allocations CIS240417 and CIS250040.

## 8. References

[1] M. Lavechin, R. Bousbib, H. Bredin, E. Dupoux, and A. Cristia, “An open-source voice type classifier for child-centered daylong recordings,” in Proc. Interspeech 2020, 2020, pp. 3072–3076.

[2] J. Li, M. Hasegawa-Johnson, and N. L. McElwain, “Analysis of acoustic and voice quality features for the classification of infant and mother vocalizations,” Speech communication, vol. 133, pp. 41–61, 2021.

[3] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0: A framework for self-supervised learning of speech representations,” Advances in neural information processing systems, vol. 33, pp. 12 449–12 460, 2020.

[4] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.

[5] Y. Gong, C.-I. Lai, Y.-A. Chung, and J. Glass, “Ssast: Selfsupervised audio spectrogram transformer,” in Proceedings ofthe AAAI Conference on Artificial Intelligence, vol. 36, no. 10, 2022, pp. 10 699–10 709.

[6] J. Li, M. Hasegawa-Johnson, and N. L. McElwain, “Towards robust family-infant audio analysis based on unsupervised pretraining of wav2vec 2.0 on large-scale unlabeled family audio,” in Proc. Interspeech 2023, 2023, pp. 1035–1039.

[7] X. Fan, J. Li, M. Hasegawa-Johnson, and N. L. McElwain, “Bandsplit self-supervised mamba for infant-centered audio analysis,” in Proc. Interspeech 2025, 2025, pp. 2795–2799.

[8] R. Lahiri, T. Feng, R. Hebbar, C. Lord, S. H. Kim, and S. Narayanan, “Robust self supervised speech embeddings for child-adult classification in interactions involving children with autism,” in Proc. Interspeech 2023, 2023, pp. 3557–3561.

[9] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[10] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li, N. Kanda, T. Yoshioka, X. Xiao et al., “Wavlm: Large-scale selfsupervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[11] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen et al., “Lora: Low-rank adaptation of large language models.” ICLR, vol. 1, no. 2, p. 3, 2022.

[12] Z. Song, J. Zhuo, Y. Yang, Z. Ma, S. Zhang, and X. Chen, “Lora-whisper: Parameter-efficient and extensible multilingual asr,” arXiv preprint arXiv:2406.06619, 2024.

[13] T. Xu, K. Huang, P. Guo, Y. Zhou, L. Huang, H. Xue, and L. Xie, “Towards rehearsal-free multilingual asr: A lora-based case study on whisper,” arXiv preprint arXiv:2408.10680, 2024.

[14] W. Liu, Y. Qin, Z. Peng, and T. Lee, “Sparsely shared lora on whisper for child speech recognition,” in ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 11 751–11 755.

[15] Y. Gong, S. Khurana, L. Karlinsky, and J. Glass, “Whisper-at: Noise-robust automatic speech recognizers are also strong audio event taggers,” in Proc. Interspeech 2023, 2023.

[16] A. Xu, K. Huang, T. Feng, L. Shen, H. Tager-Flusberg, and S. Narayanan, “Exploring speech foundation models for speaker diarization in child-adult dyadic interactions,” arXiv preprint arXiv:2406.07890, 2024.

[17] A. Xu, T. Feng, H. Tager-Flusberg, C. Lord, and S. Narayanan, “Data efficient child-adult speaker diarization with simulated conversations,” in ICASSP 2025-2025 IEEE International Con ference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025, pp. 1–5.

[18] Z. Ma, G. Yang, Y. Yang, Z. Gao, J. Wang, Z. Du, F. Yu, Q. Chen, S. Zheng, S. Zhang et al., “An embarrassingly simple approach for llm with strong asr capacity,” arXiv preprint arXiv:2402.08846, 2024.

[19] B. Islam, N. L. McElwain, J. Li, M. I. Davila, Y. Hu, K. Hu, J. M. Bodway, A. Dhekne, R. Roy Choudhury, and M. Hasegawa-Johnson, “Preliminary technical validation of LittleBeats™: A multimodal sensing platform to capture cardiac physiology, motion, and vocalizations,” Sensors, vol. 24, no. 3, p. 901, 2024.

[20] P. Boersma, “Praat: doing phonetics by computer [computer program],” http://www. praat. org/, 2011.

[21] G.-T. Lin, S.-W. Li, and H.-y. Lee, “Listen, adapt, better wer: Source-free single-utterance test-time adaptation for automatic speech recognition,” in Proc. Interspeech 2022, 2022, pp. 2198– 2202.