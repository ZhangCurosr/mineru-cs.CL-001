# Motor, Cognitive, or Corpus? What Survives Cross-Lingual Transfer in Speech-Based Parkinson's Disease Detection

Serli Kopar1,2, Sam Gijsen1,2,3, Abner Hernandez4, Paula Andrea Perez-Toro4, Kerstin Ritter1,2,3

1Hertie Institute for AI in Brain Health, University of Tübingen, Tübingen, Germany

2Tübingen AI Center, University of Tübingen, Tübingen, Germany

3Charité-Universitätsmedizin, Department of Psychiatry and Psychotherapy, Berlin, Germany

4Pattern Recognition Lab, Friedrich-Alexander-Universität Erlangen-Nürnberg, Erlangen, Germany

Abstract—Self-supervised learning (SSL) speech representations achieve strong performance for Parkinson's disease (PD) detection within individual corpora. However, it remains unclear whether these models capture disease-related characteristics or exploit dataset-specific confounds, particularly since most SSL backbones are pretrained exclusively on healthy speech. To investigate this question, we perform a layer-wise analysis of nine SSL speech backbones using a low-capacity logistic regression probe across three languages. We structure the evaluation as multiple scenarios that progressively introduce distribution shifts in participant identity, recording conditions, language, and pathology. Our results reveal two key findings. First, layer selection is highly corpus-dependent: the optimal representation layer is determined primarily by the source dataset rather than by the SSL architecture itself. Second, the transferred discriminative signal lacks pathological specificity: classifiers trained to detect PD assign similarly high probabilities to both PD and dementia speech in the target corpus. These results highlight critical limitations that must be addressed before speech-based pathology recognition models can be reliably deployed in clinical settings.

Index Terms—Parkinson's disease, self-supervised learning, multilingual speech representations, cross-lingual transfer, motor speech disorders.

## I. INTRODUCTION

Speech and language are increasingly used to characterise motor and cognitive changes in Parkinson's disease (PD). Alterations in phonation, articulation, fluency, and lexical access can reflect the combined impact of motor impairment and nonmotor symptoms, including cognitive decline [1]. Clinically, motor severity is assessed using instruments such as the MDS-UPDRS [2] and Hoehn & Yahr scale [3], based on speech tasks including sustained vowel phonation (VOWEL), diadochokinetic repetition (DDK), and passage reading (READ). Cognitive impairment is evaluated using neuropsychological batteries such as CERAD-NB+ [4] or screening tools such as the MMSE [5], together with language tasks including Verbal Fluency and the Boston Naming Test.

Automated speech analysis has emerged as a promising complement to these assessments [6]. Recent work increasingly relies on self-supervised learning (SSL) speech representations. Most studies either fine-tune the entire SSL model for PD classification [7] or freeze the pretrained encoder and train a lightweight classifier on the extracted embeddings [8]. While end-to-end fine-tuning often achieves strong withincorpus performance, the resulting models can exploit arbitrary label-correlated cues, making it difficult to determine whether performance reflects pathology-relevant information or adaptation to dataset-specific characteristics. In contrast, frozen representations with lightweight classifiers allow a direct assessment of what information is already encoded in pretrained speech models. Because current SSL models are trained almost exclusively on healthy speech, it remains unclear whether they capture PD-related speech characteristics without task-specific adaptation.

This question is particularly important because SSL speech representations encode many factors unrelated to disease, including corpus identity [9], speaker identity, and gender [10]. Existing work has primarily sought to suppress these factors through techniques such as domain-adversarial training [11]. Rather than attempting to remove presumed confounds, we ask a different question: Under clinically relevant distribution shifts, is the discriminative signal learned from frozen SSL representations specific to PD? Prior work typically evaluates PD speech classifiers only against healthy controls [12]-[16] leaving it unclear whether they capture PD-specific patterns or more general markers of neurodegeneration.

To answer this question, we introduce a five-scenario evaluation framework that begins with a no-shift reference condition (REF) using intra-corpus cross-validation, and progressively introduces increasingly challenging distribution shifts. Starting from REF, we evaluate take-re-take recordings within the same corpus (S1), recording-condition shifts (S2), language shifts (S3), task shifts while holding language constant (S4), and combined language-and-task shifts (S5). The probing layer is selected separately for each corpus, backbone, and task in the REF setting and then fixed throughout all transfer experiments. Across all conditions, we use frozen SSL speech encoders with a logistic regression probe to measure the linearly decodable information in the representations. We evaluate three PD speech corpora in Spanish (ES), German (DE), and Czech (CZ). We then transfer PD-trained classifiers to the independent, cognitively characterised German TREND cohort to compare predictions for participants with PD and dementia.

## II. METHODS

## A. Datasets

1) Special Session Corpora: We used three cross-sectional corpora from the session organisers [17]. Each corpus contains the same three tasks, recorded in one session per participant: i) DDK, ii) READ and iii) VOWEL. Characteristics of all three cohorts are summarised in Table I.

a) Czech (CZ): This corpus [18] contains 100 participants (50 PD, 50 HC), each completing two full repetitions of all three tasks within one session. We treat the two repetitions as separate samples for Scenario 1 (S1, +Re-Take). For the VOWEL task, we discard non-/a/ recordings to avoid contamination, leaving 89 participants (48 HC, 41 PD).

b) Spanish (ES, ES-e): This corpus [19] has two participant-disjoint subsets recorded under different acoustic conditions. The primary subset (ES) contains 100 participants (50 PD, 50 HC), each providing recordings for all three tasks. We apply the same VOWEL filtering as in CZ, retaining only participants with two /a/-vowel recordings; no participants were removed in this subset. The second subset (ES-e) contains 40 participants (20 PD, 20 HC) recorded in a more acoustically challenging setting, with only DDK and READ recordings; these were used for Scenario 2 (S2, +Condition).

c) German (DE): This corpus [20] consists of 176 participants (88 PD and 88 HC), each providing one recording for each of the three tasks. This corpus anchors the crosspathology transfer setting (S4, +Task), as it shares the same language as the TREND cohort.

2) In-house TREND corpus: To extend the diagnostic scope beyond PD, we utilise the TREND [21] dataset 1, including recordings from non-overlapping dementia and PD participants. HCs were selected from individuals who did not convert to PD, dementia, or prodromal mild cognitive impairment (MCI) during the study period. We matched HCs to clinical subjects using a sex-matched Hungarian algorithm [22], [23]; applying tiered tolerances for age and education. This resulted in the TREND corpus, comprising dementia (36 Dem, 36 HC) and PD (18 PD, 18 HC) groups, as summarised in Table II. Unlike the special-session corpora, the TREND protocol included five CERAD-NB+ tasks and one MMSE assessment. The relationship between acoustic features and cognitive performance across these tasks has previously been demonstrated by the authors [24], and we follow the same preprocessing protocol.

## B. Feature Extraction

To enable a rigorous comparison across feature representations, we extract utterance-level representations from handcrafted acoustic features and frozen SSL backbones spanning three axes of variation: (i) capacity, (ii) ASR fine-tuning, and (iii) pretraining language.

TABLE I: Cohort overview across special session corpora. PD: Parkinson's disease; HC: healthy controls.
<table><tr><td>Field</td><td>ES</td><td> $\mathbf { E S  – e }$ </td><td>DE</td><td>CZ</td></tr><tr><td>N (PD/HC)</td><td>50/50</td><td>20/20</td><td>88/88</td><td>50/50</td></tr><tr><td>Sex (M/F) PD Age</td><td>25/25</td><td>9/11</td><td>47/41</td><td>30/20</td></tr><tr><td></td><td> $6 1 . 1 \pm 9 . 6$ </td><td> $6 1 . 2 \pm 1 4 . 3$ </td><td> $6 6 . 5 \pm 9 . 0$ </td><td> $6 3 . 4 \pm 9 . 5$ </td></tr><tr><td>MDS-UPDRS-III†</td><td> $3 7 . 6 \pm 1 8 . 2$ </td><td> $4 0 . 7 \pm 2 2 . 2$ </td><td> $2 2 . 7 \pm 1 0 . 9$ </td><td> $2 0 . 1 \pm 1 0 . 9$ </td></tr><tr><td>Time post diag.</td><td> $1 0 . 7 \pm 9 . 2$ </td><td> $9 . 2 \pm 6 . 4$ </td><td> $0 . 6 \pm 0 . 5$ </td><td> $6 . 7 \pm 4 . 7$ </td></tr><tr><td>Sex (M/F)</td><td>25/25</td><td>11/9</td><td> $4 4 / 4 4$ </td><td>30/20</td></tr><tr><td>OH Age</td><td> $6 1 . 0 \pm 9 . 5$ </td><td> $6 2 . 6 \pm 1 0 . 0$ </td><td> $6 3 . 2 \pm 1 4 . 0$ </td><td>61.6±11.2</td></tr></table>

† For the Spanish corpus the newer UPDRS-III version is used (range 0–132) versus the older version (range 0–108).

TABLE II: TREND cohort overview across two diagnosed conditions with their matched HC groups.
<table><tr><td rowspan="2"></td><td colspan="2">Dementia</td><td colspan="2">PD</td></tr><tr><td>Diagnosed</td><td>HC</td><td>Diagnosed</td><td>HC</td></tr><tr><td>N</td><td>36</td><td>36</td><td>18</td><td>18</td></tr><tr><td>Sex (M/F)</td><td>23/13</td><td>23/13</td><td>16/2</td><td>16/2</td></tr><tr><td>Age</td><td> $8 1 . 4 \pm 7 . 1$ </td><td> $7 8 . 2 \pm 6 . 2$ </td><td> $7 6 . 9 \pm 5 . 5$ </td><td> $7 6 . 3 \pm 4 . 9$ </td></tr><tr><td>Age range</td><td>[66, 92]</td><td>[66, 89]</td><td>[68, 87]</td><td>[68, 84]</td></tr><tr><td>Education (y)</td><td> $1 5 . 2 \pm 2 . 9$ </td><td> $1 5 . 5 \pm 3 . 1$   $8 4 . 6 \pm 8 . 7$ </td><td> $1 5 . 6 \pm 2 . 4$   $7 4 . 9 \pm 1 0 . 0$ </td><td> $1 5 . 3 \pm 2 . 6$ </td></tr><tr><td>CERAD total score</td><td> $6 2 . 9 \pm 1 3 . 0$ </td><td></td><td></td><td> $8 4 . 6 \pm 8 . 5$ </td></tr><tr><td>MMSE</td><td> $2 4 . 9 \pm 2 . 8$ </td><td> $2 8 . 3 \pm 1 . 6$ </td><td> $2 7 . 2 \pm 1 . 5$ </td><td> $2 7 . 8 \pm 2 . 2$ </td></tr></table>

1. Handcrafted features: We extract the extended Geneva Minimalistic Acoustic Parameter Set (eGeMAPS) [25] using openSMILE [26], yielding 88-dimensional utterance-level representations. The feature set includes prosodic (e.g., pitch, loudness, speech rate), spectral/articulatory (e.g., formants, MFCCs, spectral slope), and voice-quality (e.g., jitter, shimmer, harmonics-to-noise ratio) descriptors, providing theorydriven and interpretable features.

2. Capacity (Base vs. Large): We use Base (HuBERT-B, WavLM-B) and Large (HuBERT-L, WavLM-L) variants of HuBERT [27] and WavLM [28].

3. Fine-tuning (pretrained vs. ASR-fine-tuned): We include pretrained W2V2-B [29] and HuBERT-L alongside their ASRfine-tuned counterparts (W2V2-B-FT and HuBERT-L-FT).

4. Pretraining language (monolingual vs. multilingual): We include the English-only pretrained model W2V2-B alongside the multilingual pretrained models XLS-R [30] (128 languages) and MMS [31] (1,406 languages).

For each model, we extract frame-level representations from every transformer layer (including CNN output as layer 0) and obtain utterance-level embeddings by temporal mean pooling. We evaluate each corpus-task-backbone-layer combination using repeated 5-fold cross-validation with 10 random seeds. Within each fold, embeddings are standardised using the training split before fitting a logistic regression classifier [32]. The layer with the highest mean balanced accuracy (BA) is selected in REF and used for all subsequent scenarios.

![](images/544fea9da3db61f65a446d96c7544822201b4c0ae77b0854deef9f1af72e144e.jpg)  
Fig. 1: Five scenarios for evaluating cross-lingual transfer in speech-based PD detection. REF uses 5-fold intra-corpus cross-validation to select the SSL backbone layer, which is then frozen while the binary PD classifier is trained on the full training corpus and evaluated on the target corpus without adaptation. Scenarios (S1-S5) introduce progressively larger train-test distribution shifts. The Train and Test columns show one representative train→test example per scenario. The Changes column summarises the factors that differ from REF. Abbreviations: Rec. = recording condition; Lang. = language; Path. = pathology; DE = German corpus; ES = Spanish corpus; ES-e = Spanish corpus under noisy recording conditions; CZ = Czech corpus; TREND = in-house German cohort including PD and dementia patients.

## C. Five Scenarios (S1–S5)

To assess robustness in cross-lingual, cross-corpus PD detection, we define five scenarios that progressively introduce additional sources of distribution shift, shown in Figure 1. All scenarios are anchored to within-corpus reference setting (REF, intra-corpus CV). For each combination, the same layer selected under REF is used across all scenarios. This ensures that performance differences reflect distribution shift rather than changes in layer selection or representation choice.

• S1 (+Re-Take) isolates the mildest distribution shift. It reuses the same speaker-disjoint folds as REF, replacing only the test recordings with a second take while keeping the corpus, task, language, and recording condition fixed. The binary PD classification performance change $( B A _ { \mathrm { S } 1 } - B A _ { \mathrm { R E F } } )$ therefore reflects only repeated acquisition from the same speakers within a session.

• S2 (+Condition) adds recording-condition shift. We use the Spanish ES (clean) and ES-e (noisy) corpora, which contain non-overlapping participants performing the same READ and DDK speech tasks. Robustness is evaluated through bidirectional cross-condition transfer (train on ES, test on ES-e, and vice versa), quantifying sensitivity to recording conditions.

• S3 (+Language) introduces cross-lingual, cross-corpus transfer. Models are trained on a single source corpus and evaluated independently on two target corpora (e.g., DE → ES and $D E  C Z )$ . This scenario quantifies generalisation across corpora.

• S4 (+Task) holds language fixed while introducing task shift. Models are transferred by inference only to the TREND corpus, which contains unseen clinical protocols (CERAD NB+ and MMSE). Beyond robustness to a new speech task, this scenario tests whether classifier outputs capture PD-specific signal.

• S5 (+Task +Language) combines the shifts introduced in S3 and S4. Models are trained on ES or CZ and transferred to the TREND corpus, introducing both language and task shift simultaneously. This represents the most challenging, deployment-oriented evaluation setting.

## III. RESULTS

## A. Layer Selection

Optimal layer selection is corpus-dependent and least stable for large backbones. Table III reports each selected layer across 10 seeds and 5-fold cross-validation, together with the standard deviation (σ) of normalised optimal-layer depth. Optimal layer choice varies substantially across corpora, especially for larger backbones. On DDK, WavLM-L selects layer 24 for DE, layer 1 for ES, and layer 22 for CZ (σ = 0.43), with a similar pattern across tasks (e.g., HuBERT-L reaches σ = 0.37 on READ and σ = 0.36 on VOWEL).

Figure 2 shows BA across layers for READ. Layer-wise performance appears to vary by corpus rather than following a uniform trend. For DE, BA tends to increase with depth before plateauing. For ES, it remains relatively stable across layers. For CZ, performance appears to follow a weak unimodal pattern, with mid-depth layers performing slightly better than the deepest ones. These trends are broadly consistent across backbone variance axis (capacity, fine-tuning and pretraining), suggesting that layer-wise behaviour is driven more by corpus than by architecture.

## B. Scenario 1 (+Re-Take)

Performance across recording takes depends on the interplay between speech task and backbone. Figure 3 reports the change in binary PD classification balanced accuracy relative to REF $\Delta \mathrm { B A } = \mathrm { B A } _ { \mathrm { S 1 } } - \mathrm { B A } _ { \mathrm { R E F } }$ , where only the test recordings are replaced by each participant's second take. Thus, values close to zero indicate robustness to repeated recordings.

TABLE III: Argmax layer per backbone. Optimal layer (argmax of mean balanced accuracy across layers; 10 seeds × 5-fold CV) per backbone, task, and corpus. σ is the standard deviation of normalised layer depth (layer ÷ total number of layers) across the three corpora (DE, ES, CZ) at fixed backbone and task.
<table><tr><td rowspan="2"></td><td colspan="4">DDK</td><td colspan="4">READ</td><td colspan="4">VOWEL</td></tr><tr><td>DE</td><td>ES</td><td>CZ</td><td>σ</td><td>DE</td><td>ES</td><td>CZ</td><td>σ</td><td>DE</td><td>ES</td><td>CZ</td><td>σ</td></tr><tr><td>W2V2-B</td><td>5</td><td>0</td><td>5</td><td>0.20</td><td>4</td><td>6</td><td>5</td><td>0.07</td><td>0</td><td>1</td><td>3</td><td>0.10</td></tr><tr><td>W2V2-FT</td><td>7</td><td>0</td><td>5</td><td>0.25</td><td>6</td><td>4</td><td>8</td><td>0.14</td><td>0</td><td>1</td><td>9</td><td>0.34</td></tr><tr><td>HuB-B</td><td>9</td><td>0</td><td>7</td><td>0.32</td><td>6</td><td>10</td><td>7</td><td>0.14</td><td>3</td><td>4</td><td>4</td><td>0.04</td></tr><tr><td>HuB-L</td><td>24</td><td>1</td><td>15</td><td>0.39</td><td>24</td><td>3</td><td>18</td><td>0.37</td><td>3</td><td>2</td><td>21</td><td>0.36</td></tr><tr><td>HuB-L-FT</td><td>15</td><td>1</td><td>11</td><td>0.25</td><td>18</td><td>8</td><td>13</td><td>0.17</td><td>0</td><td>17</td><td>7</td><td>0.29</td></tr><tr><td>WavLM-B</td><td>7</td><td>0</td><td>7</td><td>0.27</td><td>9</td><td>5</td><td>6</td><td>0.14</td><td>5</td><td>5</td><td>2</td><td>0.12</td></tr><tr><td>WavLM-L</td><td>24</td><td>1</td><td>22</td><td>0.43</td><td>21</td><td>2</td><td>9</td><td>0.33</td><td>13</td><td>9</td><td>16</td><td>0.12</td></tr><tr><td>XLS-R</td><td>16</td><td>10</td><td>16</td><td>0.12</td><td>20</td><td>3</td><td>10</td><td>0.29</td><td>22</td><td>8</td><td>5</td><td>0.31</td></tr><tr><td>MMS</td><td>9</td><td>13</td><td>10</td><td>0.07</td><td>23</td><td>11</td><td>6</td><td>0.30</td><td>14</td><td>20</td><td>12</td><td>0.14</td></tr></table>

Highlighting:group max (per backbone family) andcolumn max

![](images/dcf71fb907165cb152ae422eae8023acbb852192b27e171944b5b7691ed4104c.jpg)  
Fig. 2: Layer selection results for the READ task in the REF setting, where the selected backbone layer is fixed for all downstream transfer experiments. The three plots compare: i) Capacity: HuBERT-B vs. HuBERT-L and WavLM-B vs. WavLM-L; ii) Fine-tuning: W2V2-B vs. W2V2-B-FT and HuBERT-L vs. HuBERT-L-FT; iii) Pretraining: W2V2-B vs. XLS-R and MMS. The READ task is shown because it achieved the highest overall balanced accuracy (BA).

Overall, mean performance change is small $( - 1 . 9 \pm 4 . 5$ BA points, bottom right). READ shows the smallest mean performance change (0.1 ±2.0), followed by DDK (0.3± 4.8). DDK is also the only task where both multilingual backbones improve consistently, with XLS-R gaining 7.0 ±3.9 BA points and MMS 6.0±4.1. VOWEL exhibits the greatest sensitivity to recording re-takes. Across all models, the average performance decreases by 3.8±5.5 BA points on ES and 4.2±3.3 BA points on CZ. However, the effect varies considerably across SSL backbones and corpora. For instance, HuBERT-B improves on ES VOWEL by 7.0 ± 5.2 BA points, while its performance drops by 6.1 ± 4.2 BA points on CZ. The largest degradations are observed for the ASR-fine-tuned models: HuBERT-L-FT decreases by 15.0 BA points on ES, and W2V2-B-FT decreases by 9.9 BA points on CZ. Overall, these results show that the impact of recording re-takes is strongly dependent on both the backbone and the corpus, with VOWEL exhibiting the highest variability.

<table><tr><td></td><td rowspan=1 colspan=1>ES</td><td rowspan=1 colspan=2>CZ    CZ</td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>eGeMAPS</td><td rowspan=1 colspan=1>1.0 ±1.0</td><td rowspan=1 colspan=1>-1.1 ±5.1</td><td rowspan=1 colspan=1>-8.0 ±5.0</td><td rowspan=1 colspan=1>4.0 ±4.0</td><td rowspan=1 colspan=1>-1.0 ±5.1</td></tr><tr><td rowspan=1 colspan=1>W2V2-B</td><td rowspan=1 colspan=1>-5.0 ±4.3</td><td rowspan=1 colspan=1>-0.4 ±5.1</td><td rowspan=1 colspan=1>3.0 ±4.2</td><td rowspan=1 colspan=1>0.0 ±3.1</td><td rowspan=1 colspan=1>-0.6 ±3.3</td></tr><tr><td rowspan=1 colspan=1>W2V2-B-FT</td><td rowspan=1 colspan=1>-5.0 ±4.3</td><td rowspan=1 colspan=1>-9.9 ±6.8</td><td rowspan=1 colspan=1>-4.0 ±5.0</td><td rowspan=1 colspan=1>-3.0 ±3.2</td><td rowspan=1 colspan=1>-5.5 ±3.1</td></tr><tr><td rowspan=1 colspan=1>HuBERT-B</td><td rowspan=1 colspan=1>7.0 ±5.2</td><td rowspan=1 colspan=1>-6.1 ±4.2</td><td rowspan=1 colspan=1>-4.0 ±4.3</td><td rowspan=1 colspan=1>2.0 ±2.7</td><td rowspan=1 colspan=1>-0.3 ±5.9</td></tr><tr><td rowspan=1 colspan=1>HuBERT-L</td><td rowspan=1 colspan=1>-5.0 ±4.3</td><td rowspan=1 colspan=1>-1.1 ±5.3</td><td rowspan=1 colspan=1>1.0 ±4.7</td><td rowspan=1 colspan=1>1.0 ±3.3</td><td rowspan=1 colspan=1>-1.0 ±2.8</td></tr><tr><td rowspan=1 colspan=1>HuBERT-L-FT</td><td rowspan=1 colspan=1>-15.0±5.7</td><td rowspan=1 colspan=1>-9.0 ±4.2</td><td rowspan=1 colspan=1>3.0 ±3.3</td><td rowspan=1 colspan=1>-1.0 ±3.7</td><td rowspan=1 colspan=1>-5.5 ±8.1</td></tr><tr><td rowspan=1 colspan=1>WavLM-B</td><td rowspan=1 colspan=1>-5.0 ±2.8</td><td rowspan=1 colspan=1>-4.3 ±4.3</td><td rowspan=1 colspan=1>-2.0 ±4.2</td><td rowspan=1 colspan=1>0.0 ±3.4</td><td rowspan=1 colspan=1>-2.8 ±2.3</td></tr><tr><td rowspan=1 colspan=1>WavLM-L</td><td rowspan=1 colspan=1>-3.0 ±4.0</td><td rowspan=1 colspan=1>-1.5 ±5.7</td><td rowspan=1 colspan=1>1.0 ±4.4</td><td rowspan=1 colspan=1>1.0 ±2.2</td><td rowspan=1 colspan=1>-0.6 ±2.0</td></tr><tr><td rowspan=1 colspan=1>XLS-R</td><td rowspan=1 colspan=1>-3.0 ±5.0</td><td rowspan=1 colspan=1>-4.5 ±4.9</td><td rowspan=1 colspan=1>7.0 ±3.9</td><td rowspan=1 colspan=1>-2.0 ±2.4</td><td rowspan=1 colspan=1>-0.6 ±5.2</td></tr><tr><td rowspan=1 colspan=1>MMS</td><td rowspan=1 colspan=1>-5.0 ±3.9</td><td rowspan=1 colspan=1>-4.3 ±4.9</td><td rowspan=1 colspan=1>6.0 ±4.1</td><td rowspan=1 colspan=1>-1.0 ±2.2</td><td rowspan=1 colspan=1>-1.1 ±5.0</td></tr><tr><td rowspan=1 colspan=1>Mean↓</td><td rowspan=1 colspan=1>-3.8 ±5.5</td><td rowspan=1 colspan=1>-4.2 ±3.3</td><td rowspan=1 colspan=1>0.3 ±4.8</td><td rowspan=1 colspan=1>0.1 ±2.0</td><td rowspan=1 colspan=1>-1.9 ±4.5</td></tr></table>

Fig. 3: S1 (+Re-Take) Results: Change in balanced accuracy relative to REF $( \Delta \mathrm { B A } = \mathrm { B A } _ { \mathrm { S 1 } } - \mathrm { B A } _ { \mathrm { R E F } } )$ . S1 differs from REF only in the test recordings, which are replaced by each participant's second recording; training data and participant splits are unchanged. Values close to zero indicate robust performance across recording takes. Results are reported as mean ± bootstrapped SD $( n ~ = ~ 2 0 0 0$ test-set resamples). Columns correspond to speech task and corpus. The rightmost column (Mean) reports the average ∆BA for each backbone across all tasks and corpora, while the bottom row reports the average ∆BA for each corpus-task combination across backbones. Cell colour encodes the sign and magnitude of ∆BA.

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=5>DDK           ReadC→N   N→C   C→N   N→C  Mean→</td></tr><tr><td rowspan=1 colspan=1>eGeMAPS</td><td rowspan=1 colspan=1>-59.0±7.9 -</td><td rowspan=1 colspan=1>49.0±4.5</td><td rowspan=1 colspan=1>-20.5±9.1</td><td rowspan=1 colspan=1>-1.0 ±5.1</td><td rowspan=1 colspan=1>-32.4±26.5</td></tr><tr><td rowspan=1 colspan=1>W2V2-B-</td><td rowspan=1 colspan=1>-7.5 ±8.1</td><td rowspan=1 colspan=1>4.0 ±4.9</td><td rowspan=1 colspan=1>-10.5±6.6</td><td rowspan=1 colspan=1>-3.0 ±4.7</td><td rowspan=1 colspan=1>-4.2 ±6.3</td></tr><tr><td rowspan=1 colspan=1>W2V2-B-FT-</td><td rowspan=1 colspan=1>-6.0 ±8.1</td><td rowspan=1 colspan=1>-6.0 ±4.6</td><td rowspan=1 colspan=1>0.5 ±7.5 -</td><td rowspan=1 colspan=1>1.0 ±4.7</td><td rowspan=1 colspan=1>-3.1 ±3.4</td></tr><tr><td rowspan=1 colspan=1>HuBERT-B</td><td rowspan=1 colspan=1>-20.0±6.3</td><td rowspan=1 colspan=1>-6.0 ±4.9</td><td rowspan=1 colspan=1>-31.0±5.3</td><td rowspan=1 colspan=1>-6.0 ±4.2</td><td rowspan=1 colspan=1>-15.8±12.1</td></tr><tr><td rowspan=1 colspan=1>HuBERT-L---</td><td rowspan=1 colspan=1>13.5±7.2</td><td rowspan=1 colspan=1>4.0 ±4.0</td><td rowspan=1 colspan=1>-21.0±6.5</td><td rowspan=1 colspan=1>-11.0±5.7</td><td rowspan=1 colspan=1>-10.4±10.5</td></tr><tr><td rowspan=1 colspan=1>HuBERT-L-FT</td><td rowspan=1 colspan=1>-21.0±6.7</td><td rowspan=1 colspan=1>-4.0 ±4.7</td><td rowspan=1 colspan=1>-22.0±6.5</td><td rowspan=1 colspan=1>-7.0 ±4.7</td><td rowspan=1 colspan=1>-13.5±9.3</td></tr><tr><td rowspan=1 colspan=1>WavLM-B</td><td rowspan=1 colspan=1>-45.5±7.9</td><td rowspan=1 colspan=1>-20.0±3.9</td><td rowspan=1 colspan=1>-20.5±6.8 -</td><td rowspan=1 colspan=1>16.0±4.7</td><td rowspan=1 colspan=1>-25.5±13.5</td></tr><tr><td rowspan=1 colspan=1>WavLM-L</td><td rowspan=1 colspan=1>-21.0±8.1</td><td rowspan=1 colspan=1>-2.0 ±4.5</td><td rowspan=1 colspan=1>-10.0±8.0</td><td rowspan=1 colspan=1>-7.0 ±5.0</td><td rowspan=1 colspan=1>-10.0±8.0</td></tr><tr><td rowspan=1 colspan=1>XLS-R-</td><td rowspan=1 colspan=1>-2.5 ±7.0</td><td rowspan=1 colspan=1>1.0 ±4.0</td><td rowspan=1 colspan=1>-24.5±6.8</td><td rowspan=1 colspan=1>-5.0 ±4.7</td><td rowspan=1 colspan=1>-7.7 ±11.4</td></tr><tr><td rowspan=1 colspan=1>MMS-</td><td rowspan=1 colspan=1>5.0 ±6.6 -</td><td rowspan=1 colspan=1>3.0 ±4.1</td><td rowspan=1 colspan=1>-3.5 ±6.7</td><td rowspan=1 colspan=1>-9.0 ±4.4</td><td rowspan=1 colspan=1>-2.6 ±5.8</td></tr><tr><td rowspan=1 colspan=1>Mean↓</td><td rowspan=1 colspan=1>-19.1±19.7</td><td rowspan=1 colspan=1>-8.1 ±15.9</td><td rowspan=1 colspan=1>-16.3±10.0</td><td rowspan=1 colspan=1>-6.6 ±4.6</td><td rowspan=1 colspan=1>-12.5±14.3</td></tr></table>

![](images/d62ae5bfc279d9c38aaeee77c6bd1d665955540e672e71f06e6bb55c69471177.jpg)  
Fig. 4: S2 (+Condition) Results: Change in balanced accuracy relative to REF $( \Delta \mathrm { B A } = \mathrm { B A } _ { \mathrm { S 2 } } - \mathrm { B A } _ { \mathrm { R E F } } )$ under recordingcondition shift between clean (C) ES and noisy (N) ES-e corpora with non-overlapping speakers. C→N denotes training on clean and testing on noisy data; N→C denotes the reverse. Results are reported as mean ± bootstrapped SD $( n = 2 0 0 0$ resamples). Columns correspond to speech task and corpus. The rightmost column (Mean) averages ∆BA per backbone across all tasks and corpora, while the bottom row averages across backbones per corpus-task pair. Cell colour encodes the sign and magnitude of ∆BA.

## C. Scenario 2 (+Condition)

Cross-condition transfer is asymmetric: training on noisy recordings transfers better to clean recordings than vice versa. Building on S1 (+Re-Take), S2 (+Condition) introduces a recording-condition shift by training and testing across the clean (ES) and noisy (ES-e) variants of the same Spanish corpus with non-overlapping speakers, while keeping language and task fixed. Figure 4 shows ∆BA (S2 - REF) with bootstrapped standard deviations for both transfer directions (Clean→Noisy and Noisy→Clean). Unlike the modest overall degradation in S1 $( - 1 . 9 \pm 4 . 5 )$ , recording-condition mismatch causes a substantially larger performance drop, with an overall mean of $- 1 2 . 5 \pm 1 4 . 3$ (bottom right).

Two patterns emerge. First, transfer is consistently asymmetric: training on noisy data generalises better to clean recordings than the reverse. Across backbones, mean ∆BA improves from $- 1 9 . 1 \pm 1 9 . 7 \mathrm { { \ t o \ - 8 . 1 } \pm 1 5 . 9 }$ on DDK and from $- 1 6 . 3 \pm 1 0 . 0 \mathrm { { \ t o \ } } - 6 . 6 \pm 4 . 6$ on READ. The largest improvements are observed for base models, WavLM-B on DDK $( - 4 5 . 5 \to - 2 0 . 0 )$ and HuBERT-B on READ $( - 3 1 . 0 $ $- 6 . 0 )$ . MMS and W2V2-B-FT are the most condition-robust overall (row means $- 2 . 6 \pm 5 . 8$ and $- 3 . 1 \pm 3 . 4 .$ , respectively).

![](images/96b6354313f275e7852598b5ae3ff55749f530ddc4d682eae9109c0577dd5b3a.jpg)

![](images/afa5914ba5bc76ea919e9c91aa92186aa72b71160f2298aa4179db726528b7aa.jpg)

![](images/9fdc8032c2df8bbad95053e65602a2e496badd3f3ef4fdb55d29d88e97583f5f.jpg)  
Fig. 5: S3 (+Language) Results: Seven monolingual pretrained SSL backbones (W2V2-B, W2V2-B-FT, HuBERT-B, HuBERT-L, HuBERT-L-FT, WavLM-B, WavLM-L), two multilingual SSL backbones (XLS-R, MMS), and handcrafted eGeMAPS features. Horizontal lines indicate group-wise mean values.

Second, the handcrafted eGeMAPS baseline is the most condition-sensitive $( - 3 2 . 4 \pm 2 6 . 5 )$ , with the largest individual degradations (—59.0 and -49.0 on DDK), highlighting the greater overall mean robustness of SSL representations to recording-condition mismatch.

## D. Scenario 3 (+Language)

Backbone performance depends on the recorded task. S3 (+Language) introduces a cross-lingual evaluation setting in which each backbone is trained on one corpus and tested individually on the other two, yielding six balanced accuracy (BA) values per backbone and task. Figure 5 reports binary PD classification performance for three groups: monolingual (Mo) SSL models, multilingual (Mu) SSL models, and handcrafted eGeMAPS features (eG). The black horizontal lines indicate the mean BA within each group.

Overall, eG shows lower mean performance across tasks, suggesting weaker cross-corpus transfer. Although this setup directly evaluates cross-lingual generalisation, no consistent advantage of multilingual (Mu) over monolingual (Mo) SSL backbones is observed. Consistent with earlier findings, the recording task has a pronounced effect across SSL backbones: WavLM-L, in particular, exhibits non-overlapping performance values for VOWEL versus READ. A similar pattern observed for XLS-R when comparing its VOWEL and DDK performance. Taken together, these results indicate that task and backbone choice have a stronger influence on performance than the monolingual versus multilingual distinction alone.

Across scenarios, performance relative to REF shows a progressive decline, with mean ∆BA of $- 1 . 9 \pm 4 . 5$ (S1-REF), −12.5±14.3 (S2-REF), and −16.3±10.6 (S3-REF), indicating increasing degradation under more challenging conditions.

## E. Scenarios 4 (+Task) & 5 (+Task + Language)

Cross-task transfer to TREND detects disorder, not PD specifically. Out of 540 transfer combinations (10 backbones × 6 TREND tasks × 3 PD tasks × 3 training corpora), only 9 (6 trained on DE, 2 on ES, and 1 on $\mathbf { C Z } )$ achieved at least moderate separation between PD and HC, defined as $A U C ~ > ~ 0 . 6$ and $B A ~ > ~ 0 . 6$ We therefore restrict the specificity analysis to these 9 cases, since evaluating PDspecificity is only meaningful when there is some evidence of baseline PD classification performance. Importantly, the nonspecificity pattern described below is consistent across all 9 remaining combinations.

Figure 6 shows the predicted PD probability distributions across the four TREND subgroups for the highest BA classifier trained on each corpus (DE, ES, CZ). BA is computed for the PD and HC(PD) comparison. All three classifiers achieve only modest separation between PD and matched controls (DE = 0.67, ES = 0.66, CZ = 0.64). However, performance does not generalise to specificity in the PD vs. dementia comparison. This is most evident in the German transfer setting (S4 +Task), where the classifier significantly separates PD from HC(PD) and dementia from HC(Dem) $( p < 0 . 0 1 )$ ， but fails to distinguish PD from dementia according to both permutation testing [33] and the Mann-Whitney U test [34]. Overall, these results suggest a disorder-general signal rather than PD-specific discrimination.

CZ: HuBERT-B, DDK→Word List Recognition  
![](images/4f98136ce70c41f8befbab62956226f86e14d4fc6dc1a8364e2275bda94d1e18.jpg)

![](images/28a6245ab66f3659a7543535a0b2e3a44158fa55f1715f6d266ebd12c5b6ebce.jpg)

![](images/34affbe290a6646d48b427117e57a8dc57e839b040b9a134c05f79921ab76ceb.jpg)  
Fig. 6: S4 (+Task) and S5 (+Task + Language) Results: Predicted PD probability distributions for the highest-balancedaccuracy (BA) transfer combination of each training corpus. Layers were selected using only within-corpus REF data; no TREND data was used during layer selection. Each panel shows p(PD) across the four TREND cohort groups. Box colours correspond to the respective SSL backbone. Significance brackets indicate Mann-Whitney U and permutation test results $( n = 1 0 0 0 )$ ; significance levels are denoted by \*\*\*p < 0.001, \*\*p < 0.01, and $^ { * } p < 0 . 0 5$ . Abbreviations: PD = Parkinson's disease; HC(PD) = matched healthy control; Dem = dementia; HC(Dem) = matched healthy control.

Additionally, we trained a binary logistic regression classifier from scratch to distinguish the PD from dementia, only using the demographics features as input. Age, sex, and education alone already discriminate PD from dementia with AUC values of 0.73–0.75. We than adjusted the PD probability scores for these covariates using leave-one-subject-out (LOSO) regression. The residualised performance drops to chance level $( \mathrm { A U C } = 0 . 5 0 – 0 . 5 5 )$ , and all confidence intervals include 0.5, supporting previous claim of non-specificity.

## IV. DISCUSSION AND LIMITATIONS

In this study, we show that speech-based PD detection is driven more by corpus-specific factors than by PD-related motor or cognitive pathology. Under cross-lingual and crosstask transfer, models largely preserve a generic patient-healthy separation rather than disease-specific structure. This is further supported by the fact that the selected backbone layer varies substantially across corpora, especially for larger backbones.

We observe a consistent degradation in binary PD classification under increasingly severe distribution shifts. Identitypreserving re-takes. S1 (+Re-Take) have a small degradation of performance (—1.9 BA), while recording-condition (S2) and cross-corpus (S3) transfers lead to a larger drops (—12.5 and -16.3 BA). Cross-lingual shifts are challenging, even multilingual pretraining not showing a clear advantage. If the classifier captured robust motor impairment, greater stability across re-takes and partial cross-lingual robustness would be expected. Instead, the sensitivity across settings suggests that learned representations are not purely motoric. These findings align with prior work showing that language and corpus factors confound pathology classification [17] and that SSL embeddings tend to cluster by dataset rather than disease [35].

A key clinical implication emerges in transfer to the TREND dataset. Although transferred models moderately separate PD from healthy controls, they fail to distinguish PD from dementia. This is particularly concerning because most speechbased PD studies evaluate performance only against healthy controls, leaving diagnostic specificity to other neurological conditions largely untested. To our knowledge, this is the first systematic evaluation of cross-lingual PD classifiers in a dementia setting. While our results suggest that strong intra-corpus performance may not generalise under clinically meaningful shifts, conclusions are limited by our modest cohort sizes. We therefore interpret our findings as absence of evidence for PD-specificity, not evidence of its absence.

Overall, we view this work as a proof of concept for crossdisease transfer evaluation. It addresses the central question posed in our title, showing that what survives across transfer settings is primarily driven by the corpus, reflecting a general divergence from healthy speech, rather than a robust speech signature specific to Parkinson's disease.

## REFERENCES

[1] B. R. Bloem, M. S. Okun, and C. Klein, "Parkinson's Disease," The Lancet, vol. 397, no. 10291, pp. 2284–2303, 2021.

[2] C. G. Goetz, B. C. Tilley, S. R. Shaftman et al., "Movement Disorder Society-Sponsored Revision of the Unified Parkinson's Disease Rating Scale (MDS-UPDRS): Scale Presentation and Clinimetric Testing Results," Movement Disorders, vol. 23, no. 15, pp. 2129–2170, 2008.

[3] M. M. Hoehn and M. D. Yahr, "Parkinsonism: Onset, Progression and Mortality," Neurology, vol. 17, no. 5, pp. 427–442, 1967.

[4] J. C. Morris, A. Heyman, R. C. Mohs, J. P. Hughes, G. van Belle, G. Fillenbaum, E. D. Mellits, and C. Clark, “The Consortium to Establish a Registry for Alzheimer's Disease (CERAD). Part I. Clinical and Neuropsychological Assessment of Alzheimer's Disease," Neurology, vol. 39, no. 9, pp. 1159–1165, 1989.

[5] M. F. Folstein, S. E. Folstein, and P. R. McHugh, “"Mini-Mental State": A Practical Method for Grading the Cognitive State of Patients for the Clinician," Journal of Psychiatric Research, vol. 12, no. 3, pp. 189–198, 1975.

[6] D. Kovac, J. Mekyska, V. Aharonson, P. Harar, Z. Galaz, S. Rapcsak, J. R. Orozco-Arroyave, L. Brabenec, and I. Rektorova, “Exploring digital speech biomarkers of hypokinetic dysarthria in a multilingual cohort," Biomedical Signal Processing and Control, vol. 88, p. 105667, 2024. [Online]. Available: https://www.sciencedirect.com/ science/article/pii/S174680942301100X

[7] H. Sedigh Malekroodi, N. Madusanka, B.-i. Lee, and M. Yi, "Speech-Based Parkinson's Detection Using Pre-Trained Self-Supervised Automatic Speech Recognition (ASR) Models and Supervised Contrastive Learning," Bioengineering, vol. 12, no. 7, 2025. [Online]. Available: https://www.mdpi.com/2306-5354/12/7/728

[8] O. Klempí and R. Krupička, “Analyzing Wav2Vec 1.0 Embeddings for Cross-Database Parkinson's Disease Detection and Speech Features Extraction," Sensors, vol. 24, no. 17, p. 5520, Aug. 2024.

[9] E. J. Ibarra, J. D. Arias-Londoño, M. Zañartu, and J. I. Godino-Llorente, “Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation, journal = Bioengineering," vol. 10, no. 11, 2023. [Online]. Available: https://www.mdpi.com/2306-5354/10/11/1316

[10] A. Y. F. Chiu, K. C. Fung, R. T. Y. Li, J. Li, and T. Lee, “A Large-Scale Probing Analysis of Speaker-Specific Attributes in Self-Supervised Speech Representations," 2026. [Online]. Available: https://arxiv.org/abs/2501.05310

[11] M. Siniukov, E. Xing, S. A. Isfahani, and M. Soleymani, "Towards a Generalizable Speech Marker for Parkinson's Disease Diagnosis," 2025. [Online]. Available: https://arxiv.org/abs/2501.03581

[12] K. Yokoi, Y. Iribe, N. Kitaoka, T. Tsuboi, K. Hiraga, Y. Satake, M. Hattori, Y. Tanaka, M. Sato, A. Hori, and M. Katsuno, "Analysis of spontaneous speech in Parkinson's disease by natural language processing," Parkinsonism & Related Disorders, vol. 113, p. 105411, Aug 2023.

[13] S. Dhanalakshmi, S. Das, and R. Senthil, "Speech features-based Parkinson's disease classification using combined SMOTE-ENN and binary machine learning," Health Technology, vol. 14, no. 2, pp. 393–406, Mar 2024.

[14] M. L. Quatra, J. R. Orozco-Arroyave, and S. M. Siniscalchi, "Bilingual Dual-Head Deep Model for Parkinson's Disease Detection from Speech," ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1-5, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID: 276961433

[15] M. A. Hossain and F. Amenta, "Machine Learning-Based Classification of Parkinson's Disease Patients Using Speech Biomarkers," Journal of Parkinson's Disease, vol. 14, pp. 95 – 109, 2023. [Online]. Available: https://api.semanticscholar.org/CorpusID:266681923

[16] K. M. Alhawiti, "Multi-Modal Decentralized Hybrid Learning for Early Parkinson's Detection Using Voice Biomarkers and Contrastive Speech Embeddings," Sensors (Basel, Switzerland), vol. 25, 2025. [Online]. Available: https://api.semanticscholar.org/CorpusID:283066469

[17] A. Hernandez, E. Yeo, K. Choi, C.-J. Li, Z. Yue, R. K. Das, J. Rusz, M. M. Doss, J. R. Orozco-Arroyave, T. Arias-Vergara, A. Maier, E. Nöth, D. R. Mortensen, D. Harwath, and P. A. Perez-Toro, "Adapting Self-Supervised Speech Representations for Cross-lingual Dysarthria

Detection in Parkinson's Disease," arXiv preprint arXiv:2603.22225, 2026.

[18] C. D. Rios-Urrego, J. Rusz, and J. R. Orozco-Arroyave, “Automatic speech-based assessment to discriminate Parkinson's disease from essential tremor with a cross-language approach," npj Digital Medicine, vol. 7, p. 37, 2024. [Online]. Available: https://doi.org/10.1038/ s41746-024-01027-6

[19] P. A. Pérez-Toro, J. C. Vasquez-Correa, T. Arias-Vergara, P. Klumpp, M. Schuster, E. Nöth, and J. R. Orozco-Arroyave, “Emotional State Modeling for the Assessment of Depression in Parkinson's Disease," in Text, Speech, and Dialogue, K. Ekštein, F. Pártl, and M. Konopík, Eds. Cham: Springer International Publishing, 2021, pp. 457–468.

[20] T. Bocklet, E. Nöth, G. Stemmer, H. Ruzickova, and J. Rusz, “"Detection of persons with Parkinson's disease by acoustic, vocal, and prosodic analysis," in 2011 IEEE Workshop on Automatic Speech Recognition & Understanding, 2011, pp. 478–483.

[21] TREND Study Group. (2026) Tübinger Erhebung von Risikofaktoren zur Erkennung von Neurodegeneration (TREND). University Hospital Tübingen. Accessed: 2026-01-03. [Online]. Available: https://www. trend-studie.de/

[22] “On implementing 2D rectangular assignment algorithms, author=Crouse, David F." IEEE Transactions on Aerospace and Electronic Systems, vol. 52, no. 4, pp. 1679–1696, 2016.

[23] P. Virtanen et al., "SciPy 1.0: fundamental algorithms for scientific computing in Python," Nature Methods, vol. 17, pp. 261–272, 2020.

[24] S. Kopar, R. P. Rane, C. Mychajliw, L. Federmann, G. Eschweiler, D. Berg, S. Gijsen, P. A. Perez-Toro, and K. Ritter, "Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy," 2026. [Online]. Available: https://arxiv.org/abs/2605.27189

[25] F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. André, C. Busso, L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan, and K. P. Truong, “The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research and Affective Computing," IEEE Transactions on Affective Computing, vol. 7, no. 2, pp. 190–202, 2016.

[26] F. Eyben, M. Wöllmer, and B. Schuller, “openSMILE: the Munich versatile and fast open-source audio feature extractor," in Proceedings of the 18th ACM international conference on Multimedia, 2010, pp. 1459–1462.

[27] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, "HuBERT: Self-supervised speech representation learning by masked prediction of hidden units," IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 29, pp. 3451–3460, 2021.

[28] S. Chen, C. Wang, Z. Chen, Y. Wu, S. Liu, Z. Chen, J. Li et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[29] A. Baevski, Y. Zhou, A. Mohamed, and M. Auli, "wav2vec 2.0: A framework for self-supervised learning of speech representations," in Advances in Neural Information Processing Systems (NeurIPS), vol. 33, 2020, pp. 12 449–12 460.

[30] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal, K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau, and M. Auli, “XLS-R: Self-supervised cross-lingual speech representation learning at scale," in Proceedings of Interspeech, 2022, pp. 2278–2282.

[31] V. Pratap, A. Tjandra, B. Shi, P. Tomasello, A. Babu, S. Kundu, Z. Ni, A. Vyas, M. Fazel-Zarandi, A. Baevski et al., “Scaling Speech Technology to 1,000+ Languages," Journal of Machine Learning Research (JMLR), vol. 25, 2024, preprint available on arXiv:2305.13516.

[32] F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and E. Duchesnay, "scikit-learn: Machine Learning in Python," pp. 2825–2830, 2011.

[33] G. A. Churchill and R. W. Doerge, "Naive application of permutation testing leads to inflated type I error rates," Genetics, vol. 178, no. 1, pp. 609–610, Jan 2008.

[34] H. B. Mann and D. R. Whitney, "On a test of whether one of two random variables is stochastically larger than the other," The Annals of Mathematical Statistics, vol. 18, no. 1, pp. 50–60, 1947. [Online]. Available: https://doi.org/10.1214/aoms/1177730491

[35] E. J. Ibarra, J. D. Arias-Londoño, M. Zañartu, and J. I. Godino-Llorente, “Towards a Corpus (and Language)-Independent Screening of Parkinson's Disease from Voice and Speech through Domain Adaptation," Bioengineering, vol. 10, no. 11, p. 1316, 2023.