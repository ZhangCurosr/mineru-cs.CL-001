# MD-ProTector: Positioning Multiple Data-Driven Prototypes for LLM-Generated Text Detection

Jinmo Han Jimin Hong Chanyeong Moon

Ju Yeon Kang Seonuk Kim Nam Soo Kim

Department of Electrical and Computer Engineering and INMC Seoul National University, Seoul, Republic of Korea {jinmo, jimin, chanyeong}@hi.snu.ac.kr {juyeon, seonuk}@hi.snu.ac.kr nkim@snu.ac.kr

## Abstract

As LLM-generated content becomes more sophisticated, detection systems for distinguishing those texts from human-written text must operate at scale while handling diverse writing styles, domains, languages, and generator models. Input-only encoder detectors are suitable for practical deployment setting, but standard binary classification supplies only the class label and does not explicitly organize the substantial variation within either class. We propose MD-ProTector, which represents each class with multiple trainable reference vectors in the encoder embedding space, referred to as prototypes. These prototypes provide separate decision boundaries for different groups of texts within the same class. However, adding multiple prototypes alone does not determine which variation each prototype should represent. MD-ProTector addresses this problem with Prototype Positioning loss, which separates class-level structure from the within-class variation that differentiates individual prototypes. Evaluated across five settings from three large-scale benchmarks covering domain, generator, language, and adversarial variation, MD-ProTector achieves the highest AvgRec on MAGE CDCM and RAID and the highest AU-ROC and lowest FPR95 on RAID among the compared encoder-based methods.

## 1 Introduction

As large language models have become increasingly sophisticated and widely accessible, detecting whether text is human-written or LLM-generated has become essential for ensuring credibility in digital communication (Kwon and Jang, 2025). This is particularly important for blocking automated fake news and phishing attacks, as well as maintaining academic integrity (Najjar et al., 2025). In large-scale deployment scenarios, detection systems must be applied to massive volumes of content that span diverse writing styles, domains, and generator models (Li et al., 2024). Treating such diversity is important for practical detection systems operating in real-world cases including various domain, writing style, generator, and adversarial editing cases (Wu et al., 2024).

![](images/638477efd912ffb5d40d5f6aa3f5a14838d8016a21c72119f6ab77cd022965a8.jpg)  
Figure 1: Prototype-Based Inference. Given an input text x, MD-ProTector encodes it into a normalized embedding z and compares it with the machine and human prototype banks. The detection score is $S ( z ) = s _ { 1 } ( z ) - s _ { 0 } ( z )$ , where $s _ { 0 } ( z )$ and $s _ { 1 } ( z )$ are the maximum similarities to the machine and human prototypes, respectively.

In these practical deployment settings, watermarks or model-internal scores such as loglikelihoods are not guaranteed to be available (Christ et al., 2024; Mitchell et al., 2023). Under these conditions, detectors based on lightweight text encoders provide a practical solution. They operate directly on the input text without requiring access to the generation pipeline or model internals (Kuznetsov et al., 2024). This also allows one encoder pipeline to be adapted across target domains without access to the generator internals

(Rodriguez et al., 2022).

One of the simplest designs for encoder-based detectors is to attach a binary classification head and train it using a cross-entropy loss (Ippolito et al., 2020). Despite such intra-class diversity in writing style and domain, all texts are reduced to two categories during training under this design. The standard binary classification objective is a coarse supervision signal that does not explicitly consider such intra-class diversity (Cui et al., 2016). Several prior works have attempted to move beyond standard binary classification formulations by adding structural constraints, but still overlook intra-class diversity within at least one of the two classes, which can limit detection performance (Guo et al., 2024b; Zeng et al., 2025). This observation motivates detectors that represent both humanwritten and LLM-generated texts with multiple local representatives rather than a single global class representation.

A natural extension is to represent each class with multiple trainable reference vectors in the embedding space, which we call prototypes. However, the number of prototypes alone does not determine which pattern of within-class variation each prototype should represent. Without a prototype-specific objective, different prototypes may remain redundant or capture overlapping patterns.

We therefore propose MD-ProTector, an inputonly encoder detector that learns data-driven positions for separate human and machine prototypes. MD-ProTector separates the direction shared within each class from the variation that distinguishes groups of samples within that class. Prototype Positioning loss uses this variation to give different prototypes distinct roles, while complementary objectives keep each prototype aligned with the corresponding human or machine class. The learned prototype banks then directly define the human–machine detection score shown in Figure 1.

We evaluate MD-ProTector under mixed domain and generator conditions, adversarial perturbations, multilingual generalization, held-out domains and held-out generator models. Among input-only encoder detectors trained under the same data, backbone, and validation protocol, MD-ProTector ranks within the top two in AvgRec across all five settings. It achieves the highest AvgRec on MAGE CDCM and RAID, together with the highest AU-ROC and lowest FPR95 on RAID, while DeTeCtive remains stronger on M4. Ablation studies show

Prototype Positioning performs beter than other prototype multiplicity methods such as direct prototype repulsion and positioning before removal of the class-shared direction.

Our contributions are as follows:

• We develop an input-only LLM-generated text detector in which separate human and machine prototype banks directly define the detection score.

• We introduce the Prototype Positioning loss to place each prototype to capture within-class variation in data-driven manner.

• We evaluate MD-ProTector across five controlled input-only encoder settings, where it leads AvgRec on MAGE and RAID and attains the highest AUROC and lowest FPR95 on RAID.

## 2 Related Works

## 2.1 LLM-Generated Text Detection

Prior work on LLM-generated text detection includes watermarking, zero-shot statistical methods, and supervised classifiers, depending on model access and the types of detection signals used (Wu et al., 2025a). Watermarking injects identifiable patterns during generation, enabling content source verification by model providers (Kirchenbauer et al., 2023). Zero-shot statistical detectors, such as GLTR, DetectGPT, Fast-DetectGPT, and Binoculars, use token probabilities, probability curvature, or cross-model likelihood ratios without task-specific training (Gehrmann et al., 2019; Mitchell et al., 2023; Bao et al., 2024; Hans et al., 2024). BISCOPE, ImBD, DetectAnyLLM, and PAWN further refine model-based scoring through memorization, style-aligned discrepancy, task-oriented discrepancy learning, or learned token weighting (Guo et al., 2024a; Chen et al., 2025; Fu et al., 2025; Miralles-González et al., 2026). Rewrite-based methods instead use the amount or pattern of changes produced by an auxiliary LLM (Mao et al., 2024; Hao et al., 2025). These methods can provide strong detection signals, but require access to scoring language models, auxiliary generation calls, or cooperation from the generator.

As a practical alternative in large-scale deployment scenarios, supervised detectors can operate directly on input text and are suitable for in-the-wild deployment (Bakhtin et al., 2019; Uchendu et al.,

2020; Wang et al., 2023). These detectors typically rely on encoder architectures such as BERT (Devlin et al., 2019) or RoBERTa (Liu et al., 2019). Ghostbuster builds a classifier from features extracted by several weaker language models, RADAR improves paraphrase robustness through adversarial training, and MoSEs models stylistic references with input-dependent threshold estimation (Verma et al., 2024; Hu et al., 2023; Wu et al., 2025b). Despite their efficiency, supervised detectors can still degrade when the domain or generator distribution changes (Bakhtin et al., 2019; Li et al., 2024; Wu et al., 2024).

Recent work therefore imposes stronger structure on the representation space. DeTeCtive organizes text instances through author- and styleaware contrastive supervision and performs KNN inference (Guo et al., 2024b). DSVDD takes a oneclass approach, compacting machine-generated embeddings and treating human-written text as outof-distribution (Zeng et al., 2025). SAMP represents both classes with multiple prototypes using source-model supervision (Xu et al., 2026). While these methods introduce instance-level structure, one-class compactness, or source-aware prototypes, binary supervision alone does not specify how internal variation should form distinct training targets for multiple human and machine prototypes. MD-ProTector addresses this gap by constructing a separate positioning target for each prototype from the hub-removed residuals of its associated samples.

## 2.2 Prototype-Based Representation Learning

Prototype-based methods represent each class by one or more representative points in an embedding space and classify inputs based on their similarity to these prototypes. A canonical example is Prototypical Networks, which compute class prototypes as the mean embeddings of support examples and classify queries by distance in a metric-learning framework (Snell et al., 2017). ProtoFewRoBERTa applies this episodic formulation to few-shot detection of AI-generated reviews, while ProtoryNet learns sentence-level reference patterns and classifies documents from their prototype trajectories (Agrahari et al., 2025; Hong et al., 2023). These methods establish prototypes as data-driven class summaries or interpretable reference patterns.

Prototype-based representations have also been applied beyond standard classification. Prototypical Contrastive Learning estimates prototypes as latent cluster variables for representation learning (Li et al., 2021). OOD methods use class prototypes, diversified prototypes, or mixtures of prototypes to represent the known data distribution and score unfamiliar inputs (Chen et al., 2024; Jia et al., 2025; Lu et al., 2024). Multiple normal prototypes have similarly been used in anomaly detection, and multi-prototype modeling has been applied to openset noisy-label learning (Dong et al., 2024; Zhang et al., 2025).

Prior multi-prototype methods learn prototypes from full sample embeddings or through assignment and separation objectives, leaving the respective roles of the class-shared direction and the variation that differentiates individual prototypes underdetermined. Without this distinction, the learning objective does not specify which component should preserve the class decision and which component should organize multiple representatives within the class. MD-ProTector resolves this ambiguity by preserving the shared direction through the class hub and using assignment-weighted, huborthogonal residuals to position each prototype.

## 3 Proposed Method

MD-ProTector jointly learns an encoder and separate prototype banks for human-written and LLMgenerated text. Prototype Positioning organizes each prototype according to the residual variation of its associated samples, while Prototype-to-Class and Sample-to-Prototype preserve class alignment and sample association. The learned prototype banks are used directly for detection.

## 3.1 Problem Formulation and Encoder

Given an input text $x _ { i } ,$ each sample has a binary label $y _ { i } \in \{ 0 , 1 \}$ , where $y _ { i } = 0$ denotes LLMgenerated text and $y _ { i } = 1$ denotes human-written text. A lightweight encoder $f _ { \theta }$ maps the input into token-level representations, which are meanpooled and normalized as

$$
z _ { i } = \mathrm { n o r m } \left( f _ { \theta } ( x _ { i } ) \right) ,\tag{1}
$$

where norm $. ( v ) = v / \| v \| _ { 2 }$ denotes $\ell _ { 2 }$ normalization.

For a mini-batch B in which both classes are represented, let $B _ { c } = \{ i \in B : y _ { i } = c \}$ . The class hub is

$$
h _ { c } = \mathrm { n o r m } \left( \frac { 1 } { | B _ { c } | } \sum _ { i \in B _ { c } } z _ { i } \right) .\tag{2}
$$

![](images/a50e99eae502857419b3f31bfb6be1050b8cef66d76ca916fdb7f134a2496aa1.jpg)  
Figure 2: Overview of the training objectives. Circles denote class hubs, stars denote learnable prototypes, dots denote sample embeddings, and diamonds denote assignment-weighted aggregates of sample residuals. Prototypeto-Class aligns prototypes with their class hubs. Sample-to-Prototype associates samples with the prototype bank of their ground-truth class. Prototype Positioning aligns the residual vector of each prototype with the aggregate constructed for that prototype after removing the class-hub direction. In panel (c), $p _ { c , s } ^ { \perp }$ denotes the residual vector of another prototype with $s \neq r$

It represents the direction shared by class-c samples in the current mini-batch.

For each class $c \in \{ 0 , 1 \}$ , we maintain R learnable prototypes:

$$
\mathcal { P } _ { c } = \{ p _ { c , 1 } , \ldots , p _ { c , R } \} , \qquad \| p _ { c , r } \| _ { 2 } = 1 .\tag{3}
$$

The full prototype bank is $\mathcal { P } = \mathcal { P } _ { 0 } \cup \mathcal { P } _ { 1 }$ . The objectives below preserve class-level alignment while allowing different groups of same-class samples to orient the residual component of each prototype.

## 3.2 Data-Driven Prototype Initialization

Before training, we extract embeddings from the training set and apply K-Means separately within each class (MacQueen, 1967). The resulting centroids initialize $\mathcal { P } _ { c } ^ { ( 0 ) } = \{ p _ { c , r } ^ { ( 0 ) } \} _ { r = 1 } ^ { R }$ . The centroids are normalized and subsequently optimized as learnable parameters together with the encoder. This initialization places the prototype banks in the observed class distributions rather than at random directions.

## 3.3 Training Objectives

For compact notation, define

$$
\phi ( u , v ) = \exp \left( u ^ { \top } v / \tau \right) ,\tag{4}
$$

where $\tau$ is a temperature parameter.

Prototype-to-Class Loss. The first objective aligns each prototype with the hub of its own class:

$$
\mathcal { L } _ { \mathrm { P 2 C } } = - \frac { 1 } { 2 R } \sum _ { c = 0 } ^ { 1 } \sum _ { r = 1 } ^ { R } \log \frac { \phi ( p _ { c , r } , h _ { c } ) } { \sum _ { d = 0 } ^ { 1 } \phi ( p _ { c , r } , h _ { d } ) } .\tag{5}
$$

This objective preserves the class-level orientation of each prototype bank.

Sample-to-Prototype Loss. We compute a soft assignment over the prototypes of the ground-truth class:

$$
q _ { i , r } = \frac { \phi ( z _ { i } , p _ { y _ { i } , r } ) } { \sum _ { k = 1 } ^ { R } \phi ( z _ { i } , p _ { y _ { i } , k } ) } .\tag{6}
$$

The assignment is treated as a stop-gradient target in the Sample-to-Prototype loss:

$$
\begin{array} { c } { { \displaystyle \mathcal { L } _ { \mathrm { S 2 P } } = - \frac { 1 } { | B | } \sum _ { i \in B } \sum _ { r = 1 } ^ { R } \mathrm { s g } ( q _ { i , r } ) } } \\ { { \displaystyle \phantom { \sum _ { i \in B } ^ { R } \sum _ { i \in B } ^ { R } \sum _ { r = 1 } ^ { R } } \cdot \log \frac { \phi \left( z _ { i } , p _ { y _ { i } , r } \right) } { \sum _ { p \in \mathcal { P } } \phi \left( z _ { i } , p \right) } . } } \end{array}\tag{7}
$$

Here, sg(·) denotes the stop-gradient operator. This objective associates each sample with its groundtruth prototype bank while separating it from the opposite-class bank.

Prototype Positioning Loss. Sample-to-Prototype associates samples with prototypes but does not provide a prototype-specific target for within-class variation. We therefore remove the class-hub component from the samples and prototypes:

$$
z _ { i } ^ { \bot } = z _ { i } - ( z _ { i } ^ { \top } h _ { y _ { i } } ) h _ { y _ { i } } ,\tag{8}
$$

$$
p _ { c , r } ^ { \perp } = \operatorname { n o r m } \left( p _ { c , r } - ( p _ { c , r } ^ { \top } h _ { c } ) h _ { c } \right) .\tag{9}
$$

Here, $p _ { c , r } ^ { \perp }$ is the normalized residual vector of prototype $p _ { c , r }$ after removing its class-hub component.

Using the assignments in Equation 6, we construct a residual aggregate for each prototype:

$$
g _ { c , r } ^ { \perp } = \sum _ { i \in B _ { c } } q _ { i , r } z _ { i } ^ { \perp } , \qquad \bar { z } _ { c , r } ^ { \perp } = \mathrm { n o r m } \left( g _ { c , r } ^ { \perp } \right) .\tag{10}
$$

The vector $\bar { z } _ { c , r } ^ { \perp }$ summarizes the residual variation of class-c samples associated with prototype $p _ { c , r }$

Let $\mathcal { P } ^ { \perp } = \{ p _ { d , k } ^ { \perp } \} _ { d , k }$ denote the set of prototype residual vectors. The Prototype Positioning loss is

$$
\mathcal { L } _ { \mathrm { P P } } = - \frac { 1 } { 2 R } \sum _ { c = 0 } ^ { 1 } \sum _ { r = 1 } ^ { R } \log \frac { \phi ( \bar { z } _ { c , r } ^ { \perp } , p _ { c , r } ^ { \perp } ) } { \sum _ { p ^ { \perp } \in \mathcal { P } ^ { \perp } } \phi ( \bar { z } _ { c , r } ^ { \perp } , p ^ { \perp } ) } .\tag{11}
$$

Equation 11 is a softmax cross-entropy over the prototype residual vectors, with a separate dataderived target constructed for each prototype. Residual vectors from both classes appear in the denominator and therefore compete in the shared embedding space. Because $p _ { c , r } ^ { \perp }$ is normalized, Prototype Positioning controls the direction of the residual component. Prototype-to-Class preserves class alignment, and Sample-to-Prototype connects the resulting prototype to same-class samples.

Final Training Objective. The final objective is

$$
\mathcal { L } _ { \mathrm { t r a i n } } = \mathcal { L } _ { \mathrm { P 2 C } } + \mathcal { L } _ { \mathrm { S 2 P } } + \mathcal { L } _ { \mathrm { P P } } .\tag{12}
$$

We jointly optimize the encoder and prototype parameters and renormalize the prototypes after each update.

## 3.4 Inference

Given an input text x, we compute $z =$ norm $( f _ { \boldsymbol { \theta } } ( \boldsymbol { x } ) )$ and score each class by its most similar prototype:

$$
s _ { c } ( z ) = \operatorname* { m a x } _ { r \in \{ 1 , \ldots , R \} } z ^ { \top } p _ { c , r } .\tag{13}
$$

The detection score and prediction are

$$
S ( z ) = s _ { 1 } ( z ) - s _ { 0 } ( z ) , \qquad \hat { y } = { \bf 1 } \{ S ( z ) > \delta \} .\tag{14}
$$

Appendix B.2 evaluates a weighted within-class alternative using the same frozen model parameters.

## 4 Experiments

## 4.1 Experimental Setup

Evaluation Settings. We evaluate MD-ProTector on MAGE, RAID, and M4, which cover distinct deployment scenarios. MAGE is used to evaluate domain and generator generalization: MAGE Cross-Domain Cross-Model (MAGE CDCM) tests mixed domain/generator conditions, while MAGE Unseen Domains and MAGE Unseen Models test leave-one-out domain and leave-one-out generatorfamily generalization (Li et al., 2024). M4 evaluates multilingual and unseen-language generalization (Wang et al., 2024). RAID evaluates robustness to adversarial attacks and decoding-related variations (Dugan et al., 2024). Each MAGE leaveone-out scenario trains a separate detector, yielding 10 domain-shift and 7 generator-family-shift evaluations. Detailed dataset statistics and split construction are provided in Appendix A.

Baselines. We compare input-only encoder detectors under a common data and model-access protocol. Binary CE and SupCon provide standard classification and supervised contrastive references, while DeTeCtive and DSVDD introduce structured representation objectives through hierarchical contrastive learning with KNN inference and machine-class compactness, respectively (Guo et al., 2024b; Zeng et al., 2025). All methods use the same data splits, encoder backbone, training budget, checkpoint selection, and validation-based threshold selection. The comparison isolates the detector formulation without access to model internals or auxiliary generation.

Evaluation Metrics. Following MAGE (Li et al., 2024), we use Average Recall (AvgRec), the mean of HumanRec and MachineRec, as the primary metric and report the values in the main tables. AvgRec evaluates the two class recalls with equal weight at the fixed decision threshold, preventing high recall on one class from masking failure on the other. We choose $\delta$ on the validation split to maximize AvgRec and keep it fixed on the test split. Section 4.2 discusses AUROC and FPR95, while complete F1, Accuracy, AUROC, AUPR, FPR95, and per-scenario results are provided in Appendix D.

<table><tr><td>Method</td><td>MAGE CDCM</td><td>RAID</td><td>M4</td></tr><tr><td>Binary CE</td><td>90.79 (83.18/98.40) 94.77</td><td>86.81 (74.60/99.02) 77.67</td><td>76.68 (54.56/98.80) 83.84</td></tr><tr><td>SupCon</td><td>(92.98/96.56) 94.84</td><td>(58.94/96.39) 87.68</td><td>(72.37/95.30) 92.74</td></tr><tr><td>DeTeCtive</td><td>(91.87/97.81) 94.43</td><td>(80.17/95.20) 86.17</td><td>(87.92/97.57) 81.08</td></tr><tr><td>DSVDD</td><td>(95.18/93.67) 95.14</td><td>(76.36/95.97) 88.18</td><td>(63.13/99.03) 86.03</td></tr><tr><td>MD-ProTector</td><td>(95.81/94.47)</td><td>(82.52/93.84)</td><td>(76.87/95.20)</td></tr></table>

Table 1: Benchmark-level evaluation results. Each cell reports AvgRec with HumanRec/MachineRec in parentheses. MAGE CDCM evaluates mixed domain and generator conditions, RAID evaluates adversarial and decoding robustness, and M4 evaluates language shift. Bold and underline indicate the best and second-best AvgRec within each setting, respectively.
<table><tr><td>Method</td><td>MAGE Unseen Domains</td><td>MAGE Unseen Models</td></tr><tr><td rowspan="2">Binary CE</td><td>67.99 (37.04/98.94)</td><td>89.78</td></tr><tr><td></td><td>(85.85/93.71)</td></tr><tr><td rowspan="2">SupCon</td><td>75.11</td><td>90.92</td></tr><tr><td>(56.82/93.41)</td><td>(93.49/88.34)</td></tr><tr><td rowspan="2">DeTeCtive</td><td>76.72</td><td>91.69</td></tr><tr><td>(55.75/97.69) 79.08</td><td>(92.08/91.30)</td></tr><tr><td rowspan="2">DSVDD</td><td>(63.46/94.71)</td><td>90.71 (95.22/86.19)</td></tr><tr><td>78.59</td><td>91.34</td></tr><tr><td>MD-ProTector</td><td>(61.46/95.72)</td><td>(95.63/87.05)</td></tr></table>

Table 2: MAGE leave-one-out evaluation results. Each cell reports AvgRec with HumanRec/MachineRec in parentheses. MAGE Unseen Domains and MAGE Unseen Models report averages over 10 independently trained leave-one-out-domain scenarios and 7 independently trained leave-one-generator-family-out scenarios, respectively. Bold and underline indicate the best and second-best AvgRec within each setting.

Implementation Details. Unless otherwise specified, all encoder-based methods use the same backbone encoder within each evaluation setting. Pretrained encoder checkpoints are loaded from the HuggingFace model hub and used with the HuggingFace Transformers implementation (Wolf et al., 2020). We use 125M Unsupervised SimCSE-RoBERTa as the default encoder (Gao et al., 2021). All models are trained with a batch size of 256 using AdamW with a learning rate of $2 \times 1 0 ^ { - 5 }$ . Models are trained for 30 epochs with 2,000 warmup steps. The checkpoint with the best AvgRec on the validation set is selected for evaluation. For MD-ProTector, the number of prototypes per class is set to R = 8. All experiments are conducted on a single NVIDIA B200 GPU with mixed BF16 precision.

## 4.2 Evaluation Results

Tables 1 and 2 report AvgRec together with the recall for each class under the five evaluation settings.

Mixed and Adversarial Conditions. MD-ProTector obtains the highest AvgRec on both MAGE CDCM and RAID. On MAGE CDCM, it reaches 95.14 AvgRec with balanced recalls of 95.81 for human and 94.47 for machine text. Its AUROC of 98.41 and FPR95 of 4.89 are also second-best, indicating that the AvgRec gain is accompanied by strong score-level separation under mixed domain and generator conditions. On RAID, MD-ProTector achieves the highest AvgRec (88.18), HumanRec (82.52), and AUROC (95.41), together with the lowest FPR95 (27.78).

Held-Out Generator and Domain Shifts. Under unseen-generator-family evaluation, MD-ProTector obtains the second-highest AvgRec of 91.34 and the lowest FPR95 of 11.44. Its Human-

<table><tr><td>Variant AvgRec</td></tr><tr><td>Prototype Positioning (R = 8, τ = 0.15)</td></tr><tr><td>Full objective 95.14</td></tr><tr><td>w/o LPP 94.78</td></tr><tr><td>PP → Simple Prototype Repulsion 94.55</td></tr><tr><td>PP w/o residual 94.33</td></tr><tr><td>Prototype Initialization</td></tr><tr><td>K-Means 95.14</td></tr><tr><td>Random 94.50</td></tr><tr><td>Number of Prototypes (τ = 0.15)</td></tr><tr><td>R = 1 94.50 R = 2 95.03</td></tr><tr><td>R = 4 94.88</td></tr><tr><td>R = 8 95.14</td></tr><tr><td>94.53</td></tr><tr><td>R = 16</td></tr><tr><td>R = 32 94.36 Temperature (R = 8)</td></tr><tr><td>τ = 0.07 94.91</td></tr><tr><td>τ = 0.10 95.07</td></tr><tr><td>τ = 0.15 95.14</td></tr><tr><td>τ = 0.20 95.05</td></tr><tr><td>93.97</td></tr><tr><td>τ = 0.50 Encoder Backbone</td></tr><tr><td>unsup-simcse-roberta-base 95.14</td></tr><tr><td>roberta-base 94.66</td></tr><tr><td></td></tr><tr><td>sup-simcse-roberta-base 94.01</td></tr><tr><td>e5-base 93.70</td></tr><tr><td>bert-base-uncased 93.60</td></tr><tr><td>unsup-simcse-bert-base 93.24</td></tr><tr><td></td></tr><tr><td>bge-base-en-v1.5 92.86</td></tr></table>

Table 3: Ablation studies on the MAGE CDCM dataset. All entries report AvgRec. The default configuration uses the full objective, $R = 8 , \tau = 0 . 1 5$ , K-Means initialization, and the unsupervised SimCSE-RoBERTa encoder. “PP w/o residual” denotes Prototype Positioning without removing the class hub direction.

Rec of 95.63 is the highest among the evaluated methods, although MachineRec remains lower than that of DeTeCtive. Under unseen-domain evaluation, MD-ProTector again ranks second in AvgRec at 78.59, narrowly below DSVDD at 79.08. DSVDD retains stronger AUROC and FPR95 in this setting, showing that the prototype organization improves class-balanced performance more consistently than score ordering under every type of domain shift.

Language Shift. M4 remains the most challenging setting for MD-ProTector. It obtains the secondhighest AvgRec of 86.03, improving HumanRec to 76.87 compared with 54.56 for Binary CE, 72.37 for SupCon, and 63.13 for DSVDD, while retaining 95.20 MachineRec. However, its AUROC and FPR95 remain below Binary CE and DSVDD. The remaining error is therefore concentrated in representing human-written text in unseen languages, rather than in detecting machine-generated text. Appendix B.2 reports that a weighted average of prototype similarities improves the frozen checkpoint AvgRec without retraining. Across the five settings, MD-ProTector ranks within the top two in AvgRec.

## 4.3 Ablation Studies

Table 3 examines Prototype Positioning together with the main configuration choices on MAGE CDCM. The objective ablation retains Prototype-to-Class and Sample-to-Prototype and uses the same initialization, prototype count, and temperature across all variants. Removing L<sub>PP</sub> lowers AvgRec from 95.14 to 94.78. Replacing Prototype Positioning with simple prototype repulsion yields 94.55, while positioning prototypes without removing the class-hub direction yields 94.33. These comparisons support the proposed formulation, in which each prototype is positioned using the residual variation of its associated samples.

K-Means initialization improves AvgRec from 94.50 to 95.14 relative to random initialization, indicating that the observed class distributions provide a useful starting point for optimization. Increasing the number of prototypes from R = 1 to R = 2 raises AvgRec from 94.50 to 95.03, and reaches its maximum at R = 8 and declines with larger prototype banks. Once multiple prototypes are available, their organization remains important rather than capacity itself.

AvgRec remains between 94.91 and 95.14 for $\tau \in [ 0 . 0 7 , 0 . 2 0 ]$ and decreases to 93.97 at τ = 0.50. Across encoder backbones, AvgRec ranges from 92.86 to 95.14, with unsupervised SimCSE-RoBERTa obtaining the highest value.

## 4.4 Prototype Analysis

Figure 3 shows the learned prototypes distributed across multiple occupied regions of the human and machine embedding spaces. All prototypes show their own test sample covers, showing that both banks retain multiple active representatives and avoid complete assignment collapse.

![](images/84524d3fed2ef1299e87b476f8ffe1d96d007e8938dbcd20f2aea27764c207f0.jpg)

![](images/b9f724061cd74de01df7184ffd153e2b13c5d10051b79cb0baac5bf4ea620701.jpg)  
(b) Colored by generator model

Figure 3: Prototype visualization on the MAGE CDCM dataset. Both panels show the same t-SNE projection of normalized test embeddings and learned prototypes. The left panel colors samples by domain, and the right panel colors samples by generator model. Dots denote test samples. Blue and red stars indicate human and machine prototypes, respectively.  
(a) Machine prototypes
<table><tr><td>Prototype (n)</td><td>Top domain</td><td>Top generator</td><td>Writing cues</td></tr><tr><td>M1 (424)</td><td rowspan="3">SQuAD 19.8% Yelp 62.7%</td><td rowspan="3">OPT-6.7B 11.6% BLOOM-7B 7.5%</td><td>Quotation 77.6%, Instruction 78.1%</td></tr><tr><td>M5 (386) M7 (311)</td><td>Review 58.8%</td></tr><tr><td>SciGen 74.6% GPT-3.5-Turbo 6.4%</td><td>Academic 49.2%</td></tr><tr><td></td><td colspan="3">(b) Human prototypes</td></tr><tr><td>Prototype (n) H0 (147)</td><td colspan="3">Top domain</td></tr><tr><td>H6 (510)</td><td colspan="3">Yelp 93.2% First person 98.6%, Review 85.7% TLDR 41.4%, SciGen 40.6% Academic 33.5%</td></tr></table>

Table 4: Selected groups of texts assigned to prototypes. Top domain and generator are the largest shares among assigned texts. Writing-cue percentages are the fractions containing each cue. Complete summaries and cue definitions are provided in Appendix C.

We use M0–M7 and H0–H7 to identify the machine and human prototypes. Table 4 summarizes selected groups characterized by instructional, review, and academic writing cues. The same cues appear in both classes and describe recurring patterns within each class. The largest generator share among the machine prototypes is 24.8%, indicating that each group includes texts from multiple generators. These results show that the prototype banks organize distinct same-class text groups across domain and generator boundaries. Complete cue definitions and summaries for all prototypes are provided in Appendix C.

## 5 Conclusion

In this work, we introduced MD-ProTector, an input-only encoder detector that represents humanwritten and LLM-generated text with separate banks of trainable prototypes. By separating classlevel alignment from prototype-specific residual positioning, MD-ProTector organizes multiple prototypes using the variation observed within each class while retaining direct prototype-based inference. Across five controlled settings, the method achieves the highest AvgRec on MAGE CDCM and RAID. On RAID, it also attains the highest AU-ROC and lowest FPR95 among the compared methods. Ablations on MAGE CDCM further show that residual positioning yields stronger performance than the single-prototype, direct-repulsion, and raw-space positioning variants. These results support data-derived positioning as an effective mechanism for organizing multiple class prototypes for LLM-generated text detection.

## Limitations

This work assumes a fixed-label binary detection setting in which each input is classified as either human-written or LLM-generated, and does not address more complex scenarios such as partial generation, human–machine co-editing, or estimating degrees of machine involvement. In addition, the number of prototypes per class is treated as an empirically fixed design choice. Adaptive mechanisms for adjusting prototype cardinality based on data characteristics are not explored. Finally, our experiments are conducted under fixed training, validation, and test splits. In practical deployment, the distribution of generators, prompts, writing styles, and adversarial perturbations may change over time. While MD-ProTector initializes and optimizes prototypes from training data, continual prototype adaptation under temporal distribution drift remains future work.

## Ethics Statement

The datasets utilized do not include private data or non-public personally identifiable information. The development of reliable text detection systems is crucial for maintaining trust in digital information. However, we acknowledge that detection technologies can potentially be used in a dual-use manner—adversaries might use our detector as a discriminator to train more sophisticated generators that evade detection. Additionally, while we strove to use diverse datasets, the "Human" class in our training data is sourced from web texts (e.g., Reddit, Wikipedia), which may contain inherent biases. Users should be cautious when deploying this model in sensitive contexts, as false positives could unfairly penalize human writers.

## References

Shifali Agrahari, Sujit Kumar, and Ranbir Singh Sanasam. 2025. Can you really trust that review? ProtoFewRoBERTa and DetectAIRev: A prototypical few-shot method and multi-domain benchmark for detecting AI-generated reviews. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 2118–2140, Mumbai, India. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics.

Anton Bakhtin, Sam Gross, Myle Ott, Yuntian Deng, Marc’Aurelio Ranzato, and Arthur Szlam.

2019. Real or fake? learning to discriminate machine from human generated text. arXiv preprint arXiv:1906.03351.

Guangsheng Bao, Yanbin Zhao, Zhiyang Teng, Linyi Yang, and Yue Zhang. 2024. Fast-detectGPT: Efficient zero-shot detection of machine-generated text via conditional probability curvature. In International Conference on Learning Representations.

Jiaqi Chen, Xiaoye Zhu, Tianyang Liu, Ying Chen, Xinhui Chen, Yiwen Yuan, Chak Tou Leong, Zuchao Li, Long Tang, Lei Zhang, Chenyu Yan, Guanghao Mei, Jie Zhang, and Lefei Zhang. 2025. Imitate before detect: Aligning machine stylistic preference for machine-revised text detection. Proceedings of the AAAI Conference on Artificial Intelligence, 39(22):23559–23567.

Jun-Kun Chen, Jilin Mei, Liang Chen, Fangzhou Zhao, Yan Xing, and Yu Hu. 2024. Proto-OOD: Enhancing OOD object detection with prototype feature similarity. arXiv preprint arXiv:2409.05466.

Miranda Christ, Sam Gunn, and Or Zamir. 2024. Undetectable watermarks for language models. In Proceedings of Thirty Seventh Conference on Learning Theory, volume 247 of Proceedings ofMachine Learning Research, pages 1125–1139. PMLR.

Yin Cui, Feng Zhou, Yuanqing Lin, and Serge Belongie. 2016. Fine-grained categorization and dataset bootstrapping using deep metric learning with humans in the loop. In IEEE Conference on Computer Vision and Pattern Recognition, pages 1153–1162, Las Vegas, NV, USA. IEEE.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Zhijin Dong, Hongzhi Liu, Boyuan Ren, Weimin Xiong, and Zhonghai Wu. 2024. Reconstructionbased multi-normal prototypes learning for weakly supervised anomaly detection. arXiv preprint arXiv:2408.14498.

Liam Dugan, Alyssa Hwang, Filip Trhlík, Andrew Zhu, Josh Magnus Ludan, Hainiu Xu, Daphne Ippolito, and Chris Callison-Burch. 2024. RAID: A shared benchmark for robust evaluation of machinegenerated text detectors. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 12463– 12492, Bangkok, Thailand. Association for Computational Linguistics.

Jiachen Fu, Chun-Le Guo, and Chongyi Li. 2025. DetectAnyLLM: Towards generalizable and robust detection of machine-generated text across domains

and models. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 11229– 11238.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple contrastive learning of sentence embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Sebastian Gehrmann, Hendrik Strobelt, and Alexander Rush. 2019. GLTR: Statistical detection and visualization of generated text. In Proceedings ofthe 57th Annual Meeting ofthe Associationfor Computational Linguistics: System Demonstrations, pages 111–116, Florence, Italy. Association for Computational Linguistics.

Hanxi Guo, Siyuan Cheng, Xiaolong Jin, Zhuo Zhang, Kaiyuan Zhang, Guanhong Tao, Guangyu Shen, and Xiangyu Zhang. 2024a. BISCOPE: AI-generated text detection by checking memorization of preceding tokens. In Advances in Neural Information Processing Systems, volume 37, pages 104065–104090. Curran Associates, Inc.

Xun Guo, Shan Zhang, Yongxin He, Ting Zhang, Wanquan Feng, Haibin Huang, and Chongyang Ma. 2024b. DeTeCtive: Detecting AI-generated text via multi-level contrastive learning. In Advances in Neural Information Processing Systems, volume 37, pages 88320–88347. Curran Associates, Inc.

Abhimanyu Hans, Avi Schwarzschild, Valeriia Cherepanova, Hamid Kazemi, Aniruddha Saha, Micah Goldblum, Jonas Geiping, and Tom Goldstein. 2024. Spotting LLMs with binoculars: Zero-shot detection of machine-generated text. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 17519–17537. PMLR.

Wei Hao, Ran Li, Weiliang Zhao, Junfeng Yang, and Chengzhi Mao. 2025. Learning to rewrite: Generalized LLM-generated text detection. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6421–6434, Vienna, Austria. Association for Computational Linguistics.

Dat Hong, Tong Wang, and Stephen Baek. 2023. ProtoryNet: Interpretable text classification via prototype trajectories. Journal ofMachine Learning Research, 24(264):1–39.

Xiaomeng Hu, Pin-Yu Chen, and Tsung-Yi Ho. 2023. RADAR: Robust AI-text detection via adversarial learning. In Advances in Neural Information Processing Systems, volume 36. Curran Associates, Inc.

Daphne Ippolito, Daniel Duckworth, Chris Callison-Burch, and Douglas Eck. 2020. Automatic detection of generated text is easiest when humans are fooled. In Proceedings of the 58th Annual Meeting of

the Association for Computational Linguistics, pages 1808–1822, Online. Association for Computational Linguistics.

Yulong Jia, Jiaming Li, Ganlong Zhao, Shuangyin Liu, Weijun Sun, Liang Lin, and Guanbin Li. 2025. Enhancing out-of-distribution detection via diversified multi-prototype contrastive learning. Pattern Recognition, 161:111214.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. 2023. A watermark for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 17061–17084. PMLR.

Kristian Kuznetsov, Eduard Tulchinskii, Laida Kushnareva, German Magai, Serguei Barannikov, Sergey Nikolenko, and Irina Piontkovskaya. 2024. Robust AI-generated text detection by restricted embeddings. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 17036–17055, Miami, Florida, USA. Association for Computational Linguistics.

Soonchan Kwon and Beakcheol Jang. 2025. A comprehensive survey of fake text detection on misinformation and LM-generated texts. IEEE Access, 13:25301–25324.

Junnan Li, Pan Zhou, Caiming Xiong, and Steven Hoi. 2021. Prototypical contrastive learning of unsupervised representations. In International Conference on Learning Representations.

Yafu Li, Qintong Li, Leyang Cui, Wei Bi, Zhilin Wang, Longyue Wang, Linyi Yang, Shuming Shi, and Yue Zhang. 2024. MAGE: Machine-generated text detection in the wild. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 36–53, Bangkok, Thailand. Association for Computational Linguistics.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. arXiv preprint arXiv:1907.11692.

Haodong Lu, Dong Gong, Shuo Wang, Jason Xue, Lina Yao, and Kristen Moore. 2024. Learning with mixture of prototypes for out-of-distribution detection. In International Conference on Learning Representations.

J. MacQueen. 1967. Some methods for classification and analysis of multivariate observations. In Proceedings ofthe Fifth Berkeley Symposium on Mathematical Statistics and Probability, pages 281–297.

Chengzhi Mao, Carl Vondrick, Hao Wang, and Junfeng Yang. 2024. RAIDAR: Generative AI detection via rewriting. In International Conference on Learning Representations.

Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, and David Camacho. 2026. Not all tokens are created equal: Perplexity attention weighted networks for AI-generated text detection. Information Fusion, 125:103465.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D. Manning, and Chelsea Finn. 2023. DetectGPT: Zero-shot machine-generated text detection using probability curvature. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 24950–24962. PMLR.

Ayat A. Najjar, Huthaifa I. Ashqar, Omar A. Darwish, and Eman M. Hammad. 2025. Detecting aigenerated text in educational content: Leveraging machine learning and explainable AI for academic integrity. arXiv preprint arXiv:2501.03203.

Juan Diego Rodriguez, Todd Hay, David Gros, Zain Shamsi, and Ravi Srinivasan. 2022. Cross-domain detection of GPT-2-generated technical text. In Proceedings ofthe 2022 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, pages 1213–1233, Seattle, United States. Association for Computational Linguistics.

Jake Snell, Kevin Swersky, and Richard Zemel. 2017. Prototypical networks for few-shot learning. In Advances in Neural Information Processing Systems, volume 30, pages 4077–4087. Curran Associates, Inc.

Adaku Uchendu, Thai Le, Kai Shu, and Dongwon Lee. 2020. Authorship attribution for neural text generation. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pages 8384–8395, Online. Association for Computational Linguistics.

Vivek Verma, Eve Fleisig, Nicholas Tomlin, and Dan Klein. 2024. Ghostbuster: Detecting text ghostwritten by large language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1702–1717, Mexico City, Mexico. Association for Computational Linguistics.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weaklysupervised contrastive pre-training. arXiv preprint arXiv:2212.03533.

Yuxia Wang, Jonibek Mansurov, Petar Ivanov, Jinyan Su, Artem Shelmanov, Akim Tsvigun, Chenxi Whitehouse, Osama Mohammed Afzal, Tarek Mahmoud, Toru Sasaki, Thomas Arnold, Alham Fikri Aji, Nizar Habash, Iryna Gurevych, and Preslav Nakov. 2024. M4: Multi-generator, multi-domain, and multilingual black-box machine-generated text detection. In Proceedings of the 18th Conference of the European

Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1369–1407, St. Julian’s, Malta. Association for Computational Linguistics.

Zecong Wang, Jiaxi Cheng, Chen Cui, and Chenhao Yu. 2023. Implementing BERT and fine-tuned RoBERTa to detect AI-generated news by ChatGPT. arXiv preprint arXiv:2306.07401.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, and 3 others. 2020. Transformers: State-of-the-art natural language processing. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online. Association for Computational Linguistics.

Junchao Wu, Shu Yang, Runzhe Zhan, Yulin Yuan, Lidia Sam Chao, and Derek Fai Wong. 2025a. A survey on LLM-generated text detection: Necessity, methods, and future directions. Computational Linguistics, 51(1):275–338.

Junchao Wu, Runzhe Zhan, Derek Wong, Shu Yang, Xinyi Yang, Yulin Yuan, and Lidia Chao. 2024. DetectRL: Benchmarking LLM-generated text detection in real-world scenarios. In Advances in Neural Information Processing Systems, volume 37, pages 100369–100401. Curran Associates, Inc.

Junxi Wu, Jinpeng Wang, Zheng Liu, Bin Chen, Dongjian Hu, Hao Wu, and Shu-Tao Xia. 2025b. MoSEs: Uncertainty-aware AI-generated text detection via mixture of stylistics experts with conditional thresholds. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 5786–5805, Suzhou, China. Association for Computational Linguistics.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighoff, Defu Lian, and Jian-Yun Nie. 2024. C-Pack: Packed resources for general Chinese embeddings. In Proceedings ofthe 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 641–649, Washington, DC, USA. Association for Computing Machinery.

Yan Xu, Wenzhong Yang, Yabo Yin, Hongzhen Lv, Zhenhua Wang, Jingfeng He, Xiangyi Jia, and Xianfeng Wang. 2026. SAMP: Source-aware multiprototype learning for machine-generated text detection. Research Square. Preprint.

Cong Zeng, Shengkun Tang, Yuanzhou Chen, Zhiqiang Shen, Wenchao Yu, Xujiang Zhao, Haifeng Chen, Wei Cheng, and Zhiqiang Xu. 2025. Human texts are outliers: Detecting LLM-generated texts via outof-distribution detection. In Advances in Neural Information Processing Systems, volume 38. Curran Associates, Inc.

Yue Zhang, Yiyi Chen, Chaowei Fang, Qian Wang, Jiayi Wu, and Jingmin Xin. 2025. Learning from openset noisy labels based on multi-prototype modeling. Pattern Recognition, 157:110902.

## A Evaluation Set and Protocol Details

This section reports dataset statistics and protocol details for MAGE, M4, and RAID. All datasets are cast as binary human-written versus machinegenerated text detection, and human-written text is treated as the positive class in our metric implementation. Validation data are used only to select checkpoints, thresholds, and hyperparameters. Test labels are not used for any of these choices.

MAGE Splits. MAGE CDCM uses the crossdomain cross-model setting of MAGE (Li et al., 2024). The processed split contains 10 source domains: CMV, ELI5, HellaSwag, ROC, SciGen, SQuAD, TLDR, WP, XSum, and Yelp. Machinegenerated texts are produced by 27 model variants grouped into seven Generator Model Families: LLaMA, BigScience, FLAN-T5, GLM-130B, EleutherAI, OpenAI, and OPT. For MAGE Unseen Domains, each leave-one-domain-out scenario excludes one source domain from training and validation and evaluates on that domain. For MAGE Unseen Models, each leave-one-generator-familyout scenario excludes one Generator Model Family from training and validation and evaluates on that family. Specifically, we used LLaMA-7B, BLOOM-7B, FLAN-T5-Small, GLM-130B, GPT-J, GPT-3.5-Turbo and OPT-125M model for the unseen-model setting evaluation. The test sizes for the leave-one-out scenarios are listed in Table 6.

M4 Split. M4 is a multi-generator, multidomain, and multilingual benchmark for blackbox machine-generated text detection (Wang et al., 2024). We use the SemEval-2024 M4 Subtask A multilingual protocol with the multilingual train, development, and test splits. The development split is used to select the checkpoint and threshold and is not merged into training. The training split spans nine sources, while the multilingual test split contains Arabic, German, Italian, and English. The test-time generator labels include BLOOMZ, ChatGPT, Cohere, Davinci, Dolly, JAIS-30B, and LLaMA2 fine-tuned.

RAID Split. RAID is designed to evaluate detector robustness across domains, generators, decoding strategies, and adversarial perturbations (Dugan et al., 2024). In our experiments, we use the RAID split preprocessed by Zeng et al. (2025). We apply a 90%/10% split to the processed RAID training data, stratified by binary label. The resulting train split is used for fitting, valid is used to select the checkpoint and threshold, and the processed RAID test split is used for final evaluation.

RAID covers eight domains: abstracts, books, news, poetry, recipes, Reddit, reviews, and Wiki. The machine-generated side spans several generator families, including GPT-series, Cohere, LLaMA, Mistral, and MPT variants, alongside human-written text. The train and validation splits contain clean examples, while the final test split contains both clean and perturbed examples. The perturbations include paraphrase, homoglyph, whitespace, zero-width-space, synonym, and spelling or formatting perturbations. Thus, RAID evaluates test-time robustness to adversarial and decoding-related variation, while model selection is performed only on clean validation data.

MAGE Leave-one-out Settings. For MAGE Unseen Domains and MAGE Unseen Models, each leave-one-out scenario trains a separate detector. Aggregate metrics are computed as arithmetic means over scenario-level results:

$$
m _ { \mathrm { U D } } = \frac { 1 } { 1 0 } \sum _ { d \in \mathcal { D } _ { \mathrm { U D } } } m _ { d } , \qquad m _ { \mathrm { U M } } = \frac { 1 } { 7 } \sum _ { g \in \mathcal { G } _ { \mathrm { U M } } } m _ { g } .
$$

Thus, MAGE leave-one-out results are scenariowise macro averages. All metrics in Appendix D are reported as percentages.

## B Ablation Definitions and Additional Analyses

This appendix provides the exact definitions of the nonstandard variants evaluated in Table 3 and reports the weighted-inference and mini-batch-hub analyses. The ablation results and their interpretation are presented in Section 4.3 and are not repeated here.

## B.1 Ablation Definitions

Prototype Repulsion. We replace Prototype Positioning with a direct same-class prototype repulsion loss. Because the prototypes are normalized, their inner products correspond to cosine similarity:

$$
\mathcal { L } _ { \mathrm { r e p } } = \frac { 1 } { 2 R ( R - 1 ) } \sum _ { c = 0 } ^ { 1 } \sum _ { r = 1 } ^ { R } \sum _ { k \neq r } \left[ \operatorname* { m a x } \left( 0 , p _ { c , r } ^ { \top } p _ { c , k } \right) \right] ^ { 2 } .\tag{15}
$$

<table><tr><td colspan="3">MAGE CDCM</td></tr><tr><td>Split</td><td>Total</td><td>Human</td><td>Machine</td></tr><tr><td>Train</td><td>319,071</td><td>93,318</td><td>225,753</td></tr><tr><td>Valid</td><td>56,792</td><td>28,799</td><td>27,993</td></tr><tr><td>Test</td><td>56,819</td><td>28,741</td><td>28,078</td></tr></table>

<table><tr><td colspan="3">M4 Multilingual</td></tr><tr><td>Split</td><td>Total</td><td>Human Machine</td></tr><tr><td>Train</td><td>172,417</td><td>83,846 88,571</td></tr><tr><td>Valid</td><td>4,000 2,000</td><td>2,000</td></tr><tr><td>Test</td><td>42,378 20,238</td><td>22,140</td></tr></table>

<table><tr><td colspan="4">RAID</td></tr><tr><td>Split</td><td>Total</td><td>Human</td><td>Machine</td></tr><tr><td>Train</td><td>303,247</td><td>8,571</td><td>294,676</td></tr><tr><td>Valid</td><td>33,694</td><td>952</td><td>32,742</td></tr><tr><td>Test</td><td>112,317</td><td>3,232</td><td>109,085</td></tr></table>

Table 5: Split sizes after preprocessing. Each panel reports the total number of examples and the counts for each class for one evaluation setting. MAGE CDCM denotes the cross-domain cross-model setting of MAGE. For M4, Valid corresponds to the official development split and is not merged into training. For RAID, Train and Valid are obtained by a deterministic 90%/10% split of the processed RAID training data.
<table><tr><td colspan="3">MAGE Unseen Domains</td></tr><tr><td>Domain</td><td>Total Human</td><td>Machine</td></tr><tr><td>CMV</td><td>4,917 2,403</td><td>2,514</td></tr><tr><td>ELI5</td><td>6,351 3,156</td><td>3,195</td></tr><tr><td>HellaSwag</td><td>6,347 3,292</td><td>3,055</td></tr><tr><td>ROC</td><td>6,462 3,275</td><td>3,187</td></tr><tr><td>SciGen</td><td>4,789 2,538</td><td>2,251</td></tr><tr><td>SQuAD</td><td>5,004 2,508</td><td>2,496</td></tr><tr><td>TLDR</td><td>4,977 2,535</td><td>2,442</td></tr><tr><td>WP</td><td>6,236 3,099</td><td>3,137</td></tr><tr><td>XSum</td><td>6,537 3,283</td><td>3,254</td></tr><tr><td>Yelp</td><td>5,199 2,652</td><td>2,547</td></tr></table>

<table><tr><td colspan="4">MAGE Unseen Models</td></tr><tr><td>Generator Model Family</td><td>Total</td><td>Human</td><td>Machine</td></tr><tr><td>GLM-130B</td><td>1,838</td><td>919</td><td>919</td></tr><tr><td>LLaMA</td><td>7,420</td><td>3,710</td><td>3,710</td></tr><tr><td>BigScience</td><td>5,386</td><td>2,693</td><td>2,693</td></tr><tr><td>FLAN-T5</td><td>9,320</td><td>4,660</td><td>4,660</td></tr><tr><td>OpenAI</td><td>13,284</td><td>6,642</td><td>6,642</td></tr><tr><td>EleutherAI</td><td>2,884</td><td>1,442</td><td>1,442</td></tr><tr><td>OPT</td><td>16,024</td><td>8,012</td><td>8,012</td></tr></table>

Table 6: Test sizes for the MAGE leave-one-out evaluations. The left panel lists leave-one-domain-out scenarios, and the right panel lists leave-one-generator-family-out scenarios. Each row corresponds to an independently trained leave-one-out evaluation scenario.

This objective separates same-class prototype vectors without using sample-dependent positioning targets. All other objectives and training settings remain unchanged.

Raw-Space Prototype Positioning. We also evaluate Prototype Positioning without removing the class-hub direction. For each prototype, the assignment-weighted aggregate is formed directly from the normalized sample embeddings:

$$
g _ { c , r } = \sum _ { i \in \mathcal { B } _ { c } } q _ { i , r } z _ { i } , \qquad \bar { z } _ { c , r } = \mathrm { n o r m } _ { \varepsilon } ( g _ { c , r } ) .\tag{16}
$$

The corresponding objective is

$$
\mathcal { L } _ { \mathrm { P P - r a w } } = - \frac { 1 } { 2 R } \sum _ { c = 0 } ^ { 1 } \sum _ { r = 1 } ^ { R } \log \frac { \phi ( \bar { z } _ { c , r } , p _ { c , r } ) } { \sum _ { p \in \mathcal { P } } \phi ( \bar { z } _ { c , r } , p ) } .\tag{17}
$$

This variant retains prototype-specific sample aggregation while omitting the decomposition into class-shared and residual components.

Encoder Backbones. The encoder rows in Table 3 cover backbones from the SimCSE, RoBERTa, BERT, E5, and BGE families (Gao et al., 2021; Liu et al., 2019; Devlin et al., 2019; Wang et al., 2022; Xiao et al., 2024). All backbone variants use the same prototype configuration and training protocol.

<table><tr><td>Setting</td><td>Hard max</td><td>Weighted</td></tr><tr><td>MAGE CDCM</td><td>95.14</td><td>95.08</td></tr><tr><td>RAID</td><td>88.18</td><td>88.25</td></tr><tr><td>M4</td><td>86.03</td><td>88.54</td></tr></table>

Table 7: AvgRec under hard-maximum and weighted prototype inference.

## B.2 Weighted Prototype Inference

The main method scores each class using its maximum prototype similarity. We additionally evaluate a weighted class score using the same temperature as the within-class assignments in Equation 6:

$$
w _ { c , r } ( z ) = \frac { \exp ( z ^ { \top } p _ { c , r } / \tau ) } { \sum _ { k = 1 } ^ { R } \exp ( z ^ { \top } p _ { c , k } / \tau ) } ,\tag{18}
$$

$$
\widetilde { s } _ { c } ( z ) = \sum _ { r = 1 } ^ { R } w _ { c , r } ( z ) { z } ^ { \top } p _ { c , r } ,\tag{19}
$$

$$
\widetilde { S } ( z ) = \widetilde { s } _ { 1 } ( z ) - \widetilde { s } _ { 0 } ( z ) .\tag{20}
$$

All model parameters remain frozen; only the rule used to combine prototype similarities within each class is changed.

Weighted inference leaves MAGE CDCM and RAID nearly unchanged and improves M4 without retraining. Its effect is therefore most pronounced under the M4 evaluation setting.

<table><tr><td>Dataset</td><td>M/H per batch</td><td>Hub cosine M/H</td></tr><tr><td>MAGE CDCM</td><td>181 / 75</td><td>0.9995 / 0.9963</td></tr><tr><td>RAID</td><td>249 / 7</td><td>1.0000 / 0.9883</td></tr><tr><td>M4</td><td>132 / 124</td><td>0.9996 /0.9994</td></tr></table>

Table 8: Mean mini-batch composition and cosine similarity between mini-batch hubs and the corresponding full-training-data class directions. Batch counts are rounded to the nearest sample.

## B.3 Mini-Batch Hub Stability

We sample frozen training embeddings using the same sampler and batch size as training and compare each mini-batch hub with the corresponding class direction computed from the full training set.

The mini-batch hubs remain closely aligned with the corresponding full-data directions across all three datasets. This alignment is also maintained for the human class in RAID, despite its substantially smaller batch count.

## C Details of Prototype Analysis

This appendix reports the residual-to-full preference analysis, defines the writing cues used in Section 4.4, and provides complete summaries of assigned texts together with their split-half stability. The cue definitions are fixed before prototype-wise aggregation.

## C.1 Residual-to-Full Prototype Preference

For a class-c vector, define its normalized hubremoved representation as

$$
R _ { c } ( v ) = \mathrm { n o r m } \left( v - ( v ^ { \top } h _ { c } ) h _ { c } \right) .\tag{21}
$$

For each test sample $i ,$ we compare the same-class full-space and hub-removed preferences,

$$
\boldsymbol { r } _ { i } ^ { \mathrm { f u l l } } = \arg \operatorname* { m a x } _ { \boldsymbol { r } } \boldsymbol { z } _ { i } ^ { \top } \boldsymbol { p } _ { y _ { i } , \boldsymbol { r } } ,\tag{22}
$$

$$
r _ { i } ^ { \perp } = \arg \operatorname* { m a x } _ { r } R _ { y _ { i } } ( z _ { i } ) ^ { \top } R _ { y _ { i } } ( p _ { y _ { i } , r } ) .\tag{23}
$$

The analysis uses the same R = 8 hard-max checkpoint and 7,200-sample MAGE CDCM test capture, containing 3,600 machine and 3,600 human samples. The two preferences agree for 98.9% of machine samples and 98.3% of human samples. Thus, the prototype preferred after removing the class hub is almost always the same prototype selected in the full space at inference.

## C.2 Writing Cue Definitions

We compute 21 post-hoc measurements for every MAGE CDCM test sample. Table 9 defines the cue names used in the main paper, and Tables 10 and 11 report the complete prototype summaries. The analysis implementation fixes the lexicons, regular expressions, tokenizer, and sentence-segmentation rules before prototype-wise aggregation.

<table><tr><td>Writing cue</td><td>Measured as</td><td>Writing cue</td><td>Measured as</td></tr><tr><td>Words</td><td>Number of word tokens</td><td>Instruction</td><td>Imperative or procedural expressions</td></tr><tr><td>Sentences</td><td>Number of detected sentences</td><td>Attribution</td><td>Reporting or attribution expressions</td></tr><tr><td>Sentence length</td><td>Word count divided by sentence count</td><td>Headings</td><td>Section-heading patterns</td></tr><tr><td>Lexical diversity</td><td>Unique word types divided by word count</td><td>Lists</td><td>List bullets or enumerated items</td></tr><tr><td>Token repetition</td><td>Fraction of word tokens occurring more than once</td><td>Templates</td><td>Bracketed template or pipeline markers</td></tr><tr><td>Single-use vocab- ulary</td><td>Fraction of word types occurring once</td><td>Academic</td><td>Academic-register expressions</td></tr><tr><td>Word length</td><td>Mean characters per word</td><td>Review vocabu- lary</td><td>Any whole-word match to star(s), service, staff, food, price, hotel, restaurant, rec-</td></tr><tr><td>Punctuation</td><td>Punctuation characters divided by text length</td><td>First person</td><td>ommend, delicious, favorite, or ordered Any of I, me, my, mine, we, us, our, ours</td></tr><tr><td>Phrase repetition</td><td>Repeated word trigrams divided by avail- able trigrams</td><td>Second person</td><td>(case-insensitive) Any of you, your, yours (case-insensitive)</td></tr><tr><td>Questions</td><td>Question-form markers</td><td>Links/markup</td><td></td></tr><tr><td>Quotation</td><td>Quotation or quoted-speech markers</td><td></td><td>URLs or markup patterns</td></tr></table>

Table 9: The 21 post-hoc measurements used in the prototype analysis. Continuous cues use word and sentence statistics. Binary cues use fixed lexicons or regular expressions.

The review-vocabulary cue uses the fixed, caseinsensitive term list shown in Table 9; it is not a learned review classifier.

## C.3 Stability of Prototype Descriptions

We recompute the 21-dimensional featureassociation vector for every prototype over 50 label-by-domain-stratified split halves. Within each half, each feature is standardized across the 16 prototypes before comparing the two vectors for the same prototype. The mean split-half cosine is 0.837 for machine prototypes and 0.908 for human prototypes. All 16 prototypes receive at least 20 hard assignments in the 7,200-sample test capture.

## C.4 Complete Prototype Summaries

Tables 10 and 11 summarize the texts assigned to all machine and human prototypes. The cue names match the inventory in Table 9.

<table><tr><td>Prototype (n)</td><td>Top domain</td><td>Top generator</td><td>Distinctive characteristics</td></tr><tr><td>M0 (525)</td><td>XSum 34.7%</td><td>OPT-30B 11.6%</td><td></td></tr><tr><td>M1 (424)</td><td>SQuAD 19.8%</td><td>OPT-6.7B 11.6%</td><td>Quotation occurs in 77.6% and instruction expressions in 78.1% of assigned texts.</td></tr><tr><td>M2 (541)</td><td>TLDR 14.8%</td><td>LLaMA-13B 17.4%</td><td></td></tr><tr><td>M3 (479)</td><td>HellaSwag 22.8%</td><td>FLAN-T5-x1 15.0%</td><td>Academic expressions occur in 0.2% of assigned texts and 6.9% of the remaining machine texts.</td></tr><tr><td>M4 (459)</td><td>ELI5 29.8%</td><td>text-davinci-003 22.4%</td><td>Template markers occur in 0.9% of assigned texts and 6.1% of the remaining machine texts.</td></tr><tr><td>M5 (386) M6 (475)</td><td>Yelp 62.7%</td><td>BLOOM-7B 7.5%</td><td>Review vocabulary occurs in 58.8% of assigned texts.</td></tr><tr><td></td><td>ROCStories 30.3%</td><td>GPT-3.5-Turbo 24.8%</td><td></td></tr><tr><td>M7 (311)</td><td>SciGen 74.6%</td><td>GPT-3.5-Turbo 6.4%</td><td>Academic expressions occur in 49.2% of assigned texts.</td></tr></table>

Table 10: Complete summaries of texts assigned to the machine prototypes. Top domain and top generator denote the largest metadata shares within each assigned group. Distinctive characteristics report cue frequencies or comparisons with the remaining machine texts.
<table><tr><td>Prototype (n)</td><td>Top domain</td><td>Distinctive characteristics</td></tr><tr><td>H0 (147)</td><td>Yelp 93.2%</td><td>First-person expressions occur in 98.6% and review vocabulary in 85.7% of assigned texts.</td></tr><tr><td>H1 (235)</td><td>ELI5 98.7%</td><td>Links or markup occur in 14.0% of assigned texts.</td></tr><tr><td>H2 (242)</td><td>SQuAD 97.9%</td><td></td></tr><tr><td>H3 (71)</td><td>Yelp 95.8%</td><td></td></tr><tr><td>H4 (583)</td><td>WritingPrompts 44.3%</td><td>First-person expressions occur in 94.0% of assigned texts.</td></tr><tr><td>H5 (1370) H6 (510)</td><td>ROCStories 30.3%</td><td>Template markers occur in 15.0% of assigned texts.</td></tr><tr><td></td><td>TLDR 41.4%, SciGen 40.6%</td><td>Academic expressions occur in 33.5% of assigned texts.</td></tr><tr><td>H7 (442)</td><td>XSum 37.6%</td><td>First-person expressions occur in 83.7% of assigned texts.</td></tr></table>

Table 11: Complete summaries of texts assigned to the human prototypes. Top domain denotes the largest metadata share within each assigned group. Distinctive characteristics report cue frequencies or comparisons with the remaining human texts.

## D Full Results

We provide full results for all methods, including aggregate evaluation settings and detailed MAGE leave-one-out scenarios. Each table reports AvgRec, HumanRec, MachineRec, F1, Accuracy, AU-ROC, AUPR, and FPR95. MAGE-UD and MAGE-UM denote MAGE Unseen Domains and MAGE Unseen Models, respectively.

<table><tr><td>Setting</td><td>Scenario</td><td>AvgRec</td><td>HumanRec</td><td>MachineRec</td><td>F1</td><td>Acc</td><td>AUROC</td><td>AUPR</td><td>FPR95↓</td></tr><tr><td>Aggregate settings</td><td colspan="7"></td></tr><tr><td>MAGE CDCM Mixed</td><td></td><td>90.79</td><td>83.18</td><td>98.40</td><td>90.05</td><td>90.71</td><td>98.36</td><td>98.52</td><td>4.78</td></tr><tr><td>M4</td><td>Language shift</td><td>76.68</td><td>54.56</td><td>98.80</td><td>70.00</td><td>77.67</td><td>95.80</td><td>95.55</td><td>18.63</td></tr><tr><td>RAID</td><td>Adversarial/decoding</td><td>86.81</td><td>74.60</td><td>99.02</td><td>71.80</td><td>98.31</td><td>89.48</td><td>77.32</td><td>93.90</td></tr><tr><td>MAGE-UD</td><td>Average</td><td>67.99</td><td>37.04</td><td>98.94</td><td>51.14</td><td>67.55</td><td>89.30</td><td>90.79</td><td>56.25</td></tr><tr><td>MAGE-UM</td><td>Average</td><td>89.78</td><td>85.85</td><td>93.71</td><td>89.41</td><td>89.78</td><td>97.03</td><td>96.93</td><td>12.37</td></tr><tr><td colspan="2">MAGE Unseen Domains</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UD ROC</td><td></td><td>53.78</td><td>7.85</td><td>99.72</td><td>14.52</td><td>53.16</td><td>88.75</td><td>88.25</td><td>46.38</td></tr><tr><td>UD</td><td>HellaSwag</td><td>66.87</td><td>37.24</td><td>96.50</td><td>53.02</td><td>65.76</td><td>91.04</td><td>89.00</td><td>28.25</td></tr><tr><td>UD</td><td>XSum</td><td>55.86</td><td>12.85</td><td>98.86</td><td>22.55</td><td>55.67</td><td>70.42</td><td>74.85</td><td>96.22</td></tr><tr><td>UD</td><td>Yelp</td><td>67.70</td><td>35.78</td><td>99.61</td><td>52.56</td><td>67.05</td><td>91.23</td><td>93.50</td><td>69.61</td></tr><tr><td>UD</td><td>TLDR</td><td>61.80</td><td>24.54</td><td>99.06</td><td>39.12</td><td>61.10</td><td>94.20</td><td>93.10</td><td>19.25</td></tr><tr><td>UD</td><td>SciGen</td><td>69.66</td><td>40.39</td><td>98.93</td><td>57.15</td><td>67.91</td><td>93.65</td><td>94.50</td><td>24.21</td></tr><tr><td>UD</td><td>WP</td><td>75.59</td><td>51.73</td><td>99.46</td><td>67.94</td><td>75.74</td><td>87.75</td><td>92.07</td><td>91.81</td></tr><tr><td>UD</td><td>CMV</td><td>79.57</td><td>60.13</td><td>99.01</td><td>74.62</td><td>80.01</td><td>91.82</td><td>94.42</td><td>81.50</td></tr><tr><td>UD</td><td>SQuAD</td><td>67.68</td><td>35.49</td><td>99.88</td><td>52.34</td><td>67.61</td><td>89.16</td><td>92.44</td><td>83.20</td></tr><tr><td>UD</td><td>ELI5</td><td>81.43</td><td>64.45</td><td>98.40</td><td>77.62</td><td>81.53</td><td>94.94</td><td>95.79</td><td>22.10</td></tr><tr><td colspan="2">MAGE Unseen Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UM</td><td>LLaMA</td><td>91.33</td><td>85.74</td><td>96.93</td><td>90.82</td><td>91.33</td><td>98.17</td><td>97.84</td><td>7.12</td></tr><tr><td>UM</td><td>FLAN-T5</td><td>85.04</td><td>83.61</td><td>86.48</td><td>84.82</td><td>85.04</td><td>92.67</td><td>92.62</td><td>33.43</td></tr><tr><td>UM</td><td>EleutherAI</td><td>92.02</td><td>84.26</td><td>99.79</td><td>91.35</td><td>92.02</td><td>99.37</td><td>99.57</td><td>0.76</td></tr><tr><td>UM</td><td>GLM-130B</td><td>91.62</td><td>86.07</td><td>97.17</td><td>91.13</td><td>91.62</td><td>98.25</td><td>98.32</td><td>6.31</td></tr><tr><td>UM</td><td>OPT</td><td>90.88</td><td>86.50</td><td>95.26</td><td>90.46</td><td>90.88</td><td>97.60</td><td>97.31</td><td>9.67</td></tr><tr><td>UM</td><td>BigScience</td><td>89.77</td><td>82.14</td><td>97.40</td><td>88.92</td><td>89.77</td><td>97.84</td><td>97.88</td><td>8.76</td></tr><tr><td>UM</td><td>OpenAI</td><td>87.78</td><td>92.65</td><td>82.91</td><td>88.35</td><td>87.78</td><td>95.28</td><td>95.00</td><td>20.52</td></tr></table>

Table 12: Full results for Binary CE.

<table><tr><td>Setting</td><td>Scenario</td><td>AvgRec</td><td>HumanRec</td><td>MachineRec</td><td>F1</td><td>Acc</td><td>AUROC</td><td>AUPR</td><td>FPR95↓</td></tr><tr><td>Aggregate settings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">MAGE CDCM Mixed</td><td>94.77</td><td>92.98</td><td>96.56</td><td>94.71</td><td>94.75</td><td>95.06</td><td>96.79</td><td>31.27</td></tr><tr><td>M4</td><td>Language shift</td><td>83.84</td><td>72.37</td><td>95.30</td><td>81.54</td><td>84.35</td><td>89.23</td><td>92.21</td><td>72.72</td></tr><tr><td>RAID</td><td>Adversarial/decoding</td><td>77.67</td><td>58.94</td><td>96.39</td><td>42.00</td><td>95.32</td><td>78.04</td><td>45.77</td><td>88.26</td></tr><tr><td>MAGE-UD</td><td>Average</td><td>75.11</td><td>56.82</td><td>93.41</td><td>67.01</td><td>74.84</td><td>75.15</td><td>83.68</td><td>84.16</td></tr><tr><td>MAGE-UM</td><td>Average</td><td>90.92</td><td>93.49</td><td>88.34</td><td>91.40</td><td>90.92</td><td>91.89</td><td>93.90</td><td>32.59</td></tr><tr><td colspan="2">MAGE Unseen Domains</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UD</td><td>ROC</td><td>55.72</td><td>12.06</td><td>99.37</td><td>21.41</td><td>55.12</td><td>55.71</td><td>75.80</td><td>94.35</td></tr><tr><td>UD</td><td>HellaSwag</td><td>72.64</td><td>52.58</td><td>92.70</td><td>65.99</td><td>71.89</td><td>72.77</td><td>82.82</td><td>90.23</td></tr><tr><td>UD</td><td>XSum</td><td>65.99</td><td>35.58</td><td>96.40</td><td>51.14</td><td>65.86</td><td>66.00</td><td>79.24</td><td>92.52</td></tr><tr><td>UD</td><td>Yelp</td><td>50.08</td><td>37.22</td><td>62.94</td><td>43.07</td><td>49.82</td><td>49.15</td><td>51.03</td><td>95.43</td></tr><tr><td>UD</td><td>TLDR</td><td>72.57</td><td>47.14</td><td>97.99</td><td>63.24</td><td>72.09</td><td>72.68</td><td>85.15</td><td>90.73</td></tr><tr><td>UD</td><td>SciGen</td><td>84.35</td><td>72.66</td><td>96.05</td><td>82.49</td><td>83.65</td><td>84.52</td><td>91.29</td><td>82.44</td></tr><tr><td>UD</td><td>WP</td><td>92.04</td><td>85.87</td><td>98.21</td><td>91.51</td><td>92.08</td><td>92.13</td><td>95.49</td><td>65.25</td></tr><tr><td>UD</td><td>CMV</td><td>91.64</td><td>86.27</td><td>97.02</td><td>91.10</td><td>91.76</td><td>91.98</td><td>95.07</td><td>64.68</td></tr><tr><td>UD</td><td>SQuAD</td><td>79.92</td><td>60.89</td><td>98.96</td><td>75.20</td><td>79.88</td><td>79.99</td><td>89.49</td><td>87.35</td></tr><tr><td>UD</td><td>ELI5</td><td>86.17</td><td>77.92</td><td>94.43</td><td>84.90</td><td>86.22</td><td>86.53</td><td>91.35</td><td>78.62</td></tr><tr><td colspan="2">MAGE Unseen Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UM</td><td>LLaMA</td><td>93.48</td><td>93.29</td><td>93.67</td><td>93.46</td><td>93.48</td><td>94.18</td><td>95.78</td><td>30.22</td></tr><tr><td>UM</td><td>FLAN-T5</td><td>80.98</td><td>94.25</td><td>67.70</td><td>83.21</td><td>80.98</td><td>85.20</td><td>87.79</td><td>41.14</td></tr><tr><td>UM</td><td>EleutherAI</td><td>96.05</td><td>92.30</td><td>99.79</td><td>95.89</td><td>96.05</td><td>96.08</td><td>97.99</td><td>35.18</td></tr><tr><td>UM</td><td>GLM-130B</td><td>94.29</td><td>92.38</td><td>96.19</td><td>94.18</td><td>94.29</td><td>94.51</td><td>96.31</td><td>36.86</td></tr><tr><td>UM</td><td>OPT</td><td>93.26</td><td>93.58</td><td>92.94</td><td>93.28</td><td>93.26</td><td>93.82</td><td>95.35</td><td>27.57</td></tr><tr><td>UM</td><td>BigScience</td><td>93.15</td><td>92.94</td><td>93.35</td><td>93.13</td><td>93.15</td><td>93.39</td><td>95.07</td><td>33.84</td></tr><tr><td>UM</td><td>OpenAI</td><td>85.21</td><td>95.69</td><td>74.72</td><td>86.61</td><td>85.21</td><td>86.06</td><td>88.98</td><td>23.31</td></tr><tr><td>MAGE CDCM</td><td>Mixed</td><td>94.84</td><td>91.87</td><td>97.81</td><td>94.71</td><td>94.80</td><td>95.81</td><td>97.36</td><td>23.07</td></tr><tr><td>M4</td><td>Language shift</td><td>92.74</td><td>87.92</td><td>97.57</td><td>92.27</td><td>92.96</td><td>92.74</td><td>95.38</td><td>59.62</td></tr><tr><td>RAID</td><td>Adversarial/decoding</td><td>87.68</td><td>80.17</td><td>95.20</td><td>46.85</td><td>94.77</td><td>91.08</td><td>62.01</td><td>67.70</td></tr><tr><td>MAGE-UD</td><td>Average</td><td>76.72</td><td>55.75</td><td>97.69</td><td>67.09</td><td>76.42</td><td>79.49</td><td>88.39</td><td>82.17</td></tr><tr><td>MAGE-UM</td><td>Average</td><td>91.69</td><td>92.08</td><td>91.30</td><td>91.85</td><td>91.69</td><td>92.87</td><td>94.61</td><td>27.14</td></tr><tr><td>MAGE Unseen Domains</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UD</td><td>ROC</td><td>53.43</td><td>7.08</td><td>99.78</td><td>13.20</td><td>52.80</td><td>54.32</td><td>75.86</td><td>94.53</td></tr><tr><td>UD</td><td>HellaSwag</td><td>78.62</td><td>66.01</td><td>91.23</td><td>75.81</td><td>78.15</td><td>82.84</td><td>87.35</td><td>81.69</td></tr><tr><td>UD</td><td>XSum</td><td>61.16</td><td>24.03</td><td>98.28</td><td>38.23</td><td>60.99</td><td>64.78</td><td>78.90</td><td>92.83</td></tr><tr><td>UD</td><td>Yelp</td><td>77.81</td><td>56.30</td><td>99.33</td><td>71.74</td><td>77.38</td><td>80.09</td><td>89.81</td><td>87.32</td></tr><tr><td>UD</td><td>TLDR</td><td>70.70</td><td>42.17</td><td>99.22</td><td>59.01</td><td>70.16</td><td>73.52</td><td>86.22</td><td>90.49</td></tr><tr><td>UD</td><td>SciGen</td><td>80.49</td><td>63.83</td><td>97.16</td><td>76.74</td><td>79.49</td><td>83.38</td><td>90.87</td><td>83.99</td></tr><tr><td>UD</td><td>WP</td><td>88.55</td><td>78.28</td><td>98.82</td><td>87.23</td><td>88.61</td><td>91.12</td><td>95.10</td><td>70.38</td></tr><tr><td>UD</td><td>CMV</td><td>90.66</td><td>82.40</td><td>98.93</td><td>89.80</td><td>90.85</td><td>92.74</td><td>95.86</td><td>63.16</td></tr><tr><td>UD</td><td>SQuAD</td><td>78.77</td><td>58.33</td><td>99.20</td><td>73.31</td><td>78.72</td><td>82.38</td><td>90.72</td><td>85.63</td></tr><tr><td>UD</td><td>ELI5</td><td>86.99</td><td>79.02</td><td>94.96</td><td>85.84</td><td>87.04</td><td>89.74</td><td>93.26</td><td>71.69</td></tr><tr><td>MAGE Unseen Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UM</td><td>LLaMA</td><td>93.77</td><td>93.58</td><td>93.96</td><td>93.76</td><td>93.77</td><td>95.41</td><td>96.27</td><td>6.73</td></tr><tr><td>UM</td><td>FLAN-T5</td><td>86.63</td><td>90.88</td><td>82.38</td><td>87.18</td><td>86.63</td><td>87.98</td><td>90.19</td><td>41.95</td></tr><tr><td>UM</td><td>EleutherAI</td><td>95.46</td><td>90.98</td><td>99.93</td><td>95.25</td><td>95.46</td><td>96.46</td><td>98.21</td><td>28.71</td></tr><tr><td>UM</td><td>GLM-130B</td><td>93.58</td><td>90.32</td><td>96.84</td><td>93.36</td><td>93.58</td><td>94.63</td><td>96.43</td><td>38.17</td></tr><tr><td>UM</td><td>OPT</td><td>93.32</td><td>93.00</td><td>93.65</td><td>93.30</td><td>93.32</td><td>94.24</td><td>95.53</td><td>16.50</td></tr><tr><td>UM</td><td>BigScience</td><td>93.32</td><td>90.61</td><td>96.03</td><td>93.13</td><td>93.32</td><td>94.38</td><td>96.07</td><td>35.10</td></tr><tr><td>UM</td><td>OpenAI</td><td>85.77</td><td>95.21</td><td>76.33</td><td>87.00</td><td>85.77</td><td>87.01</td><td>89.55</td><td>22.82</td></tr></table>

Table 13: Full results for SupCon.

Table 14: Full results for DeTeCtive.

<table><tr><td>Setting</td><td>Scenario</td><td>AvgRec</td><td>HumanRec</td><td>MachineRec</td><td>F1</td><td>Acc</td><td>AUROC</td><td>AUPR</td><td>FPR95↓</td></tr><tr><td>Aggregate settings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAGE CDCM</td><td>Mixed</td><td>94.43</td><td>95.18</td><td>93.67</td><td>94.54</td><td>94.44</td><td>98.47</td><td>98.56</td><td>6.15</td></tr><tr><td>M4</td><td>Language shift</td><td>81.08</td><td>63.13</td><td>99.03</td><td>76.90</td><td>81.89</td><td>95.20</td><td>96.32</td><td>12.53</td></tr><tr><td>RAID</td><td>Adversarial/decoding</td><td>86.17</td><td>76.36</td><td>95.97</td><td>48.89</td><td>95.41</td><td>87.47</td><td>51.98</td><td>96.52</td></tr><tr><td>MAGE-UD</td><td>Average</td><td>79.08</td><td>63.46</td><td>94.71</td><td>72.57</td><td>78.90</td><td>92.11</td><td>92.02</td><td>33.25</td></tr><tr><td>MAGE-UM</td><td>Average</td><td>90.71</td><td>95.22</td><td>86.19</td><td>91.38</td><td>90.71</td><td>96.71</td><td>96.79</td><td>13.33</td></tr><tr><td colspan="2">MAGE Unseen Domains</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UD</td><td>ROC</td><td>59.50</td><td>20.06</td><td>98.93</td><td>33.13</td><td>58.96</td><td>87.08</td><td>86.51</td><td>45.09</td></tr><tr><td>UD</td><td>HellaSwag</td><td>83.92</td><td>80.01</td><td>87.82</td><td>83.65</td><td>83.77</td><td>92.36</td><td>90.10</td><td>23.54</td></tr><tr><td>UD</td><td>XSum</td><td>62.12</td><td>28.21</td><td>96.04</td><td>42.69</td><td>61.97</td><td>83.12</td><td>82.25</td><td>63.92</td></tr><tr><td>UD</td><td>Yelp</td><td>81.10</td><td>66.52</td><td>95.68</td><td>77.95</td><td>80.80</td><td>91.98</td><td>93.14</td><td>48.02</td></tr><tr><td>UD</td><td>TLDR</td><td>70.76</td><td>45.05</td><td>96.48</td><td>60.70</td><td>70.28</td><td>87.36</td><td>87.99</td><td>58.11</td></tr><tr><td>UD</td><td>SciGen</td><td>85.64</td><td>78.17</td><td>93.11</td><td>84.84</td><td>85.20</td><td>95.24</td><td>95.45</td><td>18.84</td></tr><tr><td>UD</td><td>WP</td><td>86.65</td><td>76.93</td><td>96.37</td><td>85.19</td><td>86.71</td><td>97.27</td><td>97.19</td><td>11.57</td></tr><tr><td>UD</td><td>CMV</td><td>89.23</td><td>84.14</td><td>94.31</td><td>88.53</td><td>89.34</td><td>96.60</td><td>96.71</td><td>16.91</td></tr><tr><td>UD</td><td>SQuAD</td><td>83.67</td><td>70.33</td><td>97.00</td><td>81.16</td><td>83.63</td><td>95.26</td><td>95.62</td><td>25.00</td></tr><tr><td>UD</td><td>ELI5</td><td>88.27</td><td>85.17</td><td>91.36</td><td>87.84</td><td>88.29</td><td>94.86</td><td>95.24</td><td>21.56</td></tr><tr><td colspan="2">MAGE Unseen Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UM</td><td>LLaMA</td><td>94.30</td><td>95.66</td><td>92.94</td><td>94.38</td><td>94.30</td><td>98.35</td><td>98.34</td><td>6.44</td></tr><tr><td>UM</td><td>FLAN-T5</td><td>79.39</td><td>94.87</td><td>63.91</td><td>82.15</td><td>79.39</td><td>91.83</td><td>91.91</td><td>36.52</td></tr><tr><td>UM</td><td>EleutherAI</td><td>97.16</td><td>95.08</td><td>99.24</td><td>97.10</td><td>97.16</td><td>99.04</td><td>99.34</td><td>0.55</td></tr><tr><td>UM</td><td>GLM-130B</td><td>94.72</td><td>95.10</td><td>94.34</td><td>94.74</td><td>94.72</td><td>98.40</td><td>98.52</td><td>5.66</td></tr><tr><td>UM</td><td>OPT</td><td>91.20</td><td>94.22</td><td>88.18</td><td>91.46</td><td>91.20</td><td>96.92</td><td>97.01</td><td>13.27</td></tr><tr><td>UM</td><td>BigScience</td><td>92.65</td><td>94.65</td><td>90.64</td><td>92.79</td><td>92.65</td><td>97.44</td><td>97.55</td><td>9.84</td></tr><tr><td>UM</td><td>OpenAI</td><td>85.54</td><td>96.99</td><td>74.09</td><td>87.02</td><td>85.54</td><td>94.96</td><td>94.85</td><td>21.02</td></tr><tr><td colspan="2">Aggregate settings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MAGE CDCM</td><td>Mixed</td><td>95.14</td><td>95.81</td><td>94.47</td><td>95.23</td><td>95.15</td><td>98.41</td><td>98.37</td><td>4.89</td></tr><tr><td>M4</td><td>Language shift</td><td>86.03</td><td>76.87</td><td>95.20</td><td>84.42</td><td>86.45</td><td>92.35</td><td>93.08</td><td>57.00</td></tr><tr><td>RAID</td><td>Adversarial/decoding</td><td>88.18</td><td>82.52</td><td>93.84</td><td>42.25</td><td>93.51</td><td>95.41</td><td>71.41</td><td>27.78</td></tr><tr><td>MAGE-UD</td><td>Average</td><td>78.59</td><td>61.46</td><td>95.72</td><td>71.13</td><td>78.34</td><td>86.90</td><td>89.10</td><td>59.41</td></tr><tr><td>MAGE-UM</td><td>Average</td><td>91.34</td><td>95.63</td><td>87.05</td><td>91.91</td><td>91.34</td><td>96.48</td><td>96.16</td><td>11.44</td></tr><tr><td colspan="2">MAGE Unseen Domains</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UD</td><td>ROC</td><td>56.68</td><td>13.92</td><td>99.44</td><td>24.33</td><td>56.10</td><td>86.11</td><td>86.41</td><td>55.26</td></tr><tr><td>UD</td><td>HellaSwag</td><td>75.84</td><td>59.51</td><td>92.18</td><td>71.37</td><td>75.23</td><td>78.73</td><td>82.41</td><td>89.89</td></tr><tr><td>UD</td><td>XSum</td><td>61.29</td><td>25.68</td><td>96.90</td><td>39.89</td><td>61.13</td><td>68.73</td><td>72.74</td><td>92.58</td></tr><tr><td>UD</td><td>Yelp</td><td>83.29</td><td>71.68</td><td>94.90</td><td>81.19</td><td>83.05</td><td>88.71</td><td>91.76</td><td>78.03</td></tr><tr><td>UD</td><td>TLDR</td><td>75.06</td><td>53.89</td><td>96.23</td><td>68.42</td><td>74.66</td><td>81.49</td><td>86.25</td><td>87.85</td></tr><tr><td>UD</td><td>SciGen</td><td>83.82</td><td>72.89</td><td>94.76</td><td>82.11</td><td>83.17</td><td>95.89</td><td>95.21</td><td>12.67</td></tr><tr><td>UD</td><td>WP</td><td>91.29</td><td>85.25</td><td>97.32</td><td>90.71</td><td>91.32</td><td>96.67</td><td>97.32</td><td>16.67</td></tr><tr><td>UD</td><td>CMV</td><td>90.30</td><td>85.06</td><td>95.54</td><td>89.67</td><td>90.42</td><td>95.56</td><td>96.41</td><td>22.37</td></tr><tr><td>UD</td><td>SQuAD</td><td>80.18</td><td>62.76</td><td>97.60</td><td>76.00</td><td>80.14</td><td>85.27</td><td>89.54</td><td>80.55</td></tr><tr><td>UD</td><td>ELI5</td><td>88.17</td><td>84.00</td><td>92.33</td><td>87.61</td><td>88.19</td><td>91.87</td><td>92.99</td><td>58.20</td></tr><tr><td colspan="2">MAGE Unseen Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>UM</td><td>LLaMA</td><td>94.20</td><td>96.01</td><td>92.40</td><td>94.31</td><td>94.20</td><td>98.36</td><td>97.77</td><td>6.45</td></tr><tr><td>UM</td><td>FLAN-T5</td><td>83.12</td><td>96.12</td><td>70.13</td><td>85.06</td><td>83.12</td><td>93.13</td><td>91.96</td><td>26.07</td></tr><tr><td>UM</td><td>EleutherAI</td><td>96.78</td><td>94.24</td><td>99.31</td><td>96.69</td><td>96.78</td><td>99.07</td><td>99.32</td><td>0.96</td></tr><tr><td>UM</td><td>GLM-130B</td><td>94.23</td><td>95.32</td><td>93.14</td><td>94.29</td><td>94.23</td><td>96.47</td><td>96.52</td><td>6.42</td></tr><tr><td>UM</td><td>OPT</td><td>93.02</td><td>94.88</td><td>91.15</td><td>93.14</td><td>93.02</td><td>96.48</td><td>96.32</td><td>9.09</td></tr><tr><td>UM</td><td>BigScience</td><td>93.00</td><td>95.95</td><td>90.05</td><td>93.20</td><td>93.00</td><td>97.68</td><td>97.71</td><td>8.47</td></tr><tr><td>UM</td><td>OpenAI</td><td>85.05</td><td>96.90</td><td>73.20</td><td>86.63</td><td>85.05</td><td>94.19</td><td>93.50</td><td>22.63</td></tr></table>

Table 15: Full results for DSVDD.

Table 16: Full results for MD-ProTector.