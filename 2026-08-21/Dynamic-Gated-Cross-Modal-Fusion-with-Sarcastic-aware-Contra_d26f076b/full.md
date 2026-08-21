# Dynamic Gated Cross-Modal Fusion with Sarcastic-aware Contrastive Regularization for Multimodal Sarcasm Detection

1<sup>st</sup> Hao Guo School of Computer and Information Anhui Polytechnic University Wuhu, China sifan10077@gmail.com

4<sup>th</sup> Zhifa Geng   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
zhifageng77@gmail.com

School of Computer and Information Anhui Polytechnic University Wuhu, China subinhuang@ahpu.edu.cn

5<sup>th</sup> Sanmin Liu   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
sanmin.liu@ahpu.edu.cn   
3<sup>rd</sup> Junjie Chen   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
jorji.chen@gmail.com   
6<sup>th</sup> Chao Kong<sup>†</sup>   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
kongchao@ahpu.edu.cn

Abstract—Multimodal sarcasm detection aims to identify sarcastic intent from multimodal content, where inconsistencies between literal meaning and contextual cues often signal irony. This task has attracted increasing research attention. However, accurate detection remains challenging due to instance-dependent modality contributions and misleading semantic consistency, where surface-level alignment masks underlying contradictory intent. Existing methods often rely on fixed fusion strategies and treat sarcasm as generic cross-modal mismatch, limiting their ability to capture subtle sarcasm cues and instance-specific modality interactions. To address these challenges, we propose a novel MSD framework that integrates Dynamic Gated Cross-Modal Fusion with Sarcastic-aware Contrastive Regularization (SaCR). Specifically, a bidirectional gated interaction module performs cross-modal feature filtering and adaptively calibrates textual and visual contributions at the instance level. A dynamic fusion gate further balances modality importance to generate more robust multimodal representations. Furthermore, SaCR is introduced as a label-aware contrastive regularization objective that encourages semantic consistency for non-sarcastic samples while suppressing misleading consistency in sarcastic cases. The proposed framework is trained end-to-end with a multi-objective learning strategy that jointly optimizes multimodal classification and auxiliary unimodal supervision. Extensive experiments on MMSD and MMSD2.0 demonstrate that the proposed method consistently outperforms strong baselines.

Index Terms—multi-modal sarcasm detection, sarcasm detection, incongruity perception

## I. INTRODUCTION

Multimodal sarcasm detection (MSD) aims to identify sarcastic intent from paired textual and visual content [1]–[4].

![](images/adbe18ab84c65e19fb5927db3ae8ab30af33a319289e37033972010b8d930a6c.jpg)  
(a) just got over the flu now i 'm in the hospital fun

![](images/dd67c7a35b5827df8bd704c494c8c2de8642170d0113f6496f7f2ac49ef034df.jpg)  
(b) no jesus on this easter cup ? ! ? ! time to throw a fit !

Fig. 1. Illustrative examples of surface-level consistency masking underlying incongruity in multimodal sarcasm. (a) The text ”fun” and the hospital setting are semantically compatible but sentimentally contradictory. (b) The text ”no Jesus” and the cup image share a literal subject, yet convey ironic intent. Our method aims to capture such hidden incongruity beyond literal alignment.

Unlike general multimodal classification, MSD is particularly challenging because sarcastic evidence is often both instancedependent and deceptively aligned across modalities [5]–[7]. In some cases, sarcasm is mainly triggered by visual contradiction, whereas in others it relies more on textual framing or subtle cross-modal interplay. Moreover, sarcastic image-text pairs may appear semantically compatible at the literal level while conveying contradictory underlying intent, which makes naive alignment-based modeling unreliable [8], [9].

Recent MSD research has progressed from early multimodal fusion and incongruity modeling to graph-based interaction, debiasing strategies, and large vision-language model based approaches [2], [8], [10]–[13]. However, existing methods still suffer from two limitations. First, many approaches rely on fixed or insufficiently adaptive fusion schemes, which may dilute the dominant sarcastic cues in a given instance and fail to capture instance-specific modality contributions [5], [6], [14]. Second, they often treat sarcasm as generic crossmodal mismatch, without explicitly modeling a common but underexplored phenomenon in sarcasm: misleading semantic consistency that masks underlying intent contradiction [7]–[9]. As a result, models can overemphasize superficial alignment while overlooking the actual evidence of irony.

As illustrated in Figure 1, the sarcastic signal may originate from different modalities. In Figure 1(a), the textual cue “fun” contradicts the hospital scene, resulting in textdominant sarcasm. In contrast, Figure 1(b) shows a case where sarcasm emerges from visual and contextual interpretation despite surface-level semantic consistency. These examples highlight that sarcasm is both modality-dependent and often masked by superficial alignment, posing significant challenges for multimodal modeling.

To address these challenges, we propose a unified framework that integrates Dynamic Gated Cross-Modal Fusion with Sarcastic-aware Contrastive regularization (SaCR). The former performs bidirectional text-image interaction and adaptively calibrates modality contributions for each instance, while the latter imposes label-aware constraints on cross-modal similarity to distinguish genuine semantic alignment from sarcasminduced misleading consistency. The whole framework is trained end-to-end under a unified learning objective.

The main contributions of this paper are summarized as follows:

• We propose a multimodal sarcasm detection framework with dynamic gated cross-modal fusion, which adaptively balances textual and visual evidence in an instance-aware manner.

• We introduce a sarcastic-aware contrastive regularization strategy that regularizes cross-modal similarity according to sarcasm labels, encouraging genuine consistency in non-sarcastic samples while suppressing misleading consistency in sarcastic ones.

• Experiments on MMSD and MMSD2.0 verify the effectiveness of the proposed framework, and ablation studies further confirm the contribution of each component.

## II. RELATED WORK

Early sarcasm detection methods focused primarily on textbased approaches, but with the rise of social media, these methods proved insufficient. Schifanella et al. [1] were the first to combine text and visual information for sarcasm detection, followed by Cai et al. [2] who constructed the MMSD dataset and proposed a hierarchical fusion model. Later, Xu et al. [10] and Pan et al. [15] introduced models like D&R Net and Att-BERT that leveraged modality incongruity for sarcasm detection. Building on this, Liang et al. [14] and Qiao et al. [7] further enhanced modality modeling using graph networks and additional visual cues such as OCR text. These studies improve cross-modal reasoning, but most of them still rely on predefined or insufficiently adaptive fusion schemes.

Recent work has focused on improving robustness and mitigating dataset bias in MSD. Qin et al. [5] re-annotated MMSD to address false cues and class imbalance, and proposed Multiview-CLIP. Chen et al. [16] further improved the Vision-Language interaction with an interactive encoder and memory-augmented predictor. Jia et al. [12] explored debiasing via contrastive regularization, and Tang et al. [17] leveraged retrieval to prompt LVLMs(e.g., LLaVA) for MSD. Wei et al. [18] proposed $G ^ { 2 } \mathrm { S A M }$ to integrate fine-grained multimodal graphs for stronger reasoning, while Yuan et al. [19] introduced ESAM by decoupling modalities and imposing sentimental congruity constraints with outlier masking. Although these methods improve robustness and multimodal reasoning, they still largely model sarcasm as generic crossmodal mismatch and do not explicitly address two challenges central to MSD: instance-dependent modality contribution and surface-level semantic consistency masking underlying intent contradiction.

## III. METHODOLOGY

In this section, we introduce the proposed framework for multimodal sarcasm detection. An overview of the architecture is illustrated in Fig. 2. Given a multimodal input consisting of a text-image pair, our framework is organized as a unified architecture with multiple cooperative components. First, modalityspecific representations are extracted using a pre-trained vision-language encoder. Then, a dynamic gated cross-modal fusion module performs bidirectional interactions between text and image and adaptively calibrates their contributions at the instance level. To further reduce over-reliance on misleading cross-modal consistency, we incorporate a sarcastic-aware contrastive regularization objective during training, which applies label-aware constraints on cross-modal similarity. Finally, the model is trained end-to-end with a multi-objective learning strategy that jointly optimizes the multimodal classification loss, auxiliary unimodal classification losses, and the sarcasticaware contrastive regularization term.

## A. Representation Extraction

Given a multimodal input consisting of text $x ^ { t }$ and image $x ^ { v }$ , we employ a pretrained vision–language model, CLIP [20], to extract representations. The parameters of both the text and vision encoders are fine-tuned during training. Specifically, global textual and visual features are obtained from the text encoder $f ^ { t }$ and vision encoder $f ^ { v }$ of CLIP, respectively. The extracted features $\tilde { h } ^ { t }$ and $\tilde { h } ^ { v }$ are then projected into a shared feature space via learnable linear transformations:

$$
\tilde { h } ^ { t } = W _ { t } f ^ { t } ( x ^ { t } ) , \quad \tilde { h } ^ { v } = W _ { v } f ^ { v } ( x ^ { v } ) .\tag{1}
$$

## B. Dynamic Gated Cross-Modal Fusion

Given the projected textual and visual representations $\tilde { h } ^ { t }$ and $\tilde { h } ^ { v }$ , we propose a dynamic gated cross-modal fusion module to capture bidirectional cross-modal interactions and adaptively balance modality contributions across instances.

a) Bidirectional Cross-Modal Attention: Bidirectional cross-modal attention has been widely adopted to model interactions between textual and visual modalities. However, most existing approaches aggregate attended cross-modal information indiscriminately, implicitly treating stronger semantic alignment as more reliable evidence. Such an assumption may be problematic for multimodal sarcasm detection, where surface-level consistency between text and image can obscure the underlying contradictory intent. Consequently, naively aggregating cross-modal information may emphasize misleading cues rather than truly informative sarcastic signals.

![](images/03351cb70a38b7ecb4fdcda5478ff8868358467ef85bf3332e7e75d3d9c3649c.jpg)  
Fig. 2. Illustration of the proposed framework for multimodal sarcasm detection

To address this issue, we introduce a value-gated bidirectional cross-modal attention mechanism, where learnable gates are applied to the value representations before crossmodal aggregation. Given the projected textual representation $\tilde { h } ^ { t }$ and visual representation $\tilde { h } ^ { v }$ , we first compute gated value representations for each modality:

$$
\bar { h } ^ { v } = \tilde { h } ^ { v } \odot \sigma ( W _ { g } ^ { v } \tilde { h } ^ { v } + b _ { g } ^ { v } ) , \quad \bar { h } ^ { t } = \tilde { h } ^ { t } \odot \sigma ( W _ { g } ^ { t } \tilde { h } ^ { t } + b _ { g } ^ { t } ) ,\tag{2}
$$

where $W _ { g } ^ { v } , W _ { g } ^ { t } , b _ { g } ^ { v }$ , and $b _ { g } ^ { t }$ are learnable parameters, $\sigma ( \cdot )$ denotes the sigmoid activation function, and $\odot$ denotes elementwise multiplication.

The gated representations are then used as value inputs in a symmetric bidirectional attention scheme:

$$
h ^ { t  { v } } = \mathrm { A t t n } ( \tilde { h } ^ { t } , \tilde { h } ^ { v } , \bar { h } ^ { v } ) , \quad h ^ { v  t } = \mathrm { A t t n } ( \tilde { h } ^ { v } , \tilde { h } ^ { t } , \bar { h } ^ { t } ) .\tag{3}
$$

Through this design, the model explicitly captures bidirectional semantic dependencies between modalities while selectively suppressing less informative or potentially misleading cross-modal signals at the value level. This is particularly beneficial for sarcasm detection, where superficial multimodal consistency may obscure the underlying incongruity.

b) Dynamic Fusion Gate: Although value-gated attention effectively filters cross-modal information at a finegrained level, the relative importance of textual and visual modalities still varies significantly across different instances. In multimodal sarcasm detection, some samples are primarily driven by textual cues, while others rely more heavily on visual contradictions, making fixed fusion strategies suboptimal.

To address this challenge, we propose a dynamic fusion gate that adaptively balances the contributions of text-aware and image-aware representations in an instance-specific manner.

Given the text-aware visual representation $h ^ { t  v }$ and the image-aware textual representation $h ^ { v  t }$ obtained from gated bidirectional attention, we first concatenate them and compute a fusion gate α as:

$$
\begin{array} { r } { \pmb { \alpha } = \sigma ( W _ { f } [ h ^ { t  v } ; h ^ { v  t } ] + b _ { f } ) , } \end{array}\tag{4}
$$

where $W _ { f }$ and $b _ { f }$ are learnable parameters and $\sigma ( \cdot )$ denotes the sigmoid activation function. The final fused representation $h ^ { f }$ is then obtained via gated combination:

$$
h ^ { f } = \alpha \odot h ^ { t  v } + ( 1 - \alpha ) \odot h ^ { v  t } .\tag{5}
$$

This dynamic fusion mechanism allows the model to selectively emphasize the modality that provides more discriminative evidence for each individual sample.

The fused representation $h ^ { f }$ is used as the final multimodal representation and serves as the sole input to the downstream sarcasm classifier during inference.

## C. Sarcastic-aware Contrastive Regularization

While dynamic gated cross-modal fusion enables instanceadaptive integration of textual and visual cues, effective sarcasm detection further requires explicitly modeling the semantic inconsistency that characterizes sarcastic expressions. In particular, sarcastic samples often exhibit surface-level semantic consistency between modalities while conveying contradictory underlying intent, which cannot be fully captured by classification objectives alone.

To this end, we introduce a sarcastic-aware contrastive regularization strategy as an auxiliary regularization objective. Instead of uniformly encouraging cross-modal alignment, the proposed contrastive objective imposes label-aware constraints on the similarity between textual and visual representations, promoting semantic consistency for non-sarcastic samples while discouraging misleading consistency in sarcastic cases.

Formally, let $h ^ { t } = \mathrm { N o r m } ( \tilde { h } ^ { t } )$ and $h ^ { v } = \mathrm { N o r m } ( \tilde { h } ^ { v } )$ denote the $\ell _ { 2 } \cdot$ -normalized projected textual and visual representations extracted by the backbone encoder. We compute the cosine similarity $s = \cos ( h ^ { t } , h ^ { v } )$ and define a label-dependent contrastive loss as:

$$
\mathcal { L } _ { \mathrm { c o n } } = \left\{ \begin{array} { l l } { \operatorname* { m a x } ( 0 , m - s ) , } & { y = 0 , } \\ { \operatorname* { m a x } ( 0 , s + m ) , } & { y = 1 , } \end{array} \right.\tag{6}
$$

where $y \in \{ 0 , 1 \}$ denotes the sarcasm label (0 for non-sarcastic and 1 for sarcastic), and $m > 0$ is a predefined margin. This formulation encourages higher cross-modal similarity for nonsarcastic samples while reducing over-reliance on potentially misleading cross-modal consistency in sarcastic ones, thereby helping the model distinguish genuine semantic alignment from sarcasm-related deceptive compatibility.

By incorporating sarcastic-aware contrastive regularization into the training process, the model learns more discriminative cross-modal representations that are sensitive to the unique semantic contradictions of sarcasm, complementing the proposed dynamic fusion mechanism.

## D. Multi-objective Training Strategy

To effectively integrate dynamic gated cross-modal fusion and sarcastic-aware contrastive regularization, we adopt a unified multi-objective training strategy. All components are optimized jointly in an end-to-end manner.

Given a training sample with label $y ,$ the overall training objective consists of three complementary loss terms:

a) Classification Loss: The fused representation $h ^ { f } ,$ , obtained via dynamic gated cross-modal fusion, is fed into a multimodal classifier. We apply the standard cross-entropy loss CE:

$$
\mathcal { L } _ { \mathrm { m m } } = \mathrm { C E } ( \hat { y } ^ { f } , y ) ,\tag{7}
$$

where $\hat { y } ^ { f }$ denotes the predicted label distribution from the fused representation.

b) Auxiliary Unimodal Supervision: To preserve modality-specific discriminative information and stabilize multimodal learning, we introduce auxiliary classification losses on the projected unimodal representations $\tilde { h } ^ { t }$ and $\tilde { h } ^ { v }$ These unimodal classifiers share the same supervision label y but are only used during training.

$$
\mathcal { L } _ { \mathrm { u n i } } = \mathrm { C E } ( \hat { y } ^ { t } , y ) + \mathrm { C E } ( \hat { y } ^ { v } , y ) ,\tag{8}
$$

where $\hat { y } ^ { t }$ and $\hat { y } ^ { v }$ are predictions from text-only and image-only classifiers, respectively.

c) Sarcastic-aware Contrastive Regularization: Following the formulation described earlier, we apply a label-aware contrastive loss $\mathcal { L } _ { \mathrm { c o n } }$ between textual and visual representations. The loss enforces high cross-modal similarity for non-sarcastic samples and penalizes excessive similarity for sarcastic ones, serving as a semantic inconsistency regularizer.

TABLE I  
STATISTICS OF MMSD AND MMSD2.0 DATASETS
<table><tr><td>Dataset</td><td>Split</td><td></td><td>Total Sarcastic Non-sarcastic</td><td></td></tr><tr><td rowspan="3">MMSD</td><td>Train</td><td>19,816</td><td>8,642</td><td>11,174</td></tr><tr><td>Val</td><td>2,410</td><td>959</td><td>1,451</td></tr><tr><td>Test</td><td>2,409</td><td>959</td><td>1,450</td></tr><tr><td rowspan="3">MMSD2.0 Val</td><td>Train</td><td>19,816</td><td>9,576</td><td>10,240</td></tr><tr><td></td><td>2,410</td><td>1,042</td><td>1,368</td></tr><tr><td>Test</td><td>2,409</td><td>1,037</td><td>1,372</td></tr></table>

d) Overall Objective: The final training objective is defined as:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { m m } } + \mathcal { L } _ { \mathrm { u n i } } + \mathcal { L } _ { \mathrm { c o n } } .\tag{9}
$$

This multi-objective formulation enables the model to jointly learn instance-adaptive cross-modal fusion, modalityaware representations, and sarcasm-sensitive semantic constraints.

## IV. EXPERIMENTS

## A. Experiment Setting

a) Datasets: In this study, we evaluate our method on two widely used multimodal sarcasm detection benchmarks: MMSD [2] and its refined version MMSD2.0 [5]. Both datasets consist of image–text pairs annotated with instancelevel sarcasm labels, while MMSD2.0 [5] provides improved annotation quality and reduced noise. The statistics of the two datasets are summarized in Table I.

b) Baselines: We compare our model with three types of baselines: (1) Text-modality models, including TextCNN [21], SMSD [22], and BERT [23]; (2) Image-modality models, including ResNet [24] and ViT [25]; (3) Multimodal models, including DIP [26], Multi-view CLIP [5], MoBA [27], G<sup>2</sup>SAM [18], TFCD [9], DGLF [28], LLaVA+RAG [17], and ESAM [19]. We additionally report the performance of GPT-5.4 under a zero-shot setting as a representative large multimodal model for reference. For a fair comparison, all methods are evaluated using the same data splits and evaluation metrics as previous work.

## B. Main Results

As shown in Table II, our method achieves competitive or superior performance across most evaluation metrics on both MMSD and MMSD2.0. On the MMSD benchmark, our model achieves the best F1 score, surpassing the strongest baseline by a clear margin. Notably, similar performance gains are observed on MMSD2.0, which is an improved version of MMSD with refined re-annotations and reduced spurious cues, indicating that the improvements are not limited to the original dataset. Compared with recent CLIP-based and large vision–language models, our approach demonstrates superior effectiveness, highlighting the limitation of fixed cross-modal alignment strategies for sarcasm detection. We attribute these gains to the proposed instance-adaptive fusion mechanism, together with the sarcastic-aware contrastive regularization that explicitly suppresses misleading semantic consistency between modalities.

TABLE II  
PERFORMANCE OF SELECTED BASELINES AND OUR METHOD(%).
<table><tr><td rowspan="2">Modality</td><td rowspan="2">Method</td><td colspan="4">MMSD</td><td colspan="4">MMSD2.0</td></tr><tr><td>Acc.%</td><td>P%</td><td>R%</td><td>F1%</td><td>Acc.%</td><td>P%</td><td>R%</td><td>F1%</td></tr><tr><td rowspan="3">Text</td><td>TextCNN</td><td>80.03</td><td>74.29</td><td>76.39</td><td>75.32</td><td>71.61</td><td>64.62</td><td>75.22</td><td>69.52</td></tr><tr><td>SMSD</td><td>80.90</td><td>76.46</td><td>75.18</td><td>75.82</td><td>73.56</td><td>68.45</td><td>71.55</td><td>69.97</td></tr><tr><td>BERT</td><td>83.60</td><td>78.50</td><td>82.51</td><td>80.45</td><td>76.52</td><td>74.48</td><td>73.09</td><td>73.91</td></tr><tr><td rowspan="2">Image</td><td>ResNet</td><td>64.76</td><td>54.41</td><td>70.80</td><td>61.53</td><td>65.50</td><td>61.17</td><td>54.39</td><td>57.58</td></tr><tr><td>ViT</td><td>67.83</td><td>57.93</td><td>70.07</td><td>63.40</td><td>72.02</td><td>65.26</td><td>74.83</td><td>69.72</td></tr><tr><td rowspan="9">Multimodal</td><td>DIP</td><td>89.59</td><td>87.76</td><td>86.58</td><td>87.17</td><td>80.96</td><td>78.02</td><td>77.56</td><td>77.79</td></tr><tr><td>Multi-view CLIP</td><td>88.33</td><td>82.66</td><td>88.65</td><td>85.55</td><td>85.64</td><td>80.33</td><td>88.24</td><td>84.10</td></tr><tr><td>MoBA</td><td>88.96</td><td>82.84</td><td>88.12</td><td>85.40</td><td>85.83</td><td>80.42</td><td>88.67</td><td>84.34</td></tr><tr><td>G²SAM</td><td>90.48</td><td>87.95</td><td>89.02</td><td>88.48</td><td>79.43</td><td>72.04</td><td>78.07</td><td>78.07</td></tr><tr><td>TFCD</td><td>89.57</td><td>84.83</td><td>89.43</td><td>88.13</td><td>86.54</td><td>82.46</td><td>87.95</td><td>84.31</td></tr><tr><td>DGLF</td><td>89.43</td><td>85.81</td><td>89.27</td><td>87.51</td><td>86.82</td><td>81.90</td><td>89.85</td><td>85.69</td></tr><tr><td>LLaVA+RAG</td><td>89.97</td><td>89.26</td><td>89.58</td><td>89.42</td><td>86.43</td><td>87.00</td><td>86.30</td><td>86.34</td></tr><tr><td>ESAM</td><td>90.11</td><td>86.87</td><td>89.54</td><td>88.19</td><td>85.87</td><td>83.12</td><td>86.05</td><td>84.56</td></tr><tr><td>GPT-5.4 (zeroshot)</td><td>71.05</td><td>76.51</td><td>75.50</td><td>71.01</td><td>72.85</td><td>78.79</td><td>75.78</td><td>72.55</td></tr><tr><td>Ours</td><td></td><td>92.62</td><td>91.96</td><td>92.82</td><td>92.33</td><td>89.66</td><td>89.36</td><td>89.74</td><td>89.51</td></tr></table>

TABLE III  
EXPERIMENT RESULTS OF ABLATION STUDY (%).
<table><tr><td rowspan="2">Variant</td><td colspan="2">MMSD</td><td colspan="2">MMSD2.0</td></tr><tr><td>Acc.%</td><td>F1%</td><td>Acc.%</td><td>F1%</td></tr><tr><td>Full</td><td>92.62</td><td>92.33</td><td>89.66</td><td>89.36</td></tr><tr><td>w/o CMI</td><td>87.91</td><td>87.42</td><td>85.10</td><td>84.97</td></tr><tr><td>w/o BiXAtt (only v→t)</td><td>87.48</td><td>87.03</td><td>81.86</td><td>81.70</td></tr><tr><td>w/o BiXAtt (only t→v)</td><td>81.84</td><td>81.07</td><td>80.53</td><td>80.37</td></tr><tr><td>w/o VGate</td><td>92.08</td><td>91.77</td><td>88.71</td><td>88.59</td></tr><tr><td>w/o DFGate</td><td>90.35</td><td>89.80</td><td>88.34</td><td>88.28</td></tr><tr><td>w/o SaCR</td><td>91.28</td><td>91.00</td><td>88.54</td><td>88.49</td></tr><tr><td>w/o UniAux</td><td>91.31</td><td>90.92</td><td>89.16</td><td>89.03</td></tr></table>

## C. Ablation Study

We perform ablations under the same backbone, training hyper-parameters, and evaluation protocol, and only change the specified module/loss. We compare Full with the following variants: (1) w/o CMI, which removes cross-modal interaction and uses fixed fusion of unimodal features; (2) w/o BiXAtt, which removes one cross-attention direction; (3) w/o VGate, which disables value gating in cross-attention; (4) w/o DF-Gate, which replaces the dynamic fusion gate with a fixed fusion strategy; (5) w/o SaCR, which removes sarcastic-aware contrastive regularization; and (6) w/o UniAux, which removes unimodal auxiliary classification supervision. Performance is reported with the same metrics as the main results.

Table III shows that removing any component consistently degrades performance on both MMSD and MMSD2.0. In particular, removing cross-modal interaction (w/o CMI) causes a large drop in F1 (5.40% on MMSD and 4.54% on MMSD2.0), highlighting the necessity of explicit crossmodal reasoning for sarcasm-related incongruity. Disabling bidirectionality further hurts performance, and the only t→v variant yields the lowest scores (F1 81.07/80.37%), suggesting that symmetric text–image interactions are important. w/o VGate reduces F1 by 1.05/0.92%, indicating that value-level gating helps suppress misleading or redundant attended cues. Replacing the dynamic fusion gate with fixed fusion (w/o DFGate) results in a clear degradation (3.02/1.23%), validating instance-adaptive modality balancing. Removing SaCR also lowers F1 (1.82/1.02%), showing the benefit of sarcasm-aware contrastive regularization. Finally, removing UniAux also degrades performance, suggesting that auxiliary unimodal supervision provides additional training signals for representation learning. Overall, cross-modal interaction and dynamic fusion contribute the most, and the consistent trends on MMSD2.0 confirm robustness.

## D. Case Study and Visualization

Fig. 3 presents two correctly classified sarcastic samples and one failure case, together with Grad-CAM visualizations on the image encoder. In the first example, the text describes a “lovely, clean, pleasant” train ride, but the image shows obvious litter on the floor, and Grad-CAM highlights the litter region as the key evidence for sarcasm recognition. In the second example, the text expresses a positive statement (“let’s start spending on education!”), while the meme image contains an explicit ironic cue (e.g., “THANKS ... VERY ... USEFUL”), and Grad-CAM focuses on the meme text area, which supports the sarcastic prediction. This failure suggests a human-centric saliency bias: the text mentions “humans”, and the Grad-CAM map concentrates on the face region, while the model underattends other cues that better support the intended irony. These cases show that our model captures typical sarcasm cues, but it can over-rely on cross-modal alignment in some scenarios and overlook the true evidence of sarcasm.

![](images/ce3fdc387ae5982f4e6a751af9dcae38aff987dbd1624ccd4efe65011392163b.jpg)  
Fig. 3. Example of case study and visualization.

## V. CONCLUSION

In this paper, we address multimodal sarcasm detection from the perspective of two key challenges: instance-dependent modality contribution and misleading surface-level crossmodal consistency. To this end, we propose a unified framework that combines dynamic gated cross-modal fusion with sarcastic-aware contrastive regularization. The proposed model performs bidirectional text-image interaction, adaptively balances multimodal evidence for each instance, and applies label-aware regularization on cross-modal similarity during training. Experiments on MMSD and MMSD2.0 show the effectiveness of the proposed framework, while ablation and case studies further illustrate the role of each component. In future work, we will explore finer-grained visual-textual grounding and more robust sarcasm modeling strategies that are less sensitive to superficial alignment cues.

## REFERENCES

[1] R. Schifanella, P. de Juan, J. Tetreault et al., “Detecting sarcasm in multimodal social platforms,” in Proceedings of the 24th ACM International Conference on Multimedia, 2016, pp. 1136–1145.

[2] Y. Cai, H. Cai, and X. Wan, “Multi-modal sarcasm detection in twitter with hierarchical fusion model,” in Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019, pp. 2506–2515.

[3] X. Zhuang, F. Zhou, and Z. Li, “Multi-modal sarcasm detection via knowledge-aware focused graph convolutional networks,” ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 21, no. 5, 2025.

[4] F. Wang, J. Yang, J. Chen et al., “Xinsight: Integrative stage-consistent psychological counseling support agents for digital well-being,” in Proceedings of the ACM Web Conference. ACM, 2026, p. 9297–9308.

[5] L. Qin, S. Huang, Q. Chen et al., “MMSD2.0: Towards a reliable multimodal sarcasm detection system,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 10 834–10 845.

[6] Y. Tian, N. Xu, R. Zhang et al., “Dynamic routing transformer network for multimodal sarcasm detection,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 2468–2480.

[7] Y. Qiao, L. Jing, X. Song, X. Chen et al., “Mutual-enhanced incongruity learning network for multi-modal sarcasm detection,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2023, pp. 9507–9515.

[8] Y. Li, W. Zhang, Z. Lin et al., “Incorporating communication style and interaction of speakers for sarcasm explanation in dialogue,” in Proceedings of SIGIR ’25, 2025, pp. 1350–1359.

[9] Z. Zhu, X. Zhuang, Y. Zhang et al., “TFCD: Towards multi-modal sarcasm detection via training-free counterfactual debiasing,” in Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24), 2024, pp. 6687–6695.

[10] N. Xu, Z. Zeng, and W. Mao, “Reasoning with multimodal sarcastic tweets via modeling cross-modality contrast and semantic association,” in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 2020, pp. 3777–3786.

[11] B. Liang, C. Lou, X. Li et al., “Multi-modal sarcasm detection with interactive in-modal and cross-modal graphs,” in Proceedings of the 29th ACM International Conference on Multimedia, 2021, pp. 4707–4715.

[12] M. Jia, C. Xie, and L. Jing, “Debiasing multimodal sarcasm detection with contrastive learning,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 16, pp. 18 354–18 362, 2024.

[13] J. Chen, X. Liu, S. Huang et al., “Seeing sarcasm through different eyes: Analyzing multimodal sarcasm perception in large vision-language models,” IEEE Transactions on Computational Social Systems, pp. 1–18, 2025.

[14] B. Liang, C. Lou, X. Li, M. Yang, L. Gui et al., “Multi-modal sarcasm detection via cross-modal graph convolutional network,” in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2022, pp. 1767–1777.

[15] H. Pan, Z. Lin, P. Fu et al., “Modeling intra and inter-modality incongruity for multi-modal sarcasm detection,” in Findings ofthe Association for Computational Linguistics: EMNLP 2020, 2020, pp. 1383–1392.

[16] J. Chen, H. Yu, S. Huang et al., “Interclip-mep: Interactive clip and memory-enhanced predictor for multi-modal sarcasm detection,” ACM Transactions on Multimedia Computing, Communications and Applications, vol. 22, no. 2, 2026.

[17] B. Tang, B. Lin, H. Yan et al., “Leveraging generative large language models with visual instruction and demonstration retrieval for multimodal sarcasm detection,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2024, pp. 1732–1742.

[18] Y. Wei, S. Yuan, H. Zhou et al., “G2SAM: Graph-based global semantic<sup>ˆ</sup> awareness method for multimodal sarcasm detection,” Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 8, pp. 9151– 9159, 2024.

[19] S. Yuan, Y. Wei, H. Zhou et al., “Enhancing semantic awareness by sentimental constraint with automatic outlier masking for multimodal sarcasm detection,” IEEE Transactions on Multimedia, vol. 27, pp. 5376–5386, 2025.

[20] A. Radford, J. W. Kim, C. Hallacy et al., “Learning transferable visual models from natural language supervision,” in Proceedings of the International Conference on Machine Learning, 2021, pp. 8748–8763.

[21] Y. Kim, “Convolutional neural networks for sentence classification,” in Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, 2014, pp. 1746–1751.

[22] T. Xiong, P. Zhang, H. Zhu et al., “Sarcasm detection with self-matching networks and low-rank bilinear pooling,” in Proceedings of the 2019 World Wide Web Conference, 2019, pp. 2115–2124.

[23] J. Devlin, M.-W. Chang, K. Lee et al., “BERT: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of NAACL-HLT, 2019, pp. 4171–4186.

[24] K. He, X. Zhang, S. Ren et al., “Deep residual learning for image recognition,” in 2016 IEEE Conference on Computer Vision and Pattern Recognition, 2016, pp. 770–778.

[25] A. Dosovitskiy, L. Beyer, A. Kolesnikov et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020, arXiv:2010.11929 [cs.CV].

[26] C. Wen, G. Jia, and J. Yang, “DIP: Dual incongruity perceiving network for sarcasm detection,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023, pp. 2540– 2550.

[27] Y. Xie, Z. Zhu, X. Chen et al., “MoBA: Mixture of bi-directional adapter for multi-modal sarcasm detection,” in Proceedings of the 32nd ACM International Conference on Multimedia, 2024, pp. 4264–4272.

[28] Z. Zhu, K. Shen, Z. Chen et al., “DGLF: A dual graph-based learning framework for multi-modal sarcasm detection,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024, pp. 2900–2912.