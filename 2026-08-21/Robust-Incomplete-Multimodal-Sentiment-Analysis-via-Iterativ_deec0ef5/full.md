# Robust Incomplete Multimodal Sentiment Analysis via Iterative Proxy Correction

1<sup>st</sup> Zhifa Geng   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
zhifageng77@gmail.com   
4<sup>th</sup> Junjie Chen   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
jorji.chen@gmail.com   
2<sup>nd</sup> Subin Huang<sup>∗†</sup>   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
subinhuang@ahpu.edu.cn   
5<sup>th</sup> Sanmin Liu   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
sanmin.liu@ahpu.edu.cn   
3<sup>rd</sup> Hao Guo   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
sifan10077@gmail.com   
6<sup>th</sup> Chao Kong<sup>†</sup>   
School of Computer and Information   
Anhui Polytechnic University   
Wuhu, China   
kongchao@ahpu.edu.cn

Abstract—Multimodal sentiment analysis aims to infer affective states by integrating language, visual, and acoustic cues. However, real-world multimodal inputs are often incomplete or corrupted, which can weaken cross-modal complementarity and introduce misleading information into downstream fusion. Existing proxy-based methods for incomplete MSA commonly rely on one-shot proxy construction to compensate for degraded language information, but the generated proxy may be coarse or unreliable at initialization. Prematurely injecting such a proxy into multimodal reasoning can propagate initial errors and compromise sentiment prediction. To address this limitation, we propose an iterative proxy correction framework for robust incomplete MSA. Our method constructs a language-oriented proxy from non-language modalities and progressively refines it under multimodal context through gated residual correction. The corrected proxy is then adaptively fused with the observed language representation according to an estimated language reliability score, allowing the model to balance proxy-based compensation and trustworthy linguistic evidence. In addition, we introduce a stage-wise latent correction objective that uses the complete language representation as a training-time semantic anchor to stabilize the proxy refinement trajectory. Extensive experiments on MOSI, MOSEI, and SIMS under diverse missingmodality settings demonstrate that the proposed framework consistently outperforms competitive baselines and achieves robust sentiment prediction under incomplete inputs.

Index Terms—Incomplete multimodal sentiment analysis, proxy correction, multimodal fusion

## I. INTRODUCTION

Multimodal sentiment analysis (MSA) aims to infer human affective states by integrating complementary cues from language, vision, and acoustics [1]. Recent MSA models exploit cross-modal interactions to achieve strong performance, but usually assume that all modalities are complete and reliable [2]–[4]. In real-world scenarios, however, multimodal observations are often incomplete or corrupted due to background noise, sensor failures, transmission instability, or privacy constraints. Under such conditions, the challenge goes beyond simple information loss: degraded modalities may still participate in cross-modal interaction and inject misleading information into multimodal fusion, thereby distorting multimodal representations and biasing sentiment prediction.

To improve robustness under incomplete inputs, existing methods for incomplete MSA mainly focus on compensating for missing or degraded modality information, and can be broadly divided into reconstruction-based and representationbased paradigms. Reconstruction-based methods [5]–[8] explicitly recover missing signals or features before prediction, while representation-based methods [9]–[11] directly learn decision-oriented latent representations without raw-signal recovery. Within the latter paradigm, language-centered methods are widely adopted because textual information usually provides explicit and task-relevant semantic cues [12], [13]. However, this strategy implicitly assumes that language representations are sufficiently reliable, which may not hold when textual inputs are missing, noisy, or semantically insufficient. This has motivated language-oriented proxy-based methods [14], [15], which construct auxiliary latent representations to compensate for degraded language information while preserving languageguided sentiment inference.

However, as illustrated in Figure 1(a), many existing proxybased methods follow a one-shot construction paradigm, where the proxy is directly injected into subsequent multimodal interaction once it is generated. This paradigm implicitly assumes that the initialized proxy is sufficiently reliable. However, under incomplete conditions, the proxy is often inferred from degraded observations and may remain coarse, noisy, or only partially informative. Prematurely involving such an imperfect proxy in cross-modal fusion can propagate and amplify its initial errors during subsequent reasoning, ultimately degrading sentiment prediction.

![](images/19f8a78f88e807297eb76137a9a9c9e2384d9241e34d74331733e57cfe6a038d.jpg)  
Fig. 1: Comparison between existing proxy-based methods and our iterative proxy correction method.

To address this issue, we propose an iterative proxy correction framework for incomplete multimodal sentiment analysis, as shown in Figure 1(b). Instead of treating the proxy as a fixed substitute, we regard it as a language-oriented auxiliary representation that should be progressively refined before downstream decision-making. Specifically, the proxy is first initialized from non-language modalities and then iteratively updated under multimodal context through gated residual correction. The corrected proxy is further adaptively fused with the observed language representation according to its estimated reliability before downstream reasoning. Moreover, we introduce a stage-wise latent correction objective that uses the complete language representation only during training as a semantic anchor, encouraging the proxy to become progressively more stable and informative throughout the refinement process. The main contributions of this work are summarized as follows:

• We identify the error-propagation risk of one-shot proxy construction in incomplete MSA and propose iterative proxy correction to improve proxy reliability before downstream fusion.

• We develop a gated residual correction mechanism to progressively refine a language-oriented proxy under multimodal context, guided by a stage-wise latent correction objective.

• Extensive experiments under diverse incomplete-modality settings demonstrate the robustness and effectiveness of our method against competitive incomplete MSA baselines.

## II. RELATED WORK

In real-world scenarios, multimodal inputs are often incomplete due to sensor failure, noise, occlusion, or privacy constraints. Existing studies on incomplete multimodal learning mainly fall into two categories: reconstruction-based methods and representation-based methods.

Reconstruction-based methods. The first line reconstructs absent signals through cross-modal generation or imputation.

Early efforts explored multimodal autoencoding and adversarial generation for missing-view completion [16], [17]. More recent MSA methods exploit semantic correlations across modalities for targeted recovery. For example, MPLMM [8] adopts prompt-based generation for lightweight recovery, TRML [6] generates semantically aligned virtual missing modalities based on CLIP, and HyPLe-MKD [5] combines hybrid prompt learning with multilevel distillation for missing modality generation and fusion.

Representation-based methods. The second line avoids explicit reconstruction and instead emphasizes robust fusion under incomplete inputs. Representative methods improve incomplete-modality learning through adaptive fusion and selfdistillation [12], [18]. Recent studies further enhance this direction with proxy modality, prototype guidance, and selfsupervised pre-training [11], [14]. For instance, MIG-HCL [9] builds uni- and multimodal interaction graphs with hybrid contrastive learning, while $\mathrm { T } ^ { \mathrm { 2 } } \mathrm { D R }$ [10] improves incomplete multimodal learning through deficiency-resistant attention, shared feature prediction, and capability-aware fusion under mixed missing scenarios.

## III. METHOD

## A. Multimodal Input

The overall pipeline for the proposed iterative proxy correction framework for incomplete MSA is illustrated in $\mathrm { F i g \mathrm { - } }$ ure 2. Given an incomplete multimodal utterance $\begin{array} { r l } { { \mathcal { X } } ^ { m } } & { { } = } \end{array}$ $\{ X _ { l } ^ { m } , X _ { v } ^ { m } , X _ { a } ^ { m } \}$ , the goal is to predict its sentiment label y. During training, the corresponding complete sample $\mathcal { X } ^ { c } =$ $\{ X _ { l } ^ { c } , X _ { v } ^ { c } , X _ { a } ^ { c } \}$ is available only for auxiliary supervision. We encode the incomplete language, visual, and acoustic modalities into a shared latent space:

$$
H _ { l } ^ { m } = E _ { l } ( X _ { l } ^ { m } ) , \quad H _ { v } ^ { m } = E _ { v } ( X _ { v } ^ { m } ) , \quad H _ { a } ^ { m } = E _ { a } ( X _ { a } ^ { m } ) ,\tag{1}
$$

where $E _ { l } ( \cdot ) , E _ { v } ( \cdot )$ , and $E _ { a } ( \cdot )$ are modality-specific encoders. After token compression, $H _ { l } ^ { m } , H _ { v } ^ { m } , H _ { a } ^ { m } \in \mathbb R ^ { N \times d }$ , where N is the number of tokens and d is the hidden dimension.

## B. Iterative Proxy Correction

Directly relying on the observed language representation $H _ { l } ^ { m }$ can be suboptimal when important semantics are corrupted or missing. We therefore construct a language proxy in latent space. To avoid trivial re-parameterization of the observed language stream, the proxy is initialized solely from non-language modalities, encouraging it to capture complementary cues unavailable from the corrupted language input.

Let $Q \in \mathbb { R } ^ { N \times d }$ denote a set of learnable query tokens. The initial proxy is generated as

$$
S ^ { ( 0 ) } = G ( [ Q ; H _ { a } ^ { m } ; H _ { v } ^ { m } ] ) ,\tag{2}
$$

where $[ \cdot ; \cdot ]$ denotes token-wise concatenation and $G ( \cdot )$ is implemented as a Transformer-based proxy generator. Since $H _ { a } ^ { m }$ and ${ \cal { H } } _ { v } ^ { m }$ are encoded from incomplete inputs, $S ^ { ( 0 ) }$ is only a coarse estimate from available non-language evidence.

![](images/77aefe1a34c834852e0b41a5ab83848fa2e9a9cc845161d3b18b83bceb58109e.jpg)  
Fig. 2: Overall architecture of the proposed method.

To improve this estimate, we refine the proxy iteratively under multimodal context. At the t-th step, we form

$$
C ^ { ( t ) } = [ S ^ { ( t ) } ; H _ { l } ^ { m } ; H _ { a } ^ { m } ; H _ { v } ^ { m } ] .\tag{3}
$$

A residual correction term and an update gate are then computed as

$$
\Delta ^ { ( t ) } = \phi _ { t } ( C ^ { ( t ) } ) , \qquad M ^ { ( t ) } = \sigma ( \psi _ { t } ( C ^ { ( t ) } ) ) ,\tag{4}
$$

where $\phi _ { t } ( \cdot )$ and $\psi _ { t } ( \cdot )$ are learnable mappings, $\sigma ( \cdot )$ is the sigmoid function, and both $\Delta ^ { ( t ) }$ and $M ^ { ( t ) }$ have the same shape as $S ^ { ( t ) }$ . The proxy is updated by

$$
\begin{array} { r } { S ^ { ( t + 1 ) } = \mathrm { L N } \Big ( S ^ { ( t ) } + M ^ { ( t ) } \odot \Delta ^ { ( t ) } \Big ) , } \end{array}\tag{5}
$$

where $\odot$ denotes element-wise multiplication and $\mathrm { L N } ( \cdot )$ denotes layer normalization. After $T$ correction steps, the final corrected proxy is

$$
S ^ { * } = S ^ { ( T ) } .\tag{6}
$$

## C. Language-Proxy Fusion

Although the corrected proxy $S ^ { * }$ provides complementary semantic evidence, the observed language representation $H _ { l } ^ { m }$ may still retain useful sentiment cues. We therefore adaptively fuse them using an estimated language reliability score:

$$
r = \sigma ( f ( \mathrm { P o o l } ( H _ { l } ^ { m } ) ) ) ,\tag{7}
$$

where $f ( \cdot )$ is a scoring network and $r \in [ 0 , 1 ]$ measures the estimated reliability of the observed language representation. The final fused language representation is computed as

$$
H _ { d } = ( 1 - r ) S ^ { * } + r H _ { l } ^ { m } .\tag{8}
$$

In this way, the model trusts $H _ { l } ^ { m }$ more when it is reliable, and otherwise shifts more weight to the corrected proxy.

## D. Sentiment Prediction

The fused language representation is integrated with the visual and acoustic features through the downstream multimodal reasoning backbone:

$$
F = \mathcal { F } ( H _ { d } , H _ { v } ^ { m } , H _ { a } ^ { m } ) ,\tag{9}
$$

where $\mathcal F ( \cdot )$ denotes the inherited fusion network. The final sentiment prediction is obtained as

$$
\begin{array} { r } { \hat { \boldsymbol y } = W _ { o } \operatorname { P o o l } ( \boldsymbol { F } ) + \boldsymbol b _ { o } . } \end{array}\tag{10}
$$

## E. Training Objective

The model is optimized with a sentiment prediction loss and a stage-wise correction loss. The prediction loss is

$$
\mathcal { L } _ { \mathrm { s p } } = \| \hat { y } - y \| _ { 2 } ^ { 2 } .\tag{11}
$$

To supervise progressive proxy correction, we use the complete language representation $H _ { l } ^ { c }$ as a semantic anchor:

$$
\mathcal { L } _ { \mathrm { c o r r } } = \sum _ { t = 0 } ^ { T } \lambda _ { t } \| S ^ { ( t ) } - H _ { l } ^ { c } \| _ { 2 } ^ { 2 } ,\tag{12}
$$

where $\textstyle \sum _ { t = 0 } ^ { T } \lambda _ { t } = 1$ and larger weights are assigned to later stages.

Following the base framework, we keep the remaining auxiliary objectives unchanged and summarize them as $\mathcal { L } _ { \mathrm { a u x } } ,$ with implementation details following [15]. The final loss is

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { s p } } + \theta \mathcal { L } _ { \mathrm { c o r r } } + \mathcal { L } _ { \mathrm { a u x } } ,\tag{13}
$$

where θ controls the strength of the correction supervision.

## IV. EXPERIMENT

## A. Benchmarks

We evaluate our method on three widely used multimodal sentiment benchmarks: MOSI, MOSEI, and SIMS, covering both English and Chinese scenarios. MOSI contains 2,199 English utterances from online movie reviews, each annotated with text, audio, vision, and a sentiment score in [−3, +3]. MOSEI is a larger English benchmark with 22,856 utterances from more diverse speakers and topics, making it more challenging for robustness evaluation. SIMS is a Chinese benchmark with 2,281 manually annotated segments collected from movie and TV dialogues, featuring richer conversational context and more subtle sentiment expressions.

## B. Evaluation Metrics and Implementation Details

We use MAE and Corr for regression evaluation. For MOSI and MOSEI, we additionally report binary classification results (Acc-2 and F1) under both negative/positive and negative/nonnegative settings; when presented as “a/b”, the two values correspond to these settings, respectively. We also report Acc-5 and Acc-7 on MOSI/MOSEI, and Acc-3 and Acc-5 on SIMS.

All models are implemented in PyTorch and optimized with Adam. The learning rate and weight decay are both set to 1 × 10<sup>−4</sup>, the batch size is 64, and the default number of training epochs is 200. Experiments are conducted on a single NVIDIA RTX 4090 GPU with 8 workers. To evaluate robustness under incomplete multimodal inputs, we vary the missing rate from 0.0 to 0.9. Each setting is run three times with seeds 1111, 1112, and 1113, and we report the average results.

## C. Baselines

We compare our method with several representative multimodal sentiment models, including Self-MM [19], CENet [20], TETFN [21], MMIM [22], MISA [13], ALMT [23], and LNLN [15]. These baselines cover diverse design paradigms, such as unimodal-multimodal joint learning, text-centered interaction, mutual-information maximization, shared-specific disentanglement, adaptive multimodal fusion, and robustness modeling under missing modalities.

## D. Robustness Comparison

Tables I and II report the results on MOSI, MOSEI, and SIMS, where the best and second-best results are highlighted in bold and underlined, respectively. Overall, our method achieves the best or highly competitive performance across all three benchmarks, demonstrating strong effectiveness and robustness. On MOSI, our method achieves the best or tiedbest results on all metrics. Compared with the strongest baseline LNLN, it consistently improves Acc-7, Acc-5, Acc-2, F1, and Corr, while matching the best MAE, showing clear advantages in both classification and regression. On MOSEI, our method obtains the best results on five out of six metrics and ranks second on Acc-7 with only a marginal gap. In particular, it achieves the best MAE and Corr, with Corr improving from 0.535 to 0.595, indicating a stronger ability to model fine-grained sentiment intensity. On SIMS, our method also delivers robust performance, achieving the best results on Acc-5, Acc-3, MAE, and Corr, while remaining competitive on Acc-2 and F1. Although it does not obtain the best F1 score, the overall results still suggest a better balance between classification accuracy and regression quality.

![](images/a2255395343b2dc24a1dcd83d5cc6feafc99a7df6ba2c790ea84478c2a6bbacf.jpg)  
Fig. 3: (a)–(c) show the F1 curves on MOSI, MOSEI, and SIMS. (d)–(f) show the MAE curves on MOSI, MOSEI, and SIMS.

Figure 3 further shows that increasing missing rates consistently deteriorate the performance of all compared methods, highlighting the difficulty of incomplete multimodal sentiment analysis. Nevertheless, our method exhibits better resistance to such degradation and maintains more stable performance across a wide range of missing conditions.

## E. Effects of Different Modalities

To assess the contribution of each modality under incomplete conditions, we conduct modality ablation experiments on MOSI and MOSEI. During training, random modality dropping is applied to simulate incomplete inputs, while at test time we evaluate three unimodal settings (T, A, and V), three bimodal settings (T+A, T+V, and A+V), and a random-missing setting (Random), with results reported in Table III. The results show that language is the most informative modality on both datasets, as text-involved settings consistently outperform A, V, and A+V, indicating that linguistic information provides the primary basis for sentiment prediction. In contrast, the relatively weak performance of non-textual settings suggests that audio and visual cues alone are insufficient for reliable sentiment understanding. The strong results of T+A and T+V further indicate that these modalities mainly serve as complementary signals when language is available. Under the random-missing setting, our model still maintains strong performance, demonstrating its robustness to stochastic modality absence.

## F. Effects of Different Components

To assess the contribution of each component, we conduct ablation studies on MOSI and SIMS. FULL denotes the complete model. w/o Proxy removes the proxy branch and directly predicts from the observed language representation. w/o Iterative disables iterative correction and uses only a one-shot proxy. w/o $\mathcal { L } _ { c o r r }$ removes the stage-wise correction loss while keeping the rest of the framework unchanged. All variants are trained and evaluated under the same settings, and the results are reported in Table IV. Overall, FULL achieves the most balanced performance, especially on SIMS, where all three ablated variants—w/o Proxy, w/o Iterative, and w/o $\mathcal { L } _ { c o r r } .$ —show clear and consistent degradation. Among them, w/o Proxy leads to the largest drop, highlighting the importance of proxy-based compensation under incomplete conditions. Both w/o Iterative and w/o $\mathcal { L } _ { c o r r }$ also underperform FULL, indicating that iterative refinement and stage-wise correction supervision are both beneficial for improving proxy quality. On MOSI, the gains are relatively smaller, but FULL still achieves the best results on the main classification metrics and remains competitive on the others. These results confirm that each proposed component contributes to the effectiveness of the overall framework.

TABLE I: Results on the MOSI and MOSEI datasets (baseline results from LNLN [15]).
<table><tr><td rowspan="2">Method</td><td colspan="6">MOSI</td><td colspan="6">MOSEI</td></tr><tr><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>MISA</td><td>29.85</td><td>33.08</td><td>71.49/70.00</td><td>71.28/70.33</td><td>1.085</td><td>0.524</td><td>40.84</td><td>39.39</td><td>71.27/75.82</td><td>63.85/68.73</td><td>0.780</td><td>0.503</td></tr><tr><td>Self-MM</td><td>29.55</td><td>34.67</td><td>70.51/69.26</td><td>66.60/67.54</td><td>1.070</td><td>0.512</td><td>44.70</td><td>45.38</td><td>73.89/77.42</td><td>68.92/72.31</td><td>0.695</td><td>0.498</td></tr><tr><td>MMIM</td><td>31.30</td><td>33.77</td><td>69.14/67.06</td><td>66.65/64.04</td><td>1.077</td><td>0.507</td><td>40.75</td><td>41.74</td><td>73.32/75.89</td><td>68.72/70.32</td><td>0.739</td><td>0.489</td></tr><tr><td>CENET</td><td>30.38</td><td>37.25</td><td>71.46/67.73</td><td>68.41/64.85</td><td>1.080</td><td>0.504</td><td>47.18</td><td>47.83</td><td>74.67/77.34</td><td>70.68/74.08</td><td>0.685</td><td>0.535</td></tr><tr><td>TETFN</td><td>30.30</td><td>34.34</td><td>69.76/67.68</td><td>65.69/63.29</td><td>1.087</td><td>0.507</td><td>40.30</td><td>47.70</td><td>69.76/67.68</td><td>65.69/63.29</td><td>1.087</td><td>0.508</td></tr><tr><td>ALMT</td><td>30.30</td><td>33.42</td><td>70.40/68.39</td><td>72.57/71.80</td><td>1.083</td><td>0.498</td><td>40.92</td><td>41.64</td><td>76.64/77.54</td><td>77.14/78.03</td><td>0.674</td><td>0.481</td></tr><tr><td>LNLN</td><td>34.26</td><td>38.27</td><td>72.55/70.94</td><td>72.73/71.25</td><td>1.046</td><td>0.527</td><td>45.42</td><td>46.17</td><td>76.30/78.19</td><td>77.77/79.95</td><td>0.692</td><td>0.530</td></tr><tr><td>Ours</td><td>34.52</td><td>38.55</td><td>73.27/71.93</td><td>73.03/71.93</td><td>1.046</td><td>0.532</td><td>47.10</td><td>47.94</td><td>78.20/78.94</td><td>78.46/80.00</td><td>0.664</td><td>0.595</td></tr></table>

TABLE II: Results on the SIMS dataset (baseline results from LNLN [15]).
<table><tr><td>Method</td><td>Acc-5</td><td>Acc-3</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>MISA</td><td>31.53</td><td>56.87</td><td>72.71</td><td>66.30</td><td>0.539</td><td>0.348</td></tr><tr><td>Self-MM</td><td>32.28</td><td>56.75</td><td>72.81</td><td>68.43</td><td>0.508</td><td>0.376</td></tr><tr><td>MMIM</td><td>31.81</td><td>52.76</td><td>69.86</td><td>66.21</td><td>0.544</td><td>0.339</td></tr><tr><td>CENET</td><td>22.29</td><td>53.17</td><td>68.13</td><td>57.90</td><td>0.589</td><td>0.107</td></tr><tr><td>TETFN</td><td>33.42</td><td>56.91</td><td>73.58</td><td>68.67</td><td>0.505</td><td>0.387</td></tr><tr><td>ALMT</td><td>20.00</td><td>45.36</td><td>69.66</td><td>72.76</td><td>0.561</td><td>0.364</td></tr><tr><td>LNLN</td><td>34.64</td><td>57.14</td><td>72.73</td><td>79.43</td><td>0.514</td><td>0.397</td></tr><tr><td>Ours</td><td>35.13</td><td>58.28</td><td>73.05</td><td>76.29</td><td>0.498</td><td>0.404</td></tr></table>

![](images/87469ff261e23ae71bc1a97fd0baafdc67da84d3a24f61adc3cf20dcb47e279d.jpg)  
Fig. 4: Case study on three MOSI samples. Our method correctly predicts the first two cases where LNLN fails, suggesting its advantage in handling incomplete multimodal cues. Both methods fail on the third case, revealing the difficulty of highly ambiguous samples.

## G. Case Study

As shown in Figure 4, we present three case studies from MOSI. In the first two cases, our method yields correct predictions whereas LNLN fails, showing that iterative proxy correction can better recover sentiment-relevant semantics from incomplete inputs by leveraging complementary multimodal cues. In the third case, both methods produce incorrect predictions, indicating that severe ambiguity or conflicting cross-modal evidence remains challenging. These examples qualitatively demonstrate the advantage of our method in handling incomplete multimodal inputs, while also revealing the difficulty of extremely hard cases.

## V. CONCLUSION

In this paper, we investigate incomplete multimodal sentiment analysis from the perspective of proxy reliability. Existing proxy-based methods typically rely on one-shot proxy construction, making downstream reasoning sensitive to initial proxy errors. To address this limitation, we propose an iterative proxy correction framework that progressively refines a language-oriented proxy under multimodal context before prediction. The proxy is initialized from non-language modalities, updated through gated residual correction, and adaptively fused with the observed language representation via an estimated reliability score. We further introduce a stagewise correction objective to regularize the refinement process. Extensive experiments verify the effectiveness and robustness of the proposed method under incomplete conditions.

TABLE III: Modality ablation results on the MOSI and MOSEI datasets.
<table><tr><td rowspan="2">Method</td><td colspan="6">MOSI</td><td colspan="6">MOSEI</td></tr><tr><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>T</td><td>45.58</td><td>51.70</td><td>84.91/82.75</td><td>84.84/82.68</td><td>0.731</td><td>0.790</td><td>52.39</td><td>53.81</td><td>85.75/84.46</td><td>85.7/84.14</td><td>0.548</td><td>0.769</td></tr><tr><td>A</td><td>22.84</td><td>23.08</td><td>58.49/57.38</td><td>61.27/59.64</td><td>1.371</td><td>0.280</td><td>41.38</td><td>41.38</td><td>63.84/71.28</td><td>51.85/59.93</td><td>0.830</td><td>0.152</td></tr><tr><td>V</td><td>22.98</td><td>24.83</td><td>58.59/56.80</td><td>51.78/53.90</td><td>1.376</td><td>0.230</td><td>42.46</td><td>42.46</td><td>65.30/71.02</td><td>61.06/58.99</td><td>0.811</td><td>0.244</td></tr><tr><td>T+A</td><td>45.53</td><td>51.31</td><td>85.16/83.19</td><td>84.97/83.07</td><td>0.739</td><td>0.790</td><td>52.29</td><td>53.68</td><td>85.77/84.65</td><td>85.73/84.20</td><td>0.549</td><td>0.764</td></tr><tr><td>T+V</td><td>45.40</td><td>51.31</td><td>84.86/82.60</td><td>84.70/82.56</td><td>0.737</td><td>0.790</td><td>52.48</td><td>53.96</td><td>86.08/84.85</td><td>85.99/84.85</td><td>0.545</td><td>0.768</td></tr><tr><td>A+V</td><td>22.40</td><td>24.30</td><td>59.25/58.31</td><td>56.45/57.04</td><td>1.350</td><td>0.198</td><td>42.52</td><td>42.52</td><td>65.38/71.22</td><td>60.40/60.12</td><td>0.812</td><td>0.244</td></tr><tr><td>Random</td><td>34.52</td><td>38.55</td><td>73.27/71.93</td><td>73.03/71.93</td><td>1.046</td><td>0.532</td><td>47.10</td><td>47.94</td><td>78.20/78.94</td><td>78.46/80.00</td><td>0.664</td><td>0.595</td></tr></table>

TABLE IV: Component ablation results on the MOSI and SIMS datasets
<table><tr><td rowspan="2">Method</td><td colspan="6">MOSI</td><td colspan="6">SIMS</td></tr><tr><td>Acc-7</td><td>Acc-5</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td><td>Acc-5</td><td>Acc-3</td><td>Acc-2</td><td>F1</td><td>MAE</td><td>Corr</td></tr><tr><td>FULL</td><td>34.52</td><td>38.55</td><td>73.27/71.93</td><td>73.03/71.93</td><td>1.046</td><td>0.532</td><td>35.13</td><td>58.28</td><td>73.05</td><td>76.29</td><td>0.498</td><td>0.404</td></tr><tr><td>w/o Proxy</td><td>34.53</td><td>38.57</td><td>72.29/71.89</td><td>72.29/71.84</td><td>1.053</td><td>0.527</td><td>30.93</td><td>54.98</td><td>70.05</td><td>62.08</td><td>0.569</td><td>0.238</td></tr><tr><td>w/o Iterative</td><td>34.27</td><td>38.32</td><td>73.19/71.84</td><td>72.37/71.69</td><td>1.051</td><td>0.536</td><td>30.98</td><td>55.23</td><td>70.58</td><td>69.74</td><td>0.565</td><td>0.241</td></tr><tr><td>wlo Lcorr</td><td>34.23</td><td>38.25</td><td>72.37/71.29</td><td>71.48/70.45</td><td>1.051</td><td>0.529</td><td>31.05</td><td>55.03</td><td>70.09</td><td>62.12</td><td>0.570</td><td>0.236</td></tr></table>

## REFERENCES

[1] W. Li, J. Chen, B. Li et al., “Tactic: Translation agents with cognitivetheoretic interactive collaboration,” arXiv preprint arXiv:2506.08403, 2025.

[2] Y. Liu, X. Zhang, B. Zhang et al., “Multimodal sentiment analysis based on label semantic guidance under social links,” Pattern Recognition, vol. 171, p. 112277, 2026.

[3] S. Afroze, M. R. Hossain, M. M. Hoque et al., “Mulmosent: Multimodal sentiment analysis for a low-resource language using textual-visual cross-attention and fusion,” Information Fusion, vol. 131, p. 104129, 2026.

[4] X. Wang, Y. Chang, L. Kou et al., “Public behavior and emotion correlation mining driven by aspect from news corpus,” IEEE Transactions on Neural Networks and Learning Systems, vol. 36, no. 5, pp. 8632–8645, 2025.

[5] Y. Zhai, Q. Yang, C. Wang et al., “Hybrid prompt learning and multilevel knowledge distillation for multimodal sentiment analysis with missing modalities,” Expert Systems with Applications, vol. 315, p. 131746, 2026.

[6] X. Zhao, S. Poria, X. Li et al., “Toward robust multimodal sentiment analysis using multimodal foundational models,” Expert Systems with Applications, vol. 276, p. 126974, 2025.

[7] X. Zheng, F. Wang, Y. Nie et al., “3d smoke scene reconstruction guided by vision priors from multimodal large language models,” arXiv preprint arXiv:2604.05687, 2026.

[8] Z. Guo, T. Jin, and Z. Zhao, “Multimodal prompt learning with missing modalities for sentiment analysis and emotion recognition,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 1726–1736.

[9] P. Gong, J. Liu, X. Zhang et al., “Towards robust sentiment analysis with multimodal interaction graph and hybrid contrastive learning,” Pattern Recognition, vol. 169, p. 111870, 2026.

[10] H. Lin, X. Tang, H. Li et al., “t<sup>2</sup>dr: A two-tier deficiency-resistant framework for incomplete multimodal learning,” in Findings of the Associationfor Computational Linguistics: ACL 2025. Vienna, Austria: Association for Computational Linguistics, Jul. 2025, pp. 8602–8616.

[11] Y. Wang, H. Jian, J. Zhuang et al., “Sslmm: Semi-supervised learning with missing modalities for multimodal sentiment analysis,” Information Fusion, vol. 120, p. 103058, 2025.

[12] M. Li, D. Yang, Y. Lei et al., “A unified self-distillation framework for multimodal sentiment analysis with uncertain missing modalities,” in Proceedings of the AAAI conference on artificial intelligence, 2024, pp. 10 074–10 082.

[13] D. Hazarika, R. Zimmermann, and S. Poria, “Misa: Modality-invariant and-specific representations for multimodal sentiment analysis,” in Proceedings ofthe 28th ACM international conference on multimedia, 2020, pp. 1122–1131.

[14] A. Zhu, M. Hu, X. Wang et al., “Proxy-driven robust multimodal sentiment analysis with incomplete data,” in Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Jul. 2025, pp. 22 123–22 138.

[15] H. Zhang, W. Wang, and T. Yu, “Towards robust multimodal sentiment analysis with incomplete data,” in The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024, pp. 55 943–55 974.

[16] L. Tran, X. Liu, J. Zhou et al., “Missing modalities imputation via cascaded residual autoencoder,” in 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 4971–4980.

[17] C. Shang, A. Palmer, J. Sun et al., “Vigan: Missing view imputation with generative adversarial networks,” in 2017 IEEE International conference on big data (Big Data), 2017, pp. 766–775.

[18] J. Zeng, J. Zhou, and T. Liu, “Robust multimodal sentiment analysis via tag encoding of uncertain missing modalities,” IEEE Transactions on Multimedia, vol. 25, pp. 6301–6314, 2022.

[19] W. Yu, H. Xu, Z. Yuan et al., “Learning modality-specific representations with self-supervised multi-task learning for multimodal sentiment analysis,” in Proceedings of the AAAI conference on artificial intelligence, 2021, pp. 10 790–10 797.

[20] D. Wang, S. Liu, Q. Wang et al., “Cross-modal enhancement network for multimodal sentiment analysis,” IEEE Transactions on Multimedia, vol. 25, pp. 4909–4921, 2022.

[21] D. Wang, X. Guo, Y. Tian et al., “Tetfn: A text enhanced transformer fusion network for multimodal sentiment analysis,” Pattern Recognition, vol. 136, p. 109259, 2023.

[22] W. Han, H. Chen, and S. Poria, “Improving multimodal fusion with hierarchical mutual information maximization for multimodal sentiment analysis,” in Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 2021, pp. 9180–9192.

[23] H. Zhang, Y. Wang, G. Yin et al., “Learning language-guided adaptive hyper-modality representation for multimodal sentiment analysis,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 2023, pp. 756–767.