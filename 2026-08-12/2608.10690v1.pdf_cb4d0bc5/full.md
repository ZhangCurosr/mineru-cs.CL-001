# Can Released LLM Vocabularies Support Token-Level Estimation of Hidden Corpora?

Qingjie Zhang<sup>1,2</sup>, Xingzhang Ren<sup>2</sup>, Zixuan Chen<sup>1</sup>, Jinfeng Li<sup>3</sup>, Yuefeng Chen<sup>3</sup>, Yitong Yang<sup>3</sup>, Hui Xue<sup>3</sup>, Dayiheng Liu<sup>2\*</sup>, and Han Qiu<sup>1\*</sup>

<sup>1</sup>Tsinghua University <sup>2</sup>Qwen Team, Alibaba Group <sup>3</sup>Alibaba Group Emails: {qj-zhang24@mails., qiuhan@}tsinghua.edu.cn <sup>\*</sup>Corresponding authors

## Abstract

Pretraining corpus composition shapes LLM capabilities, but it often remains hidden even when model weights are released. Prior work has inferred corpus mixtures or traced specific token groups from released tokenizer vocabularies; in contrast, we estimate corpus ratios for arbitrary target tokens. We first show that BPE tokenizers trained on different corpora share stable token ID–ratio distributions, motivating distribution transfer from known corpora to a target tokenizer trained on hidden corpora. We then propose Quantile-Guided Density Estimation (QGDE), which approximates this distribution with multiple quantile trends and uses local density weighting to produce token-level estimates. In controlled settings and a realistic setting using the released SmolLM tokenizer, QGDE achieves mean relative errors as low as 3.00% for token-level estimation and 3.08% after aggregation into category-level mixtures. These results suggest that released tokenizer vocabularies provide a useful signal for finegrained corpus estimation beyond coarse composition inference. The code of QGDE is available at https://github.com/qingjiesjtu/QGDE.

## 1 Introduction

The composition of pretraining corpora is essential for interpreting LLM performance, since it shapes basic capabilities such as multilingual and domainspecific performance (Xie et al., 2023; Hoffmann et al., 2022; Li et al., 2024; Petty et al., 2024). However, even models with released weights often leave the corpus opaque (Touvron et al., 2023; Bi et al., 2024; Liu et al., 2024). A more accessible signal is the tokenizer vocabulary, which is often released and commonly trained with byte-pair encoding (BPE) (Sennrich et al., 2016) on corpus statistics, as in the ChatGPT (Singh et al., 2025), Qwen (Bai et al., 2023; Yang et al., 2025), DeepSeek (Bi et al., 2024; Liu et al., 2024) families.

![](images/36d05fd4d82892412a139bfc50f9f5cb49fc2b701bad06f775b5bc3a65de9762.jpg)  
Figure 1: Overview of our work. QGDE transfers IDratio structure from known corpora to estimate token and category ratios of hidden corpora.

Prior work has shown that released BPE vocabularies can reveal corpus information<sup>1</sup>. Hayase et al. (2024) recover corpus mixture from BPE merge rules, but it remains coarse-grained rather than estimating the ratio of each token. Zhang et al. (2025) use token IDs to speculate about the prevalence of polluted Chinese tokens, but their setting targets a specific token group rather than general tokens. This motivates our central question:

Can released tokenizer vocabularies support general token-level estimation of hidden corpora?

Figure 1 illustrates our work. To estimate token ratios, we first compare BPE vocabularies trained on different corpora and find that their token ID– ratio distributions share a stable global shape. This motivates transferring this distribution from known corpora to a target tokenizer trained on hidden corpora. We then propose Quantile-Guided Density Estimation (QGDE), which fits multiple quantile trends to approximate the known ID–ratio distribution and enables token-level estimates for the target tokenizer. Our main contributions are:

• We show that token ID–ratio distributions are transferable across BPE vocabularies trained on different languages and domains, providing a usable signal for estimating hidden corpus ratios from released vocabularies.

• We propose QGDE, a general token-level estimator that fits multiple quantile trends to approximate the transferable distribution and uses local density weighting to predict corpus ratios for arbitrary target tokens.

• In both controlled settings and a realistic setting, QGDE achieves relative errors as low as 3.00% for token-level estimation and 3.08% for category-level estimation after aggregation, surpassing three compared methods.

## 2 Background

## 2.1 Related Work

Training data transparency and tokenizer signals. LLMs are trained on corpora whose exact composition is often undisclosed, making it difficult to interpret the data sources, language coverage, and domain coverage behind released models (Bommasani et al., 2023; Longpre et al., 2024; Dodge et al., 2021; Xu et al., 2024). Existing transparency methods often rely on model outputs or query access to reveal memorized examples, extract training data, or detect benchmark overlap (Carlini et al., 2021, 2022; Yang et al., 2023). In contrast, released tokenizers provide a corpus signal because they are learned from corpus statistics: frequent adjacent token pairs are merged earlier, and the resulting vocabulary and token IDs can encode distributional traces of the tokenizer-training corpus (Sennrich et al., 2016; Hayase et al., 2024; Zhang et al., 2025).

From coarse mixture to token-level estimation. Data Mixture Inference (DMI) (Hayase et al., 2024) treats released BPE merge rules as a composition signal: it compares tokenizer merge statistics with candidate corpora and estimates a small vector of category proportions. This output is inherently coarse-grained, over languages, domains, or data sources, rather than a ratio estimate for each token. PoCTrace (Zhang et al., 2025) exploits the relation between token IDs and frequency: it fits a single ID–ratio median trend to estimate the prevalence of polluted Chinese tokens. However, its use case remains tied to a specific token group and a coarse trend estimate. We instead study general tokenlevel corpus ratio estimation for arbitrary tokens.

## 2.2 Problem Formulation

Let $\mathcal { V } ^ { * } = \{ ( v _ { i } , t _ { i } ) \} _ { i = 1 } ^ { N }$ denote the released vocabulary of a target tokenizer, where $v _ { i }$ is a token string and $t _ { i }$ is its token ID. The tokenizer-training corpus $\mathcal { C } ^ { \ast }$ is hidden. Let $r _ { i } ^ { * }$ denote the corpus ratio of $v _ { i }$ in $\mathcal { C } ^ { \ast }$ . The hidden corpus therefore induces an unknown ID-to-ratio mapping $f ^ { * } ( t _ { i } ) = r _ { i } ^ { * }$

We assume access to one or more known corpora D. For each known corpus or mixture of known corpora, token ratios are observable after training and applying a BPE tokenizer (Sennrich et al., 2016; Hayase et al., 2024). These known corpora provide empirical ID–ratio pairs that can be used to approximate the unknown mapping $f ^ { * }$

Our goal is to learn an estimator $\widehat { f }$ from the known ID–ratio pairs, such that for arbitrary tokens in the released target vocabulary,

$$
{ \widehat { f } } ( t _ { i } ) = { \widehat { r } } _ { i } \approx f ^ { * } ( t _ { i } ) = r _ { i } ^ { * } .\tag{1}
$$

After estimating token-level ratios, we aggregate them over any token set $S \subseteq \mathcal { V } ^ { * }$ , e.g., ${ \hat { R } } ( S ) =$ $\textstyle \sum _ { i \in S } { \widehat { r } } _ { i }$ . When token sets are defined by languages or domains, this yields corpus mixture estimates.

## 3 Is ID-Ratio Relationship Transferable?

The formulation above relies on a basic assumption: ID–ratio pairs observed in known corpora should provide useful evidence for the hidden target corpus. Since BPE token IDs reflect merge order and thus corpus statistics (Sennrich et al., 2016; Hayase et al., 2024; Zhang et al., 2025), the key question is whether the ID–ratio relationship is transferable across tokenizers trained on different corpora.

We compare ID–ratio distributions under two controlled settings: same domain but different languages, and same language but different domains. Specifically, we train BPE tokenizers on English, French, Japanese, and Chinese slices from mC4 (Xue et al., 2021) for the language setting, and on FineWeb (Web) (Penedo et al., 2024), Wikipedia (Wiki) (Wikimedia Foundation, 2023), CodeParrot (Code) (Hugging Face, 2021), and OpenWeb-Math (Math) (Paster et al., 2024) for the domain setting. For each tokenizer, we remove the initial vocabulary tokens, since special tokens and alphabet symbols do not reflect BPE merge order. We then compute each token’s ratio as its count divided by the total token count.

We plot the token ID–ratio distributions in log– log space in Figure 2. The scatter plots show that the distributions are visually similar, even for English and Japanese, and for Code and Wiki, which differ substantially in language or domain structure (see Figure 7 for all eight corpora). This suggests a transferable ID–ratio distribution.

![](images/5abf89993c5217ef71263dbc90032b49f1a62eb86f60825d5173404a626dbf24.jpg)  
Figure 2: Token ID–ratio distributions are transferable across different corpora. Left: ID–ratio scatter plots in log–log space. Middle: Similarity across different languages. Right: Similarity across different domain.

To quantify this similarity, we convert each scatter plot into a two-dimensional probability density over (log ID, log ratio) (see Appendix B) and define a directional transfer similarity score. Let $P _ { S }$ and $P _ { T }$ denote the ID–ratio densities of a source tokenizer S and a target tokenizer T. We compute

$$
\mathrm { S i m } ( S  T ) = \exp ( - \frac { D _ { \mathrm { K L } } ( P _ { T } \| P _ { S } ) } { H ( P _ { T } ) } ) ,\tag{2}
$$

where $H ( P _ { T } )$ is the entropy of the target density and $D _ { \mathrm { K L } }$ denotes Kullback–Leibler divergence (Murphy, 2022). This score normalizes the transfer divergence by the target entropy and maps it to (0, 1], where larger values indicate that the source distribution better explains the target distribution.

The heatmap in Figure 2 shows that token ID– ratio distributions are broadly transferable. In the language setting, transfer is strongest between languages with more similar linguistic structures: English and French are nearly interchangeable, and Japanese and Chinese are also highly similar. Transfer across these groups is weaker, but still substantial, with similarities around 0.8 in the harder English/French-to-Japanese/Chinese directions. In the domain setting, all pairs remain highly similar, with the largest gap appearing between Code and Wiki, mirroring the larger structural gap between programming language and encyclopedic text. Additional single- and mixed-source transfer profiles are reported in Appendix C. Overall, the ID–ratio relationship has a stable global shape, motivating an estimator that exploits the shared global ID–ratio trend across corpora.

## 4 Quantile-Guided Density Estimation

Since token ID–ratio distributions share a global shape, we estimate token ratios by transferring this distributional structure from known corpora with observed token ratios. Such transfer is inevitably imperfect: the key is to preserve the shared ID– ratio distribution.

Inspired by prior work (Zhang et al., 2025) that uses quantile regression to obtain ratio ranges from a single median or boundary curve, we fit multiple quantile trends to better approximate the shared ID–ratio distribution. We then use local density weighting to convert these quantile trends into a token point estimate, as illustrated in Figure 3.

## 4.1 Fitting ID–Ratio Trends with Quantiles

Following Zipf’s law (Piantadosi, 2014; Saichev et al., 2009), which states that word frequency is approximately inversely proportional to frequency rank, a log–log transformation turns this inverse relationship into an approximately linear trend. We therefore model token ID and corpus ratio in log– log space, where quantile regression can fit linear trends at different quantile levels.

Given a tokenizer trained on known corpora with observed token counts, we represent each token j as $( x _ { j } , y _ { j } ) = ( \log t _ { j } , \log r _ { j } )$ , where $t _ { j }$ is the token ID and $r _ { j }$ is the token’s corpus ratio. A single median curve, corresponding to the 0.5 quantile, can capture the central tendency of this relation. However, it reduces tokens with similar IDs to a single typical ratio, discarding their local ratio variation. This local variation is exactly the distributional signal neededfor token-level estimation.

We instead model the shared ID–ratio relationship with a family of quantile trends. Let $\tau$ denote a set of candidate quantile levels. For each $\tau \in \mathcal { T } .$ we fit a log-linear quantile curve over the known

![](images/764893cb67fbed235e7f214cd27d12c04219eabf2bb559c1f623de0aace61a22.jpg)  
Figure 3: Overview of quantile-guided density estimation. Left: the ID–ratio distribution in log–log space. Middle: multiple quantile trends approximate the global ID–ratio distribution and produce candidate estimates at a target token ID. Right: Gaussian density kernels around each candidate assign local weights to the estimates.

ID–ratio points using standard quantile regression (Regression, 2017), $q _ { \tau } ( x ) = a _ { \tau } + b _ { \tau } x$ (see details in Appendix A). For a target token with ID $t _ { i } ,$ the τ-th trend gives a candidate log-ratio estimate:

$$
z _ { i , \tau } = q _ { \tau } ( \log t _ { i } ) = a _ { \tau } + b _ { \tau } \log t _ { i } .\tag{3}
$$

The quantile family generalizes single-curve token-ID estimators: lower quantiles describe conservative low-ratio hypotheses, upper quantiles describe high-ratio hypotheses, and intermediate quantiles describe the dense central region. Thus, at each token ID, a dense quantile family defines multiple plausible candidate estimates. The remaining question is which quantile levels best approximate the shared ID–ratio distribution.

## 4.2 Selecting Quantile Anchors

To avoid redundant or poorly supported trends, we select a small set of representative quantile anchors $\mathcal { T } _ { K } ^ { \star } = \left\{ \tau _ { 1 } , \ldots \ldots , \tau _ { K } \right\}$ . These anchors should cover the global ID–ratio distribution without collapsing onto redundant regions or passing through consistently sparse regions. We therefore select anchors by directly maximizing their coverage over the known ID–ratio points $\mathcal { P } = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n }$

For a candidate anchor set T<sub>K</sub>, a point is covered if it falls within a vertical band of width $h _ { y }$ around at least one selected quantile trend. We define Quantile Anchor Coverage C as

$$
C ( \mathcal T _ { K } ) = \sum _ { ( x _ { j } , y _ { j } ) \in \mathcal P } \mathbf { 1 } \Big [ \operatorname* { m i n } _ { \tau \in \mathcal T _ { K } } | y _ { j } - q _ { \tau } ( x _ { j } ) | < h _ { y } \Big ] .\tag{4}
$$

This score counts each known ID–ratio point at most once, even if it is close to multiple selected trends, so redundant anchors receive little extra benefit. We choose the anchor set by maximizing this Quantile Anchor Coverage:

$$
\mathcal { T } _ { K } ^ { \star } = \arg \operatorname* { m a x } _ { \mathcal { T } _ { K } \subset \mathcal { T } , | \mathcal { T } _ { K } | = K } C ( \mathcal { T } _ { K } ) .\tag{5}
$$

In implementation, we perform a grid search over quantile levels. Because the maximization depends on both the selected quantile set and the number of anchors K, Section 5 analyzes which anchors are selected and how many anchors are needed.

## 4.3 Local Density Weighting

After selecting quantile anchors, each target token has multiple candidate estimates $\{ z _ { i , \tau } \} _ { \tau \in \mathcal { T } _ { K } ^ { \star } }$ These candidates are plausible under the global ID– ratio trends, but the global trends do not determine how much each candidate should contribute to a particular token. Therefore, we assign them soft weights by measuring how much local density support each candidate receives from nearby known ID–ratio points.

For target token $t _ { i } .$ , we collect nearby known ID–ratio points $\mathcal { N } _ { i } = \{ ( x _ { j } , y _ { j } ) ~ | ~ | x _ { j } - \log t _ { i } | ~ <$ $h _ { x } \}$ . Within this local ID window, we compute the unnormalized local support using Gaussian kernel (Silverman, 2018):

$$
W _ { i , \tau } = \sum _ { ( x _ { j } , y _ { j } ) \in \mathcal { N } _ { i } } \exp \left( - \frac { ( y _ { j } - z _ { i , \tau } ) ^ { 2 } } { 2 h _ { y } ^ { 2 } } \right) .\tag{6}
$$

This kernel is a soft version of the vertical neighborhood used in anchor coverage, centered at the candidate estimates $z _ { i , \tau }$ . This weighting matches the intuition in Figure 3: at a fixed token ID, Gaussian density kernels compare how strongly the surrounding points support each quantile candidate.

By normalizing the weights across all quantile trends, the estimated ratio of token $t _ { i }$ is the densityweighted average of the candidate estimates:

![](images/c6087888848da9a48b4ddcd1c6da672f6daf7ea72d83d97d7eac34df47b21238.jpg)  
Figure 4: Pairwise projection of Quantile Anchor Coverage for selecting three quantile anchors. The star marks the selected triplet (0.50, 0.70, 0.90).

$$
\widehat { y } _ { i } = \sum _ { \tau \in \mathcal { T } _ { K } ^ { \star } } \frac { W _ { i , \tau } } { \sum _ { \tau ^ { \prime } \in \mathcal { T } _ { K } ^ { \star } } W _ { i , \tau ^ { \prime } } } z _ { i , \tau } .\tag{7}
$$

Thus, the global quantile trends provide candidate estimates, and local density weighting combines them using token-specific neighborhood evidence. As a result, it turns the range-style signal used in prior token-ID frequency estimation (Zhang et al., 2025) into a fine-grained token-level point estimate.

## 5 Quantile Anchor Configuration

In Equation 5, we select quantile anchors to cover the known ID–ratio distribution while avoiding redundant or poorly supported trends. Before evaluating token-level estimates, we first examine how this Quantile Anchor Coverage (QAC) objective configures the anchors in practice. The following subsections address two practical questions.

## 5.1 Which Anchors Cover Better?

We first inspect the fixed-K selection problem. Taking $K \ = \ 3$ as an example, we grid search over candidate triplets using the QAC objective in Equation 5. Since the triplet search space is three-dimensional, Figure 4 visualizes a pairwise projection: each cell fixes two anchors $( \tau _ { a } , \tau _ { b } )$ and reports the best coverage obtained by any triplet containing that pair.

Figure 4 shows that coverage is substantially lower when anchors concentrate in sparse or redundant regions. By contrast, the high-coverage area lies around middle-to-high quantile levels. In this K = 3 configuration, the triplet (0.50, 0.70, 0.90) achieves the highest coverage, covering 53.1% of the sampled known ID–ratio points under the chosen vertical bandwidth. This indicates that QAC does not simply spread anchors uniformly across quantile levels; instead, it favors anchors whose trends jointly pass through well-supported regions of the known ID–ratio distribution.

![](images/4d757e0ae81894f7a69e57156b97f371f47ef790d8b19a5800366820c9adbde3.jpg)  
Figure 5: Effect of the number of quantile anchors on Quantile Anchor Coverage. Coverage increases with K, but marginal gains quickly saturate.

## 5.2 How Many Anchors Are Enough?

We then study how the number of quantile anchors affects coverage. For each anchor number K, we maximize $C ( \mathcal T _ { K } )$ and report both the best coverage and the marginal gain over $K - 1$ anchors in Figure 5. Coverage increases as more anchors are added, but the marginal gain decreases rapidly and becomes nearly zero at $K = 1 4$ . This saturation suggests that additional anchors eventually pass through regions that are already covered by the selected trends, rather than adding substantial new support from the known ID–ratio distribution.

Overall, QAC favors complementary anchors rather than sparse or redundant ones. Its coverage then saturates as K grows, indicating diminishing returns from adding more anchors.

## 6 Token-Level Ratio Estimation

We now test whether QGDE turns the quantile trends into accurate token-level ratio estimates.

## 6.1 Evaluation Setting

The evaluation is under different source compositions. Here, source refers to the known corpus mixture used to fit the estimator, while the target corpora are held fixed as uniform mixtures and used only for evaluation.

We use the same language and domain categories as in Section 3, but evaluate on target corpora drawn from different datasets. For languages, mC4 serves as the source side and OSCAR (Suarez et al., 2020) as the target side over English, French, Japanese, and Chinese. For domains, Web, Wiki, Code, and Math are paired with target corpora from the same broad domains: RedPajama-C4 (Weber et al., 2024), BookCorpus (Zhu et al., 2015), RedPajama-GitHub (Weber et al., 2024), and FineWebMath (Allal et al., 2025).

<table><tr><td rowspan="2">Source</td><td colspan="5">Single-source</td><td colspan="5">70%-mixed source</td><td>Target-like</td></tr><tr><td> $S _ { 1 } \mathrm { \cdot }$  -only</td><td> $S _ { 2 }$  -only</td><td> $S _ { 3 } .$  only</td><td> $S _ { 4 } .$  -only</td><td> $\mathbf { A v g . } { \pm } \mathbf { S t d . }$ </td><td> $S _ { 1 }$  -major  $S _ { 2 } .$ </td><td>major  $S _ { 3 } .$ </td><td>-major</td><td> $S _ { 4 } .$ </td><td>-major Avg.±Std.</td><td>Uniform</td></tr><tr><td colspan="10">Language Sources  $( S _ { 1 } = \mathrm { E n } , S _ { 2 } = \mathrm { F r } , S _ { 3 } = \mathrm { Z h } , S _ { 4 } = \mathrm { J a } )$ </td><td></td></tr><tr><td>Transfer</td><td>21.43</td><td>21.40</td><td>31.25</td><td>37.71</td><td> $2 7 . 9 5 { \pm } 6 . 9 2$ </td><td>22.19</td><td>19.58</td><td>33.53</td><td>22.33</td><td>24.41±5.38</td><td>21.72</td></tr><tr><td>PoCTrace</td><td>24.94</td><td>25.01</td><td>10.33</td><td>9.40</td><td>17.42±7.56</td><td>18.82</td><td>14.21</td><td>8.27</td><td>9.75</td><td>12.76±4.12</td><td>10.45</td></tr><tr><td>QGDE Avg.</td><td>5.80</td><td>5.84</td><td>6.33</td><td>5.21</td><td>5.80±0.40</td><td>5.93</td><td>3.91</td><td>4.65</td><td>4.64</td><td>4.78±0.73</td><td>4.68</td></tr><tr><td>K = 3</td><td>14.11</td><td>18.05</td><td>9.17</td><td>10.95</td><td> $1 3 . 0 7 \pm 3 . 3 8$ </td><td>17.54</td><td>10.02</td><td>7.29</td><td>10.78</td><td>11.41±3.77</td><td>11.42</td></tr><tr><td>K =</td><td>14.19</td><td>11.95</td><td>7.98</td><td>7.24</td><td>10.34±2.85</td><td>9.65</td><td>5.79</td><td>6.57</td><td>6.76</td><td>7.19±1.46</td><td>7.29</td></tr><tr><td>K =</td><td>7.46</td><td>7.91</td><td>7.99</td><td>6.08</td><td> $7 . 3 6 { \pm } 0 . 7 6$ </td><td>7.55</td><td>3.47</td><td>5.90</td><td>5.68</td><td>5.65±1.45</td><td>5.75</td></tr><tr><td>456 K =</td><td>4.48</td><td>4.44</td><td>7.08</td><td>4.65</td><td>5.16±1.11</td><td>8.36</td><td>4.61</td><td>4.55</td><td>4.18</td><td>5.43±1.70</td><td>3.94</td></tr><tr><td>K = 78</td><td>3.46</td><td>3.44</td><td>6.27</td><td>5.03</td><td>4.55±1.19</td><td>4.24</td><td>3.26</td><td>4.90</td><td>4.47</td><td>4.22±0.60</td><td>4.44</td></tr><tr><td>K =</td><td>4.69</td><td>4.20</td><td>5.49</td><td>4.95</td><td>4.83±0.46</td><td>3.83</td><td>2.92</td><td>4.03</td><td>3.40</td><td>3.54±0.43</td><td>3.45</td></tr><tr><td>K = 9</td><td>3.95</td><td>3.76</td><td>5.63</td><td>4.03</td><td>4.34±0.75</td><td>3.03</td><td>2.79</td><td>3.89</td><td>3.50</td><td>3.31±0.42</td><td>3.39</td></tr><tr><td>K = 10</td><td>3.85</td><td>3.14</td><td>4.84</td><td>4.07</td><td>3.98±0.61</td><td>3.53</td><td>2.79</td><td>3.39</td><td>3.52</td><td>3.31±0.31</td><td>3.51</td></tr><tr><td>K 二 11</td><td>3.20 2.91</td><td>3.17</td><td>5.10</td><td>3.54</td><td>3.75±0.79</td><td>3.57</td><td>2.82</td><td>3.51</td><td>3.09</td><td>3.25±0.31</td><td>3.00</td></tr><tr><td>K = 12</td><td>3.54</td><td>2.94</td><td>5.29</td><td>3.76 3.98</td><td>3.72±0.97</td><td>3.09</td><td>2.77</td><td>3.71</td><td>3.22</td><td>3.20±0.34</td><td>3.09</td></tr><tr><td>K = 13</td><td></td><td>3.43</td><td>5.49</td><td></td><td>4.11±0.82</td><td>3.26</td><td>2.76</td><td>3.90</td><td>3.44</td><td>3.34±0.41</td><td>3.28</td></tr><tr><td>K = 14</td><td>3.78</td><td>3.66</td><td>5.64</td><td>4.23</td><td>4.33±0.79</td><td>3.50</td><td>2.93</td><td>4.09</td><td>3.67</td><td>3.55±0.42</td><td>3.55</td></tr><tr><td colspan="10">=Math,</td><td></td></tr><tr><td>Transfer</td><td>13.76</td><td>13.75</td><td>53.02</td><td>12.78</td><td> $2 3 . 3 3 \pm 1 7 . 1 5$ </td><td>10.42</td><td>17.59</td><td>20.96</td><td>9.97</td><td>14.73±4.70</td><td>14.19</td></tr><tr><td>PoCTrace</td><td>30.41</td><td>27.50</td><td>37.72</td><td>21.13</td><td> $2 9 . 1 9 { \pm } 5 . 9 6 $ </td><td>27.51</td><td>26.34</td><td>25.22</td><td>22.20</td><td>25.32±1.97</td><td>24.32</td></tr><tr><td>QGDE Avg.</td><td>11.45</td><td>8.06</td><td>16.77</td><td>24.28</td><td> ${ \bf 1 5 . 1 4 \pm 6 . 1 2 }$ </td><td>7.34</td><td>6.31</td><td>8.71</td><td>11.28</td><td>8.41±1.86</td><td>7.44</td></tr><tr><td>K = 3</td><td>29.08</td><td>18.66</td><td>28.44</td><td>31.44</td><td> $2 6 . 9 1 \pm 4 . 8 9$ </td><td>22.21</td><td>17.38</td><td>24.22</td><td>23.79</td><td>21.90±2.71</td><td>22.24</td></tr><tr><td>K = 4</td><td>25.12</td><td>17.29</td><td>27.43</td><td>32.78</td><td> $2 5 . 6 6 \pm 5 . 5 7$ </td><td>19.59</td><td>10.85</td><td>21.70</td><td>22.15</td><td>18.57±4.56</td><td>11.89</td></tr><tr><td>K = 5</td><td>17.87</td><td>10.45</td><td>26.06</td><td>31.42</td><td> $2 1 . 4 5 { \pm } 7 . 9 7$ </td><td>4.66</td><td>4.89</td><td>14.52</td><td>17.99</td><td>10.52±5.87</td><td>12.76</td></tr><tr><td>K = 6</td><td>18.60</td><td>11.22</td><td>23.76</td><td>30.94</td><td> $2 1 . 1 3 { \pm } 7 . 2 1$ </td><td>5.91</td><td>4.85</td><td>7.64</td><td>15.04</td><td>8.36±3.98</td><td>5.42</td></tr><tr><td>K=8</td><td>13.19</td><td>7.03</td><td>20.64</td><td>31.67</td><td> $1 8 . 1 3 { \pm } 9 . 1 8 $ </td><td>4.47</td><td>4.48</td><td>4.73</td><td>10.96</td><td>6.16±2.78</td><td>4.44</td></tr><tr><td></td><td>5.33</td><td>4.52</td><td>20.43</td><td>31.81</td><td> $1 5 . 5 2 { \pm } 1 1 . 3 4 $ </td><td>4.52</td><td>4.69</td><td>4.67</td><td>11.45</td><td>6.33±2.96</td><td>4.44</td></tr><tr><td>K = 9</td><td>5.15</td><td>4.50</td><td>16.77</td><td>31.28</td><td> $1 4 . 4 3 \pm 1 0 . 8 9$ </td><td>4.48</td><td>4.89</td><td>4.47</td><td>11.02</td><td>6.21±2.78</td><td>4.52</td></tr><tr><td>K = 10</td><td>4.49</td><td>4.66</td><td>8.63</td><td>30.05</td><td> $1 1 . 9 6 { \scriptstyle \pm 1 0 . 5 8 }$ </td><td>4.45</td><td>4.84</td><td>4.45</td><td>5.00</td><td>4.69±0.24</td><td>4.76</td></tr><tr><td>K = 11</td><td>4.70</td><td>4.59</td><td>8.81</td><td>10.63</td><td> $7 . 1 8 { \pm } 2 . 6 2$ </td><td>4.49</td><td>5.01</td><td>4.58</td><td>4.49</td><td>4.64±0.22</td><td>4.99</td></tr><tr><td>K = 12</td><td>4.54</td><td>4.67</td><td>7.32</td><td>10.53</td><td> $6 . 7 6 { \pm } 2 . 4 4$ </td><td>4.45</td><td>4.71</td><td>4.46</td><td>4.45</td><td>4.52±0.11</td><td>4.67</td></tr><tr><td></td><td>4.63</td><td>4.57</td><td>7.61</td><td>9.54</td><td> $6 . 5 9 { \pm } 2 . 1 0 $ </td><td>4.45</td><td>4.60</td><td>4.51</td><td>4.51</td><td>4.52±0.05</td><td>4.60</td></tr><tr><td>K = 13 K = 14</td><td>4.65</td><td>4.52</td><td>5.30</td><td>9.27</td><td> $5 . 9 4 \pm 1 . 9 5$ </td><td>4.46</td><td>4.56</td><td>4.53</td><td>4.51</td><td>4.52±0.04</td><td>4.54</td></tr></table>

Table 1: Mean relative error (MRE) (%) of token-level ratio estimation across language and domain source mixtures. Single-source uses one known corpus; 70%-mixed uses a 70%-dominant known-corpus mixture, with the remaining three sources mixed equally; Target-like uses the same uniform mixture as the target corpora. Darker green indicates higher error in the QGDE K-ablation rows.

We report mean relative error in Table 1, where lower values indicate better estimates. We compare QGDE with two baselines: direct ID-ratio transfer, which copies the source ratio profile by token position, and PoCTrace (Zhang et al., 2025), which estimates ratios from a single median ID– ratio trend.

## 6.2 Token-Level Results

Table 1 yields four main observations.

QGDE outperforms baselines. QGDE consistently improves token-level estimation over both baselines. Direct ID-ratio transfer copies the source ratio profile and therefore remains higherror, showing that distributional transferability does not justify token-wise ratio copying. PoC-Trace avoids direct copying by fitting a median ID–ratio trend, but a single trend cannot represent the vertical spread of plausible ratios at each token ID. In contrast, QGDE combines multiple quantile trends with local density weighting. The later Krows are substantially lower than both baselines in most source-mixture settings, especially for mixed known-corpus settings.

More quantile anchors help, then saturate. The ablation on K confirms that multiple quantile anchors are necessary, but that their benefit saturates. Moving from K = 3 to larger K sharply reduces error, especially in the domain block: the single-source average decreases from 26.91 to 5.94, and the 70%-mixed average decreases from 21.90 to 4.52. Later rows fluctuate within a narrower band, so the best K is not universal across source compositions; nevertheless, K = 14 provides a reasonable high-coverage default once the QAC gain has saturated. This echoes the QAC analysis in Section 5: once additional anchors provide little new coverage of the known ID–ratio distribution, they also yield diminishing improvements in token-level estimation error.

Mixed sources help, but exact ratios matter less. Mixed sources are generally preferable to singlesource settings because they expose the estimator to a broader ID–ratio distribution. This is most visible in the domain block, where the QGDE average drops from 15.14 for single-source settings to 8.41 for 70%-mixed settings. Within the mixed-source group, however, the exact dominant component matters much less: at K = 14, the 70%-mixed columns have low standard deviation, with Std. of 0.42 for languages and 0.04 for domains. Even matching the target mixture exactly (Target-like at the last column) is not always optimal; what matters more is using a mixed source that covers multiple source components.

Language ratios are easier to predict than domain ratios. The language block reaches low error with fewer anchors, whereas the domain block has much larger errors at small K and only approaches a similar range after more anchors are used. This difference reflects how closely the source and target ID–ratio relationships match. In the language setting, mC4 and OSCAR differ as corpora, but the language-specific signals that shape token IDs are largely stable across them. In the domain setting, the matched source and target corpora are less aligned. Source and target corpora may differ in collection pipelines, so the same broad domain label does not guarantee a similar ID–ratio relationship.

## 7 Aggregating Token Ratios into Mixtures

A useful token-level estimator should also support corpus mixture estimation. We test this by aggregating QGDE’s token-level estimates into language or domain proportions.

## 7.1 Aggregation Procedure

We convert target token ratios into category proportions by distributing each estimated token ratio $\widehat { r _ { i } }$ according to how the token appears across known source categories. For example, a token that appears mostly in one category contributes most of its estimated ratio to that category, while a token that appears across all categories is split according to its relative counts in the known corpora. Formally, we define the category assignment weight as

$$
\pi _ { c , i } = { \frac { n _ { c , i } } { \sum _ { c ^ { \prime } \in { \mathcal { C } } } n _ { c ^ { \prime } , i } } } , \qquad c \in { \mathcal { C } } .\tag{8}
$$

Here, C is the set of source categories, and $n _ { c , i }$ is the count of target token $v _ { i }$ in the known corpus for category c.

We then normalize the estimated token ratios over token set $\mathcal { T }$ and distribute each token ratio to categories using $\pi _ { c , i }$ . The estimated mixture proportion of category c is therefore

$$
\widehat { \alpha } _ { c } = \sum _ { i \in \mathbb { Z } } \frac { \widehat { r } _ { i } } { \sum _ { j \in \mathbb { Z } } \widehat { r } _ { j } } \pi _ { c , i } , \qquad c \in \mathcal { C } .\tag{9}
$$

## 7.2 Category-Level Results

Figure 6 evaluates category-level mixture estimation under uniform language and domain targets. We compare QGDE aggregation with DMI (Hayase et al., 2024), a source-independent baseline that specifically designed for estimating macro mixture proportions from tokenizer statistics.

Token-level estimates support mixture estimation. The figure shows that QGDE token-level ratios can be aggregated into meaningful categorylevel proportions. Across both language and domain targets, the QGDE pies track the ground-truth mixtures closely than the baseline estimates. This indicates that the token-level signal recovered by QGDE is not only useful for individual token prediction, but also remains informative after aggregation to categories.

QGDE outperforms baseline DMI. DMI gives visibly skewed estimates: in the language setting, it overestimates English and French while underestimating Chinese and Japanese; in the domain setting, it strongly overestimates Web and underestimates the remaining categories. QGDE substantially reduces these errors, lowering language error from 9.09 to about 3.0 and domain error from 15.14 to a much smaller range.

Anchor gains are weaker after aggregation. The effect of increasing K is less pronounced at the category level than at the token level. In the language setting, error decreases from K = 3 to K = 8, but changes only slightly at K = 14. In the domain setting, the trend is even less monotonic, because aggregation can shift estimated ratios among overlapping domain vocabularies even when token-level predictions improve. Thus, additional anchors still help by improving the underlying token estimates, but their gains are partially smoothed or redistributed by category-level aggregation. The full source-composition and K-sweep results are reported in Table 5.

![](images/11d7e39f0d74daae7fd1c9193928b8691b816959fd8a0597d4f58faf07fe7fb2.jpg)  
Figure 6: Category-level mixture estimation, where QGDE matches the ground truth more closely in both language and domain settings. Each setting compares the ground truth, DMI, and QGDE at three K values; center labels report mean relative error (MRE).

## 8 Validation on the SmolLM Tokenizer

Beyond the controlled language and domain settings, we also test QGDE in a realistic setting: a released tokenizer along with training corpora. Such validation is uncommon because LLM releases rarely include training corpora. SmolLM (Allal et al., 2025) is a useful exception.

We use the SmolLM tokenizer as the target tokenizer. Token-level ground truth is computed from the training corpus, and category-level ground truth uses the released component proportions: FineWebedu 87.30%, Cosmopedia-v2 11.11%, and Pythonedu 1.59%. Since the exact component corpora cannot be used as known corpora, we fit the estimators on corpora matched to the three components: RedPajama-C4 (Weber et al., 2024), a Wikipedia– arXiv mixture (Wikimedia Foundation, 2023; Cornell University, 2020), and CodeParrot (Hugging Face, 2021), respectively. We evaluate three knowncorpus mixture ratios over these proxies: a uniform mixture, a pretrain-like mixture based on (Soldaini et al., 2024), and a target-like mixture that matches the released SmolLM component proportions as a diagnostic setting. Table 2 shows the results.

At the token-level, QGDE outperforms both direct ID-ratio transfer and PoCTrace, reaching 5.72–5.78 MRE. The small spread across Uniform, Pretrain-like, and Target-like source mixture also echoes the controlled experiments: once the knowncorpus mixture covers the relevant components, the exact mixture ratio is less important.

<table><tr><td>Source</td><td>Uniform</td><td>Pretrain-like</td><td>Target-like</td></tr><tr><td colspan="4">Token-level</td></tr><tr><td>Transfer</td><td>9.06</td><td>10.53</td><td>7.71</td></tr><tr><td>PoCTrace</td><td>13.84</td><td>18.68</td><td>20.33</td></tr><tr><td>QGDE</td><td>5.78</td><td>5.73</td><td>5.72</td></tr><tr><td colspan="4">Category-level</td></tr><tr><td>DMI</td><td>9.11</td><td>9.11</td><td>9.11</td></tr><tr><td>QGDE</td><td>6.08</td><td>5.90</td><td>5.93</td></tr></table>

Table 2: Validation on the released SmolLM tokenizer, where QGDE achieves the lowest error for both tokenlevel and category-level estimation.

At the category level, QGDE also improves over the source-independent DMI baseline, reducing error from 9.11 to about 5.9–6.1. These estimates are less exact than in the controlled category-level experiments because the component proportions are highly imbalanced; in particular, Python-edu accounts for only 1.59%. Nevertheless, QGDE still recovers a better mixture estimate.

## 9 Conclusion

Released LLM vocabularies provide a useful signal for estimating hidden corpus composition beyond coarse category proportions. We show that BPE tokenizers share stable token ID–ratio distributions across corpora, and introduce QGDE to transfer this structure through quantile trends and local density weighting. Across controlled settings and the released SmolLM tokenizer, QGDE achieves relative errors as low as 3.00% for token-level estimation and 3.08% after aggregation into categorylevel mixtures.

## Limitations

Scarcity of ground truth for released LLM tokenizers. ChatGPT, Qwen, and DeepSeek release tokenizer vocabularies, but not their training corpora. This prevents direct evaluation of token-level ratio estimates on these models. We therefore validate QGDE in controlled settings and on SmolLM, a rare released tokenizer with available training data. These experiments show that the approach is effective when ground truth is available. Broader validation will become possible if more LLM training corpora are released.

## Ethics Statement

ACL Ethics Policy is respected in this work. This work studies corpus ratio estimation from released tokenizer vocabularies and known corpora. We use publicly available or controlled corpora for research purposes, and we respect the terms, conditions, and copyright requirements of the corresponding data sources. No human subjects or private personal data are involved. The proposed methods are intended for research use, transparency analysis, and auditing of corpus composition signals from released tokenizers.

We adhere to the Association for Computational Linguistics (ACL) guidelines on responsible NLP research<sup>2</sup>, with particular attention to transparency, research-use framing, and responsible handling of corpus-derived evidence.

## Use of AI Assistants

The authors used AI assistants for language polishing, LaTeX editing, and phrasing suggestions during paper preparation. All substantive claims, experimental results, analyses, citations, and final text were reviewed and verified by the authors.

## References

Loubna Ben Allal, Anton Lozhkov, Elie Bakouch, Gabriel Martín Blázquez, Guilherme Penedo, Lewis Tunstall, Andrés Marafioti, Hynek Kydlícek,ˇ Agustín Piqueres Lajarín, Vaibhav Srivastav, et al. 2025. Smollm2: When smol goes big–data-centric training of a small language model.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. 2024. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954.

Rishi Bommasani, Kevin Klyman, Shayne Longpre, Sayash Kapoor, Nestor Maslej, Betty Xiong, Daniel Zhang, and Percy Liang. 2023. The foundation model transparency index. arXiv preprint arXiv:2310.12941.

Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang. 2022. Quantifying memorization across neural language models. In The Eleventh International Conference on Learning Representations.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, et al. 2021. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650.

Cornell University. 2020. arXiv dataset. https://www. kaggle.com/datasets/Cornell-University/arxiv.

Jesse Dodge, Maarten Sap, Ana Marasovic, William´ Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. 2021. Documenting large webtext corpora: A case study on the colossal clean crawled corpus. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 1286–1305.

Jonathan Hayase, Alisa Liu, Yejin Choi, Sewoong Oh, and Noah A Smith. 2024. Data mixture inference attack: Bpe tokenizers reveal training data compositions. Advances in Neural Information Processing Systems, 37:8956–8983.

Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. 2022. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556.

Hugging Face. 2021. CodeParrot. https://huggingface. co/codeparrot.

Jeffrey Li, Alex Fang, Georgios Smyrnis, Maor Ivgi, Matt Jordan, Samir Gadre, Hritik Bansal, Etash Guha, Sedrick Keh, Kushal Arora, et al. 2024. Datacomplm: In search of the next generation of training sets for language models. Advances in Neural Information Processing Systems, 37:14200–14282.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Shayne Longpre, Robert Mahari, Anthony Chen, Naana Obeng-Marnu, Damien Sileo, William Brannon, Niklas Muennighoff, Nathan Khazam, Jad Kabbara, Kartik Perisetla, et al. 2024. A large-scale audit of dataset licensing and attribution in ai. Nature Machine Intelligence, 6(8):975–987.

Kevin P. Murphy. 2022. Probabilistic Machine Learning: An Introduction. MIT Press.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. 2024. Openwebmath: An open dataset of high-quality mathematical web text.

Guilherme Penedo, Hynek Kydlícek, Anton Lozhkov,ˇ Margaret Mitchell, Colin Raffel, Leandro Von Werra, Thomas Wolf, et al. 2024. The fineweb datasets: Decanting the web for the finest text data at scale.

Jackson Petty, Sjoerd van Steenkiste, and Tal Linzen. 2024. How does code pretraining affect language model task performance? arXiv preprint arXiv:2409.04556.

Steven T Piantadosi. 2014. Zipf’s word frequency law in natural language: A critical review and future directions. Psychonomic bulletin & review, 21(5):1112– 1130.

Quantile Regression. 2017. Handbook of quantile regression. Boca Raton, FL, USA: CRC.

Alexander I Saichev, Yannick Malevergne, and Didier Sornette. 2009. Theory of Zipf ’s law and beyond, volume 632. Springer Science & Business Media.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 1715–1725.

Bernard W Silverman. 2018. Density estimation for statistics and data analysis. Routledge.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. 2025. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267.

Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk, David Atkinson, Russell Authur, Ben Bogin, Khyathi Chandu, Jennifer Dumas, Yanai Elazar, et al. 2024. Dolma: An open corpus of three trillion tokens for language model pretraining research. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15725–15788.

Pedro Ortiz Suarez, Laurent Romary, and Benoît Sagot. 2020. A monolingual approach to contextualized word embeddings for mid-resource languages. In Proceedings of the 58th annual meeting of the association for computational linguistics, pages 1703– 1714.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Maurice Weber, Daniel Y Fu, Quentin Anthony, Yonatan Oren, Shane Adams, Anton Alexandrov, Xiaozhong Lyu, Huu Nguyen, Xiaozhe Yao, Virginia Adams, et al. 2024. Redpajama: an open dataset for training large language models.

Wikimedia Foundation. 2023. Wikipedia dumps. https: //dumps.wikimedia.org/.

Sang Michael Xie, Hieu Pham, Xuanyi Dong, Nan Du, Hanxiao Liu, Yifeng Lu, Percy S Liang, Quoc V Le, Tengyu Ma, and Adams Wei Yu. 2023. Doremi: Optimizing data mixtures speeds up language model pretraining. volume 36, pages 69798–69818.

Cheng Xu, Shuhao Guan, Derek Greene, M Kechadi, et al. 2024. Benchmark data contamination of large language models: A survey. arXiv preprint arXiv:2406.04244.

Linting Xue, Noah Constant, Adam Roberts, Mihir Kale, Rami Al-Rfou, Aditya Siddhant, Aditya Barua, and Colin Raffel. 2021. mt5: A massively multilingual pre-trained text-to-text transformer. In Proceedings ofthe 2021 conference ofthe North American chapter of the association for computational linguistics: Human language technologies, pages 483–498.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Shuo Yang, Wei-Lin Chiang, Lianmin Zheng, Joseph E Gonzalez, and Ion Stoica. 2023. Rethinking benchmark and contamination for language models with rephrased samples. arXiv preprint arXiv:2311.04850.

Qingjie Zhang, Di Wang, Haoting Qian, Liu Yan, Tianwei Zhang, Ke Xu, Qi Li, Minlie Huang, Hewu Li, and Han Qiu. 2025. Speculating llms’ chinese training data pollution from their tokens. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26124–26144.

Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books.

## A Quantile Trend Fitting Details

In Section 4.1, QGDE fits one log-linear trend for each quantile level over the known ID–ratio points using quantile regression (Regression, 2017). This section gives the optimization objective used to fit those trends.

Let $\mathbf { \mathcal { P } } = \{ ( x _ { j } , y _ { j } ) \} _ { j = 1 } ^ { n }$ denote the known ID– ratio points, where $x _ { j } = \log t _ { j } , y _ { j } = \log r _ { j } , t _ { j }$ is the token ID, and $r _ { j }$ is the observed ratio in the known corpus. For each quantile level $\tau \in \mathcal T$ , we estimate the trend coefficients by

$$
\left( a _ { \tau } , b _ { \tau } \right) = \arg \operatorname* { m i n } _ { a , b } \sum _ { j = 1 } ^ { n } \rho _ { \tau } \left( y _ { j } - a - b x _ { j } \right) ,\tag{10}
$$

where $\rho _ { \tau } ( \cdot )$ is the asymmetric loss:

$$
\rho _ { \tau } ( u ) = \left\{ \begin{array} { l l } { \tau u , } & { u \geq 0 , } \\ { ( \tau - 1 ) u , } & { u < 0 . } \end{array} \right.\tag{11}
$$

The fitted trend is $\begin{array} { r } { q _ { \tau } ( x ) = a _ { \tau } + b _ { \tau } x } \end{array}$ . The asymmetric loss encourages approximately a τ fraction of known points to lie below the trend, so different $\tau$ values trace different vertical levels of the ID– ratio distribution. A single median trend captures only the central tendency; QGDE fits a family of trends so that the vertical spread of plausible ratios is preserved before local density weighting.

## B Transfer Similarity Computation

In Section 3, Figure 2 quantifies whether ID–ratio distributions transfer across tokenizers. We compute this score by converting each scatter plot into a probability density $P _ { D }$ over ID–ratio space.

For each tokenizer D, each token $v _ { i }$ is represented as $( x _ { i } , y _ { i } ) = ( \log t _ { i } , \log r _ { i } )$ , where $t _ { i }$ is its token ID and $r _ { i }$ is its corpus ratio. We place all tokenizers on a shared grid $\boldsymbol { B }$ and convert each tokenizer into a smoothed two-dimensional histogram:

$$
P _ { D } ( b ) = \frac { \sum _ { v _ { i } \in V _ { D } } \mathbf { 1 } [ ( x _ { i } , y _ { i } ) \in b ] + \epsilon } { | V _ { D } | + \epsilon | B | } , \qquad b \in \mathcal { B } .\tag{12}
$$

Here, ϵ is a small smoothing constant. Setting $D = S { \mathrm { ~ o r ~ } } D = T$ gives the source and target densities $P _ { S }$ and $P _ { T }$ used in the main text.

The score is directional: $\mathrm { S i m } ( S \to T )$ asks how well the source density $P _ { S }$ explains the target density $P _ { T }$ . We therefore compute $D _ { \mathrm { K L } } ( P _ { T } \Vert P _ { S } )$ , normalize it by the target entropy $H ( P _ { T } )$ , and map the result to (0, 1], where higher values indicate stronger transfer similarity (Murphy, 2022).

<table><tr><td>Source Ratio</td><td colspan="4">Target</td></tr><tr><td>Web:Wiki:Code:Math</td><td>Web</td><td>Wiki</td><td>Code</td><td>Math</td></tr><tr><td>100:0:0:0</td><td>1.00</td><td>0.93</td><td>0.87</td><td>0.95</td></tr><tr><td>0:100:0:0</td><td>0.94</td><td>1.00</td><td>0.85</td><td>0.91</td></tr><tr><td>0:0:100:0</td><td>0.86</td><td>0.78</td><td>1.00</td><td>0.93</td></tr><tr><td>0:0:0:100</td><td>0.96</td><td>0.90</td><td>0.92</td><td>1.00</td></tr><tr><td>70:10:10:10</td><td>0.96</td><td>0.93</td><td>0.84</td><td>0.91</td></tr><tr><td>10:70:10:10</td><td>0.93</td><td>0.97</td><td>0.82</td><td>0.89</td></tr><tr><td>10:10:70:10</td><td>0.94</td><td>0.96</td><td>0.82</td><td>0.89</td></tr><tr><td>10:10:10:70</td><td>0.97</td><td>0.91</td><td>0.87</td><td>0.94</td></tr><tr><td>25:25:25:25</td><td>0.95</td><td>0.95</td><td>0.83</td><td>0.91</td></tr></table>

Table 3: Directional transfer similarity of domains.
<table><tr><td rowspan="2">Source Ratio En:Fr:Ja:Zh</td><td colspan="4">Target</td></tr><tr><td>En</td><td>Fr</td><td>Ja</td><td>Zh</td></tr><tr><td>100:0:0:0</td><td>1.00</td><td>0.99</td><td>0.78</td><td>0.80</td></tr><tr><td>0:100:0:0</td><td>0.99</td><td>1.00</td><td>0.78</td><td>0.80</td></tr><tr><td>0:0:100:0</td><td>0.85</td><td>0.85</td><td>1.00</td><td>0.96</td></tr><tr><td>0:0:0:100</td><td>0.85</td><td>0.85</td><td>0.96</td><td>1.00</td></tr><tr><td>70:10:10:10</td><td>0.95</td><td>0.95</td><td>0.89</td><td>0.88</td></tr><tr><td>10:70:10:10</td><td>0.91</td><td>0.91</td><td>0.94</td><td>0.93</td></tr><tr><td>10:10:70:10</td><td>0.84</td><td>0.84</td><td>0.97</td><td>0.96</td></tr><tr><td>10:10:10:70</td><td>0.84</td><td>0.84</td><td>0.97</td><td>0.96</td></tr><tr><td>25:25:25:25</td><td>0.87</td><td>0.87</td><td>0.97</td><td>0.95</td></tr></table>

Table 4: Directional transfer similarity of languages.

## C Single-source and Mixed-source Transfer Similarity

In Section 3, the main transfer analysis compares single-category tokenizers. Because the token-level experiments also use mixed known-corpus sources, Table 3 and Table 4 report transfer similarity from both single-source and mixed-source tokenizers to single-category targets. Rows specify the source tokenizer’s training mixture, and columns specify the target tokenizer whose ID–ratio density is explained.

The mixed-source profiles support the tokenlevel results in Table 1. First, ID–ratio distributions remain broadly transferable after mixing: most mixed-source similarities are still above 0.8, and many domain similarities are close to or above 0.9. Second, mixed sources retain useful similarity to multiple targets rather than only to a single self-matched target. This helps explain why mixed known-corpus sources are generally more stable for token-level estimation. Third, the same structure as in the main heatmap remains visible: English– French and Japanese–Chinese form stronger language clusters, while Code is the hardest domain target because its ID–ratio distribution differs more from natural language corpora.

![](images/e40f9fd97b84db3b691d4c7cd8d3c0a792112a8dd6ff91131742250f774c18ac.jpg)  
Figure 7: Token ID–ratio scatter plots for all eight controlled tokenizers. The top row shows language-specific mC4 tokenizers, and the bottom row shows domain-specific English tokenizers.

## D Additional ID–Ratio Scatter Plots

In Section 3, Figure 2 shows representative ID– ratio scatter plots to motivate transferability. Figure 7 provides the full set of eight controlled tokenizers: the top row shows language-specific mC4 tokenizers, and the bottom row shows domainspecific English tokenizers.

Across both rows, the point clouds follow a similar downward log–log shape, while their thickness, location, and tail behavior vary by language and domain. These patterns support the main-text conclusion that the ID–ratio relationship is broadly shared but not identical across corpora.

## E Category-Level Estimation Details

In Section 7, Figure 6 shows selected categorylevel estimates for the uniform language and domain targets. We present the complete sourcecomposition and K-sweep results in this section, showing that the main conclusions do not depend on a single displayed K or source mixture.

Table 5 follows the same layout as Table 1: single-source columns use one known corpus, 70%- mixed columns use one dominant known corpus with the other three mixed in equally, and the Target-like column uses the same uniform mixture as the target. We report category-level mean relative error, scaled by 100.

QGDE remains below DMI. Across both language and domain targets, QGDE gives substantially lower category-level error than the sourceindependent DMI baseline. In the language setting, DMI has error 9.09, while the QGDE averages are around 3.1–3.3 across source mixtures. In the domain setting, DMI has error 15.14, while QGDE stays around 5.3–5.5.

Anchor gains are weaker after aggregation. The K-sweep is less monotonic than in token-level estimation. For languages, increasing K improves the early rows but the results quickly cluster near 3.1. For domains, most QGDE rows remain in a narrow band around 5.3–5.5. This supports the observation in Section 7 that category-level aggregation smooths and redistributes token-level improvements, so additional anchors have a weaker visible effect after aggregation.

Language mixtures remain easier. The language block consistently has lower error than the domain block. This matches the main results: language categories provide more separable tokenlevel evidence, whereas domain categories share more vocabulary and therefore make aggregate mixture estimation harder.

<table><tr><td rowspan="2">Source</td><td colspan="4">Single-source</td><td colspan="5">70%-mixed source</td><td>Target-like</td></tr><tr><td> $S _ { 1 }$  -only</td><td> $S _ { 2 } .$  -only</td><td> $S _ { 3 }$  -only  $S _ { 4 } .$ </td><td>-only Avg.±Std.</td><td> $S _ { 1 }$ </td><td>-major  $S _ { 2 } .$  -major</td><td> $S _ { 3 } .$  -major</td><td> $S _ { 4 }$  -major</td><td> $\mathbf { A v g . } { \pm } \mathbf { S t d . }$ </td><td>Uniform</td></tr><tr><td colspan="10">Language Sources</td></tr><tr><td>DMI</td><td></td><td></td><td></td><td></td><td>9.09</td><td>(source-independent)</td><td>3.16</td><td></td><td></td><td></td></tr><tr><td>QGDE Avg.</td><td>3.25 3.70</td><td>3.21</td><td>3.22 3.33</td><td>3.14 3.43</td><td> $\mathbf { 3 . 2 1 { \pm 0 . 0 4 } }$   $3 . 5 1 { \pm } 0 . 1 4$ </td><td>3.20 3.59</td><td>3.14</td><td>3.14 3.47</td><td> $\mathbf { 3 . 1 6 \pm 0 . 0 2 }$   $3 . 4 4 \pm 0 . 1 4$ </td><td>3.12</td></tr><tr><td> $K = 3$   $K = 4$ </td><td>3.62</td><td>3.58 3.53</td><td>3.26</td><td>3.23</td><td> $3 . 4 1 \pm 0 . 1 7$ </td><td>3.47 3.27</td><td>3.21 3.16</td><td>3.25</td><td> $3 . 2 7 { \pm } 0 . 0 8$ </td><td>3.42 3.17</td></tr><tr><td>K = 5</td><td>3.43</td><td>3.41</td><td>3.26</td><td>3.17</td><td> $3 . 3 2 { \pm } 0 . 1 1$ </td><td>3.39 3.25</td><td>3.15</td><td>3.14</td><td> $3 . 2 2 { \pm } 0 . 0 7$ </td><td>3.11</td></tr><tr><td></td><td>3.32</td><td>3.31</td><td>3.22</td><td>3.08</td><td> $3 . 2 3 { \pm } 0 . 1 0$ </td><td>3.31 3.29 3.21</td><td>3.11</td><td>3.08</td><td> $3 . 1 7 { \pm } 0 . 0 8$ </td><td>3.07</td></tr><tr><td>K=0</td><td>3.20</td><td>3.18</td><td>3.19</td><td>3.11</td><td> $3 . 1 7 { \pm } 0 . 0 3$ </td><td>3.13 3.11</td><td>3.14</td><td>3.12</td><td> $3 . 1 2 { \stackrel { \textstyle - } { \pm } } 0 . 0 1$ </td><td>3.11</td></tr><tr><td> $K = 8$ </td><td>3.18</td><td>3.08</td><td>3.18</td><td>3.11</td><td> $3 . 1 3 { \pm } 0 . 0 4$ </td><td>3.15 3.13 3.08</td><td>3.14</td><td>3.09</td><td> $3 . 1 3 { \pm } 0 . 0 2$ </td><td>3.10</td></tr><tr><td>K = 9</td><td>3.09</td><td>3.01</td><td>3.19</td><td>3.07</td><td> $3 . 0 9 { \pm } 0 . 0 6 $ </td><td>3.07</td><td>3.12</td><td>3.09</td><td> $3 . 0 9 { \pm } 0 . 0 2$ </td><td>3.09</td></tr><tr><td> $K = 1 0$ </td><td>3.10</td><td>3.10</td><td>3.18</td><td>3.08</td><td> $3 . 1 1 \pm 0 . 0 4$ </td><td>3.10</td><td>3.09</td><td>3.10</td><td> $3 . 0 9 { \pm } 0 . 0 0$ </td><td>3.09</td></tr><tr><td>K = 11</td><td>3.11</td><td>3.11</td><td>3.20</td><td>3.07</td><td> $3 . 1 2 { \pm } 0 . 0 5$ </td><td>3.12</td><td>3.11</td><td>3.08</td><td> $3 . 1 0 { \pm } 0 . 0 2$ </td><td>3.04</td></tr><tr><td>K = 12</td><td>3.05</td><td>3.04</td><td>3.22</td><td>3.10</td><td> $3 . 1 0 { \pm } 0 . 0 7$ </td><td>3.10 3.06 3.05</td><td>3.13</td><td>3.09</td><td> $3 . 0 8 { \pm } 0 . 0 3$ </td><td>3.06</td></tr><tr><td> $K = 1 3$ </td><td>3.08</td><td>3.06</td><td>3.23</td><td>3.12</td><td> $3 . 1 2 { \pm } 0 . 0 7$ </td><td>3.08</td><td>3.15</td><td>3.10</td><td> $3 . 1 0 { \pm } 0 . 0 3$ </td><td>3.08</td></tr><tr><td> $K = 1 4$ </td><td>3.17</td><td>3.14</td><td>3.24</td><td></td><td> $3 . 1 7 { \pm } 0 . 0 4$ </td><td>3.06</td><td></td><td>3.11</td><td>3.12±0.03</td><td></td></tr><tr><td></td><td></td><td></td><td>3.13</td><td></td><td>3.12</td><td>3.08</td><td>3.15</td><td></td><td></td><td>3.09</td></tr><tr><td colspan="9"> $\mathbf { D o m a i n } \ S \mathbf { o u r c e s } \ ( S _ { 1 } = \mathbf { W e b } , S _ { 2 } = \mathbf { W i k i } , S _ { 3 } = \mathbf { M a t h } , S _ { 4 } = \mathbf { C o d e } )$ </td></tr><tr><td>DMI</td><td colspan="10"></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>15.14  ${ \bf 5 . 3 9 { \pm } 0 . 0 4 }$ </td><td>(source-independent)</td><td></td><td></td><td></td><td>5.37</td></tr><tr><td>QGDE Avg.  $K = 3$ </td><td>5.44 5.25</td><td>5.37 5.44</td><td>5.40 5.27</td><td>5.34 5.24</td><td> $5 . 3 0 { \pm } 0 . 0 8$ </td><td>5.44 5.33</td><td>5.37 5.40</td><td>5.48 5.40</td><td>5.39 5.26</td><td> ${ \bf 5 . 4 2 \pm 0 . 0 5 }$   $5 . 3 5 { \pm } 0 . 0 6$ </td></tr><tr><td> $K = 4$ </td><td>5.54</td><td>5.26</td><td>5.39</td><td>5.27</td><td> $5 . 3 7 { \pm } 0 . 1 1 $ </td><td>5.23</td><td>5.40 5.42</td><td>5.33</td><td> $5 . 3 4 \pm 0 . 0 8$ </td><td>5.32</td></tr><tr><td>2281 K = 5</td><td>5.50</td><td>5.35</td><td>5.29</td><td>5.27</td><td> $5 . 3 5 { \pm } 0 . 0 9$ </td><td>5.60 5.32</td><td>5.51</td><td>5.32</td><td> $5 . 4 4 \pm 0 . 1 2$ </td><td>5.28 5.29</td></tr><tr><td>K =</td><td>5.47</td><td>5.47</td><td>5.41</td><td>5.30</td><td> $5 . 4 2 { \pm } 0 . 0 7$ </td><td>5.50 5.36</td><td>5.46</td><td>5.27</td><td> $5 . 4 0 { \pm } 0 . 0 9$ </td><td>5.43</td></tr><tr><td></td><td>5.51</td><td>5.39</td><td>5.35</td><td>5.34</td><td> $5 . 4 0 { \scriptstyle \pm 0 . 0 7 }$ </td><td>5.44 5.39</td><td>5.51</td><td>5.42</td><td> $5 . 4 4 \pm 0 . 0 5$ </td><td>5.46</td></tr><tr><td></td><td>5.53</td><td>5.41</td><td>5.30</td><td>5.34</td><td> $5 . 4 0 { \pm } 0 . 0 9$ </td><td>5.46 5.39</td><td>5.52</td><td>5.43</td><td> $5 . 4 5 { \pm } 0 . 0 5$ </td><td>5.39</td></tr><tr><td>K = 9</td><td>5.42</td><td>5.30</td><td>5.40</td><td>5.38</td><td> $5 . 3 7 { \pm } 0 . 0 5$ </td><td>5.43 5.39</td><td>5.52</td><td>5.43</td><td> $5 . 4 4 \pm 0 . 0 5$ </td><td>5.37</td></tr><tr><td> $K = 1 0$ </td><td>5.38</td><td>5.34</td><td>5.42</td><td>5.37</td><td> $5 . 3 8 \pm 0 . 0 3$ </td><td>5.47 5.38</td><td>5.54</td><td>5.49</td><td> $5 . 4 7 { \pm } 0 . 0 6$ </td><td>5.37</td></tr><tr><td>K = 11</td><td>5.35</td><td>5.34</td><td>5.50</td><td>5.38</td><td> $5 . 3 9 { \pm } 0 . 0 6$ </td><td>5.47 5.36</td><td>5.53</td><td>5.43</td><td> $5 . 4 5 { \pm } 0 . 0 6$ </td><td>5.36</td></tr><tr><td>K = 12</td><td>5.36</td><td>5.35</td><td>5.56</td><td>5.37</td><td> $5 . 4 1 \pm 0 . 0 9$ </td><td>5.45 5.35</td><td>5.48</td><td>5.43</td><td> $5 . 4 3 { \pm } 0 . 0 5$ </td><td>5.36</td></tr><tr><td> $K = 1 3$ </td><td>5.45</td><td>5.40</td><td>5.48</td><td>5.38</td><td> $5 . 4 3 { \pm } 0 . 0 4$ </td><td>5.47 5.32</td><td>5.45</td><td>5.40</td><td> $5 . 4 1 { \pm } 0 . 0 6$ </td><td>5.40</td></tr><tr><td> $K = 1 4$ </td><td>5.48</td><td>5.36</td><td>5.47</td><td>5.44</td><td> $5 . 4 4 \pm 0 . 0 5$ </td><td>5.49 5.36</td><td>5.46</td><td>5.43</td><td> $5 . 4 4 \pm 0 . 0 5$ </td><td>5.41</td></tr></table>

Table 5: Mean relative error (MRE) (%) of category-level mixture estimation under uniform targets. The sourcemixture columns follow the same layout as Table 1. DMI is source-independent and therefore shown as a single merged value. Darker green indicates higher error in the QGDE K-ablation rows.