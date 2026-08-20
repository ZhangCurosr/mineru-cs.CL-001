# Comment-level Topic Drift Analysis in the Reddit Corpus

Steven Morse and Dan Runfola and Trenton W. Ford William & Mary {stmorse, dsmillerrunfol, twford}@wm.edu

## Abstract

We present a novel application of embeddingbased dynamic topic modeling techniques to detect and quantify topic drift at the comment level in a massive corpus. By leveraging pretrained language models to generate contextualized semantic embeddings for short text, we analyzed 12.7 billion Reddit comments spanning 2006 to 2022. Using unsupervised methods on these embeddings, we identify dynamically evolving topic clusters over time. Our primary contribution is a methodology for analysis of semantic drift and discourse evolution in the embedding space itself. We also demonstrate modifications to existing methods that enable this analysis at scale, and we propose and demonstrate a null model comparison test to filter spurious dynamics. Key findings suggest that politically and socially contentious topics exhibit significant directional drift in embedding space, with inter-topic distances changing systematically over time beyond what the null model can explain, whereas domains such as music and sports remain comparatively stable.

## 1 Introduction

Attention-based language models encode language into a high-dimensional embedding space that captures both semantic relationships and long-range contextual dependencies, enabling a nuanced representation of meaning. This has enabled the study of topic analysis as computation in a vector space (Churchill and Singh, 2022; Wu et al., 2024). Beyond static clustering, embedding-based methods enable quantitative comparisons among words, sentences, and topics (Grootendorst, 2022). Yet most pipelines remain descriptive: they list topics or track keyword changes, but they do not estimate whether topics move meaningfully through embedding space over time or compare those movements across topics. That gap matters because the contextualized semantics of topic discourse shifts with events and norms; without a time-resolved geometric model, we cannot measure or forecast that change.

We address this gap by treating change in topical discourse as trajectories in embedding space and asking when observed movement exceeds stochastic variation. Concretely, we embed comments, cluster within monthly windows, align similar clusters across time to recover topic paths, and test those paths against a null model to separate genuine drift from noise. At web-scale over multiple decades, this yields interpretable measures of displacement and direction and exposes inter-topic convergence and divergence — for example, two topics may retain the same representative words (for example, “police” and “racism”) but grow closer in semantic space, implying a change in semantic similarity of content not detectable by methods treating topics as bags of words. The result is a practical framework for tracking discourse change, evaluating interventions, and informing downstream tasks such as retrieval, moderation, and forecasting.

## 1.1 Related work

Topic modeling identifies and organizes latent themes in a collection of texts (Vayansky and Kumar, 2020), and it categorizes broadly into approaches based on distributional language frameworks and those leveraging deep neural network embeddings of language (Churchill and Singh, 2022). Within the distributional paradigm, considerable research has focused on modeling temporal dynamics and, as a natural extension, analyzing changes in topic distributions (“drift”) (Khan and Wakabayashi, 2021; Knights et al., 2009; Liu et al., 2013). Within the neural, or algorithmic paradigm, similar extensions exist, with most studies focused at the word (or token) level (Bamler and Mandt, 2017; Periti and Montanelli, 2024). In this section, we review this literature and introduce our own contribution: the study of drift at the sentence (or in our context, comment) level, in an embedded space and at large scale.

![](images/7cab830ad3419812c88ede80d508943804740e7e390c7c418520f6b9860f0bcb.jpg)  
Figure 1: Clustering in embedded space yields time-local topics with contextualized semantic similarity over time. Aligning highly similar topic clusters over time may reveal topic drift in embedding space.

Probabilistic approaches typically approach topics as a collection of independently occurring words (“bag of words”) (Churchill and Singh, 2022). Latent Dirichlet Allocation (LDA) (Blei et al., 2003) introduced a flexible, hierarchical Bayesian framework for this task, with subsequent work exploring metrics of topic coherence and diversity (Röder et al., 2015), expansion to a dynamic setting where topics’ distributions change over time (Blei and Lafferty, 2006; Wang et al., 2008; Bhadury et al., 2016; Zhang and Lauw, 2022; Iwata et al., 2010), and analysis of distributional drift in the dynamic setting (Knights et al., 2009; Khan and Wakabayashi, 2021; Liu et al., 2013; Li et al., 2017; Fei et al., 2015). In work extending these topic models to larger datasets, we find modifications to existing methods (Yuan et al., 2015; Hassani et al., 2020), applications to marketing (Amado et al., 2018), social media (Guo et al., 2016), among many others. We also see the application of modern neural network architectures (“neural topic models” (NTMs) (Wu et al., 2024)), which in most cases extends probabilistic approaches by using vectorized BoW input to a neural net, as in (Miao et al., 2016; Srivastava and Sutton, 2017; Krishnan et al., 2018; Zhao et al., 2021).

More relevant to our work is the recent field of approaches that leverage transformer-based language models’ embeddings to capture deep contextual information about the text. Early works like word2vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) found word embeddings using shallow neural networks trained to predict the immediate context of a given word. Large Language Models (LLMs) based on the transformer architecture (Vaswani et al., 2017; Kenton and Toutanova, 2019; Zhou et al., 2024), provide even richer language embeddings. Research has since extended this transformer approach to sentencelevel embeddings (Incitti et al., 2023) — for example, the Sentence-BERT approach in (Reimers and Gurevych, 2019) involves fine-tuning a BERT model through contrastive training on a dataset of paired sentences, driving the embedding space’s geometry to capture semantic similarity. This concept is highly adaptable, as evidenced by a growing suite of tools (Thielmann et al., 2024).

These language embeddings enable new approaches in topic modeling by working in this embedding space (Wu et al., 2024). For example, we may apply unsupervised methods directly to identify topic clusters, either at word (Xie et al., 2019) or sentence level (Grootendorst, 2022), which we discuss further below. We may extend this to analysis of word dynamics (Bamler and Mandt, 2017; Yao et al., 2018; Hofmann et al., 2021) and semantic shift detection (Periti and Montanelli, 2024; Taher Harikandeh et al., 2023). Many NTM approaches also work by incorporating embeddings into a BoW model (e.g., (Bunk and Krestel, 2018)), either as a kind of “side-information” (Petterson et al., 2010) or directly as input (Dieng et al., 2019, 2020), although (Zhang et al., 2022) argues that unsupervised approaches directly in the embedding space yield better results. Note also there is a growing field of study around feature drift in LLMs (e.g. (Fastowski and Kasneci, 2024; Santos et al., 2024)), but this differs from our subject in its focus on model reliability and robustness in the context of feature drift, not topic modeling and semantic drift.

## 1.2 Contributions

We build on two embedding-based approaches to topic dynamics within this literature. BERTopic discovers time-agnostic clusters via global clustering, then inspects time-sliced keyword changes, which imposes a global view that can mask drift of topics in embedding space (Grootendorst, 2022). The complementary method in (Rahimi et al., 2024) partitions time first, clusters locally, and aligns clusters across periods using hierarchical procedures. Both methods emphasize keyword or coherence summaries rather than geometry, provide no formal test that observed dynamics exceed stochastic variation, and rely on global dimensionality reduction that is impractical at web-scale.

Our work augments this literature in three ways. First, we propose a more scalable comment-level pipeline: we perform topic clustering directly in the original embedding space within monthly windows, then perform dimensionality reduction and alignment of the resulting centroids to discover topic trajectories. Second, we introduce a likelihood-ratio test based on a random walk null model to filter genuine topic drift from noise. This framework and null together provide quantitative measures of displacement and directionality both within and between topics, enabling interpretation in the embedding space. Our third contribution is to demonstrate this analysis on a web-scale dataset, the Reddit corpus.

## 2 Data & Methodology

## 2.1 Data

We explore topic drift using embeddings of the Reddit comment corpus (2006–2022) released via Pushshift (Baumgartner et al., 2020). To analyze topical discourse in text, we restrict to comments and exclude top-level submissions and multimedia. Comments authored by deleted accounts are removed. Very short replies (including single-word comments) are retained to preserve conversational structure; each comment is truncated to at most 128 tokens to match the embedding model’s context window.

After filtering, the corpus comprises $T = 2 0 4$ monthly windows and 12.7 billion comments. The volume is highly skewed toward recent years: the most recent five-year period contains >9 billion comments, more than 160× the first five years (< 60 million).<sup>1</sup>

## 2.2 Methodological framework

This work leverages a framework consisting of three stages.

1. Contextualized embedding. The first stage uses a pretrained, transformer-based LLM to convert a short body of text into a highdimensional vector representation capturing contextual encoding. We perform this embedding using a fixed transformer across the entire dataset.

2. Topic clustering. The second stage partitions the embeddings into time-windows and performs clustering within each window to infer topic clusters by grouping texts that are close in the embedding space. Concurrent with this layer, we produce keyword representations of each topic cluster to aid in semantic analysis.

3. Topic alignment. The third stage works across time windows to align similar topic clusters over time, in a locally-time-aware way. Given these aligned topic groups, we analyze the resulting “trajectories” of topics in embedding space, specifically, their absolute drift and drift relative to each other. We also conduct a dimensionality reduction prior to this alignment, and we note this differs from (Grootendorst, 2022; Rahimi et al., 2024) which apply a global dimensionality reduction (UMAP) prior to clustering, a prohibitive step for massive datasets.

We then perform analysis of the resulting aligned topic groups in the embedding space, leveraging the interpretation of distance in the embeddings space as a representation of semantic similarity.

## 2.3 Contextual Embedding

Consider a dataset $\mathcal { D } = \{ ( s _ { i } , \hat { t } _ { i } ) \}$ consisting of “mini-documents” (generally of sentence to paragraph length) $s _ { i } \in S _ { \ell }$ with timestamps $\hat { t } _ { i }$ at high granularity (e.g., seconds), and where $S _ { \ell }$ represents character strings of maximum length ℓ. Our first step is to embed each mini-document in a space which captures semantic and contextual similarity, using a transformation $E : \mathcal { S }  \mathcal { E } = \mathbb { R } ^ { d }$ . For $E ,$ we use a pretrained LLM following the SBERT and SentenceTransformer work in previous literature (Reimers and Gurevych, 2019). The resulting embeddings serve as vectorized representations of the sentences, optimized to enable accurate comparisons of semantic similarity.

![](images/bc55ccbd2c9fc7a3dbfed0b7ed92fbb2125c73596497cb8ab5fcd829f942da4b.jpg)  
(a) Topic clusters (June 2007)

![](images/28b2527aae4f18252f359615c075e695920f08a53c226b0fb891006ae8dcbb0e.jpg)  
(b) Topic clusters (June 2008)  
Figure 2: Visualization of topic clusters for two example time periods (June 2007 (a) and 2008 (b)), with each dot representing a single embedded comment. Only the top 200 closest to the centroid are shown for visual clarity; text label is the top scoring word by TF-IDF. We note trends of changing focus: increased presidential voting discussion, diversification of several topics (e.g., immigration), and increase in sexually-oriented commenting.

Specifically, we used ‘all-MiniL $\mathbf { \mathcal { M } } \mathbf { - } \mathbf { L } 6 \mathbf { - } \mathbf { v } 2 \mathbf { \cdot } 2$ which embeds each post $s _ { i }$ into $\mathbb { R } ^ { d }$ with $d = 3 8 4$ (following the methodology presented in (Rahimi et al., 2024)). The output of this step is an embedded dataset $\mathcal { D } _ { E } = \{ ( \mathbf { e } _ { i } , \hat { t } _ { i } ) \}$ consisting of timestamped, embedded mini-documents $\mathbf { e } _ { i } \in \mathcal { E }$ , with d representing the embedding dimension of the sentence transformer E.

## 2.4 Time-windowed Topic Clustering

After embedding the dataset, we partition it into discrete time windows $\mathcal { T } = \{ 0 , 1 , . . . , T \}$ such that each time window t consists of all embeddings $\mathcal { E } _ { t }$ with timestamps $\hat { t }$ in the range $[ t ^ { \mathrm { m i n } } , t ^ { \mathrm { m a x } } )$ . We set the time window as a calendar month, and within each window, we assume there is a set of M topics such that an embedding $\mathbf { e } _ { i }$ is a sample from some topic $m$ , which we denote as ${ \bf e } _ { i } ^ { ( m ) }$ . Following earlier studies (Gao et al., 2019), we further assume each topic has some underlying topic distribution for that time period, which is approximately normal in the embedding space, that is:

$$
\mathbf { e } _ { i } ^ { ( m ) } \sim \mathcal { N } ( \pmb { \mu } _ { t _ { i } } ^ { ( m ) } , \pmb { \Sigma } _ { t _ { i } } ^ { ( m ) } ) .\tag{1}
$$

with $\pmb { \mu } _ { t _ { i } } ^ { ( m ) } \in \mathbb { R } ^ { d }$ and $\pmb { \Sigma } _ { t _ { i } } ^ { ( m ) } \in \mathbb { R } ^ { d \times d }$

To identify topic assignments in an unsupervised way, we perform a clustering step within each time window $C : \mathcal { E } _ { t }  \mathcal { L } _ { t }$ which assigns a label $l _ { i }$ to each embedding e<sub>i</sub>. Any unsupervised method that yields a labeling of the inputs may be employed at this stage; however, under the normality assumption, a Gaussian mixture model is particularly wellsuited.

With this consideration in mind, we apply $k \mathrm { - }$ means clustering. K-means is well-known to approximate a GMM with spherical covariance (e.g., (MacKay, 2003)); in addition, it scales sufficiently to operate within the full d-dimensional embedding space and handle an extensive corpus. Because the embedding space encodes contextually informed semantic similarity, we may interpret the resulting clusters as topic clusters. We find that $k = 5 0$ clusters provide a consistent representation of topics across the entire dataset, and we provide a more thorough analysis of this choice in Appendix A.

The output of this step is a labeled dataset $\mathcal { D } _ { L } =$ $\left\{ \left( \mathbf { e } _ { i } , t _ { i } , l _ { i } \right) \right\}$ . Note that our clustering method allows us to associate each label $l _ { i } \in L = \{ 1 , . . . , M \}$ within a time window t with a centroid $\mathbf { c } _ { t } ^ { ( m ) }$ that estimates the topic mean, $\pmb { \mu } _ { t } ^ { ( m ) }$ . Note also each specific timestamp $\hat { t } _ { i }$ is replaced with the corresponding time window (i.e. month) $t _ { i }$

Since we perform this unsupervised clustering stage independently within each time window, we keep open the possibility of capturing different topic clusters in different windows, and avoid imposing a global view on the embedded location of any particular topic (cf. (Rahimi et al., 2024)).

To improve semantic interpretability, we assign a single-keyword summary to represent each cluster. This is implemented using the topic-frequency inverse-document-frequency (TF-IDF) approach, treating each cluster as a document, and normalizing frequency by time-window (c.f. (Grootendorst, 2022)). Specifically, define

$$
\operatorname { t f } ( t , c ) = { \frac { f _ { t , c } } { \sum _ { t ^ { \prime } \in c } f _ { t ^ { \prime } , c } } }\tag{2}
$$

$$
\operatorname { i d f } ( t , C ) = \log { \frac { N } { | \{ c : c \in C , \ t \in c \} | } }\tag{3}
$$

where $\operatorname { t f } ( t , c )$ captures how relevant term t is to document $c ,$ and id $\mathrm { \Sigma } [ ( t , C )$ captures how unique term t is across all documents C by its ratio of occurrence N in c to all documents. In our application, we treat each cluster as a document c, and all clusters in a time-window as C. In practice, since we are primarily concerned with interpretability, we use only the $| c | = 1 0 0$ mini-documents nearest the centroid $\mathbf { c } _ { i }$ as the $" c _ { i } > "$ for TF-IDF purposes. We emphasize that these keywords are only to aid in human interpretability, as we perform all analysis and validation in the embedding space itself.

## 2.5 Topic Alignment

Because clustering is performed independently within each time period, a subsequent alignment stage is required to identify clusters that represent the same underlying topic across periods. This approach contrasts with the global clustering of (Grootendorst, 2022) and follows the temporal alignment strategy of (Rahimi et al., 2024). In essence, the goal is to determine whether topic a at time t and topic b at time t + 1, produced by separate period-specific clusterings, correspond to the same semantic concept.

To accomplish this, given the labeled dataset $\mathcal { D } _ { L } = \{ ( { \bf e } _ { i } , t _ { i } , l _ { i } ) \}$ , we perform an additional unsupervised step that clusters the topic centroids $\mathbf { c } _ { t } ^ { ( m ) }$ to align similar topics over time. Because no invariant topic representation exists across periods, alignment is based on proximity in embedding space rather than on shared keywords or TF-IDF features. This approach preserves contextual semantics and enables detection of topic evolution, emergence, and disappearance over time. We specifically apply the Uniform Manifold Approximation and Projection (UMAP) algorithm to reduce the centroid embeddings to $d = 1 0$ dimensions, followed by Hierarchical DBSCAN (HDBSCAN) clustering to group temporally related topics. This combination allows flexible detection of gradual semantic drift and heterogeneous topic structures, yielding a final mapping $A : { \mathcal { C } } \to { \mathcal { G } }$ from centroids to topic groups and a dataset $D _ { A } = \{ ( \mathbf { c } _ { i } , g _ { i } ) \}$ representing each topic and its aligned group across time.

## 2.6 Validation of drift

A central question to our work is assessing topics’ drift through embedding space over time. A limited but straightforward approach to measure this is to measure a topic group’s total displacement over time. That is, for group k define its displacement $\delta _ { k }$ as

$$
\delta _ { k }  { \stackrel { \mathrm { d e f } } { = } } | | \mathbf { c } _ { t _ { \operatorname* { m a x } } } ^ { ( k ) } - \mathbf { c } _ { t _ { \operatorname* { m i n } } } ^ { ( k ) } | |\tag{4}
$$

However, this metric alone does not indicate whether an observed topic shift reflects a meaningful directional drift or merely high-dimensional sampling variance. We address this in two ways. First, we assess whether observed topic displacements reflect systematic directional drift by comparing each group’s trajectory in embedding space to a random-walk null. We compute a log–likelihood ratio statistic Λ that tests whether the mean step is nonzero given the empirical step covariance: larger Λ indicates stronger evidence of persistent, directional movement rather than stochastic fluctuation, with significance quantified by permutation p-values. Full methodological details and derivations appear in Appendix C. We present significance tests for a selection of cases in our Results.

Second, we conduct a bootstrapped permutation test to ensure the observed drift is a stable phenomenon not introduced by batched, large-scale unsupervised clustering and that resampling recovers the same aligned groups with only small outliers. We present details and discussion of this secondary test in Appendix B.

## 3 Results

## 3.1 Topic Detection

Examining individual monthly windows through TF–IDF-weighted representations, we find that the k-means clusters exhibit internal coherence and semantic interpretability, corresponding to major domains of online discourse such as politics, computing, film, dating, travel, and science. Within domains, the method also identifies coherent substructure — for example, within politics, distinct clusters emerge for electoral events, media organizations, and political figures. This approach also isolates clusters defined by short or single-word responses (e.g., “ha,” “link”). Figure 2 illustrates these trends for two example periods, with topics labeled by their top-scoring keywords. Extending the alignment procedure across all time windows (2006–2022) yields K = 1005 persistent topic groups, the largest extending across 83 monthly clusters. The two-dimensional projections in Figures 2 and 3 illustrate that embedding distance corresponds closely to semantic relatedness: topics concerning the arts (music, film, literature) cluster together, as do those on prices and the economy, while more subtle associations — such as the proximity of discussions on therapy, food, and loneliness — suggest deeper thematic and contextual linkages.

![](images/4078b0e0bcc3777ef407f73949fe0839dd4b17ec6ce0bbbae321d9b0ba9ce219.jpg)  
Figure 3: Aligned topic groups over all time periods of the Reddit corpus (2006-2022) of 12.7 billion comments, using comment-level embeddings, with each dot representing just the topic centroid of a particular time period; labeling is the top shared word across multiple topic groups (using TF-IDF). Visualization uses a projection to 2-D using UMAP.

## 3.2 Topic Drift

The visualization in Fig. 3 indicates that many topics are not static in embedding space over time, suggesting that their underlying semantic content evolves even when clusters remain sufficiently similar to be aligned within the same topic group. Figure 6 illustrates this phenomenon, showing pronounced movement among politically oriented topics compared to the relative stability of a group like “news.” To quantify this behavior in the full embedding space $( d = 3 8 4 )$ , we compute each topic group’s total displacement $\delta _ { k } ,$ , reported in Table 1. As expected, volatile domains such as politics and conspiracy theory exhibit the largest drift, while topics such as food or music remain nearly stationary.

![](images/72ab81fbe8fade5733c5839d3c417dfa1d87edd02296a76d603d99d1825f1146.jpg)  
(a)

![](images/78a9b061e37e9d14d5b0a9ab56379082039783635f4f7ab6fa08df2df8081519.jpg)

(b) “laugh”  
![](images/6f823f26bad224edb3b04c77574b584f862959851c7223ccdf8a3579cd34f3ac.jpg)  
(c) “religion”  
Figure 4: (a) Visualization of two regimes of topic groups: those with a significant level of drift (in color) and those without (in gray). (b) A topic group related to “laughing” exhibiting static movement around a center, with non-significant levels of drift $( \Lambda = 3 . 7 4 , p = 0 . 1 5 )$ . (c) A “religion” topic group with significant levels of drift $( \Lambda = 8 . 2 9 , p = 0 . 0 0 1 )$ . All plots are presented in their 2-D UMAP projection.

Using the random walk framework described in the Methods section, we estimate a likelihood ratio statistic $\Lambda _ { k }$ to test whether each group’s trajectory exhibits systematic (nonzero-mean) drift. Two regimes emerge: topic groups with low $\Lambda _ { k }$ values that fail to reject the null hypothesis of zeromean movement, and those with significant directional drift $( p < 0 . 0 5 )$ . The correlation between $\Lambda _ { k } , p \ d t$ -values, and displacement $\delta _ { k }$ highlights that some topics move considerably without coherent directionality, whereas others display persistent, directional evolution in meaning over time (Fig. 4). These results indicate that certain thematic domains undergo significant semantic change (like the groups on government, science, or religion all with large Λ and corresponding p-value < 0.01), while others remain more stable (like music or sports or humor).

## 3.3 Inter-topic Drift

We next examine inter-topic dynamics to determine whether the relative positions of topics in embedding space change systematically over time. Because distance encodes contextual similarity, such movement reflects evolving relationships in discourse — i.e., if a topic of conversation becomes more or less similar to another topic in its semantics. Figure 5 reports the mean annual change in pairwise distance within the d = 384 embedding space, restricted to topic pairs within a radius of 0.4 to ensure semantic relevance.

As shown in Fig. 5(a), the topic labeled by “racism” has become increasingly proximate to discussions of police, religion, social issues, and women, while distancing from topics on Europeans. A parallel pattern appears in Fig. 5(b), where “streaming” converges toward topics concerning music, video, and mobile technology. Figure 6 further shows several political topics drifting toward an unoccupied region of embedding space, consistent with the emergence of new contextual associations. Together, these results indicate that semantic change occurs not only within topics but also in their relative configuration, reflecting structural evolution in the organization of discourse.

![](images/91be60bc2fa3a363e2cbc6cf47cbabc5f5f3c7737e32ce609add6917af1d0bcc.jpg)  
(a) Change relative to “racism”

![](images/dca833a19b2594d73eea4cb2e8cb669eac7d44c51d78ca8fef2bc402dc06bddc.jpg)  
(b) Change relative to “streaming”

Figure 5: Average annual change in centroid distance between a selected topic and other nearby topics, annotated by the topic group’s top shared keyword across all time periods. Negative values (blue) mean the topic group is growing closer to the selected topic in embedding space, implying a change in context and semantics.
<table><tr><td>Topic k</td><td></td><td> $\delta _ { k }$   $\Lambda _ { k }$ </td><td></td><td> $p _ { k }$ </td></tr><tr><td rowspan="3">top</td><td>&quot;conspiracy&quot;</td><td>0.387</td><td>0.277</td><td rowspan="3">0.285 0.295</td></tr><tr><td>“oil”</td><td>0.374</td><td>0.794 8.656</td></tr><tr><td>“scientific” 66 government&quot;</td><td>0.354 0.353</td><td>0.003 2.776 0.004</td></tr><tr><td rowspan="5">bottom</td><td>“republican&quot; &quot;woman&quot;</td><td>0.351 0.032</td><td>2.525 0.142</td><td>0.119 0.901</td></tr><tr><td>“music”</td><td>0.038</td><td>0.052</td><td>0.897</td></tr><tr><td>“team”</td><td>0.044</td><td>0.113</td><td>0.942</td></tr><tr><td>“ “games&quot;</td><td>0.050</td><td>0.061</td><td>0.903</td></tr><tr><td>“delicious”</td><td>0.054</td><td>0.551</td><td>0.339</td></tr></table>

Table 1: Top and bottom-ranked aligned topic groups ordered by total displacement $( \delta _ { k } )$ in embedding space; log-likelihood ratio $( \Lambda _ { k } )$ and computed $p \mathrm { - }$ -value $( p _ { k } )$ of nonzero mean drift are also presented. We see movement in inherently controversial topics like conspiracy theories or politics, while sports or music discussion remains surprisingly stable.

## 4 Discussion

Modern language models enable the representation of language with geometric structure, directly capturing contextual and semantic meaning in an embedded space. This perspective reframes topic modeling as an analysis of movement — how discourse itself traverses and reshapes semantic space over time. Prior work has explored topic dynamics at the sentence level (Grootendorst, 2022) and lexical drift at the word level (Bamler and Mandt, 2017); here, we extend both by tracing comment-level trajectories within a large-scale corpus of 12.7 billion Reddit comments. The results demonstrate that discourse change manifests as measurable geometric motion, both within topics and in their relative configuration.

![](images/fd25dc54b92efb9fdac90f29b01b06a8a4a7a5b7900f0af88d399a1775a84867.jpg)  
Figure 6: Visualization of several aligned topic groups in a select region of embedding space. We observe 3-4 topic groups that appear to drift toward a central shared area of embedding space while others move away.

This finding carries practical and theoretical significance: computational systems that assume topic stationarity (e.g., retrieval, moderation, classifier calibration) will degrade unless they incorporate temporal geometry, while the inter-topic structure reveals cultural re-association over time. For instance, the convergence of a “racism” related topic group toward groups related to “police,” “religion,” and “women,” or of “streaming” toward “music,” “video,” and “phones,” reflects durable shifts in meaning rather than transient lexical overlap, offering a quantitative view of how online discourse reconfigures itself.

Three particularly notable patterns emerge from this analysis. First, directional drift is selective, as contentious domains exhibit significant nonzeromean trajectories (high $\Lambda _ { k }$ , low $p _ { k } )$ , whereas stable domains show near-zero displacement. Second, relational reorganization occurs over time, as intertopic distances change systematically, suggesting discourse evolves through a reconfiguration of the semantic network rather than independent topic shifts (Fig. 5). Third, several political topics drift toward previously unoccupied regions of the space (Fig. 6), indicating convergence toward new contextual meanings not captured by static taxonomies. Together, these results imply that topic movement in embedding space reflects real semantic evolution — quantifiable changes in how ideas relate to one another and the emergence of new conceptual alignments over time.

Methodologically, treating topics as trajectories with uncertainty-aware tests advances beyond keyword tracking by furnishing falsifiable claims about semantic change. Substantively, the geometry illuminates when concepts gravitate toward new associations (e.g., policing with racism) and when domains decouple, offering evidence for theories of polarization, agenda setting, and technological diffusion. Practically, the statistics we report (displacement, $\Lambda _ { k }$ , neighborhood drift) can serve as early-warning indicators for model and policy interventions: large positive neighborhood drift flags domains where curated lexicons or classifier boundaries will become obsolete, while stable regimes justify longer retraining cycles. These use cases point to forecasting and causal attribution as next steps, including dynamical models of attractors and exogenous shocks that can explain—and predict—the observed reconfiguration of discourse.

## 4.1 Limitations & Future work

Several limitations of the present analysis suggest directions for future research. First, the decision to include the complete Reddit comment corpus with minimal filtering was motivated by a desire for maximal coverage. This, however, entails the presence of automated or low-quality content, whose prevalence has grown substantially over time and may contribute residual noise to the inferred semantic dynamics. Future work could incorporate classifiers to identify or weight such content, allowing assessment of its influence on topic trajectories. Second, although Reddit’s scale and topical heterogeneity make it an ideal testbed for largescale semantic modeling, its discourse conventions are distinctive. Applying this framework to more domain-specific corpora — such as scientific communication, news archives, or professional forums — would clarify the extent to which the observed dynamics generalize across linguistic communities.

A further avenue for improvement lies in methodological refinement of the analysis pipeline. Future work could employ more robust embedding models, enhanced clustering algorithms, or alignment techniques that explicitly account for temporal dependencies in topic structure. Incorporating multilingual or language-aware embeddings could enable a more faithful representation of global discourse and permit analysis of cross-lingual semantic drift.

Another limitation arises from the use of nondeterministic unsupervised clustering methods, such as our batched k-means approach and related algorithms like HDBSCAN (Grootendorst, 2022; Rahimi et al., 2024). Because these methods are sensitive to data permutations that can influence topic alignment and downstream trajectory estimates, it was necessary to employ both a nullmodel comparison for statistical validation and bootstrapping tests (Appendix B) to assess the robustness of our results. This is both computationally expensive as well as challenging to implement.

This work emphasizes the interpretability and scalability of transformer-based embeddings rather than the formal generative structure of probabilistic topic models (Wu et al., 2024; Churchill and Singh, 2022). This design enables fine-grained, large-scale analysis of discourse but sacrifices some of the inferential depth available in statistical modeling. Future work may bridge these paradigms by combining embedding-based representations with probabilistic or hybrid models to better capture the mechanisms that drive topic formation and semantic change. A dynamical systems perspective, for instance, might reveal whether certain regions of the embedding space act as attractors or repellers. Moreover, integrating datasets with well-defined thematic boundaries — such as extremist language — could provide insights into the forces shaping topic trajectories.

## 4.2 Conclusion

In this work, we presented a novel application of dynamic topic modeling to detect and quantify topic drift at the sentence level within a large-scale corpus. By embedding 12.7 billion Reddit comments (2006–2022) in a contextualized semantic space, applying unsupervised clustering, and aligning clusters over different time periods we were able to explore how different topic areas drifted across semantic space over time. Our main contribution is a quantitative framework for measuring these shifts directly in embedding space, enabling the detection of significant directional drift both within and between topics.

Beyond methodological novelty, the findings reveal that linguistic and cultural change can be observed as measurable geometric motion — an empirical signature of how public discourse reorganizes itself. Topics in politics, social identity, and religion show strong directional drift, whereas domains such as sports and music remain semantically stable, illustrating that social contestation leaves a traceable imprint in language. This framework thus bridges computational linguistics and social theory, offering a reproducible, data-driven approach to studying semantic evolution, polarization, and cultural realignment at scale. Future work may extend these methods toward predictive modeling, identifying the emergence of new conceptual “attractors” and forecasting the trajectories of public discourse across digital platforms.

## References

Mohiuddin Ahmed, Raihan Seraj, and Syed Mohammed Shamsul Islam. 2020. The k-means algorithm: A comprehensive survey and performance evaluation. Electronics, 9(8):1295.

Alexandra Amado, Paulo Cortez, Paulo Rita, and Sérgio Moro. 2018. Research trends on big data in marketing: A text mining and topic modeling based literature analysis. European Research on Management and Business Economics, 24(1):1–7.

Robert Bamler and Stephan Mandt. 2017. Dynamic word embeddings. In International conference on Machine learning, pages 380–389. PMLR.

Jason Baumgartner, Savvas Zannettou, Brian Keegan, Megan Squire, and Jeremy Blackburn. 2020. The pushshift reddit dataset. In Proceedings ofthe international AAAI conference on web and social media, volume 14, pages 830–839.

Arnab Bhadury, Jianfei Chen, Jun Zhu, and Shixia Liu. 2016. Scaling up dynamic topic models. In Proceedings of the 25th International Conference on World Wide Web, pages 381–390.

David M Blei and John D Lafferty. 2006. Dynamic topic models. In Proceedings of the 23rd international conference on Machine learning, pages 113–120.

David M Blei, Andrew Y Ng, and Michael I Jordan. 2003. Latent dirichlet allocation. Journal ofmachine Learning research, 3(Jan):993–1022.

Stefan Bunk and Ralf Krestel. 2018. Welda: Enhancing topic models by incorporating local word context. In Proceedings ofthe 18th ACM/IEEE on Joint Conference on Digital Libraries, pages 293–302.

Rob Churchill and Lisa Singh. 2022. The evolution of topic modeling. ACM Computing Surveys, 54(10s):1– 35.

Adji B Dieng, Francisco JR Ruiz, and David M Blei. 2019. The dynamic embedded topic model. arXiv preprint arXiv:1907.05545.

Adji B Dieng, Francisco JR Ruiz, and David M Blei. 2020. Topic modeling in embedding spaces. Transactions ofthe Associationfor Computational Linguistics, 8:439–453.

Alina Fastowski and Gjergji Kasneci. 2024. Understanding knowledge drift in llms through misinformation. In International Workshop on Discovering Drift Phenomena in Evolving Landscapes, pages 74–85. Springer.

Yue Fei, Yihong Hong, and Jianwu Yang. 2015. Handling topic drift for topic tracking in microblogs. In Advances in Information Retrieval: 37th European Conference on IR Research, ECIR 2015, Vienna, Austria, March 29-April 2, 2015. Proceedings 37, pages 477–488. Springer.

Yang Gao, Yue Xu, Heyan Huang, Qian Liu, Linjing Wei, and Luyang Liu. 2019. Jointly learning topics in sentence embedding for document summarization. IEEE Transactions on Knowledge and Data Engineering, 32(4):688–699.

Maarten Grootendorst. 2022. Bertopic: Neural topic modeling with a class-based tf-idf procedure. arXiv preprint arXiv:2203.05794.

Lei Guo, Chris J Vargo, Zixuan Pan, Weicong Ding, and Prakash Ishwar. 2016. Big social data analytics in journalism and mass communication: Comparing dictionary-based text analysis and unsupervised topic modeling. Journalism & Mass Communication Quarterly, 93(2):332–359.

Hossein Hassani, Christina Beneki, Stephan Unger, Maedeh Taj Mazinani, and Mohammad Reza Yeganegi. 2020. Text mining in big data analytics. Big Data and Cognitive Computing, 4(1):1.

Valentin Hofmann, Janet Pierrehumbert, and Hinrich Schütze. 2021. Dynamic contextualized word embeddings. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 6970–6984.

Francesca Incitti, Federico Urli, and Lauro Snidaro. 2023. Beyond word embeddings: A survey. Information Fusion, 89:418–436.

Tomoharu Iwata, Takeshi Yamada, Yasushi Sakurai, and Naonori Ueda. 2010. Online multiscale dynamic topic models. In Proceedings of the 16th ACM SIGKDD international conference on Knowledge discovery and data mining, pages 663–672.

Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofNAACL-HLT, volume 1, page 2.

MHUR Khan and Kei Wakabayashi. 2021. Drifting and popularity: a study of time series analysis of topics. In The Seventh International Conference on Big Data, Small Data, Linked Data and Open Data, pages 16–22.

Dan Knights, Michael Mozer, and Nicolas Nicolov. 2009. Detecting topic drift with compound topic models. In Proceedings of the International AAAI Conference on Web and Social Media, volume 3, pages 242–245.

Rahul Krishnan, Dawen Liang, and Matthew Hoffman. 2018. On the challenges of learning with inference networks on sparse, high-dimensional data. In International Conference on Artificial Intelligence and Statistics, pages 143–151. PMLR.

Peipei Li, Lu He, Haiyan Wang, Xuegang Hu, Yuhong Zhang, Lei Li, and Xindong Wu. 2017. Learning from short text streams with topic drifts. IEEE Transactions on Cybernetics, 48(9):2697–2711.

Quanchao Liu, Heyan Huang, and Chong Feng. 2013. Micro-blog post topic drift detection based on lda model. In International Workshop on Behavior and Social Informatics and Computing, pages 106–118. Springer.

David JC MacKay. 2003. Information theory, inference and learning algorithms. Cambridge University Press, Cambridge, UK.

Yishu Miao, Lei Yu, and Phil Blunsom. 2016. Neural variational inference for text processing. In International Conference on Machine Learning, pages 1727–1736. PMLR.

Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. Advances in Neural Information Processing Systems, 26.

Jeffrey Pennington, Richard Socher, and Christopher D Manning. 2014. Glove: Global vectors for word representation. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 1532–1543.

Francesco Periti and Stefano Montanelli. 2024. Lexical semantic change through large language models: a survey. ACM Computing Surveys.

James Petterson, Wray Buntine, Shravan Narayanamurthy, Tibério Caetano, and Alex Smola. 2010. Word features for latent dirichlet allocation. Advances in Neural Information Processing Systems, 23.

Hamed Rahimi, Hubert Naacke, Camelia Constantin, and Bernd Amann. 2024. Antm: Aligned neural topic models for exploring evolving topics. In Abdelkader Hameurlain, Josef Küng, and Roland R. Wagner, editors, Transactions on Large-Scale Dataand Knowledge-Centered Systems LVI, volume 14656 of Lecture Notes in Computer Science, pages 76–97. Springer.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982–3992.

Michael Röder, Andreas Both, and Alexander Hinneburg. 2015. Exploring the space of topic coherence measures. In Proceedings ofthe Eighth ACM International Conference on Web Search and Data Mining, pages 399–408.

Andrei Rykov, Renato Cordeiro De Amorim, Vladimir Makarenkov, and Boris Mirkin. 2024. Inertia-based indices to determine the number of clusters in kmeans: an experimental evaluation. IEEE Access, 12:11761–11773.

Helen Santos, Anthony Schmidt, Caspian Dimitrov, Christopher Antonucci, and Dorian Kuznetsov. 2024. Adaptive contextualization in large language models using dynamic semantic drift encoding. Authorea Preprints.

Robert H Shumway, David S Stoffer, and David S Stoffer. 2000. Time series analysis and its applications, volume 3. Springer, New York.

Akash Srivastava and Charles Sutton. 2017. Autoencoding variational inference for topic models. In 5th International Conference on Learning Representations.

Seyyed Reza Taher Harikandeh, Sadegh Aliakbary, and Soroush Taheri. 2023. An embedding approach for analyzing the evolution of research topics with a case study on computer science subdomains. Scientometrics, 128(3):1567–1582.

Anton Thielmann, Arik Reuter, Christoph Weisser, Gillian Kant, Manish Kumar, and Benjamin Säfken. 2024. Stream: Simplified topic retrieval, exploration, and analysis module. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 435– 444.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all

you need. Advances in neural information processing systems, 30.

Ike Vayansky and Sathish AP Kumar. 2020. A review of topic modeling methods. Information Systems, 94:101582.

Chong Wang, David Blei, and David Heckerman. 2008. Continuous time dynamic topic models. In Proceedings ofthe Twenty-Fourth Conference on Uncertainty in Artificial Intelligence, pages 579–586.

Xiaobao Wu, Thong Nguyen, and Anh Tuan Luu. 2024. A survey on neural topic models: methods, applications, and challenges. Artificial Intelligence Review, 57(2):18.

Yu Xie, Bin Zhou, and Yang Ou. 2019. A method based on sentence embeddings for the sub-topics detection. In Journal of Physics: Conference Series, 5, page 052004. IOP Publishing.

Zijun Yao, Yifan Sun, Weicong Ding, Nikhil Rao, and Hui Xiong. 2018. Dynamic word embeddings for evolving semantic discovery. In Proceedings of the eleventh ACM International Conference on Web Search and Data Mining, pages 673–681.

Jinhui Yuan, Fei Gao, Qirong Ho, Wei Dai, Jinliang Wei, Xun Zheng, Eric Po Xing, Tie-Yan Liu, and Wei-Ying Ma. 2015. Lightlda: Big topic models on modest computer clusters. In Proceedings ofthe 24th International Conference on World Wide Web, pages 1351–1361.

Delvin Ce Zhang and Hady Lauw. 2022. Dynamic topic models for temporal document networks. In International Conference on Machine Learning, pages 26281–26292. PMLR.

Zihan Zhang, Meng Fang, Ling Chen, and Mohammad-Reza Namazi-Rad. 2022. Is neural topic modelling better than clustering? an empirical study on clustering with contextual embeddings for topics. In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 3886–3893.

He Zhao, Dinh Phung, Viet Huynh, Trung Le, and Wray Buntine. 2021. Neural topic model via optimal transport. In International Conference on Learning Representations 2022. OpenReview.

Ce Zhou, Qian Li, Chen Li, Jun Yu, Yixin Liu, Guangjing Wang, Kai Zhang, Cheng Ji, Qiben Yan, Lifang He, et al. 2024. A comprehensive survey on pretrained foundation models: A history from bert to chatgpt. International Journal ofMachine Learning and Cybernetics, pages 1–65.

## Appendix A Cluster selection

Our analysis pipeline, a modification of similar methodology in (Reimers and Gurevych, 2019; Rahimi et al., 2024), uses a k-means clustering step following time windowing and before alignment. We chose to work in the full dimensional space of the sentence embedding in order to avoid prohibitively expensive global dimensionality reduction across the entire dataset, and here present results supporting our choice of k. In the subsequent Appendix we examine the stability of the clusters.

We use the metric of average within-cluster sumof-squares (WCSS), sometimes termed inertia (e.g. (Rykov et al., 2024)), as a metric to evaluate appropriate selection of k (Ahmed et al., 2020). This is simply the average distance between a datapoint and its assigned centroid, or more formally, for a particular time period $t \in \mathcal T$

$$
\mathrm { W C S S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } | | \mathbf { e } _ { i } - \mathbf { c } _ { i } | | ^ { 2 }\tag{5}
$$

where we are using $\mathbf { c } _ { i }$ as shorthand for the centroid within t corresponding to the embedding $\mathbf { e } _ { i }$

We computed this metric for each of the 204 time periods from 2006-2022, across a small grid of $k = 3 0 , 5 0 , 7 0 , 9 0$ . Results are shown in Figure 7. We note remarkable consistency in average inertia across time periods, especially for all time periods 2009 and on. This indicates a single choice of k is appropriate for all time periods (as opposed to varying k in each time period).

Next, we computed this metric for a single time period across a larger grid of $k ,$ shown in Figure 8. We note a quantitatively sharper decrease in $1 \leq k \leq 5 0$ , with diminishing returns after $k \approx 5 0 .$ leading to a selection of $k = 5 0$ — this technique is sometimes termed the “elbow” criterion (Rykov et al., 2024). Given this and the similarity of results in Figure 7, we used $k = 5 0$ for all time periods for the analysis pipeline in the body of the paper.

## Appendix B Cluster validation

We also explored validation of the stability of the clusters themselves. Working with such large data necessitates batched and distributed approaches that by nature introduce stochasticity into the otherwise deterministic algorithms underlying k-means (Ahmed et al., 2020). We seek to validate that the drift within an aligned topic group is not attributable to noisiness introduced in our batch scheme.

![](images/59ee7dc4f8f5088bd8881a7e18978869867d1d3e152bd8d9e6185a8ac1b52a9c.jpg)  
Figure 7: Average within-cluster sum-of-squares (WCSS) for a grid of $k = 3 0 , 5 0 , 7 0 , 9 0$ across all time periods (excluding 2014-2015).

![](images/4da5ef28ccd5f77f7c3e600511baec30d6eb855e7dd761aaf54d1d518f03577f.jpg)  
Figure 8: Average within-cluster sum-of-squares (WCSS) for a grid of $1 \leq k \leq 2 2 0$ on a single demonstration time period (April 2017).

To address this, we applied permutation testing to our pipeline. Specifically, we considered a single time window of data from a central portion of the entire 17-year corpus, resampled it with replacement R times within batches of size M, to create R sets of unlabeled embeddings. We then treat these as R distinct time windows and apply our pipeline: we perform k-means clustering on each, then a global dimension reduction on the $R \times k$ learned centroids, then we align similar centroids into G topic groups.

In a deterministic clustering, with a fixed random seed in initialization, we would expect each permutation to have the exact same centroids, and the resulting aligned groups to be an exact set of $G = k$ groups, each group consisting of R centroids with identical coordinates. This is the situation in Figure ${ 9 } ( \mathrm { a } )$ where we use a single batch over the entire time window and a fixed seed during each initialization (with $R = 2 0 , k = 5 0$ , and a recovered $G = 5 0 )$ . As further validation, we apply the random walk comparison test outlined in the paper, and find zero groups with p-values below 0.05.

Although we are able to follow this deterministic approach for smaller quantities of embeddings, this becomes prohibitive as the size of the time window grows. In the dataset of this paper, this transition happens around 2015 as the size of each time window grows from hundreds of thousands to tens of millions. We adopt a batch size of $M = 1$ million embeddings and repeat the permutation testing, with the result depicted in Figure 9(b). We still recover $G = 4 9$ , though now with some centroids $( \approx 7 \%$ of the data) being classified as outliers. A visual inspection of the 2-dimensional UMAP plot can of course be deceiving, so we again apply the random walk comparison, and find zero groups with p-values below 0.01, and only 4 groups with p-values below 0.05.

We conclude that although our methodology introduces potential randomness at the clustering step, in large datasets, we here demonstrate is minimized through reasonable batching schemes and filterable by our random walk LRT approach. We note that the methods proposed in (Rahimi et al., 2024) also introduce potential noise at the dimensionality reduction step prior to clustering, by performing a global dimension reduction using a variant of UMAP called Aligned-UMAP, which we also find prohibitive for datasets at the scale used in this paper.

![](images/36c64eaf8219adb5da4566167608984e1094950d7787ba681d7a39559fe30d25.jpg)

![](images/ffe10edc01df9365346a12758d9c782d7aac69b5839f9e7907fea2de607ec52d.jpg)  
(a) Shared initialization  
(b) Different initialization  
Figure 9: Aligned topic groups for the same resampled time period, using a single shared centroid initialization in each permutation (a) and different initialization (b). Despite variation in centroids between permutations, the method recovers groups that are nearly an exact match for the original clusters, and inspection of the random walk test under both initialization schemes shows no groups have p-values below 0.01.

## Appendix C Random Walk & LR test

This appendix details the test employed to determine the significance of the observed topic displacement over time through comparison to a random walk; see (Shumway et al., 2000) for a comprehensive treatment.

Let $\{ \mathbf { c } _ { t } \} _ { t = 1 } ^ { T }$ denote the sequence of topic centroids for a given group, where each $\mathbf { c } _ { t } \in \mathbb { R } ^ { d }$ and $T \geq 2$ . Define the step (displacement) vectors:

$$
\Delta _ { t } \equiv \mathbf { c } _ { t + 1 } - \mathbf { c } _ { t } \in \mathbb { R } ^ { d } , \qquad t = 1 , \dots , T - 1 .
$$

Assume $\Delta _ { t } \overset { \mathrm { i . i . d . } } { \sim } \mathcal { N } ( \pmb { \mu } , \pmb { \Sigma } )$ with unknown mean $\pmb { \mu } \in$ $\mathbb { R } ^ { d }$ and positive-definite covariance $\pmb { \Sigma } \in \mathbb { R } ^ { d \times d }$ We test:

$$
\begin{array} { r l } & { H _ { 0 } : \ \mu = 0 \quad ( \mathrm { n o \ d i r e c t i o n a l \ d r i f t } ) , } \\ & { H _ { 1 } : \ \mu \neq { \bf 0 } \quad ( \mathrm { s y s t e m a t i c \ d r i f t } ) . } \end{array}
$$

Let $\scriptstyle { \mathcal { L } } ( \mu \mid \Delta _ { 1 : T - 1 } )$ denote the likelihood of the observed steps $\Delta _ { 1 } , \ldots , \Delta _ { T - 1 }$ under mean $\pmb { \mu }$ and covariance Σ:

$$
\mathcal { L } ( \pmb { \mu } | \Delta _ { 1 : T - 1 } ) = \prod _ { t = 1 } ^ { T - 1 } \mathcal { N } ( \Delta _ { t } ; \pmb { \mu } , \pmb { \Sigma } ) .
$$

The log–likelihood ratio comparing $H _ { 1 }$ to $H _ { 0 }$ simplifies to

$$
\Lambda = { \frac { T - 1 } { 2 } } \bar { \mu } ^ { \top } \Sigma ^ { - 1 } \bar { \mu } , \qquad \bar { \mu } \equiv { \frac { 1 } { T - 1 } } \sum _ { t = 1 } ^ { T - 1 } \Delta _ { t } ,
$$

where $\bar { \pmb { \mu } }$ is the empirical mean step, and Λ is the log–likelihood ratio statistic comparing $H _ { 1 }$ to $H _ { 0 }$ In practice, Σ is replaced by the unbiased sample covariance

$$
\hat { \pmb { \Sigma } } = \frac { 1 } { ( T - 1 ) - 1 } \sum _ { t = 1 } ^ { T - 1 } \bigl ( \Delta _ { t } - \bar { \pmb { \mu } } \bigr ) \bigl ( \Delta _ { t } - \bar { \pmb { \mu } } \bigr ) ^ { \top } .
$$

To stabilize estimation, the steps are projected via principal component analysis (PCA) onto the first $r \ : = \ : 4 5$ components (explaining > 90% of variance). Denote the projected steps by $\tilde { \Delta } _ { t } \ \in$ $\mathbb { R } ^ { r }$ and the corresponding covariance by $\hat { \Sigma } _ { r } \mathrm { : }$ ; the statistic $\Lambda$ is then computed in $\mathbb { R } ^ { r }$ by replacing $( \bar { \mu } , \hat { \Sigma } )$ with $( \bar { \mu } _ { r } , \hat { \Sigma } _ { r } )$

For each topic group indexed by $k \_ { } \in \{$ $\{ 1 , \ldots , K \}$ , let $\Lambda _ { k }$ be the observed statistic from its time-ordered steps. A permutation null is obtained by randomly shuffling the time order of $\{ \tilde { \Delta } _ { t } \}$ within group k to generate $B \ = \ 1 0 { , } 0 0 0$ replicates, yielding $\{ \Lambda _ { k } ^ { ( b ) } \} _ { b = 1 } ^ { B }$ . The Monte Carlo p-value is

$$
p _ { k } \ = \ \frac 1 B \sum _ { b = 1 } ^ { B } \mathbf 1 \Big \{ \Lambda _ { k } ^ { ( b ) } \ge \Lambda _ { k } \Big \} ,
$$

where 1{·} denotes the indicator function. Additional diagnostics are reported in Appendix B.