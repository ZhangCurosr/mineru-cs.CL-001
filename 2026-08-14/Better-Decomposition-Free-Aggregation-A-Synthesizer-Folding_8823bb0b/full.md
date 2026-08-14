# Better Decomposition, Free Aggregation: A Synthesizer-Folding Framework for Multilingual Multi-Hop Question Answering

Yilin Wang<sup>1[0009−0009−2393−2330]</sup>, Yuchun Fan<sup>1</sup>, Weidong Bao<sup>1</sup>, Zili Wei<sup>1</sup>, Shi Feng<sup>1</sup>, Tong Xiao<sup>1</sup>, Zhengtao Yu<sup>2</sup>, and Jingbo Zhu<sup>1†</sup>

<sup>1</sup> School of Computer Science and Engineering, Northeastern University, Shenyang, China

<sup>2</sup> Yunnan Key Laboratory of Artificial Intelligence, Kunming University of Science and Technology, Kunming, China wangyilin0409@gmail.com

Abstract. Multilingual retrieval-augmented generation (mRAG) equips large language models with access to globally distributed external knowledge for complex multilingual question answering. Recent approaches either translate retrieved documents into English or the query language to bridge the cross-lingual semantic gap, or decompose a complex query into sub-questions and aggregate the intermediate reasoning process. However, both lines of work sufer from two limitations. First, one-size-fitsall translation alignment, blanket translation discards culturally and linguistically native information unique to the target language, introduces translation noise, and inflates system cost. Second, greedy decomposition and aggregation, uncontrolled decomposition produces redundant sub-questions that compound errors during step-wise reasoning, and the final aggregation over reasoning paths further amplifies these errors. We address both with our method Syfer, a synthesizerfolding framework for multilingual multi-hop question answering that defers translation rather than applying it by default. Syfer first invokes a format-constrained decomposer to produce a sub-question graph in the original language, followed by a decomposition-quality check; when the check passes, sub-questions are answered sequentially under a retrievethen-answer policy in the target language, and the English translation pathway with bilingual sub-question graph alignment is activated only when the check fails. Experiments across multiple languages show that Syfer attains competitive accuracy while striking a favourable balance between performance and computational cost. The code and data will be available at https://github.com/f6ster/Syfer

Keywords: Multilingual question answering · Multi-hop reasoning · Question decomposition · Retrieval-augmented generation · Large language model.

![](images/308e5ff648f7859ca2c33869a96a185121468bf07ae32af8dfbd731b8967ca42.jpg)  
Fig. 1. Two structural issues in prior decomposition-based mRAG and how Syfer addresses them. Top: one-size-fits-all translation injects noise, and an end-of-pipeline aggregation call concentrates intermediate errors. Bottom: Syfer reasons in the query language by default, falls back to a cross-lingual pathway only when faithfulness fails, and folds aggregation into a terminal sub-question.

## 1 Introduction

Retrieval-augmented generation (RAG) grounds large language models (LLMs) in external knowledge to improve factuality and mitigate hallucination on knowledgeintensive tasks [11–13, 16]. However, in the multilingual setting, the training distribution of LLMs is highly skewed toward English and other high-resource languages, leaving a persistent capability gap on mid- and low-resource languages [6, 7, 10, 30]. This motivates multilingual RAG (mRAG), which lets the model draw on multilingual collections to compensate for capability gaps and to recover information that is unevenly distributed across languages.

Existing mRAG systems handle single-hop multilingual queries well [3, 21] but struggle on multilingual multi-hop QA, where the LLM must also collect and integrate clues scattered across documents in diferent languages. The dominant remedy translates either the query or retrieved documents to align languages [18– 20]; a more refined line decomposes the query into a bilingual sub-question graph and aggregates per-hop answers [2, 25, 27], which delivers stronger interpretability and currently represents one of the strongest design points on this task

Despite these advances, two structural issues remain, illustrated in Figure 1. (i) One-size-fits-all translation: forcibly aligning native-language documents to the query language incurs a heavy translation cost and tends to inject translation noise that is later answer as evidence. (ii) Greedy decomposition and aggregation: uncontrolled decomposition produces a sub-question graph that is often redundant and logically incoherent, where minor errors in intermediate reasoning propagate and compound. Besides, the final aggregation call is logically independent of the decomposition process, it concentrates rather than absorbs decomposition noise, breaking the reasoning chain and yielding incorrect answers.

We address both issues with Syfer, a Synthesizer-Folding framework for multilingual multi-hop question answering. Syfer keeps the decompositionbased backbone but makes two changes. First, translation becomes decomposition driven: the framework reasons in the original language by default and activates the cross-lingual pathway only when the produced sub-question graph fails a quality check, after which an English-parallel graph is decomposed and fused for recovery. Second, decomposition becomes synthesizer-folded: a trained decomposer constrains the breadth and format of sub-questions and emits a terminal sub-question in the same logical layer as its peers. This terminal sub-question serves as a synthesis question over the prior reasoning chain, so the generator answers it directly instead of performing a separate aggregation over a long intermediate trace. As a result, aggregation is folded into decomposition and decomposition quality becomes directly checkable.

We further extend the multilingual multi-hop benchmarks introduced by [25] into a more comprehensive testbed spanning five language families and nine languages across high-, mid-, and low-resource regimes. Across three benchmarks and nine languages, Syfer not only attains competitive accuracy against strong baselines but also achieves a more favourable balance between performance, computational cost, and latency. The same experiments expose an interesting finding: unconditional fusion of an English-parallel sub-question graph into the reasoning trace is in fact sub-optimal on queries that decompose cleanly, since these queries already correspond to low-complexity reasoning, and a redundant English-side graph injects choices that confuse the generator rather than help it. In summary, the main contributions of this paper are as follows:

We propose Syfer, a multilingual RAG framework that introduces synthesizer folding, the end-of-pipeline aggregation call is replaced by a terminal sub-question emitted by the decomposer in the same logical layer as its peers, placing decomposition and aggregation in a single logical pass and making decomposition quality directly verifiable.

– We extend three multilingual multi-hop QA benchmarks into a unified testbed covering five language families and nine languages across high-, mid- and low-resource regimes, providing a comprehensive evaluation suite for future multilingual RAG systems.

– Across three benchmarks and nine languages, Syfer attains strong accuracy and a favourable cost-performance balance against competitive baselines. On the most challenging benchmark MuSiQue, Syfer still improves over the strongest decomposition-based baseline by +8.91 F1 (+29.8% relative) averaged over nine languages.

## 2 Related Work

## 2.1 RAG Systems for Multi-hop QA

Multi-hop question answering requires a RAG system to retrieve and integrate evidence scattered across multiple documents. Existing methods mainly follow two directions: iterative retrieval that progressively refines supporting evidence [14, 22, 24, 26], and structured retrieval over trees or graphs that guides the LLM toward clue-dense documents or relevant context spans [2, 5, 8, 27]. These methods, however, largely rely on the English-side understanding ability of the underlying LLM. In multilingual settings with parallel documents and mid/low-resource languages, such structures can introduce redundant evidence and amplify extraction noise, reducing downstream RAG efectiveness.

## 2.2 Multilingual RAG

Multilingual RAG (mRAG) has primarily improved cross-lingual access through translation and alignment strategies [3, 18–21, 30]. These approaches typically translate the query into a high-resource pivot language or translate retrieved documents into the query language. While useful for language-aligned tasks, indiscriminate translation injects noise into the evidence and increases inference cost. Closer to our setting, DaPT [25] applies question decomposition to multilingual multi-hop QA, but its unconstrained decomposition produces long intermediate contexts and remains vulnerable to lost-in-the-middle efects [17].

In contrast, Syfer keeps reasoning in the original language by default and activates cross-lingual machinery only when the in-language decomposition fails a quality check. It further constrains the decomposition format and folds aggregation into a terminal sub-question, reducing both unnecessary translation and dependence on long intermediate reasoning traces.

## 3 Methodology

Given a query $Q$ in language $L$ and a multilingual corpus ${ \mathcal { C } } ,$ Syfer retrieves evidence from $\mathcal { C }$ and uses the generator to produce the final answer A. As shown in Fig. 2, the framework consists of four stages: ofline logical-decomposition distillation, synthesizer-folded decomposition, faithfulness verification with bilingual fallback, and cross-lingual retrieval-and-answering. The following subsections describe these stages in order.

## 3.1 Logical Decomposition Distillation

A natural alternative is to invoke a frontier LLM to decompose every test query online, but this makes inference expensive and leaves decomposition quality hard to verify before answering. We instead distil a standalone decomposer on a corpus–query pool held out from evaluation, so that synthesizer folding is learned ofline.

The teacher model maps each query Q to an acyclic sub-question DAG $D ^ { * }$ with a unique terminal node and placeholder dependencies. We keep only decompositions whose filled terminal sub-question remains close to $Q$ in the retriever embedding space:

$$
\cos \bigl ( { \mathbf e } ( q _ { n } ^ { \mathrm { f i l l e d } } ) , ~ { \mathbf e } ( Q ) \bigr ) ~ \geq ~ \tau _ { \mathrm { c o n s t r a i n t } } ,\tag{1}
$$

![](images/be7ccec730d9faf9a7908511e1fb442b1172a4d4ec0945e005039e4761c604d1.jpg)  
Fig. 2. Overview of Syfer: synthesizer-folded decomposition, faithfulness verification with bilingual fallback, and cross-lingual retrieval-and-answering.

where $\mathbf { e } ( \cdot )$ is the retriever embedding. This filtering yields the supervision set ${ \mathcal { D } } _ { \mathrm { t r a i n } } = \dot { \{ ( Q ^ { ( k ) } , D ^ { * ( k ) } ) \} } _ { k = 1 } ^ { N }$

Each $D ^ { * }$ is linearised into a token sequence $y = ( y _ { 1 } , \dots , y _ { T } )$ that preserves node order and edge structure. The student decomposer with parameters $\theta$ is trained by token-level cross-entropy:

$$
\mathcal { L } _ { \mathrm { D i s t i l l } } ( \theta ) = - \mathbb { E } _ { ( Q , y ) \sim \mathcal { D } _ { \mathrm { t r a i n } } } \sum _ { t = 1 } ^ { T } \log p _ { \theta } \big ( y _ { t } | Q , y _ { < t } \big ) , \quad \theta ^ { * } = \operatorname * { a r g m i n } _ { \theta } \mathcal { L } _ { \mathrm { D i s t i l l } } ( \theta ) .\tag{2}
$$

The trained $\theta ^ { * }$ defines the inference-time decomposer in Section 3.2 and is reused, without branch-specific fine-tuning, by the bilingual fallback in Section 3.3.

## 3.2 Synthesizer-Folded Decomposition

Conventional multi-hop RAG ends with a dedicated aggregation call over $( Q , q _ { 1 : n } , a _ { 1 : n } )$ ， where most cross-lingual noise accumulates. Syfer removes this call by letting the decomposer emit a terminal sub-question that, once filled, already speaks for $Q .$ . Concretely, given $Q$ in language $L ,$

$$
D _ { L } \ = \ \mathrm { D e c o m p o s e } _ { \theta ^ { * } } ( Q ) , \qquad D _ { L } = ( V _ { L } , E _ { L } ) ,\tag{3}
$$

with $V _ { L } = \{ q _ { 1 } , \dots , q _ { n } \}$ and $E _ { L } \subseteq V _ { L } \times V _ { L }$ . An edge $( q _ { i } , q _ { j } ) \in E _ { L }$ indicates that the answer $a _ { i }$ is required to instantiate $q _ { j } ;$ substituting $a _ { i }$ for each placeholder $\# i \ : ( i < j )$ in $q _ { j }$ yields the filled form $q _ { i } ^ { \mathrm { f i l l e d } }$ . The graph is acyclic and exposes a unique terminal node $q _ { n }$ . Trained under Eq. (1), the terminal node is no longer an arbitrary leaf but a learned slot whose filled form encodes both Q and the prior sub-answers, so that answering it against the corpus is, by construction, equivalent to aggregating.

## 3.3 Faithfulness Verification and Bilingual Fallback

Even a well-trained decomposer is not fully reliable, for dificult queries, the filled terminal sub-question may drift away from the original query Q. Syfer therefore verifies the terminal sub-question before retrieval and uses bilingual fallback only when the monolingual decomposition appears unfaithful. Formally,

$$
\begin{array} { r l } & { \mathrm { s c o r e } ( D _ { L } ) = \cos \bigl ( { \mathbf e } ( q _ { n } ^ { \mathrm { f i l e d } } ) , { \mathbf e } ( Q ) \bigr ) , } \\ & { \mathrm { R o u t e } ( D _ { L } ) = \left\{ \begin{array} { l l } { D _ { L } , } & { \mathrm { s c o r e } ( D _ { L } ) \geq \tau _ { \mathrm { c o n s t r a i n t } } , } \\ { \mathrm { B i l i n g u a l F a l l b a c k } ( D _ { L } , Q ) , } & { \mathrm { s c o r e } ( D _ { L } ) < \tau _ { \mathrm { c o n s t r a i n t } } . } \end{array} \right. } \end{array}\tag{4}
$$

Thus, translation is not a default operation but a recovery path for queries whose in-language decomposition cannot be trusted.

When fallback is triggered, Syfer translates Q into English, decomposes the English query with the same decomposer, and aligns the English DAG with the original-language DAG by node similarity:

$$
\left( q _ { i } ^ { L * } , q _ { j } ^ { \mathrm { e n } * } \right) = \arg \operatorname* { m a x } _ { \substack { q _ { i } ^ { L } \in V _ { L } , q _ { j } ^ { \mathrm { e n } } \in V _ { \mathrm { e n } } } } \sin \mathopen { } \mathclose \bgroup \left( \mathbf { e } ( q _ { i } ^ { L } ) , \mathbf { e } ( q _ { j } ^ { \mathrm { e n } } ) \aftergroup \egroup \right) .\tag{5}
$$

Node pairs above the alignment threshold $\tau _ { \mathrm { a l i g n } }$ are fused, yielding a bilingual graph

$$
D _ { F } = \mathrm { F u s e } ( D _ { L } , D _ { \mathrm { e n } } ) ,\tag{6}
$$

## 3.4 Cross-Lingual Retrieval and Answering

The output of the previous stage is either a monolingual graph $D _ { L }$ or a fused bilingual graph $D _ { F }$ . Syfer solves either graph in topological order, so each sub-question is answered only after the sub-questions it depends on:

$$
S = ( v _ { 1 } , \ldots , v _ { m } ) = \operatorname { T o p o S o r t } ( D ) , \qquad D \in \{ D _ { L } , D _ { F } \} .\tag{7}
$$

For a monolingual node, Syfer retrieves with the target-language sub-question; for a bilingual node, it retrieves with both the target-language and English subquestions and merges the candidates:

$$
\begin{array} { r } { R _ { i } = \left\{ \begin{array} { l l } { \mathrm { R e t r i e v e } ( q _ { i } ^ { \mathrm { \mathrm { f i l e d } , } L } , \mathcal { C } ) , } & { v _ { i } \in V _ { L } \setminus V _ { \mathrm { b i } } , } \\ { \mathrm { R e t r i e v e } ( q _ { i } ^ { \mathrm { \mathrm { f i l e d } , } L } , \mathcal { C } ) \cup \mathrm { R e t r i e v e } ( q _ { j } ^ { \mathrm { \mathrm { f i l e d } , e n } } , \mathcal { C } ) , } & { v _ { i } \in V _ { \mathrm { b i } } . } \end{array} \right. } \end{array}\tag{8}
$$

Because the corpus contains document-aligned translations, the merged candidates may include several language versions of the same evidence. Syfer applies maximal marginal relevance (MMR) to keep documents that are both relevant to the current sub-question and diferent from documents already selected:

$$
d ^ { * } = \arg \operatorname* { m a x } _ { d \in R _ { i } \backslash S _ { i } } \lambda \sin ( \mathbf { e } ( q _ { i } ^ { \mathrm { { f i l e d } } } ) , \mathbf { e } ( d ) ) - ( 1 - \lambda ) \operatorname* { m a x } _ { d ^ { \prime } \in S _ { i } } \sin ( \mathbf { e } ( d ) , \mathbf { e } ( d ^ { \prime } ) ) ,\tag{9}
$$

where $S _ { i }$ is the selected document set and λ balances relevance and diversity. In plain terms, MMR prefers documents that match the sub-question while avoiding near-duplicate parallel translations.

Given the filtered evidence $R _ { i }$ , the generator produces one short answer per node, optionally using both language views for bilingual nodes:

$$
a _ { i } = \left\{ \begin{array} { l l } { \mathrm { A n s w e r } ( q _ { i } ^ { \mathrm { f i l e d } , L } , R _ { i } ) , } & { v _ { i } \in V _ { L } \setminus V _ { \mathrm { b i } } , } \\ { \mathrm { A n s w e r } ( q _ { i } ^ { \mathrm { f i l e d } , L } , q _ { j } ^ { \mathrm { f i l e d , e n } } , R _ { i } ) , } & { v _ { i } \in V _ { \mathrm { b i } } . } \end{array} \right.\tag{10}
$$

Each answer is substituted into later sub-questions, and the answer to the terminal node is returned as the final answer:

$$
A = a _ { m } .\tag{11}
$$

This completes the synthesizer-folded pipeline without a separate final aggregation call.

## 4 Experiments

## 4.1 Datasets

We evaluate Syfer on HotpotQA [29], 2WikiMultiHopQA (2Wiki) [9] and MuSiQue [23], using the same 1,000 query test split per benchmark as HippoRAG2 [8]. To assess multilingual mRAG behaviour more comprehensively, we use GPT-4o [15] to translate and extend the original English-only multihop test pool into nine languages spanning low, mid, and high-resource regimes across five language families. To ensure translation quality, we maintain a perdocument entity table during translation so that entity names remain consistent across questions, supporting passages, and sub-questions.

## 4.2 Baselines

We compare Syfer with five representative baselines of multilingual multi-hop QA: (i) Zero-shot LLM and (ii) Vanilla RAG, which answer without retrieval and with single-shot retrieval on the original query, respectively; (iii) HippoRAG2 [8], a strong structure-aware graph-RAG baseline operating over a triple-indexed knowledge graph; (iv) CrossRAG [20], a retrieve-then-translate baseline that first retrieves documents on the multilingual corpus and then translates the retrieved documents into the query language before generating the response; and (v) DaPT [25], a decomposition-based mRAG baseline that decomposes the query into a sub-question DAG and fuses each sub-question with its Englishparallel counterpart at every hop.

## 4.3 Metrics

We report Exact Match (EM) and token-level F1 on the predicted answer string against the reference answer in the query language. EM assigns a score of 1 only when the normalized prediction exactly matches the ground truth and 0 otherwise. F1 measures token-level similarity by computing the harmonic mean of Precision and Recall, where Precision reflects the proportion of predicted tokens that are correct and Recall denotes the proportion of reference tokens successfully retrieved. Following MuSiQue [23], predictions and references are normalized with language-specific tokenization, lowercasing, and punctuation rules so that scores are comparable across the nine languages.

## 4.4 Implementation Details

Backbones. To avoid model family preference, we use DeepSeek-V4 Pro [4] as the answering model for Syfer and for all baselines that require a generator. This decouples answer generation from both the decomposer family (Qwen) and the test-set translation model (GPT-4o), so that the three modeling layers do not share a common backbone and same-family confounders are ruled out in the cross-lingual evaluation.

Retrieval setup. We use BGE-m3 [1] as the multilingual retriever for all methods, indexing the union of the nine-language corpora and retrieving top- $k = 5$ documents per query. The faithfulness gating threshold of Syfer is $\tau _ { \mathrm { c o n s t r a i n t } } =$ 0.8, the cross-lingual node alignment threshold is $\tau _ { \mathrm { a l i g n } } = 0 . 6$ , and the MMR trade-of is $\lambda = 0 . 6$

Logical decomposition distillation. We sample a decomposer-training pool from the oficial training splits of the three benchmarks, disjoint from all test queries. Training-side translation and annotation are produced by Qwen3-235B-A22B-Instruct-2507 [28], used as the teacher model. For Syfer, we curate 59,688 decomposition records covering six in-distribution languages: English (En), Chinese (Zh), German (De), Spanish (Es), Swahili (Sw), and Thai (Th). French (Fr), Bengali (Bn), and Korean (Ko) are held out as out-of-distribution evaluation languages. We fine-tune Qwen3-8B [28] as the student decomposer for 2 epochs with a global batch size of 64 and learning rate of $2 \times 1 0 ^ { - 4 }$ . Training is conducted on 8×NVIDIA H800 80 GB GPUs with Intel(R) Xeon(R) Platinum 8468V CPUs.

## 4.5 Main Results

Table 1 reports EM and F1 of Syfer and the five baselines across nine languages on HotpotQA, 2WikiMultiHopQA, and MuSiQue. We organize the analysis around four observations.

Single step retrieval is fundamentally insuficient for multi-hop QA, regardless of language alignment. Vanilla RAG and CrossRAG both plateau at low EM/F1 across datasets and languages, and CrossRAG’s retrieve-thentranslate strategy fails to consistently improve over Vanilla RAG. The bottleneck is therefore not only cross-lingual mismatch but also one-shot retrieval’s inability to gather all evidence for a compositional query, motivating decomposition-based pipelines.

Table 1. EM and F1 scores across nine languages on the three multilingual multi-hop QA datasets we constructed. Avg. denotes the average score across languages. Bold values indicate the best performance within each group.
<table><tr><td rowspan="2">Method</td><td colspan="2">En</td><td colspan="2">Zh</td><td colspan="2">De</td><td colspan="2">Es</td><td colspan="2">Th</td><td colspan="2">Fr</td><td colspan="2">Bn</td><td colspan="2">K₀</td><td colspan="2">Avg.</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1 EM</td><td>F1</td><td>EM F1</td><td>EM</td><td>F1</td><td>EM F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td></td><td>F1 EM</td><td>F1</td></tr><tr><td colspan="10">HotpotQA</td><td colspan="10"></td></tr><tr><td>Zero-shot</td><td>18.6</td><td>28.2</td><td>10.2</td><td>27.9</td><td>13.1</td><td>21.0 14.3</td><td>23.6</td><td>8.8</td><td>15.4</td><td>13.2 40.0</td><td>15.1</td><td>22.7</td><td>8.2</td><td></td><td>14.0</td><td>7.7</td><td>23.7</td><td>12.1</td><td>24.1</td></tr><tr><td>Vanilla RAG</td><td>35.4</td><td>48.4</td><td>24.8</td><td>42.9</td><td>25.4</td><td>38.8 22.1</td><td>35.3</td><td>26.4</td><td>41.1</td><td>22.6</td><td>44.2</td><td>15.1</td><td>27.3</td><td>8.4</td><td>21.2</td><td>12.3</td><td>33.1</td><td>21.4</td><td>36.9</td></tr><tr><td>CrossRAG</td><td>34.6</td><td>47.5</td><td>13.4</td><td>30.2</td><td>24.6 37.5</td><td>22.3</td><td>35.3</td><td>25.8</td><td>40.5</td><td>21.5 41.4</td><td>16.3</td><td>27.7</td><td></td><td>5.3</td><td>18.2</td><td>8.9</td><td>26.9</td><td>19.2</td><td>33.9</td></tr><tr><td>HippoRAG2</td><td>44.9</td><td>57.9</td><td>22.6</td><td>47.8</td><td>36.4 49.4</td><td>35.9</td><td>51.1</td><td>34.6</td><td>46.1</td><td>20.3</td><td>45.6</td><td>33.8</td><td>47.2</td><td>22.3</td><td>35.7</td><td>15.9</td><td>36.8</td><td>29.6</td><td>46.4</td></tr><tr><td>DaPT</td><td>48.1</td><td>59.3</td><td>32.6</td><td>49.7</td><td>48.2</td><td>60.3 47.6</td><td>60.7</td><td>37.5</td><td>47.3</td><td>29.5</td><td>44.1</td><td>42.8</td><td>56.0</td><td>22.2</td><td>33.1</td><td>25.6</td><td>41.9</td><td>37.1</td><td>50.3</td></tr><tr><td>SYFER</td><td>57.5 69.9</td><td></td><td> 39.4 61.2</td><td></td><td>50.0 63.6</td><td>49.5</td><td>64.7</td><td>50.5</td><td>62.5</td><td>43.8</td><td>63.9</td><td>45.1</td><td></td><td>59.4 36.8</td><td>49.2</td><td></td><td></td><td></td><td>2 37.3 47.4 45.5 60.2</td></tr><tr><td colspan="10">2WikiMultiHopQA</td><td colspan="10"></td></tr><tr><td>Zero-shot 23.0</td><td colspan="10">25.5 30.9 24.3 34.6 17.5</td><td colspan="10"></td></tr><tr><td>Vanilla RAG</td><td colspan="10">17.8 24.9 8.6</td><td colspan="10">23.8 43.6</td></tr><tr><td>CrossRAG</td><td colspan="10"></td><td colspan="10">15.7 31.2</td></tr><tr><td></td><td colspan="10">19.2 25.4 5.4</td><td colspan="10">2.8 16.5 31.7</td></tr><tr><td>HippoRAG2</td><td colspan="10">56.0 65.0 21.5</td><td colspan="10">2.8 52.1</td></tr><tr><td>DaPT</td><td colspan="10">34.7 39.3 30.0 45.0</td><td colspan="10">42.5 34.7 41.3</td></tr><tr><td>SYFER 67.5</td><td colspan="10">75.2 42.8 61.7 64.2 72.4 64.6 73.2</td><td colspan="10">50.2 71.3 59.0 68.1</td></tr><tr><td>20.8 2.0 6.8</td><td colspan="10">MuSiQue</td><td colspan="10"></td></tr><tr><td>Zero-shot</td><td colspan="10">2.5 10.3 1.4</td><td colspan="10">30.0 1.3</td></tr><tr><td>Vanilla RAG</td><td colspan="10">19.7 30.1 10.0</td><td colspan="10">32.0 6.7</td></tr><tr><td>CrossRAG</td><td colspan="10">19.9 30.3</td><td colspan="10">28.4</td></tr><tr><td></td><td colspan="10">4.9</td><td colspan="10">6.5</td></tr><tr><td>HippoRAG2</td><td colspan="10">24.3 35.9 7.8</td><td colspan="10">16.7 35.6 16.2 29.1</td></tr><tr><td>DaPT SYFER</td><td colspan="10">28.1 35.6 19.1 34.0</td><td colspan="10">27.0 19.8 31.6</td></tr></table>

Structured graph indexing transfers poorly out of English. HippoRAG2 is the strongest English baseline, but its advantage weakens on the multilingual corpus: from English to the nine-language average, its F1 drops by 23.6% on 2Wiki (65.0 → 49.7), 19.8% on HotpotQA (57.9 → 46.4), and 19.3% on MuSiQue (35.9 → 29.0). The drop is especially visible on Zh, Th, Bn, and Ko, suggesting that graph indexing based on LLM-extracted entities and relations is fragile when applied to heterogeneous multilingual evidence. The resulting cross-lingual noise and redundant edges then propagate into downstream retrieval.

Syfer consistently enhances mRAG on multilingual multi-hop QA. DaPT applies bilingual decomposition at every hop, so noisy sub-branches can still enter the final aggregation. Syfer instead gates bilingual reasoning by decomposition faithfulness (Eq. (4)) and returns the terminal sub-question answer directly (Eq. (1)), reducing the chance that unreliable branches dominate the final prediction. Empirically, Syfer achieves the largest average gain on 2Wiki, improving over the strongest baseline by +17.3 F1 and +20.1 EM, and remains robust on the harder MuSiQue benchmark with +8.9 F1 and +6.2 EM over DaPT. The same trend appears on HotpotQA and on the held-out OOD languages Fr, Bn and Ko, suggesting that the noise-suppression mechanism transfers beyond the languages used for decomposer training.

Syfer attains a more favourable accuracy–cost balance than competing baselines. Figure 3 plots the accuracy–cost Pareto fronts of Syfer against the five baselines, and Syfer achieves a better balance between accuracy and inference cost than every competitor. Notably, HippoRAG2 additionally requires hours of ofline graph-index construction before any query can be served, an overhead that grows further as the corpus size increases.

Table 2. Ablation performance on multilingual HotpotQA. Bold values indicate the best performance within each group.
<table><tr><td rowspan="2">Variant</td><td colspan="2">Zh</td><td colspan="2">De</td><td colspan="2">Es</td><td colspan="2">Sw</td><td colspan="2">Th</td><td colspan="2">Fr</td><td colspan="2">Bn</td><td colspan="2">K₀</td><td colspan="2">Avg.</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td>Full SYFER</td><td>39.4</td><td>61.2</td><td>50.0</td><td>63.6</td><td>49.5</td><td>64.7</td><td>50.5</td><td>62.5</td><td>43.8</td><td>63.9</td><td>45.1</td><td>59.4</td><td>36.8</td><td>49.2</td><td>37.3</td><td>47.4 44.1</td><td></td><td>59.0</td></tr><tr><td>w/o Folding</td><td>33.2</td><td>52.0</td><td>49.5</td><td>62.7</td><td>50.8</td><td>63.2</td><td>35.9</td><td>44.4</td><td>31.6</td><td>47.4</td><td>44.0</td><td>56.9</td><td>26.9</td><td>38.9</td><td>30.2</td><td>46.2</td><td>37.8</td><td>51.4</td></tr><tr><td>w/o Verification</td><td>33.0</td><td>50.5</td><td>48.8</td><td>60.9</td><td>47.9</td><td>60.8</td><td>32.5</td><td>41.0</td><td>28.7</td><td>42.9</td><td>42.4</td><td>56.8</td><td>22.4</td><td>33.3</td><td>25.8</td><td>42.1</td><td>35.2</td><td>48.5</td></tr><tr><td>w/o MMR</td><td>35.9</td><td>52.8</td><td>40.5</td><td>50.6</td><td>42.9</td><td>55.0</td><td>27.1</td><td>34.2</td><td>26.6</td><td>41.5</td><td>35.2</td><td>48.0</td><td>22.6</td><td>34.4</td><td>26.1</td><td>43.9</td><td>32.1</td><td>45.1</td></tr><tr><td>Always Bilingual</td><td>31.2</td><td>47.4</td><td>43.5</td><td>54.1</td><td>44.1</td><td>56.2</td><td>29.8</td><td>37.7</td><td>27.8</td><td>43.2</td><td>36.3</td><td>49.3</td><td>17.9</td><td>28.8</td><td>22.8</td><td>37.8</td><td>31.7</td><td>44.3</td></tr></table>

![](images/e04fe9253f2c0a38905fe8b5cf4ae429289c5836af5e3ce30b9f3bb3ace1e8e3.jpg)

![](images/4767ef77b61896a8dfc42e76ac0ec50240d33e1eb789b5b768dd704503b7bf37.jpg)  
Fig. 3. Accuracy–cost Pareto fronts on the multilingual multi-hop QA pool. sub-figure (a): accuracy versus end-to-end latency per query. sub-figure (b): accuracy versus token cost per query. Syfer sits at the upper-right tip of the Pareto front in both panels.

## 4.6 Ablation Study

We ablate the four core components of Syfer on the eight-language multilingual test pool and report the average EM/F1 in Table 2. The ablated variants are: (i) w/o Folding, which restores the classical end-of-pipeline aggregation call; (ii) w/o Verification, which always commits the monolingual branch, never triggering the bilingual fallback; (iii) Always Bilingual, which keeps synthesizer folding but disables the faithfulness gate, running the bilingual branch at every hop unconditionally; (iv) w/o MMR, which replaces cross-lingual MMR with vanilla top-k.

Removing any component consistently weakens Syfer, confirming that the gains come from the joint design. Among the components, cross-lingual MMR and faithfulness-controlled bilingual routing are especially important: without them, retrieval is more easily dominated by near-duplicate parallel passages or by unnecessary bilingual branches. There is an interesting observation, Always Bilingual, which confirms that more cross-lingual signal is not always better. This suggests that when a sub-question is already answerable in the target language, forcing an English-parallel branch injects extra reasoning noise and may even make the final answer drift into English, which is penalized by the languageconditioned EM/F1 metric.

## 5 Conclusion

We present Syfer, a synthesizer-folding mRAG framework for multilingual multi-hop QA that gates bilingual fusion on a faithfulness check and folds aggregation into a terminal sub-question, making decomposition quality directly verifiable. On a unified nine-language testbed extended from three multi-hop QA benchmarks, Syfer outperforms structure-aware, translation-based, and decomposition-based baselines with a better quality–cost trade-of.

## 6 Acknowledgements

This work was supported in part by the National Science Foundation of China (Nos. 62276056 and U24A20334), the Natural Science Foundation of Liaoning Province of China (2022-KF-26-01), the Fundamental Research Funds for the Central Universities (Nos. N2216016 and N2316002), the Yunnan Fundamental Research Projects (No. 202401BC070021), and the Program of Introducing Talents of Discipline to Universities, Plan 111 (No.B16009).

## References

1. Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., Liu, Z.: Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through selfknowledge distillation. arXiv preprint arXiv:2402.03216 4(5) (2024)

2. Chen, S., Zhou, C., Yuan, Z., Zhang, Q., Cui, Z., Chen, H., Xiao, Y., Cao, J., Huang, X.: You don’t need pre-built graphs for rag: Retrieval augmented generation with adaptive reasoning structures. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 30270–30278 (2026)

3. Chirkova, N., Rau, D., Déjean, H., Formal, T., Clinchant, S., Nikoulina, V.: Retrieval-augmented generation in multilingual settings. In: Proceedings of the 1st Workshop on Towards Knowledgeable Language Models (KnowLLM 2024). pp. 177–188 (2024)

4. DeepSeek-AI: Deepseek-v4: Towards highly eficient million-token context intelligence (2026)

5. Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S., Metropolitansky, D., Ness, R.O., Larson, J.: From local to global: A graph rag approach to query-focused summarization. arXiv preprint arXiv:2404.16130 (2024)

6. Fan, Y., Li, B., Li, P., Wang, Y., Mu, Y., Yang, J., Chen, X., Weng, R., Wang, J., Cai, X., et al.: Lang: Reinforcement learning for multilingual reasoning with language-adaptive hint guidance. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 43838– 43866 (2026)

7. Fan, Y., Mu, Y., Wang, Y., Huang, L., Ruan, J., Li, B., Xiao, T., Huang, S., Feng, X., Zhu, J.: Slam: Towards eficient multilingual reasoning via selective language alignment. In: Proceedings of the 31st International Conference on Computational Linguistics. pp. 9499–9515 (2025)

8. Gutiérrez, B.J., Shu, Y., Qi, W., Zhou, S., Su, Y.: From rag to memory: Non-parametric continual learning for large language models. arXiv preprint arXiv:2502.14802 (2025)

9. Ho, X., Nguyen, A.K.D., Sugawara, S., Aizawa, A.: Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In: Proceedings of the 28th International Conference on Computational Linguistics. pp. 6609–6625 (2020)

10. Huang, K., Mo, F., Zhang, X., Li, H., Li, Y., Zhang, Y., Yi, W., Mao, Y., Liu, J., Xu, Y., et al.: A survey on large language models with multilingualism: Recent advances and new frontiers. Artificial Intelligence Review (2026)

11. Huang, L., Feng, X., Ma, W., Fan, Y., Feng, X., Gu, Y., Ye, Y., Zhao, L., Zhong, W., Wang, B., et al.: Alleviating hallucinations from knowledge misalignment in large language models via selective abstention learning. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 24564–24579 (2025)

12. Huang, L., Feng, X., Ma, W., Fan, Y., Feng, X., Ye, Y., Zhong, W., Gu, Y., Wang, B., Wu, D., et al.: Improving contextual faithfulness of large language models via retrieval heads-induced optimization. arXiv preprint arXiv:2501.13573 (2025)

13. Huang, L., Yu, W., Ma, W., Zhong, W., Feng, Z., Wang, H., Chen, Q., Peng, W., Feng, X., Qin, B., et al.: A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems 43(2), 1–55 (2025)

14. Huang, P., Liu, Z., Yan, Y., Zhao, H., Yi, X., Chen, H., Liu, Z., Sun, M., Xiao, T., Yu, G., et al.: Parammute: Suppressing knowledge-critical fns for faithful retrievalaugmented generation. Advances in Neural Information Processing Systems 38, 100378–100410 (2026)

15. Hurst, A., Lerer, A., Goucher, A.P., Perelman, A., Ramesh, A., Clark, A., Ostrow, A., Welihinda, A., Hayes, A., Radford, A., et al.: Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024)

16. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.t., Rocktäschel, T., et al.: Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems 33, 9459–9474 (2020)

17. Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P.: Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics 12, 157–173 (2024)

18. Park, J., Lee, H.: Investigating language preference of multilingual rag systems. In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 5647– 5675 (2025)

19. Qi, J., Fernández, R., Bisazza, A.: On the consistency of multilingual context utilization in retrieval-augmented generation. In: Proceedings of the 5th Workshop on Multilingual Representation Learning (MRL 2025). pp. 199–225 (2025)

20. Ranaldi, L.: Multilingual retrieval-augmented generation for knowledge-intensive question answering task. In: Findings of the Association for Computational Linguistics: EACL 2026. pp. 697–716 (2026)

21. Ranaldi, L., Ranaldi, F., Zanzotto, F.M., Haddow, B., Birch, A.: Improving multilingual retrieval-augmented language models through dialectic reasoning argumen-

tations. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 9075–9096 (2025)

22. Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N., Chen, W.: Enhancing retrievalaugmented large language models with iterative retrieval-generation synergy. In: Findings of the Association for Computational Linguistics: EMNLP 2023. pp. 9248–9274 (2023)

23. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics 10, 539–554 (2022)

24. Trivedi, H., Balasubramanian, N., Khot, T., Sabharwal, A.: Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In: Proceedings of the 61st annual meeting of the association for computational linguistics (volume 1: long papers). pp. 10014–10037 (2023)

25. Wang, Y., Fan, Y., Li, J., Zhu, Z., Mu, Y., He, Q., Xiao, T., Zhu, J.: Dapt: A dualpath framework for multilingual multi-hop question answering. In: ICASSP 2026- 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 19302–19306. IEEE (2026)

26. Wei, Z., Yang, X., Wang, Y., Wang, Z., Bao, W., Feng, S., Wang, D., Zhang, Y.: Cirag: Construction-integration retrieval and adaptive generation for multi-hop question answering. arXiv preprint arXiv:2601.06799 (2026)

27. Xiao, Y., Zhou, C., Zhang, Y., Zhang, Q., Dong, S., Chen, S., Yang, C., Huang, X.: Lag: Logic-augmented generation from a cartesian perspective. arXiv preprint arXiv:2508.05509 (2025)

28. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

29. Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R., Manning, C.D.: Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In: Proceedings of the 2018 conference on empirical methods in natural language processing. pp. 2369–2380 (2018)

30. Zhang, X., Liang, Y., Meng, F., Zhang, S., Chen, Y., Xu, J., Zhou, J.: Multilingual knowledge editing with language-agnostic factual neurons. In: Proceedings of the 31st International Conference on Computational Linguistics. pp. 5775–5788 (2025)