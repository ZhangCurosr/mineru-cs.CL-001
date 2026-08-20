# MissDiag: Diagnostic Evaluation of Incomplete-Knowledge Robustness in KGQA and KG-RAG

Hang Wang, Hang Dong, Lu Liu, Chuanru Ren

## Abstract

Knowledge graph question answering (KGQA) and knowledge-graph-based retrieval-augmented generation (KG-RAG) aim to ground answers in explicit graph evidence, but real-world knowledge graphs are often sparse, outdated, and incomplete. Existing robustness evaluations usually report aggregate changes in answer quality after evidence is removed or perturbed, which measures sensitivity to incomplete support but leaves the source of degradation under-specified: the same score change can conflate the type of missing evidence, the response of the evaluated system, and the sensitivity of the answer-matching protocol. To address this gap, we propose MissDiag, a diagnostic evaluation framework for incompleteknowledge robustness in KGQA and KG-RAG. MissDiag keeps the question and gold answer fixed while applying structurally typed missingness interventions to benchmarkprovided support graphs, enabling paired comparisons that decompose robustness changes by evidence type, system response, and evaluation protocol rather than reducing them to a single aggregate score drop. Experiments across multiple system families show that incomplete-knowledge robustness is better understood as a typed degradation phenomenon than as a uniform property: answer-adjacent evidence loss produces the largest observed degradation, source-context removal is often neutral and can be beneficial, and semantic answer matching changes absolute scores while preserving the main typed degradation patterns. By transforming aggregate robustness measurement into typed diagnostic attribution, MissDiag provides a more interpretable basis for comparing, diagnosing, and stress-testing KGQA and KG-RAG systems under incomplete knowledge.

Code will be released after the review process.

## Introduction

Knowledge graph question answering (KGQA) and knowledge-graph-based retrieval-augmented generation (KG-RAG) aim to ground answers in explicit graph evidence rather than relying only on parametric memory. This makes them important testbeds for evaluating structured reasoning, factual grounding, and answer trustworthiness. Prior work has advanced this goal through semantic parsing and querygraph construction (Berant et al. 2013; Yih et al. 2015), embedding-based and graph-based QA (Huang et al. 2019; Sun, Bedrax-Weiss, and Cohen 2019; Shi et al. 2021; Ye et al. 2022; Gu and Su 2022; Mavromatis and Karypis 2022), retrieve-and-reason systems over knowledge bases and text (Guu et al. 2020; Lewis et al. 2020), LLM-based KBQA (Xiong, Bao, and Zhao 2024), and increasingly challenging benchmarks (Dubey et al. 2019; Gu et al. 2021; Cao et al. 2022; Zhang et al. 2025). More recently, graph-grounded LLM studies have examined whether KG augmentation improves factuality, reasoning quality, retrieval eficiency, and trustworthiness in open-ended generation settings (Wang et al. 2024; Sun et al. 2024; Sui et al. 2025; Zhou et al. 2026). Across these lines of work, however, a persistent dificulty remains: real-world knowledge graphs are rarely complete. They are often sparse, outdated, and unevenly populated across entities and relations, making robustness under incomplete knowledge a central requirement for reliable KGQA and KG-RAG systems.

![](images/cff35afaf93a7136a493adbaa90255023a504d6d6764383c6d11c2fc8dedb303.jpg)  
Figure 1: Motivation for typed degradation analysis. The same QA instance can show similar aggregate degradation under structurally diferent missingness conditions. Aggregate score drops alone do not reveal which evidence type was removed, whether answer-local evidence was afected, or whether the efect is systematic.

Existing research has studied incomplete knowledge from several complementary perspectives, including contextual reasoning over partial graph structures (Mai et al. 2019), integration of textual evidence (Xiong et al. 2019; Han, Cheng, and Wang 2020; Sun et al. 2023), KG embeddings and relation prediction (Trouillon et al. 2016; Schlichtkrull et al. 2018; Sun et al. 2019; Huang et al. 2019; Saxena, Tripathi, and Talukdar 2020; Zhao et al. 2022; Zan et al. 2022; Saxena, Kochsiek, and Gemulla 2022; Guo et al. 2023), knowledge graph completion (Zhao et al. 2022; Guo et al. 2023), completion-aware reasoning pipelines (Liu et al. 2022; Ye et al. 2024; Han et al. 2025), and LLM-based or multi-agent reasoning over incomplete graphs (Xu et al. 2024; Liu et al. 2026). In parallel, evaluation studies have asked whether completion methods improve downstream QA and whether current benchmarks and metrics support reliable conclusions under incomplete or imperfect knowledge (Yu et al. 2023; Perevalov et al. 2022; Steinmetz and Sattler 2021; Zhang et al. 2025, 2026; Zhou et al. 2026; Sui et al. 2025). These studies have substantially improved our understanding of how systems recover missing facts, exploit auxiliary evidence, or reason over partial structures. Nevertheless, current evaluation practice still has a basic attribution limitation: most evaluations remove, corrupt, or recover evidence and then report the resulting change in answer quality. Such degradation-based evaluation can measure whether performance changes, but it does not explain why the change occurs.

This ambiguity is illustrated in Figure 1: the same question–answer instance may exhibit similar aggregate score drops under structurally diferent evidence-removal conditions, even though the underlying failure mechanisms are diferent. A lower score may indicate that essential supporting evidence has become unavailable; it may also indicate that the system fails to exploit retained evidence, that graph conversion or candidate construction changes the effective search space, or that the evaluation protocol fails to recognize a semantically acceptable answer. Conversely, apparent robustness or even improvement under incomplete evidence may reflect support pruning or metric behavior rather than stronger reasoning. This matters because robustness under incomplete knowledge is often used as evidence ofreasoning capability and system reliability. Ifsimilar score changes can arise from diferent causes, then aggregate degradation alone is insuficient for interpreting robustness claims. What is needed is not only a performance measurement protocol, but a diagnostic evaluation framework that can attribute degradation to diferent forms of missing evidence, compare how system families respond to the same intervention, and expose how much the conclusion depends on the evaluation metric.

To address this gap, we propose MissDiag, a diagnostic evaluation framework for incomplete-knowledge robustness in KGQA and KG-RAG. As shown in Figure 2, MissDiag starts from a benchmark-provided evaluation instance consisting of a question, a gold answer set, and a local support graph. It keeps the question and gold answer fixed while transforming the support graph into structurally distinct incomplete-evidence conditions. The primary missingness operators include random support loss, source-context loss, relation-level removal, and answer-adjacent removal. By evaluating the same system on the same question–answer pair before and after each typed intervention, MissDiag separates three factors that are usually entangled in incompleteknowledge evaluation: missingness type, system response, and evaluation sensitivity.

Rather than treating incomplete knowledge as a single robustness condition, MissDiag represents it as a typed degradation phenomenon. The framework reports robustness through degradation profiles indexed by missingness type, severity, system, and evaluation metric, with structural slices used for further analysis. This design enables paired and interpretable comparison across trained KGQA models, graph-structured prompting methods, iterative KG agents, and direct LLM baselines. It therefore allows robustness claims to be examined in terms of where degradation comes from, when it reflects genuine evidence loss, and when it should instead be attributed to system behavior or evaluation sensitivity.

Our contributions are as follows:

• We formulate incomplete-knowledge robustness evaluation as a diagnostic attribution problem, showing that aggregate score changes conflate evidence availability, system behavior, and evaluation protocol, and therefore cannot by themselves support reliable conclusions about reasoning robustness.

• We introduce MissDiag, a controlled diagnostic framework that applies structurally typed missingness interventions to benchmark support graphs while preserving paired question–answer comparisons, enabling degradation to be analyzed by missingness source rather than only by overall performance loss.

• We instantiate MissDiag across multiple system families, severity levels, structural slices, and answer-matching metrics, demonstrating that the same missing-evidence intervention can lead to degradation, near invariance, or improvement depending on system design and evaluation protocol. This reveals robustness patterns that aggregate scores obscure.

## Related Work

## KGQA and Graph-Grounded QA

KGQA aims to answer natural-language questions using structured graph evidence. Early work often relied on semantic parsing or query-graph construction to map questions into executable logical forms or graph queries (Berant et al. 2013; Yih et al. 2015). Later graph-based models retrieve and reason over local evidence subgraphs, making the support structure itself part of the answering process (Sun, Bedrax-Weiss, and Cohen 2019; He et al. 2021; Mavromatis and Karypis 2022). In parallel, retrieval-centered QA and graph-grounded LLM methods use retrieved knowledge to support generation and reasoning beyond purely parametric memory (Guu et al. 2020; Lewis et al. 2020; Wang et al. 2024; Xiong, Bao, and Zhao 2024; Sun et al. 2024). Benchmarks such as LC-QuAD 2.0, GrailQA, KQA Pro, and KGQAGen broaden evaluation beyond simple fact lookup by emphasizing compositionality, generalization, and dataset reliability (Dubey et al. 2019; Gu et al. 2021; Cao et al. 2022; Zhang et al. 2025). These studies provide the system and benchmark context for our work. MissDiag difers by treating existing systems as diagnostic subjects under controlled evidence manipulation rather than proposing another KGQA architecture.

## Incomplete-Knowledge Question Answering

Incomplete knowledge is a persistent challenge for KGQA because real-world graphs are sparse, unevenly populated, and often missing facts needed for multi-hop reasoning. Prior work has addressed this issue by reasoning over partial graph structures (Mai et al. 2019), incorporating auxiliary textual evidence (Xiong et al. 2019; Han, Cheng, and Wang 2020), using graph completion or completion-aware QA pipelines (Yu et al. 2023; Ye et al. 2024), and applying LLM-centered reasoning to incomplete graph evidence (Xu et al. 2024; Zhou et al. 2026). These approaches mainly ask how to recover or maintain answer quality when knowledge is missing. Our work asks a complementary evaluation question: when answer quality changes under incomplete knowledge, how should the change be attributed? Instead of treating missingness as a single condition, MissDiag separates diferent structural forms of evidence loss and compares their efects under a paired protocol.

![](images/97acabc22292a370d8bb7f0da21bc670888399e27be47c56a8a8307ae60562d9.jpg)  
Figure 2: Overview of MissDiag. (1) Input: an instance contains a question, gold answer set, and local support graph. (2) Typed Missingness Operators: typed operators select removable support edges for random, source-context, relation-level, and answeradjacent missingness. (3) Severity-Controlled Missingness: a shared severity budget produces an incomplete support graph. (4) Paired Evaluation: the same system is evaluated under complete and incomplete support to compute paired degradation. (5) Typed Degradation Profile: degradation values are summarized across missingness type, severity, system, and metric.

## Benchmark Reliability and Evaluation Sensitivity

Evaluation conclusions in KGQA and graph-grounded QA are sensitive to dataset construction, evidence availability, and answer-matching protocols. Dataset audits and leaderboard analyses show that KGQA benchmarks can contain annotation issues, heterogeneous dificulty, and inconsistent reporting practices (Steinmetz and Sattler 2021; Perevalov et al. 2022; Zhang et al. 2025). More broadly, adversarial evaluation and behavioral testing show that aggregate metrics can hide distinct failure mechanisms behind similar score changes (Jia and Liang 2017; Ribeiro et al. 2020; Gardner et al. 2020). Recent KG-RAG and trustworthiness studies further suggest that graph augmentation and incomplete evidence require careful evaluation design (Sui et al. 2025; Zhang et al. 2026; Zhou et al. 2026). This literature motivates our view that evaluation is not a neutral reporting layer. MissDiag extends this line of work by decomposing incomplete-knowledge evaluation into missingness type, system response, severity, and metric sensitivity, so that robustness claims can be interpreted beyond a single aggregate degradation score.

## Method

This section introduces MissDiag, a diagnostic evaluation framework for incomplete-knowledge robustness in KGQA and KG-RAG. As shown in Figure 2, the framework keeps the question, gold answer, and evaluated system fixed, transforms the support graph with severity-controlled typed missingness operators, and compares the resulting outputs against the complete-support condition. This design turns aggregate robustness changes into paired degradation profiles indexed by missingness type, severity, system, and metric. The section first formulates the diagnostic evaluation problem, then describes support graph construction, defines the missingness operators, and presents the paired degradation profile used as the main diagnostic output.

## Diagnostic Formulation

Each evaluation instance is represented as

$$
x _ { i } = ( q _ { i } , A _ { i } ^ { * } , G _ { i } ) ,\tag{1}
$$

where $q _ { i }$ is the question, $A _ { i } ^ { * }$ is the gold answer set, and $G _ { i } = ( \bar { V } _ { i } , E _ { i } )$ is the local support graph. The source entities linked to the question are denoted by $S _ { i } \subseteq V _ { i } ,$ , and the gold-answer entities aligned to graph nodes are denoted by $Y _ { i } \subseteq V _ { i }$ . Operators requiring unavailable alignments mark the instance infeasible for the corresponding operator. Given an evaluated system $f ,$ MissDiag compares the completesupport prediction $f ( q _ { i } , G _ { i } )$ with predictions obtained after transforming only the support edges. Across conditions, the question, gold answer, source entities, and node inventory remain fixed; only the retained edge evidence changes.

## Support Graph Construction

The local support graph $G _ { i }$ is constructed from the evidence associated with instance $x _ { i }$ . When the evidence is given as triples, proof paths, or a retrieved subgraph, it is converted into a labeled directed graph. Graph distances used by the missingness operators are computed on the undirected projection of this graph. For a node u and node set $B ,$ dist $( u , B )$ denotes the shortest-path distance from u to any node in B on the undirected projection.

The edge set is written as

$$
E _ { i } = E _ { i } ^ { \mathrm { s u p } } \cup E _ { i } ^ { \mathrm { c t x } } ,\tag{2}
$$

where $E _ { i } ^ { \mathrm { s u p } }$ denotes the original support evidence and $E _ { i } ^ { \mathrm { c t x } }$ denotes a possibly empty set of optional source-anchored context edges. The context edges are constructed before any missingness intervention, follow a deterministic selection rule, and remain fixed across all conditions. This ensures that all missingness operators start from the same support graph.

After an intervention, the evaluated system receives only the retained support evidence. The node inventory may be kept fixed internally for alignment and paired comparison, but isolated nodes are not exposed as additional answer hints in generative KG-RAG prompts. Additional construction details are provided in the supplementary material.

## Severity-Controlled Missingness

Let $\alpha \in [ 0 , 1 ]$ denote the missingness severity and $n _ { i } = | E _ { i } |$ the number of support edges. For each instance, the nominal removal budget is defined as

$$
\kappa _ { i } ( \alpha ) = \left\{ \begin{array} { l l } { 0 , } & { n _ { i } \leq 1 \mathrm { o r } \alpha = 0 , } \\ { \operatorname* { m i n } \bigl ( n _ { i } - 1 , \eta _ { i } ( \alpha ) \bigr ) , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{3}
$$

where $\eta _ { i } ( \alpha ) = \operatorname* { m a x } ( 1 , \operatorname { r o u n d } ( \alpha n _ { i } ) )$ . This budget preserves complete support at $\alpha = 0$ and keeps at least one support edge when removal is applied.

Each missingness operator m defines an ordered candidate edge list $\boldsymbol { L } _ { i } ^ { ( m ) }$ . Given the shared budget, the removed edge set $R _ { i } ^ { ( m , \alpha ) }$ consists of the first min $( \kappa _ { i } ( \alpha ) , | L _ { i } ^ { ( m ) } | )$ edges in $\boldsymbol { L } _ { i } ^ { ( m ) }$ . The retained support graph is

$$
\widetilde { G } _ { i } ^ { ( m , \alpha ) } = ( V _ { i } , E _ { i } \setminus R _ { i } ^ { ( m , \alpha ) } ) ,\tag{4}
$$

where $R _ { i } ^ { ( m , \alpha ) }$ is the removed edge set. I $\dot { \cdot } \kappa _ { i } ( \alpha ) > 0$ but $L _ { i } ^ { ( m ) }$ is empty, the instance is marked infeasible for operator m.

## Typed Missingness Operators

MissDiag uses four typed missingness operators: random, source-context, relation-level, and answer-adjacent missingness. Each operator first defines a candidate edge set and then orders it into $\boldsymbol { L } _ { i } ^ { ( m ) }$ , which is used by the shared removal budget. The operators are designed to probe diferent structural forms of evidence loss under the same removal budget, rather than to define mutually exclusive edge categories.

For an edge $e = ( u , r , v )$ , two structural scores are used:

$$
\tau _ { i } ( e ) = \operatorname* { m a x } \{ \mathrm { d i s t } ( u , S _ { i } ) , \mathrm { d i s t } ( v , S _ { i } ) \} ,\tag{5}
$$

$$
\rho _ { i } ( e ) = \operatorname* { m i n } \{ \mathrm { d i s t } ( u , Y _ { i } ) , \mathrm { d i s t } ( v , Y _ { i } ) \} .\tag{6}
$$

Here, $\tau _ { i } ( e )$ measures source depth and $\rho _ { i } ( e )$ measures answer distance.

Random missingness. Random missingness uses all support edges as candidates, $C _ { i } ^ { ( \mathrm { r a n d } ) } = E _ { i }$ . The ordered list $L _ { i } ^ { ( \mathrm { r a n d } ) }$ is a uniform random permutation of $C _ { i } ^ { ( \mathrm { r a n d } ) }$ . This operator serves as a quantity-matched baseline for generic support loss.

Source-context missingness. Source-context missingness targets edges incident to source entities but not incident to aligned answer entities:

$$
\begin{array} { r } { C _ { i } ^ { \mathrm { ( s r c ) } } = \{ e = ( u , r , v ) \in E _ { i } : \{ u , v \} \cap S _ { i } \neq \emptyset , } \\ { \{ u , v \} \cap Y _ { i } = \emptyset \} . } \end{array}\tag{7}
$$

The ordered list $L _ { i } ^ { ( \mathrm { s r c } ) }$ is obtained by sorting candidates by increasing $\tau _ { i } ( e )$ , with deterministic tie-breaking. This operator probes sensitivity to source-side context while excluding answer-incident evidence from its candidate set.

Relation-level missingness. Relation-level missingness uses all support edges as candidates, $C _ { i } ^ { ( \mathrm { r e l } ) } = E _ { i }$ , and groups them by relation type. Relation blocks are ordered by decreasing frequency in $\bar { E } _ { i } ,$ , and edges are removed block by block under the shared budget. This operator probes sensitivity to relation-family evidence.

Answer-adjacent missingness. Answer-adjacent missingness targets edges incident to, or one hop from, aligned answer entities:

$$
C _ { i } ^ { ( \mathrm { a n s } ) } = \{ e \in E _ { i } : \rho _ { i } ( e ) \leq 1 \} .\tag{8}
$$

The ordered list $L _ { i } ^ { ( \mathrm { a n s } ) }$ is obtained by sorting candidates by increasing $\rho _ { i } ( e )$ , with deterministic tie-breaking. This operator probes sensitivity to answer-local grounding evidence.

## Paired Degradation Profiles

Let $\mu$ be the answer-set evaluation metric used for a given comparison. MissDiag treats the metric as an explicit axis of the diagnostic profile. For system $f ,$ , the complete-support score of instance i is $\mu ( f ( q _ { i } , \mathbf { \bar { G } } _ { i } ) , \dot { A _ { i } ^ { * } } )$ , and the score under missingness type m and severity α is $\mu ( f ( q _ { i } , \widetilde { G } _ { i } ^ { ( m , \alpha ) } ) , A _ { i } ^ { * } )$ The paired degradation is defined as

$$
\delta _ { i } ^ { ( m , \alpha , \mu ) } = \mu ( f ( q _ { i } , G _ { i } ) , A _ { i } ^ { * } ) - \mu ( f ( q _ { i } , \widetilde { G } _ { i } ^ { ( m , \alpha ) } ) , A _ { i } ^ { * } ) .\tag{9}
$$

Positive values indicate performance loss after evidence removal, while negative values indicate improvement.

Let $\mathcal { T } ^ { ( m , \alpha ) }$ be the feasible instance set for operator m at severity α. The dataset-level degradation of system $f$ is

$$
\Delta _ { f } ^ { ( m , \alpha , \mu ) } = \frac { 1 } { | \mathcal { T } ^ { ( m , \alpha ) } | } \sum _ { i \in \mathcal { I } ^ { ( m , \alpha ) } } \delta _ { i } ^ { ( m , \alpha , \mu ) } .\tag{10}
$$

The main diagnostic output is the typed degradation profile

$$
\begin{array} { r } { \mathbf { D } _ { f } ^ { ( \alpha , \mu ) } = \left[ \begin{array} { l } { \Delta _ { f } ^ { ( \mathrm { r a n d } , \alpha , \mu ) } } \\ { \Delta _ { f } ^ { ( \mathrm { s r c } , \alpha , \mu ) } } \\ { \Delta _ { f } ^ { ( \mathrm { r e l } , \alpha , \mu ) } } \\ { \Delta _ { f } ^ { ( \mathrm { a n s } , \alpha , \mu ) } } \end{array} \right] . } \end{array}\tag{11}
$$

This profile summarizes how a fixed system degrades under diferent missingness types at a given severity and metric. Instances infeasible for an operator are excluded from that operator’s aggregation, with feasibility details provided in the supplementary material.

## Experiments

We evaluate whether MissDiag provides a more informative view of incomplete-knowledge robustness than a single aggregate score. Following the method design, the evaluation first compares typed degradation profiles across system families, then examines severity efects, metric sensitivity, and structural slices. The experiments are organized around four research questions:

• RQ1: Do missingness types produce distinct degradation profiles?

• RQ2: How do profiles change as severity increases?

• RQ3: Do profiles depend on the evaluation metric?

• RQ4: When is answer-adjacent degradation strongest?

## Experimental Setup

Data. We use KGQAGen-10k (Zhang et al. 2025) as the main evaluation benchmark. Its instances provide question– answer pairs and local evidence structures, which are converted into the support-graph format used by MissDiag. We link source entities from the question and align gold-answer entities to graph nodes when available. The main experiments use the development split. Dataset-specific support construction details are provided in the supplementary material.

Evaluation protocol. We use a paired fixed-input protocol. For each instance, the question, gold answer set, source entities, and node inventory are kept fixed across complete and incomplete-support conditions; only the retained support edges change. For a fixed instance, missingness type, and severity, all systems are evaluated on the same transformed support graph. The main cross-system comparison uses 1,050 examples for which the complete-support condition and all four missingness conditions are feasible. LLMfocused analyses use the corresponding feasible subset, with sample sizes reported in the relevant tables.

Systems. We evaluate four system families: (1) Trained KGQA models, including ReaRev (Mavromatis and Karypis 2022), NuTrea (Choi et al. 2023), and NSM (He et al. 2021); (2) Graph prompting methods, including MindMap (Wen, Wang, and Sun 2024), StructGPT (Jiang et al. 2023), and KG-GPT (Kim et al. 2023); (3) KG agents, including ToG (Sun et al. 2024), PoG (Chen et al. 2024), and GoG (Xu et al. 2024); and (4) Direct LLMbaselines, including Qwen2.5-7B-Instruct<sup>1</sup>, Qwen2.5-14B-Instruct<sup>2</sup>, Llama-3.1-8B-Instruct<sup>3</sup>, and Mistral-7B-Instruct-v0.3<sup>4</sup>. For graph prompting methods and KG agents, all methods use Qwen2.5-7B-Instruct as the shared LLM backbone across complete-support and missingness conditions. This controls for backbone capability and focuses the comparison on prompting, planning, and agent workflow. For all LLM-based systems, graph access is restricted to the transformed local support graph.

Missingness conditions. We compare complete support with four typed missingness conditions: random, sourcecontext, relation-level, and answer-adjacent missingness. Unless otherwise stated, the main comparison uses severity α = 0.3. The severity analysis evaluates α ∈ {0.1, 0.3, 0.5}.

Metrics and reporting. The primary metric is exact macro set-level F1. Results are reported as complete-support F1 and paired degradation ∆F1, where positive values indicate performance loss after evidence removal and negative values indicate improvement. Metric sensitivity is evaluated by comparing exact F1 with semantic F1 for direct LLM baselines. Semantic F1 replaces exact string matching with semantic equivalence matching before computing set-level precision and recall. Additional low-level details, including prompt templates and feasibility bookkeeping, are provided in the supplementary material.

## Implementation Details

Random missingness uses a fixed global seed, with perexample seeds derived from the sample identifier, severity, and operator. Non-random operators use deterministic tiebreaking when ordering candidate edges. ReaRev, NuTrea, and NSM are trained once on complete-support data using pinned oficial implementations, and their best-F1 checkpoints are reused across all missingness conditions without condition-specific retraining. For LLM-based systems, retained graph edges are serialized as textual triples and provided as the only graph evidence in the prompt. Inference uses deterministic greedy decoding without sampling, with fixed generation and reasoning budgets across evidence conditions. Experiments are conducted using NVIDIA GH200 GPUs; LLM-based experiments use PyTorch 2.9.1, CUDA 12.8, Transformers 4.46.2, and bfloat16 inference. Additional implementation details are provided in the supplementary material.

## RQ1: Typed Degradation Profiles Across Systems

Table 1 reports the main cross-system comparison on 1,050 paired development examples at α = 0.3. Each row gives the complete-support F1 ofa system and its paired ∆F1 under the four missingness conditions, directly instantiating the typed degradation profile defined in the method. Three patterns are clear: First, answer-adjacent removal is the dominant degradation condition for every system, with drops ranging from 10.3 to 21.3 F1 points. This shows that answer-local evidence loss is consistently more damaging than generic support removal across trained KGQA models, graph prompting methods, KG agents, and direct LLM baselines. Second, source-context removal behaves diferently: it is small for several prompting and direct LLM systems, and negative for several trained KGQA models. This indicates that removing source-side context can sometimes reduce distracting evidence rather than harm prediction. Third, random and relation-level removal usually produce intermediate degradation, suggesting that support quantity and relation-family loss afect performance but do not explain the full degradation pattern.

<table><tr><td rowspan="2">Category</td><td rowspan="2">System</td><td rowspan="2">Complete F1</td><td colspan="4">Paired degradation ∆F1</td></tr><tr><td>Random</td><td>Source-context</td><td>Relation-level</td><td>Answer-adjacent</td></tr><tr><td rowspan="3">Trained KGQA Models</td><td>ReaRev</td><td>80.8</td><td>4.5</td><td>-1.2</td><td>2.7</td><td>11.9</td></tr><tr><td>NuTrea</td><td>77.2</td><td>1.9</td><td>-9.9</td><td>-2.7</td><td>12.2</td></tr><tr><td>NSM</td><td>46.7</td><td>3.0</td><td>-24.1</td><td>-5.2</td><td>12.1</td></tr><tr><td rowspan="3">Graph Prompting Methods</td><td>MindMap</td><td>46.2</td><td>10.1</td><td>0.6</td><td>11.2</td><td>12.7</td></tr><tr><td>StructGPT</td><td>52.9</td><td>9.6</td><td>-0.7</td><td>8.0</td><td>13.8</td></tr><tr><td>KG-GPT</td><td>45.7</td><td>7.0</td><td>4.2</td><td>5.3</td><td>10.3</td></tr><tr><td rowspan="3">KG Agents</td><td>ToG</td><td>79.5</td><td>7.0</td><td>3.9</td><td>6.0</td><td>11.4</td></tr><tr><td>PoG</td><td>79.9</td><td>6.8</td><td>2.7</td><td>5.8</td><td>11.6</td></tr><tr><td>GoG</td><td>81.7</td><td>12.1</td><td>2.3</td><td>8.9</td><td>19.8</td></tr><tr><td rowspan="4">Direct LLM Baselines</td><td>Qwen2.5-7B-Instruct</td><td>85.3</td><td>11.1</td><td>3.0</td><td>7.4</td><td>21.3</td></tr><tr><td>Qwen2.5-14B-Instruct</td><td>90.0</td><td>6.7</td><td>1.2</td><td>5.5</td><td>17.0</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td>84.4</td><td>6.1</td><td>-0.5</td><td>4.0</td><td>15.8</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>81.2</td><td>5.4</td><td>1.6</td><td>3.8</td><td>13.3</td></tr></table>

Table 1: Main typed degradation profiles across system families at $\alpha = 0 . 3 .$ . Scores are reported as complete-support macro F1 and paired ∆F1 under each missingness type. Positive ∆F1 indicates degradation; negative values indicate improvement.

Together, these results provide the main empirical support for typed degradation profiles: under the same paired protocol, incomplete-support degradation depends on what kind of evidence is missing and how each system uses the remaining graph. A single aggregate degradation score would collapse these distinct efects and obscure the diference between harmful answer-local loss, neutral or beneficial sourcecontext removal, and intermediate random or relation-level loss.

## RQ2: Severity Efects

Figure 3 reports severity-dependent degradation for representative direct LLM baselines. The severity parameter varies over $\alpha \in \{ 0 . 1 , 0 . 3 , 0 . 5 \}$ , while the question, gold answer, metric, and missingness operators remain fixed. The curves show that increasing severity amplifies degradation, but not uniformly across missingness types. Answer-adjacent removal remains the strongest degradation condition at every severity and grows most sharply as α increases. Random and relation-level removal also increase with severity, but their degradation remains consistently below answer-adjacent removal. Source-context removal stays comparatively small and changes little across severity levels. These results show that the typed degradation pattern is not an artifact of the main α = 0.3 setting. Severity controls the scale of evidence loss, while the relative behavior of missingness types remains structurally distinct.

## RQ3: Metric Sensitivity

Table 2 compares paired degradation under exact F1 and semantic F1 for direct LLM baselines at $\alpha \ = \ 0 . 3 .$ . This analysis tests whether the typed degradation profile changes when answer matching allows semantic equivalence rather than exact surface matching. The results show that metric choice changes degradation magnitudes but not the main typed pattern. For all four direct LLM baselines, answeradjacent removal remains the largest degradation condition under both exact F1 and semantic F1. Semantic matching slightly changes individual ∆F1 values, but it does not reverse the ordering of missingness efects. This indicates that the main diagnostic conclusion depends primarily on the type of missing evidence rather than on the particular answermatching rule.

![](images/7ee254ea7f51a366e2aa7aa4e4f291853ef4281ada72cf11647604a83246223b.jpg)  
Figure 3: Severity efects on paired degradation. Curves show paired ∆F1 across missingness types for Qwen2.5- 7B-Instruct and Llama-3.1-8B-Instruct as α increases.

## RQ4: Structural Slices

Table 3 examines when answer-adjacent degradation is strongest. The analysis compares random and answeradjacent degradation across answer cardinality and supportgraph size, averaged over the four direct LLM baselines. The gap is defined as answer-adjacent ∆F1 minus random ∆F1.

<table><tr><td rowspan="2">System</td><td rowspan="2">Metric</td><td colspan="4">Paired degradation ∆F1</td></tr><tr><td>Rand</td><td>Src</td><td>Rel</td><td>Ans-adj</td></tr><tr><td rowspan="2">Qwen2.5-7B</td><td>Exact F1</td><td>11.1</td><td>3.0</td><td>7.4</td><td>21.3</td></tr><tr><td>Semantic F1</td><td>11.6</td><td>3.2</td><td>8.0</td><td>21.4</td></tr><tr><td rowspan="2">Qwen2.5-14B</td><td>Exact F1</td><td>6.7</td><td>1.2</td><td>5.5</td><td>17.0</td></tr><tr><td>Semantic F1</td><td>6.9</td><td>1.2</td><td>5.4</td><td>16.1</td></tr><tr><td rowspan="2">Llama-3.1-8B</td><td>Exact F1</td><td>6.1</td><td>-0.5</td><td>4.0</td><td>15.8</td></tr><tr><td>Semantic F1</td><td>5.1</td><td>-1.0</td><td>3.9</td><td>15.1</td></tr><tr><td rowspan="2">Mistral-7B</td><td>Exact F1</td><td>5.4</td><td>1.6</td><td>3.8</td><td>13.3</td></tr><tr><td>Semantic F1</td><td>5.1</td><td>0.9</td><td>3.2</td><td>13.5</td></tr></table>

Table 2: Metric sensitivity of typed degradation profiles for direct LLM baselines at α = 0.3. Rand, Src, Rel, and Ans-adj denote random, source-context, relation-level, and answeradjacent missingness.
<table><tr><td>Slice</td><td>Group</td><td>N</td><td>Full</td><td>Rand</td><td>Ans-adj</td><td>Gap</td></tr><tr><td rowspan="2">Answer cardinality</td><td>Single-answer</td><td>888</td><td>90.8</td><td>6.8</td><td>14.4</td><td>7.6</td></tr><tr><td>Multi-answer</td><td>163</td><td>54.5</td><td>10.4</td><td>30.2</td><td>19.9</td></tr><tr><td rowspan="3">Support size</td><td>Small support</td><td>353</td><td>80.5</td><td>12.6</td><td>26.7</td><td>14.2</td></tr><tr><td>Medium support</td><td>527</td><td>87.0</td><td>5.2</td><td>13.3</td><td>8.0</td></tr><tr><td>Large support</td><td>171</td><td>89.2</td><td>2.9</td><td>7.4</td><td>4.5</td></tr></table>

Table 3: Structural slices for direct LLM baselines at α = 0.3. Rand and Ans-adj report paired ∆F1 under random and answer-adjacent missingness. Gap is the diference between Ans-adj and Rand degradation.

Answer-adjacent degradation is most pronounced for multi-answer questions and small-support graphs. Multianswer examples show a 19.9-point gap over random removal, compared with 7.6 points for single-answer examples. The gap also decreases as support size grows, from 14.2 points on small-support graphs to 4.5 points on large-support graphs. These results show that answer-local evidence loss is especially harmful when answers are structurally harder to recover or when alternative support paths are limited.

## Discussion

Taken together, the experiments show that incompleteknowledge robustness is better characterized by typed degradation profiles than by a single aggregate score. The dominant efect comes from answer-adjacent evidence loss, while severity, metric, and structural analyses clarify when this efect is amplified or preserved. A system can appear robust under one missingness type while being highly sensitive to another. This is most visible in the contrast between source-context and answer-adjacent removal. Source-context removal is often small or even beneficial, suggesting that some source-side evidence may introduce distraction or increase the burden of evidence selection. In contrast, answeradjacent removal consistently produces the largest degradation, indicating that systems depend strongly on evidence near the aligned answer entities. These two efects would be collapsed by an aggregate missing-evidence score, even though they imply diferent failure mechanisms.

Typed profiles also make cross-system comparison more interpretable. Higher complete-support F1 does not necessarily imply stronger robustness under all forms of missingness. For example, direct LLM baselines achieve strong completesupport performance but can still show large answer-adjacent degradation. Conversely, some trained KGQA models show negative degradation under source-context removal, indicating that their behavior under incomplete evidence is not captured by complete-support accuracy alone. This highlights MissDiag’s diagnostic role in revealing how systems respond to diferent evidence losses, not only how much their average score changes.

## Conclusion

This paper studies incomplete-knowledge KGQA and KG-RAG evaluation as a diagnostic problem. A score drop under incomplete knowledge is not directly interpretable because it can conflate missing evidence, system response, and evaluation protocol. MissDiag addresses this ambiguity by keeping the question, gold answer, source entities, and node inventory fixed while applying severity-controlled typed missingness operators to support edges. Across trained KGQA models, graph prompting methods, KG agents, and direct LLM baselines, answer-adjacent evidence loss produces the most consistent degradation, whereas source-context removal can be neutral or beneficial. Severity, metric, and structural analyses further show when this typed pattern is preserved or amplified. The central implication is methodological: incompleteknowledge robustness should be reported as typed degradation profiles, rather than as one aggregate score drop.

## References

Berant, J.; Chou, A.; Frostig, R.; and Liang, P. 2013. Semantic Parsing on Freebase from Question-Answer Pairs. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, 1533–1544.

Cao, S.; Shi, J.; Pan, L.; Nie, L.; Xiang, Y.; Hou, L.; Li, J.; He, B.; and Zhang, H. 2022. KQA Pro: A Dataset with Explicit Compositional Programs for Complex Question Answering over Knowledge Base. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 6101–6119.

Chen, L.; Tong, P.; Jin, Z.; Sun, Y.; Ye, J.; and Xiong, H. 2024. Plan-on-Graph: Self-Correcting Adaptive Planning of Large Language Model on Knowledge Graphs. In Advances in Neural Information Processing Systems, volume 37.

Choi, H. K.; Lee, S.; Chu, J.; and Kim, H. J. 2023. Nutrea: Neural tree search for context-guided multi-hop kgqa. Advances in Neural Information Processing Systems, 36: 35954–35965.

Dubey, M.; Banerjee, D.; Abdelkawi, A.; and Lehmann, J. 2019. LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia. In The Semantic Web – ISWC 2019, 69–78.

Gardner, M.; Artzi, Y.; Basmov, V.; Berant, J.; Bogin, B.; Chen, S.; Dasigi, P.; Dua, D.; Elazar, Y.; Gottumukkala, A.; et al. 2020. Evaluating Models’ Local Decision Boundaries via Contrast Sets. In Findings of the Association for Computational Linguistics: EMNLP 2020, 1307–1323.

Gu, Y.; Kase, S.; Vanni, M.; Sadler, B.; Liang, P.; Yan, X.; and Su, Y. 2021. Beyond I.I.D.: Three Levels of Generalization for Question Answering on Knowledge Bases. In Proceedings ofThe Web Conference 2021, 3477–3488.

Gu, Y.; and Su, Y. 2022. ArcaneQA: Dynamic Program Induction and Contextualized Encoding for Knowledge Base Question Answering. In Proceedings of the 29th International Conference on Computational Linguistics, 1718– 1731.

Guo, Q.; Wang, X.; Zhu, Z.; Liu, P.; and Xu, L. 2023. A Knowledge Inference Model for Question Answering on an Incomplete Knowledge Graph. Applied Intelligence, 53(7): 7634–7646.

Guu, K.; Lee, K.; Tung, Z.; Pasupat, P.; and Chang, M. 2020. Retrieval Augmented Language Model Pre-Training. In Proceedings of the 37th International Conference on Machine Learning, 3929–3938.

Han, J.; Cheng, B.; and Wang, X. 2020. Open Domain Question Answering based on Text Enhanced Knowledge Graph with Hyperedge Infusion. In Findings of the Association for Computational Linguistics: EMNLP 2020, 1475–1481.

Han, R.; Liu, J.; Bi, H.; Peng, T.; and Liu, L. 2025. SCR: A Completion-then-Reasoning Framework for Multi-hop Question Answering over Incomplete Knowledge Graph. Neurocomputing, 131027.

He, G.; Lan, Y.; Jiang, J.; Zhao, W. X.; and Wen, J.-R. 2021. Improving Multi-hop Knowledge Base Question Answering by Learning Intermediate Supervision Signals. In Proceedings ofthe FourteenthACMInternational Conference on Web Search and Data Mining, 553–561. Association for Computing Machinery.

Huang, X.; Zhang, J.; Li, D.; and Li, P. 2019. Knowledge Graph Embedding Based Question Answering. In Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining, 105–113.

Jia, R.; and Liang, P. 2017. Adversarial Examples for Evaluating Reading Comprehension Systems. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, 2021–2031.

Jiang, J.; Zhou, K.; Dong, Z.; Ye, K.; Zhao, W. X.; and Wen, J.-R. 2023. StructGPT: A General Framework for Large Language Model to Reason over Structured Data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, 9237–9251.

Kim, J.; Kwon, Y.; Jo, Y.; and Choi, E. 2023. KG-GPT: A General Framework for Reasoning on Knowledge Graphs Using Large Language Models. In Findings of the Association for Computational Linguistics: EMNLP 2023, 9410–9421.

Lewis, P.; Perez, E.; Piktus, A.; Petroni, F.; Karpukhin, V.;Goyal, N.; Küttler, H.; Lewis, M.; Yih, W.-t.; Rocktäschel, T.;

et al. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33: 9459–9474.

Liu, J.; Shao, P.; Qin, W.; Liu, F.; Yang, Y.; and Hong, R. 2026. Debate over Mixed-knowledge: A Robust Multi-Agent Reasoning Framework for Incomplete Knowledge Graph Question Answering. In Proceedings oftheAAAIConference on Artificial Intelligence, volume 40, 15333–15341.

Liu, L.; Du, B.; Xu, J.; Xia, Y.; and Tong, H. 2022. Joint Knowledge Graph Completion and Question Answering. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 1098–1108.

Mai, G.; Janowicz, K.; Yan, B.; Zhu, R.; Cai, L.; and Lao, N. 2019. Contextual Graph Attention for Answering Logical Queries over Incomplete Knowledge Graphs. In Proceedings of the 10th International Conference on Knowledge Capture, 171–178.

Mavromatis, C.; and Karypis, G. 2022. ReaRev: Adaptive Reasoning for Question Answering over Knowledge Graphs. In Findings of the Association for Computational Linguistics: EMNLP 2022, 2447–2458.

Perevalov, A.; Yan, X.; Kovriguina, L.; Jiang, L.; Both, A.; and Usbeck, R. 2022. Knowledge Graph Question Answering Leaderboard: A Community Resource to Prevent a Replication Crisis. In Proceedings of the Thirteenth Language Resources and Evaluation Conference, 2998–3007.

Ribeiro, M. T.; Wu, T.; Guestrin, C.; and Singh, S. 2020. Beyond Accuracy: Behavioral Testing of NLP Models with CheckList. In Proceedings ofthe 58th Annual Meeting ofthe Associationfor Computational Linguistics, 4902–4912.

Saxena, A.; Kochsiek, A.; and Gemulla, R. 2022. Sequenceto-Sequence Knowledge Graph Completion and Question Answering. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2814–2828.

Saxena, A.; Tripathi, A.; and Talukdar, P. 2020. Improving Multi-hop Question Answering over Knowledge Graphs using Knowledge Base Embeddings. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 4498–4507.

Schlichtkrull, M.; Kipf, T. N.; Bloem, P.; van den Berg, R.; Titov, I.; and Welling, M. 2018. Modeling Relational Data with Graph Convolutional Networks. In The Semantic Web, 593–607.

Shi, J.; Cao, S.; Hou, L.; Li, J.; and Zhang, H. 2021. Transfer-Net: An Efective and Transparent Framework for Multi-hop Question Answering over Relation Graph. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 4149–4158.

Steinmetz, N.; and Sattler, K.-U. 2021. What is in the KGQA Benchmark Datasets? Survey on Challenges in Datasets for Question Answering on Knowledge Graphs. Journal on Data Semantics, 10(3): 241–265.

Sui, Y.; He, Y.; Ding, Z.; and Hooi, B. 2025. Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study over Open-Ended Question Answering.

In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 12685–12701.

Sun, H.; Bedrax-Weiss, T.; and Cohen, W. 2019. PullNet: Open Domain Question Answering with Iterative Retrieval on Knowledge Bases and Text. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2380–2390.

Sun, J.; Xu, C.; Tang, L.; Wang, S.; Lin, C.; Gong, Y.; Ni, L.; Shum, H.-Y.; and Guo, J. 2024. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph. In The Twelfth International Conference on Learning Representations.

Sun, Q.; Zhang, C.; Hu, Z.; Jin, Z.; Yu, J.; and Liu, L. 2023. Multi-hop Question Answering over Incomplete Knowledge Graph with Abstract Conceptual Evidence. Applied Intelligence, 53(21): 25731–25751.

Sun, Z.; Deng, Z.-H.; Nie, J.-Y.; and Tang, J. 2019. RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. arXiv preprint arXiv:1902.10197.

Trouillon, T.; Welbl, J.; Riedel, S.; Gaussier, É.; and Bouchard, G. 2016. Complex Embeddings for Simple Link Prediction. In Proceedings of the 33rd International Conference on Machine Learning, 2071–2080.

Wang, Y.; Lipka, N.; Rossi, R. A.; Siu, A.; Zhang, R.; and Derr, T. 2024. Knowledge Graph Prompting for Multi-Document Question Answering. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, 19206– 19214.

Wen, Y.; Wang, Z.; and Sun, J. 2024. MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models. In Proceedings of the 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 10370–10388. Association for Computational Linguistics.

Xiong, G.; Bao, J.; and Zhao, W. 2024. Interactive-KBQA: Multi-turn Interactions for Knowledge Base Question Answering with Large Language Models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 10561–10582.

Xiong, W.; Yu, M.; Chang, S.; Guo, X.; and Wang, W. Y. 2019. Improving Question Answering over Incomplete KBs with Knowledge-Aware Reader. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 4258–4264.

Xu, Y.; He, S.; Chen, J.; Wang, Z.; Song, Y.; Tong, H.; Liu, G.; Zhao, J.; and Liu, K. 2024. Generate-on-Graph: Treat LLM as both Agent and KG for Incomplete Knowledge Graph Question Answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 18410–18430.

Ye, X.; Xiao, L.; Zhang, C.; and Yamasaki, T. 2024. E-ReaRev: Adaptive Reasoning for Question Answering over Incomplete Knowledge Graphs by Edge and Meaning Extensions. In Natural Language Processing and Information Systems, 85–95.

Ye, X.; Yavuz, S.; Hashimoto, K.; Zhou, Y.; and Xiong, C. 2022. RNG-KBQA: Generation Augmented Iterative Ranking for Knowledge Base Question Answering. In Proceedings ofthe 60th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 6032–6043.

Yih, W.-t.; Chang, M.-W.; He, X.; and Gao, J. 2015. Semantic Parsing via Staged Query Graph Generation: Question Answering with Knowledge Base. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 1321–1331.

Yu, D.; Gu, Y.; Xiong, C.; and Yang, Y. 2023. CompleQA: Benchmarking the Impacts of Knowledge Graph Completion Methods on Question Answering. In Findings of the Association for Computational Linguistics: EMNLP 2023, 12748–12755.

Zan, D.; Wang, S.; Zhang, H.; Zhou, K.; Wu, W.; Zhao, W. X.; Wu, B.; Guan, B.; and Wang, Y. 2022. Complex Question Answering over Incomplete Knowledge Graph as N-ary Link Prediction. In 2022 International Joint Conference on Neural Networks (IJCNN), 1–8.

Zhang, L.; Jiang, Z.; Chi, H.; Chen, H.; ElKoumy, M.; Wang, F.; Wu, Q.; Zhou, Z.; Pan, S.; Wang, S.; and Ma, Y. 2025. Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking. In NeurIPS 2025 Datasets and Benchmarks Track.

Zhang, L.; Jiang, Z.; Chi, H.; Chen, H.; Elkoumy, M.; Wang, F.; Wu, Q.; Zhou, Z.; Pan, S.; Wang, S.; et al. 2026. Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking. Advances in Neural Information Processing Systems, 38.

Zhao, F.; Li, Y.; Hou, J.; and Bai, L. 2022. Improving Question Answering over Incomplete Knowledge Graphs with Relation Prediction. Neural Computing and Applications, 34(8): 6331–6348.

Zhou, D.; Zhu, Y.; Wang, X.; Zhou, H.; He, Y.; Chen, J.; Staab, S.; and Kharlamov, E. 2026. What Breaks Knowledge Graph based RAG? Benchmarking and Empirical Insights into Reasoning under Incomplete Knowledge. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), 2522–2538.