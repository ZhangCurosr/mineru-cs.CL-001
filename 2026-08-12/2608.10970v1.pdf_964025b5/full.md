# ReLTEx: Reliable LLM-based Taxonomy Expansion

Zeinab Ghamlouch<sup>1</sup> Mehwish Alam<sup>1</sup>

<sup>1</sup>Télécom Paris, Institut Polytechnique de Paris, France zeinab.b.ghamlouch@gmail.com mehwish.alam@telecom-paris.fr

## Abstract

Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in generating semantically relevant concepts and relations, making them promising tools for taxonomy enrichment. However, directly relying on LLM-generated expansions often leads to noisy, redundant, or hierarchically inconsistent structures, limiting their reliability for automated taxonomy expansion. In this paper, we present ReLTEx, a framework for reliable LLM-based taxonomy expansion. ReLTEx combines LLM-driven candidate generation with structure-aware validation and recursive expansion control to improve the consistency and quality of generated taxonomies by reducing hallucinations. We evaluate the proposed framework using benchmark taxonomies under a masked taxonomy expansion setting and compare multiple validation strategies. Experimental results, supported by both adapted evaluation metrics and human evaluation, demonstrate that ReLTEx produces more reliable and semantically coherent taxonomy expansions.

## 1 Introduction

Taxonomies provide structured hierarchical representations of knowledge and play a fundamental role in numerous applications, including knowledge graphs (Suchanek et al., 2024), semantic search and question answering (Navigli and Ponzetto, 2010), etc. By organizing concepts as parent–child relations, taxonomies enable semantic understanding, efficient navigation, and knowledge discovery. However, as domains evolve and new concepts continuously emerge, maintaining and expanding taxonomies manually becomes increasingly costly, time-consuming, and difficult to scale (Navigli and Ponzetto, 2010).

To address these challenges, extensive research has explored automated taxonomy expansion. Early approaches relied on lexico-syntactic patterns (Hearst, 1992), followed by distributional semantic methods (Zhang et al., 2018) and graphbased techniques (Pietrasik et al., 2024). Moreover, Pretrained Language Models (PLMs) have substantially improved taxonomy expansion by learning contextual representations of concepts and hierarchical relations (Liu et al., 2021). Despite these advances, automated taxonomy expansion remains challenging. A generated concept may be semantically ambiguous, as in “Bank”, which can refer to either a financial institution or the side of a river. Candidate generation may also introduce noise or redundancy, for example by proposing both “Question” and “Inquiry”. Moreover, an otherwise valid concept may be attached to an inappropriate parent, such as placing “Answer” under “Creative-Work” rather than “Comment”. These errors can compromise hierarchical coherence and propagate during recursive expansion.

Recent advances in Large Language Models (LLMs) have created new opportunities for taxonomy enrichment which uses prompting for taxonomy construction (Zeng et al., 2024) and hierarchical attachment prediction (Mishra et al., 2024). Because of their extensive parametric knowledge and strong generative capabilities, LLMs can infer hierarchical relations and propose semantically relevant concepts in a zero-shot setting, making them attractive for expanding existing taxonomies, particularly in domains where curated resources are scarce or rapidly evolving. However, without explicit structural verification, LLM-generated concepts may violate the taxonomy hierarchy, and such errors can accumulate during recursive expansion.

To address these challenges, we propose ReL-TEx<sup>1</sup>, a framework for reliable LLM-based taxonomy expansion. ReLTEx combines zero-shot LLM-based candidate generation with structureaware validation and recursive expansion control.

Candidate parent–child relations are verified using a path-aware classifier trained on taxonomy structures, allowing structurally inconsistent generations to be filtered before insertion into the taxonomy. Recursive stopping criteria further regulate recursive expansion and reduce the propagation of erroneous generations.

We evaluate ReLTEx on benchmark taxonomies under a masked taxonomy expansion setting, where hidden concepts must be recovered from partial taxonomies. Our evaluation consists of standard and adapted evaluation metrics, human assessment, and LLM based assessment along with an ablation study. The main contributions of this work are as follows, we propose:

• A zero-shot LLM-based taxonomy expansion framework that exploits contextual information from the taxonomy hierarchy to generate candidate concepts.

• A structure-aware validation mechanism that mitigates hallucinated and structurally inconsistent parent–child relations, improving the reliability of recursive taxonomy expansion.

• A comprehensive evaluation protocol based on adapted evaluation metrics, human evaluation, and LLM based assessment.

This paper is organized as follows. Section 2 reviews related work while Section 3 defines the taxonomy expansion problem. Section 4 presents ReLTEx. Section 5 describes the datasets and evaluation metrics, and Section 6 presents the experimental results. Finally, Section 7 concludes the paper.

## 2 Related Work

Recent methods formulate taxonomy expansion as node attachment problem, where predefined candidate concepts are inserted into an existing taxonomy by predicting their most appropriate parent node. In the following we categorize these methods into structure based and LLM-based methods.

Structure based Methods. TaxoExpan (Shen et al., 2020) leverages Graph Neural Networks and self-supervised learning to insert unseen concepts through ego-graph representations. Building on the use of structural information, TEMP (Liu et al., 2021) models taxonomy paths using PLMs and optimizes a dynamic margin ranking objective over positive and negative insertion paths. STEAM (Yu et al., 2020) adopts a multi-view co-training framework that combines distributed, contextual, and lexico-syntactic representations through minipath structures. Other approaches have similarly focused on improving hierarchical attachment through graph-based representations (Shang et al., 2020), contrastive learning strategies (Niu et al., 2024), or PLM embeddings (Takeoka et al., 2021).

Despite their effectiveness, most existing taxonomy expansion approaches (Shen et al., 2020; Margiotta et al., 2023) operate under a constrained setting in which the candidate concepts are provided in advance instead of generating entirely new concepts dynamically.

LLM-based Methods. TaxoGlimpse (Sun et al., 2024) assesses the taxonomic knowledge encoded in LLMs across general and specialized taxonomies. The results show that the performance of an LLM deteriorates for specialized domains and deeper taxonomy levels. However, their work focuses on evaluating taxonomic knowledge rather than expanding an existing hierarchy. Chain-of-Layer (Zeng et al., 2024) iteratively induces a taxonomy from a given set of entities through layerwise prompting and an ensemble-based relation filtering mechanism. FLAME (Mishra et al., 2024) leverages LLMs to predict the most appropriate parent for a query concept that is known in advance.

Taxoria (Ghamlouch and Alam, 2025) is the closest work to ours, as it recursively generates novel concepts based on an existing taxonomy without relying on a predefined candidate pool. However, its validation is based primarily on embedding-based semantic similarity and does not explicitly model the structural compatibility of a generated parent– child relation. Consequently, erroneous or weakly related generations may be accepted and propagated during recursive expansion.

Unlike prior taxonomy expansion methods that either attach predefined concepts or rely primarily on semantic similarity, ReLTEx combines openended LLM generation with path-aware structural validation and recursive expansion control. By validating each generated parent–child relation in its hierarchical context before insertion.

## 3 Problem Formulation

A taxonomy can be represented as T = (V, E), where V denotes the set of concepts (nodes) and E represents parent–child relations between the concepts. Given an initial seed taxonomy $T _ { 0 } =$ $( V _ { 0 } , E _ { 0 } )$ , the goal of taxonomy expansion is to enrich the taxonomy with semantically relevant concepts while preserving hierarchical consistency. Formally, given a node $v \in V _ { 0 }$ and its local hierarchical context, the objective is to generate a set of candidate child concepts $C _ { v } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { k } \}$ Where k denotes the number of candidate child concepts generated for node v, and $c _ { i }$ is a potential child concept of v, where $i = 1 , \ldots , k .$

The generated candidates may be hallucinated, redundant, or structurally inconsistent. Thus, the goal is to select a curated subset suitable for safe taxonomy integration. The resulting expanded concept set can therefore be expressed as $V ^ { \prime } = V _ { 0 } \cup A _ { v }$ where $A _ { v } \subseteq C _ { v }$ denotes the subset of generated candidates selected for insertion into the taxonomy.

## 4 ReLTEx

Figure 1 shows an overview of ReLTEx framework. Starting from a seed taxonomy, ReLTEx expands each taxonomy node in three stages: (A) LLMbased candidate generation, (B) structure-aware validation, and (C) recursive expansion using a stopping mechanism. The following subsections describe each stage in detail.

## 4.1 LLM-based Candidate Generation

ReLTEx performs recursive taxonomy expansion through a depth-first traversal of the taxonomy. For each target node, the framework constructs a prompt using its local hierarchical context, consisting of the path from the taxonomy root to the target node together with its existing children. The ancestor path provides the semantic context of the current taxonomy branch, while the existing children illustrate the desired level of abstraction, guiding the LLM to generate semantically relevant child concepts at the appropriate level of granularity. The language model is instructed to generate new child concepts considering the appropriate granularity level while avoiding duplicates, synonyms, and simple rephrasing of the existing or newly generated siblings (Appendix E shows the prompt). For each parent node, the LLM is prompted to generate a fixed number k of candidate child concepts. The parameter k controls the branching factor of the expanded taxonomy and can be adjusted depending on the desired level of expansion. After generation, candidate names undergo basic lexical normalization. Exact lexical duplicates under this canonical representation, as well as candidates matching existing children or sibling nodes, are removed before the remaining candidates are passed to the validation stage.

## 4.2 Structure-Aware Candidate Validation

We investigated three validation strategies: (i) semantic similarity filtering, which measures the semantic compatibility between the generated concept and its local taxonomy context; (ii) LLMbased validation, where an LLM judges whether the generated concept represents a valid, nonredundant child of the parent; and (iii) classifierguided validation, which predicts the structural validity of the parent–child relation using a classifier trained on taxonomy edges. In this subsection we focus on classifier guided validation.

Unlike semantic similarity measures, the proposed classifier learns structural compatibility from annotated taxonomy relations by jointly considering the hierarchical context, the parent concept, and the generated child concept. The validation module is formulated as a binary classification task using a DistilRoBERTa encoder fine-tuned on labeled parent–child relation pairs. The classifier is trained on positive and negative examples derived from the seed taxonomy, enabling it to discriminate between valid and invalid hierarchical relations. Given the hierarchical path $p ,$ the parent concept v, and a generated candidate child concept $c _ { i } ,$ represented as Path [SEP] Parent [SEP] Child, the classifier outputs a confidence score

$$
s ( p , v , c _ { i } ) \in [ 0 , 1 ] ,
$$

defined as the softmax probability assigned to the positive class, representing the likelihood that the relation $( v , c _ { i } )$ corresponds to a valid taxonomy edge within the hierarchical context $p .$ Candidate concepts with $s ( p , v , c _ { i } ) \geq \tau$ , where τ denotes the validation threshold, are accepted and incorporated into the taxonomy; otherwise, they are discarded.

Positive examples correspond to valid taxonomy edges, whereas negative examples are generated automatically using hierarchy-aware perturbations that produce semantically plausible but structurally incorrect relations. We construct the following hard negative examples:

• Reversed edges, obtained by reversing a valid parent–child relation, e.g., Comment → Answer becomes Answer → Comment (see Figure 1 for running example).

![](images/7f60a84211e97a308a049a1d25c11b6e45f8993863a52f80924237b1fcca1a7f.jpg)  
Figure 1: Overview of the ReLTEx framework.

• Sibling confusions, where the correct parent is replaced by one of its siblings, e.g., Comment → Answer becomes Book → Answer.

• Grandparent–child confusions, where the immediate parent is replaced by its parent, e.g., Comment → Answer becomes Creative-Work → Answer.

• Same-depth mismatches, where the child is attached to a concept at the same hierarchical level as its true parent, e.g., Article → Answer.

• Nearby hierarchy confusions, where the correct parent is replaced by a semantically related concept from a neighboring branch, e.g., Review → Answer.

• Random invalid relations, obtained by randomly pairing unrelated concepts, e.g., Organization → Answer.

Each training instance is represented as Path [SEP] Parent [SEP] Child, where the path corresponds to the ancestor sequence from the taxonomy root to the parent concept. By training on both positive examples and hierarchy-aware hard negatives, the classifier learns structural compatibility rather than relying solely on semantic similarity.

## 4.3 Recursive Expansion

Recursive generative expansion may progressively introduce semantic drift and uncontrolled taxonomy growth. For example, once an incorrect parent– child relation is accepted, subsequent expansion may continue from that erroneous node and generate concepts that increasingly diverge from the intended taxonomy branch. To mitigate this issue, ReLTEx reuses the confidence scores assigned by the classifier during child validation to regulate recursive expansion at the branch level.

Only the concepts accepted during the validation step are eligible for further recursive expansion. Let $A _ { v }$ denote the set of accepted child concepts generated for node v:

$$
A _ { v } = \{ c _ { i } \in C _ { v } \mid s ( p , v , c _ { i } ) \geq \tau \}
$$

ReLTEx first requires a minimum number of accepted children: $| A _ { v } | \ge m$ , where m is the minimum number of accepted children required to continue expanding the branch. $\mathrm { I f } \ | A _ { v } | < m$ , expansion at node v terminates. If m = 1, a branch stops when none of its generated candidates passes the validation threshold.

When $| A _ { v } | \geq m$ , ReLTEx estimates the overall reliability of the current expansion step using the mean confidence of the accepted children:

$$
S ( v ) = { \frac { 1 } { | A _ { v } | } } \sum _ { c _ { i } \in A _ { v } } s ( p , v , c _ { i } ) .
$$

Averaging considers all accepted children and provides a normalized estimate of the reliability of the expansion step, without being determined by a single exceptionally high- or low-confidence prediction.

If we have S(v) and S(parent(v)) (mean confidence of its parent node), the expansion below v continues only if

$$
S ( v ) \geq S ( \mathrm { p a r e n t } ( v ) ) - \delta ,
$$

where δ is the maximum allowable decrease in average classifier confidence between two consecutive expansion levels. For the initial expansion step, where no parent confidence score is available, this criterion is not applied. This recursive stopping mechanism recursively expands only branches whose confidence remains sufficiently stable.

## 5 Experimental Setup

## 5.1 Datasets and LLMs

We evaluate ReLTEx on two taxonomy benchmarks that differ substantially in size and structure: (i) SemEval-2016 Task 13 Environment taxonomy (Bordea et al., 2016), a standard benchmark for taxonomy expansion; (ii) Schema. $o r g ^ { 2 }$ a large-scale real-world taxonomy that covers a broad range of domains and semantic concepts. Dataset statistics are summarized in Table 1.

Table 1: Statistics of the taxonomy benchmarks. |N| denotes #nodes, |E| denotes #edges, and |D| the maximum depth (with the root counted as level 1).
<table><tr><td>Dataset</td><td>|N|</td><td>|E|</td><td>|D|</td></tr><tr><td>SemEval-env</td><td>263</td><td>262</td><td>6</td></tr><tr><td>Schema.org</td><td>1,143</td><td>1,142</td><td>6</td></tr></table>

Experiments are conducted using four opensource language models from different families: Llama3.2:3B, Mistral:7B, Qwen3:8B, and DeepSeek-R1:8B. These models were selected to cover a diverse range of capabilities, including lightweight instruction following, general-purpose generation, enhanced reasoning, and explicit reasoning via a thinking model.

## 5.2 Evaluation Metrics

Our evaluation builds upon metrics commonly adopted in taxonomy expansion literature (Liu et al., 2021; Zhang et al., 2024), including Recall@K, Mean Reciprocal Rank (MRR), and the Wu & Palmer similarity measure. Unlike traditional taxonomy expansion, ReLTEx performs generative expansion, where language models may produce multiple candidate concepts, semantically equivalent variants, or valid concepts attached under alternative parent nodes. To account for these characteristics, we adapt the classical evaluation metrics to the generative setting while preserving their original objectives. Specifically, we distinguish between concept recovery and structural attachment quality, and additionally incorporate semantic matching to account for lexical variability in generated concepts.

The primary quantitative evaluation follows a masked taxonomy expansion protocol (described in Section 6), where a subset of taxonomy nodes is hidden and treated as ground truth. The enriched taxonomy is generated from the remaining seed taxonomy, and the recovered concepts are evaluated using the metrics defined below.

Let Y denote the set of hidden concepts, with $n = | Y |$ . For each hidden concept $y \in Y$ , let $v _ { y }$ denote the original parent node from which y was masked. Let $r a n k ( y )$ denote the ranking position of $y$ within the accepted candidate set $A _ { v _ { y } }$ (see Section 4 for details).

We define the set of locally recovered hidden concepts as

$$
\Gamma _ { \mathrm { l o c a l } } = \left\{ y \in Y \mid y \in A _ { v _ { y } } \right\} ,
$$

where a hidden concept is considered recovered if it appears among the accepted generated children of its original parent node.

Recall@K (R@K) measures the proportion of hidden concepts that are successfully recovered under their original parent nodes:

$$
R ^ { \mathbb { Q } } K = { \frac { | \Gamma _ { \mathrm { l o c a l } } | } { n } } .
$$

SoftRecall@K (SR@K). Exact lexical matching may underestimate taxonomy expansion quality when semantically equivalent concepts are generated. To account for this, we introduce SR@K.

$$
S R @ K ( y ) = \left\{ \begin{array} { l l } { 1 , } & { \mathrm { i f ~ } \underset { c _ { i } \in A _ { v _ { y } } } { \operatorname* { m a x } } \sin ( y , c _ { i } ) \geq \rho , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.
$$

where $\begin{array} { r l r } { y } & { { } \in } & { Y } \end{array}$ is the hidden concept, sim(·, ·) denotes cosine similarity between embeddings generated by the BAAI/bge-small-en-v1.5 model (Xiao et al., 2023). We set $\rho = 0 . 8 5$ , which provides a conservative balance between recognizing semantically equivalent concepts while avoiding matches between only loosely related concepts. The final SR@K score is computed as

$$
S R @ K = \frac { 1 } { n } \sum _ { y \in Y } S R @ K ( y ) .
$$

Mean Reciprocal Rank (MRR). Accepted candidate concepts are ranked according to decreasing classifier confidence score $s ( p , v , c _ { i } )$ We report two complementary variants of MRR: (i)

$M R R _ { \mathrm { o v e r a l l } }$ and (ii) $M R R _ { \mathrm { f o u n d } } . \quad M R R _ { \mathrm { o v e r a l l } }$ measures ranking quality over all hidden concepts. Recovered concepts contribute the reciprocal of their ranking position, while hidden concepts that are not recovered receive a score of zero, thereby penalizing failure to recover hidden concepts:

$$
M R R _ { \mathrm { o v e r a l l } } = \frac { 1 } { n } \sum _ { y \in Y } \left\{ \frac { 1 } { r a n k ( y ) } , \right. \mathrm { i f } y \in A _ { v _ { y } } ,
$$

$M R R _ { \mathrm { f o u n d } }$ evaluates ranking quality only for successfully recovered hidden concepts:

$$
M R R _ { \mathrm { f o u n d } } = \frac { 1 } { | \Gamma _ { \mathrm { l o c a l } } | } \sum _ { y \in \Gamma _ { \mathrm { l o c a l } } } \frac { 1 } { r a n k ( y ) } .
$$

Wu & Palmer (WuP) Similarity. To evaluate the structural quality of node generation, we adapt the WuP similarity to compare predicted parent placements against the original taxonomy structure. For each hidden concept $y \in Y$ , let

$$
V _ { y } ^ { \mathrm { p r e d } } = \{ v \mid y \in A _ { v } \}
$$

denote the set of parent nodes under which a hidden concept y was generated. The attachment similarity score is then defined as

$$
w ( y ) = \operatorname* { m a x } _ { v \in V _ { y } ^ { \mathrm { p r e d } } } \frac { 2 \cdot d e p t h ( L C A ( v , v _ { y } ) ) } { d e p t h ( v ) + d e p t h ( v _ { y } ) } .
$$

where $L C A ( \cdot , \cdot )$ denotes the Lowest Common Ancestor. Higher similarity values indicate that the predicted parent node is structurally closer to the original parent node. We further define the set of globally recovered hidden concepts as

$$
\Gamma _ { \mathrm { g l o b a l } } = \{ y \in Y \mid V _ { y } ^ { \mathrm { p r e d } } \neq \emptyset \} .
$$

We report two variants of the WuP score. The first variant measures attachment quality over all hidden concepts. Successfully generated concepts contribute their WuP score, while hidden concepts that are not successfully generated are scored zero.

$$
W u P _ { \mathrm { o v e r a l l } } = \frac { 1 } { n } \sum _ { y \in Y } \left\{ w ( y ) , \begin{array} { l l } { w ( y \in \Gamma _ { \mathrm { g l o b a l } } , } \\ { 0 , } \end{array} \right.
$$

The second variant evaluates attachment quality only for successfully generated concepts:

$$
W u P _ { \mathrm { f o u n d } } = \frac { 1 } { | \Gamma _ { \mathrm { g l o b a l } } | } \sum _ { y \in \Gamma _ { \mathrm { g l o b a l } } } w ( y ) .
$$

## 6 Evaluation Results

We evaluate ReLTEx using three complementary protocols: (i) Masked Taxonomy Expansion, (ii) Human Evaluation, (iii) LLM based taxonomy evaluation. .

## 6.1 Masked Taxonomy Expansion Benchmark

We evaluate concept recovery using a masked taxonomy expansion benchmark, following the standard evaluation protocol (Xu et al., 2024). We remove 20% of the nodes from the seed taxonomy and treat them as hidden test concepts. The remaining taxonomy serves as the seed taxonomy provided to ReLTEx, while the hidden concepts constitute the gold standard used for evaluation. Only leaf nodes are considered eligible for masking in order to preserve the overall taxonomy structure.

Masking is performed under two constraints. (i) Hidden nodes are selected only from those parent nodes retaining at least one visible child after masking, preventing the removal of all local structural context. (ii) A maximum of five hidden children is allowed under any single parent node. These constraints prevent a small number of high-degree parent nodes from dominating the evaluation process and ensures a more balanced distribution of hidden concepts across the taxonomy.

The generator $\mathrm { L L M } ^ { 3 }$ produces $k = 5$ candidate child concepts for each parent node. The root node is excluded from expansion. Generated parent– child candidates are then validated. Candidates are accepted only if their confidence score $\geq 0 . 9 0$ Although ReLTEx supports recursive taxonomy expansion, benchmark experiments are performed exclusively on the masked seed taxonomy. Newly generated concepts are not recursively expanded.

Since existing taxonomy expansion methods assume a predefined candidate set rather than generating new concepts, no directly comparable baseline exists for our setting. Table 2 summarizes the benchmark results under the masked taxonomy expansion setting on the SemEval Environment and Schema.org taxonomies. The higher numbers (shown in bold) indicate better performance or quality in the rest of this paper.

Overall, all evaluated models demonstrate the ability to recover hidden taxonomy concepts. As expected, the larger and more diverse Schema.org taxonomy is substantially more challenging than the SemEval Environment taxonomy, leading to lower recovery and attachment scores across all models. Nevertheless, the relative ranking of the models remains largely consistent across both benchmarks. Mistral achieves the strongest overall performance across both datasets, obtaining the highest R@K, MRR<sub>o</sub>, and $\mathsf { W u P } _ { o }$ scores on Schema.org while remaining highly competitive on SemEval. Llama3.2 performs best on the smaller SemEval taxonomy, achieving the highest R@K, S@K, and $\operatorname { W u P } _ { o } .$ , indicating strong concept recovery and structural placement. Qwen3 consistently produces competitive semantic recovery performance, achieving the highest S@K on Schema.org, suggesting a greater ability to generate semantically related concepts beyond exact lexical matches. DeepSeek-R1 generally obtains lower recovery scores and exhibits larger standard deviations across several metrics, indicating lower stability across different masking configurations.

Table 2: Results of the masked taxonomy expansion benchmark.
<table><tr><td>Metric</td><td>Llama3.2</td><td>Mistral</td><td>Qwen3</td><td>DeepSeek</td></tr><tr><td colspan="5">SemEval Environment</td></tr><tr><td> $\mathbb { R } \ @ \mathbb { K }$ </td><td> $4 4 . 2 3 \pm 2 . 1 1$ </td><td> $4 2 . 3 1 \pm 1 . 2 2$ </td><td> $4 2 . 9 5 \pm 0 . 9 9$ </td><td> $2 8 . 8 5 \pm 6 . 2 0$ </td></tr><tr><td> $\mathrm { S R } @ \mathrm { K }$ </td><td> ${ \pm 3 . 2 1 \pm 2 . 6 3 }$ </td><td> $4 6 . 4 7 \pm 1 . 8 9$ </td><td> $4 7 . 7 6 \pm 2 . 2 5$ </td><td> $4 0 . 7 0 \pm 4 . 7 8$ </td></tr><tr><td> $\mathrm { M R R } _ { o }$ </td><td> $2 3 . 4 7 \pm 3 . 1 0$ </td><td> ${ \bf 2 3 . 8 1 \pm 0 . 5 2 }$ </td><td> $2 1 . 6 9 \pm 1 . 0 8$ </td><td> $1 4 . 5 3 \pm 2 . 6 1$ </td></tr><tr><td> $\operatorname { M R R } _ { f }$ </td><td> $5 2 . 9 2 \pm 5 . 0 2$ </td><td> ${ \pm } 5 5 . 9 0 \pm 1 . 7 4$ </td><td> $5 0 . 5 4 \pm 3 . 1 1$ </td><td> $5 0 . 7 4 \pm 2 . 6 0$ </td></tr><tr><td> $\mathsf { W u P } _ { o }$ </td><td> ${ \bf 4 8 . 1 7 \pm 1 . 9 9 }$ </td><td> $4 4 . 5 1 \pm 1 . 6 6$ </td><td> $4 5 . 2 2 \pm 1 . 7 3$ </td><td> $3 1 . 7 7 \pm 5 . 9 3$ </td></tr><tr><td> $\operatorname { W u P } _ { f }$ </td><td> $9 2 . 9 3 \pm 3 . 1 8$ </td><td> $\mathbf { 9 4 . 6 4 } \pm 3 . 4 0$ </td><td> $9 6 . 6 3 \pm 1 . 9 0$ </td><td> $9 2 . 1 9 \pm 6 . 9 0$ </td></tr><tr><td colspan="5">Schema.org</td></tr><tr><td> $\mathbb { R } \ @ \mathbb { K }$ </td><td> $1 4 . 0 4 \pm 2 . 6 9$ </td><td> ${ \bf 2 3 . 2 8 \pm 0 . 8 0 }$ </td><td> $1 9 . 6 6 \pm 1 . 0 6$ </td><td> $1 1 . 0 6 \pm 2 . 5 2$ </td></tr><tr><td> $\mathrm { S R } @ \mathrm { K }$ </td><td> $2 2 . 9 9 \pm 2 . 2 4$ </td><td> $3 1 . 7 0 \pm 2 . 1 1$ </td><td> ${ \bf 3 1 . 7 9 \pm 0 . 9 6 }$ </td><td> $1 9 . 2 3 \pm 2 . 6 2$ </td></tr><tr><td> $\mathrm { M R R } _ { o }$ </td><td> $7 . 8 6 \pm 1 . 1 8$ </td><td> ${ \bf 1 1 . 9 7 \pm 0 . 4 6 }$ </td><td> $1 1 . 0 2 \pm 1 . 1 8$ </td><td> $6 . 6 3 \pm 1 . 6 7$ </td></tr><tr><td> $\operatorname { M R R } _ { f }$ </td><td> $5 2 . 1 2 \pm 4 . 8 6$ </td><td> $4 9 . 7 2 \pm 1 . 6 2$ </td><td> $5 3 . 9 4 \pm 3 . 5 6$ </td><td> ${ \pm } { \bf 5 5 . 3 9 } \pm { \bf 4 . 6 7 }$ </td></tr><tr><td> $\mathbf { W u P } _ { o } ^ { \prime }$ </td><td> $2 1 . 7 5 \pm 3 . 2 9$ </td><td> ${ \bf 2 7 . 5 7 \pm 1 . 2 8 }$ </td><td> $2 4 . 9 2 \pm 1 . 6 1$ </td><td> $1 3 . 7 5 \pm 4 . 5 2$ </td></tr><tr><td> $\operatorname { W u P } _ { f }$ </td><td> $8 0 . 2 7 \pm 1 . 1 0$ </td><td> $\mathbf { 9 2 . 3 2 \pm 3 . 0 3 }$ </td><td> $9 0 . 6 1 \pm 2 . 5 2$ </td><td> $8 5 . 7 1 \pm 3 . 0 3$ </td></tr></table>

## 6.2 Human Evaluation

LLMs may generate valid novel concepts that are absent from the ground truth. To complement the automatic evaluation, we conduct a human assessment of the generated taxonomy expansions. We evaluate the expanded taxonomy obtained from the run achieving the highest R@K.

For each taxonomy, we randomly sample 20% of the eligible parent nodes, providing representative coverage while keeping the manual annotation effort feasible. Eligible parents correspond to nonroot nodes that retain at least one visible child in the masked seed taxonomy. The same sampled parent nodes are evaluated across all four models.

To avoid model-specific bias, the four models are anonymized. We further randomize the order of both the sampled parent nodes and the generated concepts. A web-based interface<sup>4</sup> shows the taxonomy context and allows consistent annotation.

Table 3: Human evaluation results using majority voting across three annotators.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Model</td><td colspan="3">Expansion Quality</td><td colspan="2">Recovery</td></tr><tr><td>HC</td><td>GC</td><td>NR</td><td>ER</td><td>SR</td></tr><tr><td rowspan="4">SemEval Env</td><td>Llama3.2</td><td>0.978</td><td>0.978</td><td>0.978</td><td>0.200</td><td>0.300</td></tr><tr><td>DeepSeek-R1</td><td>0.960</td><td>0.960</td><td>0.920</td><td>0.200</td><td>0.300</td></tr><tr><td>Mistral</td><td>0.960</td><td>0.940</td><td>0.860</td><td>0.400</td><td>0.400</td></tr><tr><td>Qwen3</td><td>0.980</td><td>0.980</td><td>0.940</td><td>0.300</td><td>0.400</td></tr><tr><td rowspan="4">Schema.org</td><td>Llama3.2</td><td>0.898</td><td>0.932</td><td>0.864</td><td>0.103</td><td>0.144</td></tr><tr><td>DeepSeek-R1</td><td>0.988</td><td>0.988</td><td>0.940</td><td>0.076</td><td>0.143</td></tr><tr><td>Mistral</td><td>0.978</td><td>1.000</td><td>1.000</td><td>0.101</td><td>0.202</td></tr><tr><td>Qwen3</td><td>1.000</td><td>1.000</td><td>0.989</td><td>0.112</td><td>0.178</td></tr></table>

For each sampled parent node, annotators are presented with its hierarchical path, its existing children, the accepted generated children, and, when applicable, the hidden gold children removed during masking (see Appendix B for the snapshot of the annotation interface). The interface distinguishes between two evaluation settings. In recovery cases, hidden gold children exist for the current parent, allowing both recovery and enrichment to be assessed. In novel enrichment cases, no hidden gold child exists under the parent, and the evaluation focuses solely on the quality of the generated concepts. Each generated concept is independently evaluated according to the following criteria: Hierarchical Correctness (HC), Granularity Consistency (GC), Non-Redundancy (NR), Exact Recovery (ER), and Semantic Recovery (SR).

Three independent annotators evaluated all eight sample sets. Final labels are obtained using majority voting across the three annotations for each generated concept. The inter-annotator agreement was measured using Fleiss’ κ, which is 0.695.

Table 3 presents the final results. Expansionquality metrics (HC, GC, and NR) are computed over all sampled parent nodes, whereas ER and SR are evaluated only on the subset of masked concepts. Despite being measured on this more restrictive evaluation set, the recovery scores demonstrate that the generated concepts frequently recover or closely match the hidden taxonomy concepts.

Overall, the generated expansions receive consistently high human judgments for hierarchical correctness, granularity consistency, and nonredundancy across both datasets, indicating that the accepted concepts are generally well positioned within the taxonomy and remain consistent with the abstraction level of the surrounding branch.

On SemEval Environment, Qwen3 achieves the strongest performance in HC and GC, while Llama3.2 produces the highest non-redundancy score. Mistral obtains the best exact recovery and ties with Qwen3 for the highest SR, indicating a stronger ability to recover concepts removed during masking.

On Schema.org, Qwen3 achieves perfect HC and GC. Mistral reaches perfect GC and NR together with the highest SR, whereas DeepSeek-R1 also demonstrates strong expansion quality despite lower recovery performance.

## 6.3 LLM-Based Taxonomy Evaluation

We complement our evaluation using LITE (Zhang et al., 2025), an LLM-based framework for assessing the semantic and structural quality of taxonomies. LITE defines four measures: (i) Single Concept Accuracy (SCA): evaluates the semantic clarity, validity, and coherence of individual taxonomy concepts; (ii) Hierarchy Relationship Rationality (HRR): evaluates whether parent–child relations represent logically consistent hierarchical dependencies. (iii) Hierarchy Relationship Exclusivity (HRE): evaluates whether sibling concepts are semantically distinct and non-overlapping. (iv) Hierarchy Relationship Independence (HRI): evaluates the structural independence of concepts by measuring redundancy and semantic overlap among siblings.

Following LITE, we assess taxonomy quality using LLM-based judgments. While the original framework reports HRE and HRI separately, our implementation returns a single score jointly reflecting both criteria, which we report as the HRE/HRI score. LITE is further applied to the fully expanded taxonomies produced by the ReL-TEx (instead of masking). As recursive expansion increases the number of generated nodes at each level, using the benchmark generation setting (k = 5 candidates per parent) results in large taxonomies with limited benefit. Therefore, we reduce the generation to k = 3. We evaluate both the seed taxonomy and the enriched taxonomy, allowing direct comparison before and after enrichment.

We replace the proprietary evaluator in LITE with three open-source instruction-tuned LLMs deployed locally through Ollama<sup>5</sup>: Llama3.2, Qwen3:8B, and DeepSeek-R1:8B. The enriched taxonomies are generated using Mistral:7B, selected as the generator due to its strongest overall performance in the masked taxonomy expansion benchmark. The three LLMs are used exclusively as evaluators. Tables 4 and 5 report the LITE scores for the seed and the enriched taxonomy. Improvements over the seed taxonomy show that recursive enrichment preserves or enhances the quality while introducing new concepts.

Table 4: LITE evaluation on the SemEval Environment.
<table><tr><td>Evaluator</td><td>Taxonomy</td><td>SCA</td><td>HRR</td><td>HRE/HRI</td></tr><tr><td rowspan="2">Llama3.2</td><td>Seed</td><td>8.01</td><td>8.19</td><td>7.74</td></tr><tr><td>ReLTEx</td><td>7.92</td><td>8.24</td><td>7.75</td></tr><tr><td rowspan="2">Qwen3</td><td>Seed</td><td>7.42</td><td>6.82</td><td>7.83</td></tr><tr><td>ReLTEx</td><td>7.31</td><td>6.96</td><td>8.88</td></tr><tr><td rowspan="2">DeepSeek</td><td>Seed</td><td>7.71</td><td>7.43</td><td>7.03</td></tr><tr><td>ReLTEx</td><td>7.73</td><td>8.78</td><td>8.15</td></tr></table>

Table 5: LITE evaluation on the Schema.org.
<table><tr><td>Evaluator</td><td>Taxonomy</td><td>SCA</td><td>HRR</td><td>HRE/HRI</td></tr><tr><td rowspan="2">Llama3.2</td><td>Seed</td><td>8.18</td><td>8.05</td><td>7.95</td></tr><tr><td>ReLTEx</td><td>7.94</td><td>8.29</td><td>8.30</td></tr><tr><td rowspan="2">Qwen3</td><td>Seed</td><td>8.35</td><td>8.95</td><td>8.59</td></tr><tr><td>ReLTEx</td><td>8.64</td><td>8.68</td><td>8.76</td></tr><tr><td rowspan="2">DeepSeek</td><td>Seed</td><td>8.19</td><td>8.67</td><td>8.05</td></tr><tr><td>ReLTEx</td><td>8.07</td><td>8.68</td><td>7.94</td></tr></table>

Across both taxonomies, the enriched taxonomies achieve high SCA, indicating that recursive expansion preserves semantic quality. HRR is generally maintained or improved after enrichment, suggesting that generated parent–child relations remain logically coherent. Similar trends are observed for HRE/HRI, with most evaluators assigning equal or higher scores to the enriched taxonomies, particularly on the Environment taxonomy.

The ablation study, qualitative error analysis, and expanded taxonomy statistics are provided in Appendix D, Appendix C, and Appendix E, respectively.

## 7 Conclusion

We presented ReLTEx, a framework for taxonomy enrichment using LLMs. Unlike existing taxonomy expansion approaches that assume candidate concepts to be already available, ReLTEx jointly performs concept generation and structureaware validation through a classifier-based validation module and recursive expansion mechanism. The proposed evaluation protocol combines automatic benchmarking, human assessment, and LLMbased taxonomy evaluation, providing complementary perspectives on taxonomy quality beyond concept recovery alone.

## Limitations

Our experiments rely on relatively compact opensource language models to enable efficient and reproducible experimentation. Larger and more capable models may further improve the quality of generated taxonomy expansions.

In addition, the proposed classifier is a learned approximation of hierarchical validity and may occasionally accept incorrect parent–child relations. Finally, recursive LLM-based taxonomy generation remains an emerging task without a standardized benchmark or directly comparable generation baseline, making comprehensive system-level comparisons difficult.

## Ethical Considerations

The human evaluation involved three volunteer annotators following predefined annotation guidelines. The collected annotations were used solely for research purposes, and no sensitive personal information is reported in this work.

The taxonomies used are publicly available benchmark resources and contain no personal or sensitive data.

Since ReLTEx relies on large language models for concept generation, it may inherit biases present in the underlying models and their training data, potentially affecting the generated concepts and their hierarchical placement. Although the proposed validation module mitigates these issues, some errors may still remain.

## References

Georgeta Bordea, Els Lefever, and Paul Buitelaar. 2016. SemEval-2016 task 13: Taxonomy extraction evaluation (TExEval-2). In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pages 1081–1091, San Diego, California. Association for Computational Linguistics.

Zeinab Ghamlouch and Mehwish Alam. 2025. Enriching taxonomies using large language models. In ECAI 2025-28th European Conference on Artificial Intelligence (Demo Track).

Marti A Hearst. 1992. Automatic acquisition of hyponyms from large text corpora. In COLING 1992 volume 2: The 14th international conference on computational linguistics.

Zichen Liu, Hongyuan Xu, Yanlong Wen, Ning Jiang, Haiying Wu, and Xiaojie Yuan. 2021. Temp: Taxonomy expansion with dynamic margin loss through taxonomy-paths. In Proceedings of the 2021 conference on empirical methods in natural language processing, pages 3854–3863.

Daniele Margiotta, Danilo Croce, and Roberto Basili. 2023. Taxosbert: Unsupervised taxonomy expansion through expressive semantic similarity. In International Conference on Deep Learning Theory and Applications, pages 295–307. Springer.

Sahil Mishra, Ujjwal Sudev, and Tanmoy Chakraborty. 2024. Flame: Self-supervised low-resource taxonomy expansion using large language models. ACM Transactions on Intelligent Systems and Technology.

Roberto Navigli and Simone Paolo Ponzetto. 2010. Babelnet: Building a very large multilingual semantic network. In Proceedings of the 48th annual meeting of the association for computational linguistics, pages 216–225.

Yuhang Niu, Hongyuan Xu, Ciyi Liu, Yanlong Wen, and Xiaojie Yuan. 2024. Contrastive representation learning for self-supervised taxonomy completion. In IJCAI, volume 8, pages 6442–6450.

Marcin Pietrasik, Marek Reformat, and Anna Wilbik. 2024. Non-parametric path based model for taxonomy induction in knowledge graphs. In Belgium netherlands conference on artificial intelligence.

Chao Shang, Sarthak Dash, Md Faisal Mahbub Chowdhury, Nandana Mihindukulasooriya, and Alfio Gliozzo. 2020. Taxonomy construction of unseen domains via graph-based cross-domain knowledge transfer. In Proceedings of the 58th annual meeting of the Association for Computational Linguistics, pages 2198–2208.

Jiaming Shen, Zhihong Shen, Chenyan Xiong, Chi Wang, Kuansan Wang, and Jiawei Han. 2020. Taxoexpan: Self-supervised taxonomy expansion with position-enhanced graph neural network. In Proceedings ofthe web conference 2020, pages 486–497.

Fabian M Suchanek, Mehwish Alam, Thomas Bonald, Lihu Chen, Pierre-Henri Paris, and Jules Soria. 2024. Yago 4.5: A large and clean knowledge base with a rich taxonomy. In Proceedings ofthe 47th international ACM SIGIR conference on research and development in information retrieval, pages 131–140.

Yushi Sun, Hao Xin, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, and Lei Chen. 2024. Are large language models a good replacement of taxonomies? arXiv preprint arXiv:2406.11131.

Kunihiro Takeoka, Kosuke Akimoto, and Masafumi Oyamada. 2021. Low-resource taxonomy enrichment with pretrained language models. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 2747–2758.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. Preprint, arXiv:2309.07597.

Fred Xu, Song Jiang, Zijie Huang, Xiao Luo, Shichang Zhang, Yuanzhou Chen, and Yizhou Sun. 2024. Fuse: Measure-theoretic compact fuzzy set representation for taxonomy expansion. In Findings of the association for computational linguistics: ACL 2024, pages 2707–2720.

Yue Yu, Yinghao Li, Jiaming Shen, Hao Feng, Jimeng Sun, and Chao Zhang. 2020. Steam: Self-supervised taxonomy expansion with mini-paths. In Proceedings ofthe 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 1026–1035.

Qingkai Zeng, Yuyang Bai, Zhaoxuan Tan, Shangbin Feng, Zhenwen Liang, Zhihan Zhang, and Meng Jiang. 2024. Chain-of-layer: Iteratively prompting large language models for taxonomy induction from limited examples. In Proceedings of the 33rd ACM International Conference on Information and Knowledge Management, pages 3093–3102.

Chao Zhang, Fangbo Tao, Xiusi Chen, Jiaming Shen, Meng Jiang, Brian Sadler, Michelle Vanni, and Jiawei Han. 2018. Taxogen: Constructing topical concept taxonomy by adaptive term embedding and clustering. Proc. KDDI.

Lin Zhang, Zhouhong Gu, Suhang Zheng, Tao Wang, Tianyu Li, Hongwei Feng, and Yanghua Xiao. 2025. Lite: Llm-impelled efficient taxonomy evaluation. arXiv preprint arXiv:2504.01369.

Yuhang Zhang, Jiwei Qin, and Chongren Feng. 2024. Peb-taxo: Projecting entities as boxes for taxonomy expansion. Neural Processing Letters, 56(2):102.

## A Input Prompts

## A.1 Generation Prompt

The following prompt is used to generate candidate children for each expanded taxonomy node. The placeholders enclosed in braces are dynamically replaced during execution.

You are expanding a taxonomy.   
Current node: {current\_node}   
Parent path: {parent\_path}   
Existing children of this node: {   
existing\_children}   
Task:   
Suggest exactly {K} NEW suitable children of   
the current node.   
Important:   
- The new children MUST be at the SAME level   
of specificity as

the existing children.   
Use the existing children as examples of   
the correct granularity.   
Each suggestion must be a child under "{   
current\_node}".   
Rules:   
- Do NOT repeat any existing child.   
- Do NOT suggest synonyms, rephrasings, or   
formatting variants.   
- Do NOT suggest broader or more general   
concepts than the existing   
children.   
Output format:   
Return exactly {K} names.   
- One name per line.   
- No numbering, explanations, or additional   
text.   
- Do not return the parent path.

## A.2 LLM-as-a-Judge Prompt

The following prompt is used by the LLM-based validator to assess each generated candidate independently. The placeholders are populated using the current taxonomy context.

You are validating a taxonomy expansion   
candidate.   
Parent path: {parent\_path}   
Parent class: {parent}   
Existing children of the parent: {   
existing\_children}   
Candidate child: {candidate}   
Task:   
Decide whether the candidate should be   
accepted as a reasonable   
NEW child category under the parent in this   
taxonomy.   
Guidelines:   
1. Accept the candidate if it is primarily a   
meaningful type/category of the parent.   
2. Reject the candidate if it is redundant   
with an existing child.   
3. Reject the candidate if it is only   
loosely related to the parent.   
Return ONLY valid JSON with exactly these   
fields:   
{   
"accept": true or false,   
"is\_type\_of\_parent": true or false,   
"is\_redundant": true or false,   
"confidence": number between 0 and 1,   
"reason": "short explanation"   
}

A candidate is accepted only if the judge marks it as acceptable, classifies it as a type of the parent, and does not identify it as redundant:

$$
\operatorname { A c C E P T } ( c ) = a ( c ) \wedge t ( c ) \wedge \neg r ( c ) ,
$$

where $a ( c ) , \ t ( c )$ , and $r ( c )$ correspond to the accept, is\_type\_of\_parent, and is\_redundant fields, respectively.

## B Human Evaluation Interface

Figure 2 illustrates the annotation interface used throughout the human evaluation. The upper panel summarizes the local taxonomy context by displaying the hierarchical path, existing children, generated accepted children, and, when applicable, the hidden gold children. The lower panel presents the evaluation form for each generated concept. Recovery-specific criteria are automatically enabled only when hidden gold children are available, while novel enrichment cases evaluate only the validity and quality of newly generated concepts.

## C Error Analysis

To better understand the remaining limitations of ReLTEx, we manually analyzed all parent–child relations that were accepted by our pipeline but rejected by the majority of human annotators. Rather than grouping errors according to the evaluation criteria, we categorized them based on their underlying semantic causes. Table 6 presents representative examples from each category.

One category of remaining errors corresponds to incorrect hierarchical placement. In these cases, the generated concept is semantically valid but attached to an inappropriate parent within the taxonomy. For example, BankTransfer was generated under PaymentCard, although it represents an alternative payment method rather than a subtype of payment card. Similarly, overfishing was proposed as a child of marine pollution, despite representing a distinct environmental issue rather than a form of pollution.

Another category corresponds to scope mismatch. Here, the generated concept belongs to the correct semantic domain but is expressed at an incompatible level of abstraction. For instance, concepts such as Climate change mitigation strategies and Climate change adaptation measures were generated as children of Climate change, mixing intervention strategies with environmental phenomena. Similar behavior was observed in Schema.org, where Gallery was generated under MediaGallery, although it is semantically more general than its parent.

We also observed cases of semantic redundancy, where the generated concept was effectively a near-synonym of an existing taxonomy concept. For example, TattooStudio was generated despite the taxonomy already containing the semantically equivalent concept TattooParlor, resulting in unnecessary duplication.

Finally, a small number of errors correspond to malformed concepts, including incomplete or corrupted generations such as Sho Emission Trading. These failures originate from generation artifacts rather than structural reasoning errors.

## D Ablation Study

To assess the impact of the main design choices in ReLTEx, we conduct ablation studies on three components of the framework: the prompting context, the validation strategy, and the validator acceptance thresholds. The experiments are performed on the SemEval Environment taxonomy using Mistral:7B as the generator model. As the goal is to isolate the contribution of each design choice rather than compare datasets, the ablation is conducted on this single representative benchmark.

Threshold calibration is performed through a sweep over a fixed set of generated candidates. Based on the resulting recovery performance, we select acceptance thresholds of 0.83 for both semantic validators and 0.90 for the classifier. These thresholds are used throughout all experiments reported in the paper.

We compare three prompting configurations:

• Local: provides only the path to the parent node together with its existing children.

• Parent Subtree: provides the complete subtree rooted at the parent.

• Full Taxonomy: provides the entire seed taxonomy as context.

For each prompting configuration, four validation strategies are evaluated:

• Semantic V1: validates candidates using cosine similarity between the parent and the generated child.

• Semantic V2: extends Semantic V1 by additionally considering the taxonomy path and similarity to the parent’s existing children.

![](images/186360cb4a33e0e5339e3ee231a3118ffb8baf071d1afd825dc758c3e79759c2.jpg)  
(a) Recovery case (SemEval Environment).

![](images/61b88f241c85d926c7b48f0128a135b29b0df2b82f3240d6d2b3425147f9ea59.jpg)  
(b) Novel enrichment case (Schema.org).  
Figure 2: Snapshot of the annotation interface used in the human evaluation.

Table 6: Representative semantic error categories identified during qualitative analysis.
<table><tr><td>Generated Concept</td><td>Existing Context</td><td>Category</td><td>Observation</td></tr><tr><td>BankTransfer</td><td>Parent: PaymentCard</td><td>Incorrect placement</td><td>Not a subtype of the parent</td></tr><tr><td>Gallery</td><td>Parent: MediaGallery</td><td>Scope mismatch</td><td>Superclass of the parent</td></tr><tr><td>TattooStudio</td><td>Sibling: TattooParlor</td><td>Semantic redundancy</td><td>Near-synonymous existing concept</td></tr><tr><td>Sho Emission Trading</td><td>一</td><td>Malformed concept</td><td>Incomplete generated entity</td></tr></table>

• LLM Judge: uses Llama3.2 to determine whether a generated candidate represents a valid, non-redundant child.

• Classifier: the proposed DistilRoBERTabased taxonomy validator.

Table 7 summarizes the benchmark results.

Across all validation strategies, the Local prompting configuration consistently achieves the highest R@K and SR@K. Restricting the context to the parent path and its existing children provides sufficient structural information while avoiding the additional noise introduced by larger contexts. Both the Parent Subtree and Full Taxonomy settings substantially reduce recovery performance, suggesting that exposing the language model to increasingly large portions of the taxonomy does not improve concept generation.

The comparison of validation strategies further supports the proposed classifier. Across all prompting contexts, it consistently achieves the highest R@K and SR@K. Both semantic validators produce slightly lower recovery scores by rejecting a small number of valid candidates, while Semantic V2 performs nearly identically to Semantic V1, indicating that the additional taxonomy context provides little benefit in this setting. The LLM Judge is the most conservative validator, substantially reducing recovery performance by rejecting many valid candidates together with incorrect ones.

Table 7: Results of the ablation study on the SemEval Environment taxonomy. Higher values indicate better performance.
<table><tr><td>Context</td><td>Validation Strategy</td><td>R@K</td><td>SR@K</td></tr><tr><td rowspan="4">Local</td><td>Classifier</td><td>34.62</td><td>42.31</td></tr><tr><td>Semantic V1</td><td>32.69</td><td>40.38</td></tr><tr><td>Semantic V2</td><td>32.69</td><td>40.38</td></tr><tr><td>LLM Judge</td><td>15.38</td><td>19.23</td></tr><tr><td rowspan="4">Parent Subtree</td><td>Classifier</td><td>17.31</td><td>23.08</td></tr><tr><td>Semantic V1</td><td>15.38</td><td>21.15</td></tr><tr><td>Semantic V2</td><td>15.38</td><td>21.15</td></tr><tr><td>LLM Judge</td><td>3.85</td><td>3.85</td></tr><tr><td rowspan="4">Full Taxonomy</td><td>Classifier</td><td>11.54</td><td>15.38</td></tr><tr><td>Semantic V1</td><td>9.62</td><td>13.46</td></tr><tr><td>Semantic V2</td><td>9.62</td><td>13.46</td></tr><tr><td>LLM Judge</td><td>9.62</td><td>9.62</td></tr></table>

Table 8: Statistics of the seed and enriched taxonomies produced by ReLTEx using Mistral:7B. |N| denotes the number of nodes, |E| the number of edges, and |D| the maximum depth.
<table><tr><td>Dataset</td><td>Taxonomy</td><td>|N|</td><td>|E|</td><td>|D|</td></tr><tr><td rowspan="2">SemEval Env</td><td>Seed</td><td>211</td><td>210</td><td>6</td></tr><tr><td>ReLTEx</td><td>4883</td><td>4882</td><td>7</td></tr><tr><td rowspan="2">Schema.org</td><td>Seed</td><td>912</td><td>911</td><td>6</td></tr><tr><td>ReLTEx</td><td>38603</td><td>38602</td><td>7</td></tr></table>

## E Expanded Taxonomy Statistics

To provide a quantitative overview of the resulting enriched taxonomies, table 8 compares the original seed taxonomies with the enriched taxonomies produced by ReLTEx. To remain consistent with the LITE evaluation, we report the taxonomies generated using Mistral:7B, which was selected as the generator after achieving the strongest overall performance in the masked taxonomy expansion benchmark.

The adaptive recursive expansion substantially increases the size of both taxonomies while introducing one additional hierarchy level. This indicates that ReLTEx primarily enriches existing branches rather than generating excessively deep hierarchies. The larger increase observed for Schema.org also reflects its broader semantic coverage, providing more opportunities for recursive concept generation than the smaller SemEval Environment taxonomy.