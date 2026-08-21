# ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM Agents

Yichu Fang<sup>1,2</sup>, Sitong Wei<sup>3</sup>, Haozhe Hu<sup>1,2</sup>, Xiaoyu Shen<sup>2</sup>\*

<sup>1</sup>Shanghai Jiao Tong University   
<sup>2</sup>Eastern Institute of Technology, Ningbo <sup>3</sup>Xi’an Jiaotong University   
fy\_chu@sjtu.edu.cn xyshen@eitech.edu.cn

## Abstract

Agentic language models repeatedly encode tool and skill schemas that recur across requests in different combinations and orders, preventing standard prefix caching from reusing their key–value (KV) states. We introduce ReCache, a framework for independently caching resource representations while reducing their inference-time computational and memory overhead. Resource-wise attention removes cross-resource interactions and assigns resource-local positions, producing composition-invariant KV blocks. ReCache then restricts resource visibility to contributionselected layer–KV-head-group routes and retains only invocation-critical fields through structural and semantic pruning. We evaluate ReCache on a benchmark assembled from seven public tool- and skill-use datasets, including resource-disjoint tests. Resource-wise attention matches dense invocation performance (82.3% versus 82.4% Inv-F1) while providing a 3.655× time-to-first-token speedup. The complete framework reduces allocated KVtensor memory by 92.43% and accelerates attention by 1.423×. These results show that separating reusable schema encoding from selective resource access substantially reduces agentic inference costs with limited effectiveness loss. The code is available at https: //github.com/EIT-NLP/ReCache.

## 1 Introduction

Recent advances have extended large language models (LLMs) into the core reasoning module of agentic systems, where they address user requests by dynamically invoking external tools and skills (Schick et al., 2023; Liu et al., 2024a; Li et al., 2026; Xu and Yan, 2026). As the set expands, agentic systems commonly adopt progressive disclosure and retrieve only task-relevant schemas during execution (Shi et al., 2025; Zheng et al., 2026). We refer to each tool or skill schema as a resource. Although retrieval limits the active context, the same resource may recur across requests in different combinations and orders, causing repeated prefill computation for unchanged information.

![](images/935e50d73694b3e9e98271b9f326070ad6746760516f47c29be4b41c1250402f.jpg)

(a) Causal attention among resources (relative-position bins). Orange boxes mark cross-resource blocks.  
![](images/a5052b3120b3e9c63709893c8a88d610628b3a7a49eba09957139737762ac3e9.jpg)  
(b) Attention from subsequent conversation tokens onto resources, normalized and aligned by relative position.  
Figure 1: Cross-resource attention on Qwen3-4B.

KV-cache reuse provides a natural way to amortize this repeated computation. However, standard prefix caching requires reused content to form an identical prefix, which rarely holds for dynamically composed resource contexts (Ye et al., 2025; Zhang et al., 2026). Recent approaches extend KV reuse to modular or position-independent contexts (Gim et al., 2024; Hu et al., 2024; Yao et al., 2025; Yang et al., 2025c), but often require positional correction or selective recomputation to approximate fullcontext KV states. Meanwhile, transformers can jointly encode semantic and positional dependencies (Vaswani et al., 2017; Shaw et al., 2018; Su et al., 2024), and the importance of positional information varies across tasks, with some studies showing that certain tasks can maintain performance under reduced or altered positional signals (Wang et al., 2021; Haviv et al., 2022; Kazemnejad et al., 2023). In resource invocation, preserving resourceinternal semantics may therefore matter more than preserving global resource order. Supporting this intuition, Figure 1a shows that Qwen3-4B exhibits substantially stronger attention within individual resources than across resources.

Based on this observation, we introduce Re-Cache, a resource-level KV reuse and compression framework for dynamically retrieved tools and skills. Its core mechanism, resource-wise attention, removes attention between distinct resource blocks and resets positional indices within each block. The resulting KV representation is independent of neighboring resources and their absolute positions, enabling resource caches to be constructed and reused independently. Although this representation differs from conventional full-context prefill, lightweight fine-tuning adapts the model to the resulting attention and positional patterns while preserving reliable resource invocation.

While resource-wise attention enables independent KV reuse, storing and transmitting complete resource caches can still be costly as resource length and scale grow. ReCache therefore compresses resource caches from both structural and semantic perspectives. Structurally, transformer layers and KV head groups contribute unevenly to model predictions (Michel et al., 2019; Voita et al., 2019; Dehghanighobadi and Fischer, 2026). ReCache ranks layers and head groups by their marginal contribution to prediction loss and retains resource visibility only along the most important routes. This contribution-based criterion exploits the distinct sparsity patterns across sequential layers and parallel KV head groups, providing a more reliable measure of resource utility than attentionmass selection. Semantically, Figure 1b shows that the final suffix token receives the highest attention mass from subsequent conversation tokens, suggesting that it naturally aggregates information from preceding fields (Muennighoff, 2022). This observation motivates retaining the suffix token as a compact semantic anchor, similar in spirit to summary-based compression methods that aggregate prompt or segment information into summary tokens (Mu et al., 2023; Chevalier et al., 2023; Zhang et al., 2024). However, unlike general text, tool and skill schemas must preserve exact interface information, including resource identifiers, argument names, and parameter constraints; invalid names and arguments are common sources of invocation failure (Song et al., 2023; Qin et al., 2024; Hao et al., 2023). ReCache therefore retains these critical fields alongside the suffix token for semantic aggregation.

To evaluate resource reuse and generalization systematically, we construct a unified tool- and skill-use benchmark from seven public datasets. We first apply a diversity-first sampling strategy to maximize resource diversity across samples, followed by filtering of invalid traces containing hallucinated resource usages. The resulting benchmark contains both in-distribution and resource-disjoint out-of-distribution splits, enabling evaluation on both known and unseen resources.

Experiments with Qwen3 backbones show that resource-wise attention differs from Dense, the standard dense-attention baseline, by at most 0.2% across effectiveness metrics while providing a 3.655× time-to-first-token (TTFT) speedup. With structural and semantic pruning, ReCache reduces allocated KV-tensor memory by 92.43% and accelerates attention by 1.423×. Under matched structural budgets, contribution-based selection preserves higher invocation effectiveness than attention-based and layer-asymmetric alternatives, particularly on unseen resources. Direct comparisons further show that field-aware semantic pruning retains better performance than generic textcompression baselines under the evaluated configurations. Across increasing resource lengths, ReCache maintains nearly constant TTFT, TPOT, and attention latency while capping allocated KVtensor memory at 0.03 GiB.

Our contributions are threefold: (1) ReCache provides a resource-level KV reuse and compression framework for dynamically retrieved tools and skills; (2) resource-wise attention, contributionguided structural pruning, and field-aware semantic pruning jointly reduce redundancy in cache construction, layer–head-group access, and token retention; and (3) a unified tool- and skill-use benchmark with resource-disjoint out-of-distribution evaluation supports systematic analysis of efficiency and invocation performance.

## 2 Related Work

Agentic Resources. Tool-augmented language models invoke external functions described by structured schemas. Recent benchmarks also evaluate function calling over large API pools and multiturn interactions (Li et al., 2023; Liu et al., 2024a; Qin et al., 2024; Liu et al., 2024b; Wang et al., 2024a). Agent systems also use natural-language skills that package reusable procedures with invocation metadata (Xu and Yan, 2026; Jiang et al., 2026; Li et al., 2026). As these resource pools grow, the corresponding retrieval selects a task-relevant subset before inference (Shi et al., 2025; Zheng et al., 2026). Retrieval reduces the number of resources in the prompt, but it does not eliminate repeated encoding of recurring resources.

Reusable Caching. Prefix caching reuses KV states for requests sharing an identical prefix (Kwon et al., 2023; Zheng et al., 2024; Yu et al., 2025). However, its applicability to dynamically composed resources remains limited, as reordered or recombined segments may differ from the positional configurations under which their KV states were constructed. Position-independent methods address this mismatch through various strategies. EPIC (Hu et al., 2024) and CacheBlend (Yao et al., 2025) preserve original positional encodings while selectively recomputing salient tokens to improve cache integration. In particular, CacheBlend further corrects positional information across cached KV states to mitigate position mismatch during reuse. KVLink (Yang et al., 2025c) instead decouples positional information from cached content and compensates for positional shifts during reuse. KVCOMM (Ye et al., 2025), on the other hand, uses the entropy of embedding-based anchor weights to assess cache shareability. Nevertheless, the necessity of explicit positional information can vary across tasks. NoPo (Haviv et al., 2022) and NoPE (Kazemnejad et al., 2023) demonstrate that Transformers can recover positional knowledge without explicit positional encodings in various downstream tasks, while Wang et al. (Wang et al., 2021) further show that positional information exhibits task-dependent importance across classification and span prediction.

KV-Cache Compression. KV-cache compression exploits structural redundancy across tokens, layers, and attention heads. H2O evicts lowimportance states under a uniform layer budget, whereas DepthKV assigns layer-dependent retention budgets (Zhang et al., 2023; Dehghanighobadi and Fischer, 2026). DuoAttention classifies heads by their attention roles, and SPEED restricts prompt-KV visibility by depth while preserving full-depth decoding states (Xiao et al., 2025; Oh et al., 2026). These studies establish the importance of multidimensional sparsity, but their selection criteria are not optimized for resource invocation. In addition, semantic compression reduces the number of retained context tokens. Learned approaches encode spans into compact summary states (Mu et al., 2023; Chevalier et al., 2023; Zhang et al., 2024), while LLMLingua and LongLLMLingua apply query-aware token filtering (Jiang et al., 2023, 2024). ChunkKV further retains contiguous segments to preserve local coherence (Liu et al., 2026). These techniques effectively compress natural-language contexts while retaining salient information, but resources contain structured identifiers and field relationships that directly determine invocation behavior, motivating compression strategies tailored to field-level semantics.

## 3 ReCache

Problem Setup Given a dataset D, each instance $( X , Y ) \in { \mathcal { D } }$ consists of an agentic input sequence X and the target resource invocations Y required to satisfy a user request, including tool and skill invocations. The input sequence comprises the system instruction, user query, retrieved resources $\mathcal { R } = \{ R _ { 1 } , R _ { 2 } , . . . , R _ { N } \}$ , and preceding conversational history when available in multi-turn interactions. Each resource $R _ { i } = \left( t _ { i , 1 } , \ldots , t _ { i , D _ { i } } \right)$ is represented as a token sequence of length $D _ { i }$ describing its functionality, including resource identifiers, argument definitions, and associated metadata.

The objective is to predict the required invocations Y conditioned on X. As illustrated in Figure 2, ReCache improves resource-processing efficiency through three progressive stages. Resourcewise attention removes unnecessary resourceresource interactions to enable independent KV representations; structural pruning restricts resource visibility to selected layer–KV-head-group routes; and semantic pruning reduces the visible resource length by retaining only critical fields.

Resource-Wise Attention Retrieval-Augmented Generation (RAG) demonstrates that disjoint retrieved passages can be encoded independently without mutual cross-attention (Izacard and Grave,

![](images/3cfffeee2ddc8d1f6efde29c3aa4ffb5d6ad7cd872cf1818d802cb76013f2788.jpg)  
Figure 2: ReCache constructs reusable KV blocks for each $R _ { i }$ and progressively reduces their subsequent cache retention and access. (a) Resource-wise attention removes inter-resource attention and resets positions from zero within each R<sub>i</sub>. (b) Structural pruning exposes resource KV states (using $M _ { \mathrm { r e s o u r c e } } )$ only through selected layer (L )–KV-head-group (G ) routes $\Omega ^ { \star }$ (highlighted in blue). The remaining routes use $M _ { \mathrm { { c o n t e x t } } }$ . (c) Semantic pruning retains resource names, argument (arg) names and descriptions (desc.), and the final suffix token. The final mask combines these decisions, ensuring that Y attends exclusively to retained semantic fields within assigned routes during decoding.

2021; Lin et al., 2025). Extending this principle, agentic resources are also retrieved (Shi et al., 2025) and primarily rely on internal semantics, allowing their KV caches to be constructed and reused independently of retrieval combinations or positional orderings.

Defining $D _ { \mathcal { R } } ~ = ~ \sum _ { i } D _ { i }$ as the total resource length, resource-wise attention removes crossresource dependencies during cache construction. As illustrated in Figure 2(a), tokens within each resource attend only to the shared prefix and their local resource tokens, while attention links between different resources are eliminated. The resource– resource attention is thus reduced from $O ( D _ { \mathcal { R } } ^ { 2 } )$ to $O ( \sum _ { i } D _ { i } ^ { 2 } )$ , allowing each $R _ { i }$ to be independently represented and cached.

To achieve full reusability, resource KV representations must also remain positionally invariant. Following prior studies on task-dependent positional importance (Wang et al., 2021; Haviv et al., 2022; Kazemnejad et al., 2023), we hypothesize that absolute placement among retrieved resources provides limited benefit for resource invocation. Therefore, we adopt a resource-local positional layout, where each resource follows an identical relative position scheme that preserves internal token continuity while removing dependence on global resource ordering. Specifically, during cache construction, ReCache assigns resource-local positional indices $p o s ( t _ { i , j } ) = j$ while preserving the original positional encodings of the surrounding context. Consequently, each resource receives identical positional embeddings regardless of its placement among retrieved items. While frameworks like EPIC (Hu et al., 2024) incorporate position resetting as part of a broader compilation pipeline, empirical ablations confirm that this re-indexing is sufficient to preserve task effectiveness.

Structural Pruning Transformer layers and attention heads exhibit substantial redundancy (Michel et al., 2019; Voita et al., 2019; Men et al., 2024). ReCache exploits this structural sparsity by restricting resource–context visibility to influential layer–KV-head-group routes. Following objective-driven selection methods (Xiao et al., 2025; Dehghanighobadi and Fischer, 2026), route importance is measured by its marginal reduction in invocation loss.

Formally, let $[ L ] ~ = ~ \{ 1 , \dots , L \}$ and $\begin{array} { r l } { [ G ] } & { { } = } \end{array}$ $\{ 1 , \ldots , G \}$ denote the sets of transformer layers and KV head groups under grouped-query attention (GQA), respectively. A structural routing configuration is defined by a subset $\Omega \subseteq [ L ] \times [ G ]$ where each $( l , g ) \in \Omega$ represents a layer–headgroup route with access to resource KV states constructed by resource-wise attention. As illustrated in Figure 2(b), routes in Ω use $M _ { \mathrm { r e s o u r c e } }$ to attend to the retained resource tokens and the conversational context. Routes outside Ω use $M _ { \mathrm { { c o n t e x t } } }$ , which structurally prunes all resource tokens from their attention scope while preserving standard causal attention over non-resource tokens.

The key challenge is to identify a compact Ω without degrading task performance. Contrary to DepthKV (Dehghanighobadi and Fischer, 2026), which evaluates layer importance by pruning one layer at a time from a dense configuration, ReCache applies leave-one-in analysis to both layers and KV head groups<sup>1</sup>. Starting from $\Omega = \emptyset$ , where conversation tokens cannot access resource states, it measures the reduction in invocation loss obtained by activating one layer or head group.

We denote the held-out selection set by $\mathcal { D } _ { \mathrm { h } }$ and the span of the target response $Y$ by $\mathcal { T } ( Y )$ . For each $( X , Y ) \in { \mathcal { D } } _ { \mathrm { h } } .$ the invocation loss under configuration Ω is

$$
\ell _ { \Omega } ( X , Y ) = - \frac { 1 } { | { \cal T } ( Y ) | } \sum _ { t \in { \cal T } ( Y ) } \log p _ { \theta } ^ { \Omega } ( y _ { t } \mid y _ { < t } , X ) ,
$$

where $p _ { \theta } ^ { \Omega }$ denotes the model distribution induced by Ω. Restricting the objective to invocationrelated positions aligns route selection with resource identification and argument generation, while preventing unrelated response tokens from dominating the score. The selection objective $\mathcal { I } ( \Omega )$ is then the average of $\ell _ { \Omega } ( X , Y )$ over $\mathcal { D } _ { \mathrm { h } }$

For layer l, we activate all associated KV head groups using $\Omega _ { l } = \{ l \} \times [ G ]$ and define the layer contribution as

$$
s _ { l } = \mathcal { I } ( \emptyset ) - \mathcal { I } ( \Omega _ { l } ) .
$$

A positive $s _ { l }$ indicates that resource visibility at layer l improves prediction performance relative to $\Omega = \emptyset$ . Head-group contributions $s _ { g }$ are computed analogously using $\Omega _ { g } = [ L ] \times \{ g \}$ . For each structural dimension, positive contributions are normalized into contribution weights:

$$
w _ { i } = \frac { \operatorname* { m a x } ( s _ { i } , 0 ) } { \sum _ { j } \operatorname* { m a x } ( s _ { j } , 0 ) }
$$

where i denotes a layer or KV head group. A larger $w _ { i }$ indicates a greater isolated contribution to resource-dependent prediction, and guides the selection of candidate structural budgets.

Specifically, given layer and head-group budgets $K _ { L }$ and $K _ { G }$ , ReCache retains the highest-weight units according to w<sub>i</sub>:

$$
\begin{array} { r } { \mathcal { L } ^ { \star } = \mathrm { T o p K } _ { K _ { L } } ( \{ w _ { l } \} ) , } \\ { \mathcal { G } ^ { \star } = \mathrm { T o p K } _ { K _ { G } } ( \{ w _ { g } \} ) , } \end{array}
$$

and constructs the final routing configuration as

$$
\Omega ^ { \star } = { \mathcal { L } } ^ { \star } \times { \mathcal { G } } ^ { \star } .
$$

During inference, only routes in $\Omega ^ { \star }$ expose resource KV states to conversational texts to reduce unnecessary computation.

Semantic Pruning While resource-wise attention enables resource-level cache reuse and structural pruning reduces unnecessary resource visibility, each activated route still processes the full $D _ { i }$ tokens within independent KV blocks. These tokens are inherently structural (Qin et al., 2024; Bai et al., 2026) and span heterogeneous fields, including executable identifiers, parameter constraints, descriptions, and formatting elements, with varying contributions to invocation performance, motivating field-aware compression. Since resource invocation failures are predominantly associated with invalid names and incorrect arguments (Song et al., 2023; Qin et al., 2024; Hao et al., 2023), Re-Cache performs semantic pruning along the token dimension to preserve invocation-critical components while removing redundant content.

For each $R _ { i }$ , ReCache retains tokens from three fields: the resource name, argument names, and argument descriptions. These fields support resource identification, parameter-interface alignment, and argument-value constraints, respectively. Ablation results show that preserving these three fields incurs only minor performance degradation<sup>2</sup>.

Additionally, the final token is preserved as a suffix representation, whose hidden state aggregates preceding resource information. This design is related to summary-based compression approaches (Mu et al., 2023; Zhang et al., 2024), but instead of introducing new summary tokens, it reuses the final token of $R _ { i }$ as a naturally occurring boundary representation whose causal state can condition on all preceding fields (Muennighoff, 2022; Wang et al., 2024b; Zhuang et al., 2024)<sup>3</sup>. The resulting subsequence ${ \widetilde { R } } _ { i }$ has an effective length of $\widetilde { D } _ { i } \le D _ { i }$

The resulting compact KV representations remain independently reusable, while conversational texts on routes in $\Omega ^ { \star }$ attend only to the retained tokens. As illustrated by the Final ReCache Mask in Figure 2, this selective visibility further reduces resource–context attention overhead.

## 4 Experimental Setup

## 4.1 Datasets

We evaluate ReCache on a unified resource-use dataset covering both tools and skills. Since existing datasets typically focus on particular resource types or interaction patterns, we combine seven sources to broaden resource and scenario coverage, including ToolACE (Liu et al., 2024a), API-GEN (Liu et al., 2024b), ToolMind (Yang et al., 2025b), ToolRet (Shi et al., 2025), Toucan (Xu et al., 2025), SkillRouter (Zheng et al., 2026), and WildToolBench (Yu et al., 2026).

Direct aggregation, however, introduces several confounding factors. Synthetic traces may contain malformed invocations or use resources absent from the provided set. More subtly, resource knowledge can be encoded into model parameters through learned embeddings or special tokens (Hao et al., 2023; Wang et al., 2025). ToolOmni further observes that parameter memorization generalizes poorly to unseen and evolving resources (Huang et al., 2026). Frequently repeated resource names or exact candidate subsets may consequently allow the model to recall familiar interfaces rather than interpret the schema supplied at inference time. We therefore retain only parseable and grounded resource calls and adopt diversity-first sampling, which prioritizes newly observed called resources and candidate-set configurations before satisfying source-level quotas.

The final training dataset D contains 49,424 examples, while a subset $\mathcal { D } _ { \mathrm { s u b s e t } }$ with 5,000 examples is used for preliminary experiments. Evaluation is conducted on two test sets with 1,000 examples each: an in-distribution split $\mathcal { T } _ { \mathrm { I N D } }$ , where the involved resources are observed during training, and an out-of-distribution split $\tau _ { \mathrm { O O D } }$ , where the resources are unseen in D.

## 4.2 Evaluation Metrics

Effectiveness We evaluate effectiveness using the turn-averaged resource-invocation F1 (Inv-F1), where each turn may involve multiple parallel resource invocations for one instance. This metric measures whether the model can generate the complete set of required invocations for a given query. A predicted invocation is considered correct only if both the resource name and all associated arguments exactly match the corresponding gold invocation. Beyond end-to-end performance, we further analyze model behavior from two complementary perspectives. Resource identification (Resource ID) is evaluated using precision (ID-P), recall (ID-R), and turn-averaged F1 (ID-F1) over resource names, reflecting the ability to identify relevant resources. We additionally report the resource hallucination rate (Halluc.), defined as the proportion of predicted resource names that do not exist in the available resource set, which captures failures caused by generating invalid or unavailable resources. All metrics are reported in percentages (%).

Efficiency. We use time-to-first-token (TTFT) to quantify online prefill latency under a cache hit, including request-local KV materialization but excluding offline cache construction. We further report time-per-output-token (TPOT) to measure end-to-end decoding latency, and attention latency (Attn.) to expose the decoding gains attributable specifically to reduced attention computation, which can be obscured in TPOT by unchanged projection and feed-forward costs. Finally, Mem. measures the resource reduction in memory allocated to KV-cache tensors.

## 4.3 Backbone Models

We perform full fine-tuning using Qwen3 variants (Yang et al., 2025a), with Qwen3-4B $( Q _ { l } )$ as the primary backbone. Qwen3-1.7B $( Q _ { s } )$ serves as an auxiliary backbone for analyzing the sensitivity of structural budgets to model scale. We train the primary Qwen3-4B model on an NVIDIA A800 GPU and run the preliminary Qwen3-1.7B experiments on an NVIDIA RTX 5000 GPU.

![](images/6c85e08955e287ed57bf58067902046293dd3016143d38ecb373efa2531a517e.jpg)

![](images/f0548d07b538b073ca94008fbd8b9b0b075cb6910b920f36891841bf197d1c40.jpg)  
Figure 3: Structural contribution coverage in $Q _ { l }$ and $Q _ { s }$ Cumulative leave-one-in contributions of layers (top) and KV head groups (bottom); vertical lines mark the selected budgets.

## 5 Structural Budget Analysis

Rather than treating structural budgets as unexplained hyperparameters, we derive them from how resource information is used across the network. Motivated by the non-uniform structural sensitivity observed in layer and KV-cache pruning (Men et al., 2024; Dehghanighobadi and Fischer, 2026), we use the leave-one-in scores from Section 3 to profile layers and KV head groups, then validate the resulting budgets through end-to-end ablations, compared with the full-visibility configuration $\Omega _ { \mathrm { f u l l } }$ after applying resource-wise attention. For $\Omega = \mathcal { L } \times \mathcal { G }$ , structural sparsity is

$$
\rho ( \Omega ) = 1 - \frac { | \mathcal { L } | | \mathcal { G } | } { L G } ,
$$

which measures the fraction of routes without resource visibility. We also introduce contribution coverage (CC) to quantify the preserved structural capacity under a routing budget. CC is computed as the cumulative contribution weight $w _ { i }$ of the retained top-k layers or KV head groups.

Layer contributions. Figure 3 shows that layer contributions are highly concentrated: both cumulative curves rise rapidly and saturate near $K _ { L } =$ 20, where the retained layers capture 97.7% and 99.8% CC for $Q _ { l }$ and $Q _ { s }$ , respectively. This nonuniformity agrees with prior layer-pruning analyses (Fan et al., 2020; Men et al., 2024) and indicates that exposing resource states in every layer is unnecessary. We therefore use $K _ { L } = 2 0$ as the shared layer budget and jointly validate it with the selected head groups against $\Omega _ { \mathrm { f u l l } }$

Head-group contributions. Head-group profiles differ by model scale. In $Q _ { l }$ , contributions are relatively flat, yet three groups retain near-full invocation performance despite covering only 47.3% CC. This discrepancy suggests that resource-related functions are not concentrated in a single head group but distributed across multiple groups, allowing certain groups to be removed with limited impact (Michel et al., 2019; Voita et al., 2019). In contrast, $Q _ { s }$ requires seven groups, which cover 99.3% CC and recover Inv-F1 to within 0.7 points of $\Omega _ { \mathrm { f u l l } }$ The smaller model therefore distributes this computation across more parallel groups, whereas the larger model offers greater substitutability, echoing the functional specialization of KV heads observed in long-context retrieval (Wu et al., 2024; Xiao et al., 2025). Table 1 summarizes the resulting budgets, structural sparsity, and invocation performance relative to $\Omega _ { \mathrm { f u l l } }$

<table><tr><td>Model</td><td> $K _ { L } / L$ </td><td> $K _ { G } / G$ </td><td> $\mathrm { C C } _ { L }$ </td><td> $\operatorname { C C } G$ </td><td> $\rho$  ∆Inv-F1 (%)</td></tr><tr><td> $Q _ { l }$ </td><td>20/36</td><td>3/8</td><td>97.7</td><td>47.379.2</td><td>-0.2</td></tr><tr><td> $Q _ { s }$ </td><td>20/28</td><td>7/8</td><td>99.8</td><td>99.3 37.5</td><td>-0.7</td></tr></table>

Table 1: Selected structural budgets. CC and $\rho$ are percentages; $\Delta$ Inv-F1 is measured relative to $\Omega _ { \mathrm { f u l l } }$ after full-data training.

Budget choice. We select the sparsest configuration that maintains performance comparable to $\Omega _ { \mathrm { f u l l } }$ , obtaining $\Omega ^ { \star } = \Omega _ { 2 0 , 3 }$ for $Q _ { l }$ and $\Omega ^ { \star } = \Omega _ { 2 0 , 7 }$ for $Q _ { s }$ . Both configurations preserve nearly all cumulative layer contributions while assigning different head-group budgets according to modelspecific contribution distributions<sup>5</sup>.

## 6 Results

All experiments in this section use $Q _ { l }$ trained on the full training set D and are evaluated on $\mathcal { T } _ { \mathrm { I N D } }$ unless otherwise specified.

Cache Reuse Table 2 reports both the effectiveness and efficiency of resource-wise attention. We compare against Dense, and evaluate two incremental configurations: Block eliminates cross-resource attention while retaining the original positional indices, and $\Omega _ { \mathrm { f u l l } }$ further reindexes positions to construct composition-independent resource blocks. Both configurations remain comparable to Dense, with all effectiveness metrics differing by at most 0.2%. Resource-wise attention also enables resource representations to be pre-encoded with a shared prefix and reused across requests, reducing online computation and allowing $\Omega _ { \mathrm { f u l l } }$ to achieve a 3.655× TTFT speedup over Dense. Consistent with findings from other tasks (Wang et al., 2021; Haviv et al., 2022; Kazemnejad et al., 2023), these results suggest that cross-resource interactions and global resource ordering provide limited benefits in this setting, while resource-local position reindexing preserves prediction quality and enables efficient resource-level reuse.

<table><tr><td colspan="4">Method Inv-F1 ID-F1 Halluc. TTFT (ms)</td></tr><tr><td>Dense</td><td>82.4</td><td>96.0 0.0</td><td>26.319</td></tr><tr><td>Block</td><td>82.2</td><td>95.8 0.1</td><td></td></tr><tr><td> $\Omega _ { \mathrm { f u l l } }$ </td><td>82.3</td><td>96.0 0.2</td><td>7.200</td></tr></table>

Table 2: Resource-wise attention ablation.

Model Comparison Table 3 compares structural and semantic pruning strategies built on $\Omega _ { \mathrm { f u l l } }$ on $\mathcal { T } _ { \mathrm { I N D } }$ , reporting both efficiency and invocation effectiveness. To assess robustness to distribution shift, Table 4 reports effectiveness on $\tau _ { \mathrm { O O D } }$

Under matched structural budgets, contributionbased selection consistently surpasses the corresponding baselines. At the 20-layer budget, $\Omega _ { 2 0 , G }$ and $\Omega _ { \mathrm { f u l l } } .$ +SPEED (Oh et al., 2026) exhibit nearly identical efficiency, whereas $\Omega _ { 2 0 , G }$ improves Inv-F1 by 3.2% on $\mathcal { T } _ { \mathrm { I N D } }$ and $1 0 . 0 \%$ on $\tau _ { \mathrm { { O O D } } }$ . For joint layer–head-group pruning, $\mathrm { S A _ { 2 0 , 3 } }$ ranks routes by their attention mass over each resource and retains the top 20 layers and 3 KV head groups. Despite comparable efficiency, $\Omega _ { 2 0 , : }$ <sub>3</sub> improves Inv-

<table><tr><td>Method</td><td>Attn. ×</td><td>Mem. ↓</td><td>Inv-F1</td><td>ID-F1</td><td>Halluc.</td></tr><tr><td>Dense</td><td></td><td></td><td>82.4</td><td>96.0</td><td>0.0</td></tr><tr><td>Ωfull</td><td>1.001×</td><td>0.47%</td><td>82.3</td><td>96.0</td><td>0.2</td></tr><tr><td colspan="6">Structural pruning</td></tr><tr><td> $\Omega _ { 2 0 , G }$ </td><td>1.016×</td><td>44.71%</td><td>82.5</td><td>95.7</td><td>0.3</td></tr><tr><td>Ωfull + SPEED</td><td>1.013×</td><td>44.71%</td><td>79.3</td><td>92.9</td><td>2.8</td></tr><tr><td>Ω20,3</td><td>1.314×</td><td>79.27%</td><td>82.1</td><td>95.6</td><td>0.2</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + \mathrm { S A } _ { 2 0 , 3 }$ </td><td>1.302×</td><td>79.27%</td><td>79.1</td><td>93.7</td><td>1.0</td></tr><tr><td colspan="6">Semantic pruning</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + \mathrm { G i s t }$ </td><td>1.029×</td><td>99.22%</td><td>39.2</td><td>46.9</td><td>51.4</td></tr><tr><td>Ωfull + Beacon</td><td>1.020×</td><td>75.42%</td><td>78.6</td><td>92.5</td><td>1.4</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + { \bf S } { \bf M } { \bf P }$ </td><td>1.021×</td><td>63.57%</td><td>81.6</td><td>95.2</td><td>0.2</td></tr><tr><td>ReCache</td><td>1.423×</td><td>92.43%</td><td>80.3</td><td>94.9</td><td>0.2</td></tr></table>

Table 3: Results on $\mathcal { T } _ { \mathrm { I N D } }$ . Efficiency is reported relative to Dense. SMP denotes Semantic Pruning.

<table><tr><td>Method</td><td>Inv-F1</td><td>ID-F1</td><td>Halluc.</td></tr><tr><td>Dense</td><td>66.3</td><td>95.6</td><td>0.0</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } }$ </td><td>64.7</td><td>94.4</td><td>0.4</td></tr><tr><td colspan="4">Structural pruning</td></tr><tr><td> $\Omega _ { 2 0 , G }$ </td><td>64.3</td><td>94.3</td><td>0.5</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } \cdot t$  -SPEED</td><td>54.3</td><td>81.6</td><td>13.4</td></tr><tr><td> $\Omega _ { 2 0 , 3 }$ </td><td>63.2</td><td>93.3</td><td>0.5</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + \mathrm { S A } _ { 2 0 , 3 }$ </td><td>58.2</td><td>88.4</td><td>5.1</td></tr><tr><td colspan="4">Semantic pruning</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + \mathrm { G i s t }$ </td><td>9.7</td><td>17.8</td><td>79.9</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + \mathrm { B e a c o n }$ </td><td>58.5</td><td>89.7</td><td>4.2</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } } + { \bf S } { \bf M } { \bf P }$ </td><td>62.8</td><td>94.1</td><td>0.3</td></tr><tr><td>ReCache</td><td>60.8</td><td>92.8</td><td>0.6</td></tr></table>

Table 4: $\tau _ { \mathrm { O O D } }$ Results. SMP denotes Semantic Pruning.

F1 over $\mathrm { S A _ { 2 0 , 3 } }$ by 3.0% and 5.0% points on the two test sets, respectively. The OOD hallucination rates further decrease from 13.4% to 0.5% relative to SPEED and from 5.1% to 0.5% relative to $\mathrm { S A _ { 2 0 , 3 } }$ , demonstrating the stronger generalization of contribution-based routing.

For semantic pruning<sup>6</sup>, Gist (Mu et al., 2023) attains the largest standalone KV reduction but severely degrades invocation effectiveness, indicating that a single summary vector cannot preserve the fine-grained information required for resource invocation. Beacon (Zhang et al., 2024) retains more information through chunk-level summaries, yet remains less effective than field-aware Semantic Pruning and exhibits higher hallucination, particularly on $\tau _ { \mathrm { { O O D } } }$ This gap indicates that partial coverage of critical fields is insufficient for robust generalization to unseen resources. Detailed field ablations are provided in Appendix C.

Combining all strategies, ReCache achieves the highest attention speedup (1.423×) and a comparable KV reduction (92.43%) among configurations that retain competitive effectiveness. It retains Inv-F1 scores of 80.3% on $\mathcal { T } _ { \mathrm { I N D } }$ and 60.8% on $\tau _ { \mathrm { O O D } } .$ corresponding to 97.5% and 91.8% of performance of Dense, respectively.

Efficiency Scaling We further evaluate efficiency across four ranges of total resource length $D _ { \mathcal { R } }$ . As shown in Figure 4, Dense TTFT increases from tens of milliseconds to over 5,000 ms as resource contexts expand, whereas $\Omega _ { \mathrm { f u l l } }$ maintains TTFT below 50 ms through resource-level cache reuse. ReCache further reduces this prefill overhead, exhibiting an almost flat profile slightly above 5 ms across all ranges.

![](images/f49d2f6ca6ad4b82eaf6aeaa4024ed15514a7527c5e747298481baa62f632d66.jpg)  
(a) TTFT

![](images/4dccf4945cca4541d6f48629e2002d390ead07cf977475ca45079535c5c7c267.jpg)  
(b) TPOT

![](images/7e82a9c73a55f6339770763b290f4f8355fb0a97e27cb020d4964d122332e40f.jpg)  
(c) Attn.

![](images/6c3f648575b785fc9b0386665bc7576e94feca3013adeee5ca0dc5634ca44e67.jpg)  
(d) Mem.  
Figure 4: Efficiency on varying resource lengths $D _ { \mathcal { R } }$ . Requests are grouped by average resource length into Small $( 0 \leq D _ { \mathcal { R } } <$ 1K), Medium $( 1 \mathrm { K } \dot { \leq } D _ { \mathcal { R } } \dot { < } 5 \bar { \mathrm { K } } )$ , Large $( 5 \bar { \mathrm { K } } \leq D _ { \mathcal { R } } < 1 \bar { 0 } \mathrm { K } )$ , and XL $( \hat { D } _ { \mathcal { R } } \overset { \cdot } { \geq } 1 0 \mathrm { K } )$ . SMP denotes semantic pruning.

Structural and semantic pruning further improve decoding efficiency. Dense and $\Omega _ { \mathrm { f u l l } }$ reach approximately 11 ms TPOT and 5 ms attention latency, whereas ReCache maintains TPOT slightly above 6 ms and attention latency below 0.2 ms with consistently stable decoding latency across resource scales. At XL, semantic pruning $( \Omega _ { \mathrm { f u l l } } { + } S \mathbf { M P } )$ reduces attention latency below 0.5 ms compared with over 2 ms under structural pruning $( \Omega _ { 2 0 , 3 } )$ alone, as longer resource contexts amplify the benefit of reducing tokens from semantic fields. Combining both pruning strategies therefore achieves the lowest and most stable latency.

ReCache also caps allocated Mem. at 0.03 GiB, whereas Dense and $\Omega _ { \mathrm { f u l l } }$ approach 8 GiB as resource contexts grow. These bounded prefill, decoding, and memory costs demonstrate the suitability of ReCache for agentic workloads with increasingly large resource contexts.

## 7 Conclusion

We presented ReCache, a framework for reusing and compressing tool and skill schemas in agentic language models. Resource-wise attention encodes each complete schema as a resource-compositioninvariant KV block, eliminating repeated encoding across retrieval combinations. After construction, field-aware semantic pruning retains invocationcritical states, while contribution-guided structural pruning exposes them only to influential layer–KVhead-group routes. This separation preserves information during encoding while reducing the states stored and processed during inference.

Across seven tool- and skill-use datasets, independently reusable blocks match dense invocation quality and substantially reduced prefill latency. Combining structural and semantic pruning reduces allocated KV-tensor memory by 92.43% and accelerates attention by 1.423×, while retaining 97.5% of dense Inv-F1 in-distribution and 91.8% on unseen resources. The structural analysis further showed that effective routing budgets depend on model-scale redundancy, but can be selected with the same contribution-based criterion. ReCache therefore offers a practical way to scale dynamic resource contexts without repeatedly encoding or densely exposing every schema.

## Limitations

Our evaluation focuses on two Qwen3 backbones, with end-to-end experiments primarily conducted on Qwen3-4B, as full-parameter fine-tuning of larger models was beyond the computational budget of our current infrastructure. While the optimal structural budget varies across model scales, extending the analysis to larger models and other architectures remains an important direction. Since ReCache modifies attention masks and positional layouts, the current study focuses on trainable settings, while adapting it to frozen models is left for future work. In addition, ReCache is designed for environments with relatively stable system instructions and resource schemas, where resource representations can be reused across different user queries and retrieval combinations. Extending reuse to highly dynamic resource environments, where retrieval order or cross-resource dependencies carry essential information, provides another avenue for future exploration.

## References

Xiaofan Bai, Hongqiang Lin, Chao Liu, Yantao Zhang, Xuan Jin, Xipeng Cao, and Yuhong Li. 2026. Skillzip: Evaluation-free skill compression for selfevolving agents by discovering reusable structure. arXiv preprint arXiv:2608.11079.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846.

Zahra Dehghanighobadi and Asja Fischer. 2026. DepthKV: Layer-dependent kv cache pruning for long-context llm inference. arXiv preprint arXiv:2604.24647.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. 2024. LongRoPE: Extending LLM context window beyond 2 million tokens. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 11091–11104. PMLR.

Angela Fan, Edouard Grave, and Armand Joulin. 2020. Reducing transformer depth on demand with structured dropout. In International Conference on Learning Representations.

In Gim, Guojun Chen, Seung-seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. 2024. Prompt cache: Modular attention reuse for low-latency inference. Proceedings of Machine Learning and Systems, 6:325–338.

Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. 2023. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. Advances in neural information processing systems, 36:45870–45894.

Adi Haviv, Ori Ram, Ofir Press, Peter Izsak, and Omer Levy. 2022. Transformer language models without positional encodings still learn positional information. In Findings of the Association for Computational Linguistics: EMNLP 2022, pages 1382–1390.

Junhao Hu, Wenrui Huang, Weidong Wang, Haoyi Wang, Tiancheng Hu, Qin Zhang, Hao Feng, Xusheng Chen, Yizhou Shan, and Tao Xie. 2024. Epic: Efficient position-independent caching for serving large language models. arXiv preprint arXiv:2410.15332.

Shouzheng Huang, Meishan Zhang, Baotian Hu, and Min Zhang. 2026. Toolomni: Enabling open-world tool use via agentic learning with proactive retrieval and grounded execution. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 37421– 37439.

Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th conference of the european chapter of the association for computational linguistics: main volume, pages 874–880.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 13358–13376.

Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2024. Longllmlingua: Accelerating and enhancing llms in long context scenarios via prompt compression. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1658–1677.

Yanna Jiang, Delong Li, Haiyu Deng, Baihe Ma, Xu Wang, Qin Wang, and Guangsheng Yu. 2026. SoK: Agentic skills – beyond tool use in LLM agents. arXiv preprint arXiv:2602.20867.

Amirhossein Kazemnejad, Inkit Padhi, Karthikeyan Natesan Ramamurthy, Payel Das, and Siva Reddy. 2023. The impact of positional encoding on length generalization in transformers. arXiv preprint arXiv:2305.19466.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pages 611–626.

Minghao Li, Feifan Song, Bowen Yu, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. 2023. API-bank: A comprehensive benchmark for toolaugmented LLMs. arXiv preprint arXiv:2304.08244.

Xiangyi Li, Wenbo Chen, Yimin Liu, Shenghan Zheng, Xiaokun Chen, Yifeng He, Yubo Li, Bingran You, Haotian Shen, Jiankai Sun, Shuyi Wang, Qunhong Zeng, Di Wang, Xuandong Zhao, Yuanli Wang, Roey Ben Chaim, Zonglin Di, Yipeng Gao, Junwei He, Yizhuo He, Liqiang Jing, Luyang Kong, Xin Lan, Jiachen Li, Songlin Li, Yijiang Li, Yueqian Lin, Xinyi Liu, Xuanqing Liu, Haoran Lyu, Ze Ma, Bowei Wang, Runhui Wang, Tianfu Wang, Wengao Ye, Yue Zhang, Hanwen Xing, Yiqi Xue, Steven Dillmann, and Han-Chung Lee. 2026. Skillsbench: Benchmarking how well agent skills work across diverse tasks. CoRR, abs/2602.12670.

Xiaoqiang Lin, Aritra Ghosh, Bryan Kian Hsiang Low, Anshumali Shrivastava, and Vijai Mohan. 2025. Refrag: Rethinking rag based decoding. arXiv preprint arXiv:2509.01092.

Weiwen Liu, Xu Huang, Xingshan Zeng, Xinlong Hao, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Zhengying Liu, Yuanqing Yu, Zezhong Wang, Yuxian Wang, Ning Wu, Yutai Hou, Bin Wang, Chuhan Wu, Xinzhi Wang, Yong Liu, Yasheng Wang, Duyu Tang, Dandan Tu, Lifeng Shang, Xin Jiang, Ruiming Tang, Defu Lian, Qun Liu, and Enhong Chen. 2024a. ToolACE: Winning the points of LLM function calling. arXiv preprint arXiv:2409.00920.

Xiang Liu, Zhenheng Tang, Peijie Dong, Zeyu Li, Bo Li, Xuming Hu, and Xiaowen Chu. 2026. Chunkkv: Semantic-preserving kv cache compression for efficient long-context llm inference. Advances in Neural Information Processing Systems, 38:28728–28778.

Zuxin Liu, Thai Hoang, Jianguo Zhang, Ming Zhu, Tian Lan, Shirley Kokane, Juntao Tan, Weiran Yao, Zhiwei Liu, Yihao Feng, Rithesh Murthy, Liangwei Yang, Silvio Savarese, Juan Carlos Niebles, Huan Wang, Shelby Heinecke, and Caiming Xiong. 2024b. APIGen: Automated pipeline for generating verifiable and diverse function-calling datasets. Advances in Neural Information Processing Systems, 37:54463– 54482.

Xin Men, Mingyu Xu, Qingyu Zhang, Bingning Wang, Hongyu Lin, Yaojie Lu, Xianpei Han, and Weipeng Chen. 2024. ShortGPT: Layers in large language models are more redundant than you expect. arXiv preprint arXiv:2403.03853.

Paul Michel, Omer Levy, and Graham Neubig. 2019. Are sixteen heads really better than one? In Advances in Neural Information Processing Systems, volume 32.

Jesse Mu, Xiang Li, and Noah Goodman. 2023. Learning to compress prompts with gist tokens. Advances in Neural Information Processing Systems, 36:19327– 19352.

Niklas Muennighoff. 2022. Sgpt: Gpt sentence embeddings for semantic search. arXiv preprint arXiv:2202.08904.

Jungsuk Oh, Hyeseo Jeon, Hyunjune Ji, Kyongmin Kong, and Jay-Yoon Lee. 2026. Shallow prefill, deep decoding: Efficient long-context inference via layer-asymmetric kv visibility. arXiv preprint arXiv:2605.06105.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. 2024. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations (ICLR).

Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. 2023. Toolformer: Language models can teach themselves to use tools. Advances in neural information processing systems, 36:68539–68551.

Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. 2018. Self-attention with relative position representations. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pages 464–468.

Zhengliang Shi, Yuhan Wang, Lingyong Yan, Pengjie Ren, Shuaiqiang Wang, Dawei Yin, and Zhaochun Ren. 2025. Retrieval models aren’t tool-savvy: Benchmarking tool retrieval for large language models. arXiv preprint arXiv:2503.01763.

Yifan Song, Weimin Xiong, Dawei Zhu, Wenhao Wu, Han Qian, Mingbo Song, Hailiang Huang, Cheng Li, Ke Wang, Rong Yao, Ye Tian, and Sujian Li. 2023. RestGPT: Connecting large language models with real-world RESTful APIs. arXiv preprint arXiv:2306.06624.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. 2024. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Elena Voita, David Talbot, Fedor Moiseev, Rico Sennrich, and Ivan Titov. 2019. Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 5797–5808.

Benyou Wang, Lifeng Shang, Christina Lioma, Xin Jiang, Hao Yang, Qun Liu, and Jakob Grue Simonsen. 2021. On position embeddings in {bert}. In International Conference on Learning Representations.

Jun Wang, Jiamu Zhou, Muning Wen, Xiaoyun Mo, Haoyu Zhang, Qiqiang Lin, Cheng Jin, Xihuai Wang, Weinan Zhang, Qiuying Peng, and Jun Wang. 2024a. HammerBench: Fine-grained function-calling evaluation in real mobile device scenarios. arXiv preprint arXiv:2412.16516.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei. 2024b. Improving text embeddings with large language models. arXiv preprint arXiv:2401.00368.

Renxi Wang, Xudong Han, Lei Ji, Shu Wang, Timothy Baldwin, and Haonan Li. 2025. Toolgen: Unified tool retrieval and calling via generation. In International Conference on Learning Representations, volume 2025, pages 73473–73498.

Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. 2024. Retrieval head mechanistically explains long-context factuality. arXiv preprint arXiv:2404.15574.

Guangxuan Xiao, Jiaming Tang, Jingwei Zuo, Junxian Guo, Shang Yang, Haotian Tang, Yao Fu, and Song Han. 2025. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. In International Conference on Learning Representations, volume 2025, pages 37228–37253.

Renjun Xu and Yang Yan. 2026. Agent skills for large language models: Architecture, acquisition, security, and the path forward. arXiv preprint arXiv:2602.12430.

Zhangchen Xu, Adriana Meza Soria, Shawn Tan, Anurag Roy, Ashish Sunil Agrawal, Radha Poovendran, and Rameswar Panda. 2025. TOU-CAN: Synthesizing 1.5m tool-agentic data from real-world MCP environments. arXiv preprint arXiv:2510.01179.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. 2025a. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Chen Yang, Ran Le, Yun Xing, Zhenwei An, Zongchao Chen, Wayne Xin Zhao, Yang Song, and Tao Zhang. 2025b. ToolMind technical report: A large-scale, reasoning-enhanced tool-use dataset. arXiv preprint arXiv:2511.15718.

Jingbo Yang, Bairu Hou, Wei Wei, Yujia Bao, and Shiyu Chang. 2025c. Kvlink: Accelerating large language models via efficient kv cache reuse. arXiv preprint arXiv:2502.16002.

Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2025. Cacheblend: Fast large language model serving for rag with cached knowledge fusion. In Proceedings of the twentieth European conference on computer systems, pages 94–109.

Hancheng Ye, Zhengqi Gao, Mingyuan Ma, Qinsi Wang, Yuzhe Fu, Ming-Yu Chung, Yueqian Lin, Zhijian Liu, Jianyi Zhang, Danyang Zhuo, and Yiran Chen. 2025. Kvcomm: Online cross-context kv-cache communication for efficient llm-based multi-agent systems. arXiv preprint arXiv:2510.12872.

Lingfan Yu, Jinkun Lin, and Jinyang Li. 2025. Stateful large language model serving with pensieve. In Proceedings ofthe Twentieth European Conference on Computer Systems, pages 144–158.

Peijie Yu, Wei Liu, Yifan Yang, Jinjian Li, Zelong Zhang, Xiao Feng, and Feng Zhang. 2026. Benchmarking llm tool-use in the wild. arXiv preprint arXiv:2604.06185.

Peitian Zhang, Zheng Liu, Shitao Xiao, Ninglu Shao, Qiwei Ye, and Zhicheng Dou. 2024. Long context compression with activation beacon. arXiv preprint arXiv:2401.03462.

Quqing Zhang, Kai Chen, Ning Liao, Zehao Lin, Bo Tang, Feiyu Xiong, Zhiyu Li, and Xiaoxing Wang. 2026. Sparsex: Efficient segment-level kv cache sharing for interleaved llm serving. arXiv preprint arXiv:2606.01751.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Ré, Clark W. Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2O: heavy-hitter oracle for efficient generative inference of large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jeff Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Efficient execution of structured language model programs. Advances in Neural Information Processing Systems, 37:62557–62583.

Yanzhao Zheng, ZhenTao Zhang, Chao Ma, YuanQiang Yu, JiHuai Zhu, Yong Wu, Tianze Xu, Baohua Dong, Hangcheng Zhu, Ruohui Huang, and Gang Yu. 2026. Skillrouter: Skill routing for LLM agents at scale. CoRR, abs/2603.22455.

Shengyao Zhuang, Xueguang Ma, Bevan Koopman, Jimmy Lin, and Guido Zuccon. 2024. Promptreps: Prompting large language models to generate dense and sparse representations for zero-shot document retrieval. arXiv preprint arXiv:2404.18424.

## A Dataset Construction Details

This appendix describes how the heterogeneous source pools are converted into the training and evaluation data used by ReCache. The objective is to obtain broad resource-use coverage without allowing invalid traces or repeated resource configurations to confound the evaluation of resource compression. Table 5 reports the source-level statistics before and after construction.

Problems in direct aggregation. We combine ToolACE (Liu et al., 2024a), APIGEN (Liu et al., 2024b), ToolMind (Yang et al., 2025b), Toucan (Xu et al., 2025), ToolRet (Shi et al., 2025), WildTool-Bench (Yu et al., 2026), and SkillRouter (Zheng et al., 2026) because they cover complementary schema conventions, candidate-set sizes, interaction lengths, and resource types. Directly concatenating them, however, exposes two systematic problems. First, their declarations and invocations use heterogeneous formats, and generated trajectories may contain unparsable arguments, no actual invocation, or hallucinated invocations. Second, their resource distributions are highly repetitive. Across the five sources with frequency statistics, 29,232 of 37,824 source-level distinct resource names recur in multiple records (77.3%), while 9,901 of 52,621 candidate records repeat an exact previously observed candidate subset (18.8%). The imbalance is concentrated rather than uniform: recurring-name rates reach 99.96% in APIGEN, 100% in Toucan, and 99.95% in ToolMind, while 52.8% of Toucan records duplicate an existing candidate subset. Consequently, random downsampling would preserve shortcuts based on familiar resource identities and candidate configurations instead of encouraging resource-conditioned invocation.

Two-stage diversity-first pipeline. Before sampling, we normalize resource names by trimming whitespace and case, repair lightweight JSONformat errors, parse every declared resource and assistant invocation, and discard traces with no usable invocations, unresolved invocations, undeclared resource names, or sequences beyond the context budget. Repeated invocations to the same resource inside one trajectory are counted once, preventing retries from inflating frequency estimates; when source-provided quality labels are available, they are applied before sampling, including the realism, verifiability, stability, completeness, and conciseness filters for Toucan. The diversity sampler then proceeds in two stages. For each valid example x, we define the candidate-configuration signature

$$
\sigma ( x ) = \mathrm { s o r t } \{ n ( R ) : R \in \mathcal { R } ( x ) \} ,
$$

where $\mathcal { R } ( x )$ is its declared resource set and $n ( R )$ is the normalized resource name.

In Stage I, an example is accepted only if its signature has not appeared and none of its called resources has been used by a previously selected example, jointly prioritizing candidate-subset and invocation-level diversity. Since strict uniqueness cannot fill every source quota, Stage II admits additional valid examples in the same quality-ranked order until each predefined quota is reached. This second stage preserves source and scenario coverage while retaining the diversity gains established by the first pass. A final global validity pass reduces the 52,964 sampled candidates to 49,424 training examples, removing 3,540 records (6.7%).

Evaluation split construction. After the twostage selection, we construct in-distribution (IND) and out-of-distribution (OOD) candidate pools at the resource level. For each resource $R _ { i } , \mathrm { f r q } ( R _ { i } )$ is the number of valid examples in which it is declared; low-frequency resources are reserved for OOD evaluation and removed from training. An example is eligible for the OOD pool only when every declared resource belongs to this reserved set, which prevents partial overlap with training resources. IND candidates use resources observed during training but contain different examples, queries, and invocation patterns. Exact configuration signatures are deduplicated independently within both pools before sampling, preventing repeated candidate subsets from making either test condition artificially easy. We finally sample 1,000 IND and 1,000 OOD examples, yielding matched evaluation sizes while isolating generalization to unseen resources.

## B Structural Capacity Analysis

## B.1 Preliminary Budget Investigation

We investigate how computation is allocated across transformer layers and KV head groups under different model scales and training regimes when identifying the optimal structural budget Ω<sup>⋆</sup>. The analysis focuses on two factors that influence the achievable sparsity level: the amount of optimization supervision and the representational capacity of the backbone model.

<table><tr><td>Group</td><td>Source</td><td>Train pool</td><td>Distinct names</td><td>Subset dup.</td><td>Recurring names</td><td>IND cand.</td><td>OOD cand.</td></tr><tr><td rowspan="10">Tool</td><td>ToolACE</td><td>8,400</td><td>14,566</td><td>178</td><td>7,144</td><td>200</td><td>195</td></tr><tr><td>APIGEN</td><td>10,000</td><td>2,806</td><td>734</td><td>2,805</td><td>500</td><td>146</td></tr><tr><td>Toucan (all generators)</td><td>13,071</td><td>4,355</td><td>6,901</td><td>4,355</td><td>600</td><td>493</td></tr><tr><td>- Kimi-K2</td><td>3,333</td><td>1,103</td><td>1,772</td><td>1,103</td><td>200</td><td>145</td></tr><tr><td>- GPT-OSS-120B</td><td>4,184</td><td>1,412</td><td>2,162</td><td>1,412</td><td>200</td><td>130</td></tr><tr><td>- Qwen3-32B</td><td>5,554</td><td>1,840</td><td>2,967</td><td>1,840</td><td>200</td><td>218</td></tr><tr><td>ToolRet</td><td>1,150</td><td>1,593</td><td>235</td><td>431</td><td>100</td><td>150</td></tr><tr><td>ToolMind</td><td>20,000</td><td>14,504</td><td>1,853</td><td>14,497</td><td>500</td><td>190</td></tr><tr><td>WildToolBench</td><td>256</td><td></td><td>一</td><td></td><td></td><td></td></tr><tr><td>SkillRouter</td><td>87</td><td>一</td><td>一</td><td>一</td><td>一</td><td>一</td></tr><tr><td colspan="2">Sources with frequency statistics</td><td>52,621</td><td>37,824</td><td>9,901</td><td>29,232</td><td>1,900</td><td>1,174</td></tr><tr><td colspan="2">Candidate pool</td><td>52,964</td><td></td><td>一</td><td></td><td>1,900</td><td>1,174</td></tr><tr><td colspan="2">Final dataset</td><td>49,424</td><td></td><td>一</td><td></td><td>1,000</td><td>1,000</td></tr></table>

Table 5: Dataset composition and split statistics. Source rows report eligible examples before final selection; the Toucan aggregate is not double-counted in the summary row. Distinct names counts source-level unique declared tools or skills, Subset dup. counts records beyond the first occurrence of an identical candidate configuration, and Recurring names counts resources appearing in multiple records. The summary row corresponds to 18.8% duplicated candidate records and 77.3% recurring resource names. IND cand. and OOD cand. denote eligible in-distribution and out-of-distribution candidates, and dashes mark unavailable statistics

Observation 1: Head-group pruning is more sensitive to limited supervision. Under $\mathcal { D } _ { \mathrm { s u b s e t } }$ $Q _ { s }$ is substantially more sensitive to head-group sparsification than to layer sparsification. Retaining six of eight KV head groups across all layers, denoted by $\Omega _ { L , 6 } ,$ , reduces Inv-F1 by 18.7% compared to $\Omega _ { \mathrm { f u l l } }$ (Table 6). In contrast, retaining 20 of 28 layers across all head groups, denoted by $\Omega _ { 2 0 , G }$ , changes Inv-F1 by less than 0.2%. This asymmetry indicates that head-group sparsification requires stronger supervision to identify effective resource-conditioned computation patterns.

The auxiliary-loss experiment provides further evidence. Incorporating an invocation-oriented objective for $Q _ { s }$ with $\Omega _ { L , 6 }$ under $\mathcal { D } _ { \mathrm { s u b s e t } }$ increases Inv-F1 to 69.9%, indicating that direct invocation supervision partially mitigates the degradation caused by head-group sparsification. Similarly, training on the full dataset D narrows the performance gap between sparse configurations and $\Omega _ { \mathrm { f u l l } }$ Together, these results show that the viable structural budget depends jointly on retained route capacity and the available optimization signal.

Observation 2: Model scale affects the required structural budget. Despite full-data training, models with different capacities retain distinct structural requirements (Table 7). $Q _ { l }$ maintains comparable invocation performance with fewer active head groups, while $Q _ { s }$ requires broader headgroup participation. This implies that larger models can sustain resource-conditioned computation with more aggressive structural sparsification.

Overall, these observations suggest that the attainable structural sparsity is determined by the effective capacity of retained resource-visible routes, which depends on both optimization supervision and model representation capability. The larger Qwen3-4B provides greater capacity within each retained route, attributable to its larger hidden and FFN dimensions and its four query heads per KV group, compared with two in Qwen3-1.7B.

## B.2 Contribution Distribution

Table 9 reports detailed weights of individual units, the cumulative contribution coverage of layers, and head groups, respectively.

## C Semantic Pruning Ablation

Table 8 presents the semantic pruning ablation on $\mathcal { D } _ { \mathrm { s u b s e t } }$ , where all experiments are conducted under $\Omega _ { 2 0 , 3 }$ . The objective is to identify the semantic fields that should be retained before scaling the selected configuration to full-data training.

The results reveal a clear hierarchy among the evaluated semantic fields. Adding only the resource name already yields substantial improvements, reducing the hallucination rate by 97.63% relative to the suffix-only setting while increasing all resource-ID metrics above 88%. Incorporating argument names further improves Inv-F1 by 22.5%, demonstrating that parameter identifiers provide essential grounding for argument generation.

<table><tr><td colspan="4"> $\pmb { \mathcal { D } } _ { \mathbf { s u b s e t } }$ </td></tr><tr><td rowspan="2">Config Inv-F1</td><td colspan="3">Resource ID</td></tr><tr><td>ID-P ID-R</td><td>ID-F1</td><td></td></tr><tr><td> $\Omega _ { \mathrm { f u l l } }$ </td><td>72.1 92.0</td><td>90.8</td><td>90.9</td><td>0.4</td></tr><tr><td> $\Omega _ { L , 5 }$ </td><td>49.6 74.5</td><td>73.5</td><td>73.4</td><td>4.8</td></tr><tr><td> $\Omega _ { L , 6 }$ </td><td>53.4</td><td>79.6 77.9</td><td>78.3</td><td>3.2</td></tr><tr><td> $\Omega _ { L , 7 }$ </td><td>57.3</td><td>83.4 81.8</td><td>82.1</td><td>2.9</td></tr><tr><td> $\Omega _ { 1 5 , G }$ </td><td>69.3</td><td>89.2 88.3</td><td>88.3</td><td>1.1</td></tr><tr><td> $\Omega _ { 2 0 , G }$ </td><td>70.5</td><td>90.6 89.5</td><td>89.6</td><td>1.1</td></tr></table>

D
<table><tr><td rowspan="2">Config Inv-F1</td><td colspan="3">Resource ID</td><td rowspan="2">Halluc.</td></tr><tr><td>ID-P</td><td>ID-R</td><td>ID-F1</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } }$ </td><td>80.7</td><td>94.9</td><td>94.3 94.4</td><td>0.1</td></tr><tr><td> $\Omega _ { L , 5 }$ </td><td>79.5</td><td>94.4 93.6</td><td>93.8</td><td>0.6</td></tr><tr><td> $\Omega _ { L , 6 }$ </td><td>79.8</td><td>94.4 93.9</td><td>93.9</td><td>0.3</td></tr><tr><td> $\Omega _ { L , 7 }$ </td><td>80.6</td><td>94.8</td><td>94.2 94.3</td><td>0.2</td></tr><tr><td> $\Omega _ { 1 5 , G }$ </td><td>80.0</td><td>94.2</td><td>93.7 93.8</td><td>0.3</td></tr><tr><td> $\Omega _ { 2 0 , G }$ </td><td>80.6</td><td>94.7</td><td>94.1 94.2</td><td>0.4</td></tr></table>

Table 6: Structural budget ablation for $Q _ { s }$ across training data regimes with comparable budgets.

<table><tr><td>Config Inv-F1</td><td>Resource ID ID-P ID-R ID-F1</td></tr><tr><td> $Q _ { s }$   $\Omega _ { \mathrm { f u l l } }$  80.7</td><td>94.9 94.3 94.4 0.1</td></tr><tr><td> $\Omega _ { 2 0 , 5 }$  78.8  $\Omega _ { 2 0 , 7 }$  80.0</td><td>94.1 93.5 93.6 0.5 94.7 94.0 94.1 0.3</td></tr><tr><td> $Q _ { l }$ </td><td>0.2</td></tr><tr><td> $\Omega _ { \mathrm { f u l l } }$  82.3  $\Omega _ { 2 0 , 2 }$  81.9</td><td>96.4 95.8 96.0 95.3 94.7 94.7</td></tr></table>

Table 7: Structural budget ablation trained on D.
<table><tr><td>Config</td><td>Inv-F1</td><td>Resource ID</td><td></td><td>Halluc.</td></tr><tr><td> $\Omega _ { 2 0 , 3 }$ </td><td>76.1</td><td>ID-P ID-R ID-F1 93.1 92.8</td><td>92.5</td><td>0.9</td></tr><tr><td>Schema representation ablation</td><td></td><td></td><td></td><td></td></tr><tr><td>Suffix-only Schema + Resource Name</td><td>14.8</td><td>22.8 23.0</td><td>22.9</td><td>76.0</td></tr><tr><td>+ Arg Name</td><td>45.2</td><td>89.9 88.8</td><td>88.9</td><td>1.8</td></tr><tr><td></td><td>67.7</td><td>92.1 91.1</td><td>91.2</td><td>1.3</td></tr><tr><td>+ Resource Desc.</td><td>67.9 72.8</td><td>92.2 91.4</td><td>91.4</td><td>1.1</td></tr><tr><td>+ Arg Desc.</td><td></td><td>92.8 91.9</td><td>92.0</td><td>1.0</td></tr><tr><td>+ Both Desc.</td><td>73.4</td><td>93.0 92.2</td><td>92.2</td><td>1.2</td></tr></table>

Table 8: Semantic pruning ablation trained on $\mathcal { D } _ { \mathrm { s u b s e t } } . \ \mathrm { A r g }$ stands for argument, Desc. stands for description. Ω<sub>20,3</sub> denotes the routing configuration with $K _ { L } = 2 \bar { 0 } , K _ { G } = 3 .$

Within the considered schema components, argument descriptions provide additional benefits by supplying semantic constraints for parameter prediction. In contrast, resource descriptions offer limited contribution. Adding only resource descriptions on top of resource and argument names yields negligible improvement, while additionally including resource descriptions alongside argument descriptions changes all evaluation metrics by less than 0.7% compared with preserving argument descriptions alone. These results suggest that resource names, argument names, and argument descriptions constitute the core semantic components for resource invocation, while resource descriptions provide limited additional information once the executable interface has been preserved.

<table><tr><td colspan="4"> $Q _ { l } ( \mathbf { Q } \mathbf { w e n 3 - 4 B } )$ </td><td colspan="4">Layers  $Q _ { s } ( \bf { Q } { \bf { w e n } } 3 { - } 1 . 7 B )$ </td></tr><tr><td>rank unit</td><td colspan="3"></td><td colspan="4"></td></tr><tr><td rowspan="14"></td><td></td><td>wi (%)</td><td>CC (%)</td><td>rank</td><td>unit</td><td>wi (%)</td><td>CC (%)</td></tr><tr><td> $L _ { 3 4 }$   $L _ { 3 5 }$ </td><td>15.56</td><td>15.6</td><td>1</td><td> $L _ { 2 2 }$ </td><td>20.09</td><td>20.1</td></tr><tr><td>12</td><td>10.53 9.37</td><td>26.1</td><td></td><td>2  $L _ { 2 5 }$ </td><td>12.90</td><td>33.0</td></tr><tr><td>3  $L _ { 2 3 }$ </td><td></td><td>35.5</td><td></td><td>3  $L _ { 2 0 }$ </td><td>12.14</td><td>45.1</td></tr><tr><td>4  $L _ { 2 5 }$ </td><td>8.31</td><td>43.8</td><td></td><td>4  $L _ { 2 6 }$ </td><td>10.90</td><td>56.0</td></tr><tr><td>5  $L _ { 3 2 }$ </td><td>8.01</td><td>51.8</td><td></td><td>5  $L _ { 2 7 }$ </td><td>10.39</td><td>66.4</td></tr><tr><td>6  $L _ { 3 0 }$  7  $L _ { 3 3 }$ </td><td>6.95 6.89</td><td>58.7</td><td></td><td>6  $L _ { 1 7 }$ </td><td>6.82</td><td>73.2</td></tr><tr><td>8</td><td></td><td></td><td>65.6</td><td>7  $L _ { 2 1 }$ </td><td>6.70</td><td>79.9</td></tr><tr><td>9</td><td> $L _ { 2 7 }$   $L _ { 2 1 }$ </td><td>5.17 4.87</td><td>70.8</td><td>8</td><td> $L _ { 1 9 }$  4.65</td><td>84.6</td></tr><tr><td></td><td></td><td></td><td>75.7</td><td>9  $L _ { 2 4 }$ </td><td>4.00</td><td>88.6</td></tr><tr><td>10</td><td> $L _ { 2 8 }$ </td><td>4.64</td><td>80.3</td><td>10  $L _ { 2 3 }$ </td><td>2.13</td><td>90.7</td></tr><tr><td>11</td><td> $L _ { 2 4 }$  4.02</td><td></td><td>84.3</td><td>11  $L _ { 1 8 }$ </td><td>1.89</td><td>92.6</td></tr><tr><td>12 13</td><td> $L _ { 2 2 }$ </td><td>3.74</td><td>88.1</td><td>12</td><td> $L _ { \mathrm { 1 5 } }$  1.73</td><td>94.3</td></tr><tr><td></td><td> $L _ { 1 7 }$ </td><td>2.30</td><td>90.4</td><td>13</td><td> $L _ { 1 3 }$ </td><td>1.39 95.7</td></tr><tr><td>14</td><td> $L _ { 3 1 }$ </td><td>1.96</td><td>92.3</td><td>14  $L _ { 1 0 }$ </td><td>1.13</td><td>96.9</td></tr><tr><td>15</td><td> $L _ { 1 8 }$ </td><td>1.29</td><td>93.6</td><td>15  $L _ { 9 }$ </td><td>1.03</td><td>97.9</td></tr><tr><td>16</td><td> $L _ { 1 4 }$ </td><td>1.26</td><td>94.9</td><td>16  $L _ { 4 }$ </td><td>0.88</td><td>98.8</td></tr><tr><td>17</td><td> $L _ { 2 0 }$ </td><td>0.78</td><td>95.7</td><td>17</td><td> $L _ { 1 6 }$  0.56</td><td>99.3</td></tr><tr><td>18</td><td> $L _ { 3 6 }$ </td><td>0.69</td><td>96.3</td><td>18</td><td> $L _ { 1 2 }$ </td><td>0.18</td><td>99.5</td></tr><tr><td>19</td><td> $L _ { 1 9 }$ </td><td>0.68</td><td>97.0</td><td>19</td><td> $L _ { 1 1 }$ </td><td>0.15</td><td>99.7</td></tr><tr><td>20</td><td> $L _ { 1 6 }$ </td><td>0.68</td><td>97.7</td><td>20</td><td> $L _ { 8 }$ </td><td>0.14</td><td>99.8</td></tr><tr><td>21</td><td> $L _ { 1 0 }$ </td><td>0.60</td><td>98.3</td><td>21</td><td> $L _ { 2 8 }$ </td><td>0.11</td><td>99.9</td></tr><tr><td>22 23</td><td> $L _ { 5 }$ </td><td>0.33</td><td>98.6</td><td>22</td><td> $L _ { 5 }$ </td><td>0.06</td><td>100.0</td></tr><tr><td>24</td><td> $L _ { 1 1 }$ </td><td>0.29</td><td>98.9</td><td>23</td><td> $L _ { 2 }$ </td><td>0.02</td><td>100.0</td></tr><tr><td>25</td><td> $L _ { 9 }$ </td><td>0.27</td><td>99.2</td><td>24</td><td> $L _ { 1 }$ </td><td>0.00</td><td>100.0</td></tr><tr><td></td><td> $L _ { 1 2 }$ </td><td>0.26</td><td>99.5</td><td>25</td><td> $L _ { 3 }$ </td><td>0.00</td><td>100.0</td></tr><tr><td>26</td><td> $L _ { \mathrm { 1 5 } }$ </td><td>0.26</td><td>99.7</td><td>26</td><td> $L _ { 6 }$ </td><td>0.00</td><td>100.0</td></tr><tr><td>27</td><td></td><td>0.13</td><td>99.8</td><td>27</td><td> $L _ { 7 }$ </td><td>0.00</td><td>100.0</td></tr><tr><td>28</td><td> $L _ { 1 3 }$   $L _ { 8 }$ </td><td>0.07</td><td>99.9</td><td>28</td><td> $L _ { 1 4 }$ </td><td>0.00</td><td>100.0</td></tr><tr><td>29</td><td> $L _ { 3 }$ </td><td>0.07</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>30</td><td> $L _ { 1 }$ </td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>31</td><td> $L _ { 2 }$ </td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>32</td><td> $L _ { 4 }$ </td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>33</td><td></td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>34</td><td> $L _ { 6 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $L _ { 7 }$ </td><td>0.00</td><td>100.0</td><td></td><td></td><td></td><td></td></tr><tr><td>35 36</td><td> $L _ { 2 6 }$   $L _ { 2 9 }$ </td><td>0.00 0.00</td><td>100.0 100.0</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8"></td></tr><tr><td rowspan="4">rank</td><td></td><td> $Q \imath ( \mathbf { Q } \mathbf { w e n } 3 { \mathbf { - } } 4 \mathbf { B } )$ </td><td></td><td>KV head groups</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td> $Q _ { s } ( \bf { Q } { \bf { w e n } } 3 { - } 1 . 7 B )$ </td></tr><tr><td>unit</td><td>wi (%)</td><td>CC (%)</td><td>rank</td><td>unit</td><td> $_ { w _ { i } }$  (%)</td><td>CC (%)</td></tr><tr><td>12  $G _ { 4 }$   $G _ { 6 }$ </td><td>16.54</td><td></td><td>16.5</td><td>1  $G _ { 5 }$  2</td><td>22.84 19.01</td><td>22.8</td></tr><tr><td>3 4 5</td><td> $G _ { 3 }$ </td><td>16.17 14.64</td><td>32.7 47.3</td><td>3</td><td> $G _ { 3 }$   $G _ { 8 }$ </td><td>17.27</td><td>41.9 59.1</td></tr><tr><td></td><td> $G _ { 2 }$ </td><td>12.37</td><td>59.7</td><td>4</td><td> $G _ { 6 }$ </td><td>16.77</td><td>75.9</td></tr><tr><td></td><td> $G _ { 8 }$ </td><td>11.62</td><td>71.3</td><td>5</td><td> $G _ { 1 }$ </td><td>10.91</td><td>86.8</td></tr><tr><td>6</td><td> $G _ { 1 }$ </td><td>9.68</td><td>81.0</td><td>6</td><td> $G _ { 7 }$ </td><td>8.83</td><td>95.6</td></tr><tr><td>7</td><td> $G _ { 5 }$ </td><td>9.61</td><td>90.6</td><td>7</td><td> $G _ { 4 }$ </td><td>3.64</td><td>99.3</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>8</td><td> $G _ { 7 }$ </td><td>9.37</td><td>100.0</td><td>8</td><td> $G _ { 2 }$ </td><td>0.74</td><td>100.0</td></tr></table>

Table 9: Individual contribution weights and cumulative contribution coverage of structural units. Within each panel, units are ranked independently by decreasing w , and CC at rank k is the cumulative positive contribution weight through rank k. Bold entries fall within the empirical budgets $K _ { L } = 2 0$ $K _ { G } { = } 3$ for $Q _ { l }$ and $K _ { L } = 2 0$ $K _ { G } { = } 7$ for $Q _ { s }$