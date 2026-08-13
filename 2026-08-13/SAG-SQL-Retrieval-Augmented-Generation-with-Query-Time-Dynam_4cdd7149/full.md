# SAG: SQL-Retrieval Augmented Generation with Query-Time Dynamic Hyperedges

Yuchao Wu<sup>∗</sup>, Junqin Li, XingCheng Liang, Yongjie Chen, Yinghao Liang

Linyuan Mo, Guanxian Li

Zleap AI

{jomy,junqing,lensen,jinzhoulawen,leo}@zleap.com

{mo-linyuan,li guanxian}@foxmail.com

## Abstract

While retrieval-augmented generation (RAG) has proven efective at giving LLMs access to external knowledge, mainstream dense-retrieval implementations remain inherently limited in handling structured constraints and multi-hop reasoning. Graph-based methods address this by constructing knowledge graphs ofline, but they often fragment semantics, incur high maintenance, and complicate incremental updates. We propose SAG (SQL-Retrieval Augmented Generation), a structured retrieval architecture that organizes documents into an event-entity index without building a global knowledge graph. SAG represents each chunk as a semantically complete event paired with its entities, forming a latent hyperedge that preserves n-ary relations without decomposing them into triples. At query time, SAG treats shared entities as join keys to connect related chunks. This dynamically yields a query-scoped neighborhood of events, and yet every piece of evidence remains the original chunk throughout. Experiments on HotpotQA, 2WikiMultiHopQA, and MuSiQue show that SAG achieves the best retrieval and end-to-end QA performance on every benchmark, with gains that widen as reasoning-chain complexity increases. On MuSiQue, where multi-hop evidence chaining is most demanding, SAG reaches 80.36% Recall@5, outperforming the strongest baseline by 11.52 points. This work paves the way for knowledge infrastructure that enables LLM agents to retrieve and reason over continually growing organizational knowledge.<sup>1</sup>

## 1 Introduction

What an agent can do is bounded by what it can retrieve. Organizational knowledge is distributed across documents, messages, customer systems, and code; much of it is unstructured, relational, and continuously accumulating. Making this knowledge available to agents is therefore an infrastructure problem as well as a modeling problem. Multi-hop question answering makes this gap concrete. Answers often require evidence from multiple documents, whereas conventional RAG pipelines (Lewis et al., 2020; Karpukhin et al., 2020) rank passages independently by their similarity to the query. This approach works well when a single passage contains the answer, but it does not explicitly represent associations across passages. In multi-hop retrieval, failing to recover one intermediate passage can break the entire evidence chain (Yang et al., 2018; Trivedi et al., 2022; Mavi et al., 2024). The challenge is therefore not only to improve single-shot recall, but also to organize evidence so that cross-document associations can be recovered reliably.

Structure-augmented methods address this problem by constructing knowledge graphs or hierarchical representations ofline (Edge et al., 2024; Gutierrez et al., 2025). When n-ary events are decomposed´ into independently indexed triples, however, their shared event boundary may be lost. Hypergraphbased methods (Luo et al., 2025) preserve higher-order relations but still encode them in a persistent corpus-level structure. These structures support graph traversal and ranking, but the connections available during retrieval are determined before the query arrives, while ingestion and maintenance remain separate from query execution. This raises a diferent question: can the structure needed by a query be activated from an append-only index during retrieval, without constructing a global knowledge graph?

Our central claim is that multi-hop retrieval need not rely exclusively on either dense similarity or a globally constructed graph (Figure 1). The required associative structure is latent in the events described by passages and the entities shared across them. SAG exposes this structure through an event–entity index. Each chunk is independently represented as one semantically complete event linked to a set of indexing entities. Because an event binds all its entities within one retrieval unit, it defines a latent hyperedge that retains the event’s n-ary relation rather than decomposing it into independent triples. At query time, SQL joins over shared entities activate a local neighborhood of related events. This structural path is combined with dense retrieval, after which an LLM selects evidence from the compressed candidate set. SAG constructs no global corpus graph, and selected events are mapped back to their original chunks before answer generation. Under unified evaluation settings, SAG achieves the best retrieval and end-to-end QA performance on HotpotQA, 2WikiMultiHopQA, and MuSiQue. Its largest retrieval gain occurs on MuSiQue, where SAG reaches 80.36% Recall@5 and exceeds the strongest baseline by 11.52 points.

To summarize, our contributions are fourfold. (1) We introduce SAG, a retrieval architecture that activates query-specific latent hyperedges over an event–entity index through SQL joins, without constructing a global knowledge graph. (2) We formalize the event–entity representation, characterize the neighborhoods recovered through incidence joins, and compare its corpus-level connectivity with four structure-augmented baselines. (3) We evaluate SAG on three multi-hop benchmarks under a unified configuration and isolate the efects of representation, expansion, final selection, and candidate budget. (4) We evaluate robustness to embedding replacement and continual corpus growth, and show how bounded candidate activation limits the downstream SQL and LLM workloads. Beyond the benchmark study, we provide an engineering implementation inspired by SAG’s core mechanisms.<sup>2</sup> Together, these properties make SAG a step toward knowledge infrastructure for agents operating over continually growing corpora.

![](images/6f2b194de0b5ef49525e55df7135d9caf375236a7159aed1eb91e3258f6583c2.jpg)  
Figure 1: Three RAG paradigms. NaiveRAG retrieves top-� chunks by dense similarity, while GraphRAG organizes evidence in an ofline knowledge graph. SAG stores event–entity relations in an append-only index and activates query-specific latent hyperedges through SQL joins.

## 2 Related Work

Retrieval-Augmented Generation. RAG combines a parametric generator with a dense passage index to incorporate external knowledge (Lewis et al., 2020). Later work adapts when and how retrieval is performed. Self-RAG uses reflection tokens to decide when to retrieve and to evaluate retrieved evidence (Asai et al., 2024), while FLARE retrieves from predicted upcoming sentences when generation contains low-confidence tokens (Jiang et al., 2023b). Adaptive-RAG routes queries among no-retrieval, single-step, and iterative strategies according to predicted complexity (Jeong et al., 2024). IRCoT instead alternates retrieval with chain-of-thought generation, using each reasoning step to guide the next retrieval (Trivedi et al., 2023). These methods modify retrieval control or query formulation, whereas SAG changes how corpus knowledge is represented and connected.

Structure-Augmented Retrieval. GraphRAG constructs an entity graph and community summaries for answering global corpus-level questions (Edge et al., 2024). HippoRAG builds a knowledge graph from extracted relations and propagates relevance with personalized PageRank, while HippoRAG 2 integrates passages more directly and strengthens the online use of an LLM (Gutierrez et al., 2024;´ 2025). LightRAG combines incremental graph updates with low- and high-level retrieval over entities and relations (Guo et al., 2025). SiReRAG constructs similarity and relatedness trees and flattens their nodes into a unified retrieval pool (Zhang et al., 2025), whereas StructRAG selects and constructs a task-appropriate structure at inference time (Li et al., 2025). These methods either maintain corpus-level structures or reconstruct structured views for individual queries. SAG instead stores event–entity incidence records and activates query-specific associations through SQL joins without materializing a global graph. ChatDB and StructGPT query existing databases or structured sources, but do not organize unstructured corpora for retrieval (Hu et al., 2023; Jiang et al., 2023a).

Hyperedge and Higher-Order Representations. Hypergraphs and n-ary knowledge-base models represent relations among multiple entities without reducing each relation to independent binary triples (Zhou et al., 2006; Feng et al., 2019; Fatemi et al., 2020; Galkin et al., 2020; Liu et al., 2020; 2021). HyperGraphRAG extracts n-ary facts and stores a corpus-level hypergraph with vector indexes for entities and hyperedges (Luo et al., 2025). Hyper-RAG retrieves both entities and correlations from a hypergraph repository and expands them through structural difusion (Feng et al., 2026). HGRAG treats entities as nodes and passages as hyperedges, combining entity- and passage-level similarity through hypergraph difusion (Wang et al., 2026). Graph-R1 constructs a lightweight knowledge hypergraph and learns multi-turn retrieval through reinforcement learning (Luo et al., 2026). These methods materialize a corpus-level hypergraph. SAG instead stores each event and its entities as relational incidence rows. SQL joins over shared entities activate only the latent hyperedges required by the query, after which the original chunks are returned to the reader.

## 3 SAG

SAG has an ofline indexing phase and an online phase with seed retrieval, query-time expansion, and final selection (Figure 2). SQL performs exact entity joins, vector retrieval finds semantically relevant seed chunks even when the query and the passage share no exact terms, and the LLM is reserved for event-entity extraction, query-entity identification, and final selection over the compressed candidate set.

## 3.1 Event-Entity Index

SAG builds an event-entity index rather than a global knowledge graph. Each chunk is processed independently into one event � and a set of entities �(�), which together define a latent hyperedge. New documents can therefore be appended without recomputing existing records.

Event. An event is a concise statement of the chunk’s core content, with one event extracted per chunk. Decomposing a chunk into independent triples can fragment its meaning, especially when several entities jointly constrain the underlying fact. SAG instead keeps the event intact: retrieval operates on the concise event text, while the corresponding original chunk is returned as evidence.

![](images/4eb5dc0c972a9ca10386f0a828acddb95dbc8c3e0e51d329bdd3ef9f0a991e41.jpg)  
Figure 2: Architecture overview of SAG. Ofline, chunks are indexed as events and entities across SQL, vector, and full-text stores; online, initial recall is expanded at query time, and final selection operates over the compressed candidate set. The output panel illustrates the case in which the structural path reaches its five-event cap; the semantic path fills the remaining slots to return ten chunks.

Entities. Entities span 11 semantic types.<sup>3</sup> They are not treated as self-contained evidence; instead, they serve primarily as indexing and expansion keys that connect events sharing the same participant or concept. Events and entities are produced in parallel by a single LLM call per chunk and written into SQL as many-to-many rows, as well as into the vector and full-text indices.

The index deliberately avoids full entity disambiguation. It relies only on string normalization and SQL deduplication, preserving independent chunk processing and append-only ingestion. This tolerance for imperfect normalization is a deployment choice: production corpora arrive continuously and are rarely clean, and requiring global disambiguation or re-chunking before indexing would forfeit the append-only property that keeps ingestion practical as the corpus grows.

## 3.2 Latent Hyperedges

Let � denote the entity vertices and � the indexed events (hyperedges). Each event ℎ ∈ � defines a hyperedge over its entity set $V ( h ) \subseteq V$ , keeping all entities mentioned in the event together as one retrieval unit. The event text preserves the relational meaning that a bare set of entity names cannot express, while the linked original chunk remains the evidence returned to the downstream reader.

The event-entity table stores each hyperedge as a set of rows (ℎ, �) for � ∈ �(ℎ). Grouping these rows by event identifier recovers the complete entity set of the event, making the representation lossless with respect to incidence.

One expansion round (Section 3.4) walks from seed events to their entities and then back to co-incident events. The activated structure exists only for the current query, and ingesting a new chunk requires only appending new rows rather than rebuilding a global graph. Corpus statistics support this structural interpretation, with SAG showing less fragmentation than GraphRAG, LightRAG, and HyperGraphRAG on MuSiQue and more than 93% giant-component coverage on every benchmark (Appendix B). Section 5.2 further shows that hyperedges outperform triple decomposition in Recall@5.

## 3.3 Seed Retrieval

Given a query $q ,$ , SAG first defines a query-specific working corpus $D _ { q } .$ . In the full-pool configuration used for the main results, $D _ { q } = D$ . Under bounded activation, embedding retrieval first selects the top $K _ { \mathrm { p o o l } }$ chunks from the global corpus, and all subsequent event and entity operations are restricted to records associated with those chunks. This corpus-level activation budget is distinct from $K _ { \mathrm { c a n d } }$ which controls the number of events passed to the final LLM reranker.

Within $D _ { q } ,$ , SAG constructs an initial event set � through two parallel retrieval paths, each addressing a failure mode of the other.

Path A: entity-guided structured recall. A lightweight LLM call identifies key entities in the query and produces a seed set $Q = \left\{ \nu _ { i } \right\}$ . Similarity search over the entity vector index expands these seeds into an augmented set ${ \hat { Q } } \supseteq Q$ using $\mathrm { S i m } _ { \mathrm { e n t } } : V \times V  [ 0 , 1 ]$ . Because entities are short strings, SAG uses a high similarity threshold $\tau _ { \mathrm { e n t } } = 0 . 9$ by default; looser matching is more likely to conflate distinct entities than to recover genuine aliases. A SQL join then retrieves all events sharing at least one entity with ${ \widehat { Q } } \colon$

$$
R _ { \mathrm { e n t } } = \left\{ h \in H \Big \vert V ( h ) \cap \widehat { Q } \neq \emptyset \right\} .\tag{1}
$$

Path B: direct event recall via query vector. In parallel, SAG retrieves events directly by embedding similarity to the query using $\mathrm { S i m } _ { \mathrm { e v t } } : H \times \{ q \}  [ 0 , 1 ]$ . Because events are longer semantic statements, this path uses a deliberately low threshold $\tau _ { \mathrm { e v t } } = 0 . 4$ by default, to favor recall and leave precision control to later ranking. The retrieved events form $R _ { \mathrm { e v t } }$

The two paths are merged to form the seed event set:

$$
R = R _ { \mathrm { e n t } } \cup R _ { \mathrm { e v t } } .\tag{2}
$$

## 3.4 Expansion and Selection

Expansion. Seed retrieval finds events that either share entities with the query or have high direct semantic similarity. Multi-hop questions, however, may require evidence connected through intermediate entities that do not appear in the query.

Starting from �, SAG uses a reverse SQL join to collect the entities associated with the seed events. Previously unexplored entities form an expansion frontier, which is pruned to a fixed budget before the next join to bound both computation and candidate growth. SAG then retrieves previously unseen events associated with the retained frontier entities.

Each event-to-entity-to-event composition constitutes one expansion round. Expansion runs for at most � rounds, with $L = 1$ by default, and each round introduces only entities and events not visited earlier. Let $X \subseteq H$ denote the events added through expansion. The complete candidate pool is

$$
C = R \cup X ,\tag{3}
$$

Coarse ranking. SAG ranks the events in � by embedding similarity to the query and retains the top $K _ { \mathrm { c a n d } }$ events:

$$
\widehat { C } = \mathrm { T o p K } _ { K _ { \mathrm { c a n d } } } ( C , q ) .\tag{4}
$$

Dual-path output. The structural path preserves evidence recovered through cross-entity expansion, which may have low direct query similarity despite being necessary for multi-step reasoning. The semantic path retains chunks that are directly relevant to the query but may not participate in the expanded structure.

The system returns $K _ { \mathrm { o u t } } = 1 0$ chunks. In the structural path, the LLM reads $\widehat { C }$ in context and selects up to $K _ { \mathrm { e v e n t } } = 5$ events that jointly entail the answer, producing $E _ { \mathrm { s e l } }$ with $| E _ { \mathrm { s e l } } | \le K _ { \mathrm { e v e n t } }$ . These events are mapped to chunks as $D _ { \mathrm { e v t } } = \{ \phi ( h ) : h \in E _ { \mathrm { s e l } } \}$ through the event-to-chunk mapping � : $H  D$ This is a contextual selection step rather than a pointwise reranking: cross-candidate attention lets the model detect the implicit relations between events and return the chain suficient to support an answer, which a per-event score cannot capture. The semantic path produces a ranked chunk list $R _ { \mathrm { c h u n k } }$ by direct retrieval over the chunk index. After removing chunks already in $D _ { \mathrm { e v t } } .$ , it fills the remaining $K _ { \mathrm { d i r e c t } } ( q ) = K _ { \mathrm { o u t } } - | D _ { \mathrm { e v t } } |$ slots. Thus, $K _ { \mathrm { e v e n t } }$ is a cap on structural selections rather than a fixed five-chunk allocation, and $K _ { \mathrm { d i r e c t } }$ varies by query:

$$
D _ { \mathrm { o u t } } = D _ { \mathrm { e v t } } \cup \mathrm { T o p K } _ { K _ { \mathrm { d i r e c t } } ( q ) } \left( R _ { \mathrm { c h u n k } } \ \backslash \ D _ { \mathrm { e v t } } \right) .\tag{5}
$$

These explicit stages make retrieval failures traceable. Section 5.2 shows that most expansion gains arrive in the first round, contextual LLM selection outperforms pointwise reranking, and the two output paths are complementary. Appendix C presents successful and unsuccessful activation traces.

## 4 Experiments

## 4.1 Datasets and Metrics

We evaluate on three multi-hop benchmarks: MuSiQue (Trivedi et al., 2022) (answerable subset, up to four counterfactually filtered hops), 2WikiMultiHopQA (Ho et al., 2020) (inference-type multi-document questions), and HotpotQA (Yang et al., 2018) (two-hop). Following the HippoRAG 2 evaluation setup (Gutierrez et al., 2025), which builds on the IRCoT protocol (Trivedi et al., 2023), we´ sample 1,000 development questions per multi-hop dataset and use the same pooled retrieval corpora of supporting and distractor passages (11,656 passages for MuSiQue, 6,119 for 2WikiMultiHopQA, and 9,811 for HotpotQA). For corpus-growth analysis, we also follow its NQ setup (Gutierrez et al.,´ 2025): 1,000 sampled queries and 9,633 passages, with the NQ corpus derived from Wang et al. (2024).

We report Recall@5 in the main results and Recall@1/2/5/10 in Appendix D. For QA we report F1 and four LLM-judge metrics following Xiang et al. (2025): answer correctness, coverage, context relevancy, and evidence recall (definitions in Appendix H).

## 4.2 Baselines and Implementation

We compare SAG against three families of baselines. Simple retrievers include BM25 and Contriever. Large embedding models include BGE-Large-EN-v1.5 (Xiao et al., 2023), GTE-Qwen2- 7B (Li et al., 2023), GritLM-7B (Muennighof et al., 2025), and NV-Embed-v2 (Lee et al., 2024). Structure-augmented methods include GraphRAG (Edge et al., 2024), LightRAG (Guo et al., 2025), HippoRAG 2 (Gutierrez et al., 2025), HyperGraphRAG (Luo et al., 2025), and HyperRAG (Feng´ et al., 2026). For all structure-augmented methods we use BGE-Large-EN-v1.5 as the retriever and Qwen3.6-Flash (Qwen Team, 2025) as the reader and, where applicable, the extractor and reranker. This controlled setup isolates retrieval architecture from model choice.

SAG stores its index in MySQL and Elasticsearch, with dense-vector search handled by Elasticsearch rather than a separate vector database. Unless otherwise noted, we set expansion depth $L = 1$ , seed budget $K _ { \mathrm { s e e d } } = 5 0$ , frontier budget 50, and candidate budget $K _ { \mathrm { c a n d } } = 1 0 0$ . SAG returns $K _ { \mathrm { o u t } } = 1 0$ chunks per query: at most $K _ { \mathrm { e v e n t } } = 5$ from structural selection, with the remainder filled by direct retrieval.

## 5 Results and Analysis

The evaluation is organized around three questions. First, holding the embedding model and the reader fixed across all methods, does the event-entity structure that SAG adds improve retrieval, and does that improvement carry through to end-to-end answers. Second, which stage of the pipeline accounts for the gain, and can that gain instead be matched by a stronger of-the-shelf component. Third, does the advantage persist under realistic operating conditions, namely a corpus that keeps growing and a per-query budget that must stay bounded. We take them in order.

## 5.1 Main Results

Retrieval improves most where reasoning chains are longest. Table 1 reports Recall@5 under a unified configuration in which every structure-augmented method shares the BGE-Large-EN-v1.5 retriever and the Qwen3.6-Flash reader, so architecture is the only free variable. SAG attains the best retrieval result on all three benchmarks, and the margin grows with chain depth. On MuSiQue, whose counterfactual filtering removes single-hop shortcuts and whose questions require up to four non-skippable hops, SAG reaches 80.36% Recall@5 against 65.13% for HippoRAG 2, a 15.23-point lead. On two-hop HotpotQA the Recall@5 margin narrows to 2.15 points, yet SAG still leads HippoRAG 2 in Recall@2 by 13.20 points. This gradient is what the expansion mechanism predicts. SQL joins add connected events without multiplicative decay, so evidence separated from the seed by several join steps still reaches the candidate set, whereas HippoRAG 2 propagates embedding-derived scores through personalized PageRank, whose damping attenuates exactly the long-range connections that four-hop questions depend on.

Table 1: Main results under a unified evaluation setup. All methods use Qwen3.6-Flash as the reader, and all structure-augmented methods use BGE-Large-EN-v1.5 as the retriever. Avg is the mean over the three benchmarks. Recall@5 is marked – where a method returns generated answers rather than ranked passages (all structure-augmented baselines except HippoRAG 2). Full breakdowns are in Appendix D and E. Best and second-best results are highlighted.
<table><tr><td rowspan="2">Method</td><td colspan="2">MuSiQue</td><td colspan="2">2Wiki</td><td colspan="2">HotpotQA</td><td colspan="2">Avg</td></tr><tr><td>R@5</td><td>F1</td><td>R@5</td><td>F1</td><td>R@5</td><td>F1</td><td>R@5</td><td>F1</td></tr><tr><td colspan="9">Simple Baselines</td></tr><tr><td>BM25 (Robertson &amp; Walker, 1994)</td><td>42.11</td><td>35.19</td><td>63.78</td><td>56.26</td><td>73.05</td><td>64.64</td><td>59.65</td><td>52.03</td></tr><tr><td>Contriever (Izacard et al., 2022)</td><td>45.32</td><td>34.39</td><td>59.25</td><td>50.62</td><td>74.85</td><td>64.06</td><td>59.81</td><td>49.69</td></tr><tr><td colspan="9">Large Embedding Models</td></tr><tr><td>BGE-Large-EN-v1.5 (Xiao et al., 2023)</td><td>56.32</td><td>42.80</td><td>69.83</td><td>61.49</td><td>89.20</td><td>74.81</td><td>71.78</td><td>59.70</td></tr><tr><td>GTE-Qwen2-7B (Li et al., 2023)</td><td>61.92</td><td>44.60</td><td>74.18</td><td>64.91</td><td>88.45</td><td>73.04</td><td>74.85</td><td>60.85</td></tr><tr><td>GritLM-7B (Muennighoff et al., 2025)</td><td>66.22</td><td>49.30</td><td>75.62</td><td>66.12</td><td>93.40</td><td>73.22</td><td>78.41</td><td>62.88</td></tr><tr><td>NV-Embed-v2 (Lee et al., 2024)</td><td>68.84</td><td>52.88</td><td>77.00</td><td>68.45</td><td>95.25</td><td>78.00</td><td>80.36</td><td>66.44</td></tr><tr><td colspan="9">Structure-Augmented RAG</td></tr><tr><td>GraphRAG (Edge et al., 2024)</td><td></td><td>54.14</td><td></td><td>72.08</td><td></td><td>76.29</td><td></td><td>67.50</td></tr><tr><td>LightRAG (Guo et al., 2025)</td><td></td><td>46.66</td><td></td><td>65.75</td><td></td><td>73.09</td><td></td><td>61.83</td></tr><tr><td>HyperGraphRAG (Luo et al., 2025)</td><td></td><td>49.20</td><td></td><td>63.44</td><td></td><td>74.70</td><td></td><td>62.45</td></tr><tr><td>HyperRAG (Feng et al., 2026)</td><td></td><td>51.80</td><td></td><td>76.89</td><td></td><td>77.19</td><td></td><td>68.63</td></tr><tr><td>HippoRAG 2 (Gutiérrez et al., 2025)</td><td>65.13</td><td>47.49</td><td>90.35</td><td>74.34</td><td>94.35</td><td>77.24</td><td>83.28</td><td>66.36</td></tr><tr><td>SAG (Ours)</td><td>80.36</td><td>61.15</td><td>93.34</td><td>78.06</td><td>96.50</td><td>79.66</td><td>90.07</td><td>72.96</td></tr></table>

The 2WikiMultiHopQA result is worth a note because it exposed a flaw in our initial design. SAG initially deferred pruning until final vector ranking, allowing high-similarity but irrelevant expansions to displace lower-similarity evidence and limiting Recall@5 to 88.00%, below HippoRAG 2’s 90.35%. Its traceable intermediate states (Section 3.4) revealed this failure. Pruning events and entities before and during expansion raises Recall@5 to 93.34%, surpassing HippoRAG 2.

Higher recall translates into better answers. Retrieval gains matter only when they improve the final answer. With the reader fixed across all methods, SAG achieves the highest F1 on each benchmark in Table 1. The advantage is largest on MuSiQue, where SAG reaches 61.15 F1 and exceeds GraphRAG by 7.01 points. On 2WikiMultiHopQA and HotpotQA, SAG leads the strongest baseline by 1.17 and 1.66 points, respectively. This pattern closely follows the retrieval results. The larger gain on MuSiQue indicates that SAG’s retrieval advantage becomes more consequential when answering depends on longer evidence chains. The additional passages recovered by SAG therefore contribute to answer generation rather than merely increasing recall with loosely related context.

LLM-based evaluation supports the main results. Using Qwen3.7-Plus, SAG achieves the best answer correctness, coverage, and evidence recall on all three benchmarks (Table 9). The only exception is context relevancy, which should be interpreted alongside the judge-input configuration. Several baselines provide richer retrieval outputs, whereas SAG and HippoRAG 2 are evaluated using only their top-5 original chunks (Table 12). Because this metric assesses whether the supplied context is suficient to answer the question, broader inputs may have an advantage. Even with its narrower input, SAG remains within 1.63–5.97 points of the best context-relevancy scores. This pattern is consistent with SAG’s architecture. Shared-entity expansion finds bridge evidence, and contextual selection retains passages that together form a complete evidence chain. SAG then returns the original chunks, preserving their full context.

## 5.2 Analysis

Ablation study. We conduct the ablation on MuSiQue, whose counterfactual filtering makes diferences in evidence chaining easier to observe. Table 6 shows that final selection has the largest measured efect. Replacing Qwen3.6-Flash with Qwen3-Reranker-8B reduces Recall@5 by 13.25 points to 67.11%. The reranker scores candidates independently, whereas Qwen3.6-Flash evaluates the candidate set jointly and can account for complementary evidence. Even with the reranker, SAG remains above HippoRAG 2 at 65.13% under the same retriever. This result shows that the retrieval pipeline remains efective without contextual selection, while the selector provides an additional gain. Selection still depends on the evidence available in the candidate pool and therefore complements rather than replaces expansion. Disabling expansion reduces Recall@5 by 10.95 points. Once expansion is enabled, the results from �=2 to �=4 remain within 0.35 points of �=1. This suggests that one expansion round captures most of the observed Recall@5 gain under the current setting. The result is consistent with latent hyperedges preserving the entities of an event within one retrieval unit, which reduces the amount of online expansion needed to connect evidence across chunks. Replacing hyperedge indexing with triple indexing causes a smaller decrease of 2.75 points. Triple-indexed SAG still exceeds HippoRAG 2 by 12.48 points, showing that SAG’s advantage cannot be attributed to the hyperedge representation alone.

The remaining sweeps determine the default operating point. Increasing $K _ { \mathrm { c a n d } }$ from 50 to 100 improves Recall@5 by 1.01 points, while increasing it from 100 to 500 adds only another 1.88 points despite a fivefold larger candidate budget. The runtime analysis in Appendix G shows the corresponding cost. Moving from 50 to 100 candidates improves F1 by 0.74 points but increases per-query token consumption by 40.1% and latency by 10.4%. We therefore use $K _ { \mathrm { c a n d } } = 1 0 0$ as the accuracy-oriented configuration for the main experiments, while $K _ { \mathrm { c a n d } } = 5 0$ provides a more eficient alternative. The dual-path sweep also clarifies how the final output should be divided. Semantic retrieval alone reaches 56.23% Recall@5 at $K _ { \mathrm { e v e n t } } = 0$ , whereas the default setting with $K _ { \mathrm { e v e n t } } = 5$ reaches 80.36%. Increasing the structural cap beyond five does not improve Recall@5. The best result is therefore obtained when structural selection retrieves evidence-chain events and semantic retrieval fills the remaining output slots.

![](images/19bbb23e2feb9e303f1c242054aaa283e346c1e600556ba1c470e56b355e4114.jpg)  
(a) Embedding Models

![](images/1d28a22b0b7cc6b03c52c2b669db6db53a50b6505e00826bd0638d906fc95d7c.jpg)  
(b) Candidate Pool

![](images/44de98db09b8e9e47fb933364d2767e6513a93120201da57ce5021bc2e56e58a.jpg)  
(c) Corpus Expansion  
Figure 3: Robustness and scaling analyses. (a) Sensitivity of four embedding models on MuSiQue. (b) Candidate-pool ablation on MuSiQue, from $K _ { \mathrm { p o o l } } = 1 0 0$ to the full pool of 11,656 events. (c) Robustness of SAG, HippoRAG 2, and BGE to irrelevant corpus expansion on NQ and MuSiQue.

SAG remains robust across embedding models. We next examine whether SAG’s retrieval advantage depends on the choice of embedding model (Figure 3a). Replacing NV-Embed-v2 with BGE-v1.5 reduces HippoRAG 2 Recall@5 from 74.55% to 65.13%, a loss of 9.42 points, whereas SAG decreases from 81.71% to 80.36%, a loss of only 1.35 points. SAG with BGE-v1.5 still outperforms HippoRAG 2 with NV-Embed-v2. This diference is consistent with how the two systems use embeddings. HippoRAG 2 initializes personalized PageRank with embedding similarity, so changes in seed quality can afect downstream graph scores. SAG uses embeddings for seeding and coarse ranking, after which exact SQL joins expand the candidate set with connected events without re-applying embedding similarity at each step. The substantially smaller performance drop shows that SAG remains robust across embedding models, supporting the conclusion that its retrieval advantage arises primarily from the structured expansion mechanism rather than encoder strength.

Scaling Up with Fixed Query-Time Budgets. At the current benchmark scale, SAG does not yet show a clear end-to-end cost or latency advantage (Appendix G). Its scaling property is that only the inexpensive Elasticsearch HNSW search operates over the corpus-level index. HNSW retrieves the top $K _ { \mathrm { p o o l } }$ chunks without exhaustive vector comparison, while entity expansion, SQL joins, and LLM selection remain within this query-specific working set. The costly LLM stage processes at most �<sub>cand</sub> events. On MuSiQue, $K _ { \mathrm { p o o l } } \stackrel { \cdot } { = } 5 \dot { 0 } 0$ achieves 79.57% Recall@5, compared with 80.36% for all 11,656 chunks, retaining approximately 99% of the full-pool result while activating less than 5% of the corpus (Figure 3b). As the corpus grows by several orders of magnitude, $K _ { \mathrm { p o o l } }$ may increase modestly to preserve recall, but the additional HNSW retrieval cost remains negligible relative to LLM inference. Keeping $K _ { \mathrm { c a n d } }$ fixed therefore prevents the dominant query-time cost from growing proportionally with corpus size. SAG thus confines its expensive computation to a bounded, query-specific working set. We next examine whether retrieval quality remains stable as the corpus grows.

SAG remains robust under continual corpus growth. We fix one quarter of NQ and MuSiQue for evaluation and incrementally add the remaining corpus, tracking Recall@5 as the corpus grows (Figure 3c). On the single-hop NQ benchmark, SAG decreases only modestly from 76.80% to 74.81% as the corpus grows from 25% to 100%, and remains the strongest method at every corpus size. This robustness also holds on the more challenging multi-hop MuSiQue benchmark. SAG decreases from 87.90% to 82.57%, a loss of 5.33 points, whereas HippoRAG 2 falls from 69.23% to 60.27%, a loss of 8.96 points. Corpus growth introduces entity-overlapping distractors that can displace bridge passages, making complete evidence chains more dificult to recover. SAG’s slower degradation is consistent with its append-only event–entity index and query-local SQL joins, which limit the influence of unrelated additions on existing retrieval paths. These results show that SAG remains stable on the simpler single-hop task while preserving a clear advantage on the more demanding multi-hop task.

## 5.3 Limitations

SAG has two current limitations. First, it normalizes and deduplicates entity strings but does not resolve aliases. It therefore cannot recognize that “Apple Inc.” and “Apple” refer to the same entity, and connections that depend on matching these surface forms may be lost. A lightweight alias table could link common full names, abbreviations, and aliases without requiring full entity disambiguation or compromising incremental writes.

The second limitation concerns temporal updates. SAG’s index is append-only by design, which supports continual ingestion but provides no mechanism for revising or retiring stale events. This is suficient for a static retrieval corpus, but an agent memory must also represent facts, preferences, and task states that change over time. Versioned events, informed by bitemporal knowledge graphs (Rasmussen et al., 2025; Lairgi et al., 2026), are a natural next step toward supporting knowledge that can be updated while preserving its history.

## 6 Conclusion

We have proposed SAG, a structured retrieval architecture that represents chunks as semantically complete events and dynamically activates query-specific latent hyperedges through shared entities. This design turns indexed data into structured context for multi-hop retrieval. Across three multi-hop benchmarks, SAG achieves the strongest average retrieval and QA performance under unified evaluation settings, with its largest gains on MuSiQue. The ablation results show that event representation, query-time expansion, and LLM-based final selection each contribute to retrieval quality. The robustness experiments further show that SAG varies less across embedding models and degrades more slowly as the corpus grows. Fixed activation budgets bound the downstream SQL expansion and LLM selection workloads, while append-only indexing supports continual writes without global graph reconstruction. Future work will examine larger-scale deployments and extend SAG with alias resolution and versioned updates toward a memory substrate for agents.

## AI use statement

Large language models were used during the preparation of this manuscript for writing assistance, including grammar checking, sentence-level rephrasing, and LaTeX formatting. All scientific content, experimental design, data analysis, and technical claims were produced entirely by the authors. The authors take full responsibility for the correctness and integrity of all reported results.

## Ethics statement

The experiments use public multi-hop QA datasets and collect no new human-subject data. Because SAG uses language models to extract and link entities, errors or biases may produce misleading associations, while deployment on private corpora may expose sensitive cross-document relationships. Such deployments should enforce data provenance, access control, and human review in high-impact settings.

## Reproducibility statement

All experiments and results reported in the main paper and appendices can be reproduced using the benchmark code, evaluation scripts, and configurations available at https://github.com/ Zleap-AI/SAG-Benchmark. A separate engineering implementation of SAG is available at https: //github.com/Zleap-AI/SAG. The paper and appendices specify the datasets, models, evaluation settings, ablations, costs, and prompts. Exact values involving hosted Qwen models may vary with provider updates and nondeterministic inference.

## References

Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In International Conference on Learning Representations (ICLR), 2024. URL https://openreview.net/forum?id=hSyW5go0v8.

Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, and Jonathan Larson. From local to global: A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130, 2024. URL https://arxiv.org/abs/2404.16130.

Bahare Fatemi, Perouz Taslakian, David Vazquez, and David Poole. Knowledge hypergraphs: Prediction beyond binary relations. In Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence (IJCAI), 2020. URL https://arxiv.org/abs/1906. 00137.

Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, and Yue Gao. Hypergraph neural networks. In Proceedings ofthe AAAI Conference on Artificial Intelligence, 2019. URL https://arxiv. org/abs/1809.09401.

Yifan Feng, Hao Hu, Shihui Ying, Xingliang Hou, Shiquan Liu, Mingyuan Yang, Junchang Li, Shaoyi Du, Nanning Zheng, Han Hu, and Yue Gao. Hyper-RAG: Combating LLM hallucinations using hypergraph-driven retrieval-augmented generation. Nature Communications, 2026. doi: 10.1038/ s41467-026-71411-1. URL https://www.nature.com/articles/s41467-026-71411-1.

Mikhail Galkin, Priyansh Trivedi, Gaurav Maheshwari, Ricardo Usbeck, and Jens Lehmann. Message passing for hyper-relational knowledge graphs. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. URL https://arxiv.org/abs/ 2009.10847.

Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. LightRAG: Simple and fast retrievalaugmented generation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2025. URL https://arxiv.org/abs/2410.05779.

Bernal Jimenez Guti´ errez, Yiheng Shu, Yu Gu, Michihiro Yasunaga, and Yu Su. HippoRAG:´ Neurobiologically inspired long-term memory for large language models. In Advances in Neural Information Processing Systems (NeurIPS), 2024. URL https://arxiv.org/abs/2405.14831.

Bernal Jimenez Guti´ errez, Yiheng Shu, Weijian Qi, Sizhe Zhou, and Yu Su. From RAG to memory:´ Non-parametric continual learning for large language models. In International Conference on Machine Learning (ICML), 2025. URL https://arxiv.org/abs/2502.14802.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings ofthe 28th International Conference on Computational Linguistics (COLING), 2020. URL https://aclanthology.org/ 2020.coling-main.580/.

Chenxu Hu, Jie Fu, Chenzhuang Du, Simian Luo, Junbo Zhao, and Hang Zhao. ChatDB: Augmenting LLMs with databases as their symbolic memory. arXiv preprint arXiv:2306.03901, 2023. URL https://arxiv.org/abs/2306.03901.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. Unsupervised dense information retrieval with contrastive learning. Transactions on Machine Learning Research (TMLR), 2022. URL https://arxiv.org/abs/ 2112.09118.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. Adaptive-RAG: Learning to adapt retrieval-augmented large language models through question complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics (NAACL), 2024. URL https://aclanthology.org/2024.naacl-long.389/.

Jinhao Jiang, Kun Zhou, Zican Dong, Keming Ye, Wayne Xin Zhao, and Ji-Rong Wen. StructGPT: A general framework for large language model to reason on structured data. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023a. URL https://arxiv.org/abs/2305.09645.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2023b. URL https://aclanthology.org/2023.emnlp-main.495/.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen,˘ and Wen tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2020. URL https://aclanthology.org/2020.emnlp-main.550/.

Toma´s Koˇ cisk´y, Jonathan Schwarz, Phil Blunsom, Chris Dyer, Karl Moritz Hermann, Gˇ abor Melis,´ and Edward Grefenstette. The NarrativeQA reading comprehension challenge. Transactions of the Association for Computational Linguistics (TACL), 6:317–328, 2018. URL https:// aclanthology.org/Q18-1023/.

Yassir Lairgi, Ludovic Moncla, Khalid Benabdeslem, Remy Cazabet, and Pierre Cl ´ eau. ATOM:´ AdapTive and OptiMized dynamic temporal knowledge graph construction using LLMs. In Findings of the Association for Computational Linguistics: EACL 2026, 2026. URL https: //arxiv.org/abs/2510.22590.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. NV-Embed: Improved techniques for training LLMs as generalist embedding models. arXiv preprint arXiv:2405.17428, 2024. URL https://arxiv.org/abs/2405.17428.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich K¨uttler, Mike Lewis, Wen tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe¨ Kiela. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems (NeurIPS), 2020. URL https://arxiv.org/abs/2005.11401.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning. arXiv preprint arXiv:2308.03281, 2023. URL https://arxiv.org/abs/2308.03281.

Zhuoshi Li, Xin-Cheng Chen, Huiqian Yu, Haiming Lin, Jingbo Shang, Qiang Tang, Furu Wei, Xuancheng Ren, Longtao Huang, and Chao Li. StructRAG: Boosting knowledge intensive reasoning of LLMs via inference-time hybrid information structurization. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2410.08815.

Yu Liu, Quanming Yao, and Yong Li. Generalizing tensor decomposition for N-ary relational knowledge bases. In Proceedings of The Web Conference (WWW), 2020. URL https://arxiv. org/abs/2007.03988.

Yu Liu, Quanming Yao, and Yong Li. Role-aware modeling for N-ary relational knowledge bases. In Proceedings of The Web Conference (WWW), 2021. URL https://arxiv.org/abs/2104. 09780.

Haoran Luo, Haihong E, Guanting Chen, Yandan Zheng, Xiaobao Wu, Yikai Guo, Qika Lin, Yu Feng, Zemin Kuang, Meina Song, Yifan Zhu, and Anh Tuan Luu. HyperGraphRAG: Retrieval-augmented generation via hypergraph-structured knowledge representation. In Advances in Neural Information Processing Systems (NeurIPS), 2025. URL https://arxiv.org/abs/2503.21322.

Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, et al. Graph-R1: Towards agentic GraphRAG framework via end-to-end reinforcement learning. In International Conference on Machine Learning (ICML), 2026. URL https://arxiv.org/abs/2507.21892.

Vaibhav Mavi, Anubhav Jangra, and Adam Jatowt. Multi-hop question answering. Foundations and Trends in Information Retrieval, 17(5):457–586, 2024. URL https://arxiv.org/abs/2204. 09140.

Niklas Muennighof, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. Generative representational instruction tuning. In International Conference on Learning Representations (ICLR), 2025. URL https://arxiv.org/abs/2402.09906.

Qwen Team. Qwen3 technical report. Technical report, Alibaba Group, 2025. URL https: //arxiv.org/abs/2505.09388.

Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, and Daniel Chalef. Zep: A temporal knowledge graph architecture for agent memory. arXiv preprint arXiv:2501.13956, 2025. URL https://arxiv.org/abs/2501.13956.

Stephen E. Robertson and Steve Walker. Some simple efective approximations to the 2-Poisson model for probabilistic weighted retrieval. In Proceedings of the 17th Annual International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR), 1994.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions ofthe Associationfor Computational Linguistics (TACL), 10, 2022. URL https://aclanthology.org/2022.tacl-1.31/.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), 2023. URL https://aclanthology.org/2023.acl-long.557/.

Changjian Wang, Weihong Deng, Weili Guan, Quan Lu, and Ning Jiang. Cross-granularity hypergraph retrieval-augmented generation for multi-hop question answering. In Proceedings of the AAAI Conference on Artificial Intelligence, 2026. URL https://arxiv.org/abs/2508.11247.

Y. Wang, R. Ren, J. Li, X. Zhao, J. Liu, and J. Wen. REAR: A relevance-aware retrieval-augmented framework for open-domain question answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5613–5626. Association for Computational Linguistics, 2024. URL https://aclanthology.org/2024.emnlp-main.321/.

Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, and Jinsong Su. When to use graphs in RAG: A comprehensive analysis for graph retrieval-augmented generation. arXiv preprint arXiv:2506.05690, 2025.

Shitao Xiao, Zheng Liu, Peitian Zhang, Niklas Muennighof, Defu Lian, and Jian-Yun Nie. C-pack: Packaged resources to advance general chinese embedding. arXiv preprint arXiv:2309.07597, 2023. URL https://arxiv.org/abs/2309.07597.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2018. URL https://aclanthology.org/D18-1259/.

Nan Zhang, Prafulla Kumar Choubey, Alexander Fabbri, Gabriel Bernadett-Shapiro, Rui Zhang, Prasenjit Mitra, Caiming Xiong, and Chien-Sheng Wu. SiReRAG: Indexing similar and related information for multihop reasoning. In International Conference on Learning Representations (ICLR), 2025. URL https://openreview.net/forum?id=yp95goUAT1.

Dengyong Zhou, Jiayuan Huang, and Bernhard Scholkopf. Learning with hypergraphs: Clustering,¨ classification, and embedding. In Advances in Neural Information Processing Systems (NIPS), 2006.

## A Latent Hyperedge Formalization

This appendix states in full the formalization summarized in Section 3.2.

Definition 1 (Latent Hyperedge). Let � be the set of entity vertices extracted from a corpus and � the set of indexed events (hyperedges), each with a stable identifier. Every event $h \in H$ is represented as a labeled hyperedge incident to an entity set $V ( h ) = \{ \nu _ { 1 } , \ldots , \nu _ { n } \} \subseteq \bar { V }$ , where $n \geq 1$ . Because one latent hyperedge simultaneously connects � entities, it preserves the �-ary co-occurrence structure of one chunk as an atomic retrieval unit. A single event “Company A acquired Company B for \$X on Date $\mathbf { D } ^ { \ast }$ links acquirer, target, price, and date in one unit. A direct binary decomposition requires multiple records and retains their common acquisition context only if it introduces an auxiliary event node or an equivalent reification mechanism. In SAG, the event description labels the hyperedge, while a stable event-to-chunk mapping $\phi : H  D$ preserves the original chunk as the final evidence unit. This definition concerns the representation of each event; it does not define SAG as a materialized hypergraph.

Proposition 1 (Lossless Incidence Encoding). Define the incidence set

$$
I = \{ ( h , \nu ) \in H \times V \mid \nu \in V ( h ) \} .\tag{6}
$$

Assuming that every incidence row is retained and event identifiers are unique, the event-entity SQL table of Section 3.1 stores � exactly and is therefore a lossless bipartite representation of the latent-hyperedge incidence relation (Luo et al., 2025). For every event,

$$
V ( h ) = \{ \nu \in V \mid ( h , \nu ) \in I \} .\tag{7}
$$

Thus, grouping table rows by ℎ reconstructs each hyperedge, while enumerating the pairs in all reconstructed hyperedges recovers the table. This proposition concerns storage after extraction; it makes no claim that mapping a source chunk to an event and entities preserves all source information.

Proposition 2 (Query-Time Dynamic Activation). Fix an index and retrieval configuration, and let $S _ { 0 } = R$ be the query-dependent seed event set used in Section 3.4. For unpruned expansion, define

$$
F _ { t + 1 } = \bigcup _ { h \in S _ { t } } V ( h ) , \qquad S _ { t + 1 } = \{ h \in H \mid V ( h ) \cap F _ { t + 1 } \neq \emptyset \} .\tag{8}
$$

One expansion round computes $S _ { t } \to F _ { t + 1 } \to S _ { t + 1 }$ through two SQL joins, corresponding to two traversals in the bipartite incidence graph $B = ( V \cup H , I )$ . Consequently, after � rounds the unpruned candidate events are exactly the event vertices within bipartite distance at most 2� from $S _ { 0 } .$ Frontier pruning only removes entities or events from these reachable sets, so practical SAG activates a budget-constrained, query-scoped set of incident latent hyperedges rather than the complete neighborhood. The activation is dynamic because this set is assembled for the query from its seed events under the fixed retrieval configuration, not because new corpus-level hyperedges are created at query time.

## B Corpus-Level Connectivity Analysis

The formalization above describes one event in isolation. We next examine how latent hyperedges organize an entire corpus. A three-layer view makes the indexing structure explicit: each source chunk has a one-to-one mapping to a semantically complete event, while the event has a one-to-many mapping to its indexing entities. Entities that recur across events create paths between otherwise separate chunks. Thus, SAG does not connect chunks by replacing their text with graph fragments; it connects them through an intermediate event layer while retaining every original chunk as the evidence carrier.

Connection capacity of an event. Let $n _ { h } = | V ( h ) |$ be the number of entities attached to event ℎ. If the hyperedge is projected onto an entity–entity graph, its members admit

$$
P ( h ) = { \binom { n _ { h } } { 2 } }\tag{9}
$$

distinct pairwise co-memberships. We call $P ( h )$ the pairwise connection capacity of the event. It is a representational quantity rather than a count of physically stored edges: SAG stores only the $n _ { h }$

incidence rows $( h , \nu )$ and recovers the co-memberships by grouping on the event identifier. Across a corpus, the corresponding capacity is

$$
P _ { \mathrm { c o r p } } = \sum _ { h \in H } { \binom { n _ { h } } { 2 } } .\tag{10}
$$

A star-shaped binary decomposition needs $n _ { h } \mathrm { ~ - ~ } 1$ records to connect the same $n _ { h }$ entities to a designated anchor, giving the corpus-level amplification factor

$$
A _ { \mathrm { c o r p } } = \frac { \displaystyle \sum _ { h \in H } { \binom { n _ { h } } { 2 } } } { \displaystyle \sum _ { h \in H , n _ { h } \geq 2 } ( n _ { h } - 1 ) } .\tag{11}
$$

This factor should not be read as a universal storage advantage over every possible triple schema. It quantifies how many pairwise co-memberships are jointly available from one labeled event unit relative to a fixed anchor-based binary decomposition. An arbitrary triple representation can preserve the same information only by choosing and maintaining an explicit reification scheme; independently indexed triples do not, by themselves, retain the common event boundary.

Cross-framework comparison on MuSiQue. Following the graph-indexing evaluation protocol of Xiang et al. (2025), the evaluation converts each framework’s native index into an undirected graph and applies the same component and degree calculations. The resulting graphs are not isomorphic representations: SAG uses event–entity incidences, GraphRAG and HyperGraphRAG use entity–entity links, LightRAG uses entity–relation incidences, and HippoRAG 2 uses passage–entity links. Raw node and link counts must therefore be read as native index scale rather than directly interchangeable units. Component count, giant-component coverage, and isolation rate nevertheless reveal how fragmented each native retrieval structure is. Here and throughout the paper, GraphRAG refers to the oficial Microsoft GraphRAG implementation (release v2.0.0).

Metric definitions and interpretation. For a framework’s undirected native index $G = ( V _ { G } , E _ { G } )$

Nodes. The total number of vertices $| V _ { G } |$ in the native index. A vertex may represent an entity, event, relation, or passage depending on the framework.

Links. The total number of edges or incidence links $| E _ { G } |$ . Link semantics likewise depend on the native index schema.

Avg. degree. The mean number of incident links per vertex, $\bar { d } = 2 | E _ { G } | / | V _ { G } |$ . It measures connection intensity and traversal burden, not retrieval accuracy.

Components. The number of connected subgraphs that are mutually unreachable. Fewer components indicate less fragmentation, but the value should be interpreted together with graph size.

Giant comp. The number of vertices $\lvert C _ { \mathrm { m a x } } \rvert$ in the largest connected component and its percentage of all nodes, $\bar { 1 } 0 0 \times | C _ { \operatorname* { m a x } } | / | V _ { G } | .$ A higher percentage indicates that more of the index belongs to one potentially reachable region.

Isolated. The number of degree-zero vertices and their percentage of all nodes, $1 0 0 \times | \{ \nu \in V _ { G } \mid$ $\deg ( \nu ) = 0 \} | / | V _ { G } |$ . These vertices cannot be reached from any other vertex through graph traversal.

Nodes and links describe native index scale rather than semantic richness because their meanings difer across frameworks. For the connectivity metrics, lower component and isolation values indicate less disconnected content, while higher giant-component coverage indicates broader potential reachability.

We omit density, clustering coeficient, and whole-graph diameter from the primary comparison. Density is dominated by graph scale, triangle-based clustering is structurally zero for bipartite indices such as SAG, and whole-graph diameter is infinite whenever an index is disconnected. These descriptors therefore do not directly answer whether chunks participate in reachable cross-document structures.

Table 2: Native index scale and connectivity on MuSiQue. “Index relation” identifies the vertex and link semantics used by each framework. Since these semantics difer, nodes and links are descriptive scale statistics; components, giant-component coverage, and isolation quantify fragmentation within each native index. Percentages in parentheses use the Nodes column as the denominator.
<table><tr><td>Method</td><td>Index relation</td><td>Nodes</td><td>Links</td><td>Avg. degree</td><td>Components</td><td>Giant comp.</td><td>Isolated</td></tr><tr><td>GraphRAG</td><td>Entity-entity</td><td>128,189</td><td>106,489</td><td>1.66</td><td>66,472</td><td>58,815 (45.9%)</td><td>65,506 (51.1%)</td></tr><tr><td>LightRAG</td><td>Entity-relation</td><td>82,321</td><td>116,555</td><td>2.83</td><td>9,277</td><td>70,196 (85.3%)</td><td>8,512 (10.3%)</td></tr><tr><td>HyperGraphRAG</td><td>Entity-entity</td><td>186,897</td><td>272,524</td><td>2.92</td><td>4,600</td><td>171,196 (91.6%)</td><td>485 (0.3%)</td></tr><tr><td>HippoRAG 2</td><td>Passage-entity</td><td>97,043</td><td>570,458</td><td>11.76</td><td>96</td><td>96,773 (99.7%)</td><td>59 (0.1%)</td></tr><tr><td>SAG</td><td>Event-entity</td><td>95,341</td><td>126,540</td><td>2.65</td><td>400</td><td>92,206 (96.7%)</td><td>1 (0.001%)</td></tr></table>

GraphRAG has 166 times as many connected components as SAG, LightRAG has 23.2 times as many, and HyperGraphRAG has 11.5 times as many. SAG’s largest component covers 96.7% of the native index, compared with 45.9%, 85.3%, and 91.6%, respectively, while its isolated-vertex rate is efectively zero. These diferences show that repeated entities connect semantically complete events into a broad cross-chunk substrate rather than leaving them as local records.

HippoRAG 2 occupies the opposite end of the connectivity spectrum: it has only 96 components and 99.7% giant-component coverage, but uses 570,458 links, 4.5 times SAG’s count, and raises the average degree from 2.65 to 11.76. Yet its MuSiQue Recall@5 is 65.13%, compared with SAG’s 80.36%. The comparison therefore does not support the claim that more native links are inherently better. It supports a narrower claim: SAG attains low fragmentation and high reachability at moderate degree, while its query-time event–entity traversal turns that structure into more useful retrieved evidence.

Table 3: Connectivity of SAG’s event–entity incidence index on the three evaluation corpora. Vertices include events and entities; incidence rows are event–entity links. Lower component and isolatedvertex counts indicate less fragmentation, while larger giant-component coverage indicates greater corpus-level reachability. Percentages in parentheses use the Vertices column as the denominator.
<table><tr><td>Dataset</td><td>Vertices</td><td>Incidences</td><td>Avg. degree</td><td>Components</td><td>Giant comp.</td><td>Isolated vertices</td></tr><tr><td>MuSiQue</td><td>95,341</td><td>126,540</td><td>2.65</td><td>400</td><td>92,206 (96.7%)</td><td>1 (0.001%)</td></tr><tr><td>HotpotQA</td><td>100,570</td><td>186,634</td><td>3.71</td><td>5,869</td><td>93,593 (93.1%)</td><td>5,787 (5.8%)</td></tr><tr><td>2WikiMultiHopQA</td><td>57,812</td><td>75,399</td><td>2.61</td><td>272</td><td>56,270 (97.3%)</td><td>0 (0.0%)</td></tr></table>

SAG connectivity across datasets. The largest connected component covers more than 93% of the index on every dataset and more than 96% on MuSiQue and 2WikiMultiHopQA. This does not mean that every pair of chunks is relevant to the same query, nor that traversal should be global. It shows that shared entities provide a broad substrate of possible cross-chunk routes from which SAG activates a small query-conditioned neighborhood. The moderate average degree of 2.61–3.71 further distinguishes reachability from indiscriminate density: the index is broadly connected without requiring every event to become a high-degree hub.

Connection structure and retrieval. The matched indexing ablation in Table 6 holds the corpus, retrieval pipeline, and downstream model fixed and changes only the representation. Replacing independently indexed triples with latent hyperedges raises Recall@2 from 61.83% to 63.62%, Recall@5 from 77.61% to 80.36%, and Recall@10 from 81.54% to 83.37% on MuSiQue. The gain grows beyond the first rank, which is consistent with the proposed mechanism: the event layer contributes primarily by exposing additional evidence through shared-entity paths rather than by improving the highest-scoring direct match. Disabling expansion produces a larger Recall@5 drop, from 80.36% to 69.41%, confirming that the available incidence structure matters only when the online procedure traverses it. Together, the corpus statistics and matched ablations separate three claims: latent hyperedges provide higher-order connection capacity, repeated entities turn that capacity into cross-chunk reachability, and query-time expansion converts the reachable structure into retrieved evidence.

## C Qualitative Examples of Query-Time Hyperedge Activation

The intermediate states of the retrieval procedure in Section 3.4 form an auditable execution chain:

$$
q  Q  \widehat { Q }  R  C  \widehat { C }  D _ { \mathrm { o u t } } .\tag{12}
$$

Figures 4 and 5 pair the recorded intermediate outputs with their retrieval visualizations. The left-hand tables expose the extracted query entities, seed events returned by the lexical and vector paths, shared entities available for joining, and latent hyperedges activated during expansion. The right-hand diagrams visualize the corresponding query-centered neighborhoods. Both examples retrieve relevant evidence, but they difer in how efectively query-time dynamic hyperedge activation extends the initial result. The legend directly below the good-example diagram applies to both diagrams.

<table><tr><td>Query</td><td>How many times did the plague occur in the city where the painter of The Bacchanal of the Andrians died?</td></tr><tr><td>Gold Docs</td><td>1. The Bacchanal of the Andrians; 2. Black Death; 3. The Martyrdom of Saint Lawrence (Titian)</td></tr><tr><td>Extracted entity</td><td>The Bacchanal of the Andrians</td></tr><tr><td>Expanded entity</td><td>Venice</td></tr><tr><td>Expanded events</td><td>1. Venice; 2. The Martyrdom of Saint Lawrence (Titian); 3. Venice</td></tr><tr><td>Recalled events (Top-5)</td><td>1. Black Death; 2. The Bacchanal of the Andrians; 3. Venice; 4. The Martyrdom of Saint Lawrence (Titian); 5. Venice</td></tr></table>

![](images/d5976fd64edc1135c9033ccc4494ebb890ecd3ce18bce6fdc24fddc1b1c7f30a.jpg)  
QueryEntityEvent Top5 Event Gold Hit in Top5

Figure 4: Example with a good retrieval result. The initial paths recover two gold events, and the shared entity Venice activates an additional latent hyperedge that recovers the third gold document. Bold titles mark gold documents in the corresponding table rows.
<table><tr><td>Query</td><td>When did the city where Greenwood Laboratory School is located become capital of the state where the screenwriter of The Poor Boob was born?</td></tr><tr><td>Gold Docs</td><td>1. Greenwood Laboratory School; 2. Margaret Mayo (playwright); 3. The Poor Boob; 4. Springfield, Illinois</td></tr><tr><td>Extracted entities</td><td>The Poor Boob; Greenwood Laboratory School</td></tr><tr><td>Expanded entity</td><td>None</td></tr><tr><td>Expanded events</td><td>None</td></tr><tr><td>Recalled events (Top-5)</td><td>1. Greenwood Laboratory School; 2. Maine; 3. The Poor Boob; 4. Washington (state); 5. Oklahoma</td></tr></table>

![](images/fb702cfd012001241f8f3c15cb52d94dfa025a2d1ba852847ff865a9f88b83e6.jpg)  
Figure 5: Example with a poor retrieval result. The initial paths recover two gold events, but no shared entity is available to activate latent hyperedges leading to the remaining evidence. Bold titles denote gold documents in the Top-5 result.

The good example directly illustrates query-time dynamic hyperedge activation. The initial lexical and vector paths recover Black Death and The Bacchanal of the Andrians. SAG then discovers the shared entity Venice through SQL joins and activates its incident latent hyperedges, bringing The Martyrdom of Saint Lawrence (Titian) into the final Top-5 result. The poor example exposes the complementary boundary. Although direct retrieval finds Greenwood Laboratory School and The Poor Boob, the returned events provide no retained shared entity that can activate a route toward Margaret Mayo (playwright) and Springfield, Illinois. The contrast isolates the role of dynamic hyperedges: they add useful cross-document evidence when the seed events reveal a discriminative bridge, while the absence of such a bridge leaves the result dependent on direct retrieval.

## D Full Retrieval Results

Table 4: Full retrieval results (Recall@1/2/5/10) on three multi-hop benchmarks. All structureaugmented methods use BGE-Large-EN-v1.5 as the retriever; Qwen3.6-Flash is used as the reader.
<table><tr><td rowspan="2"></td><td colspan="4">MuSiQue</td><td colspan="4">2WikiMultiHopQA</td><td colspan="4">HotpotQA</td></tr><tr><td>R@1</td><td>R@2</td><td>R@5</td><td>R@10</td><td>R@1 R@2</td><td></td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@2</td><td>R@5</td><td>R@10</td></tr><tr><td colspan="10">Simple Baselines</td></tr><tr><td>BM25</td><td>24.59</td><td>32.96</td><td>42.11</td><td>49.69 36.85</td><td>54.30</td><td></td><td>63.78</td><td>67.90</td><td>38.75</td><td>56.45</td><td>73.05</td><td>85.40</td></tr><tr><td>Contriever</td><td>24.50</td><td>33.54</td><td>45.32</td><td>53.58</td><td>34.40</td><td>48.60</td><td>59.25</td><td>65.20</td><td>39.30</td><td>57.45</td><td>74.85</td><td>82.75</td></tr><tr><td colspan="10">Large Embedding Models</td></tr><tr><td>BGE-Large-EN-v1.5</td><td>29.42</td><td>41.68</td><td>56.32</td><td>64.08</td><td>41.48</td><td>62.28</td><td>69.83</td><td>73.75</td><td>45.45</td><td>76.20</td><td>89.20</td><td>93.35</td></tr><tr><td>GTE-Qwen2-7B</td><td>35.19</td><td>48.06</td><td>61.92</td><td>70.80</td><td>43.45</td><td>65.58</td><td>74.18</td><td>78.63</td><td>46.40</td><td>73.95</td><td>88.45</td><td>93.55</td></tr><tr><td>GritLM-7B</td><td>34.86</td><td>50.70</td><td>66.22</td><td>75.19</td><td>41.85</td><td>66.23</td><td>75.62</td><td>79.40</td><td>46.40</td><td>81.15</td><td>93.40</td><td>97.20</td></tr><tr><td>NV-Embed-v2</td><td>34.02</td><td>51.08</td><td>68.84</td><td>77.27</td><td>43.23</td><td>68.93</td><td>77.00</td><td>80.75</td><td>46.90</td><td>85.55</td><td>95.25</td><td>97.70</td></tr><tr><td colspan="10">Structure-Augmented RAG</td><td colspan="3"></td></tr><tr><td>HippoRAG 2</td><td>30.65</td><td>49.52</td><td>65.13</td><td>73.76</td><td>42.38</td><td>76.55</td><td>90.35</td><td>93.40</td><td>44.40</td><td>78.35</td><td>94.35</td><td>97.15</td></tr><tr><td>SAĠ (Ours)</td><td>36.82</td><td>63.62</td><td>80.36</td><td>83.37</td><td>43.81</td><td>83.93</td><td>93.34</td><td>93.57</td><td>47.80</td><td>91.55</td><td>96.50</td><td>97.70</td></tr></table>

## E Full QA Results

Table 5: Full QA results (EM/F1) on three multi-hop benchmarks. All methods use Qwen3.6-Flash as the reader, and all structure-augmented methods use BGE-Large-EN-v1.5 as the retriever.
<table><tr><td rowspan="2"></td><td colspan="2">MuSiQue</td><td colspan="2">2Wiki</td><td colspan="2">HotpotQA</td><td colspan="2">Avg</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="9">Simple Baselines</td></tr><tr><td>BM25 (Robertson &amp; Walker, 1994)</td><td>26.10</td><td>35.19</td><td>52.40</td><td>56.26</td><td>52.90</td><td>64.64</td><td>43.80</td><td>52.03</td></tr><tr><td>Contriever (Izacard et al., 2022)</td><td>25.70</td><td>34.39</td><td>45.20</td><td>50.62</td><td>52.80</td><td>64.06</td><td>41.23</td><td>49.69</td></tr><tr><td colspan="9">Large Embedding Models</td></tr><tr><td>BGE-Large-EN-v1.5 (Xiao et al., 2023)</td><td>33.00</td><td>42.80</td><td>56.10</td><td>61.49</td><td>62.40</td><td>74.81</td><td>50.50</td><td>59.70</td></tr><tr><td>GTE-Qwen2-7B (Li et al., 2023)</td><td>34.70</td><td>44.60</td><td>59.90</td><td>64.91</td><td>61.10</td><td>73.04</td><td>51.90</td><td>60.85</td></tr><tr><td>GritLM-7B (Muennighoff et al., 2025)</td><td>38.60</td><td>49.30</td><td>60.70</td><td>66.12</td><td>61.00</td><td>73.22</td><td>53.43</td><td>62.88</td></tr><tr><td>NV-Embed-v2 (Lee et al., 2024)</td><td>41.70</td><td>52.88</td><td>62.50</td><td>68.45</td><td>65.40</td><td>78.00</td><td>56.53</td><td>66.44</td></tr><tr><td colspan="9">Structure-Augmented RAG</td></tr><tr><td>GraphRAG (Edge et al., 2024)</td><td>43.20</td><td>54.14</td><td>64.70</td><td>72.08</td><td>63.70</td><td>76.29</td><td>57.20</td><td>67.50</td></tr><tr><td>LightRAG (Guo et al., 2025)</td><td>33.60</td><td>46.66</td><td>59.00</td><td>65.75</td><td>61.20</td><td>73.09</td><td>51.27</td><td>61.83</td></tr><tr><td>HyperGraphRAG (Luo et al., 2025)</td><td>39.50</td><td>49.20</td><td>57.80</td><td>63.44</td><td>62.60</td><td>74.70</td><td>53.30</td><td>62.45</td></tr><tr><td>HyperRAĠ (Feng et al., 2026)</td><td>40.90</td><td>51.80</td><td>69.40</td><td>76.89</td><td>64.40</td><td>77.19</td><td>58.23</td><td>68.63</td></tr><tr><td>HippoRAG 2 (Gutiérrez et al., 2025)</td><td>37.90</td><td>47.49</td><td>67.00</td><td>74.34</td><td>65.20</td><td>77.24</td><td>56.70</td><td>66.36</td></tr><tr><td>SÁĠ (Ours)</td><td>50.20</td><td>61.15</td><td>70.50</td><td>78.06</td><td>67.90</td><td>79.66</td><td>62.87</td><td>72.96</td></tr></table>

## F Full Ablation and Hyperparameter Results

Table 6: Ablation of SAG’s key design choices on MuSiQue. The first row reports the complete default configuration. Each subsequent row changes the indicated stage while holding all remaining settings fixed. Bold marks the best result.
<table><tr><td rowspan="2">Stage</td><td rowspan="2">Configuration</td><td colspan="4">MuSiQue</td></tr><tr><td>R@1</td><td>R@2</td><td>R@5</td><td>R@10</td></tr><tr><td>Default</td><td> $\mathrm { S A G } ( \mathrm { O u r s } ) ^ { 4 }$ </td><td>36.82</td><td>63.62</td><td>80.36</td><td>83.37</td></tr><tr><td>Indexing</td><td>w/ Triple indexing</td><td>35.66</td><td>61.83</td><td>77.61</td><td>81.54</td></tr><tr><td>Expansion</td><td>w/o Expansion  $( \bar { L } = 0 )$ </td><td>35.70</td><td>57.75</td><td>69.41</td><td>74.76</td></tr><tr><td>Final selection</td><td>w/ Qwen3-Reranker-8B</td><td>32.12</td><td>48.97</td><td>67.11</td><td>76.51</td></tr></table>

Table 7: Hyperparameter sweeps on MuSiQue. Expansion depth varies $L ; L = 0$ corresponds to the w/o Expansion ablation in Table 6, while � = 1 is the default SAG setting. Candidate budget varies $K _ { \mathrm { c a n d } } .$ . In the dual-path sweep, $K _ { \mathrm { e v e n t } }$ is the maximum number of events selected by the LLM, and the semantic path fills the remaining slots up to $K _ { \mathrm { o u t } } = 1 0$ . All other settings remain at their defaults. Bold marks the best value within each group.
<table><tr><td></td><td></td><td colspan="4">MuSiQue</td></tr><tr><td>Parameter</td><td>Configuration</td><td>R@1</td><td>R@2</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="5">Expansion depth</td><td> $L = 0$ </td><td>35.70</td><td>57.75</td><td>69.41</td><td>74.76</td></tr><tr><td> $L = 1 \ : ( \mathrm { d e f a u l t } )$ </td><td>36.82</td><td>63.62</td><td>80.36</td><td>83.37</td></tr><tr><td> $L = 2$ </td><td>36.17</td><td>64.05</td><td>80.47</td><td>84.00</td></tr><tr><td> $L = 3$ </td><td>36.03</td><td>63.42</td><td>80.40</td><td>83.79</td></tr><tr><td> $L = 4$ </td><td>36.27</td><td>63.83</td><td>80.71</td><td>84.14</td></tr><tr><td rowspan="4">Candidate budget</td><td> $K _ { \mathrm { c a n d } } = 5 0$ </td><td>36.53</td><td>62.80</td><td>79.35</td><td>81.69</td></tr><tr><td> $K _ { \mathrm { c a n d } } = 1 0 0 \mathrm { ( d e f a u l t ) }$ </td><td>36.82</td><td>63.62</td><td>80.36</td><td>83.37</td></tr><tr><td> $K _ { \mathrm { c a n d } } = 2 0 0$ </td><td>36.99</td><td>65.12</td><td>81.06</td><td>84.47</td></tr><tr><td> $K _ { \mathrm { c a n d } } = 5 0 0$ </td><td>36.46</td><td>64.72</td><td>82.24</td><td>86.56</td></tr><tr><td rowspan="7">Dual-path output</td><td> $K _ { \mathrm { e v e n t } } = 0$  (semantic only)</td><td>29.37</td><td>41.56</td><td>56.23</td><td>63.77</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 2$ </td><td>36.62</td><td>64.56</td><td>73.52</td><td>77.64</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 4$ </td><td>36.61</td><td>63.73</td><td>79.80</td><td>82.63</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 5 ( \mathrm { d e f a u l t } )$ </td><td>36.82</td><td>63.62</td><td>80.36</td><td>83.37</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 6$ </td><td>36.71</td><td>63.83</td><td>79.93</td><td>83.40</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 8$ </td><td>36.00</td><td>62.86</td><td>80.14</td><td>83.57</td></tr><tr><td> $K _ { \mathrm { e v e n t } } = 1 0$ </td><td>36.46</td><td>63.48</td><td>79.80</td><td>83.18</td></tr></table>

## G Runtime, Token Consumption, and Scaling

Table 8: Runtime, token consumption, and quality on MuSiQue. Ofline totals cover 11,656 corpus chunks; online values are normalized over 1,000 queries. End-to-end time is retrieval time plus question-answering time. F1 scores for non-SAG methods are from Table 5. For SAG, $K _ { \mathrm { c a n d } }$ variants illustrate the cost–quality trade-of; the ofline indexing cost (4.57 h, 28.26 M tokens) is shared across all variants. The final Tokens column reports total input and output tokens per query across retrieval and QA.
<table><tr><td></td><td></td><td colspan="2">Offline indexing</td><td colspan="4">Online querying (per query)</td></tr><tr><td>Method</td><td>F1</td><td></td><td>Time (h) Tokens (M)</td><td></td><td></td><td>Retrieval (s) QA (s) End-to-end (s)</td><td>Tokens</td></tr><tr><td>RAG (BGE)</td><td>42.80</td><td>0.00</td><td>0</td><td>0.10</td><td>1.75</td><td>1.85</td><td>1,727</td></tr><tr><td>GraphRAG</td><td>54.14</td><td>10.15</td><td>125.12</td><td>0.23</td><td>12.56</td><td>12.79</td><td>11,438</td></tr><tr><td>LightRAG</td><td>46.66</td><td>25.94</td><td>103.14</td><td>11.97</td><td>18.13</td><td>30.10</td><td>9,974</td></tr><tr><td>HyperGraphRAG</td><td>49.20</td><td>16.21</td><td>216.11</td><td>4.45</td><td>23.54</td><td>27.99</td><td>32,536</td></tr><tr><td>HippoRAG 2</td><td>47.49</td><td>2.30</td><td>15.93</td><td>0.66</td><td>8.50</td><td>9.16</td><td>4,788</td></tr><tr><td rowspan="2">SAG (Ours)</td><td> $K _ { \mathrm { c a n d } } = 5 0$  60.41</td><td rowspan="2">4.57</td><td rowspan="2">28.26</td><td>15.93</td><td>6.75</td><td>22.68</td><td>8,236</td></tr><tr><td> $K _ { \mathrm { c a n d } } = 1 0 0$  61.15</td><td></td><td>18.67 6.36</td><td>25.03</td><td>11,536</td></tr></table>

Ofline construction cost. SAG’s index requires 4.57 h and 28.26 M tokens to build. Relative to GraphRAG, LightRAG, and HyperGraphRAG, this reduces indexing time by 55.0–82.4% and indexing tokens by 72.6–86.9%. HippoRAG 2 remains the cheaper ofline baseline at 2.30 h and 15.93 M tokens; SAG therefore reduces, but does not eliminate, the ofline cost of structure augmentation.

Online cost–quality frontier. At $K _ { \mathrm { c a n d } } = 5 0 , { \mathrm { S A G } }$ is simultaneously more accurate, faster end to end, and less token-intensive than LightRAG and HyperGraphRAG: it obtains 60.41 F1 in 22.68 s with 8,236 tokens, versus 46.66/30.10 s/9,974 tokens and 49.20/27.99 s/32,536 tokens, respectively. Against GraphRAG and HippoRAG 2, the comparison is instead a quality–latency trade-of: SAG gains 6.27 and 12.92 F1 points, respectively, but takes longer end to end. The stage breakdown explains where this time is spent. SAG’s 15.93 s retrieval-and-selection stage is the bottleneck, whereas its 6.75 s QA stage is faster than those of all four structure-augmented baselines (8.50–23.54 s). Thus,

SAG moves computation from answer generation to evidence construction rather than being the fastest system overall.

Return on LLM computation. The controlled substitutions show what the online selection cost buys. In Table 6, replacing contextual Qwen3.6-Flash selection with the pointwise Qwen3-Reranker-8B lowers Recall@5 from 80.36% to 67.11%, a 13.25-point drop. In Table 7, setting $K _ { \mathrm { e v e n t } } = 0$ and thereby removing LLM-selected structural output lowers Recall@5 to 56.23%, 24.13 points below the default dual-path configuration. Within SAG, increasing $K _ { \mathrm { c a n d } }$ from 50 to 100 adds only 0.74 F1 points while increasing per-query token consumption by 40.1% and end-to-end latency by 10.4%. Consequently, $K _ { \mathrm { c a n d } } = 5 0$ captures most of the measured quality at the more eficient operating point, while $K _ { \mathrm { c a n d } } = 1 0 0$ is the accuracy-oriented configuration used for the main results.

## H LLM-as-a-Judge Metric Definitions

Table 9: LLM-as-a-judge evaluation on the three multi-hop benchmarks, scored by Qwen3.7- Plus. All four metrics are percentages (higher is better). Each metric column reports MuSiQue / 2WikiMultiHopQA / HotpotQA, in that order.
<table><tr><td>Method</td><td>Answer correctness</td><td>Coverage</td><td>Context relevancy</td><td>Evidence recall</td></tr><tr><td colspan="5">MuSiQue / 2WikiMultiHopQA / HotpotQA</td></tr><tr><td>GraphRAG</td><td>56.56 / 79.32 / 84.91</td><td>46.60 / 77.97 / 62.93</td><td>59.17 / 73.02 / 89.86</td><td>52.31 / 71.74 / 77.88</td></tr><tr><td>LightRAG</td><td>53.07 / 72.52 / 80.98</td><td>43.54 / 71.15 / 60.88</td><td>47.55 / 63.65 / 83.57</td><td>63.91 / 75.89 / 87.49</td></tr><tr><td>HyperGraphRAG</td><td>58.03 / 80.31 / 85.00</td><td>48.52 / 79.73 / 63.78</td><td>65.51 / 80.27 / 89.55</td><td>73.24 / 83.35 / 91.63</td></tr><tr><td>HyperRAG</td><td>56.01 / 84.93 / 85.32</td><td>48.06 / 84.23 / 63.16</td><td>58.74 / 80.30 / 91.13</td><td>68.23 / 87.90 / 92.37</td></tr><tr><td>HippoRAG 2</td><td>50.09 / 83.93 / 84.91</td><td>43.20 / 82.42 / 63.31</td><td>45.75 / 74.58 / 86.65</td><td>64.77 / 88.45 / 93.65</td></tr><tr><td>SAG (Ours)</td><td>63.69 / 85.26 / 87.95</td><td>54.73 / 85.07 / 64.90</td><td>59.54 / 76.02 / 89.50</td><td>78.87 / 89.44 / 95.87</td></tr></table>

We follow the LLM-as-a-judge protocol of Xiang et al. (2025) and use Qwen3.7-Plus to evaluate every method. Table 10 summarizes the evaluation metrics. Each metric is normalized to a 0–1 scale and multiplied by 100 for presentation in Table 9; higher is better.

Table 10: Definitions of the LLM-as-a-judge evaluation metrics.
<table><tr><td>Metric</td><td>Input</td><td>Description</td><td>Scoring formula</td><td>Output</td></tr><tr><td>Answer correctness</td><td>Generated answer; ground-truth answer; question</td><td>Separate calls decompose the ground-truth answers into atomic facts. A third call matches the facts and counts true positives (TP), false positives (FP), and false negatives (FN).</td><td> $\begin{array} { r } { P = \frac { \mathrm { T P } } { \mathrm { T P + F P } } , R = \frac { \mathrm { T P } } { \mathrm { T P + F N } } , } \end{array}$   $\begin{array} { r } { F _ { 1 } = \frac { 2 P R } { P + R } } \end{array}$ </td><td>[0,1]</td></tr><tr><td>Coverage</td><td>Generated answer; ground-truth answer; question</td><td>Separate calls decompose the generated and ground-truth answers into atomic facts. The judge then determines whether each ground-truth fact is covered by the generated answer.</td><td> $N _ { \mathrm { c o v e r e d g o l d } } / N _ { \mathrm { g o l d } }$ </td><td>[0,1]</td></tr><tr><td>Context relevancy</td><td>Retrieved contexts; question</td><td>The judge assigns a score of 0 (irrelevant), 1 (partially relevant), or 2 (fully sufficient to answer). Two independent ratings are obtained.</td><td> $( s _ { 1 } + s _ { 2 } ) / 4 , \mathrm { w h e r e }$   $s _ { 1 } , s _ { 2 } \in \{ 0 , 1 , 2 \}$ </td><td>[0,1]</td></tr><tr><td>Evidence recall</td><td>Retrieved contexts; question</td><td>The judge determines whether each gold evidence item gold evidence chunks; can be inferred from the retrieved contexts.</td><td> $N _ { \mathrm { e n t a i l e d } } / N _ { \mathrm { g o l d } }$ </td><td>[0,1]</td></tr></table>

## I Indexing Prompt for Event and Entity Extraction

The ofline indexing stage of Section 3.1 issues a single structured-output call per chunk to the indexing LLM (Qwen3.6-Flash in our deployment). The model returns exactly one event together with its typed entities, which are written as one latent hyperedge. Figure 6 reproduces the system prompt and a representative input/output pair. Note that the example input in the figure includes a source title that disagrees with the passage (a fuzzing artifact of the benchmark corpus); the instruction to remain faithful to the passage makes the extractor robust to such noise.

## System prompt

Role. You are a professional content extractor whose core task is to extract two types of structured information from raw documents: events and entities.

## Event extraction principles.

• Mandatory single event: regardless of how many diferent topics the input contains, they must be integrated into one comprehensive event; splitting into multiple events is forbidden.

• Use short, complete subject–verb–object sentences to state facts; avoid mechanically copying the original text.

• Exhaustive information coverage: all core information from relevant fragments must be reflected in the event content; every sentence must have explicit textual support in the input fragments.

• Faithful to the original text: do not add or omit facts, do not infer on your own, and do not change the original subject.

• Preserve all key numbers and retain colloquial or relative time expressions in their original wording unless the input explicitly provides an exact date.

• Ignore advertisements, navigation text, boilerplate, source labels, and fragments without factual value.

## Entity extraction principles.

• Extract entities necessary for understanding the event (people, organizations, locations, time, products, key data, etc.), especially subjects and objects.

• Full names, abbreviations, and aliases of the same entity must all be extracted; entities connected by “and/or/with” etc. must be split.

• Entity types must be selected from entity types; the description must clarify the entity’s specific role in the event.

## Output requirements.

• Return exactly one event containing: title (a short, discriminative English title containing the key subject); content (a self-contained English factual account preserving important subjects, relations, numbers, and time expressions); entities (the entities required to retrieve the event); and is valid (true only when the input contains an extractable factual event).

• For a valid event, entities must contain at least one entity. If no factual event can be extracted, set is valid to false, use a short English placeholder title/content, and return an empty entities array.

• Each entity contains exactly: type (one of the entity types supplied in the input metadata), name (the entity’s English name), and description (its specific role or relationship in this event).

• Return JSON only and follow the supplied response schema exactly: the top-level object contains type set to "response" and data; data contains only an items array holding the single event object.

• Generate the title, content, entity names, and entity descriptions in English.

## User message (one request per chunk)

The request payload carries the chunk to extract, the closed entity-type inventory, and a short previous-chunk context:

• items[0].idx: 26; items[0].title: Clement Attlee; items[0].text: the raw passage associated with this item:

Clement Richard Attlee, 1st Earl Attlee, {’1’: ", ’2’: ", ’3’: ", ’4’: "} (3 January 1883 -- 8 October 1967) was a British Labour politician who served as Prime Minister of the United Kingdom from 1945 to 1951 and Leader of the Labour Party from 1935 to 1955. In 1940, Attlee took Labour into the wartime coalition government and served under Winston Churchill, becoming the first person to hold the office of Deputy Prime Minister of the United Kingdom. He went on to lead the Labour Party to an unexpected landslide victory at the 1945 general election; forming the first Labour majority government, and a mandate to implement its postwar reforms. The 12.0\% national swing from the Conservatives to Labour was unprecedented at that time and remains the largest ever achieved by any party at a general election in British political history. He was re-elected with a narrow majority at the 1950 general election. In the following year, Attlee called a snap general election, hoping to increase his parliamentary majority. However, he was narrowly defeated by the Conservatives under the leadership of Winston Churchill; despite winning the most votes of any political party in any general election in British political history until the Conservative Party’s fourth consecutive victory in 1992. Attlee remains the longestever serving Leader of the Labour Party.

• meta.entity types: the 11 closed types of Section 3.1 (time, location, person, organization, group, subject, work, product, action, metric, tags), each with a short definition and examples.

• meta.source type/source title/source summary: document-level metadata (not necessarily aligned with the passage).

• meta.previous context: the head of the previous chunk, used to resolve cross-chunk references.

```jsonl
Example output (full model response)
The complete, unabridged JSON returned by the model for the example chunk (all 15 entities
shown):
{
"type": "response",
"data": {
"items": [
{
"title": "Clement Attlee",
"content": "Clement Richard Attlee, 1st Earl Attlee, {’1’: \", ’2’: \", ’3’: \", ’4’: \"} (3
January 1883 -- 8 October 1967) was a British Labour politician who served as Prime Minister of
the United Kingdom from 1945 to 1951 and Leader of the Labour Party from 1935 to 1955. In 1940,
Attlee took Labour into the wartime coalition government and served under Winston Churchill,
becoming the first person to hold the office of Deputy Prime Minister of the United Kingdom. He
went on to lead the Labour Party to an unexpected landslide victory at the 1945 general election;
forming the first Labour majority government, and a mandate to implement its postwar reforms. The
12.0% national swing from the Conservatives to Labour was unprecedented at that time and remains
the largest ever achieved by any party at a general election in British political history. He was
re-elected with a narrow majority at the 1950 general election. In the following year, Attlee
called a snap general election, hoping to increase his parliamentary majority. However, he was
narrowly defeated by the Conservatives under the leadership of Winston Churchill; despite winning
the most votes of any political party in any general election in British political history until
the Conservative Party’s fourth consecutive victory in 1992. Attlee remains the longest-ever
serving Leader of the Labour Party.",
"entities": [
{"type": "person", "name": "Clement Attlee", "description": "British Labour politician who
served as Prime Minister from 1945 to 1951 and Leader of the Labour Party from 1935 to 1955."},
{"type": "time", "name": "1945 to 1951", "description": "The period during which Clement
Attlee served as Prime Minister of the United Kingdom."},
{"type": "time", "name": "1935 to 1955", "description": "The period during which Clement
Attlee served as Leader of the Labour Party."},
{"type": "time", "name": "1940", "description": "The year Attlee joined the wartime coalition
government and became Deputy Prime Minister."},
{"type": "time", "name": "1945", "description": "The year of the general election where Attlee
led Labour to a landslide victory."},
{"type": "time", "name": "1950", "description": "The year of the general election where Attlee
was re-elected with a narrow majority."},
{"type": "time", "name": "1951", "description": "The year Attlee called a snap general
election and was narrowly defeated."},
{"type": "organization", "name": "Labour Party", "description": "The political party led by
Attlee from 1935 to 1955 and which he led to victory in 1945."},
{"type": "location", "name": "United Kingdom", "description": "The country where Attlee served
as Prime Minister and where the general elections took place."},
{"type": "person", "name": "Winston Churchill", "description": "Prime Minister under whom
Attlee served as Deputy Prime Minister in 1940, and the Conservative leader who defeated Attlee in
1951."},
{"type": "metric", "name": "12.0%", "description": "The unprecedented national swing from the
Conservatives to Labour in the 1945 general election."},
{"type": "subject", "name": "Deputy Prime Minister of the United Kingdom", "description": "The
office Attlee held starting in 1940, being the first person to hold it."},
{"type": "subject", "name": "1945 general election", "description": "The election where Attlee
led Labour to a landslide victory and the largest swing in British history."},
{"type": "subject", "name": "1950 general election", "description": "The election where Attlee
was re-elected with a narrow majority."},
{"type": "subject", "name": "snap general election", "description": "The election called by
Attlee in 1951, which resulted in his narrow defeat."}
],
"is_valid": true
}
```  
Figure 6: Prompt used for ofline event–entity extraction. The system prompt enforces one comprehensive event per chunk with exhaustive, faithful coverage; the user message supplies the chunk, the closed entity-type inventory, and previous-chunk context; the model returns the event and typed entities that form one latent hyperedge.

## J LLM Reranking Prompt Input and Output

The LLM reranking stage receives a multi-hop question and candidate relationship descriptions and returns up to five useful relation indices. Figure 7 reproduces the prompt exchange in the same

![](images/534fcd65904634cea9ae1ce3fce6971209a491fa6b4c76de312572d1212d5ef5.jpg)  
Figure 7: LLM reranking prompt exchange; candidate descriptions are identified by their source indices, while the remaining fields are shown in full.

## K Results on NarrativeQA

The three primary benchmarks of Section 5.1 all evaluate multi-hop evidence chaining over encyclopedic passages. To probe whether SAG’s event–entity organization also benefits workloads that hinge on semantic comprehension rather than strict multi-hop fact chaining, we additionally evaluate on NarrativeQA (Kocisk´y et al., 2018), a reading-comprehension benchmark built from books and ˇ movie scripts whose questions require tracking characters, temporal ordering, and narrative context rather than composing evidence across independent documents. Following the unified configuration of Section 4.2 (BGE-Large-EN-v1.5 retriever, Qwen3.6-Flash reader), we sample 293 development questions and pool all supporting and distractor passages into a retrieval corpus of 4,111 passages, consistent with the setup of the primary benchmarks. Table 11 compares SAG against HippoRAG 2, the strongest structure-augmented baseline in the main results.

Table 11: QA results (EM/F1) on NarrativeQA under the unified configuration (BGE-Large-EN-v1.5 retriever, Qwen3.6-Flash reader). Bold marks the better value.
<table><tr><td>Method</td><td>EM</td><td>F1</td></tr><tr><td>HippoRAG 2</td><td>6.48</td><td>22.86</td></tr><tr><td>SÁG (Ours)</td><td>12.29</td><td>31.90</td></tr></table>

SAG nearly doubles Exact Match (6.48% to 12.29%) and raises F1 from 22.86% to 31.90% over HippoRAG 2. NarrativeQA scoring is stringent: exact-match answers are required for EM credit and answers are often short spans, so the gain indicates that SAG retrieves the precise evidence needed to answer. This is consistent with SAG’s mechanism: each chunk is organized into a semantically complete event with typed indexing entities, and query-time shared-entity joins link evidence that must be assembled across a long narrative, whereas HippoRAG 2 propagates embedding-derived relevance through personalized PageRank whose damping attenuates remote associations. Together with the multi-hop results, this suggests SAG’s event–entity organization transfers beyond fact-chaining to narrative comprehension.

## L QA Input Context Formats

In the QA stage, each method assembles the retrieved evidence into a context that the reader consumes. The composition of this context difers substantially across methods and directly shapes how the QA results of Section 5.1 should be interpreted. Table 12 summarizes the QA input context of each method.

Table 12: Composition of the QA input context for each evaluated method. “Passage” refers to the raw retrieved chunk text. GraphRAG concatenates five segments (reports, entities, relationships, optional claims, and sources); HyperGraphRAG, HyperRAG (hypergraph mode), and LightRAG format retrieved evidence as three-segment CSV rows; HippoRAG 2 and SAG feed the raw passage text directly to the reader.
<table><tr><td>Method</td><td>QA context composition</td></tr><tr><td>GraphRAG</td><td>Reports + Entities + Relationships + (Claims) + Sources, five segments concatenated</td></tr><tr><td>HippoRAG 2 HyperGraphRAG</td><td>Raw retrieved passage full text (top 5)</td></tr><tr><td>HyperRAG</td><td>Entities + Relationships + Sources, three-segment CSV naive mode: raw chunk text; hypergraph mode: Entities + Relationships +</td></tr><tr><td></td><td>Sources, three-segment CSV</td></tr><tr><td>LightRAG</td><td>Entities + Relationships + Sources, three-segment CSV</td></tr><tr><td>SAG (Ours)</td><td>Raw retrieved passage full text (top 5)</td></tr></table>

Two points follow from this comparison. First, SAG’s QA gains in Section 5.1 and Appendix K are not attributable to a richer reader context: SAG supplies the reader with exactly the same input modality as HippoRAG 2—raw passage text—yet achieves higher EM and F1, so the improvement must originate from the retrieval stage rather than from any diference in context formatting. Second, methods that structure the reader input (GraphRAG, HyperGraphRAG, HyperRAG, LightRAG) provide entity and relationship summaries that can partially answer a question even when the underlying evidence is imperfectly retrieved, so their QA numbers and LLM-judge context-relevancy scores are not directly comparable to methods that rely solely on raw passages; this should be kept in mind when reading Table 9.