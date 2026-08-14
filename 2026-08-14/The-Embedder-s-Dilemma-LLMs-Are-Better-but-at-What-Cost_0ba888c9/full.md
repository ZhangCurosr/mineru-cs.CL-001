# The Embedder's Dilemma: LLMs Are Better, but at What Cost?

Adnan El Assadi Harvard University

Niklas Muennighoff Stanford University

Jinhyuk Lee Independent Researcher

## Abstract

Should you replace your text-embedding pipeline with a large language model? We answer this with a controlled, cost-aware comparison of ten LLMs across six families and 26 embedding models (118M to 14B parameters) on 37 tasks spanning classification, semantic textual similarity (STS), clustering, pair classification, and retrieval. In aggregate the two paradigms are effectively tied: the best LLM (Gemini 3.1 Pro, 77.6) and the best embedding model (77.2) differ by 0.4 points. Their strengths differ by task: LLMs lead on reasoning-heavy retrieval, embedding models lead on classification, and the two match on clustering, STS, and pair classification. Reaching that parity is expensive. An LLM costs up to 1,431× more than an embedding model of comparable quality (\$154 vs. \$0.11 per benchmark pass), and the open LLMs tested process tokens 2.5 to 736× more slowly on the same GPU. Reasoning tokens account for 28 to 81% of LLM inference cost; lower reasoning budgets preserve or improve retrieval quality for most models in our ablation. The Pareto frontier contains the leading embedding models and one LLM, Gemini 3.1 Pro. These results support a division of labour: use embedding models for similarity, classification, and clustering, and reserve LLMs for reasoningintensive retrieval. Our code, datasets, and results are publicly available at https://github.com/embeddings-benchmark/embedders-dilemma.

![](images/3759787d520f6db3b1cd3104a2cdf45fdd188ef17858d94411c74419e8ff4934.jpg)  
Figure 1: Cost vs. performance across 36 models on MTEB(LLM). The frontier contains the leading embedding models and Gemini 3.1 Pro, which extends it by 0.4 points at 1,431 × the cost of a comparable embedding.

## 1 Introduction

We show that an LLM without task-specific training now matches the best text embedding models across a broad suite of standard embedding tasks. Embedding models reach this quality through specialised contrastive training, hard-negative mining, and multi-stage distillation; our best LLM reaches it without a dedicated embedding-training pipeline.1 The deployment costs differ sharply: LLM inference is substantially more expensive and slower on the same hardware. We call this the Embedder's Dilemma: LLMs now match the best embedding models in aggregate, but at much higher cost and lower throughput.

We ask a practical question: should you replace your embedding pipeline with an LLM? Our results favour embedding models as the default and LLMs for reasoning-heavy retrieval. A hybrid pipeline captures the retrieval benefit while controlling cost and throughput.

We compare ten frontier LLMs from six families with 26 text embedding models (118M–14B parameters). The evaluation uses MTEB(LLM), a 37-task benchmark built from the Massive Text Embedding Benchmark (MTEB; Muennighoff et al., 2023), covering classification, semantic textual similarity (STS), clustering, pair classification, and retrieval. The full model list appears in §3. Each MTEB(LLM) task is a fixed subset of the corresponding MTEB task, released on Hugging Face under mteb/11m-eval-\*; both paradigms are scored on exactly that subset. MTEB(LLM) scores compare models within this benchmark; MTEB leaderboard scores use the full test sets and follow a different evaluation basis. We pair the quality comparison with exact cost accounting (API token tracking for LLMs; GPU throughput benchmarking for embedding models), producing a cost-performance Pareto frontier spanning both paradigms (Figure 1).

Contributions. We release MTEB(LLM), the first benchmark to compare LLMs and embedding models across all five MTEB task categories, with cost measured alongside quality for every model. We implement it with the MEB framework, part of an ecosystem that spans multilingual text (MMTEB; Enevoldsen et al., 2025), image (MIEB; Xiao et al., 2025), audio (MAEB; El Assadi et al., 2026a), and video (MVEB; El Assadi et al., 2026b). The Pareto frontier contains the leading embeddings and Gemini 3.1 Pro, whose small score gain comes at three orders of magnitude higher cost. We identify the thinking-token tax: reasoning accounts for 28–81% of LLM inference cost, while lower reasoning budgets preserve retrieval quality for most models in our ablation (§4.6).

Our empirical findings are:

1. In aggregate, the paradigms tie. The best LLM and embedding model differ by 0.4 points, within statistical noise (§4.2). The pattern holds across the task suite.

2. The advantage is task-specific. LLMs lead on reasoning-heavy retrieval, embedding models lead on classification, and the paradigms are statistically even on clustering, STS, and pair classification. LLMs excel when cross-document reasoning is required; embedding pipelines excel wherever geometric similarity or labelled-reference matching suffices (§3.2–4.2).

3. The cost gap is large. Gemini 3.1 Pro costs 1,431× as much as a comparable embedding model, and the gap persists across hardware scenarios. Reasoning contributes substantially to this asymmetry. Lower reasoning budgets preserve or improve retrieval for most evaluated models (§4.4–4.6).

4. Throughput lags by orders of magnitude, even on the same hardware. Served on an identical H100, open-weight LLMs process 2.5–736× fewer tokens per second than embedding models, consistent with the architectural difference between autoregressive generation and a single encoder pass (§4.5; Figure 7).

LLMs and embedding models therefore serve complementary roles. Embedding pipelines remain the efficient default for classification, similarity, and clustering; LLMs are most useful for reasoning-intensive retrieval.

## 2 Related Work

## 2.1 Text Embedding Models, Evaluation, and Cost

Dense text representations have evolved rapidly from contextualised encoders (Devlin et al., 2019; Reimers & Gurevych, 2019) to billion-parameter, instruction-tuned models (Su et al., 2023). Leading systems such as E5 (Wang et al., 2022; 2024), NV-Embed (Lee et al. 2025a), SFR-2 (Meng et al., 2024), and Gemini Embedding (Lee et al., 2025b) achieve strong performance through contrastive training on large curated corpora. MTEB (Muennighoff et al., 2023) standardised evaluation across eight task categories and now tracks hundreds of models on a public leaderboard.2 The framework has expanded across multilingual text (MMTEB; Enevoldsen et al., 2025), image (MIEB; Xiao et al., 2025), audio (MAEB; El Assadi et al., 2026a), and video (MVEB; El Assadi et al., 2026b). HUME (Assadi et al., 2025) measures human performance on MTEB tasks and benchmarks LLMs as annotators. Our study asks whether an LLM can replace an embedding pipeline. We extend this line of evaluation with exact cost and throughput accounting across both paradigms.

Schwartz et al. (2020) introduced "Green AI," arguing that accuracy focus neglects computational and environmental costs. Gonzalez (2026) constructs Pareto frontiers of fine-tuned BERT-scale encoders vs. GPT-4o on text classification, finding encoders match or exceed frontier LLMs at 1–2 orders of magnitude lower cost. The “overthinking" literature (Sui et al., 2025) documents verbose reasoning traces and their computational cost. Snell et al. (2024) show that adaptive test-time compute can outperform a uniform reasoning budget We study these questions on classification, similarity, clustering, and retrieval workloads.

## 2.2 LLMs, Cross-Document Reasoning, and Retrieval

LLMs with controllable reasoning now span closed APIs (Gemini 3.1 Pro, GPT-5 (Singh et al., 2025), Claude Opus 4.6 (Anthropic, 2026)) and open weights (Qwen3 (Yang et āl., 2025), Llama 3 (Grattafiori et al., 2024), and DeepSeek-R1 (DeepSeek-AI et al., 2025), among others). We evaluate ten models across six families (§3), including dense and mixtureof-experts models from closed and open providers. This diversity lets us distinguish recurring patterns from effects specific to one provider. Bucher & Martini (2024) find that fine-tuned encoder models significantly outperform zero-shot frontier LLMs across all tested classification benchmarks, with margins ranging from 5 to over 75 F1 points depending on task granularity, consistent with our 5.6-point embedding advantage under the kNN vs. zero-shot comparison.

Dense retrieval (Karpukhin et al., 2020) and the BEIR benchmark (Thakur et al., 2021) established strong zero-shot bi-encoder performance. Bi-encoders process queries and documents independently, without cross-attention between them. The BRIGHT benchmark (Su et al., 2024) complements BEIR with reasoning-intensive queries. Its top embedding model scores 18.3 nDCG@10, while LLM-augmented retrieval improves by up to 12.2 points, showing the limitations of independent query and document encoding on such tasks. RAG (Lewis et al., 2020; Gao et al., 2024) pairs embedding retrieval with LLM reasoning. We evaluate this reranking stage directly (Sun et al., 2023a), crossing first-stage retrievers with cross-encoder and LLM listwise rerankers on BEIR and BRIGHT (§4.3). Lu et al. (2025) show CoT reranking consistently underperforms direct-output reranking on BEIR/BRIGHT despite higher cost, the document-ranking counterpart to our finding that reduced thinking holds or improves retrieval on all six tasks. Appendix F covers LLMs as text encoders, GritLM, and the wider set of frontier models.

## 3 Experimental Setup

## 3.1 Models

LLMs. We evaluate ten frontier LLMs spanning six families: three Gemini 3 models via the Gemini API (Gemini 3.1 Pro, Gemini 3 Flash, Gemini 3.1 Flash-Lite) and seven openweight models served via OpenRouter (DeepSeek-R1, DeepSeek-V4-Flash, Qwen3.6-27B, Qwen3.6-35B-A3B, GLM-4.7, Kimi-K2.6, and MiniMax-M2.7). Together they span dense and mixture-of-experts architectures, reasoning and instruct variants, and a wide capability and cost range. Gemini 3.1 Flash-Lite provides a non-reasoning baseline; the other models support extended reasoning (chain-of-thought tokens (Wei et al., 2022), billed at the output token rate).

Embedding models. We evaluate 26 text embedding models from 118M to 14B parameters. The multi-model families are multilingual-E5 (Wang et al., 2022; 2024), Qwen3-Embedding (Zhang et al., 2025), F2LLM-v2 (Zhang et al., 2026), Jina-v5 (Akram et al., 2026), and GTE-Qwen2 (Li et al., 2023). We also include models from Tencent (Zhao et al., 2025), NVIDIA (Babakhin et al., 2025), Salesforce (Meng et al., 2024), Snowflake (Yu et al., 2024), and Google (Vera et al., 2025), together with E5-Mistral (Wang et al., 2024), GritLM (Muennighoff et al., 2024), Linq-Embed-Mistral (Choi et al., 2024), BGÉ-M3 (Chen et al., 2025), and Octen-8B. All embedding models run locally on a single NVIDIA H100 80GB HBM3 GPU. Table 1 lists every model with its score and cost; full details appear in Appendix B.

## 3.2 Tasks and Evaluation Protocol

We introduce MTEB(LLM), a 37-task benchmark covering classification, STS, clustering, pair classification, and retrieval, implemented with the MTEB framework (Muennighoff et al., 2023). It follows MTEB's task and result interfaces, allowing new models and datasets to use the same evaluation pipeline. Each MTEB(LLM) task is a new LLM-specific evaluation dataset derived from the corresponding original MTEB task (fixed seed = 42; Appendix B.5), ensuring all submissions are evaluated on identical data.

Why these tasks. We use subsets rather than MTEB(eng, v2) because generative evaluation is orders of magnitude more expensive: one pass over the full suite would cost hundreds of dollars per LLM, and corpus-in-context retrieval requires the corpus to fit in the prompt. Tasks are selected to span all five categories, cover multiple languages and domains, and keep retrieval corpora small enough for shorter-context models. MTEB's task-selection analysis was also designed around embedding models, so v2 is not automatically the right frame for a cross-paradigm comparison.

We compare complete deployment pipelines: embedding models use kNN for classification, cosine similarity for STS and retrieval, and k-means for clustering; LLMs receive zero-shot prompts. This comparison reflects a common deployment setting in which practitioners consider LLMs because labelled data are scarce. Our few-shot ablation (§4.7) tests whether a small number of in-context examples narrows the resulting gap.

Classification (8 tasks). Sentiment (IMDB, ToxicConversations, TweetSentiment), intent detection (Banking77: 77 classes; MassiveIntent: 60 intents), and multilingual classification (AmazonCounterfactual, MTOPDomain, MassiveScenario). LLM: Zero-shot structured output for each test split sample. Embedding: kNN trained on embeddings on the labelled training split, then used for test split predictions. Metric: Accuracy.

Semantic Textual Similarity (10 tasks). English benchmarks (STSBenchmark, SICK-R, STS12–16), a biomedical benchmark (BIOSSES), and multilingual benchmarks (STS17, STS22v2). LLM: Rates similarity on each task's native scale, returning a float. Embedding: Cosine similarity between embeddings. Metric: Spearman correlation.

Clustering (9 tasks). Scientific papers (ArXiv, BioRxiv, MedRxiv) and online discussions (Reddit, StackExchange, TwentyNewsgroups). LLM: All documents in a single prompt with sequential identifiers; the model receives the ground-truth cluster count k and returns cluster assignments as a JSON array. Embedding: k-means on document embeddings. Metric: V-measure.

Pair Classification (4 tasks). Duplicate detection (SprintDuplicateQuestions), paraphrase identification (TwitterURLCorpus), legal matching (LegalBenchPC), and entailment (RTE3, multilingual). LLM: Binary output per pair with reasoning. Embedding: Cosine similarity with threshold sweep (MTEB convention; best threshold chosen post-hoc on the test set). Metric: Average precision and accuracy.

Retrieval (6 tasks). A domain-diverse suite: legal (AILAStatutes; LegalBench consumercontracts QA (Guha et al., 2023)), finance (HC3-Finance (Guo et al., 2023)), public-health QA, French QA (FQuAD (d'Hoffschmidt et al., 2020)), and Danish social media (TwitterHjerne (Vejlgaard Holm et al., 2025)). LLM: Corpus-in-context (Lee et al., 2024): the full corpus is placed in the prompt with sequential identifiers; prompt caching (Gim et al., 2024) amortises the corpus prefix across queries. Corpora are deliberately small (82–415 documents) so that models with shorter context windows can be evaluated on identical data; this keeps the whole corpus readable in one prompt, which production-scale retrieval does not permit (see §5). Embedding: Cosine similarity ranking. Metric: Recall@1 (nDCG@k results in released data files).

## 3.3 Cost Methodology

Embedding costs. We measure maximum-throughput token processing on an H100 80GB at sequence length 512 (largest fitting batch) and compute cost as rGPu/T, where $r _ { \mathrm { G P U } } =$ \$2.49/hr (Lambda Labs Hǐ00 spot, March 2026) and T is tokens per hour. Costs range from \$0.001 (mE5-small) to \$0.22 (F2LLM-v2-14B); full per-model throughput and cost data appear in Table 18 (Appendix H.2).

LLM costs. Token usage is extracted from each provider's usage\_stats. We bill every non-input (generated) token at the output rate:

$$
\mathrm { C o s t } = \mathrm { ( i n p u t - c a c h e d ) } \cdot r _ { \mathrm { i n } } + \mathrm { c a c h e d } \cdot r _ { \mathrm { c a c h e } } + \mathrm { ( t o t a l - i n p u t ) } \cdot r _ { \mathrm { o u t } }\tag{1}
$$

where $r _ { \mathrm { c a c h e } } = r _ { \mathrm { i n } } / 1 0$ . Reasoning (“thinking") tokens are part of the generated total and are billed at the output rate, matching provider billing; providers report them differently, and we attribute each model's reasoning share from its usage statistics (Appendix H.4). Gemini models are priced at Gemini API rates and open-weight models at OpenRouter public rates (March/June 2026). Costs range from \$3.16 (DeepSeek-V4-Flash) to \$154.14 (Gemini 3.1 Pro); the complete token-level breakdown is in Table 17 (Appendix H.1).

## 3.4 Throughput Methodology

We serve both paradigms on the same GPU, giving the throughput comparison common hardware and removing API rate limits. This differs from our cost accounting (§3.3), which uses API rates for LLMs and GPU-rental rates for embeddings. We serve the two open-weight LLMs that fit on a single H100 (Qwen3.6-27B dense, Qwen3.6-35B-A3B MoE) under vLLM, and run embedding models batched on the same GPU, measuring both in tokens per second (configuration in Appendix H.4). Gemini is API-only, and larger open models such as DeepSeek-R1 need multiple GPUs. This end-to-end comparison uses each paradigm's standard inference procedure: encoder batching for embeddings and autoregressive decoding for LLMs (Appendix G). Per-model embedding throughput is reported in Table 18 (Appendix H.2); the comparison appears in Figure 7 (Appendix D.1).

<table><tr><td>Model</td><td></td><td>Params</td><td>Cls (8)</td><td>Clust (9)</td><td>STS (10)</td><td>PairCls (4)</td><td>Retr (6)</td><td>Overall (37)</td><td>Cost</td></tr><tr><td rowspan="12">WT</td><td>Gemini 3.1 Pro</td><td></td><td>85.2</td><td>66.6</td><td>88.5</td><td>83.2</td><td>64.5</td><td>77.6</td><td>$154</td></tr><tr><td>Gemini 3 Flash</td><td></td><td>84.1</td><td>65.7</td><td>87.6</td><td>86.3</td><td>52.3</td><td>75.2</td><td>$56</td></tr><tr><td>Qwen3.6-27B</td><td></td><td>84.8</td><td>53.6</td><td>84.9</td><td>83.6</td><td>62.4</td><td>73.9</td><td>$103</td></tr><tr><td>Qwen3.6-35B-A3B</td><td></td><td>83.8</td><td>55.3</td><td>84.0</td><td>82.9</td><td>60.4</td><td>73.3</td><td>$34</td></tr><tr><td>Kimi-K2.6</td><td></td><td>84.5</td><td>55.4</td><td>83.9</td><td>78.0</td><td>56.7</td><td>71.7</td><td>$111</td></tr><tr><td>DeepSeek-V4-Flash</td><td></td><td>81.9</td><td>43.7</td><td>82.2</td><td>81.6</td><td>53.3</td><td>68.5</td><td>$3</td></tr><tr><td>MiniMax-M2.7</td><td></td><td>80.9</td><td>48.6</td><td>81.0</td><td>82.8</td><td>48.0</td><td>68.2</td><td>$25</td></tr><tr><td>GLM-4.7</td><td></td><td>84.2</td><td>36.5</td><td>83.5</td><td>79.7</td><td>54.4</td><td>67.7</td><td>$63</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>一</td><td>82.9</td><td>21.7</td><td>85.1</td><td>83.7</td><td>49.0</td><td>64.5</td><td>$7</td></tr><tr><td>DeepSeek-R1</td><td>一</td><td>82.5</td><td>31.3</td><td>83.7</td><td>79.5</td><td>44.7</td><td>64.3</td><td>$57</td></tr><tr><td>Octen-8B</td><td>7.6B</td><td>90.1</td><td>65.1</td><td>88.7</td><td>86.1</td><td>56.0</td><td>77.2</td><td>$0.11</td></tr><tr><td rowspan="10">Emmding</td><td>Qwen3-E-8B</td><td>7.6B</td><td>90.1</td><td>65.9</td><td>88.5</td><td>86.5</td><td>54.2</td><td>77.0</td><td>$0.11</td></tr><tr><td>Qwen3-E-4B</td><td>4.0B</td><td>89.3</td><td>64.6</td><td>88.8</td><td>86.5</td><td>50.3</td><td>75.9</td><td>$0.07</td></tr><tr><td>Nemotron-8B</td><td>7.5B</td><td>84.8</td><td>64.1</td><td>86.2</td><td>86.7</td><td>54.7</td><td>75.3</td><td></td></tr><tr><td>KaLM-12B</td><td>11.8B</td><td>88.8</td><td>63.4</td><td>84.6</td><td>87.1</td><td>49.9</td><td>74.8</td><td>$0.11</td></tr><tr><td>Jina-v5-S</td><td>596M</td><td>90.4</td><td>61.5</td><td>86.9</td><td>85.1</td><td>48.0</td><td>74.4</td><td>$0.16</td></tr><tr><td>SFR-2</td><td>7.1B</td><td>90.8</td><td>66.7</td><td>77.9</td><td>85.3</td><td>48.8</td><td>73.9</td><td>$0.03 $0.14</td></tr><tr><td>Jina-v5-Nano</td><td>212M</td><td>89.6</td><td>60.4</td><td>87.0</td><td>85.0</td><td>47.4</td><td>73.9</td><td>$0.01</td></tr><tr><td>F2LLM-14B</td><td>14.0B</td><td>77.7</td><td>66.5</td><td>84.1</td><td>86.3</td><td>52.1</td><td>73.3</td><td>$0.22</td></tr><tr><td>GTE-Qwen2-7B</td><td>7.1B</td><td>86.7</td><td>65.2</td><td>81.6</td><td>86.0</td><td>45.8</td><td>73.1</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>$0.09</td></tr></table>

Table 1: Model overview and per-category results. All ten LLMs and the ten highest-scoring embedding models. Scores are category means on a 0–100 scale; Overall is their mean. Cost is one MTEB(LLM) pass using API rates for LLMs and H100 throughput at \$2.49/hr for embeddings (§3.3). Bold = best shown; full results are in Appendix B.2.

## 4 Results

## 4.1 Overall Performance

Across all 36 models, the best LLM and the best embedding model are effectively tied. Gemini 3.1 Pro leads with a mean score of 77.6 on MTEB(LLM), followed closely by Octen-8B (77.2) and Qwen3-E-8B (77.0; full ranking in Appendix B.2). Gemini 3 Flash scores 75.2, and Flash-Lite scores 64.5. The reasoning-capable Gemini models occupy the top LLM ranks.

## 4.2 Task-Category Analysis

The aggregate tie hides large differences between task categories. Table 1 reports percategory scores for the top models, and full per-category rankings for all 36 models appear in Appendix D. Figure 6 (Appendix D.1) shows this at the task level: some embedding modeI matches the best LLM on 7 of 8 classification and 7 of 10 STS tasks, but on only 1 of 6 retrieval tasks. Retrieval is where the paradigms separate; elsewhere the choice is about cost, not quality.

Retrieval (+8.5): LLMs lead. Pro leads the best embedding on retrieval (64.5 vs. 56.0) and wins five of the six retrieval tasks, losing only legal statute retrieval (Appendix C.2).

Classification (—5.6): embeddings lead. SFR-2 (Meng et al., 2024) (90.8) outscores Pro (85.2) by a wide margin. The gap widens on fine-grained tasks (Banking77: 77 classes; MassiveIntent: 60 intents), a consequence of the kNN-vs-zero-shot setup we adopt for deployment realism (§3.2); see §5 for the structural argument.

Clustering, STS, and pair classification: statistical ties. On clustering the best embedding (SFR-2, 66.7) edges Pro (66.6); on STS Qwen3-E-4B (Zhang et al., 2025) (88.8) edges Pro (88.5); on pair classification KaLM-12B (Zhao et al., 2025) (87.1) leads Pro (83.2). All three rest on geometric proximity or labelled-reference matching, which cosine similarity over learned embeddings captures directly.

![](images/15ab458b5081bdc9440672613c3d779cb63f97dab4345507f72b49312fda72ad.jpg)  
Figure 2: Cost vs. performance by task category. Each panel plots score (0–100) against cost per benchmark pass (log scale) for all 3 models; the line is the Pareto frontier over all models and the star marks the best LLM in that category. Pro extends the retrieval frontier; embedding models define the frontiers for classification, clustering, STS, and pair classification.

Statistical significance. A paired bootstrap test (10,000 resamples) places the overall difference between Pro and Octen-8B within statistical noise $( \Delta = + \dot { 0 } . 3 , \dot { p } ^ { \underline { { { \cdot } } } } 0 . 8 5 , 9 5 \% \mathrm { C I } = [ - 2 . 4 ,$ +3.1]). Category-level tests favour LLMs on retrieval and embeddings on classification; clustering, STS, and pair classification are statistically even (Appendix A, Table 2).

The category-level cost-performance frontiers show the same division (Figure 2): Pro extends the retrieval frontier, while embeddings define the frontiers for classification, clustering, STS, and pair classification.

## 4.3 Retrieve-then-Rerank

The corpus-in-context comparison above pits single-vector embeddings against LLMs directly; production systems often insert a middle stage: a cross-encoder or LLM reranker over a first-stage shortlist. We evaluate this pipeline on the standard IR benchmarks BEIR (semantic) and BRIGHT (reasoning-heavy), crossing four first-stage retrievers with crossencoder and LLM listwise rerankers (Figure 3; the full matrix is in Appendix D.3, Table 16). The result follows the same task-dependent pattern: on reasoning-heavy BRIGHT, an LLM listwise reranker improves a strong embedding first stage from 22.3 to 35.1 nDCG@10. On semantic BEIR, the strong embedding alone scores 63.1, ahead of the best reranked configuration at 60.3. Cost scales with shortlist size: reranking a top-100 shortlist costs \$10–30 per benchmark, against \$154 for reading the full corpus in context. The MoE Qwen3.6- 35B-A3B is cheaper still, reaching 33.6 nDCG@10 on BRIGHT for \$10 where Qwen3.6-27B reaches 35.1 for \$30.

## 4.4 Cost and Throughput

Embeddings dominate the Pareto frontier. The Pareto frontier contains the embedding models and Gemini 3.1 Pro, which costs 1,431 × as much as the best embedding of comparable quality (Figure 1). Pro's marginal 0.4-point aggregate edge over Octen-8B comes with a benchmark cost of \$154.14 versus \$0.11.

The thinking-token tax. The structural driver of LLM cost is internal reasoning (Equation 1; Figure 4). Reasoning tokens account for 28–81% of inference cost across the reasoning models. Flash-Lite proviðes a non-reasoning reference point and ranks below most em-

BRIGHT (reasoning)

![](images/2a043691072d4bfdf8c5c72f52f888c3ec28714f41c11d62bf1e0c4c158277b3.jpg)  
BEIR (semantic)

![](images/782466bf9833a7442164895b886191d10bcf394bf23b5e6aae7c09e8921e444e.jpg)  
Figure 3: Retrieve-then-rerank. LLM listwise reranking improves every first stage on BRIGHT. On BEIR, a strong embedding first stage outperforms all reranked configurations.  
(a) what reasoning costs

![](images/43d7fddb613e8ce51778d33af15d80ec85153aa68698c10bbea0404f004af51f.jpg)

![](images/1e43a2aa5e6ed88d2d4fba2c396a1397f033b745a43b0886bb41955c564741c7.jpg)  
Figure 4: The thinking-token tax: what reasoning costs, and what it buys. (a) API cost per benchmark pass by token type for all ten LLMs. (b) Mean retrieval score with default vs. disabled reasoning for six models from five families; labels show the reduction in generated tokens. Four models preserve or improve retrieval with 54–96% fewer generated tokens; the two Qwen models lose ground.

beddings. This spending buys little on most tasks: lower reasoning budgets preserve or improve retrieval for four of six models (§4.6).

Cost sensitivity. The 1,431× ratio assumes H100 spot pricing (\$2.49/hr). Under alternative hardware and pricing the ratio ranges from 338× (commercial embedding API at \$0.10/MTok) to 2,424 (L4 ĠPU at \$0.49/hr). The order-of-magnitude gap persists across all evaluated cost assumptions (Appendix H.3)

## 4.5 Throughput Constraints

Served on the same H100, embedding models process tokens far faster than LLMs (Figure 7, Appendix D.1). The two open-weight LLMs that fit on a single H100 (Qwen3.6-27B, Qwen3.6- 35B-A3B) sustain 5,400–5,900 tokens/second, whereas embedding models range from 14,700 tok/s (the largest, F2LLM-14B) to 4.3M tok/s (the smallest, mE5-small), a 2.5× to 736× advantage on identical hardware. With hardware and serving stack fixed, the measured gap reflects the different inference procedures: autoregressive decoding and a single encoder pass. At production scale (millions of documents and continuous ingestion), this throughput gap limits the practicality of LLM-based pipelines.

## 4.6 Ablation: Reduced Thinking

We re-evaluate Gemini 3 Flash with reasoning\_effort=low and, for the open models, with reasoning disabled at the serving layer, cutting reasoning tokens by 54–96% on the retrieval tasks (Table 14; cross-family results in Figure 4b; details in Appendix C). Two findings stand out. First, reduced reasoning preserves or improves all six Flash retrieval scores and four of six cross-family averages. Second, classification changes by less than one point on all tested tasks $( | \Delta | < 1 . \dot { 0 } )$ . The retrieval advantage largely persists under reduced reasoning budgets, indicating a limited role for additional test-time reasoning.

## 4.7 Ablation: Few-Shot Classification

To test whether labelled examples can close the classification gap, we evaluate Flash with 5 in-context examples per task (Table 15; details in Appendix C). Five-shot results are similar to or worse than the zero-shot scores on the small-label tasks. Note that the embedding kNN classifier uses the full labelled train set. The LLM prompt only gets five examples here, yet given these results, it seems unlikely more would help.

## 5 Discussion

In aggregate, the two paradigms tie on maximum performance. The LLM advantage is narrow and specific: reasoning-heavy retrieval, where bi-encoders score systematically low (Su et al., 2024). Pro leads on 5/6 retrieval datasets, showing a consistent advantage across the suite. Capability profiles show this split: LLMs lead on retrieval while embeddings provide balanced, often superior, coverage across the other four categories (Appendix D.1, Figure 8).

Viewed through the lens of information retrieval, our results reproduce the classic bi-encoder versus cross-encoder tradeoff at LLM scale (Reimers & Gurevych, 2019; Muennighoff, 2022). One design choice separates the four architectures we evaluate (Figure 5): how many documents the model is allowed to read jointly with the query. Embedding pipelines encode queries and documents independently: each document is processed once, offline, and each query requires only vector comparisons. A cross-encoder reranker reads one document at a time, an LLM listwise reranker reads the top-k shortlist at once, and an LLM in corpus-in-context mode is the extreme case, reading the entire corpus in a single forward pass. Cost follows that ordering because query-specific computation repeats for each query: reranking scales with the shortlist k, and corpus-in-context processing scales with the corpus N. Quality follows the same ordering, which is why reranking is a more economical way to add reasoning than placing the full corpus in context (§4.3).

Our retrieve-then-rerank experiment (§4.3) shows the same pattern in a production-style setting: a reranker helps most on reasoning-heavy retrieval and little on semantic retrieval. The same-hardware results attribute much of the quality-cost gap to the underlying inference procedures.

The classification pipelines receive different supervision. Embedding classifiers use kNN over the full labelled training distribution; the LLM receives label names, task descriptions, and up to five examples. Fine-grained tasks make this difference especially visible (§4.7). Leading embedding models are also contrastively trained on data overlapping these domains and label spaces, so part of the gap reflects in-domain fit rather than a general representational advantage. The comparison is therefore between deployment pipelines as practitioners meet them, not intrinsic ceilings: a classification-post-trained LLM would likely narrow the gap, and none was available among the frontier models we evaluate.

The thinking-token tax reveals a mismatch between default reasoning budgets and these tasks. Reasoning improves the overall Flash score, while lower reasoning budgets preserve retrieval quality for most evaluated models. Where direct reading comprehension suffices, the model reconsiders straightforward relevance judgements, making the default reasoning budget wasteful for most of these benchmark tasks (Lu et al., 2025). For practitioners, reducing reasoning effort preserves retrieval quality for most evaluated models while removing the largest component of LLM cost.

Deployment implications. These findings motivate a hybrid architecture: embedding models provide high-throughput candidate retrieval, followed by LLM reasoning over a shortlist (Figure 7). This division of labour supports retrieve-then-reason pipelines (Lewis et al., 2020; Gao et al., 2024); our reranker results (§4.3) show where they help. Across the four non-retrieval categories, small-to-medium embedding models closely match the best LLM at a fraction of the cost. Hardware improvements will reduce the absolute cost of both paradigms. Their different inference procedures preserve an embedding throughput advantage for similarity, classification, and clustering workloads (§3.4). Accuracy-only leaderboards obscure large cost differences between similarly scored systems. We recommend reporting Pareto frontiers and significance tests alongside accuracy (Card et al., 2020; Dehghani et al., 2021).

![](images/a35278d059b9bfcd3c486117dba70baad33874e04b135ddb2d3a0c38cc7a4909.jpg)  
Figure 5: How many documents each architecture reads jointly with the query. Top: the pipeline. Bottom: its attention mask, drawn over the same N documents at the same scale in every panel so the four are directly comparable. Each row is a token and each column a token it may read. The red region is what one forward pass reads jointly with the query; it cannot be computed before the query arrives, so cost grows with it.

Limitations. Our ten LLMs from six families are a snapshot of a fast-moving frontier; because MTEB(LLM) uses the MTEB framework, future models can be evaluated with the same pipeline. The corpus-in-context protocol places corpora of 82–415 documents entirely in the prompt, which is only feasible at small scale: production corpora require an indexed first stage, so the cost gap we report is a lower bound. We complement this setting with a bi-encoder, cross-encoder reranker, and LLM listwise reranker comparison on BEIR and BRIGHT (§4.3). Embeddings also benefit from post-hoc threshold optimisation in pair classification (MTEB convention) with no LLM analogue. Appendix G gives the full set.

## 6 Conclusion

We present MTEB(LLM), the first cost-aware comparison of LLM and embedding deployment pipelines across five MTEB task categories, evaluating ten LLMs from six families and 26 embedding models on 37 tasks. The two paradigms match in aggregate but differ by task: LLMs lead on reasoning-heavy retrieval, embeddings lead on classification, and the rest are tied. Comparable LLM quality comes with substantially higher cost and lower throughput.

Embedding pipelines therefore remain the cost-efficient default, with reasoning-intensive retrieval the clearest case for an LLM. Our retrieve-then-rerank results support a hybrid strategy: embeddings for candidate retrieval, LLMs for reasoning over the shortlist.

Pareto frontiers add deployment cost and throughput to accuracy-based rankings, and we encourage their wider use. MTEB(LLM) is released with MTEB-compatible code and datasets, so the evaluation can expand as new models appear.

## Acknowledgments

We are extremely thankful to Laude Institute for supporting this work.

## Reproducibility Statement

Code, raw results, and analysis scripts are released at https://github.com/embeddingsbenchmark/embedders-dilemma in an implementation built on the MTEB framework. The LLM-specific datasets are hosted on Hugging Face under mteb/11m-eval-\*, and each task pins its dataset to an exact revision. LLM prompt templates, schema validation logic, and token-usage extraction scripts are included in the release, and every figure and table in the paper regenerates from the released result files. The provided scripts reproduce the embedding throughput benchmarks on an NVIDIA H100 GPU. Detailed experimental configurations are described in §3 and Appendix I.

## Ethics Statement

This work compares publicly available models on standard benchmarks and does not involve human subjects or private data. All evaluated datasets are publicly hosted and have been previously released for research purposes. Cost and throughput reporting may influence deployment decisions; we encourage practitioners to consider the environmental impact of LLM-based pipelines, including the carbon footprint of chain-of-thought inference, alongside the economic factors discussed in this paper. Reducing reasoning budgets can preserve performance on some tasks while lowering compute use. LLM assistance was used during the preparation of this paper for writing polish, phrasing refinement, and code formatting (e.g., matplotlib styling); no LLM was used to originate research ideas, generate evaluation data, produce plots, or evaluate model outputs. All scientific content, experimental design, and conclusions are the authors' own.

## References

Eneko Agirre, Daniel Cer, Mona Diab, and Aitor Gonzalez-Agirre. SemEval-2012 task 6: A pilot on semantic textual similarity. In Eneko Agirre, Johan Bos, Mona Diab, Suresh Manandhar, Yuval Marton, and Deniz Yuret (eds.), \*SEM 2012: The First Joint Conference on Lexical and Computational Semantics – Volume 1: Proceedings of the main conference and the shared task, and Volume 2: Proceedings of the Sixth International Workshop on Semantic Evaluation (SemEval 2012), pp. 385–393, Montréal, Canada, 7-8 June 2012. Association for Computational Linguistics. URL https://aclanthology.org/S12-1051/.

Eneko Agirre, Carmen Banea, Daniel Cer, Mona Diab, Aitor Gonzalez-Agirre, Rada Mihalcea, German Rigau, and Janyce Wiebe. SemEval-2016 task 1: Semantic textual similarity, monolingual and cross-lingual evaluation. In Steven Bethard, Marine Carpuat, Daniel Cer, David Jurgens, Preslav Nakov, and Torsten Zesch (eds.), Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pp. 497–511, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/S16-1081. URL https://aclanthology.org/S16-1081/.

Mohammad Kalim Akram, Saba Sturua, Nastia Havriushenko, Quentin Herreros, Michael Günther, Maximilian Werk, and Han Xiao. jina-embeddings-v5-text: Task-targeted embedding distillation,2026. URL https://arxiv.org/abs/2602.15547.

Anthropic. Claude Opus 4.6. Model card, Anthropic, 2026. URL https://www. anthropic. com/claude/opus.

Adnan El Assadi, Isaac Chung, Roman Solomatin, Niklas Muennighoff, and Kenneth Enevoldsen. Hume: Measuring the human-model performance gap in text embedding tasks,2025. URL https://arxiv.org/abs/2510.10062.

Yauhen Babakhin, Radek Osmulski, Ronay Ak, Gabriel Moreira, Mengyao Xu, Benedikt Schifferer, Bo Liu, and Even Oldridge. Llama-embed-nemotron-8b: A universal text embedding model for multilingual and cross-lingual tasks, 2025. URL https://arxiv. org/abs/2511.07025.

Parishad BehnamGhader, Vaibhav Adlakha, Marius Mosbach, Dzmitry Bahdanau, Nicolas Chapados, and Siva Reddy. LLM2Vec: Large language models are secretly powerful text encoders, 2024. URL https://arxiv.org/abs/2404.05961.

Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. Nuanced metrics for measuring unintended bias with real data for text classification, 2019. URL https://arxiv.org/abs/1903.04561.

Martin Juan José Bucher and Marco Martini. Fine-tuned 'small' LLMs (still) significantly outperform zero-shot generative AI models in text classification, 2024. URL https: //arxiv.org/abs/2406.08660.

Dallas Card, Peter Henderson, Urvashi Khandelwal, Robin Jia, Kyle Mahowald, and Dan Jurafsky. With little power comes great responsibility. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 9263–9274, 2020. URL https: //arxiv.org/abs/2010.06595.

Inigo Casanueva, Tadas Temčinas, Daniela Gerz, Matthew Henderson, and Ivan Vulić. Éfficient intent detection with dual sentence encoders. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI, 2020. URL https://arxiv.org/abs/ 2003.04807.

Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia. SemEval-2017 task 1: Semantic textual similarity multilingual and cross-lingual focused evaluation. In Proceedings of the 11th International Workshop on Semantic Evaluation (SemEval-2017), 2017. URL https://arxiv.org/abs/1708.00055.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3- embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation, 2025. URL https://arxiv.org/abs/2402.03216.

Xi Chen, Ali Zeynali, Chico Camargo, Fabian Flöck, Devin Gaffney, Przemyslaw Grabowicz, Scott A. Hale, David Jurgens, and Mattia Samory. SemEval-2022 task 8: Multilingual news article similarity. In Guy Emerson, Natalie Schluter, Gabriel Stanovsky, Ritesh Kumar, Alexis Palmer, Nathan Schneider, Siddharth Singh, and Shyam Ratan (eds.), Proceedings of the 16th International Workshop on Semantic Evaluation (ŠemEval-2022), pp. 1094–1106, Seattle, United States, July 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.semeval-1.155. URL https://aclanthology.org/2022.semeval-1.155/.

Chanyeol Choi, Junseong Kim, Seolhwa Lee, Jihoon Kwon, Sangmo Gu, Yejin Kim, Minkyung Cho, and Jy yong Sohn. Linq-embed-mistral technical report, 2024. URL https://arxiv.org/abs/2412.03223.

DeepSeek-AI, Daya Guo, Dejian Yang, et al. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Mostafa Dehghani, Yi Tay, Alexey A. Gritsenko, Zhe Zhao, Neil Houlsby, Fernando Diaz, Donald Metzler, and Oriol Vinyals. The benchmark lottery, 2021. URL https: //arxiv. org/abs/2107.07002.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics, pp. 4171–4186,2019. URL https://arxiv.org/abs/1810.04805.

Martin d'Hoffschmidt, Wacim Belblidia, Quentin Heinrich, Tom Brendlé, and Maxime Vidal. FQuAD: French question answering dataset. In Findings of the Association for Computational Linguistics: EMNLP 2020, pp. 1193–1208, 2020. URL https://arxiv.org/abs/2002.06071.

B. Efron. Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1): 1–26, 1979. ISSN 00905364, 21688966. URL http://www.jstor.org/stable/2958830.

Adnan El Assadi, Isaac Chung, Chenghao Xiao, Roman Solomatin, Animesh Jha, Rahul Chand, Silky Singh, Kaitlyn Wang, Ali Sartaz Khan, Marc Moussa Nasser, Sufen Fong, Pengfei He, Alan Xiao, Ayush Sunil Munot, Aditya Shrivastava, Artem Gazizov, Niklas Muennighoff, and Kenneth Enevoldsen. Maeb: Massive audio embedding benchmark, 2026a. URL https://arxiv.org/abs/2602.16008.

Adnan El Assadi, Roman Solomatin, Isaac Chung, Chenghao Xiao, Deep Shah, Manan Dey, Shriya Sudhakar, Zacharie Bugaud, Wissam Siblini, Åyush Sunil Munot, Yashwanth Devavarapu, Rakshitha Ireddi, Michelle Yang, Márton Kardos, Niklas Muennighoff, and Kenneth Enevoldsen. Mveb: Massive video embedding benchmark, 2026b. URL https://arxiv.org/abs/2606.14958.

Kenneth Enevoldsen, Isaac Chung, Imene Kerboua, Márton Kardos, Ashwin Mathur, David Stap, Jay Gala, Wissam Siblini, Dominik Krzeminski, Genta Indra Winata, Saba Sturua, Saiteja Utpala, Mathieu Ciancone, Marion Schaeffer, Gabriel Sequeira, Diganta Misra, Shreeya Dhakal, Jonathan Rystrøm, Roman Solomatin, Ömer Çağatan, Akash Kundu, Martin Bernstorff, Shitao Xiao, Akshita Sukhlecha, Bhavish Pahwa, Rafał Poświąta, Kranthi Kiran GV, Shawon Ashraf, Daniel Auras, Björn Plüster, Jan Philipp Harries, Loïc Magne, Isabelle Mohr, Mariya Hendriksen, Dawei Zhu, Hippolyte Gisserot-Boukhlef, Tom Aarsen, Jan Kostkan, Konrad Wojtasik, Taemin Lee, Marek Šuppa, Crystina Zhang, Roberta Rocca, Mohammed Hamdy, Andrianos Michail, John Yang, Manuel Faysse, Aleksei Vatolin, Nandan Thakur, Manan Dey, Dipam Vasani, Pranjal Chitale, Simone Tedeschi, Nguyen Tai, Artem Snegirev, Michael Günther, Mengzhou Xia, Weijia Shi, Xing Han Lù, Jordan Clive, Gayatri Krishnakumar, Anna Maksimova, Silvan Wehrli, Maria Tikhonova, Henil Panchal, Aleksandr Abramov, Malte Ostendorff, Zheng Liu, Simon Clematide, Lester James Miranda, Alena Fenogenova, Guangyu Song, Ruqiya Bin Safi, Wen-Ding Li, Alessia Borghini, Federico Cassano, Hongjin Su, Jimmy Lin, Howard Yen, Lasse Hansen, Sara Hooker, Chenghao Xiao, Vaibhav Adlakha, Orion Weller, Siva Reddy, and Niklas Muennighoff. Mmteb: Massive multilingual text embedding benchmark, 2025. URL https://arxiv.org/abs/2502.13595.

Jack FitzGerald, Christopher Hench, Charith Peris, Scott Mackie, Kay Rottmann, Ana Sanchez, Aaron Nash, Liam Urbach, Vishesh Kakarala, Richa Singh, Swetha Ranganath, Laurie Crist, Misha Britan, Wouter Leeuwis, Gokhan Tur, and Prem Natarajan. MASSIVE: A 1M-example multilingual natural language understanding dataset with 51 typologicallydiverse languages, 2022. URL https://arxiv.org/abs/2204.08582.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. Retrieval-augmented generation for large language models: A survey, 2024. URL https://arxiv.org/abs/2312.10997.

Gemma Team et al. Gemma 3 technical report, 2025. URL https://arxiv.org/abs/2503. 19786.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and Bill Dolan. The third PASCAL recognizing textual entailment challenge. In Satoshi Sekine, Kentaro Inui, Ido Dagan, Bill Dolan, Danilo Giampiccolo, and Bernardo Magnini (eds.), Proceedings of the ACL-PASCAL Workshop on Textual Entailment and Paraphrasing, pp. 1–9, Prague, June 2007. Association for Computational Linguistics. URL https://aclanthology.org/W07-1401/.

In Gim, Guojun Chen, Seung seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. In Proceedings of Machine Learning and Systems, volume 6, 2024. URL https://arxiv.org/abs/2311.04934.

Alberto Andres Valdes Gonzalez. Cost-aware model selection for text classification: Multiobjective trade-offs between fine-tuned encoders and LLM prompting in production, 2026. URL https://arxiv.org/abs/2602.06370.

Aaron Grattafiori et al. The Llama 3 herd of models, 2024. URL https://arxiv. org/abs/ 2407.21783.

Neel Guha, Julian Nyarko, Daniel E. Ho, Christopher Ré, Adam Chilton, Aditya Narayana, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel N. Rockmore, Diego Zambrano, Dmitry Talisman, Enam Hoque, Faiz Surani, Frank Fagan, Gavin Correction, Gregory Bassett, Haggai Porat, Jason Livermore, Jesse Noss, Jonathan Noss, Kevin Ashley, Kevin D. Ashley, Li-Cheng Lan, Marshall Tyler, Neel Guha, Nikhil Patel, Rosamund Thalken, Thomas F. Gordon, Tomas Skorepa, Wendy Xu, and Zhaoyi Zhou. LegalBench: A collaboratively built benchmark for measuring legal reasoning in large language models. In Advances in Neural Information Processing Systems, 2023. URL https://arxiv.org/abs/2308.11462.

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. How close is ChatGPT to human experts? comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597, 2023. URL https://arxiv. org/abs/2301.07597.

Ting Jiang, Shaohan Huang, Zhongzhi Luan, Deqing Wang, and Fuzhen Zhuang. Scaling sentence embeddings with large language models, 2023. URL https://arxiv. org/abs/ 2307.16645.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing, pp. 6769–6781,2020. URL https://arxiv.org/abs/2004.04906.

Omar Khattab and Matei Zaharia. ColBERT: Efficient and effective passage search via contextualized late interaction over BERT. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, pp. 39–48, 2020. URL https://arxiv.org/abs/2004.12832.

Wuwei Lan, Siyu Qiu, Hua He, and Wei Xu. A continuously growing dataset of sentential paraphrases. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), Proceedings of thē 2017 Conference on Empirical Methods in Natural Language Processing, pp. 1224–1234, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1126. URL https://aclanthology.org/D17-1126/.

Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. NV-Embed: Improved techniques for training LLMs as generalist embedding models. In The Thirteenth International Conference on Learning Representations, 2025a. URL https://arxiv.org/abs/2405.17428.

Jinhyuk Lee, Anthony Chen, Zhuyun Dai, Dheeru Dua, Devendra Singh Sachan, Michael Boratko, Yi Luan, Sébastien M. R. Arnold, Vincent Perot, Siddharth Dalmia, Hexiang Hu, Xudong Lin, Panupong Pasupat, Aida Amini, Jeremy R. Cole, Sebastian Riedel, Iftekhar Naim, Ming-Wei Ċhang, and Kelvin Guu. Can long-context language models subsume retrieval, RAG, SQL, and more?, 2024. URL https://arxiv.org/abs/2406.13121.

Jinhyuk Lee, Feiyang Chen, Sahil Dua, Daniel Cer, Madhuri Shanbhogue, Iftekhar Naim, Gustavo Hernández Ábrego, Zhe Li, Kaifeng Chen, Henrique Schechter Vera, Xiaoqi Ren, Shanfeng Zhang, Daniel Salz, Michael Boratko, Jay Han, Blair Chen, Shuo Huang, Vikram Rao, Paul Suganthan, Feng Han, Andreas Doumanoglou, Nithi Gupta, Fedor Moiseev, Cathy Yip, Aashi Jain, Simon Baumgartner, Shahrokh Shahi, Frank Palma Gomez, Sandeep Mariserla, Min Choi, Parashar Shah, Sonam Goenka, Ke Chen, Ye Xia, Koert Chen, Sai Meher Karthik Duddu, Yichang Chen, Trevor Walker, Wenlei Zhou, Rakesh Ghiya, Zach Gleicher, Karan Gill, Zhe Dong, Mojtaba Seyedhosseini, Yunhsuan Sung, Raphael Hoffmann, and Tom Duerig. Gemini embedding: Generalizable embeddings from Gemini, 2025b. URL https://arxiv.org/abs/2503.07891.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrievalaugmented generation for knowledge-intensive NLP tasks. In Advances in Neural Information Processing Systems, volume 33, pp. 9459–9474, 2020. URL https://arxiv.org/abs/ 2005.11401.

Haoran Li, Abhinav Arora, Shuohui Chen, Anchit Gupta, Sonal Gupta, and Yashar Mehdad. Mtop: A comprehensive multilingual task-oriented semantic parsing benchmark, 2021. URL https://arxiv.org/abs/2008.09335.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. Towards general text embeddings with multi-stage contrastive learning, 2023. URL https://arxiv.org/abs/2308.03281.

Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, and Michael Bendersky. Retrieval augmented generation or long-context LLMs? a comprehensive study and hybrid approach, 2024. URL https://arxiv.org/abs/2407.16833.

Xuan Lu et al. Rethinking reasoning in document ranking: Why chain-of-thought falls short, 2025. URL https://arxiv.org/abs/2510.08985.

Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning word vectors for sentiment analysis. In Dekang Lin, Yuji Matsumoto, and Rada Mihalcea (eds.), Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pp. 142-150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL https://aclanthology.org/P11-1015/.

Marco Marelli, Stefano Menini, Marco Baroni, Luisa Bentivogli, Raffaella Bernardi, and Roberto Zamparelli. A SICK cure for the evaluation of compositional distributional semantic models. In Nicoletta Calzolari, Khalid Choukri, Thierry Declerck, Hrafn Loftsson, Bente Maegaard, Joseph Mariani, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Ninth International Conference on Language Resources and Evaluation (LREC'14), pp. 216–223, Reykjavik, Iceland, May 2014. European Language Resources Association (ELRA). URL https://aclanthology.org/L14-1314/.

Rui Meng, Ye Liu, Shafiq Rayhan Joty, Caiming Xiong, Yingbo Zhou, and Semih Yavuz. Sfr-embedding-2: Advanced text embedding with multi-stage training, 2024. URL https: //huggingface.co/Salesforce/SFR-Embedding-2\_R.

Niklas Muennighoff. SGPT: GPT sentence embeddings for semantic search, 2022. URL https://arxiv.org/abs/2202.08904.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. MTEB: Massive text embedding benchmark. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pp. 2014–2037, 2023. URL https: //arxiv.org/ abs/2210.07316.

Niklas Muennighoff, Hongjin Su, Liang Wang, Nan Yang, Furu Wei, Tao Yu, Amanpreet Singh, and Douwe Kiela. Generative representational instruction tuning, 2024. URL https://arxiv.org/abs/2402.09906.

Zhijie Nie, Zhangchi Feng, Mingxin Li, Cunwang Zhang, Yanzhao Zhang, Dingkun Long, and Richong Zhang. When text embedding meets large language model: A comprehensive survey, 2025. URL https://arxiv.org/abs/2412.09165.

James O'Neill, Polina Rozenshtein, Ryuichi Kiryo, Motoko Kubota, and Danushka Bollegala. I wish i would have loved this one, but i didn't – a multilingual dataset for counterfactual detection in product reviews, 2021. URL https://arxiv.org/abs/2104.06893.

OpenAI. GPT-4 technical report, 2023. URL https://arxiv.org/abs/2303.08774.

OpenAI. OpenAI o1 system card, 2024. URL https://arxiv.org/abs/2412.16720.

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence embeddings using siamese BERT-networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, pp. 3982–3992, 2019. URL https://arxiv.org/abs/1908.10084.

Roy Schwartz, Jesse Dodge, Noah A. Smith, and Oren Etzioni. Green ai. Communications of the ACM, 63(12):54–63, 2020. URL https://arxiv.org/abs/1907.10597.

Darsh Shah, Tao Lei, Alessandro Moschitti, Salvatore Romeo, and Preslav Nakov. Adversarial domain adaptation for duplicate question detection. In Ellen Riloff, David Chiang, Julia Hockenmaier, ānd Jun'ichi Tsujii (eds.), Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 1056–1063, Brussels, Belgium, October-November 2018. Association for Computational Linguistics. doi: 10.18653/v1/D18-1131. URL https://aclanthology.org/D18-1131/.

Aaditya Singh et al. OpenAI GPT-5 system card, 2025. URL https: //arxiv. org/abs/2601. 03267.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters, 2024. URL https:// arxiv.org/abs/2408.03314.

Gizem Soğancioğlu, Hakime Öztürk, and Arzucan Özgür. BIOSSES: A semantic sentence similarity estimation system for the biomedical domain. Bioinformatics, 33(14):i49–i58, 2017.

Jacob Mitchell Springer, Suhas Kotha, Daniel Fried, Graham Neubig, and Aditi Raghunathan. Repetition improves language model embeddings. In The Thirteenth International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2402.15449.

Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu. One embedder, any task: Instructionfinetuned text embeddings. In Findings of the Association for Computational Linguistics: ACL 2023,2023. URL https://arxiv.org/abs/2212.09741.

Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han yu Wang, Haisu Liu, Quan Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan Arik, Danqi Chen, and Tao Yu. BRIGHT: A realistic and challenging benchmark for reasoning-intensive retrieval, 2024. URL https://arxiv.org/abs/2407.12883.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Na Zou, Hanjie Chen, and Xia Hu. Stop overthinking: A survey on efficient reasoning for large language models, 2025. URL https://arxiv.org/abs/2503.16419.

Weiwei Sun, Lingyong Yan, Xinyu Ma, Pengjie Ren, Dawei Yin, and Zhaochun Ren. Is ChatGPT good at search? investigating large language models as re-ranking agents, 2023a. URL https://arxiv.org/abs/2304.09542.

Xiaofei Sun, Xiaoya Li, Jiwei Li, Fei Wu, Shangwei Guo, Tianwei Zhang, and Guoyin Wang. Text classification via large language models, 2023b. URL https: //arxiv.org/abs/2305. 08377.

Chongyang Tao, Tao Shen, Shen Gao, Junshuo Zhang, Zhen Li, Kai Hua, Wenpen Hu, Zhangwei Tao, and Shuai Ma. Llms are also effective embedding models: An in-depth overview. ACM Transactions on Information Systems, 2024.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. BEIR: A heterogenous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track,2021. URL https://arxiv.org/abs/2104.08663.

Søren Vejlgaard Holm, Lars Kai Hansen, and Martin Carsten Nielsen. Danoliteracy of generative large language models, March 2025. URL https://aclanthology.org/2025. nodalida-1.78/.

Henrique Schechter Vera, Sahil Dua, Biao Zhang, Daniel Salz, Ryan Mullins, et al. Embeddinggemma: Powerful and lightweight text representations, 2025. URL https: //arxiv.org/abs/2509.20354.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training, 2022. URL https://arxiv.org/abs/2212.03533.

Liang Wang, Nan Yang, Xiaolong Huang, Linjun Yang, Rangan Majumder, and Furu Wei Improving text embeddings with large language models, 2024. URL https: //arxiv. org/ abs/2401.00368.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information rocessing Systems, volume 35, pp. 24824–24837,2022. URL https://arxiv.org/abs/2201.11903.

Orion Weller, Kathryn Ricci, Eugene Yang, Andrew Yates, Dawn Lawrie, and Benjamin Van Durme. Rank1: Test-time compute for reranking in information retrieval, 2025. URL https://arxiv.org/abs/2502.18418.

Chenghao Xiao, Isaac Chung, Imene Kerboua, Jamie Stirling, Xin Zhang, Márton Kardos, Roman Solomatin, Noura Al Moubayed, Kenneth Enevoldsen, and Niklas Muennighoff. Mieb: Massive image embedding benchmark, 2025. URL https://arxiv.org/abs/2504. 10471.

An Yang et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

Puxuan Yu, Luke Merrick, Gaurav Nuti, and Daniel Campos. Arctic-embed 2.0: Multilingual retrieval without compromise, 2024. URL https://arxiv.org/abs/2412.04506.

Yanzhao Zhang et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models, 2025. URL https://arxiv.org/abs/2506.05176.

Ziyin Zhang, Zihan Liao, Hang Yu, Peng Di, and Rui Wang. F2LLM-v2: Inclusive, performant, and efficient embeddings for a multilingual world, 2026. URL https: //arxiv.org/abs/2603.19223.

Xinping Zhao, Xinshuo Hu, Zifei Shan, Zetian Sun, Zhenyu Liu, Dongfang Li, Shaolin Ye, Xinyuan Wei, Qian Chen, Baotian Hu, Haofen Wang, Jun Yu, and Min Zhang. KaLM-Embedding-V2: Superior training techniques and data inspire a versatile embedding model, 2025. URL https://arxiv.org/abs/2506.20923.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging LLM-as-a-judge with MT-bench and chatbot arena. In Ādvances in Neural Information Processing Systems, volume 36,2023. URL https://arxiv.org/abs/2306.05685.

## A Statistical Significance Testing

Bootstrap procedure. We implement a paired bootstrap test (Efron, 1979) comparing Gemini 3.1 Pro against the best embedding model in each comparison. The resampling unit is the task level: each bootstrap replicate draws MTEB(LLM) tasks with replacement from the full MTEB(LLM) task set, and the score difference is computed as the mean-of-means difference across the resampled task set. Task-level resampling accommodates the different metrics used across categories (accuracy, Spearman ρ, V-measure, average precision, and Recall@1). Equal task weighting matches the macro-average score.

We report unadjusted p-values for five planned category-level comparisons. Under a Bonferroni threshold of α = 0.01, classification remains significant and retrieval falls above the threshold.

Table 2 reports the full results. The overall comparison (Pro vs. Octen-8B across MTEB(LLM)) yields $\Delta \bar { = } + 0 . 3 , 9 5 ^ { \circ } / \circ \mathrm { C I } = [ - 2 . 4 , + 3 . 1 ] , p = 0 . 8 5 ^ { }$ , a statistical tie. Category-level tests identify two significant differences:

• Classification: embeddings win $( \Delta = - 5 . 6 , \mathrm { C I } = [ - 9 . 2 , - 2 . 4 ] , p < 0 . 0 1 )$

$$
\bullet \mathrm { ~ \tiny ~ R e t r i e v a l : ~ L L M s ~ w i n ~ } ( \Delta = + 8 . 5 , \mathrm { C I } = [ + 0 . 2 , + 1 6 . 8 ] , \ : p < 0 . 0 5 )
$$

STS $( p = 0 . 7 4 )$ , clustering $( p = 0 . 9 7 ) .$ , and pair classification $( p = 0 . 4 9 )$ show no significant paradigm difference.

<table><tr><td>Comparison</td><td>∆</td><td>95% CI</td><td>p</td><td>Sig.</td></tr><tr><td>Overall (Gemini 3.1 Pro vs. Octen-8B, MTEB(LLM))</td><td></td><td></td><td></td><td></td></tr><tr><td>All tasks</td><td>+0.3</td><td>[−2.4, +3.1]</td><td>0.85</td><td>No</td></tr><tr><td colspan="5">Per category (Gemini 3.1 Pro vs. best embedding)</td></tr><tr><td>Retrieval</td><td>+8.5</td><td>[+0.2, +16.8]</td><td>&lt;0.05</td><td>Yes</td></tr><tr><td>Clustering</td><td>-0.2</td><td>[−5.6, +5.1]</td><td>0.96</td><td>No</td></tr><tr><td>STS</td><td>-0.3</td><td>[−2.2, +1.8]</td><td>0.75</td><td>No</td></tr><tr><td>Pair Classification</td><td>-3.9</td><td>[-13.6, +5.9]</td><td>0.50</td><td>No</td></tr><tr><td>Classification</td><td>-5.6</td><td>[−9.2, −2.4]</td><td>&lt;0.01</td><td>Yes</td></tr></table>

Table 2: Statistical significance. Paired bootstrap test (10,000 resamples, seed 42). $\Delta =$ Gemini 3.1 Pro – best embedding; significance at $\alpha = 0 . 0 5$ . Pair classification uses Pro for consistency, although Flash scores higher.

## B Evaluated Models and Tasks

This section lists the evaluated models, category- and task-level results, and benchmark tasks and metrics.

## B.1 Model Overview

Table 3 lists all 36 evaluated models — 10 LLMs and 26 embedding models — with parameter counts, mean MTEB(LLM) scores, total benchmark costs, and paper references. Models range from mE5-small (118M parameters, \$0.001) to Gemini 3.1 Pro (\$154.14). Models without a dedicated paper are marked “–".

## B.2 Full Per-Category Rankings

Table 4 compares Pro with the best embedding in each category. Table 5 gives per-category scores for al 36 models. Pro ranks first overall largely because of retrievaǐ; on classification, it ranks below ten embedding models. The reasoning-capable Gemini models rank far above the non-reasoning Flash-Lite, which places 31st of 36 overall.

## B.3 Per-Task Scores (Representative Models)

Table 6 reports individual task scores for 5 representative models (2 LLMs, 3 embeddings) across all MTEB(LLM) tasks, alongside the best embedding score per task. The scores vary substantially by task. For example, Pro is strongest on FQuAD retrieval, and SFR-2 is strongest on SprintDuplicateQuestions. This complementarity supports the hybrid-pipeline recommendation in §5.

## B.4 Full Per-Task Scores for All Models

Tables 7–11 provide the complete score matrix for all 36 models across every MTEB(LLM) task, presented one table per category so that each fits the page upright. The tables share a common layout: models are rows, ordered by overall MTEB(LLM) score, with the ten

<table><tr><td>Model</td><td>Params</td><td>Score</td><td>Cost</td><td>Reference</td></tr><tr><td>LLM Models</td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td></td><td>77.6</td><td>$154.14</td><td></td></tr><tr><td>Gemini 3 Flash</td><td></td><td>75.2</td><td>$55.87</td><td></td></tr><tr><td>Qwen3.6-27B</td><td></td><td>73.9</td><td>$103.43</td><td></td></tr><tr><td>Qwen3.6-35B-A3B</td><td></td><td>73.3</td><td>$34.06</td><td></td></tr><tr><td>Kimi-K2.6</td><td></td><td>71.7</td><td>$110.70</td><td></td></tr><tr><td>DeepSeek-V4-Flash</td><td></td><td>68.5</td><td>$3.16</td><td></td></tr><tr><td>MiniMax-M2.7</td><td></td><td>68.2</td><td>$24.52</td><td></td></tr><tr><td>GLM-4.7</td><td></td><td>67.7</td><td>$63.31</td><td></td></tr><tr><td>Gemini 3.1 Flash Lite</td><td></td><td>64.5</td><td>$6.85</td><td></td></tr><tr><td>DeepSeek-R1</td><td></td><td>64.3</td><td>$57.38</td><td></td></tr><tr><td>Embedding Models (ranked by score)</td><td></td><td></td><td></td><td></td></tr><tr><td>Octen-8B</td><td>7.6B</td><td>77.2</td><td>$0.108</td><td></td></tr><tr><td>Qwen3-E-8B</td><td>7.6B</td><td>77.0</td><td>$0.108</td><td>Zhang et al. 2025</td></tr><tr><td>Qwen3-E-4B</td><td>4.0B</td><td>75.9</td><td>$0.069</td><td>Zhang et al. 2025</td></tr><tr><td>Nemotron-8B</td><td>7.5B</td><td>75.3</td><td>$0.115</td><td>Lee et al. 2025a</td></tr><tr><td>KaLM-12B</td><td>11.8B</td><td>74.8</td><td>$0.158</td><td>Zhao et al. 2025</td></tr><tr><td>Jina-v5-S</td><td>596M</td><td>74.4</td><td>$0.034</td><td>Akram et al. 2026</td></tr><tr><td>SFR-2</td><td>7.1B</td><td>73.9</td><td>$0.136</td><td>Meng et al. 2024</td></tr><tr><td>Jina-v5-Nano</td><td>212M</td><td>73.9</td><td>$0.010</td><td>Akram et al. 2026</td></tr><tr><td>F2LLM-14B</td><td>14.0B</td><td>73.3</td><td>$0.215</td><td>Zhang et al. 2026</td></tr><tr><td>GTE-Qwen2-7B</td><td>7.1B</td><td>73.1</td><td>$0.089</td><td>Li et al. 2023</td></tr><tr><td>Linq-Mistral</td><td>7.1B</td><td>72.8</td><td>$0.138</td><td>Choi et al. 2024</td></tr><tr><td>Qwen3-E-0.6B</td><td>596M</td><td>72.5</td><td>$0.018</td><td>Zhang et al. 2025</td></tr><tr><td>EmbGemma-300M</td><td>308M</td><td>72.2</td><td>$0.008</td><td>Lee et al. 2025b</td></tr><tr><td>F2LLM-8B</td><td>7.6B</td><td>72.2</td><td>$0.122</td><td>Zhang et al. 2026</td></tr><tr><td>F2LLM-4B</td><td>4.0B</td><td>71.9</td><td>$0.079</td><td>Zhang et al. 2026</td></tr><tr><td>F2LLM-1.7B</td><td>1.7B</td><td>71.7</td><td>$0.030</td><td>Zhang et al. 2026</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>1.5B</td><td>70.8</td><td>$0.024</td><td>Li et al. 2023</td></tr><tr><td>F2LLM-0.6B</td><td>596M</td><td>69.9</td><td>$0.017</td><td>Zhang et al. 2026</td></tr><tr><td>mE5-L-Inst</td><td>560M</td><td>69.7</td><td>$0.005</td><td>Wang et al. 2024</td></tr><tr><td>BGE-M3</td><td>568M</td><td>66.5</td><td>$0.005</td><td>Chen et al. 2025</td></tr><tr><td>mE5-L</td><td>560M</td><td>65.9</td><td>$0.005</td><td>Wang et al. 2022</td></tr><tr><td>Arctic-L-v2</td><td>568M</td><td>65.3</td><td>$0.005</td><td>Yu et al. 2024</td></tr><tr><td>mE5-B</td><td>278M</td><td>64.4</td><td>$0.002</td><td>Wang et al. 2022</td></tr><tr><td>mE5-S</td><td>118M</td><td>63.1</td><td>$0.001</td><td>Wang et al. 2022</td></tr><tr><td>E5-Mistral-7B</td><td>7.1B</td><td>52.2</td><td>$0.139</td><td>Wang et al. 2024</td></tr><tr><td>GritLM-7B</td><td>7.2B</td><td>51.2</td><td>$0.138</td><td>Muennighoff et al. 2024</td></tr></table>

Table 3: Complete model listing with mean (macro) scores across 37 MTEB(LLM) tasks, total benchmark costs, and references. LLM costs reflect actual API usage; embedding costs from H100 throughput benchmarking (\$2.49/hr).

<table><tr><td>Category</td><td>Gemini 3.1 Pro</td><td>Gemini 3 Flash</td><td>Best Emb.</td><td>Best Model</td><td>∆</td></tr><tr><td>Retrieval (6)</td><td>64.5</td><td>52.3</td><td>56.0</td><td>Octen-8B</td><td>+8.5</td></tr><tr><td>Clustering (9)</td><td>66.6</td><td>65.7</td><td>66.7</td><td>SFR-2</td><td>-0.2</td></tr><tr><td>STS (10)</td><td>88.5</td><td>87.6</td><td>88.8</td><td>Qwen3-E-4B</td><td>-0.3</td></tr><tr><td>PairCls (4)</td><td>83.2</td><td>86.3</td><td>87.1</td><td>KaLM-12B</td><td>-3.9</td></tr><tr><td>Classification (8)</td><td>85.2</td><td>84.1</td><td>90.8</td><td>SFR-2</td><td>-5.6</td></tr></table>

Table 4: Category-level performance (best LLM vs. best embedding per category). ∆ = best LLM — best embedding; bold = winner; task counts in parentheses.

LLMs listed first and the 26 embedding models below; the category's tasks are columns, with its metric given in the caption. The final column gives each model's mean over the category, matching the corresponding column of Table 1. Bold marks the best value in a column, whether a single task or the category mean. Because rows are ordered by overall score rather than by category, this column is not monotonic: it shows where a model overor under-performs its overall rank. These tables extend Table 6 to all models.

<table><tr><td>Rank Model</td><td></td><td>Type Cls (8)</td><td>Clust (9)</td><td>STS (10)</td><td>PairCls (4)</td><td>Retr (6)</td><td>Overall</td></tr><tr><td>1</td><td>Gemini 3.1 Pro</td><td>LLM</td><td>85.2</td><td>66.6</td><td>88.5</td><td>83.2</td><td>64.5 77.6</td></tr><tr><td>2 Octen-8B</td><td></td><td>Emb</td><td>90.1</td><td>65.1 88.7</td><td>86.1</td><td>56.0</td><td>77.2</td></tr><tr><td>3 Qwen3-E-8B</td><td></td><td>Emb 90.1</td><td>65.9</td><td>88.5</td><td>86.5</td><td>54.2</td><td>77.0</td></tr><tr><td>4</td><td>Qwen3-E-4B</td><td>Emb 89.3</td><td>64.6</td><td>88.8</td><td>86.5</td><td>50.3</td><td>75.9</td></tr><tr><td>5 Nemotron-8B</td><td></td><td>Emb 84.8</td><td>64.1</td><td>86.2</td><td>86.7</td><td>54.7</td><td>75.3</td></tr><tr><td>6Gemini 3 Flash</td><td>LLM</td><td>84.1</td><td>65.7</td><td>87.6</td><td>86.3</td><td>52.3</td><td>75.2</td></tr><tr><td>7 KaLM-12B</td><td>Emb</td><td>88.8</td><td>63.4</td><td>84.6</td><td>87.1</td><td>49.9</td><td>74.8</td></tr><tr><td>8 Jina-v5-S</td><td>Emb</td><td>90.4</td><td>61.5</td><td>86.9</td><td>85.1</td><td>48.0</td><td>74.4</td></tr><tr><td>9 SFR-2</td><td>Emb</td><td>90.8</td><td>66.7</td><td>77.9</td><td>85.3</td><td>48.8</td><td>73.9</td></tr><tr><td>10 Jina-v5-Nano</td><td>Emb</td><td>89.6</td><td>60.4</td><td>87.0</td><td>85.0</td><td>47.4</td><td>73.9</td></tr><tr><td>11 Qwen3.6-27B</td><td>LLM</td><td>84.8</td><td>53.6</td><td>84.9</td><td>83.6</td><td>62.4</td><td>73.9</td></tr><tr><td>12 F2LLM-14B</td><td>Emb</td><td>77.7</td><td>66.5</td><td>84.1</td><td>86.3</td><td>52.1</td><td>73.3</td></tr><tr><td>13 Qwen3.6-35B-A3B</td><td>LLM</td><td>83.8</td><td>55.3</td><td>84.0</td><td>82.9</td><td>60.4</td><td>73.3</td></tr><tr><td>14 GTE-Qwen2-7B</td><td>Emb</td><td>86.7</td><td>65.2</td><td>81.6</td><td>86.0</td><td>45.8</td><td>73.1</td></tr><tr><td>15 Linq-Mistral</td><td>Emb</td><td>82.5</td><td>61.3</td><td>83.4</td><td>86.1</td><td>51.0</td><td>72.8</td></tr><tr><td>16 Qwen3-E-0.6B</td><td>Emb</td><td>85.4</td><td>60.9</td><td>85.7</td><td>86.1</td><td>44.1</td><td>72.5</td></tr><tr><td>17 EmbGemma-300M</td><td>Emb</td><td>86.5</td><td>59.3</td><td>82.1</td><td>86.0</td><td>47.3</td><td>72.2</td></tr><tr><td>18 F2LLM-8B</td><td>Emb</td><td>75.2</td><td>65.8</td><td>83.9</td><td>86.1</td><td>50.0</td><td>72.2</td></tr><tr><td>19 F2LLM-4B</td><td>Emb</td><td>75.3</td><td>64.7</td><td>83.6</td><td>85.9</td><td>49.9</td><td>71.9</td></tr><tr><td>20 Kimi-K2.6</td><td>LLM</td><td>84.5</td><td>55.4</td><td>83.9</td><td>78.0</td><td>56.7</td><td>71.7</td></tr><tr><td>21 F2LLM-1.7B</td><td>Emb</td><td>74.7</td><td>65.1</td><td>83.8</td><td>86.1</td><td>48.5</td><td>71.7</td></tr><tr><td>22 GTE-Qwen2-1.5B</td><td>Emb</td><td>83.3</td><td>60.0</td><td>80.3</td><td>86.6</td><td>43.9</td><td>70.8</td></tr><tr><td>23 F2LLM-0.6B</td><td>Emb</td><td>72.5</td><td>62.9</td><td>83.1</td><td>85.8</td><td>45.4</td><td>69.9</td></tr><tr><td>24 mE5-L-Inst</td><td>Emb</td><td>74.4</td><td>59.1</td><td>83.9</td><td>86.1</td><td>44.8</td><td>69.7</td></tr><tr><td>25 DeepSeek-V4-Flash</td><td>LLM</td><td>81.9</td><td>43.7</td><td>82.2</td><td>81.6</td><td>53.3</td><td>68.5</td></tr><tr><td>26 MiniMax-M2.7</td><td>LLM</td><td>80.9</td><td>48.6</td><td>81.0</td><td>82.8</td><td>48.0</td><td>68.2</td></tr><tr><td>27 GLM-4.7</td><td>LLM</td><td>84.2</td><td>36.5</td><td>83.5</td><td>79.7</td><td>54.4</td><td>67.7</td></tr><tr><td>28 BGE-M3</td><td>Emb</td><td>76.2</td><td>45.9</td><td>80.4</td><td>85.9</td><td>44.3</td><td>66.5</td></tr><tr><td>29 mE5-L</td><td>Emb</td><td>73.3</td><td>48.3</td><td>80.3</td><td>83.9</td><td>43.6</td><td>65.9</td></tr><tr><td>30 Arctic-L-v2</td><td>Emb</td><td>71.4</td><td>49.7</td><td>77.2</td><td>83.9</td><td>44.6</td><td>65.3</td></tr><tr><td>31 Gemini 3.1 Flash Lite</td><td>LLM</td><td>82.9</td><td>21.7</td><td>85.1</td><td>83.7</td><td>49.0</td><td>64.5</td></tr><tr><td>32 mE5-B</td><td>Emb</td><td>71.9</td><td>47.5</td><td>79.2</td><td>84.3</td><td>38.9</td><td>64.4</td></tr><tr><td>33 DeepSeek-R1</td><td>LLM</td><td>82.5</td><td>31.3</td><td>83.7</td><td>79.5</td><td>44.7</td><td>64.3</td></tr><tr><td>34 mE5-S</td><td>Emb</td><td>69.8</td><td>47.7</td><td>78.5</td><td>83.9</td><td>35.8</td><td>63.1</td></tr><tr><td>35 E5-Mistral-7B</td><td>Emb</td><td>70.0</td><td>47.9</td><td>49.8</td><td>72.9</td><td>20.6</td><td>52.2</td></tr><tr><td>36 GritLM-7B</td><td>Emb</td><td>68.6</td><td>48.9</td><td>55.1</td><td>77.3</td><td>6.1</td><td>51.2</td></tr></table>

Table 5: Complete per-category results for all 36 complete models, ranked by overall (macro) score. Bold = best in column.

## B.5 Benchmark Task Suite

Table 12 lists all 37 tasks with languages, sample counts, and source citations. All are heldout subsets (seed = 42) derived from MTEB and MMTEB tasks, hosted on Hugging Face under mteb/11m-eval-\*; the exact dataset path and pinned revision for each task are listed in the released repository. Table 13 provides per-task token budgets for cost estimation, broken down by split.

## C Ablation Studies

This section details the reduced-thinking and few-shot experiments and provides a per-task retrieval breakdown.

## C.1 Reduced-Thinking Ablation Details

Our reduced-thinking ablation uses Gemini 3 Flash with reasoning\_effort=low, which instructs the model to minimise internal chain-of-thought reasoning. We evaluate on all 6 retrieval tasks and 3 representative classification tasks (IMDB, Banking77, ToxicConversations).

<table><tr><td>Task</td><td>Cat.</td><td>Gemini 3.1 Pro</td><td>Gemini 3 Flash</td><td>Qwen3.6-27B</td><td>Octen-8B</td><td>Qwen3-E-8B</td><td>Best Emb.</td></tr><tr><td>Classification (Accuracy)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AmazonCounterfactualClassification</td><td>Cls</td><td>84.8</td><td>81.2</td><td>90.1</td><td>93.0</td><td>92.9</td><td>93.0</td></tr><tr><td>Banking77Classification</td><td>Cls</td><td>85.0</td><td>83.1</td><td>80.5</td><td>87.3</td><td>87.2</td><td>91.6</td></tr><tr><td>ImdbClassification</td><td>Cls</td><td>98.0</td><td>97.6</td><td>97.4</td><td>97.7</td><td>97.8</td><td>97.8</td></tr><tr><td>MTOPDomainClassification</td><td>Cls</td><td>95.5</td><td>96.4</td><td>96.9</td><td>98.2</td><td>98.1</td><td>99.2</td></tr><tr><td>MassiveIntentClassification</td><td>Cls</td><td>84.9</td><td>85.4</td><td>84.6</td><td>85.7</td><td>85.8</td><td>88.9</td></tr><tr><td>MassiveScenarioClassification</td><td>Cls</td><td>79.4</td><td>75.9</td><td>76.4</td><td>89.3</td><td>89.4</td><td>93.0</td></tr><tr><td>ToxicConversationsClassification</td><td>Cls</td><td>89.6</td><td>90.0</td><td>89.2</td><td>91.9</td><td>92.0</td><td>94.4</td></tr><tr><td>TweetSentimentExtractionClassification</td><td>Cls</td><td>64.2</td><td>63.0</td><td>63.2</td><td>77.9</td><td>77.8</td><td>78.9</td></tr><tr><td>Clustering (V-measure)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ArxivClusteringP2P</td><td>Clust</td><td>60.1</td><td>60.3</td><td>54.5</td><td>61.9</td><td>62.5</td><td>64.1</td></tr><tr><td>ArxivClusteringS2S</td><td>Clust</td><td>62.5</td><td>60.1</td><td>41.6</td><td>60.5</td><td>61.1</td><td>61.5</td></tr><tr><td>BiorxivClusteringP2PV2</td><td>Clust</td><td>62.5</td><td>60.8</td><td>49.9</td><td>61.6</td><td>65.3</td><td>77.3</td></tr><tr><td>MedrxivClusteringP2PV2</td><td>Clust</td><td>52.5</td><td>50.6</td><td>48.0</td><td>55.9</td><td>55.9</td><td>60.7</td></tr><tr><td>MedrxivClusteringS2SV2</td><td>Clust</td><td>54.0</td><td>50.3</td><td>42.7</td><td>53.3</td><td>55.2</td><td>57.5</td></tr><tr><td>RedditClusteringP2P</td><td>Clust</td><td>93.9</td><td>90.5</td><td>60.1</td><td>78.7</td><td>79.8</td><td>80.8</td></tr><tr><td>StackExchangeClusteringP2PV2</td><td>Clust</td><td>42.9</td><td>49.0</td><td>46.1</td><td>59.7</td><td>59.5</td><td>60.7</td></tr><tr><td>StackExchangeClusteringV2</td><td>Clust</td><td>91.8</td><td>90.2</td><td>73.6</td><td>85.8</td><td>86.3</td><td>86.3</td></tr><tr><td>TwentyNewsgroupsClusteringV2</td><td>Clust</td><td>78.7</td><td>79.9</td><td>66.3</td><td>68.5</td><td>67.6</td><td>73.8</td></tr><tr><td>STS (Śpearmañ ρ)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BIOSSES</td><td>STS</td><td>89.6</td><td>88.7</td><td>85.0</td><td>84.4</td><td>82.0</td><td>88.8</td></tr><tr><td>SICKR</td><td>STS</td><td>86.1</td><td>86.8</td><td>76.4</td><td>87.9</td><td>88.4</td><td>91.1</td></tr><tr><td>STS12</td><td>STS</td><td>82.6</td><td>79.5</td><td>77.1</td><td>87.3</td><td>87.3</td><td>87.4</td></tr><tr><td>STS13</td><td>STS</td><td>91.9</td><td>91.5</td><td>90.6</td><td>93.9</td><td>94.0</td><td>94.6</td></tr><tr><td>STS14</td><td>STS</td><td>90.5</td><td>89.3</td><td>88.0</td><td>90.9</td><td>91.0</td><td>91.4</td></tr><tr><td>STS15</td><td>STS</td><td>94.4</td><td>93.9</td><td>91.8</td><td>94.3</td><td>94.2</td><td>94.5</td></tr><tr><td>STS16</td><td>STS</td><td>89.0</td><td>88.5</td><td>87.1</td><td>92.5</td><td>92.5</td><td>92.9</td></tr><tr><td>STS17</td><td>STS</td><td>94.2</td><td>94.1</td><td>91.8</td><td>93.6</td><td>93.7</td><td>93.7</td></tr><tr><td>STS22v2</td><td>STS</td><td>73.6</td><td>71.7</td><td>72.0</td><td>69.5</td><td>68.9</td><td>69.5</td></tr><tr><td>STSBenchmark</td><td>STS</td><td>92.6</td><td>91.7</td><td>89.6</td><td>93.2</td><td>93.4</td><td>94.9</td></tr><tr><td>PairClassification (Avg. Precision)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LegalBenchPC RTE3PC</td><td>PairCls</td><td>72.2</td><td>77.4</td><td>71.6</td><td>70.0</td><td>70.4</td><td>73.6</td></tr><tr><td></td><td>PairCls</td><td>95.4</td><td>96.7</td><td>92.9</td><td>85.4</td><td>85.7</td><td>85.7</td></tr><tr><td>SprintDuplicateQuestionsPC</td><td>PairCls</td><td>82.2</td><td>83.4</td><td>84.6</td><td>99.4</td><td>99.6</td><td>99.8</td></tr><tr><td>TwitterURLCorpusPC</td><td>PairCls</td><td>82.8</td><td>87.6</td><td>85.2</td><td>89.4</td><td>90.2</td><td>90.2</td></tr><tr><td>Retrieval (Recall@1) AILAStatutes</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FQuADRetrieval</td><td>Retr</td><td>14.5</td><td>5.7</td><td>14.2</td><td>23.2</td><td>21.8</td><td>23.2</td></tr><tr><td></td><td>Retr</td><td>92.0</td><td>88.0</td><td>96.0</td><td>69.0</td><td>68.0</td><td>72.0</td></tr><tr><td>HC3FinanceRetrieval</td><td>Retr</td><td>71.0</td><td>60.0</td><td>57.0</td><td>66.0</td><td>59.0</td><td>67.0</td></tr><tr><td>LegalBenchConsumerContractsQA</td><td>Retr</td><td>86.0</td><td>79.0</td><td>88.0</td><td>67.0</td><td>70.0</td><td>70.0</td></tr><tr><td>PublicHealthQA TwitterHjerneRetrieval</td><td>Retr Retr</td><td>90.0 33.4</td><td>49.0 32.4</td><td>86.0 33.0</td><td>81.0 29.9</td><td>78.0 28.6</td><td>85.0 31.5</td></tr></table>

Table 6: Per-task scores for representative models across all 37 MTEB(LLM) tasks. Bold = best in row (incl. best embedding). Metric per category in italics.

The reduction differs by task type: 54–94% on retrieval and 12–30% on classification. This suggests the model adaptively allocates more reasoning to retrieval (where documents must be read and compared) than classification (where the decision is more immediate).

Reduced reasoning preserves or improves all six Flash retrieval scores and four of six cross-family averages (Figure 4b). We hypothesise that excessive reasoning introduces second-guessing: the model considers multiple interpretations of a document's relevance when a more direct reading would suffice.

On classification, all three tasks show $| \Delta | < 1 . 0 ,$ indicating no measurable benefit from additional reasoning in this ablation. This result is consistent with the supervision asymmetry between kNN and zero-shot classification.

## C.2 Per-Task Retrieval Analysis

The six retrieval tasks in our benchmark span qualitatively different retrieval challenges; per-task scores for all 36 models are in Table 11. We provide a per-task breakdown of the aggregate retrieval advantage (+8.5) reported in §4.2. Pro leads five retrieval tasks, and Octen-8B leads AILAStatutes.

FQuAD (French reading-comprehension QA). Answering requires reading French passages in context and matching the supporting one. Pro (92.0) leads the best embedding (72.0) by a wide margin, the largest LLM advantage in the suite.

<table><tr><td>Model</td><td>AmazonCF</td><td>Banking77</td><td>IMDB</td><td>MTOPDomain</td><td>MassiveIntent</td><td>MassiveScenario</td><td>ToxicConvs</td><td>TweetSent</td><td>Mean</td></tr><tr><td colspan="11"></td></tr><tr><td>LLMs Gemini 3.1 Pro</td><td>84.8</td><td>85.0</td><td>98.0</td><td>95.5</td><td>84.9</td><td>79.4</td><td>89.6</td><td>64.2</td><td>85.2</td></tr><tr><td>Gemini 3 Flash</td><td>81.2</td><td>83.1</td><td>97.6</td><td>96.4</td><td>85.4</td><td>75.9</td><td>90.0</td><td>63.0</td><td>84.1</td></tr><tr><td>Qwen3.6-27B</td><td>90.1</td><td>80.5</td><td>97.4</td><td>96.9</td><td>84.6</td><td>76.4</td><td>89.2</td><td>63.2</td><td>84.8</td></tr><tr><td></td><td>88.7</td><td>77.8</td><td>97.4</td><td>96.8</td><td>79.8</td><td>76.6</td><td>91.0</td><td></td><td></td></tr><tr><td>Qwen3.6-35B-A3B</td><td></td><td></td><td>97.4</td><td></td><td></td><td></td><td></td><td>62.4</td><td>83.8</td></tr><tr><td>Kimi-K2.6</td><td>89.3</td><td>82.8</td><td></td><td>96.6</td><td>83.2</td><td>76.5</td><td>88.8</td><td>61.6</td><td>84.5</td></tr><tr><td>DeepSeek-V4-Flash</td><td>82.4</td><td>78.5</td><td>96.8</td><td>95.8</td><td>82.4</td><td>77.5</td><td>80.4</td><td>61.2</td><td>81.9</td></tr><tr><td>MiniMax-M2.7 GLM-4.7</td><td>83.6 88.7</td><td>78.4 80.4</td><td>97.0 97.2</td><td>95.2</td><td>75.1</td><td>67.0</td><td>87.6</td><td>63.0</td><td>80.9</td></tr><tr><td>Gemini 3.1 FLite</td><td>79.7</td><td>78.7</td><td>97.6</td><td>96.8 95.9</td><td>83.0 83.2</td><td>77.1</td><td>87.6</td><td>63.2</td><td>84.2</td></tr><tr><td>DeepSeek-R1</td><td>86.9</td><td>78.9</td><td>97.0</td><td>96.1</td><td>82.7</td><td>76.2 76.6</td><td>88.6 78.2</td><td>63.4</td><td>82.9</td></tr><tr><td>Embedding models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>63.4</td><td>82.5</td></tr><tr><td colspan="10"></td></tr><tr><td>Octen-8B</td><td>93.0</td><td>87.3</td><td>97.7</td><td>98.2</td><td>85.7</td><td>89.3</td><td>91.9</td><td>77.9</td><td>90.1</td></tr><tr><td>Qwen3-E-8B</td><td>92.9</td><td>87.2</td><td>97.8</td><td>98.1</td><td>85.8</td><td>89.4</td><td>92.0</td><td>77.8</td><td>90.1</td></tr><tr><td>Qwen3-E-4B</td><td>92.6</td><td>86.3</td><td>97.4</td><td>97.6</td><td>85.2</td><td>86.0</td><td>92.3</td><td>77.1</td><td>89.3</td></tr><tr><td>Nemotron-8B</td><td>83.8</td><td>83.6</td><td>97.1</td><td>96.5</td><td>83.5</td><td>83.5</td><td>88.3</td><td>62.1</td><td>84.8</td></tr><tr><td>KaLM-12B</td><td>90.4</td><td>87.7</td><td>96.3</td><td>98.5</td><td>84.6</td><td>86.7</td><td>90.2</td><td>76.1</td><td>88.8</td></tr><tr><td>Jina-v5-S</td><td>91.8</td><td>91.5</td><td>95.9</td><td>99.2</td><td>88.9</td><td>92.6</td><td>94.4</td><td>68.5</td><td>90.4</td></tr><tr><td>SFR-2</td><td>92.7</td><td>90.1</td><td>97.6</td><td>98.3</td><td>86.3</td><td>90.5</td><td>91.9</td><td>78.9</td><td>90.8</td></tr><tr><td>Jina-v5-Nano</td><td>91.3</td><td>90.2</td><td>95.5</td><td>98.0</td><td>88.4</td><td>93.0</td><td>94.4</td><td>66.3</td><td>89.6</td></tr><tr><td>F2LLM-14B</td><td>64.1</td><td>84.7</td><td>91.2</td><td>98.8</td><td>77.6</td><td>89.4</td><td>58.7</td><td>56.7</td><td>77.7</td></tr><tr><td>GTE-Qwen2-7B</td><td>86.8</td><td>84.9</td><td>96.7</td><td>98.0</td><td>83.7</td><td>85.7</td><td>88.2</td><td>70.0</td><td>86.7</td></tr><tr><td>Linq-Mistral</td><td>83.9</td><td>87.7</td><td>94.9</td><td>97.0</td><td>82.7</td><td>84.5</td><td>70.2</td><td>59.2</td><td>82.5</td></tr><tr><td>Qwen3-E-0.6B</td><td>90.5</td><td>80.8</td><td>96.3</td><td>95.8</td><td>80.1</td><td>83.8</td><td>81.9</td><td>74.3</td><td>85.4</td></tr><tr><td>EmbGemma-300M</td><td>89.6</td><td>91.6</td><td>91.9</td><td>98.6</td><td>85.6</td><td>91.6</td><td>83.4</td><td>59.8</td><td>86.5</td></tr><tr><td>F2LLM-8B</td><td>63.1</td><td>83.6</td><td>87.3</td><td>99.0</td><td>72.2</td><td>85.8</td><td>59.3</td><td>51.4</td><td>75.2</td></tr><tr><td>F2LLM-4B</td><td>61.3</td><td>82.2</td><td>86.4</td><td>99.0</td><td>72.3</td><td>86.5</td><td>60.7</td><td>54.2</td><td>75.3</td></tr><tr><td>F2LLM-1.7B</td><td>61.2</td><td>77.6</td><td>83.4</td><td>98.3</td><td>74.6</td><td>87.5</td><td>61.1</td><td>54.3</td><td>74.7</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>81.6</td><td>79.7</td><td>95.8</td><td>95.3</td><td>77.9</td><td>79.8</td><td>84.9</td><td>71.3</td><td>83.3</td></tr><tr><td>F2LLM-0.6B</td><td>59.5</td><td>73.8</td><td>79.7</td><td>97.3</td><td>73.4</td><td>87.3</td><td>59.7</td><td>49.0</td><td>72.5</td></tr><tr><td>mE5-L-Inst</td><td>69.1</td><td>77.4</td><td>96.4</td><td>91.6</td><td>70.0</td><td>69.1</td><td>64.8</td><td>56.8</td><td>74.4</td></tr><tr><td>BGE-M3</td><td>75.4</td><td>82.4</td><td>87.9</td><td>93.5</td><td>72.2</td><td>76.6</td><td>64.8</td><td>56.9</td><td>76.2</td></tr><tr><td>mE5-L</td><td>79.1</td><td>75.3</td><td>89.2</td><td>91.4</td><td>68.7</td><td>72.8</td><td>58.1</td><td>51.8</td><td>73.3</td></tr><tr><td>Arctic-L-v2</td><td>63.6</td><td>82.3</td><td>73.4</td><td>93.5</td><td>72.2</td><td>76.0</td><td>61.5</td><td>48.8</td><td>71.4</td></tr><tr><td>mE5-B</td><td>79.6</td><td>73.7</td><td>85.5</td><td>90.6</td><td>66.1</td><td>71.3</td><td>57.1</td><td>51.6</td><td>71.9</td></tr><tr><td>mE5-S</td><td>71.8</td><td>70.6</td><td>79.7 91.6</td><td>89.1</td><td>65.6</td><td>69.1</td><td>59.4</td><td>52.9</td><td>69.8</td></tr><tr><td>E5-Mistral-7B</td><td>71.8</td><td>71.4</td><td></td><td>79.3</td><td>65.8</td><td>73.4</td><td>62.2</td><td>44.9</td><td>70.0</td></tr><tr><td>GritLM-7B</td><td>74.7</td><td>70.6</td><td>79.0</td><td>84.9</td><td>65.2</td><td>68.3</td><td>61.4</td><td>45.0</td><td>68.6</td></tr></table>

Table 7: Per-task scores: Classification (Accuracy)

<table><tr><td>Model</td><td>BIOSSES</td><td>SICK-R</td><td>STS12</td><td>STS13</td><td>STS14</td><td>STS15</td><td>STS16</td><td>STS17</td><td>STS22v2</td><td>STSBench</td><td>Mean</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>89.6</td><td>86.1</td><td>82.6</td><td>91.9</td><td>90.5</td><td>94.4</td><td>89.0</td><td>94.2</td><td>73.6</td><td>92.6</td><td>88.5</td></tr><tr><td>Gemini 3 Flash</td><td>88.7</td><td>86.8</td><td>79.5</td><td>91.5</td><td>89.3</td><td>93.9</td><td>88.5</td><td>94.1</td><td>71.7</td><td>91.7</td><td>87.6</td></tr><tr><td>Qwen3.6-27B</td><td>85.0</td><td>76.4</td><td>77.1</td><td>90.6</td><td>88.0</td><td>91.8</td><td>87.1</td><td>91.8</td><td>72.0</td><td>89.6</td><td>84.9</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>84.0</td><td>75.5</td><td>75.1</td><td>89.6</td><td>85.1</td><td>91.4</td><td>85.1</td><td>92.7</td><td>73.4</td><td>87.8</td><td>84.0</td></tr><tr><td>Kimi-K2.6</td><td>86.4</td><td>82.1</td><td>72.2</td><td>89.3</td><td>85.1</td><td>91.1</td><td>83.9</td><td>90.1</td><td>70.9</td><td>87.8</td><td>83.9</td></tr><tr><td>DeepSeek-V4-Flash</td><td>78.5</td><td>75.3</td><td>75.6</td><td>87.3</td><td>84.5</td><td>89.2</td><td>82.4</td><td>91.5</td><td>69.2</td><td>87.9</td><td>82.2</td></tr><tr><td>MiniMax-M2.7</td><td>87.7</td><td>79.7</td><td>65.4</td><td>86.5</td><td>81.9</td><td>88.4</td><td>81.1</td><td>87.1</td><td>66.7</td><td>85.7</td><td>81.0</td></tr><tr><td>GLM-4.7</td><td>82.3</td><td>76.0</td><td>73.1</td><td>88.5</td><td>85.7</td><td>91.9</td><td>84.9</td><td>90.4</td><td>73.2</td><td>88.5</td><td>83.5</td></tr><tr><td>Gemini 3.1 FLite</td><td>89.8</td><td>78.0</td><td>76.0</td><td>90.5</td><td>86.3</td><td>91.4</td><td>86.5</td><td>91.4</td><td>71.2</td><td>90.3</td><td>85.1</td></tr><tr><td>DeepSeek-R1</td><td>85.0</td><td>79.7</td><td>71.9</td><td>88.7</td><td>85.6</td><td>89.9</td><td>84.2</td><td>91.5</td><td>71.4</td><td>89.2</td><td>83.7</td></tr><tr><td>Embedding models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Octen-8B</td><td>84.4</td><td>87.9</td><td>87.3</td><td>93.9</td><td>90.9</td><td>94.3</td><td>92.5</td><td>93.6</td><td>69.5</td><td>93.2</td><td>88.7</td></tr><tr><td>Qwen3-E-8B</td><td>82.0</td><td>88.4</td><td>87.3</td><td>94.0</td><td>91.0</td><td>94.2</td><td>92.5</td><td>93.7</td><td>68.9</td><td>93.4</td><td>88.5</td></tr><tr><td>Qwen3-E-4B</td><td>84.8</td><td>87.7</td><td>87.4</td><td>94.6</td><td>91.4</td><td>94.5</td><td>92.9</td><td>93.2</td><td>67.7</td><td>93.7</td><td>88.8</td></tr><tr><td>Nemotron-8B</td><td>86.3</td><td>85.4</td><td>83.4</td><td>91.5</td><td>87.9</td><td>92.1</td><td>88.7</td><td>91.4</td><td>64.3</td><td>90.8</td><td>86.2</td></tr><tr><td>KaLM-12B</td><td>87.0</td><td>81.9</td><td>81.4</td><td>89.2</td><td>85.9</td><td>90.6</td><td>86.7</td><td>89.0</td><td>65.1</td><td>89.1</td><td>84.6</td></tr><tr><td>Jina-v5-S</td><td>85.3</td><td>90.4</td><td>86.6</td><td>89.2</td><td>88.9</td><td>92.6</td><td>87.4</td><td>89.1</td><td>65.0</td><td>94.9</td><td>86.9</td></tr><tr><td>SFR-2</td><td>87.5</td><td>76.7</td><td>74.6</td><td>81.4</td><td>80.6</td><td>87.5</td><td>84.6</td><td>85.0</td><td>38.2</td><td>82.6</td><td>77.9</td></tr><tr><td>Jina-v5-Nano</td><td>87.4</td><td>91.1</td><td>86.2</td><td>89.9</td><td>89.2</td><td>93.1</td><td>85.4</td><td>88.4</td><td>64.9</td><td>94.2</td><td>87.0</td></tr><tr><td>F2LLM-14B</td><td>88.2</td><td>82.8</td><td>82.4</td><td>86.9</td><td>84.8</td><td>91.3</td><td>85.6</td><td>88.9</td><td>63.4</td><td>87.0</td><td>84.1</td></tr><tr><td>GTE-Qwen2-7B</td><td>83.0</td><td>79.7</td><td>78.1</td><td>88.6</td><td>83.4</td><td>89.5</td><td>85.5</td><td>87.1</td><td>54.7</td><td>86.2</td><td>81.6</td></tr><tr><td>Linq-Mistral</td><td>86.1</td><td>84.4</td><td>77.6</td><td>87.8</td><td>84.1</td><td>91.2</td><td>87.4</td><td>89.9</td><td>56.1</td><td>89.0</td><td>83.4</td></tr><tr><td>Qwen3-E-0.6B</td><td>84.6</td><td>84.6</td><td>83.6</td><td>92.1</td><td>87.0</td><td>91.7</td><td>89.7</td><td>87.9</td><td>64.6</td><td>91.3</td><td>85.7</td></tr><tr><td>EmbGemma-300M</td><td>83.2</td><td>81.9</td><td>79.1</td><td>85.5</td><td>84.5</td><td>89.2</td><td>84.7</td><td>86.6</td><td>57.3</td><td>88.8</td><td>82.1</td></tr><tr><td>F2LLM-8B</td><td>88.8</td><td>82.4</td><td>81.3</td><td>86.6</td><td>83.8</td><td>91.3</td><td>85.4</td><td>89.3</td><td>63.2</td><td>86.5</td><td>83.9</td></tr><tr><td>F2LLM-4B</td><td>87.5</td><td>81.9</td><td>82.3</td><td>86.1</td><td>83.8</td><td>91.5</td><td>85.2</td><td>87.9</td><td>62.8</td><td>87.1</td><td>83.6</td></tr><tr><td>F2LLM-1.7B</td><td>88.5</td><td>81.3</td><td>82.6</td><td>86.7</td><td>84.5</td><td>91.3</td><td>85.2</td><td>87.8</td><td>62.9</td><td>87.6</td><td>83.8</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>81.4</td><td>80.7</td><td>72.9</td><td>84.4</td><td>81.7</td><td>88.7</td><td>84.6</td><td>86.1</td><td>56.6</td><td>85.6</td><td>80.3</td></tr><tr><td>F2LLM-0.6B</td><td>87.8</td><td>80.0</td><td>82.7</td><td>86.7</td><td>84.1</td><td>90.6</td><td>84.9</td><td>84.0</td><td>62.3</td><td>87.6</td><td>83.1</td></tr><tr><td>mE5-L-Inst</td><td>87.0</td><td>81.8</td><td>83.5</td><td>86.8</td><td>85.1</td><td>91.8</td><td>86.6</td><td>87.0</td><td>60.8</td><td>89.0</td><td>83.9</td></tr><tr><td>BGE-M3</td><td>83.4</td><td>79.3</td><td>79.8</td><td>78.6</td><td>80.7</td><td>88.3</td><td>84.7</td><td>82.3</td><td>60.5</td><td>86.0</td><td>80.4</td></tr><tr><td>mE5-L</td><td>84.6</td><td>80.5</td><td>80.9</td><td>75.6</td><td>78.7</td><td>89.7</td><td>83.6</td><td>85.4</td><td>58.5</td><td>85.8</td><td>80.3</td></tr><tr><td>Arctic-L-v2</td><td>87.2</td><td>72.5</td><td>71.6</td><td>79.6</td><td>76.0</td><td>83.9</td><td>83.2</td><td>79.4</td><td>59.6</td><td>78.4</td><td>77.2</td></tr><tr><td>mE5-B mE5-S</td><td>86.2 84.4</td><td>78.9 78.0</td><td>79.1</td><td>75.9</td><td>79.2</td><td>88.7</td></table>

Table 8: Per-task scores: STS (Spearman ρ).

<table><tr><td>Model</td><td>ArxivP2P</td><td>ArxivS2S</td><td>BioP2P</td><td>MedP2P</td><td>MedS2S</td><td>Reddit</td><td>SE-P2P</td><td>SE-Cl</td><td>20News</td><td>Mean</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>60.1</td><td>62.5</td><td>62.5</td><td>52.5</td><td>54.0</td><td>93.9</td><td>42.9</td><td>91.8</td><td>78.7</td><td>66.6</td></tr><tr><td>Gemini 3 Flash</td><td>60.3</td><td>60.1</td><td>60.8</td><td>50.6</td><td>50.3</td><td>90.5</td><td>49.0</td><td>90.2</td><td>79.9</td><td>65.7</td></tr><tr><td>Qwen3.6-27B</td><td>54.5</td><td>41.6</td><td>49.9</td><td>48.0</td><td>42.7</td><td>60.1</td><td>46.1</td><td>73.6</td><td>66.3</td><td>53.6</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>48.5</td><td>60.4</td><td>53.2</td><td>49.4</td><td>46.2</td><td>69.2</td><td>42.2</td><td>72.7</td><td>55.7</td><td>55.3</td></tr><tr><td>Kimi-K2.6</td><td>52.6</td><td>44.2</td><td>51.9</td><td>50.6</td><td>39.1</td><td>57.4</td><td>52.1</td><td>81.8</td><td>68.9</td><td>55.4</td></tr><tr><td>DeepSeek-V4-Flash</td><td>37.4</td><td>45.5</td><td>29.3</td><td>27.0</td><td>40.9</td><td>55.2</td><td>40.3</td><td>53.5</td><td>64.4</td><td>43.7</td></tr><tr><td>MiniMax-M2.7</td><td>36.4</td><td>40.6</td><td>38.3</td><td>39.8</td><td>42.4</td><td>67.4</td><td>39.4</td><td>75.1</td><td>57.9</td><td>48.6</td></tr><tr><td>GLM-4.7</td><td>23.6</td><td>28.6</td><td>33.2</td><td>37.5</td><td>37.8</td><td>40.6</td><td>37.9</td><td>49.5</td><td>39.8</td><td>36.5</td></tr><tr><td>Gemini 3.1 FLite</td><td>14.7</td><td>18.9</td><td>17.4</td><td>19.7</td><td>21.6</td><td>28.5</td><td>21.5</td><td>25.8</td><td>27.2</td><td>21.7</td></tr><tr><td>DeepSeek-R1</td><td>22.0</td><td>25.3</td><td>32.4</td><td>31.7</td><td>33.4</td><td>26.6</td><td>33.1</td><td>42.1</td><td>34.9</td><td>31.3</td></tr><tr><td>Embedding models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Octen-8B</td><td>61.9</td><td>60.5</td><td>61.6</td><td>55.9</td><td>53.3</td><td>78.7</td><td>59.7</td><td>85.8</td><td>68.5</td><td>65.1</td></tr><tr><td>Qwen3-E-8B</td><td>62.5</td><td>61.1</td><td>65.3</td><td>55.9</td><td>55.2</td><td>79.8</td><td>59.5</td><td>86.3</td><td>67.6</td><td>65.9</td></tr><tr><td>Qwen3-E-4B</td><td>62.6</td><td>60.3</td><td>64.0</td><td>55.5</td><td>54.1</td><td>77.1</td><td>58.8</td><td>83.9</td><td>65.3</td><td>64.6</td></tr><tr><td>Nemotron-8B</td><td>59.2</td><td>59.1</td><td>65.2</td><td>52.9</td><td>52.6</td><td>79.7</td><td>58.7</td><td>84.6</td><td>65.2</td><td>64.1</td></tr><tr><td>KaLM-12B</td><td>61.6</td><td>57.4</td><td>64.1</td><td>54.7</td><td>53.2</td><td>76.1</td><td>58.5</td><td>83.3</td><td>62.0</td><td>63.4</td></tr><tr><td>Jina-v5-S</td><td>60.0</td><td>57.5</td><td>63.6</td><td>53.7</td><td>52.6</td><td>70.5</td><td>58.9</td><td>76.8</td><td>59.9</td><td>61.5</td></tr><tr><td>SFR-2</td><td>61.8</td><td>59.6</td><td>65.6</td><td>58.9</td><td>57.5</td><td>80.8</td><td>59.3</td><td>83.2</td><td>73.8</td><td>66.7</td></tr><tr><td>Jina-v5-Nano</td><td>59.1</td><td>55.4</td><td>61.6</td><td>54.6</td><td>52.1</td><td>67.4</td><td>57.3</td><td>76.8</td><td>59.7</td><td>60.4</td></tr><tr><td>F2LLM-14B</td><td>63.1</td><td>60.6</td><td>75.6</td><td>57.7</td><td>56.7</td><td>77.9</td><td>57.2</td><td>81.1</td><td>68.8</td><td>66.5</td></tr><tr><td>GTE-Qwen2-7B</td><td>64.1</td><td>61.5</td><td>66.7</td><td>57.0</td><td>54.1</td><td>80.5</td><td>60.7</td><td>84.5</td><td>58.1</td><td>65.2</td></tr><tr><td>Linq-Mistral</td><td>58.0</td><td>55.7</td><td>61.3</td><td>51.5</td><td>52.8</td><td>77.0</td><td>56.2</td><td>78.4</td><td>61.1</td><td>61.3</td></tr><tr><td>Qwen3-E-0.6B</td><td>60.2</td><td>57.6</td><td>61.8</td><td>52.9</td><td>51.1</td><td>69.9</td><td>56.0</td><td>77.3</td><td>61.7</td><td>60.9</td></tr><tr><td>EmbGemma-300M</td><td>57.9</td><td>53.9</td><td>62.8</td><td>54.4</td><td>51.0</td><td>75.9</td><td>51.9</td><td>73.7</td><td>51.8</td><td>59.3</td></tr><tr><td>F2LLM-8B</td><td>61.7</td><td>60.9</td><td>75.5</td><td>59.2</td><td>54.4</td><td>75.3</td><td>58.2</td><td>81.2</td><td>65.8</td><td>65.8</td></tr><tr><td>F2LLM-4B</td><td>61.3</td><td>59.2</td><td>74.2</td><td>59.8</td><td>54.7</td><td>74.4</td><td>55.2</td><td>79.9</td><td>63.9</td><td>64.7</td></tr><tr><td>F2LLM-1.7B</td><td>62.6</td><td>59.6</td><td>77.3</td><td>60.7</td><td>56.1</td><td>74.4</td><td>55.2</td><td>75.5</td><td>64.2</td><td>65.1</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>58.5</td><td>57.5</td><td>62.1</td><td>53.2</td><td>52.3</td><td>71.1</td><td>52.3</td><td>73.6</td><td>59.1</td><td>60.0</td></tr><tr><td>F2LLM-0.6B</td><td>61.7</td><td>58.6</td><td>71.1</td><td>58.7</td><td>54.5</td><td>69.7</td><td>56.0</td><td>76.0</td><td>59.9</td><td>62.9</td></tr><tr><td>mE5-L-Inst</td><td>56.0</td><td>53.0</td><td>57.8</td><td>51.9</td><td>51.2</td><td>71.7</td><td>53.3</td><td>76.9</td><td>60.2</td><td>59.1</td></tr><tr><td>BGE-M3</td><td>41.7</td><td>32.4</td><td>49.7</td><td>43.1</td><td>40.7</td><td>61.2</td><td>42.6</td><td>56.3</td><td>45.5</td><td>45.9</td></tr><tr><td>mE5-L</td><td>45.8</td><td>39.7</td><td>52.7</td><td>43.2</td><td>43.9</td><td>65.6</td><td>43.7</td><td>56.4</td><td>43.9</td><td>48.3</td></tr><tr><td>Arctic-L-v2</td><td>48.1</td><td>38.4</td><td>48.7</td><td>46.5</td><td>44.0</td><td>65.2</td><td>46.6</td><td>61.7</td><td>48.3</td><td>49.7</td></tr><tr><td>mE5-B</td><td>44.9</td><td>40.2</td><td>46.2</td><td>43.1</td><td>44.3</td><td>64.8</td><td>42.6</td><td>57.6</td><td>43.4</td><td>47.5</td></tr><tr><td>mE5-S</td><td>42.5</td><td>39.9</td><td>48.2</td><td>45.5</td><td>44.3</td><td>65.5</td><td>41.9</td><td>58.0</td><td>43.2</td><td>47.7</td></tr><tr><td>E5-Mistral-7B</td><td>51.2</td><td>44.6</td><td>49.9 51.7</td><td>45.6 46.5</td><td>40.0 42.7</td><td>54.7 63.9</td><td>37.3 43.9</td><td>65.3 59.8</td><td>42.4 35.5</td><td>47.9 48.9</td></tr><tr><td>GritLM-7B</td><td>55.7</td><td>40.4</td></table>

Table 9: Per-task scores: Clustering (V-measure).

<table><tr><td>Model</td><td>LegalPC</td><td>RTE3</td><td>SprintDup</td><td>TwtURL</td><td>Mean</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>72.2</td><td>95.4</td><td>82.2</td><td>82.8</td><td>83.2</td></tr><tr><td>Gemini 3 Flash</td><td>77.4</td><td>96.7</td><td>83.4</td><td>87.6</td><td>86.3</td></tr><tr><td>Qwen3.6-27B</td><td>71.6</td><td>92.9</td><td>84.6</td><td>85.2</td><td>83.6</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>69.4</td><td>93.6</td><td>85.2</td><td>83.4</td><td>82.9</td></tr><tr><td>Kimi-K2.6</td><td>69.2</td><td>88.8</td><td>78.8</td><td>75.4</td><td>78.0</td></tr><tr><td>DeepSeek-V4-Flash</td><td>67.2</td><td>90.4</td><td>86.6</td><td>82.0</td><td>81.6</td></tr><tr><td>MiniMax-M2.7</td><td>71.8</td><td>91.5</td><td>82.8</td><td>85.0</td><td>82.8</td></tr><tr><td>GLM-4.7</td><td>69.6</td><td>90.0</td><td>82.2</td><td>76.8</td><td>79.7</td></tr><tr><td>Gemini 3.1 FLite</td><td>72.4</td><td>95.0</td><td>86.6</td><td>80.8</td><td>83.7</td></tr><tr><td>DeepSeek-R1</td><td>68.8</td><td>90.0</td><td>79.6</td><td>79.6</td><td>79.5</td></tr><tr><td>Embedding models</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Octen-8B</td><td>70.0</td><td>85.4</td><td>99.4</td><td>89.4</td><td>86.1</td></tr><tr><td>Qwen3-E-8B</td><td>70.4</td><td>85.7</td><td>99.6</td><td>90.2</td><td>86.5</td></tr><tr><td>Qwen3-E-4B</td><td>72.0</td><td>84.8</td><td>99.8</td><td>89.4</td><td>86.5</td></tr><tr><td>Nemotron-8B</td><td>72.6</td><td>84.8</td><td>99.4</td><td>89.8</td><td>86.7</td></tr><tr><td>KaLM-12B</td><td>73.6</td><td>85.2</td><td>99.8</td><td>89.6</td><td>87.1</td></tr><tr><td>Jina-v5-S</td><td>68.0</td><td>85.2</td><td>99.6</td><td>87.4</td><td>85.1</td></tr><tr><td>SFR-2</td><td>67.4</td><td>84.8</td><td>99.8</td><td>89.0</td><td>85.3</td></tr><tr><td>Jina-v5-Nano</td><td>68.8</td><td>85.0</td><td>99.4</td><td>86.6</td><td>85.0</td></tr><tr><td>F2LLM-14B</td><td>72.0</td><td>84.8</td><td>99.6</td><td>88.6</td><td>86.3</td></tr><tr><td>GTE-Qwen2-7B</td><td>71.2</td><td>85.2</td><td>99.6</td><td>88.0</td><td>86.0</td></tr><tr><td>Linq-Mistral</td><td>71.0</td><td>84.8</td><td>99.8</td><td>88.6</td><td>86.1</td></tr><tr><td>Qwen3-E-0.6B</td><td>70.4</td><td>84.8</td><td>99.8</td><td>89.4</td><td>86.1</td></tr><tr><td>EmbGemma-300M</td><td>71.0</td><td>84.8</td><td>99.6</td><td>88.6</td><td>86.0</td></tr><tr><td>F2LLM-8B</td><td>72.0</td><td>84.8</td><td>99.6</td><td>88.0</td><td>86.1</td></tr><tr><td>F2LLM-4B</td><td>70.6</td><td>84.8</td><td>99.8</td><td>88.4</td><td>85.9</td></tr><tr><td>F2LLM-1.7B</td><td>71.0</td><td>84.8</td><td>99.6</td><td>89.0</td><td>86.1</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>73.0</td><td>84.8</td><td>99.4</td><td>89.2</td><td>86.6</td></tr><tr><td>F2LLM-0.6B</td><td>70.8</td><td>85.0</td><td>99.4</td><td>88.0</td><td>85.8</td></tr><tr><td>mE5-L-Inst</td><td>70.8</td><td>84.8</td><td>99.6</td><td>89.2</td><td>86.1</td></tr><tr><td>BGE-M3</td><td>71.6</td><td>84.8</td><td>99.6</td><td>87.6</td><td>85.9</td></tr><tr><td>mE5-L</td><td>64.0</td><td>85.0</td><td>98.4</td><td>88.2</td><td>83.9</td></tr><tr><td>Arctic-L-v2</td><td>64.2</td><td>84.8</td><td>99.6</td><td>86.8</td><td>83.9</td></tr><tr><td>mE5-B</td><td>65.6</td><td>84.8</td><td>98.2</td><td>88.6</td><td>84.3</td></tr><tr><td>mE5-S</td><td>62.2</td><td>85.2</td><td>99.2</td><td>89.0</td><td>83.9</td></tr><tr><td>E5-Mistral-7B</td><td>66.0</td><td>84.8</td><td>73.4</td><td>67.4</td><td>72.9</td></tr><tr><td>GritLM-7B</td><td>56.8</td><td>84.8</td><td>92.0</td><td>75.4</td><td>77.3</td></tr></table>

Table 10: Per-task scores: Pair Classification (AP / Accuracy).

<table><tr><td>Model</td><td>AILA</td><td>FQuAD</td><td>HC3Fin</td><td>ConsumerQA</td><td>PubHealth</td><td>TwtHjerne</td><td>Mean</td></tr><tr><td>LLMs</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gemini 3.1 Pro</td><td>14.5</td><td>92.0</td><td>71.0</td><td>86.0</td><td>90.0</td><td>33.4</td><td>64.5</td></tr><tr><td>Gemini 3 Flash</td><td>5.7</td><td>88.0</td><td>60.0</td><td>79.0</td><td>49.0</td><td>32.4</td><td>52.3</td></tr><tr><td>Qwen3.6-27B</td><td>14.2</td><td>96.0</td><td>57.0</td><td>88.0</td><td>86.0</td><td>33.0</td><td>62.4</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>12.9</td><td>95.0</td><td>55.0</td><td>86.0</td><td>83.0</td><td>30.5</td><td>60.4</td></tr><tr><td>Kimi-K2.6</td><td>13.2</td><td>91.0</td><td>56.0</td><td>80.0</td><td>67.0</td><td>33.0</td><td>56.7</td></tr><tr><td>DeepSeek-V4-Flash</td><td>13.3</td><td>83.0</td><td>50.0</td><td>76.0</td><td>64.0</td><td>33.4</td><td>53.3</td></tr><tr><td>MiniMax-M2.7</td><td>9.7</td><td>82.0</td><td>42.0</td><td>67.0</td><td>59.0</td><td>28.2</td><td>48.0</td></tr><tr><td>GLM-4.7</td><td>13.0</td><td>78.0</td><td>56.0</td><td>78.0</td><td>70.0</td><td>31.6</td><td>54.4</td></tr><tr><td>Gemini 3.1 FLite</td><td>10.2</td><td>84.0</td><td>46.0</td><td>77.0</td><td>45.0</td><td>31.9</td><td>49.0</td></tr><tr><td>DeepSeek-R1</td><td>11.9</td><td>57.0</td><td>43.0</td><td>62.0</td><td>64.0</td><td>30.6</td><td>44.7</td></tr><tr><td colspan="8">Embedding models</td></tr><tr><td>Octen-8B</td><td>23.2</td><td>69.0</td><td>66.0</td><td>67.0</td><td>81.0</td><td>29.9</td><td>56.0</td></tr><tr><td>Qwen3-E-8B</td><td>21.8</td><td>68.0</td><td>59.0</td><td>70.0</td><td>78.0</td><td>28.6</td><td>54.2</td></tr><tr><td>Qwen3-E-4B</td><td>21.0</td><td>61.0</td><td>56.0</td><td>64.0</td><td>75.0</td><td>24.8</td><td>50.3</td></tr><tr><td>Nemotron-8B</td><td>9.1</td><td>72.0</td><td>67.0</td><td>70.0</td><td>80.0</td><td>29.9</td><td>54.7</td></tr><tr><td>KaLM-12B</td><td>10.3</td><td>70.0</td><td>55.0</td><td>63.0</td><td>71.0</td><td>30.2</td><td>49.9</td></tr><tr><td>Jina-v5-S</td><td>14.7</td><td>64.0</td><td>48.0</td><td>59.0</td><td>77.0</td><td>25.5</td><td>48.0</td></tr><tr><td>SFR-2</td><td>11.2</td><td>67.0</td><td>57.0</td><td>56.0</td><td>74.0</td><td>27.6</td><td>48.8</td></tr><tr><td>Jina-v5-Nano</td><td>12.6</td><td>65.0</td><td>44.0</td><td>61.0</td><td>74.0</td><td>27.6</td><td>47.4</td></tr><tr><td>F2LLM-14B</td><td>12.2</td><td>63.0</td><td>67.0</td><td>54.0</td><td>85.0</td><td>31.5</td><td>52.1</td></tr><tr><td>GTE-Qwen2-7B</td><td>9.1</td><td>64.0</td><td>51.0</td><td>55.0</td><td>77.0</td><td>19.0</td><td>45.8</td></tr><tr><td>Linq-Mistral</td><td>10.6</td><td>68.0</td><td>54.0</td><td>66.0</td><td>78.0</td><td>29.1</td><td>51.0</td></tr><tr><td>Qwen3-E-0.6B</td><td>20.2</td><td>55.0</td><td>36.0</td><td>63.0</td><td>68.0</td><td>22.4</td><td>44.1</td></tr><tr><td>EmbGemma-300M</td><td>5.5</td><td>72.0</td><td>42.0</td><td>62.0</td><td>78.0</td><td>24.2</td><td>47.3</td></tr><tr><td>F2LLM-8B</td><td>8.9</td><td>63.0</td><td>63.0</td><td>51.0</td><td>85.0</td><td>28.9</td><td>50.0</td></tr><tr><td>F2LLM-4B</td><td>8.1</td><td>64.0</td><td>62.0</td><td>53.0</td><td>83.0</td><td>29.4</td><td>49.9</td></tr><tr><td>F2LLM-1.7B</td><td>5.4</td><td>66.0</td><td>55.0</td><td>54.0</td><td>82.0</td><td>28.8</td><td>48.5</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>5.3</td><td>58.0</td><td>44.0</td><td>59.0</td><td>76.0</td><td>21.2</td><td>43.9</td></tr><tr><td>F2LLM-0.6B</td><td>3.6</td><td>56.0</td><td>52.0</td><td>54.0</td><td>80.0</td><td>26.7</td><td>45.4</td></tr><tr><td>mE5-L-Inst</td><td>6.5</td><td>68.0</td><td>44.0</td><td>50.0</td><td>73.0</td><td>27.6</td><td>44.8</td></tr><tr><td>BGE-M3</td><td>7.4</td><td>62.0</td><td>40.0</td><td>58.0</td><td>74.0</td><td>24.2</td><td>44.3</td></tr><tr><td>mE5-L</td><td>4.9</td><td>69.0</td><td>40.0</td><td>51.0</td><td>73.0</td><td>23.9</td><td>43.6</td></tr><tr><td>Arctic-L-v2</td><td>3.3</td><td>61.0</td><td>42.0</td><td>59.0</td><td>76.0</td><td>26.1</td><td>44.6</td></tr><tr><td>mE5-B</td><td>3.2</td><td>64.0</td><td>28.0</td><td>47.0</td><td>68.0</td><td>23.5</td><td>38.9</td></tr><tr><td>mE5-S</td><td>3.6</td><td>57.0</td><td>26.0</td><td>43.0</td><td>64.0</td><td>20.9</td><td>35.8</td></tr><tr><td>E5-Mistral-7B</td><td>2.7</td><td>29.0</td><td>21.0</td><td>17.0</td><td>47.0</td><td>7.1</td><td>20.6</td></tr><tr><td>GritLM-7B</td><td>1.0</td><td>1.0</td><td>3.0</td><td>8.0</td><td>18.0</td><td>5.3</td><td>6.1</td></tr></table>

Table 11: Per-task scores: Retrieval (Recall@1).

<table><tr><td>Task</td><td>Lang.</td><td>N</td><td>Cls.</td><td>Metric</td><td>Source</td></tr><tr><td colspan="6">Classification (8 tasks)</td></tr><tr><td>ImdbCls</td><td>en</td><td>500</td><td>2</td><td>Acc.</td><td>Maas et al. (2011)</td></tr><tr><td>Banking77Cls</td><td>en</td><td>3k</td><td>77</td><td>Acc.</td><td>Casanueva et al. (2020)</td></tr><tr><td>AmazonCounterfactualCls</td><td>en, de, ja</td><td>809</td><td>2</td><td>Acc.</td><td>O&#x27;Neill et al. (2021)</td></tr><tr><td>MTOPDomainCls</td><td>en, de, fr</td><td>2k</td><td>11</td><td>Acc.</td><td>Li et al. (2021)</td></tr><tr><td>MassiveIntentCls</td><td>en, de, fr, ja</td><td>4k</td><td>60</td><td>Acc.</td><td>FitzGerald et al. (2022)</td></tr><tr><td>MassiveScenarioCls</td><td>en, de, fr, ja</td><td>3k</td><td>18</td><td>Acc.</td><td>FitzGerald et al. (2022)</td></tr><tr><td>ToxicConversationsCls</td><td>en</td><td>500</td><td>2</td><td>Acc.</td><td>Borkan et al. (2019)</td></tr><tr><td>TweetSentimentCls</td><td>en</td><td>500</td><td>3</td><td>Acc.</td><td>Muennighoff et al. (2023)</td></tr><tr><td colspan="6">Semantic Textual Similarity (10 tasks)</td></tr><tr><td>STSBenchmark</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Cer et al. (2017)</td></tr><tr><td>SICK-R</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Marelli et al. (2014)</td></tr><tr><td>STS12</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Agirre et al. (2012)</td></tr><tr><td>STS13</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Agirre et al. (2012)</td></tr><tr><td>STS14</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Agirre et al. (2012)</td></tr><tr><td>STS15</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Agirre et al. (2016)</td></tr><tr><td>STS16</td><td>en</td><td>500</td><td></td><td>Spearman</td><td>Agirre et al. (2016)</td></tr><tr><td>BIOSSES</td><td>en</td><td>100</td><td></td><td>Spearman</td><td>Soğancioğlu et al. (2017)</td></tr><tr><td>STS17</td><td>en, de, es, fr</td><td>1k</td><td></td><td>Spearman</td><td>Cer et al. (2017)</td></tr><tr><td>STS22v2</td><td>en, de, es, fr, ru, zh</td><td>2k</td><td></td><td>Spearman</td><td>Chen et al. (2022)</td></tr><tr><td colspan="6">Clustering (9 tasks)</td></tr><tr><td>RedditClustP2P</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>TwentyNewsgroupsV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>StackExchangeClustP2PV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>StackExchangeClustV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>ArxivClustP2P</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>ArxivClustS2S</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>BiorxivClustP2PV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>MedrxivClustP2PV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td>MedrxivClustS2SV2</td><td>en</td><td>1k</td><td></td><td>V-meas.</td><td>Muennighoff et al. (2023)</td></tr><tr><td colspan="6">Pair Classification (4 tasks)</td></tr><tr><td>SprintDuplicateQuestionsPC</td><td>en</td><td>500</td><td>2</td><td></td><td>Avg. Prec. Shah et al. (2018)</td></tr><tr><td>TwitterURLCorpusPC</td><td>en</td><td>500</td><td>2</td><td>Avg. Prec.</td><td>Lan et al. (2017)</td></tr><tr><td>LegalBenchPC</td><td>en</td><td>500</td><td>2</td><td>Avg. Prec.</td><td>Guha et al. (2023)</td></tr><tr><td>RTE3PC</td><td>de, en, fr, it</td><td>2k</td><td>2</td><td>Avg. Prec.</td><td>Giampiccolo et al. (2007)</td></tr><tr><td colspan="6">Retrieval (6 tasks)</td></tr><tr><td>AILAStatutes</td><td>en</td><td>50Q /82C</td><td></td><td>Recall@1</td><td>Enevoldsen et al. (2025)</td></tr><tr><td>FQuADRetrieval</td><td>fr</td><td>100 Q /269 C</td><td></td><td>Recall@1</td><td>d&#x27;Hoffschmidt et al. (2020)</td></tr><tr><td>HC3FinanceRetrieval</td><td>en</td><td>100Q /415C</td><td></td><td>Recall@1</td><td>Guo et al. (2023)</td></tr><tr><td>LegalBenchConsumerContractsQA en</td><td></td><td>100Q / 154C</td><td></td><td>Recall@1</td><td>Guha et al. (2023)</td></tr><tr><td>PublicHealthQA</td><td>en</td><td>100 Q / 172 C</td><td></td><td>Recall@1</td><td>HF dataset</td></tr><tr><td>TwitterHjerneRetrieval</td><td>da</td><td>77Q /262 C</td><td></td><td>Recall@1</td><td>Vejlgaard Holm et al. (2025)</td></tr></table>

Table 12: MTEB(LLM) task suite (37 tasks). N = held-out test samples (summed over languages for multilingual tasks); Q = queries, C = corpus documents. Multilingual tasks are evaluated per language and averaged. Held-out subsets (seed 42) derived from MTEB and MMTEB tasks (Muennighoff et al., 2023; Enevoldsen et al., 2025), hosted at mteb/11m-eval-\*. Token counts: Table 13.

Consumer-Contracts QA (legal). Yes/no questions over consumer-contract clauses. Pro (86.0) outperforms the best embedding (70.0), consistent with the need to interpret the clause.

Public-Health QA. Pro (90.0) edges the best embedding (85.0); factual health QA is partly captured by dense retrieval, narrowing the gap.

HC3-Finance QA. Pro (71.0) narrowly leads the best embedding (67.0) on financial QA retrieval.

TwitterHjerne (Danish social media). Retrieving relevant tweets given a query. Pro (33.4) slightly outperforms embeddings (best: 31.5); both handle keyword-centric retrieval similarly.

AILAStatutes (legal statute retrieval): where reasoning hurts. Matching case facts to relevant statutes. The best embedding (Octen-8B, 23.2) outperforms Pro (14.5): Pro over-retrieves

<table><tr><td colspan="3">Classification</td></tr><tr><td>Task</td><td>Test</td><td>Train (kNN)*</td></tr><tr><td>ImdbCls</td><td>132k</td><td>7,281k</td></tr><tr><td>Banking77Cls</td><td>38k</td><td>134k</td></tr><tr><td>AmazonCounterfactualCls</td><td>22k</td><td>258k</td></tr><tr><td>MTOPDomainCls</td><td>23k</td><td>393k</td></tr><tr><td>MassiveIntentCls</td><td>37k</td><td>433k</td></tr><tr><td>MassiveScenarioCls</td><td>30k</td><td>433k</td></tr><tr><td>ToxicConversationsCls</td><td>33k</td><td>3,239k</td></tr><tr><td>TweetSentimentCls</td><td>9k</td><td>483k</td></tr><tr><td>STS</td><td></td><td></td></tr><tr><td>STSBenchmark</td><td></td><td>12k</td></tr><tr><td>SICK-R</td><td></td><td>10k</td></tr><tr><td>STS12</td><td></td><td>14k</td></tr><tr><td>STS13</td><td></td><td>12k</td></tr><tr><td>STS14</td><td></td><td>12k</td></tr><tr><td>STS15</td><td></td><td>12k</td></tr><tr><td>STS16</td><td></td><td>14k</td></tr><tr><td>BIOSSES</td><td></td><td>7k</td></tr><tr><td>STS17</td><td></td><td>21k</td></tr><tr><td>STS22v2</td><td></td><td>1,866k</td></tr><tr><td>Clustering</td><td></td><td></td></tr><tr><td>RedditClustP2P</td><td></td><td>178k</td></tr><tr><td>TwentyNewsgroupsV2</td><td></td><td>8k</td></tr><tr><td>StackExchangeClustP2PV2</td><td></td><td>276k</td></tr><tr><td>StackExchangeClustV2</td><td></td><td>13k</td></tr><tr><td>ArxivClustP2P</td><td></td><td>226k</td></tr><tr><td>ArxivClustS2S</td><td></td><td>16k</td></tr><tr><td>BiorxivClustP2PV2</td><td></td><td>313k</td></tr><tr><td>MedrxivClustP2PV2</td><td></td><td>401k</td></tr><tr><td>MedrxivClustS2SV2</td><td></td><td>23k</td></tr><tr><td>PairClassification</td><td></td><td></td></tr><tr><td>SprintDuplicateQuestionsPC</td><td></td><td>14k</td></tr><tr><td>TwitterURLCorpusPC</td><td></td><td>18k</td></tr><tr><td>LegalBenchPC</td><td></td><td>37k</td></tr><tr><td>RTE3PC</td><td></td><td>114k</td></tr><tr><td>Task</td><td>Queries</td><td>Corpus</td></tr><tr><td></td><td></td><td></td></tr><tr><td>Retrieval AILAStatutes</td><td>33k</td><td></td></tr><tr><td>FQuADRetrieval</td><td>1k</td><td>37k 59k</td></tr><tr><td>HC3FinanceRetrieval</td><td>1k</td><td>89k</td></tr><tr><td></td><td>2k</td><td></td></tr><tr><td>LegalBenchConsumerContractsQA</td><td></td><td>83k</td></tr><tr><td>PublicHealthQA</td><td>1k</td><td>28k</td></tr><tr><td>TwitterHjerneRetrieval</td><td>4k</td><td>10k</td></tr></table>

Table 13: Token budget per task (GPT-4o tokenizer; raw text). Actual counts vary by ±15–40% across model vocabularies. \*Train split is the kNN reference corpus processed only by embedding models; LLMs process only the test split. Corpus-in-context formatting adds \~20 tokens per document for LLM retrieval.  
broadly associated provisions, while cosine similarity naturally produces a narrower, more precise ranking.

## D Per-Category Rankings and Figures

This section provides task- and model-level figures, followed by category-level rankings.   
Figure 6 is the task-level view of the paper's central claim and is referenced from §4.2.

<table><tr><td>Task</td><td>Gemini 3 Flash</td><td>Gemini 3 Flash (low)</td><td>∆</td><td>Think↓</td><td>Best Emb.</td></tr><tr><td>Retrieval</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AILAStatutes</td><td>5.7</td><td>12.0</td><td>+6.3</td><td></td><td>23.2</td></tr><tr><td>FQuADRetrieval</td><td>88.0</td><td>92.0</td><td>+4.0</td><td>54%</td><td>72.0</td></tr><tr><td>HC3FinanceRetrieval</td><td>60.0</td><td>66.0</td><td>+6.0</td><td>87%</td><td>67.0</td></tr><tr><td>LegalBenchConsumerContractsQA</td><td>79.0</td><td>83.0</td><td>+4.0</td><td>85%</td><td>70.0</td></tr><tr><td>PublicHealthQA</td><td>49.0</td><td>66.0</td><td>+17.0</td><td>94%</td><td>85.0</td></tr><tr><td>TwitterHjerneRetrieval</td><td>32.4</td><td>33.4</td><td>+1.1</td><td></td><td>31.5</td></tr></table>

Table 14: Reduced-thinking ablation (Gemini 3 Flash with reasoning\_effort=low vs. default) on the MTEB(LLM) retrieval tasks. Think ↓ = reduction in thinking tokens vs. default. Reducing thinking by 54–94% improves all six retrieval scores in this ablation.
<table><tr><td>Task</td><td>Classes</td><td>Zero-shot</td><td>5-shot</td><td>∆</td><td>Best Emb.</td></tr><tr><td>IMDB</td><td>2</td><td>0.976</td><td>0.974</td><td>-0.002</td><td>0.976</td></tr><tr><td>ToxicConversations</td><td>2</td><td>0.900</td><td>0.838</td><td>-0.062</td><td>0.810</td></tr><tr><td>TweetSentiment</td><td>3</td><td>0.700</td><td>0.682</td><td>-0.018</td><td>0.710</td></tr><tr><td>Banking77</td><td>77</td><td>0.831</td><td>0.165</td><td>-0.666</td><td>0.960</td></tr></table>

Table 15: Few-shot classification ablation (Flash, 5 in-context examples vs. zero-shot). Bold = best score per task across all methods. Five-shot prompting matches zero-shot performance on the 2–3 class tasks and lowers Banking77 performance, where the prompt contains five examples for 77 labels.

## D.1 Task- and Model-Level Figures

The figures below show task-level performance, throughput, category profiles, and crossmodel correlations.

Task Performance Ranges: Visualizing LLM Capability Overlaps  
![](images/aed69cffa67af6aea7a28fcef5fdc7c3aec039cf55b9e360ec7e78c2ce38fc5e.jpg)  
Figure 6: Per-task performance. Horizontal bars show the full embedding score range (min to max across 26 models) for each MTEB(LLM) task, grouped by category. Circle = best embedding; diamond = best LLM on that task, taken over all ten (six different LLMs hold it across the suite, Gemini 3.1 Pro on 21 of 37 tasks). Where the diamond falls inside the bar, some embedding model already matches the best LLM: this holds on 7 of 8 classification tasks and 7 of 10 STS tasks, but on only 1 of 6 retrieval tasks.

![](images/d4ec2f9c8a450c53751e21c6da19be73480fd0bf05fd551d8a0a92dcd9222a27.jpg)  
Figure 7: Same-hardware inference throughput. Two open-weight LLMs and seven representative embedding models (118M–14B) served on one H100 (tokens/second, log scale); the full 26-model embedding throughput is in Table 18. Even the slowest embedding runs ${ \sim } 2 . 5 \times$ faster than the fastest LLM; the fastest runs \~736× faster.

Capability Profile Across Task Categories  
![](images/aa586f417bb31aa5d1481756abdb181546d8fa44cd5c6025b16e0a29722e8420.jpg)  
Figure 8: Capability profiles across five task categories. Gemini 3.1 Pro (red), Qwen3.6- 27B (amber), and Octen-8B (blue). The LLMs lead on retrieval, while Octen-8B leads on classification and remains competitive across categories.

Model Capability Correlation Across Tasks

![](images/14ab96a0c9ef68a10555bebfb8e40b846c6604b571bb8304fbafb11453fc1f81.jpg)  
Figure 9: Cross-model task-behaviour correlation. Pearson correlations based on per-task MTEB(LLM) scores. LLMs and embedding models form visually distinct clusters: LLMs are strongly intercorrelated while embedding models cluster by architecture family. Lower inter-paradigm correlations are consistent with their different category profiles.

## D.2 Per-Category Model Rankings

Figures 10–15 rank all 36 evaluated models overall and within each of the five MTEB task categories. They show category-dependent performance: LLMs lead on retrieval, embedding models lead on classification, and the two paradigms are statistically indistinguishable on clustering, STS, and pair classification. Flash-Lite ranks near the bottom in every category, below the reasoning-capable Gemini models.

MTEB(LLM) Leaderboard: Overall

![](images/320f81cea80b6fdb83a8fad383c0edc35aa7a1123486f0b9d89f99a312a5a8fa.jpg)  
Figure 10: Overall rankings. Pro (77.6) narrowly leads Octen-8B (77.2); the difference lies within statistical noise (p = 0.85). Flash-Lite (64.5) ranks near the bottom.

MTEB(LLM) Leaderboard: Classification  
![](images/91cc2073aed10468ead1f701be329ba09f3b5097f147a1b3e639d654f19d38b4.jpg)  
Figure 11: Classification rankings. SFR-2 ranks first (90.8); Pro scores 85.2 and ranks below ten embedding models.

MTEB(LLM) Leaderboard: Clustering  
![](images/0b9e54ad2e0e37c2552d5df5b18c3c6d2a85417d976b47f850233ae312c6cfb2.jpg)  
Figure 12: Clustering rankings. Statistical tie; SFR-2 (66.7) edges Pro (66.6). Flash-Lite scores 21.7.  
MTEB(LLM) Leaderboard: STS

![](images/fd2ab9ecb6b140027669bcca4921fabcca4a0af27b0995115c61c8094445b5f4.jpg)  
Figure 13: STS rankings. Qwen3-E-4B (88.8) and Pro (88.5) are essentially tied; no significant paradigm difference (p = 0.74).

MTEB(LLM) Leaderboard: PairClassification  
![](images/96d190b95816386001512b80018b0b963b6d15bd9319347e4269259047a412f4.jpg)  
Figure 14: Pair Classification rankings. KaLM-12B (87.1) outperforms Pro (83.2).

MTEB(LLM) Leaderboard: Retrieval

![](images/3b1540790b693f80409c13ced7ebd92bad05d2e54d6d26f4781899a8de122dbe.jpg)  
Figure 15: Retrieval rankings. Pro leads the best embedding (64.5 vs. 56.0) and ranks first on five of six tasks.

## D.3 Full Retrieve-then-Rerank Matrix

Table 16 reports the complete matrix summarised by Figure 3 (§4.3): all four first-stage retrievers crossed with every cross-encoder and LLM listwise reranker, on BEIR and BRIGHT.

## E Qualitative Error Analysis

We present model predictions from the evaluation files to illustrate differences between LLM and embedding approaches. The examples cover classification ambiguity, STS calibration, retrieval precision, and differences between Pro and Flash.

## E.1 Classification: Over-Interpretation vs. Anchored Reference

Several classification errors involve ambiguous inputs for which the LLM infers an intent that differs from the dataset convention. With kNN, an embedding classifier can anchor its prediction to similar labelled examples, which helps on fine-grained tasks such as Banking77 and MassiveIntent.

Banking77 (77 classes): "1ocate my card."

Input: "How do I locate my card?"

Pro reasoning: "The user is asking how to locate their card, indicating that they have misplaced it or it is missing. This aligns with the intent for a lost or stolen card."

Pro prediction: lost\_or\_stolen\_card

Flash prediction: card\_arrival

Flash predicts the gold label; Pro predicts lost\_or\_stolen\_card. The LLM infers "locate" implies loss; Flash's lower-depth reasoning follows the surface-level reading. In the training data, "locate my card" aligns with delivery tracking. The kNN classifier matches to examples like "Where is my card?" and correctly predicts card\_arrival.

Banking77: "know when my card will arrive."

Input: "Is there a way to know when my card will arrive?"

Pro reasoning: "The user is asking for information regarding the timeframe or expected date for their card's delivery, which aligns with the card\_delivery\_estimate category."

<table><tr><td>First stage</td><td>Pure</td><td>BGE-Gemma</td><td>Qwen3-RR-4B</td><td>Qwen3-RR-8B</td><td>Qwen3.6-27B†</td><td>Qwen3.6-35B†</td></tr><tr><td colspan="7">BRIGHT (reasoning)</td></tr><tr><td>BM25</td><td>10.2</td><td>18.2</td><td>22.0</td><td>21.9</td><td>24.2</td><td>23.5</td></tr><tr><td>BGE-large</td><td>15.9</td><td>18.9</td><td>22.7</td><td>21.5</td><td>27.6</td><td>26.1</td></tr><tr><td>GTE-MC-v1</td><td>15.9</td><td>19.2</td><td>23.9</td><td>23.0</td><td>28.8</td><td>27.9</td></tr><tr><td>Qwen3-E-8B</td><td>22.3</td><td>20.6</td><td>26.4</td><td>24.6</td><td>35.1</td><td>33.6</td></tr><tr><td colspan="7">BEIR (semantic)</td></tr><tr><td>BM25</td><td>39.8</td><td>50.5</td><td>53.2</td><td>52.5</td><td>52.1</td><td>51.1</td></tr><tr><td>BGE-large</td><td>52.7</td><td>55.3</td><td>58.6</td><td>57.6</td><td>57.6</td><td>56.2</td></tr><tr><td>GTE-MČ-v1</td><td>53.7</td><td>55.0</td><td>58.5</td><td>58.5</td><td>57.5</td><td>56.9</td></tr><tr><td>Qwen3-E-8B</td><td>63.1</td><td>56.8</td><td>60.3</td><td>59.5</td><td>58.9</td><td>58.9</td></tr></table>

Table 16: Retrieve-then-rerank matrix. Average nDCG@10 for first-stage retrievers crossed with cross-encoder and LLM listwise (†) rerankers over 7 BRIGHT and 5 BEIR tasks. Bold = best per row. An LLM reranker improves Qwen3-E-8B on BRIGHT (22.3→35.1); on BEIR, the embedding alone scores highest (63.1).
<table><tr><td>Pro prediction: card_delivery_estimate</td><td>Incorrect (gold: card_arrival)</td></tr><tr><td>Flash prediction: card_delivery_estimate</td><td>Incorrect (gold: card_arrival)</td></tr></table>

Both models make the same error here, illustrating the ambiguity between “arrival" and "delivery estimate" without labelled context. The kNN classifier uses the full 77-class reference set to resolve such near-neighbour cases.

MassiveIntent (60 classes): “remind me about my schedule."

Input (en): "remind me about my schedule for the afternoon"

Pro reasoning: "The user is asking to be reminded about their schedule for the afternoon, which involves checking or querying their calendar events."

"Remind me" is an action verb (set a reminder) but the LLM interprets it as informationseeking (query). This example illustrates the ambiguity between action and query intents.

MassiveScenario (German): "wir brauchen rosa."

Input (de): "wir brauchen rosa" (we need pink)

Pro reasoning: "The utterance 'wir brauchen rosa' (we need pink) implies adding an item ('rosa' / pink) to a shopping list or note-taking application.'

Pro prediction: lists

The LLM correctly reasons about the surface form but miscategorises the scenario: iot refers to controlling physical smart-home devices, not arbitrary colour references.

Input (de): "wie ist die einstellung für die helligkeit an meinem bildschirm" (what is the brightness setting on my screen)

Pro reasoning: "The user is asking about the brightness setting on their screen, which involves controlling or checking the state of a device. This falls under the IoT category."

Incorrect (gold: general)

Both errors reflect the fine-grained scenario taxonomy: iot refers to controlling physical smart-home devices, not arbitrary brightness settings or colour references. The LLM correctly identifies device-related surface features but miscategorises the scenario.

AmazonCounterfactual: hypothetical "would be great."

Input: "Would be great to have on my tablet instead of carrying the book." Pro reasoning: "The review expresses a hypothetical situation ('have on my tablet') that has not actually happened, indicated by the phrase 'Would be great'. This describes an alternative condition and its potential consequence."

Pro prediction: counterfactual Incorrect (gold: not-counterfactual)

The phrase "would be" supports the LLM's hypothetical reading. The dataset convention labels forward-looking wishes as not-counterfactual. This may reflect a mismatch between a plausible linguistic interpretation and the dataset's annotation convention.

When LLMs succeed on classification.

Input: "I still have not received my new card, I ordered over a week ago."   
Pro prediction: card\_arrival Correct

In this less ambiguous example, the LLM matches the gold annotation.

## E.2 STS: Calibration and Semantic Precision

The examples below show two kinds of disagreement between LLM scores and STS annotations: sensitivity to fine distinctions and higher scores for inferred mechanistic equivalence.

## STSBenchmark: agreement on a simple paraphrase.

S1: “open again, open anew"S2: “open again or anew."   
Pro: 5.0 Gold: 5.0 |∆|: 0.0

The LLM score matches the annotation for this near-identical paraphrase.

## STSBenchmark: conservative scoring on ideological distinctions.

S1: “... a shift from a muslim identity to an islamist identity."   
S2: "... to change a muslim identity for an islamic identity."   
Pro: 4.5 Gold: 5.0 |∆|: 0.5

The LLM distinguishes “Islamist" (political ideology) from “Islamic" (religious adjective); the gold standard treats them as equivalent. This linguistically defensible distinction reduces agreement with the benchmark annotation.

## BIOSSES: conservative scoring on biomedical paraphrases.

S1: "It has recently been shown that Craf is essential for Kras G12D-induced NSCLC."

S2: "It has recently become evident that Craf is essential for the onset of Kras-driven non-small cell lung cancer."

Pro reasoning: "Both sentences state that Craf is essential for the development of Kras-driven NSCLC. Sentence 1 specifies the particular mutation (Kras G12D), whereas Sentence 2 is slightly more general."

```yaml
Pro: 3.8 Gold: 4.0 |∆|: 0.2
```

The LLM correctly notes the specificity difference (“Kras G12D" vs. “Kras-driven") and downweights accordingly. The gold annotation gives 4.0, treating the two as highly similar; the LLM's 3.8 is slightly more conservative.

## BIOSSES: LLM over-scores on mechanistic equivalence.

S1: "More recently, IDH mutations and resultant 2-hydroxyglutarate (2HG) production in leukemia cells were found to impair TET2 function.'

S2: "It has also been recently reported that mutations of the isocitrate dehydrogenase genes IDH1 and IDH2 inhibit TET catalytic activity."

Pro reasoning: "Both sentences describe the exact same biochemical pathway: mutations in IDH genes lead to the production of 2-hydroxyglutarate (2-HG), which in turn inhibits TET2 enzymatic activity."

```yaml
Pro: 3.8 Gold: 3.2 |∆|: 0.6
```

The LLM identifies a mechanistic relationship and scores the pair 3.8, compared with the gold score of 3.2. Expert adjudication would be needed to determine whether this difference reflects model calibration or annotation criteria.

## E.3 Retrieval: Reasoning Depth vs. Geometric Precision

The retrieval examples show how joint query-document processing can help on readingcomprehension tasks and how broader associations can hurt precise statute retrieval.

FQuAD: reading comprehension in context. On French reading-comprehension retrieval, the LLM reads each candidate passage in the prompt and selects the one that answers the query, reaching Recall@1 = 92.0 versus 72.0 for the best embedding. The LLM processes the query jointly with the candidate passages, which may explain part of the observed advantage.

## AILAStatutes: where reasoning hurts.

Query: Legal case about a bank employee terminated for misconduct

Pro: Returns 14 documents spanning broad legal concepts (contract law, employment law, due process). Gold: only 4 specific statutes. The LLM's associative reasoning retrieves plausible but non-relevant provisions.

Octen-8B: Returns a narrower, more precise set via cosine similarity with legal embeddings.

In this example, Pro retrieves plausible but irrelevant provisions, while the embedding model produces a narrower ranking.

## TwitterHjerne: corpus-in-context precision.

Query (Danish): "Sønnen vil gerne lave #pebernødder. De par gange jeg har prøvet det, blev de kun OK. Er der nogen, der kan anbefale en opskrift? #twitterhjerne" ("My son would like to make peppernuts. The few times I've tried, they were only OK.'Can anyone recommend à recipe? #twitterhjerne")

Pro: Returns documents [1, 2]: exact match. Recall@1 = 1.0 on this query.

Best embedding: Recall@1 = 31.5 over the full task.

For this query with explicit topical keywords, corpus-in-context retrieval returns the correct tweet first. The overall task-level advantage for Pro (33.4 vs. 31.5 for embeddings) is modest, reflecting that both paradigms handle keyword-centric retrieval similarly.

## E.4 Model Scale Effects: Flash vs. Pro on the Same Input

Banking77: Flash succeeds where Pro over-interprets. As shown above, Flash correctly predicts card\_arrival for "How do I locate my card?" while Pro predicts lost\_or\_stolen\_card. In this example, more reasoning does not improve the prediction. Pro's extended chain-of-thought introduces an interpretive step (“locate implies misplacement") that Flash's lighter-weight reasoning bypasses. Aggregate scores provide context: Pro remains slightly stronger than Flash on classification overall.

Retrieval scale effects. On FQuAD reading-comprehension retrieval, all three Gemini models outscore the best embedding, with a clear quality ladder:

• Pro (Recall@1 = 92.0): reads and matches passages precisely.

• Flash (Recall@1 = 88.0): captures most of the advantage.

• Flash-Lite (Recall@1 = 84.0): still well above embeddings, even without reasoning.

• Best embedding (Recall@1 = 72.0): limited to single-vector matching.

Flash-Lite already outperforms the best embedding on this task; Pro adds a modest gain at much higher cost. This result is consistent with the reduced-thinking ablation (§4.6).

## Summary. The examples illustrate four patterns:

1. Ambiguous label boundaries: LLM predictions can differ from dataset conventions; kNN anchors decisions to labelled examples.

2. STS calibration: model scores can differ from annotations when they attend to different semantic details.

3. Retrieval breadth: joint processing can help reading comprehension but can also retrieve broadly related material.

4. Model scale: the value of additional capacity or reasoning varies across examples and must be weighed against cost.

Estimating the frequency of these patterns requires a systematic error analysis.

## F Extended Related Work

## F.1 LLMs as Text Encoders and Judges

The boundary between generative LLMs and discriminative embedding models has blurred considerably. LLM-as-a-judge frameworks (Zheng et al., 2023) repurpose frozen LLMs for evaluation. Sun et al. (2ǒ23b) find that zero-shot LLM classification underperforms fine-tuned models on standard benchmarks and propose chain-of-thought prompting to narrow the gap. Fine-tuned encoder models outperform zero-shot frontier LLMs across the classification benchmarks tested by Bucher & Martini (2024), with margins of 5 to over 75 F1 points depending on task granularity. This is consistent with our 5.6-point advantage for kNN-equipped embedding pipelines. The LOFT benchmark (Lee et al., 2024) evaluates long-context LLMs on retrieval by placing entire corpora in the prompt (corpus-in-context prompting); our retrieval evaluation adopts the same protocol.

On the representational side, several lines of work extract embeddings from LLMs. SGPT (Muennighoff, 2022) uses weighted mean pooling over GPT decoder states. Jiang et al. (2023) use in-context learning to extract sentence embeddings from decoder-only LLMs, while BehnamGhader et al. (2024) adapt decoder-only LLMs into sentence encoders. Springer et al. (2025) show that repeating the input can improve these embeddings by enabling effective bidirectional attention. GritLM (Muennighoff et al., 2024) goes further, unifying generation and representation in a single model that achieved strong performance on both MTEB and generative benchmarks at the time of its release.

These methods combine elements of both paradigms. At inference time, they produce fixed-length vectors compared via cosine similarity, so our cost accounting treats them as embedding models. Several of our 26 evaluated models (e.g., E5-Mistral, GritLM) are built on LLM backbones and incorporate LLM-derived representations.

HUME (Assadi et al., 2025) also benchmarks LLMs as annotators on MTEB tasks and finds that they fall short of human judgment on reranking. This complements our finding that zero-shot LLM classifiers underperform embedding pipelines with kNN access to labelled references. Our study adds a controlled comparison on identical tasks, data, and metrics, together with exact cost accounting

## F.2 Frontier LLMs and Open-Weight Reasoning Models

LLMs have advanced rapidly since GPT-4 (OpenAI, 2023). OpenAI's o1 (OpenAI, 2024) demonstrated extended chain-of-thought reasoning at inference time using reinforcement learning. At the time of our evaluation, closed-source models with controllable reasoning included Gemini 3.1 Pro, GPT-5 (Singh et al., 2025), and Claude Opus 4.6 (Anthropic, 2026) These models support extended thinking tokens billed separately from output; newer models have since appeared (e.g. GPT-5.4). Open-weight alternatives include Llama 3 (Grattafiori et al., 2024), Qwen3 (Yang et al., 2025), and DeepSeek-R1 (DeepSeek-AI et al. 2025). DeepSeek-R1 shows that reinforcement-learning-based training cān elicit strong reasoning at lower inference cost. Gemma 3 (Gemma Team et al., 2025) is a compact multilingual model family.

We evaluate ten LLMs across six families (§3), including DeepSeek-R1 and open-weight Qwen3.6, GLM, Kimi, and MiniMax models alongside Gemini 3. The cost and task-level patterns recur across this set, although their magnitude varies by model. GPT-5 (Singh et al., 2025) and Claude Opus 4.6 (Anthropic, 2026) remain to be evaluated. The same evaluation framework can accommodate these and future models.

## F.3 Cross-Encoders and Late-Interaction Retrieval

Cross-encoder rerankers (models that jointly encode query and document to produce a relevance score) and late-interaction models such as ColBÉRT (Khattab & Zaharia, 2020) occupy a middle ground between bi-encoder speed and LLM reasoning depth. We evaluate this middle ground directly (§4.3) by crossing four first-stage retrievers with cross-encoder and LLM listwise rerankers on BEIR and BRIGHT (Sun et al., 2023a; Weller et al., 2025). Reranking improves reasoning-heavy retrieval, while the strongest embedding alone remains best on semantic retrieval.

Li et al. (2024) compare RAG with long-context LLMs on QA tasks. Long-context LLMs achieve higher quality at greater token cost, while a Self-Route hybrid retains 93–100% of their quality at 38–61% of the token cost, depending on the model. This result supports the hybrid recommendation in §5.

## G Extended Limitations

We expand on the six limitations summarised in §5.

(i) LLM coverage. The ten evaluated LLMs span six families, dense and mixture-of-experts architectures, and reasoning and instruct variants. They provide a snapshot of a fast-moving frontier. At current public rates, the Pareto frontier includes Gemini 3.1 Pro and the leading embeddings; the lowest-cost LLM evaluated scores below the best embedding. Closed APIs not yet covered (GPT-5 (Singh et al., 2025) and Claude Opus 4.6 (Anthropic, 2026)) may have different cost-capability profiles, so the numerical conclusions should be updated as new evaluations become available. The released evaluation pipeline makes this straightforward to extend as new models are released.

(ii) Supervision asymmetry. Classification uses labelled-reference kNN for embeddings and zero-shot prompting for LLMs. This setup reflects deployments with scarce labelled data and gives the two pipelines different supervision. Five examples preserve performance on the small-label tasks and reduce Banking77 performance (§4.7).

(iii) Small-corpus retrieval. The corpus-in-context protocol places entire corpora (82–415 documents) in the LLM prompt, a setting that grants the LLM three structural advantages: the corpus fits entirely in context, prompt caching amortises input cost across queries, and the model can reason globally over the full document set. Each is a property of the protocol rather than a measured effect; we did not vary corpus size, so we do not quantify how retrieval quality would change at larger scale. At BEIR or production scale, placing the full corpus in context is generally impractical; an indexed first stage is needed, so the small-corpus cost gap is a lower bound for this protocol. We complement corpus-incontext retrieval with bi-encoder, cross-encoder, and LLM-reranker comparisons on BEIR and BRIGHT (§4.3). Reranking benefits reasoning-heavy retrieval; the strongest embedding retains the lead on semantic retrieval (Sun et al., 2023a; Weller et al., 2025). nDCG@k results for the small-corpus setting are available in the released data files and preserve the paradigm-level ordering.

(iv) Reasoning-control granularity. Reasoning controls differ by provider: Gemini's reasoning\_effort=low provides a soft hint, while the open models support fully disabled reasoning. We use both settings. Reduced reasoning preserves or improves retrieval for four of six models across five families (Figure 4b); token accounting varies by provider

(v) Throughput comparison. We serve two open-weight LLMs (Qwen3.6-27B and Qwen3.6-35B-A3B) and the embedding models on the same H100, removing API rate limits and hardware choice as confounds. The resulting 2.5–736× gap is consistent with the difference between autoregressive decoding and a single encoder pass.

(vi) Task weighting sensitivity. Our aggregate metric weights all MTEB(LLM) tasks equally regardless of dataset size, task difficulty, or practical importance. Alternative weighting schemes (by dataset size, normalised z-scores, or task-family weighting) may shift aggregate conclusions. Aggregate numbers should therefore be interpreted alongside the reported category- and task-level results. Alternative aggregation methods such as the Borda count used by MMTEB (Enevoldsen et al., 2025) are applicable to MTEB(LLM) and may produce different paradigm orderings.

## H Cost and Throughput Details

This section reports token usage, embedding throughput, cost sensitivity, and the cost methodology underlying §4.4–4.5.

## H.1 Detailed LLM Token Usage

Table 17 provides the complete token-level breakdown for each LLM across all MTEB(LLM) tasks. For each model, we report input, standard-output, cached, reasoning, and total tokens, together with the cost decomposition. Gemini reports reasoning separately, whereas OpenRouter includes it in the completion count. The reasoning-heavy LLMs generate 8–26M reasoning tokens (Qwen3.6-27B 26M, GLM-4.7 25M, Kimi-K2.6 22M, Gemini 3 Flash 14M, Pro 9M), compared with 1–3M standard-output tokens. Reasoning accounts for 28–81% of total inference cost across these models: 81% for Qwen3.6-27B and 28% for DeepSeek-V4- Flash. The non-reasoning Flash-Lite costs \$6.85 and ranks below most embeddings.

<table><tr><td></td><td colspan="4">Tokens (M)</td><td colspan="4">Cost (USD)</td></tr><tr><td>Model</td><td> $\mathrm { I n } _ { n c }$ </td><td>Cached</td><td>Out</td><td>Think</td><td> $\mathrm { I n } _ { n c }$ </td><td>Cached</td><td>Out+Th</td><td>Total</td></tr><tr><td>Gemini 3.1 Pro</td><td>12.2</td><td>27.7</td><td>1.8</td><td>8.6</td><td>$24.39</td><td>$5.54</td><td>$124.21</td><td>$154.14</td></tr><tr><td>Gemini 3 Flash</td><td>12.4</td><td>27.5</td><td>1.8</td><td>14.3</td><td>$6.19</td><td>$1.38</td><td>$48.30</td><td>$55.87</td></tr><tr><td>Qwen3.6-27B</td><td>39.5</td><td>0.0</td><td>2.7</td><td>26.1</td><td>$11.46</td><td>$0.00</td><td>$91.97</td><td>$103.43</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>39.5</td><td>0.0</td><td>2.5</td><td>26.0</td><td>$5.53</td><td>$0.00</td><td>$28.52</td><td>$34.06</td></tr><tr><td>Kimi-K2.6</td><td>40.1</td><td>0.0</td><td>2.6</td><td>21.7</td><td>$27.43</td><td>$0.00</td><td>$83.27</td><td>$110.70</td></tr><tr><td>DeepSeek-V4-Flash</td><td>16.8</td><td>22.6</td><td>2.1</td><td>4.5</td><td>$1.65</td><td>$0.22</td><td>$1.29</td><td>$3.16</td></tr><tr><td>MiniMax-M2.7</td><td>37.9</td><td>0.0</td><td>1.6</td><td>10.0</td><td>$10.58</td><td>$0.00</td><td>$13.93</td><td>$24.52</td></tr><tr><td>GLM-4.7</td><td>38.8</td><td>0.0</td><td>2.4</td><td>24.9</td><td>$15.54</td><td>$0.00</td><td>$47.77</td><td>$63.31</td></tr><tr><td>Gemini 3.1 Flash Lite</td><td>16.5</td><td>26.9</td><td>1.4</td><td>0.0</td><td>$4.13</td><td>$0.67</td><td>$2.05</td><td>$6.85</td></tr><tr><td>DeepSeek-R1</td><td>39.6</td><td>0.0</td><td>1.2</td><td>10.6</td><td>$27.73</td><td>$0.00</td><td>$29.65</td><td>$57.38</td></tr></table>

Table 17: Detailed LLM token usage and cost across all MTEB(LLM) tasks. ${ \mathrm { I n } } _ { n c } = { \mathrm { n o n } } -$ cached input (cached billed at 10% of input rate); Think = reasoning tokens (billed at output rate); Out+Th combines standard output and thinking cost.

## H.2 Embedding Throughput and Cost

Table 18 reports the throughput (tokens/second) and derived cost per million tokens for each of the 26 embedding models, benchmarked on a single NVIDIA H100 80GB HBM3 GPU at sequence length 512 with maximum batch size. Costs range from \$0.00016/MTok (mE5-smalī, 118M parameters, 4.3M tok/s) to \$0.047/MTok (F2LLM-v2-14B, 14.7K tok/s). Total benchmark cost spans 291 × across the embedding models, compared with a 1,431× gap between the best embedding and Gemini 3.1 Pro.

Each model's total benchmark cost uses its own tokenizer's token count (4.4–5.5M tokens across vocabularies), avoiding tokenizer mismatch.
<table><tr><td>Model</td><td>Params</td><td>Tok/s</td><td>$/MTok</td><td>Score</td></tr><tr><td>mE5-S</td><td>118M</td><td>4,314,796</td><td>0.0002</td><td>63.1</td></tr><tr><td>mE5-B</td><td>278M</td><td>1,910,005</td><td>0.0004</td><td>64.4</td></tr><tr><td>mE5-L-Inst</td><td>560M</td><td>640,842</td><td>0.0011</td><td>69.7</td></tr><tr><td>BGE-M3</td><td>568M</td><td>640,425</td><td>0.0011</td><td>66.5</td></tr><tr><td>Arctic-L-v2</td><td>568M</td><td>640,405</td><td>0.0011</td><td>65.3</td></tr><tr><td>mE5-L</td><td>560M</td><td>640,126</td><td>0.0011</td><td>65.9</td></tr><tr><td>EmbGemma-300M</td><td>308M</td><td>374,988</td><td>0.0018</td><td>72.2</td></tr><tr><td>Jina-v5-Nano</td><td>212M</td><td>327,790</td><td>0.0021</td><td>73.9</td></tr><tr><td>F2LLM-0.6B</td><td>596M</td><td>189,653</td><td>0.0036</td><td>69.9</td></tr><tr><td>Qwen3-E-0.6B</td><td>596M</td><td>175,692</td><td>0.0039</td><td>72.5</td></tr><tr><td>GTE-Qwen2-1.5B</td><td>1.5B</td><td>133,567</td><td>0.0052</td><td>70.8</td></tr><tr><td>F2LLM-1.7B</td><td>1.7B</td><td>106,406</td><td>0.0065</td><td>71.7</td></tr><tr><td>Jina-v5-S</td><td>596M</td><td>91,658</td><td>0.0075</td><td>74.4</td></tr><tr><td>Qwen3-E-4B</td><td>4.0B</td><td>46,080</td><td>0.0150</td><td>75.9</td></tr><tr><td>F2LLM-4B</td><td>4.0B</td><td>40,157</td><td>0.0172</td><td>71.9</td></tr><tr><td>GTE-Qwen2-7B</td><td>7.1B</td><td>35,419</td><td>0.0195</td><td>73.1</td></tr><tr><td>Octen-8B</td><td>7.6B</td><td>29,313</td><td>0.0236</td><td>77.2</td></tr><tr><td>Qwen3-E-8B</td><td>7.6B</td><td>29,262</td><td>0.0236</td><td>77.0</td></tr><tr><td>SFR-2</td><td>7.1B</td><td>28,196</td><td>0.0245</td><td>73.9</td></tr><tr><td>Nemotron-8B</td><td>7.5B</td><td>27,808</td><td>0.0249</td><td>75.3</td></tr><tr><td>GritLM-7B</td><td>7.2B</td><td>27,730</td><td>0.0249</td><td>51.2</td></tr><tr><td>Linq-Mistral</td><td>7.1B</td><td>27,677</td><td>0.0250</td><td>72.8</td></tr><tr><td>E5-Mistral-7B</td><td>7.1B</td><td>27,522</td><td>0.0251</td><td>52.2</td></tr><tr><td>F2LLM-8B</td><td>7.6B</td><td>25,957</td><td>0.0266</td><td>72.2</td></tr><tr><td>KaLM-12B</td><td>11.8B</td><td>19,244</td><td>0.0359</td><td>74.8</td></tr><tr><td>F2LLM-14B</td><td>14.0B</td><td>14,665</td><td>0.0472</td><td>73.3</td></tr></table>

Table 18: Embedding throughput on a single NVIDIA H100 80GB (median tokens/s over the benchmark), per-MTok cost at \$2.49/hr spot, and mean (macro) MTEB(LLM) score.

## H.3 Cost Sensitivity Analysis

Table 19 shows how the LLM-to-embedding cost ratio varies under alternative hardware and pricing assumptions. We consider five scenarios for embedding inference:

1. H100 spot (\$2.49/hr, Lambda Labs, March 2026): baseline, 1,431 ×

2. H100 on-demand (\$3.99 /hr): higher price, 893×

3. A100 spot (\$1.49/hr, est. 1.5× slower): 1,594×

4. L4 spot (\$0.49/hr, est. 3× slower): much lower throughput, 2,424×

5. Commercial API (e.g. \$0.10/MTok): fixed per-token pricing, 338×

In all five scenarios, the cost ratio exceeds 300 ×.

<table><tr><td>Hardware / Pricing Scenario</td><td>Emb. Cost</td><td>LLM Cost</td><td>Ratio</td></tr><tr><td>H100 spot $2.49/hr (our setup)</td><td>$0.108</td><td>$154.14</td><td>1,431×</td></tr><tr><td>H100 on-demand $3.99/hr</td><td>$0.173</td><td>$154.14</td><td>893×</td></tr><tr><td>A100 spot $1.49/hr (est. 1.5× slower)</td><td>$0.097</td><td>$154.14</td><td>1,594×</td></tr><tr><td>L4 spot $0.49/hr (est. 3× slower)</td><td>$0.064</td><td>$154.14</td><td>2,424×</td></tr><tr><td>Commercial API $0.10/MTok</td><td>$0.457</td><td>$154.14</td><td>338×</td></tr></table>

Table 19: Cost sensitivity analysis. LLM-to-embedding cost ratio under alternative hardware and pricing scenarios. Costs compare Octen-8B with Gemini 3.1 Pro at fixed API pricing; ratios range from 338–2,424×.

Per-query economics and training amortisation. The benchmark-level ratios translate directly to per-query terms: both paradigms answer the same queries, so dividing by the query count rescales both sides equally and leaves the 29–1,431 × range unchanged. Two effects widen the gap further under sustained deployment. First, embedding training is a one-time cost amortised over the deployment lifetime (and, for public checkpoints, across all users), so its per-query contribution vanishes at scale, whereas LLM inference cost recurs in full on every query. Second, document embeddings are computed once and reused across queries, while corpus-in-context reading repeats per query and is only partially offset by prompt caching. The benchmark-pass comparison is therefore conservative: it charges embeddings their full encoding cost while granting the LLM cached-input rates.

## H.4 Detailed Cost Methodology

Embedding cost pipeline. For each model, we tokenize all MTEB(LLM) tasks with its HuggingFace AutoTokenizer. We then benchmark maximum-throughput processing on an NVÍDIA H100 80GB at sequence length 512, using the largest batch that fits in memory. Cost is (tokens $/ 1 0 ^ { 6 } ) \times ( r _ { \mathrm { G P U } } / T )$ , where $r _ { \mathrm { G P U } } = \$ 2 .49 / \mathrm { h r }$ (Lambda Labs spot pricing, March 2026) and T is sustained throughput in tokens/hour.

Token counts range from 4.4M (Gemma-based tokenizers) to 5.5M (Mistral-based), a 26% range driven by vocabulary differences. Using a single proxy tokenizer (e.g., GPT-2 at 6.75M tokens) would overestimate embedding costs by 22–54%.

LLM cost pipeline. Token usage is extracted from each provider's usage\_stats in every task's JSON result file. For Gemini, input\_tokens =prompt\_token\_count (including cached tokens), output\_tokens = candidates\_token\_count (excluding reasoning), and cached\_tokens = cached\_content\_token\_count. Total tokens equal input plus output plus reasoning. For the open models served via OpenRouter, the completion count already includes reasoning and total = input + output, with reasoning reported separately. We bill every generated token, total — input, at the output rate. This equals output plus reasoning for Ġemini and the reasoning-inclusive completion count for open models. Each model's reasoning share comes from its usage statistics (Figure 4). For multilingual tasks (STS17, STS22v2, RTE3), per-language entries report different token counts and are summed; for classification tasks with multiple entries per split, all entries share identical usage\_stats and are counted once.

Gemini API prices per MTok input/output are \$0.25/\$1.50 for Flash-Lite, \$0.50/\$3.00 for Flash, and \$2.00/\$12.00 for Pro; ċached input is billed at 10%. Open models use OpenRouter public rates (March/June 2026).

Throughput measurement. We serve both paradigms on the same NVIDIA H100 and measure sustained tokens/second. Open-weight LLMs (Qwen3.6-27B, Qwen3.6-35B-A3B) run under vLLM (BF16, tensor-parallel = 1, 256 concurrent requests, 200/100 input/output tokens); embedding models run batch inference on the same GPU. Holding hardware fixed removes API rate limits and GPU choice as explanations for the measured gap.

## I Reproducibility

All code, datasets, result files, and analysis scripts are released at https://github.com/ embeddings-benchmark/embedders-dilemma and implemented with the MTEB framework.

LLM prompt templates. Prompt design follows a consistent structure across all five task categories:

• System prompt: one sentence describing the task type and output format.

• User prompt: task-specific instruction, followed by the input text(s).

• Output schema: a Pydantic model with fields reasoning (free text, optional) and output (constrained to valid labels or numeric ranges). JSON mode is enabled for all calls.

Example system prompts: Classification: "You are a text classifier. Return a JSON with your reasoning and the predicted label from the provided list." Retrieval: "You are a retrieval assistant. Given a query and a set of numbered documents, identify the document IDs most relevant to the query. Return a JSON list of IDs in ranked order."

Schema validation and failure handling. All LLM outputs are validated against the Pydantic schema after generation. If validation fails (e.g., invalid label or malformed JSON), the prompt is retried with an appended correction instruction, up to twice per sample. The final validation failure rate is <0.3%; failed samples receive a null prediction and are counted as incorrect.

Token usage tracking. For each API call we log: prompt\_token\_count, candidates\_token\_count, cached\_content\_token\_count, total\_token\_count, and thoughts\_token\_count from GenerateContentResponse.usage\_metadata. Thinking tokens are verified as total — prompt — candidates to cross-check against thoughts\_token\_count; agreement was exact in all cases. Pricing sources and throughput configurations are documented in §H.4.