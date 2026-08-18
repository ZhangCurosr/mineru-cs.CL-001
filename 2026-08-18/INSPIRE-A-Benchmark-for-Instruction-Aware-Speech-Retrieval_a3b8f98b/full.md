# INSPIRE: A Benchmark for Instruction-Aware Speech Retrieval

Chen-An Li <sup>1</sup>, Hung-yi Lee <sup>1,2</sup>

<sup>1</sup>National Taiwan University <sup>2</sup>NTU Artificial Intelligence Center of Research Excellence (NTU AI-CoRE) {f13942069, hungyilee}@ntu.edu.tw

## Abstract

Existing speech retrieval systems rely on fixed similarity matching and cannot adapt to diverse user intents. We introduce INSPIRE, the first benchmark for instruction-aware speech retrieval, in which natural-language instructions dynamically specify relevance criteria, including semantic content, speaker identity, speaking style, environmental sounds, and their combinations. We evaluate four retrieval paradigms: large audiolanguage models, cascaded pipelines, self-supervised speech models, and contrastive audio-language models. Our results reveal that no current method robustly handles all retrieval intents. Text-based approaches perform relatively better at semantic retrieval but struggle with paralinguistic attributes, while speechbased models are moderately better at capturing acoustic properties but falter at following instructions. These findings highlight the need for unified architectures capable of instructionaware speech retrieval.

Index Terms: speech retrieval, instruction-following

## 1. Introduction

Speech retrieval, the task of searching a collection of spoken documents to find those relevant to a user’s spoken query, is traditionally built around fixed notions of similarity, matching documents that are acoustically or semantically close to a query [1, 2]. However, real-world retrieval needs are often instruction-driven. For instance, given a single reference audio clip, a customer service manager might want to find other calls exhibiting the same frustrated tone, a journalist may search for archival statements made by that exact speaker, or an investigator might look for clips recorded in a similarly noisy background. Each scenario defines relevance in entirely different ways, yet existing systems apply a single, rigid similarity function. We therefore argue that speech retrieval should move beyond similarity-only paradigms toward an instruction-aware framework that explicitly specifies relevance criteria. Despite its success in text retrieval [3, 4], instruction-following remains largely unexplored for speech retrieval.

Compared to text retrieval, instruction-aware speech retrieval is challenging because the query is a speech signal and the relevant attributes are heterogeneous. The instruction may refer to linguistic content (“what is said”), paralinguistic characteristics (“how it is said”), or environmental context (“where it is said”), and the system must weigh these cues differently depending on the instruction. This demands representations that preserve fine-grained acoustic detail without sacrificing compositional instruction following.

As illustrated in Figure 1, the same spoken query paired with different instructions should retrieve different target spoken documents. The example in Figure 1 shows contentbased retrieval, speaker-based retrieval, style-based retrieval, background-based retrieval, and composite intents that combine multiple attributes.

![](images/7622f101294474b0f9630b5c056bc52475c140ac9f002d0632d9b27b0dabcef5.jpg)  
Figure 1: Illustrating instruction-aware speech retrieval across diverse user intents such as content, speaker, style, background, and their combinations.

In this paper, we introduce INSPIRE<sup>1</sup>, a benchmark for Instruction-Aware Speech Retrieval that systematically evaluates whether retrieval systems can follow natural-language instructions when selecting relevant spoken documents. INSPIRE targets instruction-aware retrieval, where the instruction dynamically describes relevance.

Our primary contribution is formalizing instruction-aware speech retrieval as a benchmarked research problem. Specifically, we:

• Introduce INSPIRE, the first benchmark for instructionaware speech retrieval spanning semantic, speaker, style, background, and multi-attribute intents;

• Provide a unified evaluation protocol that compares diverse model families under the same instruction formulation;

• Conduct an extensive empirical study covering large audiolanguage models, cascaded pipelines combining ASR and captioning with text retrieval, self-supervised speech representations, and contrastive audio-language architectures;

• Analyze failure modes across instruction types and subsets, revealing that no existing method robustly handles all aspects of instruction-aware speech retrieval within a single unified framework.

These findings highlight the need for new architectures and training paradigms specifically tailored to instruction-aware speech retrieval. We hope this work motivates future research to enable users to search speech collections with the same naturalness and precision as in text and image retrieval.

## 2. Related Work

## 2.1. Spoken Content Retrieval

Early work focused on query-by-example spoken term detection [5–7], where users provide a spoken sample to locate specific words or phrases in speech. Subsequent research extended these matching-based methods to spoken content retrieval [1, 2, 8–12], including speech-to-speech retrieval and text-to-speech retrieval over large speech collections. These lines of work typically define relevance through acoustic or semantic similarity, with a fixed retrieval objective per spoken query. In contrast, INSPIRE considers settings where an instruction defines relevance on-the-fly, potentially across multiple attributes, requiring the model to resolve which dimension of similarity cues takes priority.

## 2.2. Spoken Question Answering

Spoken question answering (SQA) [13–19] identifies answer spans within spoken passages for a given question. While SQA captures an important semantic retrieval use case, it does not address instruction-aware retrieval, in which relevance may depend on paralinguistic attributes. INSPIRE includes SQAlike semantic intents as one subset but goes further, requiring models to handle speaker-, style-, and environment-based constraints alongside multi-attribute compositions, revealing whether models that excel at semantic retrieval generalize when relevance shifts beyond content.

## 2.3. Instruction-Aware Retrieval in Other Modalities

Instruction-aware retrieval has recently become a strong paradigm in text [3, 20–24] and image [4, 25–34] retrieval. In contrast, instruction-following has not been systematically studied for speech retrieval, motivating the INSPIRE benchmark. INSPIRE adopts the same high-level principle, using a naturallanguage instruction to specify what should be matched, but applies it in the speech domain where the query is speech and relevance may depend on both linguistic and paralinguistic cues.

## 2.4. Speech/Audio Representation Benchmarks

Prior work has systematically evaluated speech and audio representations using fixed downstream tasks and probing suites, including NOSS [35], SUPERB [36], HARES [37], MSEB [38], and MAEB [39]. These benchmarks are highly valuable for measuring whether a representation captures semantic or acoustic information, but they generally do not test instructionconditioned relevance. INSPIRE complements them by directly evaluating retrieval behavior under natural-language instructions, where the target attribute can change across queries.

## 3. Design of INSPIRE

## 3.1. Problem Formulation

We study instruction-aware speech retrieval, in which a system takes a spoken query and a natural-language instruction, then retrieves recordings from a database that satisfy the instruction.

Let $D = \{ d _ { i } \} _ { i = 1 } ^ { | D | }$ denote a speech database with spoken documents and ${ \cal Q } = \{ q _ { i } \} _ { i = 1 } ^ { | Q | }$ a set of spoken queries, where each $d _ { i }$ and q<sub>i</sub> is a speech recording whose length may range from a single sentence to a multi-turn passage depending on the retrieval scenario. Let Z denote the set of all possible naturallanguage instructions. For retrieval, we form query-instruction pairs $Q _ { Z } = \{ ( q , z ) \mid q \in Q , \ z \in Z \}$ . In each search, the system receives a spoken query q and an instruction z that specifies the user’s intent, potentially constraining the semantic content, speaker identity, vocal style, and acoustic environment.

For any pair $( q , z )$ , the database is partitioned into two disjoint sets, $D = D ^ { + } \cup D ^ { - }$ . Positive samples $d ^ { + } \in D ^ { + }$ satisfy all requirements in z relative to the spoken query q, whereas negative samples $d ^ { - } \in D$ <sup>−</sup> violate at least one requirement.

The instruction-aware speech retrieval system uses an instruction-conditioned scoring function $f ( d \ | \ q , z ) \in$ R to assign a relevance score to each spoken document d $\in { \cal D }$ . Sorting scores in descending order yields a ranked list $\pi ( q , z )$ . Successful retrieval requires $f ( d ^ { + } \mid q , z ) > f ( d ^ { - } \mid q , z )$ for all $d ^ { + } \in D ^ { + }$ and $d ^ { - } \in D ^ { - } .$ , ensuring that relevant spoken documents are ranked above irrelevant ones.

This formulation evaluates how well models interpret natural-language instructions for speech retrieval. Compared with traditional keyword-matching retrieval, instruction-aware speech retrieval supports free-form instructions that express multi-attribute goals, yielding a more realistic and challenging evaluation setting.

## 3.2. Benchmark Construction

INSPIRE consists of four subsets, each built from a distinct data source: DailyTalk [40], VCTK [41], Expresso [42], and Synthetic Data. Each subset targets a different dimension of instruction-aware retrieval, from conversational continuity to speaker and style matching to complex compositional instructions. Each subset serves as an independent retrieval corpus, in which queries are evaluated exclusively against the documents within that subset, and all retrieval metrics are computed separately for each subset. For every subset, we specify (i) the construction of spoken queries, (ii) the relevance criteria, and (iii) the selection of positive and negative documents.

## 3.2.1. DailyTalk Subset

DailyTalk is a high-quality multi-turn conversational speech dataset designed for modeling contextual dialogue in text-tospeech systems. This subset evaluates whether a retriever can use a dialogue prefix together with a textual instruction to retrieve the correct continuation. We use the first half of each dialogue as the spoken query and the second half as the target document. A document is relevant if it is the true continuation, and negative samples are drawn from unrelated dialogues.

## 3.2.2. VCTK Subset

VCTK is a multi-speaker English speech corpus commonly used in speech synthesis and voice conversion studies. In this subset, the task is to assess whether a retriever can follow a textual instruction to find utterances spoken by the same speaker as a given spoken query. We construct the subset using 80 speakers, where each spoken query serves as a reference utterance, and the goal is to retrieve other utterances from the same speaker. Hard negatives are utterances with identical content but spoken by different speakers, while standard negatives differ from the spoken query in both speaker identity and content.

## 3.2.3. Expresso Subset

Expresso is a high-quality expressive speech dataset featuring read and improvised dialogues across diverse speaking styles. This subset evaluates the model’s ability to follow a textual instruction to retrieve utterances based on constraints related to speaker, style, or both. We consider four speakers and five styles: default, whisper, laughing, sad, and confused. We use the default style only for negative documents.

Table 1: Example instructions in INSPIRE. Instructions rangefrom single-attribute criteria to complex multi-attribute combinations
<table><tr><td>Instruction Type</td><td>Instruction Example</td></tr><tr><td>Continuation</td><td>Search for the follow-up lines to the query speech.</td></tr><tr><td>Same speaker</td><td>Find all documents said by the same talker as the query.</td></tr><tr><td>Same speaking style</td><td>Return documents with a similar way of speaking to the query</td></tr><tr><td>Same speaker + Same style</td><td>Get documents where the voice matches and the speaking style matches.</td></tr><tr><td>Same speaker + Sadness style + Same env.</td><td>Get sad-style recordings from the same speaker in the same sound environment.</td></tr><tr><td>Same speaker + Footsteps env.</td><td>Retrieve clips from the same speaker that include footsteps in the background.</td></tr></table>

Table 2: Each column represents a subtask in subsets, where attribute constraints are defined asfollows: S indicates the same value as the query, T indicates a specific target value, and – indicates no constraint.
<table><tr><td>Attribute\ Subset |DailyTalk |VCTK | 1</td><td></td><td></td><td></td><td>Expresso</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Synthetic</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Semantic</td><td>S</td><td>一-</td><td>- -</td><td></td><td> - - | S</td><td></td><td>-</td><td>-</td><td>-</td><td></td><td>-</td><td>一</td><td>-</td><td>一</td><td>一</td><td>一</td><td>一</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Speaker</td><td>-</td><td>S</td><td>S</td><td>- S</td><td></td><td>S | - S</td><td></td><td></td><td></td><td>- -</td><td>S S S</td><td></td><td></td><td>S</td><td>-</td><td>-</td><td>-</td><td>S S</td><td></td><td>S</td></tr><tr><td>Speaking Style</td><td>一</td><td>-</td><td>|- S S</td><td></td><td></td><td>T | - -</td><td></td><td>S</td><td></td><td>-</td><td>S T</td><td></td><td>-</td><td>- </td><td>S</td><td></td><td>T S</td><td>S</td><td> T</td><td>S</td></tr><tr><td>Environmental Sound</td><td>1</td><td>-</td><td>一</td><td>-</td><td>一</td><td>一</td><td>一</td><td>-</td><td>一</td><td>S</td><td>一</td><td>一</td><td>S</td><td>T</td><td>S</td><td>S</td><td>T</td><td>S</td><td>S </td><td>T</td></tr></table>

Table 3: Statistics of INSPIRE. |Q| denotes the number of unique spoken queries, $| Q _ { Z } |$ the number of query-instruction pairs, |D| the number of documents, $| D ^ { + } | / | \dot { Q } z |$ the average number ofpositive documents per query-instruction pair.
<table><tr><td>Subset</td><td>|Q|</td><td>|Qz|</td><td>|D|</td><td>|D+|/|Qz|</td></tr><tr><td>DailyTalk</td><td>200</td><td>200</td><td>4,882</td><td>1.000</td></tr><tr><td>Expresso</td><td>200</td><td>800</td><td>3,861</td><td>2.000</td></tr><tr><td>VČTK</td><td>80</td><td>80</td><td>3,082</td><td>2.000</td></tr><tr><td>Synthetic</td><td>200</td><td>3,000</td><td>5,400</td><td>1.946</td></tr><tr><td>Total / Avg.</td><td>680</td><td>4,080</td><td>17,225</td><td>1.910</td></tr></table>

A document is relevant if it matches the spoken query with respect to the required attributes. Hard negatives share the same sentence as the spoken query but violate at least one specified attribute, whereas standard negatives differ in both content and attributes. We exclude same-sentence documents from the positive candidate pool to avoid trivial matches, since these pairs yield high similarity scores that fail to measure the model’s generalization beyond exact matching. Each query-instruction pair also defines an excluded document set to control the number of positives.

## 3.2.4. Synthetic Subset

We create a synthetic speech retrieval dataset based on Natural Questions [43] by converting text queries and documents into speech. This subset evaluates whether a retrieval model can satisfy multi-attribute constraints specified in text under controlled conditions. We sample 200 spoken queries and define relevance based on semantic correctness, speaker identity, speaking style, environmental sounds, or their combinations.

Speech is synthesized using the GPT-4o-mini TTS model [44] with five distinct voices and three expressive styles: happiness, anger, and sadness. To simulate realistic acoustic conditions, each utterance is mixed with one environmental sound from the ESC-50 [45] dataset. We use 15 sound effects such as car horns, rain, thunderstorms, footsteps, keyboard typing, and animal sounds.

Documents are relevant if they satisfy all required attributes. Hard negatives share the same sentence as the spoken query but violate at least one required attribute, while standard negatives differ in both content and attributes. Following Expresso, each query-instruction pair defines its own exclusion set to control positive set size and exclude same-content documents from positives, preventing trivial retrieval.

## 3.2.5. Instruction Generation

For each relevance type in INSPIRE, we generate 20 diverse instructions using GPT-5.2 [46]. Table 1 shows example instructions ranging from simple single-attribute constraints (e.g., “Find all documents said by the same talker”) to complex multiattribute combinations (e.g., “Get sad-style recordings from the same speaker in the same sound environment”). For each queryrelevance pair, we randomly sample one instruction to form the final retrieval input, promoting linguistic diversity while preserving the underlying retrieval constraints.

## 3.3. Benchmark Statistics and Quality Assessments

Tables 2 and 3 summarize INSPIRE statistics. Table 2 shows attribute constraints across subsets. DailyTalk focuses exclusively on semantic content matching; VCTK concentrates on speaker identity; Expresso introduces speaking style considerations, with tasks requiring matching speaker alone, style alone, or both simultaneously; and Synthetic incorporates all four attributes: semantic content, speaker identity, speaking style, and environmental sounds. For speaking style and environmental acoustics, constraints may require either matching the query or specifying a distinct target value, as these attributes can be controlled independently without a spoken query. In contrast, semantic content and speaker identity are inherently querydependent; the former is derived from the transcription of the input speech, while the latter requires a reference utterance to characterize the target voice.

Table 3 reports scale statistics. The benchmark contains hundreds of spoken queries expanded into thousands of queryinstruction pairs via diverse instructions, evaluated against a large document database. Crucially, the same spoken query paired with different instructions yields different target documents, which is the characteristic of our benchmark. The low average number of positives per query reflects a realistic scenario in which few relevant items are among many candidates. DailyTalk provides queries with single positives representing true continuations; VCTK and Expresso offer more constrained evaluations; and Synthetic provides the richest diversity, with multiple instruction types per query reflecting complex multiattribute matching requirements.

We conduct quality assessments on the synthetic subset across multiple dimensions. For speech recognition, we apply punctuation filtering and number normalization after transcribing with Whisper-Large [47]. This yields word error rates of 0.0314 for clean audio and 0.0337 for audio with environmental sounds. We evaluate speaker consistency using a pretrained ECAPA-TDNN [48,49] through five-fold SVM [50,51] cross-validation. This achieves 0.9998 accuracy in both conditions. We assess speaking-style quality using emotion embeddings from emotion2vec [52] with five-fold SVM crossvalidation. This obtains accuracies of 0.8657 for clean audio and 0.8780 for audio with environmental sounds. We estimate speech quality using UTMOS [53], which produces a predicted mean opinion score of 3.76 for clean synthetic speech. These results demonstrate that the synthetic subset is appropriate for evaluating instruction-aware speech retrieval.

## 4. Baseline Methods

We evaluate four baseline approaches that differ in their modality usage and in how they incorporate natural-language instruction. Each baseline computes a relevance score $f ( d \ \mid \ q , z )$ given a spoken query q, an instruction z, and a document d.

## 4.1. Large Audio-Language Models Embeddings

Since no existing retriever is designed for instruction-aware speech retrieval, we draw inspiration from work in other modalities that leverage large language models (LLMs) [54] or multimodal LLMs (MLLMs) [4, 12] as instruction-aware embedding backbones. As a baseline, we prompt large audio-language models (LALMs) to encode task-specific instructions into latent representations for retrieval. Specifically, for the query side, we concatenate the spoken query q, the instruction z, and the prompt “Summarize the above sentence and speech in one word.” as input to the LALM, and extract the query embedding from the hidden state of the last token at layer L: ${ \bf e } _ { q } = h _ { \mathrm { l a s t } } ^ { ( L ) } ( q , z )$ . For the document side, we feed the spoken document d along with the prompt “Summarize the above speech in one word.” to obtain ${ \\\mathfrak { a } } _ { d } = h _ { \mathrm { l a s t } } ^ { ( L ) } ( d )$ . Relevance is then scored via cosine similarity: $f ( d \mid q , z ) = \cos ( { \bf e } _ { q } , { \bf e } _ { d } )$

## 4.2. Cascaded Pipelines

This baseline converts speech inputs into text through automatic speech recognition and captioning, then applies standard text retrieval methods. For the spoken query q, we generate an ASR transcription $t _ { q }$ that captures the spoken content and a caption $c _ { q }$ that describes acoustic characteristics. For each document $d ,$ we similarly obtain $t _ { d }$ and c<sub>d</sub>. The query and document inputs are constructed as $\tilde { q } = [ t _ { q } ; c _ { q } ; z ]$ and $\tilde { d } = [ t _ { d } ; c _ { d } ]$ . We evaluate both sparse and dense text retrieval approaches. For sparse retrieval, we employ BM25 as $f ( d \mid q , \bar  z ) = \mathbf { B } \mathbf { M } 2 5 ( \tilde { q } , \tilde { d } )$ . For dense retrieval, we encode $\tilde { q }$ and d<sup>˜</sup>using a text encoder ϕ(·) and compute the relevance score as $f ( d \mid q , z ) = \cos ( \phi ( \tilde { q } ) , \phi ( \tilde { d } ) )$ ). However, this method relies on the quality of speech-to-text conversion and may lose acoustic information present in the original speech. Furthermore, since BM25 and some dense retrievers are not designed to interpret instructions, appending z to the query offers only a superficial form of instruction-awareness rather than a principled solution for instruction-aware retrieval.

## 4.3. Self-Supervised Speech Embeddings

This baseline operates exclusively on speech representations, encoding all utterances without conditioning on the instruction z. We encode both spoken query and document using a selfsupervised speech model $g ^ { ( \hat { L } ) } ( \cdot \dot { ) }$ , where L denotes the layer from which representations are extracted, as ${ \bf e } _ { q } = g ^ { ( L ) } ( q )$ and ${ \bf e } _ { d } ~ = ~ g ^ { ( L ) } ( d )$ The relevance score is computed as $f ( d \ )$ $q , z ) \ = \ \cos ( \mathbf { e } _ { q } , \mathbf { e } _ { d } )$ , which is independent of the instruction z. Since this approach has no mechanism to condition retrieval on the given instruction, it is inherently unable to distinguish between different instructions for the same spoken query. As such, it is not expected to perform well on instruction-aware retrieval tasks. We include it primarily as a baseline to quantify the gains achieved by instruction-conditioned methods.

## 4.4. Contrastive Audio-Language Embeddings

This baseline uses a contrastive audio-language model, such as CLAP [55, 56], for cross-modal text-audio retrieval. The spoken query is converted into text and concatenated with the instruction as $\tilde { q } ~ = ~ [ t _ { q } ; c _ { q } ; z ]$ , then encoded using the text encoder $\psi _ { \mathrm { t e x t } } ( \cdot )$ , while each document is encoded directly from speech using the audio encoder $\psi _ { \mathrm { a u d i o } } ( \cdot )$ . The relevance score is computed as $f ( d \mathbin { \mid } \ q , z ) \ = \ \cos \big ( \psi _ { \mathrm { t e x t } } ( \tilde { q } ) , \psi _ { \mathrm { a u d i o } } ( d ) \big )$ . Although this preserves acoustic information in document representations, the encoders are not trained to ground z in the joint embedding space, so concatenating z is unlikely to modulate retrieval meaningfully. We include this baseline to assess the models’ zero-shot performance on our task.

## 4.5. Reranking

We also experiment with several reranking methods. Given the top-K documents retrieved by the first-stage retriever, the reranker reassigns a relevance score to each candidate and reorders them accordingly. Drawing inspiration from work in other modalities that leverage LLMs or MLLMs as rerankers [24, 34], we explore LALM-based rerankers that take as input the instruction z, spoken query q, and candidate document d, along with the prompt “Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be $^ { \bullet } y e s ^ { \prime } o r \ : \ : ^ { \cdot } n o ^ { \prime } . ^ { \prime }$ The reranking score is computed as $\sigma ( \mathrm { l o g i t } ( \mathbf { y e s } ) - \mathrm { l o g i t } ( \mathbf { n o } ) )$ In addition, we explore text-based rerankers, where the spoken query and document are replaced by their respective transcriptions and captions $[ t _ { q } ; c _ { q } ]$ and $[ t _ { d } ; c _ { d } ]$ , while retaining the same instruction z, prompt, and scoring function.

## 5. Experiments

## 5.1. Experiment Setup

Following Section 4, we evaluate four baseline retrieval paradigms and several reranking methods.

For LALM embedding, we evaluate six LALMs spanning different parameter sizes and architectures: Audio-Flamingo-3 [57], Qwen2.5-Omni-3B/7B [58], Qwen3-Omni-30B-A3B-Instruct [59], Voxtral-Mini-3B, and Voxtral-Small-24B [60]. Following prior work [4, 12, 54], we extract embeddings from the hidden states of the last layer. For the cascaded pipeline, we use Whisper-large-v3 [47] for ASR and Qwen3-Omni-30B-A3B-Captioner [59] for captioning, with BM25 [61] as a sparse retriever, SentenceBERT [62] as a dense retriever, and E5- Mistral [21] and Qwen3-Embedding-8B [24] as instructionaware dense retrievers. For self-supervised speech embedding, we evaluate HuBERT-Large [63] and WavLM-Large [64] using last-layer representations. For contrastive audio-language embeddings, we employ LAION-CLAP [56]. For reranking, we use Qwen2.5-Omni-3B and Voxtral-Mini-3B as LALM rerankers, and Qwen3-Reranker-0.6B and 4B as text rerankers. We opt for slightly smaller models because the reranker scores each query-document pair individually, making it far more computationally expensive than the retriever.

Table 4: Main results on INSPIRE. We report Recall@10/50/100 across four subsets. Model size is reported in billions of parameters, and the best results per subset are shown in bold.
<table><tr><td></td><td>Size (B)</td><td colspan="3">DailyTalk R@10</td><td colspan="3">VCTK R@10 R@50</td><td rowspan="2">R@10</td><td rowspan="2">Expresso R@50</td><td rowspan="2">R@100</td><td rowspan="2">R@10</td><td rowspan="2">Synthetic R@50</td><td rowspan="2">R@100</td></tr><tr><td></td><td></td><td></td><td>R@50</td><td>R@100</td><td></td><td>Random Baseline</td><td>R@100</td></tr><tr><td>Random</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.24</td><td>1.06</td><td>2.10</td><td>0.35</td><td>1.84</td><td>3.63</td><td>0.30</td><td>1.50</td><td>2.96</td><td>0.20</td><td>1.00</td><td>1.98</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>LALMs Embeddings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Audio-Flamingo-3 [57]</td><td>8</td><td>30.00</td><td>57.50</td><td>69.50</td><td>1.25</td><td>6.25</td><td>7.50</td><td>2.19</td><td>7.31</td><td>10.94</td><td>3.82</td><td>6.31</td><td>7.72</td></tr><tr><td>Qwen2.5-Omni [58] Qwen2.5-Omni [58]</td><td>3</td><td>26.50</td><td>53.00</td><td>64.00</td><td>1.88</td><td>3.13</td><td>3.13</td><td>0.75</td><td>3.75</td><td>6.69</td><td>3.14</td><td>5.20</td><td>6.88</td></tr><tr><td>Qwen3-Omni [59]</td><td>7</td><td>36.00</td><td>62.50 74.50</td><td>73.00 83.00</td><td>1.25</td><td>3.13</td><td>5.63</td><td>0.63</td><td>3.38</td><td>6.13</td><td>4.18</td><td>6.00</td><td>7.47 7.88</td></tr><tr><td>Voxtral-Mini [60]</td><td>30 3</td><td>51.50 39.00</td><td>59.00</td><td>68.00</td><td>1.25 0.63</td><td>3.13 2.50</td><td>6.25</td><td>0.94 0.38</td><td>3.56</td><td>6.19 3.13</td><td>4.88</td><td>6.38</td><td>7.20</td></tr><tr><td>Voxtral-Small [60]</td><td>24</td><td>41.00</td><td>67.50</td><td>78.00</td><td>1.25</td><td>3.75</td><td>5.63 5.63</td><td>0.31</td><td>1.38</td><td></td><td>3.87</td><td>5.40</td><td>8.38</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1.44</td><td>2.81</td><td>5.17</td><td>7.23</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>Cascaded Pipelines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25 [61]</td><td></td><td>36.50</td><td>59.50</td><td>65.50</td><td>1.25</td><td>5.63</td><td>6.88</td><td>0.56</td><td>3.06</td><td>6.19</td><td>4.06</td><td>6.89</td><td>8.95</td></tr><tr><td>SentenceBERT [62] E5-Mistral [21]</td><td>0.1</td><td>32.50</td><td>50.50 78.00</td><td>59.50 85.00</td><td>0.63</td><td>3.75</td><td>4.38</td><td>0.44</td><td>2.63</td><td>5.13</td><td>4.16</td><td>6.28</td><td>7.92</td></tr><tr><td>Qwen3-Embedding [24]</td><td>7 8</td><td>55.50 62.00</td><td>81.50</td><td>89.50</td><td>1.25 1.88</td><td>5.63</td><td>10.63</td><td>1.63 0.75</td><td>5.25 5.56</td><td>8.31 9.69</td><td>6.50 7.02</td><td>8.32</td><td>9.92 11.07</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>6.25</td><td>10.63</td><td></td><td></td><td></td><td></td><td>8.95</td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>Self-Supervised Speech Embeddings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HuBERT-Large [63] WavLM-Large [64]</td><td>0.3</td><td>11.00</td><td>18.00 24.50</td><td>26.50 35.50</td><td>11.88</td><td>31.88</td><td>37.50</td><td>9.81</td><td>19.44</td><td>24.19</td><td>1.28</td><td>3.70</td><td>5.70</td></tr><tr><td></td><td>0.3</td><td>16.50</td><td></td><td></td><td>17.50</td><td>35.00</td><td>43.75</td><td>9.50</td><td>17.69</td><td>24.00</td><td>1.70</td><td>4.48</td><td>7.22</td></tr><tr><td>Contrastive Audio-Language Embeddings</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>LAION-CLAP [56]</td><td>0.2</td><td>0.00</td><td>1.50</td><td>2.00</td><td>1.94</td><td>5.63</td><td>8.94</td><td>1.88</td><td>4.38</td><td>7.50</td><td>0.80</td><td>2.22</td><td>4.35</td></tr></table>

We use recall as the primary retrieval metric. For a queryinstruction pair $( q , z )$ , let $\pi _ { K } ( q , z )$ denote the top-K ranked documents and $\mathcal { D } ^ { + }$ the set of positive documents. Recall@K is defined as:

$$
\mathrm { R e c a l l @ } K = \frac { 1 } { | Q _ { Z } | } \sum _ { ( q , z ) \in Q _ { Z } } \frac { \left| \pi _ { K } ( q , z ) \cap \mathcal { D } ^ { + } \right| } { | \mathcal { D } ^ { + } | } ,\tag{1}
$$

where $Q _ { Z }$ denotes the set of query-instruction pairs. For reranking, we adopt NDCG, which assigns greater credit to relevant documents at higher positions:

$$
\mathsf { N D C G @ K } = \frac { \mathsf { D C G @ K } } { \mathsf { I D C G @ K } } , ~ \mathsf { D C G @ K } = \sum _ { i = 1 } ^ { K } \frac { 2 ^ { r _ { i } } - 1 } { \log _ { 2 } ( i + 1 ) }\tag{2}
$$

where $r _ { i }$ is the relevance label of the document at rank i, and IDCG@K is the ideal DCG from sorting by relevance.

## 5.2. Main Results

Table 4 shows retrieval performance across all INSPIRE subsets. We also include a random baseline that uniformly samples documents, averaged over 100 runs, confirming that higher results are meaningful.

On DailyTalk, which targets semantic retrieval of conversational continuity, cascaded pipelines achieve the best performance. Instruction-aware dense retrievers such as Qwen3- Embedding obtain the strongest results, followed closely by E5-Mistral. LALMs also perform competitively but fall short of the performance of cascaded approaches. Contrastive audiolanguage models underperform due to weak semantic alignment between text and speech. These results suggest that converting speech to text and leveraging powerful text embedding models remain the most effective strategies for semantic retrieval.

On VCTK and Expresso, which emphasize speaker identity and speaking style, self-supervised speech models exhibit clear advantages. These subsets involve simpler instructions in which retrieval depends on paralinguistic features, expressiveness, prosody, emotion, and speaker traits, which are better captured acoustically. HuBERT-Large and WavLM-Large outperform most other approaches, showing that direct speech representations preserve more relevant information than cascaded methods for acoustic and paralinguistic properties.

On the Synthetic subset, which combines multiple attributes spanning semantic content, speaker characteristics, speaking style, and environmental sounds, the task proves especially challenging. Self-supervised speech embeddings achieve very low performance, confirming that instruction-agnostic representations struggle with multi-attribute retrieval. Contrastive audio-language embeddings also underperform due to diverse scenarios. LALMs and cascaded pipelines perform better but still yield relatively low results, suggesting this subset is a genuinely challenging benchmark requiring joint understanding of complex instructions and diverse acoustic properties.

In summary, our results reveal distinct strengths and limitations across different retrieval paradigms. All methods suffer significant performance degradation in complex multi-attribute scenarios, indicating substantial room for improvement in handling compositional instructions. Different approaches excel along different retrieval dimensions depending on whether the task prioritizes semantic or acoustic features. More broadly, our benchmark reveals a fundamental trade-off between semantic understanding, where cascaded methods prevail, and acoustic feature preservation, where speech-based methods excel. These findings highlight the need for unified models that effectively integrate both modalities, enabling robust performance across diverse retrieval intents while bridging the gap between semantic comprehension and acoustic representation.

## 5.3. Reranking Results

We evaluate reranking performance on top of the first-stage retrievers. For each query-instruction pair, the top-100 retrieved documents are reranked. For the LALM retrievers, we apply

![](images/59af86953d4cfc88208caaecad2569099d41ee2f18ad504ee48d718888d570dc.jpg)  
Figure 2: Performance across attribute constraints shown in radar plots for LALMs (left), cascaded pipelines (middle), and selfsupervised/contrastive models (right). We report Recall@50. Sem refers to semantic, Spk to speaker, Sty to speaking style, and Env to environmental sound. Attributes marked with \* specify a distinct target value rather than matching the query. Scores are normalized by the best-performing model per attribute, with a shared scale across all subfigures.

Table 5: Reranking performance measured by NDCG@10/50 on the top 100 documentsfromfirst-stage retrievers.
<table><tr><td></td><td colspan="2">DailyTalk</td><td colspan="2">VCTK</td><td colspan="2">Expresso</td><td colspan="2">Synthetic</td></tr><tr><td>Model</td><td>N@10</td><td>N@50</td><td>N@10</td><td>N@50</td><td>N@10</td><td>N@50</td><td>N@10</td><td>N@50</td></tr><tr><td colspan="9">LALMs Embeddings Rerank with LALMs</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>21.47</td><td>27.45</td><td>1.10</td><td>1.52</td><td>0.46</td><td>1.18</td><td>2.02</td><td>2.47</td></tr><tr><td>+ Qwen2.5-Omni-3B</td><td>38.50</td><td>41.69</td><td>0.77</td><td>1.34</td><td>1.05</td><td>1.53</td><td>3.29</td><td>3.47</td></tr><tr><td>+ Voxtral-Mini-3B</td><td>46.32</td><td>48.32</td><td>1.01</td><td>1.79</td><td>0.18</td><td>0.77</td><td>3.14</td><td>3.43</td></tr><tr><td>Voxtral-Small</td><td>26.60</td><td>32.55</td><td>1.53</td><td>2.17</td><td>0.12</td><td>0.41</td><td>2.59</td><td>3.08</td></tr><tr><td>+ Qwen2.5-Omni-3B</td><td>37.17</td><td>40.80</td><td>1.04</td><td>1.62</td><td>0.58</td><td>0.88</td><td>3.52</td><td>3.79</td></tr><tr><td>+ Voxtral-Mini-3B</td><td>45.72</td><td>48.33</td><td>1.01</td><td>1.66</td><td>0.11</td><td>0.40</td><td>3.36</td><td>3.70</td></tr><tr><td colspan="9">Cascaded Pipelines Rerank with Text Rerankers</td></tr><tr><td>BM25</td><td>25.35</td><td>30.68</td><td>1.25</td><td>2.36</td><td>0.21</td><td>0.87</td><td>2.02</td><td>2.73</td></tr><tr><td>+ Qwen3-Reranker-0.6B</td><td>22.09</td><td>26.90</td><td>1.32</td><td>2.29</td><td>0.31</td><td>1.15</td><td>1.57</td><td>2.38</td></tr><tr><td>+ Qwen3-Reranker-4B</td><td>24.77</td><td>29.05</td><td>0.97</td><td>1.29</td><td>0.37</td><td>0.94</td><td>1.27</td><td>2.11</td></tr><tr><td>Qwen3-Embedding</td><td>43.05</td><td>47.39</td><td>1.48</td><td>2.54</td><td>0.34</td><td>1.55</td><td>3.58</td><td>4.07</td></tr><tr><td>+ Qwen3-Reranker-0.6B</td><td>23.27</td><td>31.20</td><td>1.63</td><td>2.96</td><td>0.39</td><td>1.53</td><td>2.02</td><td>3.09</td></tr><tr><td>+ Qwen3-Reranker-4B</td><td>28.47</td><td>35.43</td><td>0.90</td><td>2.11</td><td>0.43</td><td>1.45</td><td>2.03</td><td>3.07</td></tr></table>

LALM-based rerankers. For the cascaded pipeline, we employ text rerankers instead.

As shown in Table 5, LALM reranking generally outperforms the first-stage LALM retrievers, with particularly great improvements on DailyTalk. In contrast, text-based reranking substantially improves performance on DailyTalk, while achieving comparable results on the other subsets. We hypothesize that this phenomenon arises because concatenating transcriptions and captions introduces excessive information that may mislead the reranker. Additionally, the instruction formats in INSPIRE are likely out of domain for text rerankers trained on standard retrieval tasks. These findings suggest that direct access to speech signals benefits LALM rerankers, whereas text rerankers require in-domain instruction-aware fine-tuning to perform effectively in this setting.

## 6. Ablations and Analysis

## 6.1. Analysis of Instruction Types

To better understand which parts of an instruction current retrievers can handle, we break down performance by instruction type, corresponding to different attribute constraints. Figure 2 reports a fine-grained comparison among LALMs, cascaded pipelines, self-supervised speech models, and contrastive audio-language models. To ensure visual clarity across tasks of varying difficulty, each radar axis is normalized by the bestperforming model on that attribute.

The results reveal a clear modality specialization. Cascaded methods consistently perform best on semantically driven instructions. For semantic continuation and answer containment, they substantially outperform random baselines and LALMs, suggesting that specialized text embeddings remain more robust for precise semantic correspondence in retrieval.

In contrast, the trend reverses for acoustic attribute match ing. Self-supervised speech models excel at satisfying sameattribute constraints related to speaker identity, speaking style, and acoustic environment, whereas the cascaded method and LALMs frequently perform at near-chance levels on these dimensions. This performance gap indicates that current textcentric and unified multimodal architectures lack sufficient ability to disentangle paralinguistic cues from linguistic content.

Finally, when instructions specify particular attribute values, self-supervised speech models show notably weaker performance, suggesting that fine-grained control in instructionaware speech retrieval still requires more targeted supervision or attribute-aware training strategies.

## 6.2. Impact of Instruction Usage

![](images/6bcebbcc0d2ecc2361a89b4860b3c6b80186d2efdeac60ae3f09f5f1012d0846.jpg)  
Figure 3: Impact ofinstruction usage on retrieval performance. Solid bars show results with instructions; hatched bars without. Models A to F are LALMs; G to J are cascaded pipelines.

To assess whether models effectively leverage instructions, we compare retrieval performance with and without instructions. Figure 3 shows results for various models under both conditions across all four subsets.

Overall, instruction gains are evident only in instructionaware text retrievers such as E5-Mistral and Qwen3- Embedding, while BM25 and SentenceBERT show minimal sensitivity due to their instruction-agnostic design. All six LALMs also exhibit limited instruction sensitivity, as they lack instruction-aware retrieval training.

On DailyTalk, instruction-aware text retrievers show substantial gains when instructions are provided. On VCTK, Expresso, and Synthetic, all models show minimal differences between conditions, indicating current approaches cannot leverage instructions for paralinguistic and multi-attribute retrieval.

These results suggest that instruction-aware training is necessary for models to benefit from instructions.

## 6.3. Comparison of Captioning Models

![](images/e0a8685baf276ebf5ec0067d03b3e54208d9019345e1463da09e5489815cffbc.jpg)  
Figure 4: Comparison of captioning models for cascaded pipelines using detailed and instruction-aware prompts.

Figure 4 reports retrieval performance in a controlled setting where only the captioner is varied while the downstream text retriever is held fixed, isolating the effect of caption quality from retriever choice. We also conduct prompt engineering ablations for the other LALMs, since the Qwen3-Omni-Captioner does not accept text input. Specifically, we prompt the other LALMs with a detailed prompt, “Generate a caption that describes the speech above, including its meaning, the speaker’s identity or role, the speaking style, and any relevant environmental sounds.”, and an instruction-aware prompt, “Generate a caption that describes the speech above and can be usedfor the following instruction: {instruction}”.

The detailed and instruction-aware prompts yield nearly identical results across most configurations. Furthermore, all evaluated methods exhibit a sharp performance floor on the VCTK and Expresso datasets. The consistently low scores on these benchmarks, regardless of the LALM or prompting strategy employed, suggest that these tasks remain inherently challenging and are currently less sensitive in the caption model, likely due to a misalignment between the speech features and the retrieval objectives.

## 6.4. Effect of Oracle Metadata

To investigate the impact of metadata quality on cascaded pipelines, we replace model-generated captions with oracle metadata containing ground-truth transcriptions, speaker IDs, speaking styles, and environmental sound labels. Table 6 summarizes the results.

Table 6: Effect of oracle metadata on cascaded text retrieval. Green indicates improvement and red indicates degradation compared to the caption baseline.
<table><tr><td rowspan="2">Model</td><td colspan="2">DailyTalk</td><td colspan="2">VCTK</td><td colspan="2">Expresso</td><td colspan="2">Synthetic</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td>BM25</td><td>36.50</td><td>59.50</td><td>1.25</td><td>5.63</td><td>0.56</td><td>3.06</td><td>4.06</td><td>6.89</td></tr><tr><td>+ Oracle</td><td>36.00</td><td>49.00</td><td>53.13</td><td>95.00</td><td>0.63</td><td>2.69</td><td>5.18</td><td>7.57</td></tr><tr><td>SentenceBERT</td><td>32.50</td><td>50.50</td><td>0.63</td><td>3.75</td><td>0.44</td><td>2.63</td><td>4.16</td><td>6.28</td></tr><tr><td>+ Oracle</td><td>14.00</td><td>25.50</td><td>0.63</td><td>1.25</td><td>0.50</td><td>3.50</td><td>1.53</td><td>3.86</td></tr><tr><td>E5-Mistral</td><td>55.50</td><td>78.00</td><td>1.25</td><td>5.63</td><td>1.63</td><td>5.25</td><td>6.50</td><td>8.32</td></tr><tr><td>+ Oracle</td><td>49.00</td><td>71.00</td><td>1.88</td><td>4.38</td><td>1.19</td><td>5.75</td><td>7.12</td><td>10.23</td></tr><tr><td>Qwen3-Embedding</td><td>62.00</td><td>81.50</td><td>1.88</td><td>6.25</td><td>0.75</td><td>5.56</td><td>7.02</td><td>8.95</td></tr><tr><td>+ Oracle</td><td>58.00</td><td>80.50</td><td>1.88</td><td>8.13</td><td>2.69</td><td>9.50</td><td>7.52</td><td>10.20</td></tr></table>

Oracle metadata degrades semantic retrieval on DailyTalk, as non-semantic attributes introduce noise that misleads text matching. On paralinguistic subsets such as VCTK and Expresso, effects are mixed but often positive. On VCTK, BM25 benefits dramatically because oracle speaker IDs provide highly distinctive tokens ideal for term matching. Oracle labels also help instruction-aware embeddings when retrieval targets these attributes directly. On Synthetic, oracle metadata improves most models but not uniformly, highlighting that even perfect metadata cannot ensure robust multi-attribute retrieval. These findings indicate that both accurate metadata extraction and instruction-aware training are necessary for effective cascaded retrieval on non-semantic tasks.

## 6.5. Comparison with Proprietary Models

Table 7: Comparison with proprietary models. Results are shown for representative open-source models and proprietary APIsfrom Google and OpenAI.
<table><tr><td rowspan="2">Model</td><td colspan="2">DailyTalk</td><td colspan="2">VCTK</td><td colspan="2">Expresso</td><td colspan="2">Synthetic</td></tr><tr><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td colspan="9">LALMs Embeddings</td></tr><tr><td>Qwen3-Omni</td><td>51.50</td><td>74.50</td><td>1.25</td><td>3.13</td><td>0.94</td><td>3.56</td><td>4.88</td><td>6.38</td></tr><tr><td>Voxtral-Small</td><td>41.00</td><td>67.50</td><td>1.25</td><td>3.75</td><td>0.31</td><td>1.44</td><td>5.17</td><td>7.23</td></tr><tr><td colspan="9">Cascaded Pipelines with Open-source Models</td></tr><tr><td>E5-Mistral</td><td>55.50</td><td>78.00</td><td>1.25</td><td>5.63</td><td>1.63</td><td>5.25</td><td>6.50</td><td>8.32</td></tr><tr><td>Qwen3-Embedding</td><td>62.00</td><td>81.50</td><td>1.88</td><td>6.25</td><td>0.75</td><td>5.56</td><td>7.02</td><td>8.95</td></tr><tr><td colspan="9">Cascaded Pipelines with Proprietary Models</td></tr><tr><td>Gemini</td><td>53.00</td><td>80.00</td><td>1.25</td><td>3.13</td><td>0.81</td><td>3.13</td><td>6.97</td><td>8.40</td></tr><tr><td>OpenAI</td><td>57.50</td><td>80.00</td><td>1.25</td><td>1.25</td><td>0.06</td><td>1.38</td><td>6.73</td><td>7.50</td></tr></table>

We compare proprietary APIs from Google and OpenAI in Table 7. Since no proprietary multimodal speech embedding model currently exists, we follow the same cascaded pipeline (Section 4.2): captioning with a proprietary MLLM and embedding with a proprietary text model. For Google, we use Gemini-3.0-Flash [65] and gemini-embedding-001 [23]; for OpenAI, GPT-4o-mini-Audio [44] and text-embedding-3-large [66].

Proprietary models are competitive on semantic retrieval but do not consistently outperform the best open-source instruction-aware text retrievers. On DailyTalk, both trail the strongest instruction-aware embedding model, suggesting that caption quality alone does not fully close the semantic retrieval gap. On VCTK and Expresso, both underperform the best opensource baselines, further confirming the limitations of cascaded pipelines under paralinguistic constraints. On Synthetic, results are comparable to the best open-source baselines, though overall recall remains low.

These findings highlight a fundamental limitation of caption-then-embed pipelines, as compressing rich speech into text often drops fine-grained acoustic cues, and further motivate instruction-aware speech representations or multimodal embeddings that directly encode acoustic attributes.

## 6.6. LALMs Pooling Strategy Comparison

Table 8: LALMs pooling strategy comparison. We compare the prompt-based last token embedding approach versus mean pooling without prompts.
<table><tr><td></td><td colspan="2">DailyTalk</td><td colspan="2">VCTK</td><td colspan="2">Expresso</td><td colspan="2">Synthetic</td></tr><tr><td>Model</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td>Audio-Flamingo-3</td><td>30.00</td><td>57.50</td><td>1.25</td><td>6.25</td><td>2.19</td><td>7.31</td><td>3.82</td><td>6.31</td></tr><tr><td>+ Mean Pooling</td><td>34.00</td><td>56.50</td><td>3.75</td><td>8.13</td><td>1.94</td><td>6.25</td><td>0.57</td><td>2.08</td></tr><tr><td>Qwen2.5-Omni-7B</td><td>36.00</td><td>62.50</td><td>1.25</td><td>3.13</td><td>0.63</td><td>3.38</td><td>4.18</td><td>6.00</td></tr><tr><td>+ Mean Pooling</td><td>27.00</td><td>44.50</td><td>0.63</td><td>2.50</td><td>1.50</td><td>4.25</td><td>1.22</td><td>3.68</td></tr><tr><td>Qwen3-Omni</td><td>51.50</td><td>74.50</td><td>1.25</td><td>3.13</td><td>0.94</td><td>3.56</td><td>4.88</td><td>6.38</td></tr><tr><td>+ Mean Pooling</td><td>15.00</td><td>25.50</td><td>1.25</td><td>5.63</td><td>0.81</td><td>3.56</td><td>0.45</td><td>1.63</td></tr><tr><td>Voxtral-Small-24B</td><td>41.00</td><td>67.50</td><td>1.25</td><td>3.75</td><td>0.31</td><td>1.44</td><td>5.17</td><td>7.23</td></tr><tr><td>+ Mean Pooling</td><td>28.00</td><td>49.00</td><td>0.63</td><td>3.13</td><td>0.38</td><td>2.00</td><td>0.48</td><td>1.52</td></tr></table>

We compare two embedding extraction strategies for LALMs. The first is the prompt-based last token approach described in Section 4.1, where we append a summarization prompt to the input and extract the hidden state of the last token as the embedding. The second is mean pooling, where we directly average all hidden states across the sequence without using the summarization prompt. Table 8 presents the results.

The prompt-based last token approach generally outperforms mean pooling on DailyTalk and Synthetic, particularly for Qwen-series models. Qwen3-Omni exhibits the largest performance gap, with the last token embedding achieving substantially higher recall than mean pooling on semantic retrieval tasks. This suggests that the summarization prompt effectively guides the model to produce more discriminative representations for content-based retrieval. On VCTK and Expresso, both strategies yield similarly low performance, indicating that neither approach enables LALMs to effectively capture paralinguistic attributes such as speaker identity and speaking style.

These results demonstrate that the prompt-based last token strategy is more suitable for leveraging LALMs in retrieval tasks, though overall performance remains limited without instruction-aware retrieval training.

## 6.7. Layer-wise Analysis of Self-Supervised Speech Models

![](images/16e6e5f6651061433b12869bd9b477e2cc2ddb6cf7ff4645785fd201c5abf382.jpg)

![](images/984c34bff8ba4e1f0ae5d60ba0abe39ed4f519a4fbc0ab08bdf9a7e4361934f2.jpg)  
Figure 5: Layer-wise analysis of self-supervised speech models.

To understand how different layers of self-supervised speech models encode information relevant to various retrieval tasks, we analyze layer-wise performance for HuBERT-Large and WavLM-Large. As described in Section 4.3, we extract representations from different layers and compute retrieval performance. Figure 5 presents the results.

For both HuBERT and WavLM, performance on VCTK peaks in the lower to middle layers and then declines in deeper layers, indicating that speaker-related information is primarily encoded early in the network. DailyTalk shows moderate performance with peaks at specific layers, suggesting that semantic information emerges more clearly in intermediate to higher layers. Expresso follows a similar but lower trend, reflecting a balance between speaker and style representations across layers.

In contrast, Synthetic consistently exhibits the lowest recall across all layers for both models, highlighting the challenge of modeling complex instructions that combine semantics, speaker, style, and environmental factors. These findings align with prior work [67], suggesting that different layers capture different types of information, and no single layer provides optimal representations for all retrieval intents in instructionaware speech retrieval.

## 6.8. Analysis for Contrastive Audio-Language Embeddings

Table 9: Modality analysis for a contrastive audio-language model. A and T denote audio and text, respectively. $X  Y$ indicates that the query is encoded with the X encoder and the document with the Y encoder.
<table><tr><td></td><td colspan="2">DailyTalk</td><td colspan="2">VCTK</td><td colspan="2">Expresso</td><td colspan="2">Synthetic</td></tr><tr><td>Modality</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td><td>R@10</td><td>R@50</td></tr><tr><td> $\mathbf A \to \mathbf A$ </td><td>5.00</td><td>13.00</td><td>16.88</td><td>31.88</td><td>13.88</td><td>20.94</td><td>3.28</td><td>8.15</td></tr><tr><td> $\mathrm { T } { \to } \mathbf { A }$ </td><td>0.00</td><td>1.00</td><td>1.25</td><td>4.38</td><td>3.13</td><td>6.44</td><td>0.45</td><td>1.70</td></tr><tr><td> $\mathbf { A } { \to } \mathbb { T }$ </td><td>0.00</td><td>1.50</td><td>1.88</td><td>4.38</td><td>1.94</td><td>5.63</td><td>0.80</td><td>2.22</td></tr><tr><td> $\mathrm { T } { \to } \mathrm { T }$ </td><td>7.00</td><td>17.00</td><td>0.63</td><td>5.00</td><td>0.81</td><td>5.50</td><td>0.72</td><td>2.70</td></tr></table>

To better understand the modality behavior of contrastive audio-language models, we analyze CLAP performance across different encoder configurations. As Section 4.4 describes, contrastive audio-language model provides separate text and audio encoders that can be combined in various ways. We evaluate four configurations based on the encoder choice for the query and document sides. For configurations that use the text encoder, we first convert spoken utterances to text via ASR and audio captioning. When the instruction z is available, we concatenate it with the text input for configurations that employ the text encoder on the query side. For configurations that use the audio encoder on the query side, we do not include z because the audio encoder does not accept textual input. Table 9 summarizes the results.

Audio-to-audio retrieval outperforms other configurations on VCTK, Expresso, and Synthetic, indicating that the contrastive audio-language model effectively captures acoustic attributes such as speaker identity, speaking style, and environmental sounds within the audio modality. Text-to-text retrieval achieves the best performance on DailyTalk, which primarily emphasizes semantic understanding rather than acoustic details.

In contrast, cross-modal retrieval consistently yields low recall across all datasets. Both cross-modal configurations underperform their single-modality counterparts, highlighting the limited cross-modal alignment of the contrastive audiolanguage model for instruction-aware speech retrieval.

## 7. Conclusion

We introduced INSPIRE, the first benchmark for instructionaware speech retrieval, formalizing a paradigm where naturallanguage instructions dynamically describe relevance across semantic, speaker, style, and environmental attributes. Our evaluation reveals a clear modality specialization gap: cascaded pipelines excel at semantic matching but struggle with paralinguistic attributes, while self-supervised speech models capture acoustic properties but lack instruction sensitivity. LALMs and pre-trained language model approaches do not bridge this gap. The most challenging multi-attribute scenarios remain largely unsolved by all methods. These findings motivate future work on unified multimodal embedding models with instructionaware training that jointly preserve fine-grained acoustic information and support compositional instruction following.

## 8. Generative AI Use Disclosure

We use generative AI tools exclusively for polishing the manuscript. We maintain full responsibility for the study’s design, data analysis, and scientific interpretations, which remain entirely independent of AI influence.

## 9. Acknowledgements

We thank Tzu-Han Lin, Yu-Xiang Lin, and Cheng-Han Chiang from National Taiwan University for their helpful discussions and feedback. This work was financially supported by the National Science and Technology Council (NSTC) in Taiwan. We thank to National Center for High-performance Computing (NCHC) of National Applied Research Laboratories (NARLabs) in Taiwan for providing computational and storage resources. This work was supported by the Ministry of Education (MOE) of Taiwan under the project Taiwan Centers of Excellence in Artificial Intelligence, through the NTU Artificial Intelligence Center of Research Excellence (NTU AI-CoRE).

## 10. References

[1] L.-s. Lee, J. Glass, H.-y. Lee, and C.-a. Chan, “Spoken content retrieval—beyond cascading speech recognition with text retrieval,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 23, no. 9, pp. 1389–1420, 2015.

[2] C.-J. Lin, G.-T. Lin, Y.-S. Chuang, W.-L. Wu, S.-W. Li, A. Mohamed, H.-y. Lee, and L.-S. Lee, “Speechdpr: End-to-end spoken passage retrieval for open-domain spoken question answering,” in 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2024, pp. 12 476–12 480.

[3] A. Asai, T. Schick, P. Lewis, X. Chen, G. Izacard, S. Riedel, H. Hajishirzi, and W.-t. Yih, “Task-aware retrieval with instructions,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 3650–3675.

[4] T. Jiang, M. Song, Z. Zhang, H. Huang, W. Deng, F. Sun, Q. Zhang, D. Wang, and F. Zhuang, “E5-v: Universal embeddings with multimodal large language models,” arXiv preprint arXiv:2407.12580, 2024.

[5] C.-W. Ao and H.-y. Lee, “Query-by-example spoken term detection using attention-based multi-hop networks,” in 2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2018, pp. 6264–6268.

[6] D. Ram, L. Miculicich, and H. Bourlard, “Cnn based query by example spoken term detection.” in Interspeech, 2018, pp. 92–96.

[7] ——, “Neural network based end-to-end query by example spoken term detection,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 28, pp. 1416–1427, 2020.

[8] M. Witbrock and A. G. Hauptmann, “Speech recognition and information retrieval: Experiments in retrieving spoken documents,” in Proceedings of the DARPA speech recognition workshop, vol. 97, 1997.

[9] J. S. Garofolo, C. G. Auzanne, E. M. Voorhees et al., “The trec spoken document retrieval track: A success story.” NIST SPECIAL PUBLICATION SP, vol. 500, no. 246, pp. 107–130, 2000.

[10] C. Chelba, T. J. Hazen, and M. Saraclar, “Retrieval and browsing of spoken content,” IEEE Signal Processing Magazine, vol. 25, no. 3, pp. 39–49, 2008.

[11] M. Larson and G. J. Jones, “Spoken content retrieval: A survey of techniques and technologies,” Foundations and Trends® in Information Retrieval, vol. 5, no. 4–5, pp. 235–422, 2012.

[12] R. Hu, Y. Xia, M. Hong, J. Zhu, B. Chen, X. Yang, M. Fang, and T. Jin, “Vela: Scalable embeddings with voice large language models for multimodal retrieval,” in Interspeech 2025, 2025, pp. 2640–2644.

[13] J. Turmo, P. R. Comas, S. Rosset, O. Galibert, N. Moreau, D. Mostefa, P. Rosso, and D. Buscaldi, “Overview of qast 2009,” in Workshop of the Cross-Language Evaluation Forum for Euro pean Languages. Springer, 2009, pp. 197–211.

[14] P. R. Comas, J. Turmo, and L. Marquez, “Sibyl, a factoid\` question-answering system for spoken documents,” ACM Transactions on Information Systems (TOIS), vol. 30, no. 3, pp. 1–40, 2012.

[15] C.-H. Li, S.-L. Wu, C.-L. Liu, and H.-y. Lee, “Spoken squad: A study of mitigating the impact of speech recognition errors on listening comprehension,” arXiv preprint arXiv:1804.00320, 2018.

[16] C.-H. Lee, S.-M. Wang, H.-C. Chang, and H.-Y. Lee, “Odsqa: Open-domain spoken question answering dataset,” in 2018 IEEE Spoken Language Technology Workshop (SLT). IEEE, 2018, pp. 949–956.

[17] G.-T. Lin, Y.-S. Chuang, H.-L. Chung, S. wen Yang, H.-J. Chen, S. A. Dong, S.-W. Li, A. Mohamed, H. yi Lee, and L. shan Lee, “Dual: Discrete spoken unit adaptive learning for textless spoken question answering,” in Interspeech 2022, 2022, pp. 5165–5169.

[18] C. You, N. Chen, F. Liu, S. Ge, X. Wu, and Y. Zou, “End-toend spoken conversational question answering: Task, dataset and model,” in Findings of the Association for Computational Linguistics: NAACL 2022, 2022, pp. 1219–1232.

[19] S. Shon, S. Arora, C.-J. Lin, A. Pasad, F. Wu, R. Sharma, W.- L. Wu, H.-y. Lee, K. Livescu, and S. Watanabe, “Slue phase-2: A benchmark suite of diverse spoken language understanding tasks,” in Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023, pp. 8906–8937.

[20] H. Su, W. Shi, J. Kasai, Y. Wang, Y. Hu, M. Ostendorf, W.-t. Yih, N. A. Smith, L. Zettlemoyer, and T. Yu, “One embedder, any task: Instruction-finetuned text embeddings,” in Findings of the Association for Computational Linguistics: ACL 2023, 2023, pp. 1102–1121.

[21] L. Wang, N. Yang, X. Huang, L. Yang, R. Majumder, and F. Wei, “Improving text embeddings with large language models,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2024, pp. 11 897–11 916.

[22] C. Lee, R. Roy, M. Xu, J. Raiman, M. Shoeybi, B. Catanzaro, and W. Ping, “NV-embed: Improved techniques for training LLMs as generalist embedding models,” in The Thirteenth International Conference on Learning Representations, 2025.

[23] J. Lee, F. Chen, S. Dua et al., “Gemini embedding: Generalizable embeddings from gemini,” arXiv preprint arXiv:2503.07891, 2025.

[24] Y. Zhang, M. Li, D. Long et al., “Qwen3 embedding: Advancing text embedding and reranking through foundation models,” arXiv preprint arXiv:2506.05176, 2025.

[25] H. Wu, Y. Gao, X. Guo, Z. Al-Halah, S. Rennie, K. Grauman, and R. Feris, “Fashion iq: A new dataset towards retrieving images by natural language feedback,” in Proceedings of the IEEE/CVF Conference on computer vision and pattern recognition, 2021, pp. 11 307–11 317.

[26] Z. Liu, C. Rodriguez-Opazo, D. Teney, and S. Gould, “Image retrieval on real-life images with pre-trained vision-and-language models,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 2125–2134.

[27] A. Baldrati, L. Agnolucci, M. Bertini, and A. Del Bimbo, “Zeroshot composed image retrieval with textual inversion,” in Proceedings ofthe IEEE/CVF International Conference on Computer Vi sion, 2023, pp. 15 338–15 347.

[28] K. Saito, K. Sohn, X. Zhang, C.-L. Li, C.-Y. Lee, K. Saenko, and T. Pfister, “Pic2word: Mapping pictures to words for zeroshot composed image retrieval,” in Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023, pp. 19 305–19 314.

[29] K. Zhang, Y. Luan, H. Hu, K. Lee, S. Qiao, W. Chen, Y. Su, and M.-W. Chang, “Magiclens: Self-supervised image retrieval with open-ended instructions,” in Forty-first International Conference on Machine Learning, 2024.

[30] X. Zhang, Y. Zhang, W. Xie, M. Li, Z. Dai, D. Long, P. Xie, M. Zhang, W. Li, and M. Zhang, “Bridging modalities: Improving universal multimodal retrieval by multimodal large language models,” in Proceedings of the Computer Vision and Pattern Recognition Conference, 2025, pp. 9274–9285.

[31] J. Zhou, Y. Xiong, Z. Liu, Z. Liu, S. Xiao, Y. Wang, B. Zhao, C. J. Zhang, and D. Lian, “Megapairs: Massive data synthesis for universal multimodal retrieval,” in Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), 2025, pp. 19 076–19 095.

[32] Z. Jiang, R. Meng, X. Yang, S. Yavuz, Y. Zhou, and W. Chen, “VLM2vec: Training vision-language models for massive multimodal embedding tasks,” in The Thirteenth International Conference on Learning Representations, 2025.

[33] R. Meng, Z. Jiang, Y. Liu et al., “Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents,” arXiv preprint arXiv:2507.04590, 2025.

[34] M. Li, Y. Zhang, D. Long et al., “Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking,” arXiv preprint arXiv:2601.04720, 2026.

[35] J. Shor, A. Jansen, R. Maor, O. Lang, O. Tuval, F. de Chaumont Quitry, M. Tagliasacchi, I. Shavitt, D. Emanuel, and Y. Haviv, “Towards learning a universal non-semantic representation of speech,” in Interspeech 2020, 2020, pp. 140–144.

[36] S. wen Yang, P.-H. Chi, Y.-S. Chuang et al., “Superb: Speech processing universal performance benchmark,” in Interspeech 2021, 2021, pp. 1194–1198.

[37] L. Wang, P. Luc, Y. Wu et al., “Towards learning universal audio representations,” in 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2022, pp. 4593–4597.

[38] G. Heigold, E. Variani, T. Bagby, C. Allauzen, J. Ma, S. Kumar, and M. Riley, “Massive sound embedding benchmark (MSEB),” in The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2025.

[39] A. E. Assadi, I. Chung, C. Xiao et al., “Maeb: Massive audio embedding benchmark,” arXiv preprint arXiv:2602.16008, 2026.

[40] K. Lee, K. Park, and D. Kim, “Dailytalk: Spoken dialogue dataset for conversational text-to-speech,” in 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[41] J. Yamagishi, C. Veaux, and K. MacDonald, “CSTR VCTK Corpus: English multi-speaker corpus for CSTR voice cloning toolkit (version 0.92),” 2019.

[42] T. A. Nguyen, W.-N. Hsu, A. D’Avirro, B. Shi, I. Gat, M. Fazel-Zarani, T. Remez, J. Copet, G. Synnaeve, M. Hassid, F. Kreuk, Y. Adi, and E. Dupoux, “Expresso: A benchmark and analysis of discrete expressive speech resynthesis,” in Interspeech 2023, 2023, pp. 4823–4827.

[43] T. Kwiatkowski, J. Palomaki, O. Redfield et al., “Natural questions: a benchmark for question answering research,” Transactions of the Association for Computational Linguistics, vol. 7, pp. 453–466, 2019.

[44] OpenAI, “Gpt-4o system card,” 2024.

[45] K. J. Piczak, “Esc: Dataset for environmental sound classification,” in Proceedings of the 23rd ACM international conference on Multimedia, 2015, pp. 1015–1018.

[46] OpenAI, “Openai gpt-5 system card,” 2025.

[47] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, “Robust speech recognition via large-scale weak supervision,” in International conference on machine learning. PMLR, 2023, pp. 28 492–28 518.

[48] B. Desplanques, J. Thienpondt, and K. Demuynck, “Ecapa-tdnn: Emphasized channel attention, propagation and aggregation in tdnn based speaker verification,” in Interspeech 2020, 2020, pp. 3830–3834.

[49] M. Ravanelli, T. Parcollet, P. Plantinga et al., “Speechbrain: A general-purpose speech toolkit,” arXiv preprint arXiv:2106.04624, 2021.

[50] J. Platt et al., “Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods,” Advances in large margin classifiers, vol. 10, no. 3, pp. 61–74, 1999.

[51] C.-C. Chang and C.-J. Lin, “Libsvm: A library for support vector machines,” ACM transactions on intelligent systems and technology (TIST), vol. 2, no. 3, pp. 1–27, 2011.

[52] Z. Ma, Z. Zheng, J. Ye, J. Li, Z. Gao, S. Zhang, and X. Chen, “emotion2vec: Self-supervised pre-training for speech emotion representation,” in Findings ofthe Associationfor Computational Linguistics: ACL 2024, 2024, pp. 15 747–15 760.

[53] T. Saeki, D. Xin, W. Nakata, T. Koriyama, S. Takamichi, and H. Saruwatari, “Utmos: Utokyo-sarulab system for voicemos challenge 2022,” in Interspeech 2022, 2022, pp. 4521–4525.

[54] T. Jiang, S. Huang, Z. Luan, D. Wang, and F. Zhuang, “Scaling sentence embeddings with large language models,” in Findings of the association for computational linguistics: EMNLP 2024, 2024, pp. 3182–3196.

[55] B. Elizalde, S. Deshmukh, M. Al Ismail, and H. Wang, “Clap learning audio concepts from natural language supervision,” in 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[56] Y. Wu, K. Chen, T. Zhang, Y. Hui, T. Berg-Kirkpatrick, and S. Dubnov, “Large-scale contrastive language-audio pretraining with feature fusion and keyword-to-caption augmentation,” in 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2023, pp. 1–5.

[57] A. Goel, S. Ghosh, J. Kim et al., “Audio flamingo 3: Advancing audio intelligence with fully open large audio language models,” arXiv preprint arXiv:2507.08128, 2025.

[58] J. Xu, Z. Guo, J. He et al., “Qwen2. 5-omni technical report,” arXiv preprint arXiv:2503.20215, 2025.

[59] J. Xu, Z. Guo, H. Hu et al., “Qwen3-omni technical report,” arXiv preprint arXiv:2509.17765, 2025.

[60] A. H. Liu, A. Ehrenberg, A. Lo et al., “Voxtral,” arXiv preprint arXiv:2507.13264, 2025.

[61] S. Robertson, H. Zaragoza et al., “The probabilistic relevance framework: Bm25 and beyond,” Foundations and trends® in information retrieval, vol. 3, no. 4, pp. 333–389, 2009.

[62] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), 2019, pp. 3982–3992.

[63] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, K. Lakhotia, R. Salakhutdinov, and A. Mohamed, “Hubert: Self-supervised speech representation learning by masked prediction of hidden units,” IEEE/ACM transactions on audio, speech, and language processing, vol. 29, pp. 3451–3460, 2021.

[64] S. Chen, C. Wang, Z. Chen et al., “Wavlm: Large-scale selfsupervised pre-training for full stack speech processing,” IEEE Journal of Selected Topics in Signal Processing, vol. 16, no. 6, pp. 1505–1518, 2022.

[65] Google, “A new era of intelligence with gemini 3,” 2025. [Online]. Available: https://blog.google/products/gemini/gemini-3

[66] “New embedding models and api updates,” 2024. [Online]. Available: https://openai.com/index/ new-embedding-models-and-api-updates/

[67] A. Pasad, J.-C. Chou, and K. Livescu, “Layer-wise analysis of a self-supervised speech representation model,” in 2021 IEEE Automatic Speech Recognition and Understanding Workshop (ASRU). IEEE, 2021, pp. 914–921.