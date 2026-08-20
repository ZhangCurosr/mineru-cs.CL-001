# OmniAlign: A Unified Multilingual Aligner for Word and Sentence Alignment

Mengpeng Yang Jingxu Yang Chao Chen Tian Xia Yabo Sun Qiang Liu

WPS Qingqiu / Wuhan, China

yangmengpeng@wps.cn, yangjingxu1@wps.cn, chenchao9@wps.cn xiatian3@wps.cn, sunyabo@wps.cn, liuqiang2@wps.cn

## Abstract

Cross-lingual sequence alignment is fundamental for building and exploiting parallel corpora, spanning mappings from documents and sentences down to words and subwords. Existing tools, however, typically specialize in a single granularity, so practitioners often need separate systems for word- and sentence-level alignment—especially in multilingual and long-text settings. We present OmniAlign, a unified multilingual aligner that supports both word-level and sentence-level alignment with a single lightweight model. Built on an encoder-only backbone with strong long-context modeling, OmniAlign induces word alignments from contextualized token similarity matrices, and obtains document-level m–n sentence alignments via sentence embeddings combined with dynamic programming. To balance fine-grained alignment accuracy and sentencerepresentation quality, we use a four-stage training pipeline: alignment-oriented continued pre-training, self-supervised learning, supervised fine-tuning on human annotations, and sentence-embedding distillation from a strong multilingual teacher. Experiments show that OmniAlign achieves highly competitive performance on both word- and sentence-alignment benchmarks and generalizes well to unseen language pairs. Surprisingly, later-stage supervised fine-tuning on short texts further improves alignment quality while retaining the long-context understanding acquired in earlier training, keeping the model robust on long-text word alignment.

Code: https://github.com/MilkDargon/OmniAlign

Model: https://huggingface.co/WPS-Qingqiu/OmniAlign

## 1 Introduction

Cross-lingual Sequence Alignment is a core foundational task in the construction and exploitation of parallel corpora. Its goal is to establish precise and consistent correspondences between text sequences across different languages. This task typically spans multiple levels of granularity—from coarse-grained alignment at the document or paragraph level, to sentence-level alignment, and further down to finegrained word- or subword-level alignment. Together, these alignment processes form the critical pathway for cross-lingual information mapping, enabling models to build stable semantic correspondences in multilingual spaces. They provide essential support for tasks such as building para-crawl pipelines, constructing translation memories in computer-assisted translation systems (Varga et al., 2008), projecting linguistic annotations and XML structural labels across languages (David et al., 2001; Hashimoto et al., 2019), detecting under-translation in neural machine translation (Tu et al., 2016), developing cross-lingual transfer NLP tools (Nicolai & Yarowsky, 2019; Mayhew et al., 2017), and improving or augmenting multilingual pre-trained models (Chi et al., 2021).

Despite the emergence of various models targeting sequence alignment at different granularities in recent years, there is still a lack of a single practical aligner that covers both word- and sentence-level needs while remaining multilingual and long-context capable. Previous studies typically focus on only one subtask of the alignment problem. For example, methods such as (Sabet et al., 2020; Dou & Neubig, 2021; Wang et al., 2022; Wu et al., 2023; Nagata et al., 2020; Lai et al., 2022; Latouche et al., 2024) primarily explore word alignment; however, most of them suffer from significant performance degradation as sequence length increases and are unable to handle alignment scenarios exceeding 512 tokens. Some approaches claim multilingual support, yet they require training separate models for different languages and do not also provide sentence-level alignment in the same model.

On the other hand, methods such as (Molfese et al., 2024; Steingrímsson et al., 2023; Thompson & Koehn,

(a) Word Alignment Example. Red boxes denote the gold alignments.  
![](images/db6704460dff94d347ae2bf943c5d909afcc60eedcdabab56990539ae7d91dac.jpg)  
(b) Sentence Alignment Example.  
Figure 1: Sequence Alignment Diagram.

2019; Liu & Zhu, 2023) rely on cross-lingual representation models like LaBSE (Feng et al., 2022) or LASER (Artetxe & Schwenk, 2019) to obtain sentence embeddings and determine sentence alignment via similarity computation. However, these methods do not provide fine-grained word-level alignment information.

To address these limitations, we propose OmniAlign—a unified multilingual aligner that supports both word-level and sentence-level alignment with a single model. Word and sentence outputs are obtained from the same encoder via task-specific alignment procedures, enabling developers to meet diverse cross-lingual alignment needs without maintaining separate systems. The main contributions of this work are as follows:

1. We introduce OmniAlign. A novel open-source multilingual sequence alignment model with only 0.3B parameters and efficient inference. It supports bidirectional word- and sentence-level alignment across eleven major languages.

2. Multi-stage training recipe. We design a multi-stage training framework that first enhances the model’s token-level semantic representations through continued pre-training. We then combine selfsupervised learning with supervised signals to optimize token-level alignment accuracy and obtain the word-alignment model. Finally, we freeze the word-alignment module and employ knowledge distillation to restore and further strengthen the model’s sentence-level representation capability.

3. Broad applicability and competitive performance. OmniAlign supports a wide range of alignment tasks encountered in real-world applications and achieves highly competitive results on both sentencelevel and word-level alignment benchmarks, remaining robust on long-text word alignment—where short-text supervised fine-tuning further improves quality—and generalizing well to unseen language pairs.

<table><tr><td>Methods</td><td>Multilingual</td><td>Long Text</td><td>Word Alignment</td><td>Sentence Alignment</td></tr><tr><td>FastAlign (Dver et al., 2013)</td><td></td><td></td><td></td><td></td></tr><tr><td>GIZA++ (Och &amp; Ney, 2003)</td><td></td><td></td><td>V</td><td></td></tr><tr><td>SimAlign (Sabet et al., 2020)</td><td></td><td></td><td></td><td></td></tr><tr><td>AwesomeAlign (Dou &amp; Neubig, 2021)</td><td></td><td></td><td>√</td><td></td></tr><tr><td>AccAlign (Wang et al., 2022)</td><td></td><td></td><td></td><td></td></tr><tr><td>WSPAlign (Wu et al., 2023)</td><td></td><td></td><td></td><td></td></tr><tr><td>BinaryAlign (Latouche et al., 2024)</td><td></td><td></td><td></td><td></td></tr><tr><td>Gale-Church (Gale &amp; Church, 1993)</td><td></td><td></td><td></td><td></td></tr><tr><td>BleuAlign (Sennrich &amp; Volk, 2011)</td><td></td><td></td><td></td><td></td></tr><tr><td>VecAlign (Thompson &amp; Koehn, 2019)</td><td></td><td></td><td></td><td></td></tr><tr><td>BertAlign (Liu &amp; Zhu, 2023)</td><td></td><td></td><td></td><td></td></tr><tr><td>SentAlign (Steingrímsson et al., 2023)</td><td></td><td></td><td></td><td></td></tr><tr><td>CrocoAlign (Molfese et al., 2024)</td><td></td><td></td><td></td><td></td></tr><tr><td>OmniAlign</td><td></td><td></td><td></td><td></td></tr></table>

Table 1: This report compares OmniAlign with a range of existing state-of-the-art sequence alignment methods. The symbol “–” indicates that the capability is independent of the specific model choice.

The remainder of this paper is organized as follows: we first describe the training methodology of

OmniAlign, then present experimental results for each stage of the model variants, and finally conclude with key findings.

## 2 OmniAlign

This section focuses on the operational mechanism of OmniAlign in sequence alignment tasks and elaborates on its training pipeline.

![](images/3b1a751d453da6192490cf4ecf4b65a47d15586c3d0633bb2eec1a992f9bb00a.jpg)

![](images/a4d8daa99b3cd099a0793812249b702e9315521fe3f675b9d4289244891a5b12.jpg)  
(b) Training Pipeline  
Figure 2: Overall architecture of OmniAlign, including the sequence alignment model (left) and the multi-stage training pipeline (right).

## 2.1 Base Model

We adopt the mGTE base model (Zhang et al., 2024) as the base model for OmniAlign, in order to support multi-granularity, cross-lingual, and variable-length sequence alignment. The main reasons are as follows:

• Encoder-only architectures provide stronger contextual modeling. Compared with Seq2Seq or decoder-only models, encoder-only models offer inherent advantages in capturing intra-sentence token dependencies. This enables more fine-grained semantic representation learning, making them particularly suitable for word-level alignment tasks.

• Cross-lingual alignment requires robust multilingual capability. Effective word- and sentence-level alignment across languages requires the model to have strong and reliable multilingual capability.

• mGTE overcomes the sequence-length limitations of traditional multilingual models. Conventional models such as mBERT (Pires et al., 2019) and the XLM family (Conneau et al., 2020) are constrained by a maximum sequence length of 512 tokens, making them insufficient for long-text alignment scenarios. In contrast, mGTE supports significantly longer input sequences, enabling alignment over extended contexts.

## 2.2 Extracting Word Alignments from Token Embeddings

To enable OmniAlign to support both word-level and sentence-level alignment, we induce word alignments from the contextualized token embeddings of multilingual pre-trained language models.

As shown in Figure 2(a), given a source text $\mathbf { x } = \langle x _ { 1 } , x _ { 2 } , \ldots , x _ { i } \rangle$ and a target text $\mathbf { y } = \langle y _ { 1 } , y _ { 2 } , \dots , y _ { j } \rangle$ , we denote the contextualized token embeddings from the m-th layer of OmniAlign as $\mathbf { s } = \langle s _ { 1 } , s _ { 2 } , \ldots , s _ { i } \rangle$ and

$\mathbf { t } = \langle t _ { 1 } , t _ { 2 } , \ldots , t _ { j } \rangle$ , respectively. Following prior work (Sabet et al., 2020; Dou & Neubig, 2021; Wang et al., 2022), we compute the dot product between s and t to obtain the similarity matrix.

$$
{ \pmb S } = { \bf s t } ^ { \mathrm { T } } ,\tag{1}
$$

each row of the similarity matrix S is normalized using a Softmax function to obtain the source-to-target probability alignment matrix $\mathbf { S } _ { x y }$ . The i-th row of $\mathbf { S } _ { x y }$ represents the alignment probabilities between token $x _ { i }$ and all tokens in $\mathbf { y } .$ Similarly, we compute the target-to-source probability alignment matrix $\mathsf { \pmb { S } } _ { y x } ,$ and the final alignment matrix is obtained by taking the intersection of the two probability matrices.

$$
\mathbf { A } = ( \mathbf { S } _ { x y } > c ) \ast ( \mathbf { S } _ { y x } ^ { T } > c ) ,\tag{2}
$$

where c denotes the threshold, and $\mathbf { A } _ { i j } = 1$ indicates that $x _ { i }$ and $y _ { j }$ are aligned. The above procedure induces alignments at the token level; if any pair of tokens is aligned, the corresponding words are aligned to produce the final word-level alignment results.

## 2.3 Extracting Sentence Alignments From Sentence Embeddings

To support sentence-level parallel corpus alignment, OmniAlign integrates a two-stage alignment algorithm inspired by Bertalign (Liu & Zhu, 2023).

Similarity Retrieval-Based 1–1 Alignment. Given a source document $\mathbf { X } = \langle X _ { 1 } , X _ { 2 } , \ldots , X _ { N } \rangle$ and a target document $\mathbf { Y } = \langle Y _ { 1 } , Y _ { 2 } , \ldots , Y _ { M } \rangle$ , sentence encoders map each sentence to a dense vector representation, yielding source sentence embeddings $\mathbf { S } = \langle S _ { 1 } , S _ { 2 } , \ldots , S _ { N } \rangle$ and target sentence embeddings $\mathbf { T } = \langle T _ { 1 } , T _ { 2 } ^ { \ ' } , \ldots , T _ { M } ^ { ' } \rangle$

We first perform maximum inner-product search to retrieve the top-k candidate target sentences for each source sentence:

$$
\begin{array} { r } { ( \mathbf { D } , \mathbf { I } ) = \mathrm { T o p K } ( \mathbf { S } \mathbf { T } ^ { \mathrm { T } } , k ) , } \end{array}\tag{3}
$$

where $\mathbf { S T } ^ { \mathrm { T } }$ denotes the similarity matrix between source and target sentences. The top-k operation yields the similarity score matrix D and the corresponding target sentence index matrix I for each source sentence.

Subsequently, a dynamically adjusted diagonal search window is constructed, within which dynamic programming is applied to determine the optimal alignment path. The state transition $( a _ { 1 } , a _ { 2 } ) \in$ $\big \{ ( 0 , 1 ) , ( 1 , 0 ) , ( 1 , 1 ) \big \}$ corresponds to target sentence insertion, source sentence deletion, and one-to-one sentence alignment, respectively. The scoring function is defined as:

$$
\displaystyle \mathrm { s c o r e } ( i , j ) = \operatorname* { m a x } _ { ( a _ { 1 } , a _ { 2 } ) } \left[ \mathrm { s c o r e } ( i - a _ { 1 } , j - a _ { 2 } ) + \delta _ { 1 - 1 } ( a _ { 1 } , a _ { 2 } , i , j ) \right] .\tag{4}
$$

Here, $\delta _ { 1 - 1 }$ is an indicator function that contributes the similarity score from D only when the alignment operation is $^ { ( 1 , 1 ) }$ and the target sentence j belongs to the top-k candidate set I of the source sentence $i ;$ otherwise, it contributes zero.

Finally, by backtracking the dynamic programming table, we obtain a sequence of one-to-one alignment anchor points for the entire document.

Anchor-Constrained m–n Alignment. Guided by the 1–1 anchors, we restrict the second-stage search to local regions around semantically adjacent anchor points, enabling flexible m–n alignments. In the second stage, more flexible alignment patterns are permitted $\left( a _ { 1 } , a _ { 2 } \right)$ (where $a _ { 1 } + a _ { 2 } \geq 2 )$ . The alignment score for this stage is defined as:

$$
\begin{array} { r } { \mathrm { s i m } ( i , j , a _ { 1 } , a _ { 2 } ) = \langle \mathbf { S } _ { a _ { 1 } } ( i ) , \mathbf { T } _ { a _ { 2 } } ( j ) \rangle \times \mathrm { L e n g t h P e n a l t y } , } \end{array}\tag{5}
$$

where, $\langle \mathbf { S } _ { a _ { 1 } } ( i ) , \mathbf { T } _ { a _ { 2 } } ( j ) \rangle$ denotes the inner product between the embedding vectors of the source text span $\mathbf { S } _ { a _ { 1 } } ( i )$ and the target text span $\mathbf { T } _ { a _ { 2 } } ( j )$ , while LengthPenalty represents a penalty term based on the token-length ratio difference. Compared to constraints based on sentence counts, this token-level ratio is more robust in scenarios involving languages with substantial structural differences, such as Chinese and English.

The final alignment results are obtained by backtracking the dynamic programming table in the second stage, and can be represented as: $\mathcal { A } = \{ ( \check { \left[ i _ { 1 } { : } i _ { 2 } \right] } , [ j _ { 1 } { : } j _ { 2 } ] ) , { : } \ldots \}$ . A denotes the document-level $m - n$ alignment set. The terms $[ i _ { 1 } { : } i _ { 2 } ] , [ j _ { 1 } { : } j _ { 2 } ]$ correspond to contiguous segments in the source document spanning from the $i _ { 1 }$ -th sentence to the i<sub>2</sub>-th sentence, and contiguous segments in the target document spanning from the j -th sentence to the $j _ { 2 } { \cdot } \mathrm { t h }$ sentence, respectively. Each pair of $\left( \left[ i _ { 1 } { : } i _ { 2 } \right] , \left[ j _ { 1 } { : } j _ { 2 } \right] \right)$ constitutes a minimal sentence-level alignment unit.

Table 2: Supported Languages Evaluated
<table><tr><td>Languages</td><td>ISO 639-1</td><td>Languages</td><td>ISO 639-1</td><td>Languages</td><td>ISO 639-1</td></tr><tr><td>Chinese</td><td>zh</td><td>Spanish</td><td>es</td><td>Italian</td><td>it</td></tr><tr><td>English</td><td>en</td><td>Japanese</td><td>ja</td><td>Korean</td><td>ko</td></tr><tr><td>French</td><td>fr</td><td>Russian</td><td>ru</td><td>German</td><td>de</td></tr><tr><td>Portuguese</td><td>pt</td><td>Romanian</td><td>ro</td><td></td><td></td></tr></table>

## 2.4 Training Pipeline

To enable OmniAlign to achieve strong performance in both word-level and sentence-level alignment, we design a four-stage training strategy (as illustrated in Fig. 2(b)).

Continued Pre-training. We first evaluate the initial mGTE base model on word alignment (see Section 3.5.2). The results show that its zero-shot alignment capability is relatively limited. Following the previous works (Pires et al., 2019; Conneau et al., 2020; Gururangan et al., 2020; Conneau & Lample, 2019), we further conduct continued pre-training on mGTE.

Specifically, given a parallel sentence pair $( x , y )$ , we construct four input variants: $x , y , [ x , y ] ,$ , and $[ y , x ]$ From each input, 15% of tokens are randomly selected as prediction targets. Among the selected tokens, 80% are replaced with the [MASK] token, 10% are left unchanged, and the remaining 10% are randomly substituted with other tokens from the vocabulary. The model is then trained to recover the original masked tokens based on the surrounding context.

$$
\mathcal { L } _ { 1 } = \log p ( x \mid x ^ { \mathsf { m a s k } } ) + \log p ( y \mid y ^ { \mathsf { m a s k } } ) + \log p ( x , y \mid [ x , y ] ^ { \mathsf { m a s k } } ) + \log p ( y , x \mid [ y , x ] ^ { \mathsf { m a s k } } ) .\tag{6}
$$

The source texts for continued pre-training are drawn from FineWeb2<sup>1</sup>, covering a mixture of short and long documents. Corresponding translations are synthesized using Qwen3-235B-A22B<sup>2</sup> to construct a parallel corpus of approximately 9 million pairs.

Self-supervised Learning. Let A<sup>ˆ</sup> denote the alignment labels for a given sentence pair x and y. During training, we derive pseudo-labels A<sup>ˆ</sup> from the model’s online predictions of alignment relations and use them to construct the training objective. For this stage, we sample 100,000 pairs from the parallel corpus constructed during continued pre-training, with an emphasis on long-text examples, for self-supervised optimization.

$$
\mathcal { L } _ { 2 } = \sum _ { i j } \hat { \mathbf { A } } _ { i j } \frac { 1 } { 2 } \left( \frac { ( \mathbf { S } _ { x y } ) _ { i j } } { n } + \frac { ( \mathbf { S } _ { y x } ^ { \mathrm { T } } ) _ { i j } } { m } \right) .\tag{7}
$$

This objective encourages the model to bring the contextual representations of aligned words closer together, while additionally enforcing a consistency constraint between $\mathbf { S } _ { x y }$ and $\begin{array} { r } { \mathbf { S } _ { y x } \mathbf { \bar { . } } } \end{array}$ Such a constraint promotes symmetry between the alignment probability matrices in both directions, thereby improving alignment quality and training stability.

Supervised Fine-tuning. In the final stage of word alignment training, the training objective remains consistent with that of the self-supervised learning stage, while the dataset used in this stage is described in Section 3.1. The key difference lies in performing supervised fine-tuning with human-annotated gold alignment labels instead of pseudo-labels, thereby further improving the model’s alignment accuracy.

Sentence Embedding Knowledge Distillation. To rapidly equip our trained word alignment model with sentence-level representation capability, we adopt a knowledge distillation approach following (Reimers & Gurevych, 2020) to efficiently recover sentence-level representations. Specifically, given a teacher model $\mathbf { T } ( \cdot )$ , it maps sentences from one or more source languages x into a shared dense vector space. In addition, we leverage translated sentences (which may span multiple languages) y , and minimize the following objective:

$$
\mathcal { L } _ { 4 } = \left[ \left( \mathbf { T } ( x ) - \mathbf { S } ( x ) \right) ^ { 2 } + \left( \mathbf { T } ( x ) - \mathbf { S } ( y ) \right) ^ { 2 } \right] ,\tag{8}
$$

where $\mathbf { S } ( \cdot )$ denotes the student model. At this stage, the construction procedure for the multilingual parallel corpus follows the same pipeline as that used during the continued pre-training phase, resulting in a total of approximately 6.82 million multilingual sentence pairs.

## 3 Experiments

## 3.1 Datasets

Word Alignment Datasets. We conduct experiments on human-annotated word alignment datasets covering nine language pairs: German–English (de–en), French–English (fr–en), Romanian–English (ro–en), Japanese–English (ja–en), Chinese–English (zh–en), Spanish–English (es–en), Portuguese–English (pt–en), Russian–English (ru–en), and Italian–English (it–en). The ja–en dataset is drawn from the word alignment corpus of the Kyoto Free Translation Task (KFTT)<sup>3</sup>, with data splits following the protocol described in (Wu et al., 2023). We use all eight development files for supervised fine-tuning, while four of the seven test files are used for evaluation; the remaining three test files are reserved for future studies. The ro–en and fr–en datasets are provided by (Mihalcea & Pedersen, 2003), and the de–en dataset is obtained from (Vilar et al., 2006). For the de–en, ro–en, and fr–en language pairs, we partition the original data into training and test sets, using 300 sentence pairs for supervised fine-tuning in the fr–en and de–en datasets and 150 sentence pairs in the ro–en dataset; all remaining sentence pairs are used for evaluation. The zh–en dataset is collected from the TsinghuaAligner website<sup>4</sup>. We adopt the test set used in (Dou & Neubig, 2021) for evaluation, while sentence pairs outside the test set are used for supervised fine-tuning. In addition, the es–en, pt–en, ru–en, and it–en datasets are sourced from (Martelli et al., 2023), where the development sets are used for supervised fine-tuning and the test sets are used for evaluation. Statistics of all datasets are summarized in Table 3.

Table 3: Statistics of the word alignment dataset, where "Avg. Tokens" denotes the statistics for the English source text in the test set.
<table><tr><td></td><td>zh-en</td><td>de-en</td><td>fr-en</td><td>ro-en</td><td>ja-en</td><td>es-en</td><td>pt-en</td><td>ru-en</td><td>it-en</td></tr><tr><td>Train Sents</td><td>40.7K</td><td>300</td><td>300</td><td>150</td><td>653</td><td>105</td><td>105</td><td>90</td><td>103</td></tr><tr><td>Test Sents</td><td>450</td><td>208</td><td>147</td><td>98</td><td>582</td><td>245</td><td>245</td><td>210</td><td>243</td></tr><tr><td>Avg. Tokens</td><td>37</td><td>21</td><td>16</td><td>30</td><td>27</td><td>19</td><td>19</td><td>13</td><td>19</td></tr></table>

Sentence Alignment Datasets. We evaluate sentence-level alignment performance on test sets spanning seven language directions, including Chinese–English (zh–en), Spanish–English (es–en), Italian–English (it–en), German–English (de–en), French–English (fr–en), Russian–English (ru–en), and German–French (de–fr). The zh-en test set is drawn from the political (pol) domain of the aligner-eval dataset<sup>5</sup>. The es–en, it–en, de–en, and fr–en test sets are obtained from (Molfese et al., 2024). In addition, the de–fr test set is sourced from the Text+Berg corpus (Volk et al., 2010).

## 3.2 Implementation

We initialize our model with mGTE<sup>6</sup>, which consists of a 12-layer Transformer encoder with a hidden size of 768, and perform continued pre-training for one epoch. During this stage, the batch size is set to 512, the maximum sequence length follows the original setting of 8192, and the learning rate is set to $2 e ^ { - 5 }$ . A held-out validation set of 10<sup>4</sup> samples is used for model selection.

The self-supervised learning stage adopts the same hyperparameter configuration as the continued pre-training stage. We hold out 500 samples as the validation set and perform one epoch of fine-tuning on the model. Based on empirical results, we induce alignments using representations from the 7th Transformer layer, with the alignment threshold set to 0.001, following the method described in (Dou & Neubig, 2021).

In the final five epochs of supervised fine-tuning for word alignment, the batch size is reduced to 64 and the learning rate is increased to 1e<sup>−4</sup>. The same alignment induction strategy is applied, using the 7th layer representations and an alignment threshold of 0.001.

To recover sentence-level representation capability, we perform Sentence Embedding Knowledge Distillation in the final stage. During this phase, parameters related to word alignment are frozen, including the embedding layer and the first seven Transformer layers, while only the last five layers are fine-tuned for five epochs. We adopt LaBSE<sup>7</sup> as the teacher model. The batch size for this stage is set to 256, the validation set contains 1,000 samples, and the learning rate is again set to $2 e ^ { - 5 }$

## 3.3 Baselines

To evaluate the effectiveness of OmniAlign, we conduct experiments on both word alignment and sentence alignment tasks.

Word Alignment Baselines. We consider two categories of word alignment baselines. Statistical approaches include FastAlign (Dyer et al., 2013) and GIZA++ (Och & Ney, 2003), two classical and widely used models implemented under the IBM Model framework. Pre-trained language model-based approaches comprise SimAlign (Sabet et al., 2020), Awesome-align (Dou & Neubig, 2021), AccAlign (Wang et al., 2022), WSPAlign (Wu et al., 2023), and BinaryAlign (Latouche et al., 2024). For WSPAlign (Bilingual), evaluation is conducted only on the de-en, fr-en, ro-en, ja-en, and zh-en language pairs, using checkpoints trained separately for each language direction. WSPAlign (Multilingual) is evaluated using its released multilingual checkpoint. For BinaryAlign (Bilingual), we evaluate checkpoints trained individually for each language direction.

Sentence Alignment Baselines. For the sentence-level alignment task, we compare OmniAlign with six existing sentence alignment tools: Gale–Church (Gale & Church, 1993), VecAlign (Thompson & Koehn, 2019), BleuAlign (Sennrich & Volk, 2011) (using translations generated by WPS iCiba<sup>8</sup>), BertAlign (Liu & Zhu, 2023), SentAlign (Steingrímsson et al., 2023), and CrocoAlign (Molfese et al., 2024). All methods are evaluated under their default configurations.

## 3.4 Main Results

Word Alignment Results. Table 4 presents a comparison between OmniAlign and various baseline methods. Using a single shared model checkpoint, OmniAlign achieves highly competitive overall performance across nine language pairs, ranking first on four datasets and second on three others. Under the setting where a single model weight is shared across all test sets, OmniAlign consistently outperforms comparable methods such as SimAlign, Awesome-align, AccAlign, and WSPAlign (Multilingual), while achieving performance comparable to the finely tuned WSPAlign (Bilingual) model. Although BinaryAlign (Bilingual) attains state-of-the-art (SOTA) results on the tasks it participates in under a fully supervised training regime, our empirical analysis indicates that its generalization capability is largely confined to the distribution of the supervised data. When evaluated on out-of-distribution samples, particularly long-form texts, its performance degrades substantially (see Table 8). In addition, BinaryAlign requires training a separate checkpoint for each language direction, which incurs higher costs in terms of training, deployment, and maintenance in practical applications.

Table 4: AER of the test set under different word alignment methods. Lower AER indicates better performance. For each language pair, the best result is bolded and the second-best one is underlined.
<table><tr><td>Methods</td><td>zh-en</td><td>de-en</td><td>fr-en</td><td>ro-en</td><td>ja-en</td><td>es-en</td><td>pt-en</td><td>ru-en</td><td>it-en</td></tr><tr><td>FastAlign (Dyer et al., 2013)</td><td>38.1</td><td>27.0</td><td>10.5</td><td>27.0</td><td>51.1</td><td>一</td><td></td><td></td><td>一</td></tr><tr><td>GIZA++ (Òch &amp; Ney, 2003)</td><td>35.1</td><td>20.6</td><td>5.9</td><td>26.4</td><td>48.0</td><td></td><td></td><td></td><td></td></tr><tr><td>SimAlign (Sabet et al., 2020)</td><td>21.6</td><td>16.6</td><td>7.5</td><td>22.3</td><td>46.6</td><td>14.2</td><td>14.1</td><td>15.4</td><td>17.7</td></tr><tr><td>AwesomeAlign (Dou &amp; Neubig, 2021)</td><td>13.3</td><td>13.3</td><td>3.8</td><td>18.7</td><td>37.4</td><td>12.0</td><td>12.7</td><td>13.5</td><td>15.7</td></tr><tr><td>AccAlign (Wang et al., 2022)</td><td>11.5</td><td>12.1</td><td>2.8</td><td>16.9</td><td>36.8</td><td>11.1</td><td>12.1</td><td>12.5</td><td>14.3</td></tr><tr><td>WSPAlign (Bilingual) (Wu et al., 2023)</td><td>13.1</td><td>11.1</td><td>2.8</td><td>10.1</td><td>19.3</td><td></td><td></td><td></td><td>一</td></tr><tr><td>WSPAlign (Multilingual) (Wu et al., 2023)</td><td>22.3</td><td>20.0</td><td>12.8</td><td>26.4</td><td>45.8</td><td>13.4</td><td>12.3</td><td>13.1</td><td>17.1</td></tr><tr><td>BinaryAlign (Bilingual) (Latouche et al., 2024)</td><td>4.8</td><td>7.8</td><td>1.9</td><td>7.4</td><td>14.3</td><td></td><td></td><td></td><td></td></tr><tr><td>OmniAlign (ours)</td><td>8.5</td><td>11.0</td><td>2.7</td><td>16.7</td><td>29.6</td><td>10.7</td><td>11.9</td><td>12.1</td><td>14.1</td></tr></table>

Sentence Alignment Results. Table 5 presents a comparison between OmniAlign and other sentence alignment methods. For fair comparison, all baseline methods that rely on sentence encoders employ LaBSE as their underlying representation model. Although OmniAlign acquires its sentence-level representation capability through knowledge distillation from LaBSE, the proposed model combined with our alignment algorithm achieves highly competitive overall performance, ranking first on four test sets and second on two others across seven evaluation datasets. Benefiting from the use of a tokenlength–based constraint rather than a character-length–based one during sentence alignment, OmniAlign achieves particularly strong performance on the zh–en test set, while exhibiting relatively weaker results on de–fr. It is worth noting that, for certain language pairs, some baseline methods still demonstrate superior performance. For example, BertAlign, leveraging its advanced alignment algorithm together with the LaBSE encoder, achieves the best scores on the en–it, en–ru, and de–fr language pairs. However, due to its modified cosine similarity function, which tends to favor many-to-many alignments, BertAlign performs comparatively worse on en–es, en–de, and en–fr.

Table 5: F1 scores of OmniAlign and sentence alignment baseline methods across language pairs. For each language pair, the best result is bolded and the second-best one is underlined.
<table><tr><td>Algorithm</td><td>en-zh</td><td>en-es</td><td>en-it</td><td>en-de</td><td>en-fr</td><td>en-ru</td><td>de-fr</td></tr><tr><td>Gale-Church (Gale &amp; Church, 1993)</td><td>0.682</td><td>0.900</td><td>0.977</td><td>0.897</td><td>0.838</td><td>0.911</td><td>0.680</td></tr><tr><td>BleuAlign (Sennrich &amp; Volk, 2011)</td><td>0.782</td><td>0.819</td><td>0.901</td><td>0.806</td><td>0.757</td><td>0.791</td><td>0.770</td></tr><tr><td>VecAlign (Thompson &amp; Koehn, 2019)</td><td>0.957</td><td>0.892</td><td>0.956</td><td>0.869</td><td>0.880</td><td>0.921</td><td>0.902</td></tr><tr><td>BertAlign (Liu &amp; Zhu, 2023)</td><td>0.969</td><td>0.897</td><td>0.984</td><td>0.900</td><td>0.909</td><td>0.938</td><td>0.939</td></tr><tr><td>SentAlign (Steingrímsson et al., 2023)</td><td>0.968</td><td>0.872</td><td>0.978</td><td>0.892</td><td>0.903</td><td>0.920</td><td>0.932</td></tr><tr><td>CrocoAlign (Molfese et al., 2024)</td><td>0.660</td><td>0.696</td><td>0.864</td><td>0.804</td><td>0.788</td><td>0.783</td><td>0.714</td></tr><tr><td>OmniAlign (Ours)</td><td>0.970</td><td>0.906</td><td>0.978</td><td>0.913</td><td>0.912</td><td>0.935</td><td>0.922</td></tr></table>

## 3.5 Ablation Study

In this section, we compare the performance of models at different training stages, the effectiveness of embedding extraction from different layers, and the performance of various embedding models in the sentence alignment task. For the sake of brevity, we abbreviate each training stage as follows:

• Continued Pre-training: S.1

• Self-supervised Learning: S.2

• Supervised Fine-tuning: S.3

• Sentence Embedding Knowledge Distillation: S.4

## 3.5.1 Models from Different Training Stages

Table 6 reports the performance of models trained under different training strategies on the sequence alignment subtasks, evaluated across multiple test sets.

In the word alignment task, model performance improves consistently with the introduction of additional training stages. Specifically, the average alignment error rate (AER) decreases from 43.0% to 19.6%, 14.3%, and finally 8.5%, indicating that each training stage contributes positively to word-level alignment capability. Although removing S.1 and S.2 in ablation studies does not lead to a significant degradation in the final evaluation metric, both stages play an important role in shaping the model’s overall capacity. Through unsupervised or self-supervised training, the model can effectively leverage long-text samples without relying on large-scale human-annotated data, thereby improving its long-sequence modeling ability. In addition, the analysis in Table 10 further shows that S.1 helps enhance the model’s foundational representations, enabling it to better handle word alignment tasks in complex scenarios.

For the sentence alignment task, we fix the dynamic programming (DP) algorithm and fine-tune only the last five layers of the model to align multilingual sentence representations into the English embedding space of LaBSE. Under this setting, the resulting alignment performance is comparable to that of LaBSE, and even surpasses it on the EN–ES test set. This improvement may be attributed to the multilingual joint alignment mechanism, which allows languages with relatively fewer resources to learn semantic representations more closely aligned with English. Furthermore, benefiting from the multilingual joint distillation strategy, OmniAlign outperforms several mainstream multilingual embedding models on sentence alignment tasks, despite operating with a substantially smaller parameter budget and embedding dimensionality. These results highlight the advantages of OmniAlign in terms of both parameter efficiency and computational efficiency.

## 3.5.2 Word Alignment Performance across Different Word Embedding Layers

In this section, we investigate the word alignment performance of mBERT, mGTE-MLM-Base, and its continued pre-trained variant across different Transformer layers, all of which are widely used in word alignment tasks. Experimental results show that, as a base model, mGTE-MLM-Base underperforms mBERT in terms of word alignment accuracy. However, mGTE-MLM-Base offers a key advantage in that its native configuration supports a substantially longer maximum sequence length than mBERT (8192 vs. 512), and this capacity can be further extended through RoPE-based scaling techniques, such as positional frequency scaling or linear interpolation (Chen et al., 2023; Peng et al., 2023).

<table><tr><td>Model</td><td>zh-en</td><td>de-en</td><td>fr-en</td></tr><tr><td colspan="4">Additive Training Stages</td></tr><tr><td>mGTE-MLM-Base</td><td>43.0</td><td>28.3</td><td>14.5</td></tr><tr><td>mGTE-MLM-Base + S.1</td><td>19.6</td><td>16.4</td><td>5.8</td></tr><tr><td> $\mathrm { m G T E \mathrm { - } M L M \mathrm { - } B a s e + S . 1 + S . 2 }$ </td><td>14.3</td><td>12.8</td><td>3.4</td></tr><tr><td> $\mathrm { m G T E \mathrm { - } M L M \mathrm { - } B a s e + S . 1 + S . 2 + S . 3 }$ </td><td>8.5</td><td>11.0</td><td>2.7</td></tr><tr><td colspan="4">Subtractive Training Stages</td></tr><tr><td>OmniAlign w/o S.3</td><td>14.3</td><td>12.8</td><td>3.4</td></tr><tr><td>OmniAlign w/o S.2</td><td>8.6</td><td>11.1</td><td>2.7</td></tr><tr><td>OmniAlign w/o S.1</td><td>8.7</td><td>11.5</td><td>2.9</td></tr><tr><td>OmniAlign</td><td>8.5</td><td>11.0</td><td>2.7</td></tr></table>

(a) Word alignment AER (%). Lower is better. Within each block, the best result is bolded and the second-best one is underlined.
<table><tr><td>Method</td><td>#Params</td><td>Emb. Dim.</td><td>en-zh</td><td>en-es</td><td>en-it</td></tr><tr><td> $\mathrm { D P + L a B S E }$ </td><td>0.47B</td><td>768</td><td>0.972</td><td>0.904</td><td>0.979</td></tr><tr><td> $\mathrm { D P + m u l t i l i n g u a l - e 5 \mathrm { - } l a r g e \mathrm { - } i n s t r u c t E ^ { \it a } }$ </td><td>0.56B</td><td>1024</td><td>0.895</td><td>0.899</td><td>0.972</td></tr><tr><td> ${ \mathrm { D P } } + { \mathrm { b g e } } { \mathrm { - m } } 3 ^ { b }$ </td><td>0.57B</td><td>1024</td><td>0.933</td><td>0.910</td><td>0.978</td></tr><tr><td> $\mathrm { D P } + \mathrm { Q w e n 3 - E m b e d d i n g - } 0 . 6 \mathrm { B } ^ { c }$ </td><td>0.6B</td><td>1024</td><td>0.939</td><td>0.905</td><td>0.971</td></tr><tr><td> $\mathrm { D P } + \mathrm { Q w e n } 3 { \mathrm { - E m b e d d i n g - } } 8 \mathrm { B } ^ { d }$ </td><td>8B</td><td>4096</td><td>0.935</td><td>0.908</td><td>0.975</td></tr><tr><td> $\mathrm { D P + g t e - m u l t i l i n g u a l - b a s e } ^ { e }$ </td><td>0.31B</td><td>768</td><td>0.927</td><td>0.908</td><td>0.973</td></tr><tr><td> $\mathrm { D P + \breve { O } m n i A l i g n \left( w / o S . 4 \right) }$ </td><td>0.31B</td><td>768</td><td>0.716</td><td>0.872</td><td>0.966</td></tr><tr><td> $\mathrm { D P } + \mathrm { O m n i A l i g n } \ ( \mathrm { d i s t i l l e d \ f r o m \ L a B S E } )$ </td><td>0.31B</td><td>768</td><td>0.970</td><td>0.906</td><td>0.978</td></tr></table>

(b) Sentence alignment F1 scores. Higher is better. For each language pair, the best result is bolded and the secondbest one is underlined.  
<sup>a</sup>https://huggingface.co/intfloat/multilingual-e5-large-instruct  
<sup>b</sup>https://huggingface.co/BAAI/bge-m3  
<sup>c</sup>https://huggingface.co/Qwen/Qwen3-Embedding-0.6B  
<sup>d</sup>https://huggingface.co/Qwen/Qwen3-Embedding-8B  
<sup>e</sup>https://huggingface.co/Alibaba-NLP/gte-multilingual-base  
Table 6: Ablation study on word and sentence alignment performance under different training strategies and embedding models.

Motivated by this property, we perform task-oriented continued pre-training on mGTE-MLM-Base specifically for word alignment, aiming to compensate for its initial shortcomings in word-level alignment performance while preserving its long-context modeling capability.
<table><tr><td>Model</td><td>Layer</td><td>zh-en</td><td>de-en</td><td>fr-en</td></tr><tr><td></td><td>6</td><td>20.5</td><td>19.4</td><td>6.5</td></tr><tr><td>mBERT</td><td>7</td><td>19.0</td><td>16.6</td><td>5.6</td></tr><tr><td></td><td>8</td><td>18.1</td><td>15.2</td><td>5.4</td></tr><tr><td></td><td>6</td><td>54.8</td><td>33.7</td><td>18.0</td></tr><tr><td>mGTE-MLM-Base</td><td>7</td><td>43.0</td><td>28.3</td><td>14.5</td></tr><tr><td></td><td>8</td><td>47.5</td><td>30.1</td><td>16.0</td></tr><tr><td></td><td>6</td><td>28.4</td><td>21.2</td><td>6.9</td></tr><tr><td>mGTE-MLM-Base + S.1</td><td>7</td><td>19.6</td><td>16.4</td><td>5.8</td></tr><tr><td></td><td>8</td><td>22.8</td><td>19.1</td><td>6.4</td></tr></table>

Table 7: Word alignment AER (%) at different layers for various base models. Within each model group, the best result is bolded.

## 3.5.3 Exploring Long Text Word Alignment Capability

As shown in Table 3, existing evaluations for word alignment tasks are predominantly conducted in shorttext settings. To systematically assess model performance under long-text word alignment conditions, we extend the original test sets to long-context scenarios using a set of heuristic rules. Specifically, zh–en–n denotes the construction of longer input sequences by concatenating n samples from the original zh–en test set.

The experimental results in Table 8 show that, as the input text length increases, the performance of previous word alignment methods degrades substantially. Moreover, these methods are typically constrained by a maximum input length of 512 tokens, which prevents them from effectively supporting word alignment in longer sequence settings. In contrast, OmniAlign degrades only mildly, with AER rising from 8.5 on zh–en–1 to 12.6 on zh–en–50. This robustness comes from a long-context backbone that can encode up to 8,192 tokens in one pass, and from early-stage training on mixed-length—especially long—texts. Supervised fine-tuning on short annotated data further improves alignment quality, and these gains also hold under long-text evaluation.

<table><tr><td>Model</td><td>zh-en-1</td><td>zh-en-3</td><td>zh-en-5</td><td>zh-en-10</td><td>zh-en-20</td><td>zh-en-50</td></tr><tr><td>Awesome-Align</td><td>13.3</td><td>13.9</td><td>15.1</td><td>24.8</td><td>一</td><td></td></tr><tr><td>ACC-Align</td><td>11.5</td><td>12.5</td><td>15.8</td><td>28.1</td><td>一</td><td></td></tr><tr><td>BinaryAlign (Bilingual)</td><td>4.8</td><td>9.6</td><td>22.8</td><td></td><td></td><td></td></tr><tr><td>OmniAlign (Ours)</td><td>8.5</td><td>8.6</td><td>8.6</td><td>9.2</td><td>10.5</td><td>12.6</td></tr></table>

Table 8: Word alignment performance under increasing input lengths. The zh-en-n setting denotes longtext inputs constructed by concatenating n samples from the original zh-en test set. Notably, zh–en–50 reaches 1,850 tokens.

## 3.5.4 Sequence alignment on unseen language pairs

To evaluate the zero-shot generalization capability of OmniAlign, we directly apply all methods to previously unseen language pairs, without performing any additional training on the target test languages. For the word alignment task, we conduct experiments on the bg–en, da–en, et–en, hu–en, nl–en, and sl–en language pairs, using test datasets sourced from (Martelli et al., 2023). For the sentence alignment task, we evaluate on the en–hu and en–nl test sets, which are obtained from Molfese et al. (2024).

Table 9 reports the alignment performance of different methods on these unseen language pairs. OmniAlign matches or outperforms the baselines on most pairs. We attribute this to training a single shared multilingual encoder rather than language-pair-specific aligners: continued pre-training and self-supervised learning on large-scale multilingual parallel data induce token-level correspondences that are not tied to a particular language pair, while supervised fine-tuning on seen pairs further sharpens these signals and transfers them to unseen directions.

(a) Word alignment AER
<table><tr><td>Test Set</td><td>AccAlign</td><td>BinaryAlign</td><td>OmniAlign</td><td colspan="3">(b) Sentence alignment F1 score</td></tr><tr><td>bg-en</td><td>11.0</td><td>11.99</td><td>10.6</td><td></td><td></td><td></td></tr><tr><td>da-en</td><td>7.1</td><td>8.45</td><td>7.1</td><td>Test Set</td><td>BertAlign</td><td>Ours</td></tr><tr><td>et-en</td><td>15.2</td><td>15.67</td><td>14.4</td><td>en-hu</td><td>0.977</td><td>0.979</td></tr><tr><td>hu-en</td><td>18.7</td><td>16.93</td><td>19.1</td><td>en-nl</td><td>0.928</td><td>0.933</td></tr><tr><td>nl-en</td><td>4.6</td><td>5.14</td><td>4.5</td><td></td><td></td><td></td></tr><tr><td>sl-en</td><td>14.9</td><td>16.85</td><td>14.7</td><td></td><td></td><td></td></tr></table>

Table 9: Alignment performance of OmniAlign and baselines on unseen language pairs

## 3.6 Case Study

To complement the quantitative benchmarks, we present qualitative case studies in Table 10, covering both word-level and sentence-level alignment under challenging conditions. Scenario 1 focuses on word alignment. In Example #1, the key difficulty is mapping an English noun such as “failure” to a Chinese negated expression (“<sup>没有成功</sup>”). Models without continued pre-training (OmniAlign w/o S.1) often align only the positive lexical item and miss the negation scope. Example #2 examines function-word and numeral expressions in a medical instruction sentence (e.g., “<sup>有</sup>” and $\smile \Sigma ^ { \prime \prime } )$ , where partial matches can look locally plausible but remain incomplete. Example #3 involves informal user-generated text with morphological variation and mild noise (e.g., “returned” and the misspelling “windering”), which stresses robustness beyond clean parallel news text. Example #4 concatenates a long technical paragraph with many multi-token named entities and domain phrases; under this setting, BinaryAlign fails to produce usable alignment outputs, whereas OmniAlign still recovers the highlighted correspondences.

Scenario 2 turns to document-level sentence alignment, where the central issue is choosing an appropriate m–n segmentation rather than maximizing span coverage. Across Examples #1–#4, BertAlign frequently merges neighboring sentences into broader many-to-many blocks, even when finer one-to-one or constrained m–n links are preferable. OmniAlign, guided by anchor-constrained dynamic programming with a token-length–based penalty, tends to produce more compact and interpretable alignment units on these cases.

Table 10: Case studies of OmniAlign and other sequence alignment methods. Blue indicates correct alignments, red denotes incorrect alignments, and black marks segments with no alignment.
<table><tr><td colspan="2">Scenario 1: Word Alignment</td></tr><tr><td>Example #1</td><td>He had to repeatedly traverse the inner palace&#x27;s chambers, yet each attempt ended in failure.</td></tr><tr><td>Testing Points</td><td>&quot;failure&quot;.</td></tr><tr><td>OmniAlign</td><td>他得不断地一再地穿过内宫里的屋子：可是他一直没有成功。</td></tr><tr><td>OmniAlign w/o S.1</td><td>他得不断地一再地穿过内宫里的屋子；可是他一直没有成功。</td></tr><tr><td>Example #2</td><td>已知有血液疾病及尿酸性肾结石的患者不推荐使用本品，二岁以下儿童不得服用。</td></tr><tr><td>Testing Points OmniAlign</td><td>有;二岁 This product is not recommended for patients with known blood disorders or uric acid kidney stones, and it should</td></tr><tr><td>BinaryAlign</td><td>not be taken by children under the age of two.</td></tr><tr><td>Example #3</td><td>This product is not recommended for patients with known blood disorders or uric acid kidney stones, and it should not be taken by children under the age of two.</td></tr><tr><td></td><td>I recently returned to d2 after several year, now I&#x27;m windering: Where do you guys sell/buy your stuff? Do you just make a game “O xxx N yyy&quot; and hope for the best? Or are there a website that&#x27;s more efficient?</td></tr><tr><td>Testing Points OmniAlign</td><td>returned; windering 时隔多年，我又重新开始玩《暗黑破坏神2》(DiabloI)，现在我想知道：大家都是在哪里进行物品交易的？是通</td></tr><tr><td></td><td>过自己创建名为“OxxxNyyy&quot;的游戏房间来交易，然后听天由命吗?还是有更高效的交易网站?</td></tr><tr><td>BinaryAlign Example #4</td><td>时隔多年，我又重新开始玩《暗黑破坏神2》(DiabloⅡ)，现在我想知道：大家都是在哪里进行物品交易的？是通 过自己创建名为“OxxxNyyy&quot;的游戏房间来交易，然后听天由命吗？还是有更高效的交易网站? 在技术演进的脉络中，预训练模型的出现无疑是里程碑式的突破。以BERT、GPT、T5等为代表的模型通过在海</td></tr><tr><td></td><td>这些模型基于Transformer架构的多头注意力机制，能够捕捉文本中的长距离依赖关系和上下文关联，为文本分 类、情感分析、机器翻译、问答系统等经典任务提供了强大的通用能力。近年来，多语言预训练模型（如XLM R、mT5、LaBSE)进一步打破了语言壁垒，通过对数百种语言的联合训练，实现了跨语言文本的理解与生成，为 全球化信息传播和跨文化交流提供了技术支撑。在产业应用层面，NLP技术已渗透到金融、医疗、教育、传媒等 多个领域，催生了一系列创新产品和服务。在金融行业，智能客服系统能够基于用户的自然语言查询快速提供账户 咨询、业务办理等服务，平均响应时间缩短至秒级，客户满意度提升30%以上；风险控制模型通过分析企业年报、 新闻舆情等文本数据，精准识别信用风险和市场波动信号，帮助金融机构降低不良贷款率。在医疗领域，医学文本 分析系统可自动提取电子病历中的关键信息(如病症、用药史、检查结果)，辅助医生进行诊断决策，减少误诊 率；多语言医疗翻译工具则为跨境医疗合作提供了语言保障，使不同国家的医护人员能够高效协作。在教育领域， 智能写作辅助系统能够实时检测文本中的语法错误、逻辑漏洞，并提供优化建议，帮助学生提升写作能力；个性化 学习平台通过分析学生的学习行为和文本交互数据，精准推送适配的学习资源，实现“因材施教”的教育理念。 (Bert、GPT、T5);(多头注意力机制);(文本分类、情感分析、机器翻译、问答系统);(XLM-R、mT5、LaBSE);(创新产</td></tr><tr><td>OmniAlign</td><td>品和服务);(智能客服系统);(风险控制模型);(多语言医疗翻译工具);(实时检测);(学习行为和文本交互数据);(教育理 念) In the context of technological evolution, the emergence of pre-trained models is undoubtedly a milestone break- through. Represented by BERT, GPT, T5, and other models, they acquire deep semantic representations and syntactic</td></tr><tr><td></td><td>structures of language through self-supervised learning on massive unlabeled text, greatly raising the performance ceiling of downstream tasks. Based on the multi-head attention mechanism of the Transformer architecture, these modeǐs can capture long-distance dependencies and contextual correlations in text, providing powerful general capa- bilities for classic tasks such as text classification, sentiment analysis, machine translation, and question-answering systems. In recent years, multilingual pre-trained models (e.g., XLM-R, mT5, LaBSE) have further broken down language barriers. Through joint training on hundreds of languages, they have realized the understanding and generation of cross-lingual text, providing technical support for global information dissemination and cross-cultural communication. At the industrial application level, NLP technology has penetrated into multiple fields such as finance, healthcare, education, and media, spawning a series of innovative products and services. In the financial industry, intelligent customer service systems can quickly provide account consultations, business processing, and other services based on users&#x27; natural language queries, reducing the average response time to seconds and increasing customer satisfaction by more than 30%; risk control models accurately identify credit risks and market fluctuation signals by analyzing text data such as corporate annual reports and news public opinion, helping financial institutions reduce the non-performing loan ratio. In the healthcare field, medical text analysis systems can automatically extract key information from electronic medical records (such as symptoms, medication history, and examination results) to assist doctors in diagnostic decisions and reduce the misdiagnosis rate; multilingual medical translation tools provide language guarantees for cross-border medical cooperation, enabling medical staff from different countries to collaborate efficiently. In the education field, intelligent writing assistance systems can real-time detect grammatical errors and logical flaws in text, and provide optimization suggestions to help students improve their writing skills; personalized learning platforms accurately push adaptive learning resources by analyzing students&#x27; learning be-</td></tr></table>

BinaryAlign In the context of technological evolution, the emergence of pre-trained models is undoubtedly a milestone breakthrough. Represented by BERT, GPT, T5, and other models, they acquire deep semantic representations and syntactic structures of language through self-supervised learning on massive unlabeled text, greatly raising the performance ceiling of downstream tasks. Based on the multi-head attention mechanism of the Transformer architecture, these models can capture long-distance dependencies and contextual correlations in text, providing powerful general capa bilities for classic tasks such as text classification, sentiment analysis, machine translation, and question-answering systems. In recent years, multilingual pre-trained models (e.g., XLM-R, mT5, LaBSE) have further broken down language barriers. Through joint training on hundreds of languages, they have realized the understanding and generation of cross-lingual text, providing technical support for global information dissemination and cross-cultural communication. At the industrial application level, NLP technology has penetrated into multiple fields such as finance, healthcare, education, and media, spawning a series of innovative products and services. In the financial industry, intelligent customer service systems can quickly provide account consultations, business processing, and other services based on users’ natural language queries, reducing the average response time to seconds and increasing customer satisfaction by more than 30%; risk control models accurately identify credit risks and market fluctuation signals by analyzing text data such as corporate annual reports and news public opinion, helping financial institutions reduce the non-performing loan ratio. In the healthcare field, medical text analysis systems can automatically extract key information from electronic medical records (such as symptoms, medication history, and examination results) to assist doctors in diagnostic decisions and reduce the misdiagnosis rate; multilingual medical translation tools provide language guarantees for cross-border medical cooperation, enabling medical staff from different countries to collaborate efficiently. In the education field, intelligent writing assistance systems can real-time detect grammat ical errors and logical flaws in text, and provide optimization suggestions to help students improve their writing skills; personalized learning platforms accurately push adaptive learning resources by analyzing students’ learning behaviors and text interaction data, realizing the educational concept of "teaching students in accordance with their aptitude".

<table><tr><td colspan="2">Scenario 2: Sentence Alignment</td></tr><tr><td>Example #1</td><td>Source (1) Their Australian born captain, the world&#x27;s top-ranked match racer Peter Gilmour, lived in Japan for three years</td></tr><tr><td></td><td>to satisfy the Cup&#x27;s crew-nationality rules. (2) Syndicate head Tatsumitsu Yamasaki, chairman of spice giant S&amp;B Foods, is hungry for a win.</td></tr><tr><td>Target</td><td>(1) 该船队的船长为世界顶尖帆船赛选手皮得·吉尔摩。</td></tr><tr><td></td><td>(2)为了达到杯赛在船员国籍方面的各项要求，这位出生在澳大利亚的选手在日本居住了3年。 (3)山崎达光是一家财团的总裁兼调味品行业的巨人S&amp;B食品株式会社董事会主席。 (4)他渴望日本船队能够获得胜利。</td></tr><tr><td>OmniAlign</td><td>[(1)]:[(1), (2)]; [(2)]:[(3), (4)]</td></tr><tr><td>BertAlign</td><td>[(1)]:[(1), (2)]; [(2)]:[(3)]; []:[(4)] Source</td></tr><tr><td>Example #2</td><td>(1) Then came the hard part: getting people to want to see the thing. (2) Here&#x27;s how they did it, from whisper to buzz to big box-office noise, in only 21 steps.</td></tr><tr><td></td><td>Target (1)随后进入最艰苦的宣传炒作阶段。</td></tr><tr><td></td><td>(2)下面就是他们的宣传炒作步骤:刚开始知者甚少，随后观众渐增，最后票房炙手可热。 (3)这一成绩的取得仅需21步：。</td></tr><tr><td>OmniAlign</td><td>[(1)]:[(1), (2)]; [(2)]:[(2), (3)]</td></tr><tr><td>BertAlign</td><td>[(1), (2)]:[(1), (2), (3)]</td></tr><tr><td>Example #3</td><td>Source (1) The situation in Japan has to change. &quot;</td></tr><tr><td></td><td>(2) At Nissan, it already has.</td></tr><tr><td></td><td>Target (1) 这种形势必须改变。</td></tr><tr><td></td><td>(2)在日产，这种变革已经开始。</td></tr><tr><td>OmniAlign</td><td>[(1)]:[(1)];[(2)]:[(2)]</td></tr><tr><td>BertAlign</td><td>[(1), (2)]:[(1), (2)]</td></tr><tr><td>Example #4</td><td>Source</td></tr><tr><td></td><td>(1)就在这种情况下，我想起十五队的队医陈清扬是北医大毕业的大夫，对针头和勾针大概还能分清，所以我去找 她看病。</td></tr><tr><td></td><td>(2)看完病回来，不到半个小时，她就追到我屋里来，要我证明她不是破鞋。</td></tr><tr><td></td><td>Target</td></tr><tr><td></td><td>(1) Under the circumstances, I recalled that the doctor at the fifteenth team, Chen Qingyang, had graduated from Beijing Medical School.</td></tr><tr><td></td><td>(2) Maybe she would be able to tell the difference between a hypodermic and a crotchet needle. (3) So I went to see her.</td></tr><tr><td></td><td>(4) Not half an hour after my visit, she chased after me to my room, wanting me to prove that she wasn&#x27;t damaged</td></tr><tr><td></td><td>goods.</td></tr><tr><td></td><td></td></tr><tr><td>OmniAlign BertAlign</td><td>[(1)]:[(1), (2), (3)]; [(2)]:[(4)]</td></tr></table>

## 4 Related work

Cross-lingual sequence alignment provides essential signals for parallel corpus construction and downstream multilingual applications (Varga et al., 2008; David et al., 2001; Hashimoto et al., 2019; Tu et al., 2016; Nicolai & Yarowsky, 2019; Mayhew et al., 2017; Chi et al., 2021), but prior work is often specialized to a single granularity. For word alignment, classical statistical models such as GIZA++ (Och & Ney, 2003) and FastAlign (Dyer et al., 2013) remain strong baselines. Recent approaches leverage multilingual contextual representations to induce alignments, including SimAlign (Sabet et al., 2020), Awesome-align (Dou & Neubig, 2021), AccAlign (Wang et al., 2022), WSPAlign (Wu et al., 2023), and BinaryAlign (Latouche et al., 2024). While these methods improve accuracy by exploiting pre-trained encoders and supervised signals, many are evaluated under short-context settings and can be limited by encoder input length.

For sentence/document alignment, early length-based algorithms such as Gale–Church (Gale & Church, 1993) align using dynamic programming, while translation-assisted methods like BleuAlign (Sennrich & Volk, 2011) rely on external MT systems. Embedding-based approaches (e.g., VecAlign (Thompson & Koehn, 2019), BertAlign (Liu & Zhu, 2023), SentAlign (Steingrímsson et al., 2023), CrocoAlign (Molfese et al., 2024)) compute cross-lingual sentence representations—often using models such as LaBSE (Feng et al., 2022) or LASER (Artetxe & Schwenk, 2019)—and combine similarity retrieval with sequence constraints. However, these methods typically focus on sentence-level alignment and do not provide fine-grained word alignments. Overall, existing tools seldom offer a unified solution that simultaneously supports multilingual, long-context, and multi-granularity alignment.

## 5 Conclusion

In this report, we present OmniAlign, a unified multilingual aligner that supports both word-level and sentence-level alignment within a single lightweight model. OmniAlign is built upon an encoder-only architecture with strong long-context modeling capability, where token- and word-level alignments are induced from contextualized representations. For document-level sentence alignment, OmniAlign further integrates a two-stage, anchor-constrained dynamic programming algorithm.

To balance fine-grained alignment accuracy and high-quality sentence representations, we design a four-stage training pipeline consisting of continued pre-training, self-supervised learning, supervised fine-tuning with human-annotated data, and sentence embedding knowledge distillation from a strong multilingual teacher model. Extensive experiments demonstrate that OmniAlign achieves highly competitive performance at both the word and sentence alignment levels. Notably, in long-text word alignment scenarios, OmniAlign maintains stable performance, whereas many existing methods suffer from severe degradation due to context-length limitations. In addition, OmniAlign exhibits strong zero-shot generalization on unseen language pairs, delivering reliable alignment results without any additional training.

## 6 Acknowledgment

We thank the Engineering and Infrastructure teams at WPS-Qingqiu, as well as the WPS-iCIBA team, for their support. In particular, we appreciate their assistance with data collection, pre-processing, and platform support.

## References

Mikel Artetxe and Holger Schwenk. Massively multilingual sentence embeddings for zero-shot crosslingual transfer and beyond. Transactions of the associationfor computational linguistics, 7:597–610, 2019.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595, 2023.

Zewen Chi, Li Dong, Bo Zheng, Shaohan Huang, Xian-Ling Mao, He-Yan Huang, and Furu Wei. Improving pretrained cross-lingual language models via self-labeled word alignment. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pp. 3418–3430, 2021.

Alexis Conneau and Guillaume Lample. Cross-lingual language model pretraining. Advances in neural information processing systems, 32, 2019.

Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. Unsupervised crosslingual representation learning at scale. In Proceedings of the 58th annual meeting of the associationfor computational linguistics, pp. 8440–8451, 2020.

Yarowsky David, Ngai Grace, Wicentowski Richard, et al. Inducing multilingual text analysis tools via robust projection across aligned corpora. In Proceedings of the First International Conference on Human Language Technology Research, pp. 1–8, 2001.

Zi-Yi Dou and Graham Neubig. Word alignment by fine-tuning embeddings on parallel corpora. arXiv preprint arXiv:2101.08231, 2021.

Chris Dyer, Victor Chahuneau, and Noah A Smith. A simple, fast, and effective reparameterization of ibm model 2. In Proceedings of the 2013 conference of the North American chapter of the associationfor computational linguistics: human language technologies, pp. 644–648, 2013.

Fangxiaoyu Feng, Yinfei Yang, Daniel Cer, Naveen Arivazhagan, and Wei Wang. Language-agnostic bert sentence embedding. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 878–891, 2022.

William A Gale and Kenneth Church. A program for aligning sentences in bilingual corpora. Computational linguistics, 19(1):75–102, 1993.

Suchin Gururangan, Ana Marasovi´c, Swabha Swayamdipta, Kyle Lo, Iz Beltagy, Doug Downey, and Noah A Smith. Don’t stop pretraining: Adapt language models to domains and tasks. arXiv preprint arXiv:2004.10964, 2020.

Kazuma Hashimoto, Raffaella Buschiazzo, James Bradbury, Teresa Marshall, Richard Socher, and Caiming Xiong. A high-quality multilingual dataset for structured documentation translation. In Proceedings of the Fourth Conference on Machine Translation (Volume 1: Research Papers), pp. 116–127, 2019.

Siyu Lai, Zhen Yang, Fandong Meng, Yufeng Chen, Jinan Xu, and Jie Zhou. Cross-align: Modeling deep cross-lingual interactions for word alignment. arXiv preprint arXiv:2210.04141, 2022.

Gaetan Latouche, Marc-André Carbonneau, and Ben Swanson. Binaryalign: Word alignment as binary sequence labeling. In Proceedings ofthe 62nd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 10277–10288, 2024.

Lei Liu and Min Zhu. Bertalign: Improved word embedding-based sentence alignment for chinese– english parallel corpora of literary texts. Digital Scholarship in the Humanities, 38(2):621–634, 2023.

Federico Martelli, Andrei Stefan Bejgu, Cesare Campagnano, Jaka Cibej, Rute Costa, Apolonija Gantar,<sup>ˇ</sup> Jelena Kallas, Svetla Peneva Koeva, Kristina Koppel, Simon Krek, et al. Xl-wa: a gold evaluation benchmark for word alignment in 14 language pairs. In Proceedings of the 9th Italian Conference on Computational Linguistics (CLiC-it 2023), pp. 272–280, 2023.

Stephen Mayhew, Chen-Tse Tsai, and Dan Roth. Cheap translation for cross-lingual named entity recognition. In Proceedings of the 2017 conference on empirical methods in natural language processing, pp. 2536–2545, 2017.

Rada Mihalcea and Ted Pedersen. An evaluation exercise for word alignment. In Proceedings of the HLT-NAACL 2003 Workshop on Building and using parallel texts: data driven machine translation and beyond, pp. 1–10, 2003.

Francesco Maria Molfese, Andrei Stefan Bejgu, Simone Tedeschi, Simone Conia, and Roberto Navigli. Crocoalign: A cross-lingual, context-aware and fully-neural sentence alignment system for long texts. In Proceedings of the 18th Conference of the European Chapter of the Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 2209–2220, 2024.

Masaaki Nagata, Katsuki Chousa, and Masaaki Nishino. A supervised word alignment method based on cross-language span prediction using multilingual bert. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 555–565, 2020.

Garrett Nicolai and David Yarowsky. Learning morphosyntactic analyzers from the bible via iterative annotation projection across 26 languages. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pp. 1765–1774, 2019.

Franz Josef Och and Hermann Ney. A systematic comparison of various statistical alignment models. Computational linguistics, 29(1):19–51, 2003.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. arXiv preprint arXiv:2309.00071, 2023.

Telmo Pires, Eva Schlinger, and Dan Garrette. How multilingual is multilingual bert? arXiv preprint arXiv:1906.01502, 2019.

Nils Reimers and Iryna Gurevych. Making monolingual sentence embeddings multilingual using knowledge distillation. arXiv preprint arXiv:2004.09813, 2020.

Masoud Jalili Sabet, Philipp Dufter, François Yvon, and Hinrich Schütze. Simalign: High quality word alignments without parallel training data using static and contextualized embeddings. arXiv preprint arXiv:2004.08728, 2020.

Rico Sennrich and Martin Volk. Iterative, mt-based sentence alignment of parallel texts. In Proceedings of the 18th Nordic conference of computational linguistics (NODALIDA 2011), pp. 175–182, 2011.

Steinþór Steingrímsson, Hrafn Loftsson, and Andy Way. Sentalign: Accurate and scalable sentence alignment. arXiv preprint arXiv:2311.08982, 2023.

Brian Thompson and Philipp Koehn. Vecalign: Improved sentence alignment in linear time and space. In Proceedings ofthe 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 1342–1348, 2019.

Zhaopeng Tu, Zhengdong Lu, Yang Liu, Xiaohua Liu, and Hang Li. Modeling coverage for neural machine translation. arXiv preprint arXiv:1601.04811, 2016.

Dániel Varga, Péter Halácsy, András Kornai, Viktor Nagy, László Németh, and Viktor Trón. Parallel corpora for medium density languages. In Recent advances in natural language processing IV: selected papersfrom RANLP 2005, pp. 247–258. John Benjamins Publishing Company, 2008.

David Vilar, Maja Popovi´c, and Hermann Ney. Aer: Do we need to “improve” our alignments? In Proceedings of the Third International Workshop on Spoken Language Translation: Papers, 2006.

Martin Volk, Noah Bubenhofer, Adrian Althaus, Maya Bangerter, Lenz Furrer, and Beni Ruef. Challenges in building a multilingual alpine heritage corpus. 2010.

Weikang Wang, Guanhua Chen, Hanqing Wang, Yue Han, and Yun Chen. Multilingual sentence transformer as a multilingual word aligner. In Findings of the Associationfor Computational Linguistics: EMNLP 2022, pp. 2952–2963, 2022.

Qiyu Wu, Masaaki Nagata, and Yoshimasa Tsuruoka. Wspalign: Word alignment pre-training via large-scale weakly supervised span prediction. arXiv preprint arXiv:2306.05644, 2023.

Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, et al. mgte: Generalized long-context text representation and reranking models for multilingual text retrieval. arXiv preprint arXiv:2407.19669, 2024.