# LabelFusion-TS: Fusing Large Language Models, Transformer Encoders, and Financial Time Series for Monetary-Policy Stance Classification

Michael Schlee<sup>1</sup>, Fabian Lukassen<sup>1</sup> Christoph Weisser<sup>2</sup>

<sup>1</sup>Centre for Statistics, Georg-August-Universität Göttingen, Germany

<sup>2</sup>Hochschule Bielefeld (HSBI) - University of Applied Sciences and Arts, Bielefeld, Germany

michael.schlee@uni-goettingen.de, fabian.lukassen@uni-goettingen.de, christoph.weisser@hsbi.de

## Abstract

Financial text is produced and interpreted within a market environment, yet financial text classifiers almost always receive text alone. We study whether financial time series are useful as an additional input on the task of classifying sentences from Federal Reserve communication as hawkish, dovish, or neutral. Our system, LabelFusion-TS, extends the LabelFusion architecture with this modality: a small voting network combines three independently trained components, a fine-tuned RoBERTa encoder, a prompted large language model (LLM), and a fused ensemble of time-series transformers over the market series of the months preceding publication. Because only about a thousand annotated sentences are available for training, the RoBERTa encoder is first pre-trained on sentences annotated automatically by the LLM and only then fine-tuned on the human labels. Trained on Federal Open Market Committee (FOMC) communication up to 2015 and evaluated on 2015–2022, the fused system achieves 70.2% weighted F1 – against 64.1% for the zero-shot LLM – and overtakes it with as few as 240 human-labelled sentences. We take this as initial evidence for market time series as an input modality in financial text classification.

## 1 Introduction

Financial language is produced and interpreted within a quantitative environment. A reader who assesses a central-bank statement, an earnings release, or a news report does so with the recent behaviour of interest rates, inflation, and asset prices in view, and identical wording can support different readings depending on that background. Market data are also unusually convenient as a model input: they are public, machine-readable, available at daily or finer resolution over several decades, and aligned in time with any dated document. Classification models in financial natural language processing, nevertheless, operate almost exclusively on text. Where the two data types are combined, the direction of use is typically the reverse of the one studied here, with textual features added to numerical models to improve forecasts of prices or volatility (Sun et al., 2026; Cao et al., 2024); market series as input to text classifiers remain little explored.

![](images/c6f89ee6fae5b0966c8e27e604b5044ce9cc7d037a207ff49220438032940ed9.jpg)  
Figure 1: The same sentence under two market regimes. Whether “the Committee will act as appropriate” announces further tightening or continued easing is not decided by the words but by the environment they are spoken into; the market series preceding the document carry exactly this information.

A step towards using both sources is the Label-Fusion architecture of Schlee et al. (2026), which fuses a prompted large language model with a finetuned transformer encoder for financial news classification. The architecture was designed to accommodate additional input sources, and the paper closes by proposing price time series as the next modality. The present work carries out this proposal.

We evaluate it on monetary-policy stance classification (Shah et al., 2023), where sentences from Federal Reserve communication are labelled hawkish, dovish, or neutral. The task suits a first test of the modality: central-bank language is deliberately hedged, so that a sentence such as “the Committee will act as appropriate” is read differently in a tightening than in an easing cycle (Figure 1), and every sentence originates from a dated public document, so the market data of the months before it can be attached without further assumptions. Our fused system exceeds a zero-shot large language model when trained on past communication and tested on future statements.

Our contributions are:

1. LabelFusion-TS, a recipe for extending intermediate fusion in financial text classification with further input modalities, instantiated here for financial time series.

2. Initial empirical evidence that financial time series can provide context for the classification of financial text across varying data availability regimes.

## 2 Related Work

LabelFusion. Given a text $x ,$ LabelFusion (Schlee et al., 2026) obtains label scores $z =$ $f _ { \mathrm { L L M } } ( x )$ from a prompted LLM and a contextual embedding $h \ = \ f _ { \mathrm { R B } } ( x )$ from a fine-tuned RoBERTa encoder and fuses them with a small MLP, $\begin{array} { r } { \hat { y } = f _ { \mathrm { M L P } } ( [ h ; z ] ) } \end{array}$ . Since views are combined as representations rather than as raw features or as separate decisions, this is a hybrid fusion in the taxonomy of Baltrušaitis et al. (2017). On Reuters-21578, the fusion beats the standalone LLM and RoBERTa classifiers in the full-data regime, while the prompted LLM dominates in the low-data regime. We keep this design unchanged and add one expert.

Monetary-policy stance classification. Shah et al. (2023) built the benchmark we use: sentences from FOMC meeting minutes, press-conference transcripts, and speeches (1996–2022), labelled hawkish, dovish, or neutral by trained annotators, with baselines up to fine-tuned RoBERTa-large and zero-shot ChatGPT. To our knowledge, no published work uses market data as an input to stance classification; the connection between central-bank communication and markets is otherwise well documented (Tetlock, 2007; Barber and Odean, 2008).

Time series, text, and distillation. Purpose-built series encoders such as PatchTST (Nie et al., 2023) treat a time series as a sequence of patches, which suits short fixed windows; the multimodal literature predominantly uses text to help forecast series, whereas we use series to help classify text. Our silver-label stage follows knowledge distillation (Hinton et al., 2015) in its self-training form (?): the LLM acts as a free annotator of in-domain unlabelled text, producing silver pseudo-labels for the student; unlike Xie et al., where injected noise lets the student surpass the teacher, in our setting a subsequent gold fine-tuning stage plays this role."

## 3 Data and Task

Benchmark. The benchmark of Shah et al. (2023) covers three forms of Federal Reserve communication published between 1996 and 2022: the minutes of FOMC meetings, the transcripts of postmeeting press conferences, and speeches by members of the Board. Trained annotators assigned every sentence the policy direction it signals, hawkish for a tightening of monetary policy, dovish for an easing, and neutral otherwise. The combined release contains 2,379 annotated sentences, of which 2,312 remain after duplicates and the few sentences with conflicting annotations are removed. Neutral is the largest class (48%), followed by dovish (27%) and hawkish (25%). The task is a singlelabel three-class classification of one sentence at a time; Appendix A shows examples.

Metric. Following the benchmark, we report weighted F1, the class-wise F1 scores averaged with weights proportional to how often each class occurs in the test set, and

$$
{ \mathrm { F } } 1 _ { \mathrm { w } } \ = \ \sum _ { c } \frac { n _ { c } } { N } \cdot \frac { 2 P _ { c } R _ { c } } { P _ { c } + R _ { c } } ,\tag{1}
$$

where $P _ { c }$ and $R _ { c }$ are the precision and recall of class $c , n _ { c }$ is the number of test sentences carrying that class and N the size of the test set. The measure therefore rewards precision and recall on every class, but the frequent neutral class dominates the average, unlike the macro average, which weights all three classes equally.

Sentence dates. The published sentences state only the year of the document they come from, while both parts of our setup require the exact day of publication: the evaluation splits the data chronologically, and the system pairs every sentence with market data from the period preceding its publication (§4). The documents themselves are published with their dates in the benchmark’s repository, which allows the missing date to be recovered by locating the document from which a sentence originates: first by exact match against the sentence lists of the individual documents and for the remaining cases by searching for the normalised sentence in the raw document text. Sentences found in documents of more than one date, mostly formulaic passages that recur across meetings, and sentences found in no document are excluded. The procedure dates 1,692 sentences, or 73% of the benchmark, with a class distribution close to that of the full release; Appendix B gives the details.

Evaluation protocol. We split the dated sentences by time: the 1,274 sentences up to September 2015 are used for training, with the most recent 15% held out for model selection, and the 418 sentences from 2015–2022 form the test set. September 2015 serves as the cut-off throughout the paper. The chronological split mirrors deployment, where a model trained on past communication reads future statements, and prevents the leak a random split would allow: sentences from the same period appearing on both sides and revealing the prevailing conditions. A further 96 benchmark sentences occur only in documents published in 2014 or earlier; their exact day is unknown, but they precede the cut-off with certainty, and we use them as additional training sentences for the sentence encoder (§4), the one component that operates on text alone and needs no date.

Market context. For each sentence, we build a window of six daily series over the 126 business days ending the day before its document date: the federal funds rate, the 2-year Treasury yield, the yield-curve slope, the log equity index, CPI yearover-year inflation, and the unemployment rate, each expressed as its change since the window start. All series are public (FRED). The window ends before the document date, and the macro releases enter with a 45-day publication lag, so it contains only what markets actually knew at the time.

Automatically annotated sentences. Expert annotation covers only a fraction of the available text: the training-period documents contain roughly ten times as many sentences as the benchmark annotates. We label 13,017 of them with a prompted LLM, using the prompt of Shah et al. (2023) (Appendix D) – the same model that is also one of the three classifiers our system combines (§4). These silver labels are used for a single purpose: pretraining the sentence encoder (§4). The selection excludes benchmark sentences, duplicates, and any sentence with word overlap above 0.85 with a test sentence, so no test material enters pre-training.

![](images/8654f29a642e55c6f17ae2ede9fc5acb3c3da51c215a1f91e26dd3a9981b53ba.jpg)  
Figure 2: LabelFusion-TS. Three experts are trained independently and frozen: a RoBERTa encoder (pretrained on LLM-annotated silver sentences, fine-tuned on gold), a prompted LLM voting zero-shot, and a small time-series transformer reading the market window of the sentence’s document. A voting MLP combines the three views, exactly as in LabelFusion with one additional input.

## 4 LabelFusion-TS

LabelFusion-TS keeps the recipe of LabelFusion: Three specialised models, called experts in the following, are trained on the task individually, then frozen, and a small voting network is trained separately on their output (Figure 2). Afterwards, we compare LabelFusion-TS with a baseline model, the so-called market expert, which is a standard Transformer encoder (Vaswani et al., 2017) trained from scratch.

Text expert. RoBERTa-large (Liu et al., 2019) is trained in two stages. First, a silver stage: two epochs on the 13k LLM-annotated training-era sentences, which teaches the encoder the task’s shape from plentiful but imperfect labels. Then a gold stage: standard fine-tuning on the human-labelled training sentences plus the 96 recovered undated ones. Its CLS embedding $h \in \mathbb { R } ^ { 1 0 2 4 }$ represents the sentence, as in LabelFusion.

LLM expert. An instruction-tuned open LLM, gemma4:31b at temperature 0, classifies each sentence zero-shot with the classification prompt that Shah et al. (2023) used for ChatGPT, reproduced in Appendix D; its vote is encoded as $z \in \{ 0 , 1 \} ^ { 3 }$ The same model with the same prompt produces the silver labels of §3.

Fusion. With all experts frozen, a voting MLP with one hidden layer combines the three views of

a sentence,

$$
\begin{array} { r } { \hat { y } = f _ { \mathrm { M L P } } \big ( \big [ h ; ~ z ; ~ m \big ] \big ) \ \in \ [ 0 , 1 ] ^ { 3 } , } \end{array}\tag{2}
$$

which is LabelFusion’s fusion with one added input; removing m recovers LabelFusion exactly. Training involves random initialisation at three places, in the text expert, the market expert, and the voting network. To make the results independent of any particular draw, every component is trained with three different random seeds and all $3 \times 3 \times 3 = 2 7$ combinations are evaluated. We report their mean and, in addition, a probability ensemble that averages, per sentence, the class distributions predicted by the 27 systems and can be deployed as a single classifier.

## 5 Results

Table 1 reports the time-split evaluation. The market expert alone reaches 37.3%. The text expert reaches 66.1% and already exceeds the zero-shot LLM (64.1%) whose silver labels it was pre-trained on. The full fusion reaches 67.4% on average, with all 27 seed combinations above the LLM, and the probability ensemble of the 27 systems reaches 70.2%, ahead of every individual component.

<table><tr><td></td><td>wF1 (%)</td></tr><tr><td>market expert alone LLM zero-shot</td><td>37.3 64.1</td></tr><tr><td>RoBERTa-large, silver→gold</td><td>66.1</td></tr><tr><td>LabelFusion-TS (ours), mean of 27 runs</td><td>67.4</td></tr><tr><td>LabelFusion-TS (ours), probability ensemble</td><td>70.2</td></tr></table>

Table 1: Stance classification on the time-split test set (418 sentences from 2015–2022; models trained on sentences up to 2015). Weighted F1 in percent, following the benchmark’s metric; means are over all seed combinations of the trained components.

Table 2 reports the amount of human labelling required by the system. LabelFusion-TS overtakes zero-shot LLM at roughly 20% of the training pool (≈ 240 sentences) and remains above it for 100%, while the LLM row remains constant by construction. Schlee et al. (2026) report a crossover only after the majority of a substantially larger training set."

## 6 Discussion

Why market context can help. The market– label connection is visible before any model is trained: grouping training-year sentences by the federal-funds-rate move over the preceding 126 business days, 66% are hawkish when the rate rose by at least 10 basis points, versus only 38% when it fell as much. Hedging often leaves the words alone underdetermined; “act as appropriate” (Table 3) appears in a dovish 2019 sentence and a neutral 2000 one. A text-only model under a chronological split cannot observe the test period’s environment, while the market window supplies it at test time by construction. This also explains why the market expert is weak alone (37.3%): all sentences of a document share one window, so it contributes to the environment rather than an independent judgement, useful only in combination with the text views. The same logic should extend to other tasks whose label semantics shift with the economic environment, such as sentiment or risk classification across volatility regimes.

<table><tr><td>gold labels</td><td>LabelFusion-TS</td><td>RoBERTa</td><td>Market</td><td>LLM</td></tr><tr><td>5% (59)</td><td>63.4</td><td>61.9</td><td>36.1</td><td>64.1</td></tr><tr><td>10% (119)</td><td>62.8</td><td>63.9</td><td>35.6</td><td>64.1</td></tr><tr><td>20% (236)</td><td>66.3</td><td>66.3</td><td>38.4</td><td>64.1</td></tr><tr><td>40% (471)</td><td>68.1</td><td>65.4</td><td>37.6</td><td>64.1</td></tr><tr><td>60% (710)</td><td>67.8</td><td>65.1</td><td>41.0</td><td>64.1</td></tr><tr><td>80% (945)</td><td>69.7</td><td>67.4</td><td>38.7</td><td>64.1</td></tr><tr><td>100% (1181)</td><td>70.2</td><td>66.1</td><td>37.3</td><td>64.1</td></tr></table>

Table 2: Label-budget sweep: weighted F1 (%) as a function of the number of gold-labelled training sentences. Silver labels and the market expert’s rate-cycle pre-training are label-free and constant across rows. LabelFusion-TS is the probability ensemble; RoBERTa and the market expert are means over their three seeds; the zero-shot LLM uses no training data and is constant.

## 7 Conclusion & Limitations

Financial text classifiers routinely ignore an input every human analyst consults: the market context. Adding it as an additional expert to an existing fusion architecture, trained with silver labels from the fusion’s own LLM and evaluated on a train-on-thepast, test-on-the-future protocol, beats a modern zero-shot LLM with as few as ∼240 human labels, and a simple seed-grid ensemble adds another three points. Components weak alone can be strong together, and the market time series is a cheap, public, time-aligned signal that there is little reason to keep leaving out. The study is limited to one task and benchmark, a seven-year test span dominated by two unusual regimes, a recovered date subset (73% of sentences), and silver labels drawn from the same LLM used as the zero-shot baseline — a different teacher could shift the picture.

## References

Tadas Baltrušaitis, Chaitanya Ahuja, and Louis-Philippe Morency. 2017. Multimodal machine learning: A survey and taxonomy. ArXiv:1705.09406.

Brad M. Barber and Terrance Odean. 2008. All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors. The Review ofFinancial Studies, 21(2):785–818.

Yupeng Cao, Zhi Chen, Qingyun Pei, Nathan Jinseok Lee, K. P. Subbalakshmi, and Papa Momar Ndiaye. 2024. ECC analyzer: Extract trading signal from earnings conference calls using large language model for stock volatility prediction.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. ArXiv:1503.02531.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. RoBERTa: A robustly optimized BERT pretraining approach. ArXiv:1907.11692.

Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. 2023. A time series is worth 64 words: Long-term forecasting with transformers. In The Eleventh International Conference on Learning Representations.

Michael Schlee, Christoph Weisser, Timo Kivimäki, Melchizedek Mashiku, and Benjamin Säfken. 2026. Labelfusion: Fusing large language models with transformer encoders for robust financial news classification. In Proceedings ofthe 7th Financial Narrative Processing Workshop (FNP 2026), Palma de Mallorca, Spain.

Agam Shah, Suvan Paturi, and Sudheer Chava. 2023. Trillion dollar words: A new financial dataset, task & market analysis. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers).

Yining Sun, Penglei Gao, Yuyao Yan, and Xi Yang. 2026. Stock price prediction with attention-based framework by integrating LLM-generated features. In Neural Information Processing (ICONIP 2025), volume 16313 of Lecture Notes in Computer Science, pages 562–576. Springer, Singapore.

Paul C. Tetlock. 2007. Giving content to investor sentiment: The role of media in the stock market. The Journal ofFinance, 62(3):1139–1168.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems 30.

## A Benchmark Examples

## B Dating Procedure

The benchmark releases sentences with labels and a year, but not the document date needed to attach market data. All source documents are public and are included in the benchmark’s repository as date-named files. We match each sentence to its document by exact text match against the perdocument sentence files and, where that fails, by containment matching of the normalised sentence against the raw document text; normalisation lowercases and strips all non-alphanumeric characters, so that differences in punctuation, quotation marks, and sentence splitting do not prevent a match. Sentences matching documents on more than one date and sentences matching no document are excluded.

Three cases illustrate the procedure. The sentence “After precipitous drops in March and April, employment rose strongly in May and June as many people returned to workfrom temporary layoffs.” occurs in the sentence file of exactly one document and receives its date, the press conference of 29 July 2020; 1,203 sentences are dated by such exact matches. The sentence “Actual or realized saving depends on the equilibrium values of the real interest rate and other economic variables.” appears in no sentence file but is found, after normalisation, in the raw text of a single document, a speech of 10 March 2005; containment matching dates 489 further sentences this way. The sentence “Consistent with its statutory mandate, the Committee seeks to foster maximum employment and price stability.” is the Committee’s standing mandate formula and occurs in 76 documents; no unique date can be assigned, and it is excluded. This yields 1,692 uniquely dated sentences of 2,312 unique sentences (73%), spanning January 1996 to September 2022, with label proportions close to the full benchmark (neutral 47%, dovish 28%, hawkish 25%). By source, 893 come from meeting minutes, 489 from speeches, and 310 from press conferences. The year column of the release disagrees with the date of the matched document for the large majority of sentences and is not used. Undated sentences whose matching documents are all dated 2014 or earlier (96 sentences, excluding those with word overlap above 0.85 with a test sentence) are added to the text expert’s training data.

<table><tr><td>Label</td><td>Date</td><td>Sentence</td></tr><tr><td>hawkish</td><td>2005-05-03</td><td>A discernable upcreep was apparent in survey measures of short- and, to a limited extent, long-term inflation expectations over recent months.</td></tr><tr><td>hawkish</td><td>2017-12-13</td><td>Many indicated that they expected cyclical pressures associated with a tightening labor market to show through to higher inflation over the medium term.</td></tr><tr><td>dovish</td><td>2011-08-09</td><td>Some participants noted that additional asset purchases could be used to provide more accommodation by lowering longer-term interest rates.</td></tr><tr><td>dovish</td><td>2019-06-19</td><td>In light of increased uncertainties and muted inflation pressures, we now emphasize that the Committee will closely monitor the implications of incoming information for the economic outlook and will act as appropriate to sustain the expansion with a strong labor market and inflation near its 2 percent objective.</td></tr><tr><td>neutral</td><td>2000-02-02</td><td>As long as the Federal Reserve is required to set and report ranges for money and debt growth, it should update them as appropriate.</td></tr><tr><td>neutral</td><td>2004-12-02</td><td>A commonly used analogy takes the U.S. economy to be an automobile, the FOMC to be the driver, and monetary policy actions to be taps on the accelerator or brake.</td></tr></table>

Table 3: Annotated sentences of the benchmark, with the document date recovered by the procedure of Appendix B. The last two rows illustrate the difficulty of the task: both are labelled neutral although they discuss policy instruments, while the phrase “act as appropriate” appears in a dovish sentence of 2019 and in a neutral one of 2000.

## C Market Window Details

All series come from FRED, the public database of the Federal Reserve Bank of St. Louis, retrieved as CSV files through its unauthenticated download interface (fred.stlouisfed.org/graph/fredgraph.csv?id= the released code performs the retrieval. Table 4 lists the six channels and their construction.

<table><tr><td>Channel</td><td>FRED series</td><td>Construction</td></tr><tr><td>policy rate</td><td>DFF</td><td>level</td></tr><tr><td>2-year yield</td><td>DGS2</td><td>level</td></tr><tr><td>yield-curve slope</td><td>DGS10, DGS2</td><td>difference</td></tr><tr><td>equity market</td><td>NASDAQCOM</td><td>logarithm of the index</td></tr><tr><td>inflation</td><td>CPIAUCSL</td><td>12-month change, 45-day lag</td></tr><tr><td>unemployment</td><td>UNRATE</td><td>level, 45-day lag</td></tr></table>

Table 4: The six market channels. Daily series are used as released; monthly series (inflation, unemployment) are stepped forward to daily frequency and shifted by 45 days to respect their publication delay.

A window spans the 126 business days ending the day before the sentence’s document date. Within the window, every channel is expressed as its change since the first day of the window, and each channel is standardised with mean and standard deviation computed on the training split only.

## D LLM Prompt

The LLM expert uses, verbatim, the prompt published by Shah et al. (2023) for their ChatGPT baseline; [sentence] is replaced by the sentence to classify.

Classification prompt (Shah et al., 2023), verbatim   
Discard all the previous instructions. Behave like you   
,→ are an expert sentence classifier. Classify the   
,→ following sentence from FOMC into 'HAWKISH',   
,→ DOVISH', or 'NEUTRAL' class. Label 'HAWKISH' if   
,→ it is corresponding to tightening of the   
,→ monetary policy, 'DOVISH' if it is corresponding   
,→ to easing of the monetary policy, or 'NEUTRAL'   
);,→ if the stance is neutral. Provide the label in   
,→ the first line and provide a short explanation   
,→ in the second line. The sentence: [sentence]

The first line of the reply is parsed as the label. The same model and prompt produce the silver labels for the 13,017 automatically annotated sentences of §3 (99.98% of replies parse).

## E Training Details

Table 5 lists the training configuration of the three experts and the voting network; Table 6 accounts for the market expert’s parameters.

<table><tr><td>Text expert encoder optimiser batching loss silver stage gold stage precision</td><td>roberta-large, max. 128 tokens AdamW, learning rate 10−5 size 16, gradient accumulation 2 class-weighted cross-entropy 2 epochs on the 13,017 silver sentences ≤ 6 epochs, epoch selection on validation mixed (fp16)</td></tr><tr><td>Market expert input</td><td>126 × 6 window, 18 patches of 7 days 4 layers, width 64, 8 heads, FF 128</td></tr><tr><td>encoder regularisation pre-training</td><td>dropout 0.2 windows every 2nd business day 1962–2015,</td></tr><tr><td>task training</td><td>rate change in 30 days (±10bp, 3 classes) Adam,  $3 \times 1 0 ^ { - 4 } , \leq$  40 epochs, validation sel.</td></tr><tr><td>Voting network</td><td></td></tr><tr><td></td><td></td></tr><tr><td>architecture inputs</td><td>MLP, one hidden layer of 64 concatenated frozen expert outputs</td></tr></table>

Table 5: Training configuration.

<table><tr><td>patch projection (42 × 64 + biases) position embeddings (18 × 64) encoder layers  $( 4 \times 3 3 , 4 7 2 )$ </td><td>2,752 1,152 133,888</td></tr><tr><td>per layer: attention per layer: feed-forward per layer: layer norms</td><td>16,640 16,576</td></tr><tr><td>output layer norm</td><td>256 128</td></tr><tr><td>classification head</td><td>195</td></tr><tr><td>total</td><td>138,115</td></tr></table>

Table 6: Parameter count of the market expert.

Every trained component uses seeds {1, 2, 3}, giving the 27 systems of §4. Label budgets subsample the gold pool stratified by class; silver data and rate-cycle pre-training stay constant across budgets. The full pipeline, all seeds and all budgets, runs in under two hours on a single T4 GPU; an end-to-end notebook reproducing every number, together with the input bundle and its construction script, will be released (anonymised for review).