# From Entity Mentions to Tone: An LLM-Based Pipeline for Media Bias Analysis

Klesti Hoxha and Olti Qirici

Department of Informatics

Faculty of Natural Sciences

University of Tirana

Tirana, Albania

{klesti.hoxha, olti.qirici}@unitir.edu.al

Abstract—This paper presents a pipeline for analyzing media bias and framing in online news. The pipeline groups articles into topics and events, adds named-entity and sentiment annotations, and compares news sources through people mentions, sourcelevel tone, and event-level coverage patterns. We apply it to 8,358 Albanian news articles collected from GDELT and compare the resulting annotations with GDELT’s automated annotations. The results show moderate agreement for sentiment and entity extraction, as well as additional person-entity pairs that can potentially support the bias analysis. We compare two annotation prompts and find that stricter sentiment-validation rules remove label-score inconsistencies but increase execution time and reduce annotation coverage. Based on these results, the simpler prompt is used for the rest of the analysis. We have provided sample analysis on source-level framing profiles, person-level tone differences across sources, and event-level gatekeeping and coverage indicators. These outputs show how the same news collection can be used to examine what sources cover, how they describe public figures, and where coverage is concentrated. The approach is particularly useful in settings where manually verified datasets or specialized language tools are limited.

Keywords—media bias, media framing, large language models, named-entity recognition, sentiment analysis, news analysis

## I. INTRODUCTION

Online news providers have become the default way for many people to be informed. They can be accessed directly or, quite often, through social media platforms. In both cases, recommender-system algorithms [1] boost specific news items or topics [2], making readers vulnerable to disinformation and selective representations of events or facts [3]. Selection may focus on individuals, organizations, or specific topics covered in the news [3], while framing can be related to the tone used when covering people, organizations, or topics in specific news items [4].

Traditionally, this problem has been addressed through manual investigations of news providers by communication experts [5]. However, this process is heavyweight, costly, and itself prone to bias. Furthermore, it is difficult to apply continuously at the level of individual media providers or news articles.

As an alternative, machine-learning algorithms, such as classifiers, have been used to identify bias or framing in media content [6]. However, these methods also rely on languagespecific training data, which are usually created by human annotators.

To address gatekeeping and reduce disinformation [7] in light of recent advances in news dissemination technologies, reliable and fact-centered tools are needed to continuously assess bias and framing at the level of individual news providers and articles. Such tools are able to enhance news aggregator applications by enabling the visualization of these assessments.

This need is especially visible in smaller language settings such as Albanian. News is published by many outlets with different editorial positions, but there are fewer ready-made resources for large-scale entity recognition, sentiment analysis, and source comparison. As a result, even basic questions can be difficult to answer consistently: which sources covered an event, which public figures were mentioned, and whether the tone differed across outlets.

For this reason, media-bias analysis should not rely on a single score. It should separate several related signals: sourcelevel tendencies, person-level tone, and event-level coverage. Source-level summaries help show whether a provider is generally more favorable or critical. Person-level summaries show how public figures are described by different outlets. Event-level indicators show how widely an event is covered and where coverage is concentrated.

Recent advances in multilingual LLMs have enabled accurate and reliable approaches to NLP tasks such as namedentity recognition (NER) and sentiment analysis [8]. Both tasks are strongly related to the automated monitoring of bias and framing in news articles. In this work, we introduce a language-independent pipeline that continuously monitors media bias and framing by using LLMs to track mentions and tone. Beyond a static evaluation of a news provider’s bias toward individuals or organizations, the pipeline compares coverage of the same event across different news outlets to identify shifts in tone and visibility. We test this system in a low-resource media environment, showing how local open-source language models can support structured sentiment and entity analysis without requiring prior language-specific adaptation.

This work makes four main contributions. First, it implements a pipeline, built with Kedro and a local Gemma model, for studying bias, framing, and gatekeeping in Albanian news. Second, it compares two annotation prompts and shows that stricter validation removes label-score inconsistencies but also slows processing and reduces annotation coverage. Third, it compares the resulting annotations with GDELT outputs on 8,358 Albanian news articles. Fourth, it uses the annotated dataset to provide a general framework for evaluating sourcelevel framing profiles, person-level tone differences, and eventlevel coverage indicators across 124 news sources, establishing a reproducible blueprint applicable to other low-resource language news environments.

The rest of the paper follows this structure. We first define the bias and framing categories used in the study and then review related work. Next, we describe the dataset, prompts, and processing pipeline. We then evaluate the annotations and present the source-, person-, and event-level analyses. The paper ends with a discussion of the main limitations of the approach and potential directions for future work.

## II. BIAS AND FRAMING CATEGORIES

From the reviewed literature, we identified several recurring categories of media bias and framing. These categories distinguish between what news outlets choose to cover, how they describe it, and how much attention they allocate to specific actors or events.

Media bias is commonly treated as a deviation from neutrality [3]. Table I summarizes the bias categories considered in this work.

TABLE I  
MEDIA BIAS CATEGORIES
<table><tr><td>Category</td><td>Description</td><td>Reference</td></tr><tr><td>Selection</td><td>Editorial &quot;gatekeeping&quot;; choosing what is deemed newsworthy or omitting spe- cific details.</td><td>[3]</td></tr><tr><td>Tone</td><td>The emotional &quot;lean&quot; or sentiment va- lence applied to specific actors or events.</td><td>[9]</td></tr><tr><td>Reporting/Source</td><td>The long-term reputation, systemic lean, and ideological history of a news do- main.</td><td>[10]</td></tr><tr><td>Depth</td><td>Uneven coverage space or word count dedicated to one side of a story.</td><td>[4]</td></tr></table>

Media framing captures how a narrative is structured once an event is selected for coverage. Table II lists the framing categories used to characterize these narrative choices.

## III. RELATED WORK

Several recent studies have used LLMs to detect bias and framing in news outlets.

TABLE IIMEDIA FRAMING CATEGORIES
<table><tr><td>Category</td><td>Description</td><td>Reference</td></tr><tr><td>Selection</td><td>Structural visibility of narrative elements; what information is presented or omitted.</td><td>[3], [9]</td></tr><tr><td>Tone</td><td>The emotional undertone, spin, or evaluative posture applied to the news narrative.</td><td>[4], [9]</td></tr><tr><td>Emphasis</td><td>The relative importance and priority given to specific themes or viewpoints within a text.</td><td>[3]</td></tr><tr><td>Targeted</td><td>Entity-specific framing shifts depending on the politician or actor involved.</td><td>[11]</td></tr></table>

Wang et al. [3] created a real-time dashboard that visualizes bias metrics such as tone, topic, and political lean. LLMs are used to quantify and categorize these metrics. The user interface allows users to browse individual events and coverage details, while also providing aggregated bias metrics at the publisher level. The developed system was evaluated with communication experts and crowdsourced participants. The results were promising, showing a high statistical correlation with human-based annotations.

In another study, Kumar et al. [4] created a framework and application that measure possible bias at the level of individual stories or events. It allows users to compare news stories side by side, highlighting possible polarization in their framing. The authors use LLMs to perform sentiment analysis and emphasize their benefits for deeper, context-based sentiment detection. LLMs are also used to compare coverage across news providers reporting on the same story.

In earlier pre-LLM work, Ye and Skiena [12] computed source-related rankings for 50,000 news sources globally. Their ranking framework included reporting bias, peer reputation, and social-media-based popularity. The results were evaluated against independent expert scorecards and showed good agreement.

Similarly, Rönnback et al. [10] used machine learning to perform large-scale labeling and classification of large datasets of news providers (domains). The outputs of their system were human-interpretable. Training was based on the GDELT dataset [13], while evaluation was conducted using a humanlabeled dataset. They also included an LLM baseline, but it performed worse than a non-LLM neural network.

LLMs have also been used for dataset augmentation and synthetic news generation. Wessel [6] used LLMs to augment a dataset of bias-indicating keywords by replacing keywords with generated alternatives. This approach aims to reduce the effect of “spurious cues” in machine-learning models. The author argues that the resulting model is less dependent on the context of the observed keywords. Tohidi et al. [9], on the other hand, used LLMs to demonstrate that generated biased news can affect public opinion. Their results showed that the way news is presented to readers can influence their perceptions, even when the underlying facts remain unchanged. This further indicates the need for better ways to address bias and framing when combating disinformation.

Despite the successful use of LLMs for detecting bias and framing, LLMs have also been shown to carry biases. Elbouanani et al. [11] found experimentally that tone (sentiment) classification differs for identical phrases when the political targets in those phrases are substituted. As a mitigation strategy, they show that replacing politician names with fictitious ones reduces bias during sentiment classification. In this way, the context in which those politicians are mentioned carries more weight in the sentiment classification.

## IV. METHOD

In this work, we develop a media-bias analysis pipeline that tracks mentions of people in news articles and measures their sentiment polarization, or tone (Fig. 1). The tracked news articles are written in Albanian, a low-resource language.

To the best of our knowledge, there is no widely used stateof-the-art NER toolkit for Albanian, and the same holds for sentiment analysis. Most previous work consists of scholarly experiments involving the creation of small-scale, humanannotated datasets [14], [15] or fine-tuned language models such as XLM-RoBERTa [16]. Therefore, using LLMs to tackle these tasks may offer practical benefits.

![](images/b09cfdcd603e81803fcbbce55cadd7bc819a3e8bf522932145934b611c4002f3.jpg)  
Fig. 1. Overview of the proposed media-bias analysis pipeline.

For our experiments, we used a dataset of 8,358 Albanian news articles from GDELT [13], published in April 2026 (Table III). To detect potential bias toward specific topics or events, the incoming news items are grouped into topics using a simple TF-IDF and k-means approach. We then identify events within these topics using TF-IDF in combination with cosine similarity and time windows. Each news item, identified by a unique URL, is enriched with NER and sentimentanalysis outputs using Gemma 4<sup>1</sup>, a local LLM.

TABLE III  
NEWS DATASET DETAILS
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>Articles</td><td>8,358</td></tr><tr><td>Unique news sources</td><td>124</td></tr><tr><td>Topic clusters</td><td>50</td></tr><tr><td>Event clusters</td><td>3,551</td></tr><tr><td>Multi-source events</td><td>1,340</td></tr><tr><td>Articles per source (min/max/mean/median)</td><td>1 / 667  / 67.4 /  23</td></tr><tr><td>Articles per topic (min/max/mean/median)</td><td>14 / 699 /  167.16 /  127.50</td></tr><tr><td>Articles per event (min/max/mean/median)</td><td>1 / 48 / 2.35 / 1</td></tr><tr><td>Article text length chars (mean/median)</td><td>1622 / 1379</td></tr></table>

The pipeline was developed using the Kedro<sup>2</sup> Python framework. It measures bias and framing by tracking tone variation in topic and event coverage, as well as tonal bias in mentions of people. It also computes possible gatekeeping indicators by analyzing which people, topics, or events are repeatedly covered or omitted across news sources.

## V. EXPERIMENTS AND RESULTS

In this section, we describe the experiments and their results. Our aim is to demonstrate the feasibility of implementing such a pipeline by presenting potential use cases.

## A. NER and Sentiment Enrichment

Our experiments were run on a cloud instance with an NVIDIA A10 GPU (24 GB), 30 vCPUs, and 200 GiB of RAM. We relied on the local LLM Gemma 4 for NER and sentimentanalysis enrichment. The model temperature was set to 0.0 to obtain deterministic outputs, and generation was limited to 250 tokens. Requests used a 90-second timeout with one retry in case of failure. The prompts instructed the model to extract sentiment and NER data from the full text of each news article and return the output in JSON format.

Listings 1 and 2 show the two prompts used for NER and sentiment enrichment. The output of the first version showed inconsistencies between the sentiment labels (positive, negative, neutral) and their corresponding scores. The second version addresses this issue by adding explicit validation rules.

You are an information extraction assistant.   
Return JSON only with this exact schema:   
{   
"sentiment": {"label": "positive|neutral|negative",   
"score": number between -1 and 1},   
"entities": {   
"PER": [{"original": string, "canonical\_en": string}],   
<sup>1</sup>https://ai.google.dev/gemma   
<sup>2</sup>https://kedro.org/

"ORG": [{"original": string, "canonical\_en": string}],   
"LOC": [{"original": string, "canonical\_en": string}]   
}   
}.   
Rules: canonical\_en must be an English canonical form, keep   
original as text surface form from article, omit   
uncertain entities, no extra keys, no markdown.   
ARTICLE:   
{article\_text}

Listing 1. Prompt v1 for NER and sentiment enrichment

You are an information extraction assistant.   
Return JSON only with this exact schema:   
{   
"sentiment": {"label": "positive|neutral|negative",   
"score": number between -1 and 1},   
"entities": {   
"PER": [{"original": string, "canonical\_en": string}],   
"ORG": [{"original": string, "canonical\_en": string}],   
"LOC": [{"original": string, "canonical\_en": string}]   
}   
}.   
Rules:   
1. Sentiment score must range from -1.0 (very negative) to   
1.0 (very positive).   
2. Label MUST match score: use ’negative’ if score < -0.3,   
’positive’ if score > 0.3, ’neutral’ otherwise.   
3. canonical\_en must be an English canonical form, keep   
original as text surface form from article.   
4. Omit uncertain entities, no extra keys, no markdown, and   
no reasoning text.   
ARTICLE:   
{article\_text}  
Listing 2. Prompt v2 for NER and sentiment enrichment

Prompt v2 was slower and produced fewer labeled articles for both sentiment and NER (Table IV). The inconsistency rate between sentiment labels and scores in the first version was 8.53%. The gain in consistency did not justify the doubled execution time, and the remaining inconsistencies can be addressed in a cleaning step.

TABLE IV  
LLM ENRICHMENT VERSIONS COMPARISON
<table><tr><td>Metric</td><td>Prompt v1</td><td>Prompt v2</td></tr><tr><td>Sentiment Labeled Articles</td><td>51.75%</td><td>37.20%</td></tr><tr><td>Inconsistent Sentiment Label / Score</td><td>8.53%</td><td>0.0%</td></tr><tr><td>NER-Labeled Articles (any entity)</td><td>39.97%</td><td>27.20%</td></tr><tr><td>Articles with Person Entities (PER)</td><td>8.25%</td><td>5.77%</td></tr></table>

In the absence of a gold-standard test dataset, we evaluated the quality of the LLM-based annotations by measuring their agreement with the annotations provided by GDELT. These annotations are also generated by an automated NLP pipeline. The NER evaluation was based only on the PERSON category because our pipeline focuses on mentions of people. For this evaluation, we considered only articles labeled by both GDELT and the LLM-based annotator. We used the following evaluation metrics.

For sentiment, we report the number of comparable articles and the label agreement rate, defined as the percentage of articles where the LLM and GDELT sentiment labels match. For NER, we compare URL–entity-type pairs and report three macro-averaged agreement rates: one from the LLM side, one from the GDELT side, and a balanced score that averages both perspectives.

Overall, the agreement rates in Table V and Table VI show moderate alignment between the LLM-based annotations and the GDELT annotations. For sentiment, Prompt v2 achieves a slightly higher label agreement rate than Prompt v1. For NER, the LLM identifies more person-entity pairs than GDELT. At the same time, the high GDELT-side agreement shows that entities identified by GDELT are also found by the LLM to a large degree.

Rather than contradicting the existing data, the LLM actually validates it. It recovers the majority of GDELT’s person mentions while successfully identifying additional ones the source overlooked. These additional pairs should not be treated as automatically correct without manual validation, but when lacking a better alternative, they can still be used for mediabias analysis because missed person mentions can reduce the quality of source-level and person-level tone comparisons. Because the analysis focuses on people mentioned in the news, less frequent mentions can still be manually verified in the later stages of the pipeline.

TABLE V  
SENTIMENT EVALUATION AGAINST GDELT
<table><tr><td>Metric</td><td>Prompt v1</td><td>Prompt v2</td></tr><tr><td>Comparable articles</td><td>3,612</td><td>3,109</td></tr><tr><td>Label agreement rate</td><td>58.72%</td><td>62.34%</td></tr></table>

TABLE VI  
PEOPLE-ONLY NER AGREEMENT RATES AGAINST GDELT
<table><tr><td>Metric</td><td>Prompt v1</td><td>Prompt v2</td></tr><tr><td>Evaluable pairs</td><td>690</td><td>483</td></tr><tr><td>Macro LLM agreement rate</td><td>36.83%</td><td>36.12%</td></tr><tr><td>Macro GDELT agreement rate</td><td>71.46%</td><td>70.41%</td></tr><tr><td>Macro balanced agreement rate</td><td>45.30%</td><td>44.44%</td></tr></table>

While the agreement rates in Table VI measure overlap from both the LLM and GDELT perspectives for personname identification, we also report precision, recall, and F1 scores to summarize the same comparison in a more standard information extraction format. The results are very similar across prompt versions. Recall is higher than precision because the LLM-based person recognizer identifies most names also found in the GDELT dataset while extracting additional names.

TABLE VII  
PEOPLE-ONLY NER PRECISION AND RECALL AGAINST GDELT
<table><tr><td>Version</td><td>LLM people</td><td>GDELT people</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>vl</td><td>1860</td><td>799</td><td>30.2%</td><td>70.3%</td><td>42.3%</td></tr><tr><td>v2</td><td>1309</td><td>560</td><td>29.8%</td><td>69.6%</td><td>41.7%</td></tr></table>

The evaluation results suggest that, for low-resource languages such as Albanian, LLM-based annotation can provide a useful starting point when no gold-standard dataset or welltrained alternative tools are available for NER and sentiment extraction. GDELT is also based on automated annotation, so these results should be read as a comparison with an existing large-scale system rather than as a definitive evaluation. The agreement with GDELT suggests that the LLM outputs are generally consistent with an established reference, while also identifying additional entities that may be useful for bias analysis.

Furthermore, although Prompt v2 reduces inconsistencies between sentiment labels and scores, it does so at the cost of slower execution and does not substantially improve the evaluation results. Therefore, we decided to continue with the dataset labeled using Prompt v1.

## B. Media Bias and Framing Analysis

Table VIII presents an excerpt from the source-level framing and bias profiles. The table summarizes how frequently each source uses positive, neutral, or negative framing, together with aggregate sentiment and bias measures. The reported measures are defined as follows.

• Pos. / Neu. / Neg. — share of the source’s labeled articles classified as positive, neutral, and negative sentiment, respectively.

• Mean score — average raw sentiment score across all labeled articles of the source; positive values indicate favorable coverage, while negative values indicate critical coverage.

• Avg. bias — source-level framing bias, computed as the average of per-topic framing bias scores (positive rate minus negative rate for each topic), then averaged across all topics covered by the source.

• Bias vs. corpus — deviation of the source’s average bias from the corpus-wide baseline (baseline = −0.192 in this study). A positive value indicates the source frames topics more favorably than the corpus average; a negative value indicates systematically more critical coverage.

The results are broadly consistent with the generally perceived editorial positions of the these news sources in the Albanian news landscape. This analysis can help highlight differences in how news providers frame the same topics.

Table IX shows an excerpt of a person-tone bias analysis. The table reveals how different news sources frame specific people in their coverage, showing whether sources systematically portray particular individuals more favorably or critically. For each person, only the lowest and highest tone-balance rows across different sources are displayed, illustrating the range of framing. The reported measures are defined as follows.

• Source — the news source that mentioned the person.

• Mentioned articles — number of labeled articles from that source in which the person was mentioned.

• Mean score — average raw sentiment score across all mentions of the person in that source’s articles; positive values indicate favorable coverage, negative values indicate critical coverage.

• Tone balance — framing balance for the person at that source, computed as positive rate minus negative rate across all mentions; ranges from −1 (uniformly negative) to +1 (uniformly positive).

• Pos. / Neg. — share of mentions of the person in that source’s articles classified as positive or negative sentiment, respectively.

This analysis can be used to compare how the same public figure is framed across news sources. For example, a news aggregator could surface cases where a person receives strongly positive coverage in one source and strongly negative coverage in another, helping readers identify possible framing differences around the same actor. While our practical testing focuses on well-known regional and international players to show effective local tracking, the basic extraction method works for any target and can easily handle any group of entities.

As noted in [11], pre-trained models may carry biases when processing mentions of politicians in news articles. However, the results can still serve as a useful baseline, and such biases can be mitigated through name anonymization or userprovided feedback.

Table X presents event-level gatekeeping and coverage indicators. It shows how widely events are covered across news sources and how concentrated that coverage is. This approach is similar to those applied by Wang et al. [3] and Kumar et al. [4]. The reported measures are defined as follows.

• Event — event identifier, sorted by article count in descending order.

• Articles — total number of articles assigned to that event.

• Covering Sources — number of distinct news sources that covered the event.

• Omitting Sources — number of active sources in the corpus that did not cover the event, computed as total active sources minus covering sources.

• Gatekeeping Score — structural selectivity index defined as 1 − <sup>covering</sup> <sup>sources</sup> ; higher values indicate narrower total active sources dissemination and stronger gatekeeping.

• Framing Polarity — event-level sentiment balance computed as positive rate minus negative rate across labeled articles for that event; positive values indicate more favorable framing, negative values indicate more critical framing.

This analysis can be used to identify events that receive broad attention and events that are covered by only a small set of sources. For example, a news aggregator could flag events with high gatekeeping scores and compare their framing polarity across sources, helping readers notice possible gaps in coverage or differences in tone.

## VI. DISCUSSION

The comparison with GDELT should be interpreted with care. GDELT is not a human-labeled gold standard, but another automated annotation system. Therefore, differences between our results and GDELT do not automatically mean that one side is wrong. The moderate agreement scores show that the two systems often overlap, but also capture different parts of the text. In particular, the LLM found many person mentions that were not present in the GDELT output, which is useful for bias analysis because missing a public figure can weaken later source and tone comparisons.

TABLE VIII  
AGGREGATE SOURCE FRAMING AND BIAS PROFILES
<table><tr><td>Source</td><td>Topics</td><td>Articles</td><td>Pos.</td><td>Neu.</td><td>Neg.</td><td>Mean score</td><td>Avg. bias</td><td>Bias vs. corpus</td></tr><tr><td>tvkoha.tv</td><td>2</td><td>10</td><td>22.7%</td><td>27.3%</td><td>50.0%</td><td>+0.100</td><td>-0.857</td><td>-0.665</td></tr><tr><td>3shi.net</td><td>4</td><td>16</td><td>40.7%</td><td>29.6%</td><td>29.6%</td><td>+0.170</td><td>+0.200</td><td>+0.392</td></tr><tr><td>tv21.tv</td><td>5</td><td>27</td><td>11.4%</td><td>50.0%</td><td>38.6%</td><td>+0.177</td><td>-0.554</td><td>-0.362</td></tr><tr><td>kosovapress.com</td><td>4</td><td>17</td><td>9.7%</td><td>25.8%</td><td>64.5%</td><td>-0.203</td><td>-0.542</td><td>-0.350</td></tr><tr><td>rtsh.al</td><td>1</td><td>33</td><td>38.5%</td><td>33.3%</td><td>28.2%</td><td>+0.108</td><td>+0.151</td><td>+0.344</td></tr><tr><td>epokaere.com</td><td>3</td><td>10</td><td>27.6%</td><td>34.5%</td><td>37.9%</td><td>+0.324</td><td>+0.111</td><td>+0.303</td></tr><tr><td>sot.com.al</td><td>18</td><td>122</td><td>9.6%</td><td>35.3%</td><td>55.1%</td><td>+0.128</td><td>-0.493</td><td>-0.301</td></tr><tr><td>noa.al</td><td>2</td><td>29</td><td>10.0%</td><td>36.7%</td><td>53.3%</td><td>+0.123</td><td>-0.475</td><td>-0.283</td></tr><tr><td>vizionplus.tv</td><td>10</td><td>59</td><td>12.0%</td><td>29.3%</td><td>58.7%</td><td>-0.088</td><td>-0.459</td><td>-0.267</td></tr><tr><td>ina-online.net</td><td>7</td><td>25</td><td>27.5%</td><td>42.5%</td><td>30.0%</td><td>+0.170</td><td>+0.071</td><td>+0.264</td></tr></table>

Corpus baseline = -0.192; only sources with at least 10 sentiment labeled articles are included.

TABLE IX  
PERSON-TONE EXTREMES BY PERSON ACROSS NEWS SOURCES
<table><tr><td>Person</td><td>Source</td><td>Mentioned articles</td><td>Mean score</td><td>Tone balance</td><td> $\mathbf { P o s . }$ </td><td> $\mathbf { N e g . }$ </td></tr><tr><td>Benjamin Netanyahu</td><td>balkanweb.com</td><td>5</td><td>+0.000</td><td>-0.800</td><td>0.0%</td><td>80.0%</td></tr><tr><td>Benjamin Netanyahu</td><td>panorama.com.al</td><td>6</td><td>+0.300</td><td>-0.333</td><td>16.7%</td><td>50.0%</td></tr><tr><td>Donald Trump</td><td>opinion.al</td><td>8</td><td>-0.300</td><td>-0.750</td><td>12.5%</td><td>87.5%</td></tr><tr><td>Donald Trump</td><td>lajmi.net</td><td>6</td><td>+0.400</td><td>+0.500</td><td>50.0%</td><td>0.0%</td></tr><tr><td>Edi Rama</td><td>syri.net</td><td>10</td><td>+0.190</td><td>-1.000</td><td>0.0%</td><td>100.0%</td></tr><tr><td>Edi Rama</td><td>kohajone.com</td><td>8</td><td>+0.600</td><td>+0.250</td><td>62.5%</td><td>37.5%</td></tr><tr><td>Kreshnik Mujeci</td><td>top-channel.tv</td><td>5</td><td>-0.140</td><td>-1.000</td><td>0.0%</td><td>100.0%</td></tr><tr><td>Kreshnik Mujeci</td><td>sot.com.al</td><td>5</td><td>-0.660</td><td>-0.800</td><td>0.0%</td><td>80.0%</td></tr><tr><td>Sali Berisha</td><td>balkanweb.com</td><td>11</td><td>+0.455</td><td>-0.818</td><td>9.1%</td><td>90.9%</td></tr><tr><td>Sali Berisha</td><td>syri.net</td><td>6</td><td>+0.667</td><td>-0.500</td><td>16.7%</td><td>66.7%</td></tr></table>

Excerpt showing minimum and maximum tone balance per person; person-source pairs with $\geq 5$ mentions are included.

TABLE X  
GATEKEEPING AND COVERAGE INDICATORS FOR MAJOR EVENTS
<table><tr><td>Event</td><td>Articles</td><td>Covering sources</td><td>Omitting sources</td><td>Gatekeeping score</td><td>Framing polarity</td></tr><tr><td>Event 1</td><td>48</td><td>21</td><td>103</td><td>0.8306</td><td>+0.0227</td></tr><tr><td>Event 2</td><td>46</td><td>35</td><td>89</td><td>0.7177</td><td>+0.1818</td></tr><tr><td>Event 3</td><td>39</td><td>19</td><td>105</td><td>0.8468</td><td>+0.0625</td></tr><tr><td>Event 4</td><td>37</td><td>22</td><td>102</td><td>0.8226</td><td>一</td></tr><tr><td>Event 5</td><td>33</td><td>19</td><td>105</td><td>0.8468</td><td>-1.0000</td></tr><tr><td>Event 6</td><td>33</td><td>10</td><td>114</td><td>0.9194</td><td>+0.0588</td></tr><tr><td>Event 7</td><td>30</td><td>18</td><td>106</td><td>0.8548</td><td>+1.0000</td></tr><tr><td>Event 8</td><td>29</td><td>23</td><td>101</td><td>0.8145</td><td>-0.0625</td></tr><tr><td>Event 9</td><td>29</td><td>22</td><td>102</td><td>0.8226</td><td>+0.7142</td></tr><tr><td>Event 10</td><td>29</td><td>14</td><td>110</td><td>0.8871</td><td>一</td></tr></table>

Rows are ranked by article count, then by source count.

The prompt comparison also shows a practical trade-off. Prompt v2 reduced inconsistencies between sentiment labels and scores, but it also processed fewer articles and took longer to run. This suggests that adding strict validation rules directly to the prompt can make the task harder for a local model. For a continuous pipeline, Prompt v1 is therefore more practical: it produces broader coverage, runs faster, and remaining inconsistencies can be handled afterward with simple rule-based checks.

Another useful benefit of LLM-based annotation is that the model helps clean up messy news data. In multilingual and low-resource situations, raw article text often has inconsistent names, different spellings, formatting errors, and mixed language forms. A traditional process would need a lot of manual cleaning before entity and sentiment analysis could be done reliably. In contrast, the LLM can often turn these surface differences into more consistent entity and sentiment results, lowering the amount of preparation needed before analysis.

The person-tone results should also be read with caution. Large tone differences across sources may reflect real editorial framing, but they may also be affected by biases already present in the model. This is especially important for political figures, where the model may have learned associations before seeing the article being analyzed. Future versions of the pipeline should test mitigation strategies such as replacing politician names with neutral placeholders [11] and adding human feedback for disputed cases.

## VII. CONCLUSION

In this paper, we presented an end-to-end NLP pipeline that uses local LLMs and the Kedro framework to track media bias, framing, and gatekeeping in online news. We applied the pipeline to 8,358 Albanian news articles from April 2026 and showed that a general-purpose open-weights model such as Gemma 4 can extract useful semantic signals in a low-resource setting without a large local training corpus. The evaluation showed that stricter prompt-level validation improves labelscore consistency and JSON schema compliance, but reduces processing speed and annotation coverage. For this reason, the simpler prompt, combined with external rule-based data cleaning, is the more practical choice for continuous ingestion pipelines.

Given the relatively low coverage rates and the computational cost of extracting NER and sentiment information from news articles, future work will train lighter-weight machinelearning models using LLM annotations and crowdsourced data.

The media-bias and framing analysis in Section V-B shows how source-level profiles summarize general framing tendencies, person-level tone balances show how public figures are treated across outlets, and event-level indicators show how widely stories are covered. Taken together, these views provide a practical basis for comparing news sources and for building tools that help readers inspect coverage differences more easily.

These results also show the practical role of LLM-based enrichment in a news-analysis workflow. The goal is not to replace editorial judgment or expert review, but to reduce the amount of material that must be inspected manually by turning large article collections into clearer signals. This is particularly useful in low-resource media settings, where continuous monitoring is difficult and language-specific NLP tools are still limited.

For this reason, LLM annotations are best understood here as a fast bootstrapping layer rather than as a state-of-the-art alternative to human validation. They provide a useful baseline from which similar media-bias analyses can be started quickly, especially when no manually normalized dataset is available. Since most people mentioned in news articles are public figures, a later human editorial validation step is also feasible:

editors can review a relatively constrained set of extracted names, aliases, and disputed tone assignments instead of annotating the full article collection from scratch.

In addition to providing dashboard visualizations, we plan to create an MCP server that would make the media-bias and framing functions available to AI-powered systems. We believe that a continuous media-bias monitoring pipeline is most useful when integrated into an existing system, such as a news aggregator, where it can show readers polarization indicators or surface events covered from different angles by other news providers.

## REFERENCES

[1] S. Raza and C. Ding, “News recommender system: a review of recent progress, challenges, and opportunities,” Artificial Intelligence Review, vol. 55, no. 1, pp. 749–800, Jan. 2022. [Online]. Available: https://doi.org/10.1007/s10462-021-10043-x

[2] K. Hoxha, “Real Time Media Bias and Framing Detection using LLMs,” in Book of Abstracts Scientific-practical Conference Innovations in Publishing, Printing and Multimedia Technologies 2026, Kaunas, Lituania, 2026.

[3] J. S. Wang, S. Haider, A. Tohidi, A. Gupta, Y. Zhang, C. Callison-Burch, D. Rothschild, and D. J. Watts, “Media Bias Detector: Designing and Implementing a Tool for Real-Time Selection and Framing Bias Analysis in News Coverage,” in Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems. Yokohama Japan: ACM, Apr. 2025, pp. 1–27. [Online]. Available: https://dl.acm.org/doi/10.1145/3706598.3713716

[4] P. K. R, B. Mohan G, A. R. S, and J. Y, “Sentinel: An Integrated Framework for News Sentiment Analysis, Bias Detection, and Coverage Comparison Using LLMs,” in 2025 Fourth International Conference on Smart Technologies, Communication and Robotics (STCR). Sathyamangalam, India: IEEE, May 2025, pp. 1–6. [Online]. Available: https://ieeexplore.ieee.org/document/11019897/

[5] M. Castillo-Campos, D. Becerra-Alonso, and H. G. Boomgaarden, “Automated Detection of Media Bias Using Artificial Intelligence and Natural Language Processing: A Systematic Review,” Social Science Computer Review, p. 08944393251331510, Apr. 2025. [Online]. Available: https://journals.sagepub.com/doi/10.1177/08944393251331510

[6] M. Wessel, “LLM-based Adversarial Dataset Augmentation for Automatic Media Bias Detection,” in Proceedings of the 9th Joint SIGHUM Workshop on Computational Linguistics for Cultural Heritage, Social Sciences, Humanities and Literature (LaTeCH-CLfL 2025), A. Kazantseva, S. Szpakowicz, S. Degaetano-Ortlieb, Y. Bizzoni, and J. Pagel, Eds. Albuquerque, New Mexico: Association for Computational Linguistics, May 2025, pp. 19–24. [Online]. Available: https://aclanthology.org/2025.latechclfl-1.3/

[7] D. M. J. Lazer, M. A. Baum, Y. Benkler, A. J. Berinsky, K. M. Greenhill, F. Menczer, M. J. Metzger, B. Nyhan, G. Pennycook, D. Rothschild, M. Schudson, S. A. Sloman, C. R. Sunstein, E. A. Thorson, D. J. Watts, and J. L. Zittrain, “The science of fake news,” Science, vol. 359, no. 6380, pp. 1094–1096, Mar. 2018. [Online]. Available: https://www.science.org/doi/abs/10.1126/science.aao2998

[8] L. Qin, Q. Chen, X. Feng, Y. Wu, Y. Zhang, Y. Li, M. Li, W. Che, and P. S. Yu, “Large language models meet NLP: a survey,” Frontiers of Computer Science, vol. 20, no. 11, p. 2011361, Mar. 2026. [Online]. Available: https://doi.org/10.1007/s11704-025-50472-3

[9] A. Tohidi, S. Haider, and D. J. Watts, “Rethinking news framing with large language models,” Scientific Reports, vol. 15, no. 1, p. 45592, Nov. 2025. [Online]. Available: https://www.nature.com/articles/ s41598-025-29519-9

[10] R. Rönnback, C. Emmery, and H. Brighton, “Automatic large-scale political bias detection of news outlets,” PLOS ONE, vol. 20, no. 5, p. e0321418, May 2025. [Online]. Available: https://journals.plos.org/ plosone/article?id=10.1371/journal.pone.0321418

[11] A. Elbouanani, E. Dufraisse, and A. Popescu, “Analyzing Political Bias in LLMs via Target-Oriented Sentiment Classification,” in Findings of the Association for Computational Linguistics: ACL 2025, W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, Eds. Vienna, Austria:

Association for Computational Linguistics, Jul. 2025, pp. 15 476–15 505. [Online]. Available: https://aclanthology.org/2025.findings-acl.799/

[12] J. Ye and S. Skiena, “MediaRank: Computational Ranking of Online News Sources,” in Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. Anchorage AK USA: ACM, Jul. 2019, pp. 2469–2477. [Online]. Available: https://dl.acm.org/doi/10.1145/3292500.3330709

[13] K. Leetaru and P. A. Schrodt, “GDELT: Global data on events, location, and tone, 1979–2012,” in Proceedings of the International Studies Association Annual Convention, San Diego, CA, 2013, p. April 2013.

[14] F. Kadriu, D. Murtezaj, F. Gashi, L. Ahmedi, A. Kurti, and Z. Kastrati, “Human-annotated dataset for social media sentiment analysis for Albanian language,” Data in Brief, vol. 43, p. 108436, Aug. 2022. [Online]. Available: https://linkinghub.elsevier.com/retrieve/ pii/S2352340922006333

[15] N. Kote, K. Kalliri, K. Kalliri, A. Haveriku, B. Muraku, and E. K. Meçe, “NER for Albanian Language: A Manually Annotated Corpus and Machine Learning Models,” in Advanced Information Networking and Applications, L. Barolli, Ed. Cham: Springer Nature Switzerland, 2025, vol. 247, pp. 153–165, series Title: Lecture Notes on Data Engineering and Communications Technologies. [Online]. Available: https://link.springer.com/10.1007/978-3-031-87769-8\_14

[16] K. P. Nuci, P. Landes, and B. Di Eugenio, “RoBERTa Low Resource Fine Tuning for Sentiment Analysis in Albanian,” in Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), N. Calzolari, M.-Y. Kan, V. Hoste, A. Lenci, S. Sakti, and N. Xue, Eds. Torino, Italia: ELRA and ICCL, May 2024, pp. 14 146–14 151. [Online]. Available: https://aclanthology.org/2024.lrec-main.1233/