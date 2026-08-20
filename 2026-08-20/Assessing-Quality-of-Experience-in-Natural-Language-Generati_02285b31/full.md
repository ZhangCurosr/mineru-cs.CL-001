# Assessing Quality of Experience in Natural Language Generation of German Text

Dinh Nam Pham<sup>1,2\*</sup>, Shushen Manakhimova<sup>2</sup>, Vivien Macketanz<sup>2</sup>, Sebastian M¨oller<sup>1,2\*</sup>

<sup>1</sup>Technische Universit¨at Berlin, Berlin, Germany. <sup>2</sup>Speech and Language Technology Lab, German Research Center for Artificial Intelligence (DFKI), Berlin, Germany.

\*Corresponding author(s). E-mail(s): dinh-nam.pham@campus.tu-berlin.de; sebastian.moeller@tu-berlin.de; Contributing authors: shushen.manakhimova@dfki.de; vivien.macketanz@dfki.de;

## Abstract

The rapid advancement of Natural Language Generation (NLG) has made the reliable evaluation of generated text increasingly critical, as these systems, such as large language models (LLMs), are now widely deployed in real-world applications. However, traditional automatic metrics fail to capture the multifaceted nature of perceived quality. In this paper, we introduce TextQ-German, a novel dataset suite for human-centered evaluation of German NLG from a Quality of Experience (QoE) perspective, covering automatic text summarization and machine translation. Through crowdsourcing studies with German speakers, we collect human quality ratings and identify relevant perceptual quality dimensions for each task. We develop automatic QoE prediction models, including transformer-based, linguistic feature-based, and hybrid approaches. Hybrid models outperform pure transformer baselines in almost all experimental settings, while linguistic features alone can approach the performance of fine-tuned lan guage models. The dataset is extended with LLM-generated outputs annotated with overall QoE scores. Final validation on held-out sets indicates generalization to unseen data. Our work contributes a publicly accessible resource for NLG evaluation and baselines for automatic QoE prediction, providing a foundation for developing NLG systems that better align with human quality perception.

Keywords: Quality of Experience, Natural Language Generation, Text Quality, Readability, Text Generation, Machine Translation

## 1 Introduction

In recent years, automatically generated text has become ubiquitous across a wide range of domains, from news summarization and customer service chatbots to text translation and personalized content creation. This rapid development has been driven by significant advances in Natural Language Generation (NLG), the subfield of artificial intelligence concerned with the automatic production of human-readable text. The scope of NLG is broad, with common tasks including machine translation, automatic text summarization, question-answering, and content generation, among others. These technologies promise to enhance communication, increase accessibility, and automate processes in unprecedented ways.

However, the quality of outputs from NLG systems can difer greatly, making their reliable evaluation a critical challenge. Evaluation of NLG most commonly relies on automatic metrics such as BLEU (Papineni et al., 2002) or ROUGE (Lin, 2004), which measure lexical overlap with reference texts. While these methods require no training of models and are thus convenient and computationally eficient, these metrics are known to sufer from significant limitations. As they neglect sentence structure and semantic context, they are prone to significant shortcomings in tasks that require nuanced understanding and reasoning. Most importantly, these automatic metrics correlate poorly with human judgment (Belz & Reiter, 2006; Novikova et al., 2017). The reliance on surface-level comparisons may miss the fundamental objective of whether the generated text is useful and satisfactory for its intended audience and the users of NLG systems.

As NLG systems increasingly serve as interfaces for real-world users, we argue that their ultimate success hinges not on a technical score but on the Quality of Experience (QoE) they provide. QoE, a concept well-established in the domain of telecommunications and multimedia, is defined as the degree of user delight or annoyance with an application or service. It encompasses the user’s entire subjective perception, including expectations, context, and emotional response. Hence, the quality of a machinegenerated text can be evaluated from the perspective of its end-user in addition to objective metrics.

We therefore aim to assess the QoE of NLG specifically for the German language, a domain that has received comparatively less attention than the English language in NLG evaluation research. To this end, we provide two key contributions: (1) a novel QoE dataset of human-annotated ratings for machine-generated outputs and (2) a suite of automatic prediction models trained on these ratings to serve as robust QoE evaluators for unseen text. We focus on two established NLG tasks, Automatic Text Summarization (ATS) and Machine Translation (MT), and investigate QoE at two complementary levels: fine-grained perceptual quality dimensions and overall perceived quality. A core part of our work involves identifying the perceptual dimensions most relevant to users and developing meaningful ways to evaluate them. Rather than prespecifying a fixed set of quality criteria, we derive task-specific dimensions from user ratings and investigate how these dimensions can be modeled automatically. Moreover, rather than relying exclusively on learned text representations, we additionally investigate which measurable linguistic properties are associated with perceived quality and whether these features can complement transformer-based representations. By grounding our evaluation in a user-centric framework, we move beyond traditional metrics and ofer a more human-oriented benchmark for the assessment of modern NLG systems.

This article consolidates and substantially extends two previous publications from the same research project. Section 3.1 covers Manakhimova et al. (2025), which introduced the original ATS and MT corpora and identified and validated their perceptual quality dimensions, while the initial dimension-level prediction experiments of Section 4.1.1 build on Pham et al. (2025). The remaining dataset subsets, model approaches, validation experiments, and linguistic features are introduced in the present work. The datasets and additional data are publicly accessible at https://github.com/DFKI-NLP/TextQ/.

## 2 Related Work

## Human-centered evaluation of NLG

The evaluation of Natural Language Generation (NLG) has traditionally relied on a combination of human judgments and automatic metrics. Reference-based metrics such as BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) provide eficient and reproducible measures. However, their ability to represent human-perceived quality is limited. In particular, automatic word-overlap metrics have been shown to correlate weakly or inconsistently with human ratings of generated text (Belz & Reiter, 2006; Novikova et al., 2017). On the other hand, learned metrics incorporate contextual representations or are trained directly on human judgments, as exemplified by BERTScore (Zhang et al., 2020) and COMET (Rei et al., 2020).

Human evaluation remains important because text quality is inherently multifaceted. Depending on the NLG task, assessments consider properties such as fluency, adequacy, coherence, consistency, relevance, and overall quality. At the same time, the terminology and experimental procedures used for these dimensions vary considerably across studies. van der Lee et al. (2019) provide recommendations for conducting human evaluations of generated text, while Howcroft et al. (2020) identify substantial inconsistencies in the definitions of quality criteria used throughout the NLG literature. Similar concerns persist in diferent tasks. For automatic text summarization, SummEval (Fabbri et al., 2021) evaluates summaries along coherence, consistency, fluency, and relevance and demonstrates that automatic metrics capture these dimensions to diferent degrees. For machine translation, the choice of annotators and evaluation protocol can substantially afect human system rankings (Freitag et al., 2021). More recently, LLM-based evaluators such as G-Eval (Liu et al., 2023) have been proposed to approximate multidimensional human evaluation. These developments motivate evaluation approaches that are explicitly grounded in human perception rather than solely in task-specific automatic scores.

## Quality of Experience and subjective quality prediction

QoE ofers a user-centered perspective as it describes the subjective quality experienced by a user and is influenced not only by technical properties of a system but also by the content, context, and characteristics and expectations of the user (International

Telecommunication Union, 2017; M¨oller & Raake, 2014). Consequently, subjective experiments require the quantification of QoE assessment, often resulting in Mean Opinion Scores (MOS) or ratings of multiple perceptual quality dimensions. These subjective judgments can serve as targets for models that estimate perceived quality automatically. This paradigm is well established for speech, audio, image, and video services. For instance, subjective video-quality methodologies are standardized in ITU T P.910, while objective models are designed to predict human-perceived quality from measurable characteristics (International Telecommunication Union, 2023).

Only recently, however, has this same perspective been turned toward text. Naderi, Mohtaj, Karan, and M¨oller (2019) formulated German text readability as a prediction problem, demonstrating that subjective perceptions of textual complexity can be modeled automatically. Moving beyond a single property such as readability to capture the overall perception of machine-generated text, our dataset includes human ratings for both overall QoE and individual attributes, enabling us to empirically derive the perceptual dimensions most relevant to ATS and MT.

## German text quality and automatic quality estimation

Related research on German text has primarily addressed readability and complexity. Subjective German text complexity datasets have enabled the development of models that predict readers’ perceived dificulty from linguistic characteristics and contextual representations (Naderi, Mohtaj, Ensikat, & M¨oller, 2019; Seife et al., 2022). The GermEval 2022 shared task further established German text complexity prediction as a regression problem, with participating approaches combining transformer representations and manually derived textual statistics (e.g., average sentence length) (Ansch¨utz & Groh, 2022; Mohtaj et al., 2022). These studies demonstrate that measurable linguistic properties, such as lexical and syntactic features, can provide useful information about how readers perceive a text. However, these works focus on a single aspect of perception, namely text complexity or readability, rather than on the broader set of factors that shape the overall quality experienced by users. In contrast, our work considers a broader spectrum of perceived quality across multiple QoE dimensions and addresses a gap in the automatic evaluation of perceived quality for German NLG. Rather than predicting a single predefined quality criterion, we model multiple empirically derived QoE dimensions together with overall perceived quality across both ATS and MT. We further combine transformer-based representations with a broader set of interpretable linguistic features and evaluate the resulting models on independently collected data spanning both conventional and LLM-based NLG systems. In this way, our work extends existing German text evaluation from individual aspects such as readability or complexity toward a broader, user-centered prediction of perceived NLG quality.

## 3 Dataset Resource

QoE assessment of NLG requires data that reliably captures how users actually perceive generated text. By quantifying the satisfaction of past users, we can develop models that predict the QoE of future samples. To this end, we created and released

TextQ-German, a dataset suite for Quality of Experience assessment of German natural language generation.

The resource is designed for three main purposes. First, it supports the analysis of perceptual quality dimensions in generated German text, which we identified through user experiments. Second, it enables the development and comparison of automatic QoE prediction models, including linguistic-feature-based models, transformer-based models, and hybrid approaches. Third, it provides LLM-based extensions and final validation sets to test whether QoE predictors remain robust beyond non-LLM-based neural text generation.

The resource covers two established NLG tasks: automatic text summarization (ATS) and machine translation (MT). It comprises six subsets: the initial ATS and MT corpora with dimension-level QoE ratings, LLM-generated extensions created after initial prediction experiments, and final validation sets for external model evaluation. Together, these subsets allow us to study both fine-grained perceptual dimensions and overall quality scores across classical and LLM-based generation scenarios. Specifically, TextQ-ATS and TextQ-MT contain the original ATS and MT corpora with dimension-level QoE ratings. TextQ-ATS-LLM and TextQ-MT-LLM extend these corpora with outputs generated by large language models and annotated with an overall QoE score. Finally, TextQ-ATS-Val and TextQ-MT-Val serve as validation sets for assessing model generalization on previously unseen data.

Section 3.1 describes the collection and creation of TextQ-ATS and TextQ-MT, including data sources, the identified QoE quality dimensions, and the curation process. Section 3.2 then explains the LLM-based datasets TextQ-ATS-LLM and TextQ-MT-LLM, which were additionally used after the initial experiments based on TextQ-ATS and TextQ-MT. Section 3.3 introduces the two held-out final validation sets, TextQ-ATS-Val and TextQ-MT-Val. Finally, Table 6 provides an overview of all subsets of TextQ-German.

## 3.1 ATS and MT Corpora

To determine which specific perceptual features, here called quality dimensions, drive users’ QoE when judging MT and ATS outputs, we performed experimental crowdsourcing studies with German-speaking participants, allowing us to both identify, validate, and quantify these dimensions.

## 3.1.1 Text Sources

The ATS and MT corpora (TextQ-ATS and TextQ-MT) draw text from distinct sources with a broad range of vocabulary and topics. For automatic text summarization, German source texts were obtained from the GeWiki corpus (Frefel, 2020), which was also used in GermEval 2020 Task 3: 2nd German Text Summarization Challenge (Frefel et al., 2020). These summaries were kindly made available to us by the organizers for use in this research. To ensure a broad range of quality levels, we additionally generated summaries internally. Summaries from both sources were produced using a range of extractive and abstractive approaches. Extractive methods included Lead-3 (Dohare et al., 2017) and TextRank (Mihalcea & Tarau, 2004) while abstractive methods included

Pointer-Generator (See et al., 2017), Transformer (Vaswani et al., 2017), Convolutional Self-Attention Networks (Yang et al., 2019), and BERT-Transformer (Devlin et al., 2019). This resulted in diverse outputs that vary in fluency, content coverage, and coherence. The selected ATS outputs span summaries of up to five sentences.

For machine translation, we sampled English-German translations from the News Translation Task of the Conference on Machine Translation 2019 (WMT19) (Barrault et al., 2019), deliberately sampling outputs from top-ranked, mid-ranked, and bottomranked systems to capture a wide quality variety.

To ensure a diverse range of quality levels and error patterns across both NLG tasks, we performed a simple error-type annotation on the summaries and translation outputs, conducted by several linguists. Based on this, we selectively included data points exhibiting various error categories (e.g., morphological, syntactic, lexical) and severities. For ATS, we also made sure to include summaries of diferent lengths with up to five sentences. This resulted in well-balanced corpora for both tasks, containing samples with varying levels of correctness and length. Both subsets were subsequently used in crowdsourcing experiments to collect human QoE ratings, as described next.

## 3.1.2 Quality Dimension Identification

To identify the perceptual quality dimensions that shape users’ QoE for MT and ATS outputs, we ran two separate crowdsourcing studies, one for each NLG task. Both studies relied on Semantic Diferential (SD) scaling, a well-established technique that uses bipolar adjective pairs as scale endpoints to capture subjective perceptions (Osgood et al., 1957).

## Polar Adjective Pairs

Before launching the main experiments, we assembled two custom sets of polar adjective pairs since the linguistic properties of translations and summaries difer substantially. Each pair comprised a positive adjective and its direct negative counterpart (e.g., simple–complicated). Drawing on our earlier error type analyses of the MT and ATS corpora, we compiled a list of adjective pairs that reflected the most common error patterns in each text type. Linguists assisted in the selection to ensure the pairs captured relevant linguistic distinctions. The initial pool contained roughly 40 adjective pairs per text task, with partial overlap between the two sets. We aimed to cover as many linguistic facets of the texts as possible.

We then ran a small-scale pre-study, asking participants to evaluate texts from our corpora using all candidate adjective pairs. In a second step, participants rated how helpful they found each pair for the evaluation task. Based on these results, we reduced each set to approximately 20 adjective pairs, which were subsequently used in the main crowdsourcing studies.

## Experiment Setup

In the main experiments, participants were shown translations or summaries and instructed to assess the language quality using the adjective sets. For each bipolar pair, the two adjectives represented the endpoints on a 7-point Likert scale from 0 to 6. Participants were told to disregard text content as much as possible, acknowledging that content and language cannot always be fully separated, and were not informed that the texts were machine-generated. They were only told that texts might contain errors.

The survey flow consisted of four stages: (1) An introductory section explaining the setup and providing an example of a bipolar adjective pair. (2) A practice text, which served two purposes: familiarizing participants with the task and checking attention. The practice texts were clearly high or low in quality. If a participant’s rating did not match the expected direction, they were asked to reconsider. This created a mild ”being observed” efect, which was shown to improve performance and reliability of participants’ responses in crowdsourcing studies (Naderi et al., 2015). (3) The main evaluation stage, where each text appeared separately and had to be rated on all adjective pairs before advancing. A slider was used to select an integer between 0 and 6 per pair. Each participant rated three texts. (4) A final optional section to provide feedback on the survey.

The study ran on the Crowdee platform (Naderi et al., 2014), a mobile-friendly crowdsourcing system with full German localization. Eligible participants were selfidentified native German speakers residing in the DACH region (Germany, Austria, Switzerland). They could take part up to five times, receiving diferent survey versions on repeat participation. Expected completion time was approximately 10 minutes. We generated around 15 survey variants per text task, covering 45 translations and 40 summaries in total. These item counts refer specifically to the quality-dimension identification experiment and do not represent the final corpus sizes. The subsequent quantification study in Section 3.1.3 evaluated additional text items. A total of 350 participants completed the MT survey, and 425 completed the ATS survey for the quality-dimension identification.

## Results and Analysis

To mitigate low-quality responses common in crowdsourcing (Naderi et al., 2015), we performed a data cleaning procedure. We discarded ratings from participants who completed the survey in 240 seconds or less (40% of the expected duration) or who gave identical slider values for every adjective pair of a given sentence, as these behaviors suggested insuficient efort. We further computed the Inconsistency Score (IS) (Naderi, 2018) based on two repeated adjective pairs per sentence, allowing us to filter out outlier responses with high variance. After cleaning, each test item retained approximately 10 to 20 valid ratings.

We then performed Exploratory Factor Analysis (EFA) separately for each text type using SPSS (IBM Corp., 2021), employing Maximum Likelihood extraction and PROMAX rotation with Kaiser Normalization to allow non-orthogonal (correlated) dimensions. Several adjective pairs exhibited low communalities or cross-loadings with a diference below 0.2, indicating they were either semantically redundant or insuficiently specific. Thus, we removed them to balance statistical fit with interpretability (W¨altermann et al., 2010). After this reduction, a four-factor structure with eight adjective pairs emerged for each text type. Goodness-of-fit was satisfactory: for MT, Pearson’s chi-squared test yielded $p = 0 . 3 6 \ ( \chi ^ { 2 } = 2 . 0 6 , \mathrm { d f } = 2 )$ , while for ATS, it yielded $p = 0 . 6 3 \ ( \chi ^ { 2 } = 0 . 9 2 , \mathrm { d f } = 2 )$

Table 1 Loadings of adjective pairs (English translations) on factors and percentage of explained variance for Machine Translation. Adapted from Manakhimova et al. (2025)
<table><tr><td>Adjective pair</td><td>F1</td><td>F2</td><td>F3</td><td>F4</td></tr><tr><td>unambiguous-ambiguous precise-vague</td><td>.757</td><td></td><td></td><td></td></tr><tr><td>complete-incomplete</td><td>.947 .822</td><td></td><td></td><td></td></tr><tr><td>clear-chaotic direct-ponderous</td><td>.580</td><td>.806</td><td></td><td></td></tr><tr><td>simple-complicated</td><td></td><td>.923</td><td></td><td></td></tr><tr><td>grammatical-ungrammatical</td><td></td><td></td><td>.958</td><td></td></tr><tr><td>neat-confusing</td><td></td><td></td><td></td><td>.915</td></tr><tr><td>% of variance</td><td>53.2</td><td>8.4</td><td>10.5</td><td>8.0</td></tr></table>

Table 2 Loadings of adjective pairs (English translations) on factors and percentage of explained variance for Automatic Text Summarization. Adapted from Manakhimova et al. (2025)
<table><tr><td>Adjective pair</td><td>F1</td><td>F2</td><td>F3</td><td>F4</td></tr><tr><td>precise-vague complete-incomplete</td><td>.884</td><td></td><td></td><td></td></tr><tr><td>coherent-incoherent</td><td>.942 .844</td><td></td><td></td><td></td></tr><tr><td>logical-illogical simple-complicated</td><td>.796</td><td>1.002</td><td></td><td></td></tr><tr><td>straightforward-complex</td><td></td><td>.783</td><td></td><td></td></tr><tr><td>unambiguous-ambiguous</td><td></td><td></td><td>.729</td><td></td></tr><tr><td>predictable-unpredictable % of variance</td><td>54.4</td><td>14.3</td><td>10.6</td><td>.704 2.6</td></tr></table>

Table 1 and Table 2 illustrate the distribution of the adjective pairs on the four factors and the explained percentage of variance for MT and ATS. Across both text types, factors F1 to F4 reflect distinct quality dimensions based on the adjective pair(s) loading onto each. For MT, F1’s four loading attributes indicate Precision, F2’s two loading attributes indicate Complexity, and the single attributes on F3 and F4 indicate Grammaticality and Transparency, respectively. For ATS, F1’s four loading attributes indicate Linguistic Logic, F2’s two loading attributes indicate Complexity, and the single attributes on F3 and F4 indicate Clarity and Predictability, respectively. Complexity is the only dimension that emerges in both tasks, with simple–complicated as a shared indicator. Notably, the dominant F1 factors of the two tasks share two adjective pairs (precise–vague, complete–incomplete) but diverge in their remaining indicators: for MT, F1 additionally captures ambiguity and clarity of phrasing, whereas for ATS it is characterized by coherence and logical consistency (coherent–incoherent, logical–illogical). We therefore retain distinct labels, as the factors reflect task-specific interpretations of accuracy: sentence-level fidelity for translation versus discourse-level cohesion for summarization. Full lists of all polar adjective pairs, including those removed during factor analysis and the reduced sets retained for subsequent quantification, are provided in Tables 23 and 24 in the Appendix. We give an overview of the identified quality dimensions for MT and ATS, as well as typical traits we associate with them in Table 3.

Table 3 Quality dimensions and their characteristics. Adapted from Manakhimova et al. (2025)
<table><tr><td>Factor</td><td>Quality dimension</td><td>Characteristics</td></tr><tr><td>MT-F1</td><td>Precision</td><td>clear and complete phrasing unambiguous meaning</td></tr><tr><td>MT &amp; ATS–F2</td><td>Complexity</td><td>easily comprehensible not circumlocutory</td></tr><tr><td>MT-F3</td><td>Grammaticality</td><td>correct spelling and punctuation no missing words</td></tr><tr><td>MT-F4</td><td>Transparency</td><td>clear and coherent reasonably structured</td></tr><tr><td>ATS-F1</td><td>Linguistic Logic</td><td>accurate phrasing cohesive</td></tr><tr><td>ATS-F3</td><td>Clarity</td><td>direct and clear language content easily understandable</td></tr><tr><td>ATS-F4</td><td>Predictability</td><td>logical and expected structure methodical and coherent</td></tr></table>

## 3.1.3 Quality Dimension Quantification

Following the identification of the quality dimensions, we conducted a second crowdsourcing study to validate the reduced measurement scheme and test whether a reduced set of adjective pairs could reliably capture the same underlying constructs. For each factor revealed by the EFA, we selected the adjective pair with the highest factor loading as a representative indicator of that dimension. This yielded four adjective pairs per text type.

We then correlated the outcomes of the two experiments per text type to quantify the quality dimensions and assess the agreement between the full and reduced measurement schemes. In this validation experiment, 425 participants rated MT outputs, and 120 participants rated ATS outputs. The discrepancy in participant counts again stemmed from unusable ratings, which required multiple iterations of the survey to accumulate suficient valid responses.

The correlation analysis followed the same procedure for both text types. We first applied Grubbs’s test (Grubbs, 1950) to detect outliers, excluding any text sample point that emerged as a significant outlier on two or more of the four factors. After this filtering step, Spearman correlation coeficients consistently fell around 0.8 for both text types, providing initial evidence of strong alignment between the two experimental rounds.

To determine whether the two sets of ratings difered significantly, we first conducted the Jarque–Bera test (Jarque & Bera, 1980), which did not indicate significant departures from normality across the factors for either text type. We then performed Levene’s test (Levene, 1960) to assess the equality of variances. For all factors except MT’s F2 (Complexity) and ATS’s F4 (Predictability), Levene’s test indicated homogeneous variances. Consequently, we applied a one-factor ANOVA (Fisher, 1925) to those factors. For the remaining two factors that violated the homogeneity assumption, we used Welch’s t-test (Welch, 1947) instead.

For ATS, the analysis showed no statistically significant diference between the identification and quantification experiments for any factor. For MT, no significant diferences emerged for any factor except F2 (Complexity). However, the observed diference for MT’s Complexity was only 0.5 points on a 7-point scale, and the correlation between the two experimental rounds on this factor remained high. Given that we cannot definitively attribute this small discrepancy to either the reduced number of adjective pairs or the natural variation between diferent crowdsourcing participant pools, we consider the diference as acceptable.

These findings demonstrate that the condensed set of four adjective pairs efectively captures the essential aspects of each quality dimension while remaining consistent across the experimental rounds. A key benefit of validating these quality dimensions is that future studies can adopt simpler, more eficient evaluation designs that impose less cognitive load on participants while still yielding detailed insights. The strong correlations between the reduced adjective-pair sets and their corresponding original factor scores indicate that the condensed evaluation preserves the underlying quality dimensions for both MT and ATS. Nevertheless, the slight diference observed for MT’s Complexity factor (F2) suggests that complexity is not interpreted the same for diferent NLG tasks. In translations, simpler language supports fluency, whereas in summaries, structural and conceptual depth help convey concise information. This points to a broader need for making quality evaluation in NLP more task-specific.

Based on the crowdsourcing studies, we curated the evaluated text items and their human QoE dimension ratings into two datasets, TextQ-ATS and TextQ-MT. The former comprises 91 text samples, the latter contains 106. Prioritizing reliability over quantity, each sample received ratings from 10 to 20 annotators, and we averaged these per item. Tables 4 and 5 provide example excerpts from TextQ-ATS and TextQ-MT, respectively. These subsets can be used to develop QoE prediction models.

## 3.2 LLM-Based Corpora Extensions

After initial experiments on TextQ-ATS and TextQ-MT as described in Section 4.3, we conducted crowdsourcing runs again to extend the datasets. Following the recent rapid trends in NLG and NLP technology, the text samples were now generated using LLMs. The source texts were drawn from the same datasets as the original corpora (Section 3.1), namely the GeWiki corpus for ATS and the WMT19 English-German test set for MT; however, the samples used for this extension were diferent from the original corpus. In addition to the four quality dimensions, an overall QoE score was collected. The extension of the corpus was generated using a mix of commercial APIbased models and locally hosted open-weight models. The OpenAI models (GPT-4o and GPT-3.5 Turbo) were accessed through the Chat Completions API, while the open-weight models were hosted locally through Ollama. The full prompts used for each model are listed in Appendix B.

Table 4 Excerpt from TextQ-ATS illustrating its structure, with mean ratings per test item and quality dimension
<table><tr><td>Text Sample führt</td><td>Logic</td><td></td><td>Complexity 3.875</td><td>Clarity 3.9375</td><td>Predictability</td></tr><tr><td>Die Bundesstrasse 75 von Travemünde bis in die Hansestadt Lübeck und ist eine der beiden Bundesstrassen Hamburg und Lübeck. Der Name Meschete war ursprünglich</td><td>Lübeck- nur</td><td>4.0625 2.2703</td><td>1.7297</td><td>3.5000</td><td>2.0541</td></tr><tr><td>ein geographischer Name, der sowohl den turkvölkischen Einwanderern, der Region und der heutigen Provinz den Namen gab. Bis stammt von der alten georgischen Region Mzcheta. Der Siedlungsschwerpunkt der Mescheten war einst in Usbekistan, den Vereinigten Staaten und der Ukraine. Und so listet beispielsweise das &quot;Met- zler Lexikon Sprache&quot; die Sprache der Mescheten unter dem Sammelbegriff&quot;muslimische Georgier&quot; zusammen. Weltweit leben heute etwa 40.000 in der Türkei.</td><td></td><td></td><td></td><td>2.5000</td><td></td></tr><tr><td>Viktor Gustav von Strauss und Torney war ein österreichischer Dichter, Kirchenlieddichter und Zeichner. Berühmtheit erlangte er als erster Botschafter der Chinesischen Volksrepublik China.</td><td></td><td>3.5294</td><td>3.544</td><td>3.8823</td><td>2.7353</td></tr></table>

Table 5 Excerpt from TextQ-MT illustrating its structure, with mean ratings per test item and quality dimension
<table><tr><td>Text Sample</td><td>Precision</td><td>Complexity</td><td>Transparency</td><td>Grammaticality</td></tr><tr><td>Die Senatorin von Massachusetts, Eliz- abeth Warren, sagte am Samstag, sie werde nach den Zwischenwahlen einen &quot;harten Blick&quot; darauf werfen, für das</td><td>2.1765</td><td>3.2941</td><td>2.8235</td><td>2.4706</td></tr><tr><td>Präsidentenamt zu kandidieren. Die Labour Party wäre nicht die erste, die eine solche Idee befürwortet, mit der grünen Party, die eine vier-Tage- Arbeitswoche während ihrer 2017 allge- meinen Wahlkampagne geschworen hat.</td><td>3.3947</td><td>4.1579</td><td>4.1316</td><td>3.5526</td></tr><tr><td>Angesichts der Erde, die schütteln, dass Indonesien ständig übersteht, bleibt das Land kläglich unvorbereitet für den Zorn der Natur.</td><td>3.5333</td><td>3.8667</td><td>4.0000</td><td>4.2000</td></tr></table>

Translations were produced with four models: the two OpenAI models, GPT-4o and GPT-3.5 Turbo, and two open-weight models, Llama 3.2 (3B) and StableLM 2 (1.6B). All four produced translations for the complete set of source items, using a single prompt to translate the input into German. The maximum output length was capped at 300 tokens to keep the generated translations comparable in length to those in the original corpus and to avoid overly long texts that would burden crowdworkers during rating. A temperature of 0.2 was applied uniformly across all models; lower or higher temperatures caused some models to produce mixed German-English output, and a single setting was kept constant across models for consistency.

Summaries were generated with a broader set of models. From OpenAI, GPT-4o was used in a length-constrained configuration (referred to as gpt-4o short), with a temperature of 0.0, a maximum of 200 output tokens, and a prompt that explic itly restricted the summary length. The open-weight summarization models comprised DeepSeek-R1 (1.5B), StableLM 2 (1.6B), Llama 3.2 (3B), SmolLM2-German-Instruct (360M, Q8), and SauerkrautLM-7B-v1 (GGUF, Q2 K), all hosted locally through Ollama. These locally hosted models used a common German summarization prompt, a temperature of 0.5, and a maximum output length of 540 tokens; source texts exceeding 2,800 characters were truncated to their first 2,800 characters before insertion. As an instruction-tuned model, SauerkrautLM-7B-v1 instead used the Alpaca-style instruction format specified in its prompt template.

The generated outputs were manually annotated for quality and error types following the same general procedure used for the original corpora. Based on this annotation, outputs were selected to cover a broad range of quality levels rather than being dominated by fluent, high-quality generations. For MT, all four models produced outputs for every source item, allowing balanced selection across systems and quality levels. For ATS, the open-weight models did not all complete every source item, so the final selection was balanced across quality categories as far as the available outputs allowed. The selected texts were then evaluated in additional crowdsourcing runs using the same four dimension-specific adjective pairs described in Section 3.1.

The overall quality score was collected as a separate rating. Alongside the dimension-specific adjective pairs, raters judged each text using an overall good–bad (gut–schlecht) item on the same 0–6 scale, where 0 represented bad quality, and 6 represented good quality.

We refer to these additional subsets as TextQ-ATS-LLM and TextQ-MT-LLM. TextQ-ATS-LLM and TextQ-MT-LLM each contain 77 rated text samples.

## 3.3 Final Validation

In the same fashion as for TextQ-ATS, TextQ-MT, TextQ-ATS-LLM, and TextQ-MT-LLM, we obtained additional samples for ATS and MT as held-out final validation sets, referred to as TextQ-ATS-Val and TextQ-MT-Val. These sets were drawn from the same generation efort as the LLM-based extensions: from the annotated output pool, we made a separate manual selection for validation. Source texts did not overlap with those used for any other TextQ-German subset, and selection was again balanced across the quality range. As a result, each validation set combines items produced by the LLMs described in Section 3.2 with items produced by the non-LLM neural systems introduced in Section 3.1, yielding a heterogeneous mix of generation systems. For TextQ-ATS-Val, the non-LLM items were mostly generated by two of these systems, BERT-Transformer and Convolutional Self-Attention Networks; for TextQ-MT-Val, the non-LLM items were drawn from the WMT19 submissions. TextQ-ATS-Val comprises 77 items and TextQ-MT-Val 76, each split approximately evenly between LLM-generated and non-LLM items. All items are annotated with the four quality dimension ratings and the overall QoE score. Each item carries the same labels as the LLM-based subsets—the four quality dimensions and the overall QoE score, collected in the same way. These two subsets serve as held-out validation sets for the final evaluation of the best QoE predictors developed for ATS and MT.

Table 6 Overview of the TextQ-German dataset suite. ATS denotes automatic text summarization, MT denotes machine translation. All labels are represented as mean rating on a 0–6 scale.
<table><tr><td>Subset</td><td>Task</td><td>Generation</td><td>Size</td><td>QoE labels</td><td></td><td></td><td>Purpose</td></tr><tr><td>TextQ-ATS</td><td>ATS</td><td>Non-LLM</td><td>91</td><td>4 dimensions</td><td></td><td></td><td>Dimension-level QoE pre- diction</td></tr><tr><td>TextQ-MT</td><td>MT</td><td>Non-LLM</td><td>106</td><td>4 dimensions</td><td></td><td></td><td>Dimension-level QoE pre- diction</td></tr><tr><td>TextQ-ATS-LLM</td><td>ATS</td><td>LLM</td><td>77</td><td>Overall score dimensions</td><td>and</td><td>4</td><td>Overall QoE prediction and dimension-level QoE data extension</td></tr><tr><td>TextQ-MT-LLM</td><td>MT</td><td>LLM</td><td>77</td><td>Overall score dimensions</td><td>and</td><td>4</td><td>Overall QoE prediction and dimension-level QoE data extension</td></tr><tr><td>TextQ-ATS-Val</td><td>ATS</td><td>LLM and Non-LLM</td><td>77</td><td>Overall score dimensions</td><td>and</td><td>4</td><td>Held-out final validation</td></tr><tr><td>TextQ-MT-Val</td><td>MT</td><td>LLM and Non-LLM</td><td>76</td><td>Overall score dimensions</td><td>and</td><td>4</td><td>Held-out final validation</td></tr></table>

Table 6 gives an overview of the datasets. The datasets of TextQ-German can be accessed at https://github.com/DFKI-NLP/TextQ/. We make them publicly available under a CC BY-NC 4.0 license for non-commercial purposes.

## 4 Methodology

The TextQ-German datasets feature scaled human ratings of text samples, ofering a valuable resource for training and developing models that predict ratings for new, unseen samples, thus facilitating automatic QoE assessment. This approach of training models on human scores and subsequently deploying them as predictors of subjective perceptual quality has been successfully applied beyond text to other media types, including video (Li et al., 2019), image (Ma et al., 2025), and speech (Wang et al., 2025). Following this, we train and evaluate models on our TextQ-German datasets for machine-generated text (MT and ATS). This section details the models and experiments that we conducted to this end.

We consider two levels of prediction. The first is dimension-level QoE prediction, where models estimate each of the perceptual dimensions identified in the crowdsourcing studies. The second is overall QoE prediction, where models estimate a single quality score for a generated text.

The experimental design follows the structure of TextQ-German. We first use the initial corpora, TextQ-ATS and TextQ-MT, to study dimension-level QoE prediction. For ATS, the target dimensions are linguistic logic, complexity, clarity, and predictability. For MT, the target dimensions are precision, complexity, transparency, and grammaticality. These experiments test whether fine-grained perceptual quality dimensions can be predicted from textual information. We then move to the overall QoE prediction. This setting reflects practical evaluation scenarios in which a single quality estimate is needed. For this purpose, we use the subsets that contain overall QoE ratings, namely TextQ-ATS-LLM and TextQ-MT-LLM.

The experiments are organized into four stages. First, we benchmark models for predicting the four QoE dimensions of ATS and MT. Second, we train models for predicting the overall QoE score. Third, we extend the experiments to also cover LLM-generated data. Finally, we evaluate the best-performing configurations on the held-out validation sets, TextQ-ATS-Val and TextQ-MT-Val. These final validation sets are not used for feature selection, hyperparameter tuning, checkpoint selection, or model selection.

All prediction tasks are formulated as regression problems. Individual participants provide discrete ratings on a 0–6 scale, but the prediction targets are item-level Mean Opinion Scores (MOS), obtained by averaging the ratings across annotators. These aggregated scores are therefore continuous-valued. For dimension-level prediction, we consider both single-target settings, where one model is trained for each quality dimension, and multi-target settings, where one model predicts all dimensions jointly. For overall QoE prediction, models output a single continuous score.

## 4.1 Models

In this section, we describe the diferent models we used for the experiments to predict QoE.

## 4.1.1 Transformer-Based Models

The size of our datasets is relatively limited for training purposes, given that human annotation is a time- and resource-demanding process. Hence, as an initial baseline, we chose to fine-tune pre-trained German language models, taking advantage of their pre-existing knowledge of German text.

More specifically, we fine-tuned five pre-trained transformer-based models derived from Huggingface<sup>1</sup>:

• <sub>bert-base-german-uncased</sub> (Bayerische Staatsbibliothek, 2025b)

• <sub>bert-base-german-cased</sub> (Bayerische Staatsbibliothek, 2025a)

• <sub>gbert-base</sub> (Chan et al., 2020)

• <sub>gbert-large</sub> (Chan et al., 2020)

• <sub>gelectra-large</sub> (Chan et al., 2020)

For each model, we replaced the output layer with a linear layer. Its output dimensionality was set to 1 when the goal was to predict either a single QoE dimension or an overall score, and to 4 when the goal was to perform simultaneous multi-target regression across all four quality dimensions.

## 4.1.2 Linguistic Feature-Based Models

Beyond the task of rating prediction, gaining insight into which linguistic features or statistical text properties can be used to estimate text QoE is of significant value.

Thus, we employed feature selection (FS) methods to identify the linguistic features most relevant to predicting perceived text quality. We then assessed the performance of models that rely exclusively on these selected features.

To this end, we implemented 121 textual features in an efort to facilitate a comprehensive search for relevant features for the text quality dimensions. Every feature is a real-valued number computable for a given text, derived from the established literature and areas such as complexity and readability assessment. To compute these features, we used the spaCy library (Montani et al., 2023) with its German pipeline for tokenisation, part-of-speech tagging, and dependency parsing. For readability indices and lexical diversity, we relied on standard implementations from the textstat library (Bansal & Aggarwal, 2026), supplemented with custom functions for German-specific measures. The features can be categorized in the following feature types: readability features, lexical richness features, syntactic features, and morphological features. A full list of all features implemented can be found in Table 25 in the Appendix. In the following, we list some of the feature types with specific examples of the features we included in the experiments:

• Traditional Readability Features: Flesch Reading Ease, Average sentence length in words, Average word length in characters, Wiener Sachtextformel, Gunning fog index, SMOG index, Coleman-Liau index, . . .

• Lexical Richness Features: Dugast’s Uber Index, Type-Token-Ratio, Lexical density, Noun variation, Modifier variation, Adverb ratio, . . .

• Syntactic Features: Noun phrases per sentence count, Verb phrases per sentence count, . . .

• Inflectional Morphology of the Verb: Infinitive-verbs-to-verbs-ratio, Participleverbs-to-verbs-ratio, Past-tense-verbs-to-finite-verbs-ratio, third-person-to-finiteverbs-ratio, . . .

• Inflectional Morphology of the Noun: Genitive-nouns-to-nouns-ratio, accusative-nouns-to-nouns-ratio, dative-nouns-to-nouns-ratio, . . .

These features were computed for all text samples of the datasets for the experiments. Using these features and the QoE dimension ratings as labels, features can be selected to fit models, such as a linear regression model.

Given the number of 121 linguistic features that we implemented and can compute for a given text, the total number of possible feature subsets is $2 ^ { 1 2 1 } - 1$ , making a brute-force search for the optimal feature set computationally infeasible. However, selecting an optimal subset serves our two objectives: (a) identifying what linguistic features are relevant for assessing the text quality and (b) improving future ML models’ performance by minimizing redundancy within the set of selected features. Therefore, we approach this as the process of feature selection (FS) and utilize commonly used FS techniques that choose the most important features.

We implemented the FS methods Recursive Feature Elimination (RFE) and Sequential Feature Selection (SFS) using linear regression as the base model. This choice was motivated by the computational eficiency, high interpretability due to the linear equation as well as comparatively higher robustness to overfitting. As SFS requires a number n of features to select, we set n to 20 and used forward selection.

Additionally, the FS techniques Lasso and Elastic net were also employed, which combine feature selection with model training, complementing the wrapper methods with an embedded approach.

In summary, for any given text, we compute 121 real-valued linguistic features. Using the methods described above, we can fit a predictive model and obtain label predictions while simultaneously gaining insight into which individual features were selected.

## 4.1.3 Hybrid Models

Beyond using linguistic features or pre-trained language models in isolation, we propose hybrid models that combine the learned representations of language models with hand-crafted linguistic features. Jointly making use of both sources of information allows the model to benefit from the rich, context-aware representations captured by transformers while also incorporating explicitly defined linguistic knowledge, such as readability, lexical diversity, and syntactic complexity.

We constructed two types of hybrid models as follows.

• Hybrid Language Model: Using the training set, a FS method first selects a set of linguistic features. For each input text, we compute this feature set while simultaneously forwarding the text through a language model backbone to extract its [CLS] token embedding. The [CLS] token is a special token prepended to the input sequence in transformer models like BERT. Its final hidden state aggregates contextual information from the entire text into a fixed-size vector whose dimensionality is determined by the language-model backbone. This vector serves as a compact semantic representation of the input. Both the linguistic feature set and this embedding are then concatenated and passed to a linear layer, and the entire network is fine-tuned end-to-end. This architecture integrates linguistic indicators with neural representations in a late fusion manner, allowing the model to learn complementary patterns from both sources.

• Hybrid SVM: The training procedure follows two stages. Stage 1: A language model is fine-tuned on the training set in the same way as the models in Section 4.1.1. Simultaneously, a FS method selects a feature set for the training set. Stage 2: We extract the fine-tuned model’s [CLS] embeddings for each training sample, concatenate them with the corresponding selected linguistic features, and train a support vector machine (SVM) on this combined representation. During inference, the SVM predicts QoE ratings for new samples by taking as input both the computed features and the embeddings produced by the frozen, fine-tuned language model. The language model weights remain frozen throughout evaluation. For the support vector regression model, we used the radial basis function (RBF) kernel, and set C to 1.0 and epsilon to 0.1.

## 4.1.4 Multi-Task Models

As an alternative to single-task models, we explored a multi-task learning (MTL) approach, motivated by evidence that MTL can improve performance in multi-target regression settings (Mohtaj et al., 2023). Our MTL architecture employs a shared pre-trained language model backbone across all tasks. All parameters of the language model are shared between the tasks, while each task has a separate task-specific linear regression head. We extract the pooled [CLS] token embedding from this shared encoder and forward it to the corresponding task-specific regression head. Each head independently predicts one target variable. During training, the losses of all tasks are summed with equal weighting, such that gradients from each task update the same shared language-model parameters. The tasks can correspond either to the QoE dimensions or to the two datasets (ATS and MT). Thus, we adopt a hard parameter sharing approach.

Figure 1 illustrates an overview of the presented models.

## 4.2 Experimental Setup

For all experiments, we maintained consistent hyperparameters and data splits to ensure fair comparability. The chosen hyperparameters were as follows: learning rate $2 \times 1 0 ^ { - 5 }$ , mean squared error (MSE) loss function, 30 epochs, batch size 8, and AdamW optimizer with weight decay 0.01.

Due to the relatively small size of our datasets, we employed a 7-fold crossvalidation. Each dataset was randomly shufled and then split into seven equally sized folds, denoted $F = \{ f _ { 1 } , \ldots , f _ { 7 } \}$ . In each cross-validation round, fold $f _ { i }$ was designated as the test set, fold $f _ { i - 1 } \ \mathrm { ~ ( o r ~ } \ f _ { 7 }$ in the case of i = 1) as the validation set, and the remaining five folds as the training set. This procedure assigned each fold to both validation and test roles exactly once. By fixing random seeds, we guaranteed that all models trained on the same dataset were evaluated on the same fold partitions and sample orders.

We monitored the validation root mean squared error (RMSE) at the end of every epoch and retained the checkpoint achieving the lowest value for evaluation on the test fold. All reported metrics represent averages over the seven test folds.

In the experiments that used linguistic features, we standardized the features by subtracting the mean and scaling to unit variance based on the training data. The test fold was then normalized using the mean and standard deviation computed from the training folds to avoid data leakage. Importantly, feature selection was repeated independently in every cross-validation round and fitted exclusively on the corresponding training partition. Neither the validation fold nor the test fold was used to determine the selected linguistic features.

Unlike the neural models, the SVM does not require a validation set for early stopping. Therefore, to maximize the available training data, we fit the SVM on the union of the training and validation folds for each cross-validation round. The feature subset used by the SVM remained the subset selected from the training partition of that round.

## 4.3 Dimension-Level QoE Prediction

With the models and experimental settings established, we now present our experiments for assessing QoE dimensions in NLG. We organized our evaluation around four modeling choices.

![](images/f80a5da9aa8ead0b04ab8873fe3220ab07ee7aaa4a31231989d1ee8243cd3c7e.jpg)  
Fig. 1 Illustration of (a) transformer-based models, (b) hybrid language models, (c) hybrid SVM, and (d) multi-task architecture. Dashed outlines group components belonging to the same processing or training stage.

## Language Model Selection

We first fine-tuned and evaluated the pre-trained language models described in Section 4.1.1 on both TextQ-MT and TextQ-ATS. Each model was equipped with a regression head of four output units to predict all QoE dimensions simultaneously in a multi-target regression setup. Based on these experiments, we identified the bestperforming language model for each of the two text tasks, which served as the backbone for all subsequent experiments.

## Multi-Label Regression Approaches

With the best language model selected, we compared three regression approaches for dimension-level QoE prediction. Each text sample has four QoE dimension values to predict, making this a multi-label regression problem. First, we retained the multi-target regression approach used before, where a single model predicts all four dimensions at once. Second, we implemented single-target regression, instantiating a separate model for each of the four dimensions with an output size of one. Third, we applied the multi-task learning (MTL) architecture described in Section 4.1.4, treating each QoE dimension as an independent task with a shared backbone.

## Linguistic Feature Selection

Beyond neural approaches, we evaluated how well traditional linguistic features can predict QoE dimensions. We applied the feature selection methods described in Section 4.1.2 to both datasets, assessing the predictive performance achievable with carefully selected hand-crafted features alone.

## Hybrid Models

Finally, we combined the best-performing language model backbone with the most efective feature selection method to train and evaluate the proposed hybrid models on both MT and ATS. These experiments allowed us to determine whether linguistic features provide a complementary predictive signal beyond what the language model learns from raw text.

## 4.4 Overall QoE Prediction

Following the dimension-level experiments, we extended our evaluation to overall QoE prediction using the subsequently newly obtained LLM-generated text corpora (TextQ-MT-LLM and TextQ-ATS-LLM), each annotated with a single overall QoE score per sample.

We first re-evaluated all five pre-trained language models on both LLM datasets in a single-target regression setup, where each model outputs a single value per text sample.

Second, we investigated whether jointly training on both datasets could improve overall score prediction. Using the previously identified best-performing language model as the backbone, we implemented two strategies. The first strategy was multitask learning, where each dataset was treated as a separate task with a shared backbone. The second strategy was simply merging both datasets into a single corpus and fine-tuning the language model on this combined data, following the standard procedure.

Finally, based on the outcomes of these experiments, we selected the optimal language model and training strategy for the hybrid models in the context of overall QoE prediction for ATS and MT.

## 4.5 LLM-Based Extension

The newly collected LLM-generated data can also be combined with the original TextQ-ATS and TextQ-MT corpora to extend dimension-level QoE assessment. To this end, we compared three model variants using the previously selected language model backbone and feature selection method: (1) the pure language model, (2) the hybrid language model, and (3) the hybrid SVM, following the same experimental protocol as before.

## 4.6 Final Evaluation

To assess generalization to completely unseen data, we conducted a final evaluation using TextQ-MT-Val and TextQ-ATS-Val. These datasets were collected via crowdsourcing after the main experiments and serve as held-out final evaluation sets, having played no role in feature selection, hyperparameter tuning, checkpoint selection, or model selection. For each text task, we applied all seven model instances obtained from the 7-fold cross-validation procedure. Each instance performed inference on all samples of the corresponding final evaluation set, and we report the average performance across the seven instances. This setup provides an estimate of how well our models generalize to new, unseen text samples.

## 5 Results and Discussion

We report root mean squared error (RMSE), mean absolute error (MAE), and the coeficient of determination $( R ^ { 2 } )$ as metrics for regression. We primarily use RMSE for model selection because it assigns greater weight to larger prediction errors, which are particularly relevant when assessing deviations from human QoE ratings. The metrics are averaged across folds and dimensions.

## 5.1 Dimension-Level QoE Prediction

## Language Models

The fine-tuning performance of each pre-trained language model is summarized in Tables 7 and 8. For the ATS dataset, gbert-large yielded the lowest RMSE, whereas gelectra-large was the best-performing model on the MT dataset. Across all models, prediction errors were systematically higher for MT than for ATS. This pattern suggests that QoE prediction may be more demanding for machine translation. Nevertheless, we acknowledge that cross-dataset diferences in data quality or annotation reliability could partially explain this discrepancy.

Table 7 Performance of the fine-tuned language models on TextQ-ATS
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>bert-base-german-uncased</td><td>0.9751</td><td>0.8168</td><td>0.0800</td></tr><tr><td>bert-base-german-cased</td><td>0.9097</td><td>0.7458</td><td>0.2047</td></tr><tr><td> $\mathtt { g b e r t - b a s e }$ </td><td>0.9997</td><td>0.8258</td><td>0.0549</td></tr><tr><td> $\mathtt { g b e r t - l a r g e }$ </td><td>0.8438</td><td>0.6948</td><td>0.2927</td></tr><tr><td> $\mathtt { g e l e c t r a - l a r g e }$ </td><td>0.8693</td><td>0.7142</td><td>0.2657</td></tr></table>

Table 8 Performance of the fine-tuned language models on TextQ-MT
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>bert-base-german-uncased</td><td>1.1449</td><td>0.9479</td><td>0.1865</td></tr><tr><td>bert-base-german-cased</td><td>1.1375</td><td>0.9499</td><td>0.1795</td></tr><tr><td>gbert-base</td><td>1.1538</td><td>0.9403</td><td>0.1716</td></tr><tr><td>gbert-large</td><td>1.0932</td><td>0.9221</td><td>0.2026</td></tr><tr><td> $\mathtt { g e l e c t r a - l a r g e }$ </td><td>0.9853</td><td>0.8350</td><td>0.3899</td></tr></table>

Moreover, while both MAE and RMSE share the same unit as the target variables, MAE is consistently lower than RMSE, as expected from their respective formulations (Willmott & Matsuura, 2005). Based on the RMSE, we selected gbert-large for ATS and gelectra-large for MT as the backbones for all subsequent experiments.

## Multi-Label Regression Approaches

Table 9 Performance of the multi-label regression approaches on TextQ-ATS
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>Multi-Target (gbert-large)</td><td>0.8438</td><td>0.6948</td><td>0.2927</td></tr><tr><td>Single-Target (gbert-large)</td><td>0.9566</td><td>0.7735</td><td>0.1051</td></tr><tr><td>Multi-Task Learning (gbert-large)</td><td>1.0080</td><td>0.8389</td><td>-0.0152</td></tr></table>

Tables 9 and 10 show the results for multi-label regression. The multi-task model performed worst on both text tasks, though only slightly. We hypothesize that the limited amount of training data restricts the model’s ability to learn complex task interactions. In our current implementation, the multi-task architecture is essentially a regressor on top of the language model embeddings. Given more data, a larger or more sophisticated prediction head might better exploit the shared representations across tasks. For ATS, the multi-target regression model achieved the lowest error, while for MT, single-target regression performed the best. We therefore retained these configurations for the remaining experiments with hybrid models.

Table 10 Performance of the multi-label regression approaches on TextQ-MT
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>Multi-Target  $\scriptstyle { \overline { { \left( \mathbf { g e l e c t r a - l a r g e } \right) } } }$ </td><td>0.9853</td><td>0.8350</td><td>0.3899</td></tr><tr><td>Single-Target (gelectra-large)</td><td>0.9657</td><td>0.8149</td><td>0.4141</td></tr><tr><td>Multi-Task Learning (gelectra-large)</td><td>1.0430</td><td>0.8783</td><td>0.3034</td></tr></table>

Table 11 Performance of feature selection methods on TextQ-ATS, RMSE values and feature counts are averaged across the four ATS quality dimensions
<table><tr><td>Feature selection method</td><td>RMSE</td><td>Average number of selected features</td></tr><tr><td>RFE</td><td>1.0523</td><td>1.00</td></tr><tr><td>SFS</td><td>0.8923</td><td>20.00</td></tr><tr><td>Lasso</td><td>1.0068</td><td>6.00</td></tr><tr><td>ElasticNet</td><td>1.0482</td><td>7.50</td></tr></table>

Table 12 Performance of feature selection methods on TextQ-MT, RMSE values, and feature counts are averaged across the four MT quality dimensions
<table><tr><td>Feature selection method</td><td>RMSE</td><td>Average number of selected features</td></tr><tr><td>RFE</td><td>1.2613</td><td>7.00</td></tr><tr><td>SFS</td><td>1.0257</td><td>20.00</td></tr><tr><td>Lasso</td><td>1.1823</td><td>5.50</td></tr><tr><td>ElasticNet</td><td>1.1981</td><td>6.75</td></tr></table>

## Linguistic Feature Selection

Tables 11 and 12 describe the performance of the feature selection methods on QoE prediction as well as the average number of features selected out of all 121 possible linguistic features. Detailed per-dimension results and the specific selected features are available in the Appendix (Tables 26 and 27). Across all methods, the selected feature subsets were notably small. Recursive Feature Elimination (RFE) was particularly aggressive for ATS, selecting a single feature per dimension, while selecting larger subsets for some MT dimensions. Sequential Feature Selection (SFS) consistently yielded the lowest RMSE values across both datasets and all quality dimensions, with average RMSE scores of 0.8923 (ATS) and 1.0257 (MT). These results approach those of the best-performing language models, which achieved 0.8438 and 0.9853, respectively.

The result that a simple linear regression model fitted on a relatively small number of carefully selected linguistic features can rival fine-tuned transformer-based models is striking. This finding strongly motivates our proposed hybrid approach, which aims to combine the complementary strengths of complex neural representations and handcrafted linguistic features. Additionally, FS methods ofer a practical advantage: they provide transparency by explicitly revealing which linguistic features drive QoE prediction. Given its clear superiority, we use SFS as the feature selection method for all subsequent hybrid model experiments.

## Hybrid Models

Table 13 Performance of the hybrid models on TextQ-ATS
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>Multi-Target (gbert-large)</td><td>0.8438</td><td>0.6948</td><td>0.2927</td></tr><tr><td>Hybrid language model</td><td>0.8557</td><td>0.70275</td><td>0.3033</td></tr><tr><td>Hybrid SVM</td><td>0.8254</td><td>0.6645</td><td>0.3664</td></tr></table>

With the best-performing language model backbone, multi-label regression approach, and feature selection method established for each text task, we evaluated the proposed hybrid models. For both tasks, SFS selected 20 linguistic features, while the respective large language-model backbones produced 1,024-dimensional [CLS] embeddings. Consequently, the concatenated representation used by the hybrid models comprised 1,044 features. Tables 13 and 14 report the performance of the hybrid language model and hybrid SVM against the respective neural baselines.

Table 14 Performance of the hybrid models on TextQ-MT
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>Single-Target (gelectra-large)</td><td>0.9657</td><td>0.8149</td><td>0.4141</td></tr><tr><td>Hybrid language model</td><td>0.9127</td><td>0.7510</td><td>0.4720</td></tr><tr><td>Hybrid SVM</td><td>0.9584</td><td>0.8091</td><td>0.4233</td></tr></table>

On the ATS dataset, the hybrid SVM achieved the best overall performance, reducing RMSE from 0.8438 to 0.8254 and increasing $R ^ { 2 }$ from 0.2927 to 0.3664. On the MT dataset, the hybrid language model performed the best, lowering RMSE from 0.9657 to 0.9127 and raising $R ^ { 2 }$ from 0.4141 to 0.4720.

These results clearly demonstrate the merit of combining neural embeddings with carefully selected linguistic features. On both datasets, at least one hybrid model outperforms the pure transformer-based baseline, showing that linguistic features provide complementary predictive information not fully captured by the language model alone. Interestingly, the optimal hybrid architecture difers by text type. For ATS, which involves multi-sentence summaries, the SVM-based hybrid excels, perhaps because the feature set captures text-level properties such as lexical diversity and readability that are particularly salient for longer texts. For MT, where inputs are mostly single sentences, the neural hybrid performs better, suggesting that contextualized embeddings are especially powerful for sentence-level tasks. Overall, these findings validate that linguistic features and neural representations encode complementary information, and their combination yields measurable improvements in QoE prediction.

## 5.2 Overall QoE Prediction

## Language Models

After the dimension-level QoE prediction, we obtained the LLM-based datasets TextQ-ATS-LLM and TextQ-MT-LLM which contain an overall score as specified in Section 3.2, enabling the prediction of overall QoE.

Table 15 Performance of the fine-tuned language models for overall QoE prediction on TextQ-ATS-LLM
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>bert-base-german-uncased</td><td>1.1406</td><td>0.8721</td><td>-0.0099</td></tr><tr><td>bert-base-german-cased</td><td>1.1558</td><td>0.8913</td><td>-0.0502</td></tr><tr><td>gbert-base</td><td>1.1483</td><td>0.8638</td><td>0.0073</td></tr><tr><td>gbert-large</td><td>0.9502</td><td>0.7304</td><td>0.2859</td></tr><tr><td>gelectra-large</td><td>1.0045</td><td>0.7368</td><td>0.1920</td></tr></table>

Table 16 Performance of the fine-tuned language models for overall QoE prediction on TextQ-MT-LLM
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td> $\overline { { R ^ { 2 } } }$ </td></tr><tr><td>bert-base-german-uncased</td><td>1.6404</td><td>1.3925</td><td>-0.4543</td></tr><tr><td>bert-base-german-cased</td><td>1.3572</td><td>1.0988</td><td>-0.0299</td></tr><tr><td>gbert-base</td><td>1.4740</td><td>1.2506</td><td>-0.1061</td></tr><tr><td>gbert-large</td><td>1.1997</td><td>1.0241</td><td>0.1598</td></tr><tr><td>gelectra-large</td><td>1.1450</td><td>0.9929</td><td>0.3246</td></tr></table>

Tables 15 and 16 report the results of fine-tuning language models for overall QoE prediction. Although these datasets were collected independently from the dimensionlevel QoE corpora, the same models performed best, namely gbert-large for ATS and gelectra-large for MT. We therefore used these again as the backbone models.

## Joint Training

Table 17 Performance of the fine-tuned language models for overall QoE prediction on TextQ-ATS-LLM and TextQ-MT-LLM
<table><tr><td rowspan="2">Model</td><td colspan="3">TextQ-ATS-LLM</td><td colspan="3">TextQ-MT-LLM</td></tr><tr><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td><td>RMSE</td><td>MAE</td><td> $R ^ { 2 }$ </td></tr><tr><td>gbert-large (ATS)</td><td>0.9502</td><td>0.7304</td><td>0.2859</td><td></td><td></td><td></td></tr><tr><td>gelectra-large (MT)</td><td></td><td></td><td></td><td>1.1450</td><td>0.9929</td><td>0.3246</td></tr><tr><td>gbert-1arge (ATS ∪ MT)</td><td>1.2303</td><td>1.0197</td><td>-2.4031</td><td>1.3214</td><td>1.1107</td><td>-0.0089</td></tr><tr><td>gelectra-large (ATS ∪ MT)</td><td>0.9520</td><td>0.7358</td><td>-0.6210</td><td>1.0066</td><td>0.8202</td><td>0.4362</td></tr><tr><td>gbert-large (multi-task learning)</td><td>1.0232</td><td>0.7567</td><td>0.1254</td><td>1.2615</td><td>1.0348</td><td>0.1091</td></tr><tr><td>gelectra-large (multi-task learning)</td><td>0.9956</td><td>0.7584</td><td>0.1650</td><td>1.1613</td><td>0.9805</td><td>0.2962</td></tr></table>

Next, we explored whether jointly training on both ATS and MT datasets for overall QoE could improve performance. For each fold, we merged and shufled the training, validation, and test splits from both datasets together. We evaluated two approaches: (1) standard fine-tuning on the merged dataset, and (2) multi-task learning with hard parameter sharing, where each dataset was treated as a separate task. Table 17 presents the performance of both approaches alongside the single-dataset baselines.

Joint training produced asymmetric results. For MT, gelectra-large fine-tuned on the merged dataset improved performance across all metrics. For ATS, however, no improvement was achieved. The best joint model (gelectra-large on merged data) closely matched the single-dataset baseline in RMSE (0.9520 vs. 0.9502), although itsR<sup>2</sup> was lower. Multi-task learning underperformed for both datasets.

We interpret these results as follows. The MT baseline had more room for improvement, with an RMSE of 1.1450 compared to 0.9502 for ATS. Joint training appears to have regularized the MT model efectively, leveraging additional data from the ATS corpus to improve generalization. Conversely, the ATS model could have already been near-optimal given its data. Adding MT data introduced noise or task interference, leading to marginal or negative efects. This pattern is consistent with observations in multi-task and transfer learning, where benefits occur primarily to tasks with higher baseline error or smaller datasets. The best-performing ATS model remained (gbert-large trained on ATS alone), while the best MT model became gelectra-large trained on the merged corpus. We therefore adopted these configurations as the backbones for the hybrid models in the next experiments.

## Hybrid Models

Table 18 Performance of the hybrid models for overall QoE prediction on TextQ-ATS-LLM
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>gbert-large</td><td>0.9502</td><td>0.7304</td><td>0.2859</td></tr><tr><td>Hybrid language model</td><td>0.9714</td><td>0.7514</td><td>0.2533</td></tr><tr><td>Hybrid SVM</td><td>0.8906</td><td>0.7106</td><td>0.3712</td></tr></table>

Table 19 Performance of the hybrid models for overall QoE prediction on TextQ-MT-LLM
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>gelectra-large (ATS ∪ MT)</td><td>1.0066</td><td>0.8202</td><td>0.4362</td></tr><tr><td>Hybrid language model (ATS ∪ MT)</td><td>1.1726</td><td>0.9609</td><td>0.3206</td></tr><tr><td>Hybrid SVM (ATS ∪ MT)</td><td>1.0870</td><td>0.9262</td><td>0.3381</td></tr></table>

Tables 18 and 19 compare the performance of the hybrid models against the previous baselines. For ATS, as was the case in the dimension-level experiments, the hybrid SVM outperformed the baseline, improving all three reported metrics. For MT, neither hybrid model surpassed the baseline.

These results demonstrate that we have successfully developed predictive models for QoE assessment of machine translation and text summarization, covering both fine-grained dimension-level scores and overall quality, while investigating a range of approaches to improve regression performance.

## 5.3 LLM-Based Extension

The TextQ-ATS-LLM and TextQ-MT-LLM datasets also include ratings for the identified QoE dimensions. We therefore extended the dimension-level prediction data by merging these with the original corpora and reran the language model and hybrid models on TextQ-ATS ∪ TextQ-ATS-LLM and TextQ-MT ∪ TextQ-MT-LLM.

TextQ-ATS ∪ TextQ-ATS-LLM  
Table 20 Performance of the models on
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>Multi-Target (gbert-large)</td><td>0.8988</td><td>0.7232</td><td>0.3636</td></tr><tr><td>Hybrid language model</td><td>0.9066</td><td>0.7220</td><td>0.3510</td></tr><tr><td>Hybrid SVM</td><td>0.8495</td><td>0.6844</td><td>0.4350</td></tr></table>

Table 21 Performance of the models on TextQ-MT ∪ TextQ-MT-LLM
<table><tr><td>Model</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>Single-Target (gelectra-large)</td><td>1.4836</td><td>1.2178</td><td>-0.1254</td></tr><tr><td>Hybrid language model</td><td>1.4487</td><td>1.2066</td><td>-0.0744</td></tr><tr><td>Hybrid SVM</td><td>1.4647</td><td>1.1964</td><td>-0.0964</td></tr></table>

For ATS, prediction performance on the combined original and LLM-generated corpus was slightly lower across all models than on TextQ-ATS alone. One possible explanation is that the LLM-generated data exhibit distinct characteristics, making the combined dataset more heterogeneous and challenging. The hybrid SVM achieved the lowest error for ATS again. With an RMSE of 0.8495, the performance remains stable despite this marginal performance degradation. For MT, however, the extended dataset proved substantially more challenging. All models exhibited substantially worse performance compared to the non-LLM-based TextQ-MT. Despite this, these models will be evaluated on the held-out validation datasets, consisting of LLM-generated and non-LLM-based text samples.

## 5.4 Final Evaluation

Finally, we evaluated the best-performing models for dimension-level QoE and overall QoE. Table 22 presents the performance of the models on the respective validation sets.

Table 22 Final validation performances on the QoE datasets
<table><tr><td>NLG Task</td><td>QoE Label</td><td>RMSE</td><td>MAE</td><td>R2</td></tr><tr><td>ATS</td><td>Dimensions</td><td>0.9197</td><td>0.7319</td><td>0.3968</td></tr><tr><td>ATS</td><td>Overall Score</td><td>0.9572</td><td>0.7457</td><td>0.4353</td></tr><tr><td>MT</td><td>Dimensions</td><td>1.5833</td><td>1.2989</td><td>-0.3177</td></tr><tr><td>MT</td><td>Overall Score</td><td>1.1824</td><td>0.9753</td><td>0.3358</td></tr></table>

Performance was generally lower on the held-out validation sets than in the prior development experiments. Yet, all metrics are relatively close to the performance the models demonstrated previously, speaking for the robustness of the dataset collection process and validating our QoE modeling. As was already indicated in the LLM-based extension experiments, dimension-level MT prediction remained the most challenging setting in the final evaluation. This may be caused by the greater heterogeneity of the MT data, which combines LLM- and non-LLM-based translations and therefore contains diferent surface-level error profiles than the original TextQ-MT corpus. Unlike ATS, where dimensions such as clarity, predictability, and linguistic logic are reflected in longer-range properties such as coherence, structure, and readability, MT outputs are mostly shorter and sentence-level, making the corresponding dimensions harder to distinguish from the text output cues alone. In particular, LLM translations may reduce obvious grammatical or readability errors, causing precision, transparency, complexity, and grammaticality to become less clearly separable. The stronger performance for the overall score suggests that MT QoE can be predicted more efectively at a coarser level, while fine-grained MT dimension prediction may require larger, more balanced data or alternative text features in future work.

The other validation sets yielded similar results to previous experiments. To interpret these results in absolute terms, recall that all QoE ratings lie on a 0 to 6 scale, where 0 represents the most negative and 6 the most positive perception. For ATS, an RMSE of approximately 0.92 to 0.96 is therefore below one point on this sevenpoint scale. Given the inherent subjectivity of human ratings and the complexity of the underlying quality dimensions, this level of accuracy is compelling. A MAE of around 0.73 to 0.75 further confirms that typical predictions deviate from human judgments by less than one scale point in absolute terms, which is also the case for overall MT QoE (0.98). The $R ^ { 2 }$ values, ranging from 0.34 to 0.44, indicate that the models capture a substantial portion of the variance in human ratings. These values may be interpreted as meaningful and acceptable. This is plausible because QoE prediction is a human-centered perceptual task, where ratings depend on subjective judgments and annotator variability rather than deterministic text properties. In comparable humancentered domains, $\mathrm { \dot { \it R } ^ { 2 } }$ values of 0.10–0.30 are often considered acceptable in social sciences and psychology (Ozili, 2022), while values above 0.15 have been described as meaningful in clinical research (Gupta et al., 2024). Considering this, our positive validation $R ^ { 2 }$ values of 0.34–0.44 indicate reasonable explanatory capability for subjective QoE modeling. The scatter plots in Figures 2 and 3 further support this interpretation, as the fitted lines show a clear positive relationship between true and predicted overall QoE scores.

![](images/2475c71703ef3dc2bd238d6f2bcb86e6cb7e8545fb651476ecbba97e236553d4.jpg)  
Fig. 2 Scatter plot of the predictions on TextQ-ATS-Val

![](images/83e804fb320cd9f04821e30275bfbc8e379860a1b402c5935638e9e9b666730d.jpg)  
Fig. 3 Scatter plot of the predictions on TextQ-MT-Val

Overall, these results demonstrate the feasibility of automatic QoE prediction for German NLG and reveal diferences across tasks and prediction targets. On the sourcedisjoint held-out sets, the models retain predictive capability for ATS dimension-level and overall QoE as well as overall MT QoE, whereas fine-grained MT dimension prediction remains challenging. These findings provide initial baselines for automatic QoE assessment using TextQ-German and identify directions for further modeling and data collection.

## 6 Conclusion

In this paper, we introduced TextQ-German, a novel dataset suite for Quality of Experience assessment of German Natural Language Generation, covering automatic text summarization and machine translation. Through crowdsourcing studies, we identified and validated four perceptual quality dimensions for each task: Precision, Complexity, Grammaticality, and Transparency for MT, and Linguistic Logic, Complexity, Clarity, and Predictability for ATS.

We developed and evaluated a range of automatic QoE prediction models, including transformer-based models, linguistic feature-based models, and hybrid approaches. Our experiments show that hybrid models, which combine neural embeddings with carefully selected interpretable linguistic features, improve over pure transformerbased baselines in most experiments. Notably, linguistic features alone, when selected through Sequential Feature Selection, achieve performance approaching that of finetuned language models, while ofering the advantage of transparency, eficiency and interpretability.

The final validation on held-out sets comprising both LLM-generated and non-LLM-based texts provides evidence of out-of-sample generalization for ATS and overall MT QoE, while fine-grained MT dimension prediction remains more challenging.

As NLG systems become increasingly integrated into real-world applications, we believe that user-centered evaluation frameworks such as QoE will play an essential role in ensuring that generated text meets the expectations and needs of its human audience.

A key limitation of our study is that the QoE assessment is operationalized through quality attributes that are directly observable in the text (complexity, grammaticality, etc.). While these are central to the user’s perception, a QoE evaluation may also involve contextual factors such as task utility, prior expectations, and emotional response, which we did not model. We therefore view our work as a novel resource and important step towards QoE assessment for NLG rather than the complete, finished realization of this. Furthermore, as obtaining human ratings is costly, the relatively small dataset size restricts the training of larger models and limits the generalizability. Despite these limitations, the strong performance of hybrid models and the positive validation results demonstrate the utility of the resource.

For future work, several promising directions emerge. Extending TextQ-German to additional NLG tasks such as question-answering, dialogue generation, or image captioning would broaden the applicability of our QoE framework. Incorporating multilingual data would allow cross-lingual comparisons and help determine whether perceptual quality dimensions are language-universal or culturally dependent. Future work could explore more sophisticated multi-task and meta-learning architectures to better exploit the complementarity between ATS and MT datasets, potentially improving performance for tasks with limited data. While our linguistic features are hand-crafted and interpretable, automatically discovering novel features or leveraging large language models as feature extractors could further improve model performance. An additional methodological consideration for future work concerns the evaluation metrics themselves. Since human ratings exhibit natural variability, a prediction model should not be expected to outperform the agreement between human annotators. Metrics that account for this inherent uncertainty, such as a version of RMSE that considers predictions within the range of human standard deviation as efectively correct, would provide a more realistic assessment of model performance. Ultimately, advancing evaluation methods will be essential for developing NLG systems whose measured performance translates into meaningful improvements in the quality experienced by their users.

Appendix

A Adjective Pairs

Table 23 Complete list of polar adjective pairs used in the experiments for the text type MT in the German original and translated into English for better understanding. Adapted from Manakhimova et al. (2025)
<table><tr><td></td><td>German original</td><td>English translation</td></tr><tr><td>Group 1: final list of adjec- tive pairs that are loading on the underlying factors, with the pairs used for the qual- ity dimension quantification highlighted in boldface</td><td>direkt - umständlich eindeutig – mehrdeutig einfach – kompliziert grammatisch – ungrammatisch klar – wirr präzise – ungenau übersichtlich – verwirrend</td><td>direct – ponderous unambiguous – ambiguous simple – complicated grammatical – ungrammatical clear - chaotic precise – vague neat – confusing</td></tr><tr><td>pairs that were removed dur- ing the factor analysis for the sake of interpretability</td><td>flüssig − holprig formell – informell geordnet – durcheinander geschrieben – gesprochen höflich - unhöflich kongruent – inkongruent konsistent - inkonsistent logisch − unlogisch menschlich – technisch muttersprachlich - fremdsprachlich persönlich - unpersönlich professionell – laienhaft</td><td>fluent - non-fluent formal – informal orderly – messy written - spoken polite - impolite congruent – incongruent consistent - inconsistent logical - illogical human - technical native – foreign-language personal - impersonal</td></tr><tr><td>pairs that were removed after the preliminary study</td><td>aktiv – passiv angemessen – unangemessen angenehm – unangenehm bedeutungsvoll – bedeutungslos bekannt - unbekannt förmlich – lässig gebildet - ungebildet gut – schlecht hochwertig - minderwertig informativ - nichtssagend kreativ - simpel lustig – ernst optimal - suboptimal</td><td>active - passive appropriate - inappropriate pleasant – unpleasant meaningful – meaningless known - unknown formal - casual educated – uneducated good – bad valuable - poor informative - bland creative - simple funny – serious optimal - suboptimal</td></tr></table>

Table 24 Complete list of polar adjective pairs used in the experiments for the text type ATS in the German original and translated into English for better understanding. Adapted from Manakhimova et al. (2025)
<table><tr><td></td><td>German original</td><td>English translation</td></tr><tr><td rowspan="2">Group 1: final list of adjec- tive pairs that are loading on the underlying factors, with the pairs used for the qual- ity dimension quantification highlighted in boldface</td><td>eindeutig – mehrdeutig einfach - kompliziert</td><td>unambiguous – ambiguous simple - complicated</td></tr><tr><td>logisch – unlogisch präzise – ungenau simpel – komplex vollständig – lückenhaft vorhersehbar – unberechenbar zusammenhängend – unzusammen-</td><td>logical - illogical precise – vague straightforward - complex complete – incomplete predictable - unpredictable coherent - incoherent</td></tr><tr><td>Group 2: list of adjective pairs that were removed dur- ing the factor analysis for the sake of interpretability</td><td>anspruchsvoll – anspruchslos ausführlich – knapp direkt - umständlich fehlerfrei – fehlerhaft fließend − holprig geordnet – ungeordnet grammatisch – ungrammatisch klar – wirr sinnvoll - sinnlos übersichtlich - verwirrend verständlich - unverständlich widerspruchsfrei – widersprüchlich</td><td>sophisticated - unsophisticated elaborate - terse direct – ponderous error-free – faulty fluent - non-fluent ordered – disordered grammatical - ungrammatical clear – chaotic meaningful – meaningless neat - confusing comprehensible – incomprehensible</td></tr><tr><td>Group 3: list of adjective pairs that were removed after the preliminary study bekannt – fremd einheitlich - uneinheitlich</td><td>wortreich - wortarm konsistent - inkonsistent tiefgehend - oberflächlich informativ - uninformativ nicht wiederholend – wiederholend geeignet – unpassend kongruent – inkongruent pragmatisch − unpragmatisch eigenständig – uneigenständig konkret - abstrakt kompatibel - inkompatibel normal – skurril beständig – wechselhaft prägnant – ungenau sinnig − unsinnig kohärent – inkohärent</td><td>consistent – contradictory verbose – concise consistent - inconsistent thorough − cursory informative - uninformative non-repetitive - repetitive suitable - unsuitable congruent – incongruent pragmatic − impragmatic independent – dependent concrete - abstract compatible - incompatible normal - eccentric stable - volatile</td></tr></table>

## B Generation Prompts

The following prompts were used to generate the LLM-based corpus extensions described in Section 3.2. Each source text was read from the input data, inserted into the prompt at the position marked <source text>, and processed individually: the OpenAI models were called through the Chat Completions API, and the open-weight models through Ollama.

Machine translation (all models).

Translate the following English text to German: <source text>

Summarization, locally hosted open-weight models (except SauerkrautLM).

Fasse den folgenden Text auf Deutsch so kurz wie m¨oglich zusammen. <source text>   
Kurze Zusammenfassung (Deutsch):

Summarization, SauerkrautLM-7B-v1 (Alpaca format).

### Anweisung:   
FASS DEN FOLGENDEN TEXT PR<sup>¨</sup>AGNANT IN 3-5 S<sup>¨</sup>ATZEN ZUSAMMEN. KONZENTRIERE DICH AUF DIE HAUPTAUSSAGEN UND WICHTIGSTEN PUNKTE. VERMEIDE WIEDERHOLUNGEN UND   
NEBENS<sup>¨</sup>ACHLICHKEITEN.   
### Eingabe:   
<source text>   
### Antwort:

Summarization, GPT-4o length-constrained variant (gpt-4o short).

Fasse den folgenden Text sehr pr¨agnant in maximal 3 sehr kurzen S¨atzen zusammen. Verwende einfache und kurze Formulierungen. Halte jeden Satz m¨oglichst unter 15 W¨ortern.   
Die Zusammenfassung soll auf Deutsch sein:   
<source text>

## C Linguistic Features

Table 25 All features used in the feature selection process, categorized by feature type
<table><tr><td>Feature type</td><td>Features</td></tr><tr><td>Readability features</td><td>Gunning Fog Index, Flesch reading ease, SMOG index, Coleman-Liau index, adjective ttr, adposition ttr, adverb ttr, aux ttr, Wiener Sachtextformel 1-4</td></tr><tr><td>Lexical richness features</td><td>Dugast&#x27;s Uber Index, Yule&#x27;s K, Simpson&#x27;s D, Herdan&#x27;s lexical diversity measure, Summer&#x27;s lexical diversity measure, Maas&#x27;s lexical diversity measure, lexical density, noun variation, content word distribution, adjective variation, adverb variation, modifier variation, verb variation, type token ratio, root type token ratio, average sentence length in words, average word length in characters, average word length in syllables, unique tokens count, bilog ttr, conjunction ttr, cttr, determiner ttr, grammatical item ttr, lexical item ttr, lexical verb ttr, noun ttr, particle ttr, pronoun ttr, proper noun ttr, verb ttr, zipf goodness of fit, zipf steepness of curve</td></tr><tr><td>Syntactic features</td><td>Noun phrases per sentence count, verb phrases per sentence count, average sentence length in characters, characters per sentence count, finite verbs count, foreign words count, infinitive verb count, noun to pronoun ratio, passive voice count, past tense count, present tense count, pronouns per sentence count, punctuation count, standard deviation tokens per sentence, verb to adverb ratio, verb to noun ratio, verbs per sentence count</td></tr><tr><td>Morphological features</td><td>Infinitive verbs to verbs ratio, participle verbs to verbs ratio, nominative nouns to nouns ratio, genitive nouns to nouns ratio, dative nouns to nouns ratio, accusative nouns to nouns ratio, haben to verb ratio, sein to verb ratio, noun token ratio, verb token ratio, finite verbs to verbs ratio, first person verbs to finite verbs ratio, second person verbs to finite verbs ratio, third person verbs to finite verbs ratio, imperative verbs to verbs ratio, particle verbs to verbs ratio, past tense verbs to finite verbs ratio, present tense verbs to finite verbs ratio, subjunctive verbs to finite verbs ratio, percentage of words with 6 or more letters, adjective count, adposition count, adverb count, aux to verb ratio, aux verb count, conjunction count, definite word count, demonstrative count, determiners count, first person pronoun count, grammatical item count, indefinite word count, interrogative count, lexical item count, noun count, numeral count, particle count, personal pronoun count, plural word count, proper noun count, second person pronoun count, singular word count, verb count, average adjective length, average adposition length, average adverb length, average aux length, average conjunction length, average determiner length, average lexical verb length, average noun length, average numeral length, average particle length, average pronoun length, average proper noun length, average verb length, percentage of monosyllabic words, percentage of words with 3 or more syllables</td></tr></table>

Table 26 Performance of feature selection methods on TextQ-ATS by quality dimension
<table><tr><td>Dimension</td><td>Method</td><td>RMSE</td><td># Feat.</td><td>Selected features</td></tr><tr><td>Logic</td><td>RFE SFS</td><td>1.2279 1.1316</td><td>1 yules k 20</td><td>average number syllables, smog index, adposition count, parti- cle count, first person pronoun count, second person pronoun count, interrogative pronoun count, indefinite word count, avg aux length, noun variation, first person verbs to finite verbs ratio, imperative verbs to verbs ratio, second person verbs to finite verbs ratio, third person verbs to finite verbs ratio, subjunctive verbs to finite verbs ratio, std dev tokens per</td></tr><tr><td></td><td>Lasso Elastic</td><td>1.1997 1.1997</td><td>1 1</td><td>sentence, avg adposition length, avg determiner length, avg numeral length, adposition ttr indefinite word&#x27;count indefinite word count</td></tr><tr><td>Complexity</td><td>RFE SFS</td><td>1.1347 0.8336</td><td>1 20</td><td>noun count smog index, type token ratio, number unique tokens, con- tent word distribution, numerical count, first person pronoun count, demonstrative count, definite word count, present tense count, avg aux length, first person verbs to finite verbs ratio, imperative verbs to verbs ratio, present tense verbs to finite verbs ratio, past tense verbs to finite verbs ratio, second per-</td></tr><tr><td></td><td>Lasso</td><td>0.9340</td><td>22</td><td>son verbs to finite verbs ratio, third person verbs to finite verbs ratio, subjunctive verbs to finite verbs ratio, std dev tokens per sentence, noun to pronoun ratio, conjunction ttr wiener sachtextformel 1, root type token ratio, corrected type token ratio, lexical density, content word distribution, verb count, aux verb count, numerical count, demonstrative count, present tense count, noun token ratio, noun phrases per sen- tence count, finite verbs to verbs ratio, std dev tokens per</td></tr><tr><td></td><td>Elastic 1.0582</td><td></td><td>28</td><td>sentence, pronouns per sentence count, avg numeral length, noun ttr, lexical items ttr, adposition ttr, aux ttr, conjunction ttr, grammatical ttr wiener sachtextformel 1, wiener sachtextformel 2, wiener sach- textformel 3, wiener sachtextformel 4, type token ratio, root type token ratio, corrected type token ratio, lexical den- sity, number unique tokens, content word distribution, aux verb count, numerical count, demonstrative count, past tense count, present tense count, adjective variation, noun token</td></tr><tr><td>Clarity</td><td>RFE</td><td>0.9368</td><td>1</td><td>ratio, noun phrases per sentence count, participle verbs to verbs ratio, finite verbs to verbs ratio, std dev tokens per sen- tence, pronouns per sentence count, avg numeral length, noun ttr, lexical items ttr, adposition ttr, aux ttr, grammatical ttr yules k adposition count, particle count, first person pronoun count, second person pronoun count, interrogative pronoun count, demonstrative count, singular word count, present tense count, first person verbs to finite verbs ratio, imperative verbs to verbs ratio, second person verbs to finite verbs ratio, std</td></tr><tr><td></td><td></td><td></td><td></td><td>dev tokens per sentence, pronouns per sentence count, avg proper noun length, avg adposition length, avg determiner length, avg numeral length, lexical items ttr, adposition ttr, conjunction ttr</td></tr><tr><td>Predictability</td><td>Elastic RFE SFS</td><td>0.9098 0.7828</td><td>00 7 20</td><td>yules k conjunction count, first person pronoun count, second person pronoun count, interrogative pronoun count, demonstrative count, singular word count, present tense count, adjective variation, first person verbs to finite verbs ratio, imperative</td></tr><tr><td></td><td></td><td></td><td></td><td>verbs to verbs ratio, present tense verbs to finite verbs ratio, past tense verbs to finite verbs ratio, second person verbs to finite verbs ratio, third person verbs to finite verbs ratio,</td></tr><tr><td></td><td></td><td></td><td></td><td>subjunctive verbs to finite verbs ratio, std dev tokens per sen- tence, pronouns per sentence count, avg proper noun length, lexical items ttr, conjunction ttr present tense count</td></tr></table>

The best RMSE within each quality dimension is highlighted in bold. A dash indicates that the method selected no features and therefore produced no valid RMSE.

Table 27 Performance of feature selection methods on TextQ-MT by quality dimension
<table><tr><td>Dimension</td><td>Method</td><td>RMSE</td><td># Feat.</td><td>Selected features</td></tr><tr><td rowspan="4">Complexity</td><td>RFE</td><td>1.1028</td><td>13</td><td>coleman liau index, average chars per word, type token ratio, root type token ratio, corrected type token ratio, herdans mea- sure, summers measure, maas measure, yules k, simpsons d, number unique tokens, verb count, aux verb count average sentence length, number unique tokens, numerical count, interrogative pronoun count, past tense count, passive</td></tr><tr><td>SFS</td><td></td><td></td><td>voice count, participle verbs to verbs ratio, imperative verbs to verbs ratio, accusative nouns to nouns ratio, dative nouns to nouns ratio, genitive nouns to nouns ratio, std dev tokens per sentence, verb to adverb ratio, characters per sentence count, avg sentence length in characters, avg conjunction length, lex- ical verb ttr, verb ttr, coniunction ttr, determiner ttr</td></tr><tr><td>Lasso</td><td>0.9291 13</td><td></td><td>average sentence length, type token ratio, numerical count, interrogative pronoun count, past tense count, haben to verb ratio, participle verbs to verbs ratio, dative nouns to nouns ratio, genitive nouns to nouns ratio, characters per sentence count, avg sentence length in characters, avg conjunction length, verb ttr</td></tr><tr><td>Elastic</td><td>0.9630 15</td><td></td><td>average sentence length, flesch reading ease, type token ratio, lexical density, numerical count, interrogative pronoun count, past tense count, haben to verb ratio, participle verbs to verbs ratio, dative nouns to nouns ratio, genitive nouns to nouns ratio, characters per sentence count, avg sentence length in characters, avg conjunction length, verb ttr</td></tr><tr><td rowspan="2">Grammaticality RFE</td><td>SFS</td><td>1.3810 1 1.0656 20</td><td>simpsons d</td><td>flesch reading ease, maas measure, number unique tokens, adposition count, numerical count, present tense count, bilog- arithmic ttr, sein to verb ratio, haben to verb ratio, verb phrases per sentence count, first person verbs to finite verbs ratio, participle verbs to verbs ratio, past tense verbs to finite</td></tr><tr><td></td><td></td><td></td><td>verbs ratio, third person verbs to finite verbs ratio, genitive nouns to nouns ratio, std dev tokens per sentence, avg adpo- sition length, avg determiner length, adjective ttr, determiner ttr</td></tr><tr><td rowspan="4">Transparency</td><td>Lasso Elastic</td><td>1.2994 1.3105</td><td>2 3</td><td>flesch reading ease, numerical count average sentence length, flesch reading ease, numerical count average sentence length, flesch reading ease, type token ratio, herdans measure, summers measure, maas measure, yules k, simpsons d, number unique tokens, bilogarithmic ttr, zipf</td></tr><tr><td>RFE SFS</td><td>1.2641</td><td>13</td><td>steepness curve, avg determiner length, determiner ttr percentage words three syllables, flesch reading ease, root type token ratio, corrected type token ratio, grammatical item count, numerical count, plural word count, past tense count,</td></tr><tr><td></td><td></td><td></td><td>passive voice count, participle verbs to verbs ratio, imperative verbs to verbs ratio, third person verbs to finite verbs ratio. genitive nouns to nouns ratio, verb to adverb ratio, charac- ters per sentence count, avg sentence length in characters, avg noun length, avg particle length, noun ttr, determiner ttr</td></tr><tr><td>Lasso</td><td>1.2579 6</td><td></td><td>flesch reading ease, lexical item count, numerical count, par- ticiple verbs to verbs ratio, dative nouns to nouns ratio, genitive nouns to nouns ratio average sentence length, flesch reading ease, lexical item count, numerical count, haben to verb ratio, participle verbs</td></tr><tr><td rowspan="2">Precision</td><td></td><td></td><td></td><td>to verbs ratio, dative nouns to nouns ratio, genitive nouns to nouns ratio yules k percentage words six letters, flesch reading ease, type token ratio, corrected type token ratio, herdans measure, maas mea-</td></tr><tr><td>SFS</td><td>1.0106 20</td><td></td><td>sure, adposition count, numerical count, demonstrative count, definite word count, bilogarithmic ttr, verb phrases per sen- tence count, infinitive verbs to verbs ratio, participle verbs to</td></tr><tr><td rowspan="2"></td><td></td><td></td><td></td><td>verbs ratio, third person verbs to finite verbs ratio, nomina- tive nouns to nouns ratio, std dev tokens per sentence, verb to adverb ratio, avg determiner length, determiner ttr</td></tr><tr><td>Lasso Elastic</td><td>1.2428 1.2428</td><td>1 1</td><td>numerical count numerical count</td></tr></table>

The best RMSE within each quality dimension is highlighted in bold.

Acknowledgements. We thank Dominik Frefel, Manfred Vogel, and Fabian M¨arki for providing the GeWiki summary corpus they developed for the German Text Summarization Challenge of SwissText & KONVENS 2020. We are also grateful to our colleagues for participating in our pre-study. We express our gratitude to Aleksandra Gabryszak for her insights into automatic text summarization and, together with Polina Danilovskaia, for their valuable support with dataset quality control and the training of an exploratory, preliminary model.

The presented research was supported by the Deutsche Forschungsgemeinschaft (DFG) through the project “Analyse und automatische Absch¨atzung der Qualit¨at maschinell generierter Texte”, project number 436813723.

## Declarations

Funding. The presented research was funded by the Deutsche Forschungsgemeinschaft (DFG) through the project “Analyse und automatische Absch¨atzung der Qualit¨at maschinell generierter Texte”, project number 436813723.

Conflict of interest. The authors declare no Conflict of interest.

Ethics approval and consent to participate. Not applicable.

Consent for publication. Not applicable.

Data availability. Our experiment code, adjective pairs, textual/linguistic features, and datasets can be accessed at https://github.com/DFKI-NLP/TextQ/. The dataset is additionally available at https://huggingface.co/datasets/nphamdinh/textq-german and https://www.kaggle.com/datasets/namphamdinh/textq-german/.

Materials availability. Not applicable.

Code availability. Available at https://github.com/DFKI-NLP/TextQ/.

Author contributions. Drafting of the manuscript, conceptualization and implementation of the prediction experiments, analysis and discussion of the model results were led by Dinh Nam Pham. Shushen Manakhimova and Vivien Macketanz jointly conceptualized and compiled the datasets, including the design and execution of the crowdsourcing studies, quality dimension identification and statistical analysis, and the LLM-based corpus extensions. Sebastian M¨oller provided supervision and acquired funding for the project. All authors read and approved the final manuscript.

## References

Ansch¨utz, M., & Groh, G. (2022, September). TUM social computing at GermEval 2022: Towards the significance of text statistics and neural embeddings in text complexity prediction. In S. M¨oller, S. Mohtaj, & B. Naderi (Eds.), Proceedings of the germeval 2022 workshop on text complexity assessment of german text (pp. 21–26). Association for Computational Linguistics. https: //aclanthology.org/2022.germeval-1.4/

Bansal, S., & Aggarwal, C. (2026). Textstat (Version 0.7.13). https://pypi.org/project/ textstat/

Barrault, L., Bojar, O., Costa-juss\`a, M. R., Federmann, C., Fishel, M., Graham, Y., Haddow, B., Huck, M., Koehn, P., Malmasi, S., Monz, C., M¨uller, M., Pal, S., Post, M., & Zampieri, M. (2019, August). Findings of the 2019 conference on machine translation (WMT19). In O. Bojar, R. Chatterjee, C. Federmann, M. Fishel, Y. Graham, B. Haddow, M. Huck, A. J. Yepes, P. Koehn, A. Martins, C. Monz, M. Negri, A. N´ev´eol, M. Neves, M. Post, M. Turchi, & K. Verspoor (Eds.), Proceedings of the fourth conference on machine translation (volume 2: Shared task papers, day 1) (pp. 1–61). Association for Computational Linguistics. https://doi.org/10.18653/v1/W19-5301

Bayerische Staatsbibliothek. (2025a). Bert-base-german-cased (revision 43cce13). https://doi.org/10.57967/hf/4377

Bayerische Staatsbibliothek. (2025b). Bert-base-german-uncased (revision b705f0e). https://doi.org/10.57967/hf/4378

Belz, A., & Reiter, E. (2006, April). Comparing automatic and human evaluation of NLG systems. In D. McCarthy & S. Wintner (Eds.), 11th conference of the European chapter of the association for computational linguistics (pp. 313– 320). Association for Computational Linguistics. https://aclanthology.org/ E06-1040/

Chan, B., Schweter, S., & M¨oller, T. (2020, December). German‘s next language model. In D. Scott, N. Bel, & C. Zong (Eds.), Proceedings of the 28th international conference on computational linguistics (pp. 6788–6796). International Committee on Computational Linguistics. https://doi.org/10.18653/v1/2020. coling-main.598

Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019, June). BERT: Pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, & T. Solorio (Eds.), Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers) (pp. 4171–4186). Association for Computational Linguistics. https://doi.org/10.18653/v1/ N19-1423

Dohare, S., Karnick, H., & Gupta, V. (2017). Text summarization using abstract meaning representation. arXiv preprint arXiv:1706.01678. http://arxiv.org/ abs/1706.01678

Fabbri, A. R., Kry´sci´nski, W., McCann, B., Xiong, C., Socher, R., & Radev, D. (2021). SummEval: Re-evaluating summarization evaluation (B. Roark & A. Nenkova, Eds.). Transactions of the Association for Computational Linguistics, 9, 391– 409. https://doi.org/10.1162/tacl a 00373

Fisher, R. (1925). Statistical methods for research workers. Oliver; Boyd.

Frefel, D. (2020, May). Summarization corpora of Wikipedia articles. In N. Calzolari, F. B´echet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk, & S. Piperidis

(Eds.), Proceedings of the twelfth language resources and evaluation conference (pp. 6651–6655). European Language Resources Association. https:// aclanthology.org/2020.lrec-1.821/

Frefel, D., Vogel, M., & M¨arki, F. (2020). 2nd german text summarization challenge. In S. Ebling, D. Tuggener, M. H¨urlimann, M. Cieliebak, & M. Volk (Eds.), Proceedings of the 5th swiss text analytics conference and the 16th conference on natural language processing, swisstext/konvens 2020, zurich, switzerland, june 23-25, 2020 [online only]. CEUR-WS.org. https://ceur- ws.org/Vol-2624/germeval-task3-paper1.pdf

Freitag, M., Foster, G., Grangier, D., Ratnakar, V., Tan, Q., & Macherey, W. (2021). Experts, errors, and context: A large-scale study of human evaluation for machine translation (B. Roark & A. Nenkova, Eds.). Transactions ofthe Association for Computational Linguistics, 9, 1460–1474. https : / / doi . org / 10 . 1162/tacl a 00437

Grubbs, F. E. (1950). Sample Criteria for Testing Outlying Observations. The Annals of Mathematical Statistics, 21 (1), 27–58. https: / /doi. org / 10. 1214 / aoms / 1177729885

Gupta, A., Stead, T. S., & Ganti, L. (2024). Determining a meaningful r-squared value in clinical medicine. Academic Medicine & Surgery.

Howcroft, D. M., Belz, A., Clinciu, M.-A., Gkatzia, D., Hasan, S. A., Mahamood, S., Mille, S., van Miltenburg, E., Santhanam, S., & Rieser, V. (2020, December). Twenty years of confusion in human evaluation: NLG needs evaluation sheets and standardised definitions. In B. Davis, Y. Graham, J. Kelleher, & Y. Sripada (Eds.), Proceedings of the 13th international conference on natural language generation (pp. 169–182). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.inlg-1.23

IBM Corp. (2021). IBM SPSS Statistics for Macintosh, Version 28.0. Armonk, NY.

International Telecommunication Union. (2017, November). Vocabulary for performance, quality of service and quality of experience (ITU-T Recommendation No. P.10/G.100). International Telecommunication Union. Geneva, Switzerland.

International Telecommunication Union. (2023, October). Subjective video quality assessment methods for multimedia applications (ITU-T Recommendation No. P.910). International Telecommunication Union. Geneva, Switzerland.

Jarque, C. M., & Bera, A. K. (1980). Eficient tests for normality, homoscedasticity and serial independence of regression residuals. Economics Letters, 6(3), 255– 259. https://doi.org/10.1016/0165-1765(80)90024-5

Levene, H. (1960, January). Robust tests for equality of variance. Stanford University Press.

Li, D., Jiang, T., & Jiang, M. (2019). Quality assessment of in-the-wild videos. Proceedings of the 27th ACM International Conference on Multimedia, 2351– 2359. https://doi.org/10.1145/3343031.3351028

Lin, C.-Y. (2004). ROUGE: A package for automatic evaluation of summaries. Text Summarization Branches Out, 74–81. https://aclanthology.org/W04-1013/

Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023, December). G-eval: NLG evaluation using gpt-4 with better human alignment. In H. Bouamor, J. Pino, & K. Bali (Eds.), Proceedings of the 2023 conference on empirical methods in natural language processing (pp. 2511–2522). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.153

Ma, C., Shi, Z., Lu, Z., Xie, S., Chao, F., & Sui, Y. (2025). A survey on image quality assessment: Insights, analysis, and future outlook. arXiv preprint arXiv:2502.08540. https://arxiv.org/abs/2502.08540

Manakhimova, S., Macketanz, V., & M¨oller, S. (2025, March). Quality of experience of german machine translation and automatic text summarization. In S. Grawunder (Ed.), Studientexte zur sprachkommunikation: Elektronische sprachsignalverarbeitung 2025 (pp. 212–222). TUDpress, Dresden. https:// www.essv.de/pdf/2025 212 222.pdf

Mihalcea, R., & Tarau, P. (2004, July). TextRank: Bringing order into text. In D. Lin & D. Wu (Eds.), Proceedings of the 2004 conference on empirical methods in natural language processing (pp. 404–411). Association for Computational Linguistics. https://aclanthology.org/W04-3252/

Mohtaj, S., Naderi, B., & M¨oller, S. (2022, September). Overview of the GermEval 2022 shared task on text complexity assessment of German text. In S. M¨oller, S. Mohtaj, & B. Naderi (Eds.), Proceedings of the germeval 2022 workshop on text complexity assessment of german text (pp. 1–9). Association for Computational Linguistics. https://aclanthology.org/2022.germeval-1.1/

Mohtaj, S., Schmitt, V., Khamsehashari, R., & M¨oller, S. (2023). Multi-task learning for German text readability assessment. CLiC-it.

M¨oller, S., & Raake, A. (Eds.). (2014). Quality of experience. Springer International Publishing. https://doi.org/10.1007/978-3-319-02681-7

Montani, I., Honnibal, M., Honnibal, M., Boyd, A., Landeghem, S. V., & Peters, H. (2023, October). Explosion/spacy: V3.7.2: Fixes for apis and requirements (Version v3.7.2). Zenodo. https://doi.org/10.5281/zenodo.10009823

Naderi, B. (2018). Motivation of workers on microtask crowdsourcing platforms (1st). Springer Publishing Company, Incorporated.

Naderi, B., Mohtaj, S., Ensikat, K., & M¨oller, S. (2019). Subjective assessment of text complexity: A dataset for german language. arXiv preprint arXiv:1904.07733. https://arxiv.org/abs/1904.07733

Naderi, B., Mohtaj, S., Karan, K., & M¨oller, S. (2019). Automated text readability assessment for german language: A quality of experience approach. 2019 Eleventh International Conference on Quality of Multimedia Experience (QoMEX), 1–3. https://doi.org/10.1109/QoMEX.2019.8743194

Naderi, B., Polzehl, T., Beyer, A., Pilz, T., & M¨oller, S. (2014). Crowdee: Mobile crowdsourcing micro-task platform for celebrating the diversity of languages. In H. Li, H. M. Meng, B. Ma, E. Chng, & L. Xie (Eds.), 15th annual conference of the international speech communication association, INTERSPEECH 2014, singapore, september 14-18, 2014 (pp. 1496–1497). ISCA.

Naderi, B., Wechsung, I., & M¨oller, S. (2015). Efect of being observed on the reliability of responses in crowdsourcing micro-task platforms. 2015 Seventh International Workshop on Quality of Multimedia Experience (QoMEX), 1–2. https: //doi.org/10.1109/QoMEX.2015.7148091

Novikova, J., Duˇsek, O., Cercas Curry, A., & Rieser, V. (2017, September). Why we need new evaluation metrics for NLG. In M. Palmer, R. Hwa, & S. Riedel (Eds.), Proceedings of the 2017 conference on empirical methods in natural language processing (pp. 2241–2252). Association for Computational Linguistics. https://doi.org/10.18653/v1/D17-1238

Osgood, C. E., Suci, G. J., & Tannenbaum, P. H. (1957). The measurement of meaning. Univer. Illinois Press.

Ozili, P. K. (2022). The acceptable r-square in empirical modelling for social science research. SSRN Electron. J.

Papineni, K., Roukos, S., Ward, T., & Zhu, W.-J. (2002, July). Bleu: A method for automatic evaluation of machine translation. In P. Isabelle, E. Charniak, & D. Lin (Eds.), Proceedings of the 40th annual meeting of the association for computational linguistics (pp. 311–318). Association for Computational Linguistics. https://doi.org/10.3115/1073083.1073135

Pham, D. N., Macketanz, V., Manakhimova, S., & M¨oller, S. (2025, September). Modeling quality of experience in German automatic text summarization and machine translation. In C. Wartena & U. Heid (Eds.), Proceedings of the 21st conference on natural language processing (konvens 2025): Workshops (pp. 169–175). HsH Applied Academics. https : / / aclanthology. org / 2025 . konvens-2.12/

Rei, R., Stewart, C., Farinha, A. C., & Lavie, A. (2020, November). COMET: A neural framework for MT evaluation. In B. Webber, T. Cohn, Y. He, & Y. Liu (Eds.), Proceedings of the 2020 conference on empirical methods in natural language processing (emnlp) (pp. 2685–2702). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.213

See, A., Liu, P. J., & Manning, C. D. (2017, July). Get to the point: Summarization with pointer-generator networks. In R. Barzilay & M.-Y. Kan (Eds.), Proceedings ofthe 55th annual meeting ofthe association for computational linguistics (volume 1: Long papers) (pp. 1073–1083). Association for Computational Linguistics. https://doi.org/10.18653/v1/P17-1099

Seife, L., Kallel, F., M¨oller, S., Naderi, B., & Roller, R. (2022, June). Subjective text complexity assessment for German. In N. Calzolari, F. B´echet, P. Blache, K. Choukri, C. Cieri, T. Declerck, S. Goggi, H. Isahara, B. Maegaard, J. Mariani, H. Mazo, J. Odijk, & S. Piperidis (Eds.), Proceedings of the thirteenth language resources and evaluation conference (pp. 707–714). European Language Resources Association. https://aclanthology.org/2022.lrec-1.74/

van der Lee, C., Gatt, A., van Miltenburg, E., Wubben, S., & Krahmer, E. (2019, October). Best practices for the human evaluation of automatically generated text. In K. van Deemter, C. Lin, & H. Takamura (Eds.), Proceedings of the 12th international conference on natural language generation (pp. 355–368).

Association for Computational Linguistics. https://doi.org/10.18653/v1/ W19-8643

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention is all you need. In I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, & R. Garnett (Eds.), Advances in neural information processing systems (Vol. 30). Curran Associates, Inc.

W¨altermann, M., Raake, A., & M¨oller, S. (2010). Quality dimensions of narrowband and wideband speech transmission. Acta Acustica united with Acustica, 96(6), 1090–1103.

Wang, S., Yu, W., Chen, X., Tian, X., Zhang, J., Lu, L., Tsao, Y., Yamagishi, J., Wang, Y., & Zhang, C. (2025, July). QualiSpeech: A speech quality assessment dataset with natural language reasoning and descriptions. In W. Che, J. Nabende, E. Shutova, & M. T. Pilehvar (Eds.), Proceedings of the 63rd annual meeting of the association for computational linguistics (volume 1: Long papers) (pp. 23588–23609). Association for Computational Linguistics. https://doi.org/10.18653/v1/2025.acl-long.1150

Welch, B. L. (1947). The generalization of ‘student’s’ problem when several diferent population variances are involved. Biometrika, 34 (1/2), 28–35. Retrieved May 4, 2026, from http://www.jstor.org/stable/2332510

Willmott, C. J., & Matsuura, K. (2005). Advantages of the Mean Absolute Error (MAE) over the Root Mean Square Error (RMSE) in assessing average model performance. Climate Research, 30, 79–82.

Yang, B., Wang, L., Wong, D. F., Chao, L. S., & Tu, Z. (2019, June). Convolutional self-attention networks. In J. Burstein, C. Doran, & T. Solorio (Eds.), Proceedings of the 2019 conference of the north American chapter of the association for computational linguistics: Human language technologies, volume 1 (long and short papers) (pp. 4040–4045). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1407

Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., & Artzi, Y. (2020). Bertscore: Evaluating text generation with BERT. 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020. https: //openreview.net/forum?id=SkeHuCVFDr