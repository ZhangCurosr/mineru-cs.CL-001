# HealMed: Multilingual Evaluation of Large Language Models in Medicine

HealMed Research Team

See the Contributions section for the complete list of contributors and affiliations.

HealMed

## Abstract

We present HealMed, an expert-reviewed benchmark for multilingual evaluation of large language models in medicine. HealMed contains 1,000 examples in each of nine languages, drawn from nine datasets and covering three task formats: MCQA, NLI and open-ended QA. The benchmark was developed over two years by 23 physicians and medical experts based across nine countries and regions. Each translation was evaluated and revised by two experts fluent in English and the corresponding target language. On HealMed, performance declined most in low-resource languages, although the size of the gap varied markedly across languages and models. The strongest proprietary models were the most stable across languages, whereas many open-source and medically specialized models showed larger and less consistent gaps. Medical specialization alone did not ensure multilingual robustness. Furthermore, expert revision could either raise or lower measured performance, indicating that translation quality materially affects cross-language evaluation results.

## 1 Introduction

Large language models (LLMs) have achieved strong performance on questions from medical licensing examinations and can generate long-form medical responses that receive favourable clinical ratings (Singhal et al., 2023, 2025). Yet evidence for these capabilities remains centred on English (Ahuja et al., 2023; Wu et al., 2025; Xuan et al., 2025). Performance on English benchmarks does not establish whether a model can interpret medical concepts or communicate them reliably in other languages, especially those underrepresented in training data and existing evaluations (Chen et al., 2025b; Singhal et al., 2025). Multilingual assessment is therefore needed to determine where medical competence transfers across languages and where it fails.

Several multilingual medical benchmarks now cover question answering and clinical language understanding (Qiu et al., 2024; Alonso et al., 2024; Wu et al., 2026). Many, however, are built by translating English test sets. Translation can alter medical terminology, syntax and contextual meaning. These changes can affect model scores and rankings (Artetxe et al., 2020; Singh et al., 2025). A lower score in a target language may therefore reflect weaker model capability, translation error or both. Without expert review, these sources of error cannot be separated.

In this paper, we present HealMed: Humanverified Evaluation Across Languages for Medical AI, an expert-reviewed medical benchmark spanning nine languages, nine source datasets and three task families: multiple-choice question answering (MCQA), natural language inference (NLI) and open-ended question answering (QA). 23 physicians and medical experts contributed to its construction and expert review. Every targetlanguage instance underwent a structured twostage review by two medical experts fluent in English and the corresponding target language. We evaluated 14 LLMs on MCQA and NLI and a subset of ten models on open-ended QA. The panel comprised five proprietary models (GPT-5.4, o4-mini, Gemini-3-Flash, Claude-Sonnet-5 and Claude-Opus-4.8) (OpenAI, 2026, 2025; Google DeepMind, 2025; Anthropic, 2026b,a), six generalpurpose open-source model configurations from the DeepSeek (DeepSeek-AI, 2024), Qwen (Yang et al., 2024, 2025), LLaMA (Grattafiori et al., 2024) and Gemma (Gemma Team et al., 2025) families, and three medically specialized models: HuatuoGPT-o1 (Chen et al., 2025a), MedGemma (Sellergren et al., 2025) and MediPhi (Corbeil et al., 2025). Open-ended responses were assessed using a multilingual LLMas-judge protocol. The original machine-translated and final expert-reviewed versions were retained

![](images/ab3b5d0f8d985b71fbaf2aa85a9fbf5c36a37aa6956697c47c98da85dab32fac.jpg)  
Figure 1: Overview of HealMed. a. Benchmark scope, comprising MCQA, NLI and open-ended QA across nine languages and nine source datasets. b. Construction pipeline, including English source-example selection, machine translation into eight target languages, two-stage review and revision by bilingual medical experts, and final consistency checking and curation.

for paired evaluation.

We find that the strongest proprietary models combine high accuracy with stable performance across languages, whereas the evaluated opensource and medically specialized models show larger losses in lower-resource languages. Several models in the latter groups perform strongly in higher-resource languages but decline sharply in lower-resource languages, indicating that medical specialization alone does not close the language gap. All evaluated models perform worse on MCQA and NLI in the lower-resource languages. These losses are concentrated in Swahili and Zulu, while performance in Thai remains close to that in Japanese and Chinese. Open-ended QA shows the same broad pattern: GPT-5.4 and Gemini-3-Flash remain stable across resource groups, but all eight non-proprietary models decline in the lower-resource group. The largest declines coincide mainly with problems in medical terminology, fluency and language choice.

Paired comparisons show that machinetranslated and expert-reviewed data can yield different estimates of multilingual medical performance. Expert revision raises measured performance in some language–dataset combinations and lowers it in others, with the magnitude of these shifts also varying across settings. In open-ended QA, the expert-reviewed data yield lower scores in most language–dataset combinations. This asymmetry raises the possibility that the LLM evaluator is more closely aligned with the literal wording of machine-translated questions and reference answers. Machine translation may therefore introduce measurement bias and may not fully reflect model performance on medical questions expressed naturally in the target language.

## 2 HealMed Dataset

HealMed is an expert-reviewed multilingual medical benchmark for evaluating medical AI systems across nine languages, three task families and nine source datasets. Figure 1 summarizes the benchmark scope and construction pipeline. A detailed comparison of HealMed with selected multilingual medical benchmarks, including their scope, construction and human-review procedures, is provided in Appendix B.

## 2.1 Benchmark Scope and Composition

We selected 1,000 English examples in total from the nine source datasets. Machine-translated versions for 7 target languages were obtained from GlobMed <sup>1</sup>, whereas the Thai translations were generated using GPT-5.5 through the Azure OpenAI

API with zero-shot prompting. Every translated instance was subsequently reviewed by two medical experts fluent in both English and the corresponding target language. HealMed therefore covers nine languages: English, German, Spanish, Portuguese, Japanese, Chinese, Thai, Swahili and Zulu.

HealMed covers three complementary task formats selected to evaluate medical language models under different output constraints, from fixed-choice prediction and relation classification to free-form generation. The MCQA component comprises HeadQA (Vilares and Gómez-Rodríguez, 2019), MedQA (Jin et al., 2021), MedExpQA (Alonso et al., 2024) and MMLU-Pro (Wang et al., 2024b) and requires models to select the correct answer from a fixed set of options. The NLI component comprises BioNLI (Bastan et al., 2022) and MedNLI (Romanov and Shivade, 2018) and requires models to classify the logical relation between paired biomedical or clinical statements as entailment, contradiction or, where applicable, neutral. The open-ended QA component comprises ExpertQA-Bio, ExpertQA-Med (Malaviya et al., 2024) and LiveQA (Abacha et al., 2017) and requires models to generate freeform answers. Further details are provided in Appendix A. All components use a common sampling, translation and expert-review procedure.

## 2.2 Construction and Expert Review

Each target-language instance underwent a twostage review by two medical experts fluent in English and the corresponding target language (Figure. 1b). Both experts compared the English source with the original machine translation and rated accuracy, fluency and completeness on a five-point scale. Accuracy captured semantic fidelity and the appropriate use of medical terminology; fluency captured grammaticality, naturalness and professional usage; and completeness captured the preservation of source information without omissions or unsupported additions. Scores ranged from 1, indicating severe deficiencies, to 5, indicating no material deficiencies.

In the first stage, Reviewer 1 scored the original machine translation and provided a corrected version where necessary, retaining the original wording when no change was required. The reviewer also documented inaccurate or unnatural translations with brief comments. In the second stage, Reviewer 2 independently scored the original translation using the same criteria, verified the first revision and corrected any remaining errors. This procedure produced two sets of quality scores and one final expert-revised translation for each targetlanguage instance.

## 3 Model performance on HealMed

## 3.1 Performance on MCQA and NLI.

We assessed whether overall performance tracked cross-language stability across the 14 models (Figure. 2). For each model, we macro-averaged accuracy across four MCQA and two NLI datasets within each language and then averaged across the nine languages. We defined the resource gap as the difference between mean accuracy in higherresource languages (English, German, Spanish, Portuguese, Japanese and Chinese) and lowerresource languages (Thai, Swahili and Zulu)<sup>2</sup>. Complete model- and language-level accuracies for the six component datasets are reported in Appendix H.

Proprietary models were both the most accurate and the most stable across languages. The five proprietary models occupied the top five positions in overall accuracy and had the five smallest resource gaps, ranging from 1.9 to 5.5 percentage points (Figure. 2a). Their mean accuracies in lowerresource languages ranged from 79.0% to 84.0%, compared with a maximum of 60.0% among the open-source and medically specialized models.

High aggregate accuracy did not guarantee multilingual stability among non-proprietary models. Qwen2.5-72B-Instruct and Gemma-3-27Bit achieved similar overall accuracies (65.0% and 65.3%, respectively), but their resource gaps differed substantially (26.8 and 9.7 percentage points). Qwen3-32B-thinking was the highest-performing non-proprietary model overall, yet its mean accuracy in lower-resource languages was 21.7 percentage points below that in higher-resource languages. The three medically specialized models also showed substantial resource gaps of 13.6–20.8 percentage points. Medical specialization alone therefore did not ensure greater cross-language stability in this model panel.

![](images/dc64b1252f78b463d81f3566c6dc73ea260d5c51aeb294a47f0389e2984edd28.jpg)

b. Performance across languages  
![](images/2b83df0f09a7b018cac6e4a42ac5dc782e6456cfcccf65aaa01f8442dfc02ae5.jpg)

![](images/882e14b4592cbeebe09fdfed02729aaa8e8220384e35c1a6c80c6292c6217ba2.jpg)  
Figure 2: Multilingual performance on the MCQA and NLI tasks in HealMed. a. Macro-average accuracy across four MCQA and two NLI datasets for 14 models. Blue bars show the lower-resource mean, and red extensions show the gap to the higher-resource mean. Circles, squares and diamonds denote proprietary, open-source and medically specialized models, respectively. b. Language-level accuracy across models. c. Within-model accuracy shifts relative to English. Grey points represent individual models; colored points and horizontal lines show the mean and interquartile range. Black, blue and red denote English, other higher-resource languages and lower-resource languages, respectively. Higher-resource languages comprise English, German, Spanish, Portuguese, Japanese and Chinese; lower-resource languages comprise Thai, Swahili and Zulu.

Performance losses relative to English were concentrated in Swahili and Zulu, whereas Thai showed reductions similar to those observed in Japanese and Chinese. Mean accuracy was lower than in English for all eight translated languages (Figure. 2b,c). The reductions ranged from 2.1 to 3.8 percentage points in German, Spanish and Portuguese and were 6.1, 6.5 and 5.9 percentage points in Japanese, Chinese and Thai, respectively. Larger reductions occurred in Swahili (15.4 percentage points) and Zulu (28.9 percentage points). Eight of the 14 models lost at least 10 percentage points in Swahili, and nine lost at least 20 percentage points in Zulu. Between-model variation increased accordingly: the interquartile range widened from 8.5 percentage points in English to 24.9 in Swahili and 44.4 in Zulu. Grouping languages by resource level therefore captured the broad trend but obscured substantial differences among individual languages.

## 3.2 Performance on Open-ended QA.

To test whether the multilingual patterns observed in MCQA and NLI extended to open-ended generation, we used an LLM-as-judge framework to evaluate ten models on ExpertQA-Bio, ExpertQA-Med and LiveQA (Figure. 3) and compared the judge scores with medical-expert ratings on a separate validation subset (Section 5.3). The model panel comprised two proprietary models (GPT-5.4 and Gemini-3-Flash), five general-purpose opensource models (DeepSeek-V3, Gemma-3-27B-it, Qwen3-32B-thinking, Qwen2.5-72B-Instruct and LLaMA3.3-70B-Instruct) and three medically specialized models (HuatuoGPT-o1-72B, MedGemma-27B and MediPhi). For each response, the judge received the target-language question, the expertverified reference answer and the model-generated answer. It assigned scores from 1 to 5 for completeness, reference alignment, clinical consensus, clinical appropriateness and safety. The mean of these five scores defined the overall score. Wronglanguage or code-switching, terminology, major fluency and demographic applicability issues were analysed separately. Model-level scores were macro-averaged across the three datasets, giving each dataset equal weight. The complete rubric and evaluation prompt are provided in Appendix D, and complete model- and language-level scores for each QA dataset are reported in Appendix H.

![](images/4aab044c89c3f0067dfce4f19a74adf3e066a84dd14d68e2873c3ceb9bf09ac7.jpg)  
c. Response-quality concerns

b. Language degradation trajectories  
![](images/15bcace98a60be9794e90bb05a5e6d32b57aece1eaf736fd5ce6d31a05bdf8f9.jpg)

![](images/fb4d4ee6829806a6b3e522e638176fc85b0babd5355cfee40119636b43c7e5a7.jpg)  
Figure 3: Open-ended QA performance on HealMed. a. Mean LLM-as-judge scores for ten models, macro-averaged across the three QA datasets. Blue and red points show higher- and lower-resource means; diamonds show means across all languages. b. Score shifts relative to each model’s English score. Positive values indicate higher scores than the English baseline. Grey lines show individual models and the dark line shows their mean. c. Proportions of responses with an overall score below 3, a safety score of 2 or lower, or specific language and applicability issues. Other higher-resource languages comprise English, German, Spanish, Portuguese, Japanese and Chinese. Bubble size and color indicate the proportion; values are shown when at least 10%.

The proprietary models were the highestscoring and most stable, whereas all eight nonproprietary models declined in lower-resource languages. GPT-5.4 scored 4.31 and 4.35 in higherand lower-resource languages, respectively, while Gemini-3-Flash scored 3.93 in both groups (Figure. 3a). Among the general-purpose open-source models, the largest declines occurred for Qwen2.5- 72B-Instruct and Qwen3-32B-thinking, at 1.35 and

1.20 points, respectively. The medically specialized models showed gaps ranging from 0.50 to 1.38 points.

Open-ended QA performance loss was concentrated in Swahili and Zulu and was characterized mainly by language-quality problems. For each model, the language-specific shift was calculated relative to that model’s own English score, such that positive values indicate a higher targetlanguage score. Although individual models scored slightly above their English baselines in some languages, the mean shift across models was negative for all eight translated languages. Relative to English, mean scores decreased by 0.09 to 0.27 points across the five other higher-resource languages and by 0.36 points in Thai, compared with 0.94 points in Swahili and 1.26 points in Zulu (Figure. 3b). The proportion of responses with an overall score below 3 was 10.3% in English, 17.4% across the other higher-resource languages and 22.8% in Thai, but red to 47.2% in Swahili and 59.7% in Zulu (Figure. 3c). In Swahili and Zulu, terminology problems affected 50.0% and 64.2% of responses, major fluency problems affected 39.6% and 52.7%, and wrong-language or code-switching problems affected 17.0% and 30.9%, respectively. Low safety scores were less frequent, at 16.6% and 22.5%, while demographic applicability problems remained below 3% in every group. Explicit refusals were analysed separately.

a. MCQA and NLI evaluation  
![](images/ea41fffb04c8082343c2aa7073a0e9293e5804a6a72325fcc234c81921887129.jpg)

b. Open-ended QA evaluation  
![](images/acfa32395ce2e7e2a71b4543156701eb1a03efa3532437269f21f805d3785aea.jpg)  
Figure 4: Evaluation shifts between expert-reviewed HealMed and machine-translated (MT) data. a. Englishadjusted mean accuracy shifts (HealMed minus MT) across 14 models and six MCQA and NLI datasets. Cells are labelled when $| \Delta | \geq 3$ percentage points (pp). b. English-adjusted mean LLM-as-judge score shifts across five models and three open-ended QA datasets. Symbols denote datasets, and horizontal lines span their mean shifts. In both panels, Overall reports the mean absolute model-level shift across the corresponding models and datasets. Positive values indicate higher performance on expert-reviewed data; English is shown unadjusted as a same-source control.

## 4 HealMed versus Machine-translated Data

Machine-translated benchmarks may conflate translation-related variation with differences in model capability. To quantify this effect and determine whether expert review changes conclusions about multilingual performance, we compared model performance on expert-reviewed HealMed with that on the corresponding machine-translated data. This paired analysis measured changes across languages, source datasets and the three task formats. For each translated language, we adjusted the difference between the two benchmark versions using English as a same-source control (Figure. 4).

## 4.1 MCQA and NLI

For MCQA and NLI, we evaluated all 14 models using accuracy, including five proprietary models (GPT-5.4, o4-mini, Gemini-3- Flash, Claude-Sonnet-5 and Claude-Opus-4.8), six general-purpose open-source models (DeepSeek-V3, Gemma-3-27B-it, LLaMA3.3-70B-Instruct, Qwen2.5-72B-Instruct and Qwen3-32B in thinking and non-thinking modes), and three medically specialized models (HuatuoGPT-o1-72B, MedGemma-27B and MediPhi).

Machine-translated and expert-reviewed data yielded different estimates of multilingual performance, with the largest discrepancies occurring in lower-resource languages. The English control showed a mean shift of +0.2 percentage points. After adjusting each language–dataset estimate by the corresponding English shift, the mean absolute shifts were 4.9 percentage points in Thai, 3.8 percentage points in Swahili and 5.8 percentage points in Zulu, exceeding those in every higherresource language (Figure. 4a). The effects also varied across datasets. Thai showed its largest shift on MMLU-Pro, at 6.9 percentage points. The largest NLI shifts occurred on MedNLI in Chinese (+3.4 percentage points) and Swahili (+3.3 percentage points), and on BioNLI in Zulu (+3.2 percentage points). These findings raise concerns that benchmarks relying solely on machine translation may conflate model limitations with translation artefacts and may not fully reflect performance on naturally phrased target-language medical questions.

## 4.2 Open-ended QA

For open-ended QA, the paired analysis included five models: DeepSeek-V3, Gemma-3- 27B-it, LLaMA3.3-70B-Instruct, Qwen2.5-72B-Instruct and Qwen3-32B-thinking. Responses on ExpertQA-Bio, ExpertQA-Med and LiveQA were evaluated using the LLM-as-judge composite score on a five-point scale.

Machine-translated QA data generally produced higher LLM-as-judge scores than the expert-reviewed version, although the difference depended on language and dataset. Across the eight translated languages, mean absolute Englishadjusted shifts ranged from 0.06 to 0.12 points and were largest in Chinese, as shown in Figure. 4b. Expert-reviewed scores were lower in 18 of the 24 language–dataset combinations. The largest dataset-specific difference occurred on Chinese LiveQA, for which the expert-reviewed version scored 0.15 points lower. ExpertQA-Bio scores were also 0.11 points lower in Spanish and Japanese, whereas LiveQA scores increased slightly in Japanese, Portuguese and Zulu. In particular, the predominance of lower scores raises the possibility that the LLM evaluator was more closely aligned with the literal phrasing of the machine-translated data.

Across task formats, the machine-translated and expert-reviewed versions yielded language- and dataset-dependent differences in measured performance, suggesting that machine-translated benchmarks may introduce translation-related measurement variation and may not fully represent performance on naturally phrased multilingual medical questions.

## 5 Analysis

We conducted three complementary analyses to characterize refusal behavior, machine-translation quality and the reliability of the LLM-based evaluator used for open-ended QA.

## 5.1 Failure Rates

Models may decline benchmark questions because of safety or scope restrictions, leaving users without a substantive answer. We therefore measured explicit refusal rates for four proprietary models on open-ended QA across nine languages. A response was classified as a failure only when the model explicitly declined to answer and provided no substantive information. Refusals were uncommon overall (Table 1). GPT-4o had the highest overall rate (0.44%), driven mainly by Swahili (1.50%) and Zulu (2.00%). GPT-5.4 and o4-mini had overall rates of 0.06% and 0.03%, respectively, whereas Gemini-3-Flash produced no explicit refusals. Refusal rates therefore remained low but varied across models and languages. These differences may reflect provider-specific policies as well as model behavior and should not be interpreted solely as differences in model capability.

<table><tr><td>Language</td><td>GPT-40</td><td>GPT-5.4</td><td>04-mini</td><td>Gemini-3</td></tr><tr><td>English</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>German</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Spanish</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Portuguese</td><td>0.00</td><td>0.25</td><td>0.00</td><td>0.00</td></tr><tr><td>Japanese</td><td>0.00</td><td>0.25</td><td>0.00</td><td>0.00</td></tr><tr><td>Chinese</td><td>0.25</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Thai</td><td>0.25</td><td>0.00</td><td>0.25</td><td>0.00</td></tr><tr><td>Swahili</td><td>1.50</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Zulu</td><td>2.00</td><td>0.00</td><td>0.00</td><td>0.00</td></tr><tr><td>Overall</td><td>0.44</td><td>0.06</td><td>0.03</td><td>0.00</td></tr></table>

Table 1: Language-specific refusal rates in open-ended QA. Values are percentages of 400 zero-shot responses per language. A refusal was counted only when the model explicitly declined to answer and provided no substantive response. Overall rates were calculated across 3,600 responses per model. Gemini-3 denotes Gemini-3-Flash.

## 5.2 Machine Translation Quality and Expert Revision

Mean expert ratings of the original machine translations exceeded 4.7 on a five-point scale for all three criteria, as shown in Table $2 ^ { 3 }$ . Across the eight target languages, mean scores were 4.72 for accuracy, 4.71 for fluency and 4.88 for completeness. Completeness was the highest-rated dimension in six languages. Swahili received the lowest mean scores across all three criteria, whereas Chinese showed the largest difference between fluency and completeness, with mean scores of 4.51 and 4.98, respectively. The mean word-level revision rate was 4.8% across the target languages, ranging from 1.0% for Spanish to 12.1% for German; Thai had the second-highest rate at 9.1%. For German, this estimate included formatting and encoding corrections as well as textual changes. Revision rates did not consistently track expert scores: German and Thai received relatively high ratings but underwent more editing, whereas Swahili received lower ratings but fewer textual changes.

Expert ratings capture perceived translation quality, whereas revision rates quantify textual intervention and may vary with reviewer correction thresholds and editing practices. Because separate expert groups assessed each language, cross-language differences should be interpreted descriptively rather than as calibrated rankings of translation quality.

<table><tr><td>Language</td><td>Acc.</td><td>Flu.</td><td>Comp.</td><td>Revision (%)</td></tr><tr><td>German</td><td>4.87</td><td>4.75</td><td>4.83</td><td>12.1</td></tr><tr><td>Spanish</td><td>4.92</td><td>4.92</td><td>4.85</td><td>1.0</td></tr><tr><td>Portuguese</td><td>4.74</td><td>4.84</td><td>4.93</td><td>2.7</td></tr><tr><td>Japanese</td><td>4.70</td><td>4.62</td><td>4.96</td><td>5.7</td></tr><tr><td>Chinese</td><td>4.73</td><td>4.51</td><td>4.98</td><td>5.2</td></tr><tr><td>Thai</td><td>4.71</td><td>4.79</td><td>4.93</td><td>9.1</td></tr><tr><td>Swahili</td><td>4.50</td><td>4.57</td><td>4.63</td><td>1.8</td></tr><tr><td>Zulu</td><td>4.60</td><td>4.68</td><td>4.92</td><td>1.1</td></tr><tr><td>Overall</td><td>4.72</td><td>4.71</td><td>4.88</td><td>4.8</td></tr></table>

Table 2: Expert assessment and revision of machinetranslated data by language. Accuracy (Acc.), fluency (Flu.) and completeness (Comp.) are mean expert ratings on five-point scales. Revision is the mean normalized word-level edit distance between the original machine translations and reviewer-submitted revisions. Each language contains 1,000 instances.

## 5.3 QA Evaluation Quality

To assess the LLM-based evaluator used for openended QA, we compared its scores with medical expert ratings for Chinese, Japanese and Thai responses. For each language, we selected 15 scorestratified questions, five from each QA dataset. Each question was answered by three models, yielding 45 paired evaluations per language. Medical experts and the LLM evaluator scored the same responses for completeness, reference alignment, clinical consensus, clinical appropriateness and safety using five-point scales. The five scores were averaged to obtain an overall score.

Agreement varied across the three validation subsets (Table 3). In Thai, expert and LLM scores differed by no more than 0.13 points across all five criteria. The LLM evaluator assigned lower scores for every criterion in Chinese, with the largest differences in reference alignment (−0.62 points) and safety (−0.49 points), and showed still larger differences in Japanese, particularly for reference alignment (−1.04 points), completeness (−0.76 points) and clinical appropriateness (−0.76 points). Reference alignment showed the largest discrepancy in both Chinese and Japanese, whereas clinical consensus was the most consistent criterion across the three subsets. At the response level, 95.6%, 88.9% and 64.4% of LLM scores were within one point of the expert scores in Thai, Chinese and Japanese, respectively; the corresponding concordance coefficients were 0.77, 0.56 and 0.11. These results show that LLM-as-judge evaluation does not fully reproduce medical expert assessment of open-ended QA. Although agreement was close in

<table><tr><td>Criterion</td><td>Expert</td><td>LLM</td><td>∆</td></tr><tr><td>Chinese</td><td></td><td></td><td></td></tr><tr><td>Completeness</td><td>4.22</td><td>3.80</td><td>-0.42</td></tr><tr><td>Reference alignment</td><td>4.24</td><td>3.62</td><td>-0.62</td></tr><tr><td>Clinical consensus</td><td>4.33</td><td>4.04</td><td>-0.29</td></tr><tr><td>Clinical appropriateness</td><td>4.31</td><td>3.93</td><td>-0.38</td></tr><tr><td>Safety</td><td>4.69</td><td>4.20</td><td>-0.49</td></tr><tr><td>Overall</td><td>4.36</td><td>3.92</td><td>-0.44</td></tr><tr><td>Japanese</td><td></td><td></td><td></td></tr><tr><td>Completeness</td><td>4.20</td><td>3.44</td><td>-0.76</td></tr><tr><td>Reference alignment</td><td>4.76</td><td>3.71</td><td>-1.04</td></tr><tr><td>Clinical consensus</td><td>4.82</td><td>4.20</td><td>-0.62</td></tr><tr><td>Clinical appropriateness</td><td>4.78</td><td>4.02</td><td>-0.76</td></tr><tr><td>Safety</td><td>4.87</td><td>4.27</td><td>-0.60</td></tr><tr><td>Overall</td><td>4.68</td><td>3.93</td><td>-0.76</td></tr><tr><td>Thai</td><td></td><td></td><td></td></tr><tr><td>Completeness</td><td>3.28</td><td>3.38</td><td>+0.10</td></tr><tr><td>Reference alignment</td><td>3.64</td><td>3.62</td><td>-0.02</td></tr><tr><td>Clinical consensus</td><td>4.12</td><td>4.11</td><td>-0.01</td></tr><tr><td>Clinical appropriateness</td><td>3.80</td><td>3.91</td><td>+0.11</td></tr><tr><td>Safety</td><td>4.36</td><td>4.22</td><td>-0.13</td></tr><tr><td>Overall</td><td>3.84</td><td>3.85</td><td>+0.01</td></tr></table>

Table 3: Criterion-level comparison of expert and LLMbased evaluations. Scores range from 1 to 5. ∆ denotes the LLM score minus the expert score and was calculated before rounding.

Thai, systematic differences remained in Chinese and Japanese, particularly for reference alignment. Within these validation subsets, LLM-based and human evaluation were therefore not interchangeable. Representative response-level comparisons between the LLM evaluator and medical experts are provided in Appendix G.

## 6 Discussion

## 6.1 What Language Does a Multilingual Model Think In?

A multilingual model does not necessarily reason in a single fixed language, nor does its reasoning language always match the language of the input. We observe that HuatuoGPT-o1-72B reasons predominantly in Japanese when answering Japanese questions, but continues to reason in English when responding to Thai questions. We examined 15 open-ended QA responses produced by HuatuoGPT-o1-72B in each of Japanese, Chinese and Thai. An explicit reasoning trace was present in 9 Japanese, 15 Chinese and 7 Thai responses, as shown in Table 4. Among responses with an observable trace, 7 of 9 Japanese traces (77.8%) were predominantly in Japanese and all 15 Chinese traces were predominantly in Chinese. By contrast, all 7 observable Thai traces were predominantly in English, even when the final answer was returned in Thai. Japanese traces commonly began with Japanese reasoning markers, whereas Thai traces frequently began with English expressions such as “Okay, let’s think”. This asymmetric behavior suggests that reasoning-language selection may depend not only on the target language, but also on factors such as language-specific training exposure and the strength of the model’s learned reasoning representations in that language (Zhong et al., 2025). Multilingual capability should therefore be understood as more than the ability to comprehend and generate answers across languages: it also concerns whether a model can carry out the underlying reasoning process in those languages.

<table><tr><td>Language</td><td>Traces</td><td>Target</td><td>English</td></tr><tr><td>Japanese</td><td>9/15</td><td>7 (77.8%)</td><td>2 (22.2%)</td></tr><tr><td>Chinese</td><td>15/15</td><td>15 (100%)</td><td>0 (0%)</td></tr><tr><td>Thai</td><td>7/15</td><td>0 (0%)</td><td>7 (100%)</td></tr></table>

Table 4: Language use in observable reasoning traces generated by HuatuoGPT-o1-72B. The “Traces” column denotes the number of responses containing an explicit reasoning trace among 15 responses examined per language. ”Target” and ”English” report the number and percentage of traces written predominantly in the target language or English, respectively.

## 6.2 Language-specific Conventions for Translating Medical Terminology

Our observation suggests that translating medical terminology is not a binary choice between preserving the English expression and replacing it with an equivalent in the target language. Instead, preferred usage depends on languagespecific clinical conventions, the type of term, and the availability of a widely accepted local equivalent. The clinicians consulted in this study described markedly different practices across languages. The Japanese clinician generally preferred medical terms to be translated into Japanese.

However, English is commonly retained for investigations, including laboratory tests such as full blood count (FBC) and liver function tests (LFTs), and imaging modalities such as CT and MRI, because these forms are shorter and commonly used in clinical communication. This preference is not universal across investigations. Thai clinical usage showed different patterns of selective English retention. Moreover, when a concept occurs only once, spelling out the term may be preferable to introducing an acronym in either language. In Thai clinical communication, the retention of English terminology is more extensive. According to the Thai clinician, many technical terms lack an authoritative Thai translation, making their English forms more standardized and less ambiguous. Transliterating long chemical or technical names into Thai may also reduce rather than improve comprehensibility, as clinicians may need to reconstruct the original English term to recognize the concept.

Translation Quality Depends on Clinical Convention. These observations indicate that revisions from translated terminology back to English should not automatically be interpreted as corrections of translation errors. They may instead reflect adaptation to the linguistic norms of clinical practice in the target language. Consequently, a translation pipeline that uniformly prioritizes target-language rendering may produce linguistically complete translations that nevertheless appear unnatural or inefficient to clinicians. Medical machine-translation systems may therefore benefit from language- and term-specific policies that account for established local equivalents, acronym conventions, term category, frequency of occurrence, and the communicative setting. The same considerations should inform human evaluation: assessments of terminology should distinguish semantic accuracy from conformity to local clinical usage rather than treating the proportion of translated terms as a direct measure of translation quality. Because these patterns were derived from feedback from a limited number of clinicians, however, they should be interpreted as qualitative observations and validated with a larger and more diverse group of practitioners.

## 7 Conclusion

We present HealMed, an expert-reviewed medical benchmark spanning nine languages and three task formats. Across 14 LLMs, multilingual performance varied substantially by language and model; the strongest proprietary models were generally more stable, whereas medical specialization did not ensure multilingual robustness. Machine-translated and expert-reviewed data also yielded different performance estimates. Cross-language performance differences should therefore not be attributed solely to model capability without accounting for translation quality.

## 8 Limitations

## 8.1 Translation-Specific Evaluation

Our evaluation primarily measures downstream task accuracy and QA quality rather than translation quality directly. Although the evaluation framework includes general criteria such as correctness, these criteria were not specifically designed to capture the distinctive properties of medical machine translation. Consequently, the current evaluation may not adequately reflect dimensions such as terminology consistency, clinical naturalness, preservation of medically relevant nuance, and conformity to language-specific clinical conventions. A translation may therefore support an accurate answer while still containing linguistic or terminological shortcomings, whereas a clinically appropriate translation may receive limited recognition from task-oriented metrics. Future work should develop and validate medical translation–specific evaluation criteria that assess both semantic fidelity and appropriateness in the clinical context, ideally incorporating judgments from clinicians and professional medical translators across languages.

## 8.2 Scope and Scale of HealMed

HealMed was designed to isolate multilingual medical performance in controlled, single-turn settings. By spanning MCQA, NLI and open-ended QA, it enables aligned comparisons across languages and direct measurement of how expert revision changes performance estimates. This scope differs from HealthBench and HealthBench-Pro, which emphasize multi-turn healthcare conversations (Arora et al., 2025; Hicks et al., 2026). HealMed does not evaluate dialogue-level capabilities such as maintaining clinical context across turns or responding to evolving user needs, and should therefore be viewed as complementary to conversational healthcare benchmarks. The overall size of the dataset is also relatively limited, which may constrain its coverage of medical specialties, clinical contexts, and linguistic variation. Nevertheless, HealMed provides a practical step towards the systematic study of multilingual medical large language models. Importantly, the proposed pipeline is extensible and could be applied to more complex benchmarks, including the HealthBench series. Future work could therefore expand both the scale and clinical complexity of HealMed to support a more comprehensive evaluation of multilingual medical reasoning and communication.

## 8.3 Human-evaluation Criteria

The scope of the human evaluation was limited by the practical difficulty of recruiting clinicians, whose availability is constrained by demanding clinical workloads. Consequently, we could not conduct human evaluation for every language or perform comprehensive case studies across the entire dataset. Instead, we evaluated selected samples in a subset of languages. Although these analyses provide useful qualitative insights into translation quality and language-specific clinical conventions, they may not fully represent the range of specialties, linguistic preferences, and regional practices within each language. Future work should involve larger and more diverse panels of clinicians and extend human evaluation across additional languages, medical domains, and case types.

## References

Asma Ben Abacha, Eugene Agichtein, Yuval Pinter, and Dina Demner-Fushman. 2017. Overview of the medical question answering task at trec 2017 liveqa. In TREC, volume 1, page 12.

Kabir Ahuja, Harshita Diddee, Rishav Hada, Millicent Ochieng, Krithika Ramesh, Prachi Jain, Akshay Nambi, Tanuja Ganu, Sameer Segal, Mohamed Ahmed, Kalika Bali, and Sunayana Sitaram. 2023. MEGA: Multilingual evaluation of generative AI. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 4232–4267, Singapore. Association for Computational Linguistics.

Iñigo Alonso, Maite Oronoz, and Rodrigo Agerri. 2024. Medexpqa: Multilingual benchmarking of large language models for medical question answering. Artificial intelligence in medicine, 155:102938.

Anthropic. 2026a. Claude Opus 4.8 System Card. System card. Accessed 17 August 2026.

Anthropic. 2026b. Claude Sonnet 5 System Card. System card. Accessed 17 August 2026.

Rahul K. Arora, Jason Wei, Rebecca Soskin Hicks, Preston Bowman, Joaquin Quiñonero-Candela, Foivos Tsimpourlas, Michael Sharman, Meghan Shah, Andrea Vallone, Alex Beutel, Johannes Heidecke, and Karan Singhal. 2025. Healthbench: Evaluating large language models towards improved human health. arXiv preprint arXiv:2505.08775.

Mikel Artetxe, Gorka Labaka, and Eneko Agirre. 2020. Translation artifacts in cross-lingual transfer learning. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7674–7684.

Mohaddeseh Bastan, Mihai Surdeanu, and Niranjan Balasubramanian. 2022. Bionli: Generating a biomedical nli dataset using lexico-semantic constraints for adversarial examples. In Findings ofthe Association for Computational Linguistics: EMNLP 2022, pages 5093–5104.

Junying Chen, Zhenyang Cai, Ke Ji, Xidong Wang, Wanlong Liu, Rongsheng Wang, and Benyou Wang. 2025a. Towards Medical Complex Reasoning with LLMs through Medical Verifiable Problems. In Findings of the Association for Computational Linguistics: ACL 2025, pages 14552–14573, Vienna, Austria. Association for Computational Linguistics.

Qingyu Chen, Yan Hu, Xueqing Peng, Qianqian Xie, Qiao Jin, Aidan Gilson, Maxwell B. Singer, Xuguang Ai, Po-Ting Lai, Zhizheng Wang, Vipina K. Keloth, Kalpana Raja, Jimin Huang, Huan He, Fongci Lin, Jingcheng Du, Rui Zhang, W. Jim Zheng, Ron A. Adelman, and 2 others. 2025b. Benchmarking large language models for biomedical natural language processing applications and recommendations. Nature Communications, 16:3280.

Jean-Philippe Corbeil, Amin Dada, Jean-Michel Attendu, Asma Ben Abacha, Alessandro Sordoni, Lucas Caccia, Francois Beaulieu, Thomas Lin, Jens Kleesiek, and Paul Vozila. 2025. A Modular Approach for Clinical SLMs Driven by Synthetic Data with Pre-Instruction Tuning, Model Merging, and Clinical-Tasks Alignment. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 19352–19374, Vienna, Austria. Association for Computational Linguistics.

DeepSeek-AI. 2024. DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437.

Fan Gao, Sherry T. Tong, Jiwoong Sohn, Jiahao Huang, Junfeng Jiang, Ding Xia, Piyalitt Ittichaiwong, Kanyakorn Veerakanjana, Hyunjae Kim, Qingyu Chen, Edison Marrese Taylor, Kazuma Kobayashi, Akiko Aizawa, and Irene Li. 2026. Med-CoReasoner: Reducing language disparities in medical reasoning via language-informed co-reasoning. arXiv preprint arXiv:2601.08267.

Gemma Team, Aishwarya Kamath, Johan Ferret, et al. 2025. Gemma 3 Technical Report. arXiv preprint arXiv:2503.19786.

Google DeepMind. 2025. Gemini 3 Flash: Model Card. Model card. Accessed 17 August 2026.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, et al. 2024. The Llama 3 Herd of Models. arXiv preprint arXiv:2407.21783.

Rebecca Soskin Hicks, Mikhail Trofimov, Dominick Lim, Rahul K. Arora, Foivos Tsimpourlas, Preston Bowman, Michael Sharman, Chi Tong, Kavin Karthik, Arnav Dugar, Akshay Jagadeesh, Khaled Saab, Johannes Heidecke, Ashley Alexander, Nate

Gross, and Karan Singhal. 2026. Healthbench professional: Evaluating large language models on real clinician chats. arXiv preprint arXiv:2604.27470.

Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits. 2021. What disease does this patient have? a large-scale open domain question answering dataset from medical exams. Applied Sciences, 11(14):6421.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. 2020. The state and fate of linguistic diversity and inclusion in the NLP world. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 6282–6293. Association for Computational Linguistics.

Chaitanya Malaviya, Subin Lee, Sihao Chen, Elizabeth Sieber, Mark Yatskar, and Dan Roth. 2024. Expertqa: Expert-curated questions and attributed answers. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3025–3045.

João Matos, Shan Chen, Siena Kathleen V. Placino, Yingya Li, Juan Carlos Climent Pardo, Daphna Idan, Takeshi Tohyama, David Restrepo, Luis Filipe Nakayama, José María Millet Pascual-Leone, Guergana K. Savova, Hugo Aerts, Leo Anthony Celi, An-Kwok Ian Wong, Danielle Bitterman, and Jack Gallifant. 2025. WorldMedQA-V: A multilingual, multimodal medical examination dataset for multimodal language models evaluation. In Findings ofthe Association for Computational Linguistics: NAACL 2025, pages 7218–7231, Albuquerque, New Mexico. Association for Computational Linguistics.

OpenAI. 2025. OpenAI o3 and o4-mini System Card. System card. Accessed 17 August 2026.

OpenAI. 2026. GPT-5.4 Thinking System Card. System card. Accessed 17 August 2026.

Pengcheng Qiu, Chaoyi Wu, Xiaoman Zhang, Weixiong Lin, Haicheng Wang, Ya Zhang, Yanfeng Wang, and Weidi Xie. 2024. Towards building multilingual language model for medicine. Nature Communications, 15(1):8384.

Alexey Romanov and Chaitanya Shivade. 2018. Lessons from natural language inference in the clinical domain. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 1586–1596.

Andrew Sellergren, Sahar Kazemzadeh, Tiam Jaroensri, et al. 2025. MedGemma Technical Report. arXiv preprint arXiv:2507.05201.

Shivalika Singh, Angelika Romanou, Clémentine Fourrier, David Ifeoluwa Adelani, Jian Gang Ngui, Daniel Vila-Suero, Peerat Limkonchotiwat, Kelly Marchisio, Wei Qi Leong, Yosephine Susanto, et al. 2025. Global mmlu: Understanding and addressing cultural

and linguistic biases in multilingual evaluation. In Proceedings of the 63rd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 18761–18799.

Karan Singhal, Shekoofeh Azizi, Tao Tu, S Sara Mahdavi, Jason Wei, Hyung Won Chung, Nathan Scales, Ajay Tanwani, Heather Cole-Lewis, Stephen Pfohl, et al. 2023. Large language models encode clinical knowledge. Nature, 620(7972):172–180.

Karan Singhal, Tao Tu, Juraj Gottweis, Rory Sayres, Ellery Wulczyn, Mohamed Amin, Le Hou, Kevin Clark, Stephen R Pfohl, Heather Cole-Lewis, et al. 2025. Toward expert-level medical question answering with large language models. Nature medicine, 31(3):943–950.

David Vilares and Carlos Gómez-Rodríguez. 2019. Head-qa: A healthcare dataset for complex reasoning. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 960–966.

Xidong Wang, Nuo Chen, Junyin Chen, Yidong Wang, Guorui Zhen, Chunxian Zhang, Xiangbo Wu, Yan Hu, Anningzhe Gao, Xiang Wan, Haizhou Li, and Benyou Wang. 2024a. Apollo: A lightweight multilingual medical LLM towards democratizing medical AI to 6b people. arXiv preprint arXiv:2403.03640.

Yubo Wang, Xueguang Ma, Ge Zhang, Yuansheng Ni, Abhranil Chandra, Shiguang Guo, Weiming Ren, Aaran Arulraj, Xuan He, Ziyan Jiang, et al. 2024b. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266– 95290.

Jiageng Wu, Bowen Gu, Ren Zhou, Kevin Xie, Doug Snyder, Yixing Jiang, Valentina Carducci, Richard Wyss, Rishi J Desai, Emily Alsentzer, et al. 2026. Bridge: benchmarking large language models for understanding real-world clinical practice texts. Nature Biomedical Engineering.

Minghao Wu, Weixuan Wang, Sinuo Liu, Huifeng Yin, Xintong Wang, Yu Zhao, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2025. The bitter lesson learned from 2,000+ multilingual benchmarks. Preprint, arXiv:2504.15521.

Weihao Xuan, Rui Yang, Heli Qi, Qingcheng Zeng, Yunze Xiao, Aosong Feng, Dairui Liu, Yun Xing, Junjue Wang, Fan Gao, Jinghui Lu, Yuang Jiang, Huitao Li, Xin Li, Kunyu Yu, Ruihai Dong, Shangding Gu, Yuekang Li, Xiaofei Xie, and 13 others. 2025. MMLU-ProX: A multilingual benchmark for advanced large language model evaluation. In Proceedings ofthe 2025 Conference on Empirical Methods in Natural Language Processing, pages 1513–1532, Suzhou, China. Association for Computational Linguistics.

An Yang, Anfeng Li, Baosong Yang, et al. 2025. Qwen3 Technical Report. arXiv preprint arXiv:2505.09388.

An Yang, Baosong Yang, Beichen Zhang, et al. 2024. Qwen2.5 Technical Report. arXiv preprint arXiv:2412.15115.

Chengzhi Zhong, Qianying Liu, Fei Cheng, Junfeng Jiang, Zhen Wan, Chenhui Chu, Yugo Murawaki, and Sadao Kurohashi. 2025. What language do non-English-centric large language models think in? In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 26333–26346, Vienna, Austria. Association for Computational Linguistics.

## Contributions

Researchers   
Yingjian Chen<sup>†</sup> University of Tokyo, Japan   
Fan Gao<sup>†</sup> University of Tokyo, Japan   
Sherry T. Tong University of Tokyo, Japan   
Haoyu Zhang University of Tokyo, Japan   
Aosong Feng Yale University, USA   
Kevin W. Jin Yale University, USA   
Xing Wu University of California, Berkeley, USA   
Jinghui Lu Smartor AI, Japan   
Michihiro Yasunaga Stanford University, USA   
Rex Ying Yale University, USA   
Heuiseok Lim Korea University, South Korea   
Jaewoo Kang Korea University, South Korea   
Chanjun Park Soongsil University, South Korea   
Hang Jiang Northeastern University, MIT, Harvard, USA   
Ethan Goh Stanford University, USA   
Hyunjae Kim Yale University, USA   
Edison Marrese-Taylor University of Tokyo, Japan   
Yusuke Iwasawa University of Tokyo, Japan   
Yutaka Matsuo University of Tokyo, Japan   
Qingyu Chen Yale University, USA   
Irene Li<sup>∗</sup> University of Tokyo, Japan

<sup>†</sup> Equal contribution.

Corresponding author. irene.li@weblab.t.u-tokyo.ac.jp

## Medical Experts

Abdul Samad Health Lab & Diagnostic Centre, Patherdewa, Deoria, Uttar Pradesh, India

Akbar Faruqi Lake Erie College of Osteopathic Medicine, Erie, Pennsylvania, USA

Cesar Caraballo Yale University, USA

Cibele Brandão Hospital de Clínicas da Universidade Federal do Paraná (UFPR), Brazil

Dhruva (Drew) Gupta Department of Medicine, Cambridge Health Alliance; Harvard Medical School, Boston, Massachusetts, USA

Eunji Jeon Mayo Clinic, USA

Gabriel Madera-Santiago University of Puerto Rico Medical Sciences Campus; BSc. Human Biology, University of Puerto Rico–Bayamón, USA

Geon Lee GU Clinic, South Korea

Hugo Toshio Itikawa Ophthalmology Resident, University of São Paulo; Noroeste do Paraná Eye Hospital, Brazil

Insook Cho Inha University, South Korea

Isabelli Martins University of Chicago, USA

Isarar Siddique Biotech Wallah Pvt Ltd, India

Israr Ahmed Health Lab & Diagnostic Centre, Patherdewa, Deoria, Uttar Pradesh, India

Jihyo Kwak Mayo Clinic, USA

Kanyakorn Veerakanjana Siriraj Informatics and Data Innovation Center (SiData+), Faculty of Medicine Siriraj Hospital, Mahidol University, Thailand

Luis Guilherme Cardoso Physician, Universidade Federal do Paraná (UFPR), Curitiba, Brazil

Minjin Kim Northgate Health Centre / Oxford University Hospitals NHS Foundation Trust, UK

Piyalitt Ittichaiwong Siriraj Informatics and Data Innovation Center (SiData+), Faculty of Medicine Siriraj Hospital, Mahidol University, Thailand

Renee Dua MD, Valley Renal Medical Group, Northridge, California, USA

Santiago Gudiño-Rosales University of California, Riverside School of Medicine, USA

Xiujie Chen Computational Biology and Medical Sciences, Graduate School of Frontier Sciences, The University of Tokyo, Japan

Zeo Lapalus University of Montreal, Canada

Zixin Xu Dokkyo Medical University, Japan

## A HealMed dataset details

## A.1 Source datasets and sample allocation

HealMed integrates nine benchmark components spanning multiple-choice question answering (MCQA), natural language inference (NLI) and open-ended question answering (QA). The source datasets for each task are described below, and their sample allocation is summarized in Table 5.

Multiple-choice question answering (MCQA). The MCQA component draws from four datasets. HeadQA consists of questions from examinations for access to specialized positions in the Spanish healthcare system and was introduced to evaluate complex reasoning in healthcare question answering (Vilares and Gómez-Rodríguez, 2019). MedQA contains questions from professional medical board examinations in the United States, mainland China and Taiwan, originally provided in English, Simplified Chinese and Traditional Chinese (Jin et al., 2021). MedExpQA is based on commented questions from the Spanish MIR examinations and includes gold explanations written by medical doctors, with explanation spans linked to individual answer options where available (Alonso et al., 2024). MMLU-Pro is a reasoning-focused extension of MMLU that introduces more complex questions, expands the number of answer options and removes questions identified as trivial or noisy (Wang et al., 2024b).

Natural language inference (NLI). The NLI component comprises BioNLI and MedNLI. BioNLI pairs biomedical mechanisms with experimental evidence extracted from scientific abstracts; positive examples represent entailment, whereas adversarial negative examples are constructed using rule-based and constrained generation strategies (Bastan et al., 2022). MedNLI is a physician-annotated clinical inference dataset in which premises are drawn from the past medical history sections of MIMIC-III clinical notes and paired with hypotheses expressing entailment, contradiction or neutral relations (Romanov and Shivade, 2018).

Open-ended question answering (QA). The QA component comprises ExpertQA-Bio, ExpertQA-Med and LiveQA. ExpertQA-Bio and ExpertQA-Med are the biology and medicine subsets of ExpertQA, respectively. They contain expert-authored questions and long-form responses evaluated and revised by domain experts, together with supporting evidence attributions (Malaviya et al., 2024). LiveQA originates from the medical QA task of the TREC 2017 LiveQA track and contains consumer health questions received by the US National Library of Medicine, with reference answers collected manually from trusted health-information sources (Abacha et al., 2017).

<table><tr><td>Task</td><td>Component</td><td>Samples per language, n</td><td>Expected model output</td></tr><tr><td rowspan="5">MCQA</td><td>HeadQA</td><td>75</td><td>Correct option</td></tr><tr><td>MedQA</td><td>75</td><td>Correct option</td></tr><tr><td>MedExpQA</td><td>75</td><td>Correct option</td></tr><tr><td>MMLU-Pro</td><td>75</td><td>Correct option</td></tr><tr><td>BioNLI</td><td>150</td><td>Relation label</td></tr><tr><td rowspan="3">NLI QA</td><td>MedNLI</td><td>150</td><td>Relation label</td></tr><tr><td>ExpertQA-Bio</td><td>40</td><td>Free-form answer</td></tr><tr><td>ExpertQA-Med</td><td>160</td><td>Free-form answer</td></tr><tr><td></td><td>LiveQA</td><td>200</td><td>Free-form answer</td></tr><tr><td>Total</td><td></td><td>1,000</td><td></td></tr></table>

Table 5: Benchmark components and sample allocation in HealMed. Sample counts denote the number of aligned examples included in each language.

## A.2 Task and data representation

Each HealMed instance contains six fields: id, language, task, source\_dataset, input and target. The id identifies aligned versions of the same source example across languages, whereas language specifies the language of the instance. The task field takes one of three values, MCQA, NLI or QA, and source\_dataset records the corresponding benchmark component.

For MCQA, the input field contains a question and a fixed set of labelled answer options, and target contains the correct option label. For NLI, input contains a premise, a hypothesis and the available relation labels, whereas target identifies the correct relation. For open-ended QA, input contains the medical question and target contains a free-form reference answer. Representative English records are shown below. The NLI premise and selected long-form answers are shortened for presentation; the released dataset retains the complete text.

```json
[
{
"id": "mcqa-headqa-0000",
"language": "en",
"task": "MCQA",
"source_dataset": "HeadQA",
"input": "Question: Motor end-plate is the junction between the motor neuron and the:\nOptions:\nA: Smooth muscle\nB: Skeletal
muscle\nC: Cardiac muscle\nD: Muscle spindle (Muscle spindle organ)\nE: Tendon",
"target": "B"
},<sub>{</sub>
"id": "nli-bionli-0060",
"language": "en",
"task": "NLI",
"source_dataset": "BioNLI",
"input": "Premise: We examined the ability of sucralfate to prevent secretagogue-induced duodenal ulcer in the rat. [...]\
nHypothesis: We conclude that tubastatin A prevents the formation of secretagogue-induced duodenal ulcer in the rat.\
nOptions:\nA: Entailment\nB: Contradiction",
"target": "B"
},
{
"id": "qa-expertqa-med-0058",
"language": "en",
"task": "QA",
"source_dataset": "ExpertQA-Med",
"input": "How can you manage your patient expectations?",
"target": "To manage patient expectations, follow these steps: develop rapport and build a trusting therapeutic relationship
with your patients. [...]"
}
]
```

## B Comparison with existing multilingual medical benchmarks

Multilingual medical benchmarks differ not only in scale and task coverage but also in the degree of cross-language experimental control that they provide. Large benchmarks assembled from existing clinical datasets offer broad coverage of tasks and clinical settings, whereas aligned translation-based benchmarks enable controlled comparisons in which the underlying content is held constant across languages. We therefore compared HealMed with six representative multilingual medical benchmarks: MMedBench (Qiu et al., 2024), XMedBench (Wang et al., 2024a), MedExpQA (Alonso et al., 2024), WorldMedQA-V (Matos et al., 2025), MultiMed-X (Gao et al., 2026) and BRIDGE (Wu et al., 2026). The comparison separates dataset scale from three design features directly relevant to multilingual evaluation: cross-language alignment, the extent of human review and whether translation quality and its effect on model performance were quantified (Table 6).

The comparison highlights complementary benchmark designs rather than a ranking based on dataset size. BRIDGE provides substantially greater scale and task diversity by harmonizing 87 tasks from 59 realworld clinical-text datasets. Its language-specific tasks, however, originate from heterogeneous sources and do not form a parallel benchmark in which the same content is evaluated across languages. HealMed instead uses an aligned design that holds source content and task allocation constant across languages. HealMed has two related aims. First, it provides a multilingual medical benchmark in which every translated instance is evaluated and revised by bilingual medical experts. Second, it examines whether benchmarks constructed using machine translation provide faithful estimates of models’ target-language medical capabilities. Retaining both the original machine-translated and expert-reviewed versions enables direct measurement of how expert revision changes model scores and conclusions across languages and datasets. Expert review therefore serves not only as a curation step, but also as the basis for auditing translation-related measurement bias. By evaluating the same models on matched machine-translated and expert-reviewed versions, HealMed quantifies how translation changes estimated language gaps and model comparisons.

<table><tr><td></td><td colspan="3">Benchmark scope</td><td>Data construction</td><td colspan="2">Human review</td><td colspan="2">Translation audit</td></tr><tr><td>Benchmark</td><td>Reported scale</td><td>Lang.</td><td>Tasks</td><td>Language design</td><td>Review coverage (%)</td><td>Reviewers, n</td><td>Quality metrics</td><td>Score effects</td></tr><tr><td>MMedBench</td><td>53,566 QA pairs</td><td>6</td><td>MCQA; rationales</td><td>Aggregated; non-aligned</td><td>14.1</td><td>3</td><td>No</td><td>No</td></tr><tr><td>XMedBench</td><td>21,326 records</td><td>6</td><td>MCQA</td><td>Native + MT; non-aligned</td><td>10.2</td><td>NR</td><td>No</td><td>No</td></tr><tr><td>MedExpQA</td><td>2,488 records</td><td>4</td><td>MCQA</td><td>MT + manual revision; aligned</td><td>100</td><td>NR</td><td>No</td><td>No</td></tr><tr><td>WorldMedQA-V</td><td>568 evaluation</td><td>4+EN</td><td>Multimodal</td><td>Local–English pairs</td><td>100</td><td>4 country 7EN</td><td>No</td><td>No</td></tr><tr><td>MultiMed-X</td><td>items 2,450 translations 7 + EN</td><td></td><td>MCQA NLI; open QA</td><td>MT + expert revision;</td><td>100</td><td>~12</td><td>No</td><td>No</td></tr><tr><td>BRIDGE</td><td>1,418,042</td><td>9</td><td>8 task types</td><td>aligned Native-source aggregation;</td><td>Varies by</td><td>NR</td><td>N/A</td><td>N/A</td></tr><tr><td>HealMed</td><td>samples 9,000 instances</td><td>9</td><td>MCQA; NLI; open QA</td><td>non-aligned MT + two-expert medical revision; aligned</td><td>source; NR 100</td><td>23</td><td>Yes</td><td>Yes</td></tr></table>

Table 6: Comparison of selected multilingual medical benchmarks. Reported scales use source-specific units and are not directly comparable. Quality metrics indicates whether aggregate translation-quality scores or revision statistics were reported; score effects indicates whether model performance was compared between machine-translated and expert-reviewed versions. WorldMedQA-V identified four country-level validators and seven contributors to English-translation validation, but did not report the number of unique reviewers because these roles may overlap. EN, English; MT, machine translation; N/A, not applicable; NR, not reported. For BRIDGE, reference standards were inherited from the source datasets, benchmark-wide human-review coverage and reviewer numbers were not reported.

## C Model details

We evaluated 14 models: five proprietary models, six general-purpose open-source models and three medically specialized open-source models. The models differed in scale and specialization. We compared overall performance and cross-language stability across the three groups and examined whether medical specialization was associated with multilingual stability. Qwen3-32B was evaluated in both thinking and non-thinking modes, which were treated as separate models throughout the analysis. Table 7 lists the model type, publicly reported parameter count and initial release date for each model. We obtained this information from official model cards, release announcements and model repositories. For models whose developers did not disclose parameter counts, the table reports Not disclosed.

<table><tr><td>Model</td><td>Parameters</td><td>Release date</td><td>Model type</td></tr><tr><td colspan="4">Proprietary models</td></tr><tr><td>GPT-5.4</td><td>Not disclosed</td><td>5 Mar 2026</td><td>Proprietary</td></tr><tr><td>04-mini</td><td>Not disclosed</td><td>16 Apr 2025</td><td>Proprietary</td></tr><tr><td>Gemini-3-Flash</td><td>Not disclosed</td><td>17 Dec 2025</td><td>Proprietary</td></tr><tr><td>Claude-Sonnet-5</td><td>Not disclosed</td><td>30 Jun 2026</td><td>Proprietary</td></tr><tr><td>Claude-Opus-4.8</td><td>Not disclosed</td><td>28 May 2026</td><td>Proprietary</td></tr><tr><td colspan="4">Open-source models</td></tr><tr><td>DeepSeek-V3</td><td>671B</td><td>26 Dec 2024</td><td>open-source, general-purpose</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>72.7B</td><td>19 Sep 2024</td><td>open-source, general-purpose</td></tr><tr><td>Qwen3-32B</td><td>32.8B</td><td>29 Apr 2025</td><td>open-source, general-purpose</td></tr><tr><td>Qwen3-32B-thinking</td><td>32.8B</td><td>29 Apr 2025</td><td>open-source, general-purpose</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>70B</td><td>6 Dec 2024</td><td>open-source, general-purpose</td></tr><tr><td>Gemma-3-27B-it</td><td>27B</td><td>12 Mar 2025</td><td>open-source, general-purpose</td></tr><tr><td colspan="4">Medically specialized models</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>72.7B</td><td>28 Dec 2024</td><td>open-source, medical</td></tr><tr><td>MedGemma-27B-text-it</td><td>27B</td><td>20 May 2025</td><td>open-source, medical</td></tr><tr><td>MediPhi</td><td>3.8B</td><td>3 Feb 2025</td><td>open-source, medical</td></tr></table>

Table 7: Models and evaluation configurations used in HealMed.

## D LLM-as-judge evaluation protocol

## D.1 Evaluation procedure

We used GPT-5.5 as an LLM judge to evaluate open-ended responses. For each instance, the judge received the target language, the expert-verified target-language question, the expert-verified reference answer and the model-generated response. The rubric was informed by the human-evaluation framework used in MultiMedQA and Med-PaLM (Singhal et al., 2023).

The evaluation comprised two reference-based dimensions and three reference-free clinical dimensions. Completeness assessed whether the response contained the essential information required to answer the question. Reference alignment assessed whether the response remained consistent with the reference answer; this dimension was implemented in the prompt as a reference-deviation score, with higher values indicating less problematic deviation. Clinical consensus assessed consistency with established scientific and clinical knowledge. Clinical appropriateness assessed whether the response contained incorrect, misleading, irrelevant or clinically unhelpful content. Safety assessed the likelihood and severity of harm if the response were followed or relied upon; this dimension was implemented as a potential-harm score, with higher values indicating lower potential harm.

Each dimension was scored from 1 to 5, with higher scores indicating better response quality. The overall score was calculated as the arithmetic mean of the five dimension scores. Four auxiliary issues were evaluated separately: wrong language or code-switching, inappropriate medical terminology, major fluency problems and demographic applicability. Each issue was labelled as yes, maybe or no and mapped to an issue indicator of 1, 0.5 or 0, respectively. These indicators were excluded from the overall score.

The judge was queried through the Azure OpenAI Chat Completions API with a maximum completion length of 4,096 tokens. Temperature and nucleus-sampling parameters were not explicitly specified. The evaluator returned structured JSON containing the five scores, their justifications and the four auxiliary flags.

## D.2 Complete evaluation prompt

The complete system and user messages used for the LLM-based evaluation are reproduced below. For each response, the placeholders were replaced with the target language, expert-reviewed target-language question, expert-verified reference answer and model-generated answer. The role headings identify the system and user messages and were not included in the message content. The original prompt uses the terms reference deviation and potential harm. Both scoring scales are positively oriented: higher scores indicate closer agreement with the reference answer and a lower risk of harm. For clarity, these dimensions are referred to in the main text as reference alignment and safety, respectively.

SYSTEM MESSAGE

You are a careful medical answer evaluator. Return only valid JSON.

USER MESSAGE

You are a medical answer evaluator.

Your task is to evaluate an LLM-generated answer to a medical question in a target language. The evaluation should follow a clinician-style rubric inspired by the human evaluation framework used in the MultiMedQA / Med-PaLM study.

You will be given:

\- The target language.

\- An expert-verified target-language question.

\- An expert-verified target-language reference answer.

\- An LLM-generated target-language answer.

Evaluate the LLM-generated answer using the following three-part framework.

1. Reference-Based Evaluation

Evaluate how well the LLM-generated answer answers the medical question and remains consistent with the expert-verified reference answer. Do not rely on surface wording, length, style, or whether the answer uses the same structure as the reference. Focus on whether the answer captures the clinically important core meaning and whether any added information is relevant, medically appropriate, and non-misleading.

Assess this using the following two dimensions.

## 1.1 Completeness

## Question:

Does the LLM-generated answer provide a sufficiently complete answer to the medical question, using the expert-verified reference answer as guidance?

## Score:

5 = Complete answer. Directly answers the question and provides all essential medical information needed for a clinically appropriate response. It does not need to include every detail from the reference answer if the omitted details are not necessary for answering the question.

4 = Mostly complete answer. Answers the question correctly and includes the main clinically important information, with only minor omissions that do not meaningfully affect usefulness, interpretation, or safety.

3 = Partially complete answer. Provides a generally relevant and partly correct answer, but omits some information that would be useful for a complete or confident response.

2 = Incomplete answer. Addresses the question only weakly, omits major clinically important information, or is too vague to be reliably useful.

1 = Minimal or no useful answer. Does not answer the question, misses the central medical point, or is mostly irrelevant.

## 1.2 Reference Deviation

## Question:

Does the LLM-generated answer contain content that contradicts the expert-verified reference answer, misleads the user, or meaningfully reduces the clinical usefulness of the answer?

## Score:

5 = No problematic deviation. The answer is consistent with the reference. Extra information, if present, is medically appropriate and does not reduce answer quality.

4 = Minor deviation only. Contains small elaborations, simplifications, or minor tangential details that do not change the medical meaning.

3 = Some deviation. Contains extra or imprecise content, but the answer remains broadly consistent with the reference and clinically usable.

2 = Meaningful deviation. Contains misleading, off-target, unsupported, or conflicting content that could reduce usefulness or affect interpretation.

1 = Major contradiction or distortion. Directly contradicts the reference, reverses the intended meaning, or gives clearly incorrect medical guidance.

Reference-based evaluation guidance:

\- Treat the reference answer as the primary anchor for the expected answer, but do not require identical wording, structure, or level of detail.

\- Do not require the answer to be as long, detailed, or comprehensive as the reference answer. - Do not penalize concise answers if they correctly answer the question and preserve the clinically important meaning.

\- Do not penalize additional medically correct and question-relevant information solely because it is absent from the reference answer.

\- Do not treat every omitted reference detail as an error. Penalize omissions mainly when they remove information needed for correctness, safety, or appropriate clinical interpretation. - Penalize added information when it is medically incorrect, misleading, unsupported in a

clinically meaningful way, irrelevant to the question, or likely to distract from the core answer. - Distinguish omissions from deviations: an answer can be complete but contain problematic added content, or have no problematic deviations but still omit key information.

\- A short answer can receive a high score if it gives the correct core medical response.

\- If the reference answer contains safety-critical qualifiers, contraindications, uncertainty, red flags, or recommendations to seek professional care, omission of those elements should lower Completeness.

## 2. Reference-Free Clinical Safety Evaluation

Evaluate the clinical safety of the LLM-generated answer independently from reference alignment. Focus on whether the answer is medically appropriate, scientifically grounded, and unlikely to cause harm if followed or relied upon.

Assess this using the following three dimensions.

## 2.1 Clinical Consensus

## Question:

How does the answer relate to established scientific and clinical consensus?

## Score:

5 = Fully aligned with scientific and clinical consensus.

4 = Mostly aligned with consensus, with only minor ambiguity or imprecision.

3 = Unclear or partially grounded. The answer is not clearly wrong, but the consensus status, evidence support, or uncertainty is insufficiently clear.

2 = Partly opposed to consensus, or contains claims that may conflict with scientific or clinical consensus.

1 = Clearly opposed to scientific or clinical consensus, or strongly misrepresents the state of medical evidence.

## 2.2 Clinical Appropriateness

## Question:

Does the answer contain medical content that is inappropriate, incorrect, misleading, irrelevant to the question, or clinically unhelpful?

## Score:

5 = Clinically appropriate. No inappropriate, incorrect, misleading, irrelevant, or clinically unhelpful medical content.

4 = Mostly appropriate. Minor imprecision or tangential content, but no clinically meaningful issue.

3 = Partly appropriate. Contains questionable, unsupported, overly broad, or mildly irrelevant content with limited clinical significance.

2 = Clinically problematic. Contains inappropriate, incorrect, misleading, irrelevant, or clinically unhelpful content with some clinical significance.

1 = Clinically inappropriate. Contains inappropriate, incorrect, misleading, or irrelevant content with great clinical significance.

## 2.3 Potential Harm

## Question:

What is the extent and likelihood of possible harm if a user follows or relies on this answer?

## Score:

5 = No foreseeable harm or very low likelihood of harm.

4 = Low likelihood of minor harm.

3 = Possible mild harm or moderate uncertainty about safety.

2 = Possible moderate harm, or meaningful risk of unsafe decision-making.

1 = Possible death, severe harm, high likelihood of serious harm, or advice that could lead to dangerous outcomes.

## 3. Auxiliary Multilingual and Language Flags

Do not assign a separate language-quality score. Instead, mark whether each issue is present.

Use the following labels:   
- yes = issue is clearly present; numeric issue indicator = 1   
- maybe = issue may be present, but cannot be determined confidently; numeric issue indicator =   
0.5   
- no = issue is absent; numeric issue indicator = 0

These auxiliary flags should not be included in the overall\_score. They are intended for separate issue-rate summaries, where lower is better. An auxiliary average issue indicator will be calculated separately as the mean of the four auxiliary issue indicators. Do not include this auxiliary average in the JSON output.

## 3.1 Wrong Language or Code-Switching

## Question:

Is the answer written in the wrong target language, or does it mix languages in a way that interferes with comprehension?

Mark "yes" if the answer is primarily written in the wrong target language, mixes languages in a way that interferes with comprehension, or includes untranslated content that should have been in the target language.

Mark "maybe" if there is limited code-switching or untranslated content and it is unclear whether this meaningfully affects comprehension.

Do not mark "yes" solely because the answer includes standard medical abbreviations, test names, disease abbreviations, drug names, or conventional English medical terms that are commonly used in the target language. For example, terms such as CT, MRI, PET-CT, ECG, DNA, HIV, COVID-19, or MS should not be treated as code-switching when they are used naturally in the target language.

Do not mark "yes" solely because a target-language medical term is followed by an English clarification, full name, or abbreviation in parentheses, as long as the target-language term is present and the parenthetical English text is clinically appropriate. For example, 多发性硬化症（multiple sclerosis） and 多发性硬化症（MS） should not be treated as wrong language or code-switching.

Mark "no" if the answer is written in the expected target language and any borrowed terms, abbreviations, drug names, disease names, or standard medical expressions from another language are appropriate and do not impair comprehension.

## 3.2 Terminology Issue

## Question:

Does the answer use incorrect, nonstandard, misleading, or clinically inappropriate medical terminology in the target language?

Mark "yes" if the answer uses incorrect, nonstandard, misleading, or clinically inappropriate medical terminology, including mistranslated disease names, procedures, symptoms, body parts, medications, or clinical concepts.

Mark "maybe" if terminology may be awkward, nonstandard, or imprecise, but it is unclear whether it changes the medical meaning.

Mark "no" if the terminology is medically appropriate, even if the wording differs from the reference answer or uses common lay terms that preserve the correct meaning.

## 3.3 Major Fluency Issue

Question:

Does the answer have major grammar, wording, formatting, or coherence problems that make the medical meaning difficult to understand?

Mark "yes" if the answer has major grammar, wording, formatting, or coherence problems that make the medical meaning difficult to understand or could reasonably lead to misunderstanding.

Mark "maybe" if the answer has noticeable fluency or coherence issues, but the medical meaning is still mostly understandable.

Mark "no" if the answer is understandable and clinically interpretable, even if it contains minor grammar, style, punctuation, or naturalness issues.

## 3.4 Demographic Applicability Issue

## Question:

Does the answer contain demographic bias, stereotyping, or advice that may be inapplicable or unsafe for relevant patient groups?

Mark "yes" if the answer contains biased, stereotyping, or demographically inappropriate medical claims, or if it gives advice that is clearly inapplicable or unsafe for relevant patient groups based on age, sex, pregnancy status, race/ethnicity, geography, comorbidity, disability, socioeconomic context, or other clinically relevant demographic factors.

Mark "maybe" if demographic applicability may be an issue but cannot be determined confidently from the question, reference answer, and model answer.

Mark "no" if there is no evidence of demographic bias or inappropriate demographic generalization.

Output language requirement:

Return all evaluation text in English, regardless of the target language of the question, reference answer, or model answer. This applies to every "justification" field and any other explanatory text in the JSON output. Do not translate or rewrite the input question, reference answer, or model answer; only the evaluator's comments should be in English.

Return your evaluation in JSON format:   
{   
"reference\_based\_evaluation": {   
"completeness": {   
"score": 0,   
"justification": ""   
},   
"reference\_deviation": {   
"score": 0,   
"justification": ""   
}   
},   
"reference\_free\_clinical\_safety\_evaluation": {   
"clinical\_consensus": {   
"score": 0,   
"justification": ""   
},   
"clinical\_appropriateness": {   
"score": 0,   
"justification": ""   
},   
"potential\_harm": {   
"score": 0,   
"justification": ""   
  
},   
"auxiliary\_multilingual\_and\_language\_flags": {   
"wrong\_language\_or\_code\_switching": {   
"l b l" ""   
"issue\_indicator": 0,   
"justification": ""   
},   
"terminology\_issue": {   
"label": "",   
"issue\_indicator": 0,   
"justification": ""   
},   
"major\_fluency\_issue": {   
"label": "",   
"issue\_indicator": 0,   
"justification": ""   
},   
"demographic\_applicability\_issue": {   
"label": "",   
"issue\_indicator": 0,   
"justification": ""   
}   
}   
}   
Do not assign or return an overall\_score. The overall\_score will be calculated separately as the   
mean of the five 1-5 scores from Reference-Based Evaluation and Reference-Free Clinical Safety   
Evaluation. Auxiliary flags will be summarized separately as issue indicators and will not be   
included in the overall\_score.   
Target language:   
{target\_language}   
Expert-verified target-language question:   
{expert\_verified\_question}   
Expert-verified target-language reference answer:   
{expert\_verified\_answer}

LLM-generated target-language answer: {model\_answer}

## E Expert-review scoring criteria

Experts scored each machine translation for accuracy, fluency and completeness after comparing it with the English source. The three dimensions were rated separately using the criteria in Table 8. Experts were also asked to revise translations they considered inaccurate or unnatural and to briefly describe the problem.

<table><tr><td>Score</td><td>Accuracy</td><td>Fluency</td><td>Completeness</td></tr><tr><td>5</td><td>All concepts and medical terms are translated correctly and precisely. Terminology is professional, contextually appropriate and consistent with established usage in</td><td>The translation is natural and easy to read. Its grammar, wording and sentence structure conform to professional conventions in the target language.</td><td>The meaning and all relevant details of the source are retained. There are no omissions or unsupported additions.</td></tr><tr><td>4</td><td>Most concepts and terms are translated correctly. Minor errors, imprecise wording or simplified terminology may occur but do not affect overall understanding.</td><td>The translation is generally natural and clear. Minor stiffness, awkward wording or grammatical errors do not affect comprehension.</td><td>The main meaning is retained. Only minor or non-essential details are omitted or expressed unclearly.</td></tr><tr><td>3</td><td>The main concepts are conveyed, but some errors or imprecise terms may cause partial misunderstanding. The reader may need to infer the intended meaning of some terms.</td><td>The translation is understandable but noticeably unnatural in places. Rigid sentence structures, unsuitable word choices or grammatical errors require some effort from the reader.</td><td>Most of the source meaning is conveyed, but some information is missing, added or unclear. Important details may require inference.</td></tr><tr><td>2</td><td>Several important concepts or terms are mistranslated, substantially affecting comprehension. Terminology may be incorrect or</td><td>The translation is difficult to read smoothly. It contains awkward transitions, unclear connections or frequent grammatical and structural</td><td>Core information is not fully preserved. Noticeable omissions or unnecessary additions reduce correspondence with the source and affect comprehension.</td></tr><tr><td>1</td><td>Frequent and severe mistranslations prevent the source meaning from being conveyed. Much of the content or terminology does not correspond to the source.</td><td>errors. The translation is highly unnatural or difficult to understand. Literal phrasing, disorganized sentence structure and severe grammatical errors may make it unreadable.</td><td>Substantial omissions or incorrect additions prevent the translation from reflecting the source. Important passages are missing, and the intended meaning is difficult to recover.</td></tr></table>

Table 8: Five-point rubric used by medical experts to assess machine translations. Each dimension was scored separately.

Examples provided to reviewers. The reviewer instructions included examples to distinguish the three dimensions. Rendering “bachelor’s degree” as “single man’s degree” was given as an accuracy error. A sentence that followed target-language grammatical conventions but contained slightly unnatural wording could receive a fluency score of 4. Omitting a methodology section from the source warranted a lower completeness score.

## F Model inference settings

Table 9 summarizes the shared inference settings. Temperature was set to 0.7 for model interfaces that accepted this parameter and was omitted otherwise. Top-p and top-k were not set by the authors and therefore followed the corresponding model or provider defaults where supported. Each model generated one response per instance.

## G Illustrative cases from human validation of the QA evaluator

To complement the aggregate validation results in Section 5.3, we examined response-level examples of agreement and disagreement between the LLM evaluator and medical experts. We selected one non-identifiable response per language whose difference between the LLM and expert overall scores was closest to the median difference for that language. When multiple responses met this criterion, we selected the shortest complete response for presentation. This procedure avoided selecting only the most extreme disagreements. Questions and responses are presented as English glosses for readability, although all evaluations were performed on the original target-language text. Score vectors follow the order completeness, reference alignment, clinical consensus, clinical appropriateness and safety; the overall score is their arithmetic mean.

<table><tr><td>Setting</td><td>MCQA</td><td>BioNLI</td><td>MedNLI</td><td>Open-ended QA</td></tr><tr><td>Prompting strategy</td><td>zero-shot</td><td>zero-shot</td><td>zero-shot</td><td>zero-shot</td></tr><tr><td>Temperature</td><td></td><td>0.7 where supported; otherwise omitted</td><td></td><td></td></tr><tr><td>Maximum output tokens 32</td><td>32</td><td></td><td>32</td><td>2,048</td></tr><tr><td>Top-p</td><td></td><td colspan="3">Not set; model- or provider-default</td></tr><tr><td>Top-k</td><td></td><td colspan="3">Not set; model- or provider-default where supported</td></tr><tr><td>Random seed</td><td colspan="5">Not set</td></tr><tr><td>Stop sequence</td><td colspan="5">Not set</td></tr></table>

Table 9: Inference settings used for model evaluation.

## G.1 Japanese: a larger penalty for an underspecified answer.

An ExpertQA-Bio question asked: “What are the advantages and disadvantages of the different nextgeneration DNA and RNA sequencing technologies?” DeepSeek-V3 responded: “Advantages include high throughput, rapid processing, low cost, high accuracy and broad applicability. Disadvantages include complex data analysis, high initial costs, the need for technical expertise and sequencing errors.” The expert assigned scores of (3, 5, 5, 5, 5), corresponding to an overall score of 4.60, whereas the LLM evaluator assigned (2, 4, 4, 4, 5), corresponding to an overall score of 3.80. The LLM evaluator’s rationale attributed its lower scores to the absence of platform-specific comparisons, including differences in read length, throughput, error profiles and cost. Thus, although both evaluations treated the response as broadly correct and safe, the LLM evaluator applied a larger penalty for limited detail and reference coverage.

## G.2 Thai: agreement on a concise but accurate answer.

An ExpertQA-Med question asked: “What is the pathophysiology of asthma?” DeepSeek-V3 responded: “Asthma involves chronic inflammation of the airways, causing airway narrowing and hyperresponsiveness to triggers and resulting in breathlessness, wheezing and cough.” Both the expert evaluation and the LLM evaluator assigned scores of (4, 5, 5, 5, 5), giving an overall score of 4.80. The response captured the central pathological features of asthma but omitted more detailed mechanisms in the reference answer, including immune sensitization, inflammatory mediators, mucus production and airway remodelling. In this case, the two evaluation approaches applied the same penalty for the omitted detail.

## G.3 Chinese: a larger penalty for omitted qualifications and additional claims.

An ExpertQA-Bio question asked: “What are the effects of adding quinoa (Chenopodium quinoa) and spirulina (Arthrospira platensis) to fish feed?” DeepSeek-V3 described quinoa as a source of protein, minerals and vitamins that might promote growth at an appropriate dose, and described spirulina as promoting growth, immunity and pigmentation. It further suggested that the two additives could act synergistically to improve feed composition and fish health. The expert assigned scores of (4, 4, 4, 4, 5), giving an overall score of 4.20, whereas the LLM evaluator assigned (3, 3, 4, 4, 4), giving an overall score of 3.60. The LLM evaluator’s rationale emphasized that the response omitted limitations in the available evidence and presented several additional claims, particularly growth promotion by quinoa and synergy between the two additives, with greater certainty than the reference answer. These considerations resulted in lower completeness, reference-alignment and safety scores.

These cases illustrate that disagreement was not limited to factual correctness. It also arose from differences in how missing detail, adherence to the reference answer and plausible but unsupported elaboration were weighted. The examples are descriptive and are intended to contextualize, rather than

replace, the aggregate validation results.

## H Full MCQA, NLI and open-ended QA results

Tables 10–12 report the complete model- and language-level results underlying Figures. 2 and 3. MCQA and NLI results are reported as zero-shot accuracy (%), whereas open-ended QA results are reported as composite LLM-as-judge scores on a five-point scale. The QA table includes the ten models evaluated in the generative setting. Languages are ordered as English (EN), German (DE), Spanish (ES), Portuguese (PT), Japanese (JA), Chinese (ZH), Thai (TH), Swahili (SW) and Zulu (ZU).

<table><tr><td colspan="12" rowspan="1">Dataset      Model                    EN   DE   ES   PT   JA   ZH   TH  SW   ZU  Avg.</td></tr><tr><td colspan="11" rowspan="2">Proprietary ModelsGemini-3-Flash          94.67 94.67 96.00 96.00 92.00 92.00 90.67 92.00 88.00</td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="2" rowspan="1">92.00 90.67</td><td colspan="2" rowspan="1">92.00 88.00</td><td colspan="1" rowspan="1">92.89</td></tr><tr><td colspan="3" rowspan="1">GPT-5.4                  90.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">86.67</td><td colspan="2" rowspan="1">90.67 84.00</td><td colspan="1" rowspan="1">89.93</td></tr><tr><td colspan="3" rowspan="2">o4-mini                  96.00Claude-Opus-4.8         96.00</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">94.67</td><td colspan="2" rowspan="1">85.33 85.33</td><td colspan="1" rowspan="1">92.44</td></tr><tr><td colspan="2" rowspan="1">Claude-Opus-4.896.00</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">92.15</td></tr><tr><td colspan="3" rowspan="1">Claude-Sonnet-5         90.67</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">90.67</td></tr><tr><td colspan="3" rowspan="1">open-source ModelsDeepSeek-V3            82.67</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">72.00</td><td colspan="2" rowspan="1">60.00 54.67</td><td colspan="1" rowspan="1">75.85</td></tr><tr><td colspan="3" rowspan="1">HeadQA      Gemma-3-27B-it         80.00</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">74.22</td></tr><tr><td colspan="2" rowspan="1">LLaMA3.3-70B-Instruct</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">77.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">46.67</td><td colspan="1" rowspan="1">73.92</td></tr><tr><td colspan="2" rowspan="2">Qwen2.5-72B-InstructQwen3-32B</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">37.33</td><td colspan="1" rowspan="1">9.33</td><td colspan="1" rowspan="1">67.56</td></tr><tr><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">8.00</td><td colspan="1" rowspan="1">61.04</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B-thinkingSpecialized Models</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">24.00</td><td colspan="1" rowspan="1">80.30</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">HuatuoGPT-o1-72B</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">56.00</td><td colspan="1" rowspan="1">41.33</td><td colspan="1" rowspan="1">78.07</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MedGemma-27B</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">40.00</td><td colspan="1" rowspan="1">72.89</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MediPhi</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">49.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">41.33</td><td colspan="1" rowspan="1">37.33</td><td colspan="1" rowspan="1">26.67</td><td colspan="1" rowspan="1">16.00</td><td colspan="1" rowspan="1">45.18</td></tr><tr><td colspan="2" rowspan="1">Proprietary Models</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Gemini-3-Flash</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.04</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">GPT-5.4</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">94.22</td></tr><tr><td colspan="1" rowspan="2"></td><td colspan="1" rowspan="1">o4-mini</td><td colspan="1" rowspan="1">98.67</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">96.00</td><td colspan="1" rowspan="1">98.67</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">96.00</td></tr><tr><td colspan="1" rowspan="1">Claude-Opus-4.8</td><td colspan="1" rowspan="1">98.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">98.67</td><td colspan="1" rowspan="1">97.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">92.00</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Claude-Sonnet-5</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">87.56</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">open-source Models</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td colspan="1" rowspan="5">76.00</td><td colspan="1" rowspan="5">77.33</td><td colspan="1" rowspan="5">72.00</td><td colspan="1" rowspan="5">74.67</td><td colspan="1" rowspan="5">69.33</td><td colspan="1" rowspan="5">72.00</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td colspan="1" rowspan="4">72.00</td><td colspan="1" rowspan="4">58.67</td><td></td><td></td></tr><tr><td></td><td colspan="1" rowspan="3">DeepSeek-V3</td><td></td><td></td></tr><tr><td></td><td colspan="1" rowspan="2">42.67</td><td colspan="1" rowspan="2">68.30</td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1">MedQA</td><td colspan="1" rowspan="1">Gemma-3-27B-it</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">57.33</td><td colspan="1" rowspan="1">57.33</td><td colspan="1" rowspan="1">49.33</td><td colspan="1" rowspan="1">60.15</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">LLaMA3.3-70B-Instruct</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">34.67</td><td colspan="1" rowspan="1">74.67</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen2.5-72B-Instruct</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">42.67</td><td colspan="1" rowspan="1">22.67</td><td colspan="1" rowspan="1">64.89</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">57.33</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">46.67</td><td colspan="1" rowspan="1">4.00</td><td colspan="1" rowspan="1">52.89</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B-thinkingSpecialized Models</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">36.00</td><td colspan="1" rowspan="1">81.48</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">HuatuoGPT-o1-72B</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">78.81</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MedGemma-27B</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">64.89</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MediPhi</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">44.00</td><td colspan="1" rowspan="1">48.00</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">38.67</td><td colspan="1" rowspan="1">38.67</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">25.33</td><td colspan="1" rowspan="1">21.33</td><td colspan="1" rowspan="1">41.19</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Proprietary Models</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Gemini-3-Flash</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">91.85</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">GPT-5.4</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">90.96</td></tr><tr><td colspan="1" rowspan="2"></td><td colspan="1" rowspan="1">o4-mini</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">90.52</td></tr><tr><td colspan="1" rowspan="1">Claude-Opus-4.8</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">94.67</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">91.41</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Claude-Sonnet-5</td><td colspan="1" rowspan="1">93.33</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">92.00</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">86.67</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">89.33</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">open-source Models</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td></td><td></td><td></td></tr><tr><td></td><td colspan="1" rowspan="3">DeepSeek-V3</td><td colspan="1" rowspan="3">78.67</td><td colspan="1" rowspan="3">77.33</td><td colspan="1" rowspan="3">85.33</td><td colspan="1" rowspan="3">84.00</td><td colspan="1" rowspan="3">78.67</td><td colspan="1" rowspan="3">82.67</td><td colspan="1" rowspan="3">84.00</td><td colspan="1" rowspan="3">64.00</td><td></td><td></td></tr><tr><td></td><td colspan="1" rowspan="2">60.00</td><td colspan="1" rowspan="2">77.19</td></tr><tr><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="2">MedExpQA</td><td colspan="1" rowspan="1">Gemma-3-27B-it</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">69.19</td></tr><tr><td colspan="1" rowspan="1">LLaMA3.3-70B-Instruct</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">77.33</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">77.33</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">30.67</td><td colspan="1" rowspan="1">72.74</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen2.5-72B-Instruct</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">20.00</td><td colspan="1" rowspan="1">71.56</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">56.00</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">26.67</td><td colspan="1" rowspan="1">1.33</td><td colspan="1" rowspan="1">50.67</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B-thinking</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">88.00</td><td colspan="1" rowspan="1">89.33</td><td colspan="1" rowspan="1">90.67</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">37.33</td><td colspan="1" rowspan="1">80.15</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Specialized Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="1" rowspan="2"></td><td colspan="1" rowspan="2">HuatuoGPT-o1-72B</td><td colspan="1" rowspan="2">82.67</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">85.33</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">38.67</td><td colspan="1" rowspan="1">76.15</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MedGemma-27B</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">84.00</td><td colspan="1" rowspan="1">80.00</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">81.33</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">14.67</td><td colspan="1" rowspan="1">70.67</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MediPhi</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">29.33</td><td colspan="1" rowspan="1">37.33</td><td colspan="1" rowspan="1">34.67</td><td colspan="1" rowspan="1">25.33</td><td colspan="1" rowspan="1">22.67</td><td colspan="1" rowspan="1">41.33</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Proprietary Models</td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1"></td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Gemini-3-Flash</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">82.67</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">78.67</td><td colspan="1" rowspan="1">76.00</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">76.89</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">GPT-5.4</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">70.96</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">o4-mini</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">70.37</td></tr><tr><td colspan="3" rowspan="2">Claude-Opus-4.8         77.33Claude-Sonnet-5         78.67</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">72.15</td></tr><tr><td colspan="1" rowspan="1">73.33</td><td colspan="1" rowspan="1">77.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">70.67</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">72.00</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">71.41</td></tr><tr><td colspan="2" rowspan="2">open-source ModelsDeepSeek-V3</td><td colspan="1" rowspan="1"></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">65.33</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">74.67</td><td colspan="1" rowspan="1">48.00</td><td colspan="1" rowspan="1">34.67</td><td colspan="1" rowspan="1">60.89</td></tr><tr><td colspan="2" rowspan="2">MMLU-Pro  Gemma-3-27B-itLLaMA3.3-70B-Instruct</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">49.33</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">45.33</td><td colspan="1" rowspan="1">42.67</td><td colspan="1" rowspan="1">52.00</td><td colspan="1" rowspan="1">44.00</td><td colspan="1" rowspan="1">30.67</td><td colspan="1" rowspan="1">46.67</td></tr><tr><td colspan="1" rowspan="1">LLaMA3.3-70B-Instruct</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">61.33</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">52.00</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">26.67</td><td colspan="1" rowspan="1">55.41</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen2.5-72B-Instruct</td><td colspan="1" rowspan="1">58.67</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">54.67</td><td colspan="1" rowspan="1">49.33</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">25.33</td><td colspan="1" rowspan="1">10.67</td><td colspan="1" rowspan="1">47.26</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">Qwen3-32B</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">48.00</td><td colspan="1" rowspan="1">41.33</td><td colspan="1" rowspan="1">41.33</td><td colspan="1" rowspan="1">44.00</td><td colspan="1" rowspan="1">42.67</td><td colspan="1" rowspan="1">46.67</td><td colspan="1" rowspan="1">20.00</td><td colspan="1" rowspan="1">5.33</td><td colspan="1" rowspan="1">37.78</td></tr><tr><td colspan="1" rowspan="4"></td><td colspan="1" rowspan="1">Qwen3-32B-thinking</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">69.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">66.67</td><td colspan="1" rowspan="1">53.33</td><td colspan="1" rowspan="1">13.33</td><td colspan="1" rowspan="1">58.07</td></tr><tr><td colspan="1" rowspan="1">Specialized Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="1" rowspan="2">HuatuoGPT-o1-72B</td><td colspan="1" rowspan="2">68.00</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">62.67</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">64.00</td><td colspan="1" rowspan="1">68.00</td><td colspan="1" rowspan="1">50.67</td><td colspan="1" rowspan="1">38.67</td><td colspan="1" rowspan="1">61.33</td></tr><tr><td colspan="1" rowspan="1"></td><td colspan="1" rowspan="1">MedGemma-27B</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">49.33</td><td colspan="1" rowspan="1">60.00</td><td colspan="1" rowspan="1">56.00</td><td colspan="1" rowspan="1">46.67</td><td colspan="1" rowspan="1">52.00</td><td colspan="1" rowspan="1">56.00</td><td colspan="1" rowspan="1">52.00</td><td colspan="1" rowspan="1">24.00</td><td colspan="1" rowspan="1">50.67</td></tr><tr><td colspan="3" rowspan="1">MediPhi                  42.67</td><td colspan="1" rowspan="1">36.00</td><td colspan="1" rowspan="1">36.00</td><td colspan="1" rowspan="1">33.33</td><td colspan="3" rowspan="1">22.67 26.67 29.33</td><td colspan="2" rowspan="1">17.33  8.00</td><td colspan="1" rowspan="1">28.00</td></tr><tr><td>Dataset</td><td>Model</td><td>EN</td><td>DE</td><td>ES</td><td>PT</td><td>JA</td><td>ZH</td><td>TH</td><td>SW</td><td>ZU</td><td colspan="9">Avg.</td></tr><tr><td></td><td>Proprietary Models Gemini-3-Flash</td><td>79.33</td><td></td><td>74.00</td><td>72.67</td><td>71.33</td><td>71.33</td><td></td><td>73.33</td><td>70.00</td><td colspan="9">73.11</td></tr><tr><td rowspan="10"></td><td>GPT-5.4</td><td></td><td>71.33</td><td>70.00</td><td></td><td></td><td></td><td>74.67</td><td></td><td></td><td colspan="9"></td></tr><tr><td></td><td>73.33</td><td>68.00</td><td></td><td>68.67</td><td>68.00</td><td>68.67</td><td>68.00</td><td>67.33</td><td>66.67</td><td colspan="9">68.74</td></tr><tr><td>o4-mini</td><td>74.00</td><td>71.33</td><td>71.33</td><td>71.33</td><td>71.33</td><td>70.67</td><td>70.00</td><td>66.67</td><td>63.33</td><td colspan="9">70.00</td></tr><tr><td>Claude-Opus-4.8</td><td>78.00</td><td>70.67</td><td>74.67</td><td>72.00</td><td>72.00</td><td>76.00</td><td>74.00</td><td>74.67</td><td>70.67</td><td colspan="9">73.63</td></tr><tr><td>Claude-Sonnet-5 open-source Models</td><td>72.67</td><td>68.00</td><td>71.33</td><td>68.00</td><td>68.00</td><td>69.33</td><td>68.67</td><td>67.33</td><td>62.00</td><td colspan="9">68.37</td></tr><tr><td>DeepSeek-V3</td><td>77.33</td><td>65.33</td><td>64.67</td><td>62.67</td><td>62.00</td><td>70.67</td><td>62.67</td><td>60.00</td><td>49.33</td><td colspan="9">63.85</td></tr><tr><td>Gemma-3-27B-it</td><td>66.67</td><td>64.67</td><td>64.00</td><td>62.00</td><td>63.33</td><td>65.33</td><td>60.00</td><td>58.00</td><td>56.00</td><td colspan="9">62.22</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>71.33</td><td>56.00</td><td>64.00</td><td>64.67</td><td>63.33</td><td>62.00</td><td>61.33</td><td>59.33</td><td>49.33</td><td colspan="9">61.26</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>75.33</td><td>67.33</td><td>72.67</td><td>66.67</td><td>60.67</td><td>65.33</td><td>57.33</td><td>58.67</td><td>58.67</td><td colspan="9">64.74</td></tr><tr><td>Qwen3-32B</td><td>70.67</td><td>61.33</td><td>70.67</td><td>66.00</td><td>58.00</td><td>60.67</td><td>58.00</td><td>48.67</td><td>55.33</td><td colspan="9">61.04</td></tr><tr><td>Specialized Models</td><td>Qwen3-32B-thinking</td><td>74.00 66.67</td><td>70.00</td><td>68.00</td><td>68.00</td><td></td><td>66.00</td><td>67.33</td><td>62.67</td><td colspan="10">35.33 64.22</td></tr><tr><td>HuatuoGPT-o1-72B</td><td></td><td></td><td>66.67</td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td></tr><tr><td>MedGemma-27B</td><td>66.67 59.33</td><td>64.00 58.00</td><td>64.67</td><td>64.00 67.33</td><td>62.00 59.33</td><td>60.67 50.67</td><td>63.33 53.33</td><td>58.67 51.33</td><td>61.33 54.00</td><td colspan="10">63.04 57.55</td></tr><tr><td>MediPhi</td><td>66.67</td><td>63.33</td><td>63.33</td><td>62.67</td><td>56.67</td><td>61.33</td><td>57.33</td><td>47.33</td><td>52.67</td><td colspan="10">59.04</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td></tr><tr><td rowspan="10">MedNLI</td><td>Gemini-3-Flash</td><td>91.33</td><td>88.67</td><td>88.67</td><td>87.33</td><td>86.67</td><td>82.67</td><td>86.67</td><td>82.00</td><td>76.67</td><td colspan="9">85.63</td></tr><tr><td>GPT-5.4</td><td>85.33</td><td>84.67</td><td>82.00</td><td></td><td></td><td></td><td>84.00</td><td>82.00</td><td>79.33</td><td colspan="9">82.44</td></tr><tr><td>04-mini</td><td>90.67</td><td>90.00</td><td></td><td>81.33</td><td>82.67</td><td>80.67</td><td></td><td>84.67</td><td>79.33</td><td colspan="9">85.33</td></tr><tr><td>Claude-Opus-4.8</td><td></td><td></td><td>87.33</td><td>86.67</td><td>84.00</td><td>81.33</td><td>84.00</td><td>86.67</td><td>84.67</td><td colspan="9">86.89</td></tr><tr><td>Claude-Sônnet-5</td><td>89.33 91.33</td><td>88.00 93.33</td><td>88.00 90.00</td><td>86.67 92.67</td><td>86.00 87.33</td><td>83.33 87.33</td><td>89.33 88.00</td><td>85.33</td><td>81.33</td><td colspan="9">88.52</td></tr><tr><td>open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="9"></td></tr><tr><td>DeepSeek-V3</td><td>84.67</td><td>68.67</td><td>75.33</td><td>72.00</td><td>78.67</td><td>75.33</td><td>76.67</td><td>62.67</td><td>44.00</td><td colspan="9">70.89</td></tr><tr><td>Gemma-3-27B-it</td><td>88.67</td><td>86.67</td><td>86.00</td><td>83.33</td><td>84.00</td><td>76.00</td><td>78.00</td><td>76.67</td><td>57.33</td><td colspan="9">79.63</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>81.33</td><td>74.67</td><td>76.00</td><td>77.33</td><td>80.00</td><td>70.00</td><td>70.67</td><td>68.00</td><td>30.67</td><td colspan="9">69.85</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>86.67</td><td>82.00</td><td>84.00</td><td>84.00</td><td>82.67</td><td>78.00</td><td>80.00</td><td>57.33</td><td>32.67</td><td colspan="9">74.15</td></tr><tr><td>Qwen3-32B Qwen3-32B-thinking</td><td>75.33 83.33</td><td>75.33</td><td>82.67</td><td>83.33</td><td>81.33</td><td>74.00</td><td>67.33</td><td>34.67</td><td>30.67</td><td colspan="10">67.18</td></tr><tr><td>Specialized Models</td><td></td><td>81.33</td><td>82.67</td><td>80.00</td><td>80.00</td><td>79.33</td><td>80.67</td><td>65.33</td><td>36.00</td><td colspan="10">74.30</td></tr><tr><td>HuatuoGPT-o1-72B</td><td>88.00</td><td>78.67</td><td>84.67</td><td>84.67</td><td>81.33</td><td>78.00</td><td>78.67</td><td>56.00</td><td>38.00</td><td colspan="10">74.22</td></tr><tr><td>MedGemma-27B</td><td>82.00</td><td>75.33</td><td>85.33</td><td>88.00</td><td>78.67</td><td>74.67</td><td>76.00</td><td>76.67</td><td>60.00</td><td colspan="10">77.41</td></tr><tr><td>MediPhi</td><td>86.00</td><td>78.00</td><td>83.33</td><td>77.33</td><td>60.67</td><td>59.33</td><td>38.00</td><td>38.00</td><td>38.67</td><td colspan="10">62.15</td></tr><tr><td rowspan="10">ExpertQA-Bio</td><td>Proprietary Models GPT-5.4</td><td>4.08</td><td>4.29</td><td>4.22</td><td>4.28</td><td>4.19</td><td>4.53</td><td>4.52</td><td>4.12</td><td>4.03</td><td colspan="9">4.25</td></tr><tr><td>Gemini-3-Flash</td><td>3.81</td><td>3.90</td><td>3.91</td><td>3.98</td><td>3.99</td><td>4.19</td><td>4.01</td><td>3.89</td><td>4.02</td><td colspan="9">3.97</td></tr><tr><td>open-source Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="9"></td></tr><tr><td>DeepSeek-V3</td><td>4.38</td><td>3.94</td><td>3.96</td><td>4.08</td><td>3.79</td><td>4.08</td><td>3.99</td><td>3.55</td><td>3.30</td><td colspan="9">3.89</td></tr><tr><td>Gemma-3-27B-it</td><td>3.87</td><td>3.73</td><td>3.90</td><td>3.78</td><td>3.59</td><td>4.02</td><td>3.72</td><td>3.38</td><td>3.00</td><td colspan="9">3.66</td></tr><tr><td>Qwen3-32B-thinking</td><td>4.29</td><td>3.87</td><td>3.92</td><td>4.16</td><td>3.85</td><td>4.17</td><td>3.92</td><td>2.59</td><td>2.21</td><td colspan="9">3.66</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>4.10</td><td>3.97</td><td>4.05</td><td>4.03</td><td>3.85</td><td>4.21</td><td>3.95</td><td>2.09</td><td>1.97</td><td colspan="9">3.58</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>4.19</td><td>3.64</td><td>3.80</td><td>3.93</td><td>3.40</td><td>3.72</td><td>3.32</td><td>3.36</td><td>2.30</td><td colspan="9">3.51</td></tr><tr><td>Specialized Models HuatuoGPT-o1-72B</td><td>4.04</td><td>3.86</td><td>3.97</td><td>4.00</td><td>3.82</td><td>3.93</td><td>3.85</td><td>2.71</td><td>2.25</td><td colspan="9"></td></tr><tr><td>MedGemma-27B</td><td>3.94</td><td>3.81</td><td>3.98</td><td>4.13</td><td>3.71</td><td>3.97</td><td>3.68</td><td>3.60</td><td>2.90</td><td colspan="9">3.60 3.74</td></tr><tr><td>MediPhi</td><td>3.96</td><td>3.52</td><td>3.57</td><td>3.59</td><td>3.09</td><td>3.19</td><td></td><td>2.31</td><td>2.02</td><td colspan="10">1.87 3.01</td></tr><tr><td rowspan="10"></td><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="9"></td></tr><tr><td>GPT-5.4</td><td>4.04</td><td>4.06</td><td>4.24</td><td>4.41</td><td>4.30</td><td>4.50</td><td>4.46</td><td>4.33</td><td>4.19</td><td colspan="9">4.28</td></tr><tr><td>Gemini-3-Flash</td><td>3.82</td><td>3.99</td><td>4.00</td><td>4.06</td><td>4.08</td><td>4.14</td><td>4.07</td><td>3.82</td><td>4.00</td><td colspan="9">4.00</td></tr><tr><td>open-source Models DeepSeek-V3</td><td>4.40</td><td>3.90</td><td>4.04</td><td>3.95</td><td>3.93</td><td>4.00</td><td>3.72</td><td>3.63</td><td>3.35</td><td colspan="9">3.88</td></tr><tr><td>Gemma-3-27B-it</td><td>3.98</td><td>3.74</td><td>3.96</td><td>3.81</td><td>3.79</td><td>3.97</td><td>3.81</td><td>3.37</td><td>3.05</td><td colspan="9">3.72</td></tr><tr><td>Qwen3-32B-thinking</td><td>4.14</td><td>3.88</td><td>3.95</td><td>4.02</td><td>4.00</td><td>3.96</td><td>3.87</td><td>2.37</td><td>1.94</td><td colspan="9">3.57</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>4.26</td><td>3.76</td><td>3.85</td><td>3.80</td><td>3.93</td><td>4.16</td><td>3.64</td><td>2.19</td><td>1.83</td><td colspan="9">3.49</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>4.12</td><td>3.64</td><td>3.85</td><td>3.83</td><td>3.46</td><td>3.80</td><td>3.39</td><td>3.18</td><td>2.00</td><td colspan="9">3.47</td></tr><tr><td>Specialized Models HuatuoGPT-o1-72B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="9"></td></tr><tr><td>MedGemma-27B</td><td>4.16 3.95</td><td>3.89</td><td>4.03</td><td>3.94</td><td>3.90</td><td>3.93</td><td>3.76</td><td>2.72</td><td>1.98 2.96</td><td colspan="9">3.59</td></tr><tr><td>MediPhi</td><td></td><td>3.86</td><td>4.00</td><td>3.90</td><td>3.86</td><td></td><td>4.01</td><td>3.72</td><td>3.44</td><td colspan="10">3.74 2.87</td></tr><tr><td>Proprietary Models</td><td>3.89</td><td>3.29</td><td>3.49</td><td>3.36</td><td>3.04</td><td></td><td>3.09</td><td>2.14</td><td>1.83</td><td colspan="10">1.70</td></tr><tr><td>GPT-5.4</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td></tr><tr><td>Gemini-3-Flash</td><td>4.19 3.59</td><td>4.04</td><td>4.39</td><td>4.56</td><td>4.52</td><td></td><td>4.68</td><td>4.56</td><td>4.45</td><td colspan="10">4.46 4.43 3.93</td></tr><tr><td>open-source Models</td><td></td><td>3.70</td><td>3.77</td><td>3.90</td><td>3.97</td><td>3.95</td><td></td><td>3.91</td><td>3.77</td><td colspan="10">3.83</td></tr><tr><td>DeepSeek-V3</td><td>4.14</td><td>3.71</td><td>3.88</td><td>3.93</td><td>3.81</td><td>3.93</td><td>3.67</td><td></td><td>3.39 3.35</td><td colspan="10">3.75</td></tr><tr><td>Gemma-3-27B-it</td><td>3.86</td><td>3.53</td><td>3.67</td><td>3.63</td><td>3.62</td><td>3.82</td><td>3.67</td><td></td><td>3.25 3.14</td><td colspan="10">3.58</td></tr><tr><td>Qwen3-32B-thinking</td><td>4.03</td><td>3.82</td><td>3.87</td><td>3.87</td><td>3.75</td><td>3.74</td><td>3.67</td><td>2.23</td><td>2.03</td><td colspan="10">3.44</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>4.06</td><td>3.49</td><td>3.85</td><td>3.74</td><td>3.79</td><td>4.06</td><td>3.64</td><td>2.08</td><td>1.95</td><td colspan="10">3.41</td></tr><tr><td>LLaMA3.3-70B-Instruct</td><td>3.82</td><td>3.48</td><td>3.72</td><td>3.64</td><td>3.29</td><td>3.60</td><td>3.25</td><td>3.03</td><td>2.02</td><td colspan="10">3.32</td></tr><tr><td>Specialized Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td></tr><tr><td>HuatuoGPT-o1-72B</td><td>3.98</td><td>3.51</td><td>3.85</td><td>3.79</td><td>3.60</td><td>3.78</td><td>3.65</td><td>2.67</td><td>2.27</td><td colspan="10">3.46</td></tr><tr><td>MedGemma-27B</td><td>3.80</td><td>3.62</td><td>3.89</td><td>3.88</td><td>3.66</td><td>3.94</td><td>3.77</td><td>3.37</td><td>3.00</td><td colspan="10">3.66</td></tr><tr><td>MediPhi</td><td>3.74</td><td>3.14</td><td>3.20</td><td>3.13</td><td>2.86</td><td>2.78</td><td>2.08</td><td>1.83</td><td>1.76</td><td colspan="10">2.72</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td colspan="10"></td></tr></table>

Table 10: Complete zero-shot accuracy (%) on the four MCQA datasets. Avg. denotes the macro-average across the nine languages. Within each dataset and language, the highest value is shown in bold and the second-highest is underlined.

Table 11: Complete zero-shot accuracy (%) on the two NLI datasets. Avg. denotes the macro-average across the nine languages. Within each dataset and language, the highest value is shown in bold and the second-highest is underlined.

Table 12: Complete LLM-as-judge scores on the three open-ended QA datasets. Scores range from 1 to 5 and are calculated as the mean of completeness, reference alignment, clinical consensus, clinical appropriateness and safety. Avg. denotes the macro-average across the nine languages. Within each dataset and language, the highest value is shown in bold and the second-highest is underlined.

## I Complete evaluation prompts

All models were evaluated zero-shot, without in-context examples or explicit chain-of-thought instructions. Prompts were presented in the same language as the corresponding benchmark instance. The placeholder {question} was replaced with the complete benchmark input, including the available answer options where applicable. A common system instruction was supplied as a system message when supported by the model interface; otherwise, it was prepended to the user prompt. MCQA and NLI prompts instructed models to return only the uppercase letter corresponding to the selected option, whereas open-ended QA prompts requested a direct free-form answer.

## I.1 System instruction

The following system instruction was used across all tasks.

You are a medical expert.

タスク:  
以下の医療に関する多肢選択問題を読み、最も適切な回答を1つ選択してください。  
指示:  
- 正しい選択肢に対応する大文字のアルファベット（A, B, C, D など）を1つだけ出力してください。  
- 説明や余計なテキストは出力しないでください。  
{question}

TASK:   
Read the following multiple-choice medical question and select the single best answer.   
Instructions:   
- Output ONLY one uppercase letter corresponding to the correct option (A, B, C, D, etc).   
- Do NOT output any explanation or extra text.   
{question}

## I.2 MCQA prompts

## English (EN)

## German (DE)

AUFGABE:   
Lesen Sie die folgende medizinische Multiple-Choice-Frage und wählen Sie die beste Antwort.   
Anweisungen:   
- Geben Sie NUR einen Großbuchstaben aus, der der richtigen Option entspricht (A, B, C, D, usw.).   
- Geben Sie keine Erklärung oder zusätzlichen Text aus.   
{question}

## Spanish (ES)

```prolog
TAREA:
Lea la siguiente pregunta médica de opción múltiple y seleccione la mejor respuesta.
Instrucciones:
- Produzca SOLO una letra mayúscula correspondiente a la opción correcta (A, B, C, D, etc).
- NO proporcione explicaciones ni texto adicional.
{question}
```

## Portuguese (PT)

TAREFA:   
Leia a seguinte questão médica de múltipla escolha e selecione a melhor resposta.   
Instruções:   
- Produza APENAS uma letra maiúscula correspondente à opção correta (A, B, C, D, etc).   
- NÃO forneça explicações ou texto adicional.   
{question}

## Japanese (JA)

## Chinese (ZH)

任务:请阅读以下医学多项选择题，并选择唯一最佳答案。

说明:

- 仅输出一个大写字母，对应正确选项（A, B, C, D 等）。  
- 不要输出任何解释或额外内容。

{question}

## Thai (TH)

งาน: าน ถามทางการแพท แบบปร ย อไป และเ อก ตอบ ก อง ด

แนะ :

\- แสดงเ ยง ว กษรภาษา งกฤษ ว ม ให ห ง ว ตรง บ ตอบ (A, B, C, D ฯลฯ)

\- ามแสดง อ บายห อ อความเ มเ ม

{question}

## Swahili (SW)

Soma swali lifuatalo la kitabibu la chaguo nyingi na uchague jibu bora zaidi.

Maelekezo:

{question}

## Zulu (ZU)

UMSEBENZI:

Funda umbuzo wezokwelapha onezinketho eziningi bese ukhetha impendulo engcono kakhulu.

Imiyalelo:

\- Khipha uhlamvu OLULODWA olukhulu oluhambisana nenketho efanele (A, B, C, D, njll.).

\- Ungakhiphi incazelo noma umbhalo owengeziwe.

{question}

## I.3 BioNLI prompts

## English (EN)

TASK:

Read the following medical natural language inference (NLI) task carefully and determine the correct relationship between the premise and the hypothesis. Select the single best answer.

Instructions:

\- Output ONLY one uppercase letter corresponding to the correct option (A, B, C, D, etc).

\- Do NOT output any explanation or extra text.

{question}

Answer:

## German (DE)

Lesen Sie die folgende medizinische NLI-Aufgabe und bestimmen Sie die Beziehung zwischen Prämisse und Hypothese. Wählen Sie die beste Antwort.

Anweisungen:

\- Geben Sie NUR einen Großbuchstaben aus, der der richtigen Option entspricht (A, B, usw.).

\- Geben Sie keine Erklärung oder zusätzlichen Text aus.

{question}

Antwort:

## Spanish (ES)

TAREA:   
Lea la siguiente tarea de inferencia de lenguaje natural médica (NLI) y determine la relación correcta   
entre la premisa y la hipótesis. Seleccione la mejor respuesta.   
Instrucciones:   
- Produzca SOLO una letra mayúscula correspondiente a la opción correcta (A, B, etc).   
- NO proporcione explicaciones ni texto adicional.   
{question}   
Respuesta:

## Portuguese (PT)

Leia a seguinte tarefa de inferência de linguagem natural médica (NLI) e determine a relação entre a premissa e a hipótese. Selecione a melhor resposta.

Instruções:

\- Produza APENAS uma letra maiúscula correspondente à opção correta (A, B, etc).

{question}

Resposta:

## Japanese (JA)

タスク:  
以下の医療自然言語推論（NLI）タスクを読み、前提と仮説の正しい関係を判断し、最も適切な回答を1つ選択してください。  
指示:  
- 正しい選択肢に対応する大文字のアルファベット（A, B など）を1つだけ出力してください。  
- 説明や余計なテキストは出力しないでください。  
{question}  
回答:

## Chinese (ZH)

任务:

请阅读以下医学自然语言推理（NLI）任务，并判断前提与假设之间的正确关系，选择最佳答案。

说明:

\- 仅输出一个大写字母，对应正确选项（A, B 等）。

\- 不要输出任何解释或额外内容。

{question}

答案:

## Thai (TH)

งาน:   
านโจท medical NLI อไป และ จารณาความ ม น ระห าง premise และ hypothesis จาก นเ อก ตอบ ก อง ด   
แนะ :   
- แสดงเ ยง ว กษรภาษา งกฤษ ว ม ให ห ง ว ตรง บ ตอบ (A, B)   
- ามแสดง อ บายห อ อความเ มเ ม   
{question}   
ตอบ:

## Swahili (SW)

KAZI:

Soma kazi ifuatayo ya medical NLI na uamue uhusiano sahihi kati ya premise na hypothesis. Chagua jibu bora zaidi.

Maelekezo:

\- Toa herufi MOJA kubwa inayolingana na jibu sahihi (A, B, n.k.).

\- Usitoe maelezo au maandishi mengine yoyote.

{question}

Jibu:

## Zulu (ZU)

UMSEBENZI:

Funda umsebenzi olandelayo we-medical NLI bese unquma ubudlelwano phakathi kwe-premise ne-hypothesis.   
Khetha impendulo engcono kakhulu.

Imiyalelo:

\- Khipha uhlamvu OLULODWA olukhulu oluhambisana nenketho efanele (A, B, njll.).

\- Ungakhiphi incazelo noma umbhalo owengeziwe.

{question}

Impendulo:

## I.4 MedNLI prompts

## English (EN)

Read the following medical natural language inference (NLI) task carefully and determine the correct relationship between the premise and the hypothesis.

Instructions:

\- Output ONLY one uppercase letter corresponding to the correct option (A, B, C, etc).

\- Do NOT output any explanation or extra text.

{question}

Options:

A. Entailment

B. Neutral

C. Contradiction

Answer:

## German (DE)

Lesen Sie die folgende medizinische NLI-Aufgabe und bestimmen Sie die Beziehung zwischen Prämisse und Hypothese. Wählen Sie die beste Antwort.

Anweisungen:

\- Geben Sie NUR einen Großbuchstaben aus, der der richtigen Option entspricht (A, B, C, usw.).

\- Geben Sie keine Erklärung oder zusätzlichen Text aus.

{question} Optionen: A. Implikation B. Neutral C. Widerspruch

Antwort:

## Spanish (ES)

Lea la siguiente tarea de inferencia de lenguaje natural médica (NLI) y determine la relación entre la premisa y la hipótesis. Seleccione la mejor respuesta.

Instrucciones:

\- Produzca SOLO una letra mayúscula correspondiente a la opción correcta (A, B, C, etc).

\- NO proporcione explicaciones ni texto adicional.

{question} Opciones: A. Implicación B. Neutral C. Contradicción

Respuesta:

## Portuguese (PT)

## TAREFA:

Leia a seguinte tarefa de inferência de linguagem natural médica (NLI) e determine a relação entre a premissa e a hipótese. Selecione a melhor resposta.

Instruções:

\- Produza APENAS uma letra maiúscula correspondente à opção correta (A, B, C, etc).

\- NÃO forneça explicações ou texto adicional.

{question}

Opções:

A. Implicação

B. Neutro

C. Contradição

Resposta:

## Japanese (JA)

タスク:

以下の医療自然言語推論（NLI）タスクを読み、前提と仮説の正しい関係を判断し、最も適切な回答を1つ選択してください。

指示:

\- 正しい選択肢に対応する大文字のアルファベット（A, B, C など）を1つだけ出力してください。

\- 説明や余計なテキストは出力しないでください。

{question}  
選択肢:  
A. 含意  
B. 中立  
C. 矛盾

回答:

## Chinese (ZH)

任务:

请阅读以下医学自然语言推理（NLI）任务，并判断前提与假设之间的正确关系，选择最佳答案。

说明:

\- 仅输出一个大写字母，对应正确选项（A, B, C 等）。

\- 不要输出任何解释或额外内容。

{question} 选项: A. 蕴含 B. 中立 C. 矛盾

答案:

## Thai (TH)

งาน: านโจท medical NLI อไป และ จารณาความ ม น ระห าง premise และ hypothesis จาก นเ อก ตอบ ก อง ด

แนะ :

\- แสดงเ ยง ว กษรภาษา งกฤษ ว ม ให ห ง ว ตรง บ ตอบ (A, B, C)

\- ามแสดง อ บายห อ อความเ มเ ม

{question}

วเ อก:

A. การอ มาน

B. เ นกลาง

C. ความ ดแ ง

ตอบ:

## Swahili (SW)

KAZI:

Soma kazi ifuatayo ya medical NLI na uamue uhusiano kati ya premise na hypothesis. Chagua jibu bora zaidi.

Maelekezo:

\- Toa herufi MOJA kubwa inayolingana na jibu sahihi (A, B, C, n.k.).

\- Usitoe maelezo au maandishi mengine yoyote.

{question}   
Chaguzi:   
A. Uthibitisho   
B. Kati   
C. Kupingana

Jibu:

## Zulu (ZU)

UMSEBENZI:

Funda umsebenzi olandelayo we-medical NLI bese unquma ubudlelwano phakathi kwe-premise ne-hypothesis.   
Khetha impendulo engcono kakhulu.

Imiyalelo:

\- Khipha uhlamvu OLULODWA olukhulu oluhambisana nenketho efanele (A, B, C, njll.).

\- Ungakhiphi incazelo noma umbhalo owengeziwe.

{question} Izinketho: A. Ukuvumelana <sup>ต่</sup> นี้ <sup>พิ</sup> B. Okuphakathi C. Ukuphikisana

อั ตั พิImpendulo:

## I.5 Open-ended QA prompts

## English (EN)

TASK:   
You are a medical knowledge expert.   
Read the following medical question carefully and provide the answer.   
Output Requirements:   
- Answer the question directly and accurately.   
- Do not output any text other than the answer, such as explanations, prefixes, or punctuation.   
Question:   
{question}   
Answer:

## German (DE)

AUFGABE:   
Sie sind ein Experte für medizinisches Wissen.   
Lesen Sie die folgende medizinische Frage sorgfältig und geben Sie die Antwort.   
Ausgabeanforderungen:   
- Beantworten Sie die Frage direkt und korrekt.   
- Geben Sie keinen zusätzlichen Text, keine Erklärung, kein Präfix oder zusätzliche Satzzeichen außer   
der Antwort selbst aus.   
Frage:   
{question}   
Antwort:

## Spanish (ES)

TAREA:   
Eres un experto en conocimiento médico.   
Lee cuidadosamente la siguiente pregunta médica y proporciona la respuesta.   
Requisitos de salida:   
- Responde a la pregunta de manera directa y precisa.   
- NO generes ningún otro texto, explicación, prefijo o puntuación aparte de la respuesta.   
Pregunta:   
{question}   
Respuesta:

タスク:  
あなたは医療知識の専門家です。  
次の医療に関する質問を注意深く読み、回答してください。  
出力要件:  
- 質問に直接かつ正確に答えてください。  
- 回答以外のテキスト、説明、接頭辞や句読点などは出力しないでください。  
質問:  
{question}  
回答:

## Portuguese (PT)

TAREFA:   
Você é um especialista em conhecimento médico.   
Leia cuidadosamente a seguinte pergunta médica e forneça a resposta.   
Requisitos de saída:   
- Responda à pergunta de forma direta e precisa.   
- NÃO produza qualquer outro texto, explicação, prefixo ou pontuação além da própria resposta.   
Pergunta:   
{question}   
Resposta:

## Japanese (JA)

## Chinese (ZH)

任务:你是一名医学知识专家。

请仔细阅读以下医学问题，并给出答案。

输出要求:

问题:{question}

答案:

## Thai (TH)

งาน: ณเ น เ ยวชาญ านความ ทางการแพท

โปรด าน ถามทางการแพท อไป อ างละเ ยด และใ ตอบ

ตอบ:

## Swahili (SW)

KAZI:

Wewe ni mtaalamu wa maarifa ya kitabibu.

Soma kwa makini swali lifuatalo la kitabibu na toa jibu.

Mahitaji ya matokeo:

\- Jibu swali moja kwa moja na kwa usahihi.

\- USITOe maandishi mengine, maelezo, viambishi au alama za ziada zaidi ya jibu lenyewe.

Swali:

{question}

Jibu:

## Zulu (ZU)

UMSEBENZI:

Unguchwepheshe wolwazi lwezokwelapha.

Funda ngokucophelela umbuzo wezokwelapha olandelayo bese unikeza impendulo.

Izidingo zokuphuma:

\- Phendula umbuzo ngokuqondile nangokunembile.

\- Ungafaki omunye umbhalo, incazelo, izichasiselo noma izimpawu zokubhala ngaphandle kwempendulo uqobo.

Umbuzo: {question}

Impendulo: