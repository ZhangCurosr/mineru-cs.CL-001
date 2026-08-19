# The IOL-AI Challenge: An Open Challenge towards Advancing Linguistic Reasoning

Eduardo Sánchez<sup>⋆1</sup>, Rita Berrada<sup>⋆2,3</sup>, Dan-Mircea Mirea<sup>⋆4</sup>, Sara Rajaee<sup>5</sup>, Alexander Piperski<sup>6</sup>, Ana Meta Dolinar<sup>5</sup>, Boris Iomdin<sup>7</sup>, Andrey Nikulin<sup>8</sup>, Mariya Shmatova<sup>9</sup>, Marzieh Fadaee<sup>3</sup>, and Julia Kreutzer<sup>♦3</sup>

<sup>1</sup>University College London & Meta, <sup>2</sup>CentraleSupélec & McGill University, <sup>3</sup>Cohere Labs, <sup>4</sup>Princeton University, <sup>5</sup>University of Amsterdam, <sup>6</sup>Stockholm University, <sup>7</sup>Faculty of Liberal Arts and Sciences, <sup>8</sup>Universidade Federal de Goiás, <sup>9</sup>Independent

## Abstract

Reasoning in LLMs is overwhelmingly studied in domains that provide a model with rules: mathematics and code. Linguistic puzzles invert this: the solver must first discover the system before reasoning within it. We present the IOL-AI Challenge, an open-science competition run on the unseen problems of the International Linguistics Olympiad (IOL) 2026 Individual Contest, evaluated both automatically and, for the first time, by members of the oficial IOL Jury under the same rubrics applied to human contestants. The challenge drew 731 submissions from 46 teams under a strict compute budget (one T4, 30 mins). We additionally benchmark 15 unconstrained frontier and open models, with Claude Opus 4.8 earning a jury score equivalent to a gold medal, while both resource-constrained systems we submitted for jury grading scored in the range of the bottom 5% of contestants. Capability was not determined by scale: 14B submissions outperform models twice their size, and gains come from decoding and output-handling rather than model capacity. We also found that automatic metrics rank systems exactly as the jury does, but compress the scale, upscoring weak systems by 13 points and understating strong ones. Our analysis shows that while frontier models might have prior knowledge about some of the problem languages, it does not significantly help them solve the linguistic reasoning tasks, leaving linguistic reasoning as a strong benchmarking proxy for generalizable reasoning skills.

## 1 Introduction

Reasoning has become one of the central paradigms for improving the capabilities of large language models (DeepSeek-AI, 2025; Marjanovic et al., 2026; Qwen Team, 2026; Gemma Team, 2026). Rather than relying only on larger models, more data, or longer pretraining, recent systems increasingly improve their performance by allocating compute at inference time and generating intermediate reasoning (Snell et al., 2025; Geiping et al., 2025), exploring alternative paths (Qiu et al.,

2026; Brown et al., 2024), verifying solutions, and revising their answers (Hong et al., 2024; Zhang et al., 2025; Liu et al., 2025).

The advances have been impressive yet the way we study reasoning has remained remarkably narrow. The dominant tests of reasoning are mathematics (Shen et al., 2025; Chen et al., 2025a; Zhang et al., 2026) and coding (Chen et al., 2025b; He et al., 2026), which are domains with formal structures, unambiguous answers, and abundant automatically verifiable feedback. However, realworld problems often do not begin with a well-specified system of rules. Before solving the problem, a solver may first need to discover what the relevant structure is. Most reasoning benchmarks abstract away this step: they provide the rules, the formalism, or the problem space, and evaluate how well a system can reason within it. A more fundamental question is whether a system can infer a useful structure from evidence and then reason within the discovered structure. This ability is critical for generalization: Solving new problems in unfamiliar domains requires identifying regularities, forming hypotheses, and testing them against new evidence.

Natural language ofers a particularly compelling domain in which to explore this question. Human language is an intricate system of compositional structure that is highly variable, irregular, and context-dependent. Linguistic problems (sometimes referred to as linguistic puzzles)—a genre of puzzles where the solver is typically expected to disentangle linguistic phenomena of an unfamiliar language based on a limited corpus (Zaliznjak, 1963; Derzhanski & Payne, 2010)—provide a natural testbed for this ability. To understand a language that one has never encountered before, one must infer latent structure from sparse observations: identify recurring units, discover grammatical transformations, form hypotheses about meaning, and use these hypotheses to predict novel forms (Aycock et al., 2025).

The majority of linguistic problems come from linguistic olympiads, the largest of which is the International Linguistics Olympiad (IOL)<sup>1</sup>. This annual competition unites secondary school students from many countries (46 at the 2026 edition), who compete in solving linguistic problems. Each problem challenges contestants (1) to analyze linguistic data from one or more languages they have likely never seen before, (2) to deduce a set of rules that can fully explain the data, and (3) to use these rules to translate into and out of the unfamiliar language. Contestants’ solutions are graded by an international body of linguists, polyglots and past IOL contestants called the IOL Jury, all of whom have substantial expertise in composing and evaluating linguistic problems.

Prior work has uniformly reported that even frontier reasoning models consistently underperform at solving such puzzles (Garnham & Shareghi, 2026; Kocmi et al., 2025; Lian et al., 2025). We take past benchmarking eforts three steps further: (1) We engage the open-science community to advance linguistic reasoning of open and resource-constrained models in the form of a timebound challenge with a leaderboard: The IOL-AI Challenge. (2) We partner with organizers of the IOL to benchmark models on completely new and unseen problems from the IOL 2026 Individual Contest, which removes any concerns of benchmark leakage even for closed models. (3) We obtain expert-level human evaluations from the oficial IOL jury—which allows us to identify expert-level judgments of the state of linguistic reasoning, where previously we had to rely on shallow automatic matching with oficial solutions.

Our study thus constitutes the most recent, most thorough and in-depth evaluation of linguistic reasoning capabilities of LLMs at scale. In contrast to prior work, we find that proprietary models have caught up with the task, but open models lag wide behind, and under resource constraints mostly fail the task—indicating that generalizable reasoning is still a very open problem.

## 2 Related Work

Reasoning Capabilities The evaluation of the reasoning capabilities of large reasoning models has recently become a principle of AI benchmarks to measure their progress (Qwen Team, 2026; Gemma Team, 2026). Reasoning datasets cover a wide range of tasks including, but not limited to, mathematics and theorem proving (Zhang & Math-AI, 2025; Wang et al., 2026; Yang et al., 2023), code generation (Jimenez et al., 2024; Merrill et al., 2026), science and multidisciplinary (Rein et al., 2024; Center for AI Safety et al., 2026), and visual puzzles (Chollet et al., 2026). Solving problems from the International Mathematical Olympiad (IMO)<sup>2</sup>, which is widely known as the most dificult math contest, has been a big goal in mathematical reasoning. In 2025, Huang & Yang (2025) showed how using a verification-and-refinement pipeline, which proved useful in mathematics given the easily verifiable correctness of the individual logic steps in the solution, drastically increases the performance of SotA modelson IMO problems and achieved a gold-meda score.

Linguistic Reasoning Benchmarks Şahin et al. (2020) proposed PuzzLing Machines including Linguistic Olympiad examples. Later, Bean et al. (2024) gathered a series of translation and linguistic reasoning problems from the UK Linguistic Olympiad, presenting the LingOly benchmark. Similarly, Linguini, IOLBENCH, and LINGOLY-TOO have been built using IOL problems, including a diverse set of low-resource languages (Sánchez et al., 2025; Goyal & Dan, 2025; Khouja et al., 2026). While prior benchmarks focus only on correct final answers, LOBSTER (Lian et al., 2025) has provided step-by-step solutions for each problem using first an advanced LLM to generate them, and then human experts to post-edit solutions.

Advances in Linguistic Reasoning with AI Advanced LLMs still struggle with linguistic reasoning, yet eforts to improve their performance remain surprisingly limited. Trying to fill this gap, Zhu et al. (2025) has shown that LLMs’ performance improves in a multi-turn step-by-step setting. Moreover, providing analogical exemplars in the context enhances linguistic reasoning performance (Ramji & Ramji, 2025). Garnham & Shareghi (2026) study test-time scaling of large-scale LLMs and find that while benchmark performance can increase by a few points, it remains well below math and commonsense reasoning benchmarks.

## 3 Linguistic Reasoning at IOL

IOL started in 2003, growing steadily from just 36 contestants from 6 countries at its first edition in Borovets, Bulgaria to a record of 255 contestants from 46 countries at the current edition in Bucharest, Romania. Each participating country can send one or two teams of 4 contestants, usually selected through the country’s national olympiad. National olympiads<sup>3</sup> vary in size and similarity to the IOL, some having a diferent format in terms of number and length of linguistics puzzles. There are other local or international linguistic olympiads, such as the Online Olympiad in Linguistics.<sup>4</sup> The goals of IOL and other linguistic olympiads alike are: to popularize linguistics at the pre-university level, to raise awareness of linguistic diversity and low-resource languages globally, and to promote problem-solving and linguistic skills beyond traditional school subjects. For a detailed history of linguistic olympiads and a discussion of their declared goals, the reader is referred to Martins (2022).

<table><tr><td>Problem Structure Prompt Type Context</td><td></td><td></td><td>Question</td></tr><tr><td>Rosetta Stone</td><td>Translation</td><td>and their Komnzo translations: ...</td><td>Here are some sentences in English Translate into English in all possible ways: ...</td></tr><tr><td>Chaos-and-order</td><td>Fill-in Blanks</td><td>Sakurabiat speakers. ...</td><td>Here is the family tree of a family of Fill in the blanks (i-vi). There is only one possible answer for each blank, but each answer is not necessarily just one word.</td></tr><tr><td>Chaos-and-order</td><td>Match Letters</td><td>Yélì Dnye and their English transla- dences. tions in arbitrary order: ...</td><td>Here are some words and phrases in Determine the correct correspon-</td></tr></table>

Table 1: Examples from the IOL 2026 Individual Contest, structured into context and question to prompt LLMs.

## 3.1 Linguistic Problems

The concept of linguistic problems goes back to a workbook by Gleason (1955) and was first formalized by Zaliznjak (1963); more recent presentations of the genre include Derzhanski & Payne (2010); Martins (2022); Silva & Costa (2022); Nikulin (2024); Neacșu (2024). A core idea behind this concept is the principle of self-suficiency, that is, the problem must be solvable without prior knowledge of any foreign languages or advanced linguistic concepts. Low-resource languages are typically chosen in order to minimize the risk of any solver being familiar with the language, thus ofering them unfair advantage. Solvers (student contestants in the case of IOL) are given enough information about a language, including but not limited to a few sentences and their translations, a numeral system, or a script sample. To solve a given problem, students need to infer the relevant grammatical, phonological, and lexical patterns from the data, and later apply it to answer the assignments. By default, contestants are expected to provide a theoretical explanation for their answers, using only concepts from secondary education curricula (such as consonant, sufix, or verb). Contestants are not required nor expected to have received prior training in linguistics— even though in recent years there has been a surge in training materials, solving problems assigned in past installments of linguistics olympiads remains the most common way of preparing for IOL (Belikov et al., 2006; Derzhanski & Velinov, 2012; 2013; Radev, 2013a;b; Silva & Costa, 2022; Neacșu, 2024).

Linguistic problems are often classified by the subdiscipline of linguistics they illustrate—e.g., phonetics/phonology, morphology, syntax, lexical semantics, kinship terminology, number terms (Derzhanski & Veneva, 2018), computational concepts (Littell et al., 2013), corpus linguistics (Iomdin et al., 2013), writing systems. They can also be classified by problem structure (Martins, 2022; Nikulin, 2024; Neacșu, 2024), the most common structures being “Rosetta stone problems” (Bozhanov & Derzhanski, 2013), which involve translation or transcription assignments based on a bilingual corpus; and “chaos-and-order problems”, which involve matching assignments.

## 3.2 The IOL Problem Set

The IOL problem set consists of five problems meant to be solved individually by each contestant within 6 hours (the Individual Contest) and one problem designed to be solved collaboratively by national teams of 4 (the Team Contest). Any working language may be requested for the competition and assignments are equivalent in all working languages (Derzhanski 2013). The problem set is crafted and multilingually rendered by the IOL Problem Committee in the year prior to each IOL.

The individual problem set at IOL 2026 included a problem on phonetics and orthography in Central

Alaskan Yup’ik (P1); a chaos-and-order problem on lexical semantics, in particular semantics of color terms, in Yélî Dnye (P2); a Rosetta stone problem on syntax in Iquito (P3); a chaos-andorder problem on kinship terminology in Sakurabiat (P4); and a Rosetta stone problem on verb morphology in Komnzo (P5).

## 3.3 Grading

Each problem is graded by a subset of the IOL jury. In the weeks leading to IOL, after the problem set has been assigned, each grading group develops a grading scheme for their problem, deciding on which criteria to grade and how to weight them. Grading schemes allocate points to assignments and the theory part (the explanation of the rules) separately, as criteria: each assignment is usually one criterion, whereas the theory portion contains multiple criteria, each corresponding to a main rule that the solvers need to identify.

At IOL, each contestant’s submission for a problem is graded independently by at least two jurors within the grading group, with translation assistance from other jurors or external language specialists if needed. In the event of a discrepancy between two or more jurors regarding any single criterion, the discrepancy must be resolved through discussion until consensus is achieved.

## 4 The IOL-AI Challenge 2026

The challenge was set up as an open-science competition preceding the IOL (outreach eforts described in Section A), with a compute-restricted testing environment and a truly unseen test set that no participants had access to. Submissions were accepted for a month until July 26, 2026 when the on-site IOL competition started.

## 4.1 Technical Setup

The Task The dataset includes the data from the 5 problems of the English version of the IOL 2026 Individual Contest, a total of 14 sub-assignments or tasks that these problems collectively contain. We distinguish 3 types of tasks: (1) Translation, where given a context, it is asked to translate a sentence from “Solverese” (here: English) to an unseen extremely low-resource language, (2) Fill-in Blanks, where given a set of examples, the model is asked to fill the missing forms, and (3) Match Letters, where the task is to match a set of phrases to their correspondences. Table 1 presents an example of test data with diferent types. We split the tasks into a development (8 tasks) and a test set (6 tasks) balanced across task types, where the development set is used for computing automatic scores on a public leaderboard, while scores on the test set are not revealed during the competition, but only after its end, as part of a private leaderboard that contains scores for the complete set of tasks. One task is excluded from automatic scoring as it requires only an explanation that cannot be graded automatically. Submissions eligible to participate in the human evaluation had to provide a short textual explanation with every task solution as well.

Format We follow Linguini in the task format (Sánchez et al., 2025) where every data point has a context containing the given problem and a query for the asked question. Most problems require returning a list of answers, such as filling in a list of blanks, or an ordered list of letters. Reference solutions may also contain multiple valid answers for each problem, if there are any ambiguities. Any visual clues are turned into text, and diagrams into JSON format, following the setup in (Kocmi et al., 2025).

Automatic Evaluation Following previous work on linguistic reasoning (Sánchez et al., 2025; Bean et al., 2024), we employ ChrF (Popović, 2015) and exact match (EM) to evaluate the final answers. We report the geometric mean (GM) of these metrics because ChrF will likely overestimate true answer quality (e.g., partial matches can be obtained from copying words from the task), and EM will likely underestimate it (e.g. answer formatting or minor morphological variations). ChrF and EM are first computed on each individual example (taking the highest score across reference solutions if there are multiple), then averaged across examples.<sup>5</sup>

Implementation The challenge runs on Hugging Face (HF), built on the open-source competitions framework. A submission is a public HF model repository holding a self-contained inference script alongside the model weights themselves, that is run on an evaluation sandbox without network access and writes answers and optionally explanations into an output file. This evaluation sandboxing is necessary to prevent leaking the IOL-AI data. The dataset was uploaded directly by members of the IOL jury and problem committee (AP, DMM) to a private HF dataset. Neither participants nor other organizers ever saw the data. Each submission runs as an isolated HF Job on a single NVIDIA T4 GPU (16 GB VRAM, 8 vCPUs, 30 GB RAM) under a 30-minute wall-clock limit.

After each run, teams receive the score on the dev split (scored server-side based on the output file);   
failed submissions additionally return an excerpt from the error log (e.g., “time limit exceeded”).   
Teams have up to 10 submissions per day and select two for the private leaderboard.

## 4.2 Baselines

We provided four basic baselines, submitted through the same pipeline as participant entries. They relied on four diferent open-weights models of diferent scales, with slightly diferent amounts of prompt customization. Section C describes them in detail. The baselines were visible to participants on the leaderboard throughout the competition. Table 2 reports their automatic scores.

<table><tr><td>Baseline</td><td>chrF</td><td>EM</td><td>GM</td></tr><tr><td>Qwen2.5-14B-Instruct-AWQ</td><td>19.70</td><td>4.49</td><td>9.40</td></tr><tr><td>Tiny Aya Global (3.35B)</td><td>16.09</td><td>4.10</td><td>8.12</td></tr><tr><td>Qwen2.5-7B-Instruct (NF4)</td><td>19.65</td><td>2.56</td><td>7.10</td></tr><tr><td>Qwen2.5-1.5B-Instruct (starter)</td><td>12.51</td><td>1.28</td><td>4.00</td></tr></table>

Table 2: Automatic evaluation scores of organizer baselines. EM = exact match, GM = geometric mean.

## 4.3 Extended Model Benchmarking

To complement the resource-constrained submissions, we also benchmark 15 existing reasoning models (details in Section D), including “frontier” models (open and proprietary), i.e., the largest, most resource-consuming, most recent, and reasoning-heavy models, and open “mid-size” models to bridge the gap to the IOL resource constraints. Given that the latter ones are individually not competitive with the former, we aggregate the answers of four mid-sized models (Gemma 4 31B, Qwen 3.6 27b, DeepSeek R1 32B, GLM 4.7 Flash) in a best-of-n ensemble (BoN; oracle selection: choosing the best scoring model for each task) to measure the ceiling of what these models can currently achieve, inspired by earlier ensemble success (Garnham & Shareghi, 2026).

## 4.4 Human Evaluation at the IOL

We consider the top 10 challenge submissions with provided explanations along with the best closedand open-source reasoning models for human evaluation, before selecting five systems to undergo human evaluation (see Section 5.2). Human evaluation is performed by a subset of the IOL jury (at least 2 graders per problem) using a very similar procedure to the grading of contestant solutions at IOL, including anonymization with respect to the model information and resolution of discrepancies between jury members through discussion. Details are described in Section G.

## 5 Results

We first discuss the results from the automatic evaluation stage (Section 5.1), covering a larger set of models, before zooming in on the five selected submissions for the human evaluation (Section 5.2). Lastly we perform an analysis on the knowledge leveraged by top-scoring models (Section 5.3).

## 5.1 Automatic Evaluation

## 5.1.1 Submissions from the Open Challenge

We received 731 submissions (80.7% ran successfully) from 46 external teams over the one-month window. We describe submission statistics in more detail in Section H, and focus on the top 10 here, listed in Table 3. Nine of the top ten entries run a quantized 14B Qwen model, the largest model to comply with the compute constraint. The best submission more than doubles our strongest baseline with the same model (GM 9.40 19.79), showing that decoding and output handling are efective levers under constrained model size.

<table><tr><td>#</td><td>Team</td><td>chrF</td><td>EM</td><td>GM</td><td>Model</td></tr><tr><td>1</td><td>arvindcr4*</td><td>31.57</td><td>12.41</td><td>19.79</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>2</td><td>BigRatz*</td><td>30.75</td><td>11.55</td><td>18.85</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>3</td><td>hhhar</td><td>29.14</td><td>11.35</td><td>18.19</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>4</td><td>friedspaghetti</td><td>28.98</td><td>11.07</td><td>17.91</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>5</td><td>mastermind</td><td>27.78</td><td>10.07</td><td>16.73</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>6</td><td>Hul</td><td>28.41</td><td>9.84</td><td>16.72</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>7</td><td>jbuaba</td><td>27.68</td><td>9.84</td><td>16.51</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>8</td><td>attractordynamics</td><td>27.56</td><td>9.12</td><td>15.85</td><td>Qwen2.5-14B AWQ</td></tr><tr><td>9</td><td>LōPE-ÑTU</td><td>23.78</td><td>9.62</td><td>15.12</td><td>Qwen3.5-4B Q8 GGUF</td></tr><tr><td>10</td><td>srikarkashyap</td><td>26.99</td><td>7.98</td><td>14.67</td><td>Qwen3-14B AWQ</td></tr></table>

Table 3: Automatic evaluation scores for the top 10 from the leaderboard of the IOL-AI challenge. Submissions with ⋆ were selected for jury evaluation. Dashed lines indicate first, second, and third place winners on the leaderboard.

Technique adoption and measured efects We inspected the archived repositories and tagged the techniques used in each submission. Where a team submitted runs both with and without a technique, we measure its efect as the within-team diference in mean private score (n = teams with runs of both kinds), this controls for stronger teams simply trying more things. Greedy decoding was near-universal (38 teams), and 21 teams additionally used temperature sampling, mostly for retries. Self-consistency voting (24 teams) adds +2.08 points (n = 21). Retry or reformat loops (17 teams) add +1.40 (n = 14). Chain-of-thought prompting was widely tried (33 teams) but costs 1.55 (n = 25) and none of the top ten submissions used it. JSON-structured output (16 teams) is the most damaging at 4.66 (n = 10). Fine-tuning or LoRA on past puzzles brought no gain, and neither did retrieval, consistent with problems that are self-contained by construction.

Leading Approaches Leading submissions converge on a common recipe. They lower the repetition penalty from its default greedy-mode value of 1.05 to 1.0, a silent parameter that penalizes generating the same characters several times in a row, a frequent pattern in these languages. None uses chain-of-thought prompting: it was tried repeatedly, and runs without it consistently scored higher. All top entries anchor on a greedy pass, and when they sample, draw retries at temperature 0.5 as time allows, replacing the greedy answer only on agreement. Token budgets are set adaptively from the wall-clock time remaining, and the submission file is written incrementally. Part of this convergence reflects copying, since submission repositories were public throughout the competition: the 1st and 5th entries run the same script, difering only in whitespace, and their score diference reflects run-to-run variance rather than any diference in method. We return to this in the Limitations. The individual pipelines are described in Section H.

## 5.1.2 Extended Models

<table><tr><td>Model</td><td>chrF</td><td>EM</td><td>GM</td><td>LFMR</td></tr><tr><td colspan="5">Proprietary Models</td></tr><tr><td>Claude Opus 4.8 ★</td><td>82.9</td><td>68.1</td><td>75.1</td><td>100.0</td></tr><tr><td>Gemini 3.6 Flash ★</td><td>70.6</td><td>45.0</td><td>56.4</td><td>100.0</td></tr><tr><td>GPT 5.6 Sol</td><td>63.3</td><td>43.1</td><td>52.2</td><td>100.0</td></tr><tr><td colspan="5">Open Large Models</td></tr><tr><td>Kimi K2.6</td><td>57.1</td><td>35.8</td><td>45.2</td><td>100.0</td></tr><tr><td>GLM 5.2</td><td>58.0</td><td>35.0</td><td>45.0</td><td>92.9</td></tr><tr><td>DeepSeek V4 Pro</td><td>53.6</td><td>31.5</td><td>41.0</td><td>92.9</td></tr><tr><td>Qwen 3.7 Max</td><td>52.2</td><td>29.4</td><td>39.2</td><td>100.0</td></tr><tr><td>Inkling</td><td>48.0</td><td>24.3</td><td>34.1</td><td>100.0</td></tr><tr><td>Nemotron 3 Ultra</td><td>41.5</td><td>20.1</td><td>28.9</td><td>64.3</td></tr><tr><td>Command A+</td><td>34.1</td><td>11.1</td><td>19.4</td><td>100.0</td></tr><tr><td>MiniMax M3</td><td>14.4</td><td>7.9</td><td>10.6</td><td>78.6</td></tr><tr><td colspan="5">Open Mid-Size Models</td></tr><tr><td>Qwen 3.8 27B (think)</td><td>51.2</td><td>30.9</td><td>39.8</td><td>92.9</td></tr><tr><td>Glimmer 30B (high)</td><td>48.7</td><td>30.3</td><td>38.4</td><td>85.7</td></tr><tr><td>Qwen 3.6 27B (think)</td><td>37.1</td><td>19.2</td><td>26.7</td><td>92.9</td></tr><tr><td>Gemma 4 31B</td><td>37.0</td><td>14.6</td><td>23.2</td><td>92.9</td></tr><tr><td>Qwen 3.6 27B</td><td>25.2</td><td>14.9</td><td>19.4</td><td>64.3</td></tr><tr><td>DeepSeek R1 32B</td><td>23.7</td><td>3.0</td><td>8.5</td><td>100.0</td></tr><tr><td>GLM 4.7 Flash</td><td>6.6</td><td>0.0</td><td>0.0</td><td>21.4</td></tr><tr><td>BoN Open (oracle) *</td><td>48.5</td><td>23.2</td><td>33.5</td><td>100.0</td></tr></table>

Table 4: Automatic evaluation scores for the extended model set on the IOL 2026 problems (after custom answer parsing). LFMR = line format match rate. Model outputs that were chosen for the jury are marked with ⋆.

Table 4 contains the automatic evaluation scores for our selected 15 individual models and the ensemble. We report line format match rate (LFMR) besides the quality scores to indicate where low scores are due to low format compliance. Note that all submissions undergo a custom answer parsing process that e.g. strips markdown (Section D). As expected from evaluations in prior work, proprietary models perform strongest, with a noticeable 19-point gap between Claude-Opus-4.8 and the following Gemini-3.6-Flash and GPT5.6-Sol. Open (large) models follow after a 7 point gap, with a large range in scores, going down to 10.6. Of the mid-sized models, Gemma4-31B-It scores highest (23.2), and the ensemble gains another 10 points on top of it, catching up with some of the larger open models. Performance is therefore not strictly determined by scale: as Figure 1 shows, resource-constrained submissions at 14B double the organizer baselines and match or exceed the unconstrained open mid-size tier, though all open systems remain far below the frontier. As linguistic reasoning is not a typical post-training task, and most models struggle with the output format, “tweaking” inference goes a long way.

![](images/a3f33e04bf0554f68e82c5a4c9668376dbabb710539ea11a28e11c8296304ff5.jpg)  
Figure 1: Distribution of automatic scores (GM) by system tier, spanning from organizer baselines and resource constrained competition submissions to open mid-size, open large, and proprietary frontier models.

## 5.1.3 Comparison across Tasks

Among the top 10 submissions (Figure 2a), dificulty separates cleanly by problem. Iquito (P3) is the hardest: both of its tasks score 0% exact match for all ten teams. Yup’ik (P1) and Yélî Dnye (P2) follow closely, with no team solving Yup’ik writing or either Yélî Dnye translation direction, and only the closed-form items (Yélî Dnye letter and picture matching, Yup’ik letter matching) earning partial credit. On these unsolved tasks, chrF often remains in the 23 to 53 range, indicating that answers contain some correct components. Submissions perform best on Sakurabiat (P4), whose short, closed items (birth order, number matching) are answered by nearly every team. Komnzo (P5) shows the largest asymmetry within a single problem: Komnzo English is the bestsolved task across all submissions, while English Komnzo receives 0% throughout. Teams are thus able to analyze Komnzo forms well enough to translate them into English, but not to produce correct forms in the language itself.

Frontier models (Figure 2b) score higher on every problem but rank them diferently. Yélî Dnye (P2) becomes the hardest problem (about 20% geomean), with both translation directions near 11%, whereas Iquito (P3), unsolved by all submissions, moves to the middle of the range. Sakurabiat (P4) remains the easiest (about 55%), and the Komnzo (P5) asymmetry persists: English Komnzo stays low (about 28.5%) even for models that succeed in the opposite direction. On most problems, scores follow overall model strength, but translation tasks diverge from this pattern: only proprietary models obtain non-zero scores on Yélî Dnye English, while the same models fail

Komnzo English, where several open models earn substantial credit.

![](images/c4e7f842e39351be713940eb7b4584555a6541ed047793b586e5fdb83624b71a.jpg)  
(a) Top 10 submissions from the IOL-AI challenge.

![](images/4988d35b056125518b8376375624f4d326051d72a5b39769752e593b912fa571.jpg)  
(b) Frontier models.  
Figure 2: Breakdown of scores (geomean %) across the individual tasks of the 5 IOL 2026 problems, for the top 10 challenge submissions (top) and frontier models (bottom).

## 5.2 Human Evaluation by the IOL Jury

We selected the top 2 frontier models according to their automatic evaluation scores (Claude Opus 4.8 and Gemini-3.6-Flash) for human evaluation. From the top submissions we selected the first one, and then manually went through subsequent ranks and verified the explanations, and found that many of them were either templated or meaningless (copies of the same sentence). The submission ranked 11th stood out as it provided structured explanations with schemata, so we selected it to provide a contrastive solution. From the open mid-size models, we submitted the ensemble, representing the ceiling of what a team of mid-size models can achieve if assembled right. In this way, we hope to capture the full spectrum from heavily tuned, severely resource-constrained systems, over an ensemble of mid-size open models, to proprietary frontier models.

<table><tr><td></td><td colspan="2">Total Score</td><td colspan="2">P1</td><td colspan="2">P2</td><td colspan="2">P3</td><td colspan="2">P4</td><td colspan="2">P5</td></tr><tr><td>Model Name</td><td>Jury</td><td>Aut.</td><td>Jury</td><td>Aut.</td><td>Jury</td><td>Aut.</td><td>Jury</td><td>Aut.</td><td>Jury</td><td>Aut.</td><td>Jury</td><td>Aut.</td></tr><tr><td>Claude Opus 4.8</td><td>79.5</td><td>75.1</td><td>18.5</td><td>14.5</td><td>10.5</td><td>16.0</td><td>15.5</td><td>12.2</td><td>19</td><td>23.2</td><td>16</td><td>9.2</td></tr><tr><td>Gemini 3.6 Flash</td><td>60.5</td><td>56.4</td><td>13</td><td>10.8</td><td>9</td><td>11.5</td><td>15</td><td>9.2</td><td>13</td><td>19.0</td><td>10.5</td><td>5.8</td></tr><tr><td>BoN Open (oracle)</td><td>20.3</td><td>33.5</td><td>4.5</td><td>3.4</td><td>2.3</td><td>6.7</td><td>4.5</td><td>7.2</td><td>3.5</td><td>7.3</td><td>5.5</td><td>8.9</td></tr><tr><td>arvindcr4 (#1)</td><td>5.9</td><td>19.8</td><td>0.8</td><td>1.3</td><td>2.1</td><td>5.8</td><td>0.5</td><td>2.6</td><td>2</td><td>6.3</td><td>0.5</td><td>3.8</td></tr><tr><td>cabanosss (#11)</td><td>2.5</td><td>14.6</td><td>1</td><td>1.0</td><td>1</td><td>5.3</td><td>0</td><td>2.2</td><td>0</td><td>2.3</td><td>0.5</td><td>3.8</td></tr></table>

Table 5: Jury evaluation results by problem for the selected AI submissions compared to the automatic GM scores (Aut.).

## 5.2.1 Quantitative Comparison

The scores given by the IOL jurors showed the same rank-order of the models as the automatic score (Table 5). Notably, Opus 4.8 received a score equivalent to that of an IOL gold medalist (lowest gold medal this year: 70.0), a score that would have placed it in fourth place in this year’s ranking. An example of its explanations for P5 are shown side-by-side with the reference solution in Figure 3. Gemini 3.6 received a score equivalent to that of a silver medal (lowest silver medal this year: 58.4). The rest of the models were below the threshold for honorable mentions (35.3), meaning they all performed worse than 50% of contestants this year. In fact, the two models from the challenge would have both ranked in the bottom 5% of scores. Agreement between the two protocols is near-perfect in rank (Spearman ρ = 1.00, Pearson $r = 0 . 9 9 , n = 5 )$ but not in magnitude. The two proprietary models score higher with the jury than automatically (+4.4 and +4.1 points), whereas the ensemble and both challenge submissions score substantially lower ( 13.2, 13.9 and 12.1); the jury’s range therefore spans 77.0 points against 60.5 automatically. Figure 4 breaks these diferences down by problem: the automatic metric sits above the jury on nearly every problem for the weaker systems, while the direction flips for the two proprietary models on P1, P3 and P5. Automatic scores overestimate the quality of the weaker submissions, due to absence of explanation scoring and overestimating influence of ChrF in the geometric mean; conversely, strong systems are under-credited, because a correct analysis stated in prose earns jury points but nothing automatically.

## 5.2.2 Qualitative Comparison

Beyond the quantitative diferences, the IOL jurors reported several recurring qualitative diferences between the AI solutions and the human solutions they usually grade:

1. Local rather than consistent explanations. Because our prompting procedure elicited

1. object:

• Aspect/Tense:

Twenty-Third International Linguistics Olympiad (2026) Individual Contest Solutions

Problem 5. Rules:

• Word order: time adverb verb

<table><tr><td rowspan=1 colspan=1>aspect</td><td rowspan=1 colspan=1>imperfective(action is extended in time)</td><td rowspan=1 colspan=1>perfective(action is short-lived)</td></tr><tr><td rowspan=1 colspan=1>tense</td><td rowspan=1 colspan=1>- non-past (I am calling themnow)– past (I was calling them a weekago)</td><td rowspan=1 colspan=1>– recent past (I called them yes-terday)– past (I called them a week ago)</td></tr></table>

• Structure of the verb form:

Each verb = [OBJECT prefix] + [aspect stem] + (durative -w-) + [SUBJECT/TENSE suffix]. Time adverbs: zena “now” (present), kayé “yesterday” (recent past), nümä “a week ago” (remote past)

(durative vs. perfective):
<table><tr><td rowspan=1 colspan=1>verb</td><td rowspan=1 colspan=1>durative</td><td rowspan=1 colspan=1>perfective</td></tr><tr><td rowspan=1 colspan=1>hold</td><td rowspan=1 colspan=1>fath</td><td rowspan=1 colspan=1>faf</td></tr><tr><td rowspan=1 colspan=1>meet</td><td rowspan=1 colspan=1>wagr</td><td rowspan=1 colspan=1>waf</td></tr><tr><td rowspan=1 colspan=1>call</td><td rowspan=1 colspan=1>bräkn</td><td rowspan=1 colspan=1>às</td></tr><tr><td rowspan=1 colspan=1>see</td><td rowspan=1 colspan=1>mar</td><td rowspan=1 colspan=1>mar</td></tr><tr><td rowspan=1 colspan=1>return</td><td rowspan=1 colspan=1>brig</td><td rowspan=1 colspan=1>brim</td></tr></table>

<table><tr><td rowspan=1 colspan=5>object:</td><td rowspan=1 colspan=1>aspect</td><td rowspan=1 colspan=1>imperfective</td><td rowspan=1 colspan=1>perfective</td></tr><tr><td></td><td rowspan=1 colspan=2>aspect</td><td rowspan=1 colspan=1>imperfective</td><td rowspan=1 colspan=1>perfective</td><td rowspan=1 colspan=1>see</td><td rowspan=1 colspan=1>-mar-</td><td rowspan=1 colspan=1>-mar-</td></tr><tr><td></td><td rowspan=1 colspan=2>1st person sg.</td><td rowspan=1 colspan=1>wo-</td><td rowspan=1 colspan=1>ZW-</td><td rowspan=1 colspan=1>return</td><td rowspan=1 colspan=1>-brig-</td><td rowspan=1 colspan=1>-brim-</td></tr><tr><td></td><td rowspan=1 colspan=2>1st person pl.</td><td rowspan=1 colspan=1>n-</td><td rowspan=1 colspan=1>nzn-</td><td rowspan=1 colspan=1>hold</td><td rowspan=1 colspan=1>-fath-</td><td rowspan=1 colspan=1>-faf-</td></tr><tr><td></td><td rowspan=2 colspan=1>2nd person pl.3rd person pl.</td><td></td><td rowspan=2 colspan=1>e-</td><td rowspan=2 colspan=1>th-</td><td rowspan=1 colspan=1>meet</td><td rowspan=1 colspan=1>-wagr-</td><td rowspan=1 colspan=1>-waf-</td></tr><tr><td></td><td></td><td rowspan=1 colspan=1>call</td><td rowspan=1 colspan=1>-bräkn-</td><td rowspan=1 colspan=1>-S-</td></tr></table>

Durative forms insert before the suffix; perfective forms do not

Object prefixes:
<table><tr><td rowspan="5">6. subject:</td><td>1st person sg.</td><td>(-aé &gt; -a)</td></tr><tr><td>1st person pl.</td><td>-é -e (-ae &gt; -ake)</td></tr><tr><td>2nd person pl. = 3rd person pl.</td><td>-th (-wth &gt; -wrth)</td></tr></table>

<table><tr><td rowspan=1 colspan=1>object</td><td rowspan=1 colspan=1>durative</td><td rowspan=1 colspan=1>perfective</td></tr><tr><td rowspan=1 colspan=1>me (1sg)</td><td rowspan=1 colspan=1>wO-</td><td rowspan=1 colspan=1>zwe-</td></tr><tr><td rowspan=1 colspan=1>us (1pl)</td><td rowspan=1 colspan=1>n-</td><td rowspan=1 colspan=1>nzn- (from 13 nzn+äs+th)</td></tr><tr><td rowspan=1 colspan=1>them/you-all (2/3pl)</td><td rowspan=1 colspan=1>e-</td><td rowspan=1 colspan=1>the-/thã-</td></tr></table>

(a) IOL solution explanation.  
For the “th-” prefix, vowel = e (the-) if subject is 1st person, ä (thä-) if 3rd person (cf. 4 we→you-al = the-, 8 they→you-al = thä-). e+ä merges to ä (14: zwe+äs → zwäs).  
(independent of aspect):

<table><tr><td rowspan=1 colspan=1>subject</td><td rowspan=1 colspan=1>present</td><td rowspan=1 colspan=1>recent (yesterday)</td><td rowspan=1 colspan=1>remote (week ago)</td></tr><tr><td rowspan=1 colspan=1>1(1sg)</td><td rowspan=1 colspan=1>-é</td><td rowspan=1 colspan=1>-é</td><td rowspan=1 colspan=1>-a</td></tr><tr><td rowspan=1 colspan=1>we (1pl)</td><td rowspan=1 colspan=1>-e</td><td rowspan=1 colspan=1>-e</td><td rowspan=1 colspan=1>-ake</td></tr><tr><td rowspan=1 colspan=1>2pl/3pl</td><td rowspan=1 colspan=1>-rth</td><td rowspan=1 colspan=1>-th</td><td rowspan=1 colspan=1>-ath</td></tr></table>

- 16 (2pl→me, hold, perfective recent): zwe + faf + -th → - 17 (1sg→you-al , cal , perfective remote): the + äs + -a → thäs+a → - 18 (2pl→me, hold, durative present): wo + fath + w + -rth → - 19 (3pl→us, meet, perfective remote): nzn + waf + -ath →

(b) Claude 4.8 Opus explanation.

18. Youpl are holding me now. zena wofathwrthFigure 3: Schematic explanations provided in oficial IOL solutions for P5 in comparison to the top scoring model’s solution explanation that we elicit through prompting.

![](images/c973e0a7216bc7537edf20e8e5b5e1352e332967d9483a34be5f80e8d8b26ae6.jpg)  
Figure 4: Diference between automatic and jury scores per problem for the five systems selected for human evaluation. Cla = Claude Opus 4.8, Gem = Gemini 3.6 Flash, BoN = BoN Open (oracle), #1 = arvindcr4, #11 = cabanosss.

a separate explanation for each task within a problem, the models sometimes gave diferent, and occasionally contradictory, analyses of the same phenomenon. Strikingly, in P2–Yélî Dnye, the Gemini model proposed an incorrect theory in which nouns for objects were formed by reduplication from color terms for earlier assignments. However, for assignment 4 (which was unusual in that it explicitly asked for an explanation of how color terms are derived), the model fully switched its explanation to the canonical one (that some color terms are formed by reduplication while others are formed by comparison). This suggests that the model was in principle able to recover the right set of rules, but that it did not get to this explanations in the other assignments.

2. Verbosity: explanation traces and examples. The models were generally more verbose than human contestants. They often gave entire explanation traces, something that contestants are explicitly instructed not to do, and which they can learn not to do from seeing canonical solutions. The models also often tried to explain a rule by giving an example sentence instead of the specific rule application (e.g. in P3–Iquito, giving an example verb in the past tense and an example verb in the future tense, rather than stating the sufixes used to mark each tense), which is likewise not rewarded with any marks.

3. Hallucinations. Lower-performing models sometimes hallucinated data (e.g., new terms such as completely black in P2–Yélî Dnye, based on black and completely white which appear in the problem; or sounds that do not appear in the problem, such as [E] or syllable-initial consonant clusters in P1–Yup’ik), outputted impossible answers to assignments (e.g., wrong number of syllables due to miscounting in P1), or provided non-sensical explanations and examples (e.g., arvindcr4 writing in P5-Komnzo for example, “thafa” (hold) with “th” sufix indicates the subject, and “wath” indicates the object).

4. Underspecified rules. Lower-performing models tended to hedge in their explanations using words such as typically or often (e.g., arvindcr4 in P3–Iquito: The verb typically comes after the subject, which is always true and hence typically is not needed). At IOL, such qualifications matter: a rule is expected to state precisely what the data support.

5. Undergeneralization. The frontier models tended to miss broad patterns that human contestants sometimes correctly delineated. For instance in P5–Komnzo, no model parsed out morpheme -a- appearing in sentences in non-recent past. This suggests AI models could have dificulty generalizing from sparse data during linguistic reasoning.

6. Overfitting or convoluted explanations. Explanations were often more complex than those of human contestants (e.g., in P5–Komnzo, Gemini incorrectly parsed the -s- stem from nznästh and zwäsath and added a phonological modification (The punctual stem for “call” -näs- reduces to -äs- when preceded by the labial glide /w/ in the 1s punctual prefix zw- (e.g., zw-näs-ath → zwäsath).)

7. Technical terminology without a complete analysis. The frontier models (Claude and Gemini) sometimes used technical linguistic terminology (e.g., iambic feet in P1–Yup’ik, durative aspect in P5–Komnzo, or a specific kinship-system annotation (MB for mother’s brother and FZ for father’s sister in P4–Sakurabiat). However, these terms were not always used consistently or completely. (e.g., writing MB’s son instead of the usual MBS in kinship terminology).

Overall, we found that models do not fail to identify linguistic structure altogether—the strongest models clearly are able to achieve medal-level scores. However, there are diferences in how that structure is used across a problem: human solutions tend to build a compact analysis that explains the data as a whole, while model solutions more often reason from one assignment to the next. This diference is easy to miss when evaluation considers only the final answer, but becomes much more visible when the theory explanations themselves are graded.

On top of the diferences, the jury also highlighted similarities to human contestants’ solutions, such as the order in which phenomena are described in the explanations (e.g., tending to start with word order in P5–Komnzo), or the rank-order of dificulty of the diferent phenomena. Indeed, in line with the jurors’ impressions, the dificulty level of all theory and explanation criteria across the 5 grading schemes (measured as the % models or contestants that met the criterion, i.e., solved that subunit of the problem correctly) correlated strongly between humans and models (Spearman ρ = 0.57; Figure 5). This link was observed despite the fact that some problems seemed overall easier for models (P5–Komnzo), whereas others were easier for human contestants (P3–Iquito).

![](images/69eb452dce6237d7b7055fc52b6a3ba13520c9174caf1cd79aaa4ed08d16a0e5.jpg)  
Figure 5: Comparison of the dificulty of criteria in the oficial IOL grading schemes between human contestants and models. Dificulty is measured as the percentage of solutions that meet the criterion and receive the respective mark, split into human contestants and AI models.

This finding suggests that linguistic reasoning comes in diferent levels of dificulty that is to some extend intrinsic, irrespective of whether the solver is human or machine.

## 5.3 Analysis: Knowledge Probing

A key question is which knowledge successful models rely on (not only on reasoning capabilities)— one may speculate whether they post-trained on linguistic tasks, or have included grammar books in their pretraining more than others. We cannot perform training data analysis for most benchmarked models due to missing data releases, but we can estimate proxies of their leverage of prior knowledge by relating the following estimates to problem solving success:

1. Measure availability of online resources in or about the problem languages (web crawl statistics, Wikipedia article length, number of Glottolog (Hammarström et al., 2026) listed resources). This estimates how much the models could have had prior exposure with the language or words in that language.

2. Probe models about their explicit knowledge about the languages by turning the contextual language facts of the problem context in a multiple-choice task, and the lexical items from the problem language into a language identification task (also multiple-choice). This gives us proxies about (a) prior language knowledge and (b) lexical familiarity or ability to deduce language identity from single words.

3. Remove the language name and background information from the problem context and measure how scores change across problem languages. This tell us which models are able to leverage this information to bias their predictions, and how much they rely on it for solving the problems.

<table><tr><td>Model</td><td>Language Context</td><td>Lexicon</td><td>Mean</td></tr><tr><td>GPT 5.6 Sol</td><td>90.9</td><td>75.0</td><td>83.0</td></tr><tr><td>Claude Opus 4.8</td><td>84.8</td><td>52.5</td><td>68.7</td></tr><tr><td>Kimi K2.6</td><td>93.9</td><td>72.5</td><td>83.2</td></tr><tr><td>Qwen 3.7 Max</td><td>90.9</td><td>72.5</td><td>81.7</td></tr><tr><td>DeepSeek V4 Pro</td><td>93.9</td><td>67.5</td><td>80.7</td></tr><tr><td>Inkling</td><td>84.8</td><td>60.0</td><td>72.4</td></tr><tr><td>Nemotron 3 Ultra</td><td>90.9</td><td>52.5</td><td>71.7</td></tr><tr><td>MiniMax M3</td><td>75.8</td><td>42.5</td><td>59.1</td></tr><tr><td>Command A+</td><td>72.7</td><td>37.5</td><td>55.1</td></tr></table>

(a) Accuracy of frontier models on 4-way MCQ knowledge probe tasks derived from IOL 2026 problems. Language Context: questions targeting language context, Lexicon: identifying real forms from the problem language of a set of four options.

<table><tr><td>Pairing</td><td>Pearson</td><td>Spearman</td></tr><tr><td>Language Context vs GM</td><td>0.273</td><td>0.195</td></tr><tr><td>Lexicon vs GM</td><td>0.339</td><td>0.355</td></tr><tr><td>Mean vs GM</td><td>0.380</td><td>0.430</td></tr></table>

(b) Correlating knowledge probe accuracy with overall IOL 2026 performance (automatic score, GM).

Table 6: Results of knowledge probing tasks for a subset of benchmarked models.

Presence on the web All problem languages would fall under the category of (extremely) “lowresourced” languages in NLP (Joshi et al., 2020) beyond the scope of mainstream AI technology development (Blasi et al., 2022). However, Yup’ik (esu) and Yélî Dnye (yle) documents can be found in FineWeb-2 (Penedo et al., 2025) and GlotCC-V1 (Kargaran et al., 2025) web crawls,<sup>6</sup> but with less than 1MB each. Yup’ik and Komnzo are each part of publicly available (small) speech or text corpora (Ardila et al., 2020; Döhler, 2021; He et al., 2024). This means that there is a slight chance that models have encountered words in these languages before. Short Wikipedia articles in the English Wikipedia exist for all five languages, with the ones about Yup’ik and Yélî Dnye being the most detailed, containing grammar and phonology descriptions (Iquito and Komnzo: only phonology; Sakurabiat: shorter phonology, grammar)—which means that models might recall some linguistic knowledge about these languages, if they have suficiently memorized this data. Iquito and Yup’ik have the largest number of resources listed in Glottolog (around 50– 60)—resources that can be accessed if publicly available, but are not as likely to be included in pretraining data. To summarize, Yup’ik is the most present in public web sources, and Iquito (P3) is the most documentationally rich but has a small general-web footprint. Hence, LLMs are likely to be most familiar with contents in Yup’ik (P1) and Yélî Dnye (P2) (if at all) and their grammars and phonology. According to jury scores (Table 5), Yup’ik and Sakurabiat were the best solved problems, but models struggled the most with Yélî Dnye, which indicates that web presence is not a reliable predictor of dificulty for this problem set.

Direct knowledge probes When we directly a probe a subset of 9 of the large benchmarked models for either contextual or lexicon knowledge about each problem language (details in Section K), we find that they all score fairly highly (random guess baseline would score at 25%), as shown in Table 6a. All models score consistently higher on predicting contextual information about the languages (72–94%) than identifying lexical items (37–75%), indicating that they likely know more about the language than about words in the language. Comparing their mean accuracy in probing tasks with their ranking according to automatic (GM) scores on IOL 2026, we find a positive correlation between the two (Pearson: 0.38, Spearman: 0.43). Of the two sub-tasks, correlation is higher for the lexicon probe, indicating this might be a more reliable predictor for task performance overall. This finding is strictly correlational and does not allow causal conclusions—it might be an efect of scale, generally stronger reasoning or better memorization of e.g. Wikipedia, rather than indicating any linguistics-specific training.

Linguistic reasoning without language meta-information Each IOL task comes with a rich context and also background information about the language, e.g. language family, script, geographic region where it is spoken. Models (and humans) might leverage this knowledge to transfer from related languages they might be more familiar with. When we drop all mentions of the language and remove this context, we observe on average a loss in performance (Table 7), but it is not significant according to 95% confidence intervals with three repeated runs. Inspecting the efect per problem language (Table 8), however, reveals that this removal of context is significantly hurting average model performance for Yélî Dnye (P2), with 6/7 models dropping an average of 5.8 percentage points. For this problem set, which is also the most dificult for all models, the presented language context is consistently the most helpful, because the language name itself contains a cue,<sup>7</sup> but overall the efect is small and noisy.

<table><tr><td>Model</td><td>chrF</td><td>EM</td><td>Geo</td><td> $\mathrm { c h r F ^ { a n o n } }$ </td><td> $\mathrm { E M ^ { a n o n } }$ </td><td> $\mathrm { G e o ^ { a n o n } }$ </td><td> $\Delta \mathrm { G e o }$ </td><td>95% CI</td></tr><tr><td>GPT 5.6 Sol</td><td>65.6</td><td>47.6</td><td>55.8</td><td>63.6</td><td>39.7</td><td>50.2</td><td>-5.6</td><td> $[ - 1 6 . 2 , + 3 . 9 ]$ </td></tr><tr><td>Claude Opus 4.8</td><td>69.5</td><td>52.0</td><td>60.1</td><td>64.7</td><td>50.5</td><td>57.2</td><td>-3.0</td><td> $[ - 1 5 . 2 , + 1 0 . 0 ]$ </td></tr><tr><td>Command A+</td><td>29.6</td><td>6.6</td><td>14.0</td><td>22.9</td><td>3.4</td><td>8.0</td><td>-5.9</td><td> $[ - 1 8 . 0 , + 3 . 3 ]$ </td></tr><tr><td>Qwen 3.7 Max</td><td>52.8</td><td>27.4</td><td>38.0</td><td>42.1</td><td>16.5</td><td>26.3</td><td>-11.7</td><td>[-23.3, +0.6]</td></tr><tr><td>Kimi K2.6</td><td>51.8</td><td>31.2</td><td>40.2</td><td>51.3</td><td>30.4</td><td>39.5</td><td>-0.7</td><td>[-16.9, +12.9]</td></tr><tr><td>Nemotron 3 Ultra</td><td>36.7</td><td>18.0</td><td>25.4</td><td>36.8</td><td>19.7</td><td>26.7</td><td>+1.2</td><td>[-19.8, +18.2]</td></tr><tr><td>DeepSeek V4 Pro</td><td>54.2</td><td>28.7</td><td>39.4</td><td>57.9</td><td>38.4</td><td>46.8</td><td>+7.4</td><td> $[ - 1 1 . 0 , + 2 8 . 2 ]$ </td></tr></table>

Table 7: Automatic score change when language mentions and context are dropped from the task descriptions (without custom answer parsing that would raise all scores). Scores are aggregated across three repeated runs with diferen random seeds.
<table><tr><td>Language</td><td>Geo</td><td> $\overline { { \mathrm { G e o } ^ { \mathrm { a n o n } } } }$ </td><td> $\overline { { \Delta \mathrm { G e o } } }$ </td><td>95% CI</td><td># models ↓</td></tr><tr><td>Yél Dnye</td><td>21.0</td><td>15.2</td><td>-5.8</td><td>[-9.9, -1.9]</td><td>6/7</td></tr><tr><td>Komnzo</td><td>48.8</td><td>45.0</td><td>-3.8</td><td>[-10.9, +4.0]</td><td>5/7</td></tr><tr><td>Sakurabiat</td><td>42.4</td><td>41.4</td><td>-1.0</td><td>[-7.0, +4.0]</td><td>3/7</td></tr><tr><td>Central Alaskan Yup&#x27;ik</td><td>51.6</td><td>51.9</td><td>+0.3</td><td>[-9.7, +9.8]</td><td>3/7</td></tr><tr><td>Iquito</td><td>40.8</td><td>41.7</td><td>+0.9</td><td>[-3.7, +5.6]</td><td>3/7</td></tr></table>

Table 8: Automatic score change on each problem for an average across three runs from models in Table 7.

Qualitative inspection of reasoning traces For models that disclose their reasoning traces (not proprietary models), we can also inspect whether they refer to typological or language family information about the problem language (given or not). A qualitative inspection reveals that language meta-information is indeed frequently present in the reasoning traces across models. This pattern is strongest for Sakurabiat problems (P4), where models refer to Tupian kinship structures and respective relational nouns, but as shown above there is no consistent efects on task scores when this association is removed. For Yélî Dnye (P2), however, models refer to priors from Papuan/O ceanic/Austronesian languages, exploring answer paths beyond the explicitly stated information (the problem states that it is a language isolate). This analysis shows that reasoning models will try to leverage prior knowledge in their reasoning, but we cannot conclude that it will help them to solve the task.

Overall, this analysis confirms that even when models have some knowledge about the problem languages or are familiar with words of these languages, it does not determine their ability to solve the respective linguistic problems. The contextual meta-information given with each problem might bias their priors, but the dificulty, as per design of the IOL, remains in deducing patterns from the presented linguistic context, and generalizing them to new forms.

## 6 Conclusion

We introduced the IOL-AI challenge, the first evaluation of language models on genuinely unseen linguistic-olympiad problems, graded by the expert jury that grades the human contestants. Because the problems were unreleased, no model could have seen them or their solutions during training, ruling out benchmark leakage by construction. The problem languages themselves have some public documentation, but our knowledge probing shows that what models know about them does not determine success. Solving these problems requires inferring the linguistic system from the given data. This makes linguistic reasoning a true test of model reasoning capabilities.

The open challenge showed that inference-time techniques have a large impact under tight resource constraints. The winning submission more than doubled our strongest baseline with the same 14B model, with the margin coming from decoding and answer handling rather than model capacity. Prior work reported that frontier models uniformly underperform on linguistic puzzles. This is no longer the case: for the first time, models reached medal level. Claude Opus 4.8 scored above this year’s gold-medal cutof, equivalent to fourth place among 255 contestants, and Gemini 3.6 Flash reached silver. But this progress is confined to proprietary systems whose training and reasoning remain inaccessible, so it does little to advance open research on reasoning. No resource-constrained submission cleared the honorable-mention threshold, both scoring in the bottom 5% of contestants, and even an oracle ensemble of mid-size open models lands well short of the frontier. Closing this gap with open, eficient systems is the central problem this challenge poses.

Expert grading also changes what the scores mean: it is under the jury’s rubrics, not string match ing, that frontier models reached medal level. Comparing the two scoring methods, automatic metrics rank systems exactly as the jury does $( \rho = 1 . 0 0 , r = 0 . 9 9 )$ , but compress the scale, inflating weak systems by 13 points and under-crediting strong ones, whose correct analyses in prose earn nothing automatically. We release jury evaluations, model outputs, and scores<sup>8</sup> to support progress on both linguistic reasoning systems and their measurement.

## Limitations

Format Requirements We noticed that most models, including frontier models, struggle with the required answer format, outputting a list of individual solutions. As a consequence, many submissions to the IOL-AI challenge focused on working around format artifacts, and we had to craft post-hoc format fixing rules for frontier model outputs to make sure their output format does not perturb automatic score rankings. Future benchmarking for linguistic reasoning should re-evaluate which prompting style is most favorable for LLMs across model scales.

Test Set Size & Diversity The evaluation on the IOL 2026 Individual Contest problems focuses on depth rather than breadth. As a consequence, the total number of test samples is small for testing ML models and the statistical power low. Since these tasks can be assumed to be out-of-domain for most models, the risk for high variance across multiple outputs under stochastic decoding is high, especially with long reasoning traces. Small diferences in automatic scoring should therefore not be over-interpreted. The score diferences between the AI solutions participating in human evaluation completely align with the jury ranking, so we can be confident that diferences of that magnitude (min 5 points in GM) tend to be meaningful and human noticeable.

Oracle Ensemble The BoN Open ensemble uses oracle selection against the reference answers. It therefore reports an upper bound on what this pool of mid-size open models could achieve under perfect routing, rather than the performance of a deployable system. A further caveat is that a single model (Gemma4-31B-It) supplies the majority of the selected answers, so the ensemble’s advantage is only partly attributable to model diversity.

Unequal Generation Budgets Within the mid-size tier, DeepSeek-R1-32B was run with a 32,768- token generation budget against 64,000 for the other three models (Table 11), so its scores are not strictly comparable within that tier and likely understate it.

Time Limitation The open-science competition ran for a short time window of a month, and a longer duration might have gathered more submissions, or have allowed participants to improve their submissions further.

Independence of Submissions We required all submissions to be public repositories on HuggingFace, so it was possible for participants to search for each other’s submission and potentially copy it. While teaming up after individual submissions was explicitly forbidden, this remained a loophole. As a consequence, we have many similar submissions, and top-scoring submissions might not credit the work of other participants that laid the foundation. Nevertheless, building on top of other peoples’ work and improving upon it is also in the spirit of advancing the state of the art.

## Acknowledgments

We thank the participants of the challenge for their contributions to advancing the understanding of linguistic reasoning challenges for LLMs. This work would not have been possible without the generous support of additional members of the IOL 2026 Jury – Mihai-Alexandru Bratu, Aida Davletova, Jan Petr, Eimear McKnight, Stanislav Gurevich – who evaluated the submissions. We especially thank Mihai, Aida and Jan for providing written impressions of the solutions. In addition, the dataset for this challenge would have not existed without the work of the greater IOL Problem Committee and the authors of this year’s problems—DMM, Eimear McKnight, Lai Otsuka, and Vesko Milev. We also thank Cohere as sponsors of the compute resources for the challenge.

## References

Rosana Ardila, Megan Branson, Kelly Davis, Michael Kohler, Josh Meyer, Michael Henretty, Reuben Morais, Lindsay Saunders, Francis Tyers, and Gregor Weber. Common voice: A massively-multilingual speech corpus. In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), Proceedings of the Twelfth Language Resources and Evaluation Conference, pp. 4218–4222, Marseille, France, May 2020. European Language Resources Association. ISBN 979-10-95546-34-4. URL https://aclanthology.org/2020.lrec-1.520/.

Seth Aycock, David Stap, Di Wu, Christof Monz, and Khalil Sima’an. Can LLMs really learn to translate a low-resource language from one grammar book? In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=aM BSY2ebPw.

Andrew Michael Bean, Simeon Hellsten, Harry Mayne, Jabez Magomere, Ethan A Chi, Ryan Andrew Chi, Scott A. Hale, and Hannah Rose Kirk. LINGOLY: A benchmark of olympiad-level linguistic reasoning puzzles in low resource and extinct languages. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview.net/forum?id=cLga8GStdk.

Vladimir I. Belikov, Elena V. Muravenko, and Mixail E. Alexeev (eds.). Zadači lingvističeskix olimpiad. 1965–1975. Izdatel’stvo MCNMO, Moscow, 2006.

Damian Blasi, Antonios Anastasopoulos, and Graham Neubig. Systematic inequalities in language technology performance across the world’s languages. In Smaranda Muresan, Preslav Nakov, and Aline Villavicencio (eds.), Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5486–5505, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022.acl-long.376. URL https: //aclanthology.org/2022.acl-long.376/.

Bozhidar Bozhanov and Ivan Derzhanski. Rosetta stone linguistic problems. In Ivan Derzhanski and Dragomir Radev (eds.), Proceedings of the Fourth Workshop on Teaching Natural Language Processing. 51th Annual Meeting of the Association for Computational Linguistics, pp. 1–8. Association for Computational Linguistics, 2013. URL https://aclanthology.org/W13-3401.pdf.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling, 2024. URL https://arxiv.org/abs/2407.21787.

Center for AI Safety, Scale AI, and HLE Contributors Consortium. A benchmark of expert-level academic questions to assess AI capabilities. Nature, 649:1139–1146, 2026. doi: 10.1038/s41586 -025-09962-4. URL https://arxiv.org/abs/2501.14249.

Feng Chen, Allan Raventos, Nan Cheng, Surya Ganguli, and Shaul Druckmann. Rethinking finetuning when scaling test-time compute: Limiting confidence improves mathematical reasoning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum?id=jvVQeSMeGM.

Simin Chen, Pranav Pusarla, and Baishakhi Ray. Dycodeeval: Dynamic benchmarking of reasoning capabilities in code large language models under data contamination. In Forty-second International Conference on Machine Learning, 2025b. URL https://openreview.net/forum?id=3B ZyQqbytZ.

Francois Chollet, Mike Knoop, Gregory Kamradt, Bryan Landers, and Henry Pinkard. Arc-agi-2: A new challenge for frontier ai reasoning systems, 2026. URL https://arxiv.org/abs/2505.1 1831.

DeepSeek-AI. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025. URL https://arxiv.org/abs/2501.12948.

Ivan Derzhanski. Multilingual editing of linguistic problems. In Ivan Derzhanski and Dragomir Radev (eds.), Proceedings of the Fourth Workshop on Teaching Natural Language Processing. 51th Annual Meeting of the Association for Computational Linguistics, pp. 27–34. Association for Computational Linguistics, 2013.

Ivan Derzhanski and Thomas Payne. The Linguistic Olympiads: Academic competitions in linguistics for secondary school students. In Linguistics at School: Language Awareness in Primary and Secondary Education, pp. 213–226. Cambridge University Press, 2010.

Ivan Derzhanski and Aleksandar Velinov. Lingvističen kaleĭdoskop. Prosveta, 2012.

Ivan Derzhanski and Aleksandar Velinov. Lingvistična mozaĭka: Dopŭlneno izdanie. Prosveta, 2013.

Ivan Derzhanski and Milena Veneva. Linguistic problems on number names. In Proceedings of the Third International Conference on Computational Linguistics in Bulgaria (CLIB 2018), pp. 169–176, 2018. URL https://aclanthology.org/2018.clib-1.21/.

Christian Döhler. Komnzo text corpus, April 2021. URL https://doi.org/10.5281/zenodo.469 5271.

Jamie Garnham and Ehsan Shareghi. Could language models win the international linguistics olympiad? In 30th Conference on Computational Natural Language Learning, 2026. URL https://openreview.net/forum?id=dioc2KgOKg.

Jonas Geiping, Sean Michael McLeish, Neel Jain, John Kirchenbauer, Siddharth Singh, Brian R. Bartoldson, Bhavya Kailkhura, Abhinav Bhatele, and Tom Goldstein. Scaling up test-time compute with latent reasoning: A recurrent depth approach. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id= S3GhJooWIC.

Gemma Team. Gemma 4 technical report, 2026. URL https://arxiv.org/abs/2607.02770.

Henry Allan Gleason. Workbook in descriptive linguistics. Holt, Rinehart and Winston, New York, 1955.

Satyam Goyal and Soham Dan. IOLBENCH: Benchmarking LLMs on linguistic reasoning, 2025. URL https://arxiv.org/abs/2501.04249.

Harald Hammarström, Robert Forkel, Martin Haspelmath, and Sebastian Bank. Glottolog 5.3. In Leipzig: Max Planck Institute for Evolutionary Anthropology, 2026. URL https://doi.org/10 .5281/zenodo.18840935.

Taiqi He, Kwanghee Choi, Lindia Tjuatja, Nathaniel Robinson, Jiatong Shi, Shinji Watanabe, Graham Neubig, David Mortensen, and Lori Levin. Wav2Gloss: Generating interlinear glossed text from speech. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 568–582, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.34. URL https://aclanthology.org/2024.acl-long.34/.

Yifeng He, Luning Yang, Christopher Castro Gaw Gonzalo, and Hao Chen. Evaluating program semantics reasoning with type inference in system \$f\$. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https: //openreview.net/forum?id=IA9RmaP0aw.

Ruixin Hong, Hongming Zhang, Xinyu Pang, Dong Yu, and Changshui Zhang. A closer look at the self-verification abilities of large language models in logical reasoning. In Kevin Duh, Helena Gomez, and Steven Bethard (eds.), Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 900–925, Mexico City, Mexico, June 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.naacl-long.52. URL https: //aclanthology.org/2024.naacl-long.52/.

Yichen Huang and Lin F. Yang. Winning gold at IMO 2025 with a model-agnostic verification-andrefinement pipeline, 2025. URL https://arxiv.org/abs/2507.15855.

Boris Iomdin, Alexander Piperski, and Anton Somin. Linguistic problems based on text corpora. In Ivan Derzhanski and Dragomir Radev (eds.), Proceedings of the Fourth Workshop on Teaching Natural Language Processing. 51th Annual Meeting of the Association for Computational Linguistics, pp. 9–17. Association for Computational Linguistics, 2013.

Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. SWE-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/f orum?id=VTF8yNQM66.

Pratik Joshi, Sebastin Santy, Amar Budhiraja, Kalika Bali, and Monojit Choudhury. The state and fate of linguistic diversity and inclusion in the NLP world. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 6282–6293, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.560. URL https://aclanthology.org/2020.ac l-main.560/.

Amir Hossein Kargaran, François Yvon, and Hinrich Schütze. Glotcc: An open broad-coverage commoncrawl corpus and pipeline for minority languages, 2025. URL https://arxiv.org/ab s/2410.23825.

Jude Khouja, Lingyi Yang, Karolina Korgul, Simeon Hellsten, Vlad A. Neacșu, Harry Mayne, Ryan Othniel Kearns, Andrew M. Bean, and Adam Mahdi. LINGOLY-TOO: Disentangling reasoning from knowledge with templatised orthographic obfuscation. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum ?id=CQIkN2uuBr.

Tom Kocmi, Sweta Agrawal, Ekaterina Artemova, Eleftherios Avramidis, Eleftheria Briakou, Pinzhen Chen, Marzieh Fadaee, Markus Freitag, Roman Grundkiewicz, Yupeng Hou, Philipp Koehn, Julia Kreutzer, Saab Mansour, Stefano Perrella, Lorenzo Proietti, Parker Riley, Eduardo Sánchez, Patricia Schmidtova, Mariya Shmatova, and Vilém Zouhar. Findings of the WMT25 multilingual instruction shared task: Persistent hurdles in reasoning, generation, and evaluation. In Barry Haddow, Tom Kocmi, Philipp Koehn, and Christof Monz (eds.), Proceedings of the Tenth Conference on Machine Translation, pp. 414–435, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-341-8. doi: 10.18653/v1/2025.wmt-1.23. URL https://aclanthology.org/2025.wmt-1.23/.

Da-Chen Lian, Ri-Sheng Huang, Pin-Er Chen, Chunki Lim, You-Kuan Lin, Guan-Yu Tseng, Zhen-Yu Lin, Pin-Cheng Chen, and Shu-Kai Hsieh. LOBSTER: Linguistics olympiad benchmark for structured evaluation on reasoning. In Kai-Wei Chang, Ke-Han Lu, Chih-Kai Yang, Zhi-Rui Tam, Wen-Yu Chang, and Chung-Che Wang (eds.), Proceedings of the 37th Conference on Computational Linguistics and Speech Processing (ROCLING 2025), pp. 193–229, National Taiwan University, Taipei City, Taiwan, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-379-1. URL https://aclanthology.org/2025.rocling-main.23/.

Patrick Littell, Lori Levin, Jason Eisner, and Dragomir R. Radev. Introducing computational concepts in a Linguistics Olympiad. In Ivan Derzhanski and Dragomir Radev (eds.), Proceedings of the Fourth Workshop on Teaching Natural Language Processing. 51th Annual Meeting of the Association for Computational Linguistics, pp. 18–26. Association for Computational Linguistics, 2013. URL https://aclanthology.org/W13-3403/.

Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, and Dong Yu. Trust, but verify: A self-verification approach to reinforcement learning with verifiable rewards. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=gA3fFAEXNT.

Sara Vera Marjanovic, Arkil Patel, Vaibhav Adlakha, Milad Aghajohari, Parishad BehnamGhader, Mehar Bhatia, Aditi Khandelwal, Austin Kraft, Benno Krojer, Xing Han Lù, Nicholas Meade, Dongchan Shin, Amirhossein Kazemnejad, Gaurav Kamath, Marius Mosbach, Karolina Stanczak, and Siva Reddy. Deepseek-r1 thoughtology: Let’s think about LLM reasoning. Transactions on Machine Learning Research, 2026. ISSN 2835-8856. URL https://openreview.net/forum?i d=BZwKsiRnJI.

Eduardo Cardoso Martins. Olimpíadas de linguística: mosaico de uma prática social baseada em problemas. PhD thesis, Universidade de Brasília, Brasília, 2022.

Mike A Merrill, Alexander Glenn Shaw, Nicholas Carlini, Boxuan Li, Harsh Raj, Ivan Bercovich, Lin Shi, Jeong Yeon Shin, Thomas Walshe, E. Kelly Buchanan, Junhong Shen, Guanghao Ye, Haowei Lin, Jason Poulos, Maoyu Wang, Marianna Nezhurina, Di Lu, Orfeas Menis Mastromichalakis, Zhiwei Xu, Zizhao Chen, Yue Liu, Robert Zhang, Leon Liangyu Chen, Anurag Kashyap, Jan-Lucas Uslu, Jefrey Li, Jianbo Wu, Minghao Yan, Song Bian, Vedang Sharma, Ke Sun, Steven Dillmann, Akshay Anand, Andrew Lanpouthakoun, Bardia Koopah, Changran Hu, Etash Kumar Guha, Gabriel H. S. Dreiman, Jiacheng Zhu, Karl Krauth, Li Zhong, Niklas Muennighof, Robert Kwesi Amanfu, Shangyin Tan, Shreyas Pimpalgaonkar, Tushar Aggarwal, Xiangning Lin, Xin Lan, Xuandong Zhao, Yiqing Liang, Yuanli Wang, Zilong Wang, Changzhi Zhou, David Heineman, Hange Liu, Harsh Trivedi, John Yang, Junhong Lin, Manish Shetty, Michael Yang, Nabil Omi, Negin Raoof, Shanda Li, Terry Yue Zhuo, Wuwei Lin, Yiwei Dai, Yuxin Wang, Wenhao Chai, Shang Zhou, Dariush Wahdany, Ziyu She, Jiaming Hu, Zhikang Dong, Yuxuan Zhu, Sasha Cui, Ahson Saiyed, Arinbjörn Kolbeinsson, Christopher Michael Rytting, Ryan Marten, Yixin Wang, Jenia Jitsev, Alex Dimakis, Andy Konwinski, and Ludwig Schmidt. Terminal-bench: Benchmarking agents on hard, realistic tasks in command line interfaces. In The Fourteenth International Conference on Learning Representations, 2026. URL https://openreview.net/forum?id=a7Qa4CcHak.

Vlad A. Neacșu. Linguistics Olympiad: Training guide. Language Science Press, 2024. doi: 10.528 1/zenodo.10947862. URL https://zenodo.org/doi/10.5281/zenodo.10947862.

Andrey Nikulin. Las olimpiadas de lingüística, o cómo llevar la lingüística a la secundaria. In María Mare y Gonzalo Espinosa (ed.), Aportes Disciplinares II SAEL, pp. 105–117. Sociedad Argentina de Estudios Lingüísticos (SAEL), 2024.

Guilherme Penedo, Hynek Kydlíček, Vinko Sabolčec, Bettina Messmer, Negar Foroutan, Amir Hossein Kargaran, Colin Rafel, Martin Jaggi, Leandro Von Werra, and Thomas Wolf. Fineweb2: One pipeline to scale them all – adapting pre-training data processing to every language, 2025. URL https://arxiv.org/abs/2506.20920.

Maja Popović. chrF: character n-gram F-score for automatic MT evaluation. In Ondřej Bojar, Rajan Chatterjee, Christian Federmann, Barry Haddow, Chris Hokamp, Matthias Huck, Varvara Logacheva, and Pavel Pecina (eds.), Proceedings of the Tenth Workshop on Statistical Machine Translation, pp. 392–395, Lisbon, Portugal, September 2015. Association for Computational Linguistics. doi: 10.18653/v1/W15-3049. URL https://aclanthology.org/W15-3049/.

Shengxuan Qiu, Haochen Huang, Shuzhang Zhong, Pengfei Zuo, and Meng Li. HyPER: Bridging exploration and exploitation for scalable LLM reasoning with hypothesis path expansion and reduction. In Forty-third International Conference on Machine Learning, 2026. URL https: //openreview.net/forum?id=G29kBVeIZt.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.a i/blog?id=qwen3.5.

Dragomir R. Radev (ed.). Puzzles in logic, languages and computation: The green book. Springer, 2013a.

Dragomir R. Radev (ed.). Puzzles in logic, languages and computation: The red book. Springer, 2013b.

Raghav Ramji and Keshav Ramji. Inductive linguistic reasoning with large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Findings of the Association for Computational Linguistics: ACL 2025, pp. 22783–22810, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.1171. URL https://aclanthology.org/2025.findings-acl.1 171/.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling, 2024. URL https://openreview.net /forum?id=Ti67584b98.

Gözde Gül Şahin, Yova Kementchedjhieva, Phillip Rust, and Iryna Gurevych. PuzzLing Machines: A Challenge on Learning From Small Data. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), Proceedings ofthe 58th Annual Meeting ofthe Association for Computational Linguistics, pp. 1241–1254, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.115. URL https://aclanthology.org/2020.acl-main.115/.

Eduardo Sánchez, Belen Alastruey, Christophe Ropers, Pontus Stenetorp, Mikel Artetxe, and Marta R. Costa-jussà. Linguini: A benchmark for language-agnostic linguistic reasoning, 2025. URL https://openreview.net/forum?id=QiyQJqpcYe.

Maohao Shen, Guangtao Zeng, Zhenting Qi, Zhang-Wei Hong, Zhenfang Chen, Wei Lu, Gregory W. Wornell, Subhro Das, David Daniel Cox, and Chuang Gan. Satori: Reinforcement learning with chain-of-action-thought enhances LLM reasoning via autoregressive search. In Forty-second International Conference on Machine Learning, 2025. URL https://openreview.net/forum?i d=j4FXxMiDjL.

Suelen Érica Costa da Silva and Priscilla Tulipa da Costa. Linguística por problemas: explorando a linguagem de forma investigativa nas escolas. LED, Belo Horizonte, 2022.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more efective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum ?id=4FWAwZtd2n.

Yiming Wang, Pei Zhang, Jialong Tang, Hao-Ran Wei, Baosong Yang, Rui Wang, Chenshu Sun, Feitong Sun, Jiran Zhang, Junxuan Wu, Qiqian Cang, Yichang Zhang, Fei Huang, Junyang Lin, Fei Huang, and Jingren Zhou. Polymath: Evaluating mathematical reasoning in multilingual contexts. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview.net/forum?id=B1vCImy6yI.

Kaiyu Yang, Aidan M Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godil, Ryan Prenger, and Anima Anandkumar. Leandojo: Theorem proving with retrieval-augmented language models. In Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2023. URL https://openreview.net/forum?id=g7OX2sOJtn.

Andrej A. Zaliznjak. Lingvističeskie zadači. In Tatjana N. Mološnaja (ed.), Issledovanija po strukturnoj tipologii. Izdatel‘stvo AN SSSR, Moscow, 1963.

Fuxiang Zhang, Jiacheng Xu, Chaojie Wang, Ce Cui, Yang Liu, and Bo An. Incentivizing LLMs to self-verify their answers. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=MBDWO29Qq6.

Jie Zhang, Cezara Petrui, Kristina Nikolić, and Florian Tramèr. Realmath: A continuous benchmark for evaluating language models on research-level mathematics. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https://openreview.net/forum?id=RBssYVpQEr.

Yifan Zhang and Team Math-AI. American invitational mathematics examination (aime) 2025, 2025.

Hongpu Zhu, Yuqi Liang, Wenjing Xu, and Hongzhi Xu. Evaluating large language models for in-context learning of linguistic patterns in unseen low resource languages. In Hansi Hettiarachchi, Tharindu Ranasinghe, Paul Rayson, Ruslan Mitkov, Mohamed Gaber, Damith Premasiri, Fiona Anting Tan, and Lasitha Uyangodage (eds.), Proceedings of the First Workshop on Language Models for Low-Resource Languages, pp. 414–426, Abu Dhabi, United Arab Emirates, January 2025. Association for Computational Linguistics. URL https://aclanthology.org/2 025.loreslm-1.31/.

## A Accessibility and Engagement

To reduce barriers for participation, we hosted an online submission 101 session with a demo notebook that provides a sample submission based on a small open model. The compute provided in a T4 GPU session on Colab (session time limits are 12h for free usage) is comparable to that available in the competition, which allowed competitors to optimize model choices and decoding strategies for the time and compute limits. In addition, we provided participants with a list of resources about linguistic reasoning with AI, such as pointers to benchmarks and papers.

We organized this challenge intentionally outside of any academic venue to broaden participation and extend it beyond an academic audience or tie it to the requirement of paper submissions (classic shared task setup). Instead, we advertised it heavily on social media (X, LinkedIn, Bluesky) and in open science communities.We hosted a kick-of fireside chat with a discussion around the skills needed for linguistic reasoning both from a human perspective and the machine learning perspective. Additionally, we reached out to authors of related works and benchmarks to invite them to participate and/or share it with their network.

## B Technical Details

We substantially adapted the HF competition framework for the IOL-AI competition: First, submission evaluation is re-routed from per-submission Spaces to HF Jobs after undocumented platform rate limits. Second, every submission repository is automatically archived for reproducibility and post-hoc human evaluation. All modifications are publicly available in the competition Space repository.

## C Baselines

Qwen2.5-1.5B-Instruct is the example submission published on the competition platform and website to show participants what is expected: a minimal script with a minimal prompt, asking the model to answer each item on its own line. Qwen2.5-7B-Instruct scales this example to a larger model, quantized to 4-bit at load time with bitsandbytes under the same prompt, showing that quantization fits the compute constraints. Qwen2.5-14B-Instruct-AWQ illustrates the second quantization option, shipping pre-quantized AWQ weights, to fit the T4; here the prompt is not minimal but a reasoning prompt instructing the model on the output format expected for each task type. Finally, Tiny Aya Global, a 3.35B open multilingual model, uses an even more detailed prompt. Together, the baselines show that larger models reach higher scores, but that careful prompt design lets a smaller model outperform a larger one.

Section F.2 lists the full evaluation prompts.

## D Extended Model Benchmarking

We benchmark additional strong models that are outside the resource constraints of the competition. We prioritize models scoring highly on the Artificial Analysis Intelligence Index<sup>9</sup> and diversity across model providers.<sup>10</sup> The evaluation hyperparameters can be found in Table 11. We leverage Linguini (Sánchez et al., 2025) as a development set to tune task and explanation prompt and

postprocessing with an answer extraction script to further boost scores, and then use the IOL 2026 test set to select models for human evaluation. We optimize their explanation output format for readability of the jury, to make it closer to typical IOL solutions. Section F contain the prompts for benchmarking with and without explanations. Table 9 shows the results of the automatic evaluation for the extended model list, before and after custom answer parsing.
<table><tr><td rowspan="2">Model</td><td colspan="2">Original</td><td rowspan="2"></td><td colspan="5">+ custom answer parsing</td></tr><tr><td>chrF</td><td>EM GM</td><td>LFMR</td><td>chrF</td><td>EM</td><td>GM</td><td>LFMR</td></tr><tr><td>Proprietary Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>claude-opus-4-8 ★</td><td>70.6</td><td>52.8</td><td>61.0</td><td>92.9</td><td>82.9</td><td>68.1</td><td>75.1</td><td>100.0</td></tr><tr><td>gemini-3.6-flash ★</td><td>65.8</td><td>37.3</td><td>49.6</td><td>92.9</td><td>70.6</td><td>45.0</td><td>56.4</td><td>100.0</td></tr><tr><td>gpt5.6-sol</td><td>63.3</td><td>43.1</td><td>52.2</td><td>92.9</td><td>63.3</td><td>43.1</td><td>52.2</td><td>100.0</td></tr><tr><td>Open Large Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>kimi-k2-6</td><td>57.1</td><td>35.8</td><td>45.2</td><td>92.9</td><td>57.1</td><td>35.8</td><td>45.2</td><td>100.0</td></tr><tr><td>glm-5-2</td><td>43.5</td><td>21.4</td><td>30.5</td><td>71.4</td><td>58.0</td><td>35.0</td><td>45.0</td><td>92.9</td></tr><tr><td>deepseek-v4-pro</td><td>49.6</td><td>27.5</td><td>36.9</td><td>64.3</td><td>53.6</td><td>31.5</td><td>41.0</td><td>92.9</td></tr><tr><td>qwen-3-7-max</td><td>49.3</td><td>26.1</td><td>35.9</td><td>85.7</td><td>52.2</td><td>29.4</td><td>39.2</td><td>100.0</td></tr><tr><td>inkling</td><td>45.3</td><td>20.6</td><td>30.5</td><td>85.7</td><td>48.0</td><td>24.3</td><td>34.1</td><td>100.0</td></tr><tr><td>nemotron-3-ultra</td><td>41.5</td><td>20.1</td><td>28.9</td><td>64.3</td><td>41.5</td><td>20.1</td><td>28.9</td><td>64.3</td></tr><tr><td>command-a-plus</td><td>33.1</td><td>8.3</td><td>16.5</td><td>100.0</td><td>34.1</td><td>11.1</td><td>19.4</td><td>100.0</td></tr><tr><td>minimax-m3</td><td>4.8</td><td>1.3</td><td>2.5</td><td>14.3</td><td>14.4</td><td>7.9</td><td>10.6</td><td>78.6</td></tr><tr><td colspan="2">Open Mid-Size Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>gemma4-31b-it</td><td>37.2</td><td>14.6</td><td>23.3</td><td>85.7</td><td>37.0</td><td>14.6</td><td>23.2</td><td>92.9</td></tr><tr><td>qwen3.6-27b</td><td>23.8</td><td>13.4</td><td>17.8</td><td>57.1</td><td>25.2</td><td>14.9</td><td>19.4</td><td>64.3</td></tr><tr><td>deepseek-r1-32b</td><td>23.9</td><td>3.0</td><td>8.5</td><td>85.7</td><td>23.7</td><td>3.0</td><td>8.5</td><td>100.0</td></tr><tr><td>glm-4.7-flash</td><td>6.6</td><td>0.0</td><td>0.0</td><td>14.3</td><td>6.6</td><td>0.0</td><td>0.0</td><td>21.4</td></tr><tr><td>BoN Open (oracle) </td><td></td><td></td><td></td><td></td><td>48.5</td><td>23.2</td><td>33.5</td><td>100.0</td></tr></table>

Table 9: Evaluation scores for existing models on the IOL 2026 problem set, before and after answer parsing. The BoN Open (oracle) ensemble builds directly on custom answer parsed answers. EM = exact match, GM = geometric mean, LFMR = line format match rate. Model outputs that were chosen for the jury are marked with ⋆.
<table><tr><td></td><td>Language</td><td>chrF</td><td>EM</td><td>GM</td></tr><tr><td>P1</td><td>Yup&#x27;ik</td><td>49.1</td><td>7.1</td><td>18.7</td></tr><tr><td>P2</td><td>Yél Dnye</td><td>25.1</td><td>17.9</td><td>21.2</td></tr><tr><td>P3</td><td>Iquito</td><td>83.5</td><td>25.0</td><td>45.7</td></tr><tr><td>P4</td><td>Sakurabiat</td><td>32.8</td><td>27.8</td><td>30.2</td></tr><tr><td>P5</td><td>Komnzo</td><td>63.6</td><td>50.0</td><td>56.4</td></tr></table>

Table 10: Per-problem automatic scores for the BoN Open (oracle) ensemble. Scores are on a 0–100 scale per problem, so they are directly comparable to jury scores expressed as a percentage of the 20 points available per problem.

## E The Challenge of Format Following

Automatic scoring is strict about format: model answers are compared to gold by position, so a missing or shifted line can zero out a whole row even when later items are right.

Submissions. Format following was a major focus in the open challenge. 34 teams used regex answer extraction, and many added per-task format prompts or retry loops. Teams also dropped chain-of-thought on scoring runs. CoT was tried by 33 teams but hurt on average, and none of the top 6 kept it: reasoning text made extraction harder and burned wall-clock time. The podium used answers-only prompts and invested in parsers instead. Because format was already enforced in the sandbox, post-hoc extraction barely moves submission scores.

![](images/2ac043f9d89f1c30d84dc16a19c199b68214c327a89c5f402f750f3461db6652.jpg)  
(a) IOL 2026 frontier: format failures before extract.

![](images/2eed5940329e716095b3343527134366278f13640d5968c30a3fd8f29caabf58.jpg)  
(b) IOL 2026 frontier: format failures after extract.  
Figure 6: Format-failure prevalence for IOL 2026 frontier runs before and after light answer extraction.

Frontier models. Format errors are still common in raw frontier explain outputs. Light postprocessing removes most of them (Figure 6): empty answers cannot be fixed, but superficial nonconformities such as included item labels or JSON format as a one-liner are easily repaired. Table 9 reports scores before and after custom answer parsing. Parsing never lowers the score. At best it raises GM by about 14 points (e.g. GLM-5.2 from 30.5 to 45.0; Claude Opus from 61.0 to 75.1). Already clean runs stay unchanged.

## F Evaluation Prompts

## F.1 Benchmarking Task Prompts

For Linguini evaluations, we use the task prompt given in Figure 7.

You are solving a linguistic puzzle. All the information you need is contained   
in the context below - no prior knowledge of this language is required.   
Context:   
[CONTEXT]   
Question:   
[QUESTION]   
Answer with only the requested word or phrase. Do not include explanations in   
your final answer.  
Figure 7: Linguini evaluation prompt.

For out-of-the-box benchmarking of frontier/open models where explanations are required for human evaluation, we use the task prompt given in Figure 8. We verified on Linguini that the added explanation requirement does not have substantial impact on task performance of the top-scoring models participating in the human evaluation.

You are solving a linguistic puzzle. All the information you need is contained   
in the context below - no prior knowledge of this language is required.   
Context:   
[CONTEXT]   
Question:   
[QUESTION]   
Give your final answer , then an explanation of how you arrived at it.   
In the final answer , only include the requested word(s) or phrase(s), without   
markdown.   
In the explanation , summarize your linguistic analysis , and use tables or   
schemata (in markdown) to illustrate derived rules.   
Do not write a long chain -of-thought or lengthy narration.   
Use this format exactly:   
Answer:   
<your final answer only - the requested word(s) or phrase(s)>   
Explanation:   
<your explanation >  
Figure 8: Task prompt for Linguini benchmarking of out-of-the-box models, including explanations.

For IOL we slightly modified it because we also allowed for answers in JSON format, see Figure 9.

## F.2 Baseline System Prompts

The four organizer baselines (Table 2) use a chat format with a separate system prompt, unlike the single task prompts above. We list the system prompt and the user template for each; {context} and {query} are filled in per test item.

## G Grading of AI Solutions by the IOL Jury

In contrast to the in-person grading at IOL, due to time constraints at the event, grading of the AI solutions is done remotely in the week following IOL. 1–3 jury members from each grading group (that graded each of the 5 problems at IOL) volunteered to grade the AI solutions for each problem. Each jury member received pdf printouts of the AI solutions and explanations, anonymized with respect to model information.

Using the same grading schemes that were used at IOL (which were crafted by each grading group in the months leading to the event), each jury member then graded each AI paper individually, assigning points for each criterion specified in the grading scheme in a large spreadsheet. The spreadsheet is designed so that individual jurors can work separately in their own sheets, and afterwards a comparison sheet highlights any discrepancies between the grades of 2 or more jurors. At IOL, each contestant’s paper is graded by at least 2 jurors and discrepancies are resolved largely by in-person discussion. For the challenge, all problems were graded twice except problem 3, which was graded 3 times. For problem 4, the IOL jury chair served as a second grader despite not being part of that grading group. Discrepancies were resolved largely by virtual discussion (through text), and in a couple of cases by decision of the jury chair.

You are solving an International Linguistics Olympiad problem. All the   
information you need is contained in the context below - no prior   
knowledge of this language is required.   
Context:   
[CONTEXT]   
Question:   
[QUESTION]   
Answer every numbered (or lettered) item in the question , in order.   
Return either a JSON list of strings , or one answer per line (no numbering).   
Give your final answer , then an explanation of how you arrived at it.   
In the final answer , only include the requested word(s) or phrase(s), without   
markdown.   
In the explanation , summarize your linguistic analysis , and use tables or   
schemata (in markdown) to illustrate derived rules.   
Do not write a long chain -of-thought or lengthy narration.   
Use this format exactly:   
Answer:   
<your final answer only - the requested word(s) or phrase(s)>   
Explanation:   
<your explanation >  
Figure 9: Final IOL task prompt for benchmarking out-of-the-box models, including explanations.

SYSTEM: You solve International Linguistics Olympiad problems. Answer every   
numbered item. Put each answer on its own line , in order , with no   
numbering and no extra text.   
USER:   
{context}   
{query}  
Figure 10: Qwen2.5-1.5B-Instruct (starter) and Qwen2.5-7B-Instruct (NF4) baseline prompt.

Once all discrepancies were resolved, the scores for the 5 problems were summed up into one total score (out of 100) per submission.

## H Submissions

Submission Statistics Participants could form teams of up to three, but most competed individually: only four teams of two were formed. Submissions were back-loaded, with 43% arriving in the final 48 hours. 590 (80.7%) submissions ran successfully and received a score. The 141 failures fall into four families: library and environment mismatches between the submission repository and the evaluation image (54, 38.3%), errors in the participant inference script or at runtime (32, 22.7%), exceeding the wall-clock or memory limits (27, 19.1%), and ofline packaging mistakes such as code still attempting to reach the Hub at evaluation time (19, 13.5%); the remaining 9 (6.4%) fall outside these families. Common fixes included shipping model weights inside the repository, bundling updated wheels for ofline install, quantizing to fit the T4 (AWQ, NF4, or GGUF), and, against the timeout, using adaptive token budgets and writing submission.csv incrementally so a killed run still scores.

![](images/2fbf94ba30e1ed31a68e8be011e1cb4e4d720f87d571131a4285d52578974316.jpg)  
Figure 11: Qwen2.5-14B-Instruct-AWQ baseline prompt

Leading Pipelines The seven highest-ranked entries all reuse our minimal system prompt verbatim (Section F.2), and all lower the repetition penalty except Hul (6), which keeps the default. The entries that sample draw up to 8 retries (ranks 1, 4, 5) or up to 24 (ranks 2, 3) at temperature 0.5, for as long as time allows, and a retry only overrides the greedy answer when several samples agree on the same alternative; friedspaghetti (4) counts near-identical strings as the same answer, so formatting variants do not split the vote. Hul and jbuaba (7) never sample. Token budgets are either derived from the time remaining (ranks 1, 3, 4, 5, between 192 and 900 tokens per problem) or fixed at 512 tokens (ranks 2, 6, 7). Incremental writing means a complete submission file is on disk before the model even loads, and is rewritten after every pass or row, so a crash or timeout still leaves a valid file. Post-processing is where the entries difer most: most pad or trim the answers to the expected number of items, Hul normalises them by task type, and jbuaba answers matching problems by reading the model’s probability for each option letter and assigning options so that none repeats.

Submission Model Use Nine of the top ten entries run a 14B Qwen model: eight use Qwen2.5- 14B-Instruct-AWQ and one uses Qwen3-14B; the remaining entry, at rank 9, runs a 4.2B Qwen3.5 model in 8-bit GGUF. Across the 42 teams with a scored final submission, model choice is similarly concentrated: 29 teams selected Qwen2.5-14B-Instruct-AWQ for their final submissions.<sup>11</sup> This concentration has two causes: Qwen2.5-14B-Instruct-AWQ is the largest model that fits on the T4 within the time limit, and it was the model presented in our submission 101 session. Most teams therefore adopted it as a starting point and worked on the pipeline around it. The best submission more than doubles our strongest baseline with the same model (GM 9.40 19.79), so the margin at the top comes from decoding and output handling rather than model capacity.

SYSTEM:   
You solve International Linguistics Olympiad problems by reasoning from the   
data in CONTEXT you are given to solve the problems in QUERY.   
There are common TASK TYPES that we specify below , but you may meet a TASK   
TYPE you have never seen: read the instruction and the examples , and   
answer the QUERY in the same form they use.   
Common TASK TYPES and what to return:   
\`translation \`: return the translated form only , in the language the task asks   
for;   
\`fill\_blanks \`: return only the missing form for each indicated blank (beware:   
this could be many different things: a word , a part of a word or a   
phonetic transcription ---pay close attention to what part of the CONTEXT   
is missing in QUERY);   
\`match\_letters \`: return only the option letter (for example A, B, C);   
\`text\_to\_num \`: return the number in digits;   
\`num\_to\_text \`: return the number written out in words , in the language asked;   
any other type: return exactly what the instruction asks for, nothing else.   
As the first part of your answer , reason step by step about (1) the linguistic   
rules that can be deduced from the given examples in CONTEXT , and (2) how   
to apply them to the given problems in QUERY , and (3) in what format   
answers need to be returned (words , numbers , phonetic transcriptions , ...)   
Then write a draft of the final answer. Subsequently , compare it with the   
format requirements again , and verify it's compliant with the deduced   
rules , and it is complete , i.e. has an answer for each element in QUERY.   
If necessary , correct and refine. Finally , write a line that says exactly   
\`FINAL ANSWERS:\` and, below it, write the answers to the items requested   
in QUERY (not those in CONTEXT), one answer per line (separated by newline   
) in the order the items are asked for in the QUERY -- the bare answer   
only , no numbering , no quotes , no extra text , according to the given TASK   
TYPE.   
USER:   
CONTEXT:{context}   
TASK TYPE:\`{task\_type}\`   
QUERY:{query}  
Figure 12: Tiny Aya Global baseline prompt

## I Evaluation Hyper-parameters

Table 11 reports the evaluation configurations that were used for benchmarking frontier and open models. We set temperature and top p to whatever was the recommended configuration in model cards and the reasoning efort and token limit as high as possible.

## J Linguini Scores

Table 12 contains automatic evaluation results for the Linguini benchmark (Sánchez et al., 2025).

<table><tr><td>Model</td><td>Provider</td><td>temp</td><td>top_p</td><td>Reasoning Effort</td><td>Max Tokens</td></tr><tr><td>Claude Opus 4.8</td><td>Bedrock</td><td>1.0</td><td>1.0</td><td>high</td><td>128000</td></tr><tr><td>GPT-5.6</td><td>OpenAI</td><td>1.0</td><td>1</td><td>high</td><td>128000</td></tr><tr><td>Kimi K2.6</td><td>TogetherAI</td><td>1.0</td><td>0.95</td><td>N/A</td><td>256000</td></tr><tr><td>GLM-5.2</td><td>TogetherAI</td><td>1.0</td><td>0.95</td><td>max</td><td>128000</td></tr><tr><td>DeepSeek V4 Pro</td><td>TogetherAI</td><td>1.0</td><td>1</td><td>high</td><td>128000</td></tr><tr><td>Qwen3.7-Max</td><td>TogetherAI</td><td>0.6</td><td>0.95</td><td>N/A</td><td>65536</td></tr><tr><td>Inkling</td><td>TogetherAI</td><td>1.0</td><td>0.95</td><td>high</td><td>256000</td></tr><tr><td>Nemotron 3 Ultra</td><td>TogetherAI</td><td>0.3</td><td>0.95</td><td>high</td><td>65536</td></tr><tr><td>Command A Plus</td><td>Cohere</td><td>0.6</td><td>0.95</td><td>N/A</td><td>64000</td></tr><tr><td>MiniMax M3</td><td>TogetherAI</td><td>1.0</td><td>0.95</td><td>N/A</td><td>131072</td></tr><tr><td>Gemini 3.6 Flash</td><td>Google API</td><td>N/A</td><td>N/A</td><td>high</td><td>65536</td></tr><tr><td>Gemma 4 31B</td><td>Self-hosted</td><td>1.0</td><td>0.95</td><td>N/A</td><td>64000</td></tr><tr><td>Qwen3.6 27B</td><td>Self-hosted</td><td>0.6</td><td>0.95 (k=20)</td><td>N/A</td><td>64000</td></tr><tr><td>DeepSeek R1 32B</td><td>Self-hosted</td><td>0.6</td><td>0.95</td><td>N/A</td><td>32768</td></tr><tr><td>GLM 4.7 Flash</td><td>Self-hosted</td><td>0.6</td><td>0.95 (k=40)†</td><td>N/A</td><td>64000</td></tr></table>

Table 11: Hyper-parameters for evaluation. Self-hosted models were served with vLLM; “Max Tokens” is the gen eration budget, not the model’s context window. †GLM 4.7 Flash additionally requires repetition\_penalty=1.05; without it the model degenerates into repetition loops, which is reflected in its low format-compliance and near-zero scores.

## K Knowledge Probing Details

In order to turn the language context per problem into probing tasks, we analyze each problem with an LLM, and prompt it to extract the relevant information (either contextual information, or lexical items from the problem language). Table 13 shows examples. All tasks and answer options are verified by a human annotator who edited especially alternative response options to avoid making the right answer option too obvious. In total they span 33 questions about language context, and 40 about lexical items.

<table><tr><td colspan="4"></td><td colspan="4">+ custom answer parsing</td></tr><tr><td>Model</td><td>chrF EM</td><td>Original GM</td><td>LFMR</td><td>chrF</td><td>EM</td><td>GM</td><td>LFMR</td></tr><tr><td>Closed Models</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>GPT 5.6 Sol</td><td>72.0</td><td>26.9</td><td>44.0</td><td>90.6</td><td>78.5</td><td>33.8 51.5</td><td></td><td>98.8</td></tr><tr><td>Claude Opus 4.8 (Bedrock)</td><td>54.8</td><td>6.9</td><td>19.4</td><td>81.9</td><td>71.1</td><td>26.9</td><td>43.7</td><td>94.4</td></tr><tr><td>Gemini 3.6 Flash</td><td>63.7</td><td>10.6</td><td>26.0</td><td>99.4</td><td>78.0</td><td>33.1</td><td>50.8</td><td>98.8</td></tr><tr><td>Open Large Models</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>Qwen 3.7 Max DeepSeek V4 Pro</td><td>58.2</td><td>12.5</td><td>27.0</td><td>91.2</td><td>65.3</td><td>19.4</td><td>35.6</td><td>96.3</td></tr><tr><td>Kimi K2.6</td><td>48.9</td><td>10.0</td><td>22.1</td><td>66.9</td><td>55.7</td><td>19.4</td><td>32.9</td><td>74.4</td></tr><tr><td></td><td>40.1</td><td>11.2</td><td>21.2</td><td>51.2</td><td>44.4</td><td>18.8</td><td>28.8</td><td>57.5</td></tr><tr><td>GLM 5.2</td><td>47.3</td><td>8.1</td><td>19.6</td><td>66.9</td><td>55.9</td><td>16.9</td><td>30.7</td><td>79.4</td></tr><tr><td>Nemotron 3 Ultra</td><td>39.1</td><td>8.1</td><td>17.8</td><td>50.6</td><td>40.3</td><td>9.4</td><td>19.4</td><td>61.9</td></tr><tr><td>Inkling</td><td>49.8</td><td>3.8</td><td>13.7</td><td>57.5</td><td>59.4</td><td>13.8</td><td>28.6</td><td>95.0</td></tr><tr><td>Command A+</td><td>40.3</td><td>1.9</td><td>8.7</td><td>64.4</td><td>41.6</td><td>2.5</td><td>10.2</td><td>85.6</td></tr><tr><td>MiniMax M3</td><td>17.7</td><td>3.1</td><td>7.4</td><td>24.4</td><td>24.9</td><td>5.0</td><td>11.2</td><td>41.9</td></tr><tr><td>Open Mid-Size Models</td><td colspan="4"></td><td colspan="4"></td></tr><tr><td>Qwen 3.8 27B (think)</td><td>50.4</td><td>9.4</td><td>21.7</td><td>73.1</td><td>53.7</td><td>11.9</td><td>25.2</td><td>88.1</td></tr><tr><td>Muse Glimmer 30B (high)</td><td>49.9</td><td>8.8</td><td>20.9</td><td>91.2</td><td>52.3</td><td>10.0</td><td>22.9</td><td>92.5</td></tr><tr><td>Gemma 4 31B</td><td>45.4</td><td>3.1</td><td>11.9</td><td>93.8</td><td>50.7</td><td>6.2</td><td>17.8</td><td>93.8</td></tr><tr><td>Qwen 3.6 27B</td><td>44.0</td><td>1.9</td><td>9.1</td><td>83.1</td><td>50.0</td><td>5.0</td><td>15.8</td><td>95.0</td></tr><tr><td>Qwen 3.6 27B (think)</td><td>41.1</td><td>2.5</td><td>10.1</td><td>79.4</td><td>45.1</td><td>5.6</td><td>15.9</td><td>87.5</td></tr><tr><td>DeepSeek R1 32B</td><td>29.1</td><td>0.0</td><td>0.0</td><td>83.8</td><td>32.7</td><td>0.6</td><td>4.5</td><td>85.0</td></tr><tr><td>GLM 4.7 Flash</td><td>5.6</td><td>0.6</td><td>1.9</td><td>18.1</td><td>7.4</td><td>0.6</td><td>2.1</td><td>21.2</td></tr><tr><td>BoN Open (oracle)</td><td>50.5</td><td>4.4</td><td>14.9</td><td>90.0</td><td>57.5</td><td>8.8</td><td>22.4</td><td>100.0</td></tr></table>

Table 12: Out-of-the-box benchmarking scores for existing models on Linguini. EM = exact match, GM = geometric mean, LFMR = line format match rate.

<table><tr><td>Task</td><td>Problem Language</td><td>Question</td><td>Answer Options</td></tr><tr><td rowspan="4">Language Context</td><td>Yup&#x27;ik</td><td>Region where it is spoken?</td><td>A. General Central Yup&#x27;ik; B. eastern Alaska; C. northern Alaska; D. south- western Alaska</td></tr><tr><td>Yél Dnye</td><td>Island / place where it is spoken?</td><td>A. Brazil; B. New Britain; C. Rossel Island; D. Norton Sound</td></tr><tr><td>Iquito</td><td>Does the language have tones?</td><td>A. not known; B. yes; C. no; D. in some local dialects</td></tr><tr><td>Sakurabiat Komnzo</td><td>Approximate number of speakers? A. 3; B. 300000; C. 13; D. 13000 Village where it is spoken?</td><td>A. Tupian; B. everywhere in south-eastern Papua New Guinea; C. Rouku; D. Jipai</td></tr><tr><td rowspan="5">Lexicon</td><td>Yup&#x27;ik</td><td>Real word/phrase?</td><td>A. at&#x27;aiqnt; B. aknaepvn; C. oykey; D. nanevpak</td></tr><tr><td>Yélì Dnye</td><td>Real word/phrase?</td><td>A. iikkinin; B. otak; C. kinikini; D. hiîtnêg:</td></tr><tr><td>Iquito</td><td>Real word/phrase?</td><td>A. nat&#x27;rarkaq; B. paapaaja; C. ni- ieeani; D. paaaaajp</td></tr><tr><td>Sakurabiat</td><td>Real word/phrase?</td><td>A. abitop; B. okokty; C. ibptoa; D. nat&#x27;rarkaq</td></tr><tr><td>Komnzo</td><td>Real word/phrase?</td><td>A. zsmawmk; B. shatäzw; C. tep&#x27;liuq; D. zwäsath</td></tr></table>

Table 13: Examples of knowledge probing tasks derived from the problem statements from IOL 2026. Bold-faced answers are the correct answers. Language Context = 4-way MCQ over meta-facts from the IOL blurbs; Lexicon = 4-way MCQ over attested forms (distractors: scrambled or other-language). Questions are shortened.