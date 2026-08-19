# What Aggregate Scores Miss: Measuring Item-Level Regressions in Commercial LLM API Migrations

Xiaonan Xu<sup>a,∗</sup>, Wenjing Wu<sup>b</sup>

<sup>a</sup>College of Computing, Georgia Institute of Technology, Atlanta, GA, 30332, USA <sup>b</sup>Department of Computer Science, University of Colorado Boulder, Boulder, CO, 80309, USA

## Abstract

Context: Software systems that depend on commercial large language model APIs must migrate to successor versions when vendors deprecate older models. Migration decisions typically rely on aggregate benchmark scores, which compress heterogeneous item-level behaviour into a single net figure. Objective: We measure what that compression conceals. Method: On three pairwise upgrades in the GPT-5.4 to GPT-5.6 Sol product sequence, we query 900 public benchmark items (graduate-level knowledge, olympiad mathematics, instruction following) 50 times per item per model, classify each item as reliably improved, reliably regressed, practically equivalent, or inconclusive under falsediscovery-rate control and a practical-significance threshold, and calibrate the results against a label-permutation null. Results: Across all nine migration– benchmark cells, reliable improvements and reliable regressions coexist. Edges with aggregate gains of up to 7.3 percentage points contain up to 8.3% reliably regressed items; edges with aggregate losses contain up to 10.7% reliably improved items. On the instruction-following benchmark, the gap between strict and loose scoring widens by 3.9 percentage points on the latest migration: a 3.9-point regression under strict scoring shrinks to 0.04 points under loose scoring. Conclusion: Migration decisions based on aggregate scores alone miss

substantial bidirectional item-level change. The complete response-level archive and per-item scoring outputs are released.

Keywords: LLM evaluation, backward compatibility, regression testing, API migration, repeated sampling

## 1. Introduction

Commercial large language model (LLM) APIs have become external software dependencies in a growing share of production systems [1]. These APIs are controlled by the vendor: models are retired on published deprecation schedules, and downstream consumers must migrate to successor versions or lose access [2]. Library-migration studies show that forced dependency updates carry real cost for client projects [3], and that updates preserving interfaces can still break client behaviour [4]. Each migration replaces the model behind every downstream call site. Whether the new version preserves the test-case-level behaviour of the old one, a property known in software engineering as backward compatibility, determines whether the migration is safe.

Standard benchmark reporting compresses the answer to that question into aggregate scores. A net gain of two percentage points on a benchmark may consist of 100 items answered more reliably and 80 answered less reliably; the aggregate reports only the net balance. This compression has practical consequences: in July 2026, OpenAI’s Thibault Sottiaux publicly stated that, before the GPT-5.6 Sol launch, the team had focused on average and median usage and missed cases in which long-tail usage was substantially higher [5]. The stochastic nature of LLM outputs adds a second dificulty. A single correctto-incorrect flip between two model versions may reflect sampling noise alone. Ma et al. [2] identify non-determinism as one of three fundamental obstacles to applying regression testing to LLM APIs, and in empirical measurements on open-weight models, single-draw evaluation missed 42% of reliably changed items [6]. Backward compatibility at the item level therefore requires estimating pass-probability changes and calibrating them against a permutation null.

A longitudinal study of 18 GPT models fits ability trajectories to singledraw responses, so estimated probability shifts share one direction within each comparison [7]. A repeated-sampling study on open-weight 7–8B models measures bidirectional reliable churn at K=10, leaving frontier commercial models as an open question [6]. An industry evaluation compares GPT-5.5 and GPT-5.6 on 711 enterprise workflows at 5 runs per task with Holm-corrected significance tests [8]. Our study extends this repeated-sampling approach to frontier commercial API migrations at K=50 with permutation-null calibration of the complete classification procedure.

We compare three pairwise upgrades in the GPT-5.4 to GPT-5.6 Sol product sequence on 900 public benchmark items spanning graduate-level knowledge, olympiad mathematics, and instruction following, with all request parameters held constant except the model identifier. Each model–item pass probability is estimated from 50 independent trials, and item-level judgements are calibrated against a permutation null at matched sample size. On the instruction-following benchmark we also compare the oficial strict and loose verifier scores to test whether measured regressions depend on exact output compliance. We find that reliable improvements and reliable regressions coexist in all nine migration– benchmark cells: edges with a positive aggregate change contain up to 8.3% reliably regressed items, and edges with a negative aggregate change contain up to 10.7% reliably improved items. On the instruction-following benchmark, the strict–loose scoring gap widens by 3.9 percentage points on the latest migration: the same edge shows a 3.9-point regression under strict scoring and a 0.04-point regression under loose scoring.

The contributions of this paper are as follows.

1. We measure item-level backward compatibility across the GPT-5.4 to GPT-5.6 Sol product sequence, querying 900 public benchmark items 50 times per item per model. Item-level judgements are calibrated against a permutation null that reruns the complete classification procedure at matched sample size.

2. We show that reliable improvements and reliable regressions coexist in all nine migration–benchmark cells, including edges where the aggregate score

improved.

3. We quantify how migration conclusions change between strict and loose scoring: on the instruction-following benchmark, the latest migration widens the gap between strict and loose scoring, causing the regression observed under strict scoring to shrink under loose scoring.

4. We release the complete response-level archive with per-item scoring outputs, enabling verification and alternative rescoring without re-querying mutable commercial APIs.

## 2. Related Work

Aggregate evaluation and its limits. The dominant practice in LLM evaluation is to report aggregate scores on static benchmarks. Chang et al. [9] survey evaluation practices and identify recurring concerns: sensitivity to prompt format, inconsistency across evaluation pipelines, and the limited diagnostic value of a single aggregate figure. Benchmark saturation and data contamination amplify these concerns as models and training corpora co-evolve [10], and a mapping of generative-AI metrics to software quality characteristics concludes that no single aggregate captures the relevant quality dimensions [11]. Aggregate scores remain useful summaries of overall capability, but they reduce heterogeneous item-level behaviour to a single net figure that conflates uniform improvement with a mixture of gains and losses. Our study takes this compression as its object of measurement and quantifies what the aggregate hides.

Model-update regression in LLM APIs. Classical regression testing assumes deterministic test outcomes, an assumption already strained within conventional software by flaky tests [12]; stochastic LLM outputs invalidate it entirely. Chen et al. [13] provided early large-scale evidence that commercial GPT behaviour shifts across time-stamped snapshots, comparing two snapshots at the task level. Ma et al. [2] reframed the phenomenon as a software engineering problem, identifying three properties of commercial LLM APIs that break classical regressiontesting assumptions: vendor-controlled updates, prompt sensitivity, and nondeterministic outputs. Echterhof et al. [14] formalised negative flips (items correct before an update and incorrect after) and proposed compatibility-aware training to reduce them, addressing the problem from the model developer’s side. A production-oriented migration framework [15] uses Bayesian candidate comparison to support replacement decisions when a model reaches end-oflife. Dong et al. [16] showed that instruction-following accuracy on GPT-4o snapshots drops when prompts are paraphrased, decomposing aggregate scores along the prompt-variation axis. Our work shares the decomposition goal but varies the model version rather than the prompt, and measures pass-probability changes from repeated sampling rather than from prompt variants.

Item-level measurement and repeated sampling. Three recent studies address item-level version comparison directly. A longitudinal study covering 18 GPT models from GPT-3.5 to GPT-5.2 [7] fits a dynamic item response theory model to single-draw binary responses, estimating latent ability trajectories and localising probability changes across dificulty and discrimination regions. Because the model assigns a single ability parameter per snapshot, all item-level probability shifts share the same sign within a comparison; the authors note that heterogeneous directions require analysis of observed response flips, which the singledraw design does not support. The repeated-sampling approach of [6] addresses this directly: each item is queried K=10 times, and a reliable-change index with a permutation null separates true changes from sampling noise on open-weight Llama and Qwen models (7–8B parameters). That study reports 21–28% total reliable churn; single-draw evaluation missed 42% of reliably changed items and falsely flagged 25% of unchanged items. It identifies frontier commercial models and additional benchmarks as open questions. Toloka [8] evaluates GPT-5.5 against GPT-5.6 on 711 frozen enterprise workflows with 5 runs per task, applying stratified bootstrap and Holm-corrected significance tests to detect aggregate regressions and identify strong per-task flips. The present study extends the repeated-sampling line to frontier commercial API migrations, raises the per-item trial count to K=50, applies false-discovery-rate control with a practical-significance threshold, and calibrates item-level judgements against a permutation null.

Output variability. Repeated queries to the same commercial model return varying answers. Ouyang et al. [17] measure this directly for ChatGPT code generation, finding semantic variation across identical requests even at temperature zero, and Kim and Ming [18] compare output reliability and similarity across models on software-development tasks. A systematic catalogue attributes such divergence to sampling, silent updates, numerical rounding, and expert routing [19]. Estimating per-item pass probabilities from multiple generations, rather than judging single draws, follows the direction formalised by Zhang et al. [20]. Community guidelines for LLM-based empirical studies likewise identify output non-determinism and model evolution as threats to reproducibility and call for archived interaction traces [21], a practice the released response archive follows.

## 3. Method

## 3.1. Research questions

We study whether aggregate benchmark scores reliably reflect the backward compatibility of commercial LLM API upgrades. Three research questions structure the study.

RQ1. On each migration edge in the current GPT product line, what proportion of benchmark items shows reliable improvement, and what proportion shows reliable regression, beyond what sampling noise alone would produce?

RQ2. How does the aggregate score change on each edge relate to the underlying item-level changes, and how much bidirectional change does the aggregate figure conceal?

RQ3. How do migration conclusions change between strict and loose scoring of the same responses?

## 3.2. Study design

The unit of comparison is a migration edge: an ordered pair of models that an API consumer moves between when following the vendor’s product line. We measure three edges over three models, GPT-5.4, GPT-5.5, and GPT-5.6 Sol: the two consecutive flagship migrations (5.4→5.5 and 5.5→Sol) and the direct migration 5.4→Sol, which the vendor’s guidance also supports. The three models are served under the API identifiers gpt-5.4, gpt-5.5, and gpt-5.6-sol.

All request parameters are held constant across models; only the model identifier varies. This mirrors a dependency upgrade in which client code is unchanged and only the dependency version moves. Reasoning efort is set explicitly to medium for every model, the shared tier across the three models’ efort ranges. Temperature is not accepted by these models. No output-length cap is imposed. Each item is queried K = 50 times per model in independent single-turn calls.

## 3.3. Benchmarks and item selection

Measuring item-level change requires items on which the models under study retain headroom: an item answered correctly, or incorrectly, in every trial by every model carries no information about migration-induced change. We therefore selected benchmarks by a screening procedure defined before the main collection. Eleven public, automatically scorable candidates spanning knowledge, mathematics, and instruction following were screened with 30 randomly drawn items each, three models, and K = 10 repetitions per item. For each candidate we computed the fraction of items on which at least one model’s observed accuracy fell in [0.1, 0.9]. One benchmark per category was selected, requiring that at least 35% of items meet this criterion, and taking the candidate with the highest proportion per slot. Widely used benchmarks, including GPQA Diamond, MMLU-Pro, and IFEval, fell below the threshold on these models, with proportions between 7% and 27%; the full screening table is reported in the appendix. Screening data served benchmark selection only and did not enter the main analysis.

The selected benchmarks are SuperGPQA [22] (knowledge; 500 items drawn at random from the full pool before collection), Omni-MATH hard [23] (mathematics; 100 items drawn at random from the 452-item pool before collection), and IFBench [24] (instruction following; all 300 prompts, with the prompt as the unit of analysis), for a total of 900 items. The sampled item lists were fixed before collection and are included in the released archive.

Prompting follows the benchmark type. Knowledge and mathematics items use a minimal task statement and a final-answer-line convention, frozen after piloting. IFBench prompts are used verbatim from the oficial release, since their wording constitutes the task constraints; they are scored by the oficial verifier [24, 25].

## 3.4. Data collection

Collection proceeded in per-model phases in release order (GPT-5.4, then GPT-5.5, then GPT-5.6 Sol), with item–repetition order randomised within each phase. Collection ran from 30 July to 13 August 2026. All phases ran at a fixed request concurrency of 36. Transient API errors were retried with exponential backof until success; the final response matrix is complete.

## 3.5. Scoring

Scoring is decoupled from collection and deterministic given the archived responses. Every response is assigned to one of six mutually exclusive categories: semantically correct, semantically wrong, format failure, refusal, truncation, and API error; given the retry policy, every response in the final matrix falls in the first five categories. API errors are excluded from the accuracy denominator and their occurrence rate is reported separately; refusals, format failures, and model-side truncations remain in the denominator as behavioural failures.

For knowledge and mathematics items, a response is scored correct when a two-stage parser extracts an answer from the final answer line and that answer matches the reference; mathematical equivalence is established by exact match followed by symbolic normalisation. IFBench responses are scored by the oficial verifiers [24], which provide a strict and a loose reading; the loose reading re-applies each verification function after a fixed set of surface-format normalisations, so the two readings difer only in format tolerance. The strict reading is primary. Responses that a downstream system could not parse under the agreed format are counted as failures, since for an API consumer an unparseable reply is operationally indistinguishable from a wrong one.

## 3.6. Analysis

For item i under model $m ,$ the observed accuracy is $\hat { p } _ { m , i }$ , the fraction of the K trials scored correct under the primary (strict) rule. For each edge, source s to target $t ,$ we report per benchmark the aggregate change $\Delta = \mathrm { m e a n } _ { i } ( \hat { p } _ { t , i } ) -$ mean<sub>i</sub> $( \hat { p } _ { s , i } )$ ; the reliable-improvement share $P ^ { + }$ ; the reliable-regression share $P ^ { - } ;$ and the reliable-change share $P ^ { c h g } = P ^ { + } + P ^ { - }$

Item-level judgements require both statistical significance and a minimum efect size. A Fisher exact test on the $2 \times K$ correct/incorrect counts, with Benjamini–Hochberg control of the false discovery rate at $5 \%$ within each migration benchmark cell, establishes significance. A practical-significance threshold $\hat { p } _ { t , i ^ { - } }$ $\hat { p } _ { s , i } | \geq \varepsilon$ with $\varepsilon = 0 . 2$ (a 20-percentage-point shift in pass probability) establishes a minimum efect size. Items meeting both criteria are classified as reliably improved or reliably regressed by sign. Items whose 95% confidence interval for the diference lies entirely within $( - \varepsilon , \varepsilon )$ are classified as practically equivalent; the remainder are classified as inconclusive.

Observed shares are calibrated against a permutation null. For each item we pool the 2K outcomes of the two models on an edge, randomly reassign version labels with K outcomes per side, and rerun the complete classification procedure; 1,000 replications yield the null distribution of $P ^ { + }$ , $P ^ { - }$ , and $P ^ { c h g }$ under zero true change at matched sample size. Reported shares are presented alongside the 95th percentile of this null, so that reliable change is claimed only where it exceeds what label noise alone produces.

For the secondary analysis on IFBench, let $R _ { m }$ denote the mean diference

Table 1: Aggregate change and item-level classification for each migration edge and benchmark. $\Delta$ is the change in mean strict accuracy in percentage points. $P ^ { + } , P ^ { - }$ , and $P ^ { c h g }$ are the shares of items classified as reliably improved, reliably regressed, and reliably changed under the Fisher $/ \mathrm { B H } / \varepsilon { = } 0 . 2$ criteria; Equiv. and Incon. are the practically-equivalent and inconclusive shares. Null is the 95th percentile of $P ^ { c h g }$ under the permutation null (1,000 replications). All shares in percent; $\Delta$ is computed before rounding
<table><tr><td>Edge</td><td>Benchmark</td><td> $\Delta$ </td><td> $P ^ { + }$ </td><td> $P ^ { - }$ </td><td> $P ^ { c h g }$ </td><td>Equiv.</td><td>Incon.</td><td>Null95</td></tr><tr><td rowspan="3">5.4→5.5</td><td>SuperGPQA</td><td>+2.3</td><td>8.2</td><td>5.0</td><td>13.2</td><td>77.0</td><td>9.8</td><td>0.0</td></tr><tr><td>Omni-MATH hard</td><td>+7.3</td><td>17.0</td><td>6.0</td><td>23.0</td><td>59.0</td><td>18.0</td><td>0.0</td></tr><tr><td>IFBench</td><td>+1.9</td><td>11.3</td><td>8.3</td><td>19.7</td><td>55.7</td><td>24.7</td><td>0.0</td></tr><tr><td rowspan="3">5.5→Sol</td><td>SuperGPQA</td><td>+1.4</td><td>7.6</td><td>4.4</td><td>12.0</td><td>80.4</td><td>7.6</td><td>0.0</td></tr><tr><td>Omni-MATH hard</td><td>-4.1</td><td>9.0</td><td>15.0</td><td>24.0</td><td>67.0</td><td>9.0</td><td>0.0</td></tr><tr><td>IFBench</td><td>-3.9</td><td>6.7</td><td>13.3</td><td>20.0</td><td>57.0</td><td>23.0</td><td>0.0</td></tr><tr><td rowspan="3">5.4→Sol</td><td>SuperGPQA</td><td>+3.8</td><td>10.8</td><td>4.8</td><td>15.6</td><td>76.8</td><td>7.6</td><td>0.0</td></tr><tr><td>Omni-MATH hard</td><td>+3.2</td><td>10.0</td><td>5.0</td><td>15.0</td><td>70.0</td><td>15.0</td><td>0.0</td></tr><tr><td>IFBench</td><td>-2.0</td><td>10.7</td><td>13.3</td><td>24.0</td><td>54.3</td><td>21.7</td><td>0.0</td></tr></table>

between the loose and strict verifier scores across prompts. Each edge reports $\Delta R = R _ { t } - R _ { s }$ , the change in this gap across the migration.

## 4. Results

## 4.1. Aggregate scores and item-level changes

Mean strict accuracy was 63.4%, 45.7%, and 62.8% for GPT-5.4 on SuperG-PQA, Omni-MATH hard, and IFBench; 65.7%, 53.0%, and 64.6% for GPT-5.5; and 67.1%, 48.9%, and 60.7% for Sol. Table 1 summarises the aggregate change and the full item-level classification for each migration edge and benchmark; Figure 1 shows the reliable-improvement and reliable-regression shares alongside the aggregate change.

On the 5.4→5.5 edge, aggregate scores rose on all three benchmarks (+2.3, +7.3, and +1.9 percentage points on SuperGPQA, Omni-MATH hard, and IF-Bench). All three benchmarks also show reliable regression: 5.0% of SuperG-

![](images/903842cbeb4dcb6bc749cc8a1f1c0fc0a72fa269d1ae74f9b62389a6603be63c.jpg)  
Figure 1: Item-level reliable improvement and regression alongside aggregate change for each migration edge and benchmark. Bars extending right (left) show the share of items reliably improved (regressed); the central marker shows the aggregate change ∆

PQA items, 6.0% of Omni-MATH items, and 8.3% of IFBench items regressed reliably despite a positive aggregate change.

On the 5.5→Sol edge, the three benchmarks diverge. SuperGPQA continued to rise (+1.4) while Omni-MATH hard (−4.1) and IFBench (−3.9) declined. Both declining benchmarks still contain reliably improved items: 9.0% of Omni-MATH items and 6.7% of IFBench items improved reliably even as the aggregate fell.

On the direct 5.4→Sol edge, SuperGPQA showed the largest gain (+3.8) and the highest improvement share (10.8%), while IFBench showed a net loss (−2.0) with 13.3% of items reliably regressed and 10.7% reliably improved. Reliablechange shares on this edge difer from the sums of the two consecutive edges, reflecting item-level movements that cancel across consecutive migrations.

## 4.2. Permutation-null calibration

Under zero true change, the 95th percentile of the reliable-change share was 0.0% in every migration–benchmark cell: across 1,000 label permutations at the study’s sample sizes, the Fisher/BH/ε criteria produced zero spurious reliable changes. Every reliably changed item in Table 1 therefore lies strictly above the noise floor. Prior work on open-weight 7–8B models reported total churn of 21–28% at K=10 [6]; the shares of 12.0–24.0% observed here are of comparable magnitude yet rest on a calibrated zero baseline.

## 4.3. Strict–loose scoring gap on IFBench

IFBench’s oficial verifiers provide a strict and a loose reading of every response. The gap $R _ { m }$ between the two readings was 6.3 percentage points for GPT-5.4, 5.1 for GPT-5.5, and 9.0 for Sol, yielding a $\Delta R$ of −1.2 points on the 5.4→5.5 edge, +3.9 on 5.5→Sol, and +2.7 on 5.4→Sol. The widening on the 5.5→Sol edge accounts for nearly the entire aggregate decline: Sol’s strict score falls 3.9 points below GPT-5.5’s, while its loose score falls 0.04 points; the regression is concentrated in exact constraint compliance. On SuperGPQA and Omni-MATH hard, responses are scored by answer extraction and exact match; extraction failures were negligible, and the six-category breakdown in the appendix reports their rates.

## 4.4. Supplementary analyses

The appendix reports five supplementary analyses on the same data: a single-draw comparison that contrasts raw correct-to-incorrect and incorrectto-correct flips from one randomly selected trial per item with the K=50 classification; a sensitivity analysis of the practical-significance threshold over $\varepsilon \in$ {0.10, 0.15, 0.20, 0.25}; token-cost distributions per model and edge; the sixcategory response breakdown; and dificulty stratification using metadata provided by each benchmark.

## 5. Discussion

Across all nine migration–benchmark cells, reliable improvements and reliable regressions occur together. The permutation null confirms that the observed shares exceed what sampling noise alone produces at K=50. Aggregate scores therefore provide a necessary but insuficient basis for migration decisions:

on the six cells where the aggregate improved, a gate based on the aggregate alone would have accepted the migration while 4.4–8.3% of items regressed reliably.

The IFBench results raise a separate concern. Because the two readings difer only in format tolerance, the widening of the strict–loose gap on the 5.5-to-Sol edge locates part of the observed strict-scoring regression in exact format compliance rather than in the underlying task. For a system that parses outputs programmatically, format non-compliance is a functional regression; a consumer that tolerates format variation faces a smaller compatibility cost from the same migration. Migration risk therefore depends on the acceptance criteria applied to model outputs, and capturing both perspectives requires reporting both readings.

The repeated-sampling study on open-weight models [6] asked whether its findings extend to frontier commercial models; the present measurements provide a direct comparison point at K=50 on three GPT versions. The Toloka evaluation [8] identified per-task regressions on the same product line at K=5 with Holm-corrected tests; our higher per-item trial count and permutation null complement that evidence by separating reliable item-level changes from sampling artefacts at higher statistical resolution.

The appendix compares single-draw evaluation with the repeated-sampling judgements directly: a single binary observation per model cannot reach significance under the item-level test, so none of the 457 reliable item-level changes across the nine cells is detectable from one draw, and the appendix additionally reports how raw single-draw flips distribute over the K=50 classes. The gap between single-draw and K=50 judgements sets a lower bound on the information lost when migration decisions rely on single-draw evaluation. Organisations that treat model upgrades as dependency updates can use the item-level regression shares reported here as reference rates when designing acceptance tests for their own workloads.

## 6. Threats to Validity

Three limitations bound the scope of these measurements. The benchmark screening step selects benchmarks on which the studied models retain headroom; the reported reliable-change shares are therefore estimates for informative benchmarks, not population rates over arbitrary workloads. The study covers one vendor’s product line (GPT-5.4 through GPT-5.6 Sol) and three public benchmarks spanning knowledge, mathematics, and instruction following; other vendors, task types, and production workloads may show diferent patterns. The practical-significance threshold $\varepsilon = 0 . 2$ and the trial count $K { = } 5 0$ together determine which item-level changes are detectable: smaller genuine changes remain in the inconclusive category and are reported as such.

## 7. Conclusion

Across three pairwise upgrades in the GPT-5.4 to GPT-5.6 Sol product sequence, with 900 public benchmark items, 50 trials per item, and calibration against a permutation null, every migration–benchmark cell contains both reliably improved and reliably regressed items; aggregate gains of up to 7.3 percentage points accompany up to 8.3% reliably regressed items. The strict–loose scoring gap widens on the latest migration for instruction following, indicating that part of the measured regression concentrates on format compliance. The complete response-level archive and per-item scoring outputs are released to support verification and alternative analyses.

## Data Availability

The complete response-level archive (request parameters, raw responses, returned model identifiers, usage metadata, and timestamps) and the per-item scoring outputs are available at https://github.com/WenJing95/gpt-regression-data. All reported statistics follow the procedures specified in Section 3 and can be recomputed from this archive without re-querying commercial APIs.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the writing process

During the preparation of this work the authors used Claude (Anthropic) and ChatGPT (OpenAI) in order to improve the language and readability of the manuscript. After using these tools, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## CRediT authorship contribution statement

Xiaonan Xu: Conceptualization, Methodology, Investigation, Data curation, Writing – original draft. Wenjing Wu: Software, Validation, Formal analysis, Writing – review & editing.

## References

[1] X. Hou, Y. Zhao, Y. Liu, Z. Yang, K. Wang, L. Li, X. Luo, D. Lo, J. Grundy, H. Wang, Large language models for software engineering: A systematic literature review, ACM Transactions on Software Engineering and Methodology 33 (8) (2024) 220:1–220:79.

[2] W. Ma, C. Yang, C. Kästner, (Why) is my prompt getting worse? Rethinking regression testing for evolving LLM APIs, in: Proceedings of the IEEE/ACM 3rd International Conference on AI Engineering – Software Engineering for AI (CAIN), 2024, pp. 166–171.

[3] R. G. Kula, D. M. German, A. Ouni, T. Ishio, K. Inoue, Do developers update their library dependencies? an empirical study on the impact of security advisories on library migration, Empirical Software Engineering 23 (1) (2018) 384–417.

[4] D. Jayasuriya, V. Terragni, J. Dietrich, K. Blincoe, Understanding the impact of APIs behavioral breaking changes on client applications, Proceedings of the ACM on Software Engineering 1 (FSE) (2024) 1238–1261.

[5] T. Sottiaux, Sol community update: GPT-5.6 Sol usage quotas, X (formerly Twitter), https://x.com/thsottiaux/status/2082317452755751098, post of 29 July 2026. Accessed 16 August 2026 (2026).

[6] J.-P. Cacioli, Beyond the mean: Within-model reliable change detection for LLM evaluation, arXiv preprint arXiv:2604.27405 (2026).

[7] Anonymous, Longitudinal evaluation of large language models, under review for Transactions on Machine Learning Research (TMLR), Paper 8871. Submitted 2026-05-11. https://openreview.net/forum?id=INuSvLC7Bq (2026).

[8] Toloka AI, GPT-5.6 got smarter. Then it kept acting., https://toloka. ai/blog/gpt-5.6-got-smarter-then-it-kept-acting/, blog post, July 2026. Accessed 16 August 2026 (2026).

[9] Y. Chang, X. Wang, J. Wang, Y. Wu, L. Yang, K. Zhu, H. Chen, X. Yi, C. Wang, Y. Wang, et al., A survey on evaluation of large language models, ACM Transactions on Intelligent Systems and Technology 15 (3) (2024) 39:1–39:45.

[10] S. Chen, Y. Chen, Z. Li, Y. Jiang, Z. Wan, Y. He, D. Ran, T. Gu, H. Li, T. Xie, B. Ray, Benchmarking large language models under data contamination: A survey from static to dynamic evaluation, in: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, 2025, pp. 10080–10098. doi:10.18653/v1/2025.emnlp-main.511.

[11] L. Yu, E. Alégroth, P. Chatzipetrou, T. Gorschek, Measuring the quality of generative AI systems: Mapping metrics to quality characteristics— snowballing literature review, Information and Software Technology 186 (2025) 107802.

[12] Q. Luo, F. Hariri, L. Eloussi, D. Marinov, An empirical analysis of flaky tests, in: Proceedings of the 22nd ACM SIGSOFT International Sympo sium on Foundations of Software Engineering (FSE), 2014, pp. 643–653.

[13] L. Chen, M. Zaharia, J. Zou, How is ChatGPT’s behavior changing over time?, arXiv preprint arXiv:2307.09009 (2023).

[14] J. M. Echterhof, F. Faghri, R. Vemulapalli, T.-Y. Hu, C.-L. Li, O. Tuzel, H. Pouransari, MUSCLE: A model update strategy for compatible LLM evolution, in: Findings of the Association for Computational Linguistics: EMNLP 2024, 2024, pp. 7320–7332.

[15] E. Casey, D. Roberts, D. Sim, I. Beaver, When your LLM reaches endof-life: A framework for confident model migration in production systems, arXiv preprint arXiv:2604.27082 (2026).

[16] J. Dong, Y. Zhang, Y. Liu, Z. Zhong, T. Wei, C. Zhang, H. Qiu, Revisiting the reliability of language models in instruction-following, in: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026, pp. 7784–7812. doi:10.18653/v1/2026.acl-long.354.

[17] S. Ouyang, J. M. Zhang, M. Harman, M. Wang, An empirical study of the

non-determinism of ChatGPT in code generation, ACM Transactions on Software Engineering and Methodology 34 (2) (2025) 42:1–42:28.

[18] D.-K. Kim, H. Ming, Assessing output reliability and similarity of large language models in software development: A comparative case study approach, Information and Software Technology 185 (2025) 107787.

[19] G. Coqueret, J. Llull, F. Oswald, C. Pérignon, C. Scheuch, L. Vilhuber, Randomness in large language models: What researchers need to know (and report), arXiv preprint arXiv:2607.24372 (2026).

[20] W. Zhang, H. Cai, W. Chen, Beyond the singular: Revealing the value of multiple generations in benchmark evaluation, in: Findings of the Association for Computational Linguistics: ACL 2026, 2026, pp. 10033–10043.

[21] S. Baltes, F. Angermeir, C. Arora, M. M. Barón, C. Chen, L. Böhme, F. Calefato, N. Ernst, D. Falessi, B. Fitzgerald, et al., Evaluation guidelines for empirical studies in software engineering involving LLMs, arXiv preprint arXiv:2508.15503 (2025).

[22] M.-A.-P. Team, SuperGPQA: Scaling LLM evaluation across 285 graduate disciplines, arXiv preprint arXiv:2502.14739 (2025).

[23] B. Gao, F. Song, Z. Yang, Z. Cai, Y. Miao, Q. Dong, L. Li, C. Ma, L. Chen, R. Xu, et al., Omni-MATH: A universal olympiad level mathematic benchmark for large language models, arXiv preprint arXiv:2410.07985 (2024).

[24] V. Pyatkin, S. Malik, V. Graf, H. Ivison, S. Huang, P. Dasigi, N. Lambert, H. Hajishirzi, Generalizing verifiable instruction following, in: Advances in Neural Information Processing Systems 38 (NeurIPS 2025) Datasets and Benchmarks Track, 2025.

[25] J. Zhou, T. Lu, S. Mishra, S. Brahma, S. Basu, Y. Luan, D. Zhou, L. Hou, Instruction-following evaluation for large language models, arXiv preprint arXiv:2311.07911 (2023).

## Appendix A. Benchmark screening

Table A.2 reports the benchmark screening described in Section 3.3: for each of the eleven candidate benchmarks, 30 randomly drawn items were queried K=10 times by each of the three models, and an item counts as estimable when at least one model’s observed accuracy falls in [0.1, 0.9]. The screening run for ComplexBench did not complete.

Table A.2: Benchmark screening: estimable items out of 30 per candidate (K=10, three models). One benchmark per slot was selected, requiring a share of at least 35% and taking the highest-share candidate per slot
<table><tr><td>Slot</td><td>Candidate</td><td>Estimable</td><td>Share (%)</td><td>Outcome</td></tr><tr><td rowspan="4">Knowledge</td><td>SuperGPQA</td><td>13/30</td><td>43.3</td><td>Selected</td></tr><tr><td>GPQA Diamond</td><td>8/30</td><td>26.7</td><td>Not selected</td></tr><tr><td>SuperGPQA hard</td><td>7/30</td><td>23.3</td><td>Not selected</td></tr><tr><td>MMLU-Pro</td><td>2/30</td><td>6.7</td><td>Not selected</td></tr><tr><td rowspan="3">Mathematics</td><td>Omni-MATH hard</td><td>21/30</td><td>70.0</td><td>Selected</td></tr><tr><td>OlymMATH</td><td>16/30</td><td>53.3</td><td>Not selected</td></tr><tr><td>OlympiadBench text math</td><td>11/30</td><td>36.7</td><td>Not selected</td></tr><tr><td rowspan="4">Instruction following</td><td>IFBench</td><td>17/30</td><td>56.7</td><td>Selected</td></tr><tr><td>ComplexBench</td><td>11/30</td><td>36.7</td><td>Incomplete</td></tr><tr><td>IFEval</td><td>8/30</td><td>26.7</td><td>Not selected</td></tr><tr><td>IFEval++</td><td>3/30</td><td>10.0</td><td>Not selected</td></tr></table>

## Appendix B. Single-draw comparison

For each item and model, one trial was drawn uniformly at random from the K=50 archived trials under a fixed seed, and the classification procedure of Section 3.6 was applied to the resulting single observations. A comparison of two single binary observations never reaches significance under the Fisher/Benjamini– Hochberg criteria, so every item on every edge is classified as inconclusive and none of the 457 reliable item-level changes is recovered. Table B.3 reports the raw correct-to-incorrect and incorrect-to-correct flips observed in the single draws; Table B.4 cross-tabulates these flips against the $K { = } 5 0$ classification, aggregated over the nine migration–benchmark cells.

Table B.3: Single-draw comparison. Reliable changes under the full K=50 procedure, reliable changes detected from single draws, and raw flips between the single draws of source and target model (I→C: incorrect to correct; C→I: correct to incorrect)
<table><tr><td>Edge</td><td>Benchmark</td><td>Items</td><td>K=50 rel. changes</td><td>Single-draw rel. changes</td><td>I→C</td><td>C→I</td></tr><tr><td rowspan="3">5.4→5.5</td><td>SuperGPQA</td><td>500</td><td>66</td><td>0</td><td>27</td><td>20</td></tr><tr><td>Omni-MATH hard</td><td>100</td><td>23</td><td>0</td><td>14</td><td>4</td></tr><tr><td>IFBench</td><td>300</td><td>59</td><td>0</td><td>42</td><td>19</td></tr><tr><td rowspan="3">5.5→Sol</td><td>SuperGPQA</td><td>500</td><td>60</td><td>0</td><td>28</td><td>19</td></tr><tr><td>Omni-MATH hard</td><td>100</td><td>24</td><td>0</td><td>6</td><td>10</td></tr><tr><td>IFBench</td><td>300</td><td>60</td><td>0</td><td>25</td><td>47</td></tr><tr><td rowspan="3">5.4→Sol</td><td>SuperGPQA</td><td>500</td><td>78</td><td>0</td><td>38</td><td>22</td></tr><tr><td>Omni-MATH hard</td><td>100</td><td>15</td><td>0</td><td>9</td><td>3</td></tr><tr><td>IFBench</td><td>300</td><td>72</td><td>0</td><td>32</td><td>31</td></tr></table>

Table B.4: Raw single-draw flips cross-tabulated against the K=50 classification, aggregated over the nine migration–benchmark cells (2,700 item–edge pairs)
<table><tr><td>K=50 classification</td><td> $\mathrm { I } {  } \mathrm { C }$ </td><td> $\mathrm { C } {  } \mathrm { I }$ </td><td>Unchanged</td><td>Total</td></tr><tr><td>Reliably improved</td><td>120</td><td>8</td><td>127</td><td>255</td></tr><tr><td>Reliably regressed</td><td>10</td><td>87</td><td>105</td><td>202</td></tr><tr><td>Practically equivalent</td><td>25</td><td>26</td><td>1817</td><td>1868</td></tr><tr><td>Inconclusive</td><td>66</td><td>54</td><td>255</td><td>375</td></tr><tr><td>Total</td><td>221</td><td>175</td><td>2304</td><td>2700</td></tr></table>

## Appendix C. Sensitivity to the practical-significance threshold

Table C.5 reports the reliable-improvement, reliable-regression, and reliablechange shares of each migration–benchmark cell when the practical-significance

threshold varies over $\varepsilon \in \{ 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , 0 . 2 5 \}$ , holding the Fisher/Benjamini– Hochberg criterion fixed. The rows with $\varepsilon = 0 . 2 0$ correspond to Table 1.

## Appendix D. Token consumption

Table D.6 summarises per-request token counts for each benchmark and model over the 135,000 archived responses; Table D.7 reports the change in mean total tokens per request on each migration edge.

## Appendix E. Response category breakdown

Table E.8 reports the six-category classification of Section 3.5 for all archived responses. Truncation and API error do not occur in the final response matrix; refusals occur only on IFBench.

## Appendix F. Dificulty stratification

Table F.9 reports mean strict accuracy per model and the per-edge change within strata defined by benchmark-native metadata: SuperGPQA provides dificulty and discipline labels, and Omni-MATH hard provides a dificulty rating. IFBench provides no comparable dificulty metadata and is therefore not stratified.

Table C.5: $P ^ { + } , P ^ { - }$ , and $P ^ { c h g }$ per migration–benchmark cell for four values of the practicalsignificance threshold $\varepsilon .$ All shares in percent
<table><tr><td>Edge</td><td>Benchmark</td><td>ε</td><td> $P ^ { + }$ </td><td> $P ^ { - }$  1</td><td> $P ^ { c h g }$ </td></tr><tr><td rowspan="10">5.4→5.5</td><td rowspan="4">SuperGPQA</td><td>0.10</td><td>8.4</td><td>5.2</td><td>13.6</td></tr><tr><td>0.15</td><td>8.4</td><td>5.2</td><td>13.6</td></tr><tr><td>0.20</td><td>8.2</td><td>5.0</td><td>13.2</td></tr><tr><td>0.25</td><td>8.0</td><td>4.6</td><td>12.6</td></tr><tr><td rowspan="4">Omni-MATH hard</td><td>0.10</td><td>19.0</td><td>6.0</td><td>25.0</td></tr><tr><td>0.15</td><td>19.0</td><td>6.0</td><td>25.0</td></tr><tr><td>0.20</td><td>17.0</td><td>6.0</td><td>23.0</td></tr><tr><td>0.25</td><td>15.0</td><td>3.0</td><td>18.0</td></tr><tr><td rowspan="4">IFBench</td><td>0.10</td><td>12.7</td><td>9.0</td><td>21.7</td></tr><tr><td>0.15</td><td>12.7</td><td>9.0</td><td>21.7</td></tr><tr><td>0.20</td><td>11.3</td><td>8.3</td><td>19.7</td></tr><tr><td>0.25</td><td>10.0</td><td>7.3</td><td>17.3</td></tr><tr><td rowspan="9">5.5→Sol</td><td rowspan="3">SuperGPQA</td><td>0.10</td><td>8.0</td><td>4.4</td><td>12.4</td></tr><tr><td>0.15</td><td>8.0</td><td>4.4</td><td>12.4</td></tr><tr><td>0.20</td><td>7.6</td><td>4.4</td><td>12.0</td></tr><tr><td rowspan="3"></td><td>0.25</td><td>6.6</td><td>3.8</td><td>10.4</td></tr><tr><td>0.10</td><td>10.0</td><td>16.0</td><td>26.0</td></tr><tr><td>0.15 0.20</td><td>10.0 9.0</td><td>16.0 15.0</td><td>26.0 24.0</td></tr><tr><td rowspan="4"></td><td>0.25</td><td>6.0</td><td>12.0</td><td>18.0</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>0.10</td><td>7.0</td><td>14.0</td><td>21.0</td></tr><tr><td>0.15 0.20</td><td>7.0</td><td>14.0</td><td>21.0</td></tr><tr><td rowspan="4"></td><td></td><td>6.7</td><td>13.3 12.7</td><td>20.0 18.7</td></tr><tr><td>0.25</td><td>6.0</td><td></td><td></td></tr><tr><td>0.10 0.15</td><td>11.4</td><td>5.0</td><td>16.4</td></tr><tr><td></td><td>11.4</td><td>5.0</td><td>16.4 15.6</td></tr><tr><td rowspan="9">5.4→Sol</td><td rowspan="3"></td><td>0.20</td><td>10.8</td><td>4.8</td><td></td></tr><tr><td>0.25</td><td>10.2</td><td>4.6</td><td>14.8</td></tr><tr><td>0.10</td><td>11.0</td><td>5.0</td><td>16.0</td></tr><tr><td rowspan="3">Omni-MATH hard</td><td>0.15</td><td>11.0</td><td>5.0</td><td>16.0</td></tr><tr><td>0.20</td><td>10.0</td><td>5.0</td><td>15.0</td></tr><tr><td>0.25</td><td>8.0</td><td>3.0</td><td>11.0</td></tr><tr><td rowspan="4">IFBench</td><td>0.10</td><td>11.7</td><td>14.0</td><td>25.7</td></tr><tr><td>0.15</td><td>11.3</td><td>14.0</td><td>25.3</td></tr><tr><td>0.20</td><td>10.7</td><td>13.3</td><td>24.0</td></tr><tr><td>0.25</td><td>9.3</td><td>12.3</td><td>21.7</td></tr></table>

Table D.6: Per-request token counts by benchmark and model. Mean input and output tokens, and mean, median, 5th and 95th percentile of total tokens
<table><tr><td colspan="3"></td><td colspan="2">Mean</td><td colspan="4">Total tokens</td></tr><tr><td>Benchmark</td><td>Model</td><td>Input</td><td>Output</td><td>Mean</td><td>Median</td><td>P5</td><td></td><td>P95</td></tr><tr><td rowspan="3">SuperGPQA</td><td>GPT-5.4</td><td>543.2</td><td>555.2</td><td>1098.4</td><td></td><td>758</td><td>468</td><td>2692</td></tr><tr><td>GPT-5.5</td><td>543.2</td><td>543.5</td><td>1086.7</td><td>824</td><td></td><td>494</td><td>2767</td></tr><tr><td>Sol</td><td>543.8</td><td>409.7</td><td>953.5</td><td>702</td><td></td><td>476</td><td>2244</td></tr><tr><td rowspan="3">Omni-MATH hard</td><td>GPT-5.4</td><td>444.8</td><td>7653.3</td><td>8098.1</td><td></td><td>5732</td><td>1521</td><td>20779</td></tr><tr><td>GPT-5.5</td><td>444.8</td><td>3705.3</td><td>4150.1</td><td></td><td>3684</td><td>991</td><td>8751</td></tr><tr><td>Sol</td><td>444.8</td><td>3766.1</td><td>4211.0</td><td></td><td>3527</td><td>1139</td><td>9128</td></tr><tr><td rowspan="3">IFBench</td><td>GPT-5.4</td><td>375.1</td><td>1030.2</td><td>1405.3</td><td></td><td>997</td><td>461</td><td>3699</td></tr><tr><td>GPT-5.5</td><td>375.3</td><td>981.8</td><td>1357.1</td><td></td><td>992</td><td>466</td><td>3339</td></tr><tr><td>Sol</td><td>394.1</td><td>1452.9</td><td>1847.0</td><td></td><td>1118</td><td>443</td><td>5613</td></tr></table>

Table D.7: Mean total tokens per request on each migration edge
<table><tr><td>Edge</td><td>Benchmark</td><td>Source</td><td>Target</td><td>Change</td></tr><tr><td rowspan="3">5.4→5.5</td><td>SuperGPQA</td><td>1098.4</td><td>1086.7</td><td>-11.7</td></tr><tr><td>Omni-MATH hard</td><td>8098.1</td><td>4150.1</td><td>-3948.0</td></tr><tr><td>IFBench</td><td>1405.3</td><td>1357.1</td><td>-48.1</td></tr><tr><td rowspan="4">5.5→Sol</td><td>SuperGPQA</td><td>1086.7</td><td>953.5</td><td>-133.2</td></tr><tr><td>Omni-MATH hard</td><td>4150.1</td><td>4211.0</td><td>+60.9</td></tr><tr><td>IFBench</td><td>1357.1</td><td>1847.0</td><td>+489.8</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td rowspan="3">5.4→Sol</td><td>SuperGPQA</td><td>1098.4</td><td>953.5</td><td>-144.9</td></tr><tr><td>Omni-MATH hard</td><td>8098.1</td><td>4211.0</td><td>-3887.2</td></tr><tr><td>IFBench</td><td>1405.3</td><td>1847.0</td><td>+441.7</td></tr></table>

Table E.8: Six-category response classification by benchmark and model (counts; N is items × K=50)
<table><tr><td>Benchmark</td><td>Model</td><td>N</td><td>Correct</td><td>Wrong</td><td>Format</td><td>Refusal</td><td>Trunc.</td><td>API err.</td></tr><tr><td rowspan="3">SuperGPQA</td><td>GPT-5.4</td><td>25000</td><td>15841</td><td>8913</td><td>246</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GPT-5.5</td><td>25000</td><td>16424</td><td>8205</td><td>371</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Sol</td><td>25000</td><td>16780</td><td>7932</td><td>288</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3">Omni-MATH hard</td><td>GPT-5.4</td><td>5000</td><td>2285</td><td>2689</td><td>26</td><td>0</td><td>0</td><td>0</td></tr><tr><td>GPT-5.5</td><td>5000</td><td>2648</td><td>2326</td><td>26</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Sol</td><td>5000</td><td>2445</td><td>2555</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td rowspan="3">IFBench</td><td>GPT-5.4</td><td>15000</td><td>9418</td><td>4635</td><td>945</td><td>2</td><td>0</td><td>0</td></tr><tr><td>GPT-5.5</td><td>15000</td><td>9696</td><td>4528</td><td>771</td><td>5</td><td>0</td><td>0</td></tr><tr><td>Sol</td><td>15000</td><td>9111</td><td>4534</td><td>1352</td><td>3</td><td>0</td><td>0</td></tr></table>

Table F.9: Mean strict accuracy (%) per model and change per edge (percentage points) within benchmark-native strata
<table><tr><td>Benchmark</td><td>Stratum</td><td>Items</td><td>5.4</td><td>5.5</td><td>Sol</td><td> $\Delta _ { 5 . 4  5 . 5 }$ </td><td> $\Delta _ { \mathrm { 5 . 5  S o l } }$ </td><td> $\Delta _ { \mathrm { 5 . 4 }  \mathrm { S o l } }$ </td></tr><tr><td rowspan="14">SuperGPQA</td><td>Difficulty easy</td><td>156</td><td>64.0</td><td>69.3</td><td>72.4</td><td>+5.3</td><td>+3.1</td><td>+8.4</td></tr><tr><td>Difficulty middle</td><td>204</td><td>69.8</td><td>70.8</td><td>72.6</td><td>+0.9</td><td>+1.8</td><td>+2.7</td></tr><tr><td>Difficulty hard</td><td>140</td><td>53.2</td><td>54.3</td><td>53.3</td><td>+1.0</td><td>-1.0</td><td>+0.1</td></tr><tr><td>Agronomy</td><td>10</td><td>23.2</td><td>23.4</td><td>27.4</td><td>+0.2</td><td>+4.0</td><td>+4.2</td></tr><tr><td>Economics</td><td>17</td><td>74.0</td><td>76.7</td><td>78.6</td><td>+2.7</td><td>+1.9</td><td>+4.6</td></tr><tr><td>Education</td><td>10</td><td>50.6</td><td>67.2</td><td>72.8</td><td>+16.6</td><td>+5.6</td><td>+22.2</td></tr><tr><td>Engineering</td><td>135</td><td>66.5</td><td>70.3</td><td>70.3</td><td>+3.8</td><td>-0.0</td><td>+3.7</td></tr><tr><td>History</td><td>16</td><td>65.1</td><td>84.2</td><td>82.9</td><td>+19.1</td><td>-1.4</td><td>+17.8</td></tr><tr><td>Law</td><td>6</td><td>37.3</td><td>61.3</td><td>55.3</td><td>+24.0</td><td>-6.0</td><td>+18.0</td></tr><tr><td>Literature and Arts</td><td>41</td><td>55.9</td><td>59.9</td><td>64.0</td><td>+4.0</td><td>+4.2</td><td>+8.2</td></tr><tr><td>Management</td><td>12</td><td>57.2</td><td>57.2</td><td>64.0</td><td>0.0</td><td>+6.8</td><td>+6.8</td></tr><tr><td>Medicine</td><td>57</td><td>71.3</td><td>68.1</td><td>74.4</td><td>-3.2</td><td>+6.2</td><td>+3.0</td></tr><tr><td>Military Science</td><td>4</td><td>54.5</td><td>27.0</td><td>43.0</td><td>-27.5</td><td>+16.0</td><td>-11.5</td></tr><tr><td>Philosophy</td><td>11</td><td>71.1</td><td>72.0</td><td>78.0</td><td>+0.9</td><td>+6.0</td><td>+6.9</td></tr><tr><td>Science</td><td>180</td><td>62.8</td><td>63.4</td><td>62.9</td><td>+0.6</td><td>-0.5</td><td>+0.1</td></tr><tr><td>Sociology</td><td>1</td><td>100.0</td><td>100.0</td><td>100.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Omni-MATH hard</td><td>Difficulty 8.0</td><td>61</td><td>43.8</td><td>52.0</td><td>47.6</td><td>+8.2</td><td>-4.4</td><td>+3.8</td></tr><tr><td>Difficulty 8.5</td><td></td><td>2 49.0</td><td></td><td>50.0</td><td>49.0</td><td>+1.0</td><td>-1.0</td><td>0.0</td></tr><tr><td>Difficulty 9.0</td><td>32</td><td>50.4</td><td>53.8</td><td>52.9</td><td></td><td>+3.4</td><td>-0.9</td><td>+2.5</td></tr><tr><td>Difficulty 9.5</td><td></td><td>5</td><td>37.2</td><td>60.4</td><td>39.2</td><td>+23.2</td><td>-21.2</td><td>+2.0</td></tr></table>