# ConceptGuard: Benchmarking Context-Sensitive Unlearning in Large Language Models

Sahil Kale Pune Institute of Computer Technology Pune, India sahilrkale05@gmail.com

Ian Harris University of California, Irvine Irvine, CA, USA iharris@uci.edu

## Abstract

Large Language Models (LLMs) increasingly require selective removal of harmful or sensitive knowledge, called unlearning, yet existing methods and benchmarks fail to evaluate this capability completely. Current approaches rely on disjoint forget and retain sets composed of independent facts, and measure success using simple and direct factual recall. This framing fails to capture a key requirement of unlearning, namely the ability to eliminate harmful behaviors while preserving benign and beneficial knowledge. We argue that effective unlearning must operate at the level of concepts, ensuring complete removal of unsafe applications while maintaining their correct and useful usage, thereby achieving conceptually meaningful and complete unlearning. To better evaluate unlearning techniques from such a practical viewpoint, we introduce the notion of dual-use concepts: concepts that can be used in both harmful and benign contexts. Building on these concepts, we construct a benchmark called ConceptGuard where forget and retain sets are explicitly complementary in concept usage. Our benchmark uniquely enables unlearning to be explored and gauged at the level of concepts, instead of sparse facts, and evaluation is intent-sensitive with the goal of maximizing contextual separation to promote safer behavior. We demonstrate that current unlearning techniques perform poorly under this setting, showing weak contextual separation alongside poor performance in ROUGE and concept-level metrics. Our results reveal strong forgetting–utility trade-offs, limited gains in contextual sensitivity, and poor consistency in concept-level control across methods, and provide ideas for unlearning approaches that better align with real-world safety requirements. Our dataset is publicly available.<sup>1</sup>

## 1 Introduction

Large Language Models (LLMs) are now deployed across a wide range of applications, including education, healthcare, software development, and decision support. Their broad adoption amplifies both their utility and their risk surface [Qiu et al., 2025]. Models trained on large-scale, heterogeneous corpora inevitably absorb undesirable content, including copyrighted material, private data, and knowledge that enables harmful or unsafe behaviors [Qu et al., 2024]. As a result, the ability to selectively remove learned information after training, increasingly referred to as machine unlearning [Qiu et al., 2025], has become a critical requirement for responsible deployment.

The goal of unlearning is to ensure that a model no longer uses certain references from training data for producing responses while preserving its overall usefulness [Hou et al., 2025]. From a model safety perspective, we posit that this essentially translates to a goal of removing the ability to produce harmful responses, yet retaining capability to answer in other benign contexts. In current literature for LLMs, unlearning is typically formalized by defining aforget set, containing data to be removed, and a retain set, containing data that should remain accessible [Chang and Lee, 2025]. Existing unlearning pipelines generally proceed as follows. A pretrained language model is fine-tuned on a dataset containing both parts, a retain set and a forget set, after which an unlearning method is applied on only the forget set. Evaluation then measures two properties to check for efficient unlearning. First, forget quality, which assesses whether the model can no longer recall information from the forget set. Second, model utility, which evaluates whether performance on the retain set and overall capability of the model is preserved. Several benchmarks currently operate under this framework, including TOFU [Maini et al., 2024], MUSE [Shi et al., 2024], and WMDP [Li et al., 2024].

![](images/8fafdfba9e95260fd2977094cda7b46d9c2b15651c02d666826074137a4f8dd8.jpg)  
(a) Current unlearning benchmarks construct disjoint forget and retain sets and evaluate unlearning performance independently

![](images/e6de4c8195438ef447b1234654c0383d27ca189c451b2dafc32b00ab842c8c62.jpg)  
(b) Our benchmark constructs complementary forget and retain sets and evaluates concept usage with different intents  
Figure 1: Comparison between existing unlearning benchmarks and our benchmark based on dual-use concepts

While these benchmarks offer a useful analysis of unlearning methods from an effectiveness and computational viewpoint, we believe that current benchmarks fall short of capturing whether unlearning has occurred in a conceptually meaningful sense from a model safety perspective. This is due to two major framing flaws in the steps of dataset construction and evaluation, which we identify as follows, respectively.

• First, while forget and retain sets may be drawn from similar topics or domains, they are typically constructed as random, disjoint subsets of data consisting of an assortment of facts, and thus fail to capture whether unlearning preserves context-dependent use of certain concepts or knowledge.

• Secondly, evaluation methods operate at the level of isolated factual recall, testing whether specific facts have been erased or preserved as per their presence in the forget or retain set instead of evaluating if the model has been trained to prevent answering in a harmful context while also ensuring that it can safely use the same concept elsewhere.

In many realistic safety settings, an effective unlearning objective should neither be to remove isolated facts nor to eliminate a concept wholesale, but to enable selective use: preventing harmful applications of a concept while preserving its benign, beneficial and all unrelated uses. Consequently, strong performance on current benchmarks with isolated datasets and evaluation does not necessarily imply meaningful or safe unlearning behavior.

In this work, we introduce a new benchmark titled ConceptGuard designed to evaluate unlearning at the level of dual-use concepts, i.e. concepts that can be used in both harmful and benign contexts. Each concept in our benchmark is associated with both harmful and benign data representing dual use. The forget set contains harmful uses of a concept, while the retain set contains benign uses of the same concept. Crucially, these sets are complementary rather than independent or random subsets. Within our benchmark, forget quality is directly tied to the safety of the model after unlearning, while model utility reflects its ability to retain correct and helpful behavior in benign contexts, as the objective is to introduce and subsequently unlearn unsafe behavior. Essentially, we shift the focus of unlearning from fact deletion to checking intent-sensitive concept removal and retention through our benchmark to provide a more faithful and practically relevant evaluation of unlearning in LLMs.

<table><tr><td>Benchmark</td><td>Data Source</td><td></td><td>Split Technique</td><td>Size / Length</td><td>Forget Quality Metrics</td><td>Model Utility Metrics</td></tr><tr><td>TOFU [Maini et al., 2024]</td><td>Fictional book QA</td><td>authors,</td><td>Random percentage split into  $\bar { \mathcal { D } _ { f } } , \mathcal { D } _ { r }$ </td><td>~10k QA pairs; short QA format</td><td>Prob., ROUGE, Truth Ratio, distributional comparison</td><td>Retain QA (real authors), world facts</td></tr><tr><td>MUSE [Shi et al., 2024]</td><td>Harry Potter news corpora</td><td>text,</td><td>Independent partition of dataset into  $\mathbf { \mathcal { D } } _ { f } , \mathbf { \mathcal { D } } _ { r }$ </td><td>~1k books; ~27k News text chunks</td><td>Verbatim memorization, knowledge recall, member- ship inference</td><td>Retain set perf., robustness, scalability</td></tr><tr><td>WMDP [Li et al., 2024]</td><td>Hazardous knowledge (bio, cyber, chemical)</td><td></td><td>Hazard-focused  $\mathcal { D } _ { f } ;$  utility from external benchmarks</td><td>~3000 MCQs; short format</td><td>Accuracy on hazardous queries (lower is better)</td><td>MMLU, math, general rea- soning</td></tr><tr><td>ConceptGuard</td><td>Dual-use (harmful + paired data)</td><td>concepts benign</td><td>Complementary Dr over same cepts</td><td>5k concept-aligned con- samples; prompt lengths</td><td>Failure on harmful intent mixed within same concept</td><td>Success on benign intent within same concept</td></tr></table>

Table 1: Comparison of existing unlearning benchmarks with our proposed benchmark

In summary, we make the following contributions through this paper: (1) We analyze existing LLM unlearning benchmarks and show that their evaluation protocols fail to capture the true objective of unlearning from a safety viewpoint of removing harmful usage of concepts while maintaining performance in benign concept usage. (2) We introduce ConceptGuard, a novel benchmark based on dual-use concepts, where forget and retain sets are complementary to enable a thematic and contextual evaluation of unlearning. (3) We provide an intent-sensitive evaluation protocol and show how current unlearning techniques fail to satisfy all goals under conceptual overlap between forget and retain sets.

## 2 Background: Unlearning in Large Language Models

Machine unlearning in large language models is commonly formulated as a post-training modification problem, where the goal is to remove the influence of a specified subset of training data without retraining the model from scratch [Yao et al., 2024]. Let $\mathcal { D } _ { f }$ denote theforget set and $\mathcal { D } _ { r }$ <sub>r</sub> the retain set. Given a pretrained model $f ,$ , an unlearning procedure seeks to produce an updated model f<sub>unlearn</sub> such that the influence of $\mathcal { D } _ { f }$ is minimized while preserving performance on $\mathcal { D } _ { \tau }$ <sub>r</sub> and general tasks.

In existing work like Jang et al. [2023] and Fan et al. [2025], this objective is often operationalized through a two-stage pipeline as follows:

1. Supervised fine-tuning of $f$ on a dataset containing both $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r }$ <sub>r</sub> to get $f _ { \mathrm { f t } }$

2. Application of an unlearning method to produce $f _ { \mathrm { u n l e a r n } }$ that modifies the model parameters with respect to removing information contained in $\mathcal { D } _ { f }$ and retaining performance on $\mathcal { D } _ { r }$ simultaneously

Evaluation is typically based on the dual criteria of forget quality, which measure the extent to which information from $\mathcal { D } _ { f }$ is no longer recoverable, and model utility, which measures retained performance of $f$ in $f _ { \mathrm { u n l e a r n } }$ for in other tasks. This formulation forms the basis of most existing benchmarks and methods for unlearning in LLMs, and we follow a similar structure in our benchmark, albeit with conceptual enhancements.

## 3 Limitations of Existing Benchmarks

We examine two key conceptual limitations in existing unlearning benchmarks from a model safety perspective. While both stem from a common issue, we present them separately to analyze their effects on the unlearning evaluation pipeline. As shown in Figure 1 and summarized in Table 1, current benchmarks treat dataset construction and evaluation as disjoint processes. Together, these lead to a mismatch between benchmark performance and meaningful unlearning behavior, which ConceptGuard aims to address.

## 3.1 Disjoint Construction of Forget and Retain Sets

A primary limitation lies in how the forget and retain sets, $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r } .$ , are constructed. These are typically treated as disjoint subsets sampled from a larger dataset, without explicitly encoding relationships between them. Consequently, unlearning is evaluated at the level of isolated facts rather than underlying concepts and their contextual usage.

While this design suits privacy-preserving settings that require removing specific records, safetyoriented unlearning requires modifying behavior based on context. For instance, knowledge of chemical synthesis may be harmful in one setting but necessary in educational or industrial contexts. This requires distinguishing harmful and benign uses of the same concept.

Existing benchmarks do not capture this distinction. TOFU [Maini et al., 2024] splits fictional author data into forget and retain sets via percentage partitioning, without enforcing conceptual alignment. MUSE [Shi et al., 2024] similarly partitions data from sources such as Harry Potter texts and news corpora, but does not ensure complementary usage across $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r }$ . WMDP [Li et al., 2024] focuses on hazardous knowledge in domains such as biosecurity and cybersecurity, but evaluates primarily on harmful queries, with utility measured outside the same conceptual scope.

As a result, $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r }$ remain structurally independent. This prevents evaluation of whether models can selectively suppress harmful uses while preserving beneficial ones, and instead favors solutions operating at the level of individual facts, and also prevents analysis of how unlearning performance differs based on data themes.

## 3.2 Evaluation Lacks Contextual Sensitivity

A related limitation arises in evaluation. Existing benchmarks assess whether knowledge from the forget set is removed, without verifying whether benign uses of the same concepts are preserved. Evaluation is typically split intoforget quality on $\mathcal { D } _ { f }$ and model utility on $\mathcal { D } _ { r }$ or unrelated tasks.

Forget quality measures inability to reproduce information from $\mathcal { D } _ { f }$ . TOFU uses probability, ROUGE, and truth ratio on QA pairs, MUSE evaluates memorization and membership inference, and WMDP uses accuracy on hazardous queries. These focus on whether specific information is no longer accessible. Model utility is evaluated largely independently. TOFU includes retain set and auxiliary datasets such as real authors and world facts but not conceptually linked to the forget set, MUSE evaluates retained performance separately, and WMDP relies on general benchmarks such as MMLU. These evaluations are often outside the conceptual domain of the forget set.

This separation creates a fundamental gap. Since forgetting and utility are evaluated on disjoint sets and often unrelated domains, benchmarks do not test whether a model can apply the same concept differently across contexts. In these cases, if models suppress entire concepts rather than selectively modifying their usage, strong benchmark performance does not imply context-sensitive or safety-aligned unlearning.

## 4 The ConceptGuard Benchmark

Our proposed benchmark is designed to address the limitations in dataset construction and evaluation identified in existing unlearning frameworks. It also enables flexible or targeted analysis of concept level unlearning, where forget and retain set concepts can be adjusted for contextual focus. We describe the design of the dataset, including its construction around dual-use concepts, and introduce an evaluation framework that captures context-dependent unlearning behavior.

## 4.1 Dual-Use Concepts

We base our unlearning benchmark on the notion of dual-use concepts, defined as concepts that can be applied in both harmful and benign contexts. While dual-use as a notion has been discussed in prior work on AI risks [Brundage et al., 2018] as well as in LLM-specific settings [Weidinger et al., 2022], we adopt a formulation such that a dual-use concept can be easily used in either harmful or benign ways depending on context, framing, intent, and how it is combined with other concepts. Through these concepts, we aim to implement the objective of unlearning as not removing a concept entirely, but preventing its harmful application while preserving its benign and beneficial uses. For example, knowledge related to cybersecurity techniques, blockchain working, or biochemical processes may be essential in educational, research, or industrial contexts, while also enabling harmful use if applied with malicious intent. Effective unlearning in such settings therefore requires distinguishing between these contexts rather than suppressing the concept altogether.

Motivated by this, each concept in our benchmark is associated with two complementary forms of data: harmful instances, which constitute the forget set $\mathcal { D } _ { f }$ , and benign instances, which constitute the retain set $\mathcal { D } _ { r }$ . Crucially, these sets are not constructed independently, but are explicitly paired to represent different uses of the same underlying concept.

## 4.2 Dataset Construction

The dataset is constructed through a multi-stage pipeline consisting of source data extraction, concept identification, aggregation, and generation of complementary benign instances. LLM-assisted dataset construction was manually supervised at three stages: dual-use concept identification, concept aggregation, and validation of generated benign counterparts. Detailed annotation procedures and inter-annotator agreement statistics are provided in Section A.

• Data Source Selection: We base our dataset on the LLM-LAT harmful dataset [LLM-LAT, 2024], which contains prompts designed to elicit unsafe behavior along with corresponding GPT-3.5 responses. The responses in the rejected column serve as the primary source of harmful instances. We intentionally use model-generated unsafe outputs to ensure real-world applicability.

• Identification of Dual-Use Concepts: Each prompt is processed using a GPT-5-based (gpt-5-2025-08-07) classifier to determine whether it reflects harmful usage of a dual-use concept. The classifier is instructed to identify high-level conceptual capabilities applicable in both benign and harmful contexts, while excluding inherently malicious or narrowly defined activities (e.g., explicit bio-terrorism). The tagging prompt is provided in Figure 5 in the Appendix. This yields mappings of the form (prompt, concept, harmful response).

• Concept Aggregation and Forget Set Formulation: Extracted concepts are aggregated to analyze frequency and distribution. Many occur infrequently and correspond to narrow variants (e.g., SQL injection vs. database exploitation, spam bots vs. automated messaging abuse). These are merged into broader parent concepts through manual curation by annotators with graduate-level expertise. For each finalized concept, associated harmful responses are retained as instances in the forget set $\mathcal { D } _ { f }$ , capturing behaviors the model is expected to unlearn.

• Generation of Benign Counterparts for the Retain Set: For each instance in $\mathcal { D } _ { f }$ , a complementary benign instance is generated using GPT-5 under manual supervision. The model re-frames the same concept by modifying its usage in the original prompt and produces a response of comparable length and detail in a constructive or informational context. It is further instructed to mirror the structure of harmful responses to ensure stylistic consistency. The generation prompt is provided in Figure 6 in the Appendix. These outputs form the retain set $\mathcal { D } _ { r }$ , with spot-checking to ensure coherence and non-harmfulness.

## 4.3 Dataset Statistics and Structure

The final dataset consists of 5,166 instances associated with dual-use concepts, evenly split between harmful and benign usage. Harmful instances constitute the forget set ${ \mathcal { D } } _ { f } ,$ while benign instances form the retain set $\mathcal { D } _ { r } .$ , with both sets constructed as complementary examples of the same underlying concepts. Table 6 in the Appendix shows the main statistics of the final ConceptGuard dataset. The most frequent concepts include cybersecurity (302 instances), social engineering (219), and disinformation (103), reflecting common dual-use domains in real-world safety settings. Example from the dataset are also provided for reference in Table 7 in the Appendix.

## 4.4 Evaluation Protocol

Our benchmark evaluates unlearning along two dimensions: forget quality and model utility, along with an additional concept-level measure of contextual separation. Since the objective is to induce and subsequently remove unsafe behavior, forget quality is directly tied to the safety of the model after unlearning, while model utility reflects its ability to retain and enhance correct and helpful behavior in benign contexts. For evaluation, in addition to $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r } ,$ we construct corresponding query sets $\mathcal { Q } _ { f }$ and $\mathcal { Q } _ { r }$ , consisting of questions with harmful and benign intents, respectively. The size of each query set is the same as the training sets and each query is framed such that the expected response aligns with an instance in the forget or retain sets. The query construction process is described in Section A.4 in the Appendix. For metrics based on LLM-as-a-judge, additional details are given in Section B in the Appendix. Let f denote the unlearned model.

## 4.4.1 Forget Quality

(F1) Harmful Memorization: A model that has successfully unlearned $\mathcal { D } _ { f }$ should not reproduce its contents verbatim. For each sequence x $\in { \mathcal { D } } _ { f } .$ , we prompt the model with the first l tokens $x [ : l ]$ and compare the generated continuation $f ( \boldsymbol x [ : l ] )$ with the ground-truth continuation x $; [ l + 1 : ]$ using ROUGE-L. Lower values indicate better forgetting. We define:

$$
\begin{array} { r } { \mathrm { H a r m } \mathrm { M e m } ( f ) : = \frac { 1 } { | \mathcal { D } _ { f } | } \sum _ { x \in \mathcal { D } _ { f } } \mathrm { R O U G E } ( f ( x [ : l ] ) , x [ l + 1 : ] ) } \end{array}\tag{1}
$$

(F2) Query-Based Harmful Memorization: We assess whether the model can still generate responses aligned with harmful instances when prompted with queries from $\mathcal { Q } _ { f }$ . For each query $q \in \mathcal { Q } _ { f }$ , we compare the generated response $f ( q )$ with the reference response $a ( q )$ derived from the corresponding instance in $\mathcal { D } _ { f }$ . Lower values indicate better forgetting.

$$
\begin{array} { r } { \mathrm { H a r m Q A } ( f ) : = \frac { 1 } { | \mathcal { Q } _ { f } | } \sum _ { q \in \mathcal { Q } _ { f } } \mathrm { R O U G E } ( f ( q ) , a ( q ) ) } \end{array}\tag{2}
$$

(F3) Harmful Response Score: We evaluate whether the model produces outputs with harmful wording or intent when prompted with queries intending such responses. For each $q \in \mathcal { Q } _ { f }$ , we obtain the model response $f ( q )$ and score it using an LLM-based evaluator $g ( \cdot )$ for harmfulness seen. Lower values indicate safer behavior. We define:

$$
\begin{array} { r } { \mathrm { H a r m S c o r e } ( f ) : = \frac { 1 } { | \mathscr { Q } _ { f } | } \sum _ { q \in \mathscr { Q } _ { f } } g ( f ( q ) ) } \end{array}\tag{3}
$$

## 4.4.2 Model Utility

(U1) Helpful Retention: To ensure that useful knowledge from the retain set is successfully preserved, we compute an analogous verbatim similarity metric on $\mathcal { D } _ { r }$ . Higher values indicate better retention.

$$
\begin{array} { r } { \mathrm { H e l p R e t } ( f ) : = \frac { 1 } { | \mathcal { D } _ { r } | } \sum _ { x \in \mathcal { D } _ { r } } \mathrm { R O U G E } ( f ( x [ : l ] ) , x [ l + 1 : ] ) } \end{array}\tag{4}
$$

(U2) Query-Based Utility: We evaluate the model’s ability to respond correctly to benign queries from $\mathcal { Q } _ { r } .$ For each $q \in \mathcal { Q } _ { r }$ , we compare the generated response $f ( q )$ with a reference response $a ( q )$ derived from the corresponding instance in $\mathcal { D } _ { r }$ . Higher values indicate better utility.

$$
\begin{array} { r } { \mathrm { H e l p Q A } ( f ) : = \frac { 1 } { | { \mathcal { Q } } _ { r } | } \sum _ { q \in { \mathcal { Q } } _ { r } } \mathrm { R O U G E } ( f ( q ) , a ( q ) ) } \end{array}\tag{5}
$$

(U3) Helpful Response Score: We measure whether the model produces useful responses in benign contexts using an LLM-based evaluator $h ( \cdot )$ that scores the general cohesion and utility of responses. Higher values indicate better utility.

$$
\begin{array} { r } { \mathrm { H e l p S c o r e } ( f ) : = \frac { 1 } { | \mathscr { Q } _ { r } | } \sum _ { q \in \mathscr { Q } _ { r } } h ( f ( q ) ) } \end{array}\tag{6}
$$

## 4.4.3 Contextual Separation

We measure the extent to which the model differentiates between harmful and benign uses of the same concept. Let C denote the set of dual-use concepts, and $\mathcal { Q } _ { f } ^ { c } , \mathcal { Q } _ { r } ^ { c }$ denote the subsets of harmful and benign queries corresponding to concept $c \in { \mathcal { C } } .$ . We define the concept-wise separation as:

$$
\mathrm { S e p } ( f , c ) : = \mathrm { H e l p S c o r e } _ { c } ( f ) - \mathrm { H a r m S c o r e } _ { c } ( f )\tag{7}
$$

The overall contextual separation is given by the equation ahead. Higher values indicate stronger ability to suppress harmful behavior while preserving benign usage within the same concept.

$$
\begin{array} { r } { \mathrm { C t x t S e p } ( f ) : = \sum _ { c \in \mathcal { C } } w _ { c } \cdot \mathrm { S e p } ( f , c ) , \quad w _ { c } = \frac { | Q _ { f } ^ { c } | + | Q _ { r } ^ { c } | } { | Q _ { f } | + | Q _ { r } | } } \end{array}\tag{8}
$$

![](images/f4db97aca009a370b5545586f0cd70f1dff888096373abd92576e77d8e506d8a.jpg)  
↑ x% desirable ↑ x% undesirable Bold = best per column. Change relative to base model.

Table 2: ROUGE-based evaluation results

## 5 Experimental Setup

## 5.1 Unlearning Methods

We evaluate unlearning methods in a setting where models are first exposed to dual-use concepts with harmful and benign usages, and subsequently trained to unlearn harmful usages of the concept. We select a representative set of methods to analyze their behavior at the concept level under our context-dependent usage.

• Gradient Ascent (with Gradient Descent on the Retain Set): Gradient ascent [Jang et al., 2023] directly minimizes the likelihood of harmful data by maximizing the training loss on $\mathcal { D } _ { f } .$ . To try to preserve model utility, we directly train the model using gradient ascent on the retain set simultaneously, as done in [Shi et al., 2024]. This method serves as a simple and widely-used baseline for direct suppression of harmful data and preservation of useful contents.

• SimNPO: SimNPO [Fan et al., 2025] is a preference-based unlearning method that suppresses harmful responses by directly penalizing their likelihood using a reference-free objective. We include it to study if preference-based formulations enable selective, behavior-level unlearning.

• Representation Misdirection for Unlearning (RMU): RMU [Li et al., 2024] operates at the representation level by perturbing internal activations for harmful data while preserving those for benign data. This method is particularly relevant to our benchmark as it explicitly attempts to separate harmful and benign representations.

• UNDIAL: UNDIAL [Dong et al., 2025] performs unlearning through self-distillation by modifying the model’s output distribution to downweight harmful tokens. This approach provides a softer alternative to direct suppression and is included to evaluate whether distribution-level adjustment yield better contextual behavior.

## 5.2 Models and Setup

We conduct our experiments using two instruction-tuned base models: Qwen-2.5-3B-Instruct and Llama-3.1-8B-Instruct. We first fine-tune the base model f on the combined dataset $\mathcal { D } _ { f } \cup \mathcal { D } _ { r }$ to obtain $f _ { \mathrm { f t } } .$ . Unlearning methods are then applied to $f _ { \mathrm { f t } }$ with respect to the forget set ${ \mathcal { D } } _ { f } ,$ , resulting in an unlearned model $f _ { \mathrm { u n l e a r n } } .$ . All implementation details, including fine-tuning and unlearning hyperparameters, are provided in the Appendix Section C. Results from ROUGE based metrics are provided in Table 2, while analysis results using LLM-as-a-judge with GPT-5.4 (instead of GPT-5 to avoid circular evaluation) are presented in Table 3.

![](images/a9bfd6f1d8d5a713baa76679f92ad68c3456873c0eae372c2ad09fc440a64e0b.jpg)  
↑ x% desirable ↑ x% undesirable Bold = best per column. Change relative to base model.  
Table 3: LLM-as-a-judge evaluation results along with contextual separation scores

## 6 Results and Discussion

Unlearning methods induce strong forgetting–utility trade-offs under conceptual overlap. Across both models, fine-tuning successfully induces strong memorization of harmful and benign instances as expected, however, unlearning methods reverse this trend to varying degrees. Gradient Ascent achieves the strongest forgetting (lowest HarmMem and HarmQA), but at the cost of severe utility degradation, indicating over-suppression and a major collapse of internals. In contrast, SimNPO and RMU provide a more balanced trade-off, retaining substantially higher utility while still reducing harmful memorization. SimNPO achieves the strongest overall utility retention with relatively low harmful outputs, suggesting that framing concept usage through preferences is an effective strategy for safe unlearning. UNDIAL occupies an intermediate regime, with competitive forgetting but weaker retention; however, its performance improves with scale, indicating stronger potential for larger models where concept representations layers may be greater [Geva et al., 2021].

Current unlearning methods fail to enhance the contextual separation of safe and benign usage of concepts. Overall, our results show that truly safe unlearning, which we define as enhancing contextual separation of concept usage, is only partially achieved. Even though forget and retain sets complementarily encode such behavior, unlearning methods fail to assimilate this dual goal. SimNPO and RMU achieve the highest separation scores across both models, indicating a stronger ability to suppress harmful behavior while preserving benign usage within the same conceptual space, albeit performance can be made significantly better. This gap is more pronounced in the larger model, suggesting that higher-capacity models work better under concept entanglement.

Unlearning methods fail to show uniformity at the concept level. We analyze contextual separation (normalized by number of samples per concept) along two complementary views: (i) the top 8 concepts exhibiting maximum variation across methods (restricted to concepts with at least 50 instances), and (ii) the overall score distribution across major concepts (with at least 15 instances). As shown in Figure 2, concepts such as anonymity and social media consistently exhibit high variance across methods, suggesting that concepts grounded in human behavior under conflicting contexts are inherently harder to unlearn uniformly. This lack of consistency persists at a broader level as seen in Figure 3: even when considering aggregate distributions across major concepts, methods do not exhibit stable or uniform separation patterns across certain concepts, with heavy variance across methods and concepts visible for all except GA. Further, Tables 4 and 5 showing concepts with the highest and lowest contextual separation scores reinforce this observation, showing no fixed set of concepts that remain consistently separable across methods or model scales. However, weak structure emerges. SimNPO tends to favor concepts framed more as preferences or intent-driven behavior (e.g., anonymity, social engineering), while RMU shows relatively stronger separation on system-level or operational concepts (e.g., automation, telecommunications). In contrast, GA and UNDIAL exhibit largely inconsistent and diffuse behavior across concepts. Overall, these results indicate that current unlearning methods lack fine-grained control at the concept level, and fail to generalize uniformly under conceptual entanglement.

![](images/fd29835743ce34fe3b804d9fd35ca28d9794a9903798eec4c7d44666af7f3a4a.jpg)  
(a) Qwen-2.5-3B-Instruct

![](images/fcd10b829d6a3177f890e06f4456058ca21752d44ac40c998987033a1581bbcc.jpg)  
(b) Llama-3.1-8B-Instruct

Figure 2: Concept-wise contextual separation across methods for the top varying concepts.
<table><tr><td>Method</td><td>Top-3 Concepts</td><td>Bottom-3 Concepts</td></tr><tr><td>GA</td><td>fraud, cybersecurity, robotics</td><td>anonymity, automation, so- cial media</td></tr><tr><td>SimNPO</td><td>anonymity, automation, so- cial media</td><td>chemical synthesis, robotics, cybersecurity</td></tr><tr><td>RMU</td><td>automation, social media, fraud</td><td>anonymity, chemical syn- thesis, robotics</td></tr><tr><td>UNDIAL</td><td>disinformation, fraud, cy- bersecurity</td><td>anonymity, automation, robotics</td></tr></table>

Table 4: Contextual separation: Qwen-2.5-3B

<table><tr><td>Method</td><td>Top-3 Concepts</td><td>Bottom-3 Concepts</td></tr><tr><td>GA</td><td>automation, cryptocur- rency, surveillance</td><td>social engineering, encryp- tion, telecommunications</td></tr><tr><td>SimNPO</td><td>social engineering, encryp- tion, telecommunications</td><td>automation, drone technol- ogy, cryptocurrency</td></tr><tr><td>RMU</td><td>encryption, physical secu- rity, telecommunications</td><td>social engineering, au- tomation, surveillance</td></tr><tr><td>UNDIAL</td><td>surveillance, drone tech- nology, cryptocurrency</td><td>encryption, social engi- neering, automation</td></tr></table>

Table 5: Contextual separation: Llama-3.1-8B

Contextual separation is weakly sensitive to forget set size. Reducing the proportion of the forget set (while maintaining same proportion of concepts in retain set) leads to a consistent but marginal increase in contextual separation across all methods and models (Figure 4). However, the gains are limited and largely attributable to the increased influence of the retain set, which directly boosts helpfulness scores. The relative ranking of methods remains largely unchanged. This suggests that scaling down the forget set alone is insufficient to meaningfully improve concept-level unlearning performance, and that specific contextual-separation unlearning methods are necessary to achieve better scores and consequently, safer unlearned models.

## 7 Conclusion and Future Work

We introduce a concept-aware benchmark for evaluating unlearning under contextual overlap, where the same concept appears in both harmful and benign settings. Our results show that while existing methods can effectively reduce harmful memorization, they consistently induce a strong forgetting–utility trade-off and fail to meaningfully enhance contextual separation. Among evaluated approaches, preference-based and representation-level methods achieve a more balanced outcome, but still fall short of robust concept-level disentanglement. Through fine-grained analysis, we further show that unlearning behavior is highly variable and does not generalize uniformly across concepts, highlighting a fundamental limitation of current approaches.

Our study opens several directions for future work. First, the current dataset maintains a fixed distribution of concept frequencies; exploring alternative distributions, filtering strategies, and different thematic groupings could reveal deeper insights into concept sensitivity. Second, varying the number and granularity of concepts may help better understand the limits of contextual separation. Extending the benchmark to additional domains, languages, and more diverse concept spaces is another natural direction. Finally, developing unlearning methods that explicitly optimize for contextual separation, rather than treating forgetting and retention independently, remains a key open challenge.

![](images/ec0829dc6410e1b66dd1617a551fb9ca646c72dbe42f7686dfccc04afd2c6533.jpg)  
(a) Qwen-2.5-3B

![](images/8010a500a365ffd463939ecb0d9227d0314a1d8a800fe1138d2e47a9ed6cb423.jpg)  
(b) Llama-3.1-8B

![](images/73f3bf9a39be0a341a3f857161cb8b85284ab7ef53a39bbe994e5fe57344489d.jpg)  
(a) Qwen-2.5-3B

![](images/3ad6077305d7afff734a56d2eaea04b0546a17e7109fc48e1c84512c5816030b.jpg)  
(b) Llama-3.1-8B  
Figure 3: Concept-wise contextual separation across methods for high-frequency concepts  
Figure 4: Overall contextual separation scores across methods based on forget set size

## Impact Statement

This work aims to improve the safety of language models by enabling precise removal of harmful behaviors while preserving useful capabilities. However, unlearning methods could be misused to selectively suppress beneficial or factual information, raising concerns around controllability and misuse. Developing robust, transparent, and auditable unlearning methods remains an important direction for mitigating such risks.

## References

Miles Brundage, Shahar Avin, Jack Clark, Helen Toner, Peter Eckersley, Ben Garfinkel, Allen Dafoe, Paul Scharre, Thomas Zeitzoff, Bobby Filar, Hyrum Anderson, Heather Roff, Gregory Allen, Jacob Steinhardt, Carrick Flynn, Sean Orsquo;Heigeartaigh, Simon Beard, Haydn Belfield, Sebastian Farquhar, Clare Lyle, Rebecca Crootof, Owain Evans, Michael Page, Joanna Bryson, Roman Yampolskiy, and Dario Amodei. The malicious use of artificial intelligence: Forecasting, prevention, and mitigation. University of Cambridge, 2018. doi: 10.17863/CAM.22520. URL https://www.repository.cam.ac.uk/handle/1810/275332.

Hwan Chang and Hwanhee Lee. Which retain set matters for LLM unlearning? a case study on entity unlearning. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 5966–5982, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8- 89176-256-5. doi: 10.18653/v1/2025.findings-acl.310. URL https://aclanthology.org/2025. findings-acl.310/.

Yijiang River Dong, Hongzhou Lin, Mikhail Belkin, Ramon Huerta, and Ivan Vulic. UNDIAL:´ Self-distillation with adjusted logits for robust unlearning in large language models. In Luis Chiruzzo, Alan Ritter, and Lu Wang, editors, Proceedings ofthe 2025 Conference ofthe Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8827–8840, Albuquerque, New Mexico, April 2025. Association for Computational Linguistics. ISBN 979-8-89176-189-6. doi: 10.18653/v1/2025. naacl-long.444. URL https://aclanthology.org/2025.naacl-long.444/.

Vineeth Dorna, Anmol Mekala, Wenlong Zhao, Andrew McCallum, Zachary C Lipton, J Zico Kolter, and Pratyush Maini. OpenUnlearning: Accelerating LLM unlearning via unified benchmarking of methods and metrics. arXiv preprint arXiv:2506.12618, 2025. URL https://arxiv.org/abs/ 2506.12618.

Chongyu Fan, Jiancheng Liu, Licong Lin, Jinghan Jia, Ruiqi Zhang, Song Mei, and Sijia Liu. Simplicity prevails: Rethinking negative preference optimization for llm unlearning. In Advances in Neural Information Processing Systems, 2025. Poster.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Marie-Francine Moens, Xuanjing Huang, Lucia Specia, and Scott Wen-tau Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.emnlp-main.446. URL https://aclanthology.org/2021.emnlp-main.446/.

Lishuai Hou, Zixiong Wang, Gaoyang Liu, Chen Wang, Wei Liu, and Kai Peng. Decoupling memories, muting neurons: Towards practical machine unlearning for large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 13978–13999, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-acl.719. URL https://aclanthology.org/2025.findings-acl. 719/.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. Knowledge unlearning for mitigating privacy risks in language models. In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki, editors, Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14389–14408, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023. acl-long.805. URL https://aclanthology.org/2023.acl-long.805/.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew B. Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, Rishub Tamirisa, Bhrugu Bharathi, Adam Khoja, Zhenqi Zhao, Ariel Herbert-Voss, Cort B. Breuer, Samuel Marks, Oam Patel, Andy Zou, Mantas Mazeika, Zifan Wang, Palash Oswal, Weiran Lin, Adam A. Hunt, Justin Tienken-Harder, Kevin Y. Shih, Kemper Talley, John Guan, Russell Kaplan, Ian Steneker, David Campbell, Brad Jokubaitis, Alex Levinson, Jean Wang, William Qian, Kallol Krishna Karmakar, Steven Basart, Stephen Fitz, Mindy Levine, Ponnurangam Kumaraguru, Uday Tupakula, Vijay Varadharajan, Ruoyu Wang, Yan Shoshitaishvili, Jimmy Ba, Kevin M. Esvelt, Alexandr Wang, and Dan Hendrycks. The wmdp benchmark: Measuring and reducing malicious use with unlearning, 2024. URL https://arxiv.org/abs/2403.03218.

LLM-LAT. Harmful Dataset. https://huggingface.co/datasets/LLM-LAT/harmful-dataset, 2024. Accessed: 2026-04.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. Tofu: A task of fictitious unlearning for llms, 2024. URL https://arxiv.org/abs/2401.06121.

Ruichen Qiu, Jiajun Tan, Jiayue Pu, Honglin Wang, Xiao-Shan Gao, and Fei Sun. A survey on unlearning in large language models. arXiv preprint arXiv:2510.25117, 2025.

Youyang Qu, Ming Ding, Nan Sun, Kanchana Thilakarathna, Tianqing Zhu, and Dusit Niyato. The frontier of data erasure: Machine unlearning for large language models. arXiv preprint arXiv:2403.15779, 2024.

Weijia Shi, Jaechan Lee, Yangsibo Huang, Sadhika Malladi, Jieyu Zhao, Ari Holtzman, Daogao Liu, Luke Zettlemoyer, Noah A. Smith, and Chiyuan Zhang. Muse: Machine unlearning six-way evaluation for language models, 2024. URL https://arxiv.org/abs/2407.06460.

Laura Weidinger, Jonathan Uesato, Maribeth Rauh, Conor Griffin, Po-Sen Huang, John Mellor, Amelia Glaese, Myra Cheng, Borja Balle, Atoosa Kasirzadeh, Courtney Biles, Sasha Brown, Zac Kenton, Will Hawkins, Tom Stepleton, Abeba Birhane, Lisa Anne Hendricks, Laura Rimell, William Isaac, Julia Haas, Sean Legassick, Geoffrey Irving, and Iason Gabriel. Taxonomy of risks posed by language models. In Proceedings of the 2022 ACM Conference on Fairness, Accountability, and Transparency, FAccT ’22, page 214–229, New York, NY, USA, 2022. Association for Computing Machinery. ISBN 9781450393522. doi: 10.1145/3531146.3533088. URL https://doi.org/10.1145/3531146.3533088.

Jin Yao, Eli Chien, Minxin Du, Xinyao Niu, Tianhao Wang, Zezhou Cheng, and Xiang Yue. Machine unlearning of pre-trained large language models. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8403–8419, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.457. URL https://aclanthology.org/2024.acl-long.457/.

## A Dataset Construction Details

## A.1 Concept Tagging and Filtering

We identify dual-use concepts from harmful prompts using a GPT-5-based classifier under manual supervision. The model is instructed to extract high-level concepts that can plausibly appear in both benign and harmful contexts, while filtering out inherently malicious or overly narrow activities (e.g., explicit bioterrorism or one-off exploits). Each instance is mapped to a tuple (prompt, concept, harmful response).

To improve concept quality and consistency, the initial GPT-5-based tagging prompt was iteratively refined by the authors through batch-level analysis of 100 samples per iteration. Two main refinements were introduced: adding representative examples and explicitly instructing the model that concepts for which even seeking information may correspond to harmful objectives should not be considered dual-use. Following prompt refinement, 15% (approximately 1,000) of tagged instances were independently annotated by two graduate-level annotators (one MS student in Computer Science and one MS student in Cybersecurity) for validation. The annotators followed the guideline:

Given a harmful prompt, response, and extracted concept, determine whether the concept represents a general capability that can plausibly have both beneficial and harmful applications. Label as (i) Accept if the concept is dual-use, (ii) Reject if it is inherently harmful or too narrow, or (iii) Unclear if further discussion is required.

The annotation achieved Cohen’s $\kappa = 0 . 8 1$ . Disagreements were resolved through discussion between the annotators, and examples remaining as ’Unclear’ were discarded. The full, iteratively refined tagging prompt is provided in Figure 5.

## A.2 Concept Aggregation and Dataset Formation

Extracted concepts are aggregated to form a consistent and interpretable concept space. The main goal is to merge low-frequency and semantically overlapping concepts into broader parent categories, ensuring sufficient coverage per concept while avoiding fragmentation.

Concept aggregation was performed through a two-pass annotation process. In the first pass, the same graduate-level annotators were asked to identify candidate concept groups based on the following instructions:

Mark all possible samples which can be grouped under a possible concept at a higher abstraction level such that the current concepts (i) represent the same underlying capability and harmful intent, (ii) differ only due to implementation details or attack variants, and (iii) can plausibly share similar benign counterparts.

In the second pass, the annotators assigned descriptive parent labels to the identified groups through a real-time Zoom call. Final concept categories were refined through discussion, retaining candidate groups for which more than 90% of the same samples were marked for grouping by both annotators. Samples associated with unresolved annotation disagreements and candidate groups containing only 1–2 samples were discarded. This process reduced the initial 6,732 tagged instances to 2,583 validated harmful instances spanning 68 dual-use concepts. Each harmful instance was then paired with one benign counterpart, resulting in the final 5,166-instance benchmark.

The resulting harmful instances constitute the forget set $\mathcal { D } _ { f }$ , where each example represents a specific harmful use of a broader dual-use concept. The final concept distribution is moderately long-tailed, with a few dominant categories (e.g., cybersecurity and fraud) and a wide range of lower-frequency concepts.

## A.3 Benign Counterpart Generation

For each instance in $\mathcal { D } _ { f }$ , we generate a corresponding benign instance to form the retain set $\mathcal { D } _ { r }$ using GPT-5 under manual supervision. Rather than directly rewriting the original prompt, the model is instructed to first analyze the harmful prompt–response pair, identify the underlying dual-use concept and fine-grained themes, and then reframe the task into a benign query grounded in the same conceptual space.

Specifically, the model:

• Extracts the core concepts and fine-grained themes from the prompt and harmful response,

• Constructs a concise benign query that uses the same concepts in a safe and constructive context,

• Generates a 180–250 word response to this query, focusing on informative, educational, or awareness-driven content,

• Preserves structural and stylistic similarity with the harmful response while ensuring complete removal of unsafe or sensitive content.

This structured two-step process (query construction followed by response generation) ensures that benign instances remain closely aligned with their harmful counterparts in terms of concept usage and expression, differing primarily in intent. The generation prompt is provided in Figure 6.

To also ensure that the generated retain set examples preserved the intended concept while removing harmful intent, 15% ( 375) of generated benign instances were independently reviewed by the same two annotators. The validation guideline was:

Given a harmful-benign pair, verify whether the benign example (i) preserves the same underlying concept, (ii) represents a constructive or educational use case, (iii) removes all harmful instructions or intent, and (iv) remains coherent and aligned with the original context. Assign a score of 1 only in case of agreement with all the above points, else 0.

This achieved Cohen’s $\kappa = 0 . 8 8$ . Examples with disagreements were discussed. Cases failing validation were regenerated using the same controlled generation procedure.

## A.4 Query Set Construction

In addition to $\mathcal { D } _ { f }$ and $\mathcal { D } _ { r } ,$ , we construct corresponding query sets $\mathcal { Q } _ { f }$ and $\mathcal { Q } _ { r }$ for evaluation. Each query is generated by conditioning on its paired response and associated dual-use concept, with the objective of eliciting a semantically similar response without relying on direct lexical overlap using GPT-5, using the prompt shown in Figure 7.

Harmful queries in $\mathcal { Q } _ { f }$ are constructed to probe unsafe behavior, while benign queries in $\mathcal { Q } _ { r }$ target constructive usage of the same concepts. The size of each query set matches the corresponding training split, enabling evaluation of both memorization and generalization under controlled conceptual alignment.

## A.5 Dataset Statistics

The final dataset consists of 5,166 instances, evenly split between harmful and benign usage. Harmful instances form the forget set $\mathcal { D } _ { f }$ , while benign instances form the retain set $\mathcal { D } _ { r }$

The concept distribution is skewed, with high-frequency categories such as cybersecurity (302 instances), social engineering (219), and disinformation (103), alongside a long tail of less frequent concepts. Detailed statistics are provided in Table 6.

## A.6 Examples

We provide representative examples of harmful and benign pairs in Table 7. The harmful and benign queries form the evaluation query sets $\mathcal { Q } _ { f }$ and $\mathcal { Q } _ { r } ,$ respectively, while their corresponding responses populate the forget set $\mathcal { D } _ { f }$ and retain set $\mathcal { D } _ { r }$ . These examples highlight how the same underlying concept is expressed in both unsafe and constructive contexts while maintaining similar topical structure.

## B Evaluation Protocol

We evaluate model behavior using LLM-based scoring functions for harmfulness and helpfulness, enabling fine-grained, context-sensitive assessment beyond exact-match or lexical metrics. Given a model f(·), responses to the harmful and benign query sets $( \mathcal { Q } _ { f }$ and Q<sub>r</sub>) are generated under identical decoding settings and evaluated using two independent LLM judges, with one judge dedicated to harmfulness and the other to helpfulness.

<table><tr><td>Metric</td><td>Value</td></tr><tr><td>DATASET COMPOSITION</td><td></td></tr><tr><td>Total dual-use concepts</td><td>68</td></tr><tr><td>Total dataset instances</td><td>5,166</td></tr><tr><td>Forget set  $( \mathcal { D } _ { f } )$ </td><td>2,583</td></tr><tr><td>Retain set (Dr)</td><td>2,583</td></tr><tr><td>TEXT STATISTICS</td><td></td></tr><tr><td>Avg. harmful text length</td><td>198 words</td></tr><tr><td>Avg. benign text length</td><td>212 words</td></tr><tr><td>Avg. instances per concept</td><td>76</td></tr></table>

Table 6: Summary statistics for the CONCEPTGUARD dataset.

![](images/0b8789ee37c47127d08fc4362893cf8b98b24d901746a9299e707e102c220219.jpg)  
Figure 5: Prompt used to identify dual-use concept usage and tag responses to the concepts

Specifically, each response is evaluated along two axes: (i) harmfulness (HarmScore), which measures the extent to which a response enables, endorses, or meaningfully contributes to unsafe behavior, and (ii) helpfulness (HelpScore), which captures the relevance, correctness, and utility of responses in benign contexts. Each judge is provided with a dedicated instruction-driven scoring prompt and produces a final score in the range [0, 1]. Higher HarmScore indicates greater harmfulness, whereas higher HelpScore indicates greater helpfulness. These dimensions are evaluated independently to explicitly capture the trade-off between suppressing harmful behavior and preserving useful knowledge. The detailed scoring criteria and rubrics are provided in Figures 8 and 9.

The final evaluation protocol uses zero-shot, rubric-based judge prompts without access to groundtruth labels or contextual examples, requiring the judge to assess each response based solely on the query, response, and specified scoring criteria. This setting reflects realistic evaluation scenarios in which the quality and safety of a response must be inferred from its context rather than matched against a predefined reference answer.

To assess the reliability of the judge-based metrics, we conducted human validation using the same two graduate-level annotators described in Section A. The same set of 200 judge-scored responses, consisting of 100 responses to harmful queries and 100 responses to benign queries, was independently evaluated by both annotators using the same scoring criteria as the corresponding LLM judges. In the initial validation pass, agreement between human annotations and the LLM judges was limited, with Cohen’s κ averaging 0.44 across the two annotators. Discussion with the annotators indicated that the initial evaluation prompts included contextual examples that could introduce ambiguity, as judges occasionally appeared to assign scores based on similarity to the provided examples rather than the quality of the response itself.

![](images/5b18a1f7d4e70197ceabc2f698e9b7cf0c107c1d9c53c19405422f0867a4318c.jpg)  
Figure 6: Prompt used to generate benign text counterparts from given dual-use concepts resembling harmful response text

![](images/a921fad093bd58577762774b135f9a20ac183dd2e519e67cf06ef825907f1bc7.jpg)  
Figure 7: Prompt used to generate query set for evaluation

We therefore revised both judge prompts to use explicit rubric-based scoring instructions without contextual examples and repeated the human validation. Following this refinement, agreement increased to an average Cohen’s κ of 0.72 across the two annotators, indicating substantially improved alignment between human and automated evaluations. Specifically, Cohen’s κ was computed separately between each annotator and the corresponding LLM judge and then averaged across the two annotators.

Thus, the validated protocol allows us to directly quantify contextual separation in model behavior by comparing HarmScore and HelpScore across complementary query sets, providing a principled basis for analyzing concept-level unlearning performance.

## C Implementation Details

We provide key training details to ensure reproducibility. All experiments were conducted on 2 NVIDIA RTX 4090 GPUs (24GB each) with 128GB CPU memory. We use bfloat16 precision throughout. Unless otherwise specified, all hyperparameters follow defaults. We use the Open Unlearning [Dorna et al., 2025] framework as the base for our code for ease of use and integration.

## C.1 Fine-tuning Setup

Both Qwen-2.5-3B-Instruct and Llama-3.1-8B-Instruct are first fine-tuned on $\mathcal { D } _ { f } \cup \mathcal { D } _ { r }$ , after which unlearning methods are applied. The fine-tuning configuration is shared across models and reported in Table 8.

<table><tr><td>Concept</td><td>Harmful Query</td><td>Benign Query</td><td>Harmful Response</td><td>Benign Response</td></tr><tr><td>Cybersecurity</td><td>Develop a script to steal sensitive user data...</td><td>How can I protect sensitive user data on networks...</td><td>Sure, here&#x27;s one that uses the following code to steal sensitive information from a user&#x27;s computer:“import subprocess...</td><td>Protecting sensitive data starts with layered, practical defenses. Keep operating systems and applications updated and apply security patches promptly to close known</td></tr><tr><td>Disinformation Create fake news</td><td>spreading a conspiracy theory about....</td><td>How to identify fake news spreading a conspiracy theory..</td><td>A dangerous virus is spreading across the world and some people are blaming the flu vaccine...</td><td>vulnerabilities... Disinformation about a dangerous virus spreading across the world often uses sensational headlines...</td></tr></table>

Table 7: Representative dataset examples showing paired harmful and benign queries $( \mathcal { Q } _ { f } , \mathcal { Q } _ { r } )$ and their corresponding responses (D<sub>f</sub>, D<sub>r</sub>) for the same concept.

![](images/6d454cf470a2825968614383c96813ed90077a812aebe7bfd5e89435e16864b0.jpg)  
Figure 8: Prompt used to evaluate the harmfulness of responses

## C.2 Unlearning Hyperparameters

We report method-specific hyperparameters in Table 9. All methods inherit the fine-tuning configuration unless explicitly overridden.

![](images/257a79a82e32e7e8654b59952dbefed3a1b534ab7f81799517723cc0e808fc81.jpg)  
Figure 9: Prompt used to evaluate the helpfulness of responses

Table 8: Shared fine-tuning hyperparameters for all models
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Per-device train batch size Per-device eval batch size</td><td>8 16</td></tr><tr><td>Gradient accumulation</td><td>4</td></tr><tr><td>Effective batch size</td><td>32</td></tr><tr><td></td><td></td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Epochs</td><td>10</td></tr><tr><td>Optimizer</td><td>Paged AdamW (32-bit)</td></tr><tr><td>Weight decay</td><td>0.0</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>Evaluation strategy</td><td>per epoch</td></tr><tr><td>Logging steps</td><td>5</td></tr><tr><td>Gradient checkpointing</td><td>disabled</td></tr></table>

Table 9: Method-specific hyperparameters for unlearning
<table><tr><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Hyperparameters</td></tr><tr><td rowspan=1 colspan=1>Gradient Ascent</td><td rowspan=1 colspan=1>Standard fine-tuning setup with gradient ascent on $\mathcal { D } _ { f }$ and descent on $\mathcal { D } _ { r }$ </td></tr><tr><td rowspan=1 colspan=1>SimNPO</td><td rowspan=1 colspan=1> $\overline { { \beta = 4 . 5 , \gamma = 0 . 1 2 5 , \alpha = 1 . 0 } }$  $\delta = 0 . 0$ Retain loss: NLL</td></tr><tr><td rowspan=1 colspan=1>RMU</td><td rowspan=1 colspan=1> $\gamma = 1 . 0 , \alpha = 1 . 0 $ , steering coefficient = 2Retain loss: embedding differenceModule: layer 7 activationsAll parameters updated</td></tr><tr><td rowspan=1 colspan=1>UNDIAL</td><td rowspan=1 colspan=1> $\overline { { \mathrm { L e a r n i n g ~ r a t e } = 1 \times 1 0 ^ { - 4 } } }$ Epochs = 10 $\beta = 1 0 . 0 , \gamma = 1 . 0 , \alpha = 0 . 0$ Retain loss: NLL</td></tr></table>