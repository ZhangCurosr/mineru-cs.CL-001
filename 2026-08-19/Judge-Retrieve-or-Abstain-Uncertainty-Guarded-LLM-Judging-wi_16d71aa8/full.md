# Judge, Retrieve, or Abstain: Uncertainty-Guarded LLM Judging with Provable Risk Guarantees

Sher Badshah1,2, Ali Emami³, Hassan Sajjad1,2

1Dalhousie University 2New York University Abu Dhabi 3Emory University   
{sh545346,hsajjad}@dal.ca   
ali.emami@emory.edu

## Abstract

Using LLMs as judges has become standard practice for evaluating model outputs at scale. This is particularly common for subjective, open-ended tasks such as assessing helpfulness or alignment, where no single reference answer exists. However, objective tasks introduce a distinct reliability challenge for reference-free LLM judging. In the absence of a reference answer, the judge evaluates factual correctness either through its parametric knowledge or through tool augmentation. Although the former enables efficient evaluation, the judge may hallucinate or lack sufficient evidence for its verdict. Conversely, tool augmentation can provide additional evidence but introduces extra computational cost and requires an appropriate mechanism to determine when and how that evidence should be used reliably. More importantly, neither approach alone provides formal control over the risk of accepted verdicts or guarantees their reliability at a specified level. We propose a risk-controlled framework that calibrates uncertainty thresholds on a held-out set so that the false discovery rate among accepted verdicts remains below a user-specified level α with high probability, using finite-sample Clopper-Pearson intervals. When the parametric mode is not sufficiently confident, the instance is routed to a retrieval-augmented mode, where the judge gathers web evidence and re-evaluates the instance under a second calibrated threshold. The finite-sample guarantee carries over to this two-threshold routing without additional assumptions. Across open-domain QA benchmarks and judges of varying scales, the framework maintains the target error rate while achieving substantially higher coverage than single-mode baselines.

## 1 Introduction

LLM-as-a-judge has emerged as a scalable alternative to human evaluation for assessing language model outputs (Zheng et al., 2023). It enables practitioners to evaluate thousands of model outputs at a fraction of the cost and time required for human annotation (Gu et al., 2024; Tan et al., 2025). However, most existing work employs LLM judges for subjective evaluation (Badshah et al., 2025), such as scoring outputs for helpfulness, harmlessness, or style in reward modeling, where no single correct answer exists and approximate agreement suffices (Yuan et al., 2024; Liu et al., 2023). Extending LLM-as-a-judge to objective, factual evaluation remains an open challenge.

In factual and open-domain QA, answers are either correct or incorrect, and the judge must make this determination reliably. However, obtaining ground-truth annotations for this purpose is expensive at scale (Chiang & Lee, 2023; Mañas et al., 2024), requires domain expertise for specialized questions, and is entirely infeasible when LLMs operate as autonomous agents answering open-ended queries in real-world environments where no pre-collected labels exist. LLM-as-a-judge offers a reference-free alternative that eliminates this dependency. Nevertheless, utilizing LLM judges to assess correctness without explicit references introduces two key reliability challenges (Schroeder & Wood-Doughty, 2025; Badshah & Sajjad, 2025).

First, LLM judges are bounded by their training data and may lack knowledge of recent events, rare entities, or specialized domains (Wen et al., 2025). Šuch limitations can prevent reliable distinction between correct answers and plausible fabrications. Second, even if the judge has the relevant knowledge, hallucinations may still cause it to produce confident but incorrect verdicts (Huang et al., 2025). In both cases, the failures are silent as the judge provides no indication that its verdict is unreliable. Without a mechanism to quantify and act on this uncertainty, incorrect evaluations can propagate undetected into downstream decisions.

Selective evaluation provides a natural mitigation strategy by allowing the judge to abstain when uncertain and produce a verdict only when it is likely to be correct. Uncertainty quantification methods can identify unreliable verdicts (Shorinwa et al., 2025). However, applying a heuristic threshold to these scores provides no formal guarantee on the error rate among accepted evaluations (Badshah et al., 2026a). Recent work on risk-controlled selective prediction addresses this gap by calibrating statistically valid thresholds that bound the False Discovery Rate (FDR) among selected outputs with high probability (Wang et al., 2026; 2025a). However, existing methods are designed to filter examinee answers in Question-Answering (QA) tasks rather than control errors in a meta-evaluation setting. Moreover, earlier methods treat abstention as the only recourse for uncertain instances. This discards cases where the judge's uncertainty stems from a knowledge gap or hallucination that could be resolved with additional evidence (Badshah et al., 2026b).

In this paper, we propose a risk-controlled selective evaluation framework for LLM-as-ajudge that addresses the above limitations. Inspired by LLM-based agents that leverage external tools to overcome their parametric limitations, we equip the judge with web search as a tool to bridge its knowledge gaps. Our framework operates in two modes. In Mode 1, the judge model evaluates using only its parametric knowledge without any toolaugmentation. If the judge is confident, the verdict is accepted. If uncertain, the instance is routed to Mode 2 where the judge retrieves relevant evidence from the web and re-evaluates with this augmented context. If the judge remains uncertain after retrieval, it abstains and flags the instance for human review. Both thresholds are jointly calibrated on a held-out set with known ground-truth labels via Clopper-Pearson upper confidence bounds (Clopper & Pearson, 1934). We prove that the finite-sample FDR guarantee for a fixed threshold pair extends to the two-threshold routing setting without aăditional distributional assumptions. Our contributions are:

• We formulate risk-controlled selective evaluation for LLM-as-a-judge on factual QA.

• We introduce a two-mode routing mechanism that equips the judge with adaptive web retrieval to resolve uncertainty before resorting to abstention.

• We show that the Bernoulli structure required for the Clopper-Pearson bound extends to the two-threshold routing setting.

• Experiments across open-domain QA benchmarks and judges of varying scales demonstrate that adaptive retrieval recovers substantial cóverage over single-mode baselines while controlling the judge's error rate.

## 2 Related Work

LLM-as-a-Judge. Using LLMs to evaluate the outputs of other models has gained widespread adoption as a cost-effective alternative to human annotation. Strong LLMs can approximate human preferences on subjective dimensions (Liu et al., 2023). As a result, they have been adopted for reward modeling and reinforcement learning (Son et al., 2024). More recently, LLM judges have been extended to objective tasks such as factual evaluation. SAFE (Wei et al., 2024) uses an LLM agent with web search to verify individual facts in long-form responses. Similarly, SAGE (Badshah et al., 2026b) employs a tool-augmented agent that iteratively retrieves and synthesizes external evidence for reference-free QA evaluation. These works improve judge accuracy but provide no formal guarantee on the reliability of verdicts.

Uncertainty quantification for LLM evaluation. Existing methods generally fall into two categories. White-box methods estimate uncertainty from the judge's token probabilities. The most common choice is the predictive entropy of the output distribution (Malinin & Gales, 2021). These probabilities are often well cālibrated with respect to correctness (Kadavath et al., 2022). Black-box methods, in contrast, estimate uncertainty by querying the judge repeatedly and measuring how consistent its verdicts are (Wang et al., 2023; 2024). Both signals correlate with judge accuracy and can flag low-confidence verdicts (Badshah et al., 2026a). However, they remain heuristic. For instance, a threshold chosen by hand or tuned on a validation set gives no formal guarantee on the error rate among accepted verdicts.

Uncertainty-based routing and tool use. A parallel line of work uses uncertainty to determine when additional resources are needed. FrugalGPT (Chen et al., 2023) and AutoMix (Aggarwal et al., 2024) use confidence to route queries across models with different capabilities, trading off cost against quality. In contrast, FLARE (Jiang et al., 2023) uses token-level uncertainty to decide when to retrieve external information during generation. More recently, SAGE (Badshah et al., 2026b) introduced search-augmented retrieval for LLM judges, where the judge acts as an agent that gathers targeted evidence when evaluating free-form answers. However, these approaches often rely on unconditional retrieval, which introduces unnecessary computation. Furthermore, they do not provide formal guarantees on the error rate of the resulting evaluations.

Risk-controlled selective evaluation. Conformal prediction provides distribution-free, finite-sample coverage guarantees by constructing prediction sets calibrated on held-out data (Angelopoulos & Bates, 2023). Conformal risk control generalizes this beyond coverage to bound arbitrary loss functions, including the FDR, at user-specified levels (Angelopoulos et al., 2024). Recent work has applied these guarantees to question answering. SConU builds prediction sets that cover admissible answers with high probability (Wang et al., 2025b). Similarly, COIN (Wang et al., 2025a) calibrates an uncertainty threshold that bounds the FDR among accepted answers using Clopper-Pearson upper confidence bounds (Clopper & Pearson, 1934). Closer to our setting, Trust or Escalate (Jung et al., 2025) and SCOPE (Badshah et al., 2026a) adapt risk-controlled selective prediction to pairwise evaluation and provide guarantee on human agreement among accepted judgments. However, both target subjective preference tasks where the judge compares two responses rather than evaluating factual correctness (i.e., objective evaluation).

Our framework introduces risk-controlled retrieval-augmented evaluation, where retrieval is selectively activated based on calibrated uncertainty thresholds. Unlike unconditional retrieval approaches, it provides formal FDR guarantees for the accepted verdicts.

## 3 Methodology

We propose a risk-controlled framework for LLM-as-a-judge evaluation. The framework calibrates uncertainty thresholds on a held-out set with human labels so that, among all instances the judge evaluates, the probability of producing an incorrect verdict is bounded by a user-specified risk level α (Angelopoulos & Bates, 2023; Angelopoulos et al., 2024). When the judge is uncertain, the framework first attempts to improve the evaluation through web retrieval; if the judge remains uncertain after retrieval, it abstains and flags the instance for human review 1.

## 3.1 Problem Formulation

Task. Let F : $\mathcal { X }  \mathcal { V }$ be an examinee LLM that, given a question $\mathbf { x } \in \mathcal { X } ,$ , generates a candidate answer $\hat { y } \in \mathcal { V }$ , and let $y ^ { \ast } \in \mathcal { V }$ be the corresponding ground-truth answer. A judge

LLM $\mathcal { I } { : } \mathcal { X } \times \mathcal { Y }  \{ 0 , 1 \}$ evaluates whether Î correctly answers x, producing a binary verdict $v \in \{ 0 , 1 \}$ , where $v = 1$ means the judge deems Î correct.

Admission function. The ground-truth evaluation label $e ^ { * } \in \{ 0 , 1 \}$ indicates whether is truly correct. It can be obtained from human annotation, or by applying a task-specific relevance function $\mathcal { A } _ { \mathrm { r e f } } : \mathcal { V } \times \mathcal { V }  [ 0 , 1 ]$ with a threshold $\lambda _ { A } \colon$

$$
e ^ { * } = \mathbf { 1 } [ \mathcal { A } _ { \mathrm { r e f } } ( \hat { y } , y ^ { * } ) \geq \lambda _ { A } ] .\tag{1}
$$

A verdict v is admissible if it matches the ground truth:

$$
\mathcal { A } ( v , e ^ { * } ) = \mathbf { 1 } [ v = e ^ { * } ] .\tag{2}
$$

Selective evaluation. Rather than always outputting a verdict, the judge is allowed to abstain when it is uncertain. We define a failure indicator $W = \mathbf { 1 } [ \stackrel { \prime } { v } \neq e ^ { * } ]$ . Given an uncertainty score U (defined in §3.2), the judge outputs a verdict only when $U \overset { \cdot } { \leq } t$ for some threshold t. The false discovery rate (FDR) is:

$$
R ( t ) = \mathbb { E } [ W \mid U \leq t ] ,\tag{3}
$$

i.e., the error rate among instances the judge chooses to evaluate.

Goal. For a user-specified risk level $\alpha \in ( 0 , 1 )$ and significance level $\delta \left( \mathrm { e . g . } , 0 . 0 5 \right)$ , we seek a threshold t such that:

$$
\Pr ( R ( t ) \leq \alpha ) \geq 1 - \delta ,\tag{4}
$$

where the probability is over the randomness of the calibration data. With high confidence $( 1 - \delta )$ , the error rate among the judge's non-abstained outputs is at most α.

Calibration data. We hold out N calibration samples $\mathcal { D } _ { \mathrm { c a l } } = \{ ( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } ) \} _ { i = 1 } ^ { N }$ with known ground-truth labels $e _ { i } ^ { * }$ , drawn i.i.d. from the data-generating distribution $\dot { \mathcal { D } } .$

## 3.2 Uncertainty Quantification (UQ)

Accurate uncertainty estimation is critical for selective evaluation: the better the uncertainty score separates correct from incorrect verdicts, the higher the coverage (fewer abstentions) at a given risk level.

## 3.3 Risk-Bounded Selective Evaluation

We now describe how to calibrate the uncertainty threshold to satisfy the guarantee in Eq. (4). This section presents the base case where the judge uses only its parametric knowledge (no retrieval); we extend to retrieval-augmented judging in §3.4.

Empirical failure analysis. For each calibration instance, we obtain the judge's verdict $v _ { i }$ and uncertainty score $U _ { i }$ Given a candidate threshold $t ,$ we compute the number of instances the judge would evaluate (selection size):

$$
\hat { m } _ { \mathrm { c a l } } ( t ) = \sum _ { i = 1 } ^ { N } { \bf 1 } \{ U _ { i } \leq t \} ,\tag{5}
$$

and the number of errors among them (error count):

$$
\hat { w } _ { \mathrm { c a l } } ( t ) = \sum _ { i = 1 } ^ { N } \mathbf { 1 } \{ U _ { i } \leq t \} \cdot ( 1 - c _ { i } ) ,\tag{6}
$$

where $c _ { i } ~ = ~ \mathcal { A } ( v _ { i } , e _ { i } ^ { * } )$ is the correctness indicator. The empirical FDR is $\hat { r } _ { \mathrm { c a l } } ( t ) ~ =$ $\hat { w } _ { \mathrm { c a l } } ( t ) / \hat { m } _ { \mathrm { c a l } } ( t )$

Upper confidence bound (UCB). The empirical FDR is an unbiased point estimate of the true failure rate, but may underestimate it on any given calibration set due to sampling variability. To obtain a rigorous bound, we leverage the Clopper-Pearson exact method. Since the calibration instances are i.i.d. and the selection event $\{ U _ { i } \leq t \}$ is a deterministic function of $\left( \mathbf { x } _ { i } , \hat { y } _ { i } \right)$ , the failure indicators of selected instances are i.i.d. Bernoulli with parameter $R ( t )$ . We can therefore apply the one-sided Clopper-Pearson bound:

$$
\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) = \mathrm { B e t a l n v } \big ( 1 - \delta ; \hat { w } _ { \mathrm { c a l } } ( t ) + 1 , \hat { m } _ { \mathrm { c a l } } ( t ) - \hat { w } _ { \mathrm { c a l } } ( t ) \big ) ,\tag{7}
$$

which satisfies $\mathrm { P r } ( R ( t ) \leq \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) ) \geq 1 - \delta$ (Appendix A.1).

Threshold selection. We select the largest threshold whose UCB stays below the risk level α:

$$
\hat { t } = \operatorname* { s u p } \big \{ t : \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ^ { \prime } ) \leq \alpha \ \mathrm { f o r } \ \mathrm { a l l } \ t ^ { \prime } \leq t \big \} .\tag{8}
$$

This maximizes the number of instances the judge evaluates while targeting Eq. (4).

## 3.4 Adaptive Retrieval via Two-Mode Routing

Inspired by LLM-based agents that leverage external tools to overcome the base model limitations, we equip the judge with web search as a tool to bridge its knowledge gaps. When the judge is not confident based on its parametric knowledge alone, it retrieves relevant evidence from the web and re-evaluates with this augmented context. This recovers coverage on exactly the instances that the single-mode framework would abstain on, turning abstention into an opportunity for informed evaluation

Two evaluation modes. For each instance $( \mathbf { x } _ { i } , \hat { y } _ { i } )$ , we define:

• Mode 1 (direct): J evaluates using only its internal knowledge, producing verdict $v _ { 1 } ^ { ( i ) }$ with uncertainty $U _ { 1 } ^ { ( i ) }$

• Mode 2 (retrieval-augmented): The system queries a web search engine using the question $\mathbf { x } _ { i } ,$ retrieving the top-k results, each containing a title, text snippet, and source URL. These are concatenated into an evidence passage $\mathcal { E } _ { i }$ appended to the judge's prompt. The judge $\mathcal { I }$ re-evaluates with this context, producing $v _ { 2 } ^ { ( i ) }$ with uncertainty $U _ { 2 } ^ { ( i ) }$

We denote the correctness of each mode as $c _ { k } ^ { ( i ) } = \mathcal { A } ( v _ { k } ^ { ( i ) } , e _ { i } ^ { * } )$ for $k \in \{ 1 , 2 \}$

Routing logic. At test time, given calibrated thresholds $( \hat { t } _ { 1 } , \hat { t } _ { 2 } )$

$$
\begin{array} { r } { \mathrm { o u t p u t } = \left\{ \begin{array} { l l } { v _ { 1 } , } & { \mathrm { i f } U _ { 1 } \leq \hat { t } _ { 1 } , } \\ { v _ { 2 } , } & { \mathrm { i f } U _ { 1 } > \hat { t } _ { 1 } \mathrm { a n d } U _ { 2 } \leq \hat { t } _ { 2 } , } \\ { \mathrm { A B S T A I N , } } & { \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}\tag{9}
$$

Retrieval is invoked only when Mode 1 is uncertain, so confident instances incur zero retrieval cost.

Joint threshold calibration. During calibration, we pre-compute both modes for all N instances in $\mathcal { D } _ { \mathrm { c a l } }$ . For a candidate pair $\left( t _ { 1 } , t _ { 2 } \right)$ , the selection size counts instances accepted via either mode:

$$
\hat { m } _ { \mathrm { c a l } } ( t _ { 1 } , t _ { 2 } ) = \sum _ { i = 1 } ^ { N } \Big [ \mathbf { 1 } \big \{ U _ { 1 } ^ { ( i ) } \leq t _ { 1 } \big \} + \mathbf { 1 } \big \{ U _ { 1 } ^ { ( i ) } > t _ { 1 } \mathrm { a n d } U _ { 2 } ^ { ( i ) } \leq t _ { 2 } \big \} \Big ] .\tag{10}
$$

The error count among selected instances is:

$$
\hat { w } _ { \mathrm { c a l } } ( t _ { 1 } , t _ { 2 } ) = \sum _ { i = 1 } ^ { N } \left[ \mathbf { 1 } \{ U _ { 1 } ^ { ( i ) } \leq t _ { 1 } \} ( 1 - c _ { 1 } ^ { ( i ) } ) + \mathbf { 1 } \{ U _ { 1 } ^ { ( i ) } > t _ { 1 } \land U _ { 2 } ^ { ( i ) } \leq t _ { 2 } \} ( 1 - c _ { 2 } ^ { ( i ) } ) \right] .\tag{11}
$$

The empirical FDR and UCB follow as before:

$$
\hat { r } _ { \mathrm { c a l } } ( t _ { 1 } , t _ { 2 } ) = \frac { \hat { w } _ { \mathrm { c a l } } ( t _ { 1 } , t _ { 2 } ) } { \hat { m } _ { \mathrm { c a l } } ( t _ { 1 } , t _ { 2 } ) } ,\tag{12}
$$

$$
\begin{array} { r } { \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t _ { 1 } , t _ { 2 } ) = \mathrm { B e t a l n v } \left( 1 - \delta ; \ \hat { w } _ { \mathrm { c a l } } + 1 , \ \hat { m } _ { \mathrm { c a l } } - \hat { w } _ { \mathrm { c a l } } \right) . } \end{array}\tag{13}
$$

We select the threshold pair that maximizes coverage subject to the risk constraint:

$$
\bigl ( \hat { t } _ { 1 } , \hat { t } _ { 2 } \bigr ) = \underset { t _ { 1 } , t _ { 2 } } { \arg \operatorname* { m a x } } \hat { m } _ { \mathrm { c a l } } \bigl ( t _ { 1 } , t _ { 2 } \bigr ) \quad \mathrm { s . t . } \quad \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t _ { 1 } , t _ { 2 } ) \leq \alpha .\tag{14}
$$

In practice, this is solved by search over the sorted unique uncertainty values on the calibration set. For each candidate $t _ { 1 } ,$ , cumulative sums enable evaluating all $t _ { 2 }$ candidates in $O ( | T _ { 2 } | )$ time, yielding $O ( | T _ { 1 } | \cdot | \overline { { T _ { 2 } } } | )$ total complexity (Algorithm 1). Because this search evaluates many threshold pairs, we additionally report a conservative variant that corrects for it via a Bonferroni adjustment $( \delta ^ { \prime } = \delta / ( | \mathcal { T } _ { 1 } | ^ { - } | \mathcal { T } _ { 2 } \overset { \cdot } { | } ) )$ (see Appendix D.4).

## 3.5 Theoretical Guarantee

The central theoretical question is whether the single-mode Clopper-Pearson guarantee (Eq. 4) extends to the two-mode routing setting. It does, because the routing decision for each instance is entirely determined by its uncertainty scores and the fixed thresholds, introducing no additional randomness.

Lemma 1 (Bernoulli structure under routing). Let $\left( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } \right)$ be i.i.d. samples from D, and let $\left( t _ { 1 } , t _ { 2 } \right)$ be fixed thresholds. Under the routing policy (Eq. 9), the failure indicators $\{ W _ { i } \}$ over the selected subset are i.i.d. Bernoulli $\left( R ( t _ { 1 } , t _ { 2 } ) \right)$ 1

The proof (Appendix A.2) shows that the selection indicator and the correctness of selected instances are both deterministic functions of $\left( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } \right)$ . Since the original data is i.i.d., conditioning on selection preserves independence and identical distribution. Combined with the Clopper-Pearson bound, this yields a guarantee for any fixed threshold pair:

$$
\begin{array} { r } { \operatorname* { P r } \Bigl ( R \bigl ( t _ { 1 } , t _ { 2 } \bigr ) \le \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } \bigl ( t _ { 1 } , t _ { 2 } \bigr ) \Bigr ) \ge 1 - \delta . } \end{array}\tag{15}
$$

In particular, any fixed pair whose bound satisfies $\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t _ { 1 } , t _ { 2 } ) \leq \alpha$ controls the error rate at α with confidence $1 - \delta$

## 4 Experiments

Datasets and models. We evaluate on four open-domain QA benchmarks: TriviaQA (Joshi et al., 2017), Natural Questions (NQ) (Kwiatkowski et al., 2019), HotpotQA (Yang et al. 2018), and PopQA (Mallen et al., 2022). From each benchmark's standard held-out split, we sample 2,000 instances using seed 42.

We use two candidate models: Qwen3-8B (Qwen Team, 2025b) and LLaMA-3.1-70B (Meta AI, 2024). For judges, we employ four models: Qwen3-4B, Qwen3-8B, and Qwen3-14B (Qwen Team, 2025b) and LLaMA-3.1-8B-Instruct (Meta AI, 2024). Each judge outputs a binary verdict (True/False) and a brief explanation.

Admission functions. To evaluate whether the judge's verdict is correct, we first determine whether the candidate's answer is actually right by comparing it against the reference answer. This ground-truth label $e ^ { * }$ is computed using three admission functions: (1) Exact Match (EM), (2) Token F1 (threshold $\geq \hat { 0 } . 5 )$ , and (3) LLM Evaluator (Badshah & Sajjad, 2025; Badshah et al., 2025), which uses Qwen2.5-7B-Instruct (Qwen Team, 2025a) as a reference-based evaluator to judge semantic equivalence.

Algorithm 1 Calibration (offline) Algorithm 2 Test-time evaluation (online)   
Require: Calibration set $\mathcal { D } _ { \mathrm { c a l } } ,$ judge $\overline { { \mathcal { I } } }$ Require: Instance $( \mathbf { x } , \hat { y } ) .$ judge $\mathcal { I }$   
Require: Risk level α, significance level δ Require: Thresholds $\hat { t } _ { 1 } , \hat { t } _ { 2 }$   
Ensure: Calibrated thresholds $\hat { t } _ { 1 } , \hat { t } _ { 2 }$ Ensure: Verdict $v \in \{ 0 , 1 \}$ or ABSTAIN   
1: for each $( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } ) \in \mathcal { D } _ { \mathrm { c a l } }$ do 1: $v _ { 1 } , U _ { 1 } \gets \mathcal { I } ( \mathbf { x } , \hat { y } )$   
2: $v _ { 1 } ^ { ( i ) } , U _ { 1 } ^ { ( i ) }  \mathcal { I } ( \mathbf { x } _ { i } , \hat { y } _ { i } )$ 2: if $U _ { 1 } \leq \hat { t } _ { 1 }$ then   
3: $\bar { \mathcal { E } _ { i } } \gets \dot { \mathrm { R E T R I E V E } } ( \mathbf { x } _ { i } )$ 3: return $v _ { 1 }$ Confident   
4: end if   
4: $v _ { 2 } ^ { ( i ) } , U _ { 2 } ^ { ( i ) } \gets \mathcal { I } ( \mathbf { x } _ { i } , \hat { y } _ { i } , \mathcal { E } _ { i } )$   
$5 { : }$ $\mathbf { \Phi } _ { c _ { 1 } ^ { ( i ) } }  \mathbf { \bar { \mathcal { A } } } ( v _ { 1 } ^ { ( i ) } , e _ { i } ^ { * } ) ; \mathbf { \Phi } _ { c _ { 2 } ^ { ( i ) } }  A ( v _ { 2 } ^ { ( i ) } , e _ { i } ^ { * } )$ 5: $\mathcal { E }  \mathrm { R E T R I E V E } ( \mathbf { x } )$   
6: end for 6: $v _ { 2 } , U _ { 2 } \gets \mathcal { T } ( \mathbf { x } , \boldsymbol { \hat { y } } , \boldsymbol { \hat { \mathcal { E } } } )$   
7: if $U _ { 2 } \leq \hat { t } _ { 2 }$ then   
7: $\mathcal { T } _ { 1 }$ ← sorted unique $\{ U _ { 1 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ 8: return v2 Retrieval helped   
9: end if   
8: $\mathcal { T } _ { 2 }$ ← sorted unique $\{ U _ { 2 } ^ { ( i ) } \} _ { i = 1 } ^ { N }$   
9: $\hat { t } _ { 1 } , \hat { t } _ { 2 } \gets \mathrm { N U L L } ; \ m ^ { * } \gets 0$ 10: return ABSTAIN ▶ Human review   
10: for $t _ { 1 }$ in $\mathcal { T } _ { 1 }$ do   
11: for $t _ { 2 }$ in $\mathcal { T } _ { 2 }$ do   
12: Compute $\hat { m } _ { \mathrm { c a l } } , \hat { w } _ { \mathrm { c a l } }$ Cost analysis. Let $p$ be the fraction of in-  
13: $R ^ { + } \gets$ BetaInv $( 1 { - } \delta ; \hat { w } _ { \mathrm { c a l } } { + } 1 , \hat { m } _ { \mathrm { c a l } } { - } \hat { w } _ { \mathrm { c a l } } )$ stances accepted by Mode 1. Test-time re-  
14: if $R _ { . } ^ { + } \leq \alpha$ and $\hat { m } _ { \mathrm { c a l } } > m ^ { * }$ then trieval is invoked for only $( 1 - p )$ of instances.   
15: $\hat { t } _ { 1 } \gets t _ { 1 } ; \hat { t } _ { 2 } \gets t _ { 2 } ; m ^ { * } \gets \hat { m } _ { \mathrm { c a l } }$ $\mathrm { A t } ~ \alpha = 0 . 2 0$ with Qwen3-14B, $p \approx 0 . 4 5$ on   
16: end if TriviaQA, so nearly half of evaluations incur   
17: end for zero retrieval cost. On harder benchmarks   
18: end for p decreases, but the framework still avoids   
19: return $\hat { t } _ { 1 } , \hat { t } _ { 2 }$ retrieval for the most confident instances.

Uncertainty quantification. We use predictive entropy (PE) (Malinin & Gales, 2021) as the uncertainty measure. PE is computed from the judge's token-level log-probabilities at the verdict position via a single forward pass (Kadavath et al., 2022). When the judge's hidden states are available, PÉ can be replāced by supervised scores without changing the guarantee. We report a probe and a LoRA+Prompt variant in Appendix D.2.

Retrieval. For Mode $^ { 2 , }$ we query a web search engine using the original question $( \mathrm { i . e . , }$ the question provided to the candidate) and retrieve the top-k results, with $k = 3$ as the default setting. The retrieved results are concatenated into a single passage and appended to the judge prompt as additional context. We further study the effect of retrieval depth by varying $k \in \left\{ 3 , 6 , 9 \right\}$ on one candidate-judge pair (LLaMA-3.1-70B evaluated by Qwen3-8B).

Calibration protocol. We run 100 random 50/50 calibration/test splits using the Clopper– Pearson exact binomial bound. We set $\delta = 0 . 0 5$ and evaluate $\alpha \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 1 5 , 0 . 2 0 , \dot { 0 } . 2 5 \}$ Appendix D.1 reports sensitivity to both δ and the calibration-set size.

Evaluation metrics. We report: (1) FDR, the fraction of accepted instances with incorrect verdicts $( w _ { \mathrm { t e s t } } / m _ { \mathrm { t e s t } } ) ;$ (2) Coverage, the fraction of instances accepted rather than abstained on $( m _ { \mathrm { { t e s t } } } / N _ { \mathrm { { t e s t } } } ) ;$ and (3) Routing distribution which is the split across Mode 1, Mode 2, and abstention.

## 5 Results

We evaluate whether the framework delivers on its two central promises: (1) the FDR guarantee holds empirically across judges, candidates, and datasets, and (2) adaptive retrieval recovers coverage that would otherwise be lost to abstention. We use predictive entropy and LLM evaluator for the main results, and additionally analyze the sensitivity of our framework to the choice of admission function and retrieval depth (§D.1).

![](images/c6d36920b51db9104aaafaf07d8c6a57182ec368fd8170a5ebe5b296c256eed3.jpg)  
Figure 1: Test-time FDR (mean±std over 100 splits) across all 32 configurations. Rows correspond to the two candidate models; columns to the four datasets. Each line represents a different judge (see legend). The dashed diagonal marks FDR = α.

## 5.1 FDR control holds across all configurations

The most basic requirement of our framework is that the empirical FDR remains at or below the target α. Figure 1 tests this across all candidate-judge-dataset configurations. In every case, the FDR stays at or below α, satisfying the finite-sample guarantee (Eq. 15).

Three observations are noteworthy. First, all four judges provide valid FDR control regardless of their parameter count, showing that the guarāntee does not require a strong judge. Second, FDR tracks close to α with narrow confidence bands (±0.01–0.02), which indicates stable calibration across 100 random splits. Third, the gap between FDR and α is smallest on TriviaQA and PopQA, which are relatively easy tasks where judges are most accurate. This demonstrates that the framework is tightest when the judge has sufficient knowledge to evaluate.

## 5.2 Coverage scales with risk tolerance and judge capability

Table 1 reports coverage across five risk levels. Coverage scales smoothly with α and is governed by two factors: dataset difficulty and judge capability. On easy benchmarks (TriviaQA, PopQA), even the smallest judge reaches near-ull coverage by α = 0.25, while on knowledge-intensive tasks (NQ-Open, HotpotQA) only the larger judges (Qwen-14B, Llama-8B) surpass 80% at α = 0.20 for the Qwen-8B candidate — Qwen-4B remains below 15% on these sāme benchmarks. Evaluating the stronger Llama-70B candidate is consistently harder: coverage drops across all judges compared to Qwen-8B, with the best judge reaching only 53% on HotpotQA at α = 0.20. This gap reflects both the answer diversity of the 70B model and the increased knowledge demands it places on the judge.

## 5.3 Adaptive retrieval recovers substantial coverage

Figure 2 breaks down accepted instances by routing mode. It is depicted that retrieval is the dominant acceptance mechanism. Across various configurations, Mode 2 accounts for the majority of non-abstained verdicts. This does not necessarily mean the judge lacks the knowledge to evaluate correctly, but rather that its uncertainty signal in Mode 1 is not confident enough to satisfy the calibrated threshold. Crucially, these are instances where a single-mode framework would abstain entirely. By grounding the judge in retrieved evidence, Mode 2 reduces uncertainty below the second threshold and converts these abstentions into accepted verdicts. This trade-off also depends on the underlying judge model. Models with better-calibrated uncertainty estimates allow a larger fraction of instances to be resolved by Mode 1 directly. For instance, at α = 0.20 on TriviaQA, Qwen3- 14B resolves 45% of evaluations parametrically versus 9% for Qwen3-4B. On harder tasks like HotpotQA and NQ-Open, all judges rely predominantly on retrieved evidence.

Table 1: Coverage of the joint two-mode framework (two-threshold routing with retrieval) at α ∈ {.05, .10, .15, .20, .25} (mean over 100 splits) across all datasets.
<table><tr><td rowspan="2">Cand.</td><td rowspan="2">Judge</td><td colspan="4">TriviaQA .10</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2">NQ-Open .15</td><td rowspan="2">.20</td><td rowspan="2">.25</td><td rowspan="2">.05</td><td rowspan="2">.10</td><td rowspan="2">PopQA .15</td><td rowspan="2">.20</td><td rowspan="2">.25</td><td rowspan="2">.05 .10</td><td rowspan="2">HotpotQA</td><td rowspan="2">.15</td><td rowspan="2">.20</td><td rowspan="2">.25</td></tr><tr><td></td><td>.05</td><td>.15 .20</td><td>.05 .10</td></tr><tr><td rowspan="4">Owe-8BB</td><td>Qwen-4B</td><td>.03</td><td>.39</td><td>.66</td><td>.91</td><td>1.0</td><td>.00</td><td>.03</td><td>.08</td><td>.14</td><td>.28 .00</td><td>.01</td><td>.91</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.09</td><td>.14</td><td>.30</td></tr><tr><td>Qwen-8B</td><td>.12</td><td>.63</td><td>.91</td><td>1.0</td><td>1.0</td><td>.00</td><td>.02</td><td>.24</td><td>.82</td><td>1.0</td><td>.00</td><td>.87</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.49 .74</td><td>.96</td></tr><tr><td>Qwen-14B</td><td>.21</td><td>.78</td><td>.98</td><td>1.0</td><td>1.0</td><td>.00</td><td>.05</td><td>.56</td><td>.81</td><td>.99</td><td>.19</td><td>.89</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.02</td><td>.26</td><td>.64</td><td></td><td>1.0</td></tr><tr><td>Llama-8B</td><td>.13</td><td>.61</td><td>.86</td><td>1.0</td><td>1.0</td><td>.00</td><td>.20</td><td>.61</td><td>.83</td><td>1.0</td><td>.02</td><td>.88</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.01</td><td>.30</td><td>.65</td><td>.86 .85</td><td>1.0</td></tr><tr><td rowspan="4">LIla0B</td><td>Qwen-4B</td><td>.05</td><td>.37</td><td>.80</td><td>.98</td><td>1.0</td><td>.00</td><td>.05</td><td>.18</td><td>.32</td><td>.48</td><td>.00</td><td>.00</td><td>.11</td><td>.35</td><td>.92</td><td>.00</td><td>.09</td><td>.20</td><td></td><td>.55</td></tr><tr><td>Qwen-8B</td><td>.17</td><td>.44</td><td>.83</td><td>.99</td><td>1.0</td><td>.01</td><td>.10</td><td>.25</td><td>.47</td><td>.73</td><td>.00</td><td>.06</td><td>.11</td><td>.48</td><td>.98</td><td>.00</td><td>.11</td><td>.31</td><td>.34 .47</td><td>.65</td></tr><tr><td>Qwen-14B</td><td>.12</td><td>.46</td><td>.85</td><td>1.0</td><td>1.0</td><td>.00</td><td>.05</td><td>.23</td><td>.51</td><td>.75</td><td>.00</td><td>.04</td><td>.31</td><td>.77</td><td>.99</td><td>.00</td><td>.07</td><td>.31</td><td>.53</td><td>.76</td></tr><tr><td>Llama-8B</td><td>.04</td><td>.32</td><td>.67</td><td>.86</td><td>1.0</td><td>.00</td><td>.02</td><td>.13</td><td>.32</td><td>.59</td><td>.00</td><td>.07</td><td>.23</td><td>.77</td><td>.98</td><td>.00</td><td>.02</td><td>.11</td><td>.27</td><td>.51</td></tr></table>

Although retrieval substantially expands coverage, it does not always improve the judge's confidence. For 14–29% of instances, retrieval increases rather than decreases uncertainty (Appendix E.1). The second calibrated threshold identifies these cases and routes them to abstention. Consequently, only retrieval-based verdicts that satisfy the desired risk level are accepted.

![](images/57d51e5176e49bf0feeb9622f37634f2a491d7a1e4f1fd1ecb31c6d852705604.jpg)  
Figure 2: Routing distribution across configurations. Within each group of four bars, judges are ordered left to right: Qwen3-4B, Qwen3-8B, Qwen3-14B, Llama-3.1-8B.

## 5.4 Single-mode vs. joint two-mode routing

Table 2 isolates the contribution of adaptive retrieval by comparing Mode 1 Only against the full joint framework. On TriviaQA, Mode 1 already achieves high coverage for strong judges (87% for Qwen3-14B), so retrieval provides a modest but consistent gain. The effect is more pronounced: on NQ-Open, coverage for Qwen3-8B increases from 7% to 82%; on HotpotQA, coverage for Llama-8B increases from 40% to 85%. Similarly, the weakest judge (Qwen3-4B) benefits substantially on PopQA, with coverage increasing from 4% to 100%.

Table 2: Coverage at α = 0.20: Mode 1 Only vs. Joint (two-threshold routing with retrieval).
<table><tr><td></td><td></td><td colspan="2">TriviaQA</td><td colspan="2">NQ-Open</td><td colspan="2">PopQA</td><td colspan="2">HotpotQA</td></tr><tr><td>Cand.</td><td>Judge</td><td>M1 Only Joint</td><td></td><td>M1 Only Joint</td><td></td><td>M1 Only</td><td>Joint</td><td>M1 Only Joint</td><td></td></tr><tr><td></td><td>Qwen-4B</td><td>.47</td><td>.91</td><td>.01</td><td>.14</td><td>.04</td><td>1.0</td><td>.00</td><td>.14</td></tr><tr><td></td><td>Qwen-8B</td><td>.59</td><td>1.0</td><td>.07</td><td>.82</td><td>.03</td><td>1.0</td><td>.05</td><td>.74</td></tr><tr><td>Owe-8B</td><td>Qwen-14B</td><td>.87</td><td>1.0</td><td>.14</td><td>.81</td><td>.15</td><td>1.0</td><td>.35</td><td>.86</td></tr><tr><td></td><td>Llama-8B</td><td>.77</td><td>1.0</td><td>.22</td><td>.83</td><td>.21</td><td>1.0</td><td>.40</td><td>.85</td></tr><tr><td></td><td>Qwen-4B</td><td>.67</td><td>.98</td><td>.07</td><td>.32</td><td>.05</td><td>.35</td><td>.09</td><td>.34</td></tr><tr><td>LI1m0B</td><td>Qwen-8B</td><td>.82</td><td>.99</td><td>.22</td><td>.47</td><td>.08</td><td>.48</td><td>.27</td><td>.47</td></tr><tr><td></td><td>Qwen-14B</td><td>.84</td><td>1.0</td><td>.12</td><td>.51</td><td>.15</td><td>.77</td><td>.24</td><td>.53</td></tr><tr><td></td><td>Llama-8B</td><td>.48</td><td>.86</td><td>.02</td><td>.32</td><td>.12</td><td>.77</td><td>.02</td><td>.27</td></tr></table>

## 5.5 Robustness to retrieval non-stationarity

Web search results change over time, which §3.5 identified as the main practical threat to the determinism the guarantee assumes for Mode 2. We measure the effect directly on the main configuration (Qwen3-8B candidate, Qwen3-8B judge, LLM evaluator, $k \overset { \cdot } { = } 3 )$ We re-query the same 2,000 questions on TriviaQA and HotpotQA roughly three months after the original snapshot and recompute the Mode 2 verdicts on the fresh evidence. The retrieved pages turn over substantially. The mean per-item Jaccard overlap between the original (τ0) and re-queried (τ1) ranked top-3 URL lists is 0.30 for TriviaQA and 0.24 for HotpotQA. When ignoring the retrieval rank and considering only URL membership, the corresponding position-blind top-3 overlaps are 0.47 and 0.40, respectively.

To evaluate whether the calibrated thresholds remain valid after retrieval drift, we calibrate $( \hat { t } _ { 1 } , \hat { t } _ { 2 } )$ on the original data and apply them to test instances evaluated with the updated Mode 2 verdicts (while keeping Mode 1 unchanged). We repeat this evaluation over 100 random 50/50 splits. Table 3 summarizes the results. Across all settings, the empirical FDR remains below the target level α. Similarly, coverage remains largely stable after redeployment. This indicates that drift changes which pages are returned more than the overall quality of the evidence. Therefore, the calibrated thresholds remain effective after redeployment.

Table 3: FDR and coverage under retrieval drift from the original (τ0) to the re-queried snapshot $( \tau _ { 1 } , \sim 3$ months later). $\Delta = \tau _ { 1 } - \tau _ { 0 }$
<table><tr><td></td><td colspan="6">TriviaQA</td><td colspan="6">HotpotQA</td></tr><tr><td></td><td colspan="3">FDR</td><td colspan="3">Coverage</td><td colspan="3">FDR</td><td colspan="3">Coverage</td></tr><tr><td>α</td><td>T0</td><td>T1</td><td>Δ</td><td>T0</td><td>T1</td><td>∆</td><td>T0</td><td>T1</td><td>∆</td><td>T0</td><td>T1</td><td>∆</td></tr><tr><td>.05</td><td>.03</td><td>.04</td><td>.00</td><td>.12</td><td>.17</td><td>+.05</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td></tr><tr><td>.10</td><td>.08</td><td>.09</td><td>.00</td><td>.63</td><td>.68</td><td>+.06</td><td>.05</td><td>.03</td><td>-.01</td><td>.04</td><td>.02</td><td>-.02</td></tr><tr><td>.15</td><td>.14</td><td>.13</td><td>.00</td><td>.91</td><td>.94</td><td>+.04</td><td>.13</td><td>.13</td><td>.00</td><td>.49</td><td>.49</td><td>.00</td></tr><tr><td>.20</td><td>.16</td><td>.15</td><td>-.01</td><td>1.0</td><td>1.0</td><td>.00</td><td>.18</td><td>.18</td><td>.00</td><td>.74</td><td>.72</td><td>-.02</td></tr><tr><td>.25</td><td>.16</td><td>.15</td><td>-.01</td><td>1.0</td><td>1.0</td><td>.00</td><td>.23</td><td>.23</td><td>.00</td><td>.96</td><td>.96</td><td>.00</td></tr></table>

## 6 Conclusion

We introduced a risk-controlled selective evaluation framework for LLM-as-a-judge with finite-sample guarantees on the reliability of accepted verdicts. The framework uses a calibrated two-mode policy that routes uncertain instances to a second evaluation mode and achieves higher coverage than single-mode abstention. More importantly, the Clopper-Pearson FDR guarantee remains valid under the joint two-threshold policy without requiring additional distributional assumptions. Across all tested configurations, the observed FDR remains at or below the specified target. Future work will consider graded and subjective evaluation settings, as well as black-box judges without access to logits.

## Acknowledgments

We acknowledge the support of the Natural Sciences and Engineering Research Council of Canada (NSERC), the Canada Foundation for Innovation (CFI), and Research Nova Scotia. Advanced computing resources are provided by ACENET, the regional partner in Atlantic Canada, and the Digital Research Alliance of Canada.

## References

Pranjal Aggarwal, Aman Madaan, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal,'Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Manaal Faruqui, and Mausam Mausam. Automix: Automatically mixing language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 131000–131034. Curran Associates, Inc., 2024. doi: 10. 52202/079017-4164. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/ecda225cb187b40ea8edc1f46b03ffda-Paper-Conference.pdf.

Anastasios N Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. Foundations and Trends® in Machine Learning, 16(4):494–591, 2023.

Anastasios Nikolas Angelopoulos, Stephen Bates, Adam Fisch, Lihua Lei, and Tal Schuster. Conformal risk control. Ín The Twelth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=33XGfHLtZg.

Sher Badshah and Hassan Sajjad. Reference-guided verdict: LLMs-as-judges in automatic evaluation of free-form QA. In Chen Zhang, Emily Allaway, Hua Shen, Lesly Miculicich, Yinqiao Li, Meryem M'hamdi, Peerat Limkonchotiwat, Richard He Bai, Santosh T.y.s.s., Sophia Simeng Han, Surendrabikram Thapa, and Wiem Ben Rim (eds.), Proceedings of the 9th Widening NLP Workshop, pp. 251–267, Suzhou, China, November 2025. Association for Computational Linguistics. ISBN 979-8-89176-351-7. doi: 10.18653/v1/2025.winlp-main. 37. URL https://aclanthology.org/2025.winlp-main.37/.

Sher Badshah, Moamen Moustafa, and Hassan Sajjad. CLEV: LLM-based evaluation through lightweight efficient voting for free-form question-answering. In Kentaro Inui, Sakriani Sakti, Haofen Wang, Derek F. Wong, Pushpak Bhattacharyya, Biplab Banerjee, Asif Ekbal, Tanmoy Chakraborty, and Dhirendra Pratap Singh (eds.), Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conferènce of the Asia-Pacific Chapter of the Association for Computational Linguistics, pp. 1513–1531, Mumbai, India, December 2025. The Asian Federation of Natural Language Processing and The Association for Computational Linguistics. ISBN 979-8-89176-303-6. URL https:// aclanthology.org/2025.findings-ijcnlp.93/.

Sher Badshah, Ali Emami, and Hassan Sajjad. SCOPE: Selective Conformal Optimized Pairwise LLM Judging. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. PMLR, 2026a.

Sher Badshah, Ali Emami, and Hassan Sajjad. SAGE: A Search-AuGmented Evaluation of large language models on free-form QA. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1466–1491, San Diego, California, United States, July 2026b. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-1ong.66. URL https: //aclanthology.org/2026.acl-long.66/.

Lingjiao Chen, Matei Zaharia, and James Zou. Frugalgpt: How to use large language models while reducing cost and improving performance, 2023. URL https://arxiv.org/abs/ 2305.05176.

Cheng-Han Chiang and Hung-yi Lee. Can large language models be an alternative to human evaluations? In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15607–15631, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.870. URL https://aclanthology.org/2023. acl-long.870/.

C. J. Clopper and E. S. Pearson. The use of confidence or fiducial limits illustrated in the case of the binomial. Biometrika, 26(4):404–413, 1934. ISSN 00063444, 14643510. URL http://www.jstor.org/stable/2331986.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, et ai. A survey on llm-as-a-judge. The Innovation, 2024.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2), January 2025. ISSN 1046-8188. doi: 10.1145/3703155. URL https://doi.org/10.1145/3703155.

Zhengbao Jiang, Frank Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 7969–7992, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.495. URL https://aclanthology.org/2023.emnlp-main.495/.

Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension, 2017. URL https://arxiv.org/abs/1705.03551.

Jaehun Jung, Faeze Brahman, and Yejin Choi. Trust or escalate: LLM judges with provable guarantees for human agreement. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=UHPnqSTBPO.

Saurav Kadavath, Tom Conerly, Amanda Askell, Tom Henighan, Dawn Drain, Ethan Perez, Nicholas Schiefer, Zac Hatfield-Dodds, Nova DasSarma, Éli Tran-Johnson, Scott Johnston, Sheer El-Showk, Andy Jones, Nelson Elhage, Tristan Hume, Anna Chen, Yuntao Bai, Sam Bowman, Stanislav Fort, Deep Ganguli, Danny Hernandez, Josh Jacobson, Jackson Kernion, Shauna Kravec, Liane Lovitt, Kamal Ndousse, Catherine Olsson, Sam Ringer, Dario Amodei, Tom Brown, Jack Clark, Nicholas Joseph, Ben Mann, Sam McCandlish, Chris Olah, and Jared Kaplan. Language models (mostly) know what they know, 2022. URL https://arxiv.org/abs/2207.05221.

Sanyam Kapoor, Nate Gruver, Manley Roberts, Katherine Collins, Arka Pal, Umang Bhatt, Adrian Weller, Samuel Dooley, Micah Goldblum, and Andrew Wilson. Large language models must be taught to know what they don't know. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Informātion Processing Systems, volume 37, pp. 85932–85972. Čurran Associates, Inc., 2024. doi: 10.52202/079017-2729. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/file/9c20f16b05f5e5e70fa07e2a4364b80e-Paper-Conference.pdf.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466, 2019. doi: 10.1162/tacl\_a\_00276. URL https://aclanthology.org/Q19-1026.

Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, and Chenguang Zhu. Geval: NLG evaluation using gpt-4 with better human alignment. In Houda Bouamor,

Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 2511–2522, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.153. URL https: //aclanthology.org/2023.emnlp-main.153/.

Andrey Malinin and Mark Gales. Uncertainty estimation in autoregressive structured prediction. In International Conference on Learning Representations, 2021. URL https: //openreview.net/forum?id=jN5y-zb5Q7m.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. When not to trust language models: Investigating effectiveness and limitations of parametric and non-parametric memories. arXiv preprint, 2022.

Oscar Mañas, Benno Krojer, and Aishwarya Agrawal. Improving automatic vqa evaluation using large language models. In Proceedings of the AAAī Conference on Artificial Intelligence, volume 38, pp. 4171–4179,2024. URL https://arxiv.org/pdf/2310.02567v2.

Meta AI. The llama 3 herd of models, 2024. URL https: //arxiv.org/abs/2407.21783.

Qwen Team. Qwen2.5 technical report, 2025a. URL https://arxiv.org/abs/2412.15115.

Qwen Team. Qwen3 technical report, 2025b. URL https://arxiv.org/abs/2505.09388.

Kayla Schroeder and Zach Wood-Doughty. Can you trust llm judgments? reliability of llm-as-a-judge, 2025. URL https://arxiv.org/abs/2412.12509.

Ola Shorinwa, Zhiting Mei, Justin Lidard, Allen Z. Ren, and Anirudha Majumdar. A survey on uncertainty quantification of large language models: Taxonomy, open research challenges, and future directions. ACM Comput. Surv., 58(3), September 2025. ISSN 0360-0300. doi: 10.1145/3744238. URL https://doi.org/10.1145/3744238.

Guijin Son, Hyunwoo Ko, Hoyoung Lee, Yewon Kim, and Seunghyeok Hong. Llm-as-ajudge and reward model: What they can and cannot do, 2024. URL https: //arxiv.org/ abs/2409.11239.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang Wang, Raluca Ada Popa, and Ion Stoica. Judgebench: A benchmark for evaluating llm-based judges, 2025. URL https://arxiv.org/abs/2410.12784.

Qingni Wang, Yue Fan, and Xin Wang. SAFER: Risk-constrained sample-then-filter in large language models. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust (eds.), International Conference on Learning Representations, volume 2026, pp. 53723–53747,2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/file/ 582e9771ac8527cb6390e5e9444a0fee-Paper-Conference.pdf.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.

Zhiyuan Wang, Jinhao Duan, Lu Cheng, Yue Zhang, Qingni Wang, Xiaoshuang Shi, Kaidi Xu, Heng Tão Shen, and Xiaofeng Zhu. ConU: Conformal uncertainty in large language models with correctness coverage guarantees. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 6886–6898, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.404. URL https://aclanthology. org/2024.findings-emnlp.404/.

Zhiyuan Wang, Jinhao Duan, Qingni Wang, Xiaofeng Zhu, Tianlong Chen, Xiaoshuang Shi, and Kaidi Xu. COIN: Uncertainty-guarding selective question answering for foundation models with provable risk guarantees, 2025a. URL https://arxiv.org/abs/2506.20178.

Zhiyuan Wang, Qingni Wang, Yue Zhang, Tianlong Chen, Xiaofeng Zhu, Xiaoshuang Shi, and Kaidi Xu. SConU: Selective conformal uncertainty in large language models. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 19052–19075, Vienna, Austria, July 2025b. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/2025.acl-long.934. URL https://aclanthology.org/2025.acl-long.934/.

Jerry Wei, Chengrun Yang, Xinying Song, Yifeng Lu, Nathan Hu, Jie Huang, Dustin Tran, Daiyi Peng, Ruibo Liu, Da Huang, Cosmo Du, and Quoc V. Le. Long-form factuality in large language models. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 80756–80827. Curran Associates, Inc., 2024. doi: 10. 52202/079017-2567. URL https://proceedings.neurips.cc/paper\_files/paper/2024/ file/937ae0e83eb08d2cb8627fe1def8c751-Paper-Conference.pdf.

Bingbing Wen, Jihan Yao, Shangbin Feng, Chenjun Xu, Yulia Tsvetkov, Bill Howe, and Lucy Lu Wang. Know your limits: A survey of abstention in large language models. Transactions of the Association for Computational Linguistics, 13:529–556, 06 2ō25. ISSN 2307-387X. doi: 10.1162/tacl\_a\_00754. URL https://doi.org/10.1162/tacl\_a\_00754.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multihop question answering, 2018. URL https://arxiv.org/abs/1809.09600.

Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Żhang, Rui Wang, and Gongshen Liu. R-judge: Benchmarking safety risk awareness for LLM agents. In Yaser Al-Onaizan, Mohit Bansal, and Yun-Nung Chen (eds.), Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 1467–1490, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.findings-emnlp.79. URL https://aclanthology.org/ 2024.findings-emnlp.79/.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Žhang, Joseph E Gonzalez, and Ion Stoica. Judging llm-as-a-judge with mt-bench and chatbot arena. In A. Oh, T. Naumann, A. Ġloberson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 46595–46623. Curran Associates, Inc., 2023. URL https://openreview.net/forum?id=uccHPGDlao.

## A Proofs

## A.1 Clopper-Pearson UCB Validity

We show that $\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t )$ in Eq. (7) satisfies $\operatorname* { P r } ( R ( t ) \leq \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) ) \geq 1 - \delta .$

Proof. Under the Bernoulli structure, the observed failure count among $\hat { m } _ { \mathrm { c a l } } ( t )$ selected instances follows $\hat { w } _ { \mathrm { c a l } } ( t ) \sim \mathrm { B i n o m i a l } ( \hat { m } _ { \mathrm { c a l } } ( t ) , R ( t ) )$ . Define the CDF of the empirical failure rate:

$$
D ( r \mid R ( t ) ) = \operatorname* { P r } \left( { \hat { R } } ( t ) \leq r \mid R ( t ) \right) ,\tag{16}
$$

where $\hat { R } ( t ) = \hat { w } _ { \mathrm { c a l } } ( t ) / \hat { m } _ { \mathrm { c a l } } ( t )$ . By definition:

$$
\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) = \operatorname* { s u p } \{ R \in [ 0 , 1 ] : D ( \hat { r } _ { \mathrm { c a l } } ( t ) \mid R ) \geq \delta \} .\tag{17}
$$

Since $D ( r \mid R )$ is monotonically decreasing in R (higher true rate makes low empirical rates less likely), $R \dot { ( t ) } > \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t )$ implies $D ( \hat { r } _ { \mathrm { c a l } } ( t ) \mid R ( t ) ) < \delta .$ Therefore:

$$
\operatorname* { P r } \Bigl ( R ( t ) > \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) \Bigr ) \le \operatorname* { P r } ( D ( \hat { r } _ { \mathrm { c a l } } ( t ) \mid R ( t ) ) < \delta ) \le \delta ,\tag{18}
$$

where the last step uses the super-uniformity property: $\operatorname* { P r } ( D ( \hat { r } _ { \mathrm { c a l } } ( t ) \mid R ( t ) ) < \delta ) \leq \delta$ for discrete distributions. Hence $\mathrm { P r } ( R ( t ) \leq \hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t ) ) \geq 1 - \delta .$ □

## A.2 Bernoulli Structure under Two-Mode Routing

Proof of Lemma 1. For fixed thresholds $\left( t _ { 1 } , t _ { 2 } \right)$ , we define the selection indicator:

$$
I _ { i } : = \mathbf { 1 } \Big \{ U _ { 1 } ^ { ( i ) } \leq t _ { 1 } \Big \} + \mathbf { 1 } \Big \{ U _ { 1 } ^ { ( i ) } > t _ { 1 } \mathrm { ~ a n d ~ } U _ { 2 } ^ { ( i ) } \leq t _ { 2 } \Big \} .\tag{19}
$$

Both $U _ { 1 } ^ { ( i ) } = U ( \mathcal { I } ( \mathbf { x } _ { i } , \hat { y } _ { i } ) )$ and $U _ { \gamma } ^ { ( i ) } = U ( \mathcal { I } ( \mathbf { x } _ { i } , \hat { y } _ { i } , \mathcal { E } _ { i } ) )$ are deterministic functions of $\left( \mathbf { x } _ { i } , \hat { y } _ { i } \right)$ the judge's computation and the retrieval process $\mathcal { E } _ { i }$ are both determined by the input. Hence ${ \breve { I } } _ { i }$ is a deterministic function of $\left( \mathbf { x } _ { i } , \hat { y } _ { i } \right)$

The correctness of the selected verdict is:

$$
\begin{array} { r l } & { c _ { i } = \mathbf { 1 } \Big \{ U _ { 1 } ^ { ( i ) } \leq t _ { 1 } \Big \} c _ { 1 } ^ { ( i ) } } \\ & { ~ + \mathbf { 1 } \Big \{ U _ { 1 } ^ { ( i ) } > t _ { 1 } \wedge U _ { 2 } ^ { ( i ) } \leq t _ { 2 } \Big \} c _ { 2 } ^ { ( i ) } , } \end{array}\tag{20}
$$

which is a deterministic function of $\left( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } \right)$ , since the routing decision depends on $\left( \mathbf { x } _ { i } , \hat { y } _ { i } \right)$ and correctness additionally depends on $y _ { i } ^ { * }$

Since $\left( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } \right)$ are i.i.d. from $\mathcal { D } \mathrm { : }$

1. Independence: The selection events $\{ I _ { i } = 1 \}$ depend on disjoint instances, so they are mutually independent. Conditioning on selection preserves independence across instances.

2. Identical distribution: Each selected instance satisfies $\left( \mathbf { x } _ { i } , \hat { y } _ { i } , y _ { i } ^ { * } \right) \sim \mathcal { D } _ { t _ { 1 } , t _ { 2 } }$ , where $\mathcal { D } _ { t _ { 1 } , t _ { 2 } } : = \mathcal { D } \ | \ I = 1$ is the conditional distribution.

Note that $I _ { i } \in \{ 0 , 1 \}$ because the two indicator terms are mutually exclusive: the second requires $U _ { 1 } ^ { ( i ) } > t _ { 1 } ,$ which precludes the first.

The failure indicators $\begin{array} { r c l } { W _ { i } } & { = } & { 1 - c _ { i } } \end{array}$ over the selected subset are therefore i.i.d. Bernoulli $\left( R ( t _ { 1 } , t _ { 2 } ) \right)$ , where $\ln ( t _ { 1 } , t _ { 2 } ) = \mathbb { E } [ W \mid I = 1 ]$ □

Scope and limitations. The guarantee is tied to the retrieval snapshot used during calibration. Because retrieval relies on a non-stationary web index, a query may return different evidence at deployment than it did at calibration. Such retrieval drift affects the guarantee only if it changes the conditional error rate of Mode 2 verdicts. As long as this error rate remains stable, the calibrated thresholds continue to transfer. We evaluate this scenario in Section 5.5 by re-querying the web several months after calibration and find that FDR control is preserved. If the quality of the retrieved evidence changes substantially, recalibration restores the guarantee. Likewise, any change to the task, candidate model, or judge model violates the exchangeability assumption and requires recalibration.

## B Experimental Details

## B.1 Notation

Table 4 provides a complete reference of all symbols.

## B.2 Implementation

Candidate generation. We generate candidate answers using greedy decoding for both Qwen3-8B and LLaMA-3.1-70B. For Qwen3-8B, thinking mode is disabled (enable\_thinking=False) to prevent the <think> block from consuming the generation budget.

Table 4: Notation reference.
<table><tr><td>Symbol</td><td>Description</td></tr><tr><td> $\boldsymbol { \mathbf { x } } , \boldsymbol { \hat { y } } , \boldsymbol { y } ^ { * }$ </td><td>Question, candidate answer, ground truth</td></tr><tr><td> $F , ~ \mathcal { T }$ </td><td>Examinee LLM, judge LLM</td></tr><tr><td> $v \in \{ 0 , 1 \}$ </td><td>Evaluation verdict</td></tr><tr><td> $e ^ { * } \in \{ 0 , \mathrm { i } \}$ </td><td>Ground-truth evaluation label</td></tr><tr><td> $\boldsymbol { \mathcal { A } } ( \boldsymbol { v } , \boldsymbol { \dot { e } } ^ { * } )$ </td><td>Admission function:  $\mathbf { 1 } [ v = e ^ { * } ]$ </td></tr><tr><td> $\mathcal { A } _ { \mathrm { r e f } } , \lambda _ { A }$ </td><td>Relevance function and threshold</td></tr><tr><td> $W$   $U , U _ { 1 } , U _ { 2 }$ </td><td>Failure indicator:  ${ \bf 1 } [ v \neq e ^ { * } ]$ </td></tr><tr><td></td><td>Uncertainty (general / Mode 1 / Mode 2)</td></tr><tr><td> $v _ { 1 } , \ v _ { 2 }$ </td><td>Verdict from Mode 1 / Mode 2</td></tr><tr><td> $c _ { 1 } , \ c _ { 2 }$ </td><td>Correctness of Mode 1 / Mode 2</td></tr><tr><td>3</td><td>Retrieved external evidence</td></tr><tr><td> $\mathcal { D } _ { \mathrm { c a l } }$ </td><td>Calibration set (N instances)</td></tr><tr><td> $\hat { m } _ { \mathrm { c a l } }$ </td><td>Selection size on calibration set</td></tr><tr><td> $\hat { w } _ { \mathrm { c a l } }$ </td><td>Error count on calibration set</td></tr><tr><td> $\hat { r } _ { \mathrm { c a l } }$ </td><td>Empirical FDR</td></tr><tr><td> $R ( t )$   ${ \hat { R } } ^ { \mathrm { u p p e r } }$ </td><td>True FDR</td></tr><tr><td> $\ ? 1 \overline { { - \delta } }$ </td><td>Clopper-Pearson UCB</td></tr><tr><td> $\hat { t } _ { 1 } , \ \hat { t } _ { 2 }$   $\alpha$ </td><td>Calibrated thresholds (Mode 1, 2) User-specified risk level</td></tr><tr><td> $\delta$ </td><td></td></tr><tr><td></td><td>Significance level (default 0.05)</td></tr><tr><td> $K$ </td><td>Sampling size for black-box UQ</td></tr><tr><td> $\mathrm { C o v e r a g e } ( t )$ </td><td>Coverage:  $| \{ i : U _ { i } \leq t \} | / N _ { \mathrm { t e s t } }$ </td></tr></table>

Judge inference. For predictive entropy (PE), we perform a single forward pass at the verdict token position and extract the log-probabilities for the True and False tokens. PE is computed as binary entropy: $- p _ { \mathrm { t r u e } } \log \bar { p _ { \mathrm { t r u e } } } - ( 1 - p _ { \mathrm { t r u e } } ) \log ( 1 - p _ { \mathrm { t r u e } } )$ . This requires no additional inference beyond the greedy decoding pass used to produce the verdict.

Retrieval. We query the Serper API with the original question and retrieve the top-k results. Each result contains a title, text snippet, and source URL. Results are concatenated into a single evidence passage appended to the Mode 2 judge prompt.

Calibration. We run 100 random 50/50 calibration/test splits. For each split, we perform a grid search over all unique uncertainty values on the calibration set to find the threshold pair $( \hat { t } _ { 1 } , \hat { t } _ { 2 } )$ that maximizes correctly accepted instances subject to the Clopper-Pearson UCB constraint $\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ( t _ { 1 } , t _ { 2 } ) \leq \alpha$ . The grid search uses cumulative sums for $O ( | T _ { 1 } | \cdot | T _ { 2 } | )$ complexity.

## B.3 Prompt Templates

## B.3.1 Judge Prompts

Mode 1 (parametric evaluation). The judge receives a system prompt and a user message containing the question and candidate answer. The prompt includes a one-shot example to establish the output format.

Mode 1: Judge System Prompt   
You are a helpful assistant acting as an impartial judge. You will be given a Question and a   
Candidate Answer. Your task is to judge whether the Čandidate Answer is correct using your   
own knowledge. If the Candidate Answer is correct, choose 'True', otherwise, choose 'False'.   
Provide a brief explanation for your decision.   
Provide your response in the following format:   
Decision: [True/False]   
Explanation: [Your brief explanation]   
Example:   
Question: What is the capital of France?   
Candidate Answer: Paris is the capital of France.   
Decision: True   
Explanation: Paris is indeed the capital of France.

![](images/c0011bafda3d7da186ac4bdef9ae92d2b18520e7018c7afa296cc9016bf01e6f.jpg)

Mode 2 (retrieval-augmented evaluation). The judge additionally receives retrieved web search results. Each result is formatted as: [i] Title / Snippet / Source: URL.  
Mode 2: Judge System Prompt   
You are a helpful assistant acting as an impartial judge. You will be given a Question, a   
Candidate Answer, and Web Search Results. Your task is to judge whether the Candidate   
Answer is correct using the provided search results. If the search results support the Candidate   
Answer, choose 'True'. If the search results contradict it, choose 'False'. If the search results   
are inconclusive, use your best judgment. Provide a brief explanation for your decision.   
Provide your response in the following format:   
Decision: [True/False]   
Explanation: [Your brief explanation]

## B.4 LLM Evaluator Prompt

The LLM evaluator (Qwen2.5-7B-Instruct) determines whether the candidate answer is semantically equivalent to any reference answer (Badshah & Sajjad, 2025), serving as the admission function for computing the ground-truth label $e ^ { * }$

## LLM Evaluator (Admission Function)

You are a helpful assistant acting as an impartial evaluator. Given a question, a candidate answer, and reference answers, determine if the candidate answer is correct.

The candidate answer is correct if it conveys the same meaning as ANY of the reference answers, even if the candidate answer is longer, contains additionaǐ explanation, uses synonyms, abbreviations, or alternate names. Focus on whether the final answer or conclusion matches a reference, not on surface form.

Question: {question}

Candidate Answer: {candidate\_answer}

Reference Answers: {references}

Is the candidate answer correct? Respond with ONLY “True" or “False".

## C Additional Results

This appendix provides additional results that complement the main paper, including raw judge accuracy, an analysis of the joint framework across different risk levels, and per-dataset coverage and FDR curves.

## C.1 Judge Verdict Accuracy

Table 5 reports the raw verdict accuracy of each judge in Mode 1 (parametric) and Mode 2 (retrieval-augmented) before any calibration or selective prediction. Retrieval consistently improves accuracy across all configurations, with the largest gains on PopQA (up to 44 percentage points), showing that retrieval provides useful evidence.

Table 5: Judge verdict accuracy (%) before calibration. Mode 1 uses parametric knowledge only whereas Mode 2 uses retrieval-augmented evaluation (k = 3). LLM evaluator as admission function.
<table><tr><td colspan="2"></td><td colspan="2">TriviaQA</td><td colspan="2">NQ-Open</td><td colspan="2">PopQA</td><td colspan="2">HotpotQA</td></tr><tr><td>Cand.</td><td>Judge</td><td>M1</td><td>M2</td><td>M1</td><td>M2</td><td>M1</td><td>M2</td><td>M1</td><td>M2</td></tr><tr><td></td><td>Qwen-4B</td><td>68</td><td>79</td><td>52</td><td>71</td><td>47</td><td>84</td><td>54</td><td>69</td></tr><tr><td>ww-8B</td><td>Qwen-8B</td><td>71</td><td>84</td><td>54</td><td>79</td><td>44</td><td>88</td><td>58</td><td>76</td></tr><tr><td></td><td>Qwen-14B</td><td>78</td><td>86</td><td>61</td><td>77</td><td>65</td><td>88</td><td>67</td><td>78</td></tr><tr><td></td><td>Llama-8B</td><td>77</td><td>82</td><td>63</td><td>78</td><td>68</td><td>88</td><td>68</td><td>78</td></tr><tr><td></td><td>Qwen-4B</td><td>75</td><td>81</td><td>62</td><td>69</td><td>60</td><td>75</td><td>63</td><td>68</td></tr><tr><td></td><td>Qwen-8B</td><td>78</td><td>81</td><td>63</td><td>71</td><td>61</td><td>77</td><td>65</td><td>69</td></tr><tr><td>L1am0B</td><td>Qwen-14B</td><td>78</td><td>82</td><td>64</td><td>72</td><td>66</td><td>77</td><td>67</td><td>72</td></tr><tr><td></td><td>Llama-8B</td><td>72</td><td>78</td><td>63</td><td>70</td><td>64</td><td>77</td><td>64</td><td>68</td></tr></table>

## C.2 FDR and Coverage Across Risk Levels

Table 6 reports FDR and coverage for the joint two-mode framework (two-threshold routing with retrieval) at all evaluated α levels for the Qwen-8B candidate across all four datasets and all four judges (LLM evaluator, PE, k =3).

## C.2.1 Coverage Curves

Figure 3 shows coverage as a function of α across all 32 configurations.

## C.2.2 Per-Dataset FDR

Figures 4–7 provide per-dataset FDR plots for detailed examination.

Table 6: Full results for the joint two-mode framework at all α levels across all four datasets. Candidate: Qwen3-8B, LLM evaluator, PE, k = 3.
<table><tr><td colspan="10"></td><td colspan="4"></td></tr><tr><td>Judge</td><td>Metric</td><td>.05</td><td>.10</td><td>.15</td><td>.20 .25</td><td></td><td>.30</td><td>.35</td><td>.40</td><td>.45</td><td>.50</td><td>.55</td><td>.60</td></tr><tr><td colspan="10">TriviaQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-4B</td><td>FDR Cov.</td><td>.03 .03</td><td>.08 .39</td><td>.13 .66</td><td>.18 .91</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td><td>.21 1.0</td></tr><tr><td>Qwen-8B</td><td>FDR</td><td>.03</td><td>.08</td><td>.14</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td><td>.16</td></tr><tr><td>Qwen-14B</td><td>Cov. FDR</td><td>.12 .04</td><td>.63 .09</td><td>.91 .13</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td><td>1.0 .14</td></tr><tr><td></td><td>Cov. FDR</td><td>.21</td><td>.78</td><td>.98</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Llama-8B</td><td>Cov.</td><td>.03 .13</td><td>.09 .61</td><td>.14 .86</td><td>.18</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td><td>.18 1.0</td></tr><tr><td colspan="10">1.0 NQ-Open</td><td></td><td></td><td></td><td></td></tr><tr><td>FDR .00 Qwen-4B</td><td>Cov.</td><td>.00</td><td>.05 .03</td><td>.11 .08</td><td>.17 .14</td><td>.22 .28</td><td>.27 .93</td><td>.29 1.0</td><td>.29 1.0</td><td>.29 1.0</td><td>.29 1.0</td><td>.29 1.0</td><td>.29 1.0</td></tr><tr><td>Qwen-8B</td><td>FDR Cov.</td><td>.00 .00</td><td>.05 .02</td><td>.13 .24</td><td>.18 .82</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td><td>.22 1.0</td></tr><tr><td>Qwen-14B</td><td>FDR Cov.</td><td>.00 .00</td><td>.05 .05</td><td>.13 .56</td><td>.18 .81</td><td>.22 .99</td><td>.23</td><td>.23</td><td>.23 1.0</td><td>.23 1.0</td><td>.23 1.0</td><td>.23 1.0</td><td>.23 1.0</td></tr><tr><td>Llama-8B</td><td>FDR</td><td>.00</td><td>.09</td><td>.13</td><td>.18</td><td>.22</td><td>1.0 .22</td><td>1.0 .22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td></tr><tr><td></td><td>Cov.</td><td>.00</td><td>.20</td><td>.61</td><td>.83</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td colspan="10">PopQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10"></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen-4B</td><td>FDR Cov.</td><td>.00 .00</td><td>.01 .01</td><td>.13 .91</td><td>.16 1.0</td><td>.16</td><td>.16</td><td>.16 1.0</td><td>.16 1.0</td><td>.16 1.0</td><td>.16 1.0</td><td>.16 1.0</td><td>.16 1.0</td></tr><tr><td>Qwen-8B</td><td>FDR</td><td>.00</td><td>.09</td><td>.12</td><td>.12</td><td>1.0 .12</td><td>1.0 .12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td></tr><tr><td></td><td>Cov.</td><td>.00</td><td>.87</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Qwen-14B</td><td>FDR</td><td>.02</td><td>.09</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td><td>.12</td></tr><tr><td></td><td>Cov. FDR</td><td>.19</td><td>.89</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Llama-8B</td><td>Cov.</td><td>.00</td><td>.09</td><td>.12</td><td>.13</td><td>.13</td><td>.13</td><td>.13</td><td>.13</td><td>.13 1.0</td><td>.13 1.0</td><td>.13 1.0</td><td>.13 1.0</td></tr><tr><td></td><td></td><td>.02</td><td>.88</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">HotpotQA</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="10">FDR .00 .05</td><td></td><td>.31</td><td></td><td>.31</td><td>.31</td></tr><tr><td>Qwen-4B</td><td>Cov.</td><td>.00</td><td>.04</td><td>.10 .09</td><td>.15 .14</td><td>.21 .30</td><td>.27 .58</td><td>.31 .86</td><td>.31 .86</td><td>.31 .86</td><td>.86</td><td>.86</td><td>.86</td></tr><tr><td>Qwen-8B</td><td>FDR Cov.</td><td>.00</td><td>.05</td><td>.13</td><td>.18</td><td>.23</td><td>.24</td><td>.24</td><td>.24</td><td>.24</td><td>.24</td><td>.24</td><td>.24</td></tr><tr><td></td><td>FDR</td><td>.00 .02</td><td>.04 .08</td><td>.49 .13</td><td>.74 .18</td><td>.96 .22</td><td>1.0 .22</td><td>1.0 .22</td><td>1.0 .22</td><td>1.0 .22</td><td>1.0 .22</td><td>1.0 .22</td><td>1.0 .22</td></tr><tr><td>Qwen-14B</td><td>Cov.</td><td>.02</td><td>.26</td><td>.64</td><td>.86</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Llama-8B</td><td>FDR</td><td>.01</td><td>.08</td><td>.13</td><td>.18</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td><td>.22</td></tr><tr><td></td><td>Cov.</td><td>.01</td><td>.30</td><td>.65</td><td>.85</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr></table>

## D Ablations and Robustness

We evaluate the framework across several dimensions. This includes pipeline design choices, the calibration procedure, the uncertainty score, and the choice of no-retrieval baseline. In all cases, the finite-sample FDR guarantee is preserved, with conservative choices reducing coverage rather than compromising risk control.

## D.1 Sensitivity Analyses

Correctness criterion (admission function). Table 7 compares three admission functions across three risk levels on a representative pair (Qwen3-8B candidate, Qwen3-14B judge). The LLM evaluator consistently achieves the highest coverage, with the gap most pronounced on knowledge-intensive benchmarks: on NQ-Open at $\alpha = 0 . 1 5 ,$ it reaches 56% while both exact match and token F1 remain at 0%. This gap arises because string-matching criteria penalize semantically correct but surface-different answers, inflating the judge's apparent error rate. FDR control holds under all three criteria.

![](images/4ba3a026e8e40cfc2dc15c9dc36a03570bdfa38dcf84bc33fb76f56ac2b8418c.jpg)

Figure 3: Coverage (mean±std over 100 splits) across all 32 configurations. Rows: candidates; columns: datasets.  
![](images/303180daecd5f43e1d02f94192af0c18c17c54649d973c1a248c7b7862cee4f5.jpg)  
Figure 4: FDR control on TriviaQA.

![](images/57705c54edb8b6a9c87f02e9be23e5e95a3638028dce2086ebc9d119669621a6.jpg)  
Figure 5: FDR control on NQ-Open.

![](images/5150eb1d140758535fd2553c3aef37be04aa4a481194aee475ad5611e0e63cb8.jpg)  
Figure 6: FDR control on PopQA.

![](images/3710de1bf9304759dbc6da965ddf6027f86bb84ca0f686bc5d4d3ba7b9c00874.jpg)  
Figure 7: FDR control on HotpotQA.

Table 7: Sensitivity to admission function. Candidate: Qwen3-8B, Judge: Qwen3-14B, UQ: PE (white-box). FDR and coverage at α ∈ {.15, .20, .25}.
<table><tr><td></td><td colspan="6">Exact Match</td><td colspan="6">Token F1 FDR</td><td colspan="6">LLM Evaluator</td></tr><tr><td>Dataset</td><td colspan="2">FDR</td><td colspan="2">.25</td><td colspan="2">Coverage .20</td><td colspan="2">.15</td><td>.25</td><td colspan="2">Coverage</td><td colspan="2"></td><td colspan="2">FDR</td><td colspan="2">Coverage</td></tr><tr><td></td><td>.15</td><td>.20</td><td></td><td>.15</td><td></td><td>.25</td><td>.20</td><td></td><td>.15</td><td>.20</td><td>.25</td><td>.15</td><td>.20</td><td>.25</td><td>.15</td><td>.20</td><td>.25</td></tr><tr><td>TriviaQA</td><td>.13</td><td>.14</td><td>.14</td><td>.97</td><td>1.0</td><td>1.0</td><td>.13</td><td>.18</td><td>.18 .72</td><td>.98</td><td>1.0</td><td>.13</td><td>.14</td><td>.14</td><td>.98</td><td>1.0</td><td>1.0</td></tr><tr><td>NQ-Open</td><td>.00</td><td>.05</td><td>.22</td><td>.00</td><td>.11</td><td>.69</td><td>.00 .00</td><td>.18</td><td>.00</td><td>.00</td><td>.45</td><td>.13</td><td>.18</td><td>.22</td><td>.56</td><td>.81</td><td>.99</td></tr><tr><td>PopQA</td><td>.12</td><td>.12</td><td>.12</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.13 .13</td><td>.13</td><td>.99</td><td>1.0</td><td>1.0</td><td>.12</td><td>.12</td><td>.12</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>HotpotQA</td><td>.04</td><td>.18</td><td>.23</td><td>.03</td><td>.57</td><td>.86</td><td>.04</td><td>.17 .23</td><td>.04</td><td>.47</td><td>.83</td><td>.13</td><td>.18</td><td>.22</td><td>.64</td><td>.86</td><td>1.0</td></tr></table>

Retrieval depth. Table 8 ablates the number of retrieved passages $( k \in \{ 3 , 6 , 9 \} )$ on a representative pair (Llama-70B candidate, Qwen3-8B judge). Coverage and FDR are stable across retrieval depths, with differences within 1–2 percentage points. k = 3 already provides sufficient evidence.

Confidence level δ. Table 9 varies the confidence level. Tightening δ from 0.05 to 0.01 trades coverage for confidence, with the cost concentrated at low α and on the harder dataset; FDR stays controlled throughout.

Table 8: Retrieval depth ablation at $\alpha { = } 0 . 2 0 .$ Candidate: Llama-70B, Judge: Qwen3-8B, LLM evaluator.
<table><tr><td></td><td colspan="2"> $k = 3$ </td><td colspan="2"> $k = 6$ </td><td colspan="2"> $k = 9$ </td></tr><tr><td>Dataset</td><td>FDR</td><td>Cov.</td><td>FDR</td><td>Cov.</td><td>FDR</td><td>Cov.</td></tr><tr><td>TriviaQA</td><td>.18</td><td>.99</td><td>.17</td><td>1.0</td><td>.17</td><td>.99</td></tr><tr><td>NQ-Open</td><td>.18</td><td>.47</td><td>.18</td><td>.46</td><td>.18</td><td>.46</td></tr><tr><td>PopQÃ</td><td>.18</td><td>.48</td><td>.18</td><td>.41</td><td>.17</td><td>.36</td></tr><tr><td>HotpotQA</td><td>.18</td><td>.47</td><td>.18</td><td>.46</td><td>.18</td><td>.46</td></tr></table>

Table 9: Sensitivity to the confidence level δ (coverage). Candidate: Qwen3-8B, Judge: Qwen3-8B, LLM evaluator, k =3, PE. FDR is controlled at the target α under both settings.
<table><tr><td></td><td colspan="5">TriviaQA</td><td colspan="5">HotpotQA</td></tr><tr><td> $\delta \setminus \alpha$ </td><td>.05</td><td>.10</td><td>.15</td><td>.20</td><td>.25</td><td>.05</td><td>.10</td><td>1.15</td><td>.20</td><td>.25</td></tr><tr><td>0.05</td><td>.12</td><td>.63</td><td>.91</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.49</td><td>.74</td><td>.96</td></tr><tr><td>0.01</td><td>.04</td><td>.53</td><td>.87</td><td>1.0</td><td>1.0</td><td>.00</td><td>.01</td><td>.37</td><td>.69</td><td>.93</td></tr></table>

Calibration-set size. Table 10 sweeps the calibration fraction $n _ { \mathrm { c a l } } / n$ over the 2,000 items per dataset. Coverage on the easier TriviaQA is stable across the range, while on HotpotQA it degrades gracefully as the calibration set shrinks; FDR is controlled at every size.

Table 10: Sensitivity to calibration-set size (coverage). $n _ { \mathrm { c a l } } / n$ is the fraction of the 2,000 items used for calibration. Candidate: Qwen3-8B, Judge: Qwen3-8B, LLM evaluator, k =3, PE. FDR is controlled at the target α for every size.
<table><tr><td></td><td colspan="5">TriviaQA</td><td colspan="5">HotpotQA</td></tr><tr><td> $n _ { \mathrm { c a l } } / n \setminus \alpha$ </td><td>.05</td><td>.10</td><td>.15</td><td>.20</td><td>.25</td><td>.05</td><td>.10</td><td>.15</td><td>.20</td><td>.25</td></tr><tr><td>0.10</td><td>.02</td><td>.38</td><td>.81</td><td>.96</td><td>.99</td><td>.00</td><td>.03</td><td>.25</td><td>.55</td><td>.83</td></tr><tr><td>0.25</td><td>.08</td><td>.54</td><td>.89</td><td>.99</td><td>1.0</td><td>.00</td><td>.04</td><td>.41</td><td>.70</td><td>.92</td></tr><tr><td>0.50</td><td>.12</td><td>.63</td><td>.91</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.49</td><td>.74</td><td>.96</td></tr><tr><td>0.75</td><td>.15</td><td>.66</td><td>.92</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.53</td><td>.77</td><td>.98</td></tr></table>

## D.2 Alternative Uncertainty Estimators

The FDR guarantee depends only on the binary correctness indicator. As a result, the choice of uncertainty score affects coverage but not whether the target risk level is controlled. In addition to the white-box predictive entropy (PE) used in the main paper, we evaluate two supervised uncertainty scores derived from the judge's last-layer hidden state at the verdict position. Following Kapoor et al. (2024), we consider a small MLP (Probe) and a LoRA+Prompt adapter fine-tuned to predict verdict correctness.

Both models are trained using 5-fold cross-validation on the calibration set so that every calibration instance receives an out-of-sample uncertainty score before conformal calibration. Table 11 reports the resulting coverage. All three uncertainty scores satisfy the target FDR level across every setting.

LoRA+Prompt consistently achieves the highest coverage, particularly on datasets where PE is less informative. For example, coverage increases from 0.04 to 0.44 on HotpotQA at $\alpha { = } 0 . 1 0$ and from 0.00 to 0.71 on PopQA at $\stackrel { \smile } { \alpha } = 0 . 0 5$ . The Probe also improves over PE on more challenging datasets at moderate risk levels (e.g., NQ-Open at $\alpha { \stackrel { . } { = } } 0 . 1 5 , 0 . 2 4  0 . 4 2 )$ although it performs worse at the lowest risk levels.

## D.3 Baseline Comparison

To isolate the contribution of retrieved evidence, Table 12 compares our joint two-mode policy against three controls that keep the identical calibration but change what the accepted verdict is based on. Direct accepts only Mode 1 (no retrieval). Self-eval adds a second Mode 1 pass in which the judge is asked whether its own initial verdict is correct, using no new evidence. Empty runs the Mode 2 prompt with an empty evidence slot. FDR is controlled for every variant. The joint policy dominates all three at nearly every operating point, and the gap is largest on the knowledge-intensive HotpotQA (0.74 versus at most 0.07 for the no-retrieval controls at $\alpha { = } 0 . 2 0 )$ . A second evaluation pass without new evidence (Self-eval) barely helps, confirming that the coverage gains come from the retrieved evidence rather than from simply re-querying the judge.

Table 11: Coverage under three uncertainty scores: the white-box predictive entropy (PE) used in the main paper and two supervised scores, a Probe and a LoRA+Prompt adapter on the judge's hidden states. Candidate: Qwen3-8B, Judge: Qwen3-8B, LLM evaluator, k= 3. FDR is controlled at the target α for all three.
<table><tr><td rowspan="2">α</td><td colspan="3">TriviaQA</td><td colspan="3">NQ-Open</td><td colspan="3">PopQA</td><td colspan="3">HotpotQA</td></tr><tr><td>PE</td><td>Probe</td><td>LoRA</td><td>PE</td><td>Probe</td><td>LoRA</td><td>PE</td><td>Probe</td><td>LoRA</td><td>PE</td><td>Probe</td><td>LoRA</td></tr><tr><td>.05</td><td>.12</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.02</td><td>.00</td><td>.31</td><td>.71</td><td>.00</td><td>.00</td><td>.02</td></tr><tr><td>.10</td><td>.63</td><td>.14</td><td>.70</td><td>.02</td><td>.07</td><td>.28</td><td>.87</td><td>.82</td><td>.94</td><td>.04</td><td>.02</td><td>.44</td></tr><tr><td>.15</td><td>.91</td><td>.90</td><td>.94</td><td>.24</td><td>.42</td><td>.64</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.49</td><td>.47</td><td>.76</td></tr><tr><td>.20</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.82</td><td>.79</td><td>.86</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.74</td><td>.78</td><td>.95</td></tr><tr><td>.25</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.0</td><td>.96</td><td>.99</td><td>1.0</td></tr></table>

Table 12: Baseline comparison (coverage). Candidate: Qwen3-8B, Judge: Qwen3-8B, LLM evaluator, k =3, PE. FDR is controlled āt the target α for every variant.
<table><tr><td></td><td colspan="5">TriviaQA</td><td colspan="5">HotpotQA</td></tr><tr><td>Variant \α</td><td>.05</td><td>.10</td><td>.15</td><td>.20</td><td>.25</td><td>.05</td><td>.10</td><td>.15</td><td>.20</td><td>.25</td></tr><tr><td>Direct</td><td>.03</td><td>.28</td><td>.47</td><td>.59</td><td>.74</td><td>.00</td><td>.00</td><td>.01</td><td>.05</td><td>.17</td></tr><tr><td>Self-eval</td><td>.02</td><td>.32</td><td>.49</td><td>.60</td><td>.75</td><td>.00</td><td>.00</td><td>.02</td><td>.07</td><td>.19</td></tr><tr><td>Empty</td><td>.04</td><td>.38</td><td>.63</td><td>.83</td><td>.99</td><td>.00</td><td>.09</td><td>.32</td><td>.57</td><td>.84</td></tr><tr><td>Joint (ours)</td><td>.12</td><td>.63</td><td>.91</td><td>1.0</td><td>1.0</td><td>.00</td><td>.04</td><td>.49</td><td>.74</td><td>.96</td></tr></table>

## D.4 Conservative Joint Calibration

We selected the deployed threshold pair $( \hat { t } _ { 1 } , \hat { t } _ { 2 } )$ using the calibration data rather than fixing them beforehand. Specifically, we choose the pair that maximizes coverage while satisfying $\hat { R } _ { 1 - \delta } ^ { \mathrm { u p p e r } } ~ \leq ~ \alpha ~ ( \mathrm { E q . } ~ 1 4 )$ Although the Clopper-Pearson bound holds for each candidate pair on its own, certifying the selected pair requires accounting for this search. The conservative variant addresses this directly by evaluating all $| \mathcal { T } _ { 1 } | \ | \mathcal { T } _ { 2 }$ candidate pairs using the corrected confidence level $\delta ^ { \prime } = \delta / ( | \mathbf { \bar { \mathcal { T } } } _ { 1 } | ^ { \mathit { \prime } } | \mathcal { T } _ { 2 } | )$ . By a union-bound argument, the guarantee then holds simultaneously for every candidate pair, implying that the selected pair satisfies $\operatorname* { P r } ( R ( { \hat { t } } _ { 1 } , { \hat { t } } _ { 2 } ) \leq \alpha ) \geq 1 { \overset { \cdot } { - } } \delta .$ In practice, however, we find that the simpler uncorrected selection already provides effective risk control (see Section 5). For this reason, we use the coverage-maximizing thresholds in our main experiments, while treating the conservative procedure as a theoretically valid alternative.

One subtlety is that the grid is data-dependent, since $\mathcal { T } _ { 1 }$ and $\mathcal { T } _ { 2 }$ are the observed uncertainty values, making both the candidate pairs and their count random. Because each mode contributes at most N distinct thresholds, replacing $\vert \mathcal { T } _ { 1 } \vert \vert \mathcal { T } _ { 2 } \vert$ with the data-independent bound $N ^ { 2 }$ (equivalently, correcting at level $\bar { \delta } / N ^ { 2 } )$ makes the union bound hold over a fixed family and leaves the conclusion unchanged at negligible cost; pre-specifying the grid removes the dependence altogether.

Table 13 compares the coverage of the main-paper calibration (Pointwise, per-pair level δ) with this Conservative variant (grid-corrected level δ'). FDR is controlled under both. The correction is nearly free on the easier datasets (TriviaQA, PopQA) but costs substantial coverage on the harder ones (NQ-Open, HotpotQA), where the empirical FDR sits close to α. Bonferroni is a worst-case union bound and adjacent grid thresholds select almost the same instances, so it is very loose. The truly attainable jointly-valid coverage lies between the two rows which is much closer to the pointwise figures.

Table 13: Conservative joint calibration (coverage). Per dataset, Point. and Cons. denote the pointwise and conservative coverage, and $\Delta \stackrel { \smile } { = } \mathrm { C o n s . - P o i n t . }$ (computed on unrounded values). Candidate: Qwen3-8B, Judge: Qwen3-8B, LLM evaluator, k = 3, PE. FDR is controlled at the target α under both variants.
<table><tr><td></td><td colspan="3">TriviaQA</td><td colspan="3">NQ-Open</td><td colspan="3">PopQA</td><td colspan="3">HotpotQA</td></tr><tr><td>α</td><td>Point.</td><td>Cons.</td><td>∆</td><td>Point.</td><td>Cons.</td><td>∆</td><td>Point.</td><td>Cons.</td><td>∆</td><td>Point.</td><td>Cons.</td><td>Δ</td></tr><tr><td>.05</td><td>.12</td><td>.00</td><td>-.12</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td><td>.00</td></tr><tr><td>.10</td><td>.63</td><td>.01</td><td>-.62</td><td>.02</td><td>.00</td><td>-.02</td><td>.87</td><td>.00</td><td>-.87</td><td>.04</td><td>.00</td><td>-.04</td></tr><tr><td>.15</td><td>.91</td><td>.64</td><td>-.27</td><td>.24</td><td>.00</td><td>-.24</td><td>1.0</td><td>.93</td><td>-.07</td><td>.49</td><td>.00</td><td>-.49</td></tr><tr><td>.20</td><td>1.0</td><td>.93</td><td>-.07</td><td>.82</td><td>.05</td><td>-.76</td><td>1.0</td><td>1.0</td><td>.00</td><td>.74</td><td>.30</td><td>-.45</td></tr><tr><td>.25</td><td>1.0</td><td>1.0</td><td>.00</td><td>1.0</td><td>.80</td><td>-.20</td><td>1.0</td><td>1.0</td><td>.00</td><td>.96</td><td>.72</td><td>-.24</td></tr></table>

## E Analysis

This section moves beyond aggregate metrics to analyze how retrieval improves evaluation, including its impact on judge uncertainty, examples where it corrects incorrect verdicts, and the residual failure modes.

## E.1 Effect of Retrieval on Uncertainty

Retrieval does not lower the judge's uncertainty uniformly. Table 14 reports, per dataset, the fraction of items on which Mode 2 uncertainty is higher than Mode 1 $( U _ { 2 } > U _ { 1 } )$ versus lower, with the mean signed change $\Delta U = U _ { 2 } { \stackrel { \cdot } { - } } U _ { 1 }$ within each group. On most items retrieval reduces uncertainty, but on 14–29% it raises it, concentrated on the multi-hop HotpotQA and NQ-Open, where retrieved snippets are only partially relevant or sources disagree. These are exactly the items the second threshold $\hat { t } _ { 2 }$ routes to abstention when retrieval fails to resolve the judge's uncertainty.

Table 14: Effect of retrieval on judge uncertainty. Results are obtained with Qwen3-8B as both candidate and judge, the LLM evaluator, and $k = 3$ retrieval using predictive entropy (PE). The "raised" and "lowered" columns report the fraction of instances where retrieval increases or decreases uncertainty, respectively. Mean ∆U is averaged within each group.
<table><tr><td>一</td><td>TriviaQA</td><td>NQ-Open</td><td>PopQA</td><td>HotpotQA</td></tr><tr><td>% raised  $( U _ { 2 } > U _ { 1 } )$ </td><td>20.1</td><td>27.9</td><td>14.3</td><td>29.1</td></tr><tr><td rowspan="2">mean ΔU % lowered  $( U _ { 2 } < U _ { 1 } )$ </td><td>+0.10</td><td>+0.11</td><td>+0.10</td><td>+0.12</td></tr><tr><td>68.0</td><td>68.8</td><td>84.6</td><td>69.7</td></tr><tr><td>mean ∆U</td><td>-0.07</td><td>-0.09</td><td>-0.10</td><td>-0.12</td></tr></table>

## E.2 Qualitative Examples

Tables 15 and 16 show representative cases where the judge issues an incorrect verdict in Mode 1 but corrects it after retrieval in Mode 2. In each case, Mode 1 confidently hallucinated while retrieved evidence grounded the judge in verifiable facts.

## E.3 Failure Modes

When retrieval does not fix an incorrect Mode 1 verdict, the residual errors fall into four qualitative classes. (i) Off-topic retrieval: an ambiguous or under-specified question yields search queries that miss the relevant evidence. (ii) Insufficient evidence: the retrieved snippets are only partially relevant and lack the specific fact needed to verify the candidate. (iii) Judge reasoning error: the snippets contain the relevant evidence, but the judge still returns a wrong verdict. For example defaulting to False in the absence of an exact lexical match. (iv) Admission function disagreement: the candidate is semantically close to the gold answer but the admission function scores it as different. The two-threshold policy sends many such cases to abstention rather than accepting them, which is why coverage rather than FDR absorbs the difficulty on the hardest datasets.

Table 15: Examples where retrieval corrects the judge's verdict. Candidate: Qwen3-8B, Judge: Qwen3-14B.
<table><tr><td>Question</td><td>Candidate</td><td>Gold</td><td>Mode 1 (Wrong)</td><td>Mode 2 (Corrected)</td></tr><tr><td>Who voices Jack Kahuna Laguna in SpongeBob SquarePants?</td><td>Steve Carell</td><td>Johnny Depp</td><td>&quot;Steve Carell provided the voice... This is a well-documented fact.&quot; Verdict: True</td><td>&quot;The web search results indicate that Johnny Depp, not Steve Carell, provided the voice... Verdict: False</td></tr><tr><td>Which North African country&#x27;s flag is just the colour green?</td><td>Chad</td><td>Libya</td><td>&quot;The flag of Chad is indeed a single solid green rectangle... Verdict: True</td><td>&quot;The search results indicate that Libya&#x27;s flag was a single green field...&#x27; Verdict: False</td></tr><tr><td>&#x27;The Great American Chocolate Bar&#x27; is better known as?</td><td>Milky Way</td><td>Hershey</td><td>&quot;The Great American Chocolate Bar is indeed known as the Milky Way bar...&quot; Verdict: True</td><td>&quot;The search results indicate that it is a nickname for the Hershey Milk Chocolate Bar... Verdict: False</td></tr></table>

Table 16: Retrieved evidence (k =3) for the examples in Table 15. The first result in each case directly contains the correct answer.
<table><tr><td>Question</td><td>#</td><td>Retrieved Snippet</td></tr><tr><td rowspan="4">Who voices Jack Kahuna Laguna in SpongeBob?</td><td>1</td><td>“... American actor and musician Johnny Depp guest starred in the episode as the voice of Jack</td></tr><tr><td></td><td>Kahuna Laguna, a surf guru.. . &quot; — Wikipedia</td></tr><tr><td>2 3</td><td>&quot;Jack Kahuña Laguna hās a striking resemblance to his voice actor, Johnny Depp.&quot; — Fandom &quot;. . . it guest starred Johnny Depp as the legendary surfer, Jack Kahuna Laguna.&quot; — IMDb</td></tr><tr><td></td><td></td></tr><tr><td rowspan="4">Which N. African country&#x27;s flag is just green?</td><td>1</td><td>&quot;The flag of the Libyan Arab Jamahiriya.. . consisted of a simple green field. It was the only national flag. . . with only one colour.&quot; — Wikipedia</td></tr><tr><td></td><td>&quot;Republic of Ireland and the Ivory Coast the flags are the same.. .&quot; — Facebook</td></tr><tr><td>23</td><td>&quot;Green is an important colour in Islam... &quot; —Reddit</td></tr><tr><td>1</td><td>&quot;The classic HERSHEY&#x27;S Milk Chocolate bar has been referred to as &#x27;The Great American</td></tr><tr><td rowspan="4">&#x27;The Great American Chocolate Bar&#x27; is better known as?</td><td></td><td>Chocolate Bar.&#x27; &quot; — Hershey</td></tr><tr><td></td><td>&quot;Hershey refers to it as &#x27;The Ġreat American Chocolate Bar&#x27;. First sold in 1900.&quot; — Wikipedia</td></tr><tr><td>23</td><td>&quot;The Great American Chocolate Bar campaign served the company well...&quot; — Hershey</td></tr><tr><td></td><td>Archives</td></tr></table>