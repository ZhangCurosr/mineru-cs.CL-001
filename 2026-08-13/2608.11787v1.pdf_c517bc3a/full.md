# GRPO for Financial Advice Generation: Outperforming Commercial LLMs under CATE Evaluation

Ofir Ben Shoham, Shrutendra Harsola, Vignesh Subrahmaniam, Shravan Mohan, Yakov Gazman, Oded Vainas

Intuit

{ofir\_benshoham, shrutendra\_harsola, vignesh\_subrahmaniam, shravan\_mohan, yakov\_gazman, oded\_vainas}@intuit.com

## Abstract

Generating actionable financial advice from business records demands that models integrate numerical reasoning, domain knowledge, and sound judgment, while avoiding recommendations that could harm the business. Direct supervision is difficult: historical decisions are not necessarily optimal, and high-quality free-form labels are expensive to obtain. We formulate financial advice generation as a reinforcement learning problem and fine-tune an open-weight language model using Group Relative Policy Optimization (GRPO). Our reward is an LLMas-a-judge rubric that scores each recommendation across multiple binary dimensions of advice quality, augmented with a safety gate for harm prevention.

Since LLM-based evaluation alone cannot confirm whether improvements reflect genuine business value rather than adaptation to the judge, we complement it with a judgeindependent audit based on a standard doublyrobust Conditional Average Treatment Effect (CATE) estimator. Under this observational off-policy audit, our trained LLM achieves approximately twice the estimated gross-profit lift of the strongest evaluated commercial baseline (0.0228 vs. 0.0104), together with the lowest downside rate and the least negative tail risk of any policy evaluated. Notably, the two evaluations do not rank the baselines identically: the untrained base model places last on the judge rubric but second on the causal audit, indicating that the audit captures a signal the judge does not. Our results demonstrate that GRPO with a finance-grounded reward signal can produce substantially more useful business recommendations than commercial LLMs, and that a judge-independent causal audit is a valuable complement to, rather than a confirmation of, LLM-as-a-judge assessment in financial NLP.

## 1 Introduction

Businesses generate large volumes of financial records, yet translating those records into concrete, actionable advice remains a difficult problem. A useful recommendation must be grounded in the specific numbers of the business, aligned with its financial goals, realistically actionable, and safe to implement. These requirements make financial advice generation challenging: outputs are openended, quality is multidimensional, and a bad recommendation carries real business risk.

Supervised approaches face a fundamental obstacle: historical business decisions are not optimal gold labels. In addition, collecting high-quality free-form annotations from domain experts is expensive and does not scale. In contrast, Reinforcement Learning offers a natural alternative: instead of imitating past decisions, a model can be trained directly to maximize a reward that captures the properties we care about (Kaelbling et al., 1996).

Therefore, we formulate financial advice generation as an RL problem and fine-tune an openweight language model using Group Relative Policy Optimization (GRPO) (Shao et al., 2024). Following the RLAIF paradigm (Lee et al., 2023), our reward is provided by an LLM-as-a-judge rubric that scores each recommendation across multiple binary dimensions of advice quality, augmented with a safety gate for harm prevention.

A key concern with judge-based training is reward hacking: a model may learn to satisfy the judge without producing recommendations that are genuinely useful. To address this, we complement judge-based evaluation with a judge-independent audit based on a standard doubly-robust Conditional Average Treatment Effect (CATE) estimator (Abrevaya et al., 2015), which estimates the expected impact of following a recommendation on gross profit from observational data. Under this audit, our GRPO-trained model achieves approximately twice the gross-profit lift of the strongest evaluated commercial baseline, with the lowest downside rate and least negative tail risk of any policy evaluated. The audit further reveals that judge scores and estimated business value are only loosely coupled across the baselines, which supports its use as a genuinely independent check rather than a corroboration of the judge.

In this work, our main contributions are: (1) rubric-grounded, safety-gated GRPO for financial advice: we formulate free-form business financial advice generation as an RL problem and finetune an open-weight LLM with a finance-specific, safety-gated rubric reward; (2) a natural-languageto-action causal audit: we map generated advice into a fixed business-action taxonomy and apply a standard doubly-robust CATE estimator to audit whether recommendations correspond to historically beneficial actions; and (3) a dual evaluation against commercial and open-weight baselines, using both judge-based rubric scores and a judgeindependent score from a standard doubly-robust CATE estimator, with uncertainty, downside-rate, and tail-risk metrics; we show that the two evaluations agree on our policy but diverge on the baselines, and argue that this divergence is itself evidence that the audit is not a proxy for the judge.

## 2 Related Work

Reinforcement learning from human or AI feedback has been widely used to align LLM outputs with desired properties (Lee et al., 2023). GRPO (Shao et al., 2024) avoids the need for a separate value model by computing relative advantages within a group of sampled outputs. While GRPO was originally introduced for mathematical reasoning, it has since been applied to broader domains using structured rubric rewards. Most closely to our setting, Bhattarai et al. (2026) use multi-criterion rubric rewards with GRPO for scientific reasoning. In the financial domain, RL has been applied to tasks such as alpha factor screening (Jiang et al., 2025) and trading, but open-ended business advice generation has not been addressed.

Financial NLP work has largely focused on benchmarking general-purpose LLMs, finding that they still struggle with money-related reasoning (Rosero et al., 2025; Klimaszewski et al., 2025) and that traditional models can outperform generative LLMs when numeric signals are central (Drinkall et al., 2025). These results highlight the difficulty of the financial domain, which motivates training a model specifically for business advice, both in terms of advice quality and in terms of the latency and cost overhead of relying on general-purpose

commercial LLMs.

To train such a model without gold labels, we rely on an LLM judge as a reward signal. However, improvements driven by a judge reward may reflect adaptation to the judge’s preferences rather than genuine business value, and LLM-based evaluators are known to overestimate systems that align with their style (Lee et al., 2023). We therefore complement judge-based evaluation with a judgeindependent audit based on a standard doublyrobust CATE estimator (Robins et al., 1994; Bang and Robins, 2005; Dudík et al., 2011; Abrevaya et al., 2015).

## 3 Method

## 3.1 Problem Formulation

We are given a business financial state s, summarizing a business’s recent financials (revenue, cost of goods sold, operating expenses, vendor and product line items, and recent trends), together with a target goal g (we focus on gross profit, though the framework supports any financial KPI, such as revenue, quick ratio, or cash flow). The objective is to produce a single, structured recommendation a that, if implemented, would improve g. Each recommendation is emitted as a JSON object with a recommendation (the concrete action), a reasoning field, and a quantified expected\_impact.

We treat advice generation as a reinforcement learning problem and learn a policy $\pi _ { \theta } ( a \mid s , g )$ that maps a state and goal to a recommendation. This framing is motivated by two properties of the domain. First, the historical actions logged for each business are not optimal targets, as they reflect the constraints and incomplete information available at decision time, so supervised imitation would propagate suboptimal behavior. Second, recommendations are open-ended natural language, ruling out value-based methods over a fixed action set. RL instead lets us optimize a scalar reward that directly encodes what makes a recommendation useful.

## 3.2 GRPO Training

We optimize $\pi _ { \theta }$ with Group Relative Policy Optimization (Shao et al., 2024). For each prompt $( s , g )$ we sample a group of K candidate recommendations $\{ a _ { 1 } , \ldots , a _ { K } \}$ from the current policy, score each with the reward function of Section 3.3 to obtain rewards $\{ R _ { 1 } , \ldots , R _ { K } \}$ , and compute a group-relative advantage by standardizing rewards within the group,

<table><tr><td>Aspect</td><td>#</td><td>Binary criterion</td></tr><tr><td>Safety (gate)</td><td>1</td><td>Would not cause significant harm to the business</td></tr><tr><td>Specificity</td><td>2</td><td>Cites a concrete number from the state</td></tr><tr><td></td><td>3</td><td>Names a specific entity (vendor, account, line item)</td></tr><tr><td>Actionability</td><td>4</td><td>Implementableimmediately without further analysis</td></tr><tr><td>Data grounding</td><td>5</td><td>All stated facts are quoted from or derived from the state</td></tr><tr><td>Reasoning</td><td>6</td><td>Identifies a non-obvious pattern or anomaly</td></tr><tr><td></td><td>7</td><td>Acknowledges the primary risk of the action</td></tr><tr><td>Impact</td><td>8 9</td><td>Expected impact is quantified Impact estimate is grounded in</td></tr><tr><td></td><td></td><td>a state value Impact is directionally correct</td></tr><tr><td></td><td>10</td><td></td></tr><tr><td>Relevance</td><td>11</td><td>Targets an item that can materi- ally move the goal metric</td></tr></table>

Table 1: The eleven binary rubric criteria: a safety gate (criterion 1) plus ten quality criteria (criteria 2– 11) across six aspects. The relevance criterion is goalconditioned, adapting to the active target metric.

$$
A _ { i } = { \frac { R _ { i } - \operatorname { m e a n } ( R _ { 1 } , \dots , R _ { K } ) } { \operatorname { s t d } ( R _ { 1 } , \dots , R _ { K } ) } } .\tag{1}
$$

The policy is then updated with the clipped GRPO objective, which avoids training a separate value model. Figure 1 illustrates this loop. We additionally anchor the policy to the base model with a KL penalty weighted by β, which we found necessary to prevent policy drift during training (Section 3.3).

## 3.3 Reward Design

The reward is provided by an LLM judge (Claude Opus 4.5) that scores a recommendation against a rubric of eleven binary criteria: a safety gate plus ten quality criteria organized into six aspects, summarized in Table 1. Each criterion is judged independently as 0 or 1.

The safety gate is critical: if implementing the recommendation would cause significant harm to the business (e.g., eliminating a primary revenue stream or creating unsustainable cash-flow risk), the gate hard-zeros the reward,

$$
c ( a ) = \left\{ \begin{array} { l l } { 0 } & { \mathrm { i f ~ u n s a f e , } } \\ { \displaystyle \frac { 1 } { D } \sum _ { d = 1 } ^ { D } \mathbf { 1 } [ d \mathrm { ~ s a t i s f i e d } ] } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{2}
$$

where $D = 1 0$ is the number of quality criteria. The rubric is goal-conditioned: the relevance criterion adapts to the active target metric.

Two penalties shape the final reward. Recommendations whose JSON cannot be parsed receive a fixed penalty of −0.5. In addition, to discourage degenerate over-long reasoning, a thinking-length penalty $p ( a ) \in [ 0 , 0 . 2 ]$ is applied that is zero below a soft cap on reasoning length and ramps linearly to its maximum at a hard cap. The final reward is

$$
\ R ( a ) = c ( a ) - p ( a ) .\tag{3}
$$

Both terms address failure modes we observed in early runs: without the KL anchor the policy drifted from the base model, and without the length penalty reasoning traces grew unboundedly until JSON outputs collapsed.

## 3.4 Judge-Independent Causal Audit

The training reward comes from an LLM judge. As a result, a higher judge score may reflect recommendations that appeal to the judge rather than genuinely better business outcomes. To test for this, we audit the learned policy with a judge-independent estimate of its effect on real outcomes. This estimate is computed entirely from logged observational data.

The key step is to bridge from free-text recommendations to a quantity for which causal effects can be estimated. An independent action mapper (Section 4.2) assigns each recommendation to a discrete business action; the resulting action is treated as the treatment, the business-state embedding as the conditioning covariates, and the realized year-over-year gross-profit growth as the outcome. We then estimate the conditional average treatment effect (CATE) of the policy’s actions using a standard doubly-robust (AIPW) estimator (Robins et al., 1994; Bang and Robins, 2005; Dudík et al., 2011; Abrevaya et al., 2015), and report the estimated lift alongside a downside rate and a tail-risk measure. We use these estimates as a judge-independent policy-ranking audit rather than as evidence of realized, deployed business impact. The estimator, its propensity and outcome models, and the action mapper are detailed in Section 4.

![](images/b8330f3a1a752cf8071891acf09dbd4d1380be2612578229cb968b00f81224db.jpg)  
Figure 1: GRPO training loop. For each business state and goal $( s , g )$ , the policy samples K candidate recommendations, each is scored by the LLM-as-a-judge rubric reward, and group-relative advantages within the group drive the policy update.

## 4 Experiments

## 4.1 Experimental Setup

Data and model. We train on logged business financial states paired with the goal of improving gross profit. The base policy is Qwen3.5-27B, finetuned with LoRA under DeepSpeed ZeRO-2. Personally identifying entities are removed from the training data and replaced with synthetic surrogates: rather than leaving opaque mask placeholders in the state, the pipeline substitutes realistic synthetic vendor, account, and product names, so that the model is trained on well-formed business text while no real customer entity is retained. GRPO uses K=12 candidate generations per prompt, a maximum completion length of 8,000 tokens, a KL coefficient $\beta { = } 0 . 0 0 1$ , learning rate $5 \times 1 0 ^ { - 5 }$ with a cosine schedule, and the doubly-robust GRPO loss variant. The judge reward model is Claude Opus 4.5. We select the best checkpoint on a held-out validation reward.

Evaluation protocols. We use two complementary evaluations. The first is the LLM-as-a-judge rubric score (Section 3.3), reported on a held-out set of 500 businesses with 5 independent trials per business $_ { ( n = 2 , 5 0 0 }$ per model); we report the mean over per-trial means with a 95% confidence interval. The second is the causal off-policy evaluation (Section 3.4), which we instantiate below. We compare against commercial LLMs (Claude Opus 4.5/4.6, Claude Sonnet 4.5, GPT-5.4) and the untrained Qwen3.5-27B base model.

## 4.2 Causal Estimator Instantiation

We estimate the effect of a policy’s recommended action on year-over-year gross-profit growth from logged data. Treatment is defined per action: for a target action $^ { a , }$ let $A \in \{ 0 , 1 \}$ indicate whether a was taken, so that the treated arm consists of logged states in which a was taken and the control arm consists of states in which a different catalogue action was taken. A single multi-label propensity model supplies $e ( x ) ~ = ~ P ( A { = } 1 ~ | ~ x )$ for every action, and each action is stratified independently. For a business with covariates $x$ (a 768-dimensional embedding of the business state, encoded with a sentence-embedding model), let Y denote the realized year-over-year gross-profit growth. We fit a propensity model eˆ (a multi-label MLP over the state embedding) and an outcome model with separate treated and control heads $\hat { \mu } _ { 1 } , \hat { \mu } _ { 0 }$ (neural networks predicting $Y$ under the action and under its absence). The doubly-robust AIPW pseudooutcome for each logged business is

$$
\begin{array} { l } { \displaystyle \phi _ { i } = \big ( \hat { \mu } _ { 1 } ( x _ { i } ) - \hat { \mu } _ { 0 } ( x _ { i } ) \big ) + \frac { A _ { i } } { \hat { e } ( x _ { i } ) } \big ( Y _ { i } - \hat { \mu } _ { 1 } ( x _ { i } ) \big ) } \\ { \displaystyle - \frac { 1 - A _ { i } } { 1 - \hat { e } ( x _ { i } ) } \big ( Y _ { i } - \hat { \mu } _ { 0 } ( x _ { i } ) \big ) , } \end{array}\tag{4}
$$

which is unbiased if either the propensity or the outcome model is correctly specified (Bang and Robins, 2005).

What is computed per recommendation. The audit proceeds in two stages, and it is worth stating precisely which quantity each stage produces. In stage one, on logged data where the treatment $A _ { i }$ and the outcome $Y _ { i }$ are both observed, we evaluate the AIPW pseudo-outcome of Eq. 4 for every logged business and average it within each (action, propensity-decile) stratum. Strata that fail minimum-support or effective-sample-size gates are marked invalid and excluded. This yields a table of stratum-level effect estimates spanning the 49 actions that clear the eligibility threshold and 10 propensity bins per action. In stage two, each heldout recommendation is mapped to an action, the propensity of its business state for that action is predicted and assigned to a bin, and the recommendation receives the corresponding stratum-level effect. Equation 4 is therefore evaluated only on logged businesses; a held-out recommendation receives a stratum-level lookup rather than its own pseudooutcome, because advice that was never acted upon has no observed $A _ { i }$ or $Y _ { i }$ . The estimand is thus the mean stratum-level effect of the (action, propensitybin) cells into which a policy’s recommendations fall, and policies differ only through which actions they select and how their states distribute across propensity bins.

The three reported metrics are computed over this set of per-recommendation values. The estimated lift (DR-CATE) is their mean. The downside rate (DR%) is the fraction whose assigned stratum effect is negative. The conditional value-at-risk $\mathrm { { C V a R } _ { 0 . 1 0 } }$ is the mean over the worst 10%. Higher lift is better; a lower downside rate and a less negative $\mathrm { { C V a R } _ { 0 . 1 0 } }$ are better.

The audit follows the same protocol as the judge evaluation. Each policy is evaluated on 500 randomly sampled held-out business states over 5 independent runs with different random seeds. We compute the three metrics within each run, and report the mean across runs together with a 95% confidence interval over the five run-level estimates. Reporting uncertainty at the level of runs rather than individual recommendations means the intervals reflect run-to-run variation in the policy’s own sampling, which is the variation that matters when comparing policies. The mapper, the propensity model, and the outcome model are held fixed across all policies, so every system is scored by the same instruments.

Action mapper. Recommendations are mapped to the action taxonomy by a separate LLM classifier, distinct from and independent of the Claude Opus 4.5 reward judge, so that mapping noise cannot directly inflate the audit. Given a recommendation and the fixed action catalogue, the mapper returns the single best-matching action together with a confidence score; recommendations that fall below a confidence threshold, or that match no catalogue action, are assigned a dedicated no-match label and excluded from scoring rather than forced onto an ill-fitting action. We summarize the mapper here and will release a synthetic, anonymized version with the reproducibility artifacts.

<table><tr><td>Item</td><td>Value</td></tr><tr><td>Evaluation unit</td><td>company-month</td></tr><tr><td>Sample per run</td><td>500 business states</td></tr><tr><td>Runs per policy</td><td>5 (different seeds)</td></tr><tr><td>Action catalogue</td><td>60 actions</td></tr><tr><td>Action categories</td><td>10</td></tr><tr><td>Outcome</td><td>YoY gross-profit growth</td></tr><tr><td>Causal estimator</td><td>doubly-robust AIPW (DR-CATE)</td></tr><tr><td>Propensity model</td><td>multi-label MLP</td></tr><tr><td>Outcome model</td><td>neural, treated/control heads</td></tr><tr><td>State embedding</td><td>768-dim sentence embedding</td></tr><tr><td>Holdout split</td><td>company-level</td></tr><tr><td>Free-text mapping</td><td>LLM mapper (no-match option)</td></tr></table>

Table 2: Evaluation/data card for the judge-independent causal audit.

A compact summary of the evaluation setup is given in Table 2.

## 4.3 Results

Judge evaluation. Table 3 reports the rubric scores. After GRPO training, Qwen3.5-27B reaches 9.514, the top score on the leaderboard, ahead of Claude Opus 4.6 (9.365), GPT-5.4 (8.949), and Claude Sonnet 4.5 (8.712), and well above the untrained base model (8.457). Two caveats apply to this table. First, the judge belongs to the Claude family, which may introduce a small self-preference bias in favor of Claude baselines; this makes the comparison conservative for our model. Second, and more importantly, our policy was trained against this same judge, so its rubric score is in part a measure of successful optimization rather than an independent assessment. We therefore read Table 3 as evidence that training achieved its objective, and treat the judgeindependent causal audit below as the primary comparative evidence.

Causal evaluation. Table 4 reports the causal off-policy results. Our policy attains an estimated gross-profit lift of 0.0228 [0.0211, 0.0246], approximately twice that of the strongest commercial baseline, Claude Opus 4.6 (0.0104, a ratio of 2.20×), together with the lowest downside rate (0.155 vs.

<table><tr><td>Model</td><td>Score</td><td>95% CI</td></tr><tr><td>Qwen3.5-27B-GRPO (ours)</td><td>9.514</td><td>[9.505, 9.524]</td></tr><tr><td>Claude Opus 4.6</td><td>9.365</td><td>[9.354, 9.376]</td></tr><tr><td>Claude Opus 4.5</td><td>8.982</td><td>[8.954, 9.010]</td></tr><tr><td>GPT-5.4</td><td>8.949</td><td>[8.911, 8.988]</td></tr><tr><td>Claude Sonnet 4.5</td><td>8.712</td><td>[8.644, 8.780]</td></tr><tr><td>Qwen3.5-27B (base)</td><td>8.457</td><td>[8.384, 8.531]</td></tr></table>

Table 3: LLM-as-a-judge rubric scores, n=2,500 per model. Scores are the mean number of the D=10 quality criteria satisfied, i.e. the quality term c(a) of Eq. 2 rescaled to [0, 10]; the safety gate zeroes the score of any recommendation judged unsafe. Our GRPO-trained model ranks first.

0.232) and the least negative tail risk $( \mathrm { C V a R _ { 0 . 1 0 } = }$ $- 0 . 0 7 3 \ \mathrm { v s . } - 0 . 1 0 0 )$ , meaning that even its worstdecile recommendations are less harmful on average. Our policy ranks first on all three metrics, and its confidence interval does not overlap that of any other policy on any metric; under a Welch t-test over the five per-trial means, every pairwise comparison with our policy is significant at $p < 0 . 0 1$

The commercial baselines separate clearly under the audit. Claude Opus 4.6 is the only commercial system with a lift that is both positive and comfortably distinguishable from zero $( p <$ 0.001); GPT-5.4 is positive but noisier (0.0082, $p \ = \ 0 . 0 1 5$ , and the widest interval of any policy); Claude Sonnet 4.5 is not distinguishable from zero (0.0028 [0.0000, 0.0055], $p = 0 . 0 5 1 )$ ; and Claude Opus 4.5 is estimated to be negative $( - 0 . 0 0 2 5 \left[ - 0 . 0 0 4 3 , - 0 . 0 0 0 7 \right] )$ , i.e. its recommendations map to actions that are, on average, associated with slightly worse subsequent gross-profit growth. Downside rate tracks this ordering closely: the three weakest policies place roughly a third of their recommendations on actions with negative estimated effect, against 15.5% for ours.

Effect of GRPO training. Comparing our policy against its own starting point isolates the contribution of GRPO from that of the base model. Training raises the estimated lift from 0.0170 to 0.0228, a relative improvement of 34% (Welch t-test over the five run-level estimates, $p = 0 . 0 0 0 9$ ; because all policies are evaluated under the same five seeds, a paired test on the same estimates gives $p = 0 . 0 0 3 )$ while cutting the downside rate from 0.194 to 0.155 $( p = 0 . 0 0 2 )$ and improving tail risk from −0.094 $\mathrm { t o - 0 . 0 7 3 } \ ( p = 0 . 0 0 2 )$ . The gain is therefore not merely a property of the base checkpoint: GRPO with the safety-gated rubric reward moves the policy toward actions with higher estimated effect and away from actions with negative estimated effect, which is precisely the behavior the safety gate was designed to induce.

The two evaluations do not rank identically. The audit and the judge agree that our GRPO policy is best, but they disagree elsewhere, and the disagreement is informative. The untrained Qwen3.5- 27B base model ranks last on the judge rubric (8.457, Table 3) yet second on the causal audit (0.0170, Table 4), above every commercial system. Conversely, Claude Opus 4.5 scores well on the rubric (8.982) but has a negative estimated lift. We read this as evidence that the two evaluations measure genuinely different things. The rubric rewards well-formed advice: concrete figures, named entities, quantified impact, acknowledged risk. The audit rewards advice that maps to actions historically associated with gross-profit growth, and is indifferent to how well that advice is written. Commercial models are strong at the former and only middling at the latter, whereas the base model produces less polished recommendations that nonetheless land on reasonable actions. This divergence is the clearest available evidence that the audit is not a proxy for the judge; had it simply re-scored rubric quality, the base model would have ranked last on both. It also means the joint result carries more weight than either evaluation alone: our policy is the only system that ranks first under both.

## 5 Discussion

Three findings stand out. First, GRPO with a finance-specific reward and explicit safety gate lifts an open-weight model above strong commercial LLMs on the judge rubric (Table 3). This indicates that targeted reward design can substitute for raw scale on a narrow, high-value task. Second, the causal audit, which is independent of the judge, moves in the same direction (Table 4): the GRPO policy attains a higher estimated gross-profit lift, a lower downside rate, and a less negative tail risk than every commercial baseline, with nonoverlapping confidence intervals throughout. Because the audit is computed from logged outcomes and does not rely on the reward judge, this agreement makes pure judge adaptation a less likely explanation for the gains, although it does not establish realized impact.

Third, and less expected, the two evaluations rank the other systems quite differently. That the untrained base model is last on the rubric but second on the audit shows the audit is not a restatement of the judge, and it also cautions against reading either evaluation as a complete measure of advice quality on its own: a policy can produce recommendations that are well-formed but point at unhelpful actions, or poorly-formed but point at sensible ones. Only a system that scores well on both, as ours does, is supported by both lines of evidence.

<table><tr><td>Policy</td><td>Lift (DR-CATE) ↑</td><td>DR%↓</td><td> $\mathbf { C V a R _ { 0 . 1 0 } } \uparrow$ </td></tr><tr><td>Qwen3.5-27B-GRPO (ours)</td><td> $\mathbf { 0 . 0 2 2 8 \pm 0 . 0 0 1 7 }$ </td><td> $\mathbf { 0 . 1 5 5 \pm 0 . 0 1 7 }$ </td><td> $\mathbf { - 0 . 0 7 3 \pm 0 . 0 0 3 }$ </td></tr><tr><td>Qwen3.5-27B (base)</td><td> $0 . 0 1 7 0 \pm 0 . 0 0 2 4$ </td><td> $0 . 1 9 4 \pm 0 . 0 1 8$ </td><td> $- 0 . 0 9 4 \pm 0 . 0 0 9$ </td></tr><tr><td>Claude Opus 4.6</td><td> $0 . 0 1 0 4 \pm 0 . 0 0 3 1$ </td><td> $0 . 2 3 2 \pm 0 . 0 2 1$ </td><td> $- 0 . 1 0 0 \pm 0 . 0 0 8$ </td></tr><tr><td>GPT-5.4</td><td> $0 . 0 0 8 2 \pm 0 . 0 0 5 6$ </td><td> $0 . 3 2 7 \pm 0 . 0 4 0$ </td><td> $- 0 . 1 2 4 \pm 0 . 0 0 7$ </td></tr><tr><td>Claude Sonnet 4.5</td><td> $0 . 0 0 2 8 \pm 0 . 0 0 2 8$ </td><td> $0 . 3 2 0 \pm 0 . 0 2 7$ </td><td> $- 0 . 1 1 4 \pm 0 . 0 0 6$ </td></tr><tr><td>Claude Opus 4.5</td><td> $- 0 . 0 0 2 5 \pm 0 . 0 0 1 8$ </td><td> $0 . 3 6 2 \pm 0 . 0 2 1$ </td><td> $- 0 . 1 1 6 \pm 0 . 0 0 6$ </td></tr></table>

Table 4: Judge-independent causal off-policy evaluation: estimated gross-profit lift (DR-CATE), downside rate (DR%), and tail risk $\mathrm { ( C V a R _ { 0 . 1 0 } ) }$ . Policies are ordered by lift. Each entry is the mean over 5 independent runs (500 sampled business states per run, different seed per run) ± the half-width of a 95% confidence interval computed over the five run-level estimates. Higher lift and higher (less negative) $\mathrm { C V a R _ { 0 . 1 0 } }$ are better; lower DR% is better. Our GRPO policy ranks first on all three metrics, with no confidence-interval overlap against any other policy on any metric.

The safety gate and the risk-aware metrics are complementary. The gate suppresses harmful recommendations during training, while the downside rate and tail-risk metrics verify after training that the resulting policy places less mass on actions with negative estimated effects; the drop in downside rate from 0.194 (base) to 0.155 (ours) is direct evidence that this mechanism operates as intended.

## 6 Conclusions

We formulate financial advice generation as a reinforcement learning problem and train an openweight model with GRPO using a finance-specific reward that incorporates explicit safety constraints. The resulting model outperforms the strongest evaluated commercial baseline on both the judge rubric and the CATE audit, and is the only system to rank first under both. It achieves approximately twice the estimated gross-profit lift while maintaining the lowest downside rate and least negative tail risk of any policy evaluated, and improves over its own starting checkpoint by 34% in estimated lift. We also find that the judge and the causal audit rank the baselines differently, which suggests that rubric quality and estimated business value are distinct axes and that reporting only one of them can be misleading. These results suggest that careful reward design, combined with a judge-independent causal audit to reduce the risk of reward hacking, provides a practical approach for building safer and more effective financial advice models. Future work will explore richer action taxonomies, evaluation by financial experts, and support for multi-step advice.

## Limitations

Because randomized experiments are not available in this setting, our causal audit is computed from logged observational data, in common with offpolicy evaluation more broadly. We therefore treat it as a judge-independent, complementary signal alongside the rubric score rather than as a standalone measure, and the two evaluations are designed to be read together. A second limitation is coverage. Recommendations that receive no confident match in the 60-action catalogue are excluded from the audit rather than forced onto an ill-fitting action. This keeps the mapping conservative, but it means each policy is audited on the subset of its own output that the catalogue can express, and a policy whose unmatched recommendations differ systematically in quality from its matched ones would be scored on a non-representative subset. A richer catalogue would narrow this gap. Our audit also scores a single recommended action per business state; extending it to multi-step advice, where actions compound over a longer horizon, is a natural direction for future work. Finally, our study targets gross profit as the primary goal metric; while the framework supports other financial KPIs, we leave a broad multi-KPI evaluation to future work.

## References

Jason Abrevaya, Yu-Chin Hsu, and Robert P Lieli. 2015. Estimating conditional average treatment effects. Journal of Business & Economic Statistics, 33(4):485–505.

Heejung Bang and James M Robins. 2005. Doubly robust estimation in missing data and causal inference models. Biometrics, 61(4):962–973.

Manish Bhattarai, Ismael Boureima, Nishath Rajiv Ranasinghe, Scott Pakin, and Dan O’Malley. 2026. Rubric-grounded rl: Structured judge rewards for generalizable reasoning. arXiv preprint arXiv:2605.08061.

Felix Drinkall, Janet Pierrehumbert, and Stefan Zohren. 2025. Forecasting credit ratings: A case study where traditional methods outperform generative llms. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 118–133.

Miroslav Dudík, John Langford, and Lihong Li. 2011. Doubly robust policy evaluation and learning. arXiv preprint arXiv:1103.4601.

Zuoyou Jiang, Li Zhao, Rui Sun, Ruohan Sun, Zhongjian Li, Jing Li, Daxin Jiang, Zuo Bai, and Cheng Hua. 2025. Alpha-r1: Alpha screening with llm reasoning via reinforcement learning. arXiv preprint arXiv:2512.23515.

Leslie Pack Kaelbling, Michael L Littman, and Andrew W Moore. 1996. Reinforcement learning: A survey. Journal of artificial intelligence research, 4:237–285.

Mateusz Klimaszewski, Pinzhen Chen, Liane Guillou, Ioannis Papaioannou, Barry Haddow, and Alexandra Birch. 2025. Avenibench: Accessible and versatile evaluation of finance intelligence. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 111–117.

Harrison Lee, Samrat Phatale, Hassan Mansoor, Thomas Mesnard, Johan Ferret, Kellie Lu, Colton Bishop, Ethan Hall, Victor Carbune, Abhinav Rastogi, and 1 others. 2023. Rlaif vs. rlhf: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267.

James M Robins, Andrea Rotnitzky, and Lue Ping Zhao. 1994. Estimation of regression coefficients when some regressors are not always observed. Journal ofthe American statistical Association, 89(427):846– 866.

Alexei Gustavo Figueroa Rosero, Paul Grundmann, Julius Freidank, Wolfgang Nejdl, and Alexander Loeser. 2025. Evaluating financial literacy of large language models through domain specific languages for plain text accounting. In Proceedings ofthe Joint Workshop ofthe 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 63–75.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300.