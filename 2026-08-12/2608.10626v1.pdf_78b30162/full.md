# Dual-Loop Self-Evolution via Verifiable Emotion Feedback for Multi-Turn Empathetic Dialogue

Yi Wei<sup>1,2∗</sup>, Shuo Jiang<sup>1∗</sup>, Huaixia Dou<sup>1</sup>, Jie Zhu<sup>1†</sup>, Junhui Li<sup>3</sup>, Lifan Guo<sup>1</sup>, Feng Chen<sup>1</sup>, Chi Zhang<sup>1</sup>

<sup>1</sup>Qwen DianJin Team, Alibaba Cloud Computing

<sup>2</sup>Beihang University

<sup>3</sup>School of Computer Science and Technology, Soochow University

## Abstract

Large language models have demonstrated conversational capabilities, yet empathetic competence remains challenging. Empathetic support is inherently multi-turn and pathdependent: users disclose concerns gradually, emotions evolve over time, and early responses shape trust and receptivity. Reinforcement learning with verifiable emotion rewards provides scalable supervision for long-horizon interactions. However, existing methods evolve the dialogue policy while keeping its training interaction distribution fixed, creating a mismatch between policy competence and training experience. We introduce a dual-loop self-evolution framework driven by verifiable emotion feedback. With the user simulator and verifier frozen, the inner loop optimizes the multi-turn policy using continuous emotion rewards, while the outer loop uses the same outcomes to estimate policy-relative interaction utility and adapt experience. To obtain estimates from sparse, stochastic rollouts, the framework holds the scenario and interaction state constant within each group and prioritizes conditions whose group pass rates lie near the policy’s competence boundary. A hierarchical controller shares evidence across support intents, while uncertainty-guided exploration and uniform rehearsal prevent premature exclusion. The resulting distribution generates trajectories, closing both loops without increasing the rollout budget. On SAGE, our framework raises Qwen3-8B Overall from 53.87 to 79.24 and outperforms protocol-matched uniform emotion-reward reinforcement learning by 7.23 points.

## 1 Introduction

Large language models (LLMs) have become capable conversational systems, yet reliable empathetic support remains dificult (Rashkin et al. 2019; Liu et al. 2021; Cheng et al. 2022). Empathy is inherently multi-turn: concerns surface incrementally over turns, afect rises and falls with the conversation, and each response constrains what the user is willing to reveal next (Zhou et al. 2023; Wan, Labeau, and Clavel 2025; Wu et al. 2026; Zhang et al. 2026). A reply that appears appropriate in isolation may still derail the subsequent conversation. An efective policy must therefore learn how its actions influence an evolving user over a complete trajectory, rather than merely imitate locally plausible supportive text.

Learning such behavior requires coherent long-horizon experience and reliable outcome supervision. Human-authored support dialogues are costly to collect, dificult to standardize, and limited in the range of user reactions they expose (Liu et al. 2021; Cheng et al. 2022). Ofline augmentation improves coverage but remains fixed once generated (Zheng et al. 2023; Ye et al. 2025). Interactive user simulation offers a scalable alternative: a policy can observe how diferent responses change later disclosure and emotion, provided the simulated user preserves its persona, context, and hidden need across turns (Yoon et al. 2024; Gromada et al. 2025; Dou et al. 2025; Zhang et al. 2026). Verifiable emotion rewards further turn these reactions into trajectory-level supervision, enabling reinforcement learning (RL) to optimize long-term afective consequences rather than only match static references (Wang et al. 2026b).

![](images/16f7e1dee4fe5bc51cd97c916e6f6300f6850d19efc35ba859b92832b1dbce7c.jpg)  
Figure 1: Training paradigms. Static curricula lag behind competence; unconstrained self-play creates moving targets; ours adapts the interaction distribution with stable role-play.

However, existing methods still treat the training interaction distribution as a predefined and fixed external condition (Wang et al. 2026b). Emotion rewards continually update the dialogue policy, but do not change how subsequent experience is allocated. Consequently, each condition receives the same expected rollout budget whether the current policy has mastered it, repeatedly fails on it, or still finds it informative. As competence evolves, the learning value of an interaction changes with it, while the allocation of expensive multi-turn experience remains static. This creates a growing mismatch between what the policy can currently do and what it is asked to practice.

Simply enlarging the interaction pool or prescribing an easy-to-hard schedule does not resolve this mismatch. Interaction dificulty is not an intrinsic, permanent property of a scenario. The same hesitant or emotionally activated user may be excessive for an early policy, informative for an improving policy, and redundant for a mature one. A curriculum fixed before training cannot remain aligned with this moving competence boundary (Bengio et al. 2009; Kumar, Packer, and Koller 2010). Adaptive curriculum and replay methods recognize that training utility is learner-dependent (Graves et al. 2017; Schaul et al. 2016; Jiang, Grefenstette, and Rocktäschel 2021; Shi et al. 2026; Jiang et al. 2025), but existing formulations do not directly address sparse, stochastic outcomes from long multi-turn interactions with a role-playing user.

Self-play ofers an apparently natural solution by evolving both the assistant and its simulated partner (Chen et al. 2024; Dai et al. 2026). In empathetic dialogue, however, the simulator participates in both trajectory generation and outcome formation. Unconstrained adaptation can make users arbitrarily resistant, overly accepting, or inconsistent with their original persona and hidden need (Yoon et al. 2024; Dou et al. 2025). Reward changes would then reflect both assistant improvement and simulator drift, obscuring credit assignment and turning evaluation into a moving target. A more dificult user is not necessarily a more useful or faithful teacher.

To this end, we propose a dual-loop self-evolution framework that jointly advances the dialogue policy and the experience distribution through shared verifiable emotion feedback; Figure 1 contrasts this design with static curricula and unconstrained self-play. The loops are nested but have distinct targets: the inner loop uses continuous emotion rewards to improve how the policy responds, while the outer loop reuses group outcomes to identify what the policy should practice next and reallocates subsequent interactions accordingly. The updated distribution produces new trajectories that again supervise both processes, turning a fixed interaction pool into a capability-aligned online curriculum grounded in comparable multi-turn outcomes.

Experiments show that our framework achieves 79.24 SAGE Overall, surpassing uniform emotion-reward RL by 7.23 points, or a 10.0% relative gain. The improvement requires no additional rollouts and holds across complementary evaluation protocols and component ablations.

Our contributions are threefold:

• We identify curriculum–policy misalignment as a central challenge in multi-turn emotion-reward RL: policy competence evolves while its interaction distribution remains fixed.

• We introduce a dual-loop self-evolution framework in which shared emotion feedback jointly drives policy improvement and capability-aligned adaptation of the training experience distribution.

• Multi-run, cross-benchmark, and component-level evaluations demonstrate substantial performance gains and improved rollout-budget eficiency over uniform emotionreward training.

## 2 Related Work

Empathetic dialogue and afective reinforcement learning. Empathetic dialogue has progressed from emotiongrounded generation to long-horizon support over latent needs and strategies (Rashkin et al. 2019; Lin et al. 2019; Majumder et al. 2020; Liu et al. 2021; Cheng et al. 2022), with work on planning, augmentation, interpretable reasoning, strategy prediction, and personalization (Zhou et al. 2023; Zheng et al. 2023; Zhang et al. 2024b; Kang et al. 2024; Wan, Labeau, and Clavel 2025; Shi, Hao, and Kong 2025; Ye et al. 2025). RL encourages value-sensitive support (Kim et al. 2025; Zhao et al. 2025; Wang et al. 2025), while LLM simulators enable scalable interaction but require faithful role-play and assessment (Yoon et al. 2024; Gromada et al. 2025; Dou et al. 2025). SAGE supplies evolving emotional personas (Zhang et al. 2026), and verifiable emotion rewards optimize policies (Wang et al. 2026b); we instead align the interaction distribution with the evolving policy.

Adaptive curricula and self-evolving training. Curriculum, self-paced, and replay methods allocate computation by learner-dependent utility (Bengio et al. 2009; Kumar, Packer, and Koller 2010; Jiang et al. 2015; Schaul et al. 2016; Jiang, Grefenstette, and Rocktäschel 2021); LLM self-improvement adds self-play, self-judgment, and online optimization (Chen et al. 2024; Yuan et al. 2024; Ahmadian et al. 2024). Recent work adapts prompts or rollouts using success, uncertainty, variance, and progress (Shi et al. 2026; Jiang et al. 2025, 2026; Liu et al. 2026; Li et al. 2026; Zeng et al. 2026; Nguyen et al. 2026), or evolves tasks, users, evaluators, and harnesses (Dai et al. 2026; Wang et al. 2026a; Yang et al. 2026; Chen et al. 2026). We instead freeze role-play and evolve a hierarchical distribution over multi-turn conditions.

## 3 Dual-Loop Self-Evolution

Figure 2 presents the complete training pipeline. Our framework organizes multi-turn empathetic-dialogue training as two nested feedback loops driven by the same verified emotion outcomes. The inner loop optimizes the policy within the interaction distribution currently supplied by the controller; the outer loop observes completed group outcomes and revises that distribution as the policy changes. The two loops are logically nested rather than parallel learners.

The remainder of this section follows one complete feedback cycle. Section 3.1 formalizes scenario-conditioned rollouts and the nested objectives, while Section 3.2 constructs the controllable interaction space. Sections 3.3–3.4 describe shared-condition policy learning and verified group feedback. Section 3.5 converts sparse outcomes into robust policy-relative utility, and Section 3.6 closes the loop by sampling the next interaction distribution.

## 3.1 Task Setup and Dual-Loop Framework

Training begins from a set X of complete emotional-support scenarios. A scenario $x \in \mathcal { X }$ specifies the user’s persona, precipitating event, current dilemma, response tendencies, and hidden support need; an intent map h associates it with a support intent $c = h ( x )$ . We preserve this semantic core throughout training. To control how the same concern unfolds in dialogue, we additionally choose a simulator-side interaction state z from a finite space $\mathcal { Z }$ . The pair $e = ( x , z )$ is one concrete rollout environment: x determines what the user is experiencing, while z determines how the user discloses and reacts.

![](images/b0d64f5468ac92eeac677adbcda4f4567c63274716b7a45f95b4c7c44253c902.jpg)  
Figure 2: Overview of the proposed framework. A complete scenario remains uniformly sampled and is paired with a simulatorside interaction state conditioned on its native support intent. Trajectories within a rollout group share the same environment. Continuous emotion reward optimizes the dialogue policy, while thresholded group outcomes update hierarchical intent–state statistics and the subsequent interaction distribution.

For a sampled condition $( x , z )$ , a user simulator $\mathcal { U }$ and the dialogue policy $\pi _ { \theta }$ interact for at most H turns. Their alternating user and assistant utterances form a trajectory

$$
\tau = ( u _ { 1 } , a _ { 1 } , \dots , u _ { T } , a _ { T } ) \sim P ( \tau | \boldsymbol { x } , z , \pi _ { \theta } , \mathcal { U } ) ,
$$

where $T \leq H$ is the realized number of turns. After the interaction ends, an emotion verifier evaluates the complete trajectory and returns a scalar outcome $R ( \tau )$ . The policy observes only the natural-language conversation: support-intent labels, interaction-state labels, controller statistics, and sampling probabilities remain simulator-side metadata. Consequently, improved performance must be expressed through dialogue behavior rather than direct access to a dificulty label.

At training stage t, the outer loop represents its current curriculum by a conditional distribution $p _ { t } ( z \mid c )$ over interaction states for each support intent, initialized uniformly before feedback accumulates. Sampling from this distribution produces the conditions under which the inner loop optimizes the policy:

$$
\begin{array} { r l } { \underset { \theta } { \operatorname* { m a x } } } & { \mathbb { E } _ { { x } \sim \mathrm { U n i f } ( \mathcal { X } ) , { z } \sim p _ { t } ( \cdot | h ( x ) ) , \tau } } \\ & { [ R ( \tau ) ] . } \end{array}\tag{1}
$$

Meanwhile, verified outcomes update controller history $\mathcal { H } _ { t }$ and hence the next interaction distribution:

$$
\begin{array} { r l } & { \mathcal { H } _ { t + 1 } = \mathrm { U p d a t e } ( \mathcal { H } _ { t } ; \mathcal { D } _ { t } , R _ { t } ) , } \\ & { p _ { t + 1 } = \mathrm { C o n t r o l l e r } ( \mathcal { H } _ { t + 1 } ) . } \end{array}\tag{2}
$$

Here $\mathcal { D } _ { t }$ denotes the completed rollout groups collected at stage t, $R _ { t }$ their verified outcomes, and $\mathcal { H } _ { t }$ the controller’s accumulated evidence. Updating $\mathcal { H } _ { t }$ changes $p _ { t + 1 }$ , which in turn changes the experience used by the next policy updates. Crucially, the outer-loop target is policy-relative interaction utility—the value of allocating another rollout group to a condition under the current policy—rather than a permanent dificulty label attached to a scenario.

## 3.2 A Controllable Interaction Space

The framework requires interaction states to be behaviorally realizable, compatible with the underlying scenario, hidden from the assistant, and reusable within a rollout group. We preserve each complete scenario and define three taskoperational axes: disclosure readiness controls when details and deeper needs are revealed; emotional activation controls the intensity of negative-afect expression; and relational trust controls whether questions and support are accepted. These axes capture information release, afective expression, and relational feedback rather than clinical traits.

We discretize the axes at a granularity that remains behaviorally distinguishable and statistically estimable: disclosure has three levels (delayed, conditional, proactive), activation has two bounded levels (moderate, high), and trust has four levels from skepticism to sustained engagement. Their Cartesian product gives $| \mathcal { Z } | = 3 \times 2 \times 4 = 2 4$ interaction states. A state $z = ( d , a , r )$ is rendered by composing one natural-language behavioral clause for each axis, $\phi ( z ) = \phi _ { d } ( d ) \oplus \phi _ { a } ( a ) \oplus \phi _ { r } ( r )$ for $z ~ = ~ ( d , a , r )$ . These clauses govern disclosure pace, afective expression, and receptivity, but cannot overwrite the persona, event, or hidden need specified by x.

Estimating a separate curriculum value for every scenario– state pair would fragment evidence across many nearly related conditions. We therefore retain individual scenarios for rollout generation but organize controller evidence by support intent. Combining intent c with state z defines the reusable unit $m = \ : ( c , z )$ . This factorization lets diferent scenarios with the same support objective contribute evidence about a shared interaction state, while preserving intent-specific diferences. Scenario identities and intent frequencies remain fixed; only the conditional state distribution $p _ { t } ( z \mid c )$ evolves. Section 4.4 later separates the contribution of this interaction space from that of adaptive allocation.

Behavioral realization and information boundaries. Each state is translated into simulator-side behavioral guidance rather than exposed as a symbolic label. Delayed, conditional, and proactive disclosure regulate when relevant facts and deeper needs become available. Moderate and high activation regulate the intensity of negative afect without inventing an unsupported crisis. The four trust levels range from rejecting generic reassurance to sustained engagement with situation-specific support. Every composed instruction additionally requires the simulator to preserve the original persona, event, and hidden support need, and forbids mentioning axis names, levels, state identifiers, or dificulty labels.

## 3.3 Group-Shared Multi-Turn Policy Learning

For each scenario, the controller selects one state z, and group-relative policy optimization draws $\tau _ { 1 : K } \stackrel { \mathrm { ~ i . i . d . } } { \sim } P ( \tau \mid$ $x , z , \pi _ { \theta } , \mathcal { U } )$ under the shared condition $( x , z )$ . Holding the environment fixed within a group prevents diferences in user behavior from confounding group-relative advantages, so rollout variation primarily reflects policy and generation stochasticity.

Abstracting from stabilization terms held fixed across all controlled systems, the method-level policy reward is the continuous verifier outcome, $r _ { k } ^ { \mathrm { p o l i c y } } = R ( \tau _ { k } )$ . GRPO computes group-relative advantages from these rewards and applies the standard clipped policy objective with KL regularization (Shao et al. 2024; Schulman et al. 2017). Building on this inner optimization, our outer allocation loop reuses the same outcomes to organize subsequent training experience.

## 3.4 Emotion-Thresholded Group Feedback

The policy and controller consume diferent transformations of the same verified outcome. Given an emotion criterion $\eta ,$ the controller computes

$$
y _ { k } = \mathbb { I } [ R ( \tau _ { k } ) \geq \eta ] , \qquad p _ { m } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } y _ { k } ,\tag{3}
$$

where $y _ { k }$ records whether trajectory k passes the criterion, and $p _ { m }$ is the pass rate of the shared unit $m = ( h ( x ) , z )$ in that group. We use the pass rate because it directly reveals whether the current policy produces both successful and unsuccessful behavior under a controlled condition. Thresholding afects only outer-loop allocation: the continuous value $\ln ( \tau _ { k } )$ remains the inner-loop reward and therefore retains fine-grained diferences between trajectories. One complete group contributes one pass-rate observation, preventing correlated trajectories from masquerading as independently sampled environments.

## 3.5 Robust Policy-Relative Utility Estimation

Controller evidence is uneven: some intent–state units are observed repeatedly, whereas others have only a few completed groups. Trusting a sparse empirical rate can turn an early random success or failure into a persistent sampling bias. For unit $( c , z )$ , let $S _ { c , z }$ be the cumulative sum of group pass rates and $N _ { c , z }$ the number of complete groups. We stabilize its estimate with a leave-one-intent prior formed from the same state in the other support intents:

$$
\begin{array} { l l } { { S _ { z } ^ { - c } = \displaystyle \sum _ { c ^ { \prime } \neq c } S _ { c ^ { \prime } , z } , \qquad N _ { z } ^ { - c } = \displaystyle \sum _ { c ^ { \prime } \neq c } N _ { c ^ { \prime } , z } , } } \\ { { \hat { p } _ { z } ^ { - c } = \displaystyle \frac { S _ { z } ^ { - c } + \alpha / 2 } { N _ { z } ^ { - c } + \alpha } , \qquad \hat { p } _ { c , z } = \displaystyle \frac { S _ { c , z } + \lambda \hat { p } _ { z } ^ { - c } } { N _ { c , z } + \lambda } . } } \end{array}\tag{4}
$$

Here $\hat { p } _ { z } ^ { - c }$ summarizes how state z behaves outside intent $c ,$ and $\hat { p } _ { c , z }$ is the resulting unit estimate. The coeficient α supplies a neutral prior when pooled evidence is scarce, while λ controls how strongly the unit borrows that evidence. Excluding c avoids counting its observations in both terms. As $N _ { c , z }$ grows, the unit’s own history naturally dominates.

The stabilized pass rate is then converted into an allocation score. We assign maximal boundary value when success and failure coexist, and add an uncertainty bonus that revisits under-observed units (Auer, Cesa-Bianchi, and Fischer 2002):

$$
\begin{array} { r l } & { B _ { c , z } = \operatorname* { m a x } ( 0 , 1 - 2 | \hat { p } _ { c , z } - \frac { 1 } { 2 } | ) , } \\ & { U _ { c , z } = \sqrt { \log ( N + 1 ) / ( N _ { c , z } + 1 ) } , } \\ & { A _ { c , z } = \operatorname* { m a x } ( A _ { \operatorname* { m i n } } , B _ { c , z } + \beta U _ { c , z } ) . } \end{array}\tag{5}
$$

Here $B _ { c , z }$ measures proximity to the current success boundary, $U _ { c , z }$ expresses uncertainty from limited visits, and $A _ { c , z }$ combines the two with exploration weight $\beta$ while retaining a minimum score $A _ { \mathrm { m i n } }$ . In $\begin{array} { r } { U _ { c , z } , N = \bar { \sum } _ { c ^ { \prime } , z ^ { \prime } } N _ { c ^ { \prime } , z ^ { \prime } } } \end{array}$ denotes the total number of completed groups across all units, so that under-observed units receive a larger exploration bonus. Near $\hat { p } _ { c , z } ~ = ~ 0 . 5$ , successful and unsuccessful trajectories coexist, providing contrasting behavior under the same interaction condition. Unlike raw reward variance, which is scale-sensitive and can mix learnability with simulator noise or outliers, $B _ { c , z }$ is bounded and aligned with the verified success criterion. The controller can therefore estimate policyrelative interaction utility directly from outcomes already produced by training.

The controller first samples states uniformly to establish evidence, then activates feedback-guided allocation. Incomplete groups and groups with non-finite outcomes do not update its history. Thereafter,

$$
p _ { t } ( z \mid c ) = ( 1 - \epsilon ) \frac { A _ { c , z } ^ { 1 / \gamma } } { \sum _ { z ^ { \prime } \in \mathcal { Z } } A _ { c , z ^ { \prime } } ^ { 1 / \gamma } } + \epsilon \frac { 1 } { | \mathcal { Z } | } ,\tag{6}
$$

where $\gamma$ is a sampling temperature and ϵ reserves uniform rehearsal. Thus, no state is permanently removed, and conditions that were previously misestimated or excessive can re-enter training as the policy develops.

Algorithm 1 Dual-Loop Self-Evolution with Verifiable   
Emotion Feedback   
Require: scenarios X with intent map h, interaction states   
${ \mathcal { Z } } ,$ policy $\pi _ { \theta } ,$ group size $K$   
1: for each training batch do   
2: for each uniformly sampled scenario x do   
3: set $c = h ( x )$ and sample z uniformly during ini  
tialization, otherwise from $p _ { t } ( z \mid c )$   
4: generate K trajectories under the shared environ  
ment $( x , z )$   
5: score each trajectory with continuous emotion out  
come $R _ { k }$   
6: update π<sub>θ</sub> using group-relative advantages from   
$R _ { 1 : K }$   
7: set $y _ { k } = \mathbb { I } [ R _ { k } \geq \eta ]$ and $p _ { m } = K ^ { - 1 } \sum _ { k }$ y<sub>k</sub>   
8: if the group is complete and finite, accumulate $p _ { m }$   
for state $z$ and unit $( c , z )$   
9: end for   
10: end for

## 3.6 Coupled Self-Evolution

Every complete group thus closes both feedback paths: continuous outcomes update the policy, thresholded group outcomes update the allocation, and the revised distribution generates the next round of experience. Because the simulator, scenario semantics, and verifier stay frozen, experience evolves without a drifting role-playing target. Algorithm 1 summarizes the end-to-end process.

## 4 Experiments

## 4.1 Experimental Setup

Our experiments answer two questions: whether capabilityaligned allocation improves empathetic dialogue under a fixed rollout budget, and whether this improvement comes from adaptive allocation rather than a larger state space or generic sample prioritization. The 500 training scenarios, development split, and 100-scenario SAGE test set are instancedisjoint. Each training scenario retains one of eight native support-intent labels and is composed online with one of 24 interaction states, yielding 192 controller units. Development data is used for configuration and checkpoint selection; test instances never enter policy training, controller updates, or model selection.

We use complementary automatic, interactive, and human evaluations. SAGE measures final user emotion and success/failure over 100 multi-turn scenarios (Zhang et al. 2026). ESC-Eval evaluates 331 fixed-role interactions with the oficial InternLM2 and ESC-RANK protocol (Zhao et al. 2024). EIBench covers 213 held-out scenarios spanning Support, Defense, Repair, and Charm (Zhu et al. 2026). ESConv evaluates 2,895 reference responses using strategy accuracy, BLEU-2, ROUGE-L, BERTScore, and Distinct-2 (all multiplied by 100). To separate training feedback from evaluation, DeepSeek-V3 is used only for training simulation and verification; transfer is measured independently by InternLM2+ESC-RANK in ESC-Eval, Qwen3-Max in

EIBench, reference-based ESConv metrics, and blinded human ratings. RLVER and ours are independently trained three times; remaining fixed checkpoints are evaluated three times under shared seeds. Table 1 reports sample standard deviations for aggregate automatic metrics (Liu et al. 2023; Zhang et al. 2024a).

We also conduct a blinded human evaluation. Three annotators independently rate every system dialogue for 100 randomly sampled held-out scenarios, with dialogue and system order randomized. Ratings use a 0–4 overall-support scale covering empathy, contextual relevance, helpfulness, and coherence, and are averaged across annotators and then scenarios. Ordinal Krippendorf’s α is 0.78.

We compare all systems under a controlled protocol with the same Qwen3-8B initialization, frozen simulator and verifier, data, rollout budget, group size, and optimization settings. Qwen3-8B applies no RL; RLVER performs uniform emotion-reward RL; Interaction State Only adds the 24 states without adaptive allocation; VCRL prioritizes withingroup reward variance (Jiang et al. 2025); and Ours uses the complete controller. All RL systems therefore receive the same GRPO budget, isolating how training interactions are allocated.

## 4.2 Main Results

Against protocol-matched controls, dual-loop self-evolution raises mean SAGE Overall from 72.01 under uniform emotion-reward RL to 79.24 across three independent training runs. Its weakest run (78.11) still exceeds the strongest uniform run (73.76). Interaction states alone reach 69.42 and variance-based allocation reaches 68.51, showing that neither a larger condition space nor generic reward dispersion explains the gain.

SAGE directly evaluates the emotional consequence of a complete multi-turn interaction: Avg. is final user emotion, while Succ. and Fail. summarize trajectories reaching the oficial high- and low-emotion regions. Our framework improves Avg. and success while reducing failure to 11.33%, indicating more reliable long-horizon emotional recovery rather than an isolated response-quality gain. The particularly large SAGE margin is consistent with the method’s target: reallocating complete interaction conditions as the policy’s multi-turn competence changes.

The remaining benchmarks test whether this gain transfers beyond the training-facing emotion outcome. Ours leads all five ESConv metrics, including a Distinct-2 increase from 28.18 to 33.58 over RLVER. Better strategy accuracy, reference overlap, semantic similarity, and lexical diversity show that stronger interactive outcomes are accompanied by broader response quality rather than repetitive reassurance. Ours also attains the highest ESC-Eval aggregate score (2.55) and improves EIBench from −9.79 to −7.55, extending the gain to fixed-role dialogue quality and wider emotional-intelligence behavior. Crucially, adding interaction states without adaptive allocation lowers EIBench from the base model’s −22.40 to −40.60, whereas the complete framework reaches −7.55. This contrast shows that state diversity is not inherently beneficial: the controller must convert it into policy-relevant training experience. Human overall quality likewise rises from 3.0 to 3.5. Together, the results connect the framework’s central advantage in multiturn emotional recovery to complementary improvements in strategy, diversity, and interaction quality.

<table><tr><td></td><td colspan="3">SAGE</td><td colspan="4">ESConv</td><td></td><td>ESC-Eval</td><td>EIBench</td><td>Human</td></tr><tr><td>Counselor training</td><td> $\operatorname { A v g } . \uparrow$ </td><td></td><td>Succ. ↑ Fail. ↓</td><td></td><td>Acc. ↑ B-2 ↑</td><td>R-L↑</td><td>BS↑</td><td>D-2↑</td><td>Avg. ↑</td><td>Overall ↑</td><td>Avg. ↑</td></tr><tr><td>Qwen3-8B</td><td> $5 3 . 8 7 \pm 0 . 3 5$ </td><td>20.00</td><td>29.00</td><td>17.58</td><td>3.09</td><td>10.18</td><td>85.08</td><td>26.37</td><td> $2 . 3 9 \pm 0 . 0 3$ </td><td> $\cdot 2 2 . 4 0 \pm 0 . 6 2$ </td><td>2.3</td></tr><tr><td>RLVER</td><td> $7 2 . 0 1 \pm 1 . 8 8$ </td><td>43.00</td><td>15.33</td><td>16.72</td><td>2.75</td><td>9.83</td><td>85.43</td><td>28.18</td><td> $2 . 4 9 \pm 0 . 0 3$ </td><td> $- 9 . 7 9 \pm 0 . 5 7$ </td><td>3.0</td></tr><tr><td>Interaction State Only</td><td> $6 9 . 4 2 \pm 0 . 4 0$ </td><td>30.00</td><td>19.00</td><td>16.61</td><td>2.99</td><td>10.90</td><td>85.17</td><td>24.12</td><td> $2 . 4 3 \pm 0 . 0 2$ </td><td> $- 4 0 . 6 0 \pm 0 . 7 9$ </td><td>2.7</td></tr><tr><td>VCRL</td><td> $6 8 . 5 1 \pm 0 . 4 0$ </td><td>35.00</td><td>21.00</td><td>17.01</td><td>3.12</td><td>11.07</td><td>85.22</td><td>23.67</td><td> $2 . 5 4 \pm 0 . 0 2$ </td><td> $- 1 7 . 0 8 \pm 0 . 6 3$ </td><td>2.9</td></tr><tr><td>Ours</td><td> ${ \bf 7 9 . 2 4 \pm 1 . 5 6 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>49.0011.33 18.13 3.30 11.45 85.69 33.58</td><td> ${ \bf 2 . 5 5 \pm 0 . 0 2 }$ </td><td> $\mathbf { - 7 . 5 5 \pm 0 . 3 3 }$ </td><td>3.5</td></tr></table>

Table 1: Protocol-matched Qwen3-8B results across complementary automatic, interactive, and human evaluations. Under SAGE, “Avg.” is the Overall score (mean final user emotion), i.e., the value reported as “SAGE Overall” in the text and in Table 5.

<table><tr><td>Method</td><td>Run 1</td><td>Run 2</td><td>Run 3</td><td> $\mathbf { M e a n } \pm \mathbf { S D }$ </td></tr><tr><td>RLVER</td><td>70.03</td><td>72.23</td><td>73.76</td><td> $7 2 . 0 1 \pm 1 . 8 8$ </td></tr><tr><td>Ours</td><td>81.02</td><td>78.58</td><td>78.11</td><td> ${ \bf 7 9 . 2 4 \pm 1 . 5 6 }$ </td></tr></table>

Table 2: SAGE Overall across three independent training runs. SD is the sample standard deviation across runs.

## 4.3 Run-Level Robustness and Transfer Detail

Table 2 exposes the individual SAGE results underlying the training-level statistics in Table 1. The weakest completeframework run (78.11) remains above the strongest RLVER run (73.76), so the aggregate margin is present throughout the observed independent-run range rather than being determined by one favorable optimization run.

The dimension-level ESC-Eval results in Table 3 further show that the aggregate gain is broad rather than concentrated in one rubric. The complete framework improves empathy and suggestion quality most strongly over RLVER while also maintaining or improving fluency, diversity, humanness, technical quality, and overall interaction quality.

Table 4 localizes EIBench transfer. Ours obtains the strongest aggregate reward and the best Support, Repair, and Charm values among the matched systems. Interaction states alone enlarge behavioral variation without deciding where a fixed rollout budget is useful, whereas adaptive allocation converts that diversity into stronger transfer.

## 4.4 Ablations and Training Analysis

We additionally use two diagnostic controllers to isolate allocation design choices. Scenario-only allocation maintains pass-rate statistics for individual scenario identities without the intent–state factorization. The variance-gated pass-rate controller retains our pass-rate priority but admits only highvariance groups when updating controller history; all groups still participate in policy optimization.

Table 5 maps each intervention to a specific part of the framework. Interaction State Only removes the adaptive outer loop while retaining the 24-state environment representation; the 8.69-point drop shows that additional diversity can dilute a fixed budget when it is not allocated according to policy feedback. Scenario-only allocation removes the intent–state factorization and instead estimates individual scenario identities. Its 70.03 score indicates that sparse identity-level histories cannot efectively reuse evidence across semantically related interactions.

The utility-estimation ablations examine how the controller turns outcomes into stable evidence. Replacing hard group outcomes with a soft mapping reduces Overall by 1.59 points. Removing hierarchical sharing costs 3.04 points because sparsely visited units can no longer borrow evidence from the same state under related support intents. The variance-gated controller preserves pass-rate priority but admits only high-variance groups into controller history; its KL growth and 61.45 score highlight the stability advantage of bounded, criterion-aligned pass-rate evidence over raw reward dispersion.

The remaining variants test exploration. Without the uncertainty bonus, Overall falls by 4.19 points, showing that exploiting current estimates alone can neglect conditions that remain poorly understood. Removing uniform rehearsal has a smaller 1.08-point efect, consistent with its role as a safety floor that lets previously mastered, excessive, or misestimated conditions re-enter training. A matched sensitivity study obtains 77.46, 78.11, and 77.72 for $\eta = 4 0 , 5 0 , 6 0 ,$ respectively; the controller therefore does not depend on a narrowly tuned success criterion. Each completed group requires only constant-time statistic updates and normalization over 24 state scores; the outer loop adds no trajectories, model passes, backward passes, or simulator/verifier calls.

## 4.5 Validating the Interaction-State Space

The outer loop relies on two properties of the interactionstate space: requested states should produce observable differences in user behavior, and those diferences should not alter the underlying persona, event, or support need. We test behavioral separability and semantic preservation directly.

For behavioral separability, we audit 120 multi-turn trajectories, with five examples for each of the 24 states. A held-out Qwen3-Max judge sees only the resulting dialogue and attempts to recover the requested level of each state dimension. Because the judge never observes the control labels, successful recovery indicates that the requested state is expressed in dialogue behavior rather than remaining as prompt metadata. Figure 3 shows accuracies of 86.7%, 92.5%, and 82.5% for disclosure, activation, and trust, with corresponding macro-F1 scores of 86.7%, 92.5%, and 82.1%. All three dimensions are recovered correctly in 68.3% of trajectories, a substantially stricter test of joint realization.

(a) Disclosure
<table><tr><td>Counselor training</td><td>Flu.</td><td>Div.</td><td>Emp.</td><td>Sug.</td><td>Hum.</td><td>Tech.</td><td>Overall</td><td>Macro</td></tr><tr><td>Qwen3-8B</td><td>2.78</td><td>2.55</td><td>2.46</td><td>2.31</td><td>1.72</td><td>2.64</td><td>2.27</td><td>2.39</td></tr><tr><td>RLVER</td><td>2.86</td><td>2.68</td><td>2.61</td><td>2.43</td><td>1.82</td><td>2.73</td><td>2.30</td><td>2.49</td></tr><tr><td>Interaction State Only</td><td>2.82</td><td>2.64</td><td>2.52</td><td>2.37</td><td>1.79</td><td>2.69</td><td>2.18</td><td>2.43</td></tr><tr><td>Ours</td><td>2.91</td><td>2.74</td><td>2.72</td><td>2.52</td><td>1.88</td><td>2.74</td><td>2.34</td><td>2.55</td></tr></table>

Table 3: Dimension-level results under the oficial ESC-Eval protocol. All dimensions use the original 0–4 scale; Macro is the arithmetic mean of the seven columns.

<table><tr><td>Method</td><td>Overall</td><td>Support</td><td>Defense Repair</td><td>Charm</td></tr><tr><td>Qwen3-8B</td><td>-22.40</td><td>-43.30</td><td>-21.40 -35.90</td><td>14.60</td></tr><tr><td>RLVER</td><td>-9.79</td><td>-34.27</td><td>-2.55-14.72</td><td>12.80</td></tr><tr><td>States Only</td><td>-40.60</td><td>-75.17</td><td>-12.66-51.52</td><td>-28.83</td></tr><tr><td>Ours</td><td>-7.55</td><td>-27.61</td><td>-6.31-11.64</td><td>17.52</td></tr></table>

Table 4: EIBench raw rewards by category under one shared 213-scenario evaluation protocol. Higher is better.
<table><tr><td>Variant</td><td>Overall ↑</td><td>∆</td></tr><tr><td>Complete framework</td><td>78.11</td><td></td></tr><tr><td>Environment representation</td><td></td><td></td></tr><tr><td>Interaction State Only</td><td>69.42</td><td>-8.69</td></tr><tr><td>Scenario-only allocation</td><td>70.03</td><td>-8.08</td></tr><tr><td>Utility estimation</td><td></td><td></td></tr><tr><td>Soft outcome mapping</td><td>76.52</td><td>-1.59</td></tr><tr><td>No hierarchical sharing</td><td>75.07</td><td>-3.04</td></tr><tr><td>Variance-gated pass-rate controller</td><td>61.45†</td><td>-16.66</td></tr><tr><td>Exploration and rehearsal</td><td></td><td></td></tr><tr><td>No uncertainty bonus</td><td>73.92</td><td>-4.19</td></tr><tr><td>No uniform rehearsal</td><td>77.03</td><td>-1.08</td></tr></table>

<sup>†</sup>Last stable checkpoint before KL instability.  
Table 5: Single-run component ablations on SAGE under a matched training and evaluation protocol. ∆ is measured against the complete-framework run (78.11); Table 1 separately reports three-run statistics for the complete framework.

We then verify that behavioral control preserves scenario meaning. Human reviewers inspect 30 cases; their decisions agree with the automatic judge in 90.0% of cases, with Cohen’s $\kappa = 0 . 8 9$ . Across the full audit, 92.5% of trajectories retain the original persona, event, and hidden need. The interaction states therefore induce recognizable diferences in disclosure, emotion, and trust while leaving the support problem intact. This is precisely the structure required by the outer loop: the controller reallocates distinct but semantically comparable multi-turn experiences, not nominal prompt labels.

## 4.6 Qualitative Analysis

Figure 4 localizes the aggregate gain within one complete interaction. Starting from the same profile and initial emotion, the policies encounter resistance and induce diferent subsequent user trajectories. The base policy repeats broad relationship advice and ends at 45, showing limited adaptation to information disclosed across turns. RLVER endorses the user’s hostile framing; this emotional echoing validates the immediate complaint but intensifies the conflict, and emotion falls to 10. Our policy instead grounds its response in the user’s creative labor, identifies the hidden need for recognition, and reframes the conflict around efort and artistic identity before guiding self-reflection. Emotion consequently rises to 98.

The case clarifies the capability reflected by the main results. The benefit is not simply more positive wording: the learned policy integrates disclosures across turns, distinguishes validation from escalation, and moves from surface afect toward the latent support need. This behavior illustrates the value of repeatedly allocating training to conditions where the current policy exhibits mixed success, rather than spending the same budget uniformly on already mastered or currently uninformative interactions.

## 5 Discussion and Conclusion

Experiments establish that dual-loop self-evolution converts a fixed rollout budget into higher-value multi-turn experience. Its largest gain appears on SAGE, which directly measures complete emotional recovery, while consistent improvements on ESConv, ESC-Eval, EIBench, and human ratings extend this advantage to strategy choice, diversity, and interaction quality. Ablations and state validation further connect the gain to factorized evidence sharing, uncertaintyguided exploration, and behaviorally realizable interaction states.

We introduced a nested framework in which verified emotion feedback serves two coordinated roles: continuous outcomes improve how the dialogue policy responds, while group-level outcome patterns determine what it should practice next. By adapting experience rather than increasing rollout count, the framework makes existing feedback an additional source of training eficiency. More broadly, verifiable feedback can supervise both policy behavior and the organization of the experience from which that behavior is learned.

![](images/540475e6a344024d8afe7e415075db30919ded98f4989548210eecbb51191b07.jpg)

![](images/aeb337c0707e3103ace42279cfc64689ff8dff4a7f339e6aa1c0caa08d957eb7.jpg)

![](images/7430e7404898da9d20716166f9e89bda843ce44dc2c84b5b3d613c5f4bf72a69.jpg)  
Figure 3: Behavioral validation of the interaction-state space: row-normalized confusion matrices (%) of requested vs. recovered (a) disclosure readiness, (b) emotional activation, and (c) relational trust.

![](images/0dfbf441234257e453cb1147199eda886417523537fe206f3ebf95ee3b354040.jpg)  
Figure 4: Representative trajectories under the same SAGE profile and initial emotion. The base policy remains generic, RLVER (uniform emotion-reward RL) echoes escalating hostility, and ours identifies the hidden need for recognition and guides self-reflection. User turns diverge as the frozen simulator reacts to each policy—a trajectory-level rather than turn-aligned comparison.

## References

Ahmadian, A.; Cremer, C.; Gallé, M.; Fadaee, M.; Kreutzer, J.; Pietquin, O.; Üstün, A.; and Hooker, S. 2024. Back to Basics: Revisiting REINFORCE-Style Optimization for Learning from Human Feedback in LLMs. In Proceedings ofACL, 12248–12267.

Auer, P.; Cesa-Bianchi, N.; and Fischer, P. 2002. Finitetime analysis of the multiarmed bandit problem. Machine learning, 47(2): 235–256.

Bengio, Y.; Louradour, J.; Collobert, R.; and Weston, J. 2009. Curriculum learning. In Proceedings ofICML, 41–48.

Chen, G.; Shi, Y.; Li, Y.; Li, B.; Xu, X.; Wei, H.; Ni, S.; Yang, M.; and Ye, J. 2026. EvoTrainer: Co-Evolving LLM Policies and Training Harnesses for Autonomous Agentic Reinforcement Learning. arXiv:2606.03108.

Chen, Z.; Deng, Y.; Yuan, H.; Ji, K.; and Gu, Q. 2024. Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models. In Proceedings of ICML, volume 235, 6621–6642.

Cheng, Y.; Liu, W.; Li, W.; Wang, J.; Zhao, R.; Liu, B.; Liang, X.; and Zheng, Y. 2022. Improving Multi-turn Emotional Support Dialogue Generation with Lookahead Strategy Planning. In Proceedings ofEMNLP, 3014–3026.

Dai, Y.; Gao, N.; Zhang, W.; Wang, J.; Luozichen; Wu, R.; Wang, J.; and Wang, C. 2026. SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue. In Findings ofACL, 3674– 3684.

Dou, Y.; Galley, M.; Peng, B.; Kedzie, C.; Cai, W.; Ritter, A.; Quirk, C.; Xu, W.; and Gao, J. 2025. SimulatorArena: Are User Simulators Reliable Proxies for Multi-Turn Evaluation of AI Assistants? In Proceedings ofEMNLP, 35212–35290.

Graves, A.; Bellemare, M. G.; Menick, J.; Munos, R.; and Kavukcuoglu, K. 2017. Automated Curriculum Learning for Neural Networks. In Proceedings of ICML, volume 70, 1311–1320. PMLR.

Gromada, J.; Kasicka, A.; Komkowska, E.; Krajewski, L.; Krawczyk, N.; Veyret, M.; Przybyl, B.; Rojas-Barahona, L. M.; and Szczerbak, M. K. 2025. Evaluating Conversational Agents with Persona-driven User Simulations based on Large Language Models: A Sales Bot Case Study. In Proceedings ofthe EMNLP Industry Track, 230–245.

Jiang, G.; Feng, W.; Quan, G.; Hao, C.; Zhang, Y.; Liu, G.; and Wang, H. 2025. VCRL: Variance-based Curriculum Reinforcement Learning for Large Language Models. arXiv:2509.19803.

Jiang, L.; Meng, D.; Zhao, Q.; Shan, S.; and Hauptmann, A. G. 2015. Self-Paced Curriculum Learning. In Proceedings ofAAAI, volume 29, 2694–2700.

Jiang, M.; Grefenstette, E.; and Rocktäschel, T. 2021. Prioritized level replay. In Proceedings of ICML, 4940–4950. PMLR.

Jiang, Z.; Han, J.; Li, T.; Wang, X.; Jiang, S.; Meng, X.; Wei, J.; Liang, J.; and Xiao, Y. 2026. Dificulty Is Not Enough:

Curriculum Learning for LLMs Fine-tuning Must Consider Utility. In Proceedings ofAAAI, volume 40, 31365–31373.

Kang, D.; Kim, S.; Kwon, T.; Moon, S.; Cho, H.; Yu, Y.; Lee, D.; and Yeo, J. 2024. Can Large Language Models be Good Emotional Supporter? Mitigating Preference Bias on Emotional Support Conversation. In Proceedings of ACL, 15232–15261.

Kim, J.; Mok, C.; Lee, J.; Kim, H. S.; and Jo, Y. 2025. Dialogue Systems for Emotional Support via Value Reinforcement. In Proceedings of ACL, 28733–28766.

Kumar, M. P.; Packer, B.; and Koller, D. 2010. Self-Paced Learning for Latent Variable Models. In Proceedings of NeurIPS, volume 23.

Li, R.; Huang, H.; Wei, F.; Xiong, F.; Wang, Y.; and Chu, X. 2026. AdaCuRL: Adaptive Curriculum Reinforcement Learning with Invalid Sample Mitigation and Historical Revisiting. In Proceedings ofAAAI, volume 40, 23123–23131.

Lin, Z.; Madotto, A.; Shin, J.; Xu, P.; and Fung, P. 2019. MoEL: Mixture of Empathetic Listeners. In Proceedings of EMNLP, 121–132.

Liu, S.; Zheng, C.; Demasi, O.; Sabour, S.; Li, Y.; Yu, Z.; Jiang, Y.; and Huang, M. 2021. Towards Emotional Support Dialog Systems. In Proceedings ofACL, 3469–3483.

Liu, W.; Huo, L.; Jing, Y.; Zhang, X.; and Xie, J. 2026. MRACL: Multi-Reward Space Guided Adaptive Curriculum Reinforcement Learning for LLMs. In Proceedings ofAAAI, volume 40, 37663–37672.

Liu, Y.; Iter, D.; Xu, Y.; Wang, S.; Xu, R.; and Zhu, C. 2023. G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. In Proceedings of EMNLP, 2511–2522.

Majumder, N.; Hong, P.; Peng, S.; Lu, J.; Ghosal, D.; Gelbukh, A.; Mihalcea, R.; and Poria, S. 2020. MIME: MIMicking Emotions for Empathetic Response Generation. In Proceedings ofEMNLP, 8968–8979.

Nguyen, H. T.; Nguyen, B.; Ma, W.; Zhao, Y.; She, R.; and Nguyen, V. A. 2026. Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards. In Proceedings ofICLR.

Rashkin, H.; Smith, E. M.; Li, M.; and Boureau, Y.-L. 2019. Towards Empathetic Open-domain Conversation Models: A New Benchmark and Dataset. In Proceedings of ACL, 5370– 5381.

Schaul, T.; Quan, J.; Antonoglou, I.; and Silver, D. 2016. Prioritized Experience Replay. In Proceedings of ICLR.

Schulman, J.; Wolski, F.; Dhariwal, P.; Radford, A.; and Klimov, O. 2017. Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Shao, Z.; Wang, P.; Zhu, Q.; Xu, R.; Song, J.; Bi, X.; Zhang, H.; Zhang, M.; Li, Y. K.; Wu, Y.; and Guo, D. 2024. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.

Shi, T.; Wu, Y.; Song, L.; Zhou, T.; and Zhao, J. 2026. Efficient Reinforcement Finetuning via Adaptive Curriculum Learning. Transactions on Machine Learning Research.

Shi, Y.; Hao, J.; and Kong, F. 2025. Beyond Coarse Labels: Fine-Grained Problem Augmentation and Multi-Dimensional Feedback for Emotional Support Conversation. In Findings ofEMNLP, 1634–1647.

Wan, C.; Labeau, M.; and Clavel, C. 2025. EmoDynamiX: Emotional Support Dialogue Strategy Prediction by Modelling Mixed Emotions and Discourse Dynamics. In Proceedings ofNAACL, 1678–1695.

Wang, A.; Yan, Y.; Zhou, N.; Lu, Z.; Lu, W.; Xiao, J.; Zhuang, Y.; and Shen, Y. 2026a. Code-A1: Adversarial Evolving of Code LLM and Test LLM via Reinforcement Learning. arXiv:2603.15611.

Wang, F.; Shen, X.; Yu, J.; and Xia, R. 2025. Flexible Thinking for Multimodal Emotional Support Conversation via Reinforcement Learning. In Findings ofEMNLP, 1341–1356.

Wang, P.; Ma, R.; Zhang, B.; Chen, X.; He, Z.; Luo, K.; Lv, Q.; Jiang, Q.; Xie, Z.; Wang, S.; Li, C.; Li, Y.; Ye, F.; Li, J.; Yang, Y.; Li, J.; Tu, Z.; and Li, X. 2026b. RLVER: Reinforcement Learning with Verifiable Emotion Rewards for Empathetic Agents. In Proceedings of ICLR.

Wu, H.; Tian, Z.; Huang, Z.; Hu, T.; Qiao, L.; Gao, Y.; Liu, F.; and Li, D. 2026. Emotion Trajectory-Aware Retrieval for Markov-Driven Emotion Anticipation in LLM-Based Emotional Support Conversation. In Findings of ACL, 42887– 42905.

Yang, S.; Ma, Z.; Huang, T.; Hu, Y.; Wang, Y.; and Chu, X. 2026. CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution. In Proceedings ofACL, 23015–23036.

Ye, J.; Xiang, L.; Zhang, Y.; and Zong, C. 2025. From Generic Empathy to Personalized Emotional Support: A Self-Evolution Framework for User Preference Alignment. In Findings ofEMNLP, 18826–18853.

Yoon, S.-e.; He, Z.; Echterhof, J.; and McAuley, J. 2024. Evaluating Large Language Models as Generative User Simulators for Conversational Recommendation. In Proceedings ofNAACL, 1490–1504.

Yuan, W.; Pang, R. Y.; Cho, K.; Sukhbaatar, S.; Xu, J.; and Weston, J. 2024. Self-Rewarding Language Models. In Proceedings ofICML, volume 235, 57905–57923.

Zeng, Y.; Sun, Z.; Ji, B.; Min, E.; Cai, H.; Wang, S.; Yin, D.; Zhang, H.; Chen, X.; and Wang, J. 2026. CurES: From Gradient Analysis to Eficient Curriculum Learning for Reasoning LLMs. In Proceedings ofICLR.

Zhang, B.; Ma, R.; Jiang, Q.; Wang, P.; Chen, J.; Xie, Z.; Chen, X.; Wang, Y.; Ye, F.; Li, J.; Yang, Y.; Tu, Z.; and Li, X. 2026. Sentient Agent as a Judge: Evaluating Higher-Order Social Cognition in Large Language Models. In Findings of ACL, 38196–38224.

Zhang, C.; D’Haro, L. F.; Chen, Y.; Zhang, M.; and Li, H. 2024a. A Comprehensive Analysis of the Efectiveness of Large Language Models as Automatic Dialogue Evaluators. In Proceedings ofAAAI, volume 38, 19515–19524.

Zhang, T.; Zhang, X.; Zhao, J.; Zhou, L.; and Jin, Q. 2024b. ESCoT: Towards Interpretable Emotional Support Dialogue Systems. In Proceedings of ACL, 13395–13412.

Zhao, H.; Li, L.; Chen, S.; Kong, S.; Wang, J.; Huang, K.; Gu, T.; Wang, Y.; Wang, J.; Liang, D.; Li, Z.; Teng, Y.; Xiao, Y.; and Wang, Y. 2024. ESC-Eval: Evaluating Emotion Support Conversations in Large Language Models. In Proceedings ofEMNLP, 15785–15810.

Zhao, W.; Sui, X.; Han, X.; Deng, Y.; Hu, Y.; Guo, J.; Qin, L.; Du, Q.; Wang, S.; Zhao, Y.; Qin, B.; and Liu, T. 2025. Chain of Strategy Optimization Makes Large Language Models Better Emotional Supporter. In Findings ofEMNLP, 15361– 15381.

Zheng, C.; Sabour, S.; Wen, J.; Zhang, Z.; and Huang, M. 2023. AugESC: Dialogue Augmentation with Large Language Models for Emotional Support Conversation. In Findings ofACL, 1552–1568.

Zhou, J.; Chen, Z.; Wang, B.; and Huang, M. 2023. Facilitating Multi-Turn Emotional Support Conversation with Positive Emotion Elicitation: A Reinforcement Learning Approach. In Proceedings ofACL, 1714–1729.

Zhu, R.; Huang, X.; Wu, Y.; Wang, R.; Sun, Z.; Ren, T.; Luo, W.; Qiu, B.; Ye, J.; Li, Y.; and Hu, W. 2026. EIBench: A Simulator-Based Benchmark and Turn-Credit RL for Emotion Management. arXiv:2606.15532.